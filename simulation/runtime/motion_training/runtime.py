"""One model/precision/Adam implementation for all recipes. No result imports."""
import argparse, copy, random, sys, math
from pathlib import Path
import numpy as np
import torch
from ardy.model import load_model
from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel
from g1_motion_contract import inspect_stats,validate_checkpoint
from g1_padding_contract import install_padding_contract
from g1_root_body_gradient import configure_root_body_gradient
from g1_optimizer_precision import create_adamw
from g1_training_precision import configure_training_precision,FP32_BOUNDARIES
from g1_training_launch import install_training_launch_contract
from ._compat.checkpoint_config import sequence_config_kwargs
from ._compat import legacy_loss
from .io import identity,copy_state

_LAUNCH=False
def load(cp,training=False,device='cuda',fresh_seed=None,official_frontend=False):
    global _LAUNCH
    torch.set_num_threads(2)
    if not _LAUNCH:install_training_launch_contract();_LAUNCH=True
    ck=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    # Keep the checkpoint bytes and metadata intact. Relocate only the runtime
    # copy of its normalization folder; validate_checkpoint checks its hash.
    ck=dict(ck,config=dict(ck['config']));cfg=ck['config']
    from paths import STATS
    cfg['stats_dir']=str(STATS)
    metadata=inspect_stats(cfg['stats_dir']);validate_checkpoint(ck,metadata)
    if fresh_seed is not None:
        random.seed(fresh_seed);np.random.seed(fresh_seed%(2**32));torch.manual_seed(fresh_seed);torch.cuda.manual_seed_all(fresh_seed)
    m=load_model('g1',device=device,text_encoder=False,fresh_denoiser=fresh_seed is not None and not official_frontend)
    m.training_codec_contract=cfg.get('training_codec_contract','legacy');m.checkpoint_training_manifest=cfg.get('manifest')
    legacy_loss._replace_stats(m,cfg['stats_dir']);m.generation_contract=cfg.get('generation_contract','legacy')
    opts,_=sequence_config_kwargs(argparse.Namespace(),ck)
    c=install_ardy_sequence_kernel(m,ArdySequenceKernelConfig(mode='hybrid',attention_backend='sdpa',window_size_chunks=4,num_sink_tokens=4,**opts))
    raw=m.denoiser.model
    if fresh_seed is None:legacy_loss._load_denoiser_weights(raw,ck)
    else:
        ck=dict(ck,step=0,initialization=dict(policy='fresh_denoiser',seed=fresh_seed,
            source_checkpoint_metadata_only=str(cp),official_denoiser_loaded=False,
            parameter_sources={n:'constructor_initialized_native_denoiser' for n,_ in raw.named_parameters()}))
    install_padding_contract(m,cfg.get('padding_contract','legacy'));configure_root_body_gradient(raw,cfg.get('root_body_gradient_scale',0))
    configure_training_precision(raw,m.autoencoder,FP32_BOUNDARIES);m.device=next(raw.parameters()).device;m.eval()
    for p in m.parameters():p.requires_grad_(False)
    for p in raw.parameters():p.requires_grad_(training)
    assert not cfg.get('content_memory_enabled',False)
    return m,c,ck
def optimizer(m,ck,cfg=None,resume=False):
    raw=m.denoiser.model
    cfg=cfg or {};policy='inherit' if resume else cfg.get('optimizer_state_policy','inherit')
    kwargs=dict(lr=cfg.get('lr',3e-6),betas=tuple(cfg.get('betas',[.9,.999])),eps=cfg.get('eps',1e-8),weight_decay=cfg.get('weight_decay',.01))
    opt=create_adamw(raw.parameters(),precision='fp32_master_v1',parameter_names=['module.'+n for n,p in raw.named_parameters()],**kwargs)
    if policy!='fresh':
        opt.load_state_dict(copy.deepcopy(ck['optimizer']))
        assert identity(opt.state_dict())==identity(ck['optimizer'])
        if policy=='reset':
            # Preserve FP32 masters exactly; clear only Adam history.
            opt.state.clear()
            for g in opt.param_groups:g.update(kwargs)
        else:
            assert policy=='inherit'
            for g in opt.param_groups:
                for key in ['betas','eps','weight_decay']:
                    if key in cfg:assert g[key]==kwargs[key],('optimizer_config_mismatch',key)
                if not ck.get('framework',{}).get('scheduler'):assert g['lr']==kwargs['lr'],'Inherited LR differs from requested LR'
    else:
        assert ck.get('initialization',{}).get('policy')=='fresh_denoiser'
    spec=cfg.get('scheduler',{'kind':'constant','lr':kwargs['lr']})
    saved=ck.get('framework',{}).get('scheduler')
    if saved and (resume or policy=='inherit'):
        assert saved['spec']==spec,'Schedule changed without a new registered stage'
        opt.training_schedule=copy.deepcopy(saved)
    else:opt.training_schedule=dict(spec=spec,completed=0)
    return opt

def set_update_lr(opt):
    schedule=opt.training_schedule;spec=schedule['spec'];step=schedule['completed']+1
    if spec['kind']=='constant':lr=spec['lr']
    else:
        assert spec['kind']=='warmup_cosine'
        warm=spec['warmup'];total=spec['total'];peak=spec['peak'];minimum=spec['minimum']
        lr=spec['start']+(peak-spec['start'])*(step-1)/max(warm-1,1) if step<=warm else minimum+(peak-minimum)*.5*(1+math.cos(math.pi*min((step-warm)/(total-warm),1)))
    for g in opt.param_groups:g['lr']=lr
    return lr
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(s,fallback_seed=None):
    py=s.get('python',s.get('random'))
    if py is None or 'numpy' not in s:
        if fallback_seed is None:raise ValueError('Incomplete legacy RNG: explicit new-stage fallback seed required')
        random.seed(fallback_seed);np.random.seed(fallback_seed%(2**32))
    if py is not None:random.setstate(py)
    if 'numpy' in s:np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state(s['cuda'].cpu())
def master_named(raw,opt):return [(n,opt.masters.get(i,p)) for i,(n,p) in enumerate(raw.named_parameters())]
def import_sources():
    root=Path(__file__).resolve().parent;out={}
    for n,mod in list(sys.modules.items()):
        f=getattr(mod,'__file__',None)
        if f and n.startswith(('motion_training','ardy.','ardy_adapter','sequence_kernel','g1_')):
            p=Path(f).resolve();assert p.is_relative_to(root),(n,str(p));out[n]=str(p)
    return out
