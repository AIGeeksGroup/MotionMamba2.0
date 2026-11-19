"""GT batches with frozen causal encoding and explicit future visibility."""
import functools
import concurrent.futures
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import data, losses
from .checkpoint import tensor_hash as identity
from motion_mamba2.model.decoder_history import CONTRACT, decode_training_prediction
from motion_mamba2.model.bounded_codec import encode_windows
from motion_mamba2.model.motion_model import get_three_mask_from_len
from motion_mamba2.kernels.codec_graph import FrozenCodecGraphs

@functools.lru_cache(maxsize=4096)
def text_cpu(caption, roots):
    for root in roots:
        p = Path(root)/(data.text_key(caption)+'.npy')
        if p.exists():
            a = np.load(p)
            assert a.size == 4096 and np.isfinite(a).all()
            return a.astype(np.float32, copy=False).reshape(4096)
    raise FileNotFoundError(caption)

def read_cpu(record, cfg):
    r = data.validate_record(record)
    a = np.load(r['feature_path'], mmap_mode='r')[r['crop_start']:r['crop_end']].copy()
    h, future = r['H'], r['F']
    assert a.shape == (h+52+future,414) and a.dtype == np.float32 and np.isfinite(a).all()
    rng = np.random.default_rng(r['seed'])
    cond = data.sample_control_condition(torch.from_numpy(a), None, rng=rng, horizon=52,
        generation_start=h, control=r['condition'] if r['condition']!='none' else 'root_waypoint',
        out_of_horizon=r['out_of_horizon'], condition_dropout=0)
    rawmask = cond['motion_mask'].numpy().copy()
    premask = rawmask.copy()
    if r['condition']=='none':premask.fill(False)
    pre = bool(premask[h+52:].any())
    td = bool(rng.random()<cfg['text_dropout']); cd = bool(rng.random()<cfg['constraint_dropout'])
    mask = premask.copy()
    if cd: mask.fill(False)
    return dict(record=r, a=a, rawmask=rawmask, mask=mask, text=text_cpu(r['caption'],tuple(cfg['text_roots'])),
        td=td,cd=cd,pre=pre,visible=bool(mask[h+52:].any()))

def submit_reads(client, records, cfg, workers):
    if workers == 0:return [read_cpu(r,cfg) for r in records]
    if not hasattr(client, '_cpu_prefetch_pool'):
        client._cpu_prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cpu-motion-read')
    return [client._cpu_prefetch_pool.submit(read_cpu,r,cfg) for r in records]

def materialize(values):
    return [v.result() if isinstance(v,concurrent.futures.Future) else v for v in values]

def upload(array, device):
    t = torch.from_numpy(np.stack(array))
    if device.type == 'cuda':return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)

def build_group(m,c,prepared,cfg,audit=False,serial_codec=False):
    b=len(prepared);r0=prepared[0]['record'];h,f=r0['H'],r0['F'];device=torch.device(m.device)
    assert all((p['record']['H'],p['record']['F'])==(h,f) for p in prepared)
    physical=upload([p['a'] for p in prepared],device)
    origin=physical[:,max(h-1,0),[0,2]].clone()
    physical[:,:,0]-=origin[:,None,0];physical[:,:,2]-=origin[:,None,1]
    rawmask=upload([p['rawmask'] for p in prepared],device)
    mask=upload([p['mask'] for p in prepared],device)
    observed=m.motion_rep.normalize(torch.where(rawmask,physical,torch.zeros_like(physical)))*rawmask
    dropped=torch.tensor([p['cd'] or p['record']['condition']=='none' for p in prepared],device=device)
    observed.masked_fill_(dropped[:,None,None],0)
    text=upload([p['text'] for p in prepared],device)[:,None]
    td=torch.tensor([p['td'] for p in prepared],device=device)
    text.masked_fill_(td[:,None,None],0)
    with torch.no_grad():
        known=m.motion_rep.normalize(physical[:,:h+52]);hp=None
        if serial_codec:
            if not hasattr(c,'_frozen_codec_graphs'):
                c._frozen_codec_graphs=FrozenCodecGraphs(m.hybrid)
            coder=c._frozen_codec_graphs
            if h:
                ht=coder.uniform_history(known[:,:h])
                hp=ht[:,-13:]
            context=known[:,max(0,h-52):h+52]
            targetG=coder.encode(context)[0][:,-13:]
        else:
            if h:
                pad=torch.ones(b,h,device=device,dtype=torch.bool)
                ht,_=encode_windows(m.hybrid,known[:,:h],pad.sum(1),pad);hp=ht[:,-13:]
            context=known[:,max(0,h-52):h+52];pad=torch.ones(context.shape[:2],device=device,dtype=torch.bool)
            targetG,_=m.hybrid.get_hybrid_motion_from_explicit(context,pad.sum(1),pad);targetG=targetG[:,-13:]
        clean=torch.zeros(b,(h+52+f)//4,148,device=device)
        if h:clean[:,:h//4]=ht
        clean[:,h//4:h//4+13]=targetG
    hl=torch.full((b,),h,device=device,dtype=torch.long);gl=torch.full((b,),52,device=device,dtype=torch.long)
    fl=torch.tensor([f if p['visible'] else 0 for p in prepared],device=device,dtype=torch.long)
    # Construct masks at the original padded extent, then mask invisible future rows.
    fullfl=torch.full((b,),f,device=device,dtype=torch.long)
    hm,gm,fm=get_three_mask_from_len(hl,gl,fullfl,h+52+f,device)
    visible=torch.tensor([p['visible'] for p in prepared],device=device,dtype=torch.bool)
    fm=fm & visible[:,None]
    htm=hm.reshape(b,-1,4).all(-1);gtm=gm.reshape(b,-1,4).all(-1);ftm=fm.reshape(b,-1,4).all(-1)
    m.diffusion.calc_diffusion_vars(torch.arange(m.diffusion.num_base_steps,device=device))
    ts=[];noises=[]
    for p in prepared:
        gen=torch.Generator(device=device).manual_seed(p['record']['seed'])
        ts.append(torch.randint(m.diffusion.num_base_steps,(1,),generator=gen,device=device))
        noises.append(torch.randn((1,*clean.shape[1:]),generator=gen,device=device))
    t=torch.cat(ts);noise=torch.cat(noises);x=torch.zeros_like(clean)
    x[htm]=clean[htm];x[gtm]=m.diffusion.q_sample(clean,t,noise)[gtm]
    pad_mask=torch.ones_like(hm)
    if f:pad_mask=torch.where(visible[:,None],pad_mask,hm|gm)
    c.reset_memory()
    fixed=dict(x=x,target=clean,text=text,text_mask=(~td)[:,None],timestep=t,noise=noise,
        heading=torch.atan2(physical[:,max(h-1,0),4],physical[:,max(h-1,0),3]),history_len=hl,generation_len=gl,
        future_len=fl,history_mask=hm,generation_mask=gm,future_mask=fm,history_token_mask=htm,
        generation_token_mask=gtm,future_token_mask=ftm,observed=observed,motion_mask=mask.float(),
        hybrid_converter=m.hybrid,motion_contract=CONTRACT,pad_mask=pad_mask,lengths=hl+gl+fl,
        decoder_history_hybrid=hp,supervised_motion=known[:,h:h+52],sequential_cache=False)
    receipts=[]
    if audit:
        for i,p in enumerate(prepared):
            receipts.append(dict(p['record'],text_dropped=p['td'],condition_dropped=p['cd'],F_nonzero=f>0,
                F_before_drop=p['pre'],F_visible_after_drop=p['visible'],observed_scalars=int(mask[i].sum()),
                G_goal_scalars=int(mask[i,h:h+52].sum()),effective_G_frames=52,effective_G_tokens=13,
                latent_codec_context_frames=min(h,52)+52,decoder_history_frames=min(h,52),
                target_hash=identity(targetG[i:i+1]),input_hash=identity([x[i:i+1],text[i:i+1],observed[i:i+1],mask[i:i+1]]),
                noise_hash=identity(noise[i:i+1]),physical_G_hash=identity(physical[i:i+1,h:h+52]),earlier_prefix_primed=0,
                future_visibility_contract='mask_no_visible_future_v1'))
    return fixed,receipts

def components_group(raw,f,pred):
    b=pred.shape[0];gt=f['generation_token_mask'];gen=gt[:,:,None]
    values=F.smooth_l1_loss(pred.float(),f['target'].float(),beta=1,reduction='none')
    hybrid,hn=losses.mean_valid(values,gen);rd=raw.nframe_root_dim
    assert rd==20 and pred.shape[-1]==148
    root,_=losses.mean_valid(values[:,:,:rd],gen);body,_=losses.mean_valid(values[:,:,rd:],gen)
    assert torch.equal(gt,gt[:1].expand_as(gt))
    selected=pred[:,gt[0]].float();valid=torch.ones(b,52,device=pred.device,dtype=torch.bool)
    dec=decode_training_prediction(selected,dict(motion_contract=CONTRACT,hybrid_converter=f['hybrid_converter'],
        history_mask=~valid,generation_mask=valid,decoder_history_hybrid=f.get('decoder_history_hybrid')))
    target=f['supervised_motion'].float();bs=raw.motion_rep.body_slice
    decoded=(dec[:,:,bs].float()-target[:,:,bs]).abs().flatten(1).mean(1).mean()
    obs=f['observed'][:,f['generation_mask'][0]];mask=f['motion_mask'][:,f['generation_mask'][0]].bool()
    goal,gn=losses.mean_valid((dec.float()-obs.float()).abs(),mask)
    fk=raw.motion_rep.inverse(dec,is_normalized=True)['posed_joints'][:,:,1:].float()
    positions=raw.motion_rep.inverse(dec,is_normalized=True,posed_joints_from='positions')['posed_joints'][:,:,1:].float()
    consistency=(fk-positions).norm(dim=-1).flatten(1).mean(1).mean()
    terms=dict(hybrid=hybrid,decoded_body=decoded,goal=goal,fk_consistency=consistency)
    diagnostics=torch.stack([hybrid.detach(),decoded.detach(),goal.detach(),consistency.detach(),root.detach(),body.detach()])
    return sum(terms.values()),dict(diagnostics=diagnostics),terms
