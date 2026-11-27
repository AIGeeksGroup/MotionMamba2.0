"""Load the public Motion Mamba 2.0 tensor checkpoint with strict key checks."""
from pathlib import Path
import argparse
import copy
import json
import os
import torch
from motion_mamba2.model.configuration import ROOT,load_environment,require_path
from motion_mamba2.model.variants import read_model_config

def _frontend(config,device):
    from motion_mamba2.model.loading import instantiate_from_dict
    return instantiate_from_dict(config,overrides=dict(device=device,text_encoder=None))

def load_model(config='configs/model.yaml',paths='configs/paths.env',device='cuda:0',*,text_encoder=False,minimal_baseline=False):
    values=load_environment(paths)
    checkpoint=require_path(values,'MM2_CHECKPOINT')
    require_path(values,'MM2_CODEC_CHECKPOINT')
    require_path(values,'MM2_NORMALIZATION_PATH',directory=True)
    require_path(values,'MM2_SKELETON_PATH',directory=True)
    from motion_mamba2.model.kernel_launch import install_forward_launch_contract
    from motion_mamba2.model.decoder_history import install_decoder_history
    from motion_mamba2.model.padding import install_padding_contract
    from motion_mamba2.model.integration import SequenceModelConfig,install_motion_mamba2_sequence_kernel
    spec=read_model_config(config, minimal_baseline=minimal_baseline)
    if not minimal_baseline:
        install_forward_launch_contract(spec['execution']['mamba_forward_launch'])
    model=_frontend(spec['frontend'],device)
    representation=spec['representation']
    if model.skeleton.nbjoints!=34 or model.gen_horizon_len!=52 or model.motion_rep.fps!=25:
        raise ValueError('The public checkpoint requires the G1 34-joint, 25-FPS, 52-frame architecture')
    # The codec's body/local-root statistics must match the inference representation.
    from tempfile import TemporaryDirectory
    import numpy as np
    from motion_mamba2.model.representation import MotionMamba2MotionRep
    folder=TemporaryDirectory(prefix='motion_mamba2_normalization_')
    expected=spec['normalization']
    for name in ['mean','std']:
        array=np.asarray(expected[name],dtype=expected['dtype'])
        if not np.isfinite(array).all():raise ValueError('Nonfinite normalization')
        np.save(Path(folder.name)/(name+'.npy'),array,allow_pickle=False)
    old=model.motion_rep
    rep=MotionMamba2MotionRep(old.skeleton,old.fps,stats_path=folder.name)
    for field in ['body_stats','local_root_stats']:
        for stat in ['mean','std']:
            if not torch.equal(getattr(getattr(old,field),stat).cpu(),getattr(getattr(rep,field),stat).cpu()):
                raise ValueError('Normalization disagrees with the frozen codec')
    raw=model.denoiser.model
    model.motion_rep=raw.motion_rep=rep
    model.autoencoder._motion_rep=rep
    for converter in [model.hybrid,raw.hybrid]:
        converter._motion_rep=rep
        converter.__dict__.pop('motion_rep',None)
    model._inference_stats_folder=folder
    install_decoder_history(model,representation['decoder_history_frames'])
    options=dict(spec['sequence']);options['mamba_dtype']=getattr(torch,options['mamba_dtype'])
    client=install_motion_mamba2_sequence_kernel(model,SequenceModelConfig(**options))
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if set(weights)!={'root_model','body_model','denoiser_aux'}:
        raise ValueError('Expected the public tensor-only MotionMamba2.pt format')
    if not all(isinstance(v,torch.Tensor) for part in weights.values() for v in part.values()):
        raise ValueError('Checkpoint contains non-tensor payloads')
    raw=model.denoiser.model
    for branch in ['root_model','body_model']:
        target=getattr(raw,branch)
        current=target.state_dict()
        if set(current)!=set(weights[branch]):raise ValueError(f'Weight keys differ: {branch}')
        # Construction precision is part of the architecture contract.
        for key,value in weights[branch].items():
            if current[key].shape!=value.shape or current[key].dtype!=value.dtype:
                raise ValueError(f'Weight shape or precision differs: {branch}.{key}')
        target.load_state_dict(weights[branch],strict=True)
    state=raw.state_dict()
    aux={k for k in state if not k.startswith(('root_model.','body_model.'))}
    if aux!=set(weights['denoiser_aux']):raise ValueError('Auxiliary weight keys differ')
    with torch.no_grad():
        for key,value in weights['denoiser_aux'].items():
            if state[key].shape!=value.shape or state[key].dtype!=value.dtype:raise ValueError(key)
            state[key].copy_(value.to(state[key].device))
    for branch in ['root_model','body_model']:
        for key,value in weights[branch].items():
            if not torch.equal(getattr(raw,branch).state_dict()[key].cpu(),value):raise ValueError(f'Weight load failed: {key}')
    model.motion_contract=representation['motion_contract']
    model.generation_contract=representation['generation_contract']
    raw.root_model.output_projection_fp32=spec['execution']['root_output_fp32']
    install_padding_contract(model,representation['padding_contract'])
    if text_encoder:
        from motion_mamba2.model.text_encoder import ExactMotionMamba2TextEncoder
        model.text_encoder=ExactMotionMamba2TextEncoder(base_mirror=str(require_path(values,'MM2_TEXT_ENCODER_BASE',directory=True)),
            mntp_adapter=str(require_path(values,'MM2_TEXT_ENCODER_MNTP',directory=True)),
            supervised_adapter=str(require_path(values,'MM2_TEXT_ENCODER_ADAPTER',directory=True)),device=device)
    model.device=next(raw.parameters()).device
    model.eval();model.requires_grad_(False)
    return model,client,spec

def load_inference_package(settings_path,device='cuda:0',text_encoder=False):
    """Read a deployment specification generated from configs/model.yaml."""
    import tempfile
    settings_path=Path(settings_path).resolve()
    spec=json.loads(settings_path.read_text())
    values={'MM2_CHECKPOINT':str((settings_path.parent/spec['weights']).resolve()),
            'MM2_FRONTEND_DIR':str((settings_path.parent/spec['frontend_assets']).resolve())}
    if spec.get('skeleton_assets'):values['MM2_SKELETON_PATH']=str((settings_path.parent/spec['skeleton_assets']).resolve())
    for k,v in values.items():os.environ[k]=v
    return load_model(device=device,text_encoder=text_encoder)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',default=str(ROOT/'configs/paths.env'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args();m,c,s=load_model(paths=a.paths,device=a.device)
    print(json.dumps(dict(model='MotionMamba2',pcm=bool(s['sequence']['content_memory_enabled']),weights_loaded=True)))
if __name__=='__main__':main()
