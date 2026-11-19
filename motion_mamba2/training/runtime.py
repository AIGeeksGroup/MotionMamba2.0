"""Fresh generator/PCM construction with a frozen motion tokenizer."""
from pathlib import Path
import torch
from motion_mamba2.model.configuration import load_environment, require_path, read_config
from motion_mamba2.model.checkpoint import _frontend
from .precision import configure_training_precision, FP32_BOUNDARIES
from .launch import install_training_launch_contract
from .history_backward import install

def create_model(config, paths, device):
    values=load_environment(paths)
    for key in ['MM2_CODEC_CHECKPOINT', 'MM2_NORMALIZATION_PATH', 'MM2_SKELETON_PATH']:
        require_path(values,key,directory=key!='MM2_CODEC_CHECKPOINT')
    from motion_mamba2.model.kernel_launch import install_forward_launch_contract
    from motion_mamba2.model.decoder_history import install_decoder_history
    from motion_mamba2.model.padding import install_padding_contract
    from motion_mamba2.model.integration import SequenceModelConfig,install_motion_mamba2_sequence_kernel
    spec=read_config(config)
    if spec['frontend']['denoiser'].get('ckpt_path') is not None:
        raise ValueError('Generator must start with random weights')
    seq=spec['sequence']
    if not (seq['content_memory_enabled'] and seq['content_memory_read_enabled'] and seq['content_memory_slots']==2):
        raise ValueError('This recipe requires enabled two-slot PCM')
    install_training_launch_contract()
    install()
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
    model.motion_contract=representation['motion_contract']
    model.generation_contract=representation['generation_contract']
    raw.root_model.output_projection_fp32=spec['execution']['root_output_fp32']
    install_padding_contract(model,representation['padding_contract'])
    raw=model.denoiser.model
    raw.requires_grad_(True)
    raw.train()
    model.autoencoder.requires_grad_(False)
    model.autoencoder.eval()
    configure_training_precision(raw, model.autoencoder, FP32_BOUNDARIES)
    model.device=next(raw.parameters()).device
    return model,client,spec
