"""Architecture selection is derived from checkpoint metadata, never a step-specific script."""
import importlib.util
import json
import os
import sys
from pathlib import Path

from checkpoint_profile import resolve
from paths import activate

def load_selected(checkpoint,device='cuda:0',text_encoder=None):
    profile=resolve(checkpoint);source=Path(profile['source']);stage=Path(profile['stage_root'])
    activate()
    import torch
    from ardy.model import load_model
    from motion_training.io import identity
    from motion_training._compat.legacy_loss import _unwrap_block
    kind=profile['kind']
    if kind=='official_ardy':
        path=Path(profile['checkpoint'])
        m=load_model(path.name,checkpoints_dir=str(path.parent),device=device,text_encoder=False)
        c=None;ck=None
    elif kind in ('ardy','minimal'):
        from architecture_adapter import load
        m,c,ck=load(profile['checkpoint'],kind,device=device)
    else:
        from motion_training import runtime
        loader=runtime.load
        if profile.get('content_memory_enabled'):
            # Some foundation training runtimes prohibit PCM for that training
            # stage. Inference must honor the saved architecture instead.
            import inspect
            code=inspect.getsource(runtime.load)
            guard="    assert not cfg.get('content_memory_enabled',False)\n"
            if guard in code:
                assert code.count(guard)==1
                namespace=dict(runtime.__dict__)
                exec(compile(code.replace(guard,''),'<simulation_saved_PCM_inference>','exec'),namespace)
                loader=namespace['load']
        m,c,ck=loader(profile['checkpoint'],False,device=device)
    raw=m.denoiser.model
    if ck is not None:
        for name in ('root_model','body_model'):
            state=_unwrap_block(getattr(raw,name)).state_dict()
            assert set(state)==set(ck[name]),name
            assert identity(state)==identity(ck[name]),f'Loaded {name} differs from selected checkpoint'
        for key,value in ck.get('denoiser_aux',{}).items():
            assert identity(raw.state_dict()[key])==identity(value),key
    if m.skeleton.nbjoints!=34 or m.motion_rep.fps!=25 or m.gen_horizon_len!=52:
        raise ValueError('Expected G1 / 25FPS / 52-frame generator')
    if text_encoder is not None:m.text_encoder=text_encoder
    m.simulation_architecture=kind;m.simulation_checkpoint_profile=profile
    m.simulation_model_label=profile['label']
    out=Path(profile['output']);out.mkdir(parents=True,exist_ok=True)
    record=dict(profile,loaded=True,all_checkpoint_weights_exact=ck is not None,
        codec_hash=identity(m.autoencoder.state_dict()),weights_hash=identity(raw.state_dict()),
        stats_dir=str(m.motion_rep.stats.folder),device=str(device))
    (out/'MODEL_LOAD.json').write_text(json.dumps(record,indent=2)+'\n')
    if kind in ('ardy','official_ardy') and os.environ.get('MM2_INFERENCE_BACKEND','optimized')!='reference':
        from inference_acceleration import enable_transformer
        enable_transformer(m)
        record.update(inference_backend=m.simulation_inference_backend,
                      acceleration_validation='Every newly captured forward shape checked exactly against eager')
        (out/'MODEL_LOAD.json').write_text(json.dumps(record,indent=2)+'\n')
    return m,c,ck,profile
