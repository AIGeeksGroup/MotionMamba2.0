"""Inference-only PCM checkpoint loader and accelerated execution binding."""
import json
import os
from pathlib import Path
from integrity import file_sha
from integrity import identity

def load_generator(settings_path,mode=None,stats_dir=None,device='cuda:0'):
    from bootstrap import activate
    activate()
    from motion_mamba2.model.checkpoint import load_inference_package
    model,client,settings=load_inference_package(settings_path,device=device,text_encoder=False)
    if not settings['sequence'].get('content_memory_enabled'):
        raise ValueError('This release requires a checkpoint with PCM enabled')
    for b in [client.controller.root_block,client.controller.body_block]:
        if (not all(layer.mixer.content_memory is not None for layer in b.sequence_stack.layers)
                or b.content_memory_read_enabled_runtime is False or b.content_memory_write_enabled_runtime is False):
            raise ValueError('PCM read and write must both be enabled')
    spec=Path(settings_path).resolve()
    weights=Path(os.environ['MM2_CHECKPOINT']).resolve()
    output=Path(os.environ['MM2_RUNTIME_DIR'])/'generator'
    output.mkdir(parents=True,exist_ok=True)
    profile=dict(checkpoint=str(weights),checkpoint_sha256=file_sha(weights),
                 settings_sha256=file_sha(spec),kind='native',label='MotionMamba2',output=str(output))
    model.simulation_checkpoint_profile=profile
    from cfg_acceleration import enable_sequence
    gate=enable_sequence(model,client,profile)
    model.online_inference_backend='optimized'
    model.online_control_contract='g1_bounded_future_smooth_control_v1'
    model.online_masked_graphs=True
    model.simulation_model_label='MotionMamba2'
    model.simulation_architecture='native'
    (output/'MODEL_LOAD.json').write_text(json.dumps(dict(profile,pcm_enabled=True,
        pcm_read=True,pcm_write=True,cfg_branches=3,cfg_graph='three_serial_branches_one_graph_per_stack_forward',
        acceleration_gate=gate,weights_hash=identity(model.denoiser.model.state_dict())),indent=2)+'\n')
    return model,client
