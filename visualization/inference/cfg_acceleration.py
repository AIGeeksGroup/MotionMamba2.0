"""Exact, checkpoint-specific acceleration for the unified simulation entry."""
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from bootstrap import LIVE
from bootstrap import SOURCE
from bootstrap import SIM
from bootstrap import warm_embedding


def configure(client, enabled):
    # Minimal has attention state but no Mamba wrapper. Do not enable a
    # recurrent kernel on a different architecture.
    for block in (client.controller.root_block, client.controller.body_block):
        block.inference_no_diagnostics = enabled
        block.inference_no_pending = enabled
        block.inference_cfg_parallel = False
        block.inference_no_commit_diagnostics = enabled
        block.inference_serial_cfg_graph = enabled
        stack = block.sequence_stack
        stack.inference_graph = enabled
        stack.inference_graph_shared_kv = enabled
        stack.inference_fast_graph_checks = enabled
        for layer in stack.layers:
            mixer = layer.mixer
            if mixer._mamba_wrapper is not None:
                mixer._mamba_wrapper.inference_fused_adt = enabled
            mixer.attention.inference_cache = enabled
            mixer.inference_prepared_attention = enabled
            mixer.inference_dual_attention = False
            layer.inference_fused_adaln = False
            layer.inference_fused_pointwise = enabled


def enable_sequence(model, client, profile):
    from integrity import file_sha
    from integrity import identity
    import integrity as acceleration
    sys.path.append(str(LIVE))
    from g1_masked_inference import install_masked_graphs
    from g1_inference_graph_warmup import configure_graph_capacity
    from motion_mamba2.model.kernel_launch import install_forward_launch_contract
    from g1_live_replanning import ReplanningOnlineG1

    install_forward_launch_contract('mamba3_siso_angle_w4s1_v1')
    install_masked_graphs(client)
    configure_graph_capacity(client, 64)
    paths = {Path(__file__), LIVE/'g1_masked_inference.py', LIVE/'g1_live_replanning.py',
             SOURCE/'kernels/masked_cuda_graph.py'}
    paths.update(SOURCE.rglob('*.py'))
    paths.update(LIVE.rglob('*.py'))
    sources = {str(p): file_sha(p) for p in sorted(paths) if p.is_file()}
    key = dict(checkpoint_sha256=profile['checkpoint_sha256'], settings_sha256=profile['settings_sha256'], sources=sources,
               gpu=torch.cuda.get_device_name(model.device), torch=torch.__version__)
    dest = Path(profile['output'])/'ONLINE_ACCELERATION_GATE.json'
    if dest.is_file():
        saved = json.loads(dest.read_text())
        if saved.get('passed') and all(saved.get(k) == v for k, v in key.items()):
            configure(client, True)
            model.online_masked_graphs = True
            return str(dest)

    text = warm_embedding(model.device)
    mask = torch.ones(1, 1, dtype=torch.bool, device=model.device)
    controls = [{}, {'root2d':[{'frame':74,'xz':[.8,1.5]}]},
        {'root2d':[{'frame':75,'xz':[-.8,.5]}]}, {'root2d':[{'frame':180,'xz':[.8,1.5]}]},
        {'root2d':[{'frame':52,'xz':[.2,.3]},{'frame':247,'xz':[1.,2.]}]},
        dict(version='g1_joint_interaction_v1', intent='decelerate',velocity_xz=[0.,0.],event_id='stop'),
        dict(version='g1_joint_interaction_v1', intent='move',velocity_xz=[0.,.6],heading=0.,event_id='restart'), {}]
    runner = ReplanningOnlineG1(model, client)
    checks = []
    tick = time.monotonic()
    sessions = []
    try:
        for seed in (2026101013, 2026101014):
            expected = []
            for optimized in (False, True):
                configure(client, optimized)
                s = runner.new_session(seed)
                sessions.append(s)
                for i, control in enumerate(controls):
                    r = runner.step(s,text,mask,'walk',i,0,control,i)
                    value = identity([r['normalized_motion'],r['qpos'],
                                      acceleration.state_tree(s.snapshots[i+1])])
                    if not optimized:
                        expected.append(value)
                    else:
                        assert value == expected[i], ('accelerated output/state/RNG mismatch',seed,i)
                        checks.append(dict(seed=seed,window=i,exact=True))
                    assert runner.step(s,text,mask,'walk',i,0,control,i) is s.last_result
                    runner.publish(s,i+1,0)
                print('SIMULATION_ACCELERATION_PAIRED',seed,optimized,flush=True)
            for source, lookahead in ((264,0),(304,1)):
                values = []
                before = identity(acceleration.state_tree(s.snapshots[8]))
                for optimized in (False,True):
                    configure(client,optimized)
                    r=runner.prepare_replan(s,text,mask,'walk',source,0,controls[3],11,lookahead,controls[2])
                    assert identity(acceleration.state_tree(s.snapshots[8])) == before
                    trial=s.pending_replan[2]
                    values.append(identity([r['normalized_motion'],r['qpos'],
                        [[v['normalized_motion'],v['qpos']] for v in r['following_windows']],
                        acceleration.state_tree(trial.snapshots[trial.windows])]))
                    runner.cancel_replan(s)
                assert values[0]==values[1], ('accelerated replan mismatch',seed,source)
                checks.append(dict(seed=seed,source=source,lookahead=lookahead,exact=True))
        result=dict(key,passed=True,checks=checks,seconds=time.monotonic()-tick,
                    scope='Two seeds, 8 recursive windows, changing sparse masks, future goals, FIFO, stop/restart, retry, prefix replans; exact motion/state/RNG')
        dest.write_text(json.dumps(result,indent=2)+'\n')
    except Exception as exc:
        dest.write_text(json.dumps(dict(key,passed=False,error=str(exc),checks=checks),indent=2)+'\n')
        raise
    finally:
        for s in sessions:
            if hasattr(s,'archive'):shutil.rmtree(s.archive,ignore_errors=True)
    configure(client,True)
    model.online_masked_graphs=True
    print('SIMULATION_ACCELERATION_GATE_PASS',dest,flush=True)
    return str(dest)
