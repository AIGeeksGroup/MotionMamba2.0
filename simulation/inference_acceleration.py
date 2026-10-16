"""Exact, checkpoint-specific acceleration for the unified simulation entry."""
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from paths import LIVE, SOURCE, SIM, warm_embedding


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
    from checkpoint_profile import file_sha
    from motion_training.io import identity
    from motion_training import acceleration
    sys.path.append(str(LIVE))
    sys.path.append(str(SIM/'examples'))
    from g1_masked_inference import install_masked_graphs
    from g1_inference_graph_warmup import configure_graph_capacity
    from g1_inference_launch import install_forward_launch_contract
    from g1_live_replanning import ReplanningOnlineG1

    install_forward_launch_contract('mamba3_siso_angle_w4s1_v1')
    install_masked_graphs(client)
    configure_graph_capacity(client, 64)
    paths = {Path(__file__), LIVE/'g1_masked_inference.py', LIVE/'g1_live_replanning.py',
             SOURCE/'motion_training/_runtime/dit_sequence_kernel/inference_graph.py'}
    paths.update(Path(mod.__file__) for name, mod in sys.modules.items()
                 if getattr(mod, '__file__', None) and name.startswith(
                     ('g1_', 'sequence_kernel', 'dit_sequence_kernel', 'ardy_adapter', 'ardy.model')))
    sources = {str(p): file_sha(p) for p in sorted(paths) if p.is_file()}
    key = dict(checkpoint_sha256=profile['checkpoint_sha256'], sources=sources,
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


class TransformerGraphs:
    """Capture a stateless Transformer forward, preserving its actual arithmetic.

    Every new input shape is compared exactly to eager before being admitted.
    Tensor values/masks/positions are copied on every replay. Gradients always
    use the original method. No released engine is reused for other weights.
    """
    def __init__(self, block):
        self.block = block
        self.original = block.forward
        self.entries = {}
        self.replays = 0
        self.validated = 0

    def __call__(self,*args,**kwargs):
        if torch.is_grad_enabled() or self.block.training:
            return self.original(*args,**kwargs)
        position = getattr(self.block,'motion_token_embedding',None)
        token_index = kwargs.get('token_index',args[6] if len(args)>6 else None)
        if position is not None and token_index is not None:
            assert token_index.abs().max() < position.max_len, 'Position index outside saved embedding'
        def signature(x):
            if torch.is_tensor(x):return (tuple(x.shape),x.dtype,str(x.device),tuple(x.stride()))
            if isinstance(x,(tuple,list)):return tuple(signature(v) for v in x)
            if isinstance(x,dict):return tuple((k,signature(v)) for k,v in sorted(x.items()))
            return (type(x).__name__,x)
        key=(signature(args),signature(kwargs),torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'))
        def clone(x):
            if torch.is_tensor(x):return x.clone()
            if isinstance(x,tuple):return tuple(clone(v) for v in x)
            if isinstance(x,dict):return {k:clone(v) for k,v in x.items()}
            return x
        def copy(a,b):
            if torch.is_tensor(a):a.copy_(b)
            elif isinstance(a,tuple):
                for x,y in zip(a,b):copy(x,y)
            elif isinstance(a,dict):
                for k in a:copy(a[k],b[k])
        if key not in self.entries:
            if len(self.entries)>=128:
                raise RuntimeError('Transformer graph capacity exceeded; input shape needs qualification')
            a,k=clone(args),clone(kwargs)
            reference=self.original(*a,**k).clone()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.original(*a,**k)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            if position is not None:position._simulation_graph_capture=True
            try:
                with torch.cuda.graph(graph):output=self.original(*a,**k)
            finally:
                if position is not None:position._simulation_graph_capture=False
            graph.replay()
            assert torch.equal(reference,output),'Transformer CUDA graph differs from eager'
            self.entries[key]=(a,k,graph,output)
            self.validated+=1
        a,k,graph,output=self.entries[key]
        copy(a,args);copy(k,kwargs);graph.replay();self.replays+=1
        return output.clone()


def enable_transformer(model):
    import types
    for block in (model.denoiser.model.root_model, model.denoiser.model.body_model):
        position=getattr(block,'motion_token_embedding',None)
        if position is not None:
            # Preserve the bounds check outside capture; all embedding math
            # below is the official operation sequence, including int cast.
            def positional_forward(self,x,index):
                if not getattr(self,'_simulation_graph_capture',False):
                    assert index.abs().max() < self.max_len
                safe_index=torch.where(index>=0,index,index+self.pe.shape[0])
                encoding=torch.nn.functional.embedding(safe_index.int(),self.pe)
                return self.dropout(x+encoding)
            position.forward=types.MethodType(positional_forward,position)
        for p in block.parameters():p.requires_grad_(False)
        dispatch=TransformerGraphs(block)
        block.forward=dispatch
        block.simulation_graph_dispatch=dispatch
    model.simulation_inference_backend='transformer_cuda_graphs'
