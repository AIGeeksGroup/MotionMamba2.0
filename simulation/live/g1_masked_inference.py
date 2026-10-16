"""Service-only CUDA graphs for sparse future attention masks.

Preserve serial CFG GEMMs, dense Mamba recurrence, exact key/query masking,
and all committed state. Training and unsupported reads retain the eager path.
Frozen weight snapshots are never edited; only the read-only graph dispatcher
is loaded from the maintained source after an equivalence gate.
"""
import importlib
import importlib.util
from pathlib import Path
import shutil
import time
import types

import torch


def graph_allowed_mask(*, key_valid, query_valid, prefix_tokens, causal_within_chunk):
    """Same reference mask, without host sync or dynamic nonzero in capture."""
    batch, q_len = query_valid.shape
    kv_len = key_valid.shape[1]
    allowed = key_valid[:,None,:].expand(batch,q_len,kv_len)
    if causal_within_chunk:
        q = torch.arange(q_len,device=key_valid.device)[:,None]
        k = torch.arange(kv_len,device=key_valid.device)[None,:]
        allowed = allowed & (k <= prefix_tokens+q)[None]
    allowed = allowed & query_valid[:,:,None]
    if kv_len:
        # The reference lets invalid queries see only key zero before zeroing
        # their outputs. Static shape and tensor operations permit replay with
        # a *different* sparse mask; no mask values are baked into the graph.
        allowed = allowed.clone()
        allowed[:,:,0] |= ~query_valid
    return allowed


def install_masked_graphs(client):
    for block in (client.controller.root_block,client.controller.body_block):
        stack = block.sequence_stack
        if getattr(stack,'inference_masked_graph',False):continue
        module_name = stack.__class__.__module__.rsplit('.',1)[0]+'.inference_graph'
        target = importlib.import_module(module_name)
        path = Path(__file__).parent/'motion_training/_runtime/dit_sequence_kernel/inference_graph.py'
        spec = importlib.util.spec_from_file_location(module_name,path)
        maintained = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(maintained)
        target.ReadOnlyStackGraphs = maintained.ReadOnlyStackGraphs
        # Existing graphs contain the old dispatcher. Recreate scratch only;
        # accepted KV/SSM/PCM state lives on the adapter and is unchanged.
        if hasattr(stack,'_inference_graphs'):del stack._inference_graphs
        for layer in stack.layers:
            att = layer.mixer.attention
            original = att._build_allowed_mask
            def build(self, _original=original, **kwargs):
                if getattr(self,'_masked_graph_capture',False):return graph_allowed_mask(**kwargs)
                return _original(**kwargs)
            att._build_allowed_mask = types.MethodType(build,att)
        stack.inference_masked_graph = True


def warm_future_graphs(runtime, text, mask):
    """Warm bounded future shapes at every FIFO occupancy before readiness.

    Actual ten-step sampling; use discarded isolated sessions only. Each graph
    replays new mask values and state, rather than caching a prepared motion.
    """
    from g1_inference_graph_warmup import configure_graph_capacity
    configure_graph_capacity(runtime.client,256)
    session=runtime.new_session(2026101016)
    tick=time.perf_counter()
    try:
        for i in range(6):
            runtime.step(session,text,mask,'warm',i,0)
            runtime.publish(session,i+1,0)
        for boundary in range(5):
            for length in range(56,249,4):
                ctl={'root2d':[{'frame':length-1,'xz':[.8,1.5]}]}
                runtime.prepare_replan(session,text,mask,'warm',boundary*52,0,ctl,1)
                runtime.cancel_replan(session)
            print('FUTURE_GRAPH_WARMUP',boundary,'49 bounded lengths',flush=True)
        return dict(seconds=time.perf_counter()-tick,history_occupancies=5,
                    future_lengths=49,capacity_per_branch=256,
                    allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    finally:
        shutil.rmtree(session.archive,ignore_errors=True)
