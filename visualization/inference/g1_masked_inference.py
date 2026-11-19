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


from motion_mamba2.kernels.masked_graph import graph_allowed_mask
from motion_mamba2.kernels.masked_graph import install_masked_graphs


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
