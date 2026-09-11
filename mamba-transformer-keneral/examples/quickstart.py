"""Minimal end-to-end usage of the State-Transition Memory Operator.

Run with local defaults:

/home/plote/miniconda3/envs/env_isaaclab/bin/python \
    mamba_transformer_keneral/examples/quickstart.py

Run on a server/workspace with explicit overlays:

MAMBA_TRITON_OVERLAY=/path/to/triton_overlay \
MAMBA_REPO=/path/to/state-spaces-mamba \
CUDA_VISIBLE_DEVICES=0 python examples/quickstart.py

This is a shape/wiring demo, not a training script -- it shows how to
construct the operator, carry `OperatorState` + `EventMemory` across
chunks, and read the diagnostics you should log to tell whether memory is
actually being used (see `docs/mamba_integration_contract.md` Section 7).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get(
    "MAMBA_REPO", "/home/plote/mambamotion/basecode/mamba/state-spaces-mamba"
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        return 2
    torch.empty(1, device="cuda")

    sys.path.insert(0, TRITON_OVERLAY)
    sys.path.insert(0, MAMBA_REPO)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from mamba_ssm.modules.mamba3 import Mamba3
    from state_memory_operator import StateMemoryOperatorConfig, StateTransitionMemoryOperator

    d_model = 128
    mamba3 = Mamba3(
        d_model=d_model,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=1,
        is_mimo=False,
        chunk_size=16,
        layer_idx=0,
        device="cuda",
        dtype=torch.bfloat16,
    )

    config = StateMemoryOperatorConfig(
        d_model=d_model,
        memory_capacity=64,
        memory_backend="participative",  # or "fifo"
        injection_mode="input",           # or "output"
        topk=4,
    )
    operator = StateTransitionMemoryOperator(mamba3, config).cuda()

    device = torch.device("cuda")
    batch_size, chunk_len, num_chunks = 4, 16, 8

    op_state = operator.init_operator_state(batch_size, device=device)
    memory = operator.init_memory(batch_size, device=device)

    for step in range(num_chunks):
        x = torch.randn(batch_size, chunk_len, d_model, device=device, dtype=torch.bfloat16)

        # Whatever "this chunk's content" means for your task -- here just a
        # pooled first-frame stand-in. For motion, this is the natural place
        # to plug in a pooled/keyframe representation of the chunk.
        content = x.mean(dim=1).float()

        out, op_state, memory, aux = operator.forward_chunk_with_memory(
            x,
            op_state,
            memory,
            read_content=content,
            write_content=content,
            # reset_mask=episode_reset_mask,   # (batch,) bool, per-row episode boundaries
            # write_mask=commit_mask,           # (batch,) bool, e.g. only on accepted clean windows
        )

        print(
            f"step {step}: out={tuple(out.shape)} "
            f"memory_entries={memory.num_entries} "
            f"retrieval_entropy={aux.get('retrieval_entropy')} "
            f"injection_delta_norm={aux['injection_delta_norm']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
