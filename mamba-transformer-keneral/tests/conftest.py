"""Shared pytest setup for Mamba-3 GPU tests.

CUDA must be initialized before the Triton overlay is inserted onto
`sys.path` (a plain `PYTHONPATH` overlay breaks CUDA). This must run once,
before any test module imports `mamba_ssm` or `state_memory_operator`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

TRITON_OVERLAY = os.environ.get("MAMBA_TRITON_OVERLAY", "/tmp/mamba_op_triton")
MAMBA_REPO = os.environ.get(
    "MAMBA_REPO", "/home/plote/mambamotion/basecode/mamba/state-spaces-mamba"
)
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
LOCAL_ARDY = str(Path(__file__).resolve().parents[2] / "ardy")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Mamba-3 kernels"
)

if LOCAL_ARDY not in sys.path:
    sys.path.insert(0, LOCAL_ARDY)

if torch.cuda.is_available():
    torch.empty(1, device="cuda")
    for path in (TRITON_OVERLAY, MAMBA_REPO, SRC_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
