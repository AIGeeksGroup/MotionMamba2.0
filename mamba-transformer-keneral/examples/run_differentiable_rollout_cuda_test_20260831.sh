#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed: torch.cuda.is_available() is false")
print({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
PY
exec "$PY" -m pytest -q \
  "$ROOT/tests/test_differentiable_rollout.py::test_cuda_windows_6_to_8_backpropagate_through_early_recurrent_carry"
