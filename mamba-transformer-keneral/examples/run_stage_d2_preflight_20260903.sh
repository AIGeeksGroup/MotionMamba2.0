#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
OUT=/data4/zts/result/stage_d2_learned_causal_memory_v2_20260903
GPU=${D2_PREFLIGHT_GPU:-0}

export PYTHONPATH=$ROOT/src:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

cd "$ROOT"
"$PY" -m pytest -q
RUN_MAMBA_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=$GPU "$PY" -m pytest -q \
  tests/test_content_memory_trainability.py::test_real_mamba_cross_window_gradient_attached_positive_detached_negative
"$PY" -m py_compile \
  src/state_memory_operator/content_memory.py \
  src/sequence_kernel/config.py \
  src/sequence_kernel/operator.py \
  src/ardy_adapter/ardy_dit_block.py \
  src/ardy_adapter/integration.py \
  examples/train_humanml3d_ardy_ddp.py \
  examples/train_humanml3d_ardy_tiny.py \
  examples/train_learned_delayed_cue_memory_benchmark.py
git diff --check

if [[ ! -f "$OUT/results.json" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY" \
    examples/train_learned_delayed_cue_memory_benchmark.py \
    --output-dir "$OUT" --device cuda:0 --seed 20260903 \
    --steps 800 --batch-size 64 --eval-samples 512 \
    --learning-rate 3e-3 --log-every 50 \
    >"$OUT.log" 2>&1
fi

"$PY" - "$OUT/results.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("gate", {}).get("status") != "PASS":
    raise SystemExit("learned delayed-cue gate did not pass")
print(json.dumps(result["gate"], indent=2))
PY
