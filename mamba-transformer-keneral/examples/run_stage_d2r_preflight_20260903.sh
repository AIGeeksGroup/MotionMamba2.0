#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
GPU=${D2R_PREFLIGHT_GPU:-0}
BENCH=$BASE/stage_d2r_contextual_cue_v3_20260903

export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

cd "$ROOT"
"$PY" -m pytest -q
RUN_MAMBA_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=$GPU "$PY" -m pytest -q \
  tests/test_content_memory_trainability.py::test_real_mamba_cross_window_gradient_attached_positive_detached_negative
"$PY" -m compileall -q src examples tests
git diff --check

if [[ ! -f "$BENCH/results.json" ]]; then
  mkdir -p "$BENCH"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" examples/train_learned_delayed_cue_memory_benchmark.py \
    --output-dir "$BENCH" --device cuda:0 --seed 20260903 --contextual \
    --steps 800 --batch-size 64 --eval-samples 512 --learning-rate 3e-3 \
    --log-every 50 >"$BENCH/run.log" 2>&1
fi

"$PY" - "$BENCH/results.json" <<'PY'
import json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
if result.get("gate", {}).get("status") != "PASS":
    raise SystemExit("contextual delayed-cue gate failed")
print(json.dumps(result["gate"], indent=2))
PY
