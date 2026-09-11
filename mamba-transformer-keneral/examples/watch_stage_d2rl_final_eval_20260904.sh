#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
RUN_ROOT=/data4/zts/result/stage_d2rl_bones_long_20260904
TRAIN_SESSION=d2rl_long_20260904

while [[ ! -f $RUN_ROOT/LONG_TRAINING_COMPLETE ]]; do
  if ! tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    echo "[$(date -Is)] training session ended before the 5000-step completion marker" >&2
    date -Is >"$RUN_ROOT/FINAL_EVALUATION_NOT_STARTED"
    exit 1
  fi
  sleep 60
done

echo "[$(date -Is)] starting registered D2R-L step5000 selection evaluation"
cd "$ROOT"
D2_FAMILY=d2rl D2_STAGE=5000 ./examples/run_stage_d2_eval_20260903.sh

set +e
PYTHONPATH=src "$PY" examples/analyze_stage_d2_gate.py \
  --stage 5000 \
  --eval-root "$RUN_ROOT/step5000/selection_eval" \
  --full-train-root "$RUN_ROOT/step5000/full" \
  --no-delta-train-root "$RUN_ROOT/step5000/no_delta" \
  --learned-results /data4/zts/result/stage_d2r_contextual_cue_v3_20260903/results.json \
  --parity-results /data4/zts/result/stage_d2r_step0_native_parity_20260903/results.json \
  --results-output "$RUN_ROOT/step5000/D2RL_STEP5000_RESULTS.json" \
  --causal-output "$RUN_ROOT/step5000/D2RL_STEP5000_CAUSAL_INTERVENTION_RESULTS.json"
analysis_status=$?
set -e

echo "$analysis_status" >"$RUN_ROOT/step5000/analysis_exit_status"
date -Is >"$RUN_ROOT/FINAL_EVALUATION_COMPLETE"
echo "[$(date -Is)] D2R-L final evaluation complete; gate exit=$analysis_status"
