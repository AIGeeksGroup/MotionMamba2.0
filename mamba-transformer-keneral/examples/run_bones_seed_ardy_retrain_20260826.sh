#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_CACHE=true
export PYTHONPATH=/data3/zts/packages/mamba_op_triton:/data3/zts/packages/state-spaces-mamba:/data3/zts/MotionMamba2.0/mamba-transformer-keneral/src${PYTHONPATH:+:$PYTHONPATH}

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUTPUT=$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original
INITIAL=$BASE/bones_seed_gate5k_original/checkpoint_step_005000.pt
TRAIN=$BASE/bones_seed_ardy_train
VAL=$BASE/bones_seed_ardy_val
TEXT=$BASE/bones_seed_text_cache_llm2vec
LOG=$OUTPUT/train.stdout.log

mkdir -p "$OUTPUT"

latest_checkpoint() {
  find "$OUTPUT" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -print | sort | tail -n 1
}

validate_stage() {
  local expected_steps=$1
  "$PY" - "$OUTPUT/metrics.json" "$OUTPUT/original_humanml3d_ddp.pt" "$expected_steps" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, checkpoint_path, expected_steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if not metrics_path.is_file() or not checkpoint_path.is_file():
    raise SystemExit("missing metrics or final checkpoint")
metrics = json.loads(metrics_path.read_text())
required = (
    metrics.get("status") == "PASS"
    and metrics.get("steps") == expected_steps
    and metrics.get("all_gradients_finite") is True
    and metrics.get("final_eval_loss", float("inf")) <= 4.0
    and metrics.get("root_output_weight_norm", 0.0) >= 2.0
    and metrics.get("final_learning_rate") == 1e-5
)
if not required:
    raise SystemExit(f"stabilization gate failed: {metrics}")
print(
    f"stage_pass steps={expected_steps} eval={metrics['final_eval_loss']:.6f} "
    f"root_output_norm={metrics['root_output_weight_norm']:.6f}"
)
PY
}

run_stage() {
  local target_steps=$1 resume
  if validate_stage "$target_steps" >/dev/null 2>&1; then
    validate_stage "$target_steps"
    return
  fi
  resume=$(latest_checkpoint)
  [[ -n "$resume" ]] || resume=$INITIAL
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
    --model g1 --mode original \
    --manifest "$TRAIN/manifest.jsonl" --dev-manifest "$VAL/manifest.jsonl" \
    --stats-dir "$TRAIN" --text-cache "$TEXT" --output-dir "$OUTPUT" \
    --steps "$target_steps" --batch-size 8 --learning-rate 1e-5 \
    --override-resume-learning-rate --resume "$resume" --seed 123 --frames 104 \
    --history-probability 0.5 --constraint-dropout 0.1 --text-dropout 0.1 \
    --sequential-cache-probability 0 --eval-protocol explicit_history --eval-batches 4 \
    --boundary-loss-weight 10 --root-velocity-loss-weight 50 \
    --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 \
    --eval-every 1000 --save-every 5000 --keep-last-checkpoints 10 \
    --max-final-eval-loss 4 --min-root-output-weight-norm 2 \
    >>"$LOG" 2>&1
  validate_stage "$target_steps"
}

# The 10k stage is an explicit fail-fast check before spending the remaining
# 40k steps. Both stages preserve the original total-step data/RNG stream.
run_stage 10000
run_stage 50000
date -Is > "$OUTPUT/RETRAIN_COMPLETE"
echo "retrain_complete output=$OUTPUT"
