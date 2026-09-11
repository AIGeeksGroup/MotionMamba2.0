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
OUTPUT_ROOT=$BASE/bones_seed_protocol_aligned_ft_20260826
TRAIN=$BASE/bones_seed_ardy_train
VAL=$BASE/bones_seed_ardy_val
TEXT=$BASE/bones_seed_text_cache_llm2vec
mkdir -p "$OUTPUT_ROOT"

initial_for() {
  case "$1" in
    original) echo "$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt" ;;
    minimal) echo "$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt" ;;
    hybrid) echo "$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt" ;;
    *) return 1 ;;
  esac
}

eval_protocol_for() {
  [[ "$1" == original ]] && echo explicit_history || echo cache_only
}

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -print | sort | tail -n 1
}

validate_stage() {
  local mode=$1 stage=$2 output=$OUTPUT_ROOT/$mode
  "$PY" - "$output/metrics.json" "$output/checkpoint_step_$(printf '%06d' "$stage").pt" "$mode" "$stage" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics_path, checkpoint, mode, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
if not metrics_path.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise SystemExit(1)
metrics = json.loads(metrics_path.read_text())
counts = {int(key): int(value) for key, value in metrics["rollout_window_counts"].items()}
allowed = {2, 4, *range(6, 11), *range(12, 20)}
valid = (
    metrics["mode"] == mode
    and metrics["steps"] == stage
    and metrics["world_size"] == 8
    and metrics["global_batch_size"] == 32
    and metrics["train_scope"] == "denoiser"
    and metrics["sequential_rollout_curriculum"] == "long_horizon_v1"
    and metrics["sequential_rollout_steps"] == stage
    and metrics["all_gradients_finite"] is True
    and metrics["root_output_weight_norm"] >= 2.0
    and math.isfinite(metrics["final_eval_loss"])
    and set(counts).issubset(allowed)
    and sum(counts.values()) == stage
)
raise SystemExit(0 if valid else 1)
PY
}

run_stage() {
  local mode=$1 stage=$2 output=$OUTPUT_ROOT/$mode checkpoint eval_protocol
  mkdir -p "$output"
  if validate_stage "$mode" "$stage"; then
    echo "skip_completed_training mode=$mode stage=$stage"
    return
  fi
  checkpoint=$(latest_checkpoint "$output")
  local init_args=()
  if [[ -n "$checkpoint" ]]; then
    init_args=(--resume "$checkpoint" --override-resume-learning-rate)
  else
    init_args=(--init-checkpoint "$(initial_for "$mode")")
  fi
  eval_protocol=$(eval_protocol_for "$mode")
  echo "training_start mode=$mode stage=$stage $(date -Is)"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
    --standalone --nproc-per-node=8 "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
    --model g1 --mode "$mode" --manifest "$TRAIN/manifest.jsonl" \
    --dev-manifest "$VAL/manifest.jsonl" --stats-dir "$TRAIN" --text-cache "$TEXT" \
    --output-dir "$output" --steps "$stage" --batch-size 4 --learning-rate 5e-6 \
    --train-scope denoiser --seed 321 --frames 988 --history-probability 0 \
    --constraint-dropout 0.1 --text-dropout 0.1 --sequential-cache-probability 1 \
    --sequential-rollout-curriculum long_horizon_v1 --eval-protocol "$eval_protocol" \
    --eval-batches 1 --boundary-loss-weight 10 --root-velocity-loss-weight 50 \
    --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 \
    --eval-every 0 --save-every 100 --keep-last-checkpoints 12 \
    "${init_args[@]}" >>"$output/train_stage${stage}.stdout.log" 2>&1
  validate_stage "$mode" "$stage"
  cp "$output/metrics.json" "$output/metrics_step_$(printf '%06d' "$stage").json"
  echo "training_complete mode=$mode stage=$stage $(date -Is)"
}

for stage in 500 1000; do
  for mode in original minimal hybrid; do
    run_stage "$mode" "$stage"
  done
  STAGE=$stage bash "$ROOT/examples/run_bones_seed_protocol_aligned_eval_20260826.sh"
done

"$PY" "$ROOT/examples/write_bones_seed_protocol_aligned_report.py"
date -Is >"$OUTPUT_ROOT/PIPELINE_COMPLETE"
echo "protocol_aligned_pipeline_complete output=$OUTPUT_ROOT"
