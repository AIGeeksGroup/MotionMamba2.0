#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUT=$BASE/bones_seed_long_memory_stage_b_20260831
TRAIN=$BASE/bones_seed_ardy_train/manifest.jsonl
VAL=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
MIN_CKPT=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt
HYB_CKPT=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
ARDY_CKPT=$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt

export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

run_one() {
  local gpu=$1 name=$2 mode=$3 checkpoint=$4 stage=$5
  shift 5
  local output=$OUT/$name
  if [[ -f $output/checkpoint_step_$(printf '%06d' "$stage").pt ]]; then
    return
  fi
  local initialization=(--init-checkpoint "$checkpoint")
  if (( stage == 250 )); then
    initialization=(--resume "$output/checkpoint_step_000100.pt")
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
    --model g1 --mode "$mode" --manifest "$TRAIN" --dev-manifest "$VAL" \
    --stats-dir "$STATS" --text-cache "$TEXT" --output-dir "$output" \
    --steps "$stage" --batch-size 1 --learning-rate 1e-6 --seed 827321 --frames 416 \
    --history-probability 0 --constraint-dropout 0.1 --text-dropout 0.1 \
    --sequential-cache-probability 0 --eval-protocol explicit_history \
    --eval-batches 1 --eval-every 50 --save-every "$stage" \
    --keep-last-checkpoints 2 --no-final-resume-checkpoint \
    --boundary-loss-weight 10 --root-velocity-loss-weight 50 \
    --constraint-loss-weight 1 --fk-constraint-loss-weight 1 \
    --foot-skate-loss-weight 50 --differentiable-rollout-probability 0.25 \
    --differentiable-rollout-windows 8 --differentiable-rollout-diffusion-steps 1 \
    --differentiable-rollout-warm-prefix-windows 4 \
    --long-slope-loss-weight 0.001 --long-auc-loss-weight 0.0001 \
    --long-final-loss-weight 0.001 --long-slope-margin-cm-per-window 0 \
    "${initialization[@]}" --log-every 10 "$@" \
    >"$OUT/${name}_stage${stage}.log" 2>&1
}

run_group() {
  local stage=$1
  shift
  local pids=()
  while (( $# )); do
    local spec=$1
    shift
    IFS='|' read -r gpu name mode checkpoint extras <<<"$spec"
    read -r -a extra_args <<<"$extras"
    run_one "$gpu" "$name" "$mode" "$checkpoint" "$stage" "${extra_args[@]}" &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return "$status"
}

for stage in 100 250; do
  run_group "$stage" \
    "3|ardy|original|$ARDY_CKPT|" \
    "4|minimal_w4|minimal|$MIN_CKPT|--attention-window-chunks 4" \
    "5|minimal_w8|minimal|$MIN_CKPT|--attention-window-chunks 8" \
    "6|minimal_parammatch|minimal|$MIN_CKPT|--attention-window-chunks 4 --minimal-ff-size 5248" \
    "7|hybrid_carry|hybrid|$HYB_CKPT|--attention-window-chunks 4 --training-state-intervention carry"
  run_group "$stage" \
    "4|minimal_w16|minimal|$MIN_CKPT|--attention-window-chunks 16" \
    "5|hybrid_no_carry|hybrid|$HYB_CKPT|--attention-window-chunks 4 --training-state-intervention fresh_each_window"
done

date -Is >"$OUT/STAGE_B_TRAINING_COMPLETE"
