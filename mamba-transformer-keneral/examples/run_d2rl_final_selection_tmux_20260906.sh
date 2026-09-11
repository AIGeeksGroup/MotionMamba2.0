#!/usr/bin/env bash
set -euo pipefail

# Post-training evaluator for the registered D2R-L 50k selection protocol.
# Run this script inside tmux; it waits for the training completion marker.
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
TRAIN_OUT=$BASE/stage_d2rl_50k_20260906
OUT=$TRAIN_OUT/final_selection_eval
PANELS=$BASE/bones_seed_long_memory_panel_v2_20260831/selection
LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
NATIVE=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
MINIMAL=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt

export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

while [[ ! -f "$TRAIN_OUT/TRAINING_COMPLETE" ]]; do
  if ! tmux has-session -t d2rl50k 2>/dev/null; then
    echo "training tmux session ended before TRAINING_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

FINAL=$(find "$TRAIN_OUT" -maxdepth 1 -type f -name 'checkpoint_step_050000.pt' -print -quit)
[[ -n "$FINAL" && -s "$FINAL" ]] || { echo "missing final checkpoint: $TRAIN_OUT/checkpoint_step_050000.pt" >&2; exit 2; }
mkdir -p "$OUT"
sha256sum "$FINAL" > "$OUT/final_checkpoint.sha256"

run_table4() {
  local gpu=$1 name=$2 checkpoint=$3 mode=$4 dir="$OUT/$2/table4"
  [[ -f "$dir/result.json" ]] && return 0
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --mode "$mode" --model g1 --dataset-name BONES-SEED-v004 --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$TEXT" --protocol table4 \
    --samples 64 --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260831 --device cuda:0 --panel "$PANELS/table4.jsonl" --protocol-lock "$LOCK" \
    --attention-window-chunks 4 --output "$dir/result.json" --per-sample-output "$dir/per_sample.jsonl" \
    >"$dir/eval.log" 2>&1
}

run_long() {
  local gpu=$1 name=$2 checkpoint=$3 mode=$4 seconds=$5 protocol=$6 intervention=${7:-normal}
  local dir="$OUT/$name/${seconds}s_${protocol}${intervention:+__${intervention}}"
  [[ -f "$dir/result.json" ]] && return 0
  mkdir -p "$dir"
  local samples=16 selection=$seconds
  [[ "$seconds" == 9 ]] && samples=64
  [[ "$seconds" == 20 ]] && selection=40
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --mode "$mode" --model g1 --dataset-name BONES-SEED-v004 --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$TEXT" --protocol "$protocol" \
    --total-seconds "$seconds" --selection-min-seconds "$selection" --samples "$samples" --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260831 --device cuda:0 \
    --panel "$PANELS/${seconds}s_${protocol}.jsonl" --protocol-lock "$LOCK" --attention-window-chunks 4 \
    --content-memory-intervention "$intervention" --output "$dir/result.json" \
    --per-sample-output "$dir/per_sample.jsonl" --state-diagnostics-output "$dir/state.jsonl" \
    >"$dir/eval.log" 2>&1
}

wave() {
  local status=0 pid spec
  for spec in "$@"; do eval "$spec" & done
  for pid in $(jobs -pr); do wait "$pid" || status=1; done
  return "$status"
}

# Four models use exactly the same panels, seeds and evaluator settings.
wave \
  'run_table4 0 d2rl_50k "$FINAL" hybrid' \
  'run_table4 1 native_hybrid "$NATIVE" hybrid' \
  'run_table4 2 minimal_w4 "$MINIMAL" minimal' \
  'run_table4 3 ardy "$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt" original'
wave \
  'run_long 0 d2rl_50k "$FINAL" hybrid 9 in_horizon' \
  'run_long 1 native_hybrid "$NATIVE" hybrid 9 in_horizon' \
  'run_long 2 minimal_w4 "$MINIMAL" minimal 9 in_horizon' \
  'run_long 3 ardy "$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt" original 9 in_horizon'
wave \
  'run_long 0 d2rl_50k "$FINAL" hybrid 20 in_horizon' \
  'run_long 1 native_hybrid "$NATIVE" hybrid 20 in_horizon' \
  'run_long 2 minimal_w4 "$MINIMAL" minimal 20 in_horizon' \
  'run_long 3 ardy "$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt" original 20 in_horizon' \
  'run_long 4 d2rl_50k "$FINAL" hybrid 40 in_horizon' \
  'run_long 5 native_hybrid "$NATIVE" hybrid 40 in_horizon' \
  'run_long 6 minimal_w4 "$MINIMAL" minimal 40 in_horizon' \
  'run_long 7 ardy "$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt" original 40 in_horizon'
wave \
  'run_long 0 d2rl_50k "$FINAL" hybrid 40 in_horizon write_off' \
  'run_long 1 d2rl_50k "$FINAL" hybrid 40 in_horizon readout_off'

date -Is > "$OUT/FINAL_SELECTION_EVALUATION_COMPLETE"
echo "final selection evaluation complete: $OUT"
