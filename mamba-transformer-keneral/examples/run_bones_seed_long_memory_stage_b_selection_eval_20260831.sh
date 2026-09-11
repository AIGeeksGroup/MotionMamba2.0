#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
TRAIN_OUT=$BASE/bones_seed_long_memory_stage_b_20260831
DECAY_OUT=$BASE/bones_seed_long_memory_stage_b_decay_20260831
OUT=$BASE/bones_seed_long_memory_stage_b_selection_eval_20260831
PANELS=$BASE/bones_seed_long_memory_panel_v2_20260831/selection
LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
ARDY_INIT=$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt
MIN_INIT=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt
HYB_INIT=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt

export HF_HOME=/data3/zts/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

checkpoint_for() {
  local stage=$1 name=$2
  if (( stage == 0 )); then
    case "$name" in
      ardy) echo "$ARDY_INIT" ;;
      minimal_*) echo "$MIN_INIT" ;;
      hybrid_*) echo "$HYB_INIT" ;;
    esac
    return
  fi
  case "$name" in
    ardy|minimal_*) echo "$TRAIN_OUT/$name/checkpoint_step_$(printf '%06d' "$stage").pt" ;;
    hybrid_decay_*) echo "$DECAY_OUT/$name/checkpoint_step_$(printf '%06d' "$stage").pt" ;;
    hybrid_no_decay_carry) echo "$TRAIN_OUT/hybrid_carry/checkpoint_step_$(printf '%06d' "$stage").pt" ;;
    hybrid_no_decay_no_carry) echo "$TRAIN_OUT/hybrid_no_carry/checkpoint_step_$(printf '%06d' "$stage").pt" ;;
  esac
}

mode_for() {
  case "$1" in
    ardy) echo original ;;
    minimal_*) echo minimal ;;
    hybrid_*) echo hybrid ;;
  esac
}

extra_for() {
  case "$1" in
    minimal_w4) echo "--attention-window-chunks 4" ;;
    minimal_w8) echo "--attention-window-chunks 8" ;;
    minimal_w16) echo "--attention-window-chunks 16" ;;
    minimal_parammatch) echo "--attention-window-chunks 4 --minimal-ff-size 5248" ;;
    hybrid_decay_carry) echo "--attention-window-chunks 4 --mamba-output-decay-start-windows 4 --mamba-output-decay-half-life-windows 4 --mamba-output-decay-min-scale 0 --state-intervention carry" ;;
    hybrid_decay_no_carry) echo "--attention-window-chunks 4 --mamba-output-decay-start-windows 4 --mamba-output-decay-half-life-windows 4 --mamba-output-decay-min-scale 0 --state-intervention fresh_each_window" ;;
    hybrid_no_decay_carry) echo "--attention-window-chunks 4 --state-intervention carry" ;;
    hybrid_no_decay_no_carry) echo "--attention-window-chunks 4 --state-intervention fresh_each_window" ;;
    *) echo "" ;;
  esac
}

table_extra_for() {
  case "$1" in
    minimal_w8) echo "--attention-window-chunks 8" ;;
    minimal_w16) echo "--attention-window-chunks 16" ;;
    minimal_parammatch) echo "--attention-window-chunks 4 --minimal-ff-size 5248" ;;
    *) echo "--attention-window-chunks 4" ;;
  esac
}

run_table4() {
  local gpu=$1 stage=$2 name=$3
  local dir=$OUT/stage$(printf '%03d' "$stage")/$name/table4
  [[ -f $dir/result.json ]] && return
  mkdir -p "$dir"
  local checkpoint mode
  checkpoint=$(checkpoint_for "$stage" "$name")
  mode=$(mode_for "$name")
  read -r -a extra_args <<<"$(table_extra_for "$name")"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --mode "$mode" --model g1 --dataset-name BONES-SEED-v004 \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$TEXT" --protocol table4 --samples 64 --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260831 --device cuda:0 --panel "$PANELS/table4.jsonl" \
    --protocol-lock "$LOCK" --output "$dir/result.json" \
    --per-sample-output "$dir/per_sample.jsonl" --artifacts-output "$dir/artifacts.npz" \
    "${extra_args[@]}" >"$dir/eval.log" 2>&1
}

run_long() {
  local gpu=$1 stage=$2 name=$3 seconds=$4 intervention=${5:-}
  local suffix=$name
  [[ -n $intervention ]] && suffix=hybrid_decay_carry__"$intervention"
  local dir=$OUT/stage$(printf '%03d' "$stage")/$suffix/${seconds}s_in_horizon
  [[ -f $dir/result.json ]] && return
  mkdir -p "$dir"
  local checkpoint mode selection
  checkpoint=$(checkpoint_for "$stage" "$name")
  mode=$(mode_for "$name")
  selection=$seconds
  (( seconds == 20 )) && selection=40
  read -r -a extra_args <<<"$(extra_for "$name")"
  if [[ -n $intervention ]]; then
    extra_args=(--attention-window-chunks 4 --mamba-output-decay-start-windows 4 \
      --mamba-output-decay-half-life-windows 4 --mamba-output-decay-min-scale 0 \
      --state-intervention "$intervention")
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --mode "$mode" --model g1 --dataset-name BONES-SEED-v004 \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$TEXT" --protocol in_horizon --total-seconds "$seconds" \
    --selection-min-seconds "$selection" --samples "$([[ $seconds == 9 ]] && echo 64 || echo 16)" \
    --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260831 --device cuda:0 --panel "$PANELS/${seconds}s_in_horizon.jsonl" \
    --protocol-lock "$LOCK" --output "$dir/result.json" \
    --per-sample-output "$dir/per_sample.jsonl" --artifacts-output "$dir/artifacts.npz" \
    --state-diagnostics-output "$dir/state.jsonl" \
    "${extra_args[@]}" >"$dir/eval.log" 2>&1
}

run_primary_stage() {
  local stage=$1
  local names=(ardy minimal_w4 minimal_w8 minimal_w16 minimal_parammatch \
    hybrid_decay_carry hybrid_decay_no_carry \
    hybrid_no_decay_carry hybrid_no_decay_no_carry)
  local pids=() index=0 status=0
  for name in "${names[@]}"; do
    gpu=$((3 + index % 5))
    (
      run_table4 "$gpu" "$stage" "$name"
      run_long "$gpu" "$stage" "$name" 9
      if (( stage > 0 )); then
        run_long "$gpu" "$stage" "$name" 20
        run_long "$gpu" "$stage" "$name" 40
      fi
    ) & pids+=("$!")
    index=$((index + 1))
    if (( ${#pids[@]} == 5 )); then
      for pid in "${pids[@]}"; do wait "$pid" || status=1; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  (( status == 0 ))
}

run_interventions() {
  local stage=$1
  local interventions=(fresh_each_window zero_after_w4 freeze_after_w4 permute_after_w4 readout_off_after_w4 no_decay)
  local pids=() index=0 status=0
  for intervention in "${interventions[@]}"; do
    gpu=$((3 + index % 5))
    (
      run_long "$gpu" "$stage" hybrid_decay_carry 20 "$intervention"
      run_long "$gpu" "$stage" hybrid_decay_carry 40 "$intervention"
    ) & pids+=("$!")
    index=$((index + 1))
    if (( ${#pids[@]} == 5 )); then
      for pid in "${pids[@]}"; do wait "$pid" || status=1; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  (( status == 0 ))
}

run_primary_stage 0
for stage in 100 250; do
  run_primary_stage "$stage"
  run_interventions "$stage"
done

date -Is >"$OUT/SELECTION_EVALUATION_COMPLETE"
"$PY" "$ROOT/examples/check_bones_seed_long_memory_stage_b_gate.py" \
  --evaluation-root "$OUT" --training-root "$DECAY_OUT" \
  --output "$OUT/gate_summary.json" >"$OUT/gate.log" 2>&1
date -Is >"$OUT/STAGE_B_GATE_COMPLETE"
