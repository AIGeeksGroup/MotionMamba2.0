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
SOURCE=$BASE/bones_seed_generated_history_v2_ft_20260828
TRAIN_ROOT=$BASE/bones_seed_generated_history_v2_earlystop900_ft_20260828
EVAL_ROOT=$BASE/bones_seed_generated_history_v2_earlystop900_eval_20260828
TRAIN=$BASE/bones_seed_ardy_train
VAL=$BASE/bones_seed_ardy_val
TEST=$BASE/bones_seed_ardy_test/manifest.jsonl
TEXT=$BASE/bones_seed_text_cache_llm2vec
STAGE=900
REPORT=$ROOT/docs/BONES_SEED_GENERATED_HISTORY_V2_EARLYSTOP900_CONFIRMATION_20260828.md

result_pass() {
  local path=$1 expected=$2
  [[ -f $path ]] || return 1
  "$PY" - "$path" "$expected" <<'PY'
import json, sys
from pathlib import Path
row = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if row.get("status") == "PASS" and row.get("samples") == int(sys.argv[2]) else 1)
PY
}

audit_checkpoint() {
  local mode=$1 protocol output=$TRAIN_ROOT/$mode
  local source=$SOURCE/$mode/checkpoint_step_000900.pt
  mkdir -p "$output"
  if [[ -f $output/metrics_step_000900.json && -f $output/checkpoint_step_000900.pt ]]; then
    return
  fi
  [[ $mode == original ]] && protocol=explicit_history || protocol=cache_only
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
    --standalone --nproc-per-node=8 "$ROOT/examples/train_humanml3d_ardy_ddp.py" \
    --model g1 --mode "$mode" --manifest "$TRAIN/manifest.jsonl" \
    --dev-manifest "$VAL/manifest.jsonl" --stats-dir "$TRAIN" --text-cache "$TEXT" \
    --output-dir "$output" --steps "$STAGE" --batch-size 2 --learning-rate 5e-6 \
    --train-scope denoiser --seed 827321 --frames 260 --history-probability 0 \
    --constraint-dropout 0.1 --text-dropout 0.1 --sequential-cache-probability 0 \
    --sequential-rollout-curriculum generated_gate_v1 \
    --history-sampling-strategy balanced_v2 --short-window-probability 0.35 \
    --generated-history-probability-start 0.25 --generated-history-probability-1k 0.75 \
    --generated-history-probability-5k 0.85 --generated-history-diffusion-steps 10 \
    --generated-history-cfg-weight-text 2 --generated-history-cfg-weight-constraint 2 \
    --eval-protocol "$protocol" --eval-batches 1 --boundary-loss-weight 10 \
    --root-velocity-loss-weight 50 --constraint-loss-weight 1 \
    --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 --eval-every 0 \
    --save-every 0 --no-final-resume-checkpoint --resume "$source" \
    >"$output/audit_stage900.log" 2>&1
  cp "$output/metrics.json" "$output/metrics_step_000900.json"
  ln -f "$source" "$output/checkpoint_step_000900.pt"
}

run_table4() {
  local gpu=$1 mode=$2 checkpoint=$TRAIN_ROOT/$mode/checkpoint_step_000900.pt
  local stem=${mode}_table4
  result_pass "$EVAL_ROOT/$stem.json" 64 && [[ -f $EVAL_ROOT/$stem.npz ]] && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol table4 \
    --samples 64 --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
    --cfg-weight-constraint 2 --seed 20260824 --device cuda:0 \
    --output "$EVAL_ROOT/$stem.json" --artifacts-output "$EVAL_ROOT/$stem.npz" \
    >"$EVAL_ROOT/$stem.log" 2>&1
}

run_duration() {
  local gpu=$1 mode=$2 seconds=$3 protocol=$4 samples=$5 selection=$6
  local checkpoint=$TRAIN_ROOT/$mode/checkpoint_step_000900.pt
  local stem=${mode}_${seconds}s_${protocol} diag=()
  if result_pass "$EVAL_ROOT/$stem.json" "$samples" && [[ -f $EVAL_ROOT/$stem.npz ]]; then
    if [[ $mode != hybrid || $seconds != 40 || $protocol != in_horizon || -f $EVAL_ROOT/$stem.state.jsonl ]]; then
      return
    fi
  fi
  if [[ $mode == hybrid && $seconds == 40 && $protocol == in_horizon ]]; then
    diag=(--state-diagnostics-output "$EVAL_ROOT/$stem.state.jsonl")
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol "$protocol" \
    --total-seconds "$seconds" --selection-min-seconds "$selection" --samples "$samples" \
    --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260824 --device cuda:0 --output "$EVAL_ROOT/$stem.json" \
    --artifacts-output "$EVAL_ROOT/$stem.npz" "${diag[@]}" >"$EVAL_ROOT/$stem.log" 2>&1
}

for mode in original minimal hybrid; do audit_checkpoint "$mode"; done
mkdir -p "$EVAL_ROOT"
for mode in original minimal hybrid; do
  pids=()
  run_table4 0 "$mode" & pids+=("$!")
  run_duration 1 "$mode" 9 in_horizon 64 9 & pids+=("$!")
  run_duration 2 "$mode" 9 out_of_horizon 64 9 & pids+=("$!")
  run_duration 3 "$mode" 20 in_horizon 20 40 & pids+=("$!")
  run_duration 4 "$mode" 20 out_of_horizon 20 40 & pids+=("$!")
  run_duration 6 "$mode" 40 in_horizon 20 40 & pids+=("$!")
  run_duration 7 "$mode" 40 out_of_horizon 20 40 & pids+=("$!")
  for pid in "${pids[@]}"; do wait "$pid"; done
done
date -Is >"$EVAL_ROOT/EVALUATION_COMPLETE"

pids=()
for spec in 2:5 4:6 8:7; do
  reset=${spec%%:*}; gpu=${spec##*:}; stem=hybrid_40s_in_horizon_reset${reset}
  if result_pass "$EVAL_ROOT/$stem.json" 20 && [[ -f $EVAL_ROOT/$stem.state.jsonl ]]; then continue; fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode hybrid \
    --checkpoint "$TRAIN_ROOT/hybrid/checkpoint_step_000900.pt" --manifest "$TEST" \
    --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol in_horizon --total-seconds 40 \
    --selection-min-seconds 40 --samples 20 --batch-size 2 --diffusion-steps 10 \
    --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260824 \
    --periodic-state-reset-windows "$reset" \
    --state-diagnostics-output "$EVAL_ROOT/$stem.state.jsonl" --device cuda:0 \
    --output "$EVAL_ROOT/$stem.json" --artifacts-output "$EVAL_ROOT/$stem.npz" \
    >"$EVAL_ROOT/$stem.log" 2>&1 & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" "$ROOT/examples/check_bones_seed_generated_history_gate.py" \
  --eval-root "$EVAL_ROOT" --training-root "$TRAIN_ROOT" --stage "$STAGE" \
  --output "$EVAL_ROOT/gate.json"

"$PY" "$ROOT/examples/write_bones_seed_generated_history_report.py" \
  --train-root "$TRAIN_ROOT" --eval-prefix "$BASE/bones_seed_generated_history_v2_earlystop" \
  --eval-suffix "_eval_20260828" --stages "$STAGE" \
  --gate-budgets-label "Exploratory sweep selected step 900; this run is the full confirmation audit" \
  --selection-note "Step 900 was selected after an exploratory 600/700/800/900/1000 checkpoint sweep. Metrics in this report use the same held-out test protocol, so an additional untouched-seed confirmation is still required before treating the selection as final evidence." \
  --output "$REPORT"

date -Is >"$TRAIN_ROOT/CONFIRMATION_COMPLETE"
