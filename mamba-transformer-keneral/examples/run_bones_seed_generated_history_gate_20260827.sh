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
RUN_NAME=${RUN_NAME:-bones_seed_generated_history}
RUN_DATE=${RUN_DATE:-20260827}
TRAIN_ROOT=${TRAIN_ROOT:-$BASE/${RUN_NAME}_ft_${RUN_DATE}}
EVAL_PREFIX=${EVAL_PREFIX:-$BASE/${RUN_NAME}_eval_stage}
REPORT_OUTPUT=${REPORT_OUTPUT:-$ROOT/docs/BONES_SEED_GENERATED_HISTORY_GATE_${RUN_DATE}.md}
HISTORY_SAMPLING_STRATEGY=${HISTORY_SAMPLING_STRATEGY:-legacy_conditional}
SEQUENTIAL_ROLLOUT_CURRICULUM=${SEQUENTIAL_ROLLOUT_CURRICULUM:-generated_gate_v1}
SHORT_WINDOW_PROBABILITY=${SHORT_WINDOW_PROBABILITY:-0.2}
SEQUENTIAL_CACHE_PROBABILITY=${SEQUENTIAL_CACHE_PROBABILITY:-0.8}
GENERATED_PROBABILITY_START=${GENERATED_PROBABILITY_START:-0.0}
GENERATED_PROBABILITY_1K=${GENERATED_PROBABILITY_1K:-0.5}
GENERATED_PROBABILITY_5K=${GENERATED_PROBABILITY_5K:-0.75}
TRAIN_FRAMES=${TRAIN_FRAMES:-260}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
MIN_GENERATED_FRACTION=${MIN_GENERATED_FRACTION:-0.0}
GENERATED_MAX_RECOVERY_METERS=${GENERATED_MAX_RECOVERY_METERS:-0.0}
GENERATED_DEPTH_RAMP_STEPS=${GENERATED_DEPTH_RAMP_STEPS:-0}
GENERATED_MAX_SUPERVISION_POINTS=${GENERATED_MAX_SUPERVISION_POINTS:-1}
GENERATED_BPTT_WINDOWS=${GENERATED_BPTT_WINDOWS:-0}
GENERATED_STATE_CONSISTENCY_WEIGHT=${GENERATED_STATE_CONSISTENCY_WEIGHT:-0}
GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT=${GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT:-0}
MAMBA_OUTPUT_DECAY_START_WINDOWS=${MAMBA_OUTPUT_DECAY_START_WINDOWS:-0}
MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS=${MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS:-0}
MAMBA_OUTPUT_DECAY_MIN_SCALE=${MAMBA_OUTPUT_DECAY_MIN_SCALE:-0}
TRAIN=$BASE/bones_seed_ardy_train
VAL=$BASE/bones_seed_ardy_val
TEST=$BASE/bones_seed_ardy_test/manifest.jsonl
TEXT=$BASE/bones_seed_text_cache_llm2vec

if ! nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
  echo "GPU preflight failed: NVIDIA devices are unavailable" >&2
  exit 3
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -lt 8 ]]; then
  echo "GPU preflight failed: eight GPUs are required for the matched run" >&2
  exit 3
fi
mkdir -p "$TRAIN_ROOT"

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

validate_training() {
  local mode=$1 stage=$2 output=$TRAIN_ROOT/$mode
  local checkpoint=$output/checkpoint_step_$(printf '%06d' "$stage").pt
  [[ -f $checkpoint ]] || checkpoint=$output/designated_stage_$(printf '%06d' "$stage").pt
  "$PY" - "$output/metrics_step_$(printf '%06d' "$stage").json" "$checkpoint" "$mode" "$stage" \
    "$HISTORY_SAMPLING_STRATEGY" "$SEQUENTIAL_ROLLOUT_CURRICULUM" \
    "$MIN_GENERATED_FRACTION" "$GENERATED_MAX_SUPERVISION_POINTS" \
    "$GENERATED_BPTT_WINDOWS" "$GENERATED_STATE_CONSISTENCY_WEIGHT" <<'PY'
import json, math, sys
from pathlib import Path
metrics_path, checkpoint, mode, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
strategy, curriculum, min_generated_fraction = sys.argv[5], sys.argv[6], float(sys.argv[7])
max_supervision_points = int(sys.argv[8])
bptt_windows = int(sys.argv[9])
state_consistency_weight = float(sys.argv[10])
if not metrics_path.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise SystemExit(1)
m = json.loads(metrics_path.read_text())
counts = {int(k): int(v) for k, v in m["rollout_window_counts"].items()}
supervision = m.get("generated_history_supervision_audit", {})
depth_counts = {
    int(k): int(v) for k, v in supervision.get("depth_counts", {}).items()
}
max_windows = 19 if curriculum in {"generated_deep_v3", "generated_multidepth_v4"} else 5
audit_valid = not supervision and max_supervision_points == 1
audit_valid = audit_valid or (
    supervision.get("target_count", 0) == sum(depth_counts.values())
    and m.get("generated_history_steps", 0)
        <= supervision.get("target_count", 0)
        <= m.get("generated_history_steps", 0) * max_supervision_points
    and all(1 <= depth <= max_windows - 2 for depth in depth_counts)
)
valid = (
    m["mode"] == mode and m["steps"] == stage and m["world_size"] == 8
    and m["global_batch_size"] == 16 and m["train_scope"] == "denoiser"
    and m["sequential_rollout_curriculum"] == curriculum
    and m.get("history_sampling_strategy", "legacy_conditional") == strategy
    and m.get("generated_history_max_supervision_points", 1) == max_supervision_points
    and m.get("generated_history_bptt_windows", 0) == bptt_windows
    and m.get("generated_history_state_consistency_weight_requested", 0.0)
        == state_consistency_weight
    and m["all_gradients_finite"] is True and math.isfinite(m["final_eval_loss"])
    and m["root_output_weight_norm"] >= 2.0
    and set(counts).issubset(set(range(2, max_windows + 1)))
    and sum(counts.values()) == m["sequential_rollout_steps"]
    and m.get("generated_history_steps", 0) + m.get("teacher_history_steps", 0)
        == m["sequential_rollout_steps"]
    and m.get("short_window_steps", stage - m["sequential_rollout_steps"])
        + m["sequential_rollout_steps"] == stage
    and audit_valid
)
valid = valid and m["generated_history_steps"] >= int(min_generated_fraction * stage)
raise SystemExit(0 if valid else 1)
PY
}

run_training() {
  local mode=$1 stage=$2 output=$TRAIN_ROOT/$mode eval_protocol checkpoint
  mkdir -p "$output"
  if validate_training "$mode" "$stage"; then
    echo "skip_completed_training mode=$mode stage=$stage"
    return
  fi
  local init_args=()
  if [[ $stage -gt 1000 ]]; then
    checkpoint=$(find "$output" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' -print | sort | tail -n 1)
    [[ -n $checkpoint ]] || checkpoint=$output/designated_stage_001000.pt
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
    --output-dir "$output" --steps "$stage" --batch-size 2 --learning-rate "$LEARNING_RATE" \
    --train-scope denoiser --seed 827321 --frames "$TRAIN_FRAMES" --history-probability 0 \
    --constraint-dropout 0.1 --text-dropout 0.1 \
    --sequential-cache-probability "$SEQUENTIAL_CACHE_PROBABILITY" \
    --sequential-rollout-curriculum "$SEQUENTIAL_ROLLOUT_CURRICULUM" \
    --history-sampling-strategy "$HISTORY_SAMPLING_STRATEGY" \
    --short-window-probability "$SHORT_WINDOW_PROBABILITY" \
    --generated-history-probability-start "$GENERATED_PROBABILITY_START" \
    --generated-history-probability-1k "$GENERATED_PROBABILITY_1K" \
    --generated-history-probability-5k "$GENERATED_PROBABILITY_5K" \
    --generated-history-diffusion-steps 10 \
    --generated-history-cfg-weight-text 2 --generated-history-cfg-weight-constraint 2 \
    --generated-history-max-recovery-meters "$GENERATED_MAX_RECOVERY_METERS" \
    --generated-history-depth-ramp-steps "$GENERATED_DEPTH_RAMP_STEPS" \
    --generated-history-max-supervision-points "$GENERATED_MAX_SUPERVISION_POINTS" \
    --generated-history-bptt-windows "$GENERATED_BPTT_WINDOWS" \
    --generated-history-state-consistency-weight "$GENERATED_STATE_CONSISTENCY_WEIGHT" \
    --generated-history-root-endpoint-loss-weight "$GENERATED_ROOT_ENDPOINT_LOSS_WEIGHT" \
    --mamba-output-decay-start-windows "$MAMBA_OUTPUT_DECAY_START_WINDOWS" \
    --mamba-output-decay-half-life-windows "$MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS" \
    --mamba-output-decay-min-scale "$MAMBA_OUTPUT_DECAY_MIN_SCALE" \
    --eval-protocol "$eval_protocol" --eval-batches 1 \
    --boundary-loss-weight 10 --root-velocity-loss-weight 50 \
    --constraint-loss-weight 1 --fk-constraint-loss-weight 1 --foot-skate-loss-weight 50 \
    --eval-every 0 --save-every 100 --keep-last-checkpoints 5 \
    "${init_args[@]}" >>"$output/train_stage${stage}.stdout.log" 2>&1
  cp "$output/metrics.json" "$output/metrics_step_$(printf '%06d' "$stage").json"
  validate_training "$mode" "$stage"
  # Preserve each designated gate checkpoint while allowing intermediate
  # resume checkpoints to be pruned on the 503 GiB free result volume.
  ln -f "$output/checkpoint_step_$(printf '%06d' "$stage").pt" \
    "$output/designated_stage_$(printf '%06d' "$stage").pt"
  echo "training_complete mode=$mode stage=$stage $(date -Is)"
}

result_pass() {
  local path=$1 expected=$2
  [[ -f $path ]] || return 1
  "$PY" - "$path" "$expected" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if r.get("status") == "PASS" and r.get("samples") == int(sys.argv[2]) else 1)
PY
}

run_table4() {
  local gpu=$1 mode=$2 stage=$3 out=$4 checkpoint stem
  stem=${mode}_table4
  checkpoint=$TRAIN_ROOT/$mode/checkpoint_step_$(printf '%06d' "$stage").pt
  result_pass "$out/$stem.json" 64 && [[ -f $out/$stem.npz ]] && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol table4 \
    --samples 64 --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
    --cfg-weight-constraint 2 --seed 20260824 --device cuda:0 --output "$out/$stem.json" \
    --artifacts-output "$out/$stem.npz" >"$out/$stem.log" 2>&1
}

run_duration() {
  local gpu=$1 mode=$2 stage=$3 out=$4 seconds=$5 protocol=$6 samples=$7 selection=$8
  local checkpoint=$TRAIN_ROOT/$mode/checkpoint_step_$(printf '%06d' "$stage").pt
  local stem=${mode}_${seconds}s_${protocol} diag=() decay=()
  if result_pass "$out/$stem.json" "$samples" && [[ -f $out/$stem.npz ]]; then
    if [[ $mode != hybrid || $seconds != 40 || $protocol != in_horizon || -f $out/$stem.state.jsonl ]]; then
      return
    fi
  fi
  if [[ $mode == hybrid && $seconds == 40 && $protocol == in_horizon ]]; then
    diag=(--state-diagnostics-output "$out/$stem.state.jsonl")
  fi
  if [[ $mode == hybrid ]]; then
    decay=(
      --mamba-output-decay-start-windows "$MAMBA_OUTPUT_DECAY_START_WINDOWS"
      --mamba-output-decay-half-life-windows "$MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS"
      --mamba-output-decay-min-scale "$MAMBA_OUTPUT_DECAY_MIN_SCALE"
    )
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol "$protocol" \
    --total-seconds "$seconds" --selection-min-seconds "$selection" --samples "$samples" \
    --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260824 --device cuda:0 --output "$out/$stem.json" \
    --artifacts-output "$out/$stem.npz" "${decay[@]}" "${diag[@]}" >"$out/$stem.log" 2>&1
}

run_evaluation() {
  local stage=$1 out
  out=${EVAL_PREFIX}${stage}_${RUN_DATE}
  mkdir -p "$out"
  for mode in original minimal hybrid; do
    local pids=()
    run_table4 0 "$mode" "$stage" "$out" & pids+=("$!")
    run_duration 1 "$mode" "$stage" "$out" 9 in_horizon 64 9 & pids+=("$!")
    run_duration 2 "$mode" "$stage" "$out" 9 out_of_horizon 64 9 & pids+=("$!")
    run_duration 3 "$mode" "$stage" "$out" 20 in_horizon 20 40 & pids+=("$!")
    run_duration 4 "$mode" "$stage" "$out" 20 out_of_horizon 20 40 & pids+=("$!")
    run_duration 6 "$mode" "$stage" "$out" 40 in_horizon 20 40 & pids+=("$!")
    run_duration 7 "$mode" "$stage" "$out" 40 out_of_horizon 20 40 & pids+=("$!")
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
  date -Is >"$out/EVALUATION_COMPLETE"
}

run_reset_ablations() {
  local stage=$1 out checkpoint pids=()
  out=${EVAL_PREFIX}${stage}_${RUN_DATE}
  checkpoint=$TRAIN_ROOT/hybrid/checkpoint_step_$(printf '%06d' "$stage").pt
  for spec in 2:5 4:6 8:7; do
    local reset gpu stem
    reset=${spec%%:*}
    gpu=${spec##*:}
    stem=hybrid_40s_in_horizon_reset${reset}
    if result_pass "$out/$stem.json" 20 && [[ -f $out/$stem.state.jsonl ]]; then continue; fi
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
      --dataset-name BONES-SEED-v004 --model g1 --mode hybrid --checkpoint "$checkpoint" \
      --manifest "$TEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol in_horizon \
      --total-seconds 40 --selection-min-seconds 40 --samples 20 --batch-size 2 \
      --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260824 \
      --periodic-state-reset-windows "$reset" --state-diagnostics-output "$out/$stem.state.jsonl" \
      --device cuda:0 --output "$out/$stem.json" --artifacts-output "$out/$stem.npz" \
      >"$out/$stem.log" 2>&1 & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

write_report() {
  "$PY" "$ROOT/examples/write_bones_seed_generated_history_report.py" \
    --train-root "$TRAIN_ROOT" --eval-prefix "$EVAL_PREFIX" \
    --eval-suffix "_$RUN_DATE" --output "$REPORT_OUTPUT"
}

for mode in original minimal hybrid; do run_training "$mode" 1000; done
run_evaluation 1000
run_reset_ablations 1000
GATE=${EVAL_PREFIX}1000_${RUN_DATE}/gate.json
if ! "$PY" "$ROOT/examples/check_bones_seed_generated_history_gate.py" \
  --eval-root "${EVAL_PREFIX}1000_${RUN_DATE}" \
  --training-root "$TRAIN_ROOT" --stage 1000 --output "$GATE"; then
  write_report
  date -Is >"$TRAIN_ROOT/PIPELINE_STOPPED_AT_1K_GATE"
  echo "1k gate failed; 5k was not started"
  exit 0
fi

for mode in original minimal hybrid; do run_training "$mode" 5000; done
run_evaluation 5000
run_reset_ablations 5000
"$PY" "$ROOT/examples/check_bones_seed_generated_history_gate.py" \
  --eval-root "${EVAL_PREFIX}5000_${RUN_DATE}" \
  --training-root "$TRAIN_ROOT" --stage 5000 \
  --output "${EVAL_PREFIX}5000_${RUN_DATE}/gate.json" || true
write_report
date -Is >"$TRAIN_ROOT/PIPELINE_COMPLETE"
echo "generated-history pipeline complete"
