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
SOURCE=${SOURCE:-$BASE/bones_seed_generated_history_multidepth_v4_ablation_ft_20260829}
STAGE=${STAGE:-100}
OUT=${OUT:-$BASE/bones_seed_generated_history_multidepth_v4_ablation_val_stage${STAGE}_20260829}
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
HEALTHY_HYBRID=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
MAMBA_OUTPUT_DECAY_START_WINDOWS=${MAMBA_OUTPUT_DECAY_START_WINDOWS:-0}
MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS=${MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS:-0}
MAMBA_OUTPUT_DECAY_MIN_SCALE=${MAMBA_OUTPUT_DECAY_MIN_SCALE:-0}
mkdir -p "$OUT"

run_long() {
  local gpu=$1 mode=$2 seconds=$3
  local stem=${mode}_${seconds}s checkpoint=$SOURCE/$mode/checkpoint_step_$(printf '%06d' "$STAGE").pt diag=()
  if [[ $mode == hybrid && $seconds == 40 ]]; then
    diag=(--state-diagnostics-output "$OUT/$stem.state.jsonl")
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004-val --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$TEXT" --protocol in_horizon \
    --total-seconds "$seconds" --selection-min-seconds 40 --samples 16 --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260829 \
    --mamba-output-decay-start-windows "$MAMBA_OUTPUT_DECAY_START_WINDOWS" \
    --mamba-output-decay-half-life-windows "$MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS" \
    --mamba-output-decay-min-scale "$MAMBA_OUTPUT_DECAY_MIN_SCALE" \
    --device cuda:0 --output "$OUT/$stem.json" --artifacts-output "$OUT/$stem.npz" \
    "${diag[@]}" \
    >"$OUT/$stem.log" 2>&1
}

run_table4() {
  local gpu=$1 tag=$2 checkpoint=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004-val --model g1 --mode hybrid --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$TEXT" --protocol table4 \
    --samples 64 --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
    --cfg-weight-constraint 2 --seed 20260829 --device cuda:0 \
    --output "$OUT/${tag}_table4.json" >"$OUT/${tag}_table4.log" 2>&1
}

run_9s() {
  local gpu=$1 tag=$2 checkpoint=$3 protocol=$4
  local decay=()
  if [[ $tag == stage ]]; then
    decay=(
      --mamba-output-decay-start-windows "$MAMBA_OUTPUT_DECAY_START_WINDOWS"
      --mamba-output-decay-half-life-windows "$MAMBA_OUTPUT_DECAY_HALF_LIFE_WINDOWS"
      --mamba-output-decay-min-scale "$MAMBA_OUTPUT_DECAY_MIN_SCALE"
    )
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004-val --model g1 --mode hybrid --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$TEXT" --protocol "$protocol" \
    --total-seconds 9 --selection-min-seconds 9 --samples 64 --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260829 \
    "${decay[@]}" \
    --device cuda:0 --output "$OUT/${tag}_9s_${protocol}.json" \
    >"$OUT/${tag}_9s_${protocol}.log" 2>&1
}

pids=()
run_long 0 minimal 20 & pids+=("$!")
run_long 1 minimal 40 & pids+=("$!")
run_long 2 hybrid 20 & pids+=("$!")
run_long 3 hybrid 40 & pids+=("$!")
run_table4 4 healthy "$HEALTHY_HYBRID" & pids+=("$!")
run_table4 5 stage "$SOURCE/hybrid/checkpoint_step_$(printf '%06d' "$STAGE").pt" & pids+=("$!")
(run_9s 6 healthy "$HEALTHY_HYBRID" in_horizon; run_9s 6 stage "$SOURCE/hybrid/checkpoint_step_$(printf '%06d' "$STAGE").pt" in_horizon) & pids+=("$!")
(run_9s 7 healthy "$HEALTHY_HYBRID" out_of_horizon; run_9s 7 stage "$SOURCE/hybrid/checkpoint_step_$(printf '%06d' "$STAGE").pt" out_of_horizon) & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" - "$OUT" "$STAGE" <<'PY'
import json, sys
from pathlib import Path
from examples.check_bones_seed_generated_history_gate import (
    paired_bootstrap_gap, sample_root_slopes, state_audit,
)

out, stage = Path(sys.argv[1]), int(sys.argv[2])
durations = {}
long_pass = True
for seconds in (20, 40):
    minimal = json.loads((out / f"minimal_{seconds}s.json").read_text())
    hybrid = json.loads((out / f"hybrid_{seconds}s.json").read_text())
    gaps = {
        "constraint_mean_gap_cm": hybrid["constraint_error_cm_mean"] - minimal["constraint_error_cm_mean"],
        "root_ade_gap_cm": hybrid["mobench_root_ade_cm_mean"] - minimal["mobench_root_ade_cm_mean"],
        "root_slope_gap_cm_per_window": hybrid["post_window5_root_ade_slope_cm_per_window"] - minimal["post_window5_root_ade_slope_cm_per_window"],
        "constraint_slope_gap_cm_per_window": hybrid["post_window5_constraint_slope_cm_per_window"] - minimal["post_window5_constraint_slope_cm_per_window"],
    }
    paired = paired_bootstrap_gap(
        sample_root_slopes(out / f"hybrid_{seconds}s.npz", int(hybrid["generation_horizon_frames"])),
        sample_root_slopes(out / f"minimal_{seconds}s.npz", int(minimal["generation_horizon_frames"])),
        seed=20260829 + seconds,
    )
    duration_pass = max(gaps.values()) < 0
    long_pass &= duration_pass
    durations[f"{seconds}s"] = {
        "gaps": gaps,
        "paired_root_slope_gap": paired,
        "all_mean_and_slope_gaps_negative": duration_pass,
    }
retention = {}
retention_pass = True
for stem in ("table4", "9s_in_horizon", "9s_out_of_horizon"):
    healthy = json.loads((out / f"healthy_{stem}.json").read_text())
    current = json.loads((out / f"stage_{stem}.json").read_text())
    retention[stem] = {}
    for metric in ("constraint_error_cm_mean", "mobench_root_ade_cm_mean"):
        relative_change = current[metric] / max(healthy[metric], 1e-8) - 1
        metric_pass = relative_change <= 0.10
        retention_pass &= metric_pass
        retention[stem][metric] = {
            "healthy": healthy[metric],
            "current": current[metric],
            "relative_change": relative_change,
            "passed": metric_pass,
        }
state = state_audit(out / "hybrid_40s.state.jsonl")
passed = long_pass and retention_pass and state["all_finite"] and state["no_explosion"]
summary = {
    "status": "PASS" if passed else "FAIL",
    "split": "actor_disjoint_validation",
    "seed": 20260829,
    "stage": stage,
    "samples": 16,
    "criteria": {
        "hybrid_20_40_means_and_slopes_better": long_pass,
        "table4_9s_retention_within_10pct": retention_pass,
        "state_finite": state["all_finite"],
        "no_state_norm_explosion": state["no_explosion"],
    },
    "durations": durations,
    "short_retention": retention,
    "state_audit": state,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
date -Is >"$OUT/EVALUATION_COMPLETE"
