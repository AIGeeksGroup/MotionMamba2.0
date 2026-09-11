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
OUT=$BASE/bones_seed_generated_history_v2_val_long_checkpoint_sweep_20260828
TRAIN=$BASE/bones_seed_ardy_train
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
TEXT=$BASE/bones_seed_text_cache_llm2vec
mkdir -p "$OUT"

run_one() {
  local gpu=$1 stage=$2 mode=$3 seconds=$4
  local stem=stage${stage}_${mode}_${seconds}s checkpoint=$SOURCE/$mode/checkpoint_step_$(printf '%06d' "$stage").pt
  [[ -f $OUT/$stem.json ]] && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004-val --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$TRAIN" --text-cache "$TEXT" --protocol in_horizon \
    --total-seconds "$seconds" --selection-min-seconds 40 --samples 16 --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260829 \
    --device cuda:0 --output "$OUT/$stem.json" >"$OUT/$stem.log" 2>&1
}

tasks=()
for stage in 600 700 800 900 1000; do
  for mode in minimal hybrid; do
    for seconds in 20 40; do tasks+=("$stage $mode $seconds"); done
  done
done
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  (
    for ((index=gpu; index<${#tasks[@]}; index+=8)); do
      read -r stage mode seconds <<<"${tasks[$index]}"
      run_one "$gpu" "$stage" "$mode" "$seconds"
    done
  ) & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for stage in (600, 700, 800, 900, 1000):
    row = {"stage": stage, "durations": {}}
    passed = True
    for seconds in (20, 40):
        minimal = json.loads((out / f"stage{stage}_minimal_{seconds}s.json").read_text())
        hybrid = json.loads((out / f"stage{stage}_hybrid_{seconds}s.json").read_text())
        gaps = {
            "constraint_mean_gap": hybrid["constraint_error_cm_mean"] - minimal["constraint_error_cm_mean"],
            "root_ade_gap": hybrid["mobench_root_ade_cm_mean"] - minimal["mobench_root_ade_cm_mean"],
            "root_slope_gap": hybrid["post_window5_root_ade_slope_cm_per_window"] - minimal["post_window5_root_ade_slope_cm_per_window"],
            "constraint_slope_gap": hybrid["post_window5_constraint_slope_cm_per_window"] - minimal["post_window5_constraint_slope_cm_per_window"],
        }
        passed &= max(gaps.values()) < 0
        row["durations"][f"{seconds}s"] = gaps
    row["all_long_checks_pass"] = passed
    rows.append(row)
(out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
date -Is >"$OUT/SWEEP_COMPLETE"
