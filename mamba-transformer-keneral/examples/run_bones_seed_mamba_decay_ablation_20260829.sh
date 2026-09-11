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
CHECKPOINT=${CHECKPOINT:-$BASE/bones_seed_generated_history_bptt_v5_finaltarget_ft_20260829/hybrid/checkpoint_step_000100.pt}
OUT=${OUT:-$BASE/bones_seed_mamba_decay_ablation_v5stage100_20260829}
HALF_LIVES=${HALF_LIVES:-"2 4 8"}
DECAY_MIN_SCALE=${DECAY_MIN_SCALE:-0}
mkdir -p "$OUT"

run_one() {
  local gpu=$1 half_life=$2 seconds=$3
  local stem=hybrid_${seconds}s_decay_h${half_life}
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004-val --model g1 --mode hybrid --checkpoint "$CHECKPOINT" \
    --manifest "$BASE/bones_seed_ardy_val/manifest.jsonl" \
    --stats-dir "$BASE/bones_seed_ardy_train" \
    --text-cache "$BASE/bones_seed_text_cache_llm2vec" --protocol in_horizon \
    --total-seconds "$seconds" --selection-min-seconds 40 --samples 16 --batch-size 2 \
    --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260829 \
    --mamba-output-decay-start-windows 4 \
    --mamba-output-decay-half-life-windows "$half_life" \
    --mamba-output-decay-min-scale "$DECAY_MIN_SCALE" \
    --device cuda:0 --output "$OUT/$stem.json" >"$OUT/$stem.log" 2>&1
}

pids=()
gpu=0
for half_life in $HALF_LIVES; do
  for seconds in 20 40; do
    run_one "$gpu" "$half_life" "$seconds" & pids+=("$!")
    gpu=$((gpu + 1))
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" - "$OUT" "$HALF_LIVES" "$DECAY_MIN_SCALE" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
half_lives = tuple(map(int, sys.argv[2].split()))
min_scale = float(sys.argv[3])
rows = []
for half_life in half_lives:
    row = {"half_life_windows": half_life, "min_scale": min_scale, "durations": {}}
    for seconds in (20, 40):
        result = json.loads((out / f"hybrid_{seconds}s_decay_h{half_life}.json").read_text())
        row["durations"][f"{seconds}s"] = {
            key: result[key] for key in (
                "constraint_error_cm_mean",
                "mobench_root_ade_cm_mean",
                "post_window5_root_ade_slope_cm_per_window",
                "post_window5_constraint_slope_cm_per_window",
            )
        }
    rows.append(row)
(out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
date -Is >"$OUT/ABLATION_COMPLETE"
