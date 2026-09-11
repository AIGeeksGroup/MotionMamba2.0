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
OUT=$BASE/bones_seed_hybrid_multiwindow_final_eval_20260826
CHECKPOINT=$BASE/bones_seed_hybrid_multiwindow_recurrent_ft_20260826/hybrid_humanml3d_ddp.pt
MANIFEST=$BASE/bones_seed_ardy_test/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
CACHE=$BASE/bones_seed_text_cache_llm2vec
mkdir -p "$OUT"

run_table4() {
  CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode hybrid --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$CACHE" --protocol table4 \
    --samples 64 --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
    --cfg-weight-constraint 2 --seed 20260824 --device cuda:0 \
    --output "$OUT/hybrid_multiwindow_table4.json" \
    --artifacts-output "$OUT/hybrid_multiwindow_table4.npz" \
    >"$OUT/hybrid_multiwindow_table4.log" 2>&1
}

run_table5() {
  local gpu=$1 seconds=$2 protocol=$3 samples=$4 selection=$5
  local stem=hybrid_multiwindow_${seconds}s_${protocol}
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode hybrid --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$CACHE" --protocol "$protocol" \
    --total-seconds "$seconds" --selection-min-seconds "$selection" --samples "$samples" \
    --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260824 --device cuda:0 --output "$OUT/$stem.json" \
    --artifacts-output "$OUT/$stem.npz" >"$OUT/$stem.log" 2>&1
}

run_table4 & pids=("$!")
run_table5 1 9 in_horizon 64 9 & pids+=("$!")
run_table5 2 9 out_of_horizon 64 9 & pids+=("$!")
run_table5 3 20 in_horizon 20 40 & pids+=("$!")
run_table5 4 20 out_of_horizon 20 40 & pids+=("$!")
run_table5 6 40 in_horizon 20 40 & pids+=("$!")
run_table5 7 40 out_of_horizon 20 40 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 ))

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(root.glob("*.json"))
assert len(paths) == 7, len(paths)
assert all(json.loads(path.read_text())["status"] == "PASS" for path in paths)
print("multiwindow_final_evaluation=PASS")
PY

date -Is > "$OUT/EVALUATION_COMPLETE"
