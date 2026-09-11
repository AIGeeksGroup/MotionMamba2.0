#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
OUT=/data4/zts/result/humanml3d_dual_eval_20260822
PY=/data3/zts/conda_envs/ardy_env/bin/python
MANIFEST=/data4/zts/result/humanml3d_ardy_test/manifest.jsonl
STATS=/data4/zts/result/humanml3d_ardy_train
TEXT=/data3/zts/datasets/HumanML3D/HumanML3D/texts
CACHE=/data4/zts/result/humanml3d_text_cache_llm2vec_test
mkdir -p "$OUT"

run_one() {
  local gpu=$1 mode=$2 ckpt=$3 protocol=$4
  local stem="${mode}_${protocol}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --mode "$mode" --checkpoint "$ckpt" --manifest "$MANIFEST" \
    --stats-dir "$STATS" --text-root "$TEXT" --text-cache "$CACHE" \
    --protocol "$protocol" --samples 64 --batch-size 4 --diffusion-steps 10 \
    --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260818 \
    --device cuda:0 --output "$OUT/${stem}.json" \
    --artifacts-output "$OUT/${stem}.npz" >"$OUT/${stem}.log" 2>&1
}

run_one 0 original /data4/zts/result/humanml3d_ardy_full_fresh_baseline_20260821_original/original_humanml3d_ddp.pt table4 &
run_one 1 minimal /data4/zts/result/humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt table4 &
run_one 2 hybrid /data4/zts/result/humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt table4 &
run_one 3 original /data4/zts/result/humanml3d_ardy_full_fresh_baseline_20260821_original/original_humanml3d_ddp.pt table5_in &
run_one 4 minimal /data4/zts/result/humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt table5_in &
run_one 5 hybrid /data4/zts/result/humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt table5_in &
run_one 6 original /data4/zts/result/humanml3d_ardy_full_fresh_baseline_20260821_original/original_humanml3d_ddp.pt table5_out &
run_one 7 minimal /data4/zts/result/humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt table5_out &
run_one 5 hybrid /data4/zts/result/humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt table5_out &
wait
echo "completed $(date -Is)"
