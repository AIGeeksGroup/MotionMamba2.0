#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
OUT=/data4/zts/result/humanml3d_contractfix_dual_eval_20260823
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
MANIFEST=$BASE/humanml3d_ardy_test/manifest.jsonl
STATS=$BASE/humanml3d_ardy_train
TEXT=/data3/zts/datasets/HumanML3D/HumanML3D/texts
CACHE=$BASE/humanml3d_text_cache_llm2vec_test
mkdir -p "$OUT"

run_one() {
  local gpu=$1 mode=$2 ckpt=$3 protocol=$4
  local stem="${mode}_${protocol}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --mode "$mode" \
    --checkpoint "$ckpt" \
    --manifest "$MANIFEST" \
    --stats-dir "$STATS" \
    --text-root "$TEXT" \
    --text-cache "$CACHE" \
    --protocol "$protocol" \
    --samples 64 \
    --batch-size 4 \
    --diffusion-steps 10 \
    --cfg-weight-text 2 \
    --cfg-weight-constraint 2 \
    --seed 20260818 \
    --device cuda:0 \
    --output "$OUT/${stem}.json" \
    --artifacts-output "$OUT/${stem}.npz" \
    >"$OUT/${stem}.log" 2>&1
}

ARDY=$BASE/humanml3d_contractfix_full50k_20260823_original/original_humanml3d_ddp.pt
MINIMAL=$BASE/humanml3d_contractfix_full50k_20260823_minimal/minimal_humanml3d_ddp.pt
HYBRID=$BASE/humanml3d_contractfix_full50k_20260823_hybrid/hybrid_humanml3d_ddp.pt

# GPU 5 is reserved by an unrelated process. Run seven protocols first.
run_one 0 original "$ARDY" table4 &
run_one 1 minimal "$MINIMAL" table4 &
run_one 2 hybrid "$HYBRID" table4 &
run_one 3 original "$ARDY" table5_in &
run_one 4 minimal "$MINIMAL" table5_in &
run_one 6 hybrid "$HYBRID" table5_in &
run_one 7 original "$ARDY" table5_out &
wait

run_one 0 minimal "$MINIMAL" table5_out &
run_one 1 hybrid "$HYBRID" table5_out &
wait
echo "completed $(date -Is)"
