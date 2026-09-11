#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
MAN=/data4/zts/result/humanml3d_ardy_test/manifest.jsonl
ST=/data4/zts/result/humanml3d_ardy_train
TXT=/data3/zts/datasets/HumanML3D/HumanML3D/texts
CACHE=/data4/zts/result/humanml3d_text_cache_llm2vec_test
OUT=/data4/zts/result/cachefix_memory_diag_20260822
mkdir -p "$OUT"
run_one() {
  local gpu=$1 mode=$2 name=$3 ckpt=$4
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --mode "$mode" --checkpoint "$ckpt" --manifest "$MAN" --stats-dir "$ST" \
    --text-root "$TXT" --text-cache "$CACHE" --protocol table5_out --samples 64 \
    --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260818 --device cuda:0 --memory-mode explicit_history \
    --output "$OUT/${name}.json" >"$OUT/${name}.log" 2>&1
}
run_one 0 minimal v5r_minimal_explicit /data4/zts/result/humanml3d_ardy_full_v5r_minimal_20260819/minimal_humanml3d_ddp.pt &
run_one 1 hybrid v5r_hybrid_explicit /data4/zts/result/humanml3d_ardy_full_v5r_hybrid_20260819/hybrid_humanml3d_ddp.pt &
run_one 2 minimal cachefix_minimal_explicit /data4/zts/result/humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt &
run_one 3 hybrid cachefix_hybrid_explicit /data4/zts/result/humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt &
wait
