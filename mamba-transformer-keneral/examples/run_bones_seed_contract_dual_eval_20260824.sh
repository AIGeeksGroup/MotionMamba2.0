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
OUT=$BASE/bones_seed_contract_dual_eval_20260824
MANIFEST=$BASE/bones_seed_ardy_test/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
CACHE=$BASE/bones_seed_text_cache_llm2vec
mkdir -p "$OUT"

run_one() {
  local gpu=$1 mode=$2 checkpoint=$3 protocol=$4
  local stem="${mode}_${protocol}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 \
    --model g1 \
    --mode "$mode" \
    --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" \
    --stats-dir "$STATS" \
    --text-cache "$CACHE" \
    --protocol "$protocol" \
    --samples 64 \
    --batch-size 4 \
    --diffusion-steps 10 \
    --cfg-weight-text 2 \
    --cfg-weight-constraint 2 \
    --seed 20260824 \
    --device cuda:0 \
    --output "$OUT/${stem}.json" \
    --artifacts-output "$OUT/${stem}.npz" \
    >"$OUT/${stem}.log" 2>&1
}

ARDY=$BASE/bones_seed_contract_full50k_20260824_original/original_humanml3d_ddp.pt
MINIMAL=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt
HYBRID=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt

run_one 0 original "$ARDY" table4 &
run_one 1 minimal "$MINIMAL" table4 &
run_one 2 hybrid "$HYBRID" table4 &
run_one 3 original "$ARDY" table5_in &
run_one 4 minimal "$MINIMAL" table5_in &
run_one 5 hybrid "$HYBRID" table5_in &
run_one 6 original "$ARDY" table5_out &
run_one 7 minimal "$MINIMAL" table5_out &
wait

run_one 0 hybrid "$HYBRID" table5_out
echo "completed $(date -Is)"
