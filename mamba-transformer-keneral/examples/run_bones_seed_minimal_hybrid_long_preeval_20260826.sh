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
MANIFEST=$BASE/bones_seed_ardy_test/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
CACHE=$BASE/bones_seed_text_cache_llm2vec
TABLE4_OUT=$BASE/bones_seed_ardy_retrained_dual_eval_20260826
TABLE5_OUT=$BASE/bones_seed_ardy_retrained_official_table5_20260826
LONG_OUT=$BASE/bones_seed_ardy_retrained_long_table5_20260826
mkdir -p "$TABLE4_OUT" "$TABLE5_OUT" "$LONG_OUT"

checkpoint_for() {
  local mode=$1
  echo "$BASE/bones_seed_contract_full50k_20260824_${mode}/${mode}_humanml3d_ddp.pt"
}

run_table4() {
  local gpu=$1 mode=$2 stem=${2}_table4 checkpoint
  checkpoint=$(checkpoint_for "$mode")
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_humanml3d_table45.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$CACHE" --protocol table4 \
    --samples 64 --batch-size 4 --diffusion-steps 10 --cfg-weight-text 2 \
    --cfg-weight-constraint 2 --seed 20260824 --device cuda:0 \
    --output "$TABLE4_OUT/$stem.json" --artifacts-output "$TABLE4_OUT/$stem.npz" \
    >"$TABLE4_OUT/$stem.log" 2>&1
}

run_long() {
  local gpu=$1 mode=$2 protocol=$3 seconds=$4 samples=$5 selection=$6 output=$7
  local stem=${mode}_${seconds}s_${protocol} checkpoint
  checkpoint=$(checkpoint_for "$mode")
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name BONES-SEED-v004 --model g1 --mode "$mode" --checkpoint "$checkpoint" \
    --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$CACHE" --protocol "$protocol" \
    --total-seconds "$seconds" --selection-min-seconds "$selection" --samples "$samples" \
    --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 \
    --seed 20260824 --device cuda:0 --output "$output/$stem.json" \
    --artifacts-output "$output/$stem.npz" >"$output/$stem.log" 2>&1
}

run_table4 4 minimal & p0=$!
run_table4 6 hybrid & p1=$!
wait "$p0"; wait "$p1"

for protocol in in_horizon out_of_horizon; do
  run_long 4 minimal "$protocol" 9 64 9 "$TABLE5_OUT" & p0=$!
  run_long 6 hybrid "$protocol" 9 64 9 "$TABLE5_OUT" & p1=$!
  wait "$p0"; wait "$p1"
done

for seconds in 20 40; do
  for protocol in in_horizon out_of_horizon; do
    run_long 4 minimal "$protocol" "$seconds" 20 40 "$LONG_OUT" & p0=$!
    run_long 6 hybrid "$protocol" "$seconds" 20 40 "$LONG_OUT" & p1=$!
    wait "$p0"; wait "$p1"
  done
done

date -Is > "$LONG_OUT/MINIMAL_HYBRID_PREEVAL_COMPLETE"
echo "minimal_hybrid_preeval_complete"
