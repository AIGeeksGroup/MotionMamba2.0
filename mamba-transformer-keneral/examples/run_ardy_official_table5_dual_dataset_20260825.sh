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

run_one() {
  local gpu=$1 dataset=$2 model=$3 mode=$4 checkpoint=$5 manifest=$6 stats=$7 cache=$8 text_root=$9 protocol=${10} seed=${11} out=${12}
  local stem="${mode}_${protocol}"
  local text_args=()
  [[ -z "$text_root" ]] || text_args=(--text-root "$text_root")
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --dataset-name "$dataset" --model "$model" --mode "$mode" \
    --checkpoint "$checkpoint" --manifest "$manifest" --stats-dir "$stats" \
    --text-cache "$cache" "${text_args[@]}" --protocol "$protocol" \
    --samples 64 --batch-size 4 --diffusion-steps 10 \
    --cfg-weight-text 2 --cfg-weight-constraint 2 --seed "$seed" --device cuda:0 \
    --output "$out/${stem}.json" --artifacts-output "$out/${stem}.npz" \
    >"$out/${stem}.log" 2>&1
}

launch_dataset() {
  local dataset=$1 model=$2 tag=$3 manifest=$4 stats=$5 cache=$6 text_root=$7 seed=$8 out=$9
  local original=$BASE/${tag}_original/original_humanml3d_ddp.pt
  local minimal=$BASE/${tag}_minimal/minimal_humanml3d_ddp.pt
  local hybrid=$BASE/${tag}_hybrid/hybrid_humanml3d_ddp.pt
  local pids=()
  local failed=0
  mkdir -p "$out"
  run_one 0 "$dataset" "$model" original "$original" "$manifest" "$stats" "$cache" "$text_root" in_horizon "$seed" "$out" &
  pids+=("$!")
  run_one 1 "$dataset" "$model" minimal "$minimal" "$manifest" "$stats" "$cache" "$text_root" in_horizon "$seed" "$out" &
  pids+=("$!")
  run_one 2 "$dataset" "$model" hybrid "$hybrid" "$manifest" "$stats" "$cache" "$text_root" in_horizon "$seed" "$out" &
  pids+=("$!")
  run_one 3 "$dataset" "$model" original "$original" "$manifest" "$stats" "$cache" "$text_root" out_of_horizon "$seed" "$out" &
  pids+=("$!")
  run_one 4 "$dataset" "$model" minimal "$minimal" "$manifest" "$stats" "$cache" "$text_root" out_of_horizon "$seed" "$out" &
  pids+=("$!")
  run_one 6 "$dataset" "$model" hybrid "$hybrid" "$manifest" "$stats" "$cache" "$text_root" out_of_horizon "$seed" "$out" &
  pids+=("$!")
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "dataset evaluation failed: $dataset" >&2
    return 1
  fi
}

# Run the six HumanML3D jobs first, then the six BONES-SEED jobs. This keeps
# every job isolated to one GPU while leaving GPU 5's unrelated process alone.
launch_dataset \
  HumanML3D core humanml3d_contractfix_full50k_20260823 \
  "$BASE/humanml3d_ardy_test/manifest.jsonl" "$BASE/humanml3d_ardy_train" \
  "$BASE/humanml3d_text_cache_llm2vec_test" \
  /data3/zts/datasets/HumanML3D/HumanML3D/texts 20260818 \
  "$BASE/humanml3d_contractfix_official_table5_20260825"

launch_dataset \
  BONES-SEED-v004 g1 bones_seed_contract_full50k_20260824 \
  "$BASE/bones_seed_ardy_test/manifest.jsonl" "$BASE/bones_seed_ardy_train" \
  "$BASE/bones_seed_text_cache_llm2vec" "" 20260824 \
  "$BASE/bones_seed_contractfix_official_table5_20260825"

echo "completed $(date -Is)"
