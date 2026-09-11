#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUT=$BASE/stage_d2rl_frozen_5k_final_eval_20260906
PANEL=$BASE/bones_seed_long_memory_panel_v2_20260831/selection
LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec

export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/MotionMamba2.0/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

declare -A CHECKPOINT=(
  [d2rl_5k]=$BASE/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt
  [native_hybrid]=$BASE/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt
  [minimal_w4]=$BASE/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt
  [ardy]=$BASE/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt
)
declare -A MODE=(
  [d2rl_5k]=hybrid
  [native_hybrid]=hybrid
  [minimal_w4]=minimal
  [ardy]=original
)

run_one() {
  local gpu=$1 name=$2
  local dir=$OUT/$name/20s_out_of_horizon
  mkdir -p "$dir"
  if [[ -s "$dir/result.json" ]] && grep -q '"status": "PASS"' "$dir/result.json"; then
    echo "skip_completed $name"
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --mode "${MODE[$name]}" --model g1 --dataset-name BONES-SEED-v004 \
    --checkpoint "${CHECKPOINT[$name]}" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$TEXT" --protocol out_of_horizon --total-seconds 20 \
    --selection-min-seconds 40 --samples 16 --batch-size 2 --diffusion-steps 10 \
    --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260831 --device cuda:0 \
    --panel "$PANEL/20s_out_of_horizon.jsonl" --protocol-lock "$LOCK" \
    --attention-window-chunks 4 --output "$dir/result.json" \
    --per-sample-output "$dir/per_sample.jsonl" --state-diagnostics-output "$dir/state.jsonl" \
    >"$dir/eval.log" 2>&1
}

mkdir -p "$OUT"
run_one 0 d2rl_5k & p0=$!
run_one 1 native_hybrid & p1=$!
run_one 2 minimal_w4 & p2=$!
run_one 3 ardy & p3=$!
status=0
for p in "$p0" "$p1" "$p2" "$p3"; do wait "$p" || status=1; done
(( status == 0 ))
date -Is > "$OUT/FOUR_MODEL_20S_OUT_OF_HORIZON_COMPLETE"
echo "four_model_20s_out_of_horizon=PASS"
