#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUT=$BASE/stage_d3_memory_specificity_20260905/checkpoint_sweep
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
PANEL=$BASE/bones_seed_long_memory_panel_v2_20260831/selection/40s_in_horizon.jsonl
LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"
steps=(250 500 1000 2000)
for index in "${!steps[@]}"; do
  step=${steps[$index]}
  ckpt="$BASE/stage_d2rl_bones_long_20260904/step${step}/full/hybrid_humanml3d_ddp.pt"
  for intervention in normal persistent_read_permutation; do
    dir="$OUT/step${step}/${intervention}/40s_in_horizon"
    mkdir -p "$dir"
    gpu=$((index * 2))
    [[ "$intervention" == persistent_read_permutation ]] && gpu=$((gpu + 1))
    (
      CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
        --mode hybrid --model g1 --dataset-name BONES-SEED-v004 --checkpoint "$ckpt" \
        --manifest "$MANIFEST" --stats-dir "$STATS" --text-cache "$TEXT" \
        --protocol in_horizon --total-seconds 40 --selection-min-seconds 40 \
        --samples 16 --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 \
        --cfg-weight-constraint 2 --seed 20260831 --device cuda:0 --panel "$PANEL" \
        --protocol-lock "$LOCK" --attention-window-chunks 4 \
        --content-memory-intervention "$intervention" --output "$dir/result.json" \
        --per-sample-output "$dir/per_sample.jsonl" --state-diagnostics-output "$dir/state.jsonl" \
        > "$dir/eval.log" 2>&1
    ) &
  done
done
wait
