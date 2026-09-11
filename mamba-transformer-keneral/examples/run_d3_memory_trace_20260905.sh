#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral; PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result; OUT=$BASE/stage_d3_memory_specificity_20260905/memory_trace
CKPT=$BASE/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl; STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec; PANEL=$BASE/bones_seed_long_memory_panel_v2_20260831/selection/40s_in_horizon.jsonl
LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json
export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}
for spec in 0:normal 1:permute_memory; do
 gpu=${spec%%:*}; intervention=${spec##*:}; dir=$OUT/$intervention; mkdir -p "$dir"
 (CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
  --mode hybrid --model g1 --dataset-name BONES-SEED-v004 --checkpoint "$CKPT" --manifest "$MANIFEST" \
  --stats-dir "$STATS" --text-cache "$TEXT" --protocol in_horizon --total-seconds 40 --selection-min-seconds 40 \
  --samples 16 --batch-size 2 --diffusion-steps 10 --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260831 \
  --device cuda:0 --panel "$PANEL" --protocol-lock "$LOCK" --attention-window-chunks 4 \
  --content-memory-intervention "$intervention" --output "$dir/result.json" --per-sample-output "$dir/per_sample.jsonl" \
  --state-diagnostics-output "$dir/state.jsonl" --content-memory-tensor-trace-output "$dir/memory_trace.pt" > "$dir/eval.log" 2>&1) &
done
wait
