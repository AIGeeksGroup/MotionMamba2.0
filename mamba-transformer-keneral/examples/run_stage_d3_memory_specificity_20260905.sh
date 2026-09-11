#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUT=$BASE/stage_d3_memory_specificity_20260905
CKPT=$BASE/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt
MANIFEST=$BASE/bones_seed_ardy_val/manifest.jsonl
STATS=$BASE/bones_seed_ardy_train
TEXT=$BASE/bones_seed_text_cache_llm2vec
PANEL=$BASE/bones_seed_long_memory_panel_v2_20260831/selection/40s_in_horizon.jsonl
LOCK=$ROOT/configs/bones_seed_long_memory_protocol_v2.json

export HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export PYTHONPATH=$ROOT/src:/data3/zts/ardy:/data3/zts/packages/state-spaces-mamba:/data3/zts/packages/mamba_op_triton${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$OUT"
interventions=(normal one_shot_permute_memory persistent_read_permutation freeze_write_after_w4 one_shot_permute_plus_freeze_write semantic_mismatch_persistent_read_permutation write_off readout_off)
for index in "${!interventions[@]}"; do
  intervention=${interventions[$index]}
  dir="$OUT/$intervention/40s_in_horizon"
  mkdir -p "$dir"
  if [[ "$intervention" == one_shot_permute_memory ]]; then
    memory_intervention=permute_memory
  else
    memory_intervention=$intervention
  fi
  (
  CUDA_VISIBLE_DEVICES=$index "$PY" "$ROOT/examples/evaluate_ardy_official_table5.py" \
    --mode hybrid --model g1 --dataset-name BONES-SEED-v004 \
    --checkpoint "$CKPT" --manifest "$MANIFEST" --stats-dir "$STATS" \
    --text-cache "$TEXT" --protocol in_horizon --total-seconds 40 \
    --selection-min-seconds 40 --samples 16 --batch-size 2 --diffusion-steps 10 \
    --cfg-weight-text 2 --cfg-weight-constraint 2 --seed 20260831 --device cuda:0 \
    --panel "$PANEL" --protocol-lock "$LOCK" --attention-window-chunks 4 \
    --content-memory-intervention "$memory_intervention" \
    --output "$dir/result.json" --per-sample-output "$dir/per_sample.jsonl" \
    --state-diagnostics-output "$dir/state.jsonl" > "$dir/eval.log" 2>&1
  ) &
done
wait

"$PY" "$ROOT/examples/analyze_d3_memory_specificity.py" \
  --root "$OUT" --output "$ROOT/docs/D3_PERSISTENT_PERMUTATION_RESULTS.json" \
  --panel-manifest "$ROOT/docs/D3_PANEL_MANIFEST.json"
