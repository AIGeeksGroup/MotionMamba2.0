#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
BASE=/data4/zts/result
OUT=${OUT:-$BASE/bones_seed_long_memory_panel_v1_20260831}

"$PY" "$ROOT/examples/build_bones_seed_long_memory_panel.py" \
  --reference "$BASE/bones_seed_generated_history_v2_earlystop900_eval_20260828" \
  --manifest "$BASE/bones_seed_ardy_test/manifest.jsonl" \
  --text-cache "$BASE/bones_seed_text_cache_llm2vec" \
  --output-dir "$OUT" \
  --lock-output "$OUT/protocol_lock.json" \
  --seed 20260824

date -Is >"$OUT/PANEL_BUILD_COMPLETE"
