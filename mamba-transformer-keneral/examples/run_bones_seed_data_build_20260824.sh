#!/usr/bin/env bash
set -euo pipefail

ROOT=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
PY=/data3/zts/conda_envs/ardy_env/bin/python
DATA=/data4/zts/datasets/bones_seed
OUT=/data4/zts/result

"$PY" "$ROOT/examples/prepare_bones_seed_g1.py" --dataset-root "$DATA"

"$PY" "$ROOT/examples/build_bones_seed_ardy.py" \
  --dataset-root "$DATA" --output "$OUT/bones_seed_ardy_train" \
  --split train --workers 16 --min-frames 52 --stats

"$PY" "$ROOT/examples/build_bones_seed_ardy.py" \
  --dataset-root "$DATA" --output "$OUT/bones_seed_ardy_val" \
  --split val --workers 16 --min-frames 52

"$PY" "$ROOT/examples/build_bones_seed_ardy.py" \
  --dataset-root "$DATA" --output "$OUT/bones_seed_ardy_test" \
  --split test --workers 16 --min-frames 52

HF_HOME=/data3/zts/cache/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$PY" "$ROOT/examples/build_humanml3d_text_cache.py" \
  --manifest "$OUT/bones_seed_ardy_train/manifest.jsonl" \
  --manifest "$OUT/bones_seed_ardy_val/manifest.jsonl" \
  --manifest "$OUT/bones_seed_ardy_test/manifest.jsonl" \
  --output-dir "$OUT/bones_seed_text_cache_llm2vec" \
  --device cuda:0 --encoder-mode local --batch-size 32 --flush-every 1024

echo "bones_seed_data_build_complete $(date -Is)"
