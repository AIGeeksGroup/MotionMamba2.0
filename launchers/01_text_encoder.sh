#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ ! -x "$ARDY_ENV/bin/python" ]]; then
  echo '[ERROR] Run bash launchers/install.sh first.' >&2
  exit 1
fi
if [ ! -f "$ARDY_ROOT/scripts/run_text_encoder_server.py" ]; then
  echo "[ERROR] 缺少 $ARDY_ROOT/scripts/run_text_encoder_server.py" >&2
  exit 1
fi

cd "$ARDY_ROOT"
mkdir -p "$LOG_ROOT"

echo "[TEXT] physical GPU=$GPU_TEXT -> process logical cuda:0"
echo "[TEXT] URL=http://127.0.0.1:${TEXT_ENCODER_PORT}/"

exec env \
  CUDA_VISIBLE_DEVICES="$GPU_TEXT" \
  TEXT_ENCODER_DEVICE=cuda:0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  python scripts/run_text_encoder_server.py \
    --host 127.0.0.1 \
    --port "$TEXT_ENCODER_PORT" \
    --device cuda:0
