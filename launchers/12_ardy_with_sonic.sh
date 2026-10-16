#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ ! -f "$ARDY_ROOT/scripts/run_demo.py" ]; then
  echo "[ERROR] ARDY demo script is missing." >&2
  exit 1
fi
if ! port_is_listening "$TEXT_ENCODER_PORT"; then
  echo "[ERROR] Text encoder port ${TEXT_ENCODER_PORT} is not ready." >&2
  exit 1
fi
if ! port_is_listening "$SONIC_UI_PORT"; then
  echo "[ERROR] SONIC web bridge port ${SONIC_UI_PORT} is not ready." >&2
  exit 1
fi

cd "$ARDY_ROOT"
echo "[ARDY] GPU ${GPU_ARDY}; SONIC bridge enabled on ${POSE_ZMQ_PORT}"
exec env \
  CUDA_VISIBLE_DEVICES="$GPU_ARDY" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  LOCAL_CACHE=true \
  TEXT_ENCODER_MODE=api \
  TEXT_ENCODER_URL="http://127.0.0.1:${TEXT_ENCODER_PORT}/" \
  ARDY_SONIC_ENABLE=1 \
  ARDY_SONIC_WEB_URL="ws://127.0.0.1:${SONIC_UI_PORT}/ws/producer" \
  ARDY_SONIC_ZMQ_BIND="tcp://127.0.0.1:${POSE_ZMQ_PORT}" \
  ARDY_SONIC_CONTROL_BIND="tcp://127.0.0.1:${BRIDGE_CONTROL_ZMQ_PORT}" \
  ARDY_SONIC_ZMQ_TOPIC=pose \
  ARDY_SONIC_FPS=50 \
  ARDY_SONIC_DEBUG_MODE="${ARDY_SONIC_DEBUG_MODE:-ardy_streaming}" \
  ARDY_SONIC_TIME_STRETCH="${ARDY_SONIC_TIME_STRETCH:-auto}" \
  ARDY_DEFAULT_MODEL=ARDY-G1-RP-25FPS-Horizon52 \
  python scripts/run_demo.py
