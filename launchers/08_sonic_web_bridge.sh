#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

WEB_ROOT="$MM2_ROOT/ardy_sonic_web"
if [ ! -f "$WEB_ROOT/backend/app.py" ] || [ ! -f "$WEB_ROOT/frontend/index.html" ]; then
  echo "[ERROR] SONIC web project is incomplete: $WEB_ROOT" >&2
  exit 1
fi
if ss -ltn | grep -q ":${SONIC_UI_PORT} "; then
  echo "[ERROR] Port ${SONIC_UI_PORT} is already in use." >&2
  exit 1
fi

cd "$WEB_ROOT"
echo "[WEB] http://127.0.0.1:${SONIC_UI_PORT}"
exec env CUDA_VISIBLE_DEVICES="" \
  SONIC_TELEMETRY_ENDPOINT="tcp://127.0.0.1:${TELEMETRY_ZMQ_PORT}" \
  SONIC_REFERENCE_TELEMETRY_ENDPOINT="tcp://127.0.0.1:${REFERENCE_TELEMETRY_ZMQ_PORT}" \
  SONIC_CONTROL_ENDPOINT="tcp://127.0.0.1:${SIM_CONTROL_ZMQ_PORT}" \
  ARDY_BRIDGE_CONTROL_ENDPOINT="tcp://127.0.0.1:${BRIDGE_CONTROL_ZMQ_PORT}" \
  python -m uvicorn \
    --app-dir "$WEB_ROOT/backend" \
    --host 0.0.0.0 \
    --port "$SONIC_UI_PORT" \
    app:app
