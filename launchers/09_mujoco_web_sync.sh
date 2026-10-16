#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

PYTHON="$SONIC_ROOT/.venv_sim/bin/python"
SCRIPT="$SONIC_ROOT/gear_sonic/scripts/run_sim_headless_web.py"
if [ ! -x "$PYTHON" ] || [ ! -f "$SCRIPT" ]; then
  echo "[ERROR] MuJoCo environment or sync script is missing." >&2
  exit 1
fi
for port in "$TELEMETRY_ZMQ_PORT" "$SIM_CONTROL_ZMQ_PORT"; do
  if ss -ltn | grep -q ":${port} "; then
    echo "[ERROR] Port ${port} is already in use." >&2
    exit 1
  fi
done

cd "$SONIC_ROOT"
unset DISPLAY CYCLONEDDS_URI RMW_IMPLEMENTATION ROS_DOMAIN_ID
echo "[MUJOCO] GPU ${GPU_SIM}; telemetry=${TELEMETRY_ZMQ_PORT} (browser-local WebGL)"
exec env \
  CUDA_VISIBLE_DEVICES="$GPU_SIM" \
  SONIC_PHYSICS_STEP_MODE="${SONIC_PHYSICS_STEP_MODE:-fixed_lowcmd}" \
  SONIC_PHYSICS_STEPS_PER_LOWCMD="${SONIC_PHYSICS_STEPS_PER_LOWCMD:-4}" \
  SONIC_SIM_YAML_PD_JOINTS="${SONIC_SIM_YAML_PD_JOINTS:-}" \
  SONIC_SAFETY_HARNESS_SCALE="${SONIC_SAFETY_HARNESS_SCALE:-0}" \
  "$PYTHON" "$SCRIPT" \
    --debug-mode "${ARDY_SONIC_DEBUG_MODE:-ardy_streaming}" \
    --telemetry-port "$TELEMETRY_ZMQ_PORT" \
    --control-port "$SIM_CONTROL_ZMQ_PORT"
