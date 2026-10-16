#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ ! -f "$TENSORRT_ROOT/include/NvInfer.h" ]; then
  echo "[ERROR] TensorRT installation is invalid: $TENSORRT_ROOT" >&2
  exit 1
fi
if ! ss -ltn | grep -q ":${POSE_ZMQ_PORT} "; then
  echo "[WARN] ARDY publisher is not listening on ${POSE_ZMQ_PORT} yet; SONIC will reconnect." >&2
fi

cd "$SONIC_ROOT/gear_sonic_deploy"
set +u
source scripts/setup_env.sh
set -u
unset CYCLONEDDS_URI RMW_IMPLEMENTATION ROS_DOMAIN_ID
export CUDA_VISIBLE_DEVICES="$GPU_SIM"
export DEPLOY_ASSUME_YES=1
export SONIC_HISTORY_WARMUP="${SONIC_HISTORY_WARMUP:-zero}"
# In synchronized MuJoCo mode each 50 Hz command advances exactly four 5 ms
# physics steps. Real-robot deployments keep the official 500 Hz default.
export SONIC_LOWCMD_PUBLISH_HZ="${SONIC_LOWCMD_PUBLISH_HZ:-50}"
echo "[SONIC] GPU ${GPU_SIM}; official zmq_manager input on ${POSE_ZMQ_PORT}"
deploy_args=(
  --input-type zmq_manager \
  --zmq-host 127.0.0.1 \
  --zmq-port "$POSE_ZMQ_PORT" \
  --zmq-topic pose \
  --output-type zmq \
  --zmq-out-port "$REFERENCE_TELEMETRY_ZMQ_PORT" \
  --zmq-out-topic g1_debug \
  --zmq-conflate
)
if [ -n "${SONIC_POLICY_INPUT_LOGFILE:-}" ]; then
  deploy_args+=(--policy-input-logfile "$SONIC_POLICY_INPUT_LOGFILE")
fi
if [ -n "${SONIC_POLICY_OUTPUT_LOGFILE:-}" ]; then
  deploy_args+=(--policy-output-logfile "$SONIC_POLICY_OUTPUT_LOGFILE")
fi
deploy_args+=(sim)
exec bash deploy.sh "${deploy_args[@]}"
