#!/usr/bin/env bash
set -euo pipefail
export MM2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "${MM2_CONFIG:-$MM2_ROOT/simulation.env}"
set +a
cd "$MM2_ROOT"
export MM2_CHECKPOINT="$(realpath -m -- "$MM2_CHECKPOINT")"
export WORK_ROOT="$MM2_ROOT"
export ARDY_ROOT="$MM2_ROOT/ardy"
export ARDY_ENV="${MM2_PYTHON_ENV:-$MM2_ROOT/.venv}"
export SONIC_ROOT="$MM2_ROOT/GR00T-WholeBodyControl"
export PACKAGE_ROOT="$MM2_ROOT/vendor"
export TENSORRT_ROOT="${TENSORRT_ROOT:-$MM2_ROOT/vendor/TensorRT}"
export TensorRT_ROOT="$TENSORRT_ROOT"
export RUNTIME_ROOT="$MM2_ROOT/runtime"
export LOG_ROOT="$RUNTIME_ROOT/logs"
export LAUNCH_ROOT="$MM2_ROOT/launchers"
export HF_HOME="${HF_HOME:-$MM2_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$MM2_ROOT/cache/torch"
export TMPDIR="$RUNTIME_ROOT/tmp"
export TEXT_ENCODER_TMP_FOLDER="$TMPDIR/text_encoder"
export TEXT_ENCODER_PORT="${TEXT_ENCODER_PORT:-9550}"
export SONIC_UI_PORT="${SONIC_UI_PORT:-8080}"
export POSE_ZMQ_PORT="${POSE_ZMQ_PORT:-5556}"
export REFERENCE_TELEMETRY_ZMQ_PORT="${REFERENCE_TELEMETRY_ZMQ_PORT:-5557}"
export TELEMETRY_ZMQ_PORT="${TELEMETRY_ZMQ_PORT:-5560}"
export SIM_CONTROL_ZMQ_PORT="${SIM_CONTROL_ZMQ_PORT:-5558}"
export BRIDGE_CONTROL_ZMQ_PORT="${BRIDGE_CONTROL_ZMQ_PORT:-5559}"
export GPU_TEXT="$MM2_GPU_TEXT" GPU_ARDY="$MM2_GPU_VIEWER" GPU_SIM="$MM2_GPU_SIM"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export CHECKPOINTS_DIR="$MM2_ROOT/checkpoints/official"
# Resolve adapter repository IDs through the pinned Hub cache, just like the
# deployed encoder. A local adapter folder is not a full base-model snapshot.
unset TEXT_ENCODERS_DIR
export PYTHONPATH="$MM2_ROOT/vendor/viser:$ARDY_ROOT:$SONIC_ROOT:$MM2_ROOT/ardy_sonic_web/backend${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$MM2_ROOT/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python:$PYTHONPATH"
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ARDY_ENV/bin:$TENSORRT_ROOT/bin:$PATH"
mkdir -p "$RUNTIME_ROOT/tmp" "$LOG_ROOT"
port_is_listening() { ss -ltnH "sport = :$1" | grep -q .; }
