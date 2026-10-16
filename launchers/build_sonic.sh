#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
export HAS_ROS2=0
export onnxruntime_ROOT="${onnxruntime_ROOT:-$MM2_ROOT/vendor/onnxruntime}"
[[ -f "$TENSORRT_ROOT/include/NvInfer.h" ]] || { echo "Install TensorRT 10.13.3.9 under vendor/TensorRT (README.MD)" >&2; exit 2; }
[[ -f "$onnxruntime_ROOT/include/onnxruntime_cxx_api.h" ]] || { echo "Install ONNX Runtime C++ 1.16.3 under vendor/onnxruntime (README.MD)" >&2; exit 2; }
cmake -S "$SONIC_ROOT/gear_sonic_deploy" -B "$SONIC_ROOT/gear_sonic_deploy/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build "$SONIC_ROOT/gear_sonic_deploy/build" --target g1_deploy_onnx_ref -j "${MM2_BUILD_JOBS:-8}"
