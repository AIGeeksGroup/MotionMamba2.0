"""Validate installed release inputs without starting GPU simulation services."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    required = [
        "vendor/viser/viser/client/build/index.html",
        "ardy/ardy/assets/skeletons/g1skel34/joints.p",
        "simulation/assets/warm_text.npy", "simulation/assets/stats/motion_contract.json",
        "checkpoints/official/ARDY-G1-RP-25FPS-Horizon52/config.yaml",
        "checkpoints/official/ARDY-G1-RP-25FPS-Horizon52/tokenizer.safetensors",
        "vendor/TensorRT/include/NvInfer.h", "vendor/TensorRT/lib/libnvinfer.so",
        "vendor/onnxruntime/include/onnxruntime_cxx_api.h", "vendor/onnxruntime/lib/libonnxruntime.so",
        "GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
        "GR00T-WholeBodyControl/.venv_sim/bin/python",
        "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
        "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
        "GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
        "ardy_sonic_web/node_modules/three/build/three.module.js",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing: raise SystemExit("Missing setup inputs:\n" + "\n".join(missing))
    import torch
    import triton
    import viser
    if not Path(viser.__file__).resolve().is_relative_to(ROOT / "vendor/viser"):
        raise SystemExit("Launch through launchers/common.sh to use the included Viser frontend")
    if torch.__version__ != "2.6.0+cu124" or triton.__version__ != "3.5.1":
        raise SystemExit(f"Expected torch 2.6.0+cu124 / triton 3.5.1; found {torch.__version__} / {triton.__version__}")
    if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable in the installed environment")
    from checkpoint_profile import resolve
    profile = resolve(args.checkpoint)
    print(json.dumps(dict(ready=True, profile=profile, viser=str(Path(viser.__file__).resolve()),
                         torch=torch.__version__, triton=triton.__version__), indent=2))


if __name__ == "__main__": main()
