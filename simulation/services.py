"""Scoped tmux lifecycle for the exact six-service simulation chain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "mm2_" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:10] + "_"


def tmux(*args, check=True):
    return subprocess.run(["tmux", *args], check=check, capture_output=True, text=True)


def alive(name):
    return tmux("has-session", "-t", "=" + PREFIX + name, check=False).returncode == 0


def wait_port(port, service):
    deadline = time.monotonic() + int(os.getenv("STARTUP_WAIT_SECONDS", "1200"))
    while time.monotonic() < deadline:
        if not alive(service):
            raise RuntimeError(f"{service} exited; inspect runtime/logs/{service}.log")
        with socket.socket() as connection:
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(1)
    raise RuntimeError(f"{service} did not open {port}; inspect runtime/logs/{service}.log")


def start_one(name, port, args, env):
    if alive(name):
        wait_port(port, name)
        return
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} belongs to another process; refusing to replace it")
    # Offline workers do not need Hub/Git credentials. Do not serialize the
    # caller's entire environment (which may contain secrets) into tmux commands.
    explicit = {key: value for key, value in env.items() if
        key in {"PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "HOME", "TMPDIR", "NO_PROXY", "no_proxy",
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_HOME", "CUDAToolkit_ROOT",
                "WORK_ROOT", "ARDY_ROOT", "ARDY_ENV", "SONIC_ROOT", "PACKAGE_ROOT", "RUNTIME_ROOT",
                "LOG_ROOT", "LAUNCH_ROOT", "TENSORRT_ROOT", "TensorRT_ROOT", "CHECKPOINTS_DIR",
                "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "LOCAL_CACHE",
                "TORCH_HOME", "OPENBLAS_NUM_THREADS", "POSE_ZMQ_PORT", "TELEMETRY_ZMQ_PORT",
                "REFERENCE_TELEMETRY_ZMQ_PORT", "SIM_CONTROL_ZMQ_PORT", "BRIDGE_CONTROL_ZMQ_PORT"}
        or key.startswith(("MM2_", "ARDY_", "SONIC_", "TEXT_ENCODER", "GPU_"))}
    command = ["env", "-u", "HF_TOKEN", "-u", "HUGGING_FACE_HUB_TOKEN",
               *[f"{key}={value}" for key, value in explicit.items()], *args]
    log = ROOT / "runtime/logs" / (name + ".log")
    shell = shlex.join(command) + " > " + shlex.quote(str(log)) + " 2>&1"
    # Set all environment values explicitly: an existing tmux server may have
    # been started from an unrelated conda environment or Hugging Face account.
    tmux("new-session", "-d", "-s", PREFIX + name, "-c", str(ROOT), shell)
    print(f"Starting {name}; log: {log}", flush=True)
    wait_port(port, name)


def stop():
    for name in ("viewer", "policy", "sim", "web", "generator", "text"):
        if alive(name):
            tmux("kill-session", "-t", "=" + PREFIX + name)
    print("Stopped this checkout's services (other checkouts and training preserved).")


def start():
    from checkpoint_profile import resolve
    checkpoint = Path(os.environ["MM2_CHECKPOINT"]).expanduser().resolve(strict=True)
    profile = resolve(checkpoint)
    env = dict(os.environ)
    env.update(MM2_CHECKPOINT=str(checkpoint), MM2_MODEL_LABEL=profile["label"],
               MM2_INFERENCE_BACKEND=env.get("MM2_INFERENCE_BACKEND", "optimized"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LOCAL_CACHE="true")
    python = str(Path(env["ARDY_ENV"]) / "bin/python")
    launch = ROOT / "launchers"
    text_port = int(env["TEXT_ENCODER_PORT"])
    viewer_port = int(env.get("ARDY_VIEWER_PORT", "2333"))
    generator_port = int(env.get("MM2_GENERATOR_PORT", "8766"))
    ui_port = int(env["SONIC_UI_PORT"])
    start_one("text", text_port, ["bash", str(launch / "01_text_encoder.sh")], env)
    remote = profile["kind"] in ("native", "minimal")
    if remote:
        generator_env = dict(env, CUDA_VISIBLE_DEVICES=env["MM2_GPU_GENERATOR"])
        start_one("generator", generator_port, [python, "-u", str(ROOT / "simulation/serve_checkpoint.py"),
            "--checkpoint", str(checkpoint), "--output", str(Path(profile["output"]) / "generator"),
            "--port", str(generator_port), "--text-url", f"http://127.0.0.1:{text_port}/"], generator_env)
        url = f"http://127.0.0.1:{generator_port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/health", timeout=30) as response:
            health = json.load(response)
        if health["checkpoint_sha256"] != profile["checkpoint_sha256"]:
            raise RuntimeError("A different checkpoint is running; stop this checkout before switching")
        if env["MM2_INFERENCE_BACKEND"] != "reference" and health.get("inference_backend") != "optimized":
            raise RuntimeError("Acceleration not active; inspect generator log")
        env["MM2_GENERATOR_URL"] = url
    env.update(ARDY_GENERATOR_BACKEND="motion_mamba" if remote else "ardy",
               ARDY_SONIC_DEBUG_MODE="ardy_streaming", ARDY_SONIC_TIME_STRETCH="1.0",
               MM2_VIEWER_OUTPUT=str(Path(profile["output"]) / "viewer"))
    start_one("web", ui_port, ["bash", str(launch / "08_sonic_web_bridge.sh")], env)
    start_one("sim", int(env["TELEMETRY_ZMQ_PORT"]), ["bash", str(launch / "09_mujoco_web_sync.sh")], env)
    start_one("policy", int(env["REFERENCE_TELEMETRY_ZMQ_PORT"]), ["bash", str(launch / "11_sonic_policy_sync.sh")], env)
    start_one("viewer", viewer_port, ["bash", str(launch / "12_ardy_with_sonic.sh")], env)
    print(f"Viewer: http://127.0.0.1:{viewer_port}\nSONIC: http://127.0.0.1:{ui_port}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    # Backward-compatible old command; checkpoint metadata selects the model.
    parser.add_argument("--generator", choices=["ardy", "motion_mamba"])
    parser.add_argument("--debug-mode", default="ardy_streaming", choices=["ardy_streaming"])
    parser.add_argument("--time-stretch", default="1.0", choices=["1", "1.0"])
    args = parser.parse_args()
    if args.action == "stop": stop()
    else: start()
