"""Run the controlled Original/DiT/Minimal Stage-05 release comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-feature-sample", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--dit-checkpoint", required=True)
    parser.add_argument("--minimal-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["original", "dit", "minimal"],
        default=["original", "dit", "minimal"],
        help="Subset to run; existing results from other modes are preserved.",
    )
    parser.add_argument("--window-size-chunks", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    return parser.parse_args()


def _motion_metrics(path: Path, chunks: int, chunk_frames: int = 40) -> dict:
    with np.load(path, allow_pickle=True) as motion:
        joints = motion["posed_joints"]
        root = motion["root_positions"]
        finite = all(
            np.isfinite(motion[key]).all()
            for key in motion.files
            if motion[key].dtype.kind in "fciu"
        )
    joint_boundaries = [
        float(np.linalg.norm(joints[index] - joints[index - 1], axis=-1).mean())
        for index in range(chunk_frames, len(joints), chunk_frames)
    ]
    root_boundaries = [
        float(np.linalg.norm(root[index] - root[index - 1]))
        for index in range(chunk_frames, len(root), chunk_frames)
    ]
    chunk_speeds = [
        float(
            np.linalg.norm(
                np.diff(joints[index * chunk_frames : (index + 1) * chunk_frames], axis=0),
                axis=-1,
            ).mean()
        )
        for index in range(chunks)
    ]
    return {
        "finite": bool(finite),
        "shape": list(joints.shape),
        "root_displacement": float(np.linalg.norm(root[-1] - root[0])),
        "boundary_joint_mean": float(np.mean(joint_boundaries)) if joint_boundaries else None,
        "boundary_joint_max": float(np.max(joint_boundaries)) if joint_boundaries else None,
        "boundary_root_mean": float(np.mean(root_boundaries)) if root_boundaries else None,
        "boundary_root_max": float(np.max(root_boundaries)) if root_boundaries else None,
        "chunk_speed_min": float(np.min(chunk_speeds)),
        "chunk_speed_max": float(np.max(chunk_speeds)),
    }


def main() -> int:
    args = parse_args()
    if any(horizon < 1 for horizon in args.horizons):
        raise ValueError("all horizons must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = Path(__file__).with_name("ardy_sequence_kernel_generate.py")
    checkpoints = {
        "dit": args.dit_checkpoint,
        "minimal": args.minimal_checkpoint,
    }
    summary_path = output_dir / "comparison.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {
            "prompt": args.prompt,
            "seed": args.seed,
            "diffusion_steps": args.diffusion_steps,
            "results": [],
        }
    if (summary["prompt"], summary["seed"], summary["diffusion_steps"]) != (
        args.prompt,
        args.seed,
        args.diffusion_steps,
    ):
        raise ValueError("existing comparison configuration does not match")
    summary["results"] = [
        result for result in summary["results"] if result["mode"] not in args.modes
    ]
    for mode in args.modes:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for chunks in args.horizons:
            output_path = mode_dir / f"walk_turn_{chunks}chunk.npz"
            raw_metrics_path = mode_dir / f"walk_turn_{chunks}chunk_runtime.json"
            command = [
                sys.executable,
                str(generator),
                f"--{mode}",
                "--ardy-root",
                args.ardy_root,
                "--checkpoints-dir",
                args.checkpoints_dir,
                "--model",
                args.model,
                "--text-feature-sample",
                args.text_feature_sample,
                "--prompt",
                args.prompt,
                "--duration",
                str(2 * chunks),
                "--diffusion-steps",
                str(args.diffusion_steps),
                "--seed",
                str(args.seed),
                "--attention-backend",
                "sdpa",
                "--window-size-chunks",
                str(args.window_size_chunks),
                "--sink-tokens",
                str(args.sink_tokens),
                "--output",
                str(output_path),
                "--metrics-output",
                str(raw_metrics_path),
                "--no-postprocess",
            ]
            if mode != "original":
                command.extend(["--sequence-checkpoint", checkpoints[mode]])
            print(f"Running {mode} at {chunks} chunk(s)", flush=True)
            subprocess.run(command, check=True)
            runtime = json.loads(raw_metrics_path.read_text())
            result = {
                "mode": mode,
                "chunks": chunks,
                "output": str(output_path),
                "generation_seconds": runtime["generation_seconds"],
                "peak_gpu_memory_mib": runtime["peak_gpu_memory_mib"],
                **_motion_metrics(output_path, chunks),
            }
            for key in ("memory_trace", "root_cache", "body_cache"):
                if key in runtime:
                    result[key] = runtime[key]
            summary["results"].append(result)
            print(json.dumps(result), flush=True)
    summary["results"].sort(key=lambda result: (("original", "dit", "minimal").index(result["mode"]), result["chunks"]))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved comparison: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
