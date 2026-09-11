"""Numerical plausibility gate for generated CoreSkeleton27 motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-frames", type=int, default=40)
    parser.add_argument("--max-root-speed", type=float, default=5.42)
    args = parser.parse_args()
    data = np.load(args.motion, allow_pickle=True)
    root = data["root_positions"]
    rotations = data["global_rot_mats"]
    fps = int(data["fps"])
    root_speed = np.linalg.norm(np.diff(root, axis=0), axis=-1) * fps
    boundaries = list(range(args.chunk_frames, len(root), args.chunk_frames))
    boundary_speeds = {str(frame): float(root_speed[frame - 1]) for frame in boundaries}
    identity = np.eye(3)
    metrics = {
        "status": "PASS",
        "motion": str(Path(args.motion).resolve()),
        "frames": len(root),
        "finite": bool(all(np.isfinite(data[key]).all() for key in ("root_positions", "posed_joints", "global_rot_mats"))),
        "mean_root_speed": float(root_speed.mean()),
        "max_root_speed": float(root_speed.max()),
        "boundary_root_speeds": boundary_speeds,
        "rotation_orthogonality_max_abs": float(
            np.abs(np.swapaxes(rotations, -1, -2) @ rotations - identity).max()
        ),
        "rotation_determinant_max_abs": float(np.abs(np.linalg.det(rotations) - 1).max()),
        "root_speed_threshold": args.max_root_speed,
    }
    if (
        not metrics["finite"]
        or metrics["max_root_speed"] > args.max_root_speed
        or metrics["rotation_orthogonality_max_abs"] > 1e-4
        or metrics["rotation_determinant_max_abs"] > 1e-4
    ):
        metrics["status"] = "FAIL"
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
