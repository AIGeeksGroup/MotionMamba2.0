"""Build mixture-weighted 334-D dual-root stats from compatible train manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent / "ardy"))

from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep  # noqa: E402
from ardy.skeleton import CoreSkeleton27  # noqa: E402


def source_arg(value: str) -> tuple[str, Path, float]:
    try:
        name, manifest, weight = value.split("=", 2)
        return name, Path(manifest), float(weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=MANIFEST=WEIGHT") from exc


def source_moments(manifest: Path, motion_rep) -> tuple[np.ndarray, np.ndarray, int, int]:
    total = np.zeros(334, dtype=np.float64)
    squares = np.zeros(334, dtype=np.float64)
    frames = 0
    motions = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if float(record["fps"]) != 20.0:
            raise ValueError(f"{record['id']} is {record['fps']} FPS, expected 20")
        features = torch.from_numpy(np.load(record["feature_path"])).float()
        if features.ndim != 2 or features.shape[1] != 330:
            raise ValueError(f"{record['id']} has incompatible feature shape {tuple(features.shape)}")
        length = torch.tensor([len(features)])
        local_root = motion_rep.global_root_to_local_root(
            features[None, :, :5], normalized=False, lengths=length
        )[0]
        dual = torch.cat((features[:, :5], local_root, features[:, 5:]), dim=-1).double().numpy()
        total += dual.sum(axis=0)
        squares += np.square(dual).sum(axis=0)
        frames += len(dual)
        motions += 1
    if not frames:
        raise ValueError(f"empty manifest {manifest}")
    mean = total / frames
    second = squares / frames
    return mean, second, frames, motions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=source_arg)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weight_sum = sum(weight for _, _, weight in args.source)
    if weight_sum <= 0 or any(weight <= 0 for _, _, weight in args.source):
        raise SystemExit("all mixture weights must be positive")
    skeleton = CoreSkeleton27(load=True)
    rep = ArdyMotionRep(skeleton, fps=20)
    weighted_mean = np.zeros(334, dtype=np.float64)
    weighted_second = np.zeros(334, dtype=np.float64)
    details = []
    for name, manifest, raw_weight in args.source:
        weight = raw_weight / weight_sum
        mean, second, frames, motions = source_moments(manifest, rep)
        weighted_mean += weight * mean
        weighted_second += weight * second
        details.append({"name": name, "manifest": str(manifest), "weight": weight, "frames": frames, "motions": motions})
    variance = np.maximum(weighted_second - np.square(weighted_mean), 1e-12)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "mean.npy", weighted_mean.astype(np.float32))
    np.save(output / "std.npy", np.sqrt(variance).clip(min=1e-6).astype(np.float32))
    metadata = {
        "representation": "CoreSkeleton27 ARDY",
        "motion_dim": 330,
        "dual_root_stats_dim": 334,
        "fps": 20,
        "aggregation": "per-source moments combined with normalized mixture weights",
        "sources": details,
    }
    (output / "stats_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
