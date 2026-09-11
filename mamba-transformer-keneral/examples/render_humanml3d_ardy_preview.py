"""Render a compact skeleton preview from a generated ARDY manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))

from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep  # noqa: E402
from ardy.skeleton import CoreSkeleton27  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--manifest")
    sources.add_argument("--motion", action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.manifest:
        records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
        records = [
            record for record in records
            if not record.get("mirror", record.get("is_mirror", False))
        ][args.offset:args.offset + args.samples]
    else:
        records = [{"id": Path(path).stem, "motion_path": path} for path in args.motion]
    skeleton = CoreSkeleton27()
    rep = ArdyMotionRep(skeleton, fps=20)
    figure = plt.figure(figsize=(3.2 * args.frames, 3.2 * len(records)))
    parents = skeleton.joint_parents.tolist()
    for row, record in enumerate(records):
        if "motion_path" in record:
            posed = np.load(record["motion_path"])["posed_joints"]
        else:
            features = torch.from_numpy(np.load(record["feature_path"])).float()
            posed = rep.inverse(features[None], is_normalized=False)["posed_joints"][0].numpy()
        indices = np.linspace(0, len(posed) - 1, args.frames).round().astype(int)
        for column, frame in enumerate(indices):
            ax = figure.add_subplot(len(records), args.frames, row * args.frames + column + 1, projection="3d")
            joints = posed[frame]
            for joint in range(1, len(parents)):
                pair = joints[[parents[joint], joint]]
                ax.plot(pair[:, 0], pair[:, 2], pair[:, 1], color="#176b87", linewidth=2)
            ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], color="#d1495b", s=6)
            center = joints[0]
            ax.set_xlim(center[0] - 1, center[0] + 1)
            ax.set_ylim(center[2] - 1, center[2] + 1)
            ax.set_zlim(0, 2)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            ax.set_title(f"{record['id']}  f={frame}", fontsize=9)
    figure.tight_layout()
    figure.savefig(args.output, dpi=140, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
