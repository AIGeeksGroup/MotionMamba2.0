"""Build and validate ARDY-faithful HumanML3D 330-D data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))

from ardy_data.humanml3d import (  # noqa: E402
    HumanML3DConverter,
    _rows,
    load_split_ids,
    validate_ardy_features,
    write_manifest,
)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/data3/zts/datasets/HumanML3D")
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-canonicalize", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--min-frames", type=int, default=40)
    return p.parse_args()


def main() -> int:
    a = args()
    root = Path(a.dataset_root)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    selected = None if a.split == "all" else load_split_ids(root, a.split)
    converter = HumanML3DConverter(root)
    records = []
    stats_sum = np.zeros(334, dtype=np.float64)
    stats_square_sum = np.zeros(334, dtype=np.float64)
    stats_count = 0
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
    motion_rep = ArdyMotionRep(converter.skeleton, fps=converter.fps)
    skipped = {"humanact12": 0, "missing": 0, "invalid": 0, "too_short": 0}
    for row in _rows(root / "index.csv", root):
        if row.is_humanact12:
            skipped["humanact12"] += 1
            continue
        names = [row.new_name]
        if not a.no_mirror:
            names.append("M" + row.new_name)
        names = [name for name in names if selected is None or name in selected]
        for name in names:
            if a.max_samples is not None and len(records) >= a.max_samples:
                break
            mirror = name.startswith("M")
            try:
                result = converter.convert_row(row, mirror=mirror, canonicalize=not a.no_canonicalize)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                skipped["invalid"] += 1
                print(f"SKIP {row.new_name}: {exc}", file=sys.stderr)
                continue
            if result["features"].shape[0] < a.min_frames:
                skipped["too_short"] += 1
                continue
            feature_path = out / f"{name}.npy"
            np.save(feature_path, result["features"].numpy().astype(np.float32))
            validation = validate_ardy_features(result["features"], converter.skeleton, converter.fps)
            if (
                not validation["finite"]
                or validation["roundtrip_max_abs"] > 2e-4
                or validation["contact_disagreement_fraction"] > 0.01
                or validation["rotation_orthogonality_max_abs"] > 1e-4
                or validation["rotation_det_max_abs"] > 1e-4
                or validation["bone_length_max_abs"] > 1e-4
                or validation["heading_unit_max_abs"] > 1e-4
                or validation["minimum_foot_height"] < -1e-4
            ):
                raise RuntimeError(f"geometry validation failed for {name}: {validation}")
            feature_batch = result["features"][None]
            length = torch.tensor([feature_batch.shape[1]])
            local_root = motion_rep.global_root_to_local_root(
                feature_batch[:, :, :5], normalized=False, lengths=length
            )[0]
            # ARDY stores dual-root stats as [global root 5, local root 4,
            # body 325].  Model tensors remain 330-D; the extra 4-D block is
            # used internally after root prediction.
            stats_row = torch.cat((result["features"][:, :5], local_root, result["features"][:, 5:]), dim=-1)
            stats_array = stats_row.numpy().astype(np.float64)
            stats_sum += stats_array.sum(axis=0)
            stats_square_sum += np.square(stats_array).sum(axis=0)
            stats_count += len(stats_array)
            records.append({
                "id": name,
                "source_id": row.new_name,
                "feature_path": str(feature_path),
                "source_path": result["source_path"],
                "frames": int(result["features"].shape[0]),
                "fps": result["fps"],
                "scale": result["scale"],
                "mirror": mirror,
                "split": a.split,
                "validation": validation,
            })
        if a.max_samples is not None and len(records) >= a.max_samples:
            break
    write_manifest(records, out / "manifest.jsonl")
    summary = {"samples": len(records), "skipped": skipped, "split": a.split, "feature_dim": 330}
    if a.stats and stats_count:
        mean = stats_sum / stats_count
        variance = np.maximum(stats_square_sum / stats_count - np.square(mean), 1e-12)
        np.save(out / "mean.npy", mean.astype(np.float32))
        np.save(out / "std.npy", np.sqrt(variance).clip(min=1e-6).astype(np.float32))
        summary["stats_frames"] = int(stats_count)
        summary["stats_dim"] = 334
        summary["stats_source"] = "selected split only"
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
