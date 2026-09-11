"""Build actor-disjoint ARDY-G1 manifests and physical feature caches from BONES-SEED."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))

from ardy_data.bones_seed import (  # noqa: E402
    BonesSeedG1Converter,
    actor_split,
    captions_from_row,
    dual_root_stats,
    metadata_rows,
)

_CONVERTER = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data4/zts/datasets/bones_seed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=52)
    parser.add_argument("--split-seed", type=int, default=20260824)
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args()


def _init_worker():
    global _CONVERTER
    torch.set_num_threads(1)
    _CONVERTER = BonesSeedG1Converter()


def _convert_task(task):
    row, dataset_root, output, min_frames, include_stats = task
    source = Path(dataset_root) / row["move_g1_path"]
    sample_id = row["filename"]
    shard = hashlib.sha256(sample_id.encode()).hexdigest()[:2]
    destination = Path(output) / "features" / shard / f"{sample_id}.npy"
    try:
        features = _CONVERTER.convert(source)
        if len(features) < min_frames:
            return {"skip": "too_short", "id": sample_id}
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, features.numpy().astype(np.float32))
        stats = dual_root_stats(features, _CONVERTER.motion_rep) if include_stats else None
        record = {
            "id": sample_id,
            "source_id": row["take_org_name"],
            "feature_path": str(destination),
            "source_path": str(source),
            "frames": int(len(features)),
            "source_fps": 120,
            "fps": 25,
            "actor_uid": row["actor_uid"],
            "is_mirror": str(row["is_mirror"]).lower() == "true",
            "package": row["package"],
            "category": row["category"],
            "captions": captions_from_row(row),
        }
        if not record["captions"]:
            return {"skip": "no_caption", "id": sample_id}
        return {
            "record": record,
            "stats_sum": stats.sum(0) if stats is not None else None,
            "stats_square_sum": np.square(stats).sum(0) if stats is not None else None,
            "stats_count": len(stats) if stats is not None else 0,
        }
    except Exception as exc:
        return {"skip": "invalid", "id": sample_id, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = root / "metadata" / "seed_metadata_v004.csv"
    rows = [
        row for row in metadata_rows(metadata)
        if actor_split(row["actor_uid"], args.split_seed) == args.split
    ]
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    tasks = [
        (row, str(root), str(output), args.min_frames, bool(args.stats)) for row in rows
    ]
    records = []
    skipped = {"too_short": 0, "no_caption": 0, "invalid": 0}
    stats_sum = np.zeros(418, dtype=np.float64)
    stats_square_sum = np.zeros(418, dtype=np.float64)
    stats_count = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as executor:
        for index, result in enumerate(executor.map(_convert_task, tasks, chunksize=8), 1):
            if "skip" in result:
                skipped[result["skip"]] += 1
                if result["skip"] == "invalid" and skipped["invalid"] <= 20:
                    print(json.dumps(result), file=sys.stderr, flush=True)
            else:
                records.append(result["record"])
                if result["stats_count"]:
                    stats_sum += result["stats_sum"]
                    stats_square_sum += result["stats_square_sum"]
                    stats_count += result["stats_count"]
            if index % 1000 == 0:
                print(json.dumps({"processed": index, "total": len(tasks), "kept": len(records)}), flush=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")
    if args.stats and stats_count:
        mean = stats_sum / stats_count
        variance = np.maximum(stats_square_sum / stats_count - np.square(mean), 1e-12)
        np.save(output / "mean.npy", mean.astype(np.float32))
        np.save(output / "std.npy", np.sqrt(variance).clip(min=1e-6).astype(np.float32))
    summary = {
        "status": "PASS" if records and not skipped["invalid"] else "FAIL",
        "dataset": "BONES-SEED v004",
        "representation": "ARDY G1 414-D explicit / 418-D dual-root stats",
        "split": args.split,
        "split_policy": f"actor-disjoint sha256 seed={args.split_seed}, 90/5/5",
        "samples": len(records),
        "skipped": skipped,
        "stats_frames": stats_count,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
