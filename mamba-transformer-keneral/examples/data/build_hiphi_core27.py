"""Build resumable HiPHI Core27 rotations/root and ARDY 330-D features."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "ardy"))

from ardy_data.adapters.hiphi import HiPHIAdapter, HiPHIRecord  # noqa: E402
from ardy_data.hiphi import (  # noqa: E402
    HiPHIConverter,
    actor_split_map,
    cache_key,
    output_paths,
    save_conversion,
    split_audit,
)

_ADAPTER = None
_CONVERTER = None
_OUTPUT = None
_SPLITS = None
_COMMIT = None
_RESUME = False


def _init_worker(hiphi_root, output, target_fps, splits, commit, resume):
    global _ADAPTER, _CONVERTER, _OUTPUT, _SPLITS, _COMMIT, _RESUME
    torch.set_num_threads(1)
    _ADAPTER = HiPHIAdapter(hiphi_root)
    _CONVERTER = HiPHIConverter(_ADAPTER, target_fps)
    _OUTPUT = output
    _SPLITS = splits
    _COMMIT = commit
    _RESUME = resume


def _load_resumed(paths: dict[str, Path], key: str) -> dict | None:
    if not paths["item"].exists():
        return None
    record = json.loads(paths["item"].read_text())
    required = (paths["feature"], paths["rotation"], paths["root"], paths["qa"])
    if (
        record.get("status") == "pass"
        and record.get("cache_key") == key
        and all(path.exists() for path in required)
    ):
        return record
    return None


def _task(record: HiPHIRecord) -> dict:
    source = _ADAPTER.bvh_path(record)
    try:
        key = cache_key(record, source, _CONVERTER.target_fps, _COMMIT)
        paths = output_paths(_OUTPUT, record.motion_id)
        resumed = _load_resumed(paths, key) if _RESUME else None
        if resumed is not None:
            return {"record": resumed, "resumed": True}
        result = _CONVERTER.convert(record)
        result["source_path"] = str(source)
        manifest = save_conversion(
            _OUTPUT, record, result, _SPLITS[record.actor_id], key
        )
        if manifest["status"] != "pass":
            return {"error": "geometry_qa", "id": record.motion_id, "record": manifest}
        return {"record": manifest, "resumed": False}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "id": record.motion_id}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiphi-root", default="/data4/zts/datasets/HiPHI")
    parser.add_argument(
        "--output",
        default="/data4/zts/datasets/motion_fusion/staging/hiphi_core27_20fps",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--include-hoi", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard", help="Only convert records from this archive filename")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any sample fails")
    parser.add_argument("--split-seed", type=int, default=20260907)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.body_only and args.include_hoi:
        raise SystemExit("--body-only and --include-hoi are mutually exclusive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    adapter = HiPHIAdapter(args.hiphi_root)
    records = list(adapter.iter_records(body_only=not args.include_hoi, include_hoi=args.include_hoi))
    if args.shard:
        records = [record for record in records if record.archive_name == args.shard]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    splits = actor_split_map({record.actor_id for record in adapter.records}, args.split_seed)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO.parent, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    initargs = (str(args.hiphi_root), str(output), args.fps, splits, commit, args.resume)
    if args.workers == 1:
        _init_worker(*initargs)
        iterator = map(_task, records)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=initargs
        )
        iterator = executor.map(_task, records, chunksize=1)
    try:
        for index, result in enumerate(iterator, 1):
            results.append(result)
            if "error" in result or index % 25 == 0:
                print(json.dumps({"processed": index, "total": len(records), **result}), flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    kept = sorted(
        [result["record"] for result in results if "record" in result and result["record"]["status"] == "pass"],
        key=lambda record: record["sample_uid"],
    )
    errors = [{key: value for key, value in result.items() if key != "record"} for result in results if "error" in result]
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept)
    )
    for split in ("train", "val", "test"):
        (manifest_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept if record["split"] == split)
        )
    (manifest_dir / "errors.jsonl").write_text(
        "".join(json.dumps(error, sort_keys=True) + "\n" for error in errors)
    )
    leakage = split_audit(kept)
    (output / "qa").mkdir(parents=True, exist_ok=True)
    (output / "qa" / "split_audit.json").write_text(json.dumps(leakage, indent=2) + "\n")
    source_contracts = [record.get("source_fps") for record in kept]
    summary = {
        "status": "PASS" if kept and leakage["status"] == "PASS" and (not args.strict or not errors) else "FAIL",
        "selected": len(records),
        "converted": len(kept),
        "rejected": len(errors),
        "resumed": sum(bool(result.get("resumed")) for result in results),
        "target_fps": args.fps,
        "feature_dim": 330,
        "source_fps": sorted(set(source_contracts)),
        "body_only": not args.include_hoi,
        "include_hoi": args.include_hoi,
        "shard": args.shard,
        "split_audit": leakage,
        "errors": errors[:100],
    }
    (output / "qa" / "qa_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
