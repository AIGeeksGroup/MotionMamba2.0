"""Resumable HiPHI metadata, first-shard smoke, full build, and release pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "ardy"))

from ardy_data.adapters.hiphi import HiPHIAdapter  # noqa: E402
from ardy_data.hiphi import output_paths, split_audit  # noqa: E402
from ardy_data.hiphi_archive import audit_archives, write_metadata_audit  # noqa: E402


def extract_archive(archive: Path, hiphi_root: Path, done_path: Path, expected_bvh: Path | None = None) -> dict:
    identity = {"archive": str(archive), "size": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns}
    if done_path.exists():
        previous = json.loads(done_path.read_text())
        if all(previous.get(key) == value for key, value in identity.items()) and (
            expected_bvh is None or expected_bvh.exists()
        ):
            return previous
    subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(hiphi_root.parent)], check=True)
    bvh_files = list((hiphi_root / "data").glob("*/*/*/motion_actor.bvh"))
    audit = {
        **identity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bvh_count_visible": len(bvh_files),
        "sample_paths": [str(path) for path in bvh_files[:20]],
        "sample_sizes": [path.stat().st_size for path in bvh_files[:20]],
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def run_python(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(REPO / "examples" / "data" / script), *arguments], check=True)


def consolidate_items(output: Path) -> dict:
    records = []
    for path in sorted((output / "manifests" / "items").glob("*/*.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "pass":
            records.append(record)
    records.sort(key=lambda record: record["sample_uid"])
    manifest_dir = output / "manifests"
    (manifest_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    for split in ("train", "val", "test"):
        (manifest_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records if record["split"] == split)
        )
    errors = []
    for path in sorted((manifest_dir / "errors_by_shard").glob("*.jsonl")):
        errors.extend(line for line in path.read_text().splitlines() if line.strip())
    (manifest_dir / "errors.jsonl").write_text("\n".join(errors) + ("\n" if errors else ""))
    audit = split_audit(records)
    (output / "qa" / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {"records": len(records), "rejected": len(errors), "split_audit": audit}


def cleanup_extracted_shard(adapter: HiPHIAdapter, archive_name: str) -> int:
    removed = 0
    for record in adapter.records:
        if record.archive_name != archive_name:
            continue
        motion_dir = adapter.bvh_path(record).parent
        if motion_dir.exists():
            shutil.rmtree(motion_dir)
            removed += 1
        for parent in (motion_dir.parent, motion_dir.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                pass
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiphi-root", default="/data4/zts/datasets/HiPHI")
    parser.add_argument("--fusion-root", default="/data4/zts/datasets/motion_fusion")
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--include-hoi", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full", action="store_true", help="After smoke, extract all shards and build the full release")
    parser.add_argument(
        "--streaming-full",
        action="store_true",
        help="Process one shard at a time and remove only its extracted copies after conversion; archives are retained",
    )
    args = parser.parse_args()
    if args.include_hoi:
        raise SystemExit("the v1 release is body-only; use build_hiphi_core27.py for the later HOI ablation")
    if args.full and args.streaming_full:
        raise SystemExit("--full and --streaming-full are mutually exclusive")
    hiphi_root = Path(args.hiphi_root)
    fusion_root = Path(args.fusion_root)
    audit_dir = fusion_root / "staging" / "hiphi_audit"
    write_metadata_audit(hiphi_root, audit_dir)
    archive_audit = audit_archives(hiphi_root, test_readability=False)
    (audit_dir / "archive_audit.json").write_text(json.dumps(archive_audit, indent=2) + "\n")
    if archive_audit["status"] != "PASS":
        raise SystemExit("archive audit failed; resume the mirror download before preprocessing")

    manifest = pd.read_csv(hiphi_root / "data" / "archive_manifest.csv").sort_values("part_index")
    adapter = HiPHIAdapter(hiphi_root)
    first_name = str(manifest.iloc[0]["archive_name"])
    first_archive = hiphi_root / "data" / first_name
    listing = subprocess.run(
        ["tar", "--zstd", "-tf", str(first_archive)], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    (audit_dir / "first_shard_listing.txt").write_text("\n".join(listing[:100]) + "\n")
    first_done = fusion_root / "staging" / "hiphi_extract_audit" / "part_0001.extract_done.json"
    first_record = next(
        record for record in adapter.records if record.archive_name == first_name
    )
    first_extract = extract_archive(
        first_archive,
        hiphi_root,
        first_done,
        adapter.bvh_path(first_record),
    )
    (audit_dir / "first_shard_extract_audit.json").write_text(json.dumps(first_extract, indent=2) + "\n")

    parser_records = [
        record for record in adapter.iter_records(body_only=True)
        if record.archive_name == first_name
    ][:20]
    parser_contracts = []
    parser_errors = []
    for record in parser_records:
        try:
            parser_contracts.append(adapter.parse(record).metadata["bvh_contract"])
        except Exception as exc:
            parser_errors.append({"motion_id": record.motion_id, "error": f"{type(exc).__name__}: {exc}"})
    parser_summary = {
        "status": "PASS" if len(parser_contracts) == 20 and not parser_errors else "FAIL",
        "requested": 20,
        "parsed": len(parser_contracts),
        "errors": parser_errors,
        "observed_joint_counts": sorted(set(item["joint_count"] for item in parser_contracts)),
        "observed_fps": sorted(set(item["fps"] for item in parser_contracts)),
        "observed_euler_orders": sorted({order for item in parser_contracts for order in item["euler_orders"]}),
        "source_skeleton_height_m": [item["source_skeleton_height_m"] for item in parser_contracts],
    }
    (audit_dir / "parser_smoke_summary.json").write_text(json.dumps(parser_summary, indent=2) + "\n")
    if parser_summary["status"] != "PASS":
        raise SystemExit(f"first-shard parser smoke failed: {parser_summary}")

    output = fusion_root / "staging" / "hiphi_core27_20fps"
    smoke_args = [
        "--hiphi-root", str(hiphi_root), "--output", str(output), "--fps", str(args.target_fps),
        "--body-only", "--shard", first_name, "--max-samples", "100", "--workers", str(args.workers),
        "--strict",
    ]
    if args.resume:
        smoke_args.append("--resume")
    run_python("build_hiphi_core27.py", *smoke_args)
    smoke_summary = json.loads((output / "qa" / "qa_summary.json").read_text())
    if smoke_summary["converted"] < 100 or smoke_summary["status"] != "PASS":
        raise SystemExit(f"first-shard smoke failed: {smoke_summary}")
    (audit_dir / "sample_summary.json").write_text(json.dumps(smoke_summary, indent=2) + "\n")
    (audit_dir / "qa_summary.json").write_text(json.dumps(smoke_summary, indent=2) + "\n")
    smoke_records = [
        json.loads(line)
        for line in (output / "manifests" / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    originals = [record for record in smoke_records if not record["is_mirror"]]
    random_rows = random.Random(20260907).sample(originals, min(10, len(originals)))

    def geometry_score(record):
        qa = record["geometry_qa"]
        return max(
            qa["roundtrip_max_abs"] / 2e-4,
            qa["rotation_orthogonality_max_abs"] / 1e-4,
            qa["rotation_det_max_abs"] / 1e-4,
            qa["bone_length_max_abs"] / 1e-4,
            max(0.0, -qa["minimum_foot_height"]) / 1e-4,
        )

    worst_rows = sorted(originals, key=geometry_score, reverse=True)[:10]
    for label, selected in (("random", random_rows), ("worst", worst_rows)):
        render_manifest = audit_dir / f"render_{label}_selection.jsonl"
        render_manifest.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in selected))
        subprocess.run([
            sys.executable,
            str(REPO / "examples" / "render_humanml3d_ardy_preview.py"),
            "--manifest", str(render_manifest),
            "--output", str(audit_dir / f"hiphi_first_shard_{label}.png"),
            "--samples", str(len(selected)),
            "--frames", "5",
        ], check=True)
    if not args.full and not args.streaming_full:
        print("First-shard smoke passed; rerun with --full only after reviewing QA and disk strategy.")
        return 0

    if args.streaming_full:
        errors_dir = output / "manifests" / "errors_by_shard"
        errors_dir.mkdir(parents=True, exist_ok=True)
        stream_dir = fusion_root / "staging" / "hiphi_stream_audit"
        stream_dir.mkdir(parents=True, exist_ok=True)
        pipeline_digest = hashlib.sha256()
        for relative in (
            "src/ardy_data/hiphi.py",
            "src/ardy_data/parsers/hiphi.py",
            "src/ardy_data/resample.py",
            "src/ardy_data/retarget/core27.py",
        ):
            pipeline_digest.update((REPO / relative).read_bytes())
        pipeline_hash = pipeline_digest.hexdigest()
        for row in manifest.to_dict("records"):
            part_index = int(row["part_index"])
            archive_name = str(row["archive_name"])
            archive = hiphi_root / "data" / archive_name
            selected = [
                record for record in adapter.iter_records(body_only=True)
                if record.archive_name == archive_name
            ]
            stream_done = stream_dir / f"part_{part_index:04d}.stream_done.json"
            if stream_done.exists():
                previous = json.loads(stream_done.read_text())
                item_count = 0
                for record in selected:
                    item_path = output_paths(output, record.motion_id)["item"]
                    if item_path.exists() and json.loads(item_path.read_text()).get("status") == "pass":
                        item_count += 1
                if (
                    previous.get("archive_size") == archive.stat().st_size
                    and previous.get("target_fps") == args.target_fps
                    and previous.get("selected") == len(selected)
                    and previous.get("pipeline_hash") == pipeline_hash
                    and item_count == previous.get("converted")
                ):
                    continue
            done = fusion_root / "staging" / "hiphi_extract_audit" / f"part_{part_index:04d}.extract_done.json"
            extract_archive(archive, hiphi_root, done, adapter.bvh_path(selected[0]))
            build_args = [
                "--hiphi-root", str(hiphi_root), "--output", str(output), "--fps", str(args.target_fps),
                "--body-only", "--shard", archive_name, "--workers", str(args.workers), "--resume",
            ]
            run_python("build_hiphi_core27.py", *build_args)
            shard_summary = json.loads((output / "qa" / "qa_summary.json").read_text())
            if shard_summary["selected"] != shard_summary["converted"] + shard_summary["rejected"]:
                raise SystemExit(f"incomplete shard conversion, refusing cleanup: {shard_summary}")
            shutil.copy2(output / "manifests" / "errors.jsonl", errors_dir / f"part_{part_index:04d}.jsonl")
            removed = cleanup_extracted_shard(adapter, archive_name)
            stream_done.write_text(json.dumps({
                "archive": archive_name,
                "archive_size": archive.stat().st_size,
                "target_fps": args.target_fps,
                "pipeline_hash": pipeline_hash,
                "selected": shard_summary["selected"],
                "converted": shard_summary["converted"],
                "rejected": shard_summary["rejected"],
                "removed_extracted_packages": removed,
                "archive_retained": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
        consolidated = consolidate_items(output)
        if consolidated["split_audit"]["status"] != "PASS":
            raise SystemExit(f"consolidated split audit failed: {consolidated}")
        run_python("finalize_hiphi_release.py", "--staging", str(output), "--hiphi-root", str(hiphi_root))
        print(json.dumps(consolidated, indent=2))
        return 0

    required = int(manifest["uncompressed_bytes"].sum())
    free = shutil.disk_usage(hiphi_root).free
    if free < required + 50 * 10**9:
        raise SystemExit(
            f"full simultaneous extraction needs at least {required + 50 * 10**9} bytes free; have {free}. "
            "Use an explicitly approved sequential-shard extraction/build cleanup plan."
        )
    for row in manifest.to_dict("records"):
        archive = hiphi_root / "data" / str(row["archive_name"])
        done = fusion_root / "staging" / "hiphi_extract_audit" / f'part_{int(row["part_index"]):04d}.extract_done.json'
        extract_archive(archive, hiphi_root, done)
    full_args = [
        "--hiphi-root", str(hiphi_root), "--output", str(output), "--fps", str(args.target_fps),
        "--body-only", "--workers", str(args.workers), "--strict",
    ]
    if args.resume:
        full_args.append("--resume")
    run_python("build_hiphi_core27.py", *full_args)
    run_python("finalize_hiphi_release.py", "--staging", str(output), "--hiphi-root", str(hiphi_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
