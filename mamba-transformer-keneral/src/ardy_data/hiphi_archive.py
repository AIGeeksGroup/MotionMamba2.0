"""HiPHI metadata and archive integrity utilities."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


REQUIRED_METADATA_FIELDS = {
    "motion_id",
    "frame",
    "lu",
    "frame_lu",
    "duration_sec",
    "frame_count",
    "actor_id",
    "text_annotation",
    "is_hoi",
    "mirrored",
}


def discover_metadata_table(hiphi_root: str | Path) -> Path:
    """Find the released motion-level table by its schema, not its filename."""
    root = Path(hiphi_root)
    matches = []
    for path in sorted((root / "metadata").glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if REQUIRED_METADATA_FIELDS <= columns:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"expected one HiPHI motion metadata table, found {matches}")
    return matches[0]


def metadata_audit(hiphi_root: str | Path) -> dict:
    root = Path(hiphi_root)
    table = discover_metadata_table(root)
    motions = pd.read_csv(table)
    manifest = pd.read_csv(root / "data" / "archive_manifest.csv")
    motion_to_part = pd.read_csv(root / "data" / "motion_to_part.csv")

    def hours(rows: pd.DataFrame) -> float:
        return float(rows["duration_sec"].sum() / 3600.0)

    original = motions[~motions["mirrored"].astype(bool)]
    text = motions["text_annotation"].fillna("").astype(str).str.strip()
    mapping = {
        "motion_id": "motion_id",
        "duration_seconds": "duration_sec",
        "actor_id": "actor_id",
        "mirrored": "mirrored",
        "is_hoi": "is_hoi",
        "text_annotation": "text_annotation",
        "semantic_frame": "frame",
        "lexical_unit": "lu",
        "archive_part": "data/motion_to_part.csv:archive_name",
    }
    return {
        "status": "PASS",
        "metadata_table": str(table),
        "field_mapping": mapping,
        "total_motions": int(len(motions)),
        "total_release_hours": hours(motions),
        "original_only_hours": hours(original),
        "mirrored_hours": hours(motions[motions["mirrored"].astype(bool)]),
        "body_only_hours": hours(motions[~motions["is_hoi"].astype(bool)]),
        "hoi_hours": hours(motions[motions["is_hoi"].astype(bool)]),
        "body_only_original_hours": hours(original[~original["is_hoi"].astype(bool)]),
        "hoi_original_hours": hours(original[original["is_hoi"].astype(bool)]),
        "actor_count": int(motions["actor_id"].nunique()),
        "text_nonempty_count": int(text.ne("").sum()),
        "text_coverage": float(text.ne("").mean()),
        "duplicate_caption_count": int(text[text.ne("")].duplicated().sum()),
        "semantic_frame_count": int(motions["frame"].nunique()),
        "semantic_category_count": int(motions["frame_lu"].nunique()),
        "expected_archive_count": int(len(manifest)),
        "expected_package_count": int(manifest["package_count"].sum()),
        "indexed_original_count": int(len(motion_to_part)),
        "expected_compressed_bytes": int(manifest["compressed_bytes"].sum()),
        "expected_uncompressed_bytes": int(manifest["uncompressed_bytes"].sum()),
        "source_columns": list(motions.columns),
    }


def write_metadata_audit(hiphi_root: str | Path, output: str | Path) -> dict:
    audit = metadata_audit(hiphi_root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    rows = [
        ("Total motions", audit["total_motions"]),
        ("Total release hours", f'{audit["total_release_hours"]:.6f}'),
        ("Original-only hours", f'{audit["original_only_hours"]:.6f}'),
        ("Mirrored hours", f'{audit["mirrored_hours"]:.6f}'),
        ("Body-only hours", f'{audit["body_only_hours"]:.6f}'),
        ("HOI hours", f'{audit["hoi_hours"]:.6f}'),
        ("Body-only original hours", f'{audit["body_only_original_hours"]:.6f}'),
        ("HOI original hours", f'{audit["hoi_original_hours"]:.6f}'),
        ("Actors", audit["actor_count"]),
        ("Text coverage", f'{audit["text_coverage"]:.2%}'),
        ("Frame-LU categories", audit["semantic_category_count"]),
        ("Expected archives", audit["expected_archive_count"]),
        ("Compressed bytes", audit["expected_compressed_bytes"]),
        ("Uncompressed bytes", audit["expected_uncompressed_bytes"]),
    ]
    lines = ["# HiPHI Metadata Audit", "", "Values are computed from the released CSV files.", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(["", "## Field Mapping", "", "| Meaning | Released field |", "|---|---|"])
    lines.extend(f"| {name} | `{value}` |" for name, value in audit["field_mapping"].items())
    (out / "metadata_summary.md").write_text("\n".join(lines) + "\n")
    return audit


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_archives(
    hiphi_root: str | Path,
    *,
    test_readability: bool = False,
) -> dict:
    """Compare local shards with the release manifest and optional checksums."""
    root = Path(hiphi_root)
    manifest_path = root / "data" / "archive_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    expected_names = set(manifest["archive_name"].astype(str))
    actual_paths = sorted((root / "data").glob("HiPHI_data_part_*_of_*.tar.zst"))
    actual_by_name = {path.name: path for path in actual_paths}
    missing = sorted(expected_names - set(actual_by_name))
    unexpected = sorted(set(actual_by_name) - expected_names)
    size_mismatches = []
    checksum_mismatches = []
    corrupt = []
    for row in manifest.to_dict("records"):
        path = actual_by_name.get(str(row["archive_name"]))
        if path is None:
            continue
        expected_size = int(row["compressed_bytes"])
        if path.stat().st_size != expected_size:
            size_mismatches.append({"archive": path.name, "expected": expected_size, "actual": path.stat().st_size})
            continue
        for algorithm in ("sha256", "md5"):
            if algorithm in row and pd.notna(row[algorithm]) and str(row[algorithm]).strip():
                actual = _digest(path, algorithm)
                if actual.lower() != str(row[algorithm]).strip().lower():
                    checksum_mismatches.append({"archive": path.name, "algorithm": algorithm})
        if test_readability:
            result = subprocess.run(
                ["tar", "--zstd", "-tf", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode:
                corrupt.append({"archive": path.name, "error": result.stderr.strip()[-1000:]})
    status = "PASS" if not (missing or unexpected or size_mismatches or checksum_mismatches or corrupt) else "FAIL"
    return {
        "status": status,
        "expected_archives": len(expected_names),
        "actual_archives": len(actual_by_name),
        "missing_archives": missing,
        "unexpected_archives": unexpected,
        "size_mismatches": size_mismatches,
        "checksum_mismatches": checksum_mismatches,
        "corrupt_archives": corrupt,
        "actual_total_bytes": sum(path.stat().st_size for path in actual_paths),
        "expected_total_bytes": int(manifest["compressed_bytes"].sum()),
        "readability_tested": bool(test_readability),
    }
