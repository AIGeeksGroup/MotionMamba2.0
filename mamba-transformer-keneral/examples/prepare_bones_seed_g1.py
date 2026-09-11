"""Validate and extract the BONES-SEED G1 archive before feature building."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import subprocess
from pathlib import Path


EXPECTED_G1_ARCHIVE_BYTES = 23_499_973_647


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data4/zts/datasets/bones_seed")
    args = parser.parse_args()
    root = Path(args.dataset_root)
    archive = root / "g1.tar.gz"
    if not archive.exists() or archive.stat().st_size != EXPECTED_G1_ARCHIVE_BYTES:
        raise RuntimeError(
            f"G1 archive is absent/incomplete: {archive} "
            f"({archive.stat().st_size if archive.exists() else 0}/{EXPECTED_G1_ARCHIVE_BYTES})"
        )

    with gzip.open(archive, "rb") as handle:
        while handle.read(16 * 1024 * 1024):
            pass
    print("gzip_validation=PASS", flush=True)

    subprocess.run(
        ["tar", "-xzf", str(archive), "-C", str(root)], check=True
    )
    metadata_path = root / "metadata" / "seed_metadata_v004.csv"
    missing = []
    metadata_count = 0
    with metadata_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            metadata_count += 1
            path = root / row["move_g1_path"]
            if not path.is_file() and len(missing) < 20:
                missing.append(str(path))
    csv_count = sum(1 for _ in (root / "g1" / "csv").rglob("*.csv"))
    status = "PASS" if metadata_count == csv_count == 142_220 and not missing else "FAIL"
    result = {
        "status": status,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "gzip_validation": "PASS",
        "metadata_rows": metadata_count,
        "g1_csv_files": csv_count,
        "missing_metadata_paths_first20": missing,
        "free_bytes_after_extract": shutil.disk_usage(root).free,
    }
    (root / "g1_prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
