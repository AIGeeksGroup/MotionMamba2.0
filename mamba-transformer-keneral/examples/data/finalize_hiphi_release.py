"""Create the body-only HiPHI release manifests and provenance card."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ardy_data.hiphi import split_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--staging",
        default="/data4/zts/datasets/motion_fusion/staging/hiphi_core27_20fps",
    )
    parser.add_argument(
        "--release",
        default="/data4/zts/datasets/motion_fusion/releases/hiphi_body_hq_v1",
    )
    parser.add_argument("--hiphi-root", default="/data4/zts/datasets/HiPHI")
    args = parser.parse_args()
    staging = Path(args.staging)
    records = [
        json.loads(line)
        for line in (staging / "manifests" / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records = [record for record in records if record["status"] == "pass" and not record["is_hoi"]]
    audit = split_audit(records)
    if not records or audit["status"] != "PASS":
        raise SystemExit(f"refusing invalid release: records={len(records)}, split={audit['status']}")
    release = Path(args.release)
    release.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        selected = sorted(
            (record for record in records if record["split"] == split),
            key=lambda record: record["sample_uid"],
        )
        (release / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected)
        )
    (release / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    qa_source = staging / "qa" / "qa_summary.json"
    if qa_source.exists():
        shutil.copy2(qa_source, release / "qa_summary.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO.parent, text=True).strip()
    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "source": "HiPHI",
        "source_repository": "noitomrobotics/HiPHI",
        "license": "modalitynet-open-research-license",
        "processing": "HiPHI BVH -> CoreSkeleton27 -> rotation-aware 20 FPS -> ARDY 330-D",
        "body_only": True,
        "records": len(records),
        "feature_dim": 330,
        "stats_dim": 334,
        "target_fps": 20,
    }
    (release / "build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    data_card = """# HiPHI Body HQ v1

- Source: HiPHI
- Repository: `noitomrobotics/HiPHI`
- License: ModalityNet Open Research License (`LICENSE.md` in the downloaded source)
- Processing: HiPHI BVH -> CoreSkeleton27 -> rotation-aware 20 FPS -> ARDY 330-D
- Contents: body-only motions with official `text_annotation`; raw HOI remains available for later ablation
- Split: actor-disjoint and original/mirror-family-disjoint

This release is a local research derivative. It does not grant redistribution rights for raw or processed HiPHI data.
"""
    (release / "DATA_CARD.md").write_text(data_card)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
