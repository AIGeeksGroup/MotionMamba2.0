"""Audit released HiPHI metadata without relying on claimed README totals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ardy_data.hiphi_archive import write_metadata_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiphi-root", default="/data4/zts/datasets/HiPHI")
    parser.add_argument(
        "--output",
        default="/data4/zts/datasets/motion_fusion/staging/hiphi_audit",
    )
    args = parser.parse_args()
    audit = write_metadata_audit(args.hiphi_root, args.output)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
