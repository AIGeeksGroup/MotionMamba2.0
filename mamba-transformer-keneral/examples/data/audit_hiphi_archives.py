"""Validate downloaded HiPHI archives against the official manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ardy_data.hiphi_archive import audit_archives  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiphi-root", default="/data4/zts/datasets/HiPHI")
    parser.add_argument("--readability", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    audit = audit_archives(args.hiphi_root, test_readability=args.readability)
    payload = json.dumps(audit, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    print(payload, end="")
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
