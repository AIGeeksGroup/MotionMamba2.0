"""Validate exact sizes and gzip streams for all BONES-SEED archives."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path("/data4/zts/datasets/bones_seed")
EXPECTED = {
    "g1.tar.gz": 23_499_973_647,
    "soma_proportional.tar.gz": 45_468_083_856,
    "soma_uniform.tar.gz": 45_188_088_297,
}


def main() -> int:
    result = {"status": "PASS", "archives": {}}
    for name, expected in EXPECTED.items():
        path = ROOT / name
        size = path.stat().st_size if path.exists() else 0
        item = {"bytes": size, "expected_bytes": expected, "size_valid": size == expected}
        if size == expected:
            try:
                with gzip.open(path, "rb") as handle:
                    while handle.read(16 * 1024 * 1024):
                        pass
                item["gzip_valid"] = True
            except Exception as exc:
                item["gzip_valid"] = False
                item["error"] = f"{type(exc).__name__}: {exc}"
        else:
            item["gzip_valid"] = False
        if not item["size_valid"] or not item["gzip_valid"]:
            result["status"] = "FAIL"
        result["archives"][name] = item
        print(json.dumps({"archive": name, **item}), flush=True)
    (ROOT / "archive_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
