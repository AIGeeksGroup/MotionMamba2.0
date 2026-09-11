"""Create a non-destructive manifest containing only real motion prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    manifest = json.loads(input_path.read_text())
    kept = []
    skipped = []
    for record in manifest["records"]:
        sample_path = input_path.parent / record["path"]
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        if sample["prompt"].lstrip().startswith("#"):
            skipped.append(record["path"])
        else:
            kept.append(record)
    manifest["records"] = kept
    manifest["num_prompts"] = len({record["prompt_id"] for record in kept})
    manifest["filtered_comment_records"] = skipped
    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"kept": len(kept), "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
