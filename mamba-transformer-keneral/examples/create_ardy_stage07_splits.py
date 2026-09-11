"""Freeze Stage-07 train/dev/final-held-out manifests without copying tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _absolute_records(manifest_path: Path, records: list[dict]) -> list[dict]:
    result = []
    for record in records:
        item = dict(record)
        item["path"] = str((manifest_path.parent / record["path"]).resolve())
        result.append(item)
    return result


def _write(source: dict, records: list[dict], output: Path, split: str) -> None:
    manifest = dict(source)
    manifest["records"] = records
    manifest["split"] = split
    manifest["num_prompts"] = len({record["prompt_id"] for record in records})
    manifest["stage07_frozen_split"] = True
    output.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-count", type=int, default=8)
    args = parser.parse_args()
    train_path = Path(args.train_manifest).resolve()
    eval_path = Path(args.eval_manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = json.loads(train_path.read_text())
    evaluation = json.loads(eval_path.read_text())
    eval_records = evaluation["records"]
    if not 0 < args.dev_count < len(eval_records):
        raise ValueError("dev-count must leave at least one final-held-out record")
    _write(
        train,
        _absolute_records(train_path, train["records"]),
        output / "manifest_train.json",
        "train",
    )
    _write(
        evaluation,
        _absolute_records(eval_path, eval_records[: args.dev_count]),
        output / "manifest_dev.json",
        "development",
    )
    _write(
        evaluation,
        _absolute_records(eval_path, eval_records[args.dev_count :]),
        output / "manifest_final_heldout.json",
        "final-held-out",
    )
    (output / "split_policy.json").write_text(
        json.dumps(
            {
                "policy": "Stage-05 train unchanged; Stage-05 eval in manifest order",
                "development": f"first {args.dev_count} eval trajectories",
                "final_heldout": f"remaining {len(eval_records) - args.dev_count} eval trajectories",
                "final_heldout_use": "exactly once after architecture selection",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"train": len(train["records"]), "dev": args.dev_count,
                      "final_heldout": len(eval_records) - args.dev_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
