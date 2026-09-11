"""Precompute deterministic HumanML3D caption embeddings with local LLM2Vec."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))

from ardy_data import select_manifest_caption  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--text-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder-mode", choices=["local", "api"], default="local")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    records = []
    seen = set()
    for manifest in args.manifest:
        for line in Path(manifest).read_text().splitlines():
            record = json.loads(line)
            if record["id"] not in seen:
                records.append(record)
                seen.add(record["id"])
    if args.max_samples is not None:
        records = records[:args.max_samples]
    captions = [select_manifest_caption(record, args.seed, args.text_root) for record in records]
    if not all(captions):
        raise ValueError("all cached samples must have a non-empty caption")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embedding_path = output / "embeddings.npy"
    progress_path = output / "progress.json"
    if embedding_path.exists():
        embeddings = np.load(embedding_path, mmap_mode="r+")
        if embeddings.shape != (len(records), 4096):
            raise ValueError(f"existing cache shape {embeddings.shape} does not match {(len(records), 4096)}")
        start = json.loads(progress_path.read_text()).get("completed", 0) if progress_path.exists() else 0
    else:
        embeddings = np.lib.format.open_memmap(
            embedding_path, mode="w+", dtype=np.float16, shape=(len(records), 4096)
        )
        start = 0

    from ardy.model.load_model import load_text_encoder

    encoder = load_text_encoder(mode=args.encoder_mode, device=args.device)
    started = time.perf_counter()
    for batch_start in range(start, len(records), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(records))
        with torch.inference_mode():
            feature, lengths = encoder(captions[batch_start:batch_end])
        expected_shape = (batch_end - batch_start, 1, 4096)
        if lengths != [1] * (batch_end - batch_start) or feature.shape != expected_shape or not torch.isfinite(feature).all():
            raise RuntimeError(f"invalid text embedding for rows {batch_start}:{batch_end}")
        embeddings[batch_start:batch_end] = feature[:, 0].float().cpu().numpy().astype(np.float16)
        if batch_end % args.flush_every < args.batch_size or batch_end == len(records):
            embeddings.flush()
            progress_path.write_text(json.dumps({"completed": batch_end, "total": len(records)}) + "\n")
            elapsed = time.perf_counter() - started
            rate = (batch_end - start) / max(elapsed, 1e-9)
            print(json.dumps({"completed": batch_end, "total": len(records), "samples_per_second": rate}), flush=True)

    samples = [
        {"id": record["id"], "row": index, "caption": captions[index]}
        for index, record in enumerate(records)
    ]
    (output / "index.json").write_text(
        json.dumps({"format_version": 1, "dtype": "float16", "dim": 4096, "samples": samples}) + "\n"
    )
    (output / "complete.json").write_text(
        json.dumps({"status": "PASS", "samples": len(records), "device": args.device}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
