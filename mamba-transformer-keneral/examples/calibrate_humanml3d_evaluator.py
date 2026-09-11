"""Calibrate the frozen original HumanML3D evaluator on Dataset ground truth."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from humanml3d_official_evaluator import HumanML3DOfficialEvaluator, frechet_distance, r_precision  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit eligible records for a fast evaluator smoke test.")
    parser.add_argument("--full-only", action="store_true",
                        help="Exclude partial-caption segments for diagnosis.")
    return parser.parse_args()


def records(root: Path, include_segments: bool = True):
    output = []
    for sample_id in (root / "test.txt").read_text().splitlines():
        path = root / "new_joint_vecs" / f"{sample_id}.npy"
        text_path = root / "texts" / f"{sample_id}.txt"
        if not path.exists() or not text_path.exists():
            continue
        motion = np.load(path)
        if not 40 <= len(motion) < 200:
            continue
        full = []
        for line in text_path.read_text().splitlines():
            caption, token_string, start, end = line.strip().split("#")
            start_value, end_value = float(start), float(end)
            # Match the original MDM loader: NaN tags are treated as a full caption.
            if not np.isfinite(start_value) or not np.isfinite(end_value):
                start_value, end_value = 0.0, 0.0
            item = (caption, token_string.split(), start_value, end_value)
            if item[2] == 0.0 and item[3] == 0.0:
                full.append(item)
            else:
                segment = motion[int(item[2] * 20):int(item[3] * 20)]
                if include_segments and 40 <= len(segment) < 200:
                    output.append((segment, [item]))
        if full:
            output.append((motion, full))
    return output


def view(data, seed):
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    order = np_rng.permutation(len(data))
    motions, lengths, tokens = [], [], []
    for index in order[: len(order) // 32 * 32]:
        motion, captions = data[int(index)]
        _, selected_tokens, _, _ = py_rng.choice(captions)
        length = len(motion) // 4 * 4
        if np_rng.choice(("single", "single", "double")) == "double":
            length -= 4
        start = py_rng.randint(0, len(motion) - length)
        motions.append(np.pad(motion[start:start + length], ((0, 196 - length), (0, 0))))
        lengths.append(length)
        tokens.append(selected_tokens)
    return np.stack(motions).astype(np.float32), np.asarray(lengths), tokens


def main():
    args = arguments()
    evaluator = HumanML3DOfficialEvaluator(args.assets, args.device)
    data = records(Path(args.dataset), include_segments=not args.full_only)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    replications = []
    reference_embeddings = None
    for replication in range(args.replications):
        motions, lengths, tokens = view(data, args.seed + replication)
        text_embeddings = []
        embeddings = []
        for start in range(0, len(motions), args.batch_size):
            text, motion, _ = evaluator.embed(motions[start:start + args.batch_size], lengths[start:start + args.batch_size], tokens[start:start + args.batch_size])
            text_embeddings.append(text)
            embeddings.append(motion)
        text_embeddings = np.concatenate(text_embeddings)
        embeddings = np.concatenate(embeddings)
        # The original evaluator computes retrieval against 32 candidates per
        # batch. Embedding batches may be larger for throughput, but the
        # metric's candidate-set size remains fixed at 32.
        counts = np.zeros(3, dtype=np.int64)
        for start in range(0, len(embeddings), 32):
            counts += r_precision(text_embeddings[start:start + 32], embeddings[start:start + 32])
        if reference_embeddings is None:
            reference_embeddings = embeddings
        # This script calibrates the Dataset row against itself, whose FID is
        # exactly zero. Cross-crop FID is a different statistic and is not part
        # of the HumanML3D table protocol.
        fid = 0.0
        replications.append({
            "replication": replication,
            "samples": len(motions),
            "r_precision": (counts / len(motions)).tolist(),
            "fid_to_replication_0": fid,
        })
        print(json.dumps(replications[-1]), flush=True)
    values = np.asarray([item["r_precision"] for item in replications])
    fid_values = np.asarray([item["fid_to_replication_0"] for item in replications])
    result = {
        "protocol": "original_humanml3d_dataset_calibration",
        "batch_size": args.batch_size,
        "records": len(data),
        "replications": replications,
        "r_precision_mean": values.mean(0).tolist(),
        "r_precision_ci95": (1.96 * values.std(0) / np.sqrt(len(values))).tolist(),
        "fid_mean": float(fid_values.mean()),
        "fid_ci95": float(1.96 * fid_values.std() / np.sqrt(len(fid_values))),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
