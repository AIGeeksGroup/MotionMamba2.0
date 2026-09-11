"""Validate Core27 -> HumanML3D-263 conversion on real test motions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))

from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep  # noqa: E402
from ardy.skeleton import CoreSkeleton27  # noqa: E402
from humanml3d_conversion import core27_to_humanml263  # noqa: E402
from humanml3d_official_evaluator import (  # noqa: E402
    HumanML3DOfficialEvaluator,
    frechet_distance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line]
    records = [record for record in records if not record["id"].startswith("M")]
    records = records[: args.samples]
    rep = ArdyMotionRep(CoreSkeleton27(), fps=20)
    converted, reference, lengths, used = [], [], [], []
    for record in records:
        original_path = Path(args.dataset) / "new_joint_vecs" / f"{record['id']}.npy"
        if not original_path.exists():
            continue
        features = torch.from_numpy(np.load(record["feature_path"])).float()
        joints = rep.inverse(features[None], is_normalized=False)["posed_joints"][0].numpy()
        rebuilt, _ = core27_to_humanml263(joints)
        original = np.load(original_path).astype(np.float32)
        length = min(len(rebuilt), len(original), 196) // 4 * 4
        if length < 40:
            continue
        converted.append(np.pad(rebuilt[:length], ((0, 196 - length), (0, 0))))
        reference.append(np.pad(original[:length], ((0, 196 - length), (0, 0))))
        lengths.append(length)
        used.append(record["id"])

    evaluator = HumanML3DOfficialEvaluator(args.assets, args.device)
    converted_embeddings, reference_embeddings = [], []
    for start in range(0, len(converted), 32):
        _, converted_embedding, _ = evaluator.embed(
            np.stack(converted[start:start + 32]), np.asarray(lengths[start:start + 32])
        )
        _, reference_embedding, _ = evaluator.embed(
            np.stack(reference[start:start + 32]), np.asarray(lengths[start:start + 32])
        )
        converted_embeddings.append(converted_embedding)
        reference_embeddings.append(reference_embedding)
    converted_embeddings = np.concatenate(converted_embeddings)
    reference_embeddings = np.concatenate(reference_embeddings)
    cosine = np.sum(converted_embeddings * reference_embeddings, axis=1) / (
        np.linalg.norm(converted_embeddings, axis=1)
        * np.linalg.norm(reference_embeddings, axis=1)
    ).clip(1e-8)
    result = {
        "status": "PASS" if len(used) >= args.samples else "FAIL",
        "samples": len(used),
        "fid_original_vs_core_roundtrip": frechet_distance(
            reference_embeddings, converted_embeddings
        ),
        "paired_embedding_cosine_mean": float(cosine.mean()),
        "paired_embedding_cosine_min": float(cosine.min()),
        "sample_ids": used,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

