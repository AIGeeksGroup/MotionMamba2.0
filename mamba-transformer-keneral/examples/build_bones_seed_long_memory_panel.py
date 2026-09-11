"""Build actor-disjoint immutable BONES-SEED long-memory panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from long_memory_protocol import (  # noqa: E402
    assert_actor_disjoint,
    canonical_jsonl,
    canonical_sha256,
    file_sha256,
)

G1_GOAL_JOINTS = (0, 6, 13, 24, 32)


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _caption_map(text_cache: Path) -> dict[str, str]:
    index = json.loads((text_cache / "index.json").read_text())
    return {str(row["id"]): str(row["caption"]) for row in index["samples"]}


def _stable_joint(panel_id: str, goal_number: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{panel_id}:goal:{goal_number}".encode()).digest()
    return G1_GOAL_JOINTS[int.from_bytes(digest[:8], "little") % len(G1_GOAL_JOINTS)]


def _select_records(
    records: list[dict], *, minimum_frames: int, samples: int, seed: int
) -> list[dict]:
    eligible = [row for row in records if int(row["frames"]) >= minimum_frames]
    by_actor: dict[str, list[dict]] = {}
    for row in eligible:
        by_actor.setdefault(str(row["actor_uid"]), []).append(row)
    if not by_actor:
        raise ValueError(f"no records have at least {minimum_frames} frames")
    for rows in by_actor.values():
        rows.sort(key=lambda row: (str(row.get("source_id", row["id"])), str(row["id"])))
    actors = sorted(by_actor)
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    selected = []
    positions = {actor: 0 for actor in actors}
    while len(selected) < samples:
        progress = False
        for actor in actors:
            position = positions[actor]
            if position < len(by_actor[actor]):
                selected.append(by_actor[actor][position])
                positions[actor] += 1
                progress = True
                if len(selected) == samples:
                    break
        if not progress:
            break
    if len(selected) < samples:
        raise ValueError(f"only {len(selected)} eligible actor-balanced records, need {samples}")
    return selected


def _build_panel(
    records: list[dict], captions: dict[str, str], *, manifest_sha: str,
    split_name: str, stem: str, total_seconds: float, selection_seconds: float,
    samples: int, seed: int, fps: int = 25, history_frames: int = 24,
    horizon: int = 52,
) -> list[dict]:
    total_frames = 52 if stem == "table4" else int(round(total_seconds * fps))
    selection_frames = max(total_frames, int(round(selection_seconds * fps)))
    selected = _select_records(
        records, minimum_frames=selection_frames, samples=samples, seed=seed
    )
    rows = []
    for offset, record in enumerate(selected):
        augmentation_seed = seed + 1_000_003 * offset
        rng = np.random.default_rng(augmentation_seed)
        start = int(rng.integers(int(record["frames"]) - selection_frames + 1))
        generation_frames = total_frames if stem == "table4" else total_frames - history_frames
        if stem == "table4":
            protocol = "table4"
            goal_frames = [total_frames - 1]
        elif stem.endswith("in_horizon"):
            protocol = "in_horizon"
            goal_frames = list(range(horizon - 1, generation_frames, horizon))
            if not goal_frames or goal_frames[-1] != generation_frames - 1:
                goal_frames.append(generation_frames - 1)
        else:
            protocol = "out_of_horizon"
            goal_frames = [generation_frames - 1]
        panel_id = f"{split_name}:{record['id']}:{start}:{total_frames}:{protocol}"
        rows.append({
            "panel_id": panel_id,
            "id": record["id"],
            "actor_uid": record["actor_uid"],
            "source_id": record.get("source_id", record["id"]),
            "feature_path": record["feature_path"],
            "source_path": record.get("source_path"),
            "caption": captions[record["id"]],
            "augmentation_seed": augmentation_seed,
            "noise_seed": seed * 1_000_003 + offset,
            "crop_start": start,
            "crop_end": start + total_frames,
            "selection_span_frames": selection_frames,
            "total_frames": total_frames,
            "history_frames": 0 if stem == "table4" else history_frames,
            "horizon_frames": horizon,
            "protocol": protocol,
            "goal_frames": goal_frames,
            "goal_joints": [
                _stable_joint(panel_id, number, augmentation_seed)
                for number in range(len(goal_frames))
            ],
            "source_manifest_sha256": manifest_sha,
            "split_role": split_name,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--locked-final-manifest", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lock-output", required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--short-samples", type=int, default=64)
    parser.add_argument("--long-samples", type=int, default=20)
    args = parser.parse_args()

    paths = {
        "train": Path(args.train_manifest),
        "selection": Path(args.selection_manifest),
        "locked_final": Path(args.locked_final_manifest),
    }
    manifests = {name: _read_manifest(path) for name, path in paths.items()}
    actors = {
        name: sorted({str(row["actor_uid"]) for row in rows})
        for name, rows in manifests.items()
    }
    assert_actor_disjoint(*(item for item in actors.items()))
    captions = _caption_map(Path(args.text_cache))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = {
        "table4": (52 / 25, 52 / 25, args.short_samples),
        "9s_in_horizon": (9, 9, args.short_samples),
        "9s_out_of_horizon": (9, 9, args.short_samples),
        "20s_in_horizon": (20, 40, args.long_samples),
        "20s_out_of_horizon": (20, 40, args.long_samples),
        "40s_in_horizon": (40, 40, args.long_samples),
        "40s_out_of_horizon": (40, 40, args.long_samples),
    }
    panel_hashes = {}
    panel_rows = {}
    for split_offset, split_name in enumerate(("selection", "locked_final")):
        manifest_sha = file_sha256(paths[split_name])
        split_dir = output / split_name
        split_dir.mkdir(exist_ok=True)
        for stem, (duration, selection, count) in specs.items():
            rows = _build_panel(
                manifests[split_name], captions, manifest_sha=manifest_sha,
                split_name=split_name, stem=stem, total_seconds=duration,
                selection_seconds=selection, samples=count,
                seed=args.seed + split_offset * 10_000_019,
            )
            path = split_dir / f"{stem}.jsonl"
            path.write_text(canonical_jsonl(rows))
            panel_hashes[f"{split_name}/{stem}"] = canonical_sha256(rows)
            panel_rows[f"{split_name}/{stem}"] = len(rows)

    lock = {
        "protocol_version": "bones_seed_long_memory_v2",
        "dataset": "BONES-SEED-v004",
        "split_roles": {
            "selection": "actor-disjoint validation used for all choices",
            "locked_final": (
                "newly frozen confirmation set; not claimed as never-observed blind test"
            ),
        },
        "manifest_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "actor_set_sha256": {
            name: hashlib.sha256("\n".join(values).encode()).hexdigest()
            for name, values in actors.items()
        },
        "actor_counts": {name: len(values) for name, values in actors.items()},
        "panel_sha256": panel_hashes,
        "panel_rows": panel_rows,
        "fps": 25,
        "gt_prefix_frames": 24,
        "horizon_frames": 52,
        "durations_seconds": [9, 20, 40],
        "diffusion_steps": 10,
        "cfg": {"text": 2.0, "constraint": 2.0},
        "batch_size": 2,
        "dtype": "float32_model_bf16_kernel_dependent",
        "attention_window_chunks": 4,
        "attention_window_variants": [4, 8, 16],
        "mamba_decay": {"start_windows": 4, "half_life_windows": 4.0, "min_scale": 0.0},
        "training_seeds": [827321, 827322, 827323],
        "evaluation_seed": args.seed,
        "bootstrap_unit": "actor_then_motion_within_actor",
        "bootstrap_resamples": 20_000,
        "primary_metrics": [
            "root_post_w5_auc", "root_post_w5_slope", "root_final_window",
            "constraint_post_w5_auc", "constraint_post_w5_slope",
            "constraint_final_window",
        ],
        "secondary_multiple_comparison": "Holm",
        "short_retention_threshold": 0.10,
        "checkpoint_selection": "selection_only_then_locked_final_once",
        "locked_final_status": "unread",
    }
    lock_path = Path(args.lock_output)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
