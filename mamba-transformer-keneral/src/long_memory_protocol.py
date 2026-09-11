"""Immutable long-memory panels and clustered paired statistics."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def canonical_jsonl(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def canonical_sha256(rows: Sequence[dict]) -> str:
    return hashlib.sha256(canonical_jsonl(rows).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_panel(path: str | Path, *, expected_sha256: str | None = None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("panel is empty")
    panel_ids = [row.get("panel_id") for row in rows]
    if any(not value for value in panel_ids) or len(set(panel_ids)) != len(panel_ids):
        raise ValueError("panel_id must be present and unique")
    required = {
        "panel_id", "id", "actor_uid", "crop_start", "crop_end", "caption",
        "goal_frames", "goal_joints", "augmentation_seed", "noise_seed",
        "source_manifest_sha256",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"panel row {index} is missing {sorted(missing)}")
        if int(row["crop_start"]) < 0 or int(row["crop_end"]) <= int(row["crop_start"]):
            raise ValueError(f"panel row {index} has an invalid crop")
    actual = canonical_sha256(rows)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"panel hash mismatch: {actual} != {expected_sha256}")
    return rows


def assert_actor_disjoint(*named_actor_sets: tuple[str, Iterable[str]]) -> None:
    materialized = [(name, set(values)) for name, values in named_actor_sets]
    for index, (left_name, left) in enumerate(materialized):
        for right_name, right in materialized[index + 1 :]:
            overlap = sorted(left & right)
            if overlap:
                raise ValueError(
                    f"actor leakage between {left_name} and {right_name}: {overlap[:8]}"
                )


def window_summary(values: Sequence[float], *, start_index: int = 5) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) <= start_index:
        raise ValueError("window values do not cover the post-window-5 region")
    tail = array[start_index:]
    x = np.arange(len(tail), dtype=np.float64)
    slope = float(np.polyfit(x, tail, 1)[0]) if len(tail) >= 2 else 0.0
    return {
        "mean": float(array.mean()),
        "post_w5_auc": float(np.trapezoid(tail)),
        "post_w5_slope": slope,
        "final": float(array[-1]),
    }


def first_threshold_window(values: Sequence[float], threshold: float) -> int | None:
    for index, value in enumerate(values, start=1):
        if float(value) > threshold:
            return index
    return None


def paired_cluster_bootstrap(
    rows: Sequence[dict],
    *,
    value_key: str,
    actor_key: str = "actor_uid",
    motion_key: str = "id",
    resamples: int = 20_000,
    seed: int = 20260831,
) -> dict[str, float | list[float]]:
    """Bootstrap paired differences, resampling actors then motions."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError("bootstrap input contains a non-finite value")
        grouped[str(row[actor_key])][str(row[motion_key])].append(value)
    actors = sorted(grouped)
    if len(actors) < 2:
        raise ValueError("cluster bootstrap requires at least two actors")
    actor_values = {
        actor: np.asarray(
            [np.mean(grouped[actor][motion]) for motion in sorted(grouped[actor])],
            dtype=np.float64,
        )
        for actor in actors
    }
    observed_actor_means = np.asarray([values.mean() for values in actor_values.values()])
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        selected = rng.integers(0, len(actors), size=len(actors))
        means = []
        for actor_index in selected:
            values = actor_values[actors[int(actor_index)]]
            motion_draw = rng.integers(0, len(values), size=len(values))
            means.append(float(values[motion_draw].mean()))
        samples[draw] = np.mean(means)
    mean = float(observed_actor_means.mean())
    std = float(observed_actor_means.std(ddof=1))
    return {
        "mean_difference": mean,
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "effect_size": mean / max(std, np.finfo(np.float64).eps),
        "win_fraction": float(np.mean(observed_actor_means < 0.0)),
        "p_value_two_sided": float(
            min(1.0, 2.0 * min(np.mean(samples <= 0.0), np.mean(samples >= 0.0)))
        ),
        "actors": len(actors),
        "motions": int(sum(len(values) for values in actor_values.values())),
        "resamples": resamples,
        "seed": seed,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a one-dimensional sequence in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def verify_locked_final_guard(
    frozen_config: str | Path,
    *,
    checkpoint: str | Path,
    panel: str | Path,
    protocol_lock: str | Path,
    completion_marker: str | Path,
) -> dict:
    frozen = json.loads(Path(frozen_config).read_text())
    if frozen.get("status") != "FROZEN":
        raise ValueError("locked-final config is not FROZEN")
    marker = Path(completion_marker)
    if marker.exists():
        raise FileExistsError(f"locked-final evaluation already completed: {marker}")
    checkpoint_sha = file_sha256(checkpoint)
    allowed = frozen.get("checkpoint_sha256", {})
    if checkpoint_sha not in set(allowed.values()):
        raise ValueError("checkpoint is absent from FROZEN_CONFIG")
    if file_sha256(panel) not in set(frozen.get("panel_file_sha256", {}).values()):
        raise ValueError("panel is absent from FROZEN_CONFIG")
    if file_sha256(protocol_lock) != frozen.get("protocol_lock_sha256"):
        raise ValueError("protocol lock differs from FROZEN_CONFIG")
    return frozen
