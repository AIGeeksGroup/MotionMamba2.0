"""Paired actor/motion-cluster analysis for frozen long-memory panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from long_memory_protocol import holm_adjust, paired_cluster_bootstrap  # noqa: E402


DEFAULT_METRICS = (
    "root_ade_cm",
    "root_post_w5_auc_cm_windows",
    "root_post_w5_slope_cm_per_window",
    "root_final_window_cm",
    "joint_position_error_cm",
    "joint_post_w5_auc_cm_windows",
    "joint_post_w5_slope_cm_per_window",
    "joint_final_window_cm",
    "constraint_error_cm",
    "constraint_post_w5_auc_cm_windows",
    "constraint_post_w5_slope_cm_per_window",
    "constraint_final_window_cm",
)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    baseline = _read(Path(args.baseline))
    candidate = _read(Path(args.candidate))
    baseline_ids = [row["panel_id"] for row in baseline]
    candidate_ids = [row["panel_id"] for row in candidate]
    if baseline_ids != candidate_ids:
        raise ValueError("ordered panel IDs differ between paired inputs")
    results = {}
    p_values = []
    available_metrics = []
    for metric in args.metrics:
        paired = []
        for left, right in zip(baseline, candidate):
            if metric not in left or metric not in right:
                continue
            if left["actor_uid"] != right["actor_uid"] or left["motion_id"] != right["motion_id"]:
                raise ValueError("paired actor/motion metadata differs")
            paired.append({
                "panel_id": left["panel_id"],
                "actor_uid": left["actor_uid"],
                "id": left["motion_id"],
                "difference": float(right[metric]) - float(left[metric]),
            })
        if not paired:
            continue
        result = paired_cluster_bootstrap(
            paired,
            value_key="difference",
            resamples=args.resamples,
            seed=args.seed,
        )
        result["paired_rows"] = paired
        results[metric] = result
        p_values.append(result["p_value_two_sided"])
        available_metrics.append(metric)
    adjusted = holm_adjust(p_values)
    for metric, value in zip(available_metrics, adjusted):
        results[metric]["holm_adjusted_p"] = value
    summary = {
        "status": "PASS",
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "ordered_panel_ids_equal": True,
        "samples": len(baseline),
        "metrics": results,
        "difference_definition": "candidate_minus_baseline; negative favors candidate",
        "bootstrap_unit": "actor_then_motion_within_actor",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value["mean_difference"] for key, value in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
