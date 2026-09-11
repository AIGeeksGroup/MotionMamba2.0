"""Apply the predefined 1k generated-history gate and paired slope audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ("original", "minimal", "hybrid")
METRICS = ("constraint_error_cm_mean", "mobench_root_ade_cm_mean")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sample_root_slopes(path: Path, horizon: int, start_window: int = 6) -> np.ndarray:
    artifact = np.load(path)
    error = np.linalg.norm(
        artifact["generated_root"] - artifact["target_root"], axis=-1
    ) * 100.0
    windows = np.stack(
        [
            error[:, start:start + horizon].mean(axis=1)
            for start in range(0, error.shape[1], horizon)
        ],
        axis=1,
    )
    tail = windows[:, start_window - 1:]
    if tail.shape[1] < 2:
        raise ValueError("not enough post-window-5 observations for a slope")
    x = np.arange(start_window, start_window + tail.shape[1], dtype=np.float64)
    x -= x.mean()
    return (tail * x[None]).sum(axis=1) / np.square(x).sum()


def paired_bootstrap_gap(
    hybrid: np.ndarray, minimal: np.ndarray, *, seed: int
) -> dict:
    if hybrid.shape != minimal.shape:
        raise ValueError("paired slope arrays differ in shape")
    gap = hybrid - minimal
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gap), size=(20_000, len(gap)))
    bootstrap = gap[indices].mean(axis=1)
    return {
        "mean_cm_per_window": float(gap.mean()),
        "ci95_cm_per_window": [
            float(value) for value in np.percentile(bootstrap, [2.5, 97.5])
        ],
        "hybrid_better_fraction": float(np.mean(gap < 0)),
    }


def state_audit(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    layer_rows = [
        (record, block, layer)
        for record in records
        for block, layers in record["blocks"].items()
        for layer in layers
        if layer["mamba_state_norm"] is not None
    ]
    by_series: dict[tuple[int, str, int], list[float]] = {}
    for record, block, layer in layer_rows:
        key = (int(record["batch_start"]), block, int(layer["layer"]))
        by_series.setdefault(key, []).append(float(layer["mamba_state_norm"]))
    growth = [
        values[-1] / max(values[0], 1e-8)
        for values in by_series.values()
        if len(values) >= 2
    ]
    norms = [float(layer["mamba_state_norm"]) for _, _, layer in layer_rows]
    drifts = [
        float(layer["mamba_state_drift_norm"])
        for _, _, layer in layer_rows
        if layer["mamba_state_drift_norm"] is not None
    ]
    return {
        "records": len(records),
        "all_finite": bool(records) and all(record["all_finite"] for record in records),
        "max_state_norm": max(norms) if norms else None,
        "max_state_drift_norm": max(drifts) if drifts else None,
        "max_final_to_initial_norm_ratio": max(growth) if growth else None,
        "no_explosion": bool(growth) and np.isfinite(growth).all() and max(growth) <= 10.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-short-degradation", type=float, default=0.10)
    parser.add_argument(
        "--baseline-table4",
        default="/data4/zts/result/bones_seed_ardy_retrained_dual_eval_20260826/hybrid_table4.json",
    )
    parser.add_argument(
        "--baseline-9s-root",
        default="/data4/zts/result/bones_seed_ardy_retrained_official_table5_20260826",
    )
    args = parser.parse_args()
    eval_root = Path(args.eval_root)
    training_root = Path(args.training_root)

    training = {
        mode: load_json(training_root / mode / f"metrics_step_{args.stage:06d}.json")
        for mode in MODES
    }
    training_healthy = all(
        row["all_gradients_finite"]
        and np.isfinite(row["final_eval_loss"])
        and row["root_output_weight_norm"] >= 2.0
        for row in training.values()
    )
    matched_draws = (
        len({json.dumps(row["rollout_window_counts"], sort_keys=True) for row in training.values()}) == 1
        and len({row["generated_history_steps"] for row in training.values()}) == 1
    )

    retention = {}
    retention_pass = True
    baseline_paths = {
        "table4": Path(args.baseline_table4),
        "9s_in_horizon": Path(args.baseline_9s_root) / "hybrid_9s_in_horizon.json",
        "9s_out_of_horizon": Path(args.baseline_9s_root) / "hybrid_9s_out_of_horizon.json",
    }
    for stem, baseline_path in baseline_paths.items():
        baseline = load_json(baseline_path)
        current = load_json(eval_root / f"hybrid_{stem}.json")
        retention[stem] = {}
        for metric in METRICS:
            ratio = current[metric] / max(baseline[metric], 1e-8) - 1.0
            passed = ratio <= args.max_short_degradation
            retention[stem][metric] = {
                "baseline": baseline[metric],
                "current": current[metric],
                "relative_change": ratio,
                "passed": passed,
            }
            retention_pass = retention_pass and passed

    long_horizon = {}
    long_pass = True
    for seconds in (20, 40):
        hybrid = load_json(eval_root / f"hybrid_{seconds}s_in_horizon.json")
        minimal = load_json(eval_root / f"minimal_{seconds}s_in_horizon.json")
        horizon = int(hybrid["generation_horizon_frames"])
        bootstrap = paired_bootstrap_gap(
            sample_root_slopes(eval_root / f"hybrid_{seconds}s_in_horizon.npz", horizon),
            sample_root_slopes(eval_root / f"minimal_{seconds}s_in_horizon.npz", horizon),
            seed=20260827 + seconds,
        )
        root_better = (
            hybrid["post_window5_root_ade_slope_cm_per_window"]
            < minimal["post_window5_root_ade_slope_cm_per_window"]
        )
        hybrid_constraint = hybrid["post_window5_constraint_slope_cm_per_window"]
        minimal_constraint = minimal["post_window5_constraint_slope_cm_per_window"]
        constraint_better = (
            hybrid_constraint is not None
            and minimal_constraint is not None
            and hybrid_constraint < minimal_constraint
        )
        long_horizon[f"{seconds}s"] = {
            "hybrid_root_slope": hybrid["post_window5_root_ade_slope_cm_per_window"],
            "minimal_root_slope": minimal["post_window5_root_ade_slope_cm_per_window"],
            "hybrid_constraint_slope": hybrid_constraint,
            "minimal_constraint_slope": minimal_constraint,
            "root_slope_better": root_better,
            "constraint_slope_better": constraint_better,
            "paired_root_slope_gap": bootstrap,
        }
        long_pass = long_pass and root_better and constraint_better

    state = state_audit(eval_root / "hybrid_40s_in_horizon.state.jsonl")
    passed = bool(
        training_healthy and matched_draws and retention_pass and long_pass
        and state["all_finite"] and state["no_explosion"]
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "stage": args.stage,
        "continue_to_5k": passed and args.stage == 1000,
        "criteria": {
            "training_healthy": training_healthy,
            "matched_draws": matched_draws,
            "short_retention": retention_pass,
            "hybrid_long_slopes_better_than_minimal": long_pass,
            "state_finite": state["all_finite"],
            "no_state_norm_explosion": state["no_explosion"],
        },
        "short_retention": retention,
        "long_horizon": long_horizon,
        "state_audit": state,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
