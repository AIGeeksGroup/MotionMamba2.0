#!/usr/bin/env python3
"""Generate the final four-model 40 s out-of-horizon main figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/data3/zts/MotionMamba2.0/mamba-transformer-keneral")
OUT = Path("/data4/zts/result/stage_d2rl_frozen_5k_final_eval_20260906")
FIGURES = ROOT / "docs" / "figures"
MODELS = {
    "ARDY": ("ardy", "#4c4c4c", "o"),
    "Minimal-W4": ("minimal_w4", "#d55e00", "s"),
    "Native Hybrid": ("native_hybrid", "#0072b2", "^"),
    "D2R-L": ("d2rl_5k", "#009e73", "D"),
}


def load() -> dict[str, dict]:
    results = {}
    panel = None
    sample_ids = None
    for label, (directory, _, _) in MODELS.items():
        path = OUT / directory / "40s_out_of_horizon" / "result.json"
        result = json.loads(path.read_text())
        if result.get("status") != "PASS":
            raise ValueError(f"{path} is not PASS")
        if result.get("protocol") != "out_of_horizon" or result.get("total_seconds") != 40.0:
            raise ValueError(f"{path} is not the 40 s out-of-horizon protocol")
        if result.get("generation_windows") != 19 or result.get("samples") != 16:
            raise ValueError(f"{path} has unexpected panel/window count")
        if panel is None:
            panel = result["panel_sha256"]
            sample_ids = result["sample_ids"]
        if result["panel_sha256"] != panel or result["sample_ids"] != sample_ids:
            raise ValueError(f"panel mismatch in {path}")
        results[label] = result
    return results


def save_curve(results: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for label, (_, color, marker) in MODELS.items():
        values = results[label]["root_ade_cm_mean_by_window"]
        if len(values) != 19 or any(v is None for v in values):
            raise ValueError(f"incomplete Root ADE series for {label}")
        ax.plot(range(1, 20), values, color=color, marker=marker, markersize=4,
                linewidth=1.8, label=label)
    ax.axvline(4.5, color="black", linestyle="--", linewidth=1.2)
    ax.text(4.65, 0.98, "W4 eviction boundary", transform=ax.get_xaxis_transform(), va="top")
    ax.set_title("40 s out-of-horizon rollout error")
    ax.set_xlabel("Rollout window (52 frames / 2.08 s per full window)")
    ax.set_ylabel("Root ADE (cm)")
    ax.set_xticks(range(1, 20, 2))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_four_model_root_curve_out_of_horizon.png", dpi=220)
    fig.savefig(FIGURES / "final_four_model_root_curve_out_of_horizon.pdf")
    plt.close(fig)


def save_constraint_endpoint(results: dict[str, dict]) -> None:
    # Out-of-horizon places one goal at frame 975, so this metric is defined only at W19.
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for label, (_, color, marker) in MODELS.items():
        value = results[label]["constraint_error_cm_mean_by_window"]
        if len(value) != 19 or any(v is not None for v in value[:-1]) or value[-1] is None:
            raise ValueError(f"unexpected out-of-horizon constraint series for {label}")
        ax.scatter([19], [value[-1]], color=color, marker=marker, s=50, label=label)
    ax.axvline(4.5, color="black", linestyle="--", linewidth=1.2)
    ax.text(4.65, 0.98, "W4 eviction boundary", transform=ax.get_xaxis_transform(), va="top")
    ax.set_title("40 s out-of-horizon final-goal constraint error")
    ax.set_xlabel("Rollout window (only W19 has a target under this protocol)")
    ax.set_ylabel("Constraint error at W19 (cm)")
    ax.set_xlim(0.5, 19.5)
    ax.set_xticks(range(1, 20, 2))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_four_model_constraint_curve_out_of_horizon.png", dpi=220)
    fig.savefig(FIGURES / "final_four_model_constraint_curve_out_of_horizon.pdf")
    plt.close(fig)


def write_summary(results: dict[str, dict]) -> None:
    payload = {
        "status": "COMPLETE",
        "dataset": "BONES-SEED-v004",
        "protocol": "40s_out_of_horizon",
        "panel_sha256": next(iter(results.values()))["panel_sha256"],
        "samples": 16,
        "generation_windows": 19,
        "constraint_metric_note": "Only W19 has a target in out-of-horizon; W1-W18 are null by design.",
        "rows": [
            {
                "model": label,
                "checkpoint": result["checkpoint"],
                "root_ade_cm_mean_by_window": result["root_ade_cm_mean_by_window"],
                "constraint_error_cm_mean_by_window": result["constraint_error_cm_mean_by_window"],
                "root_ade_final_window_cm": result["root_ade_cm_mean_by_window"][-1],
                "constraint_error_final_window_cm": result["constraint_error_cm_mean_by_window"][-1],
            }
            for label, result in results.items()
        ],
    }
    (ROOT / "docs" / "FINAL_40S_OUT_OF_HORIZON_RESULTS.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = load()
    save_curve(results)
    write_summary(results)


if __name__ == "__main__":
    main()
