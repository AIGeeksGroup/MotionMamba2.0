#!/usr/bin/env python3
"""Plot the four available out-of-horizon final-goal constraint points."""

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
POINTS = (("table4", 2.08), ("9s_out_of_horizon", 9.0), ("20s_out_of_horizon", 20.0), ("40s_out_of_horizon", 40.0))


def load_points() -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for label, (directory, _, _) in MODELS.items():
        row = []
        for panel, _ in POINTS:
            path = OUT / directory / panel / "result.json"
            result = json.loads(path.read_text())
            if result.get("status") != "PASS":
                raise ValueError(f"{path} is not PASS")
            if result.get("constraint_error_cm_mean") is None:
                raise ValueError(f"missing constraint metric in {path}")
            row.append(float(result["constraint_error_cm_mean"]))
        values[label] = row
    return values


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    values = load_points()
    x = [point[1] for point in POINTS]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    label_offsets = {"ARDY": (0, 10), "Minimal-W4": (0, 10), "Native Hybrid": (-18, -16), "D2R-L": (18, -16)}
    for label, (_, color, marker) in MODELS.items():
        y = values[label]
        ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=1.8, label=label)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=label_offsets[label],
                        ha="center", fontsize=8, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["W1\n2.08 s", "9 s", "20 s", "40 s"])
    ax.set_xlabel("Out-of-horizon evaluation duration")
    ax.set_ylabel("Final-goal constraint error (cm, log scale)")
    ax.set_title("Constraint-error growth with out-of-horizon duration")
    all_values = [value for row in values.values() for value in row]
    ax.set_ylim(min(all_values) * 0.65, max(all_values) * 1.25)
    ax.grid(which="both", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for stem in (
        "final_four_model_constraint_curve_out_of_horizon",
        "final_four_model_constraint_duration_curve_out_of_horizon",
    ):
        fig.savefig(FIGURES / f"{stem}.png", dpi=220)
        fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)

    payload = {
        "status": "COMPLETE",
        "dataset": "BONES-SEED-v004",
        "metric": "constraint_error_cm_mean",
        "protocol": "out_of_horizon endpoint across durations",
        "durations_seconds": [p[1] for p in POINTS],
        "panel_note": "W1/9s use 64-sample panels; 20s/40s use the 16-sample selection panels.",
        "rows": [
            {"model": label, "values_cm": values[label], "source_panels": [p[0] for p in POINTS]}
            for label in MODELS
        ],
    }
    (ROOT / "docs" / "FINAL_CONSTRAINT_DURATION_RESULTS.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
