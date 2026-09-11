"""Compare per-motion HiPHI and BONES Core27 kinematic QA distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    "foot slip left p95": ("foot_slip_left", "p95"),
    "foot slip right p95": ("foot_slip_right", "p95"),
    "joint jerk p99": ("joint_jerk", "p99"),
    "root speed p99": ("root_speed", "p99"),
    "root acceleration p99": ("root_accel", "p99"),
    "penetration minimum": ("penetration_depth_min",),
    "contact ratio": ("foot_contact_ratio",),
    "ground offset": ("ground_offset_m",),
}


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nested(record: dict, path: tuple[str, ...]):
    value = record.get("kinematic_qa", record.get("qa", {}).get("kinematic", {}))
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value)


def summarize(rows: list[dict], path: tuple[str, ...]) -> dict:
    values = np.asarray([value for record in rows if (value := nested(record, path)) is not None])
    if not len(values):
        return {"count": 0, "median": None, "p95": None}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def duration_summary(rows: list[dict]) -> dict:
    values = np.asarray([float(row["frames"]) / float(row["fps"]) for row in rows])
    return {"count": int(len(values)), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiphi-manifest", required=True)
    parser.add_argument("--bones-manifest", required=True)
    parser.add_argument(
        "--output",
        default="/data4/zts/datasets/motion_fusion/staging/hiphi_audit/hiphi_vs_bones_quality_report.md",
    )
    args = parser.parse_args()
    sources = {
        "HiPHI body": records(Path(args.hiphi_manifest)),
        "BONES-SEED": records(Path(args.bones_manifest)),
    }
    summary = {
        metric: {name: summarize(rows, path) for name, rows in sources.items()}
        for metric, path in METRICS.items()
    }
    summary["duration seconds"] = {name: duration_summary(rows) for name, rows in sources.items()}
    lines = [
        "# HiPHI vs BONES Quality Report",
        "",
        "Metrics are summarized per motion. Values are diagnostic flags, not automatic rejection thresholds.",
        "",
        "| Metric | HiPHI median | HiPHI p95 | BONES median | BONES p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in summary.items():
        hiphi = values["HiPHI body"]
        bones = values["BONES-SEED"]
        fmt = lambda value: "N/A" if value is None else f"{value:.6g}"
        lines.append(f"| {metric} | {fmt(hiphi['median'])} | {fmt(hiphi['p95'])} | {fmt(bones['median'])} | {fmt(bones['p95'])} |")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
