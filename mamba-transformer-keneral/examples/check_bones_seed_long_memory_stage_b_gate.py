"""Apply the preregistered BONES-SEED 100/250-step long-memory gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from long_memory_protocol import holm_adjust, paired_cluster_bootstrap  # noqa: E402


LONG_METRICS = (
    "root_post_w5_auc_cm_windows",
    "root_post_w5_slope_cm_per_window",
    "constraint_post_w5_auc_cm_windows",
    "constraint_post_w5_slope_cm_per_window",
)
REPORT_METRICS = (
    "root_ade_cm",
    "root_post_w5_auc_cm_windows",
    "root_post_w5_slope_cm_per_window",
    "root_final_window_cm",
    "constraint_error_cm",
    "constraint_post_w5_auc_cm_windows",
    "constraint_post_w5_slope_cm_per_window",
    "constraint_final_window_cm",
)
SHORT_METRICS = ("constraint_error_cm_mean", "mobench_root_ade_cm_mean")
RETENTION_CONFIGS = (
    "ardy",
    "minimal_w4",
    "minimal_w8",
    "minimal_w16",
    "minimal_parammatch",
    "hybrid_decay_carry",
    "hybrid_decay_no_carry",
    "hybrid_no_decay_carry",
    "hybrid_no_decay_no_carry",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _last_pretty_json(path: Path) -> dict:
    lines = path.read_text().splitlines()
    starts = [index for index, line in enumerate(lines) if line == "{"]
    for start in reversed(starts):
        try:
            value, _ = json.JSONDecoder().raw_decode("\n".join(lines[start:]))
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no final JSON summary in {path}")


def _paired(root: Path, stage: int, baseline: str, candidate: str, seconds: int) -> dict:
    stem = f"stage{stage:03d}"
    left = _jsonl(root / stem / baseline / f"{seconds}s_in_horizon/per_sample.jsonl")
    right = _jsonl(root / stem / candidate / f"{seconds}s_in_horizon/per_sample.jsonl")
    if [row["panel_id"] for row in left] != [row["panel_id"] for row in right]:
        raise ValueError(f"ordered panel IDs differ for {stem} {baseline} {candidate} {seconds}s")
    metrics = {}
    for metric in REPORT_METRICS:
        rows = []
        for before, after in zip(left, right):
            if before["actor_uid"] != after["actor_uid"] or before["motion_id"] != after["motion_id"]:
                raise ValueError("paired actor/motion metadata differs")
            rows.append({
                "actor_uid": before["actor_uid"],
                "id": before["motion_id"],
                "difference": float(after[metric]) - float(before[metric]),
            })
        metrics[metric] = paired_cluster_bootstrap(
            rows, value_key="difference", resamples=20_000,
            seed=20260831 + stage + seconds,
        )
    adjusted = holm_adjust([
        metrics[metric]["p_value_two_sided"] for metric in REPORT_METRICS
    ])
    for metric, p_value in zip(REPORT_METRICS, adjusted):
        metrics[metric]["holm_adjusted_p_all_reported_metrics"] = p_value
    return {
        "baseline": baseline,
        "candidate": candidate,
        "difference_definition": "candidate_minus_baseline; negative favors candidate",
        "metrics": metrics,
        "passed_direction": all(
            metrics[metric]["mean_difference"] < 0 for metric in LONG_METRICS
        ),
    }


def _short_retention(root: Path, stage: int) -> dict:
    result = {}
    for name in RETENTION_CONFIGS:
        panels = {}
        for panel in ("table4", "9s_in_horizon"):
            baseline = _json(root / "stage000" / name / panel / "result.json")
            candidate = _json(root / f"stage{stage:03d}" / name / panel / "result.json")
            metrics = {}
            for metric in SHORT_METRICS:
                before = float(baseline[metric])
                after = float(candidate[metric])
                relative_degradation = (after - before) / max(abs(before), 1e-12)
                metrics[metric] = {
                    "healthy_start": before,
                    "candidate": after,
                    "relative_degradation": relative_degradation,
                    "passed": relative_degradation <= 0.10,
                }
            panels[panel] = {
                "metrics": metrics,
                "passed": all(value["passed"] for value in metrics.values()),
            }
        result[name] = {
            "panels": panels,
            "passed": all(value["passed"] for value in panels.values()),
        }
    return {
        "configs": result,
        "passed": all(value["passed"] for value in result.values()),
    }


def _state_audit(root: Path, stage: int) -> dict:
    path = root / f"stage{stage:03d}/hybrid_decay_carry/40s_in_horizon/state.jsonl"
    records = _jsonl(path)
    finite = bool(records) and all(record["all_finite"] for record in records)
    bounded = True
    checked_series = 0
    by_batch: dict[int, list[dict]] = {}
    for record in records:
        by_batch.setdefault(int(record["batch_start"]), []).append(record)
    for batch_records in by_batch.values():
        mature = [record for record in batch_records if int(record["completed_windows"]) >= 4]
        if not mature:
            bounded = False
            continue
        for block in ("root", "body"):
            layer_count = len(mature[0]["blocks"][block])
            for layer in range(layer_count):
                sizes = {
                    int(record["blocks"][block][layer]["persistent_state_bytes"])
                    for record in mature
                }
                checked_series += 1
                bounded = bounded and len(sizes) == 1
    return {
        "records": len(records),
        "all_finite": finite,
        "persistent_bytes_constant_after_w4": bounded,
        "checked_block_layer_series": checked_series,
        "passed": finite and bounded and checked_series > 0,
    }


def _training_audit(training_root: Path, stage: int) -> dict:
    rows = {}
    for name in ("hybrid_decay_carry", "hybrid_decay_no_carry"):
        training_rows = [
            row for row in _jsonl(training_root / name / "train.jsonl")
            if "loss" in row and int(row["step"]) <= stage
        ]
        rollout_steps = sum(bool(row.get("differentiable_rollout")) for row in training_rows)
        final_metrics = _last_pretty_json(training_root / f"{name}_stage{stage}.log")
        rows[name] = {
            "logged_steps": len(training_rows),
            "last_logged_step": int(training_rows[-1]["step"]),
            "all_gradients_finite": bool(final_metrics["all_gradients_finite"]),
            "all_losses_finite": all(
                math.isfinite(float(row["loss"])) for row in training_rows
            ),
            "logged_differentiable_rollout_steps": rollout_steps,
            "differentiable_rollout_steps": int(
                final_metrics["differentiable_rollout_steps"]
            ),
            "checkpoint_step": stage,
        }
        rows[name]["passed"] = (
            rows[name]["last_logged_step"] == stage
            and rows[name]["all_gradients_finite"]
            and rows[name]["all_losses_finite"]
            and rows[name]["differentiable_rollout_steps"] > 0
        )
    return {"models": rows, "passed": all(row["passed"] for row in rows.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    root = Path(args.evaluation_root)
    training_root = Path(args.training_root)
    stages = {}
    for stage in (100, 250):
        long_vs_minimal = {
            f"{seconds}s": _paired(
                root, stage, "minimal_w4", "hybrid_decay_carry", seconds
            )
            for seconds in (20, 40)
        }
        causal = {}
        for intervention in ("fresh_each_window", "permute_after_w4"):
            candidate = f"hybrid_decay_carry__{intervention}"
            causal[intervention] = {
                f"{seconds}s": _paired(
                    root, stage, candidate, "hybrid_decay_carry", seconds
                )
                for seconds in (20, 40)
            }
        criteria = {
            "short_retention": _short_retention(root, stage),
            "hybrid_vs_minimal_direction": all(
                value["passed_direction"] for value in long_vs_minimal.values()
            ),
            "carry_vs_fresh_and_permute_direction": all(
                value["passed_direction"]
                for intervention in causal.values()
                for value in intervention.values()
            ),
            "state": _state_audit(root, stage),
            "training": _training_audit(training_root, stage),
        }
        passed = (
            criteria["short_retention"]["passed"]
            and criteria["hybrid_vs_minimal_direction"]
            and criteria["carry_vs_fresh_and_permute_direction"]
            and criteria["state"]["passed"]
            and criteria["training"]["passed"]
        )
        stages[str(stage)] = {
            "passed": passed,
            "criteria": criteria,
            "hybrid_decay_carry_vs_minimal_w4": long_vs_minimal,
            "same_checkpoint_causal_interventions": causal,
        }
    directions_consistent = all(stage["passed"] for stage in stages.values())
    summary = {
        "status": "PASS" if directions_consistent else "FAIL",
        "gate": "BONES-SEED preregistered 100/250-step mechanism gate",
        "stages": stages,
        "directions_consistent_and_both_stages_pass": directions_consistent,
        "next_stage_authorized": directions_consistent,
        "bootstrap_unit": "actor_then_motion_within_actor",
        "bootstrap_resamples": 20_000,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "stage_pass": {key: value["passed"] for key, value in stages.items()},
        "next_stage_authorized": summary["next_stage_authorized"],
    }, indent=2))
    return 1 if args.fail_on_gate and summary["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
