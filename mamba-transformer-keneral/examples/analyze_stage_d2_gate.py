"""Audit the preregistered D2 step-100 selection gate and causal panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from long_memory_protocol import paired_cluster_bootstrap  # noqa: E402


LONG_METRICS = (
    "root_post_w5_auc_cm_windows",
    "root_post_w5_slope_cm_per_window",
    "constraint_post_w5_auc_cm_windows",
    "constraint_post_w5_slope_cm_per_window",
    "root_final_window_cm",
    "constraint_final_window_cm",
)
SHORT_METRICS = ("mobench_root_ade_cm_mean", "constraint_error_cm_mean")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _paired(baseline: Path, candidate: Path, *, seed: int) -> dict:
    left = _jsonl(baseline)
    right = _jsonl(candidate)
    if [row["panel_id"] for row in left] != [row["panel_id"] for row in right]:
        raise ValueError(f"ordered panel IDs differ: {baseline} vs {candidate}")
    metrics = {}
    for metric in LONG_METRICS:
        pairs = []
        for before, after in zip(left, right):
            if (
                before["actor_uid"] != after["actor_uid"]
                or before["motion_id"] != after["motion_id"]
            ):
                raise ValueError("paired actor/motion metadata differs")
            pairs.append({
                "actor_uid": before["actor_uid"],
                "id": before["motion_id"],
                "difference": float(after[metric]) - float(before[metric]),
            })
        metrics[metric] = paired_cluster_bootstrap(
            pairs, value_key="difference", resamples=20_000, seed=seed
        )
    return {
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "samples": len(left),
        "difference_definition": "candidate_minus_baseline; negative favors candidate",
        "metrics": metrics,
    }


def _result_dir(root: Path, variant: str, panel: str) -> Path:
    return root / variant / panel


def _validate_pair(left: Path, right: Path) -> dict:
    a = _json(left / "result.json")
    b = _json(right / "result.json")
    fields = (
        "panel_sha256", "sample_ids", "motion_ids", "actor_uids",
        "diffusion_steps", "cfg_weight_text", "cfg_weight_constraint",
        "attention_window_chunks",
    )
    mismatches = [field for field in fields if a.get(field) != b.get(field)]
    if mismatches:
        raise ValueError(f"protocol mismatch {left} vs {right}: {mismatches}")
    return {"status": "PASS", "checked_fields": list(fields)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=100)
    parser.add_argument(
        "--eval-root",
        default=None,
    )
    parser.add_argument(
        "--old-control-root",
        default=(
            "/data4/zts/result/"
            "bones_seed_long_memory_stage_b_selection_eval_20260831/stage000"
        ),
    )
    parser.add_argument(
        "--full-train-root",
        default=None,
    )
    parser.add_argument(
        "--no-delta-train-root",
        default=None,
    )
    parser.add_argument("--learned-results", default=None)
    parser.add_argument("--parity-results", default=None)
    parser.add_argument("--results-output", default=None)
    parser.add_argument(
        "--causal-output", default=None
    )
    args = parser.parse_args()
    if args.stage < 100:
        parser.error("--stage must be at least 100 optimizer steps")

    eval_root = Path(
        args.eval_root
        or (
            "/data4/zts/result/stage_d2_bones_selection_eval_20260903"
            if args.stage == 100
            else "/data4/zts/result/stage_d2_bones_step250_selection_eval_20260903"
        )
    )
    old_root = Path(args.old_control_root)
    full_train = Path(
        args.full_train_root
        or (
            "/data4/zts/result/stage_d2_bones_step100_full_v2_20260903"
            if args.stage == 100
            else "/data4/zts/result/stage_d2_bones_step250_full_20260903"
        )
    )
    no_delta_train = Path(
        args.no_delta_train_root
        or (
            "/data4/zts/result/stage_d2_bones_step100_no_delta_v2_20260903"
            if args.stage == 100
            else "/data4/zts/result/stage_d2_bones_step250_no_delta_20260903"
        )
    )
    results_output = Path(
        args.results_output
        or REPO / f"docs/D2_STEP{args.stage}_RESULTS.json"
    )
    causal_output_path = Path(
        args.causal_output
        or REPO
        / (
            "docs/D2_CAUSAL_INTERVENTION_RESULTS.json"
            if args.stage == 100
            else "docs/D2_STEP250_CAUSAL_INTERVENTION_RESULTS.json"
        )
    )
    evaluation_manifest = _json(eval_root / "evaluation_manifest.json")
    analysis_provenance = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha(Path(__file__).resolve()),
    }
    learned = _json(
        Path(args.learned_results)
        if args.learned_results
        else REPO / "docs/D2_LEARNED_CAUSAL_MEMORY_RESULTS.json"
    )
    parity = _json(
        Path(args.parity_results)
        if args.parity_results
        else Path("/data4/zts/result/stage_d2_step0_native_parity_20260903/results.json")
    )

    panels = ("table4", "9s_in_horizon", "9s_out_of_horizon", "20s_in_horizon", "40s_in_horizon")
    aggregate = {}
    protocol_audit = {}
    for variant in ("d2_full", "d2_no_delta"):
        aggregate[variant] = {}
        for panel in panels:
            result = _json(_result_dir(eval_root, variant, panel) / "result.json")
            aggregate[variant][panel] = {
                key: result.get(key)
                for key in (
                    "samples", "frames", "total_seconds", "mobench_root_ade_cm_mean",
                    "constraint_error_cm_mean", "post_window5_root_ade_slope_cm_per_window",
                    "post_window5_constraint_slope_cm_per_window", "latency_seconds_per_sample",
                    "peak_gpu_memory_mib", "content_memory_norm_max",
                    "content_memory_bytes_max_per_layer",
                    "content_memory_read_residual_scale_max_abs",
                )
            }
            protocol_audit[f"{variant}:{panel}"] = _validate_pair(
                _result_dir(eval_root, "d2_full", panel),
                _result_dir(eval_root, variant, panel),
            )

    short_health = {}
    short_pass = True
    for variant in ("d2_full", "d2_no_delta"):
        short_health[variant] = {}
        for panel in ("table4", "9s_in_horizon"):
            baseline_dir = old_root / "hybrid_no_decay_carry" / panel
            candidate_dir = _result_dir(eval_root, variant, panel)
            protocol_audit[f"native:{variant}:{panel}"] = _validate_pair(
                baseline_dir, candidate_dir
            )
            baseline = _json(baseline_dir / "result.json")
            candidate = _json(candidate_dir / "result.json")
            values = {}
            for metric in SHORT_METRICS:
                before = float(baseline[metric])
                after = float(candidate[metric])
                degradation = (after - before) / max(abs(before), 1e-12)
                values[metric] = {
                    "native": before,
                    "candidate": after,
                    "relative_degradation": degradation,
                    "passed_le_5_percent": degradation <= 0.05,
                }
                short_pass = short_pass and degradation <= 0.05
            short_health[variant][panel] = values

    comparisons = {}
    full_40 = _result_dir(eval_root, "d2_full", "40s_in_horizon")
    for index, baseline in enumerate(("native_hybrid", "minimal_w4", "d2_no_delta")):
        baseline_40 = _result_dir(eval_root, baseline, "40s_in_horizon")
        protocol_audit[f"{baseline}:d2_full:40s"] = _validate_pair(
            baseline_40, full_40
        )
        comparisons[f"d2_full_minus_{baseline}"] = _paired(
            baseline_40 / "per_sample.jsonl",
            full_40 / "per_sample.jsonl",
            seed=20260903 + index,
        )

    native_metrics = comparisons["d2_full_minus_native_hybrid"]["metrics"]
    root_screen = any(
        native_metrics[key]["mean_difference"] < 0
        for key in (
            "root_post_w5_auc_cm_windows",
            "root_post_w5_slope_cm_per_window",
        )
    )
    constraint_screen = any(
        native_metrics[key]["mean_difference"] < 0
        for key in (
            "constraint_post_w5_auc_cm_windows",
            "constraint_post_w5_slope_cm_per_window",
        )
    )
    not_all_worse = not all(
        native_metrics[key]["mean_difference"] > 0 for key in LONG_METRICS[:4]
    )
    minimal_metrics = comparisons["d2_full_minus_minimal_w4"]["metrics"]
    minimal_screen = all(
        minimal_metrics[key]["mean_difference"] < 0
        for key in (
            "root_post_w5_auc_cm_windows",
            "constraint_post_w5_auc_cm_windows",
        )
    )

    training = {}
    trainability_pass = True
    for variant, root in (("d2_full", full_train), ("d2_no_delta", no_delta_train)):
        metrics = _json(root / "metrics.json")
        fractions = metrics["d2_gradient_update_fraction"]
        diagnostics = metrics["d2_last_memory_diagnostics"]
        passed = (
            bool(metrics["all_gradients_finite"])
            and all(float(value) >= 0.5 for value in fractions.values())
            and float(diagnostics["memory_slot_norm"]) > 0
            and math.isfinite(float(diagnostics["memory_slot_norm"]))
            and abs(float(diagnostics["read_residual_scale"])) > 0
        )
        trainability_pass = trainability_pass and passed
        training[variant] = {
            "checkpoint": str((root / "hybrid_humanml3d_ddp.pt").resolve()),
            "checkpoint_sha256": _sha(root / "hybrid_humanml3d_ddp.pt"),
            "trainer_status": metrics["status"],
            "trainer_status_explanation": (
                "informational random-minibatch loss trend heuristic; not the D2 preregistered gate"
            ),
            "first_50_mean_loss": metrics["first_50_mean_loss"],
            "last_50_mean_loss": metrics["last_50_mean_loss"],
            "gradient_update_fraction": fractions,
            "last_memory_diagnostics": diagnostics,
            "passed_trainability": passed,
        }

    interventions = {}
    causal_worse = False
    for index, name in enumerate(
        ("zero_delta", "permute_delta", "write_off", "readout_off", "permute_memory")
    ):
        candidate = _result_dir(eval_root, f"d2_full__{name}", "40s_in_horizon")
        protocol_audit[f"normal:{name}:40s"] = _validate_pair(full_40, candidate)
        comparison = _paired(
            full_40 / "per_sample.jsonl",
            candidate / "per_sample.jsonl",
            seed=20261000 + index,
        )
        interventions[name] = comparison
        if name in {"write_off", "readout_off", "permute_memory"}:
            causal_worse = causal_worse or any(
                comparison["metrics"][key]["mean_difference"] > 0
                for key in (
                    "root_post_w5_auc_cm_windows",
                    "constraint_post_w5_auc_cm_windows",
                )
            )

    permutation_audit = _jsonl(
        _result_dir(eval_root, "d2_full__permute_memory", "40s_in_horizon")
        / "state.jsonl.content_interventions.jsonl"
    )
    permutation_contract_pass = bool(permutation_audit) and all(
        row["completed_windows"] == 4
        and row["attention_checksum_before"] == row["attention_checksum_after"]
        and row["memory_checksum_before"] != row["memory_checksum_after"]
        for row in permutation_audit
    )

    delta_harm = any(
        interventions[name]["metrics"][key]["mean_difference"] > 0
        and interventions[name]["metrics"][key]["ci95"][0] > 0
        for name in ("zero_delta", "permute_delta")
        for key in LONG_METRICS[:4]
    )
    full_beats_no_delta = any(
        comparisons["d2_full_minus_d2_no_delta"]["metrics"][key]["ci95"][1] < 0
        for key in LONG_METRICS[:4]
    )
    delta_established = delta_harm or full_beats_no_delta
    delta_removal_improves = any(
        interventions[name]["metrics"][key]["mean_difference"] < 0
        and interventions[name]["metrics"][key]["ci95"][1] < 0
        for name in ("zero_delta", "permute_delta")
        for key in LONG_METRICS[:4]
    )

    step100_gate_checks = {
        "trainability": trainability_pass,
        "learned_causal_benchmark": learned["gate"]["status"] == "PASS",
        "step0_native_parity": parity["gate"]["status"] == "PASS",
        "short_horizon_relative_degradation_le_5_percent": short_pass,
        "root_slope_or_auc_better_than_native": root_screen,
        "constraint_slope_or_auc_better_than_native": constraint_screen,
        "four_primary_long_metrics_not_all_worse": not_all_worse,
        "root_and_constraint_auc_better_than_minimal_w4": minimal_screen,
        "content_causal_ablation_worse": causal_worse,
        "permutation_contract_and_nonidentical_rows": permutation_contract_pass,
    }
    step100_gate_status = "PASS" if all(step100_gate_checks.values()) else "FAIL"

    read_write_causal = any(
        interventions[name]["metrics"][key]["ci95"][0] > 0
        for name in ("write_off", "readout_off")
        for key in (
            "root_post_w5_auc_cm_windows",
            "constraint_post_w5_auc_cm_windows",
        )
    )
    trajectory_causal = all(
        interventions["permute_memory"]["metrics"][key]["ci95"][0] > 0
        for key in (
            "root_post_w5_auc_cm_windows",
            "constraint_post_w5_auc_cm_windows",
        )
    )
    stable_minimal_auc = all(
        minimal_metrics[key]["ci95"][1] < 0
        for key in (
            "root_post_w5_auc_cm_windows",
            "constraint_post_w5_auc_cm_windows",
        )
    )
    full_rows = _jsonl(full_40 / "per_sample.jsonl")
    nonpositive_post_w5_slopes = all(
        _mean(full_rows, key) <= 0
        for key in (
            "root_post_w5_slope_cm_per_window",
            "constraint_post_w5_slope_cm_per_window",
        )
    )
    final_gate_checks = {
        "step100_screening_contract_still_passes": step100_gate_status == "PASS",
        "d2_full_auc_better_than_minimal_w4_with_ci": stable_minimal_auc,
        "post_w5_root_and_constraint_slopes_nonpositive": nonpositive_post_w5_slopes,
        "trajectory_specific_permute_memory_causality_with_ci": trajectory_causal,
        "write_or_read_causality_with_ci": read_write_causal,
    }
    gate_checks = step100_gate_checks if args.stage == 100 else final_gate_checks
    gate_status = "PASS" if all(gate_checks.values()) else "FAIL"

    causal_output = {
        "schema_version": "d2_causal_intervention_v1",
        "stage": args.stage,
        "status": (
            "PASS"
            if read_write_causal and trajectory_causal and permutation_contract_pass
            else "FAIL"
        ),
        "checkpoint": evaluation_manifest["checkpoints"]["d2_full"],
        "checkpoint_sha256": evaluation_manifest["checkpoint_sha256"]["d2_full"],
        "protocol": evaluation_manifest,
        "analysis_provenance": analysis_provenance,
        "normal_per_sample": str((full_40 / "per_sample.jsonl").resolve()),
        "comparisons": interventions,
        "permutation_audit": permutation_audit,
        "content_write_or_read_mean_degradation_observed": causal_worse,
        "content_write_or_read_causality_established": read_write_causal,
        "trajectory_specific_behavioral_causality_established": trajectory_causal,
        "content_write_or_read_causality_with_ci_established": read_write_causal,
        "delta_state_independent_contribution_established": delta_established,
        "delta_removal_or_permutation_significantly_improves": delta_removal_improves,
    }
    step100_output = {
        "schema_version": f"d2_step{args.stage}_gate_v1",
        "stage": args.stage,
        "status": gate_status,
        "decision": (
            "CONTINUE_TO_STEP250"
            if args.stage == 100 and gate_status == "PASS"
            else "ELIGIBLE_FOR_STAGE_E"
            if args.stage == 250 and gate_status == "PASS"
            else "STOP"
        ),
        "protocol": evaluation_manifest,
        "analysis_provenance": analysis_provenance,
        "locked_final_actors_read": False,
        "training": training,
        "aggregate_results": aggregate,
        "short_horizon_health": short_health,
        "paired_actor_cluster_bootstrap_40s": comparisons,
        "protocol_audit": protocol_audit,
        "causal_results": str(causal_output_path.resolve()),
        "gate_checks": gate_checks,
        "step100_screening_checks": step100_gate_checks,
        "delta_state_independent_contribution_established": delta_established,
        "delta_removal_or_permutation_significantly_improves": delta_removal_improves,
        "claim_boundary": (
            "Trajectory-specific state-conditioned persistent content-memory "
            "write/read causality is established on the BONES selection panel; "
            "delta-state independent contribution is not established."
            if trajectory_causal and read_write_causal and not delta_established
            else "Trajectory-specific state-conditioned persistent content-memory "
            "write/read causality and delta-state variant evidence are established "
            "on the BONES selection panel."
            if trajectory_causal and read_write_causal and delta_established
            else "Persistent content-memory trainability is established, but the "
            "registered real-motion causal chain is incomplete."
        ),
    }
    for path, value in (
        (causal_output_path, causal_output),
        (results_output, step100_output),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": gate_status, "checks": gate_checks}, indent=2))
    return 0 if gate_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
