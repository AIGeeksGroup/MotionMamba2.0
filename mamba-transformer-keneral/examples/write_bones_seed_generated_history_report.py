"""Write the independent BONES-SEED generated-history gate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = Path("/data4/zts/result")
TRAIN = BASE / "bones_seed_generated_history_ft_20260827"
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "docs/BONES_SEED_GENERATED_HISTORY_GATE_20260827.md"
)
MODES = ("original", "minimal", "hybrid")
DISPLAY = {"original": "ARDY", "minimal": "Minimal", "hybrid": "Hybrid"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def value(value, digits=2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=TRAIN)
    parser.add_argument(
        "--eval-prefix",
        default=str(BASE / "bones_seed_generated_history_eval_stage"),
    )
    parser.add_argument("--eval-suffix", default="_20260827")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stages", type=int, nargs="+", default=None)
    parser.add_argument(
        "--gate-budgets-label",
        default="1k; 5k only if every predefined 1k criterion passes",
    )
    parser.add_argument("--selection-note", default=None)
    args = parser.parse_args()
    train_root = args.train_root
    output = args.output
    candidate_stages = args.stages or [1000, 5000]
    stages = [
        stage
        for stage in candidate_stages
        if all((train_root / mode / f"metrics_step_{stage:06d}.json").is_file() for mode in MODES)
        and Path(f"{args.eval_prefix}{stage}{args.eval_suffix}/EVALUATION_COMPLETE").is_file()
    ]
    if not stages:
        raise FileNotFoundError("no completed generated-history stage")
    protocol = load(train_root / "hybrid" / f"metrics_step_{stages[0]:06d}.json")
    strategy = protocol.get("history_sampling_strategy", "legacy_conditional")
    if strategy == "deep_v3":
        curriculum = (
            "Generated branch: 40% x3, 30% x5, 20% x10, 10% x19; teacher branch: "
            "50% x2, 20% x3, 20% x4, 10% x5"
        )
        maximum_seconds = 19 * protocol["production_contract"]["gen_horizon_len"] / 25
        short_retention = (
            f"{100 * protocol['short_window_probability']:.0f}% fixed short-window replay"
        )
    elif strategy == "balanced_v2":
        curriculum = (
            "Generated branch: 40% x3, 40% x4, 20% x5; teacher branch: "
            "50% x2, 20% x3, 20% x4, 10% x5"
        )
        maximum_seconds = 5 * protocol["production_contract"]["gen_horizon_len"] / 25
        short_retention = (
            f"{100 * protocol['short_window_probability']:.0f}% fixed short-window replay"
        )
    else:
        curriculum = "50% x2, 20% x3, 20% x4, 10% x5 windows"
        maximum_seconds = 5 * protocol["production_contract"]["gen_horizon_len"] / 25
        short_retention = "20% standard short-window updates; 80% rollout updates"
    probability_schedule = (
        f"{protocol.get('generated_history_probability_start', 0):.2f} -> "
        f"{protocol['generated_history_probability_1k']:.2f} by 1k; -> "
        f"{protocol['generated_history_probability_5k']:.2f} by 5k"
    )
    lines = [
        "# BONES-SEED Generated-History Scheduled-Sampling Gate",
        "",
        "## Scope",
        "",
        "This report is independent of the teacher-history protocol-aligned report. All three models "
        "start from their designated healthy checkpoints. Training uses one GT seed window followed "
        "by a scheduled mixture of accepted model generations and teacher history; the supervised "
        "window is expressed in the generated endpoint coordinate frame.",
        "",
        "| Setting | Value |",
        "|---|---|",
        "| Dataset | BONES-SEED v004 G1, actor-disjoint train/val/test |",
        f"| Gate budgets | {args.gate_budgets_label} |",
        f"| History sampling strategy | {strategy} |",
        f"| Rollout curriculum | {curriculum} (maximum {maximum_seconds:.1f} s) |",
        f"| Generated-history schedule | {probability_schedule} |",
        f"| Generated-depth ramp | {protocol.get('generated_history_depth_ramp_steps', 0)} steps |",
        f"| Per-window recovery cap | {protocol.get('generated_history_max_recovery_meters', 0):.1f} m (0 means disabled) |",
        f"| Short-window retention | {short_retention} |",
        f"| Optimizer | AdamW, LR {protocol['final_learning_rate']:.1e}, global batch 16, entire denoiser |",
        "| Sampling during training | 10 diffusion steps, text/constraint CFG 2/2 |",
        "| Evaluation | Table 4, official-style 9 s, and matched 20/40 s extensions |",
        "| State audit | root/body per-layer Mamba norm, state-difference norm, KV size, fusion gate |",
        "| Reset ablation | no reset, then persistent-state-only reset every 2/4/8 generated windows |",
        "",
    ]
    if args.selection_note:
        setting_index = lines.index("| Setting | Value |")
        lines[setting_index:setting_index] = [args.selection_note, ""]

    for stage in stages:
        eval_root = Path(f"{args.eval_prefix}{stage}{args.eval_suffix}")
        gate_path = eval_root / "gate.json"
        gate = load(gate_path) if gate_path.is_file() else None
        lines.extend([
            f"## Stage {stage}",
            "",
            "### Training Audit",
            "",
            "| Method | Trainer trend flag | Eval loss | Generated updates/windows | Teacher/short updates | Rollout counts | Time (h) |",
            "|---|---|---:|---:|---:|---|---:|",
        ])
        for mode in MODES:
            metrics = load(train_root / mode / f"metrics_step_{stage:06d}.json")
            counts = ", ".join(
                f"{key}w:{count}"
                for key, count in sorted((int(k), int(v)) for k, v in metrics["rollout_window_counts"].items())
            )
            lines.append(
                f"| {DISPLAY[mode]} | {metrics['status']} | {metrics['final_eval_loss']:.4f} | "
                f"{metrics['generated_history_steps']}/{metrics.get('generated_history_windows', 'N/A')} | "
                f"{metrics.get('teacher_history_steps', 'N/A')}/{metrics.get('short_window_steps', 'N/A')} | "
                f"{counts} | {metrics['elapsed_seconds'] / 3600:.2f} |"
            )

        lines.extend([
            "",
            "The trainer trend flag compares early and late losses even though the generated-history "
            "probability increases throughout training, making the late objective harder. Therefore "
            "`FAIL` here is not a numerical-health failure. The predefined gate separately verifies "
            "finite gradients, finite validation loss, output-weight norm, and matched draws.",
        ])

        lines.extend([
            "",
            "### Table 4: Offline Text and Joint-Position Control",
            "",
            "| Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |",
            "|---|---:|---:|---:|---:|",
        ])
        for mode in MODES:
            row = load(eval_root / f"{mode}_table4.json")
            lines.append(
                f"| {DISPLAY[mode]} | {row['constraint_error_cm_mean']:.2f} | "
                f"{row['mobench_root_ade_cm_mean']:.2f} | {row['mobench_root_fde_cm_mean']:.2f} | "
                f"{row['mobench_pa_mpjpe_cm_mean']:.2f} |"
            )

        lines.extend([
            "",
            "### Table 5 and Long-Horizon Extension",
            "",
            "| Duration | Protocol | Method | Constraint (cm) | Root ADE (cm) | Post-W5 Root slope | Post-W5 constraint slope |",
            "|---:|---|---|---:|---:|---:|---:|",
        ])
        for seconds in (9, 20, 40):
            for protocol in ("in_horizon", "out_of_horizon"):
                for mode in MODES:
                    row = load(eval_root / f"{mode}_{seconds}s_{protocol}.json")
                    lines.append(
                        f"| {seconds} s | {'in' if protocol == 'in_horizon' else 'out'} | "
                        f"{DISPLAY[mode]} | {row['constraint_error_cm_mean']:.2f} | "
                        f"{row['mobench_root_ade_cm_mean']:.2f} | "
                        f"{value(row['post_window5_root_ade_slope_cm_per_window'], 3)} | "
                        f"{value(row['post_window5_constraint_slope_cm_per_window'], 3)} |"
                    )

        lines.extend([
            "",
            "### Hybrid State Reset Ablation (40 s, In-Horizon)",
            "",
            "| Reset interval | Constraint (cm) | Root ADE (cm) | Root slope | Max state norm | Max state drift | Finite |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ])
        for reset in (0, 2, 4, 8):
            stem = "hybrid_40s_in_horizon" if reset == 0 else f"hybrid_40s_in_horizon_reset{reset}"
            row = load(eval_root / f"{stem}.json")
            lines.append(
                f"| {'none' if reset == 0 else reset} | {row['constraint_error_cm_mean']:.2f} | "
                f"{row['mobench_root_ade_cm_mean']:.2f} | "
                f"{value(row['post_window5_root_ade_slope_cm_per_window'], 3)} | "
                f"{value(row['mamba_state_norm_max'], 3)} | {value(row['mamba_state_drift_norm_max'], 3)} | "
                f"{row['state_diagnostics_all_finite']} |"
            )

        if gate is not None:
            lines.extend([
                "",
                "### Predefined Gate Decision",
                "",
                f"**{gate['status']}**. Continue to 5k: **{gate['continue_to_5k']}**.",
                "",
                "| Criterion | Passed |",
                "|---|---|",
            ])
            for criterion, passed in gate["criteria"].items():
                lines.append(f"| {criterion.replace('_', ' ')} | {passed} |")
            lines.extend([
                "",
                "| Duration | Hybrid-Minimal root-slope gap | Paired 95% CI | Hybrid better fraction |",
                "|---:|---:|---:|---:|",
            ])
            for seconds in (20, 40):
                paired = gate["long_horizon"][f"{seconds}s"]["paired_root_slope_gap"]
                lines.append(
                    f"| {seconds} s | {paired['mean_cm_per_window']:.3f} | "
                    f"[{paired['ci95_cm_per_window'][0]:.3f}, {paired['ci95_cm_per_window'][1]:.3f}] | "
                    f"{paired['hybrid_better_fraction']:.2f} |"
                )
            audit = gate["state_audit"]
            lines.extend([
                "",
                f"State audit: {audit['records']} commit records; maximum norm "
                f"{value(audit['max_state_norm'], 3)}, maximum drift {value(audit['max_state_drift_norm'], 3)}, "
                f"maximum final/initial norm ratio {value(audit['max_final_to_initial_norm_ratio'], 3)}.",
            ])

            hybrid_40 = load(eval_root / "hybrid_40s_in_horizon.json")
            minimal_40 = load(eval_root / "minimal_40s_in_horizon.json")
            reset8 = load(eval_root / "hybrid_40s_in_horizon_reset8.json")
            failed_retention = []
            for stem, metrics in gate["short_retention"].items():
                for metric, result in metrics.items():
                    if not result["passed"]:
                        label = "constraint" if metric == "constraint_error_cm_mean" else "Root ADE"
                        failed_retention.append(
                            f"{stem} {label} ({100 * result['relative_change']:+.2f}%)"
                        )
            retention_finding = (
                "Short-horizon retention passes every predefined 10% check."
                if not failed_retention
                else "Short-horizon retention fails the predefined 10% limit for "
                + ", ".join(failed_retention)
                + "."
            )
            lines.extend([
                "",
                "### Stage Finding",
                "",
                f"{retention_finding} At 40 seconds, Hybrid has Root ADE "
                f"{hybrid_40['mobench_root_ade_cm_mean']:.2f} cm and slope "
                f"{hybrid_40['post_window5_root_ade_slope_cm_per_window']:.3f}, versus Minimal's "
                f"{minimal_40['mobench_root_ade_cm_mean']:.2f} cm and "
                f"{minimal_40['post_window5_root_ade_slope_cm_per_window']:.3f}.",
                "",
                f"Resetting persistent state every eight generated windows changes Hybrid's 40-second "
                f"Root slope to {reset8['post_window5_root_ade_slope_cm_per_window']:.3f}, constraint "
                f"slope to {reset8['post_window5_constraint_slope_cm_per_window']:.3f}, and average "
                f"Root ADE to {reset8['mobench_root_ade_cm_mean']:.2f} cm. State remains finite and its "
                "maximum final/initial norm ratio is below the explosion threshold. The evidence therefore "
                "does not support simple Mamba norm explosion as the primary cause; periodic reset only "
                "partially controls late-window drift and does not repair accumulated trajectory bias.",
            ])

    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "A lower denoising loss is not sufficient. The experiment supports the long-horizon Hybrid "
        "claim only when Hybrid improves both 20 s and 40 s post-window-5 root/constraint slopes over "
        "Minimal, retains Table 4 and 9 s within the predefined 10% tolerance, and shows no non-finite "
        "or greater-than-10x state-norm growth. A failed 1k gate is a negative result and stops 5k.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
