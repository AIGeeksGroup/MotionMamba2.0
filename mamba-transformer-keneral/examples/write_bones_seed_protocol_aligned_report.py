"""Write the BONES-SEED protocol-aligned long-horizon fine-tune report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MODES = ("original", "minimal", "hybrid")
STAGES = (500, 1000)
DISPLAY = {
    "original": "ARDY",
    "minimal": "Minimal",
    "hybrid": "Hybrid",
}
METRICS = (
    ("constraint_error_cm_mean", "Constraint (cm)", 2),
    ("mobench_root_ade_cm_mean", "Root ADE (cm)", 2),
    ("mobench_root_fde_cm_mean", "Root FDE (cm)", 2),
    ("mobench_pa_mpjpe_cm_mean", "PA-MPJPE (cm)", 2),
    ("mobench_jitter_m_per_frame2_mean", "Jitter", 5),
    ("mobench_angle_mae_deg_mean", "Angle MAE (deg)", 2),
    ("skate_percent_proxy", "Skate (%)", 2),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def panel_hash(rows: list[str]) -> str:
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def fmt(value: float, decimals: int, best: bool = False) -> str:
    rendered = f"{value:.{decimals}f}"
    return f"**{rendered}**" if best else rendered


def metric_table(results: dict, stage: int, stem: str, protocol_label: bool) -> list[str]:
    header = ["Method"]
    if protocol_label:
        header.append("Protocol")
    header.extend(label for _, label, _ in METRICS)
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * (1 + int(protocol_label)) + "---:|" * len(METRICS))
    stems = (
        ("9s_in_horizon", "in-horizon"),
        ("9s_out_of_horizon", "out-of-horizon"),
    ) if protocol_label else ((stem, ""),)
    for current_stem, label in stems:
        for mode in MODES:
            row = [DISPLAY[mode]]
            if protocol_label:
                row.append(label)
            for key, _, decimals in METRICS:
                value = results[(stage, mode, current_stem)][key]
                best = value == min(
                    results[(stage, candidate, current_stem)][key]
                    for candidate in MODES
                )
                row.append(fmt(value, decimals, best))
            lines.append("| " + " | ".join(row) + " |")
    return lines


def gap_slope_summary(eval_root: Path, results: dict, stage: int, seconds: int) -> dict:
    sample_windows = {}
    for mode in ("minimal", "hybrid"):
        artifact = np.load(eval_root / f"{mode}_{seconds}s_in_horizon.npz")
        horizon = int(results[(stage, mode, f"{seconds}s_in_horizon")]["generation_horizon_frames"])
        root_error = np.linalg.norm(
            artifact["generated_root"] - artifact["target_root"], axis=-1
        ) * 100.0
        sample_windows[mode] = np.stack(
            [
                root_error[:, start:start + horizon].mean(axis=1)
                for start in range(0, root_error.shape[1], horizon)
            ],
            axis=1,
        )
    gap = sample_windows["hybrid"] - sample_windows["minimal"]
    x = np.arange(6, gap.shape[1] + 1, dtype=np.float64)
    centered = x - x.mean()
    sample_slopes = (gap[:, 5:] * centered[None]).sum(axis=1) / np.square(centered).sum()
    rng = np.random.default_rng(20260826 + stage + seconds)
    bootstrap = np.asarray(
        [rng.choice(sample_slopes, len(sample_slopes), replace=True).mean() for _ in range(20_000)]
    )
    mean_gap = gap.mean(axis=0)
    return {
        "slope": float(sample_slopes.mean()),
        "ci": tuple(float(value) for value in np.percentile(bootstrap, [2.5, 97.5])),
        "final_gap": float(mean_gap[-1]),
        "first_worse": (
            None
            if not np.any(mean_gap[5:] > 0)
            else int(np.flatnonzero(mean_gap[5:] > 0)[0] + 6)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data4/zts/result")
    parser.add_argument(
        "--output",
        default=(
            "/data3/zts/MotionMamba2.0/mamba-transformer-keneral/docs/"
            "BONES_SEED_PROTOCOL_ALIGNED_LONG_HORIZON_20260826.md"
        ),
    )
    args = parser.parse_args()
    base = Path(args.base)
    train_root = base / "bones_seed_protocol_aligned_ft_20260826"
    output = Path(args.output)

    training = {}
    results = {}
    eval_roots = {}
    hashes = {}
    for stage in STAGES:
        counts = []
        for mode in MODES:
            path = train_root / mode / f"metrics_step_{stage:06d}.json"
            metrics = load_json(path)
            checkpoint = train_root / mode / f"checkpoint_step_{stage:06d}.pt"
            assert metrics["steps"] == stage and metrics["all_gradients_finite"] is True
            assert metrics["sequential_rollout_curriculum"] == "long_horizon_v1"
            assert metrics["sequential_rollout_steps"] == stage
            assert metrics["global_batch_size"] == 32 and metrics["final_learning_rate"] == 5e-6
            assert checkpoint.is_file() and checkpoint.stat().st_size > 0
            training[(stage, mode)] = metrics
            counts.append(metrics["rollout_window_counts"])
        assert counts[0] == counts[1] == counts[2], (stage, counts)

        eval_root = base / f"bones_seed_protocol_aligned_eval_stage{stage}_20260826"
        assert (eval_root / "EVALUATION_COMPLETE").is_file()
        eval_roots[stage] = eval_root
        specs = ["table4"] + [
            f"{seconds}s_{protocol}"
            for seconds in (9, 20, 40)
            for protocol in ("in_horizon", "out_of_horizon")
        ]
        for stem in specs:
            rows = []
            for mode in MODES:
                result = load_json(eval_root / f"{mode}_{stem}.json")
                assert result["status"] == "PASS"
                assert result["diffusion_steps"] == 10
                assert result["cfg_weight_text"] == result["cfg_weight_constraint"] == 2.0
                assert f"checkpoint_step_{stage:06d}.pt" in result["checkpoint"]
                results[(stage, mode, stem)] = result
                rows.append(result["sample_ids"])
            assert rows[0] == rows[1] == rows[2]
            hashes[(stage, stem)] = panel_hash(rows[0])

    gap_summaries = {
        (stage, seconds): gap_slope_summary(eval_roots[stage], results, stage, seconds)
        for stage in STAGES
        for seconds in (20, 40)
    }
    ablation = load_json(
        base / "bones_seed_hybrid_multiwindow_eval_20260826" /
        "hybrid_recurrent_stage500_summary.json"
    )

    lines = [
        "# BONES-SEED Protocol-Aligned Long-Horizon Evaluation",
        "",
        "## Status and Experimental Role",
        "",
        "This is the primary three-model long-horizon fine-tune track. ARDY, Minimal, and ",
        "Hybrid start from their healthy matched 50k checkpoints and receive the same 500-step ",
        "gate followed by the same 1000-step total confirmation. The 1000-step checkpoint is ",
        "pre-designated as the main result; the 500-step result is retained to test stability ",
        "rather than to select the better-looking checkpoint.",
        "",
        "The earlier Hybrid-only 500-step recurrent-path run is reported only as a training-strategy ",
        "ablation. It is not mixed into any three-model main table.",
        "",
        "## Unified Fine-Tune Protocol",
        "",
        "| Setting | Value |",
        "|---|---|",
        "| Dataset/split | BONES-SEED v004 G1, actor-disjoint train/val/test |",
        "| Initialization | Each architecture's healthy matched 50k checkpoint |",
        "| Train scope | Entire denoiser for all three models |",
        "| Optimizer budget | 500-step gate, then resume to 1000 total steps |",
        "| Learning rate | `5e-6` |",
        "| Global batch | 32 (8 GPUs x 4 samples) |",
        "| Supervision | Teacher history; loss on the final 52-frame window |",
        "| Curriculum | 50% x2 windows; 20% x4; 15% uniform x6-10; 15% uniform x12-19 |",
        "| ARDY history | Native explicit history |",
        "| Minimal/Hybrid history | Canonical persistent cache/state commits |",
        "| Control/dropout | Joint-position control; text/control dropout 0.1/0.1 |",
        "| Loss weights | boundary 10, root velocity 50, constraint 1, FK 1, foot skate 50 |",
        "",
        "The data, sampled motion IDs, augmentation seeds, rollout-depth draws, update count, ",
        "batch size, learning rate, losses, and evaluation panels are matched. The internal history ",
        "mechanism is necessarily architecture-native: changing ARDY into a persistent-state model ",
        "would erase the architectural comparison.",
        "",
        "## Training Audit",
        "",
        "| Stage | Method | Eval loss | Root output norm | Trainable params | Time (min) | Rollout counts |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for stage in STAGES:
        for mode in MODES:
            metrics = training[(stage, mode)]
            counts = ", ".join(
                f"{key}w:{value}" for key, value in sorted(
                    ((int(key), value) for key, value in metrics["rollout_window_counts"].items())
                )
            )
            lines.append(
                f"| {stage} | {DISPLAY[mode]} | {metrics['final_eval_loss']:.4f} | "
                f"{metrics['root_output_weight_norm']:.4f} | {metrics['trainable_parameters']:,} | "
                f"{metrics['elapsed_seconds'] / 60:.1f} | {counts} |"
            )

    for stage in STAGES:
        lines.extend([
            "",
            f"## Stage {stage}: Table 4 Offline Text and Joint-Position Control",
            "",
        ])
        lines.extend(metric_table(results, stage, "table4", False))
        lines.extend([
            "",
            f"## Stage {stage}: ARDY-Aligned 9-Second Table 5",
            "",
        ])
        lines.extend(metric_table(results, stage, "9s", True))

    lines.extend([
        "",
        "## 20/40-Second Supplement",
        "",
        "All rows use the same 20 naturally long test motions. Slopes start after window 5; lower ",
        "is better. The 20/40-second evaluations are duration extensions, not official ARDY rows.",
        "",
        "| Stage | Duration | Method | Protocol | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | Post-W5 Root slope | Post-W5 constraint slope |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for stage in STAGES:
        for seconds in (20, 40):
            for protocol in ("in_horizon", "out_of_horizon"):
                stem = f"{seconds}s_{protocol}"
                for mode in MODES:
                    row = results[(stage, mode, stem)]
                    root_slope = row["post_window5_root_ade_slope_cm_per_window"]
                    constraint_slope = row["post_window5_constraint_slope_cm_per_window"]
                    constraint_slope_cell = (
                        f"{constraint_slope:.3f}" if constraint_slope is not None else "N/A"
                    )
                    lines.append(
                        f"| {stage} | {seconds} s | {DISPLAY[mode]} | "
                        f"{'in' if protocol == 'in_horizon' else 'out'} | "
                        f"{row['constraint_error_cm_mean']:.2f} | {row['mobench_root_ade_cm_mean']:.2f} | "
                        f"{row['mobench_root_fde_cm_mean']:.2f} | "
                        f"{root_slope:.3f} | "
                        f"{constraint_slope_cell} |"
                    )

    lines.extend([
        "",
        "## Hybrid vs Minimal Stability",
        "",
        "Negative deltas favor Hybrid. The gap slope is computed per paired sample from window 6 ",
        "onward and bootstrapped with 20,000 paired resamples.",
        "",
        "| Stage | Duration | Protocol | Hybrid-Minimal constraint (cm) | Hybrid-Minimal Root ADE (cm) | Post-W5 Root-gap slope (cm/window) | 95% CI | Final-window Root gap (cm) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for stage in STAGES:
        for seconds in (20, 40):
            for protocol in ("in_horizon", "out_of_horizon"):
                stem = f"{seconds}s_{protocol}"
                hybrid = results[(stage, "hybrid", stem)]
                minimal = results[(stage, "minimal", stem)]
                if protocol == "in_horizon":
                    gap = gap_summaries[(stage, seconds)]
                    slope = f"{gap['slope']:.3f}"
                    ci = f"[{gap['ci'][0]:.3f}, {gap['ci'][1]:.3f}]"
                    final_gap = f"{gap['final_gap']:.2f}"
                else:
                    slope = ci = final_gap = "N/A"
                lines.append(
                    f"| {stage} | {seconds} s | {'in' if protocol == 'in_horizon' else 'out'} | "
                    f"{hybrid['constraint_error_cm_mean'] - minimal['constraint_error_cm_mean']:.2f} | "
                    f"{hybrid['mobench_root_ade_cm_mean'] - minimal['mobench_root_ade_cm_mean']:.2f} | "
                    f"{slope} | {ci} | {final_gap} |"
                )

    comparisons = []
    short_long_comparisons = []
    for stage in STAGES:
        wins = 0
        total = 0
        short_wins = 0
        short_total = 0
        long_wins = 0
        long_total = 0
        for seconds in (9, 20, 40):
            for protocol in ("in_horizon", "out_of_horizon"):
                stem = f"{seconds}s_{protocol}"
                for key in ("constraint_error_cm_mean", "mobench_root_ade_cm_mean"):
                    is_win = (
                        results[(stage, "hybrid", stem)][key]
                        < results[(stage, "minimal", stem)][key]
                    )
                    wins += is_win
                    total += 1
                    if seconds == 9:
                        short_wins += is_win
                        short_total += 1
                    else:
                        long_wins += is_win
                        long_total += 1
        comparisons.append((stage, wins, total))
        short_long_comparisons.append(
            (stage, short_wins, short_total, long_wins, long_total)
        )
    stable = all(wins == comparisons[0][1] for _, wins, _ in comparisons)
    lines.extend([
        "",
        "## Predefined Stability Check",
        "",
    ])
    for stage, wins, total in comparisons:
        lines.append(
            f"- At {stage} steps, Hybrid beats Minimal in {wins}/{total} matched "
            "duration/protocol comparisons across constraint error and Root ADE."
        )
    lines.append(
        "- The sign pattern is "
        + ("stable" if stable else "not stable")
        + " between the 500-step gate and 1000-step confirmation."
    )
    for stage, short_wins, short_total, long_wins, long_total in short_long_comparisons:
        lines.append(
            f"- At {stage} steps, Hybrid wins {short_wins}/{short_total} 9-second comparisons "
            f"but only {long_wins}/{long_total} comparisons at 20/40 seconds."
        )

    hybrid_500_20 = results[(500, "hybrid", "20s_in_horizon")]
    hybrid_1000_20 = results[(1000, "hybrid", "20s_in_horizon")]
    hybrid_500_40 = results[(500, "hybrid", "40s_in_horizon")]
    hybrid_1000_40 = results[(1000, "hybrid", "40s_in_horizon")]
    lines.extend([
        "",
        "## Main Finding",
        "",
        "The confirmation is stable but negative for the intended long-horizon Hybrid claim. ",
        "Hybrid remains better than Minimal at nine seconds, but loses every matched constraint-error ",
        "and Root-ADE comparison at 20 and 40 seconds at both training stages. ARDY is the strongest ",
        "model on those long-duration control metrics.",
        "",
        f"Continuing Hybrid from 500 to 1000 steps increases 20-second in-horizon constraint error "
        f"by {hybrid_1000_20['constraint_error_cm_mean'] - hybrid_500_20['constraint_error_cm_mean']:.2f} cm "
        f"and Root ADE by {hybrid_1000_20['mobench_root_ade_cm_mean'] - hybrid_500_20['mobench_root_ade_cm_mean']:.2f} cm. ",
        f"At 40 seconds the increases are "
        f"{hybrid_1000_40['constraint_error_cm_mean'] - hybrid_500_40['constraint_error_cm_mean']:.2f} cm "
        f"and {hybrid_1000_40['mobench_root_ade_cm_mean'] - hybrid_500_40['mobench_root_ade_cm_mean']:.2f} cm. ",
        "The falling teacher-history validation loss therefore does not predict generated-history ",
        "rollout quality for Hybrid.",
        "",
        "These results do not justify a 50k fresh run with the same teacher-history objective. The ",
        "next defensible training experiment is a matched generated-history/scheduled-sampling ",
        "objective, with short-window retention, before spending the full pretraining budget.",
    ])

    lines.extend([
        "",
        "## Training-Strategy Ablation: Hybrid-Only Recurrent Fine-Tune",
        "",
        "This earlier run updates only Hybrid's Mamba path and fusion gates (104,664,080 parameters) ",
        "for 500 steps with a 50% x2 / 50% x6-10 curriculum. ARDY and Minimal did not receive ",
        "the same adaptation, so these values are an ablation and are excluded from every main table.",
        "",
        "| 40 s in-horizon metric | Old Hybrid checkpoint | Hybrid-only recurrent FT |",
        "|---|---:|---:|",
        f"| Constraint error (cm) | 166.89 | {ablation['constraint_error_cm_mean']:.2f} |",
        f"| Root ADE (cm) | 160.81 | {ablation['root_ade_cm_mean']:.2f} |",
        f"| Hybrid-Minimal post-W5 Root-gap slope | {ablation['baseline_post_window5_gap_slope_cm_per_window']:.3f} | {ablation['candidate_post_window5_gap_slope_cm_per_window']:.3f} |",
        f"| Final-window Hybrid-Minimal Root gap (cm) | {ablation['baseline_final_window_gap_cm']:.2f} | {ablation['candidate_final_window_gap_cm']:.2f} |",
        "",
        "## Interpretation Boundary",
        "",
        "The unified fine-tune uses teacher-forced history and truncated state commits. It evaluates ",
        "protocol-aligned adaptation but does not remove the generated-history exposure gap. A fresh ",
        "50k long-rollout pretraining run remains a separate, more expensive experiment and should only ",
        "be launched if the 500/1000-step conclusions are sufficiently stable.",
        "",
        "R-Precision, FID, FVMD, and TM-Sim remain `N/A` because BONES-SEED has no locally calibrated ",
        "official learned evaluator. Geometry and physical values are fixed-panel proxies.",
        "",
        "## Provenance",
        "",
        "- Trainer: `examples/train_humanml3d_ardy_ddp.py`",
        "- Training launcher: `examples/run_bones_seed_protocol_aligned_ft_20260826.sh`",
        "- Evaluation launcher: `examples/run_bones_seed_protocol_aligned_eval_20260826.sh`",
        "- Training artifacts: `/data4/zts/result/bones_seed_protocol_aligned_ft_20260826/`",
        "- Stage-500 evaluation: `/data4/zts/result/bones_seed_protocol_aligned_eval_stage500_20260826/`",
        "- Stage-1000 evaluation: `/data4/zts/result/bones_seed_protocol_aligned_eval_stage1000_20260826/`",
        "- Hybrid-only ablation: `/data4/zts/result/bones_seed_hybrid_multiwindow_eval_20260826/`",
        "- Evaluation seed: `20260824`",
        "",
        "## Panel Hashes",
        "",
        "| Stage | Protocol | SHA-256 |",
        "|---:|---|---|",
    ])
    for stage in STAGES:
        for stem in ("table4", "9s_in_horizon", "9s_out_of_horizon", "20s_in_horizon", "40s_in_horizon"):
            lines.append(f"| {stage} | {stem} | `{hashes[(stage, stem)]}` |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"protocol_aligned_report=PASS output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
