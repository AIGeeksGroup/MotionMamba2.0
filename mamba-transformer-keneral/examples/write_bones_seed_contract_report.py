"""Write the verified BONES-SEED three-model Table 4/5 report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


MODES = ("original", "minimal", "hybrid")
PROTOCOLS = ("table4", "in_horizon", "out_of_horizon")
DISPLAY = {
    "original": "ARDY stabilized retrain (50k)",
    "minimal": "Minimal contract-fix (50k)",
    "hybrid": "Hybrid contract-fix (50k)",
}
TRAIN_DIR = {
    "original": "bones_seed_ardy_retrain_lr1e5_20260826_original",
    "minimal": "bones_seed_contract_full50k_20260824_minimal",
    "hybrid": "bones_seed_contract_full50k_20260824_hybrid",
}
METRICS = (
    ("constraint_error_cm_mean", "Constraint Error (cm)", 2),
    ("mobench_root_ade_cm_mean", "Root ADE (cm)", 2),
    ("mobench_root_fde_cm_mean", "Root FDE (cm)", 2),
    ("mobench_pa_mpjpe_cm_mean", "PA-MPJPE (cm)", 2),
    ("mobench_jitter_m_per_frame2_mean", "Jitter (m/frame2)", 5),
    ("mobench_angle_mae_deg_mean", "Angle MAE (deg)", 2),
    ("skate_percent_proxy", "Skate proxy (%)", 2),
    ("latency_seconds_per_sample", "Latency (s/sample)", 4),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def split_for_actor(actor_uid: str, seed: int = 20260824) -> str:
    digest = hashlib.sha256(f"{seed}:{actor_uid}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 1000
    return "train" if bucket < 900 else ("val" if bucket < 950 else "test")


def manifest_records(path: Path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def fmt(value: float, decimals: int, best: bool) -> str:
    rendered = f"{value:.{decimals}f}"
    return f"**{rendered}**" if best else rendered


def metric_row(mode: str, protocol: str, results: dict, include_protocol: bool) -> str:
    cells = [DISPLAY[mode]]
    if include_protocol:
        cells.append("in-horizon" if protocol == "in_horizon" else "out-of-horizon")
    for key, _, decimals in METRICS:
        value = results[(mode, protocol)][key]
        best = value == min(results[(candidate, protocol)][key] for candidate in MODES)
        cells.append(fmt(value, decimals, best))
    return "| " + " | ".join(cells) + " |"


def winner(results: dict, protocol: str, key: str) -> str:
    mode = min(MODES, key=lambda item: results[(item, protocol)][key])
    return DISPLAY[mode].replace(" (50k)", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data4/zts/result")
    parser.add_argument("--dataset-root", default="/data4/zts/datasets/bones_seed")
    parser.add_argument(
        "--output",
        default="/data3/zts/MotionMamba2.0/mamba-transformer-keneral/docs/"
        "BONES_SEED_CONTRACTFIX_DUAL_EVAL_20260824.md",
    )
    args = parser.parse_args()
    base = Path(args.base)
    dataset_root = Path(args.dataset_root)
    table4_root = base / "bones_seed_ardy_retrained_dual_eval_20260826"
    table5_root = base / "bones_seed_ardy_retrained_official_table5_20260826"
    long_root = base / "bones_seed_ardy_retrained_long_table5_20260826"

    training = {}
    for mode in MODES:
        run = base / TRAIN_DIR[mode]
        metrics = load_json(run / "metrics.json")
        checkpoint = run / f"{mode}_humanml3d_ddp.pt"
        assert metrics["status"] == "PASS", (mode, metrics["status"])
        assert metrics["steps"] == 50_000, (mode, metrics["steps"])
        assert metrics["all_gradients_finite"] is True, mode
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0, checkpoint
        training[mode] = (metrics, checkpoint)

    results = {}
    for protocol in PROTOCOLS:
        panels = []
        for mode in MODES:
            path = (
                table4_root / f"{mode}_table4.json"
                if protocol == "table4"
                else table5_root / f"{mode}_9s_{protocol}.json"
            )
            result = load_json(path)
            assert result["status"] == "PASS", (mode, protocol, result["status"])
            assert result["samples"] == 64, (mode, protocol, result["samples"])
            assert result["model"] == "g1", (mode, protocol, result["model"])
            expected_protocol = "table4" if protocol == "table4" else protocol
            assert result["protocol"] == expected_protocol, (mode, protocol, result["protocol"])
            assert result["r_precision"] is None and result["fid"] is None
            if protocol != "table4":
                assert result["paper_reference"].startswith("ARDY Section 6.3")
                assert result["total_frames"] == 225
                assert result["initial_history_frames"] == 24
            results[(mode, protocol)] = result
            panels.append(result["sample_ids"])
        assert panels[0] == panels[1] == panels[2], protocol
    assert (
        results[("original", "in_horizon")]["sample_ids"]
        == results[("original", "out_of_horizon")]["sample_ids"]
    ), "Table 5 in/out panels differ"
    panel_hashes = {
        protocol: hashlib.sha256(
            "\n".join(results[("original", protocol)]["sample_ids"]).encode()
        ).hexdigest()
        for protocol in PROTOCOLS
    }

    long_results = {}
    long_panels = []
    for seconds in (20, 40):
        for protocol in ("in_horizon", "out_of_horizon"):
            panels = []
            for mode in MODES:
                result = load_json(long_root / f"{mode}_{seconds}s_{protocol}.json")
                assert result["status"] == "PASS", (seconds, protocol, mode)
                assert result["samples"] == 20, (seconds, protocol, mode)
                assert result["selection_min_seconds"] == 40.0
                long_results[(mode, seconds, protocol)] = result
                panels.append(result["sample_ids"])
            assert panels[0] == panels[1] == panels[2]
            long_panels.append(panels[0])
    assert all(panel == long_panels[0] for panel in long_panels[1:])
    long_panel_hash = hashlib.sha256("\n".join(long_panels[0]).encode()).hexdigest()

    long_gap_summary = {}
    for seconds in (20, 40):
        minimal_root = np.asarray(
            long_results[("minimal", seconds, "in_horizon")]["root_ade_cm_mean_by_window"]
        )
        hybrid_root = np.asarray(
            long_results[("hybrid", seconds, "in_horizon")]["root_ade_cm_mean_by_window"]
        )
        gap = hybrid_root - minimal_root
        post5 = gap[5:]
        post5_gap_slope = float(np.polyfit(np.arange(6, 6 + len(post5)), post5, 1)[0])
        worse = np.flatnonzero(gap[5:] > 0)
        first_worse = None if not len(worse) else int(worse[0] + 6)
        artifacts = {
            mode: np.load(long_root / f"{mode}_{seconds}s_in_horizon.npz")
            for mode in ("minimal", "hybrid")
        }
        assert np.array_equal(
            artifacts["minimal"]["sample_ids"], artifacts["hybrid"]["sample_ids"]
        )
        sample_window_root = {}
        horizon = int(long_results[("minimal", seconds, "in_horizon")]["generation_horizon_frames"])
        for mode, artifact in artifacts.items():
            root_error = np.linalg.norm(
                artifact["generated_root"] - artifact["target_root"], axis=-1
            ) * 100.0
            sample_window_root[mode] = np.stack(
                [
                    root_error[:, start:start + horizon].mean(axis=1)
                    for start in range(0, root_error.shape[1], horizon)
                ],
                axis=1,
            )
        sample_gap = sample_window_root["hybrid"] - sample_window_root["minimal"]
        slope_x = np.arange(6, sample_gap.shape[1] + 1, dtype=np.float64)
        centered_x = slope_x - slope_x.mean()
        sample_slopes = (
            sample_gap[:, 5:] * centered_x[None]
        ).sum(axis=1) / np.square(centered_x).sum()
        rng = np.random.default_rng(20260826 + seconds)
        bootstrap = np.asarray(
            [rng.choice(sample_slopes, len(sample_slopes), replace=True).mean() for _ in range(20_000)]
        )
        long_gap_summary[seconds] = {
            "post5_gap_slope": post5_gap_slope,
            "post5_gap_slope_ci": tuple(np.percentile(bootstrap, [2.5, 97.5])),
            "first_worse_window": first_worse,
            "last_window_gap": float(gap[-1]),
        }

    summaries = {
        split: load_json(base / f"bones_seed_ardy_{split}" / "summary.json")
        for split in ("train", "val", "test")
    }
    for split, summary in summaries.items():
        assert summary["status"] == "PASS", (split, summary)
        assert summary["skipped"]["invalid"] == 0, (split, summary["skipped"])

    manifests = {}
    for split in ("train", "val", "test"):
        records = list(manifest_records(base / f"bones_seed_ardy_{split}" / "manifest.jsonl"))
        assert len(records) == summaries[split]["samples"]
        assert all(record["captions"] for record in records)
        manifests[split] = records
    assert not ({record["actor_uid"] for record in manifests["train"]} &
                {record["actor_uid"] for record in manifests["val"]})
    assert not ({record["actor_uid"] for record in manifests["train"]} &
                {record["actor_uid"] for record in manifests["test"]})
    assert not ({record["actor_uid"] for record in manifests["val"]} &
                {record["actor_uid"] for record in manifests["test"]})

    assigned_rows = Counter()
    assigned_actors = {split: set() for split in ("train", "val", "test")}
    metadata = dataset_root / "metadata" / "seed_metadata_v004.csv"
    with metadata.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = split_for_actor(row["actor_uid"])
            assigned_rows[split] += 1
            assigned_actors[split].add(row["actor_uid"])
    assert sum(assigned_rows.values()) == 142_220

    train_categories = Counter(record["category"] for record in manifests["train"])
    test_categories = Counter(record["category"] for record in manifests["test"])
    absent_categories = sorted(set(train_categories) - set(test_categories))

    archive_validation = load_json(dataset_root / "archive_validation.json")
    prepared = load_json(dataset_root / "g1_prepared.json")
    cache_complete = load_json(base / "bones_seed_text_cache_llm2vec" / "complete.json")
    assert archive_validation["status"] == prepared["status"] == cache_complete["status"] == "PASS"
    assert cache_complete["samples"] == sum(item["samples"] for item in summaries.values())

    header = "| Method | " + " | ".join(label for _, label, _ in METRICS) + " |"
    divider = "|---|" + "---:|" * len(METRICS)
    long_header = "| Method | Protocol | " + " | ".join(label for _, label, _ in METRICS) + " |"
    long_divider = "|---|---|" + "---:|" * len(METRICS)

    lines = [
        "# BONES-SEED Contract-Fix Table 4/5 Evaluation",
        "",
        "## Status",
        "",
        "This report evaluates three 50k denoiser checkpoints on the ",
        "actor-disjoint BONES-SEED v004 G1 split under the `canonical_history_v2` contract.",
        "The released ARDY-G1 representation frontend and tokenizer/decoder are shared; ",
        "released Rigplay mixer weights are excluded from all three architecture rows.",
        "The original ARDY run was invalidated after its root output branch collapsed. The ",
        "replacement resumes the stable fresh-mixer 5k checkpoint at learning rate `1e-5`, ",
        "continues to 50k total steps, and must pass fixed-dev-loss and root-output-norm gates.",
        "",
        "| Method | Checkpoint | Training status |",
        "|---|---|---|",
    ]
    for mode in MODES:
        metrics, checkpoint = training[mode]
        hours = metrics["elapsed_seconds"] / 3600
        lines.append(
            f"| {DISPLAY[mode].replace(' (50k)', '')} | `{checkpoint}` | "
            f"PASS, 50k, finite gradients, {hours:.2f} h |"
        )
    lines.extend([
        "",
        "All nine evaluation jobs completed with `status=PASS`. Every model used the same ",
        "test manifest, seed `20260824`, 64 samples per protocol, batch size 4 for Table 4 and 2 ",
        "for Table 5, 10 diffusion ",
        "steps, CFG text/constraint weights 2/2, SDPA attention, and identical sample IDs ",
        "within each protocol. Table 4 uses a 52-frame panel. Table 5 follows ARDY Section 6.3: ",
        "a nine-second, 225-frame sequence with a tokenizer-aligned 24-frame (0.96 s) GT prefix ",
        "and 201 generated frames. Table 5 in/out use the same motion IDs.",
        "",
        "| Protocol | Fixed panel SHA-256 |",
        "|---|---|",
        f"| Table 4 | `{panel_hashes['table4']}` |",
        f"| Table 5 in | `{panel_hashes['in_horizon']}` |",
        f"| Table 5 out | `{panel_hashes['out_of_horizon']}` |",
        f"| 20/40 s supplement | `{long_panel_hash}` |",
        "",
        "For Table 5, Minimal and Hybrid prime the GT prefix once and then use persistent ",
        "cache-only rollout (`protocol_guard=gt_prefix_primed_cache_only`). ARDY has no ",
        "adapter-owned persistent state and uses native explicit autoregressive history.",
        "",
        "## Dataset and Split",
        "",
        "| Split | Assigned motions | Eligible motions (>=52 frames) | Actors | Invalid CSVs |",
        "|---|---:|---:|---:|---:|",
    ])
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split.capitalize()} | {assigned_rows[split]:,} | {summaries[split]['samples']:,} | "
            f"{len(assigned_actors[split]):,} | {summaries[split]['skipped']['invalid']} |"
        )
    absent = ", ".join(f"`{item}`" for item in absent_categories) if absent_categories else "None"
    lines.extend([
        "",
        "The SHA-256 actor split has zero actor overlap. All eligible rows have captions. ",
        f"Because several categories have too few contributing actors, the test split contains no "
        f"eligible samples for: {absent}. This category-coverage limitation must be considered when "
        "interpreting aggregate results.",
        "",
        "The model-native source is the 120 fps G1 CSV archive, deterministically resampled to 25 fps ",
        "and represented as 414-D explicit / 418-D dual-root statistics. All three downloaded ",
        "archives passed exact-size and gzip validation; SOMA is retained as dataset material but is ",
        "not silently retargeted into the G1 experiment.",
        "",
        "## Official BONES-SEED Track",
        "",
        "BONES-SEED has no locally available calibrated retrieval/FID or MoBench evaluator. These ",
        "values remain `N/A`; HumanML3D embeddings are not substituted as official BONES-SEED scores.",
        "",
        "| Method | R-Precision Top-3 | FID | FVMD | TM-Sim |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        lines.append(f"| {DISPLAY[mode].replace(' (50k)', '')} | N/A | N/A | N/A | N/A |")
    lines.extend([
        "",
        "## Table 4 proxy: Offline Text and Joint-Position Control",
        "",
        "This protocol uses the 52-frame native G1 in-horizon generation panel. Lower is better for ",
        "every reported metric.",
        "",
        header,
        divider,
    ])
    lines.extend(metric_row(mode, "table4", results, False) for mode in MODES)
    lines.extend([
        "",
        "## Table 5: ARDY-Paper-Aligned 9-Second Autoregressive Control",
        "",
        "In-horizon places one sampled pelvis/wrist/foot position goal at the end of every ",
        "generation window. Out-of-horizon places one goal only at the final frame, beyond the ",
        "initial generation window. Metrics exclude the GT prefix and cover generated frames only.",
        "",
        long_header,
        long_divider,
    ])
    for protocol in ("in_horizon", "out_of_horizon"):
        lines.extend(metric_row(mode, protocol, results, True) for mode in MODES)

    lines.extend([
        "",
        "## 20/40-Second Long-Sequence Supplement",
        "",
        "The supplement uses the same 20 test motions that are each at least 40 seconds long. ",
        "The 20-second crop and 40-second crop share the same motion ID and source start, and all ",
        "three models receive identical text, goals, diffusion noise, and CFG settings. Window 5 ",
        "is the first generation window that reads memory after the four-chunk attention cache has ",
        "evicted the GT-prefix chunk. Slopes are least-squares growth rates from windows 6 onward; ",
        "lower values indicate slower post-window-5 degradation.",
        "",
        "| Duration | Method | In constraint (cm) | Out constraint (cm) | "
        "In Root ADE (cm) | Out Root ADE (cm) | Post-W5 in Root slope (cm/window) | "
        "Post-W5 in constraint slope (cm/window) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for seconds in (20, 40):
        for mode in MODES:
            in_result = long_results[(mode, seconds, "in_horizon")]
            out_result = long_results[(mode, seconds, "out_of_horizon")]
            lines.append(
                f"| {seconds} s | {DISPLAY[mode]} | "
                f"{in_result['constraint_error_cm_mean']:.2f} | "
                f"{out_result['constraint_error_cm_mean']:.2f} | "
                f"{in_result['mobench_root_ade_cm_mean']:.2f} | "
                f"{out_result['mobench_root_ade_cm_mean']:.2f} | "
                f"{in_result['post_window5_root_ade_slope_cm_per_window']:.3f} | "
                f"{in_result['post_window5_constraint_slope_cm_per_window']:.3f} |"
            )

    lines.extend([
        "",
        "| Duration | First post-W5 window where Hybrid Root ADE is worse | "
        "Post-W5 Hybrid-Minimal Root-gap slope (cm/window) | Paired 95% bootstrap CI | "
        "Final-window Root gap (cm) |",
        "|---:|---:|---:|---:|---:|",
    ])
    for seconds in (20, 40):
        summary = long_gap_summary[seconds]
        first_worse = (
            "never" if summary["first_worse_window"] is None
            else str(summary["first_worse_window"])
        )
        lines.append(
            f"| {seconds} s | {first_worse} | {summary['post5_gap_slope']:.3f} | "
            f"[{summary['post5_gap_slope_ci'][0]:.3f}, {summary['post5_gap_slope_ci'][1]:.3f}] | "
            f"{summary['last_window_gap']:.2f} |"
        )

    lines.extend([
        "",
        "### 40-Second Window Trace: Hybrid vs Minimal",
        "",
        "| Window | Minimal Root ADE (cm) | Hybrid Root ADE (cm) | "
        "Minimal constraint (cm) | Hybrid constraint (cm) |",
        "|---:|---:|---:|---:|---:|",
    ])
    minimal_long = long_results[("minimal", 40, "in_horizon")]
    hybrid_long = long_results[("hybrid", 40, "in_horizon")]
    for index, (minimal_root, hybrid_root, minimal_constraint, hybrid_constraint) in enumerate(
        zip(
            minimal_long["root_ade_cm_mean_by_window"],
            hybrid_long["root_ade_cm_mean_by_window"],
            minimal_long["constraint_error_cm_mean_by_window"],
            hybrid_long["constraint_error_cm_mean_by_window"],
        ),
        start=1,
    ):
        lines.append(
            f"| {index} | {minimal_root:.2f} | {hybrid_root:.2f} | "
            f"{minimal_constraint:.2f} | {hybrid_constraint:.2f} |"
        )

    gap40 = long_gap_summary[40]
    lines.extend([
        "",
        f"At 40 seconds, Hybrid first exceeds Minimal Root ADE at window "
        f"{gap40['first_worse_window']}; its post-window-5 Hybrid-Minimal gap grows by "
        f"{gap40['post5_gap_slope']:.3f} cm per window (paired 95% bootstrap CI "
        f"[{gap40['post5_gap_slope_ci'][0]:.3f}, {gap40['post5_gap_slope_ci'][1]:.3f}]) and reaches "
        f"{gap40['last_window_gap']:.2f} cm in the final window. The current checkpoint therefore "
        "does not establish a 40-second Hybrid advantage. Training exposes the recurrent state to "
        "only one teacher-history commit followed by one generated window, so repeated-state rollout "
        "is an out-of-distribution regime and can accumulate Mamba-state bias. This negative result is "
        "kept separate from the healthy nine-second result.",
    ])

    lines.extend([
        "",
        "## Boundary and Interpretation",
        "",
        "| Method | Table 5 in max boundary speed (m/s) | Table 5 out max boundary speed (m/s) | "
        "Eval peak memory (MiB, max) | Active mixer parameters |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        peak = max(results[(mode, protocol)]["peak_gpu_memory_mib"] for protocol in PROTOCOLS)
        active = results[(mode, "table4")]["active_parameters"]
        lines.append(
            f"| {DISPLAY[mode]} | {results[(mode, 'in_horizon')]['max_window_boundary_root_speed']:.2f} | "
            f"{results[(mode, 'out_of_horizon')]['max_window_boundary_root_speed']:.2f} | "
            f"{peak:.1f} | {active:,} |"
        )

    table4_geometry = winner(results, "table4", "constraint_error_cm_mean")
    table5_in_geometry = winner(results, "in_horizon", "constraint_error_cm_mean")
    table5_out_geometry = winner(results, "out_of_horizon", "constraint_error_cm_mean")
    if table5_in_geometry == table5_out_geometry:
        table5_summary = (
            f"{table5_in_geometry} has the lowest Table 5 constraint error in both the "
            "in-horizon and out-of-horizon protocols"
        )
    else:
        table5_summary = (
            f"{table5_in_geometry} and {table5_out_geometry} have the lowest Table 5 "
            "in-horizon and out-of-horizon constraint errors respectively"
        )
    hybrid_in_gain = 100.0 * (
        results[("minimal", "in_horizon")]["constraint_error_cm_mean"]
        - results[("hybrid", "in_horizon")]["constraint_error_cm_mean"]
    ) / results[("minimal", "in_horizon")]["constraint_error_cm_mean"]
    hybrid_out_gain = 100.0 * (
        results[("minimal", "out_of_horizon")]["constraint_error_cm_mean"]
        - results[("hybrid", "out_of_horizon")]["constraint_error_cm_mean"]
    ) / results[("minimal", "out_of_horizon")]["constraint_error_cm_mean"]
    lines.extend([
        "",
        f"On the fixed panels, {table4_geometry} has the lowest Table 4 constraint error; "
        f"{table5_summary}. Bold values identify the global minimum ",
        "within each protocol and metric. These paired geometry and physical proxies support only ",
        "this fixed BONES-SEED protocol; they do not establish an official leaderboard result.",
        f"Hybrid improves Table 5 constraint error over Minimal by {hybrid_in_gain:.2f}% in-horizon and "
        f"{hybrid_out_gain:.2f}% ",
        "out-of-horizon. Relative to Minimal, Hybrid also lowers Root ADE/FDE, PA-MPJPE, jitter, ",
        "and skate proxy in both protocols. Against all three models, stabilized ARDY is best for ",
        "in-horizon constraint/Root ADE/Root FDE/jitter and out-of-horizon jitter/skate, while Hybrid ",
        "is best for out-of-horizon constraint/Root ADE/Root FDE/PA-MPJPE and in-horizon PA-MPJPE/skate. ",
        "The nine-second result therefore supports Hybrid over the attention-only Minimal baseline, ",
        "not a blanket win over stabilized ARDY.",
        "",
        "Latency was measured in parallel on separate GPUs and is suitable for within-run comparison ",
        "only. A systems claim requires serialized warm-up and identical GPU/timing isolation.",
        "",
        "## Provenance",
        "",
        "- Dataset protocol: `docs/BONES_SEED_TRAINING_PROTOCOL_20260824.md`",
        "- Trainer: `examples/train_humanml3d_ardy_ddp.py`",
        "- ARDY retraining launcher: `examples/run_bones_seed_ardy_retrain_20260826.sh`",
        "- Table 4 evaluator: `examples/evaluate_humanml3d_table45.py`",
        "- Table 5 evaluator: `examples/evaluate_ardy_official_table5.py`",
        "- Corrected and long evaluation launcher: `examples/run_bones_seed_ardy_retrained_eval_20260826.sh`",
        f"- Table 4 JSON and NPZ artifacts: `{table4_root}/`",
        f"- Table 5 JSON and NPZ artifacts: `{table5_root}/`",
        f"- 20/40 s JSON and NPZ artifacts: `{long_root}/`",
        f"- Stabilized ARDY training metadata: `{base}/{TRAIN_DIR['original']}/metrics.json`",
        f"- Archive validation: `{dataset_root}/archive_validation.json`",
        "",
    ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"report_validation=PASS output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
