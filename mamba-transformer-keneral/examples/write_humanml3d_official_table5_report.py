"""Write the HumanML3D report with ARDY-paper-aligned Table 5 results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASE = Path("/data4/zts/result")
OUTPUT = Path(
    "/data3/zts/MotionMamba2.0/mamba-transformer-keneral/docs/"
    "HUMANML3D_CONTRACTFIX_DUAL_EVAL_20260823.md"
)
MODES = ("original", "minimal", "hybrid")
DISPLAY = {
    "original": "ARDY fresh mixer (50k)",
    "minimal": "Minimal contract-fix (50k)",
    "hybrid": "Hybrid contract-fix (50k)",
}
TRAIN_DIR = {
    mode: BASE / f"humanml3d_contractfix_full50k_20260823_{mode}"
    for mode in MODES
}
TABLE4_ROOT = BASE / "humanml3d_contractfix_dual_eval_20260823"
TABLE5_ROOT = BASE / "humanml3d_contractfix_official_table5_20260825"
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


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt(value: float, decimals: int, best: bool) -> str:
    text = f"{value:.{decimals}f}"
    return f"**{text}**" if best else text


def row(mode: str, protocol: str, results: dict, with_protocol: bool) -> str:
    cells = [DISPLAY[mode]]
    if with_protocol:
        cells.append("in-horizon" if protocol == "in_horizon" else "out-of-horizon")
    for key, _, decimals in METRICS:
        value = results[mode, protocol][key]
        best = value == min(results[candidate, protocol][key] for candidate in MODES)
        cells.append(fmt(value, decimals, best))
    return "| " + " | ".join(cells) + " |"


def winner(results: dict, protocol: str, key: str) -> str:
    mode = min(MODES, key=lambda candidate: results[candidate, protocol][key])
    return DISPLAY[mode].replace(" (50k)", "")


def main() -> int:
    test_manifest = BASE / "humanml3d_ardy_test" / "manifest.jsonl"
    test_records = [json.loads(line) for line in test_manifest.read_text().splitlines()]
    eligible_20s = sum(int(record.get("frames", 0)) >= 400 for record in test_records)
    eligible_40s = sum(int(record.get("frames", 0)) >= 800 for record in test_records)
    assert eligible_20s == eligible_40s == 0
    training = {}
    for mode in MODES:
        metrics = load(TRAIN_DIR[mode] / "metrics.json")
        checkpoint = TRAIN_DIR[mode] / f"{mode}_humanml3d_ddp.pt"
        assert metrics["status"] == "PASS" and metrics["steps"] == 50_000
        assert metrics["all_gradients_finite"] is True
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0
        training[mode] = metrics, checkpoint

    results = {}
    for protocol in ("table4", "in_horizon", "out_of_horizon"):
        ids = []
        for mode in MODES:
            path = (
                TABLE4_ROOT / f"{mode}_table4.json"
                if protocol == "table4"
                else TABLE5_ROOT / f"{mode}_{protocol}.json"
            )
            result = load(path)
            assert result["status"] == "PASS" and result["samples"] == 64
            if protocol != "table4":
                assert result["protocol"] == protocol
                assert result["total_frames"] == 180
                assert result["initial_history_frames"] == 20
                assert result["generated_frames"] == 160
                assert result["paper_reference"].startswith("ARDY Section 6.3")
            assert result["r_precision"] is None and result["fid"] is None
            results[mode, protocol] = result
            ids.append(result["sample_ids"])
        assert ids[0] == ids[1] == ids[2], protocol
    assert results["original", "in_horizon"]["sample_ids"] == results[
        "original", "out_of_horizon"
    ]["sample_ids"]

    panel_hash = {
        protocol: hashlib.sha256(
            "\n".join(results["original", protocol]["sample_ids"]).encode()
        ).hexdigest()
        for protocol in ("table4", "in_horizon", "out_of_horizon")
    }
    header = "| Method | " + " | ".join(label for _, label, _ in METRICS) + " |"
    divider = "|---|" + "---:|" * len(METRICS)
    long_header = "| Method | Protocol | " + " | ".join(label for _, label, _ in METRICS) + " |"
    long_divider = "|---|---|" + "---:|" * len(METRICS)

    lines = [
        "# HumanML3D Contract-Fix Table 4/5 Evaluation",
        "",
        "## Status",
        "",
        "This report evaluates three fresh 50k checkpoints under the ",
        "`canonical_history_v2` contract. Table 4 retains the fixed 40-frame offline panel. ",
        "Table 5 has been replaced by the ARDY Section 6.3 generation protocol.",
        "",
        "| Method | Checkpoint | Training status |",
        "|---|---|---|",
    ]
    for mode in MODES:
        metrics, checkpoint = training[mode]
        lines.append(
            f"| {DISPLAY[mode].replace(' (50k)', '')} | `{checkpoint}` | "
            f"PASS, 50k, finite gradients, {metrics['elapsed_seconds'] / 3600:.2f} h |"
        )
    lines.extend([
        "",
        "All nine jobs completed with `status=PASS`. Each protocol uses 64 fixed samples, ",
        "batch size 4, 10 diffusion steps, CFG text/constraint weights 2/2, and identical ",
        "sample IDs across the three models. Table 5 uses a nine-second sequence at 20 fps: ",
        "20 GT history frames followed by 160 generated frames in four 40-frame windows.",
        "",
        "| Protocol | Fixed panel SHA-256 |",
        "|---|---|",
        f"| Table 4 | `{panel_hash['table4']}` |",
        f"| Table 5 in | `{panel_hash['in_horizon']}` |",
        f"| Table 5 out | `{panel_hash['out_of_horizon']}` |",
        "",
        "For Table 5, Minimal and Hybrid prime the GT prefix exactly once and then use ",
        "persistent cache-only rollout. ARDY uses native explicit autoregressive history.",
        "",
        "## Official HumanML3D Track",
        "",
        "Official R-Precision and FID remain unavailable because generated output is ",
        "CoreSkeleton27/330-D while the official evaluator requires HumanML3D 22-joint/263-D. ",
        "The prior conversion audit found round-trip FID bias `14.9845` and paired embedding ",
        "cosine mean `0.8807`, so learned scores are not approximated.",
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
        "This protocol uses the fixed 40-frame in-horizon panel. Lower is better for every metric.",
        "",
        header,
        divider,
    ])
    lines.extend(row(mode, "table4", results, False) for mode in MODES)
    lines.extend([
        "",
        "## Table 5: ARDY-Paper-Aligned 9-Second Autoregressive Control",
        "",
        "In-horizon places one sampled pelvis/wrist/foot position goal at the end of each ",
        "two-second generation window. Out-of-horizon places one goal only at the final frame. ",
        "Metrics exclude the one-second GT prefix and cover the generated eight seconds.",
        "",
        long_header,
        long_divider,
    ])
    for protocol in ("in_horizon", "out_of_horizon"):
        lines.extend(row(mode, protocol, results, True) for mode in MODES)

    lines.extend([
        "",
        "## 20/40-Second Supplement Status",
        "",
        f"The HumanML3D test manifest contains {eligible_20s} motions of at least 20 seconds and ",
        f"{eligible_40s} motions of at least 40 seconds. A paired-reference 20/40-second experiment ",
        "cannot therefore be run on HumanML3D without looping or stitching clips and fabricating ",
        "ground truth. Those synthetic variants are intentionally not reported. The strict 20/40-second ",
        "supplement is run on the same 20 naturally long BONES-SEED test motions and is documented in ",
        "`docs/BONES_SEED_CONTRACTFIX_DUAL_EVAL_20260824.md`.",
        "",
        "The nine-second HumanML3D protocol contains four generation windows, equal to the configured ",
        "four-chunk attention-cache capacity. It validates four-window rollout but does not isolate the ",
        "Hybrid Mamba state after attention-cache eviction.",
    ])

    lines.extend([
        "",
        "## Boundary and Interpretation",
        "",
        "| Method | Table 5 in max boundary speed (m/s) | Table 5 out max boundary speed (m/s) | Eval peak memory (MiB, max) | Active mixer parameters |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        peak = max(results[mode, p]["peak_gpu_memory_mib"] for p in ("table4", "in_horizon", "out_of_horizon"))
        lines.append(
            f"| {DISPLAY[mode]} | {results[mode, 'in_horizon']['max_window_boundary_root_speed']:.2f} | "
            f"{results[mode, 'out_of_horizon']['max_window_boundary_root_speed']:.2f} | "
            f"{peak:.1f} | {results[mode, 'table4']['active_parameters']:,} |"
        )
    in_winner = winner(results, "in_horizon", "constraint_error_cm_mean")
    out_winner = winner(results, "out_of_horizon", "constraint_error_cm_mean")
    if in_winner == out_winner:
        constraint_summary = (
            f"{in_winner} has the lowest constraint error in both Table 5 protocols"
        )
    else:
        constraint_summary = (
            f"{in_winner} and {out_winner} have the lowest in-horizon and "
            "out-of-horizon constraint errors respectively"
        )
    lines.extend([
        "",
        f"{constraint_summary} on these fixed panels. Bold values are per-protocol ",
        "minima. These geometry and physical proxies do not constitute official leaderboard scores.",
        "Hybrid does not win goal error on HumanML3D at nine seconds, but it has the lowest ",
        "out-of-horizon Root ADE, PA-MPJPE, jitter, Angle MAE, and skate proxy, plus the lowest ",
        "in-horizon skate proxy. The evidence therefore supports better long-rollout continuity ",
        "and physical proxies, not a blanket long-horizon control-accuracy win.",
        "",
        "Latency jobs ran concurrently on separate GPUs and support within-run comparison only.",
        "",
        "## Provenance",
        "",
        "- ARDY paper protocol: Section 6.3 / Table 5, DOI `10.1145/3811284`",
        "- Table 4 evaluator: `examples/evaluate_humanml3d_table45.py`",
        "- Table 5 evaluator: `examples/evaluate_ardy_official_table5.py`",
        "- Table 5 launcher: `examples/run_ardy_official_table5_dual_dataset_20260825.sh`",
        f"- Table 4 artifacts: `{TABLE4_ROOT}/`",
        f"- Table 5 artifacts: `{TABLE5_ROOT}/`",
        "",
    ])
    OUTPUT.write_text("\n".join(lines))
    print(f"report_validation=PASS output={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
