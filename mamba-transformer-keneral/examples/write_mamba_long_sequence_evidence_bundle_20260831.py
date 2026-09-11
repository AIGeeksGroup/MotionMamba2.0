"""Write the final machine-readable long-sequence evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path("/data3/zts/MotionMamba2.0/mamba-transformer-keneral")
RESULT = Path("/data4/zts/result")
OUT = RESULT / "mamba_long_sequence_evidence_20260831"
STAGE_A = RESULT / "bones_seed_long_memory_stage_a_20260831"
STAGE_B = RESULT / "bones_seed_long_memory_stage_b_20260831"
DECAY = RESULT / "bones_seed_long_memory_stage_b_decay_20260831"
EVAL = RESULT / "bones_seed_long_memory_stage_b_selection_eval_20260831"
BENCH = RESULT / "bones_seed_long_sequence_scaling_v2_20260831"
PANELS = RESULT / "bones_seed_long_memory_panel_v2_20260831"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = read_json(EVAL / "gate_summary.json")
    benchmark = read_json(BENCH / "summary.json")
    protocol = read_json(REPO / "configs/bones_seed_long_memory_protocol_v2.json")
    key_metrics = {}
    for stage in (100, 250):
        stage_key = str(stage)
        key_metrics[stage_key] = {
            "hybrid_decay_carry_vs_minimal_w4": gate["stages"][stage_key][
                "hybrid_decay_carry_vs_minimal_w4"
            ],
            "same_checkpoint_causal_interventions": gate["stages"][stage_key][
                "same_checkpoint_causal_interventions"
            ],
            "criteria": gate["stages"][stage_key]["criteria"],
        }
    summary = {
        "verdict": "NOT SUPPORTED",
        "claim": "stable Mamba-specific long-memory or system-efficiency advantage",
        "reason": (
            "The preregistered 100/250 gate failed: Hybrid carry did not beat "
            "Minimal-W4 on every 20/40-second Root/constraint AUC and slope, and "
            "carry did not beat same-checkpoint permutation on every post-W5 slope."
        ),
        "protocol_version": protocol["protocol_version"],
        "selection_only": True,
        "locked_final_status": "unread",
        "stage_c_1k": "not_run_hard_stop",
        "stage_d_5k": "not_run_hard_stop",
        "learned_metrics": {
            "r_precision": None,
            "fid": None,
            "fvmd": None,
            "tm_sim": None,
            "reason": "compatible official learned evaluator unavailable",
        },
        "tests": {
            "pytest": "115 passed, 6 skipped",
            "cuda_rollout": "1 passed",
            "compileall": "PASS",
        },
        "stage_a_complete": (STAGE_A / "STAGE_A_COMPLETE").exists(),
        "stage_b_training_complete": (STAGE_B / "STAGE_B_TRAINING_COMPLETE").exists(),
        "stage_b_decay_training_complete": (
            DECAY / "STAGE_B_DECAY_TRAINING_COMPLETE"
        ).exists(),
        "stage_b_gate": gate,
        "key_metrics": key_metrics,
        "system_benchmark": benchmark,
        "artifacts": {
            "protocol_lock": str(
                (REPO / "configs/bones_seed_long_memory_protocol_v2.json").resolve()
            ),
            "panels": str(PANELS.resolve()),
            "stage_a": str(STAGE_A.resolve()),
            "stage_b_no_decay": str(STAGE_B.resolve()),
            "stage_b_decay": str(DECAY.resolve()),
            "selection_evaluation": str(EVAL.resolve()),
            "benchmark": str(BENCH.resolve()),
            "audit_report": str(
                (REPO / "docs/MAMBA_LONG_SEQUENCE_EVIDENCE_AUDIT_20260831.md").resolve()
            ),
            "reproduction": str(
                (REPO / "docs/MAMBA_LONG_SEQUENCE_REPRODUCTION_20260831.md").resolve()
            ),
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    ledger = []
    for name_dir in sorted(path for path in STAGE_A.iterdir() if path.is_dir()):
        metrics = read_json(name_dir / "metrics.json")
        ledger.append({
            "stage": "A_20_step_smoke",
            "name": name_dir.name,
            "status": metrics["status"],
            "mechanism_pass": bool(metrics.get("all_gradients_finite")),
            "differentiable_rollout_steps": metrics.get("differentiable_rollout_steps"),
            "peak_memory_mib": metrics.get("peak_memory_mib"),
            "artifact": str(name_dir.resolve()),
        })
    for root, label in ((STAGE_B, "B_no_decay"), (DECAY, "B_registered_decay")):
        for name_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            metrics_path = name_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = read_json(metrics_path)
            ledger.append({
                "stage": label,
                "name": name_dir.name,
                "steps": 250,
                "status": metrics["status"],
                "all_gradients_finite": metrics.get("all_gradients_finite"),
                "differentiable_rollout_steps": metrics.get("differentiable_rollout_steps"),
                "peak_memory_mib": metrics.get("peak_memory_mib"),
                "artifact": str(name_dir.resolve()),
            })
    for stage in (100, 250):
        for name_dir in sorted(path for path in (EVAL / f"stage{stage:03d}").iterdir() if path.is_dir()):
            results = sorted(name_dir.glob("*/result.json"))
            ledger.append({
                "stage": f"B_selection_eval_step_{stage}",
                "name": name_dir.name,
                "panels": [path.parent.name for path in results],
                "all_outputs_finite": all(read_json(path)["status"] == "PASS" for path in results),
                "artifact": str(name_dir.resolve()),
            })
    ledger.append({
        "stage": "B_gate",
        "name": "preregistered_gate",
        "status": gate["status"],
        "next_stage_authorized": gate["next_stage_authorized"],
        "artifact": str((EVAL / "gate_summary.json").resolve()),
    })
    ledger.append({
        "stage": "system_benchmark",
        "name": "real_checkpoint_root_blocks",
        "status": benchmark["status"],
        "scope": benchmark["scope"],
        "artifact": str((BENCH / "summary.json").resolve()),
    })
    (OUT / "ablation_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger)
    )

    checksum_paths = []
    checksum_paths.extend(sorted(PANELS.rglob("*.jsonl")))
    checksum_paths.extend(sorted(EVAL.rglob("result.json")))
    checksum_paths.extend(sorted(EVAL.rglob("per_sample.jsonl")))
    checksum_paths.extend(sorted(EVAL.rglob("state.jsonl")))
    checksum_paths.extend(sorted(EVAL.rglob("*.interventions.jsonl")))
    checksum_paths.extend(sorted(EVAL.rglob("*.npz")))
    checksum_paths.extend(sorted(BENCH.glob("*")))
    checksum_paths.extend(sorted(STAGE_B.rglob("checkpoint_step_*.pt")))
    checksum_paths.extend(sorted(DECAY.rglob("checkpoint_step_*.pt")))
    checksum_paths.extend([
        REPO / "configs/bones_seed_long_memory_protocol_v2.json",
        REPO / "docs/MAMBA_LONG_SEQUENCE_EVIDENCE_AUDIT_20260831.md",
        REPO / "docs/MAMBA_LONG_SEQUENCE_REPRODUCTION_20260831.md",
        OUT / "summary.json",
        OUT / "ablation_ledger.jsonl",
    ])
    unique_paths = sorted({path.resolve() for path in checksum_paths if path.is_file()})
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in unique_paths)
    )
    print(json.dumps({
        "verdict": summary["verdict"],
        "ledger_rows": len(ledger),
        "checksums": len(unique_paths),
        "output": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
