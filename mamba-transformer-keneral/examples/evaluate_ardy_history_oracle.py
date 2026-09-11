"""Evaluate oracle chunk routing with the frozen Stage-05 Minimal model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))

from train_ardy_minimal_cached import (  # noqa: E402
    TrajectoryDataset,
    _branch_zero,
    _call,
    _current_loss,
    _move,
    _reset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--minimal-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-trajectories", type=int, default=16)
    parser.add_argument("--window-size-chunks", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--diffusion-indices", default="first,middle,last")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_diffusion_indices(spec: str, steps: int) -> list[int]:
    aliases = {"first": 0, "middle": steps // 2, "last": steps - 1}
    result = []
    for item in spec.split(","):
        token = item.strip().lower()
        if not token:
            continue
        index = aliases[token] if token in aliases else int(token)
        if index < 0 or index >= steps:
            raise ValueError(f"diffusion index {index} outside [0, {steps})")
        result.append(index)
    result = sorted(set(result))
    if not result:
        raise ValueError("at least one diffusion index is required")
    return result


def _load_minimal(raw, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != "minimal":
        raise ValueError("--minimal-checkpoint must contain mode='minimal'")
    raw.root_model.load_state_dict(checkpoint["root_model"], strict=True)
    raw.body_model.load_state_dict(checkpoint["body_model"], strict=True)
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "sha256": _sha256(checkpoint_path),
        "training": checkpoint.get("training"),
    }


def _commit_selected_history(
    raw,
    sample: dict[str, Any],
    current_chunk: int,
    selected_chunks: Iterable[int],
) -> None:
    selected = sorted(set(selected_chunks))
    if any(chunk < 0 or chunk >= current_chunk for chunk in selected):
        raise ValueError("selected history must precede the current chunk")
    if not selected:
        return
    steps = len(sample["calls"]) // sample["num_chunks"]
    clean_call = _call(sample, current_chunk, steps - 1)
    if int(clean_call["diffusion_timestep"].max()) != 0:
        raise RuntimeError("oracle history must use accepted timestep-zero chunks")
    root_device = raw.root_model._stack_parameter().device
    body_device = raw.body_model._stack_parameter().device
    root_inputs = _move(_branch_zero(clean_call["root"]["inputs"]), root_device)
    body_inputs = _move(_branch_zero(clean_call["body"]["inputs"]), body_device)
    history_valid = clean_call["outer_inputs"]["history_token_mask"][:1].bool()
    tokens_per_chunk = sample["chunk_frames"] // raw.num_frames_per_token
    with torch.no_grad():
        for history_chunk in selected:
            selection = torch.zeros_like(root_inputs[1], dtype=torch.bool)
            start = history_chunk * tokens_per_chunk
            selection[:, start : start + tokens_per_chunk] = True
            if bool((selection.cpu() & ~history_valid).any()):
                raise RuntimeError("oracle selected invalid/non-history tokens")
            raw.root_model.prepare_accepted_commit(selection, chunk_id=history_chunk)
            raw.root_model(*root_inputs)
            raw.body_model.prepare_accepted_commit(selection, chunk_id=history_chunk)
            raw.body_model(*body_inputs)


def _evaluate_subset(
    raw,
    sample: dict[str, Any],
    current_chunk: int,
    diffusion_index: int,
    selected_chunks: tuple[int, ...],
    device: torch.device,
    autocast_enabled: bool,
) -> dict[str, Any]:
    _reset(raw)
    _commit_selected_history(raw, sample, current_chunk, selected_chunks)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        total, root_loss, body_loss, _, _ = _current_loss(
            raw, sample, current_chunk, diffusion_index, device
        )
    if not torch.isfinite(total):
        raise RuntimeError("non-finite oracle denoising loss")
    result = {
        "chunks": list(selected_chunks),
        "loss": float(total),
        "root_loss": float(root_loss),
        "body_loss": float(body_loss),
    }
    _reset(raw)
    return result


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"cases": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "cases": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _paired_summary(
    candidate: list[float], baseline: list[float]
) -> dict[str, float | int]:
    if len(candidate) != len(baseline):
        raise ValueError("paired metrics require equal case counts")
    gains = [base - value for value, base in zip(candidate, baseline)]
    relative = [gain / max(abs(base), 1e-12) for gain, base in zip(gains, baseline)]
    tolerance = 1e-12
    return {
        "cases": len(candidate),
        "mean_absolute_gain": statistics.fmean(gains),
        "median_absolute_gain": statistics.median(gains),
        "mean_relative_gain": statistics.fmean(relative),
        "median_relative_gain": statistics.median(relative),
        "fraction_strictly_better": sum(gain > tolerance for gain in gains) / len(gains),
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    policies = ["no_history", "recent", "oldest", "full", "oracle_top1"]
    metrics = {
        policy: _summary([case[policy]["loss"] for case in cases])
        for policy in policies
    }
    top2_cases = [case for case in cases if case["oracle_top2"] is not None]
    metrics["oracle_top2"] = _summary(
        [case["oracle_top2"]["loss"] for case in top2_cases]
    )
    comparisons = {
        "oracle_top1_vs_full": _paired_summary(
            [case["oracle_top1"]["loss"] for case in cases],
            [case["full"]["loss"] for case in cases],
        ),
        "oracle_top1_vs_recent": _paired_summary(
            [case["oracle_top1"]["loss"] for case in cases],
            [case["recent"]["loss"] for case in cases],
        ),
    }
    if top2_cases:
        comparisons["oracle_top2_vs_full"] = _paired_summary(
            [case["oracle_top2"]["loss"] for case in top2_cases],
            [case["full"]["loss"] for case in top2_cases],
        )
    top1_distances = Counter(
        distance for case in cases for distance in case["oracle_top1"]["distances"]
    )
    top2_distances = Counter(
        distance
        for case in top2_cases
        for distance in case["oracle_top2"]["distances"]
    )
    return {
        "policy_loss": metrics,
        "paired_comparisons": comparisons,
        "oracle_top1_distance_counts": {
            str(key): top1_distances[key] for key in sorted(top1_distances)
        },
        "oracle_top2_distance_counts": {
            str(key): top2_distances[key] for key in sorted(top2_distances)
        },
        "oracle_top1_recent_fraction": sum(
            case["oracle_top1"]["distances"] == [1] for case in cases
        )
        / len(cases),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    sys.path.insert(0, str(Path(args.ardy_root).resolve()))
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model(
        args.teacher_model,
        device=args.device,
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )
    install_ardy_sequence_kernel(
        model,
        ArdySequenceKernelConfig(
            mode="minimal",
            attention_backend="sdpa",
            window_size_chunks=args.window_size_chunks,
            num_sink_tokens=args.sink_tokens,
        ),
    )
    raw = model.denoiser.model
    initialization = _load_minimal(raw, args.minimal_checkpoint, device)
    if any("mamba" in type(module).__name__.lower() for module in raw.modules()):
        raise RuntimeError("frozen Minimal unexpectedly instantiated Mamba")
    if "mamba_ssm" in sys.modules:
        raise RuntimeError("frozen Minimal unexpectedly imported mamba_ssm")
    raw.eval()
    for parameter in raw.parameters():
        parameter.requires_grad_(False)

    dataset = TrajectoryDataset(args.manifest, args.max_trajectories)
    diffusion_indices = _resolve_diffusion_indices(
        args.diffusion_indices, dataset.manifest["diffusion_steps"]
    )
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    with torch.inference_mode():
        for trajectory_index, sample in enumerate(dataset.samples):
            for current_chunk in range(1, sample["num_chunks"]):
                available = tuple(
                    range(max(0, current_chunk - args.window_size_chunks), current_chunk)
                )
                singleton_subsets = [(chunk,) for chunk in available]
                pair_subsets = list(itertools.combinations(available, 2))
                for diffusion_index in diffusion_indices:
                    subset_results = {
                        subset: _evaluate_subset(
                            raw,
                            sample,
                            current_chunk,
                            diffusion_index,
                            subset,
                            device,
                            autocast_enabled,
                        )
                        for subset in singleton_subsets + pair_subsets
                    }
                    no_history = _evaluate_subset(
                        raw, sample, current_chunk, diffusion_index, (), device, autocast_enabled
                    )
                    full = _evaluate_subset(
                        raw,
                        sample,
                        current_chunk,
                        diffusion_index,
                        available,
                        device,
                        autocast_enabled,
                    )
                    recent = subset_results[(available[-1],)]
                    oldest = subset_results[(available[0],)]
                    top1 = min(
                        (subset_results[subset] for subset in singleton_subsets),
                        key=lambda result: result["loss"],
                    )
                    top2 = (
                        min(
                            (subset_results[subset] for subset in pair_subsets),
                            key=lambda result: result["loss"],
                        )
                        if pair_subsets
                        else None
                    )
                    timestep = _call(sample, current_chunk, diffusion_index)[
                        "diffusion_timestep"
                    ]
                    case = {
                        "trajectory_index": trajectory_index,
                        "prompt_id": sample["prompt_id"],
                        "seed": sample["seed"],
                        "current_chunk": current_chunk,
                        "diffusion_index": diffusion_index,
                        "diffusion_timestep": int(timestep[0]),
                        "available_chunks": list(available),
                        "no_history": no_history,
                        "recent": recent,
                        "oldest": oldest,
                        "full": full,
                        "oracle_top1": {
                            **top1,
                            "distances": [current_chunk - chunk for chunk in top1["chunks"]],
                        },
                        "oracle_top2": (
                            None
                            if top2 is None
                            else {
                                **top2,
                                "distances": [
                                    current_chunk - chunk for chunk in top2["chunks"]
                                ],
                            }
                        ),
                        "singleton_candidates": [
                            subset_results[subset] for subset in singleton_subsets
                        ],
                        "pair_candidates": [subset_results[subset] for subset in pair_subsets],
                    }
                    cases.append(case)
                    print(
                        json.dumps(
                            {
                                "case": len(cases),
                                "prompt_id": sample["prompt_id"],
                                "chunk": current_chunk,
                                "diffusion_index": diffusion_index,
                                "full": full["loss"],
                                "recent": recent["loss"],
                                "oracle_top1": top1["loss"],
                                "oracle_top1_chunks": top1["chunks"],
                                "oracle_top2": None if top2 is None else top2["loss"],
                                "oracle_top2_chunks": None if top2 is None else top2["chunks"],
                            }
                        ),
                        flush=True,
                    )

    _reset(raw)
    elapsed = time.perf_counter() - started
    result = {
        "format_version": 1,
        "phase": "stage06r_gate_a_oracle_history",
        "config": {
            **vars(args),
            "manifest": str(Path(args.manifest).resolve()),
            "output_dir": str(output_dir),
            "git_commit": _git_commit(),
            "diffusion_indices_resolved": diffusion_indices,
        },
        "minimal_initialization": initialization,
        "dataset": {
            "trajectories": len(dataset),
            "num_chunks": dataset.manifest["num_chunks"],
            "diffusion_steps": dataset.manifest["diffusion_steps"],
            "teacher_model": dataset.manifest["teacher_model"],
            "teacher_revision": dataset.manifest["teacher_revision"],
            "replay_verification": dataset.manifest["replay_verification"],
        },
        "aggregate": _aggregate(cases),
        "elapsed_seconds": elapsed,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "cases": cases,
    }
    (output_dir / "oracle_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
