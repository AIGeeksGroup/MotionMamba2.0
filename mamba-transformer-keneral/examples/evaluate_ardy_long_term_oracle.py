"""Gate A: frozen-Minimal oracle ceiling for out-of-window accepted K/V."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from train_ardy_minimal_cached import _branch_zero, _move, _reset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-root", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--teacher-model", default="Ardy-Core-RP-20FPS-Horizon40")
    parser.add_argument("--minimal-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size-chunks", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
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


def _load_samples(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text())
    samples = [
        torch.load(
            manifest_path.parent / record["path"],
            map_location="cpu",
            weights_only=False,
        )
        for record in manifest["records"]
    ]
    if not samples:
        raise ValueError("long-horizon manifest is empty")
    return manifest, samples


def _load_minimal(raw, path: str, device: torch.device) -> dict[str, str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != "minimal":
        raise ValueError("checkpoint is not Stage-05 Minimal")
    raw.root_model.load_state_dict(checkpoint["root_model"], strict=True)
    raw.body_model.load_state_dict(checkpoint["body_model"], strict=True)
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path)}


def _clean_call(sample: dict[str, Any]) -> dict[str, Any]:
    call = max(sample["calls"], key=lambda item: item["diffusion_call_index"])
    if int(call["diffusion_timestep"].max()) != 0:
        raise RuntimeError("last captured call is not accepted timestep zero")
    return call


def _commit_all_history(raw, sample: dict[str, Any]) -> None:
    call = _clean_call(sample)
    root_device = raw.root_model._stack_parameter().device
    body_device = raw.body_model._stack_parameter().device
    root_inputs = _move(_branch_zero(call["root"]["inputs"]), root_device)
    body_inputs = _move(_branch_zero(call["body"]["inputs"]), body_device)
    history_valid = call["outer_inputs"]["history_token_mask"][:1].bool()
    tokens_per_chunk = sample["chunk_frames"] // raw.num_frames_per_token
    with torch.no_grad():
        for chunk in range(sample["target_chunk"]):
            selection = torch.zeros_like(root_inputs[1], dtype=torch.bool)
            start = chunk * tokens_per_chunk
            selection[:, start : start + tokens_per_chunk] = True
            if bool((selection.cpu() & ~history_valid).any()):
                raise RuntimeError("long-term oracle selected invalid history tokens")
            raw.root_model.prepare_accepted_commit(selection, chunk_id=chunk)
            raw.root_model(*root_inputs)
            raw.body_model.prepare_accepted_commit(selection, chunk_id=chunk)
            raw.body_model(*body_inputs)


def _clone_states(raw):
    if raw.root_model._sequence_state is None or raw.body_model._sequence_state is None:
        raise RuntimeError("history state was not initialized")
    return (
        raw.root_model._sequence_state.clone(),
        raw.body_model._sequence_state.clone(),
    )


def _activate_chunks(block, full_state, chunks: tuple[int, ...]) -> None:
    keep = set(chunks)
    state = full_state.clone()
    for layer_state in state.layers:
        cache = layer_state.attention_cache
        cache._chunks = [chunk for chunk in cache._chunks if chunk.chunk_id in keep]
        if tuple(cache.chunk_ids) != tuple(sorted(chunks)):
            raise RuntimeError(
                f"activated cache IDs {cache.chunk_ids} do not match {sorted(chunks)}"
            )
    block._sequence_state = state


def _evaluate_subset(
    raw,
    call: dict[str, Any],
    root_state,
    body_state,
    chunks: tuple[int, ...],
    device: torch.device,
    autocast_enabled: bool,
) -> dict[str, Any]:
    chunks = tuple(sorted(chunks))
    _activate_chunks(raw.root_model, root_state, chunks)
    _activate_chunks(raw.body_model, body_state, chunks)
    root_inputs = _move(call["root"]["inputs"], device)
    body_inputs = _move(call["body"]["inputs"], device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        root_prediction = raw.root_model(*root_inputs)
        body_prediction = raw.body_model(*body_inputs)
        root_loss = F.mse_loss(
            root_prediction.float(), call["root"]["teacher_output"].to(device).float()
        )
        body_loss = F.mse_loss(
            body_prediction.float(), call["body"]["teacher_output"].to(device).float()
        )
        total = root_loss + body_loss
    if not torch.isfinite(total):
        raise RuntimeError("non-finite long-term oracle loss")
    return {
        "chunks": list(chunks),
        "loss": float(total),
        "root_loss": float(root_loss),
        "body_loss": float(body_loss),
    }


def _summary(records: list[dict[str, Any]], policy: str) -> dict[str, float | int]:
    values = [record[policy]["loss"] for record in records]
    return {
        "cases": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _paired(records: list[dict[str, Any]], policy: str) -> dict[str, float | int]:
    gains = [
        record["minimal_recent"]["loss"] - record[policy]["loss"]
        for record in records
    ]
    relative = [
        gain / max(abs(record["minimal_recent"]["loss"]), 1e-12)
        for gain, record in zip(gains, records)
    ]
    by_trajectory: dict[tuple[int, int], list[float]] = {}
    for record, gain in zip(records, gains):
        by_trajectory.setdefault(
            (record["horizon"], record["prompt_id"]), []
        ).append(gain)
    trajectory_gains = [statistics.fmean(values) for values in by_trajectory.values()]
    rng = random.Random(0)
    bootstrap = sorted(
        statistics.fmean(rng.choice(trajectory_gains) for _ in trajectory_gains)
        for _ in range(10000)
    )
    return {
        "mean_absolute_gain": statistics.fmean(gains),
        "median_absolute_gain": statistics.median(gains),
        "mean_relative_gain": statistics.fmean(relative),
        "fraction_strictly_better": sum(gain > 1e-12 for gain in gains) / len(gains),
        "trajectory_bootstrap_ci95_low": bootstrap[249],
        "trajectory_bootstrap_ci95_high": bootstrap[9749],
        "positive_trajectories": sum(gain > 0 for gain in trajectory_gains),
        "trajectories": len(trajectory_gains),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    policies = (
        "minimal_recent",
        "random_old",
        "latest_evicted",
        "oracle_top1",
        "oracle_top2",
    )
    horizons = sorted({record["horizon"] for record in records})
    result = {"overall": {}, "by_horizon": {}}
    for policy in policies:
        result["overall"][policy] = _summary(records, policy)
        if policy != "minimal_recent":
            result["overall"][policy]["paired_vs_minimal"] = _paired(records, policy)
    for horizon in horizons:
        selected = [record for record in records if record["horizon"] == horizon]
        result["by_horizon"][str(horizon)] = {}
        for policy in policies:
            item = _summary(selected, policy)
            if policy != "minimal_recent":
                item["paired_vs_minimal"] = _paired(selected, policy)
            result["by_horizon"][str(horizon)][policy] = item
    result["oracle_top1_distance_counts"] = {
        str(distance): count
        for distance, count in sorted(
            Counter(
                record["target_chunk"] - record["oracle_top1"]["chunks"][0]
                for record in records
            ).items()
        )
    }
    return result


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    manifest, samples = _load_samples(args.manifest)
    maximum_horizon = max(sample["num_chunks"] for sample in samples)

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
            window_size_chunks=maximum_horizon,
            num_sink_tokens=args.sink_tokens,
        ),
    )
    raw = model.denoiser.model
    initialization = _load_minimal(raw, args.minimal_checkpoint, device)
    if any("mamba" in type(module).__name__.lower() for module in raw.modules()):
        raise RuntimeError("Gate-A Minimal unexpectedly instantiated Mamba")
    if "mamba_ssm" in sys.modules:
        raise RuntimeError("Gate-A Minimal unexpectedly imported mamba_ssm")
    raw.eval()
    for parameter in raw.parameters():
        parameter.requires_grad_(False)
    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    started = time.perf_counter()
    records = []
    with torch.inference_mode():
        for sample in samples:
            _reset(raw)
            _commit_all_history(raw, sample)
            root_state, body_state = _clone_states(raw)
            target = sample["target_chunk"]
            recent = tuple(range(target - args.window_size_chunks, target))
            old = tuple(range(0, target - args.window_size_chunks))
            if not old:
                raise RuntimeError("Gate A sample has no out-of-window chunks")
            for call in sorted(sample["calls"], key=lambda item: item["diffusion_call_index"]):
                candidate = {}
                for old_chunk in old:
                    chunks = tuple(sorted(recent + (old_chunk,)))
                    candidate[old_chunk] = _evaluate_subset(
                        raw,
                        call,
                        root_state,
                        body_state,
                        chunks,
                        device,
                        autocast_enabled,
                    )
                best_old = min(candidate, key=lambda chunk: candidate[chunk]["loss"])
                top1 = candidate[best_old]
                pair_candidates = []
                for second in old:
                    if second == best_old:
                        continue
                    chunks = tuple(sorted(recent + (best_old, second)))
                    pair_candidates.append(
                        _evaluate_subset(
                            raw,
                            call,
                            root_state,
                            body_state,
                            chunks,
                            device,
                            autocast_enabled,
                        )
                    )
                top2 = min(pair_candidates, key=lambda item: item["loss"])
                rng = random.Random(
                    args.seed
                    + 100000 * sample["num_chunks"]
                    + 100 * sample["prompt_id"]
                    + call["diffusion_call_index"]
                )
                random_chunk = rng.choice(old)
                record = {
                    "horizon": sample["num_chunks"],
                    "prompt_id": sample["prompt_id"],
                    "seed": sample["seed"],
                    "target_chunk": target,
                    "diffusion_index": call["diffusion_call_index"],
                    "diffusion_timestep": int(call["diffusion_timestep"][0]),
                    "recent_chunks": list(recent),
                    "eligible_old_chunks": list(old),
                    "minimal_recent": _evaluate_subset(
                        raw,
                        call,
                        root_state,
                        body_state,
                        recent,
                        device,
                        autocast_enabled,
                    ),
                    "random_old": candidate[random_chunk],
                    "latest_evicted": candidate[old[-1]],
                    "oracle_top1": top1,
                    "oracle_top2": top2,
                    "singleton_candidates": list(candidate.values()),
                    "greedy_pair_candidates": pair_candidates,
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "horizon": record["horizon"],
                            "prompt_id": record["prompt_id"],
                            "diffusion_index": record["diffusion_index"],
                            "minimal": record["minimal_recent"]["loss"],
                            "oracle_top1": top1["loss"],
                            "oracle_top1_old": [
                                chunk for chunk in top1["chunks"] if chunk in old
                            ],
                            "oracle_top2": top2["loss"],
                            "oracle_top2_old": [
                                chunk for chunk in top2["chunks"] if chunk in old
                            ],
                        }
                    ),
                    flush=True,
                )
            _reset(raw)

    result = {
        "format_version": 1,
        "phase": "stage06s_gate_a_raw_old_kv_oracle",
        "config": {**vars(args), "git_commit": _git_commit()},
        "minimal_initialization": initialization,
        "dataset": {
            "manifest": str(Path(args.manifest).resolve()),
            "horizons": manifest["horizons"],
            "num_prompts": manifest["num_prompts"],
            "capture_diffusion_indices": manifest["capture_diffusion_indices"],
            "replay_verification": manifest["replay_verification"],
        },
        "semantics": {
            "short_term_window": args.window_size_chunks,
            "raw_injection": "exact accepted old token K/V in addition to recent W",
            "top2_search": "greedy: oracle Top-1 plus best second old chunk",
            "no_recent_long_term_duplication": True,
        },
        "aggregate": _aggregate(records),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "records": records,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
