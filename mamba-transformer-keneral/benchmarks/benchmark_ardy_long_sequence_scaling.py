"""Serialized real-block long-sequence scaling benchmark."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples"))
sys.path.insert(0, str(REPO.parents[0] / "ardy"))
for path in (
    os.environ.get("MAMBA_REPO", "/data3/zts/packages/state-spaces-mamba"),
    os.environ.get("MAMBA_TRITON_OVERLAY", "/data3/zts/packages/mamba_op_triton"),
):
    sys.path.insert(0, path)

from train_humanml3d_ardy_tiny import _load_denoiser_weights  # noqa: E402


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _state_bytes(block) -> tuple[int, int]:
    attention = recurrent = 0
    state = getattr(block, "_sequence_state", None)
    if state is None:
        return attention, recurrent
    for layer in state.layers:
        for chunk in layer.attention_cache.chunks:
            attention += sum(
                tensor.numel() * tensor.element_size()
                for tensor in (chunk.key, chunk.value, chunk.token_valid)
            )
        mamba = layer.mamba_state
        if mamba is not None:
            for name in ("angle_dt_state", "ssm_state", "k_state", "v_state"):
                tensor = getattr(mamba, name, None)
                if isinstance(tensor, torch.Tensor):
                    recurrent += tensor.numel() * tensor.element_size()
        for name in ("state_summary", "state_delta"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor):
                recurrent += tensor.numel() * tensor.element_size()
    return attention, recurrent


def _load_block(mode: str, checkpoint_path: str, device: torch.device, window: int):
    from ardy.model import load_model
    from ardy_adapter import ArdySequenceKernelConfig, install_ardy_sequence_kernel

    model = load_model("g1", device=str(device), text_encoder=False)
    if mode != "original":
        install_ardy_sequence_kernel(
            model,
            ArdySequenceKernelConfig(
                mode=mode,
                attention_backend="sdpa",
                window_size_chunks=window,
                num_sink_tokens=4,
            ),
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _load_denoiser_weights(model.denoiser.model, checkpoint)
    block = model.denoiser.model.root_model.eval()
    active = sum(
        parameter.numel()
        for part in (
            model.denoiser.model.root_model,
            model.denoiser.model.body_model,
        )
        for parameter in part.parameters()
    )
    return model, block, active


def _input_dim(block) -> int:
    projection = block.input_linear
    return projection.in_features if isinstance(projection, nn.Linear) else int(block.latent_dim)


def _run_once(
    mode: str,
    block,
    chunks: list[torch.Tensor],
    text: torch.Tensor,
    *,
    tokens_per_window: int,
) -> dict:
    if hasattr(block, "reset_memory"):
        block.reset_memory()
    batch = chunks[0].shape[0]
    text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
    timestep = torch.zeros(batch, dtype=torch.long, device=text.device)
    heading = torch.zeros(batch, device=text.device)
    cuda_ms = []
    wall_seconds = []
    history = []
    for index, chunk in enumerate(chunks):
        if mode == "original":
            history.append(chunk)
            value = torch.cat(history, dim=1)
            token_index = torch.arange(
                value.shape[1], device=value.device
            )[None].expand(batch, -1) - (value.shape[1] - tokens_per_window)
        else:
            value = chunk
            token_index = torch.arange(
                tokens_per_window, device=value.device
            )[None].expand(batch, -1)
            block.prepare_accepted_commit(
                torch.ones(batch, tokens_per_window, dtype=torch.bool, device=value.device),
                chunk_id=index,
            )
        pad = torch.ones(batch, value.shape[1], dtype=torch.bool, device=value.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(value.device)
        wall_start = time.perf_counter()
        start_event.record()
        output = block(
            value,
            pad,
            text,
            text_mask,
            timestep,
            heading,
            token_index,
        )
        end_event.record()
        torch.cuda.synchronize(value.device)
        wall_seconds.append(time.perf_counter() - wall_start)
        cuda_ms.append(start_event.elapsed_time(end_event))
        if not torch.isfinite(output.float()).all():
            raise RuntimeError("benchmark produced non-finite output")
    attention_bytes, recurrent_bytes = _state_bytes(block)
    return {
        "cuda_ms_by_window": cuda_ms,
        "wall_seconds_by_window": wall_seconds,
        "attention_kv_bytes": attention_bytes,
        "mamba_state_bytes": recurrent_bytes,
    }


def _fit(records: list[dict], mode: str, seed: int, resamples: int = 2000) -> dict:
    rows = [row for row in records if row["mode"] == mode]
    x = np.asarray([row["windows"] for row in rows], dtype=np.float64)
    y = np.asarray([row["full_sequence_cuda_ms_median"] for row in rows], dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    alpha = np.polyfit(np.log(x), np.log(np.maximum(y, 1e-9)), 1)[0]
    rng = np.random.default_rng(seed)
    boot_a = []
    boot_alpha = []
    raw_by_row = [np.asarray(row["full_sequence_cuda_ms_raw"]) for row in rows]
    for _ in range(resamples):
        sampled = np.asarray(
            [values[rng.integers(0, len(values), size=len(values))].mean() for values in raw_by_row]
        )
        boot_a.append(np.polyfit(x, sampled, 1)[0])
        boot_alpha.append(
            np.polyfit(np.log(x), np.log(np.maximum(sampled, 1e-9)), 1)[0]
        )
    memory = np.asarray([row["persistent_state_bytes"] for row in rows], dtype=np.float64)
    memory_slope = np.polyfit(x, memory, 1)[0]
    capacity_rows = [row for row in rows if row["windows"] >= 4]
    capacity_x = np.asarray([row["windows"] for row in capacity_rows], dtype=np.float64)
    capacity_memory = np.asarray(
        [row["persistent_state_bytes"] for row in capacity_rows], dtype=np.float64
    )
    post_capacity_slope = np.polyfit(capacity_x, capacity_memory, 1)[0]
    return {
        "time_a_cuda_ms_per_window": float(a),
        "time_b_cuda_ms": float(b),
        "time_a_ci95": [float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5))],
        "loglog_alpha": float(alpha),
        "loglog_alpha_ci95": [float(np.percentile(boot_alpha, 2.5)), float(np.percentile(boot_alpha, 97.5))],
        "persistent_memory_slope_bytes_per_window": float(memory_slope),
        "persistent_memory_slope_after_w4_bytes_per_window": float(
            post_capacity_slope
        ),
        "persistent_state_bytes_constant_after_w4": bool(
            np.all(capacity_memory == capacity_memory[0])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ardy-checkpoint", required=True)
    parser.add_argument("--minimal-checkpoint", required=True)
    parser.add_argument("--hybrid-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens-per-window", type=int, default=13)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--attention-window-chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    checkpoints = {
        "ardy_explicit_full_history": args.ardy_checkpoint,
        "minimal_sliding_w4": args.minimal_checkpoint,
        "hybrid_w4_carry": args.hybrid_checkpoint,
    }
    records = []
    max_windows = max(args.windows)
    for mode_name, checkpoint in checkpoints.items():
        architecture = (
            "original" if mode_name.startswith("ardy")
            else "minimal" if mode_name.startswith("minimal") else "hybrid"
        )
        model, block, active = _load_block(
            architecture, checkpoint, device, args.attention_window_chunks
        )
        generator = torch.Generator(device=device).manual_seed(args.seed)
        chunks = [
            torch.randn(
                args.batch,
                args.tokens_per_window,
                _input_dim(block),
                generator=generator,
                device=device,
                dtype=next(block.parameters()).dtype,
            )
            for _ in range(max_windows)
        ]
        text = torch.randn(
            args.batch,
            int(block.num_text_tokens),
            int(block.embed_text.in_features),
            generator=generator,
            device=device,
            dtype=block.embed_text.weight.dtype,
        )
        for windows in args.windows:
            for _ in range(args.warmup):
                with torch.inference_mode():
                    _run_once(
                        architecture, block, chunks[:windows], text,
                        tokens_per_window=args.tokens_per_window,
                    )
            torch.cuda.reset_peak_memory_stats(device)
            raw = []
            for _ in range(args.repeats):
                with torch.inference_mode():
                    raw.append(
                        _run_once(
                            architecture, block, chunks[:windows], text,
                            tokens_per_window=args.tokens_per_window,
                        )
                    )
            full_cuda = [sum(row["cuda_ms_by_window"]) for row in raw]
            full_wall = [sum(row["wall_seconds_by_window"]) for row in raw]
            first_cuda = [row["cuda_ms_by_window"][0] for row in raw]
            steady_cuda = [
                float(np.mean(row["cuda_ms_by_window"][1:]))
                if windows > 1 else row["cuda_ms_by_window"][0]
                for row in raw
            ]
            attention_bytes = max(row["attention_kv_bytes"] for row in raw)
            recurrent_bytes = max(row["mamba_state_bytes"] for row in raw)
            records.append({
                "scope": "real_checkpoint_root_block",
                "mode": mode_name,
                "windows": windows,
                "batch": args.batch,
                "tokens_per_window": args.tokens_per_window,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "first_window_cuda_ms_median": float(np.median(first_cuda)),
                "steady_window_cuda_ms_median": float(np.median(steady_cuda)),
                "full_sequence_cuda_ms_median": float(np.median(full_cuda)),
                "full_sequence_cuda_ms_p10": float(np.percentile(full_cuda, 10)),
                "full_sequence_cuda_ms_p90": float(np.percentile(full_cuda, 90)),
                "full_sequence_wall_seconds_median": float(np.median(full_wall)),
                "full_sequence_cuda_ms_raw": full_cuda,
                "full_sequence_wall_seconds_raw": full_wall,
                "tokens_per_second": float(args.batch * windows * args.tokens_per_window / (np.median(full_cuda) / 1000.0)),
                "frames_per_second": float(args.batch * windows * args.tokens_per_window * 4 / (np.median(full_cuda) / 1000.0)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "attention_kv_bytes": attention_bytes,
                "mamba_state_bytes": recurrent_bytes,
                "persistent_state_bytes": attention_bytes + recurrent_bytes,
                "active_parameters": active,
                "checkpoint": str(Path(checkpoint).resolve()),
            })
        del model, block
        torch.cuda.empty_cache()
    fits = {
        mode: _fit(records, mode, args.seed + index)
        for index, mode in enumerate(checkpoints)
    }
    environment = {
        "gpu": torch.cuda.get_device_name(device),
        "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "mamba_ssm": _package_version("mamba-ssm"),
        "triton": _package_version("triton"),
        "platform": platform.platform(),
        "device": str(device),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    csv_rows = [{key: value for key, value in row.items() if not isinstance(value, list)} for row in records]
    with (output / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "status": "PASS",
        "scope": "real trained root blocks; quality evaluation is separate",
        "environment": environment,
        "fits": fits,
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(fits, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
