"""Exact Stage-04 FlashAttention-2 acceptance case with SDPA parity."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import statistics
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_attention_kernel import (  # noqa: E402
    MinimalAttentionKernel,
    MinimalAttentionKernelConfig,
    flash2_attention_available,
)


def _sdpa_math_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    )


def _latency_ms(kernel, current, cache, iterations: int, *, sdpa_math: bool) -> float:
    samples = []
    with torch.no_grad():
        for index in range(5 + iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with _sdpa_math_context(sdpa_math):
                kernel.forward_chunk(current, cache=cache, commit=False)
            torch.cuda.synchronize()
            if index >= 5:
                samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, reason = flash2_attention_available()
    if not available:
        raise RuntimeError(reason)

    torch.manual_seed(41)
    device = torch.device("cuda:0")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    common = dict(
        d_model=128,
        num_heads=4,
        window_size_chunks=2,
        use_attention_sinks=True,
        num_sink_tokens=2,
        allow_backend_fallback=False,
    )
    sdpa = MinimalAttentionKernel(
        MinimalAttentionKernelConfig(**common, attention_backend="sdpa")
    ).to(device=device, dtype=dtype).eval()
    flash2 = MinimalAttentionKernel(
        MinimalAttentionKernelConfig(**common, attention_backend="flash2")
    ).to(device=device, dtype=dtype).eval()
    flash2.load_state_dict(sdpa.state_dict())
    caches = [sdpa.init_cache(batch_size=1), flash2.init_cache(batch_size=1)]
    histories = [
        torch.randn(1, 16, 128, device=device, dtype=dtype) for _ in range(3)
    ]
    for kernel, cache in zip((sdpa, flash2), caches):
        for chunk_id, history in enumerate(histories):
            _, pending, _ = kernel.forward_chunk(history, cache=cache, commit=False)
            kernel.commit(cache, pending, chunk_id=chunk_id)

    current_sdpa = torch.randn(
        1, 16, 128, device=device, dtype=dtype, requires_grad=True
    )
    current_flash2 = current_sdpa.detach().clone().requires_grad_(True)
    with _sdpa_math_context(True):
        out_sdpa, _, _ = sdpa.forward_chunk(
            current_sdpa, cache=caches[0], commit=False
        )
    out_flash2, _, aux = flash2.forward_chunk(
        current_flash2, cache=caches[1], commit=False
    )
    loss = out_flash2.float().square().mean()
    loss.backward()

    error = (out_flash2.float() - out_sdpa.float()).abs()
    max_absolute_error = error.max().item()
    mean_absolute_error = error.mean().item()
    relative_error = (
        error.mean() / out_sdpa.float().abs().mean().clamp_min(1e-8)
    ).item()
    torch.testing.assert_close(out_flash2, out_sdpa, atol=4e-3, rtol=4e-3)
    assert aux["backend_used"] == "flash2"
    assert aux["query_tokens"] == 16 and aux["kv_tokens"] == 50
    assert caches[1].chunk_ids == (1, 2)
    assert current_flash2.grad is not None and torch.isfinite(current_flash2.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in flash2.parameters()
    )

    torch.cuda.reset_peak_memory_stats(device)
    sdpa_latency = _latency_ms(
        sdpa, current_sdpa.detach(), caches[0], args.iterations, sdpa_math=True
    )
    sdpa_peak = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    flash2_latency = _latency_ms(
        flash2,
        current_flash2.detach(),
        caches[1],
        args.iterations,
        sdpa_math=False,
    )
    flash2_peak = torch.cuda.max_memory_allocated(device)
    report = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "shape": {"batch": 1, "d_model": 128, "heads": 4, "chunk_len": 16},
        "history_chunks": 2,
        "sink_tokens": 2,
        "query_tokens": aux["query_tokens"],
        "kv_tokens": aux["kv_tokens"],
        "commit_false_cache_ids": list(caches[1].chunk_ids),
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "relative_error": relative_error,
        "sdpa_median_latency_ms": sdpa_latency,
        "flash2_median_latency_ms": flash2_latency,
        "sdpa_peak_allocated_mib": sdpa_peak / 2**20,
        "flash2_peak_allocated_mib": flash2_peak / 2**20,
        "flash2_backward_finite": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
