"""Small latency/memory smoke benchmark for SDPA and FlashAttention backends."""

from __future__ import annotations

import argparse
import itertools
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
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_backend(
    backend: str,
    *,
    device: torch.device,
    batch_size: int,
    d_model: int,
    num_heads: int,
    chunk_length: int,
    window_chunks: int,
    sink_tokens: int,
    warmup: int,
    iterations: int,
) -> dict:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    config = MinimalAttentionKernelConfig(
        d_model=d_model,
        num_heads=num_heads,
        window_size_chunks=window_chunks,
        use_attention_sinks=sink_tokens > 0,
        num_sink_tokens=sink_tokens,
        attention_backend=backend,
        allow_backend_fallback=True,
    )
    kernel = MinimalAttentionKernel(config).to(device=device, dtype=dtype).eval()
    cache = kernel.init_cache(batch_size=batch_size)
    with torch.no_grad():
        for chunk_id in range(window_chunks):
            history = torch.randn(batch_size, chunk_length, d_model, device=device, dtype=dtype)
            _, pending, _ = kernel.forward_chunk(history, cache=cache)
            kernel.commit(cache, pending, chunk_id=chunk_id)

    x = torch.randn(batch_size, chunk_length, d_model, device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    aux = None
    with torch.no_grad():
        for index in range(warmup + iterations):
            synchronize(device)
            started = time.perf_counter()
            _, _, aux = kernel.forward_chunk(x, cache=cache, commit=False)
            synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if index >= warmup:
                latencies_ms.append(elapsed_ms)

    element_bytes = torch.empty((), dtype=dtype).element_size()
    cache_bytes = 2 * batch_size * num_heads * cache.num_history_tokens * config.resolved_head_dim * element_bytes
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "requested": backend,
        "used": aux["backend_used"],
        "fallback": aux["backend_fallback_reason"],
        "chunk_length": chunk_length,
        "window_chunks": window_chunks,
        "cached_tokens": cache.num_history_tokens,
        "kv_tokens_per_query": aux["kv_tokens"],
        "median_latency_ms": statistics.median(latencies_ms),
        "tokens_per_second": batch_size * chunk_length / (statistics.median(latencies_ms) / 1000.0),
        "cache_bytes": cache_bytes,
        "peak_gpu_bytes": peak_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--window-chunks", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the Stage-1 CUDA smoke matrix: B=1/4, L=16/32/64, W=1/2/4, sinks=2/4.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"device={device} torch={torch.__version__}")
    if args.matrix:
        cases = itertools.product((1, 4), (16, 32, 64), (1, 2, 4), (2, 4))
    else:
        cases = ((args.batch_size, args.chunk_length, args.window_chunks, args.sink_tokens),)
    for batch_size, chunk_length, window_chunks, sink_tokens in cases:
        for backend in ("sdpa", "flash2"):
            result = run_backend(
                backend,
                device=device,
                batch_size=batch_size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                chunk_length=chunk_length,
                window_chunks=window_chunks,
                sink_tokens=sink_tokens,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            peak = (
                "n/a"
                if result["peak_gpu_bytes"] is None
                else f"{result['peak_gpu_bytes'] / 2**20:.2f} MiB"
            )
            print(
                f"requested={result['requested']} used={result['used']} "
                f"batch={batch_size} chunk={result['chunk_length']} "
                f"window={result['window_chunks']} sinks={sink_tokens} "
                f"cached_tokens={result['cached_tokens']} kv_tokens={result['kv_tokens_per_query']} "
                f"median={result['median_latency_ms']:.3f} ms "
                f"throughput={result['tokens_per_second']:.1f} token/s "
                f"cache={result['cache_bytes'] / 1024:.1f} KiB peak_gpu={peak}"
            )
            if result["fallback"]:
                print(f"  fallback: {result['fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
