"""Standalone minimal-kernel example with explicit read/commit lifecycle.

Run from the repository root:

    PYTHONPATH=src python examples/minimal_attention_kernel.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
# This script deliberately has the same basename as the package requested by
# the public example contract. Put `src/` ahead of the script directory so
# Python resolves the package rather than this file during the import below.
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))

from minimal_attention_kernel import (  # noqa: E402
    MinimalAttentionKernel,
    MinimalAttentionKernelConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone minimal attention kernel")
    parser.add_argument("--backend", choices=("sdpa", "flash2"), default="sdpa")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=6)
    parser.add_argument("--window-size-chunks", type=int, default=3)
    parser.add_argument("--sink-tokens", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunks < 1 or args.chunk_length < 1:
        raise ValueError("--chunks and --chunk-length must be positive")
    if args.window_size_chunks < 0 or args.sink_tokens < 0:
        raise ValueError("--window-size-chunks and --sink-tokens must be non-negative")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(0)
    config = MinimalAttentionKernelConfig(
        d_model=64,
        num_heads=4,
        window_size_chunks=args.window_size_chunks,
        use_attention_sinks=args.sink_tokens > 0,
        num_sink_tokens=args.sink_tokens,
        attention_backend=args.backend,
        allow_backend_fallback=True,
        causal_within_chunk=False,
    )
    kernel = MinimalAttentionKernel(config).to(device=device, dtype=dtype).eval()
    cache = kernel.init_cache(batch_size=2)

    for chunk_id in range(args.chunks):
        # This can represent any generic downstream sequence chunk. The first
        # call is read-only; committing is a separate caller decision.
        x = torch.randn(2, args.chunk_length, config.d_model, device=device, dtype=dtype)
        with torch.no_grad():
            output, pending, aux = kernel.forward_chunk(x, cache=cache, commit=False)
        assert cache.newest_chunk_id != chunk_id
        kernel.commit(cache, pending, chunk_id=chunk_id)
        print(
            f"chunk={chunk_id} output={tuple(output.shape)} "
            f"backend={aux['backend_used']} cached_chunks={cache.num_chunks} "
            f"history_tokens={cache.num_history_tokens} sink_tokens={cache.num_sink_tokens} "
            f"oldest={cache.oldest_chunk_id} newest={cache.newest_chunk_id}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
