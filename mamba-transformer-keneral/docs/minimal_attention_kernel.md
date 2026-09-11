# Minimal Attention Kernel

Last updated: 2026-08-14.

## Positioning and dependencies

`MinimalAttentionKernel` is a compatibility code name for the baseline
attention backend. It is not a proposed MotionMamba kernel and carries no
custom-kernel novelty claim. The baseline deliberately uses standard attention,
chunk-level sliding-window K/V, and attention sinks.

The package is ARDY-independent and Mamba-independent. Importing or
instantiating it does not import `mamba_ssm`, `Mamba3`,
`StateTransitionMemoryOperator`, or `EventMemory`. A downstream model only
needs `[batch, chunk_tokens, model_dim]` tensors and an explicit cache object.

Local backend audit (2026-08-14):

- `/home/plote/miniconda3/envs/beingh/bin/python` has PyTorch
  `2.10.0+cu128` and FlashAttention `2.8.3`.
- FlashAttention exports `flash_attn_func`, whose separate Q/K/V interface
  supports `Q length != K/V length`.
- No FlashAttention 4 Python package/interface was found. FA4 is the intended
  future optimized native backend, but is not implemented or validated.
- CUDA was not visible in the local execution session, so CPU correctness is
  verified with PyTorch SDPA; FlashAttention tests skip explicitly unless a
  compatible CUDA device and dtype are available.

Changing FA2 to FA4 is intended to be a backend substitution with identical
cache/read/commit semantics, not an architecture change.

## Architecture

```text
Current Chunk
    |
    v
Attention
    ^
    |
Sliding Window KV
+
Attention Sinks
```

More precisely:

```text
                  Persistent
              Attention Sinks
                    |
                    v
Previous Chunks -> Sliding-Window KV
                    |
                    | cached K,V
                    v
Current Chunk ---> Attention ---> Output
       |                |
       +--- current K,V--+
                        |
                   explicit commit
                        |
                        v
                Sliding-Window KV
```

For a current chunk `X_t`, queries come only from the current chunk. Keys and
values are concatenated in this order:

```text
learned static sink K/V + recent committed chunk K/V + current chunk K/V
```

Current K/V always participate in the current self-attention computation.
`commit=False` only means that they are not persisted for later calls.

Within a chunk, attention is bidirectional by default. Optional causal mode
masks only future positions inside the current chunk; all sink/history entries
remain visible. Across chunks, history is one-way because only already
committed chunks can be read.

## Public API

```python
from minimal_attention_kernel import (
    MinimalAttentionKernel,
    MinimalAttentionKernelConfig,
)

config = MinimalAttentionKernelConfig(
    d_model=D,
    num_heads=H,
    window_size_chunks=4,
    use_attention_sinks=True,
    num_sink_tokens=4,
    attention_backend="flash2",
)
kernel = MinimalAttentionKernel(config)
cache = kernel.init_cache(batch_size=B, device=x.device, dtype=x.dtype)

out, pending, aux = kernel.forward_chunk(x, cache=cache, commit=False)
kernel.commit(cache, pending, chunk_id=0)
```

`forward_chunk(..., commit=True, chunk_id=...)` is a convenience form, but
the explicit two-call form is preferred when a caller must decide whether a
chunk is final/accepted. The cache is an explicit mutable object; it is never
hidden in module-global or module-instance state.

### Inputs and outputs

- Input `x`: `[B, L, D]`, where `L` is the current chunk length.
- Optional `current_mask`: valid current tokens per batch row.
- Output `out`: `[B, L, D]`; it does not include history or sink positions.
- `pending`: projected current K/V eligible for a later explicit commit.
- `aux`: requested/used backend and cache diagnostics.

To use the backend with another dataset or model, instantiate one backend and
one cache per attention layer, feed that layer's current chunk, and carry only
the cache across chunks. The backend does not assume motion data, text
conditioning, diffusion, root/body structure, or ARDY. Set
`window_size_chunks=0` and `use_attention_sinks=False` for a no-persistent-
history attention ablation. Simply construct `MinimalAttentionKernel`, or a
`SequenceKernel` whose config has `mode="minimal"`, to keep Mamba disabled; no
Mamba object or package is needed.

## Cache representation and sliding window

One `ChunkSlidingWindowKVCache` belongs to one attention layer/operator.
Downstream multi-layer models create one cache per layer. Every projected K/V
tensor has shape:

```text
[batch, heads, chunk_tokens, head_dim]
```

Normal chunks are stored as distinct `CachedChunk` entries so variable chunk
lengths and chunk-level FIFO eviction are preserved. With
`window_size_chunks=W`, only the latest `W` normal chunks remain. A zero
window is valid and stores no normal history.

Sinks are learned static tokens owned by the kernel. Their projected K/V are
computed on demand for the current weights and are not stored as episode data
inside the cache. They are therefore always present when enabled, never
evicted, and remain after full or per-row reset.

## Read versus commit

Read and commit are separate operations:

1. `forward_chunk` projects current Q/K/V, reads sinks and committed history,
   and returns output plus a `PendingChunkKV` handle.
2. The cache is unchanged when `commit=False`.
3. `kernel.commit(cache, pending, chunk_id)` validates ownership/shapes and
   appends one whole chunk, then evicts old normal chunks by FIFO.

This permits any caller to perform repeated read-only passes and commit only
an accepted representation. The kernel has no diffusion-specific knowledge.

For the ARDY Minimal baseline, every noisy denoising evaluation is read-only.
Only the accepted clean autoregressive chunk commits, exactly once. Root/body
commit is transactional: if either stage fails, both states roll back. ARDY's
separated CFG branches read the same persistent batch-B cache without turning
the 2B/3B guidance batch into persistent state. These are correctness
properties, not paper novelty claims.

## Reset semantics

`kernel.reset_cache(cache)` drops every normal chunk. A direct `cache.reset()`
is also valid when weights have not changed. `kernel.reset_cache(cache,
reset_mask)` clears
history only for selected batch rows. Per-row reset is represented by a
valid-row mask on each cached chunk; invalid rows have zeroed K/V and are
masked out during attention, so no previous-episode data can be read. Entries
that become invalid for every row are removed. Learned static sinks remain.

## Attention backends

- `sdpa`: PyTorch `scaled_dot_product_attention`, always the correctness
  reference.
- `flash2`: FlashAttention 2 `flash_attn_func` when installed and compatible.
  If the package/kernel is unavailable, the implementation falls back to SDPA
  when configured to allow fallback; diagnostics report the backend actually
  used. Explicit `allow_backend_fallback=False` raises instead.
- `flash4`: reserved configuration name which currently raises
  `NotImplementedError`. It is the intended final optimized native backend,
  but no FA4 implementation, validation, latency, memory, or quality result is
  claimed.

Both backends use the same Q/K/V concatenation and mask semantics. Arbitrary
per-row history masks (needed by asynchronous batch reset) use SDPA because
FlashAttention's dense API does not accept that mask.

## Scope

This package intentionally contains:

```text
Attention + chunk-level sliding-window KV cache + learned attention sinks
```

It contains no ARDY adapter, DiT, diffusion loop or timestep conditioning,
motion/root/body representation, Mamba import, hybrid fusion, Deep Forcing,
participative eviction, simulation, or robotics code.

These standard components form the reusable Minimal baseline. MotionMamba 2.0
begins only after Mamba-specific token/state modeling is added. A future
integrated/custom Attention–Mamba kernel belongs to that future mechanism, not
to this package.

## Known limitations

- This is one attention operator, not a Transformer/DiT stack. A downstream
  multi-layer model must instantiate one kernel and one cache per layer.
- The implementation concatenates retained chunks for each read. It uses
  FlashAttention for the attention computation when possible, but does not yet
  use a fused paged-KV-cache append kernel.
- Per-row asynchronous reset introduces an arbitrary validity mask and therefore
  falls back to SDPA for that read when the dense FlashAttention backend is
  requested.
- Commit detaches K/V by default, which is appropriate for inference and
  prevents accidental unbounded graphs. Recurrent training can request
  `detach=False`, accepting the associated graph-memory cost.
- Cached values are projected activations. They must be reset/recomputed after
  an optimizer step that changes projection weights; carrying them across
  parameter updates would make them stale. The cache records the K/V projection
  parameter mutation counters and rejects stale reads/commits until the caller
  performs a full `kernel.reset_cache(cache)`.
- FA2 parity, backward, latency, and peak allocation were subsequently
  validated on A800 and are recorded in the Stage-01 closeout. FA4 remains
  unimplemented and unvalidated.

## Local verification record

Executed on 2026-08-14 with
`/home/plote/miniconda3/envs/beingh/bin/python` (PyTorch `2.10.0+cu128`):

```bash
PYTHONPATH=src python -m pytest tests/test_minimal_attention_kernel.py -q
# 25 passed, 1 skipped

PYTHONPATH=src python examples/minimal_attention_kernel.py --backend sdpa
# 8 chunks completed; W=3 stayed bounded at 3 chunks / 18 history tokens.

PYTHONPATH=src python examples/minimal_attention_kernel.py --backend flash2 --chunks 2
# completed through the documented CPU -> SDPA fallback.
```

The local skip is retained because that workstation session has no usable
CUDA driver. It is an environment-specific skip, not the final project status:
the exact A800 FA2 acceptance later passed and is recorded in the Stage-01
closeout.

Small CPU smoke benchmark (`B=2`, `D=128`, `H=4`, chunk length 32, `W=4`,
four sinks, 20 measured iterations after five warmups):

```text
requested  used  median latency  throughput       cache K/V  peak GPU
SDPA       SDPA  0.296 ms        216,332 token/s  256 KiB    n/a
FA2        SDPA  0.293 ms        218,723 token/s  256 KiB    n/a
```

The second row is a fallback sanity check, not a FlashAttention performance
number. CPU timings are preliminary and machine/load dependent.
