# Minimal Attention Kernel v1 — Implementation Validation

Audit date: 2026-08-14. Scope: actual code under
`src/minimal_attention_kernel/`; no ARDY, DiT, diffusion, or Mamba work.

## Audit method

The four implementation files and their tests were read directly. This record
does not infer behavior from the earlier design document.

## Verified implementation semantics

### Projections and tensor layout

`MinimalAttentionKernel` owns independent `q_proj`, `k_proj`, `v_proj`, and
`out_proj` linear layers, all mapping `d_model -> d_model`. Input has shape
`[B, L, d_model]`. `_split_heads` reshapes and transposes projections to
`[B, H, L, head_dim]`, where `head_dim = d_model / H`. `_merge_heads` reverses
that operation before `out_proj`.

For every call:

```text
Q_current = q_proj(current)                   [B,H,L,Dh]
K_current = k_proj(current)                   [B,H,L,Dh]
V_current = v_proj(current)                   [B,H,L,Dh]

K_all = cat(K_sink, K_history, K_current)     token axis
V_all = cat(V_sink, V_history, V_current)     token axis
```

Queries come only from the current chunk. Current K/V are always included,
including when `commit=False` and `do_read=False`, so this remains
self-attention rather than pure cross-attention.

### Dedicated Learnable Attention Sink Tokens

When enabled, `sink_tokens` is an `nn.Parameter` with shape
`[num_sink_tokens, d_model]`, initialized from a scaled normal distribution.
Each forward expands it across the batch and projects it with the same K/V
projections used for normal tokens. These are **Dedicated Learnable Attention
Sink Tokens**:

- not copied from the first chunk;
- not episode-derived;
- not stored in `ChunkSlidingWindowKVCache`;
- not counted toward `window_size_chunks`;
- recomputed from current parameters on each forward;
- unaffected by episode cache reset or FIFO eviction.

### History, chunk boundaries, and FIFO

Every accepted chunk is a distinct `CachedChunk` containing `chunk_id`, K, V,
and `[B,L]` token-validity mask. Variable lengths are preserved. `get()`
concatenates retained chunks only for the attention read. After append, the
list is sliced to the latest `window_size_chunks`, so eviction is by whole
chunk and `W=0` retains no episode history.

### Read and commit

`forward_chunk(..., commit=False)` does not call cache append. It returns a
`PendingChunkKV` containing this call's current projected K/V. Those K/V were
already included in the current attention calculation. `do_read=False` omits
committed history from K/V construction but still includes sinks and current
K/V. `commit()` is a separate mutation that validates producer ownership and
appends the pending chunk.

Default commit detaches K/V, preventing accidental unbounded cross-chunk
graphs. `detach=False` explicitly retains a recurrent training graph.

### Reset

`reset()` removes all `CachedChunk` objects. `reset_mask: BoolTensor[B]`
invalidates and zeros selected rows in every chunk, preserves unreset rows,
and removes chunks that become invalid for all rows. The concatenated
validity mask prevents reset history from participating in attention. Since
sinks are model parameters rather than cache entries, reset never deletes
them.

### Masking and within-chunk direction

Default current attention is bidirectional. Optional causal mode allows the
entire sink/history prefix and only current positions up through the query's
position. Token validity is combined into an explicit boolean SDPA mask.
Arbitrary per-row masks require SDPA fallback because the dense FA2 API does
not accept them.

### Backend dispatch

The audited implementation had a correct lazy FlashAttention-2 call through
`flash_attn.flash_attn_func`, with Q/K/V transposed to FA2's `[B,L,H,Dh]`
layout and support for unequal Q/KV lengths. It fell back to SDPA for missing
FA2, CPU/fp32, or arbitrary masks. One terminology defect was found: the
public backend string was named `"flash"`, not the required explicit
`"flash2"`; closeout work changes that name and reserves `"flash4"` as an
unimplemented future backend.

### Mamba isolation

The package `__init__.py` imports only its own config/cache/attention/operator
modules. FlashAttention is imported lazily inside backend probes/execution.
There is no import of `state_memory_operator`, `ardy_adapter`, or `mamba_ssm`.
A clean subprocess test verifies `mamba_ssm` remains absent after import,
forward, backward, cache creation, and commit.

## Gaps found by the v1 audit

The previous 19-pass suite established the main behavior, but the new hard
gate correctly requires stronger evidence. These gaps must be closed before
the Stage-1 closeout:

1. Rename public `"flash"` backend to `"flash2"` and reserve `"flash4"`
   without claiming implementation.
2. Add the specified deterministic reference configuration (`B=1`, `H=2`,
   `Dh=16`, `L=8`, `W=2`, two sinks).
3. Verify FIFO/variable-length retention against the actual concatenated K/V
   tensors, not metadata alone.
4. Add repeated-read/single-commit, current self-attention, and optimizer-step
   tests.
5. Add an executable stale-projected-K/V guard after optimizer updates, rather
   than relying only on documentation.
6. Expand the real CUDA FA2 test to backward and logged error metrics. CUDA
   execution remains subject to an accessible working NVIDIA driver.

## Audit-gap resolution

All functional gaps above were closed in code and CPU/reference tests:

- public backends are now `sdpa`, `flash2`, and reserved-unimplemented
  `flash4`;
- the required SDPA oracle configuration passes with
  `max_abs_error=2.384e-7`, `mean_abs_error=4.93e-8`;
- FIFO and variable-length tests compare retrieved K/V tensor slices against
  the exact committed tensors;
- seven read-only passes leave cache tensors unchanged, followed by one
  explicit commit;
- a real optimizer step changes weights, makes the old cache fail loudly as
  stale, and a full `kernel.reset_cache(cache)` safely rebinds it;
- the CUDA FA2 test now covers unequal Q/KV lengths, history, sinks, sliding
  eviction, `commit=False`, forward parity/error logging, backward, and final
  explicit commit.

The last item later executed successfully on an A800 with PyTorch 2.4.0,
CUDA 12.1, and existing FlashAttention 2.7.3. Against forced SDPA math it
reported max/mean/relative error `6.7139e-4 / 1.0192e-4 / 5.8379e-4`, finite
backward, 0.9815 ms median FA2 latency, and 17.8228 MiB peak allocation for the
exact requested point. No dependency installation was required.
