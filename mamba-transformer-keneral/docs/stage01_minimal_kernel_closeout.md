# Stage 01 Closeout — Minimal Attention Kernel v1

Date: 2026-08-14.

Paper-positioning clarification: “kernel” is the historical implementation
module name. Stage 1 is a standard reference streaming-attention backend and
is not the MotionMamba custom-kernel contribution. Numerical and engineering
evidence in this historical closeout is unchanged; see
`docs/paper_architecture_scope.md`.

Overall status: **PASS, INCLUDING EXACT CUDA FA2 ACCEPTANCE POINT.**

Stage 1 contains no ARDY, DiT, diffusion, or Mamba integration. All
backend-agnostic, SDPA-reference, cache, training, reset, isolation, and exact
requested FA2 gates pass, so the generic minimal architecture is frozen.

## 1. Exact final architecture

```text
Dedicated Learnable Attention Sink Tokens --project K/V--+
                                                        |
Retained committed chunks -------------------- cached K/V|
                                                        v
Current chunk --Q,K,V--> [sink | history | current] attention --> output
       |                                                        
       +---------------- PendingChunkKV ------------------------+
                                                                |
                                                       explicit commit
                                                                |
                                                                v
                                               chunk FIFO cache (latest W)
```

This is one reusable attention operator/layer. It intentionally does not own
Transformer residuals, normalization, FFN, DiT, or a model stack.

## 2. Exact public API

```python
from minimal_attention_kernel import (
    MinimalAttentionKernel,
    MinimalAttentionKernelConfig,
)

config = MinimalAttentionKernelConfig(
    d_model=128,
    num_heads=4,
    window_size_chunks=2,
    use_attention_sinks=True,
    num_sink_tokens=2,
    attention_backend="flash2",  # or "sdpa"; "flash4" is reserved
)
kernel = MinimalAttentionKernel(config).to(device=device, dtype=dtype)
cache = kernel.init_cache(batch_size=B)

output, pending, aux = kernel.forward_chunk(
    current_chunk,
    cache=cache,
    do_read=True,
    commit=False,
)
kernel.commit(cache, pending, chunk_id=chunk_id)

# Episode/trajectory boundary, and mandatory after optimizer.step():
kernel.reset_cache(cache)

# Asynchronous episode reset (only while projection version is unchanged):
kernel.reset_cache(cache, reset_mask=torch.tensor([True, False], device=device))
```

`forward_chunk(..., commit=True, chunk_id=...)` is a convenience form. The
two-call form is the canonical read-many/commit-once interface.

## 3. Exact tensor shapes

For input `x: [B,L,d_model]`, `H=num_heads`, `Dh=d_model/H`:

```text
Q_current, K_current, V_current: [B,H,L,Dh]
K_sink, V_sink:                 [B,H,S,Dh]
K_history, V_history:           [B,H,sum(L_i),Dh]
K_all, V_all:                   [B,H,S+sum(L_i)+L,Dh]
attention output before merge:  [B,H,L,Dh]
final output:                   [B,L,d_model]
token-valid masks:              [B,tokens]
```

Each `CachedChunk` retains its own `[B,H,L_i,Dh]` tensors and `[B,L_i]`
validity mask. Concatenation occurs only on read. Unequal Q/KV lengths are an
intended invariant, not a special case.

## 4. Sink semantics

Status: **PASS**.

The sinks are **Dedicated Learnable Attention Sink Tokens**:

- `nn.Parameter [S,d_model]`, projected by the current layer's K/V projections;
- recomputed on every forward;
- persistent across arbitrarily many normal-chunk FIFO evictions;
- not episode-derived and not copied from the first historical tokens;
- not stored in the cache and not counted toward `W`;
- unchanged by full/per-row episode reset.

The sink persistence test commits 20 chunks into `W=2`, then resets and
compares sink parameters plus projected K/V exactly before versus after.

## 5. KV sliding-window semantics

Status: **PASS**.

The unit of storage and eviction is `CachedChunk`, never an arbitrary token.
For `W=3` after commits `0..4`, retained IDs and exact concatenated K/V are
verified as chunks `(2,3,4)`. Variable lengths `(8,13,6)` are supported; with
`W=2`, the retrieved tensor is verified slice-by-slice as length-13 chunk 1
followed by length-6 chunk 2. `W=0` stores no normal history.

## 6. Read/commit semantics

Status: **PASS**.

Current K/V always participate in current self-attention. `commit=False`
means only "do not persist." Seven consecutive read-only calls verify that
chunk IDs and cached tensors remain unchanged; one subsequent explicit commit
adds exactly one chunk. `do_read=False` ignores committed history while still
using current self-attention and configured sinks.

## 7. Reset semantics

Status: **PASS**.

- Full reset removes all episode-derived K/V.
- `[B]` reset masks invalidate and zero only selected rows.
- Invalid historical keys are masked from attention.
- Rows not reset retain identical outputs for an identical current chunk.
- A reset row matches a fresh cache and differs from its pre-reset output.
- Dedicated learned sinks remain because they are model parameters.

## 8. Training/cache lifetime semantics

Status: **PASS**.

Safe lifecycle:

```text
reset cache -> process trajectory/chunks -> loss -> backward
-> optimizer.step() -> mandatory full kernel.reset_cache(cache)
```

Commit detaches by default. `detach=False` retains the cross-chunk graph when
explicitly requested. The cache records PyTorch mutation counters for K/V
projection parameters. After a real optimizer update, stale reads and commits
raise until `kernel.reset_cache(cache)` clears history and rebinds the cache.
Partial reset is rejected after a weight change because unreset rows would
still contain stale projected K/V.

## 9. SDPA reference parity

Status: **PASS**.

Deterministic oracle configuration:

```text
B=1, H=2, Dh=16, d_model=32, current L=8, W=2, sinks=2
history chunks: two chunks of length 8
```

Reference explicitly computes `softmax(QK^T/sqrt(Dh))V` over
`[K_sink,K_history,K_current]` and the matching V concatenation, followed by
the same output projection.

```text
max_abs_error  = 0.0000002384
mean_abs_error = 0.0000000493
shapes/values  = finite and exact within atol=1e-6, rtol=1e-5
```

## 10. FA2 CUDA parity

Status: **PASS on NVIDIA A800-SXM4-80GB**.

The implemented CUDA test uses `B=1, d_model=128, H=4, L=16, W=2, sinks=2,
fp16`, fills three history chunks to exercise eviction, verifies
`Q_len=16 != KV_len=50`, checks `commit=False`, logs max/mean/relative error
against identical-weight SDPA, performs FA2 backward, and explicitly commits
the accepted pending K/V.

Stage-04 environment and result:

```text
torch 2.4.0+cu121, flash_attn 2.7.3, fp16
Q=16, KV=50, retained history chunk IDs=(1,2), sinks=2
max absolute error  0.0006713867
mean absolute error 0.0001019189
relative error      0.0005837861
SDPA math median    0.989734 ms
FA2 median          0.981524 ms
SDPA peak allocated 17.826660 MiB
FA2 peak allocated  17.822754 MiB
FA2 backward        finite
```

The SDPA side was forced to the CUDA math backend so parity did not compare
FA2 against PyTorch SDPA silently dispatching to another flash kernel. Exact
reproduction command:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/verify_flash2_stage04.py \
  --iterations 20 --dtype fp16
```

## 11. Forward/backward/optimizer result

Status: **PASS (SDPA and FA2)**.

Forward produces finite `[B,L,d_model]` output. Backward produces finite input
and parameter gradients. A real SGD step has finite loss/gradients and changes
`k_proj.weight`; the old cache is then correctly rejected as stale.

## 12. No-Mamba isolation

Status: **PASS**.

A clean subprocess imports only `minimal_attention_kernel`, instantiates,
forwards, backwards, creates a cache, and commits. It verifies
`"mamba_ssm" not in sys.modules` both before and after execution.

## 13. Small latency/memory benchmark

CPU fallback sanity result (`B=2,D=128,H=4,L=32,W=4,S=4`, five warmups,
20 measured calls):

```text
requested  used  median   throughput       persistent cache K/V  peak GPU
SDPA       SDPA  0.296ms  216,332 token/s  256 KiB               n/a
FA2        SDPA  0.293ms  218,723 token/s  256 KiB               n/a
```

The FA2 row is explicitly a CPU fallback, not an FA2 performance claim.

The full CUDA matrix remains optional benchmark work. The exact Stage-04
acceptance point passed; the matrix script supports (`B=1/4`, `L=16/32/64`,
`W=1/2/4`, `S=2/4`) and reports requested/actual backend, latency, tokens/s,
peak GPU allocation, and cache bytes:

```bash
PYTHONPATH=src python benchmarks/benchmark_minimal_attention_kernel.py \
  --device cuda --matrix --warmup 10 --iterations 50
```

## 14. Known limitations

1. Single attention operator, not a Transformer/DiT stack.
2. Retained chunk tensors are gathered/concatenated per call; no fused paged
   inference-only K/V append path yet.
3. Arbitrary per-row validity masks use SDPA fallback with the dense FA2 API.
4. Detached commit is the default; recurrent BPTT must opt into graph cost.
5. Cache cannot survive projection-weight updates; this is now enforced.
6. Dense FA2 falls back to SDPA for arbitrary per-row masks by design.
7. `flash4` is a reserved name that raises `NotImplementedError`; FA4 is not
   implemented or claimed.

## 15. Acceptance matrix

```text
PASS     shape + finite
PASS     backward (SDPA reference path)
PASS     optimizer step / weights change / stale-cache rejection
PASS     no-history attention
PASS     current bidirectional self-attention
PASS     previous history changes output
PASS     do_read=False
PASS     commit=False leaves cache unchanged
PASS     explicit commit
PASS     read many times / commit once
PASS     FIFO whole-chunk eviction, including exact K/V contents
PASS     variable chunk lengths and exact offsets
PASS     dedicated sink persistence
PASS     sinks excluded from W
PASS     window_size_chunks=0
PASS     full reset
PASS     per-row reset and output-dependence/no-leakage checks
PASS     cache clone
PASS     cache detach
PASS     Q_len != KV_len (SDPA)
PASS     SDPA explicit-reference parity
PASS     SDPA-math vs FA2 CUDA parity/error metrics
PASS     FA2 backward
PASS     no Mamba import
PASS     exact requested CUDA SDPA/FA2 latency/peak-memory point
TODO     optional expanded CUDA performance matrix
```

Executed suite:

```text
25 passed, 1 skipped
```

The single skip is the combined real-CUDA FA2 parity/backward test. No FAIL
items remain in the architecture-independent Stage-1 implementation.
