# D3c Read-Path Audit

Date: 2026-09-06

## Current D3b representation mismatch

The D3b donor bank stored `SequenceKernelState.persistent_memory` after an
accepted commit. Its tensor shape was `[batch, K=2, d_model]`, usually in the
layer parameter dtype, serialized as CPU `float32`. It was keyed by
`(trajectory_id, completed_windows, block, layer)`.

The normal noisy read path is:

```text
state.persistent_memory  [B,K,D], durable committed slots
        |
        | PersistentContentMemory.read(slots, enabled=True)
        | optional LayerNorm when normalize_slots=True
        | returned tensor is cast to x.dtype
        v
memory_content / read_ready  [B,K,D], same device and dtype as mixer input
        |
        | MinimalAttentionKernel.forward_chunk(extra_content=memory_content)
        | current layer k_proj/v_proj are applied here
        v
memory_key, memory_value     [B,H,K,d_head]
        |
        | concatenated with sink/history/current K,V
        | run_attention(...)
        v
A_base = attention without extra_content
A_mem  = attention with memory K/V
        |
        | attention_difference = A_mem - A_base
        | output = A_base + tanh(read_gate) * difference
        v
attention_output -> state-aware/native gated fusion with Mamba output
        -> DiT residual layer output
```

At the `SequenceKernel` level, `read_ready` has shape `[B,K,D]`, the same
device/dtype as `x` after the explicit `.to(x.dtype)` conversion, and is not
detached in the normal forward path. The durable slots remain the source of
the write proposal. The read path is run during noisy diffusion forwards with
`commit=False`; accepted history prime/commit uses a separate clean forward
and must not use an external donor view.

## Phase and alignment audit

- `completed_windows=0` is the initial accepted-history state.
- After the fourth generated commit, `completed_windows=4`; the next noisy
  generation window is the first post-W4 read. This is an epoch label, not a
  claim that a commit itself is read by the same forward.
- Root and body each own independent states and layers. External lookup must
  use the same `(block, layer)` and recipient batch row.
- The ARDY CFG wrapper internally expands guidance rows, but the sequence
  cache and persistent memory are ensured at the user batch size before the
  denoiser call. External memory is therefore resolved at user-row batch size;
  it must never be applied to CFG-expanded rows directly.
- Attention W4 KV, Mamba state, state summary and state delta are separate
  fields and must remain recipient-owned.
- Normal content read and external read share the downstream K/V projection,
  attention, read-scale, fusion and DiT residual code. Only the read-ready
  tensor source may differ.

## D3b hook location and risk

D3b injected `content_memory_read_view_runtime` immediately before
`SequenceKernel.forward_chunk`. At that point it replaced `read_slots`, but
the hook accepted serialized durable slots rather than an explicitly captured
read-ready tensor. The clean accepted commit was later protected by clearing
the view, but this does not prove that the read representation, epoch, dtype,
or batch-row identity matched normal inference. D3c therefore records both
durable slots and exact read-ready tensors at the point immediately before
Attention consumes them.

## Required D3c contract

The new external level is `read_ready`. It is opt-in, read-only, noisy-read
only, and validated for shape, dtype, device, stride and contiguous layout.
Accepted commits automatically clear the external view. Projected memory K/V
are debug outputs only; causal replay continues to compute them with the
recipient layer's own `k_proj/v_proj`.
