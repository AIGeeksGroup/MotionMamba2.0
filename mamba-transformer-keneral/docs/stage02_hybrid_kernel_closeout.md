# Stage 02 Closeout — Minimal + Existing Mamba Hybrid Kernel

Date: 2026-08-14.

Paper-positioning clarification: this generic gated-add Hybrid is a validated
integration/ablation scaffold, not the finalized MotionMamba 2.0
Contribution-1 mechanism. The future contribution targets token/state
interaction or delta modeling; its corresponding custom kernel is later
Contribution 2. Historical validation evidence below is unchanged.

Overall status: **PASS**.

Stage 2 adds a thin generic integration layer. It does not modify files under
`src/state_memory_operator/`, does not merge attention K/V with recurrent
state/EventMemory, and does not touch ARDY, DiT, or diffusion code.

## Final architecture

```text
                              current chunk
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
       closed MinimalAttentionKernel    existing Mamba3ChunkWrapper
       sink + recent KV + current KV       recurrent Mamba3State
                    |                             |
                    +--------------+--------------+
                                   |
               attention + sigmoid(gate_logit) * mamba
                                   |
                                 output
```

`mode="minimal"` is exactly the Stage-1 attention member and has no hybrid
fusion parameter or Mamba module. `mode="hybrid"` registers both independent
paths. The only Stage-1 code adjustment was device-name canonicalization:
an unindexed `torch.device("cuda")` cache now binds to the kernel parameter's
concrete `cuda:N`. Attention, sink, cache, read/commit, and reset semantics
are unchanged.

## Public interface

```python
config = SequenceKernelConfig(mode="minimal" | "hybrid", attention=attention_config)
kernel = SequenceKernel(config, mamba3=existing_mamba3)  # mamba3 only for hybrid
state = kernel.init_state(batch_size=B, device=device, dtype=dtype)

output, pending, aux = kernel.forward_chunk(
    x,
    state=state,
    do_read=True,
    commit=False,
    enable_attention_path=True,
    enable_mamba_path=None,
)
kernel.commit(state, pending, chunk_id=chunk_id)
kernel.reset_state(state, reset_mask=None)
```

The full hybrid fusion is explicit gated addition. Hybrid-only ablation calls
can select attention-only or Mamba-only. Minimal mode rejects Mamba objects.

## Memory semantics

- Attention cache: projected explicit K/V from the most recent `W` committed
  chunks; sinks remain static learned parameters.
- Mamba state: the existing opaque `Mamba3State` recurrent tuple.
- EventMemory: remains a separate optional mechanism owned by the existing
  `StateTransitionMemoryOperator`, not this minimal hybrid baseline.
- A read returns `PendingSequenceKernelState`; neither memory changes until
  explicit commit. A full or per-row reset resets both mechanisms separately.

## Validation evidence

### Existing Mamba before integration

Unchanged `tests/test_smoke.py`:

```text
local CPU-only: 4 passed, 5 skipped (pre-existing CUDA markers)
RTX 3090 CUDA:  9 passed in 493.50s
```

The CUDA run covered full/chunk consistency, async row reset, determinism,
forward/backward, EventMemory policies/compaction, and the complete existing
operator path. See `docs/existing_mamba_stage02_audit.md`.

### Generic interface and Stage-1 regression

```text
local selected suite:       38 passed, 7 skipped in 3.05s
remote Stage-1 + CPU suite: 33 passed, 1 skipped in 7.93s
remote final Stage-1/Stage-2 suite: 35 passed, 1 skipped in 42.12s
```

The skips are environment/marker skips: local CUDA Mamba/FA2 tests and remote
FA2 (FlashAttention is not installed there). Stage-1's already documented FA2
environment blocker remains truthful and is not reclassified as a pass.

### Real hybrid CUDA acceptance

On an NVIDIA RTX 3090 with the existing Mamba-3 source and an isolated
compatible Triton overlay:

```text
Stage-04 re-audit, tests/test_sequence_kernel_mamba.py: 1 passed in 35.22s
```

This single end-to-end test asserts finite hybrid forward/backward and input,
attention, Mamba, and fusion-gate gradients; exact output shape; attention KV
handoff; real Mamba state shape/carry; controlled attention-history and
recurrent-state effects; and per-row reset of both memories.

## Hard-gate acceptance matrix

```text
PASS  Stage-1 minimal semantics unchanged and Stage-1 regression passing
PASS  existing Mamba source/tests unchanged and passing on CUDA
PASS  minimal/hybrid interface exists
PASS  minimal imports and instantiates zero Mamba modules
PASS  minimal forward works
PASS  hybrid and supported Mamba-only forward work
PASS  hybrid CUDA forward/backward and gradients are finite
PASS  attention KV-cache handoff works in hybrid
PASS  real Mamba recurrent-state carry works in hybrid
PASS  full and per-row reset semantics work for both memories
PASS  controlled ablations prove attention history affects output
PASS  controlled ablations prove Mamba recurrent state affects output
PASS  attention KV, Mamba state, and EventMemory remain distinct
```

## Known limitations

1. Real Mamba-3 execution remains CUDA/Triton-only under the upstream kernel.
2. Hybrid padded recurrent stepping is not defined; Stage-1 attention masks
   remain available, but Mamba consumes the supplied chunk as a dense sequence.
3. Fusion is intentionally a simple learned scalar gated addition baseline.
4. The exact Stage-1 FA2 acceptance point later passed on A800; an expanded
   performance matrix remains optional and does not change Stage-2 semantics.

All Stage-2 acceptance criteria pass. The Stage-2 hard gate is closed, so
Stage 3 may now begin without changing these generic kernel semantics.
