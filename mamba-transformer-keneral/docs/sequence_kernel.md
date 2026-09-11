# Generic Sequence Kernel v1

`sequence_kernel` exposes two deliberately distinct construction modes:

```text
minimal = closed Stage-1 attention implementation
hybrid  = minimal attention + existing Mamba-3 + explicit gated addition
```

Paper positioning: `minimal` is the standard reusable baseline backend. The
current `hybrid` is an integration/ablation scaffold proving that independent
attention K/V and Mamba recurrent state can coexist. Its scalar gated addition
is not the finalized MotionMamba 2.0 Contribution-1 architecture. See
`docs/paper_architecture_scope.md`.

## Construction

```python
from sequence_kernel import SequenceKernel, SequenceKernelConfig

minimal = SequenceKernel(
    SequenceKernelConfig(mode="minimal", attention=attention_config)
)

hybrid = SequenceKernel(
    SequenceKernelConfig(
        mode="hybrid",
        attention=attention_config,
        fusion="gated_add",
        fusion_gate_logit_init=0.0,
    ),
    mamba3=existing_mamba3,
)
```

Minimal construction performs no Mamba import. Hybrid construction lazily
uses the existing `Mamba3ChunkWrapper`; it does not duplicate or redesign the
Mamba kernel.

## State and chunk lifecycle

```python
state = kernel.init_state(batch_size=B, device=x.device, dtype=x.dtype)

output, pending, aux = kernel.forward_chunk(
    x,
    state=state,
    do_read=True,
    commit=False,
)

# Accept one denoising/proposal result as durable sequence memory.
kernel.commit(state, pending, chunk_id=chunk_id)

# Episode reset: all rows, or selected batch rows.
kernel.reset_state(state)
kernel.reset_state(state, reset_mask)
```

Repeated `commit=False` calls do not change either committed memory. This is
the interface AR diffusion can later use for several noisy denoising
evaluations followed by one clean accepted commit.

`SequenceKernelState` intentionally has two fields:

- `attention_cache`: explicit recent token-level K/V chunks.
- `mamba_state`: compressed recurrent Mamba state, hybrid only.

They are never stored inside one another. A commit accepts both candidates at
the same boundary; reset applies the appropriate mechanism to each state.

## Current hybrid scaffold and ablations

When both paths are enabled:

```text
output = attention_output + sigmoid(fusion_gate_logit) * mamba_output
```

The gate is a learned scalar initialized to a logit of zero (weight 0.5 by
default). `enable_attention_path` and `enable_mamba_path` support controlled
tests and research ablations. A Mamba-only call is supported in hybrid mode;
it is not a third construction mode.

Future Contribution 1 is expected to model the interaction or delta between
token-level attention representation and state-level Mamba representation. No
final operator is specified here. Future Contribution 2 is the corresponding
integrated/custom kernel after that operator is finalized; it is not this
generic gated-add implementation and is not part of Minimal.

## Current constraints

- The Mamba path uses the existing SISO Mamba-3 CUDA/Triton requirements.
- Nontrivial per-row `current_mask` is an attention-path feature; padded
  recurrent stepping is not yet defined and should not be treated as skipped
  Mamba time. Use unpadded hybrid chunks or lengths grouped across the batch.
- Attention caches retain the Stage-1 projection-version rule and must be
  fully reset after an optimizer update.
- EventMemory remains part of `StateTransitionMemoryOperator`; it is not
  silently inserted into this baseline hybrid interface.
