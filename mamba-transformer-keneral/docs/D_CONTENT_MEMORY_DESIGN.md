# D Content Memory Design

The D-stage proposal is the minimal extension of the existing state-aware
Hybrid path:

    committed Mamba state S_t
      -> existing StateProjector -> s_t
      -> delta_t = s_t - s_(t-1)
      -> [s_t ; delta_t]
           -> existing state_to_address (address branch)
           -> content_decoder + gated write (content branch)
           -> K fixed persistent content slots
           -> future Attention K/V read

src/state_memory_operator/content_memory.py implements this design. The
content decoder does not treat delta_t as semantic content directly; it learns
a proposal from absolute state and transition jointly. The memory is in model
dimension so Attention can map each slot to one K/V token. The
SequenceKernelConfig.content_memory_enabled flag is explicit and defaults off.

All noisy forwards are read-only. forward_chunk() returns pending Mamba,
Attention, and content-memory state. commit() installs all pending pieces in
one accepted-window transaction. reset_state() clears content slots with the
same per-row reset mask as recurrent state and recent KV.

The intervention API is explicit on `SequenceKernel.forward_chunk()`:
`content_memory_delta` supplies a zeroed or row-permuted write delta,
`content_memory_write_enabled=False` is write-off, and
`content_memory_read_enabled=False` is readout-off. `content_memory_use_state_delta=False`
is the matched `D-no-delta` configuration. `permute_rows()` supports a
cross-trajectory content-slot permutation. These switches alter only the
content branch; the C address branch retains its configured state semantics.

An all-zero initial slot tensor is not supplied to Attention as zero K/V
tokens, so the opt-in mechanism is exactly baseline-equivalent before the first
nonzero write. This avoids an otherwise subtle change to attention softmax
normalization at initialization.
