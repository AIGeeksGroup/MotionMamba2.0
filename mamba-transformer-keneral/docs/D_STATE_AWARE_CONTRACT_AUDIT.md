# Stage D State-Aware Contract Audit

Date: 2026-09-02
Status: IMPLEMENTED / GATE NOT RUN

## Contract reuse

The D module reuses the existing StateProjector and StateTransition semantics.
The state summary is s_t = P(S_t) and the transition is exactly
delta_t = P(S_t) - P(S_(t-1)); no parallel projector path was added. The
existing [s, delta] -> state_to_address branch remains owned by
TokenStateInteraction for state-query or state-KV modulation.

## Address and content paths

- Address: existing state_query/state_kv_modulation consumes committed [s, delta]
  and changes query bias or recent-KV key scales only.
- Content: PersistentContentMemory decodes the same [s, delta] into fixed slots.
  It is an opt-in SequenceKernelConfig path and is disabled by default.
- Read: prior committed slots are projected to Attention K/V as extra content
  tokens. Readout can be disabled for causal intervention.
- Write: a pending slot proposal is formed after the next state summary and is
  installed only by the same SequenceKernel.commit() that accepts the clean
  Attention/Mamba window.

## Initialization and capacity

The decoder and write gate are zero/closed initialized; read gate starts near
zero (sigmoid(-12)). Slot count K is fixed by config and does not grow with
horizon.

## Tests

`tests/test_content_memory.py` covers fixed shape, finite neutral proposal,
reset, carry, row permutation, exact readout-off, noisy-forward immutability,
single accepted-window commit, write-off, zero-delta, `D-no-delta`, and exact
neutral-init parity. `SequenceKernel` attaches a monotonic commit version to
each pending transaction, so a stale or already accepted pending state is
rejected. CUDA/Mamba integration tests must pass before any training gate is
authorized.

## Gate boundary

This is an implementation and contract result only. No D-full, D-no-delta,
zero-delta, permute-delta, write-off, or readout-off real-motion result has been
claimed. Causal training/evaluation remains blocked until the registered C
100/250-step gate is executed and documented.
