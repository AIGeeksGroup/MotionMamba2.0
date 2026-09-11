# Stage 2A — Existing Mamba Audit

Status: **PASS**

This audit was completed before adding the generic hybrid integration layer.
No files under `src/state_memory_operator/` or the existing smoke test were
modified.

## Existing implementation

- `Mamba3ChunkWrapper.forward_chunk(x, state, reset_mask=None)` calls the
  existing Mamba-3 SISO combined Triton kernel and returns `(output,
  next_state)`.
- `Mamba3State` is deliberately opaque outside `detach()`, `clone()`, and
  `apply_reset_mask()`:
  - `angle_dt_state`: `[B, nheads, num_rope_angles]`, fp32
  - `ssm_state`: `[B, nheads, headdim, d_state]`, fp32
  - `k_state`: `[B, 1, nheads, d_state]`, compute dtype
  - `v_state`: `[B, nheads, headdim]`, compute dtype
- The existing package also exposes `StateProjector`, State-Q/K addressing,
  content-V payload encoding, `EventMemory`, and
  `StateTransitionMemoryOperator`.

## Unchanged baseline results

- Local CPU-only environment: `4 passed, 5 skipped` in `tests/test_smoke.py`;
  all skips are the pre-existing CUDA marker.
- Remote NVIDIA RTX 3090, CUDA execution: `9 passed in 493.50s` in the same
  unchanged `tests/test_smoke.py`.
- The remote run used copied source and a temporary isolated compatible
  Triton overlay. It did not install packages or modify the remote Python
  environment.

The CUDA result covers full-vs-chunk consistency, per-row state reset,
determinism, forward/backward, EventMemory FIFO/participative behavior and
compaction, and the full state-transition operator with memory.
