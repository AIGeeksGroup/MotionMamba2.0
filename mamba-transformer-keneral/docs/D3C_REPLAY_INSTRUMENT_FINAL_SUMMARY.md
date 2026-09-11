# D3c Exact Read-State Capture / Replay Parity: Final Summary

Date: 2026-09-06
Decision: **STOP before donor experiments**
`D3C_REPLAY_INSTRUMENT_VALID = false`
`D3C_READY_FOR_D3B_RERUN = false`

## 1. Exact read path

Normal Attention consumes `memory_content = PersistentContentMemory.read(persistent_memory).to(x.dtype)`, with optional slot LayerNorm, on the layer's mixer device and dtype. The current layer's `k_proj/v_proj` then compute memory K/V, which are concatenated with sink/history/current K/V. `A_mem - A_base` is multiplied by `tanh(read_gate)` and fused with the Mamba path before the DiT residual. Durable slots and read-ready content are therefore distinct representation levels even when `normalize_slots=False`.

## 2. D3b mismatch and P0 result

D3b captured serialized durable slots keyed by completed commit index and injected them at the pre-read view hook. It did not record read-ready dtype/device/stride, explicit read epoch, or phase. The exact ARDY own replay consequently failed, while a repeated normal run matched exactly.

D3c's deterministic FP32/SDPA P0 fixture compares durable slots, read-ready tensor, memory K/V, A_base, A_mem, read-scaled attention, fused output, and DiT output. Every stage has zero max absolute difference and matching SHA-256. `FIRST_DIVERGENCE_STAGE = NONE` in the controlled shadow path.

## 3. P1/P2 gate

P1 full ARDY recorded same-run replay and P2 cross-run exact-read own replay were not run because the current capture path still lacks the exact pre-Attention read-ready bank and read-epoch/phase metadata. The previous D3b own replay remains a plumbing-failure artifact, not a P2 result.

## 4. Required next repair

Instrument `SequenceKernel.forward_chunk` at the `memory_content` assignment, saving both durable slots and read-ready tensors with `(trajectory_id, read_epoch, phase, stream, layer, dtype, device, stride, CFG/user-row mapping)`. Ensure external level `read_ready` bypasses `PersistentContentMemory.read` but shares the same K/V projection path, and disable it during accepted clean commits. Then run P1 before any cross-run or donor condition.

No donor, common/specific, layerwise, no-delta, training, Stage E, or locked-final operation was performed in D3c.
