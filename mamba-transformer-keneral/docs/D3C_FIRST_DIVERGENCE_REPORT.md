# D3c First-Divergence Report

`FIRST_DIVERGENCE_STAGE = NONE`

| Stage | Normal checksum | Replay checksum | Max abs diff | First mismatch |
|---|---|---|---:|---|
| durable_slots | `892207e2a9ec` | `892207e2a9ec` | 0.0 | None |
| read_ready_tensor | `892207e2a9ec` | `892207e2a9ec` | 0.0 | None |
| memory_K | `0960ac1703d7` | `0960ac1703d7` | 0.0 | None |
| memory_V | `347bae7dd237` | `347bae7dd237` | 0.0 | None |
| A_base | `307402de0359` | `307402de0359` | 0.0 | None |
| A_mem | `502f33ec5fde` | `502f33ec5fde` | 0.0 | None |
| read_scaled_A | `e3ce74b7ec02` | `e3ce74b7ec02` | 0.0 | None |
| fused_output | `e3ce74b7ec02` | `e3ce74b7ec02` | 0.0 | None |
| DiT_output | `4f6d0822bc87` | `4f6d0822bc87` | 0.0 | None |

P0 compares the exact read-ready tensor and all downstream FP32 SDPA stages with identical hidden input, cache, projections, scale and seed.

The fixture is intentionally minimal and deterministic; it is an instrument calibration, not a model-quality or donor-causality result.
