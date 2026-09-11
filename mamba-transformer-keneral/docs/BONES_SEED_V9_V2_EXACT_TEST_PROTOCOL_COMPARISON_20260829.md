# BONES-SEED v9 vs v2 Exact Test-Protocol Comparison (2026-08-29)

## Protocol Verification

**PASS.** v9 uses the exact ordered v2 test sample panels, seed 20260824, diffusion/CFG settings, generation durations, window counts, control-goal counts, persistent-memory protocol, and GT-prefix priming contract.

v9 Hybrid retains its trained inference setting: Mamba readout decay starts after window 4 with half-life 4. This is a model parameter rather than a change to the test panel or control protocol.

There is no endpoint-v9 ARDY checkpoint. The ARDY row is the healthy v2 step900 checkpoint already evaluated on this exact protocol; it is held fixed as the common architecture baseline.
ARDY uses its native explicit-history contract, while Minimal/Hybrid use persistent cache; all cross-architecture sample, duration, window, goal, diffusion, and CFG fields are matched.

## Table 4: Offline Text and Joint-Position Control

| Version | Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |
|---|---|---:|---:|---:|---:|
| v2 | ARDY | 51.62 | 56.48 | 69.17 | 17.34 |
| v2 | Minimal | 54.16 | 63.27 | 77.58 | 19.70 |
| v2 | Hybrid | 40.40 | 43.27 | 50.53 | 17.79 |
| v9 | Minimal | 41.63 | 48.08 | 58.02 | 16.60 |
| v9 | Hybrid | 38.83 | 40.45 | 48.45 | 17.15 |

## Table 5: ARDY-Aligned 9-Second Control

| Protocol | Version | Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |
|---|---|---|---:|---:|---:|---:|
| in | v2 | ARDY | 60.60 | 57.66 | 79.91 | 37.87 |
| in | v2 | Minimal | 45.27 | 40.19 | 50.54 | 23.08 |
| in | v2 | Hybrid | 36.45 | 28.60 | 45.63 | 17.76 |
| in | v9 | Minimal | 36.73 | 28.87 | 42.90 | 19.25 |
| in | v9 | Hybrid | 34.58 | 25.96 | 43.86 | 16.21 |
| out | v2 | ARDY | 75.33 | 57.82 | 73.78 | 34.85 |
| out | v2 | Minimal | 67.48 | 51.01 | 64.72 | 24.09 |
| out | v2 | Hybrid | 51.03 | 30.91 | 48.31 | 18.45 |
| out | v9 | Minimal | 57.49 | 35.54 | 54.48 | 21.83 |
| out | v9 | Hybrid | 50.71 | 28.43 | 47.30 | 17.29 |

## Table 5 Long-Horizon Extension

| Duration | Protocol | Version | Method | Constraint (cm) | Root ADE (cm) | Post-W5 Root slope | Post-W5 constraint slope |
|---:|---|---|---|---:|---:|---:|---:|
| 20 s | in | v2 | ARDY | 295.12 | 276.77 | 26.761 | 23.286 |
| 20 s | in | v2 | Minimal | 193.92 | 194.07 | -3.390 | -0.463 |
| 20 s | in | v2 | Hybrid | 137.04 | 130.11 | -4.784 | -6.626 |
| 20 s | in | v9 | Minimal | 122.51 | 116.63 | -1.202 | -2.039 |
| 20 s | in | v9 | Hybrid | 141.71 | 131.66 | -0.631 | -5.942 |
| 20 s | out | v2 | ARDY | 640.07 | 427.76 | 38.672 | N/A |
| 20 s | out | v2 | Minimal | 573.38 | 364.12 | 28.279 | N/A |
| 20 s | out | v2 | Hybrid | 478.77 | 328.73 | 10.088 | N/A |
| 20 s | out | v9 | Minimal | 345.91 | 220.69 | 12.328 | N/A |
| 20 s | out | v9 | Hybrid | 338.80 | 230.70 | 6.770 | N/A |
| 40 s | in | v2 | ARDY | 620.63 | 602.21 | 127.569 | 129.445 |
| 40 s | in | v2 | Minimal | 164.45 | 166.92 | 8.040 | 7.977 |
| 40 s | in | v2 | Hybrid | 137.47 | 134.89 | -1.238 | -1.311 |
| 40 s | in | v9 | Minimal | 138.04 | 139.43 | -0.856 | -1.253 |
| 40 s | in | v9 | Hybrid | 121.64 | 118.21 | -6.079 | -6.237 |
| 40 s | out | v2 | ARDY | 3220.55 | 1100.50 | 218.248 | N/A |
| 40 s | out | v2 | Minimal | 800.26 | 525.89 | 34.055 | N/A |
| 40 s | out | v2 | Hybrid | 756.11 | 532.39 | 31.463 | N/A |
| 40 s | out | v9 | Minimal | 543.69 | 383.86 | 13.997 | N/A |
| 40 s | out | v9 | Hybrid | 436.29 | 384.47 | 19.662 | N/A |

## Direct Hybrid Comparison

Negative deltas mean v9 improves over v2 on the identical test panel.

| Protocol | Constraint delta (cm) | Root ADE delta (cm) | Root-slope delta | Constraint-slope delta |
|---|---:|---:|---:|---:|
| table4 | -1.57 | -2.82 | N/A | N/A |
| 9s in horizon | -1.87 | -2.64 | N/A | N/A |
| 9s out of horizon | -0.31 | -2.48 | N/A | N/A |
| 20s in horizon | +4.67 | +1.55 | +4.153 | +0.684 |
| 20s out of horizon | -139.98 | -98.03 | -3.318 | N/A |
| 40s in horizon | -15.83 | -16.68 | -4.841 | -4.926 |
| 40s out of horizon | -319.82 | -147.92 | -11.801 | N/A |

## Result

v9 is better overall on the identical v2 test panels, but it is not a uniform win on every long-horizon metric.

- v9 Hybrid improves all reported Table 4 and 9-second constraint/Root metrics over v2 Hybrid and is also better than the fixed ARDY baseline.
- At 40 seconds in-horizon, v9 Hybrid improves constraint/Root ADE from 137.47/134.89 cm to 121.64/118.21 cm and improves both post-window-5 slopes from -1.238/-1.311 to -6.079/-6.237. It also remains clearly better than v9 Minimal on both means and slopes.
- At 20 seconds in-horizon, v9 Hybrid regresses by 4.67 cm constraint and 1.55 cm Root ADE relative to v2 Hybrid; its mean errors and Root slope are also worse than v9 Minimal, although its constraint slope is better.
- The 20/40-second out-of-horizon mean errors improve substantially over v2, but v9 Hybrid Root ADE is slightly worse than v9 Minimal at both durations, and its 40-second Root slope is worse.

Accordingly, v9 is the stronger 40-second and overall checkpoint on this exact panel, while v2 step900 remains the stronger 20-second in-horizon checkpoint. This test-panel result must remain separate from the actor-disjoint validation gate, where v9 still fails the strict all-slopes criterion.

## Artifacts

- v2 reference: `/data4/zts/result/bones_seed_generated_history_v2_earlystop900_eval_20260828`
- v9 evaluation: `/data4/zts/result/bones_seed_generated_history_endpoint_v9_v2_test_protocol_20260829`
- Fixed ARDY checkpoint: `/data4/zts/result/bones_seed_generated_history_v2_earlystop900_ft_20260828/original/checkpoint_step_000900.pt`
- Protocol assertions: `/data4/zts/result/bones_seed_generated_history_endpoint_v9_v2_test_protocol_20260829/protocol_verification.json`
