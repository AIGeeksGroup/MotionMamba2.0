# Final Table 5: BONES-SEED 9-second evaluation

A 24-frame GT prefix is followed by 201 generated frames. All four methods use the same ordered 64-sample panels and evaluation settings. Lower is better.

## In-horizon

| Model | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle (deg) | Skate (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY | 23.64 | 13.52 | 21.94 | 16.00 | 0.01975 | 18.05 | 37.56 | 0.19 |
| Minimal-W4 | 27.04 | 18.88 | 32.25 | 16.73 | 0.02614 | 16.38 | 46.60 | 1.24 |
| Native Hybrid | 25.34 | 16.53 | 30.78 | 15.50 | 0.02062 | 15.74 | 37.91 | 2.45 |
| D2R-L | 23.45 | 14.56 | 25.77 | 15.93 | 0.02107 | 15.26 | 33.57 | 4.14 |

## Out-of-horizon

| Model | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle (deg) | Skate (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY | 31.57 | 19.52 | 27.93 | 17.16 | 0.01985 | 18.35 | 34.74 | 0.19 |
| Minimal-W4 | 40.88 | 22.27 | 36.61 | 18.60 | 0.02462 | 16.03 | 45.99 | 1.07 |
| Native Hybrid | 24.92 | 13.67 | 22.51 | 13.87 | 0.01968 | 15.92 | 37.20 | 2.38 |
| D2R-L | 24.75 | 13.14 | 19.53 | 14.60 | 0.02008 | 15.08 | 32.92 | 4.38 |

These are BONES-SEED geometry/constraint proxy metrics, not official learned FID, R-Precision, FVMD, or TM-Sim scores.
