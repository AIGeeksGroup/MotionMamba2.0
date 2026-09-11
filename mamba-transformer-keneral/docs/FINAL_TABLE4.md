# Final Table 4: BONES-SEED short-horizon evaluation

All methods use the same ordered 64-sample, 52-frame panel, prompts, constraints, diffusion settings, and noise seeds. Lower is better.

| Model | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle (deg) | Skate (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY | 31.65 | 21.34 | 17.19 | 18.58 | 0.02183 | 21.67 | 36.62 | 0.05 |
| Minimal-W4 | 37.58 | 31.07 | 32.61 | 15.58 | 0.02495 | 18.06 | 44.45 | 0.29 |
| Native Hybrid | 27.06 | 20.91 | 21.31 | 14.50 | 0.02444 | 17.61 | 36.93 | 0.60 |
| D2R-L | 26.75 | 21.26 | 21.59 | 14.49 | 0.02452 | 17.66 | 36.68 | 1.03 |

These are BONES-SEED geometry/constraint proxy metrics. The unavailable official learned FID, R-Precision, FVMD, and TM-Sim metrics are not claimed.
