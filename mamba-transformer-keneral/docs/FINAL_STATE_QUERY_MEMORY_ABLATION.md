# Final state_query / persistent-memory ablation

Lower is better. The 40-second differences use 20,000 exact paired actor-cluster bootstrap resamples; negative differences favor the candidate.

## Table 4 (52 frames)

| Model | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle (deg) | Skate (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native Hybrid | 27.06 | 20.91 | 21.31 | 14.50 | 0.02444 | 17.61 | 36.93 | 0.60 |
| Native Hybrid + state_query | 25.69 | 19.76 | 19.95 | 14.49 | 0.02464 | 17.62 | 36.24 | 0.70 |
| Native Hybrid + persistent memory | 26.86 | 20.90 | 21.25 | 14.48 | 0.02444 | 17.63 | 36.85 | 0.97 |
| D2R-L | 26.75 | 21.26 | 21.59 | 14.49 | 0.02452 | 17.66 | 36.68 | 1.03 |

## 40-second long-horizon means

| Model | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle (deg) | Skate (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native Hybrid | 41.94 | 36.82 | 56.14 | 20.32 | 0.01633 | 14.81 | 26.92 | 11.04 |
| Native Hybrid + state_query | 39.20 | 35.36 | 56.31 | 19.75 | 0.01867 | 15.08 | 34.52 | 13.76 |
| Native Hybrid + persistent memory | 32.10 | 27.35 | 39.91 | 16.51 | 0.01825 | 13.58 | 24.06 | 18.15 |
| D2R-L | 32.43 | 28.73 | 42.34 | 17.65 | 0.01960 | 14.63 | 30.26 | 20.49 |

## 40-second post-W5 AUC comparisons

| Comparison | Root AUC difference (95% CI) | Constraint AUC difference (95% CI) |
|---|---:|---:|
| Native Hybrid + state_query minus Native Hybrid | -23.22 [-50.36, +1.72] | -43.75 [-78.11, -11.23] |
| Native Hybrid + persistent memory minus Native Hybrid | -130.97 [-236.39, -67.01] | -134.39 [-248.51, -72.66] |
| D2R-L minus Native Hybrid | -122.54 [-209.11, -56.05] | -140.12 [-228.59, -80.07] |
| D2R-L minus Native Hybrid + state_query | -99.33 [-173.01, -41.48] | -96.37 [-177.43, -33.49] |
| D2R-L minus Native Hybrid + persistent memory | +8.43 [-45.85, +62.05] | -5.73 [-55.46, +48.69] |

Provenance limitation: the two new single-component variants were initialized from the exact Native Hybrid 50k checkpoint, but the historical D2R-L checkpoint was continued through the D2R step-250 lineage. Therefore the results quantify predictive differences, but do not constitute a perfectly parent-matched 2x2 causal decomposition of state_query and persistent memory.
