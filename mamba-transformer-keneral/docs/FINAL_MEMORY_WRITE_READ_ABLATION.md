# Final persistent-memory write/read ablation

These results were re-evaluated in the final run with the frozen D2R-L checkpoint and exact 40-second ordered panel. Differences are intervention minus normal; positive values indicate degradation. Confidence intervals use 20,000 paired actor-cluster bootstrap resamples.

| Intervention | Root post-W5 AUC difference (95% CI) | Constraint post-W5 AUC difference (95% CI) |
|---|---:|---:|
| write_off | +120.55 [+53.39, +214.55] | +117.93 [+45.38, +219.90] |
| readout_off_after_W4 | +69.93 [+26.09, +124.84] | +69.68 [+22.45, +129.97] |
