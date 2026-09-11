# Final D2R-L 5k main-experiment summary

Generated: 2026-09-07T09:45:57.420776+08:00

All reported long-horizon confidence intervals are based on exact ordered sample pairs and 20,000 actor-then-motion cluster bootstrap resamples with fixed seed 20260907. Lower errors are better.

## Answers to the registered questions

1. **Final checkpoint and continuation.** The frozen checkpoint is `/data4/zts/result/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt` (SHA-256 `f197e8d7c12fca6c82db0d36a1e31ccf34163074ec8ec9aa847f0a25ca696020`). It is the historical total-step-5000 artifact whose recorded lineage is D2R step250 to total step5000. It is not represented as a fresh direct continuation from Native Hybrid 50k.

2. **Table 4 short-horizon quality.** Yes under the registered 5% compatibility criterion: D2R-L minus Native is +1.65% Root ADE and -1.15% constraint error. Individual secondary metrics remain visible in `FINAL_TABLE4.md`; this is not a claim of winning every metric.

3. **Table 5 four-model comparison.** Complete in-horizon and out-of-horizon 9-second rows for ARDY, Minimal-W4, Native Hybrid, and D2R-L are reported in `FINAL_TABLE5.md`. The values are BONES-SEED geometry proxies, not unavailable official learned metrics.

4. **20s/40s versus Native Hybrid.** The confirmatory 40-second paired AUC comparison is Root -122.54 [-209.11, -56.05]; Constraint -140.12 [-228.59, -80.07]. The dual-AUC conclusion is significant in favor of D2R-L. The 20-second statistics are retained in `FINAL_20S_RESULTS.json`.

5. **Error propagation curves.** Yes. The Native-minus-D2R-L Root gap grows from +2.33 cm at W5 to +15.44 cm at W19; the Constraint gap grows from +2.65 cm to +18.49 cm. The figures report every 40-second rollout window and mark the W4 eviction boundary. The gap is not strictly monotonic at every intermediate window, but the late-horizon separation is substantially larger.

6. **state_query-only contribution.** Relative to Native Hybrid at 40 seconds: Root -23.22 [-50.36, +1.72]; Constraint -43.75 [-78.11, -11.23].

7. **persistent-memory-only contribution.** Relative to Native Hybrid at 40 seconds: Root -130.97 [-236.39, -67.01]; Constraint -134.39 [-248.51, -72.66].

8. **Is full D2R-L overall best?** Not uniquely in the structural ablation. Full minus state_query-only is Root -99.33 [-173.01, -41.48]; Constraint -96.37 [-177.43, -33.49]. Full minus persistent-memory-only is Root +8.43 [-45.85, +62.05]; Constraint -5.73 [-55.46, +48.69]. Thus full significantly outperforms state_query-only, but is statistically tied with persistent-memory-only, whose aggregate Root ADE is also lower. Because D2R-L has a different historical initialization lineage, these differences are not a perfectly parent-matched factorial attribution.

9. **Memory write/read intervention.** This final run re-evaluated both write-off and readout-off-after-W4 using the frozen checkpoint and exact 40-second panel. Both Root and Constraint AUC degradation intervals are strictly above zero; see `FINAL_MEMORY_WRITE_READ_ABLATION.md`.

10. **Final model decision.** `FINAL_MAIN_MODEL = D2R-L`. This follows the predeclared combination of <=5% short-horizon degradation and dual-metric significant 40-second improvement over Native Hybrid. The claim is limited to the reported BONES-SEED geometry/constraint protocol.

## Claim boundary

The evidence can support short-horizon compatibility, long-horizon error reduction, and causal use of the memory read/write path when their corresponding intervals pass. It does not establish official HumanML3D/BONES-SEED learned metrics, an independent state-delta benefit, or trajectory-specific memory identity. No locked-final actors were read, and no 50k continuation result is used.
