# D2R Contextual-Delta Remediation: Final Summary

Date: 2026-09-03
Final decision: **STOP at step 250; not eligible for Stage E**
Data: BONES-SEED v004 selection actors only; locked-final actors were not read

## Outcome

D2R fixed the old D2 training contract rather than extending the failed
checkpoint. It restarted from the healthy Native Hybrid 50k parent, used a
six-window W6 target after W1 eviction, attached the W1 write, retained four
accepted writes in BPTT, added explicit pooled-text conditioning of normalized
delta, bounded the write rate at 0.25, and lowered LR from `5e-4` to `1e-4`.

This remediation succeeded for persistent memory but not for delta. At step
250, permuting memory and disabling writes or reads significantly worsened the
40s post-W5 AUC. D2R therefore establishes trajectory-specific, fixed-capacity
content-memory use on the BONES selection panel. However, D2R-full did not beat
D2R-no-delta, permuting delta was null, and zeroing delta significantly
improved both Root and Constraint AUC. The allowed mechanism name is therefore
**state-conditioned persistent content memory**, not `delta-state-driven`.

The final gate still fails because absolute post-W5 Root and Constraint slopes
remain positive. No additional training and no Stage E/F/G/H/I run was
started.

Official HumanML3D/BONES learned FID and R-Precision remain unavailable; all
reported real-motion values are preregistered geometry/constraint proxies.

## Why Old D2 Failed

Old D2 supervised W3 with only three rollout windows and BPTT depth 1, before
W1 could be evicted from W4. Its W1 write was detached. It also decoded the
roughly 12x larger absolute state and small delta through one linear map with
no explicit text x delta interaction. A mean write gate near 0.713 retained
only about `0.287^4 = 0.0068` of early content after four writes, while memory
and proposal norms grew to about 35.92/38.13. The old model could learn a
generic correction without learning durable, trajectory-specific delta use.

Adding delta was therefore not sufficient: information helps only when it is
scaled, interpreted under context, preserved through eviction, and optimized
by a post-eviction objective.

## Safety And Training Audit

| Check | D2R-full result |
|---|---:|
| Step-0 raw output/loss/cache/geometry difference | exactly 0 |
| Initial write-gate mean | 0.124935 |
| Initial read scale | 0 |
| Decoder/write/read nonzero-gradient updates | 249/249/250 of 250 |
| Final write-gate mean / max | 0.1665 / 0.2020 |
| Final memory slot norm | 17.19 |
| Final read residual scale | -0.002388 |
| 52-frame maximum relative degradation vs Native | 0.22% |
| 9s maximum relative degradation vs Native | 0.73% |

All gradients and states were finite. The write gate remained below its 0.25
architectural cap and the 16,384 bytes/layer content-memory allocation stayed
constant with horizon. The trainer's conservative random-minibatch trend flag
is `FAIL` (`14.84` first-50 mean versus `15.15` last-50 mean), so it is retained
as an audit warning. The fixed selection health panels show that the healthy
short-horizon baseline was not damaged, and the failed final gate prevents
further training.

## Selection Results

Lower is better.

| Model | Table 4 Root / Constraint (cm) | 9s Root / Constraint (cm) | 20s Root / Constraint (cm) | 40s Root / Constraint (cm) |
|---|---:|---:|---:|---:|
| Native Hybrid-W4 50k | 20.91 / 27.06 | 16.47 / 25.23 | 21.03 / 27.84 | 36.83 / 41.94 |
| Minimal-W4 50k | n/a | n/a | 31.13 / 37.19 | 60.85 / 62.47 |
| D2R-no-delta step250 | 20.99 / 26.97 | 16.42 / 25.06 | 20.44 / 27.14 | 35.94 / 40.45 |
| D2R-full step250 | 20.96 / 27.02 | 16.59 / 25.25 | 20.56 / 27.47 | 36.03 / 40.70 |

The D2R-full absolute 40s post-W5 slopes are `+2.4465` Root and
`+2.4474` Constraint cm/window, so the required nonpositive-slope check fails.

## Paired 40s Evidence

Differences are candidate minus baseline; negative favors D2R-full. CIs are
actor-clustered 95% bootstrap intervals over 16 motions and seven actors.

| Comparison | Root post-W5 AUC | Constraint post-W5 AUC |
|---|---:|---:|
| D2R-full - Native | -14.82 [-25.14, -2.58] | -22.32 [-33.10, -10.21] |
| D2R-full - Minimal-W4 | -353.41 [-438.06, -280.39] | -307.96 [-390.47, -232.31] |
| D2R-full - no-delta | -0.66 [-5.15, 3.48] | +1.59 [-2.38, 5.20] |

D2R-full significantly improves AUC relative to Native and Minimal-W4, but it
does not significantly improve either AUC relative to no-delta.

## Causal Interventions

Differences are intervention minus normal D2R-full; positive means the
intervention is worse.

| Intervention | Root post-W5 AUC | Constraint post-W5 AUC | Interpretation |
|---|---:|---:|---|
| Write off | +15.67 [5.85, 24.94] | +21.78 [12.30, 30.96] | write causality PASS |
| Readout off after W4 | +8.46 [3.14, 14.61] | +12.54 [5.09, 20.09] | read causality PASS |
| Permute memory after W4 | +3.21 [0.25, 6.46] | +2.92 [0.18, 6.25] | trajectory specificity PASS |
| Permute delta | +0.41 [-2.28, 3.40] | +0.30 [-2.25, 3.20] | no delta evidence |
| Zero delta | -3.80 [-6.84, -1.32] | -3.17 [-5.90, -0.71] | delta removal improves |

The memory permutation audit changed every memory checksum, changed no
Attention KV checksum, ran exactly once after W4, and rejects B=1. Thus the
memory result is not explained by accidentally permuting the recent KV cache.

## Required Answers

1. **Old initialization dead-start?** Yes. Decoder, write/read gates,
   StateProjector, and Mamba task gradients were exactly zero even with attached
   old-history writes.
2. **Step-0 Native parity?** Yes. Raw output, denoising loss, Attention KV,
   Mamba state, and 52-frame geometry differences were exactly zero.
3. **Decoder task gradient?** Yes. Real-Mamba CUDA attached history gave
   `6.74e-6`; detached history gave zero. BONES updated it on 249/250 steps.
4. **Write gate reads previous memory?** Yes. It consumes
   `[state, delta, flatten(M_previous)]`; the regression test changes only
   previous memory and observes a changed gate.
5. **Fixed capacity?** Yes. K=2 and 16,384 bytes/layer at batch size 2 for all
   horizons; Attention remains W4.
6. **Synthetic delayed cue?** PASS: correct 1.0, write/read-off 0.5,
   permute-memory 0.0, permute-context 0.0, Minimal-W4 0.5098.
7. **More stable than Native?** AUC is significantly lower, but absolute
   slopes remain positive; quality improves without proving nonaccumulation.
8. **Better than Minimal-W4?** Yes for Root/Constraint AUC with CIs below zero;
   no for the required absolute nonpositive slopes.
9. **Trajectory-specific causal use?** Yes on this selection panel:
   permute-memory, write-off, and readout-off all have AUC CIs above zero.
10. **Independent delta contribution?** No. Full versus no-delta is null,
    permute-delta is null, and zero-delta is significantly better.

## Gate And Claim Boundary

| Final check | Result |
|---|---|
| Step-100 screening contract still passes | PASS |
| Root/Constraint AUC better than Minimal-W4 with CI | PASS |
| Absolute post-W5 Root and Constraint slopes <= 0 | **FAIL** |
| Permute-memory Root/Constraint AUC CIs > 0 | PASS |
| Write/read causal AUC CI > 0 | PASS |

Final decision: `STOP`. The defensible result is:

> On the BONES selection panel, fixed-capacity state-conditioned persistent
> content memory is causally used after W4 eviction and improves long-horizon
> AUC, while errors still accumulate and projected delta has no demonstrated
> independent benefit.

This is selection evidence, not locked-final evidence. HumanML3D official
metrics remain blocked by the unresolved Core27-to-HumanML263 bridge.

## Artifacts

- `docs/D2R_STEP100_RESULTS.json`
- `docs/D2R_CAUSAL_INTERVENTION_RESULTS.json`
- `docs/D2R_STEP250_RESULTS.json`
- `docs/D2R_STEP250_CAUSAL_INTERVENTION_RESULTS.json`
- `docs/D2R_CONTEXTUAL_CAUSAL_MEMORY_RESULTS.json`
- `/data4/zts/result/stage_d2r_bones_step250_selection_eval_20260903`
- `/data4/zts/result/stage_d2r_bones_step250_full_20260903`
- `/data4/zts/result/stage_d2r_bones_step250_no_delta_20260903`

Step-250 checkpoint SHA-256: D2R-full
`45b5cfacd0c0e734c1782ef667154ed837142a0c34ad4c136e22d62ba6146daa`;
D2R-no-delta
`f284da037e1c88acfcadc7e8ebf75d75ee8196b20b6a194eb4b3c0bc5bddc503`.
