# D2 Native-Preserving Persistent Content Memory: Final Summary

Date: 2026-09-03  
Final status: **STOP after step 250; not eligible for Stage E**  
Data split: BONES-SEED v004 selection actors only; locked-final actors were not read

## Outcome

D2 fixed the original content-memory dead start, preserved the healthy Native
Hybrid exactly at migration step 0, passed the learned delayed-cue mechanism
benchmark, and trained normally through the accepted-window BPTT path. The
step-100 screening gate passed and therefore authorized the registered
step-250 continuation.

At step 250, D2-full improved 40-second mean Root ADE and constraint error from
the healthy Native Hybrid's `36.83/41.94 cm` to `33.60/37.74 cm`. It also had
substantially lower post-W5 Root/Constraint AUC than Minimal-W4. This is useful
long-rollout quality evidence, but it is not the requested trajectory-specific
causal-memory proof: both post-W5 slopes remain positive, permuting content
memory has negligible effect, and actor-cluster CIs for write/read ablations
cross zero. The preregistered final scientific gate therefore fails and no
E/F/G/H/I experiment was started.

Official learned BONES FID/R-Precision remain **N/A**. Every number below is a
geometry, constraint, physical, latency, or state proxy from the registered
ARDY protocol.

## Selection Results

Lower is better. Table 4 is the 52-frame joint-control proxy. The 9-second row
uses the paper-duration in-horizon protocol; 20/40 seconds are its registered
duration extensions.

| Model | Stage | Table 4 Root / Constraint (cm) | 9s Root / Constraint (cm) | 20s Root / Constraint (cm) | 40s Root / Constraint (cm) |
|---|---:|---:|---:|---:|---:|
| Native Hybrid-W4 | 50k parent | 20.91 / 27.06 | 16.47 / 25.23 | 20.99 / 27.79 | 36.83 / 41.94 |
| Minimal-W4 | 50k parent | 31.07 / 37.58 | 18.88 / 27.04 | 31.13 / 37.19 | 60.85 / 62.47 |
| D2-no-delta | 100 | 20.94 / 26.97 | 16.35 / 24.94 | 20.41 / 27.07 | 35.65 / 40.40 |
| D2-full | 100 | 20.95 / 27.06 | 16.31 / 24.93 | 20.41 / 27.13 | 35.75 / 40.47 |
| D2-no-delta | 250 | 21.04 / 27.02 | 16.94 / 25.97 | 19.69 / 25.88 | 34.96 / 39.06 |
| D2-full | 250 | 21.06 / 27.01 | 16.90 / 25.95 | 19.18 / 25.32 | 33.60 / 37.74 |

At step 250 the largest registered short-horizon degradation relative to
Native is `2.93%`, below the `5%` safety threshold. The out-of-horizon 9-second
Root/Constraint results are `13.87/25.00 cm` for D2-full and `13.70/25.06 cm`
for D2-no-delta.

## Long-Horizon Statistics

Differences are paired candidate-minus-baseline actor-cluster bootstrap values;
negative favors D2-full. CIs are 95%.

| Step-250 comparison at 40s | Root post-W5 AUC | Root slope | Constraint post-W5 AUC | Constraint slope |
|---|---:|---:|---:|---:|
| D2-full - Native Hybrid | -16.32 [-122.37, 47.63] | +0.163 [-0.682, 0.790] | -32.69 [-136.94, 30.62] | +0.102 [-0.722, 0.737] |
| D2-full - Minimal-W4 | -354.90 [-508.36, -246.68] | +1.749 [0.871, 2.625] | -318.32 [-448.32, -220.62] | +2.293 [1.331, 3.284] |
| D2-full - D2-no-delta | -15.70 [-39.20, -0.29] | -0.122 [-0.285, -0.016] | -14.56 [-37.96, 0.55] | -0.122 [-0.295, 0.001] |

D2-full's absolute post-W5 Root/Constraint slopes are still positive at
`2.387/2.327 cm per window`. Minimal-W4 is flatter only because it has already
accumulated much larger errors; D2-full has decisively lower AUC but has not
eliminated continuing error growth.

## Causal Interventions

All interventions use the same D2-full step-250 checkpoint, 16 ordered 40s
samples, prompts, constraints, CFG, diffusion noise, W4 KV, and 10 diffusion
steps. Differences below are ablation-minus-normal AUC; positive means the
ablation is worse.

| Intervention | Root AUC difference (95% CI) | Constraint AUC difference (95% CI) |
|---|---:|---:|
| write off for all accepted writes | +36.68 [-20.58, 143.77] | +37.55 [-21.00, 148.00] |
| readout off after W4 | +21.83 [-18.03, 94.84] | +23.60 [-17.94, 99.97] |
| permute memory once after W4 | +0.26 [-1.22, 1.69] | +0.56 [-1.05, 2.12] |
| zero delta for all accepted writes | +0.62 [-1.67, 3.16] | +0.92 [-1.61, 3.90] |
| permute delta for all accepted writes | +0.45 [-1.24, 2.42] | +0.67 [-1.25, 2.82] |

The permutation ran exactly once before W5 in each batch. Every Attention KV
checksum was unchanged, every memory checksum changed, and B=1 permutation is
rejected. Thus the intervention implementation is valid and memory rows are
not identical, but the behavioral permutation result does not establish
trajectory-specific content use. Write/read ablations have adverse means but
their actor-cluster CIs do not establish repeatable causality at this panel
size.

## Required Questions

1. **Did the old D initialization dead-start?** Yes. Under both detached and
   attached W1/W2 history, decoder, write gate, read gate, StateProjector, and
   Mamba gradients were exactly zero. Only ordinary Attention and the unrelated
   state scale received gradients.
2. **Does fixed D2 preserve Native Hybrid at step 0?** Yes. Root/body raw
   output, denoising loss, Attention KV, Mamba state, and 52-frame geometry all
   had exactly zero difference. Accepted memory was already nonzero while both
   new read scales were zero.
3. **Does the content decoder receive task-loss gradient?** Yes. The real Mamba
   CUDA attached-history test produced a decoder norm of `6.74e-6`; the
   detached control was exactly zero. BONES training had nonzero decoder
   gradients on `249/250` updates.
4. **Does the write gate depend on previous memory?** Yes. The production gate
   consumes `[state, delta, flatten(M_previous)]`, and the fixed-input
   previous-memory regression changes its output.
5. **Is capacity fixed?** Yes. K is always 2. At evaluation batch size 2 each
   layer uses `16,384` content-memory bytes regardless of horizon; W4 Attention
   storage is also bounded after four chunks.
6. **Did learned delayed cue beat causal ablations?** Yes. Correct accuracy was
   `1.0`; write-off/readout-off were `0.5`, memory permutation was `0.0`, and
   Minimal-W4 was `0.5039` after W1 eviction.
7. **Is BONES 40s post-W5 behavior more stable than Native?** At step 100, all
   four paired slope/AUC differences favored D2-full with 95% CIs excluding
   zero. At step 250, mean Root/Constraint errors improve further, but the
   paired Native differences have wide CIs and both absolute slopes remain
   positive. The evidence is therefore positive but not a final stability
   proof.
8. **Does D2-full beat Minimal-W4?** It decisively beats Minimal-W4 on 40s
   Root and Constraint AUC with CIs excluding zero, but not on slope. The
   Minimal curve is flatter at a much worse error plateau.
9. **Do permutation/read/write prove trajectory-specific causal use?** No.
   Memory permutation is behaviorally negligible. Write/read adverse means are
   not significant under actor-cluster bootstrap. Software isolation is proven;
   real-motion trajectory-specific causality is not.
10. **Does delta have an independent contribution?** Step 100 did not show it.
    At step 250, D2-full beats D2-no-delta on Root AUC and Root slope with CIs
    excluding zero, which satisfies the registered variant-ablation criterion,
    but zero/permute-delta interventions remain null and the Constraint CIs
    touch zero. This is limited variant evidence, not a robust delta mechanism
    claim.

## Training Audit

Both variants resumed the registered step-100 state and reached 250 with the
same parent, LR `5e-4`, seed `9032026`, global batch 4, three-window generated
history, diffusion depth 10, BPTT depth 1, W4 cache, K=2, and
`train_scope=state_memory`. Decoder/write/read nonzero-gradient fractions were
`0.996/0.996/1.000`; all gradients and states were finite.

The trainer's informational status is `FAIL` because its random-minibatch loss
heuristic rose from about `1.096` to `1.932`. This was not silently relabeled:
memory/proposal norms also grew to `35.92/38.13` in D2-full. Selection metrics
improved, but the growth reinforces the decision not to extend training after
the failed final causal gate.

Step-250 checkpoint SHA-256:

- D2-full: `5ce930aaaba1f5f9d06913d5baeef5305e979d53f7a827a30f42390b52d2de0b`
- D2-no-delta: `a34c75ba3664bb54b31ee6228d62e76b505e891a84956e6acb052779d1384cc3`

## Reproduction and Artifacts

```bash
./examples/run_stage_d2_preflight_20260903.sh
./examples/run_stage_d2_step0_parity_20260903.sh
./examples/run_stage_d2_step100_20260903.sh
PYTHONPATH=src python examples/analyze_stage_d2_gate.py --stage 100
./examples/run_stage_d2_step250_20260903.sh
D2_STAGE=250 ./examples/run_stage_d2_eval_20260903.sh
PYTHONPATH=src python examples/analyze_stage_d2_gate.py --stage 250
```

Structured summaries are in `docs/D2_STEP100_RESULTS.json`,
`docs/D2_CAUSAL_INTERVENTION_RESULTS.json`, `docs/D2_STEP250_RESULTS.json`, and
`docs/D2_STEP250_CAUSAL_INTERVENTION_RESULTS.json`. Raw selection results,
per-sample per-window rows, state diagnostics, intervention checksum audits,
logs, and environment manifests are under:

```text
/data4/zts/result/stage_d2_bones_selection_eval_20260903
/data4/zts/result/stage_d2_bones_step250_selection_eval_20260903
```

The latest audited evaluator implementation is commit `c91ef87`. Historical
negative runs, checkpoints, and reports were retained rather than overwritten.

## Claim Boundary

The allowed conclusion is:

> We implemented a trainable, fixed-capacity state-conditioned persistent
> content memory and validated its software/lifecycle contract. It improves
> BONES selection long-rollout proxies, but real-motion trajectory-specific
> long-term causal usage remains unestablished.

The stronger claim that carried Mamba state was decoded into
trajectory-specific content that causally improves motion after W4 eviction is
not supported by the final intervention gate.

## D2R Remediation Addendum (2026-09-03)

The report above is the immutable result of the original D2 experiment. A new
D2R experiment restarted from the healthy Native Hybrid 50k parent and fixed
the identified supervision, detachment, conditioning, write-retention, and
scale problems. It did not resume or overwrite the original D2 checkpoint.

D2R materially changes the mechanism conclusion: at step 250, write-off,
readout-off, and post-W4 memory permutation all significantly worsen BONES 40s
post-W5 AUC, so trajectory-specific persistent-memory use is now established
on the selection panel. Delta itself is not established: D2R-full is not
better than no-delta, permute-delta is null, and zero-delta significantly
improves both primary AUCs. Absolute Root/Constraint slopes remain positive,
so the final decision is still `STOP` and Stage E was not authorized.

The superseding remediation analysis is in `docs/D2R_FINAL_SUMMARY.md`; raw
and structured D2R artifacts use the `stage_d2r_*` and `D2R_*` names so the two
experimental histories cannot be confused.
