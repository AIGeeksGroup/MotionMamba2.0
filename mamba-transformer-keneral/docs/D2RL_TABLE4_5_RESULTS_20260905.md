# D2R-L Table 4/5 Results

Date: 2026-09-05

## Scope And Status

This report summarizes the matched D2R-L continuation trained to **5,000
optimizer steps** from the D2R step-250 checkpoints. `D2R-L-full` uses the
text-conditioned state-delta content write; `D2R-L-no-delta` is the matched
ablation. Both use the same BONES-SEED v004 selection actors, prompts,
constraints, diffusion noise, CFG weights, and evaluation panels.

The results are **BONES-SEED geometry/constraint proxies** under the registered
ARDY Table 4/5 protocol. BONES-SEED official learned FID, R-Precision, FVMD,
and TM-Sim evaluators were unavailable and are therefore not reported as
official metrics. Locked-final actors were not read.

## Paper-Ready Table 4

Table 4 is the 52-frame joint-position/control panel. Lower is better. Values
are means over 64 selection samples.

| Method | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native Hybrid-W4, 50k parent | 27.06 | 20.91 | n/a | n/a | n/a | n/a | n/a | n/a |
| D2R-L-full, 5k continuation | 26.75 | 21.26 | 21.59 | 14.49 | 0.02452 | 17.66 | 36.68 | 1.020 |
| D2R-L-no-delta, 5k continuation | **25.89** | **20.42** | **20.57** | **14.51** | 0.02462 | 17.67 | **36.63** | 1.048 |

The Native values are the registered 50k parent reference from the fixed
baseline report; D2R-L-full and D2R-L-no-delta were evaluated together on the
same Table 4 panel (SHA-256 `be23b88c53976ec40395905d385715bed8c4767cf1cc8ed8c13469a7571a7f31`).
The no-delta row is lower on the aggregate Table 4 metrics; this table does not
support a delta advantage claim.

## Paper-Ready Table 5: Standard 9-Second Protocol

Table 5 uses a 24-frame (0.96 s) GT prefix and 201 generated frames at 25 fps.
In-horizon places a goal at each generation window; out-of-horizon places the
goal only at the final frame. Values are means over 64 samples and exclude the
GT prefix. Lower is better.

| Method | Protocol | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D2R-L-full, 5k continuation | In-horizon | 23.51 | 14.55 | 25.68 | 15.93 | 0.02109 | 15.26 | 33.64 | 4.170 |
| D2R-L-no-delta, 5k continuation | In-horizon | **22.79** | **14.18** | **25.19** | **15.16** | **0.02088** | **14.85** | 33.90 | 4.314 |
| D2R-L-full, 5k continuation | Out-of-horizon | **24.75** | **13.14** | **19.53** | 14.60 | 0.02008 | 15.08 | **32.92** | 4.182 |
| D2R-L-no-delta, 5k continuation | Out-of-horizon | 25.93 | 14.43 | 21.21 | **13.84** | **0.01987** | **14.75** | 32.63 | 4.345 |

The standard Table 5 panel hashes are `9007499704361c63d579f9f1eb6aed187b1aad99a47da32d980b225f9e8d4653` (in-horizon) and
`3696795df96fa1e1d3a8543c1f7c0d47bb661ab60de55802841f5444c102cfa4`
(out-of-horizon). The no-delta ablation is better on most 9-second quality
metrics; full retains a lower out-of-horizon Root ADE, Root FDE, constraint
error, and skate proxy than no-delta.

## Table 5 Long-Horizon Supplement

These are duration extensions of the Table 5 protocol, not replacements for
the paper's standard 9-second table. Each row uses 16 ordered motions and the
same panel for full/no-delta and baseline controls. Window 5 is the first
post-eviction read after the four-chunk attention cache has dropped the oldest
window. Lower is better.

| Duration | Method | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Post-W5 Root slope | Post-W5 Constraint slope |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 s | Native Hybrid-W4, 50k | 27.84 | 21.03 | 38.03 | 14.76 | 0.01660 | 14.44 | 29.82 | 3.060 | 3.056 |
| 20 s | Minimal-W4, 50k | 37.19 | 31.13 | 43.60 | 19.47 | 0.02226 | 16.10 | 47.21 | 1.605 | 1.181 |
| 20 s | D2R-L-full, 5k continuation | 23.53 | 15.72 | 25.47 | 15.91 | 0.01966 | 14.78 | 30.33 | 1.496 | 1.172 |
| 20 s | D2R-L-no-delta, 5k continuation | 24.79 | 16.79 | 27.61 | 15.67 | 0.01956 | 14.73 | 30.33 | 1.710 | 1.680 |
| 40 s | Native Hybrid-W4, 50k | 41.94 | 36.82 | 56.14 | 20.32 | 0.01633 | 14.81 | 26.92 | 2.503 | 2.495 |
| 40 s | Minimal-W4, 50k | 62.47 | 60.85 | 62.27 | 24.84 | 0.02531 | 18.98 | 48.17 | 0.682 | 0.122 |
| 40 s | D2R-L-full, 5k continuation | **32.43** | **28.73** | **42.34** | **17.65** | 0.01960 | **14.63** | **30.26** | 1.263 | 1.102 |
| 40 s | D2R-L-no-delta, 5k continuation | 33.79 | 29.50 | **42.35** | 17.73 | **0.01962** | 14.80 | 30.56 | **1.212** | **1.050** |

At 40 s, D2R-L-full improves Root AUC and Constraint AUC relative to both
Native and Minimal controls, but its absolute post-W5 slopes remain positive.
It is slightly better than no-delta on final Root/Constraint means, while the
paired actor-cluster confidence intervals for full-minus-no-delta AUC include
zero:

| 40 s paired comparison | Root post-W5 AUC difference (95% CI) | Constraint post-W5 AUC difference (95% CI) |
|---|---:|---:|
| D2R-L-full minus Native | -122.54 [-209.67, -54.62] | -140.12 [-230.51, -79.92] |
| D2R-L-full minus Minimal-W4 | -461.13 [-570.64, -371.68] | -425.82 [-525.50, -345.97] |
| D2R-L-full minus no-delta | -11.82 [-28.40, 9.25] | -20.80 [-41.28, 4.04] |

## Causal Delta/Memory Ablations

The following differences are intervention minus normal D2R-L-full at 40 s;
positive means the intervention is worse. CIs are actor-clustered bootstrap
intervals over 16 motions and seven actors.

| Intervention | Root AUC difference (95% CI) | Constraint AUC difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Write off | +120.86 [54.61, 215.11] | +118.26 [45.65, 220.89] | write usage causal |
| Readout off after W4 | +69.33 [24.37, 124.06] | +69.08 [21.08, 129.15] | read usage causal |
| Permute memory after W4 | +0.05 [-38.34, 25.64] | +0.02 [-36.19, 24.47] | trajectory-specific permutation not established |
| Permute delta | -8.89 [-23.76, 2.23] | -8.63 [-23.84, 3.28] | delta intervention null |
| Zero delta | -9.63 [-30.44, 3.44] | -7.20 [-26.01, 6.44] | no evidence delta is beneficial |

The write/read interventions remain significant, but the step5000
permute-memory result is not significant. Consequently the long-training run
does not satisfy the full trajectory-specific causal gate, and it does not
establish an independent delta contribution. The final registered decision is
`STOP`; no Stage E experiment is authorized by these results.

## Training And Reproduction Metadata

| Item | Value |
|---|---|
| Parent | D2R step250 full/no-delta checkpoints |
| Total training | 5,000 optimizer steps |
| Variants | full delta and matched no-delta |
| Learning rate | `1e-4` |
| Global batch size | 4 |
| Rollout/BPTT | six windows, W6 target, four accepted-history BPTT windows |
| Attention cache | W4 |
| Content memory | K=2, max 16,384 bytes/layer at eval batch size 2 |
| Selection samples | 64 for Table 4/9s; 16 for 20/40 s |
| Diffusion | 10 steps; CFG text/constraint `2/2` |
| Checkpoint SHA-256 (full) | `f197e8d7c12fca6c82db0d36a1e31ccf34163074ec8ec9aa847f0a25ca696020` |
| Checkpoint SHA-256 (no-delta) | `b7018ef75cc48012a74b6033bd21d68de638fe1cd12f9aad23c8b7eca4e9c5f5` |
| Evaluation artifact root | `/data4/zts/result/stage_d2rl_bones_long_20260904/step5000` |
| Selection analysis | `D2RL_STEP5000_RESULTS.json` and `D2RL_STEP5000_CAUSAL_INTERVENTION_RESULTS.json` |

All 500/1000/2000/5000 training safety gates passed. The trainer's random
minibatch trend is retained as an informational `FAIL` warning, but all
gradients/states were finite, gradient-update fractions increased from about
`0.998` at step500 to `0.9998` at step5000, write gates stayed below 0.25,
and short-horizon health degradation stayed within the registered 5% limit.

## Paper Claim Boundary

The defensible statement is:

> Long training improves BONES-SEED long-horizon geometry proxies for the
> state-conditioned persistent-memory model, and write/read interventions show
> that the memory branch is used. The 5,000-step run does not establish a
> statistically independent benefit from the projected state delta, nor does
> it establish trajectory-specific memory permutation causality at this panel
> size.

Do not report these proxy values as official HumanML3D or BONES-SEED FID,
R-Precision, FVMD, or TM-Sim scores.
