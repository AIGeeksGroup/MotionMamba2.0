# BONES-SEED Generated-History Scheduled-Sampling Gate

## Scope

This report is independent of the teacher-history protocol-aligned report. All three models start from their designated healthy checkpoints. Training uses one GT seed window followed by a scheduled mixture of accepted model generations and teacher history; the supervised window is expressed in the generated endpoint coordinate frame.

| Setting | Value |
|---|---|
| Dataset | BONES-SEED v004 G1, actor-disjoint train/val/test |
| Gate budgets | 1k; 5k only if every predefined 1k criterion passes |
| History sampling strategy | balanced_v2 |
| Rollout curriculum | Generated branch: 40% x3, 40% x4, 20% x5; teacher branch: 50% x2, 20% x3, 20% x4, 10% x5 (maximum 10.4 s) |
| Generated-history schedule | 0.25 -> 0.75 by 1k; -> 0.85 by 5k |
| Short-window retention | 35% fixed short-window replay |
| Optimizer | AdamW, LR 5.0e-06, global batch 16, entire denoiser |
| Sampling during training | 10 diffusion steps, text/constraint CFG 2/2 |
| Evaluation | Table 4, official-style 9 s, and matched 20/40 s extensions |
| State audit | root/body per-layer Mamba norm, state-difference norm, KV size, fusion gate |
| Reset ablation | no reset, then persistent-state-only reset every 2/4/8 generated windows |

## Stage 1000

### Training Audit

| Method | Trainer trend flag | Eval loss | Generated updates/windows | Teacher/short updates | Rollout counts | Time (h) |
|---|---|---:|---:|---:|---|---:|
| ARDY | FAIL | 0.6297 | 348/631 | 332/320 | 2w:161, 3w:193, 4w:221, 5w:105 | 0.11 |
| Minimal | FAIL | 0.4324 | 348/631 | 332/320 | 2w:161, 3w:193, 4w:221, 5w:105 | 0.26 |
| Hybrid | FAIL | 0.3155 | 348/631 | 332/320 | 2w:161, 3w:193, 4w:221, 5w:105 | 0.42 |

The trainer trend flag compares early and late losses even though the generated-history probability increases throughout training, making the late objective harder. Therefore `FAIL` here is not a numerical-health failure. The predefined gate separately verifies finite gradients, finite validation loss, output-weight norm, and matched draws.

### Table 4: Offline Text and Joint-Position Control

| Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |
|---|---:|---:|---:|---:|
| ARDY | 50.80 | 55.04 | 66.84 | 16.38 |
| Minimal | 52.46 | 62.06 | 77.30 | 19.42 |
| Hybrid | 39.56 | 43.37 | 53.91 | 18.99 |

### Table 5 and Long-Horizon Extension

| Duration | Protocol | Method | Constraint (cm) | Root ADE (cm) | Post-W5 Root slope | Post-W5 constraint slope |
|---:|---|---|---:|---:|---:|---:|
| 9 s | in | ARDY | 45.15 | 40.96 | N/A | N/A |
| 9 s | in | Minimal | 50.13 | 45.19 | N/A | N/A |
| 9 s | in | Hybrid | 41.05 | 32.48 | N/A | N/A |
| 9 s | out | ARDY | 59.51 | 45.11 | N/A | N/A |
| 9 s | out | Minimal | 79.80 | 57.94 | N/A | N/A |
| 9 s | out | Hybrid | 58.69 | 35.28 | N/A | N/A |
| 20 s | in | ARDY | 251.40 | 234.10 | 18.673 | 17.220 |
| 20 s | in | Minimal | 217.62 | 225.72 | -13.139 | -11.354 |
| 20 s | in | Hybrid | 120.23 | 113.83 | -0.113 | 2.030 |
| 20 s | out | ARDY | 466.95 | 324.71 | 31.815 | N/A |
| 20 s | out | Minimal | 569.80 | 374.52 | 25.146 | N/A |
| 20 s | out | Hybrid | 586.38 | 389.13 | 15.685 | N/A |
| 40 s | in | ARDY | 447.86 | 429.44 | 74.454 | 76.167 |
| 40 s | in | Minimal | 171.36 | 176.89 | 5.128 | 5.106 |
| 40 s | in | Hybrid | 141.89 | 140.45 | -3.468 | -3.857 |
| 40 s | out | ARDY | 1642.63 | 582.29 | 93.597 | N/A |
| 40 s | out | Minimal | 770.80 | 517.42 | 35.369 | N/A |
| 40 s | out | Hybrid | 966.21 | 644.04 | 39.108 | N/A |

### Hybrid State Reset Ablation (40 s, In-Horizon)

| Reset interval | Constraint (cm) | Root ADE (cm) | Root slope | Max state norm | Max state drift | Finite |
|---:|---:|---:|---:|---:|---:|---|
| none | 141.89 | 140.45 | -3.468 | 6661.840 | 3976.752 | True |
| 2 | 139.95 | 137.16 | 2.989 | 6809.315 | 4050.787 | True |
| 4 | 143.19 | 139.12 | 2.538 | 6807.731 | 4007.888 | True |
| 8 | 145.54 | 141.78 | -1.041 | 6673.266 | 4013.413 | True |

### Predefined Gate Decision

**FAIL**. Continue to 5k: **False**.

| Criterion | Passed |
|---|---|
| training healthy | True |
| matched draws | True |
| short retention | False |
| hybrid long slopes better than minimal | False |
| state finite | True |
| no state norm explosion | True |

| Duration | Hybrid-Minimal root-slope gap | Paired 95% CI | Hybrid better fraction |
|---:|---:|---:|---:|
| 20 s | 13.026 | [-15.108, 41.228] | 0.35 |
| 40 s | -8.596 | [-18.613, -2.005] | 0.70 |

State audit: 200 commit records; maximum norm 6661.840, maximum drift 3976.752, maximum final/initial norm ratio 2.932.

### Stage Finding

Short-horizon retention fails the predefined 10% limit for 9s_in_horizon constraint (+14.97%), 9s_in_horizon Root ADE (+20.81%), 9s_out_of_horizon constraint (+13.62%), 9s_out_of_horizon Root ADE (+20.59%). At 40 seconds, Hybrid has Root ADE 140.45 cm and slope -3.468, versus Minimal's 176.89 cm and 5.128.

Resetting persistent state every eight generated windows worsens Hybrid's 40-second Root slope to -1.041 and constraint slope to -0.735, but worsens average Root ADE to 141.78 cm. State remains finite and its maximum final/initial norm ratio is below the explosion threshold. The evidence therefore does not support simple Mamba norm explosion as the primary cause; periodic reset only partially controls late-window drift and does not repair accumulated trajectory bias.

## Interpretation Rule

A lower denoising loss is not sufficient. The experiment supports the long-horizon Hybrid claim only when Hybrid improves both 20 s and 40 s post-window-5 root/constraint slopes over Minimal, retains Table 4 and 9 s within the predefined 10% tolerance, and shows no non-finite or greater-than-10x state-norm growth. A failed 1k gate is a negative result and stops 5k.
