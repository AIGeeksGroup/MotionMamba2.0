# BONES-SEED Generated-History Scheduled-Sampling Gate

## Scope

This report is independent of the teacher-history protocol-aligned report. All three models start from their designated healthy checkpoints. Training uses one GT seed window followed by a scheduled mixture of accepted model generations and teacher history; the supervised window is expressed in the generated endpoint coordinate frame.

| Setting | Value |
|---|---|
| Dataset | BONES-SEED v004 G1, actor-disjoint train/val/test |
| Gate budgets | 1k; 5k only if every predefined 1k criterion passes |
| History sampling strategy | deep_v3 |
| Rollout curriculum | Generated branch: 40% x3, 30% x5, 20% x10, 10% x19; teacher branch: 50% x2, 20% x3, 20% x4, 10% x5 (maximum 39.5 s) |
| Generated-history schedule | 0.15 -> 0.55 by 1k; -> 0.75 by 5k |
| Generated-depth ramp | 500 steps |
| Per-window recovery cap | 4.0 m (0 means disabled) |
| Short-window retention | 50% fixed short-window replay |
| Optimizer | AdamW, LR 3.0e-06, global batch 16, entire denoiser |
| Sampling during training | 10 diffusion steps, text/constraint CFG 2/2 |
| Evaluation | Table 4, official-style 9 s, and matched 20/40 s extensions |
| State audit | root/body per-layer Mamba norm, state-difference norm, KV size, fusion gate |
| Reset ablation | no reset, then persistent-state-only reset every 2/4/8 generated windows |

## Stage 1000

### Training Audit

| Method | Trainer trend flag | Eval loss | Generated updates/windows | Teacher/short updates | Rollout counts | Time (h) |
|---|---|---:|---:|---:|---|---:|
| ARDY | FAIL | 0.3945 | 178/679 | 297/525 | 2w:143, 3w:137, 4w:70, 5w:79, 10w:37, 19w:9 | 0.12 |
| Minimal | FAIL | 0.5731 | 178/679 | 297/525 | 2w:143, 3w:137, 4w:70, 5w:79, 10w:37, 19w:9 | 0.27 |
| Hybrid | FAIL | 0.5749 | 178/679 | 297/525 | 2w:143, 3w:137, 4w:70, 5w:79, 10w:37, 19w:9 | 0.43 |

The trainer trend flag compares early and late losses even though the generated-history probability increases throughout training, making the late objective harder. Therefore `FAIL` here is not a numerical-health failure. The predefined gate separately verifies finite gradients, finite validation loss, output-weight norm, and matched draws.

### Table 4: Offline Text and Joint-Position Control

| Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |
|---|---:|---:|---:|---:|
| ARDY | 50.89 | 55.01 | 68.00 | 17.46 |
| Minimal | 43.42 | 50.71 | 61.41 | 18.24 |
| Hybrid | 44.32 | 46.87 | 52.60 | 16.97 |

### Table 5 and Long-Horizon Extension

| Duration | Protocol | Method | Constraint (cm) | Root ADE (cm) | Post-W5 Root slope | Post-W5 constraint slope |
|---:|---|---|---:|---:|---:|---:|
| 9 s | in | ARDY | 37.03 | 32.24 | N/A | N/A |
| 9 s | in | Minimal | 35.82 | 28.57 | N/A | N/A |
| 9 s | in | Hybrid | 36.17 | 27.29 | N/A | N/A |
| 9 s | out | ARDY | 49.33 | 36.38 | N/A | N/A |
| 9 s | out | Minimal | 62.05 | 37.20 | N/A | N/A |
| 9 s | out | Hybrid | 53.26 | 29.93 | N/A | N/A |
| 20 s | in | ARDY | 177.28 | 175.54 | 20.032 | 20.051 |
| 20 s | in | Minimal | 121.18 | 117.60 | -2.302 | -2.058 |
| 20 s | in | Hybrid | 157.82 | 147.20 | -2.853 | -6.116 |
| 20 s | out | ARDY | 306.36 | 281.08 | -6.533 | N/A |
| 20 s | out | Minimal | 377.16 | 246.47 | 12.696 | N/A |
| 20 s | out | Hybrid | 424.09 | 272.05 | 15.804 | N/A |
| 40 s | in | ARDY | 186.60 | 193.66 | 2.932 | 4.625 |
| 40 s | in | Minimal | 122.46 | 123.94 | -1.061 | -1.118 |
| 40 s | in | Hybrid | 166.11 | 160.15 | 2.381 | 2.224 |
| 40 s | out | ARDY | 378.27 | 324.91 | 1.724 | N/A |
| 40 s | out | Minimal | 514.64 | 400.62 | 12.323 | N/A |
| 40 s | out | Hybrid | 730.55 | 469.41 | 33.731 | N/A |

### Hybrid State Reset Ablation (40 s, In-Horizon)

| Reset interval | Constraint (cm) | Root ADE (cm) | Root slope | Max state norm | Max state drift | Finite |
|---:|---:|---:|---:|---:|---:|---|
| none | 166.11 | 160.15 | 2.381 | 7005.188 | 3563.691 | True |
| 2 | 161.74 | 156.40 | 3.808 | 7216.118 | 3888.292 | True |
| 4 | 150.93 | 146.16 | 1.565 | 7221.135 | 3867.821 | True |
| 8 | 159.77 | 154.27 | 0.322 | 7093.261 | 3674.716 | True |

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
| 20 s | -0.551 | [-16.486, 14.992] | 0.30 |
| 40 s | 3.443 | [-0.203, 7.492] | 0.35 |

State audit: 200 commit records; maximum norm 7005.188, maximum drift 3563.691, maximum final/initial norm ratio 3.172.

### Stage Finding

Short-horizon retention fails the predefined 10% limit for table4 constraint (+11.40%), table4 Root ADE (+15.88%). At 40 seconds, Hybrid has Root ADE 160.15 cm and slope 2.381, versus Minimal's 123.94 cm and -1.061.

Resetting persistent state every eight generated windows improves Hybrid's 40-second Root slope to 0.322 and constraint slope to -0.066, but worsens average Root ADE to 154.27 cm. State remains finite and its maximum final/initial norm ratio is below the explosion threshold. The evidence therefore does not support simple Mamba norm explosion as the primary cause; periodic reset only partially controls late-window drift and does not repair accumulated trajectory bias.

## Interpretation Rule

A lower denoising loss is not sufficient. The experiment supports the long-horizon Hybrid claim only when Hybrid improves both 20 s and 40 s post-window-5 root/constraint slopes over Minimal, retains Table 4 and 9 s within the predefined 10% tolerance, and shows no non-finite or greater-than-10x state-norm growth. A failed 1k gate is a negative result and stops 5k.
