# BONES-SEED Generated-History Scheduled-Sampling Gate

## Scope

This report is independent of the teacher-history protocol-aligned report. All three models start from their designated healthy checkpoints. Training uses one GT seed window followed by a scheduled mixture of accepted model generations and teacher history; the supervised window is expressed in the generated endpoint coordinate frame.

| Setting | Value |
|---|---|
| Dataset | BONES-SEED v004 G1, actor-disjoint train/val/test |
| Gate budgets | 1k; 5k only if every predefined 1k criterion passes |
| Rollout curriculum | 50% x2, 20% x3, 20% x4, 10% x5 windows (maximum 10.4 s) |
| Generated-history schedule | 0 -> 0.50 by 1k; 0.50 -> 0.75 by 5k |
| Short-window retention | 20% standard short-window updates; 80% rollout updates |
| Optimizer | AdamW, LR 5e-6, global batch 16, entire denoiser |
| Sampling during training | 10 diffusion steps, text/constraint CFG 2/2 |
| Evaluation | Table 4, official-style 9 s, and matched 20/40 s extensions |
| State audit | root/body per-layer Mamba norm, state-difference norm, KV size, fusion gate |
| Reset ablation | no reset, then persistent-state-only reset every 2/4/8 generated windows |

## Stage 1000

### Training Audit

| Method | Trainer trend flag | Eval loss | Generated-history updates | Rollout counts | Time (h) |
|---|---|---:|---:|---|---:|
| ARDY | FAIL | 0.3607 | 96 | 2w:399, 3w:167, 4w:162, 5w:86 | 0.10 |
| Minimal | FAIL | 0.3302 | 96 | 2w:399, 3w:167, 4w:162, 5w:86 | 0.19 |
| Hybrid | FAIL | 0.2955 | 96 | 2w:399, 3w:167, 4w:162, 5w:86 | 0.28 |

The trainer trend flag compares early and late losses even though the generated-history probability increases throughout training, making the late objective harder. Therefore `FAIL` here is not a numerical-health failure. The predefined gate separately verifies finite gradients, finite validation loss, output-weight norm, and matched draws.

### Table 4: Offline Text and Joint-Position Control

| Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |
|---|---:|---:|---:|---:|
| ARDY | 61.26 | 67.15 | 78.67 | 18.19 |
| Minimal | 41.41 | 48.75 | 60.23 | 17.80 |
| Hybrid | 41.67 | 45.09 | 49.81 | 16.63 |

### Table 5 and Long-Horizon Extension

| Duration | Protocol | Method | Constraint (cm) | Root ADE (cm) | Post-W5 Root slope | Post-W5 constraint slope |
|---:|---|---|---:|---:|---:|---:|
| 9 s | in | ARDY | 38.18 | 33.70 | N/A | N/A |
| 9 s | in | Minimal | 38.23 | 31.23 | N/A | N/A |
| 9 s | in | Hybrid | 36.17 | 27.89 | N/A | N/A |
| 9 s | out | ARDY | 53.34 | 38.75 | N/A | N/A |
| 9 s | out | Minimal | 61.12 | 38.83 | N/A | N/A |
| 9 s | out | Hybrid | 52.21 | 29.99 | N/A | N/A |
| 20 s | in | ARDY | 208.58 | 189.62 | 41.062 | 43.889 |
| 20 s | in | Minimal | 113.43 | 110.39 | -2.230 | -3.170 |
| 20 s | in | Hybrid | 161.43 | 148.20 | 4.255 | 1.056 |
| 20 s | out | ARDY | 1098.81 | 544.37 | 112.607 | N/A |
| 20 s | out | Minimal | 370.31 | 232.85 | 21.168 | N/A |
| 20 s | out | Hybrid | 377.49 | 255.75 | 8.184 | N/A |
| 40 s | in | ARDY | 462.54 | 443.68 | 93.331 | 96.269 |
| 40 s | in | Minimal | 126.08 | 128.74 | -0.655 | -1.413 |
| 40 s | in | Hybrid | 150.19 | 144.54 | 1.715 | 1.576 |
| 40 s | out | ARDY | 3059.10 | 1103.12 | 212.758 | N/A |
| 40 s | out | Minimal | 558.45 | 385.09 | 16.560 | N/A |
| 40 s | out | Hybrid | 618.68 | 415.29 | 28.339 | N/A |

### Hybrid State Reset Ablation (40 s, In-Horizon)

| Reset interval | Constraint (cm) | Root ADE (cm) | Root slope | Max state norm | Max state drift | Finite |
|---:|---:|---:|---:|---:|---:|---|
| none | 150.19 | 144.54 | 1.715 | 6984.300 | 3330.290 | True |
| 2 | 165.94 | 162.37 | 4.506 | 7038.993 | 3859.347 | True |
| 4 | 151.89 | 147.72 | 1.840 | 6984.300 | 3595.002 | True |
| 8 | 154.03 | 149.49 | 0.403 | 6984.300 | 3431.650 | True |

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
| 20 s | 6.485 | [-4.528, 17.856] | 0.40 |
| 40 s | 2.370 | [-1.425, 6.160] | 0.25 |

State audit: 200 commit records; maximum norm 6984.300, maximum drift 3330.290, maximum final/initial norm ratio 3.111.

### Stage Finding

Hybrid retains strong 9-second behavior, but Table 4 Root ADE changes by 11.49% and exceeds the predefined 10% retention limit. At 40 seconds, Hybrid has Root ADE 144.54 cm and slope 1.715, versus Minimal's 128.74 cm and -0.655.

Resetting persistent state every eight generated windows reduces Hybrid's 40-second Root slope to 0.403 and constraint slope to -0.526, but worsens average Root ADE to 149.49 cm. State remains finite and its maximum final/initial norm ratio is below the explosion threshold. The evidence therefore does not support simple Mamba norm explosion as the primary cause; periodic reset only partially controls late-window drift and does not repair accumulated trajectory bias.

## Interpretation Rule

A lower denoising loss is not sufficient. The experiment supports the long-horizon Hybrid claim only when Hybrid improves both 20 s and 40 s post-window-5 root/constraint slopes over Minimal, retains Table 4 and 9 s within the predefined 10% tolerance, and shows no non-finite or greater-than-10x state-norm growth. A failed 1k gate is a negative result and stops 5k.
