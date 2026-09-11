# BONES-SEED Generated-History Scheduled-Sampling Gate

## Scope

This report is independent of the teacher-history protocol-aligned report. All three models start from their designated healthy checkpoints. Training uses one GT seed window followed by a scheduled mixture of accepted model generations and teacher history; the supervised window is expressed in the generated endpoint coordinate frame.

Step 900 was selected after an exploratory 600/700/800/900/1000 checkpoint sweep. Metrics in this report use the same held-out test protocol, so an additional untouched-seed confirmation is still required before treating the selection as final evidence.

| Setting | Value |
|---|---|
| Dataset | BONES-SEED v004 G1, actor-disjoint train/val/test |
| Gate budgets | Exploratory sweep selected step 900; this run is the full confirmation audit |
| History sampling strategy | balanced_v2 |
| Rollout curriculum | Generated branch: 40% x3, 40% x4, 20% x5; teacher branch: 50% x2, 20% x3, 20% x4, 10% x5 (maximum 10.4 s) |
| Generated-history schedule | 0.25 -> 0.75 by 1k; -> 0.85 by 5k |
| Generated-depth ramp | 0 steps |
| Per-window recovery cap | 0.0 m (0 means disabled) |
| Short-window retention | 35% fixed short-window replay |
| Optimizer | AdamW, LR 5.0e-06, global batch 16, entire denoiser |
| Sampling during training | 10 diffusion steps, text/constraint CFG 2/2 |
| Evaluation | Table 4, official-style 9 s, and matched 20/40 s extensions |
| State audit | root/body per-layer Mamba norm, state-difference norm, KV size, fusion gate |
| Reset ablation | no reset, then persistent-state-only reset every 2/4/8 generated windows |

## Stage 900

### Training Audit

| Method | Trainer trend flag | Eval loss | Generated updates/windows | Teacher/short updates | Rollout counts | Time (h) |
|---|---|---:|---:|---:|---|---:|
| ARDY | FAIL | 0.6757 | 295/536 | 315/290 | 2w:152, 3w:168, 4w:197, 5w:93 | 0.10 |
| Minimal | FAIL | 0.3787 | 295/536 | 315/290 | 2w:152, 3w:168, 4w:197, 5w:93 | 0.22 |
| Hybrid | FAIL | 0.3111 | 295/536 | 315/290 | 2w:152, 3w:168, 4w:197, 5w:93 | 0.37 |

The trainer trend flag compares early and late losses even though the generated-history probability increases throughout training, making the late objective harder. Therefore `FAIL` here is not a numerical-health failure. The predefined gate separately verifies finite gradients, finite validation loss, output-weight norm, and matched draws.

### Table 4: Offline Text and Joint-Position Control

| Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) |
|---|---:|---:|---:|---:|
| ARDY | 51.62 | 56.48 | 69.17 | 17.34 |
| Minimal | 54.16 | 63.27 | 77.58 | 19.70 |
| Hybrid | 40.40 | 43.27 | 50.53 | 17.79 |

### Table 5 and Long-Horizon Extension

| Duration | Protocol | Method | Constraint (cm) | Root ADE (cm) | Post-W5 Root slope | Post-W5 constraint slope |
|---:|---|---|---:|---:|---:|---:|
| 9 s | in | ARDY | 60.60 | 57.66 | N/A | N/A |
| 9 s | in | Minimal | 45.27 | 40.19 | N/A | N/A |
| 9 s | in | Hybrid | 36.45 | 28.60 | N/A | N/A |
| 9 s | out | ARDY | 75.33 | 57.82 | N/A | N/A |
| 9 s | out | Minimal | 67.48 | 51.01 | N/A | N/A |
| 9 s | out | Hybrid | 51.03 | 30.91 | N/A | N/A |
| 20 s | in | ARDY | 295.12 | 276.77 | 26.761 | 23.286 |
| 20 s | in | Minimal | 193.92 | 194.07 | -3.390 | -0.463 |
| 20 s | in | Hybrid | 137.04 | 130.11 | -4.784 | -6.626 |
| 20 s | out | ARDY | 640.07 | 427.76 | 38.672 | N/A |
| 20 s | out | Minimal | 573.38 | 364.12 | 28.279 | N/A |
| 20 s | out | Hybrid | 478.77 | 328.73 | 10.088 | N/A |
| 40 s | in | ARDY | 620.63 | 602.21 | 127.569 | 129.445 |
| 40 s | in | Minimal | 164.45 | 166.92 | 8.040 | 7.977 |
| 40 s | in | Hybrid | 137.47 | 134.89 | -1.238 | -1.311 |
| 40 s | out | ARDY | 3220.55 | 1100.50 | 218.248 | N/A |
| 40 s | out | Minimal | 800.26 | 525.89 | 34.055 | N/A |
| 40 s | out | Hybrid | 756.11 | 532.39 | 31.463 | N/A |

### Hybrid State Reset Ablation (40 s, In-Horizon)

| Reset interval | Constraint (cm) | Root ADE (cm) | Root slope | Max state norm | Max state drift | Finite |
|---:|---:|---:|---:|---:|---:|---|
| none | 137.47 | 134.89 | -1.238 | 6748.190 | 3770.729 | True |
| 2 | 148.93 | 146.57 | 3.110 | 6836.959 | 3906.900 | True |
| 4 | 143.93 | 139.89 | 2.395 | 6841.141 | 4002.742 | True |
| 8 | 149.24 | 144.60 | 0.186 | 6748.190 | 3810.913 | True |

### Predefined Gate Decision

**PASS**. Continue to 5k: **False**.

| Criterion | Passed |
|---|---|
| training healthy | True |
| matched draws | True |
| short retention | True |
| hybrid long slopes better than minimal | True |
| state finite | True |
| no state norm explosion | True |

| Duration | Hybrid-Minimal root-slope gap | Paired 95% CI | Hybrid better fraction |
|---:|---:|---:|---:|
| 20 s | -1.394 | [-24.750, 20.409] | 0.40 |
| 40 s | -9.279 | [-17.593, -2.402] | 0.75 |

State audit: 200 commit records; maximum norm 6748.190, maximum drift 3770.729, maximum final/initial norm ratio 2.970.

### Stage Finding

Short-horizon retention passes every predefined 10% check. At 40 seconds, Hybrid has Root ADE 134.89 cm and slope -1.238, versus Minimal's 166.92 cm and 8.040.

Resetting persistent state every eight generated windows changes Hybrid's 40-second Root slope to 0.186, constraint slope to 0.623, and average Root ADE to 144.60 cm. State remains finite and its maximum final/initial norm ratio is below the explosion threshold. The evidence therefore does not support simple Mamba norm explosion as the primary cause; periodic reset only partially controls late-window drift and does not repair accumulated trajectory bias.

## Interpretation Rule

A lower denoising loss is not sufficient. The experiment supports the long-horizon Hybrid claim only when Hybrid improves both 20 s and 40 s post-window-5 root/constraint slopes over Minimal, retains Table 4 and 9 s within the predefined 10% tolerance, and shows no non-finite or greater-than-10x state-norm growth. A failed 1k gate is a negative result and stops 5k.
