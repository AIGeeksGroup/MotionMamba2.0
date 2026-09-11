# BONES-SEED Protocol-Aligned Long-Horizon Evaluation

## Status and Experimental Role

This is the primary three-model long-horizon fine-tune track. ARDY, Minimal, and 
Hybrid start from their healthy matched 50k checkpoints and receive the same 500-step 
gate followed by the same 1000-step total confirmation. The 1000-step checkpoint is 
pre-designated as the main result; the 500-step result is retained to test stability 
rather than to select the better-looking checkpoint.

The earlier Hybrid-only 500-step recurrent-path run is reported only as a training-strategy 
ablation. It is not mixed into any three-model main table.

## Unified Fine-Tune Protocol

| Setting | Value |
|---|---|
| Dataset/split | BONES-SEED v004 G1, actor-disjoint train/val/test |
| Initialization | Each architecture's healthy matched 50k checkpoint |
| Train scope | Entire denoiser for all three models |
| Optimizer budget | 500-step gate, then resume to 1000 total steps |
| Learning rate | `5e-6` |
| Global batch | 32 (8 GPUs x 4 samples) |
| Supervision | Teacher history; loss on the final 52-frame window |
| Curriculum | 50% x2 windows; 20% x4; 15% uniform x6-10; 15% uniform x12-19 |
| ARDY history | Native explicit history |
| Minimal/Hybrid history | Canonical persistent cache/state commits |
| Control/dropout | Joint-position control; text/control dropout 0.1/0.1 |
| Loss weights | boundary 10, root velocity 50, constraint 1, FK 1, foot skate 50 |

The data, sampled motion IDs, augmentation seeds, rollout-depth draws, update count, 
batch size, learning rate, losses, and evaluation panels are matched. The internal history 
mechanism is necessarily architecture-native: changing ARDY into a persistent-state model 
would erase the architectural comparison.

## Training Audit

| Stage | Method | Eval loss | Root output norm | Trainable params | Time (min) | Rollout counts |
|---:|---|---:|---:|---:|---:|---|
| 500 | ARDY | 0.8967 | 3.4613 | 157,887,636 | 3.0 | 2w:246, 4w:99, 6w:13, 7w:18, 8w:18, 9w:16, 10w:14, 12w:14, 13w:12, 14w:10, 15w:9, 16w:7, 17w:7, 18w:9, 19w:8 |
| 500 | Minimal | 0.3946 | 4.6285 | 258,651,284 | 5.9 | 2w:246, 4w:99, 6w:13, 7w:18, 8w:18, 9w:16, 10w:14, 12w:14, 13w:12, 14w:10, 15w:9, 16w:7, 17w:7, 18w:9, 19w:8 |
| 500 | Hybrid | 0.3929 | 4.2304 | 363,315,364 | 9.6 | 2w:246, 4w:99, 6w:13, 7w:18, 8w:18, 9w:16, 10w:14, 12w:14, 13w:12, 14w:10, 15w:9, 16w:7, 17w:7, 18w:9, 19w:8 |
| 1000 | ARDY | 0.8594 | 3.4612 | 157,887,636 | 6.0 | 2w:483, 4w:203, 6w:31, 7w:38, 8w:35, 9w:28, 10w:29, 12w:22, 13w:23, 14w:14, 15w:19, 16w:13, 17w:22, 18w:20, 19w:20 |
| 1000 | Minimal | 0.3713 | 4.6302 | 258,651,284 | 11.8 | 2w:483, 4w:203, 6w:31, 7w:38, 8w:35, 9w:28, 10w:29, 12w:22, 13w:23, 14w:14, 15w:19, 16w:13, 17w:22, 18w:20, 19w:20 |
| 1000 | Hybrid | 0.3651 | 4.2306 | 363,315,364 | 18.9 | 2w:483, 4w:203, 6w:31, 7w:38, 8w:35, 9w:28, 10w:29, 12w:22, 13w:23, 14w:14, 15w:19, 16w:13, 17w:22, 18w:20, 19w:20 |

## Stage 500: Table 4 Offline Text and Joint-Position Control

| Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle MAE (deg) | Skate (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARDY | 52.35 | 54.25 | 66.46 | 17.24 | **0.02180** | **14.88** | 41.65 |
| Minimal | 43.87 | 50.20 | 61.52 | 17.44 | 0.02356 | 15.17 | 38.35 |
| Hybrid | **36.76** | **38.71** | **45.12** | **16.28** | 0.02429 | 15.57 | **38.29** |

## Stage 500: ARDY-Aligned 9-Second Table 5

| Method | Protocol | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle MAE (deg) | Skate (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ARDY | in-horizon | **24.05** | **17.07** | **24.45** | **15.19** | **0.01735** | **13.49** | 37.52 |
| Minimal | in-horizon | 39.48 | 31.48 | 49.31 | 19.04 | 0.02065 | 14.04 | 33.61 |
| Hybrid | in-horizon | 32.44 | 24.70 | 43.13 | 15.75 | 0.01853 | 13.49 | **31.73** |
| ARDY | out-of-horizon | **37.98** | **25.63** | **36.00** | 17.89 | **0.01612** | **13.47** | 33.13 |
| Minimal | out-of-horizon | 66.58 | 39.00 | 64.10 | 22.85 | 0.01801 | 13.63 | 31.35 |
| Hybrid | out-of-horizon | 47.80 | 27.54 | 44.90 | **17.04** | 0.01704 | 13.65 | **29.08** |

## Stage 1000: Table 4 Offline Text and Joint-Position Control

| Method | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle MAE (deg) | Skate (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARDY | 48.22 | 49.68 | 60.55 | 16.86 | **0.02259** | 15.29 | 42.52 |
| Minimal | 43.35 | 49.97 | 61.88 | 17.50 | 0.02312 | **15.00** | **37.17** |
| Hybrid | **36.83** | **38.92** | **46.16** | **16.53** | 0.02382 | 15.55 | 37.79 |

## Stage 1000: ARDY-Aligned 9-Second Table 5

| Method | Protocol | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter | Angle MAE (deg) | Skate (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ARDY | in-horizon | **28.84** | **20.61** | **31.44** | 16.90 | **0.01692** | 14.09 | 37.41 |
| Minimal | in-horizon | 39.22 | 30.98 | 48.26 | 19.15 | 0.02095 | 14.31 | 34.92 |
| Hybrid | in-horizon | 33.55 | 25.78 | 43.67 | **16.07** | 0.01923 | **13.83** | **32.28** |
| ARDY | out-of-horizon | **44.50** | **27.09** | **41.56** | 18.85 | **0.01605** | 14.10 | 33.62 |
| Minimal | out-of-horizon | 64.58 | 38.95 | 62.63 | 22.68 | 0.01803 | **13.98** | 32.25 |
| Hybrid | out-of-horizon | 51.03 | 29.35 | 48.09 | **17.32** | 0.01798 | 14.07 | **30.62** |

## 20/40-Second Supplement

All rows use the same 20 naturally long test motions. Slopes start after window 5; lower 
is better. The 20/40-second evaluations are duration extensions, not official ARDY rows.

| Stage | Duration | Method | Protocol | Constraint (cm) | Root ADE (cm) | Root FDE (cm) | Post-W5 Root slope | Post-W5 constraint slope |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 500 | 20 s | ARDY | in | 81.54 | 75.27 | 86.66 | -0.075 | -0.575 |
| 500 | 20 s | Minimal | in | 127.61 | 121.37 | 144.92 | -2.188 | -2.250 |
| 500 | 20 s | Hybrid | in | 208.58 | 187.66 | 276.75 | 11.173 | 7.496 |
| 500 | 20 s | ARDY | out | 292.32 | 222.79 | 283.80 | -5.589 | N/A |
| 500 | 20 s | Minimal | out | 338.62 | 228.32 | 332.88 | 8.293 | N/A |
| 500 | 20 s | Hybrid | out | 436.99 | 279.43 | 431.92 | 13.975 | N/A |
| 500 | 40 s | ARDY | in | 98.41 | 94.63 | 114.03 | 1.922 | 2.055 |
| 500 | 40 s | Minimal | in | 151.39 | 152.71 | 147.26 | 2.163 | 1.782 |
| 500 | 40 s | Hybrid | in | 222.22 | 214.12 | 280.96 | 4.825 | 4.463 |
| 500 | 40 s | ARDY | out | 525.19 | 320.71 | 522.31 | 21.371 | N/A |
| 500 | 40 s | Minimal | out | 661.29 | 431.70 | 663.14 | 22.268 | N/A |
| 500 | 40 s | Hybrid | out | 737.47 | 484.43 | 737.20 | 31.480 | N/A |
| 1000 | 20 s | ARDY | in | 91.28 | 80.71 | 109.22 | 5.192 | 6.516 |
| 1000 | 20 s | Minimal | in | 134.98 | 128.15 | 153.03 | -3.399 | -1.376 |
| 1000 | 20 s | Hybrid | in | 227.21 | 203.97 | 305.50 | 14.251 | 12.513 |
| 1000 | 20 s | ARDY | out | 281.66 | 220.20 | 274.49 | -3.563 | N/A |
| 1000 | 20 s | Minimal | out | 410.83 | 268.35 | 407.05 | 12.441 | N/A |
| 1000 | 20 s | Hybrid | out | 516.22 | 326.36 | 509.41 | 18.584 | N/A |
| 1000 | 40 s | ARDY | in | 106.86 | 99.71 | 174.08 | 4.619 | 5.329 |
| 1000 | 40 s | Minimal | in | 164.08 | 163.51 | 192.50 | 4.492 | 4.506 |
| 1000 | 40 s | Hybrid | in | 284.24 | 274.67 | 360.03 | 10.332 | 9.379 |
| 1000 | 40 s | ARDY | out | 355.65 | 275.38 | 355.31 | 4.192 | N/A |
| 1000 | 40 s | Minimal | out | 767.67 | 486.54 | 770.10 | 28.555 | N/A |
| 1000 | 40 s | Hybrid | out | 842.01 | 545.88 | 843.20 | 38.891 | N/A |

## Hybrid vs Minimal Stability

Negative deltas favor Hybrid. The gap slope is computed per paired sample from window 6 
onward and bootstrapped with 20,000 paired resamples.

| Stage | Duration | Protocol | Hybrid-Minimal constraint (cm) | Hybrid-Minimal Root ADE (cm) | Post-W5 Root-gap slope (cm/window) | 95% CI | Final-window Root gap (cm) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 500 | 20 s | in | 80.97 | 66.29 | 13.361 | [1.610, 25.916] | 132.04 |
| 500 | 20 s | out | 98.38 | 51.11 | N/A | N/A | N/A |
| 500 | 40 s | in | 70.83 | 61.41 | 2.662 | [-6.409, 10.162] | 118.80 |
| 500 | 40 s | out | 76.18 | 52.72 | N/A | N/A | N/A |
| 1000 | 20 s | in | 92.23 | 75.82 | 17.650 | [4.079, 33.649] | 153.94 |
| 1000 | 20 s | out | 105.39 | 58.01 | N/A | N/A | N/A |
| 1000 | 40 s | in | 120.17 | 111.16 | 5.841 | [-6.956, 20.865] | 166.76 |
| 1000 | 40 s | out | 74.34 | 59.34 | N/A | N/A | N/A |

## Predefined Stability Check

- At 500 steps, Hybrid beats Minimal in 4/12 matched duration/protocol comparisons across constraint error and Root ADE.
- At 1000 steps, Hybrid beats Minimal in 4/12 matched duration/protocol comparisons across constraint error and Root ADE.
- The sign pattern is stable between the 500-step gate and 1000-step confirmation.
- At 500 steps, Hybrid wins 4/4 9-second comparisons but only 0/8 comparisons at 20/40 seconds.
- At 1000 steps, Hybrid wins 4/4 9-second comparisons but only 0/8 comparisons at 20/40 seconds.

## Main Finding

The confirmation is stable but negative for the intended long-horizon Hybrid claim. 
Hybrid remains better than Minimal at nine seconds, but loses every matched constraint-error 
and Root-ADE comparison at 20 and 40 seconds at both training stages. ARDY is the strongest 
model on those long-duration control metrics.

Continuing Hybrid from 500 to 1000 steps increases 20-second in-horizon constraint error by 18.63 cm and Root ADE by 16.32 cm. 
At 40 seconds the increases are 62.02 cm and 60.55 cm. 
The falling teacher-history validation loss therefore does not predict generated-history 
rollout quality for Hybrid.

These results do not justify a 50k fresh run with the same teacher-history objective. The 
next defensible training experiment is a matched generated-history/scheduled-sampling 
objective, with short-window retention, before spending the full pretraining budget.

## Training-Strategy Ablation: Hybrid-Only Recurrent Fine-Tune

This earlier run updates only Hybrid's Mamba path and fusion gates (104,664,080 parameters) 
for 500 steps with a 50% x2 / 50% x6-10 curriculum. ARDY and Minimal did not receive 
the same adaptation, so these values are an ablation and are excluded from every main table.

| 40 s in-horizon metric | Old Hybrid checkpoint | Hybrid-only recurrent FT |
|---|---:|---:|
| Constraint error (cm) | 166.89 | 155.05 |
| Root ADE (cm) | 160.81 | 148.93 |
| Hybrid-Minimal post-W5 Root-gap slope | 4.899 | 4.393 |
| Final-window Hybrid-Minimal Root gap (cm) | 62.28 | 41.43 |

## Interpretation Boundary

The unified fine-tune uses teacher-forced history and truncated state commits. It evaluates 
protocol-aligned adaptation but does not remove the generated-history exposure gap. A fresh 
50k long-rollout pretraining run remains a separate, more expensive experiment and should only 
be launched if the 500/1000-step conclusions are sufficiently stable.

R-Precision, FID, FVMD, and TM-Sim remain `N/A` because BONES-SEED has no locally calibrated 
official learned evaluator. Geometry and physical values are fixed-panel proxies.

## Provenance

- Trainer: `examples/train_humanml3d_ardy_ddp.py`
- Training launcher: `examples/run_bones_seed_protocol_aligned_ft_20260826.sh`
- Evaluation launcher: `examples/run_bones_seed_protocol_aligned_eval_20260826.sh`
- Training artifacts: `/data4/zts/result/bones_seed_protocol_aligned_ft_20260826/`
- Stage-500 evaluation: `/data4/zts/result/bones_seed_protocol_aligned_eval_stage500_20260826/`
- Stage-1000 evaluation: `/data4/zts/result/bones_seed_protocol_aligned_eval_stage1000_20260826/`
- Hybrid-only ablation: `/data4/zts/result/bones_seed_hybrid_multiwindow_eval_20260826/`
- Evaluation seed: `20260824`

## Panel Hashes

| Stage | Protocol | SHA-256 |
|---:|---|---|
| 500 | table4 | `fe9d8c63b8c7f8cc45e0d4f43874cd065b29ca462a2dc119718adad3f2e0e8bd` |
| 500 | 9s_in_horizon | `7c7c980fc18fa80d080d9edf5fab554a22497be7fc43f272ae6bcea804da505e` |
| 500 | 9s_out_of_horizon | `7c7c980fc18fa80d080d9edf5fab554a22497be7fc43f272ae6bcea804da505e` |
| 500 | 20s_in_horizon | `86b0f2f94f2b6317a6b2e53d42dffe588b5857c55921d88e8120c08e561c6db2` |
| 500 | 40s_in_horizon | `86b0f2f94f2b6317a6b2e53d42dffe588b5857c55921d88e8120c08e561c6db2` |
| 1000 | table4 | `fe9d8c63b8c7f8cc45e0d4f43874cd065b29ca462a2dc119718adad3f2e0e8bd` |
| 1000 | 9s_in_horizon | `7c7c980fc18fa80d080d9edf5fab554a22497be7fc43f272ae6bcea804da505e` |
| 1000 | 9s_out_of_horizon | `7c7c980fc18fa80d080d9edf5fab554a22497be7fc43f272ae6bcea804da505e` |
| 1000 | 20s_in_horizon | `86b0f2f94f2b6317a6b2e53d42dffe588b5857c55921d88e8120c08e561c6db2` |
| 1000 | 40s_in_horizon | `86b0f2f94f2b6317a6b2e53d42dffe588b5857c55921d88e8120c08e561c6db2` |
