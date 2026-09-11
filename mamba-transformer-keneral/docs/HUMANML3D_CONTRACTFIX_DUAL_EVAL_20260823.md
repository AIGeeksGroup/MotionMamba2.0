# HumanML3D Contract-Fix Table 4/5 Evaluation

## Status

This report evaluates three fresh 50k checkpoints under the 
`canonical_history_v2` contract. Table 4 retains the fixed 40-frame offline panel. 
Table 5 has been replaced by the ARDY Section 6.3 generation protocol.

| Method | Checkpoint | Training status |
|---|---|---|
| ARDY fresh mixer | `/data4/zts/result/humanml3d_contractfix_full50k_20260823_original/original_humanml3d_ddp.pt` | PASS, 50k, finite gradients, 4.01 h |
| Minimal contract-fix | `/data4/zts/result/humanml3d_contractfix_full50k_20260823_minimal/minimal_humanml3d_ddp.pt` | PASS, 50k, finite gradients, 6.28 h |
| Hybrid contract-fix | `/data4/zts/result/humanml3d_contractfix_full50k_20260823_hybrid/hybrid_humanml3d_ddp.pt` | PASS, 50k, finite gradients, 8.11 h |

All nine jobs completed with `status=PASS`. Each protocol uses 64 fixed samples, 
batch size 4, 10 diffusion steps, CFG text/constraint weights 2/2, and identical 
sample IDs across the three models. Table 5 uses a nine-second sequence at 20 fps: 
20 GT history frames followed by 160 generated frames in four 40-frame windows.

| Protocol | Fixed panel SHA-256 |
|---|---|
| Table 4 | `af83307a719efe58762f310176c4b707333944c94fc21d458f399b7a59032c67` |
| Table 5 in | `f04db95e452b08af06e55744742206e8bfb0989c544c5dfdea1ad2598d31e3be` |
| Table 5 out | `f04db95e452b08af06e55744742206e8bfb0989c544c5dfdea1ad2598d31e3be` |

For Table 5, Minimal and Hybrid prime the GT prefix exactly once and then use 
persistent cache-only rollout. ARDY uses native explicit autoregressive history.

## Official HumanML3D Track

Official R-Precision and FID remain unavailable because generated output is 
CoreSkeleton27/330-D while the official evaluator requires HumanML3D 22-joint/263-D. 
The prior conversion audit found round-trip FID bias `14.9845` and paired embedding 
cosine mean `0.8807`, so learned scores are not approximated.

| Method | R-Precision Top-3 | FID | FVMD | TM-Sim |
|---|---:|---:|---:|---:|
| ARDY fresh mixer | N/A | N/A | N/A | N/A |
| Minimal contract-fix | N/A | N/A | N/A | N/A |
| Hybrid contract-fix | N/A | N/A | N/A | N/A |

## Table 4 proxy: Offline Text and Joint-Position Control

This protocol uses the fixed 40-frame in-horizon panel. Lower is better for every metric.

| Method | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY fresh mixer (50k) | 57.07 | 58.76 | 78.89 | **25.52** | 0.03612 | **23.38** | 34.09 | **0.0279** |
| Minimal contract-fix (50k) | 66.52 | 73.76 | 84.68 | 26.38 | 0.03422 | 26.20 | 41.95 | 0.1160 |
| Hybrid contract-fix (50k) | **48.60** | **52.44** | **62.25** | 26.78 | **0.03258** | 24.40 | **30.72** | 0.3416 |

## Table 5: ARDY-Paper-Aligned 9-Second Autoregressive Control

In-horizon places one sampled pelvis/wrist/foot position goal at the end of each 
two-second generation window. Out-of-horizon places one goal only at the final frame. 
Metrics exclude the one-second GT prefix and cover the generated eight seconds.

| Method | Protocol | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY fresh mixer (50k) | in-horizon | **35.09** | **26.02** | **33.60** | **29.38** | **0.02887** | **23.72** | 34.66 | **0.1051** |
| Minimal contract-fix (50k) | in-horizon | 46.47 | 39.27 | 50.23 | 33.61 | 0.03428 | 25.99 | 35.05 | 0.5492 |
| Hybrid contract-fix (50k) | in-horizon | 43.07 | 34.71 | 46.01 | 33.57 | 0.02889 | 23.75 | **32.29** | 1.2569 |
| ARDY fresh mixer (50k) | out-of-horizon | **57.12** | 68.59 | **49.73** | 44.69 | 0.03023 | 24.47 | 37.29 | **0.1046** |
| Minimal contract-fix (50k) | out-of-horizon | 67.61 | 66.49 | 63.83 | 43.19 | 0.03225 | 26.58 | 35.37 | 0.5428 |
| Hybrid contract-fix (50k) | out-of-horizon | 68.56 | **55.90** | 64.38 | **42.47** | **0.02765** | **24.02** | **33.38** | 1.1828 |

## 20/40-Second Supplement Status

The HumanML3D test manifest contains 0 motions of at least 20 seconds and 
0 motions of at least 40 seconds. A paired-reference 20/40-second experiment 
cannot therefore be run on HumanML3D without looping or stitching clips and fabricating 
ground truth. Those synthetic variants are intentionally not reported. The strict 20/40-second 
supplement is run on the same 20 naturally long BONES-SEED test motions and is documented in 
`docs/BONES_SEED_CONTRACTFIX_DUAL_EVAL_20260824.md`.

The nine-second HumanML3D protocol contains four generation windows, equal to the configured 
four-chunk attention-cache capacity. It validates four-window rollout but does not isolate the 
Hybrid Mamba state after attention-cache eviction.

## Boundary and Interpretation

| Method | Table 5 in max boundary speed (m/s) | Table 5 out max boundary speed (m/s) | Eval peak memory (MiB, max) | Active mixer parameters |
|---|---:|---:|---:|---:|
| ARDY fresh mixer (50k) | 5.12 | 4.11 | 1534.5 | 147,150,996 |
| Minimal contract-fix (50k) | 9.13 | 8.24 | 2303.1 | 247,914,644 |
| Hybrid contract-fix (50k) | 4.61 | 7.39 | 2982.6 | 352,578,724 |

ARDY fresh mixer has the lowest constraint error in both Table 5 protocols on these fixed panels. Bold values are per-protocol 
minima. These geometry and physical proxies do not constitute official leaderboard scores.
Hybrid does not win goal error on HumanML3D at nine seconds, but it has the lowest 
out-of-horizon Root ADE, PA-MPJPE, jitter, Angle MAE, and skate proxy, plus the lowest 
in-horizon skate proxy. The evidence therefore supports better long-rollout continuity 
and physical proxies, not a blanket long-horizon control-accuracy win.

Latency jobs ran concurrently on separate GPUs and support within-run comparison only.

## Provenance

- ARDY paper protocol: Section 6.3 / Table 5, DOI `10.1145/3811284`
- Table 4 evaluator: `examples/evaluate_humanml3d_table45.py`
- Table 5 evaluator: `examples/evaluate_ardy_official_table5.py`
- Table 5 launcher: `examples/run_ardy_official_table5_dual_dataset_20260825.sh`
- Table 4 artifacts: `/data4/zts/result/humanml3d_contractfix_dual_eval_20260823/`
- Table 5 artifacts: `/data4/zts/result/humanml3d_contractfix_official_table5_20260825/`
