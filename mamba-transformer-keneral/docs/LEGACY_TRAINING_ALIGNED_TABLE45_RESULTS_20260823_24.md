# Legacy Training-Aligned Table 4/5 Results

## Archive Status

This document restores the short-horizon Table 4/5 results produced on August
23-26, 2026. It is an additive archive: the current HumanML3D and BONES-SEED
reports are not changed. The invalid original BONES-SEED ARDY row is replaced by
the stabilized 50k retrain evaluated under the same legacy protocol and panel.

These results use the matched HumanML3D 50k checkpoints, the BONES-SEED
Minimal/Hybrid 50k checkpoints, and the stabilized BONES-SEED ARDY 50k
checkpoint with the original evaluator
`examples/evaluate_humanml3d_table45.py`. Table 4 contains one generation
window. Table 5 contains exactly two autoregressive generation windows, matching
the window-count regime for which these checkpoints were trained. It does not use
the later ARDY Section 6.3 nine-second protocol or the 20/40-second supplement.

All HumanML3D rows and the BONES-SEED Minimal/Hybrid rows are restored directly
from the preserved JSON artifacts. The BONES-SEED ARDY rows are direct outputs
of a protocol-identical reevaluation of the stabilized checkpoint. No values are
copied from the later nine-second or 20/40-second evaluations. Lower is better
for every reported metric.

## Checkpoints

| Dataset | Method | Checkpoint |
|---|---|---|
| HumanML3D | ARDY fresh mixer | `/data4/zts/result/humanml3d_contractfix_full50k_20260823_original/original_humanml3d_ddp.pt` |
| HumanML3D | Minimal contract-fix | `/data4/zts/result/humanml3d_contractfix_full50k_20260823_minimal/minimal_humanml3d_ddp.pt` |
| HumanML3D | Hybrid contract-fix | `/data4/zts/result/humanml3d_contractfix_full50k_20260823_hybrid/hybrid_humanml3d_ddp.pt` |
| BONES-SEED | ARDY stabilized retrain | `/data4/zts/result/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt` |
| BONES-SEED | Minimal contract-fix | `/data4/zts/result/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt` |
| BONES-SEED | Hybrid contract-fix | `/data4/zts/result/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt` |

## Original Protocol

| Dataset | FPS | Table 4 | Table 5 | Samples/protocol | Diffusion steps | CFG text/control |
|---|---:|---:|---:|---:|---:|---:|
| HumanML3D | 20 | 40 frames (1 window, 2.00 s) | 80 frames (2 windows, 4.00 s) | 64 | 10 | 2/2 |
| BONES-SEED v004 G1 | 25 | 52 frames (1 window, 2.08 s) | 104 frames (2 windows, 4.16 s) | 64 | 10 | 2/2 |

Minimal and Hybrid use persistent cache-only rollout
(`history_frames=0`, `protocol_guard=paper_cache_only`). ARDY uses its native
explicit autoregressive history (`protocol_guard=native_explicit_history`). Fixed
sample IDs are identical across all three models within each protocol. In-horizon
and out-of-horizon Table 5 also use the same sample IDs.

| Dataset | Protocol | Fixed panel SHA-256 |
|---|---|---|
| HumanML3D | Table 4 | `af83307a719efe58762f310176c4b707333944c94fc21d458f399b7a59032c67` |
| HumanML3D | Table 5 in/out | `c7b829d056c01bfc2b6a84a9aa06862fabe35a3fe2deb56912534eb587c14b96` |
| BONES-SEED | Table 4 | `fe9d8c63b8c7f8cc45e0d4f43874cd065b29ca462a2dc119718adad3f2e0e8bd` |
| BONES-SEED | Table 5 in/out | `4865fd4ac443e6f4d861b63bf5ebac375adb88f6a80a581eb4da49d1f097946b` |

## HumanML3D Table 4: Offline Text and Joint-Position Control

| Method | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY fresh mixer (50k) | 57.07 | 58.76 | 78.89 | **25.52** | 0.03612 | **23.38** | 34.09 | **0.0279** |
| Minimal contract-fix (50k) | 66.52 | 73.76 | 84.68 | 26.38 | 0.03422 | 26.20 | 41.95 | 0.1160 |
| Hybrid contract-fix (50k) | **48.60** | **52.44** | **62.25** | 26.78 | **0.03258** | 24.40 | **30.72** | 0.3416 |

## HumanML3D Table 5: Original Two-Window Autoregressive Control

| Method | Protocol | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY fresh mixer (50k) | in-horizon | 44.69 | 81.56 | 140.85 | 45.06 | 0.04557 | 23.82 | 33.52 | **0.0505** |
| Minimal contract-fix (50k) | in-horizon | 65.09 | 85.44 | 114.13 | 36.34 | 0.03035 | 25.36 | 35.53 | 0.2531 |
| Hybrid contract-fix (50k) | in-horizon | **40.80** | **55.14** | **78.80** | **31.99** | **0.02823** | **23.15** | **24.35** | 0.6738 |
| ARDY fresh mixer (50k) | out-of-horizon | 57.37 | 57.40 | 74.93 | 37.47 | 0.03877 | 23.65 | 32.59 | **0.0500** |
| Minimal contract-fix (50k) | out-of-horizon | 69.07 | 71.92 | 82.66 | 33.80 | 0.02942 | 25.04 | 34.82 | 0.2637 |
| Hybrid contract-fix (50k) | out-of-horizon | **46.44** | **45.46** | **49.53** | **31.33** | **0.02933** | **23.50** | **26.07** | 0.6634 |

Under this original two-window protocol, Hybrid has the lowest constraint error
and the lowest geometry/continuity proxy errors in both Table 5 variants. This is
the appropriate archived evidence for short-horizon, training-window-aligned
comparison. It is not evidence for 9/20/40-second behavior.

## BONES-SEED Table 4: Offline Text and Joint-Position Control

| Method | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY stabilized retrain (50k) | 52.91 | 55.39 | 65.89 | 18.02 | **0.02200** | 15.32 | **38.47** | **0.0276** |
| Minimal contract-fix (50k) | 41.69 | 46.88 | 54.30 | **16.50** | 0.02904 | **14.89** | 49.52 | 0.1168 |
| Hybrid contract-fix (50k) | **39.79** | **40.44** | **48.38** | 16.88 | 0.02927 | 15.76 | 47.09 | 0.3404 |

## BONES-SEED Table 5: Original Two-Window Autoregressive Control

| Method | Protocol | Constraint Error (cm) | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARDY stabilized retrain (50k) | in-horizon | 44.53 | 61.94 | 93.67 | 32.94 | 0.02938 | **15.377** | **39.78** | **0.0651** |
| Minimal contract-fix (50k) | in-horizon | 40.11 | 47.95 | 61.79 | 24.45 | 0.02698 | 15.383 | 51.35 | 0.2782 |
| Hybrid contract-fix (50k) | in-horizon | **32.86** | **39.44** | **53.55** | **23.97** | **0.02603** | 15.642 | 44.79 | 0.6901 |
| ARDY stabilized retrain (50k) | out-of-horizon | 60.07 | 55.20 | 75.97 | 32.07 | 0.03002 | 15.898 | **40.30** | **0.0620** |
| Minimal contract-fix (50k) | out-of-horizon | 46.58 | 44.12 | 51.32 | 23.28 | 0.02708 | **15.547** | 51.63 | 0.2477 |
| Hybrid contract-fix (50k) | out-of-horizon | **40.21** | **36.36** | **43.88** | **22.12** | **0.02684** | 15.916 | 46.27 | 0.6795 |

Under the preserved BONES-SEED two-window panel, Hybrid has the lowest constraint,
root, PA-MPJPE, and jitter errors in both Table 5 variants. Stabilized ARDY has
the lowest skate proxy and latency in both variants and the lowest in-horizon
angle error; Minimal has the lowest out-of-horizon angle error.

## Interpretation Boundary

This archive answers only the old, short-horizon question. The later nine-second
and 20/40-second results are a separate zero-shot length-extrapolation track and
must not replace these values. Likewise, the Hybrid-only recurrent fine-tune is an
ablation and must not be merged into this three-model table.

The original BONES-SEED ARDY checkpoint is excluded from the tables because later
diagnosis found collapse in its root output branch. Its JSON/NPZ artifacts remain
preserved in the original result directory for audit, but they are not evidence
for a valid model comparison. The stabilized retrain passes the 50k-step health
gate (`final_eval_loss=1.0054`, `root_output_weight_norm=3.4610`, finite
gradients) and is evaluated here with the same seed, sample IDs, frame counts,
diffusion steps, and CFG settings as Minimal and Hybrid.

The stabilized ARDY legacy Table 5 run also exposes a boundary diagnostic that
the aggregate table must not hide: its maximum root speed at the 52-frame window
boundary is `69.49 m/s` in-horizon and `73.96 m/s` out-of-horizon (mean boundary
speeds `8.94 m/s` and `8.03 m/s`). The training checkpoint is healthy, but the
old native explicit-history rollout/stitching path still has severe boundary
outliers on this fixed panel. Consequently, its two-window aggregate metrics
should not be read as proof of clean temporal continuity.

R-Precision, FID, FVMD, and TM-Sim remain `N/A`: the generated CoreSkeleton27/G1
representations do not have a locally calibrated official evaluator. The reported
geometry and physical metrics are fixed-panel proxies, not official leaderboard
scores. Concurrent GPU timing supports within-run comparison only.

## Provenance

- HumanML3D launcher: `examples/run_cache_contract_dual_eval_20260823.sh`
- HumanML3D JSON/NPZ artifacts: `/data4/zts/result/humanml3d_contractfix_dual_eval_20260823/`
- BONES-SEED launcher: `examples/run_bones_seed_contract_dual_eval_20260824.sh`
- BONES-SEED Minimal/Hybrid JSON/NPZ artifacts: `/data4/zts/result/bones_seed_contract_dual_eval_20260824/`
- BONES-SEED stabilized ARDY Table 4 artifacts: `/data4/zts/result/bones_seed_ardy_retrained_dual_eval_20260826/`
- BONES-SEED stabilized ARDY legacy Table 5 artifacts: `/data4/zts/result/bones_seed_ardy_retrained_legacy_table5_20260826/`
- BONES-SEED stabilized ARDY training metrics: `/data4/zts/result/bones_seed_ardy_retrain_lr1e5_20260826_original/metrics.json`
- Evaluator: `examples/evaluate_humanml3d_table45.py`
- HumanML3D seed: `20260818`
- BONES-SEED seed: `20260824`
