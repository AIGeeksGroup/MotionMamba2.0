# HumanML3D Dual Evaluation: Official and MoBench-Compatible

## Status

This report evaluates the same HumanML3D-generated samples under two explicitly
separated tracks. The panel contains 64 samples for each protocol, selected
with seed `20260818`; all three models use the same manifest, text cache,
diffusion steps (10), CFG weights (2/2), batch size (4), and generation
protocol. Sample-ID equality was checked across ARDY, Minimal, and Hybrid for
Table 4, Table 5 in-horizon, and Table 5 out-of-horizon.

The compared checkpoints are all HumanML3D-trained 50k checkpoints:

| Method | Checkpoint |
|---|---|
| ARDY fresh-mixer baseline | `/data4/zts/result/humanml3d_ardy_full_fresh_baseline_20260821_original/original_humanml3d_ddp.pt` |
| Minimal cache-fix | `/data4/zts/result/humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt` |
| Hybrid cache-fix | `/data4/zts/result/humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt` |

## Track A: HumanML3D Official Evaluation

The official HumanML3D evaluator assets are present under
`/data4/zts/humanml3d_evaluator_assets/official` and the implementation is
`src/humanml3d_official_evaluator.py`. However, the generated outputs are
CoreSkeleton27/330-D, while the official evaluator consumes the 22-joint,
263-D HumanML3D representation. The validated Core27 -> SMPL22 ->
`process_file()` conversion has a round-trip FID bias of **14.98** and mean
paired embedding cosine **0.881** (minimum **0.571**) on 64 real test motions.
This fails the representation-fidelity acceptance gate.

Therefore official generated R-Precision and FID are **N/A (blocked)** for all
three models. They are not estimated or substituted with proxy values. The
official Dataset calibration remains appendix evidence only and is not a
model result.

| Method | R-Prec. Top-3 | FID | Status |
|---|---:|---:|---|
| ARDY fresh-mixer (Table 4 / Table 5) | N/A | N/A | blocked by conversion fidelity |
| Minimal cache-fix (Table 4 / Table 5) | N/A | N/A | blocked by conversion fidelity |
| Hybrid cache-fix (Table 4 / Table 5) | N/A | N/A | blocked by conversion fidelity |

## Track B: MoBench-Compatible Unified Evaluation

The following quantities are computed from the same generated tensors and their
paired HumanML3D references. They are compatible with the metric definitions
in the MoBench paper, but they are not claimed to be the official MoBench
leaderboard implementation: the official MoBench motion encoder for FVMD and
the exact calibrated TM-Sim protocol are unavailable locally. Root ADE/FDE are
paired-reference trajectory metrics; PA-MPJPE is rigid Procrustes MPJPE; Jitter
is the mean second finite difference; Angle MAE is the mean local-rotation
geodesic error. `Skate proxy` is the existing contact/velocity proxy.

### Table 4: In-Horizon Generation

| Method | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame²) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARDY fresh-mixer (50k) | 58.76 | 78.89 | 25.52 | 0.0361 | 23.38 | 34.09 | 0.0275 |
| Minimal cache-fix (50k) | 42.44 | 50.71 | 24.69 | 0.0266 | 23.16 | 34.44 | 0.1177 |
| Hybrid cache-fix (50k) | 61.50 | 73.96 | 28.23 | 0.0367 | 23.78 | 36.53 | 0.3295 |

### Table 5: Long-Horizon Generation

| Method | Protocol | Root ADE (cm) | Root FDE (cm) | PA-MPJPE (cm) | Jitter (m/frame²) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ARDY fresh-mixer (50k) | in-horizon | 68.88 | 118.21 | 37.61 | 0.0317 | 22.92 | 35.16 | 0.0491 |
| Minimal cache-fix (50k) | in-horizon | 56.31 | 83.14 | 39.15 | 0.0256 | 22.59 | 27.69 | 0.2403 |
| Hybrid cache-fix (50k) | in-horizon | 62.12 | 72.95 | 35.61 | 0.0303 | 23.23 | 26.95 | 0.8356 |
| ARDY fresh-mixer (50k) | out-of-horizon | 59.53 | 72.85 | 34.48 | 0.0294 | 22.96 | 33.75 | 0.0499 |
| Minimal cache-fix (50k) | out-of-horizon | 55.78 | 77.55 | 38.44 | 0.0248 | 22.59 | 28.59 | 0.2522 |
| Hybrid cache-fix (50k) | out-of-horizon | 56.31 | 60.32 | 34.29 | 0.0306 | 22.84 | 28.65 | 0.8386 |

FVMD and TM-Sim are **N/A** because the official MoBench encoder and scoring
calibration are not available. These tables therefore support controlled
within-project comparisons, not claims of official MoBench scores. ADE/FDE are
especially diagnostic for paired trajectory/control tests and should not be
interpreted as a complete multimodal text-to-motion quality measure.

## Interpretation

On this fixed 64-sample panel, Minimal has the best Table 4 paired geometry
among the three models, while Hybrid is not uniformly superior in-horizon.
For long-horizon out-of-horizon generation, Hybrid has the lowest Root FDE
(60.32 cm) and PA-MPJPE (34.29 cm), with a lower skate proxy than ARDY; this is
consistent with a possible memory benefit but is not yet a paper-level claim.
All checkpoints are 50k-budget results. Multi-seed/full-test replication and a
validated Core27-to-HumanML3D conversion are required before reporting official
R-Precision/FID or claiming final convergence.

## Provenance

- Evaluation launcher: `examples/run_dual_eval_20260822.sh`
- Evaluator: `examples/evaluate_humanml3d_table45.py`
- JSON/NPZ outputs: `/data4/zts/result/humanml3d_dual_eval_20260822/`
- Conversion validation: `/data4/zts/result/humanml3d_mobench_dual_eval_20260822/conversion_validation.json`
- Official evaluator assets: `/data4/zts/humanml3d_evaluator_assets/official`
