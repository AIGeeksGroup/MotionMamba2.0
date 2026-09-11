# 40-second Out-of-Horizon Duration-Figure Audit

## Scope

This audit identifies the raw results that support the final four-model 40-second BONES-SEED out-of-horizon plots.

## Verified Source Results

All four source files are `PASS`, use `protocol: out_of_horizon`, generate 19 windows of 52 frames (40 seconds total including the 24-frame prefix), and contain the identical ordered 16-sample selection panel. They were evaluated after the D2R-L 5k checkpoint was completed.

| Display label | Source result | Checkpoint | W19 Root ADE (cm) | Final-goal constraint error (cm) |
|---|---|---|---:|---:|---:|
| ARDY | `/data4/zts/result/stage_d2rl_frozen_5k_final_eval_20260906/ardy/40s_out_of_horizon/result.json` | `bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt` | 1774.78 | 1773.58 |
| Minimal-W4 | `/data4/zts/result/stage_d2rl_frozen_5k_final_eval_20260906/minimal_w4/40s_out_of_horizon/result.json` | `bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt` | 160.53 | 161.72 |
| Native Hybrid | `/data4/zts/result/stage_d2rl_frozen_5k_final_eval_20260906/native_hybrid/40s_out_of_horizon/result.json` | `bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt` | 83.20 | 84.80 |
| D2R-L | `/data4/zts/result/stage_d2rl_frozen_5k_final_eval_20260906/d2rl_5k/40s_out_of_horizon/result.json` | `stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt` | 49.59 | 49.93 |

## Generated Figures

| Artifact | Metric and interpretation |
|---|---|
| `figures/final_four_model_root_curve_out_of_horizon.png` and `.pdf` | Root ADE at each rollout window for all four models. |
| `figures/final_four_model_constraint_curve_out_of_horizon.png` and `.pdf` | Final-goal constraint error at W1/2.08s, 9s, 20s, and 40s for all four models. |

## Why No Four-Model Constraint Curve Exists

The 40-second out-of-horizon protocol assigns one goal at frame 975, namely W19. Therefore each raw source has `constraint_error_cm_mean_by_window = [null, ..., null, value_at_W19]`; a per-window constraint curve would fabricate eighteen values that the evaluator did not define. D2R-L now has a completed 40-second out-of-horizon evaluation at `/data4/zts/result/stage_d2rl_frozen_5k_final_eval_20260906/d2rl_5k/40s_out_of_horizon/result.json`.

The four-model in-horizon figures use the same 16-sample selection split and include a constraint at every window. The out-of-horizon root plot is directly comparable across all four models; the out-of-horizon constraint plot is endpoint-only by protocol.

## Four-point duration curve

To make the endpoint result readable, a separate duration plot combines the available final-goal constraint values at W1 (52 frames, 2.08s), 9s, 20s, and 40s. The 20s evaluation was newly completed using the same 16-sample selection panel as the 40s run:

| Model | W1 / 2.08s | 9s | 20s | 40s (cm) |
|---|---:|---:|---:|---:|
| ARDY | 31.65 | 31.57 | 67.30 | 1773.58 |
| Minimal-W4 | 37.58 | 40.88 | 82.52 | 161.72 |
| Native Hybrid | 27.06 | 24.92 | 42.05 | 84.80 |
| D2R-L | 26.75 | 24.75 | 27.00 | 49.93 |

The four-point figure is the formal `figures/final_four_model_constraint_curve_out_of_horizon.png` (and `.pdf`); `final_four_model_constraint_duration_curve_out_of_horizon.*` is an explicit-name alias. It uses a logarithmic y-axis so the ARDY divergence does not hide the three lower-error trajectories. W1/9s use the 64-sample short panels from `FINAL_TABLE4.md`/`FINAL_TABLE5.md`; 20s/40s use the 16-sample long selection panels. Consequently, this is a duration-trend visualization and should be labelled as a proxy comparison, not a single fixed-panel statistical test.

## Reproduction

Run:

```bash
python examples/write_four_model_40s_out_of_horizon_main_figures_20260908.py

python examples/write_four_point_constraint_duration_figure_20260908.py
```

The script validates status, protocol, duration, window count, and exact sample order before writing the figures.
