# HumanML3D Full Training Run (2026-08-18)

## Launch gate

- Train/val/test manifests: `21096 / 1322 / 3972` samples.
- Train-only dual-root stats: finite `334-D` mean/std.
- Offline LLM2Vec cache: finite `[22418, 4096]` FP16 embeddings, `complete=PASS`.
- Repository tests after stochastic-view fix: `84 passed, 6 skipped`.
- DDP exact resume check: continuous 20 steps vs split/resume 20 steps had
  `max_parameter_abs_diff=0.0` across 268 tensors.
- Final 1k stochastic-view pilot: Minimal and Hybrid both passed finite-gradient,
  loss-decrease, and fixed-validation gates.

## Reference jobs

The two full HumanML3D reference jobs are running in detached tmux sessions:

| variant | session | GPUs | per-rank batch | global batch | steps | LR | control loss |
|---|---|---:|---:|---:|---:|---:|---|
| Minimal | `mm2_hml3d_full_minimal_20260818` | 0,1 | 16 | 32 | 50,000 | `5e-5` | boundary `10` |
| Hybrid | `mm2_hml3d_full_hybrid_20260818` | 2,3 | 16 | 32 | 50,000 | `5e-5` | boundary `10` |

Both use the same HumanML3D manifests, train-only stats, real LLM2Vec cache,
BF16, SDPA, history probability `0.5`, and checkpoint every `5000` steps.
Logs are written to `/data3/zts/results/mm2_hml3d_full_{minimal,hybrid}_20260818.log`.

## 5k checkpoint probe

The first checkpoints are valid numerical training artifacts, but their
long-horizon geometry is not yet a frozen final result:

| variant | max root speed | chunk-boundary speeds | gate |
|---|---:|---|---|
| Minimal | `24.45 m/s` | `3.27 / 2.74 m/s` | FAIL (internal spike) |
| Hybrid | `9.47 m/s` | `3.27 / 6.08 m/s` | FAIL |

The jobs remain useful as the no-velocity-loss HumanML3D reference for Table 4
/ Table 5 evaluation. A separate velocity-loss probe has been implemented and
tested, but is not mixed into these already-running reference jobs.

## Status

```text
HumanML3D full reference training: RUNNING in tmux
DDP / resume / data integrity: PASS
Kernel: not enabled
Table 4 / Table 5 benchmark: pending final checkpoints
Interactive/Bones Seed training: separate experiment, not part of this run
```
