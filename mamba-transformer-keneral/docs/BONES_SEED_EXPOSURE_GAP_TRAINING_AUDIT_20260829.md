# BONES-SEED Exposure-Gap Training Audit (2026-08-29)

## Decision

**Strict gate: FAIL. Do not start the three-model 1k/5k main run yet.**

The implementation is now complete for auditable generated-history multi-depth
training, truncated accepted-history BPTT, state/reset diagnostics, optional
teacher/generated state consistency, depth-dependent Mamba readout decay, and
generated-window root endpoint recovery. All experiments below use the
actor-disjoint BONES-SEED validation split (seed 20260829, all 16 clips with at
least 1000 frames for 20/40-second evaluation).

The current best checkpoint is:

`/data4/zts/result/bones_seed_generated_history_endpoint_v9_ft_20260829/hybrid/checkpoint_step_000100.pt`

It retains Table 4/9-second performance, keeps all recurrent state finite, and
substantially reduces the exposure gap, but Hybrid's post-window-5 Root slope
is not yet lower than Minimal at both 20 and 40 seconds.

## Implemented Training Contract

- One on-policy generated rollout is shared by all supervised depths.
- Selected depths cover early, window-five, and final prefixes without storing
  every state snapshot.
- Cache snapshots are keyed by accepted generated depth and restore the exact
  recentered runtime state.
- Multi-depth losses are normalized by target count. DDP uses `no_sync()` for
  all but the final depth in an optimizer update.
- Accepted-history BPTT replays exact sampler-committed history chunks and
  retains gradients through the newest K state writes.
- Snapshot warm starts provide truncated BPTT without rebuilding the full
  prefix graph.
- Training logs/checkpoints record target conservation, depth histograms,
  per-depth loss, raw/applied recovery, state norms, state drift, and effective
  fusion gates.
- Table 4/9-second retention, 20/40-second mean error and post-window-5 slope,
  paired bootstrap CI, and state finiteness are evaluated independently.

## Validation Trajectory

All values below are Hybrid minus Minimal. Negative values favor Hybrid. The
strict long-horizon criterion requires both mean gaps and both slope gaps to be
negative at 20 and 40 seconds.

| Variant | 20s mean C/R | 20s slope R/C | 40s mean C/R | 40s slope R/C |
|---|---:|---:|---:|---:|
| GH-v2 step900 (detached final target) | -9.51 / -7.67 | +4.019 / +3.773 | -13.87 / -13.83 | +3.146 / +3.669 |
| v4 multi-depth step100 | -11.70 / -9.72 | +2.029 / +0.967 | -18.21 / -18.67 | +2.124 / +2.924 |
| v4 multi-depth step250 | -8.37 / -6.15 | +1.992 / +1.451 | -13.41 / -13.43 | +2.116 / +2.812 |
| v4 long-biased step250 | -5.67 / -2.81 | +2.758 / +2.363 | -10.37 / -10.46 | +2.831 / +3.586 |
| v5 BPTT final-target step100 | -12.89 / -10.79 | +1.665 / +0.628 | -20.02 / -20.78 | +1.894 / +2.714 |
| v6 state consistency step100 | -13.06 / -11.04 | +1.686 / +0.584 | -19.97 / -20.74 | +1.884 / +2.719 |
| v7 decay-aligned step100 | -13.02 / -10.14 | +1.382 / +0.217 | -23.53 / -23.16 | +1.053 / +1.735 |
| v8 half-life-2 step100 | -14.69 / -11.72 | +0.484 / -0.475 | -6.46 / -5.71 | +9.101 / +9.092 |
| **v9 endpoint step100** | **-15.72 / -12.75** | **+0.669 / -0.127** | **-26.24 / -26.54** | **+0.902 / +1.686** |
| v9 endpoint step250 | -8.76 / -5.01 | +0.892 / +0.374 | -17.50 / -16.86 | +1.025 / +1.648 |
| v10 recurrent-only +100 | -15.75 / -12.76 | +0.670 / -0.117 | -26.21 / -26.51 | +0.924 / +1.720 |

`C/R` means constraint mean / Root ADE. `R/C` means Root slope /
constraint slope, in cm per generated window.

## Current Best Gate Details

For v9 step100:

- Table 4 degradation: constraint +1.06%, Root ADE +1.12%.
- 9-second in-horizon change: constraint -2.35%, Root ADE -1.86%.
- 9-second out-of-horizon degradation: constraint +1.51%, Root ADE +0.67%.
- 20-second Root-slope paired CI: [-0.278, 1.559].
- 40-second Root-slope paired CI: [0.078, 1.735].
- Maximum final/initial state-norm ratio: 3.24x.
- All state and gradient checks are finite.

The checkpoint therefore passes short retention and state health but fails the
strict 20/40-second slope criterion. Step250 and recurrent-only continuation
both regress, so step100 is the justified early-stop point.

## Exact v2 Test-Protocol Re-evaluation

The v9 step100 Minimal and Hybrid checkpoints were also evaluated on the exact
ordered v2 test panels with seed 20260824 and matching Table 4, 9-second, and
20/40-second protocols. Protocol assertions passed. These results are kept
separate from the actor-disjoint validation gate above because the v2 step900
checkpoint was selected using this test panel.

The complete Table 4/5 results and direct v9-v2 deltas are reported in
`docs/BONES_SEED_V9_V2_EXACT_TEST_PROTOCOL_COMPARISON_20260829.md`. In summary,
The report includes the fixed healthy ARDY v2-step900 baseline; no endpoint-v9
ARDY checkpoint exists. v9 improves Table 4, 9 seconds, and 40-second
in-horizon Hybrid metrics over v2 and the corresponding ARDY baseline.
At 40 seconds in-horizon, v9 Hybrid reaches 121.64 cm constraint and 118.21 cm
Root ADE with post-window-5 Root/constraint slopes of -6.079/-6.237, versus v9
Minimal's 138.04/139.43 cm and -0.856/-1.253. However, v9 Hybrid regresses
slightly at 20 seconds in-horizon and does not beat Minimal on every
out-of-horizon Root metric. It is therefore better overall on the original v2
test panel, but not a uniform win and not a replacement for the independent
validation gate.

## Eliminated Explanations

- **Short-window forgetting:** Table 4 and 9-second retention pass throughout
  v5-v10.
- **State norm explosion:** all state values remain finite, with roughly 3x
  final/initial norm growth.
- **Insufficient final-depth samples alone:** long-biased training increased
  depth-17 targets from 5 to 32 but worsened all slope gaps.
- **Detached state reads alone:** accepted-history BPTT improves all four gaps
  but does not close them.
- **Teacher/generated state consistency:** weight 0.1 produces only marginal
  movement relative to v5.
- **Simple state reset:** reset can flatten late slope but worsens mean ADE.
- **Recurrent-only continuation:** v10 is effectively unchanged from v9.

## Required Next Objective

The remaining gap is not addressed by another local next-window target. The
training objective must directly differentiate through several accepted
generated windows and penalize the post-window-5 Root/constraint growth rate.
A practical next implementation is a low-step differentiable rollout surrogate
over accepted root endpoints and recurrent states:

1. Keep the v9 short replay, BPTT, decay, and endpoint terms.
2. On a small fraction of updates, retain a 5-8-window differentiable rollout.
3. Compute Root/constraint error at each accepted endpoint relative to the
   aligned GT trajectory.
4. Penalize positive least-squares error slope after window five, not only the
   final endpoint.
5. Apply the same rollout sample IDs and target construction to ARDY, Minimal,
   and Hybrid; use architecture-specific state only where it exists.
6. Re-run the 100/250 gate before any three-model 1k/5k run.

## Reproducibility

- Main trainer: `examples/train_humanml3d_ardy_ddp.py`
- Generated-history construction: `examples/train_humanml3d_ardy_tiny.py`
- Accepted memory lifecycle: `src/ardy_adapter/diffusion_memory.py`
- Canonical differentiable commit: `src/ardy_adapter/history_commit.py`
- Runtime Mamba decay: `src/sequence_kernel/operator.py`
- Strict validation: `examples/run_bones_seed_multidepth_v4_ablation_val_20260829.sh`
- Guarded future v9 1k gate: `examples/run_bones_seed_generated_history_endpoint_v9_gate_20260829.sh`

Final regression result: **102 passed, 6 skipped**. The real CUDA recurrent
carry test also confirms that future-window loss reaches history input and
Mamba write parameters when commit detachment is disabled.
