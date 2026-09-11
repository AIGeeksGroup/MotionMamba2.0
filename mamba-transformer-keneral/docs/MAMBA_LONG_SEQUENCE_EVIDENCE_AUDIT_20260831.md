# Mamba Long-Sequence Evidence Audit (2026-08-31)

## Verdict

**NOT SUPPORTED** for a stable Mamba-specific long-memory advantage or a
Hybrid system-efficiency advantage.

The mechanism is real, and Hybrid decay-carry substantially lowers
20/40-second mean error and post-W5 AUC relative to Minimal-W4 on the
actor-disjoint selection split. The preregistered gate still fails because the
Root/constraint post-W5 slopes are not consistently lower, and carry does not
consistently beat same-checkpoint post-W4 permutation. The hard stop therefore
prevents three-seed 1k, 5k, and locked-final evaluation.

## Evidence Boundary

- Dataset: BONES-SEED v004.
- Training actors: 463; selection actors: 31; locked-final actors: 28.
- Actor sets are pairwise disjoint and hash-locked.
- All choices and the 100/250 gate use selection only.
- Locked-final is a newly frozen confirmation set, not a never-observed blind
  test. It remains unread.
- Long panels contain 16 motions across 7 actors because selection has only 16
  motions at least 40 seconds long. No valid low-overlap second crops exist.
- Bootstrap samples actor first and motion within actor, 20,000 times.
- Learned R-Precision, FID, FVMD, and TM-Sim are `N/A`; no compatible official
  learned evaluator is available. They were not synthesized from proxies.

Protocol lock:
`/data3/zts/MotionMamba2.0/mamba-transformer-keneral/configs/bones_seed_long_memory_protocol_v2.json`

Frozen panels:
`/data4/zts/result/bones_seed_long_memory_panel_v2_20260831`

## Implemented Mechanisms

- Separate 5-8-window differentiable rollout, independent of the historical
  no-grad generated-history path.
- Pure-tensor recentering and recurrent carry; windows 6-8 backpropagate into
  early rollout inputs and recurrent write/read parameters.
- Deterministic low-step diffusion, post-W5 Root/constraint AUC, slope, and
  final losses, graph-depth/gradient/memory/timing logs, and exact resume.
- Attention W4/W8/W16 and Minimal-ParamMatch controls.
- Mamba-only fresh, zero, freeze, permutation, readout-off, and no-decay
  interventions with before/after Attention and Mamba checksums.
- Per-window geometry/constraint/physical metrics, state diagnostics,
  per-sample JSONL, intervention JSONL, and NPZ artifacts.

An evaluator defect discovered during execution was fixed: ParamMatch training
used function-preserving FFN checkpoint expansion, but both evaluators used
direct strict loading. They now reuse the tested expansion loader. Valid jobs
were retained and only the failed ParamMatch job was rerun.

## Training Results

Stage A completed all seven 20-step configurations. Every run completed 20
differentiable rollout updates with finite outputs and gradients and two
nonzero early differentiable endpoints per update. Peak memory ranged from
4.16 GiB (ARDY) to 10.20 GiB (Minimal-ParamMatch). Generic training `status`
remained `FAIL` for ARDY and most Minimal variants because stochastic 20-step
mean loss was not monotonically lower. This negative result is retained and is
not treated as a mechanism failure.

Stage B completed 100 and exact-resume 250 checkpoints for ARDY, Minimal
W4/W8/W16/ParamMatch, Hybrid no-decay carry/fresh, and registered decay
carry/fresh. Registered decay is start W4, half-life 4, minimum scale 0. Both
decay Hybrid runs completed 73 differentiable rollout steps with finite
gradients. Their generic loss-trend `status` is also `FAIL`; checkpoint
selection did not use that status.

Training artifacts:

- `/data4/zts/result/bones_seed_long_memory_stage_a_20260831`
- `/data4/zts/result/bones_seed_long_memory_stage_b_20260831`
- `/data4/zts/result/bones_seed_long_memory_stage_b_decay_20260831`

## Capacity Control

| Configuration      | Active root+body parameters |
| ------------------ | --------------------------: |
| ARDY fresh-mixer   |                 147,150,996 |
| Minimal W4/W8/W16  |                 247,914,644 |
| Minimal-ParamMatch |                 352,823,444 |
| Hybrid             |                 352,578,724 |

ParamMatch differs from Hybrid by 244,720 parameters (0.069%). Expansion
copies all old FFN channels exactly, zeroes new output columns, and rejects
non-FF shape mismatches.

## Preregistered Gate

Differences are `Hybrid decay-carry - Minimal-W4`; negative favors Hybrid. CIs
are actor/motion cluster-bootstrap 95% intervals.

| Step | Duration |                      Root AUC |             Root slope |                Constraint AUC |       Constraint slope |
| ---: | -------: | ----------------------------: | ---------------------: | ----------------------------: | ---------------------: |
|  100 |     20 s |    -35.759 [-63.012, -15.303] | +0.748 [-0.109, 1.678] |    -39.727 [-63.210, -15.101] | +0.891 [-0.675, 2.573] |
|  100 |     40 s | -430.076 [-534.597, -351.926] | +0.124 [-0.796, 1.001] | -401.244 [-490.764, -327.770] | +0.877 [-0.032, 1.780] |
|  250 |     20 s |    -31.459 [-59.708, -11.437] |  +0.856 [0.009, 1.787] |    -36.526 [-60.883, -13.714] | +1.137 [-0.075, 2.604] |
|  250 |     40 s | -421.438 [-547.597, -328.143] | -0.018 [-0.945, 0.829] | -402.433 [-514.951, -309.623] | +0.631 [-0.267, 1.492] |

The AUC signal is large and consistent, but 7 of 8 slope means fail the
required negative direction. At step 250, the 20-second Root-slope CI is wholly
above zero. This is a hard gate failure, not an inconclusive pass.

All Table 4 and 9-second Root/constraint retention checks pass the 10%
threshold. Worst positive degradation is 3.70% at step 100 and 6.76% at step
250. State/output/training values are finite. For batch-2 root+body evaluation,
the mature W4 cache is constant at 30,836,352 bytes: 13,633,152 Attention bytes
plus 17,203,200 Mamba bytes.

## Causal Interventions

Carry strongly improves AUC relative to `fresh_each_window`, but slope CIs
often include zero. Relative to same-checkpoint `permute_after_w4`, the effects
are small and fail the registered all-negative direction:

- Step 100, 40 s: carry-minus-permute Root/constraint slope is
  +0.133/+0.148 cm per window.
- Step 250, 20 s: constraint slope is +0.024 cm per window.
- Step 250, 40 s: Root/constraint slope is +0.064/+0.047 cm per window.

`freeze_after_w4` is as good as or slightly better than ordinary carry on
several aggregate metrics, and `readout_off_after_w4` remains close to carry.
These controls prevent attributing all long-horizon improvement specifically
to continued post-W4 Mamba updating/readout. Separate GPU replays keep the
first four errors within 0.41 cm for post-W4 interventions, small relative to
tens-of-centimetres long errors; fresh-each-window intentionally differs from
window 1. Raw audits verify that Mamba-only mutations preserve Attention K/V.

Gate file:
`/data4/zts/result/bones_seed_long_memory_stage_b_selection_eval_20260831/gate_summary.json`

## Physical Proxies

Hybrid decay-carry has zero per-sample catastrophic Root-error events at the
registered 500 cm threshold for both stages and durations. This does not imply
physical safety: 40-second max Root speed reaches 14.60 m/s at step 100 and
17.71 m/s at step 250, while foot-skate proxy is 17.62% and 16.02%.

## System Scaling

The isolated real-checkpoint root-block benchmark is not a quality result. It
serially measures W1/2/4/8/16/32 with warmup 3 and 10 repeats on one RTX 4090.

| Mode                       | W32 median CUDA time | W32 persistent state | Log-log alpha | State constant after W4 |
| -------------------------- | -------------------: | -------------------: | ------------: | ----------------------- |
| ARDY explicit full-history |            142.29 ms |                    0 |         1.018 | yes                     |
| Minimal sliding W4         |            414.22 ms |          3,408,288 B |         0.921 | yes                     |
| Hybrid W4 carry            |          1,017.76 ms |          7,709,088 B |         0.878 | yes                     |

Both Minimal and Hybrid have bounded state and near-linear measured total time
in this range. Hybrid is slower and uses more state. Linearity is not unique to
Mamba, and no Hybrid system-efficiency advantage is supported.

Benchmark artifacts:
`/data4/zts/result/bones_seed_long_sequence_scaling_v2_20260831`

## Claim Boundary

Allowed: the multiwindow mechanism is implemented and CUDA-tested; Hybrid
decay-carry improves selection-panel long-horizon mean error and AUC; fresh
Mamba resets cause a large degradation; sliding W4 and Hybrid state are bounded.

Not allowed: stable Mamba-specific long-memory advantage; claims omitting
failed slopes; unique linear scaling; Hybrid efficiency advantage;
locked-final/test generalization; or unavailable learned-metric claims.

## Verification And Bundle

- Full suite: `115 passed, 6 skipped`.
- Mandatory CUDA rollout test: `1 passed` on RTX 4090, Torch 2.6.0+cu124.
- Static compilation: pass.
- Machine bundle: `/data4/zts/result/mamba_long_sequence_evidence_20260831`
