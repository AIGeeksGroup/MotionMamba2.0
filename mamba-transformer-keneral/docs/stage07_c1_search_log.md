# Stage 07 C1 search log

This is an append-only record of the bounded H0/H1/H2 search. Architecture
selection uses the frozen train/development split; `final-heldout` remains
unread until one candidate is selected.

## Frozen data

- Train: 64 trajectories, manifest SHA-256
  `10f9dc0010d08bcf1558612fb7b0d81cc28cc326700dca5c65f65fc9cdbf8067`.
- Development: 8 trajectories, manifest SHA-256
  `94588595c106df0b1fb6b1c641e8d6d855db5c748bd1c7da894fccbe1e944fa6`.
- Final held-out: 8 trajectories, manifest SHA-256
  `ac189a9f1b0aae151ad873b55101d1e5e65b74e62b1a8201db6b283d7859f88b`.
- Split policy SHA-256:
  `3ae55245c156f437d5172178c79bb980257429c67e840410726bcd1a62e8fb34`.

## Real Mamba control

Tiny overfit used 32 exact teacher calls and 320 updates. Mean loss fell from
1.666632 to 0.293994 (82.4%); all gradients and state statistics were finite,
the tracked parameter changed by 1.7343, state variance was 0.001054, state
carry changed the next output by 0.043715, and save/reload passed.

The selected full control used learning rate `2e-5` for 1500 updates. Dev MSE
fell from 1.57488466 to 0.34594337 (78.0%); root/body MSE were 0.06605/0.27990.
State variance was 0.801316, mean state norm 8.0615, carry effect 0.3180, and
peak allocated memory 2282 MiB. Checkpoint:
`/workspace/ardy_stage07/mamba_real_v2_lr2e5/sequence_mamba_c1_base.pt`.
Its 1/2/4-chunk generations were finite and non-collapsed. Boundary quality
was not competitive evidence (mean jumps 0.604 and 1.006 for 2/4 chunks).

## Output complementarity diagnostic

Fusion parameters were fit on the first four development trajectories and
evaluated on the other four (48 calls); final-heldout was not touched.

- Minimal MSE: 0.24398084; Mamba MSE: 0.30456382.
- Plain average: 0.19573854 (19.77% gain over Minimal).
- Calibrated scalar: 0.20385407 (16.45%).
- Affine fusion: 0.19654843 (19.44%).
- Small diagnostic elementwise gate: 0.16189051 (33.65%).
- Oracle per-example scalar: 0.16193234 (33.63%).
- Error Pearson/cosine: 0.4291/0.4282; Minimal/Mamba win rates:
  54.17%/45.83%.

This is clear complementarity, but the diagnostic gate is not treated as a
candidate architecture or novelty claim.

## H0/H1 cheap-gate attempts

### Attempt 1: parent and new parameters, conservative learning rate

Seed 113, fixed real cached case, 120 updates, parent LR `2e-5`, new-parameter
LR `2e-4`, all layers. Both H0 and H1 started at 0.696047 and ended at
approximately 0.667368 (4.12%). Mamba, interaction, and Attention gradients
were finite/nonzero, but development MSE was unchanged. Decision: **failed
tiny-overfit gate**; do not scale.

### Attempt 2: freeze parent, faster new branch

Seed 113, fixed case, 200 updates, frozen Minimal parent, new-parameter LR
`2e-3`, all layers. H0 ended at 0.66734970 and H1 at 0.66734952. Their new
branch gradients were nonzero and learned scales were nonzero, but the same
plateau remained; development MSE worsened to 0.314070 (H0) and 0.309838
(H1), from 0.306387. Decision: **failed tiny-overfit gate**. This falsifies
the narrow explanation that zero initialization merely prevented startup.

### Attempt 3: full-model capacity check

In progress. This is an allowed learning-rate refinement of the tiny gate,
not a development-set architecture selection run. H0/H1 use identical seed,
updates, data, precision, initialization, and learning-rate budget.

## H2 isolated safety gate

H2 uses an out-of-place adapter over only `ssm_state`; it preserves state
shape/dtype and leaves the other three Mamba3 cache tensors unchanged. A
bounded `tanh(beta)` scales `e k^T`, with `beta=0` exactly reproducing the
uncorrected state. Candidate state is persisted only by the existing explicit
accepted-chunk commit. Unit tests cover exact-zero initialization, shape,
dtype, out-of-place behavior, finite gradient, and nonzero correction.

The first real-host launch exposed a scalar-dtype initialization bug in the
ARDY block (`fusion_gate_logit` was absent for H2). It was reproduced, fixed
with an H2-specific FP32 scalar branch, and the focused regression passed.
The failed launch is retained as `/workspace/ardy_stage07/h2_delta_rule_tiny_safety`.
The corrected real-state safety/tiny gate completed under the `_v2` path.
Loss moved from 0.69604731 to 0.66735351 (4.12%), while development MSE
worsened from 0.30638711 to 0.38760822. Mamba, Attention, and delta-rule
parameters all received finite gradients; learned `beta` and output scales
were nonzero; peak memory was 7377.99 MiB. Decision: **failed tiny-overfit
gate** and dropped before an equal-budget pilot.

## Search-boundary decisions

- H1b was not tested because H1a failed all cheap-gate attempts.
- H3 was not implemented because neither H1 nor H2 showed positive signal.
- Layer placement search was not opened because no all-layer candidate passed
  tiny overfit; doing so would tune a failed mechanism.
- P0--P4 frozen representation probes and per-layer CKA were not used for a
  paper claim because no trainable delta candidate crossed Gate 2. The
  output-level complementarity diagnostic above remains valid evidence that
  the two *models* differ, but it does not establish that the implemented
  token-state delta is predictively useful.
- The single permitted mathematically motivated refinement corrects H1 from
  `W(g*delta)` to the specified `g*W(delta)` and uses the previously validated
  Stage-06 gate budget (`1e-3` base, `5e-3` new, 100 updates). It ended at
  0.66823018 from 0.69604731 (4.00%), and development MSE degraded from
  0.30638711 to 0.79526359. Gradients and learned scales were nonzero. The
  equation-correct refinement therefore also **failed Gate 2**. No fifth
  architecture family is opened.
