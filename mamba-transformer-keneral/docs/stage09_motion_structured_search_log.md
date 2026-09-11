# Stage 09 motion-structured search log

This is an append-only record. TRAIN and DEVELOPMENT were used; FINAL-heldout
remained unread.

## Repository preparation

- Base: `57b267ddc92a7abcaf85beb4b01d0ba576b3c17c`.
- Branch: `codex/stage09-motion-structured-c1`.
- The pre-existing `stack.py` docstring edit was preserved in stash
  `pre-stage09 user docstring change` and excluded from all commits.

## Canonical evaluator smoke and correction

The first smoke used full coupled-output per-element MSE as total. This was
rejected before gate selection because the repository has an explicit
training objective: root MSE plus body MSE. Complete-output MSE includes
history/masked positions and diluted the model-stage errors. Frozen
predictions were deterministically resummarized under the correct objective;
the old diagnostic remains recorded under `full_output`.

Canonical artifact:
`/workspace/ardy_stage09/canonical_matrix_v2/canonical_results.json`.

## Frozen cross matrix

| Candidate | Root | Body | Coupled total | Gain vs B0 |
|---|---:|---:|---:|---:|
| B0 Attention/Attention | 0.156477 | 0.265658 | 0.422135 | 0.00% |
| B1 Mamba/Mamba | 0.066065 | 0.351231 | 0.417296 | 1.15% |
| S1 Mamba/Attention | 0.066065 | 0.347763 | 0.413828 | 1.97% |
| D0 Attention/Mamba | 0.156477 | 0.283468 | 0.439944 | -4.22% |
| Minimal+Mamba average | 0.065891 | 0.223083 | 0.288975 | 31.54% |
| Minimal+Minimal average | 0.155097 | 0.264419 | 0.419517 | 0.62% |

Isolated body MSE is 0.260077 for Attention and 0.279898 for Mamba. Root
directionality is strong, and aggregate S1 beats D0 by 5.94%. The latter is
not trajectory-robust: S1 wins only trajectories 2, 5, and 6 of eight. Its
aggregate gain is dominated by large improvements on trajectories 2 and 5,
while five trajectories favor D0. Gate A's non-concentration requirement
therefore fails despite the aggregate.

## S2 tiny gate

Seed 211, one fixed real call, 500 updates, frozen parents, 139,456 trainable
body-G0 parameters. The first report incorrectly included immutable Mamba-root
loss in the reduction. The corrected body-specific loss moved 0.164267 to
0.072324 (55.97%), gradients were finite, parameter norm became 6.1023, and
save/reload passed. Decision: plumbing gate passes.

## Rejected online equal-budget attempt

An initial H0/S2 pilot used one full call per optimizer step. It was retained
as a failed run under `/workspace/ardy_stage09/{H0,S2}_seed*_v3`. This was not
equivalent to Stage-08's 4096-token batches and produced high-variance sampled
losses. It is excluded from architecture selection. Online S2 happened to
score 0.36932 mean, but it is not a valid equal-budget comparison.

Two real-path integration bugs were fixed before the corrected run:

1. G0 did not expose the delegated Attention `sequence_stack` needed by the
   history-commit helper.
2. accepted commits invoke an additional short internal DiT stack call; the
   diagnostic hook now retains only read-path features matching the parent
   output length.

Neither fix changes a parent prediction or architecture.

## Corrected H0 protocol

The exact Stage-08 G0 family (`MLP([z,state])`, width 64) was fit from frozen
canonical features for 500 updates, batch 4096, lr 1e-3, seeds 211/223/227.
Root was fitted first; body features were then extracted under the fitted
coupled root. Artifacts:
`/workspace/ardy_stage09/offline_H0_seed{211,223,227}`.

H0 totals were 0.410816, 0.411111, and 0.410447: mean
`0.410792 ± 0.000333`, only a 2.69% improvement over B0. Gate 0 requires at
least 5% in two of three seeds, so it fails in 0/3.

The offline S2 jobs had already begun concurrently. They were interrupted as
soon as the complete three-seed H0 Gate-0 result was known, in accordance with
the hard-stop rule. No result from those partial jobs is used.

## Stop decision

Gate 0 is not fully reproduced under the canonical coupled protocol: the
architecture-diverse output average remains strong, but the required H0
single-model control does not reach 5%. Independently, Gate A lacks
trajectory-robust directionality. No conditioned gate, generation,
FINAL-heldout evaluation, or new architecture was opened.
