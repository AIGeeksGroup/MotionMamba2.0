# Stage 08 C1 search log

This is an append-only Stage-08 record. Selection used TRAIN/DEVELOPMENT only;
FINAL-heldout remained unread.

## A0 and A1

- A0 fixed Minimal+Mamba average: 0.17442737 versus Minimal 0.24607701,
  a 29.12% DEVELOPMENT improvement. Decision: pass.
- Minimal seed-B: seed 83, 1500 updates, lr 5e-5, checkpoint
  `/workspace/ardy_stage08/minimal_seed83/sequence_minimal.pt`.
- A1 Minimal-A+Minimal-B average: 0.20762898 versus A 0.20827660, a 0.31%
  gain. Decision: architecture-specific complementarity passes strongly.

## Output positive control

`F0` reproduced the fixed average with max absolute error 0.0. The small
per-output gate reduced the fixed-case MSE from Minimal 0.15001257 to
0.01962164 (86.92%). TRAIN-fit gate DEVELOPMENT MSE was 0.13418588, a 45.47%
gain. Decision: plumbing/trainability pass; Stage-07 plateaus were not a
general target-alignment failure.

## Representation probes

- State sources: Mamba token hidden and actual recurrent SSM summary.
- Layers: 3 and 7 only.
- Predictor budget: 500 updates, seed 149. Probe budget: 500 updates and
  approximately 70k parameters.
- An initial diagnostic run mixed normalized delta with raw generic features
  and gave P5 2-3x more parameters. It was rejected before selection. The
  frozen corrected run standardizes all inputs using TRAIN statistics and
  parameter-matches P5.
- Corrected decision: root-only delta signal, no body/aggregate Gate-B pass.

## Late frozen-parent bridges

All variants use layer 3, 500 state-predictor updates, 500 bridge updates,
AdamW lr 1e-3, variable-length sequences with explicit padding masks, and
seeds 211/223/227.

- `G0 = MLP([z,s])`, generic fusion. Root/body parameters 198,036/139,456.
- `I1 = y_min + alpha * sigmoid(Wg[zhat,delta]) * Wout(silu(Wd delta))`,
  with `alpha=0`. Root/body parameters 107,881/336,193.
- `I2 = y_min + alpha * Wu(Attention(Wi LN(delta)))`, one 64-wide,
  four-head attention block and `alpha=0`. Root/body parameters
  85,589/92,609.

The zero-alpha I1/I2 corrections were exactly zero at initialization. Fixed
case reductions exceeded 50% for both on every seed (I1 84.97%-96.25%, I2
54.18%-99.77%). Thus innovation bridges are trainable and Gate-C capable.

Combined DEVELOPMENT results (mean ± sample std over three seeds):

| Variant | MSE | Gain vs Minimal |
|---|---:|---:|
| Minimal | 0.08217660 | 0.00% |
| G0 | 0.07271056 ± 0.00022475 | 11.52% ± 0.27% |
| I1 | 0.07805311 ± 0.00049726 | 5.02% ± 0.61% |
| I2 | 0.07634813 ± 0.00041386 | 7.09% ± 0.50% |

G0 beats I1 and I2 on all three seeds. Relative to I2, G0 lowers MSE by
approximately 4.4%-5.3%. Decision: Gate D fails. I3, deeper placement, joint
fine-tuning, generation, final-heldout, and operator-contract work are not
opened under the hard-stop rule.
