# Stage 08 token-state probe

## Protocol

Both parents remained frozen. Representations were extracted from middle
layer 3 and final layer 7 on TRAIN (57,600 valid tokens per stage/layer) and
DEVELOPMENT (7,200). Separated-CFG rows were explicitly reassembled in input
order. FINAL-heldout was not loaded.

Minimal provides `z`; Mamba provides both its per-token hidden and an actual
call-final recurrent SSM state summary. The recurrent summary is aligned to
tokens with normalized token position. A two-layer MLP fits `R(s) -> z` on
TRAIN. The innovation is `LN(z) - LN(R(s))`.

## State prediction

The selected root representation is layer-3 recurrent state: DEVELOPMENT
MSE 13.7941, R² 0.5120, cosine 0.9979. The selected body representation is
layer-3 Mamba hidden: MSE 2.8728, R² 0.7399, cosine 0.9549. Thus the actual
recurrent state is useful for root, but it is not the best state source for
body. Layer 7 was worse for both stages.

On DEVELOPMENT, root `||delta||/||z||` is 0.00155 and body is 0.04451. These
ratios use the specified normalized innovation numerator and raw token norm
denominator, so they are descriptive rather than a shared-scale energy
fraction.

## Matched P0-P5 result

All probe inputs use TRAIN-statistic feature standardization. P0-P5 are
approximately 68k-70k parameters, including a corrected 69k generic gate.

| Probe | Root MSE | Body MSE |
|---|---:|---:|
| P0 token | 0.004704 | 0.081502 |
| P1 state | 0.011804 | 0.085790 |
| P2 concat | 0.007230 | 0.084137 |
| P3 delta | 0.004987 | 0.094162 |
| P4 predicted-state + delta | 0.005455 | 0.085769 |
| P5 generic gate | 0.006796 | 0.086422 |

P3 has a local root signal but loses to token-only P0. On body, which contains
most output dimensions, explicit delta is worst; P4 does not meet the 3%
advantage threshold over matched generic controls. Aggregate evidence does
not pass Gate B. The data support representational complementarity, not a
delta-specific inductive-bias claim.

The expected smooth-motion/small-innovation interpretation was not promoted:
the available metadata and mixed root/body outcome are insufficient for that
claim.
