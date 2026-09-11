# Stage 08 complementarity analysis

## Frozen inputs

- Minimal SHA-256: `acbdcc78c6594ad690a1d3a76b129665198fd142e487a1eb64bf43163a1b949f`
- Mamba SHA-256: `cdcc3a658f221843048889a07d20329718fea963d5bf0eacddd016538fd198c1`
- TRAIN manifest SHA-256: `10f9dc0010d08bcf1558612fb7b0d81cc28cc326700dca5c65f65fc9cdbf8067`
- DEVELOPMENT manifest SHA-256: `94588595c106df0b1fb6b1c641e8d6d855db5c748bd1c7da894fccbe1e944fa6`
- FINAL-HELDOUT manifest SHA-256: `ac189a9f1b0aae151ad873b55101d1e5e65b74e62b1a8201db6b283d7859f88b`

Final-heldout has not been loaded. Fusion parameters were fit on TRAIN only;
all numbers below are one-pass DEVELOPMENT reports.

## Gate A0

Across 96 combined root/body calls:

| Policy | Development MSE | Relative gain vs Minimal |
|---|---:|---:|
| Minimal | 0.24607701 | 0.00% |
| Mamba | 0.25099943 | -2.00% |
| fixed 0.5 average | 0.17442737 | 29.12% |
| TRAIN-fit scalar | 0.17452533 | 29.08% |
| TRAIN-fit affine | 0.16785828 | 31.79% |
| TRAIN-fit elementwise diagnostic gate | 0.13339354 | 45.79% |
| oracle per-call scalar | 0.14453238 | 41.27% |

The fixed average exceeds the required 5% gain by a wide margin. Gate A0
passes. Error Pearson/cosine are 0.40064/0.40066; Minimal and Mamba win rates
are 48.96%/51.04%.

The first affine implementation used a direct tall float32 least-squares solve
and returned a clearly invalid near-zero solution. This was diagnosed as
numerical conditioning, replaced by a 3x3 float64 sufficient-statistic solve,
and recomputed from the exact frozen prediction records. The corrected TRAIN
fit is `[0.575554, 0.593500, 0.001295]`.

## Localization

Complementarity is not uniform.

- **Root:** Minimal 0.15647603, Mamba 0.06604825, fixed average 0.06588244.
  Mamba wins 71.88% of calls. Much of the aggregate gain is a stage/global
  dynamics effect rather than proof of within-token innovation.
- **Body:** Minimal 0.26007716, Mamba 0.27989804, fixed average 0.19138752.
  Despite Mamba being worse alone, the average improves Minimal by 26.41% and
  each parent wins 50% of calls. This is genuine same-target error diversity.
- **Diffusion indices 0/5/9:** fixed-average gains are 42.77%/35.60%/36.88%
  on stage-separated records. The signal is not confined to one noise level.
- **Chunks 0/1/2:** fixed-average gains are 47.45%/48.80%/31.44%. At chunk 3,
  Minimal is already much better (0.04669 versus Mamba 0.19138), and averaging
  hurts by 69.17%. A useful gate must learn horizon/chunk dependence rather
  than apply a global blend blindly.

## Gate A1: same-architecture control

The mandatory second Minimal replica used seed 83, the same teacher cache,
architecture, 1500-update budget, and learning rate as seed A. Its checkpoint
is `/workspace/ardy_stage08/minimal_seed83/sequence_minimal.pt` (SHA-256
`fe93b20b1e114efd01989f3d4c52c119a29c292c3995fe013a97cc80a284f18a`).

On the stage-separated DEVELOPMENT records, Minimal A scored 0.20827660,
Minimal B 0.21632473, and their fixed average 0.20762898: only a 0.31% gain
over A. The TRAIN-fit scalar gained 0.16%. In contrast, Minimal+Mamba gained
29.12% on the clean A0 protocol. The architecture-diverse gain exceeds the
same-architecture gain by about 28.8 percentage points, so Gate A1 passes
strongly. The output diversity is architecture-specific on this distribution;
it is not explained by ordinary same-model seed variance.

## Interpretation

A0/A1 establish a large, architecture-specific output-level ceiling. The
root result is partly a stage/global-dynamics specialization, while the body
result still shows same-target error diversity. Neither result by itself
establishes that an explicit token-state innovation parameterization is the
best way to compress the diversity; that question is resolved separately in
the matched probes and late-bridge closeout.
