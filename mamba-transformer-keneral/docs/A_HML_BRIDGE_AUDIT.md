# Stage A: HumanML3D Evaluator Bridge Audit

Date: 2026-09-02
Decision: **FAIL / STOP official-metric branch**

## Scope

The current bridge is implemented in
`src/humanml3d_conversion.py`. It maps CoreSkeleton27 joint positions to
the 22-joint HumanML3D index set and then calls the released
`motion_process.process_file()`. The conversion accepts positions only; it
does not preserve the Ardy 6D global rotations, twist, or original SMPL
retarget metadata.

## Existing calibration

The preserved 64-sample calibration is
`/data4/zts/result/humanml3d_mobench_dual_eval_20260822/conversion_validation.json`:

| Quantity | Value | Gate |
|---|---:|---|
| Round-trip FID | 14.984515 | required < 1: **FAIL** |
| Paired embedding cosine mean | 0.880730 | near-identity expected: **FAIL** |
| Paired embedding cosine minimum | 0.571276 | structural outliers present |
| Samples | 64 | complete |

The result is incompatible with reporting converted outputs as official
HumanML3D R-Precision/FID. The failure is not a numerical tolerance issue:
position-only reconstruction discards rotation/twist semantics before the
official feature extractor runs.

## Code audit

- `ArdyMotionRep.inverse()` can recover `global_rot_mats`, but the current
  bridge discards them and passes only `posed_joints` to `process_file()`.
- `SMPL22_TO_CORE` provides an index correspondence, not a bijective
  rotation/kinematic retarget map. It cannot reconstruct the missing SMPL
  rotations from positions.
- `process_file()` recomputes local rotations and foot/contact features from
  positions using HumanML3D offsets. This changes the representation before
  evaluator normalization.
- The official evaluator implementation and dataset calibration are present,
  but a rotation-preserving Core27-to-SMPL22 retarget asset is not present in
  this repository or the recorded result bundle.

## Required next action

The official HumanML3D branch is hard-stopped. Do not report R-Precision/FID
for generated Core27 outputs until a validated rotation-preserving retarget
asset and round-trip test reduce FID far below 1. The permitted fallback is a
separately named Core27-native evaluator; its scores must not be called
official HumanML3D metrics.

## Machine result

See `A_HML_BRIDGE_RESULTS.json` for the frozen measurements and gate fields.
The conversion regression harness remains available at
`examples/validate_humanml3d_conversion.py`.
