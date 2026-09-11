# Control Pipeline Audit

> Superseded for current status by [`HUMANML3D_CONTROL_TRAINING_REPORT.md`](HUMANML3D_CONTROL_TRAINING_REPORT.md).
> The original audit below records the pre-adapter state; the native AMASS
> adapter and control-training smoke are now implemented and passing.

Date: 2026-08-17
Commit audited: `91090b986c99e8ff574e4fa2c96f822ddac3b18b`

## Result

The ARDY inference stack preserves text and spatial-motion conditioning through
the Minimal/Hybrid denoiser replacement. The sequence adapter replaces only the
root/body sequence mixers; the raw ARDY denoiser still owns `motion_mask` and
`observed_motion` handling.

`examples/ardy_sequence_kernel_generate.py` now exposes `--constraints`. It
loads the standard ARDY constraint list, converts it through
`motion_rep.create_conditions_from_constraints_batched`, passes the resulting
observations/mask through the Minimal or Hybrid generation call, and preserves
the list for postprocessing.

## Verified native controls

The ARDY motion representation provides condition tensors for root 2D position,
global heading, root height, global rotations, and global joint positions.
The ARDY constraint classes and interactive UI build root waypoints, full-body
keyframes, and end-effector position/rotation constraints from that interface.
Classifier-free guidance preserves the conditional and unconditional motion
mask/observation branches.

## Full-train gate

This is an inference-path audit, not a completed control-training audit. The
current cached DiT/Minimal/C1 trainers use ARDY teacher traces and do not create
or sample control constraints. A formal control-training pipeline still needs:

1. a per-sample constraint sampler and condition-dropout policy;
2. in-horizon and out-of-horizon split definitions;
3. a training/evaluation manifest that records the masks and observations;
4. control-specific metrics and generation checks.

## HumanML3D compatibility blocker

HumanML3D provides 22-joint positions and 263-D T2M vectors. ARDY Core uses a
27-joint, 330-D representation containing root position/heading, local joint
positions, global 6D rotations, velocities, and foot contacts. No skeleton
retarget map, rotation reconstruction policy, ARDY normalization statistics, or
round-trip validation currently exists. A direct dataset adapter would silently
change the model's motion semantics and is therefore intentionally not added.

Before HumanML3D tiny overfit, implement and validate an explicit 22-to-27
retarget/converter with feature/statistics regeneration and motion round-trip
tests.
