# HumanML3D / ARDY Control Training Report

Date: 2026-08-18

## Result

The ARDY-faithful data and control smoke gates are complete and pass. The
processed 263-D HumanML3D vectors were not used. Each sample is recovered from
its native AMASS/SMPL-H `.npz`, retargeted to `CoreSkeleton27`, encoded by
`ArdyMotionRep` as 330-D, and validated by FK/round-trip checks.

## Data manifests

| split | samples | frames used for stats | excluded |
|---|---:|---:|---|
| train | 21,096 | 3,158,136 | 1,191 HumanAct12 rows, 370 sequences under 40 frames |
| val | 1,322 | n/a | 1,191 HumanAct12 rows, 16 sequences under 40 frames |
| test | 3,972 | n/a | 1,191 HumanAct12 rows, 70 sequences under 40 frames |

The train-only dual-root statistics are 334-D (`global root 5 + local root 4 +
body 325`); model motion tensors remain 330-D. Split IDs were audited against
the manifests with zero missing or extra IDs after the documented exclusions.

## Geometry gate

All retained samples are finite and pass target-skeleton FK checks. Across the
train manifest, continuous feature round-trip error is at most `1.59e-4`, bone
length error is below `1.1e-6`, rotation determinant/orthogonality errors are
below `6e-7`, and feet are grounded to numerical precision. Foot-contact is a
binary thresholded feature and is reported separately; boundary disagreements
are not mixed into the continuous round-trip metric. Preview renders were also
inspected after correcting the SMPL-H left/right arm mapping.

## Control contract

`HumanML3DArdyDataset` returns normalized motion, `observed_motion`,
`motion_mask`, caption, control type, and in/out-of-horizon metadata. Sampling
covers root trajectory, root waypoint, full keyframe, joint position, and joint
rotation controls with 10% condition dropout. A 5,000-sample audit produced:

```text
full_keyframe 1030   joint_position 1010   joint_rotation 993
root_trajectory 1005 root_waypoint 962
in_horizon 2477      out_of_horizon 2523  dropped 499
```

## Training gates

Both modes were trained on the final filtered manifest with real ARDY diffusion,
80-frame windows, 40-frame generation horizon, future constraints, and SDPA:

| gate | Minimal | Hybrid |
|---|---:|---:|
| 20-step tiny loss reduction | 73.8% | 73.2% |
| final 1k pilot loss reduction | 47.9% | 61.6% |
| fixed val loss | 1.6998 -> 0.4597 | 1.6881 -> 0.4364 |
| gradients | finite | finite |
| peak allocated memory | 5.07 GiB | 5.95 GiB |
| checkpoint strict reload | pass | pass |

The pilot's text vectors use a deterministic hash embedding to isolate motion,
control, and optimizer behavior. The real offline LLM2Vec backend was separately
loaded from the local cache and produced finite `[2, 1, 4096]` embeddings. A
full text-quality run must replace the smoke embedding with those embeddings or
precomputed caption features.

## DDP gate and generation contract

`torchrun --nproc-per-node=2` passed for both modes. Gradients were finite and
the post-step parameter checksum difference between ranks was exactly `0.0`.

The resumable DDP path was then exercised with the real LLM2Vec cache and a
boundary-continuity objective. A 2-GPU, 100-step run reached finite gradients
and reduced the sampled loss; resume from its step-100 checkpoint to step 110
completed with the same world-size contract. The trainer now records whether a
run is initialized from a checkpoint and does not incorrectly fail a fine-tune
solely because a fixed validation batch fluctuates upward.

Generation remains a separate gate. The single-GPU 500-step boundary probe
passes (`max_root_speed=5.19 m/s`, boundaries `2.65/3.16 m/s`). A 2-GPU
boundary run at `5e-5` is numerically stable but lands at `5.61 m/s`, just over
the `5.42 m/s` data-derived threshold; the 100-step run reaches `8.71 m/s`.
This shows that DDP communication and checkpoint/resume are sound, but the
full-training learning-rate/boundary configuration is not frozen yet.

Before the HumanML3D reference run, the dataset sampler was corrected so that
training visits draw a fresh crop/control/history view on every access while
the broadcast RNG state still gives exact resume. Direct indexing remains
deterministic for validation and audits. The corrected stochastic-view 1k DDP
pilot passed for both modes (global batch `32`, finite gradients, decreasing
training and fixed validation losses).

The authorized 50k-step HumanML3D reference jobs are now running detached in
tmux on two GPUs per variant. Their 5k checkpoints train normally, but a first
long-generation probe still shows internal root-speed spikes; these checkpoints
are therefore reference training artifacts, not yet the frozen Table 5 or
interactive-generation models.

## Decision

```text
Native AMASS -> CoreSkeleton27 -> 330-D adapter     PASS
Train-only stats / split integrity                   PASS
Geometry / round-trip validation                     PASS
Control training contract and sampling              PASS
Minimal tiny + 1k pilot                             PASS
Hybrid tiny + 1k pilot                              PASS
2-GPU DDP + real text + resume                      PASS
Stochastic-view 1k DDP pilot                       PASS
HumanML3D 50k reference Full Train                 RUNNING (tmux)
5k long-generation geometry probe                  NOT YET STABLE
FlashAttention-2 real-training validation           PENDING
Kernel optimization                                 WAIT
```

Full Train should begin only after selecting the real text-embedding input
path, validating the desired attention backend (FA2 or the explicitly approved
SDPA fallback), and recording the final config/checkpoint as the Golden
Reference. Kernel work remains downstream of that freeze.
