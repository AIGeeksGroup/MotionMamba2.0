# BONES-SEED G1 Training and Evaluation Protocol

## Scope

This protocol transfers the 2026-08-23 HumanML3D three-architecture study to
BONES-SEED v004 without changing the accepted `canonical_history_v2` memory
contract. The compared architectures are:

1. ARDY native Transformer with freshly initialized root/body mixers.
2. Minimal attention-only contract-fix model.
3. Hybrid Mamba+Attention contract-fix model.

All three use the released ARDY-G1 frontend, tokenizer/decoder, diffusion
process, and compatible projection initialization. Released Rigplay mixer
weights are excluded from all three architecture comparisons. Full training
updates the denoiser under the same scope used in the HumanML3D 8.23 run; it
does not retrain the released tokenizer/decoder.

## Native Representation

- Source: BONES-SEED `g1/csv` MuJoCo-compatible trajectories at 120 fps.
- Model: `ARDY-G1-RP-25FPS-Horizon52`.
- Resampling: deterministic 120 -> 25 fps time-index sampling.
- Coordinates: MuJoCo z-up/x-forward -> ARDY y-up/z-forward.
- Explicit representation: ARDY G1, 414 dimensions.
- Dual-root statistics: 418 dimensions.
- Canonicalization: first heading rotated to zero and first planar root
  position translated to the origin.

The SOMA archives are downloaded and validated as part of the complete dataset,
but are not silently retargeted to CoreSkeleton27. The G1 CSV is the official
dataset representation that exactly matches the published ARDY-G1 skeleton.

## Split

The public metadata does not define an official train/validation/test split.
This experiment therefore uses a deterministic actor-disjoint split:

- SHA-256 assignment seed: `20260824`
- Train/validation/test buckets: 90% / 5% / 5%
- Train: 125,258 motions, 463 actors
- Validation: 7,578 motions, 31 actors
- Test: 9,384 motions, 28 actors
- Actor overlap: zero for every split pair

Original and mirrored motions share `actor_uid` and therefore cannot cross
split boundaries. Every selected metadata row has at least one natural-language
description.

## Training

The three 50k runs use the same:

- seed `123`
- global batch size `32` (4 GPUs x 8 samples)
- 50,000 optimizer steps
- learning rate `5e-5`
- 104-frame training windows
- condition/text dropout `0.1/0.1`
- boundary/root-velocity/constraint/FK/foot-skate loss weights
  `10/50/1/1/50`
- G1 train/dev manifests, physical stats, and LLM2Vec text cache

Minimal and Hybrid use cache-only sequential batches with probability `0.5`
and report both explicit-history and cache-only fixed dev loss. ARDY has no
adapter-owned KV/Mamba state and uses its native explicit-history protocol.

Required gates are conversion validation, forward/backward smoke, tiny
overfit, 1k, 5k, then 50k. Every gate requires finite gradients and checkpoint
metadata consistent with `canonical_history_v2`.

The 5k gate resumes the corresponding 1k checkpoint, matching the HumanML3D
8.23 contract-fix gate protocol. The final 50k comparison is initialized fresh
for all three architectures and does not inherit gate weights.

## Evaluation

The final test panel uses 64 fixed samples per protocol, seed `20260824`, 10
diffusion steps, CFG text/constraint weights `2/2`, and identical sample IDs,
goal frames, and goal joints across architectures.

- Table 4 proxy: offline text and joint-position control, 52-frame native G1
  horizon.
- Table 5 follows ARDY Section 6.3 / Table 5 using a nine-second sequence.
  At 25 fps, each sample contains 225 frames. The tokenizer-aligned GT prefix
  is 24 frames (0.96 seconds), followed by 201 evaluated generation frames.
- Table 5 in-horizon places one pelvis/wrist/foot position goal at the end of
  every 52-frame generation window, including the truncated final window.
- Table 5 out-of-horizon places a single pelvis/wrist/foot position goal at
  the final frame, beyond the initial autoregressive generation window.

Minimal and Hybrid commit the GT prefix once through the canonical history
prime and then roll out with cache-only persistent state. Native ARDY uses its
explicit autoregressive history path. Metrics exclude the GT prefix.

Reported paired-reference metrics are constraint error, Root ADE/FDE,
PA-MPJPE, jitter, Angle MAE, foot-skate proxy, boundary root speed, latency,
active parameters, and peak memory. BONES-SEED has no locally available
official calibrated retrieval/FID or MoBench evaluator, so R-Precision, FID,
FVMD, and TM-Sim remain `N/A`; no HumanML3D evaluator or substitute embedding
space will be presented as an official BONES-SEED score.
