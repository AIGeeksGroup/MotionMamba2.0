# Architecture v1 Freeze

Date: 2026-08-14  
Status: **FROZEN FOR TRAINING**

This freeze ends architecture exploration and establishes the model contracts
used by Stage 04. A change to these contracts now requires a reproducible
correctness failure, downstream incompatibility, or training evidence—not a
new architecture preference.

## Frozen mode definitions

### Original (`--original`)

Existing released ARDY Transformer path. It uses ARDY's unchanged root/body
denoiser, diffusion sampler, CFG, motion representation, autoregressive loop,
postprocessing, and output tooling. It has no new DiT/SequenceKernel adapter.

### DiT (`--dit`)

ARDY autoregressive diffusion with the new root/body DiT denoiser. Its sequence
mixer is attention-only with `window_size_chunks=0`, no persistent cross-chunk
K/V, no attention sinks, and no Mamba.

### Minimal (`--minimal`)

ARDY autoregressive diffusion with the same DiT denoiser plus attention,
chunk-level sliding-window K/V, and dedicated learned attention sinks. Mamba
is absent from both the module tree and the operational import path.

### Hybrid (`--hybrid`)

Minimal plus the existing Mamba-3 branch through `SequenceKernel`'s frozen
gated-add fusion. Attention K/V and Mamba recurrent state remain separate.
The existing Mamba implementation is reused and is not redesigned here. This
mode is an integration scaffold, not the finalized MotionMamba Contribution-1
architecture.

## Paper/release positioning

The architecture freeze and paper-contribution boundary are different claims:

- Original, DiT, and Minimal are standard baselines. Minimal intentionally
  combines AR diffusion, DiT, native FlashAttention, sliding-window K/V, and
  attention sinks without claiming novelty.
- `MinimalAttentionKernel` is a compatibility implementation-module name, not
  a custom-kernel contribution.
- MotionMamba 2.0 begins when Mamba-specific token/state modeling is added.
- The current Hybrid gated-add implementation is scaffolding for experiments;
  future Contribution 1 targets token/state interaction or delta modeling.
- Future Contribution 2 is the integrated/custom kernel corresponding to the
  finalized Contribution-1 mechanism. It does not belong to Minimal.
- SDPA is the reference, FA2 is currently implemented/validated, and FA4 is an
  intended future native backend replacement with no current result claim.

## Frozen memory semantics

- Current-chunk Q/K/V always participate in current attention.
- Normal history capacity is measured in whole committed chunks, not tokens.
- Dedicated sink parameters are not episode-derived cache entries, do not
  consume the history window, and survive eviction/reset.
- A forward is read-only unless a pending state is explicitly committed.
- Every noisy diffusion evaluation, including the original `t=0` denoiser
  evaluation, is read-only.
- The final accepted clean/generation representation is evaluated once for
  commit. Root and body must each increment their accepted-commit count by
  exactly one or both are rolled back.
- ARDY CFG keeps persistent state at user batch B. Its 2B/3B guidance branches
  independently read the same B-row history; they never become commits.
- Full and per-row reset clear episode history without modifying learned sinks.

## Training cache lifetime

Projected K/V may live only inside one training trajectory/sample under one
fixed set of projection weights:

```text
fresh cache -> sequential chunks -> loss -> backward -> optimizer.step()
            -> discard/reset cache
```

Projected K/V must not cross `optimizer.step()`. The kernel records projection
versions and rejects stale cache use. Early Minimal training uses clean/ground-
truth previous chunks; generated-history exposure is explicitly later work.

## Frozen ownership boundaries

- ARDY owns root/body representation, constraints, CFG, diffusion schedule,
  autoregressive windowing, recentering, postprocessing, and visualization.
- `ArdyDiTSequenceBlock` adapts ARDY block inputs into DiT conditioning.
- `DiTSequenceStack` owns AdaLN-conditioned DiT layers.
- `SequenceKernel` selects minimal or hybrid mixers.
- `MinimalAttentionKernel` owns attention projections, sinks, and K/V cache.
- Existing `Mamba3ChunkWrapper` and `StateTransitionMemoryOperator` remain
  unchanged unless a concrete regression demonstrates a correctness defect.

## Stage-04 architecture audit

| Component | Status | Evidence |
|---|---|---|
| Minimal attention forward | PASS | finite shape/reference and CUDA gates |
| KV read | PASS | different committed histories change the same current input |
| KV explicit commit | PASS | read-many/commit-once and cache-ID assertions |
| Chunk eviction | PASS | exact whole-chunk K/V and variable-length tests |
| Attention sinks | PASS | learned, outside W, survive eviction/reset, receive gradients |
| Per-row reset | PASS | row isolation/output equivalence test |
| No-Mamba minimal isolation | PASS | clean package subprocess and full real ARDY A800 execution |
| Mamba existing regression | PASS | previously closed unchanged Mamba3 CUDA suite |
| Hybrid dual-path activity | PASS | attention/Mamba ablations and both gradient paths |
| DiT forward/backward | PASS | finite output/input/mixer/MLP gradients |
| ARDY adapter | PASS | factory, real CUDA placement, root/body execution |
| Diffusion read-only semantics | PASS | noisy steps have zero commits, including real separated CFG |
| Accepted-final commit | PASS | root/body/controller each exactly one per accepted chunk |
| Rollback on partial failure | PASS | exception and suppressed-commit regression tests |
| Generation entrypoints | PASS | four CLI modes and strict sequence-checkpoint loading |

`PASS` here means architecture/correctness closure. Trained DiT/Minimal
checkpoint and generation evidence is recorded separately in the completed
`docs/stage05_dit_minimal_closeout.md`.

## Explicit non-goals for v1 training

No Deep Forcing, new memory mechanism, Mamba redesign, hybrid training, FA3,
FA4 implementation, simulation, or robot experiment is part of the immediate
Minimal checkpoint stage.
