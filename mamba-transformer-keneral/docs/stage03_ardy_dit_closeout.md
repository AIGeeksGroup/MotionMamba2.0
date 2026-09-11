# Stage 03 Closeout — ARDY / DiT / Autoregressive Diffusion

Date: 2026-08-14.

Post-closeout scope note: DiT and Minimal are now classified as standard
baselines. The existing Hybrid path is an integration scaffold, not the final
MotionMamba contribution. This note does not alter the historical Stage-03
status/evidence; current trained results are in Stage 05 and the authoritative
paper boundary is `docs/paper_architecture_scope.md`.

Overall status: **FUNCTIONAL INTEGRATION PASS; TRAINED END-TO-END MOTION RUN
BLOCKED BY DATA/TRAINED CHECKPOINT.**

The architecture, API, runtime lifecycle, unit tests, real ARDY-backbone
compatibility, and real Mamba-3 CUDA path are implemented and validated. No
trained checkpoint exists for the new DiT mixers in this workspace. A complete
ARDY runtime and released teacher are now verified on A800, but motion quality
and a generated `.npz` from a trained DiT/Minimal checkpoint are not claimed.

## Preserved ARDY infrastructure

Status: **PASS (code-path audit and adapter boundary)**.

The integration mutates only the in-memory raw denoiser's `root_model` and
`body_model`. ARDY continues to own:

```text
motion representation and hybrid conversion
two-stage root/body preparation and coupling
text encoder/features
motion/future constraints
diffusion schedule, CFG wrapper, and sampler
autoregressive window loop and recentering
postprocessing and viser-compatible NPZ output
```

Neither inspected ARDY checkout was modified on disk.

## Reusable DiT denoiser

Status: **PASS**.

`dit_sequence_kernel.DiTSequenceStack` is ARDY-independent. Every layer uses
AdaLN conditioning, residual gating, MLP, and only the public generic
`SequenceKernel` interface. It has no direct FlashAttention, KV-eviction,
Mamba-3, Triton, diffusion, or motion representation logic.

`ArdyDiTSequenceBlock` adapts ARDY's real block signature into the generic
stack. Text/timestep/heading become DiT conditioning rather than cacheable
prefix tokens.

## Final configurations

```text
PASS  --original: unchanged released ARDY model path
PASS  --dit:      DiT + attention with W=0, sinks off
PASS  --minimal:  DiT + Stage-1 attention + W-chunk KV + sinks, zero Mamba
PASS  --hybrid:   minimal path + existing Mamba3 per layer
PASS  shared ARDY data/sampler/generation/postprocess launcher
PASS  flash4 is rejected rather than claimed
```

The CLI is `examples/ardy_sequence_kernel_generate.py`. New modes require a
strict mode-specific sequence checkpoint; this prevents accidental random
weight generation from being presented as a valid model.

## Mandatory diffusion cache semantics

Status: **PASS**.

The controller observes ARDY's existing `denoising_step` boundary:

- all calls with sampling `t > 0` are read-only;
- the original `t=0` denoising/sampler call is also read-only;
- its accepted result is passed once through the real raw ARDY root/body
  projections under an accepted-commit context;
- only tokens selected by `generation_token_mask` are committed;
- every root/body DiT layer commits exactly one attention candidate and, in
  hybrid mode, exactly one recurrent-state transition;
- commit contexts are cleared in `finally`, so an exception cannot arm a
  later noisy call accidentally;
- root/body state snapshots are restored if either accepted pass fails, so a
  window cannot remain half committed.
- each root and body count must increase by exactly one; a missing-but-equal
  pair of commits is rejected and rolled back;
- ARDY separated CFG keeps persistent memory at batch B and evaluates each
  of its 3B guidance branches as an independent read of the same B-row state.

The controlled two-window test verifies zero chunks after noisy steps
`3,2,1`, one chunk only after `0`, then chunk IDs `(0,1)` after the second
accepted window. It also verifies root/body commit counts and full reset.

## Validation evidence

```text
selected Mamba-free architecture suite: 44 passed, 2 CUDA skips
ARDY controller/factory suite: 6 passed, 1 CUDA skip
ARDY controller/factory suite on A800: 7 passed
real ARDY backbone minimal block: PASS, output [2,6,20], one selected commit
real ARDY Transformer block -> adapter factory: PASS, two DiT layers
real RTX 3090 ARDY-DiT hybrid test: 1 passed in 67.80s
real A800 full ARDY Minimal separated-CFG gate: PASS
```

The Stage-04 audit fixed two real full-model integration defects: replacement
frontends remained on CPU after installing into a CUDA ARDY model, and CFG's
3B read batch conflicted with the accepted B commit batch. The A800 gate now
proves zero noisy commits, exactly one root/body accepted commit, finite output,
CUDA placement, and no `mamba_ssm` import/module.

The real hybrid CUDA test uses the existing Mamba3 Triton path and checks:

- three noisy read-only forwards leave every cache empty;
- finite `[B,T,output_dim]` output and backward;
- gradients reach input, Mamba projection, and attention projection;
- accepted mask commits exactly eight generation tokens once;
- real `Mamba3State.ssm_state` has `[2,4,32,32]` shape;
- per-row reset clears only the selected attention/Mamba row.

## Acceptance matrix

```text
PASS     Stage-1/2 kernel semantics remain downstream abstractions
PASS     ARDY is a client; generic kernels have no ARDY dependency
PASS     useful ARDY pipeline components are reused
PASS     reusable DiT stack consumes SequenceKernel
PASS     minimal mode has zero Mamba modules/imports
PASS     hybrid uses the existing Mamba implementation
PASS     noisy diffusion intermediates never commit
PASS     accepted clean generation tokens commit exactly once
PASS     attention KV and Mamba state stay distinct per layer
PASS     original/dit/minimal/hybrid configuration surface exists
PASS     CPU forward/backward/reset/factory/commit tests
PASS     real ARDY backbone compatibility
PASS     real hybrid Mamba CUDA forward/backward/state/reset
BLOCKED  trained Minimal DiT checkpoint is absent
BLOCKED  trained-checkpoint ARDY generation + motion-quality evaluation
PASS     exact requested FA2 CUDA parity/backward/latency/memory point
```

## Exact blockers for a full motion result

1. The new DiT/minimal/hybrid mixers are not checkpoint-compatible with the
   released Transformer mixer and require training/distillation.
2. The public ARDY release has no motion training dataset or training pipeline;
   only inference and locally authored teacher-pseudo-data scaffolding exist.
3. Real prompt embeddings/text encoder assets have not yet been materialized
   for a meaningful distillation dataset.

The next valid step is training one mode-specific root/body checkpoint, then
running the provided launcher on a complete ARDY CUDA environment. No further
architectural rewrite is required for that step.
