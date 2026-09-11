# Paper Architecture Scope

This note defines the paper/release boundary. Compatibility code names such as
`MinimalAttentionKernel` describe implementation modules; they do not identify
a custom-kernel paper contribution.

The hierarchy is:

```text
Standard baselines: Original ARDY -> DiT -> Minimal
MotionMamba 2.0:    Minimal + Mamba-specific token/state modeling
                    -> Contribution 1 -> Contribution 2
Evaluation:         motion generation -> long horizon -> benchmark
                    -> simulation -> real robot
```

## 1. What is baseline?

- `--original` is the legacy ARDY Transformer and autoregressive-diffusion
  baseline.
- `--dit` replaces the ARDY Transformer denoiser with a standard DiT-style
  root/body denoiser. It has no persistent sliding K/V, sinks, or Mamba.
- `--minimal` is the recommended standard attention-only baseline:

  ```text
  ARDY autoregressive diffusion
  + DiT
  + native FlashAttention backend
  + chunk-level sliding-window K/V
  + attention sinks
  + no Mamba
  ```

No MotionMamba novelty is claimed for autoregressive diffusion, DiT,
FlashAttention, sliding-window K/V, or attention sinks. SDPA is the numerical
reference; FA2 is the currently implemented and validated native development
backend; FA4 is the intended future optimized native backend and is not yet
implemented or validated. Moving from FA2 to FA4 is a backend replacement, not
an architecture change. The frozen Stage-05 training/comparison used SDPA as
recorded in its closeout; FA2 validation is separate and those checkpoint
weights remain backend-compatible.

## 2. What is MotionMamba 2.0?

MotionMamba 2.0 begins only when Mamba-specific state modeling is added above
the Minimal baseline. Minimal itself is not the MotionMamba method. The current
`--hybrid` mode proves that the existing attention and Mamba paths can coexist,
but it is an integration scaffold—not the finalized paper architecture.

## 3. What is Contribution 1?

Contribution 1 is the future Hybrid Attention–Mamba modeling mechanism:

- Attention supplies token-level representation and explicit interaction.
- Mamba supplies state-level representation and recurrent propagation.
- The research target is the interaction or delta between token and state
  representations.

The mathematical operator is intentionally not defined here. The existing
parallel branches plus scalar gated addition must not be silently presented as
the final Contribution-1 mechanism.

## 4. What is Contribution 2?

Contribution 2 is the efficient integrated/custom kernel corresponding to the
finalized Contribution-1 interaction. It belongs to MotionMamba 2.0, not to
Minimal and not to native FlashAttention by itself. FA4 integration,
Attention–Mamba fusion, fused state/K/V updates, paged-K/V research, and custom
CUDA/Triton work remain separate future work. No such result is claimed by
Stage 05.
