# Stage 07 C1 Literature Map

Date: 2026-08-15

This is a bounded primary-source review for the Stage-07 H0/H1/H2 search. It
does not claim that a language-model mechanism transfers to motion without
controlled evidence.

## Decision summary

1. Parallel or interleaved Attention--Mamba is already well occupied by
   Hymba, Samba, Jamba, and Zamba. Stage-07 H0 is therefore a control, not C1.
2. Mamba-2/SSD makes matrix-state reads a principled bridge between SSMs and
   attention-like computation, but does not establish a motion-specific
   token--state innovation claim.
3. The H2 equation is a delta-rule memory update. DeltaNet, Gated DeltaNet,
   and Kimi Delta Attention create a direct novelty collision. If H2 works,
   the defensible contribution can only be its specific coupling to accepted
   clean motion chunks and AR-diffusion lifecycle, not invention of the delta
   rule.
4. H1's normalized token-minus-state residual is not the principal operator
   in the reviewed hybrids, but residual/gated fusion itself is generic. A
   claim requires evidence that the explicit delta beats H0 and that frozen
   probes show incremental predictive information.
5. Kernel friendliness favors parallel heads, interleaved blocks, SSD-style
   chunking, or low-rank outer-product delta updates. No kernel work is part of
   Stage 07.

## Primary-source map

### Mamba-2 / Structured State Space Duality

Primary source: Dao and Gu, [Transformers are SSMs: Generalized Models and
Efficient Algorithms Through Structured State Space
Duality](https://arxiv.org/abs/2405.21060), 2024.

- Mechanism: structured semiseparable matrices connect selective SSMs and
  attention-like matrix transformations; Mamba-2 uses an SSD block with a
  recurrent and chunk-parallel dual form.
- Relevant form: a matrix state can be written abstractly as
  `S_t = alpha_t S_(t-1) + v_t k_t^T`, read as `o_t = S_t q_t`.
- Combination: the paper unifies computational forms rather than proposing a
  separate softmax-Attention/Mamba fusion block.
- Motion transfer: Mamba's matrix-shaped state admits token-aligned reads and
  chunk carry, making it appropriate for H2's isolated read/correction test.
- Do not copy: language-model causal tokenization, Mamba-2 replacement, or a
  new Mamba core. This repository uses its existing Mamba3 implementation.
- Novelty risk: SSD and matrix-state read/write are prior art; they cannot be
  claimed as MotionMamba novelty.
- Kernel property: chunkwise dual form is matmul-heavy and hardware friendly.

### Hymba

Primary sources: Dong et al., [Hymba: A Hybrid-head Architecture for Small
Language Models](https://arxiv.org/abs/2411.13676), 2024; official
[NVlabs/hymba implementation](https://github.com/NVlabs/hymba).

- Mechanism: Attention heads and SSM heads operate in parallel inside one
  hybrid-head layer; their outputs are combined. Attention supplies precise
  recall while the SSM summarizes context. The complete model also uses meta
  tokens, cross-layer KV sharing, and partial sliding-window attention.
- Combination: parallel, same-layer hybrid heads rather than a sequential
  prediction-error correction.
- State read/update: standard Mamba/SSM head recurrence; token outputs are
  concatenated/fused with attention-head outputs.
- Motion transfer: strongest structural precedent for H0 and the fair control
  for any parallel MotionMamba candidate.
- Do not copy: meta tokens, language-specific KV sharing, or head allocation
  without an ARDY ablation.
- Novelty risk: very high for any claim phrased only as parallel Attention +
  Mamba.
- Kernel property: parallel head partition is regular, but mixed operators
  still require separate attention and SSM kernels absent dedicated fusion.

### Samba

Primary sources: Ren et al., [Samba: Simple Hybrid State Space Models for
Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522),
ICLR 2025; official [microsoft/Samba](https://github.com/microsoft/Samba).

- Mechanism: layer-wise combination of Mamba and sliding-window attention;
  recurrent state compresses long context and SWA recalls recent content.
- Combination: sequential/interleaved blocks rather than explicit
  token--state subtraction.
- Motion transfer: validates the short-term-attention/long-term-SSM division
  as a generic baseline concept, but Stage 06S already found too little
  out-of-window oracle value for EventMemory routing here.
- Do not copy: unlimited-context language claims or reopen EventMemory.
- Novelty risk: high for merely alternating Mamba with sliding attention.
- Kernel property: each component retains an established efficient kernel;
  the architecture is compositional rather than intrinsically fused.

### Jamba

Primary source: Lieber et al., [Jamba: A Hybrid Transformer-Mamba Language
Model](https://arxiv.org/abs/2403.19887), 2024.

- Mechanism: Transformer and Mamba blocks are interleaved; some layers also
  contain mixture-of-experts feed-forward modules.
- Combination: serial layer scheduling, with no explicit innovation error.
- Motion transfer: H0 structural control only.
- Do not copy: MoE, language-layer ratios, or long-context claims.
- Novelty risk: high for an interleaved Attention--Mamba architecture.
- Kernel property: reuses separate mature kernels and reduces the frequency
  of quadratic attention.

### Zamba

Primary source: Glorioso et al., [Zamba: A Compact 7B SSM Hybrid
Model](https://arxiv.org/abs/2405.16712), 2024; official model family code is
maintained by [Zyphra](https://github.com/Zyphra).

- Mechanism: a Mamba backbone periodically invokes a shared attention module.
- Combination: serial shared-attention blocks, not an error-based state write.
- Motion transfer: parameter-efficient H0 placement control, especially if a
  last-layer interaction is sufficient.
- Do not copy: global parameter sharing or the language training recipe.
- Novelty risk: high for sparse placement/shared Attention on a Mamba backbone.
- Kernel property: few shared attention calls reduce memory and parameter cost.

### DeltaNet and Gated DeltaNet

Primary source: Yang, Kautz, and Hatamizadeh, [Gated Delta Networks:
Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464), 2024;
official [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet).

- Mechanism: the state acts as an associative key--value map. DeltaNet reads
  the old prediction `v_hat_t = S_(t-1) k_t` and writes the innovation:

  ```text
  e_t = v_t - S_(t-1) k_t
  S_t = S_(t-1) + beta_t e_t k_t^T
  ```

  Gated DeltaNet adds data-dependent decay:

  ```text
  S_t = alpha_t S_(t-1)(I - beta_t k_t k_t^T)
        + beta_t v_t k_t^T
  ```

- Combination: the paper also evaluates hybrids with sliding-window attention
  or Mamba-2 layers, but delta correction lives inside a recurrent linear
  memory rather than between a frozen softmax-attention branch and Mamba3.
- Motion transfer: H2 is mathematically justified as online prediction-error
  correction. Accepted-clean-only writes are required by diffusion semantics.
- Do not copy: replace Mamba3 with Gated DeltaNet, claim the delta rule, or
  import language-model decay/gating hyperparameters without evidence.
- Novelty risk: direct and high. MotionMamba can at most claim a validated
  motion/ARD integration or a distinct cross-branch target, not this equation.
- Kernel property: generalized Householder/WY chunkwise formulations reduce
  sequential updates to tensor-core-friendly matrix operations.

### Kimi Delta Attention / Kimi Linear

Primary source: Kimi Team et al., [Kimi Linear: An Expressive, Efficient
Attention Architecture](https://arxiv.org/abs/2510.26692), 2025.

- Mechanism: Kimi Delta Attention extends Gated DeltaNet with finer-grained
  gating and diagonal-plus-low-rank transitions; the full model is a layerwise
  hybrid with latent attention.
- Combination: hybrid linear/latent attention, not a simple output average.
- State read/update: delta-rule finite-state memory with more expressive
  channelwise transition control.
- Motion transfer: confirms that gated innovation writes are a serious H2
  baseline and that stability/gating diagnostics are mandatory.
- Do not copy: DPLR parameterization, MLA, MoE-scale design, or its custom
  kernels during Stage 07.
- Novelty risk: very high for fine-grained gated delta attention.
- Kernel property: explicitly co-designed with a chunkwise DPLR algorithm;
  this belongs to the future Contribution-2 boundary.

## Consequences for the bounded search

```text
H0: necessary control; no novelty claim.
H1: test only explicit normalized innovation and require probe + held-out
    evidence beyond generic fusion.
H2: test as a borrowed delta-rule control/adaptation; never claim the rule.
H3: permitted only if H1 and H2 independently pass.
```

The literature does not justify a fifth family. It also reinforces the
Stage-07 requirement that the real-motion-trained Mamba control precede any
claim about token/state complementarity.
