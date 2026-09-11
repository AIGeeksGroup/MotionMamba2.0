# MotionMamba 2.0 — Baseline + C1

This branch packages the ARDY-compatible baseline interfaces and the current
Contribution-1 research scaffold. Model weights and private training artifacts
are intentionally not included; request the appropriate checkpoint separately.

## Interfaces

The generation entrypoint supports four stable modes:

```text
--original  Legacy ARDY Transformer denoiser.
--dit       DiT denoiser without persistent cross-chunk KV or Mamba.
--minimal   Recommended attention-only streaming baseline; Mamba is disabled.
--hybrid    Experimental Attention-Mamba integration scaffold; Mamba is enabled.
```

The corresponding implementation and examples live in
`mamba-transformer-keneral/`. Run its generation entrypoint with one mode and
the separately supplied sequence checkpoint, for example:

```bash
python mamba-transformer-keneral/examples/ardy_sequence_kernel_generate.py \
  --minimal \
  --sequence-checkpoint /path/to/checkpoint.pt \
  [existing ARDY generation arguments]
```

Use `--dit` or `--original` in the same position for those baselines. The
`--hybrid` path additionally requires the existing Mamba runtime and is a
research scaffold, not the final frozen paper operator.

## Minimal streaming contract

`--minimal` combines AR diffusion, DiT, a native attention backend,
chunk-level sliding-window KV, and attention sinks. It contains no Mamba
module. During autoregressive diffusion:

```text
noisy denoising step       -> read memory only
accepted clean chunk       -> commit exactly once
new sequence / selected row -> reset its cache state
```

Cache history contains accepted clean chunks only, stays bounded by the
configured chunk window, and keeps attention sinks separate from evictable
history. Root/body commits are transactional: partial failure rolls both back.

## Repository layout

- `ardy/`: ARDY base and the previously reconstructed Mamba integration.
- `mamba-transformer-keneral/`: complete DiT/Minimal implementation, C1
  experiments, tests, examples, and technical documentation.

No checkpoints, generated motion, W&B runs, teacher caches, or private server
paths should be committed to this repository.
