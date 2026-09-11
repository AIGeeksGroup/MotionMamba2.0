# ARDY DiT + Sequence Kernel Integration

ARDY remains the owner of motion representation, root/body construction,
text features, constraints, diffusion schedule and sampler, autoregressive
windowing, recentering, postprocessing, and visualization. This repository is
a downstream client that replaces only `root_model` and `body_model`.

## Configurations

```text
original  legacy ARDY Transformer baseline, unchanged
dit       DiT modernization baseline; W=0, no sinks, no Mamba
minimal   recommended standard attention baseline; sliding KV + sinks; no Mamba
hybrid    experimental MotionMamba integration scaffold; Mamba enabled
```

`dit` and `minimal` are controlled standard baselines, not MotionMamba paper
contributions. MotionMamba 2.0 begins only with Mamba-specific token/state
modeling. The current Hybrid gated-add path is structural scaffolding and must
not be presented as the finalized Contribution-1 mechanism.

Programmatic installation:

```python
from ardy_adapter import (
    ArdySequenceKernelConfig,
    install_ardy_sequence_kernel,
)

client = install_ardy_sequence_kernel(
    ardy_model,
    ArdySequenceKernelConfig(
        mode="minimal",       # original | dit | minimal | hybrid
        attention_backend="flash2",
        window_size_chunks=4,
        num_sink_tokens=4,
    ),
)

# The client performs a full state reset before each generation request.
motion = client(...same arguments as ardy_model...)
```

The command-line entry point exposes the requested flags:

```bash
PYTHONPATH=src python examples/ardy_sequence_kernel_generate.py \
  --minimal \
  --sequence-checkpoint /path/to/minimal_dit.pt \
  --attention-backend flash2 \
  --prompt "A person jumps."

PYTHONPATH=src python examples/ardy_sequence_kernel_generate.py \
  --hybrid \
  --sequence-checkpoint /path/to/hybrid_dit.pt \
  --prompt "A person jumps."
```

`--original` is the default. `--dit`, `--minimal`, and `--hybrid` are new
architectures, so the launcher requires a checkpoint containing strict
`root_model` and `body_model` state dictionaries. It will not silently use
random mixer weights for a motion-quality claim.

Stage 05 produced real mode-matching DiT and Minimal checkpoints and validated
viewer-compatible generation through 8 chunks. The release-recommended path
is `--minimal`; see `docs/stage05_dit_minimal_closeout.md` for immutable hashes
and measured commands. Hybrid remains structural only and has no Stage-05
trained checkpoint.

Backend policy is independent of these mode definitions: SDPA is the
correctness reference, FA2 is the currently implemented/validated native
development backend, and FA4 is the intended future optimized native backend.
FA4 is not implemented or validated and will not change Minimal's architecture.

## DiT block

`ArdyDiTSequenceBlock` keeps the ARDY block call signature. It converts:

- text features into a mask-aware pooled condition;
- the diffusion timestep into ARDY's existing timestep embedding;
- optional first-heading angle into a learned condition;
- motion token positions through ARDY's existing positional-encoding class.

Text/timestep/heading are combined as a DiT conditioning vector. They are not
prepended to the motion sequence and therefore are never persisted in
cross-window KV or recurrent state. Each `DiTSequenceLayer` performs AdaLN
modulation, a generic `SequenceKernel` mixer, residual gating, and an MLP.
Neither the DiT stack nor ARDY client knows how FlashAttention eviction or
Mamba Triton state is implemented.

## Diffusion memory lifecycle

The installed `ArdyDiffusionMemoryController` wraps the existing bound
`denoising_step`; it does not replace ARDY's sampler logic.

```text
committed state from prior accepted windows
              |
              +--> noisy step T   (read-only)
              +--> noisy step ... (read-only)
              +--> noisy step 1   (read-only)
              +--> noisy step 0   (read-only prediction + sampler result)
                                      |
                                      v
                            accepted clean window
                                      |
                     one extra ARDY projection/root/body pass
                                      |
                 commit generation tokens once in every DiT layer
```

The extra accepted pass reuses the real raw ARDY denoiser so root/body
projections, local-root conversion, masks, and constraint preparation remain
the existing implementation. A commit context selects only
`generation_token_mask`; history tokens, future constraints, text, timestep,
and noisy intermediate motion are excluded from committed memory.

## Reset and state ownership

Every root/body DiT layer has its own `SequenceKernelState`:

- attention cache is explicit recent token K/V;
- hybrid Mamba state is compressed recurrence;
- EventMemory is not inserted into this baseline.

`ArdySequenceKernelClient` fully resets these ephemeral states before each
top-level generation call. `reset_memory(reset_mask)` supports asynchronous
row resets for a continuing batch. States are intentionally not checkpoint
parameters or buffers.

## Constraints and limitations

- Dense ARDY token mode is required for accepted-token selection;
  `sparsify_token_seq=True` is rejected.
- Minimal attention uses exact ARDY token validity masks. Hybrid Mamba uses
  the existing dense recurrence: invalid positions are zeroed but still
  consume recurrent steps, matching the documented limitation of the earlier
  Mamba adapter.
- The adapter reuses compatible frontend/output weights, but DiT/mixer weights
  require a mode-matching trained checkpoint; Stage-05 DiT and Minimal
  checkpoints satisfy this requirement.
- `flash4` remains reserved and the launcher rejects it.
- Any future custom fused kernel belongs to the finalized MotionMamba hybrid
  token/state mechanism, not to Minimal or DiT.
