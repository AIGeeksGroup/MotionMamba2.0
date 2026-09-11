# Mamba Transformer Kernel

This repository contains the reusable ARDY attention-only baseline and the
integration scaffold for later MotionMamba research. The paper/release boundary
is explicit: **Minimal is a standard baseline; MotionMamba 2.0 begins only when
Mamba-specific token/state modeling is introduced.** See
`docs/paper_architecture_scope.md` before interpreting the code names as paper
contributions.

## Quick Start

The four stable CLI modes are:

```text
--minimal   recommended standard attention-only AR-diffusion baseline; no Mamba
--dit       DiT modernization ablation; no persistent K/V, sinks, or Mamba
--original  legacy ARDY Transformer compatibility baseline
--hybrid    experimental MotionMamba integration scaffold; contains Mamba
```

Start with `--minimal`. It deliberately combines standard components—ARDY
autoregressive diffusion, DiT, native FlashAttention, chunk-level sliding K/V,
and attention sinks. No architectural novelty is claimed for those components.
The current validated native backend is FA2; FA4 is a future backend validation,
not a completed result or an architecture change.

The repository provides an entirely independent baseline attention backend:

```text
Attention + chunk-level sliding-window KV cache + learned attention sinks
```

The compatibility package/class names retain the historical word `kernel`, but
they refer to an implementation module/backend—not the MotionMamba custom
kernel contribution. This path imports and runs without `mamba_ssm` or ARDY.
Start with
`docs/minimal_attention_kernel.md` and the standalone example:

```bash
PYTHONPATH=src python examples/minimal_attention_kernel.py
```

Minimal usage:

```python
import torch

from minimal_attention_kernel import MinimalAttentionKernel, MinimalAttentionKernelConfig

config = MinimalAttentionKernelConfig(
    d_model=256,
    num_heads=8,
    window_size_chunks=4,
    use_attention_sinks=True,
    num_sink_tokens=4,
    attention_backend="flash2",
)
kernel = MinimalAttentionKernel(config).to(device="cuda", dtype=torch.float16)
cache = kernel.init_cache(batch_size=2)

output, pending, aux = kernel.forward_chunk(x, cache=cache, commit=False)
kernel.commit(cache, pending, chunk_id=0)
```

Stage 2 exposes the attention primitive through one generic interface:

```python
from sequence_kernel import SequenceKernel, SequenceKernelConfig

minimal = SequenceKernel(
    SequenceKernelConfig(mode="minimal", attention=config)
)
state = minimal.init_state(batch_size=2)
output, pending, aux = minimal.forward_chunk(x, state=state)
minimal.commit(state, pending, chunk_id=0)
```

`mode="minimal"` imports and instantiates zero Mamba modules. For
`mode="hybrid"`, pass an existing `Mamba3` instance; the sequence backend
preserves the attention KV cache and Mamba recurrent state as separate state
objects and uses the existing gated-add scaffold. That scaffold is not the
finalized MotionMamba Contribution-1 mechanism. See `docs/sequence_kernel.md` and
`docs/stage02_hybrid_kernel_closeout.md`.

The downstream ARDY client now supports four explicit configurations without
rewriting ARDY's motion representation, diffusion sampler, constraints, or
postprocessing:

```text
--original  legacy ARDY Transformer baseline
--dit       DiT modernization baseline; no persistent K/V, sinks, or Mamba
--minimal   recommended standard attention baseline; sliding K/V + sinks; no Mamba
--hybrid    experimental MotionMamba integration scaffold; Mamba enabled
```

The release-recommended standard attention-only baseline is `--minimal`. It is
trained from the DiT checkpoint and does not instantiate or import Mamba. Use
`--original` for the released legacy comparison, `--dit` to ablate persistent
history and sinks, and `--hybrid` only with a separately trained research
Hybrid checkpoint; Stage 05 did not train Hybrid.

Real DiT and Minimal checkpoints require `--sequence-checkpoint`. For a raw
prompt, provide the exact ARDY LLM2Vec base mirror and both adapters:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/ardy_sequence_kernel_generate.py \
  --minimal \
  --sequence-checkpoint /path/to/sequence_minimal.pt \
  --ardy-root /path/to/ardy \
  --checkpoints-dir /path/to/ardy_checkpoints \
  --model nvidia/ARDY-Core-RP-20FPS-Horizon40 \
  --text-encoder-base-mirror /path/to/unsloth_llama3_8b_instruct \
  --text-encoder-mntp-adapter /path/to/llm2vec_mntp \
  --text-encoder-supervised-adapter /path/to/llm2vec_supervised \
  --prompt "A person walks forward carefully, then turns around." \
  --duration 16 --diffusion-steps 10 \
  --window-size-chunks 4 --sink-tokens 4 \
  --output outputs/minimal_walk_turn.npz
```

For exact cached-feature reproduction, replace the three text-encoder options
with `--text-feature-sample /path/to/cached_sample.pt`. The output `.npz` opens
directly in ARDY's `scripts/visualize.py`. Run
`python examples/ardy_sequence_kernel_generate.py --help` for all four modes,
and see `docs/stage05_dit_minimal_closeout.md` for checkpoint hashes, measured
training/generation results, and exact commands. FA4 and custom fused kernels
are not claimed by this release. The future custom kernel belongs only to the
future MotionMamba token/state interaction, not to Minimal.

The existing state-transition/Mamba packages below remain available, are
unchanged by the generic integration, and are not imported by the minimal
path.

It is intentionally not a research log repo. Planning notes, daily logs, and
experiment memory live elsewhere. This repo should contain only code,
interfaces, tests, and short documentation needed by collaborators who connect
the Mamba operator to training or deployment systems.

## Current Scope

The Mamba side owns, and now IMPLEMENTS (`src/state_memory_operator/`):

- Mamba-3 SISO chunk operator wrapper.
- Explicit recurrent state handoff across chunks, with per-row episode reset.
- State projection from Mamba recurrent state.
- Fixed-capacity Event Memory (FIFO and Participative-eviction backends,
  unified in one class) with per-row async reset and automatic compaction.
- State-based Q/K retrieval + content-based V (validated: state indexes
  reliably but does not carry fine-grained content -- see
  `state_qkv.py`'s docstring).
- Operator diagnostics (`aux` dict): retrieval entropy/weights/sims,
  injection norm, memory usage norm, peak GPU memory.

Downstream systems own:

- Motion / sequence data loading.
- Conditions, targets, and task losses.
- Episode boundaries and reset masks.
- Simulator or robot runtime.
- Benchmark and evaluation scripts.

## Start Here

Install (editable) and run the tests:

```bash
pip install -e .
pytest tests/test_smoke.py -v
```

Run the quickstart example:

```bash
python examples/quickstart.py
```

Minimal usage:

```python
from mamba_ssm.modules.mamba3 import Mamba3
from state_memory_operator import StateMemoryOperatorConfig, StateTransitionMemoryOperator

mamba3 = Mamba3(d_model=128, d_state=64, expand=2, headdim=64, ngroups=1,
                 is_mimo=False, chunk_size=16, layer_idx=0,
                 device="cuda", dtype=torch.bfloat16)
config = StateMemoryOperatorConfig(d_model=128, memory_backend="participative")
operator = StateTransitionMemoryOperator(mamba3, config).cuda()

op_state = operator.init_operator_state(batch_size, device="cuda")
memory = operator.init_memory(batch_size, device="cuda")

out, op_state, memory, aux = operator.forward_chunk_with_memory(
    x, op_state, memory, read_content=content, write_content=content,
)
```

Read the integration contract and ARDY replacement plan for what's validated,
what isn't, and where this connects to ARDY:

```text
docs/mamba_integration_contract.md
docs/ardy_mamba_replacement_plan.md
```

## ARDY Mamba Viser Path

For interactive visualization, do not use the MP4 render helper. Generate a
normal ARDY `.npz` with the trained Mamba denoiser blocks, then open that file
with ARDY's own `scripts/visualize.py` viser viewer:

```bash
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINTS_DIR=/path/to/ardy/checkpoints \
MAMBA_TRITON_OVERLAY=/path/to/mamba_op_triton \
MAMBA_REPO=/path/to/state-spaces-mamba \
python examples/ardy_mamba_generate_for_viser.py \
    --checkpoint /path/to/mamba_blocks_step1999.pt \
    --prompt "A person jumps." \
    --output /tmp/ardy_mamba_vis/jump \
    --launch-viser --port 2334
```

Then open:

```text
http://localhost:2334
```

Progress is tracked with `[ ]` / `[x]` checklists in those docs -- keep them
honest: update a box only when its stated acceptance criterion has actually
been verified, and correct it immediately if you find it's gone stale (this
happened once already, 2026-07-20 -> 2026-07-25, and cost real confusion).

**Current stage: Mamba-side Level 2 operator is implemented, and the first
ARDY denoiser replacement path has reached Milestone C.** `ArdyMambaBlock`
exists as a drop-in root/body denoiser block replacement, a first distillation
checkpoint has been evaluated, and interactive viewing should go through the
`.npz` + ARDY viser path above. MIMO/CUTE and cross-window recurrent memory in
ARDY remain later-stage work.
