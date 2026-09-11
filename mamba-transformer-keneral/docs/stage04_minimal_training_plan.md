# Stage 04 Minimal Training Plan

Date: 2026-08-14  
Current status: **SUPERSEDED BY COMPLETED STAGE 05**  

This file records the pre-training plan and its then-current blockers. Those
blockers were subsequently closed: the exact ARDY text bridge, real teacher
cache, cached DiT trainer, sequential Minimal trainer, real checkpoints, and
1/2/4/8-chunk generation comparison are complete. See
`docs/stage05_dit_minimal_closeout.md` for current evidence. The remaining
sections are retained as the historical gate definition, not current status.

## 1. Current verified state

Architecture v1 is frozen in `docs/architecture_v1_freeze.md`:

- `original`: released ARDY Transformer;
- `dit`: ARDY diffusion + DiT, W=0, no sinks, no Mamba;
- `minimal`: DiT + whole-chunk sliding K/V + learned sinks, no Mamba;
- `hybrid`: Minimal + existing Mamba path (not a Stage-04 training target).

The selected local Mamba-free architecture suite passes (44 passed, 2 CUDA
skips). The ARDY controller/factory suite passes (6 passed, 1 CUDA skip).
The same controller/factory suite is 7/7 on A800, including full replacement-
block CUDA placement and execution.
Real A800 Minimal separated-CFG execution passes with two noisy steps, zero
noisy commits, and exactly one accepted root/body commit. The exact FA2 case
also passes on A800. The unchanged real Mamba hybrid CUDA regression was
re-audited on RTX 3090: 1 passed in 35.22 s. Stage 04 does not modify the
upstream Mamba design.

## 2. Existing training infrastructure audit

Answer: **NO**, the existing ARDY training script cannot directly train the
new DiT/Minimal adapter.

The public ARDY checkout at `/root/workspace/ardy` is an inference release. It
contains model loading, diffusion `q_sample`, DDIM sampling, generation,
postprocessing, and visualization, but no motion Dataset/DataLoader, optimizer,
mixed-precision trainer, distributed launcher, checkpoint resume, validation
loop, or original training loss implementation.

This repository has `examples/train_ardy_mamba_distill.py`, a locally authored
single-GPU pseudo-data trainer for the old `ArdyMambaBlock`. It does not install
or optimize `ArdyDiTSequenceBlock`, does not train sequential K/V, and its loss
is x0 MSE against teacher-generated clean motion rather than direct root/body
teacher-output distillation.

Stage 04 added `examples/train_ardy_dit_distill.py` only for the cheap Phase-A
gate. It loads the real teacher, captures semantically identical root/body
inputs and targets, trains DiT with W=0, enforces cache reset after every
optimizer step, and validates checkpoint round-trip. It deliberately uses one
synthetic shape-correct batch and is not a complete training system.

Minimal remaining integration work:

1. cache real text features and teacher-generated or real motion examples;
2. generalize the Phase-A gate to shuffled train/eval samples and resume;
3. add the released x0 diffusion objective around the installed DiT blocks;
4. add a sequential-chunk sampler and per-trajectory cache lifecycle;
5. add validation generation/NPZ/metadata logging.

Do not rewrite ARDY's sampler, root/body conversion, or representation.

## 3. Available data

No real motion-training dataset or split was found in the local public ARDY
checkout, the current repository, or the accessible A800 ARDY checkout.

Available text-only prompt assets:

```text
train: configs/ardy_mamba_distill_prompts_v4_train_256.txt  (256 prompts)
eval:  configs/ardy_mamba_distill_prompts_v4_eval_64.txt    (64 prompts)
```

These are prompts, not motion examples. The old Mamba script can ask the
teacher to create one clean x0 window per prompt, but the A800 teacher was
intentionally loaded with `text_encoder=False` for cheap gates. No cached real
4096-dimensional text features or reachable text-encoder service was found.

The completed DiT gate used one deterministic synthetic batch with legal ARDY
shape: B=1, 2 history tokens, 10 generation tokens (40-frame horizon), hybrid
token dim 148, one text token of dim 4096. This proves mechanics only.

## 4. Available teacher checkpoint

```text
Hugging Face model: nvidia/ARDY-Core-RP-20FPS-Horizon40
revision: abe6c43beb28c867c950acb824b9c4ef3d63fb76
A800 path: /root/workspace/ardy_checkpoints/ARDY-Core-RP-20FPS-Horizon40
denoiser.safetensors: 623,324,336 bytes
sha256: 1019d0bf269cf8d1b3e3e9b4a384a58c112672959b071279ddb65814d77660cd
tokenizer.safetensors: 141,746,112 bytes
sha256: 58a887e299a3a6779b5b4ff361b452c4349d45b513b5957f2a961d393623abcd
```

Teacher configuration:

```text
root output shape per token: 20
body output shape per token: 128
hybrid token dim: 148
root Transformer layers: 8
body Transformer layers: 8
latent dim: 1024
FF dim: 2048
heads: 8
text conditioning: [B,1,4096]
diffusion base steps: 10
prediction parameterization: clean x0
```

The original training loss is not shipped. Clean-x0 prediction is verified by
the released sampler/API and is the objective used by the existing local
pseudo-data script; any claim about additional original losses would be
speculation.

Smoke evidence on A800:

```text
one raw teacher forward: finite [1,10,148], mean square 0.4591548
one model.denoising_step: finite [1,10,148], mean square 0.7696486
teacher-only peak allocated: 883.827 MiB
```

## 5. Training phases and hard gates

### A0 — completed distillation engineering gate

Teacher and student receive identical captured root/body motion input, text
feature, text mask, timestep, heading, token index, and validity mask. W=0,
sinks off, Mamba absent. Loss is only:

```text
MSE(student_root, teacher_root) + MSE(student_body, teacher_body)
```

Result on one legal fixed batch, 20 AdamW steps at 2e-4, bf16 autocast:

```text
initial MSE:       2.7909465
final MSE:         0.1522279
relative reduction: 94.5457%
gradients:         finite
parameters:        changed
checkpoint:        saved and exactly reloaded
peak allocated:    5939.409 MiB
measured wall time: 17.688 s
```

PASS. The transient step-2 spike means this is an overfit gate, not a selected
production learning rate.

### A1 — real prompt distillation

Materialize the 256/64 prompt split with real cached text features. For each
teacher-generated clean window, sample legal noisy x/t inputs and store or
recompute teacher root/body targets. Train DiT at W=0. Begin with plain MSE;
do not add representation losses unless the simple objective fails.

PASS requires finite training/eval loss, substantial held-out error reduction,
resume-safe checkpointing, and exact root/body input-semantic parity. If a
16–128-example subset cannot overfit, stop before scaling.

### B — DiT x0 diffusion fine-tuning

Install `mode=dit`, keep W=0, sample t from all 10 base steps, generate x_t via
the released `Diffusion.q_sample`, predict clean x0, and minimize x0 MSE. Keep
ARDY root/body preparation, masks, conditioning, and constraint behavior.

PASS requires a real `--dit` checkpoint and finite short generation through
the real ARDY loop, saved `.npz`, correct shapes, and visualization without
catastrophic collapse. Do not enable K/V before this passes.

### C — Minimal sequential K/V training

Start from the accepted DiT checkpoint and install `mode=minimal`. Use
sequential chunks C0,C1,... with clean/ground-truth history first:

```text
fresh cache -> commit clean C0 -> predict noisy C1 -> commit clean C1 -> ...
            -> loss -> backward -> optimizer.step() -> discard cache
```

Never preserve projected K/V across optimizer steps. Compare the same current
noisy chunk under different previous chunks and require different predictions.
Require nonzero finite sink gradients, bounded cache IDs/count, stable memory,
and no NaN. Generated-history exposure, rollout fine-tuning, and Deep Forcing
are later experiments, not this baseline.

### D — short autoregressive validation

Generate 1, 2, 4, and 8 chunks with a real Minimal checkpoint. Save `.npz` and
viewer output. Record success, NaN, boundary discontinuity, cache size,
oldest/newest chunk, sink count, latency, and peak memory. Then compare only
`original`, `dit`, and `minimal` on matched prompts/seeds/horizons/diffusion.

Hybrid training remains prohibited until this gate passes.

## 6. Resource plan based on measurements

All initial gates use one NVIDIA A800-SXM4-80GB and bf16 autocast. Runtime is
not projected where it has not been measured.

```text
A0 fixed-batch distill:  1 GPU, B=1, bf16, measured 5.80 GiB, 17.688 s/20 steps
A1 prompt distill:      1 GPU, B=1 start, bf16, use 5.80 GiB as measured floor;
                        measure before increasing batch, 8 GiB initial budget
B DiT x0 objective:     1 GPU, B=1 start, bf16, 5.80 GiB measured model/optimizer
                        reference; 8 GiB initial budget pending first real batch
C 2-chunk Minimal:      1 GPU, B=1, bf16, estimate 12–16 GiB by applying a
                        2x trajectory-activation allowance to the 5.80 GiB
                        measured gate; must record actual peak before scaling
D inference validation:1 GPU, B=1, bf16/fp32 model path; real untrained
                        Minimal 3-step CFG gate measured 1.892 GiB peak;
                        full text encoder/full horizon still needs measurement
```

No multi-GPU job is authorized by this plan. DDP is deferred until a 1-GPU
real-data smoke passes and actual memory/throughput are recorded.

## 7. Reproducible commands

Local architecture suite:

```bash
cd /home/plote/mambamotion/mamba_project/mamba_transformer_keneral
PYTHONPATH=src /home/plote/miniconda3/envs/sana/bin/python -m pytest -q \
  tests/test_minimal_attention_kernel.py \
  tests/test_sequence_kernel.py \
  tests/test_dit_sequence_kernel.py \
  tests/test_ardy_dit_sequence.py
```

A800 FA2 exact acceptance:

```bash
cd /root/workspace/mamba_transformer_keneral
CUDA_VISIBLE_DEVICES=0 /dev/shm/RSWM/envs/rswm/bin/python \
  benchmarks/verify_flash2_stage04.py --iterations 20 --dtype fp16
```

Real teacher + full Minimal CFG/commit/isolation gate:

```bash
cd /root/workspace/mamba_transformer_keneral
CUDA_VISIBLE_DEVICES=0 /dev/shm/ardy_env/bin/python \
  benchmarks/verify_ardy_minimal_stage04.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --teacher-model ARDY-Core-RP-20FPS-Horizon40
```

Completed Phase-A overfit/save/reload gate:

```bash
cd /root/workspace/mamba_transformer_keneral
CUDA_VISIBLE_DEVICES=0 /dev/shm/ardy_env/bin/python \
  examples/train_ardy_dit_distill.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --teacher-model ARDY-Core-RP-20FPS-Horizon40 \
  --teacher-revision abe6c43beb28c867c950acb824b9c4ef3d63fb76 \
  --steps 20 --precision bf16 \
  --output outputs/stage04/dit_distill_smoke.pt \
  --metrics-output outputs/stage04/dit_distill_smoke_metrics.json
```

## 8. Exact next command and blockers

Do not launch a longer optimizer job yet. The next command is the data gate,
not a synthetic step-count increase:

```bash
test -f /root/workspace/ardy_training_cache/v4_train_256.pt && \
test -f /root/workspace/ardy_training_cache/v4_eval_64.pt
```

Those files do not currently exist, so A1 is blocked. They must contain prompt,
real `[1,4096]` text features/mask, teacher-generated or real clean x0, and
teacher/config/revision provenance. After an adapter-aware dataset builder is
added, its exact generation command must be recorded here before execution;
inventing a non-existent command would not be reproducible.

Remaining blockers:

1. no real motion training dataset/splits;
2. no cached real text features and no verified text encoder endpoint/assets;
3. no full DiT dataset trainer/eval/resume loop beyond the completed A0 gate;
4. no sequential-chunk Minimal trainer yet;
5. no trained DiT checkpoint, so `.npz` quality/generation gates cannot start.
6. the architecture/training changes are not committed yet; the first real
   experiment must use a clean committed revision and record its hash.

## 9. Later Hybrid initialization plan (not executed)

Historical note: this initialization sketch predates the final paper boundary.
It must not be executed or presented as the MotionMamba method until the
Contribution-1 token/state interaction is formally finalized. The current
gated-add Hybrid remains only an integration scaffold.

After Minimal passes D: load the Minimal checkpoint into Hybrid, initialize the
existing Mamba/fusion contribution near zero so `Hybrid(step 0) ≈ Minimal`,
train Mamba/fusion, then jointly fine-tune. No implementation or Hybrid
training is authorized during the current stage.
