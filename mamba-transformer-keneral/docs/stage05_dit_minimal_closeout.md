# Stage 05 DiT + Minimal Release Closeout

Date: 2026-08-14  
Architecture: v1 frozen; no Hybrid training, Mamba redesign, Deep Forcing,
FA3/FA4, kernel fusion, simulation, or robotics work was performed.

## 1. Teacher-data bridge and cache

`examples/build_ardy_teacher_cache.py` runs the released ARDY autoregressive
diffusion path and hooks the actual raw denoiser plus root/body models. It
captures the exact tensors used by inference: 4096-D text feature and mask,
noisy/historical motion, masks, timestep, heading, token index, root/body
conditioning, root/body inputs, and teacher outputs. The encoder is the exact
public ARDY stack: Llama-3-8B-Instruct-compatible Unsloth base, merged MNTP
adapter, then supervised LLM2Vec adapter. It requires exactly one visible GPU
because the upstream encoder otherwise replicates the 8B model.

Teacher:

- model: `nvidia/ARDY-Core-RP-20FPS-Horizon40`
- revision: `abe6c43beb28c867c950acb824b9c4ef3d63fb76`
- installed path: `/root/workspace/ardy_checkpoints/ARDY-Core-RP-20FPS-Horizon40`
- prediction: clean x0, 10 base diffusion steps
- root/body: 20/128 output dimensions, 8 layers each, latent 1024, 8 heads

The tiny correctness cache contains 8 prompts × 10 denoising calls. Reloading
cached tensors and rerunning the teacher gave max absolute error `0.0` and
mean absolute error `0.0`. The full one-chunk caches also replayed exactly.

Cache assets and manifest SHA-256:

```text
/workspace/ardy_stage05/cache/tiny_real/manifest_train.json
  00d993fbab76894edee08e6d2dc46eb3796067b0e254c9134ce92b37907bbdb0
/workspace/ardy_stage05/cache/dit_real/manifest_train_motion.json  (256 prompts)
  ced6ab6753deaa9bda2905f52d0bb526a65c5bbf1f1fea0b5b92af35d2f12e98
/workspace/ardy_stage05/cache/dit_real/manifest_eval.json          (64 prompts)
  28e4ceb8f1ff7ff495e12ad80864f928151eed78ab5f2928452f54bf1fec2e5d
/workspace/ardy_stage05/cache/minimal_seq/manifest_train.json      (64 × 4 chunks)
  5e373ec118a721f1d65fc2501063a642e628c9037ba27372b6b4f75630833f04
/workspace/ardy_stage05/cache/minimal_seq/manifest_eval.json       (16 × 4 chunks)
  b426b46d667eb95503076ad52f80fec99cea9578fa8440789569203c4709a098
```

## 2. Real DiT training and generation

`examples/train_ardy_dit_cached.py` supplies the intentionally small training
surface: cached Dataset/DataLoader, AdamW, bf16 AMP, finite-gradient checks,
train/eval logs, checkpoint save/resume, exact save/load verification, seed,
config dump, and source revision field. Root and body are distilled with only
`MSE(student_prediction, teacher_prediction)` on semantically identical calls.

Tiny overfit (16 cached examples, 100 steps): initial loss `2.847147`, final
loss `0.125667`, a `95.5862%` reduction; gradients were finite, parameters
changed, and save/load passed.

First real DiT config/result:

```text
GPU / precision: 1 × NVIDIA A800-SXM4-80GB / bf16
train/eval:       256 / 64 prompts, all 10 diffusion calls
steps / batch:    3000 / 4
optimizer:        AdamW, lr=1e-4, weight_decay=0, grad_clip=1
seed:             17
initial eval MSE: 2.88560488
final eval MSE:   0.07085973 (root 0.01503226, body 0.05582747)
reduction:        97.5444%
measured time:    315.455 s
peak allocated:   5067.457 MiB
```

Checkpoint:

```text
/workspace/ardy_stage05/outputs/dit_real_v1/sequence_dit.pt
size:   991,502,714 bytes
sha256: 39825d5b2205ee72610dd3769a46d93df6d556cd5e8faed8ccdb5e5232dd99b9
```

The real ARDY generation loop produced finite viewer-compatible `.npz` files
with `posed_joints [T,27,3]`, rotations, root positions, contacts, heading,
fps, and text. Same-seed walk/turn/jump prompts produced distinct motion;
pairwise posed-joint MSE was `0.0156–0.0213`, and the jump sample had `0.1792 m`
vertical range. Hard Gate A passed before Minimal training began.

Representative DiT outputs:

```text
/workspace/ardy_stage05/outputs/generation/dit/walk_forward_1chunk.npz
/workspace/ardy_stage05/outputs/generation/dit/walk_turn_1chunk.npz
/workspace/ardy_stage05/outputs/generation/dit/turn_1chunk.npz
/workspace/ardy_stage05/outputs/generation/dit/jump_1chunk.npz
```

## 3. Sequential Minimal training

The four-chunk cache stores explicit accepted clean chunk boundaries. For each
training case, `examples/train_ardy_minimal_cached.py` resets state, commits
only timestep-zero accepted clean teacher history, reads it while predicting
the current noisy chunk, computes loss/backward/update, and discards projected
K/V before the next optimizer step. It initializes from DiT; the only missing
parameters are the 8 root + 8 body learned sink tensors.

Tiny sequential gate passed: loss fell `0.497237 → 0.310599`; history versus
no-history changed output by mean absolute `0.010629`, different history by
`0.029643`; sink gradients were finite/nonzero; FIFO IDs were `[1,2,3,4]`;
cache size stayed bounded and save/load passed.

First real Minimal config/result:

```text
GPU / precision: 1 × NVIDIA A800-SXM4-80GB / bf16
initialization:   real DiT checkpoint above
train/eval:       64 / 16 four-chunk teacher trajectories
steps:            1500
optimizer:        AdamW, lr=5e-5, weight_decay=0, grad_clip=1
window / sinks:   4 chunks / 4 dedicated tokens per layer
seed:             29
initial eval MSE: 0.55877899
final eval MSE:   0.35955739 (root 0.12524657, body 0.23431082)
reduction:        35.6530%
history effect:   0.013392; different-history effect 0.037554
measured time:    625.925 s
peak allocated:   5089.789 MiB
```

Checkpoint:

```text
/workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt
size:   991,773,226 bytes
sha256: acbdcc78c6594ad690a1d3a76b129665198fd142e487a1eb64bf43163a1b949f
```

## 4. Minimal generation gate and controlled comparison

All modes used prompt `A person walks forward carefully, then turns around.`,
seed 0, 10 diffusion steps, SDPA, identical ARDY representation and no optional
postprocess. Measurements are serial clean-process A800 runs; model/text cache
loading is excluded from `generation_seconds`, while motion generation and CUDA
synchronization are included.

```text
mode      chunks  finite  seconds  peak MiB  boundary joint mean  min chunk speed
original       1  yes      0.6615     883.8  n/a                  0.01381
original       2  yes      1.2844     886.3  0.0141 m             0.00416
original       4  yes      1.5515     892.0  0.0055 m             0.00088
original       8  yes      1.7829     903.0  0.0051 m             0.00083
dit            1  yes      1.0421    2215.1  n/a                  0.01238
dit            2  yes      1.4288    2219.6  0.1222 m             0.01238
dit            4  yes      3.0264    2229.3  0.1021 m             0.01238
dit            8  yes      2.9714    2248.3  0.1754 m             0.01238
minimal        1  yes      1.0458    2213.8  n/a                  0.04341
minimal        2  yes      1.9903    2217.9  0.0859 m             0.02769
minimal        4  yes      3.0245    2226.7  0.0921 m             0.01726
minimal        8  yes      5.4768    2239.8  0.0818 m             0.01098
```

Original becomes nearly static in later chunks in this controlled sample.
DiT remains dynamic but has no persistent history and the largest 8-chunk
boundary error. Minimal remains dynamic through 8 chunks and has lower
8-chunk boundary error than DiT, but its roughly 8.2 cm mean boundary jump is
still a visible first-checkpoint limitation. These measurements establish a
working release baseline, not final motion quality or a broad benchmark.

Minimal memory evidence: accepted commits are exactly `1/2/4/8`; no noisy
denoising call commits; 8-chunk root/body cache IDs are `[4,5,6,7]`, count 4,
oldest 4, newest 7, with 4 dedicated sinks. Hard Gate B passed.

Comparison and outputs:

```text
/workspace/ardy_stage05/outputs/comparison/comparison.json
sha256 c89102e68d7813bbeec8521390e23cb4226f516bb8dde6cb89cf057ef79de016
/workspace/ardy_stage05/outputs/comparison/{original,dit,minimal}/walk_turn_{1,2,4,8}chunk.npz
/workspace/ardy_stage05/outputs/qualitative_panel/turn_around/{original,dit,minimal}/walk_turn_1chunk.npz
/workspace/ardy_stage05/outputs/qualitative_panel/jump/{original,dit,minimal}/walk_turn_1chunk.npz
```

All outputs are directly accepted by ARDY `scripts/visualize.py`. The panel's
turn and jump prompts are finite and non-degenerate in all three modes. Text
conditioning is numerically active; collaborator viewer review is still
required before making subjective visual-quality claims.

## 5. Release interface and reproduction

`--original`, `--dit`, `--minimal`, and `--hybrid` remain mutually exclusive.
`--dit` and `--minimal` enforce a mode-matching sequence checkpoint. Minimal
loads no Mamba module; Hybrid is structurally available but intentionally has
no Stage-05 trained checkpoint. The recommended baseline is `--minimal`.

Cache construction (change split/prompt/output for train, eval, or four chunks):

```bash
cd /root/workspace/mamba_transformer_keneral
CUDA_VISIBLE_DEVICES=1 /dev/shm/ardy_env/bin/python examples/build_ardy_teacher_cache.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --teacher-model Ardy-Core-RP-20FPS-Horizon40 \
  --teacher-revision abe6c43beb28c867c950acb824b9c4ef3d63fb76 \
  --prompt-file configs/ardy_mamba_distill_prompts_v4_train_256.txt \
  --output-dir /workspace/ardy_stage05/cache/dit_real --split train \
  --num-chunks 1 --diffusion-steps 10 --seeds 0 \
  --text-encoder-base-mirror /workspace/ardy_stage05/hf/unsloth_llama3_8b_instruct \
  --text-encoder-mntp-adapter /workspace/ardy_stage05/hf/llm2vec_mntp \
  --text-encoder-supervised-adapter /workspace/ardy_stage05/hf/llm2vec_supervised \
  --verify-samples 16 --device cuda:0 --text-encoder-device cuda:0
```

The first train cache was generated before comment-line filtering was added to
the prompt reader. Its immutable `manifest_train_motion.json` was therefore
created non-destructively with:

```bash
/dev/shm/ardy_env/bin/python examples/filter_ardy_teacher_cache_manifest.py \
  --input /workspace/ardy_stage05/cache/dit_real/manifest_train.json \
  --output /workspace/ardy_stage05/cache/dit_real/manifest_train_motion.json
```

Current cache builds skip `#` lines directly and do not need this compatibility
step.

DiT training:

```bash
CUDA_VISIBLE_DEVICES=1 /dev/shm/ardy_env/bin/python examples/train_ardy_dit_cached.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --train-manifest /workspace/ardy_stage05/cache/dit_real/manifest_train_motion.json \
  --eval-manifest /workspace/ardy_stage05/cache/dit_real/manifest_eval.json \
  --output-dir /workspace/ardy_stage05/outputs/dit_real_v1 \
  --max-steps 3000 --batch-size 4 --learning-rate 1e-4 \
  --precision bf16 --seed 17 --eval-every 250 --save-every 1000 \
  --max-eval-batches 160
```

Minimal training:

```bash
CUDA_VISIBLE_DEVICES=1 /dev/shm/ardy_env/bin/python examples/train_ardy_minimal_cached.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --dit-checkpoint /workspace/ardy_stage05/outputs/dit_real_v1/sequence_dit.pt \
  --train-manifest /workspace/ardy_stage05/cache/minimal_seq/manifest_train.json \
  --eval-manifest /workspace/ardy_stage05/cache/minimal_seq/manifest_eval.json \
  --output-dir /workspace/ardy_stage05/outputs/minimal_real_v1 \
  --max-steps 1500 --max-train-trajectories 64 --max-eval-trajectories 16 \
  --learning-rate 5e-5 --window-size-chunks 4 --sink-tokens 4 \
  --precision bf16 --seed 29 --eval-every 150 --save-every 500
```

Controlled comparison:

```bash
CUDA_VISIBLE_DEVICES=1 /dev/shm/ardy_env/bin/python examples/run_ardy_release_comparison.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --model nvidia/ARDY-Core-RP-20FPS-Horizon40 \
  --text-feature-sample /workspace/ardy_stage05/cache/dit_real/train/prompt_00009_seed_0000000000.pt \
  --prompt "A person walks forward carefully, then turns around." \
  --dit-checkpoint /workspace/ardy_stage05/outputs/dit_real_v1/sequence_dit.pt \
  --minimal-checkpoint /workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt \
  --output-dir /workspace/ardy_stage05/outputs/comparison \
  --diffusion-steps 10 --seed 0 --horizons 1 2 4 8 \
  --window-size-chunks 4 --sink-tokens 4
```

Viewer:

```bash
/dev/shm/ardy_env/bin/python /root/workspace/ardy/scripts/visualize.py \
  /workspace/ardy_stage05/outputs/comparison/minimal/walk_turn_8chunk.npz \
  --port 2334
```

## 6. Regression, provenance, and limitations

Final local regression: `49 passed, 13 skipped`. The only new audit bug was an
unconditional package-level Mamba import that prevented standalone EventMemory
use without `mamba_ssm`; it is fixed with lazy Mamba-dependent exports and a
subprocess regression. No model architecture changed.

Implementation commit: `3f6dcd6b5dfa845fcd0db779e02da4df452770c5` (the following
provenance-only documentation commit records this immutable hash). The remote
training checkout was a code copy without `.git`, so original training configs
record `git_commit: UNAVAILABLE`; checkpoint/cache hashes above bind the exact
artifacts, and `/workspace/ardy_stage05/provenance.json` records the final local
source commits after closeout.

Remaining limitations: the cache is teacher-generated rather than the private
original ARDY training corpus; Minimal uses teacher-forced clean history, not
student rollout; evaluation covers a small fixed prompt panel, not a broad
benchmark; current Minimal boundaries remain visibly imperfect; optional
`motion_correction` was unavailable on the A800, so controlled outputs used
`--no-postprocess`; no trained Hybrid checkpoint is claimed.

## Post-Stage-05 paper/release semantics clarification

This clarification changes positioning only. It does not change the frozen
architecture, checkpoints, hashes, commands, or numerical evidence above.

1. Minimal is the recommended standard attention-only baseline, not a proposed
   custom kernel or a MotionMamba contribution. Its AR diffusion, DiT, native
   attention backend, sliding-window K/V, and attention sinks carry no novelty
   claim.
2. DiT is baseline modernization that isolates the replacement of ARDY's
   custom Transformer denoiser with a standard DiT-style denoiser.
3. SDPA is the correctness backend. FA2 is the currently implemented and
   validated native development backend. FA4 is the intended future optimized
   native backend, but it is not implemented or validated; it is a backend
   replacement rather than an architecture change. The frozen Stage-05
   training/comparison used SDPA exactly as recorded above; the checkpoint
   weights remain compatible with the separately validated FA2 backend.
4. MotionMamba 2.0 begins only when Mamba-specific state modeling and the
   token/state interaction are introduced above Minimal.
5. The future integrated/custom kernel belongs to that MotionMamba hybrid
   mechanism, not to Minimal.
6. Existing `--hybrid` code is an integration scaffold with generic gated-add
   fusion. It is not the finalized Contribution-1 architecture and no Stage-05
   Hybrid checkpoint is claimed.

The concise authoritative boundary is `docs/paper_architecture_scope.md`.

DIT_AND_MINIMAL_COMPLETE
