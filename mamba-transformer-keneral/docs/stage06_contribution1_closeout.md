# Stage 06 Contribution-1 Closeout

Date: 2026-08-14
Decision: stopped at the real-cache tiny-overfit gate

## 1. Hypothesis and bounded formulations

The tested hypothesis was that Mamba recurrent state is an address/dynamics
signal while cached attention values remain token/content payload. For current
chunk `t`, only committed `s_(t-1)` and `delta s_(t-1)` may affect retrieval;
the candidate state produced from chunk `t` becomes readable only after the
accepted clean commit.

Two direct-KV variants were implemented:

```text
state-query:
q = Wq z + tanh(beta) Ws [s_(t-1); delta s_(t-1)]

state-KV modulation:
r_j = mean_heads cosine(Ws [s_(t-1); delta s_(t-1)], mean(K_chunk_j))
K'_j = (1 + tanh(beta) tanh(r_j)) K_j
V'_j = V_j
```

Both initialize with `beta=0`, so their output is exactly the same as the
same-weight Minimal attention path. The existing gated-add Hybrid remains the
naive control and is not treated as Contribution 1.

## 2. Reused components

- Frozen Stage-05 Minimal attention, sinks, and chunk sliding K/V.
- Existing `Mamba3ChunkWrapper` and recurrent state handoff.
- Existing `StateProjector` principle and historical state-as-index evidence.
- Existing DiT stack, ARDY root/body adapter, diffusion controller, rollback,
  and real sequential teacher cache.
- Parent checkpoint:
  `/workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt`
- Parent SHA-256:
  `acbdcc78c6594ad690a1d3a76b129665198fd142e487a1eb64bf43163a1b949f`

No EventMemory was added to the proposed operator. It remains intact for a
future ablation.

## 3. New implementation

- `TokenStateInteraction` implements the two bounded direct-KV variants.
- Durable and pending sequence state now explicitly carry projected state and
  state delta.
- The attention backend accepts an optional state query offset or per-chunk
  historical-key scale. It never synthesizes values from Mamba state.
- ARDY exposes the interaction choice internally through Hybrid configuration;
  the four public release modes and their Stage-05 meanings are unchanged.
- Accepted teacher-forced history can use `detach=False` within one trajectory;
  all cache/state is discarded before the optimizer update is reused.
- `examples/train_ardy_c1_cached.py` loads the frozen Minimal checkpoint,
  records its hash, uses separate base/new-parameter learning rates, checks
  gradients, evaluates, and writes pilot checkpoints/results.

Concrete mixed-precision bugs found during the real run were fixed without
changing the model formulation:

1. autocast normalization output is explicitly aligned to the attention
   projection dtype at the mixer boundary;
2. Minimal/DiT and attention remain fp32 while only Mamba uses bf16;
3. the Mamba cast/state dtype is anchored to `in_proj.weight`, not an internal
   fp32 scalar parameter;
4. tiny C1/fusion scalar coefficients remain fp32 so optimizer updates do not
   quantize away;
5. differentiable persistent K/V history is built outside autocast so its
   dtype matches the fp32 cache.

## 4. Unit and CUDA regression

Local final suite:

```text
58 passed
15 skipped (environment-gated CUDA/Mamba tests)
```

The focused tests establish:

- Minimal default behavior is unchanged and remains Mamba-free.
- At zero state coefficient, both proposed variants are exactly equal to the
  same-weight Minimal attention output, including committed history.
- Current candidate state cannot leak into current output.
- State affects retrieval while pending values remain token-derived.
- `detach=False` history sends finite gradients to Mamba, StateProjector,
  interaction, and attention.
- State-only and state+delta are explicit configurations.
- Per-row reset covers raw state, projected summary, delta, and K/V.

Real A800 tests used official Mamba3 source with its required isolated Triton
3.5 runtime. Focused C1 query and KV forward/backward tests passed, as did the
existing naive Hybrid and ARDY lifecycle tests. The consolidated result was:

```text
4 passed in 144.55s
```

The code does not modify Mamba3 itself.

## 5. Minimal initialization

Checkpoint loading accepts only the expected new Mamba, projector, interaction,
or naive-gate keys as missing. All compatible root/body DiT, attention, sink,
MLP, and frontend parameters come from Stage 05. For the proposed variants,
`beta=0` gives exact Minimal output at step 0. The observed one-trajectory
initial eval loss was `0.3000332481` for both state-only and state+delta.

## 6. Tiny-overfit evidence

The real cache was:

```text
/workspace/ardy_stage05/cache/minimal_seq/manifest_train.json
/workspace/ardy_stage05/cache/minimal_seq/manifest_eval.json
```

The definitive matched gate used one fixed real sequential case, 100 updates,
bf16 Mamba with fp32 Minimal/attention, base learning rate `1e-3`, new-parameter
learning rate `5e-3`, seed 61, and the same Minimal initialization.

```text
variant                  first loss   final loss   reduction   final eval
state-query, state only  0.6960473    0.6678084     4.06%      0.8190935
state-query, state+delta 0.6960473    0.6684049     3.97%      0.7930969
```

Both runs had finite gradients in Mamba, StateProjector, interaction, and
attention; coefficients moved measurably away from zero; peak allocated memory
was about 5.97--6.01 GiB. Nevertheless, neither run substantially overfit the
single fixed case, and held-out eval degraded sharply.

Earlier lower-learning-rate 25/50-step diagnostics also plateaued around
`0.6675--0.6689` and did not distinguish naive, state-only, state+delta, or
state-KV. Those runs revealed and motivated the precision fixes above, but are
not used to claim an architectural result.

The definitive naive and state-KV 100-step jobs were interrupted by external
GPU memory contention during Mamba autotune/Adam-state allocation. They were
not rescheduled after the proposed state-query variants failed the mandatory
tiny-overfit gate, because the task explicitly requires stopping before scale.

Pilot artifact directories (not release checkpoints):

```text
/workspace/ardy_stage06/outputs/state_only_overfit_v4/
/workspace/ardy_stage06/outputs/state_delta_overfit_v4/
/workspace/ardy_stage06/outputs/state_delta_mixed_smoke/
```

## 7. Required ablations and selection status

- Naive gated-add: engineering forward/backward and earlier diagnostic run
  completed; definitive 100-step job was environment-OOM.
- State only: definitive 100-step gate completed and failed.
- State + delta: definitive 100-step gate completed and failed.
- Direct existing-KV interaction: implemented and CUDA-tested.
- Separate EventMemory: not run because the earlier mandatory gate failed.

There is no evidence here that delta improves on state alone. There is also no
evidence that either proposed interaction beats naive gated-add. No operator is
selected.

## 8. Gates intentionally not executed

Per the stop rule, Stage 06 did not run:

- varied-data equal-budget real-motion pilots;
- direct-KV versus EventMemory performance comparison;
- seed replication of a top candidate;
- real autoregressive generation or 1/2/4/8/16-chunk horizon scaling;
- Original/DiT/Minimal/naive/C1 final comparison;
- final C1 checkpoint training;
- `docs/c1_operator_contract.md` freeze for Contribution 2.

Creating a contract would falsely imply that a final operator had been
empirically selected. Contribution 2 remains out of scope and blocked on C1.

## 9. Reproduction commands

Real CUDA regression environment:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/dev/shm/c1_pytest:/dev/shm/c1_triton35:/root/workspace/state-spaces-mamba:/root/workspace/mamba_transformer_keneral/src \
/dev/shm/ardy_env/bin/python -m pytest -q \
  tests/test_sequence_kernel_mamba.py tests/test_ardy_dit_mamba.py
```

Definitive state+delta gate (state-only replaces the final flag with
`--no-use-state-delta`):

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/dev/shm/c1_triton35:/root/workspace/state-spaces-mamba \
/dev/shm/ardy_env/bin/python examples/train_ardy_c1_cached.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --minimal-checkpoint /workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt \
  --train-manifest /workspace/ardy_stage05/cache/minimal_seq/manifest_train.json \
  --eval-manifest /workspace/ardy_stage05/cache/minimal_seq/manifest_eval.json \
  --output-dir /workspace/ardy_stage06/outputs/state_delta_overfit_v4 \
  --interaction state_query --use-state-delta \
  --max-steps 100 --max-train-trajectories 1 --max-eval-trajectories 1 \
  --fixed-case --learning-rate 1e-3 \
  --new-parameter-lr-multiplier 5 --seed 61 --precision bf16
```

## 10. Exact failed criterion and next research question

Failed criterion: the proposed C1 variants could not substantially overfit one
fixed real sequential teacher-cache case. The measured reduction was about 4%,
not a meaningful overfit, and delta did not improve on state alone. Therefore
real-motion pilot, long-horizon evidence, naive-control superiority, delta
value, final checkpoint, and API freeze remain unestablished.

A future Stage-06 continuation should first diagnose why the fixed sequential
teacher case plateaus before changing the operator or launching more compute.
It must not treat the pilot checkpoints above as a completed C1 model.

CONTRIBUTION1_NOT_CLOSED
