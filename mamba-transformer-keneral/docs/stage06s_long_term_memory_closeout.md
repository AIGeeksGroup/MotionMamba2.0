# Stage 06S Closeout — Dual-Timescale Long-Term Event Memory

Date: 2026-08-15
Decision: stopped at failed Gate A

## 1. Final dual-timescale semantics

The tested hypothesis kept Stage-05 Minimal frozen and treated attention and
the existing state-memory machinery as different timescales:

```text
Sliding-KV + sinks: exact recent accepted token content
StateTransitionMemoryOperator/EventMemory: proposed sparse old token content
Mamba state: proposed address, never the content payload
```

The existing `StateTransitionMemoryOperator` implementation was audited and
not changed. It already contains `Mamba3ChunkWrapper`, `StateProjector`,
`StateQKVEncoder`, `ContentValueEncoder`, and `EventMemory`, with query-before-
current ordering and explicit accepted writes. No thin integration adapter was
created because the prerequisite oracle gate failed.

## 2. Sliding-KV / EventMemory ownership rule

For current chunk `t` and `W=4`, the intended non-overlapping ownership was:

```text
Sliding-KV:  C_(t-4) ... C_(t-1)
EventMemory: C_0 ... C_(t-5)
```

Gate A implemented the same observable ownership without changing production
code: a diagnostic cache retained all accepted chunks, then cloned states were
filtered by chunk ID. Every condition retained exactly the recent four chunks;
only eligible old chunks differed. Attention sinks were unchanged. Historical
K/V came from accepted clean chunks, not noisy diffusion calls.

## 3. Real long-horizon cache and replay

The bounded cache used the real ARDY autoregressive generation path, exact
Stage-05 text features, horizons 8/16/32, four prompts, and diffusion call
indices 0/5/9 for the final current chunk. It contains 12 trajectories and 36
denoising cases. Teacher replay was exact:

```text
max absolute error   0.0
mean absolute error  0.0
generation time      48.6452 s
```

Artifacts:

```text
/workspace/ardy_stage06s/long_eval/manifest.json
SHA-256 4947aa7ee9790479347bb435d50fc492a28696a137db1cd56528ffb1e26b807c
```

The frozen parent checkpoint was
`/workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt`, SHA-256
`acbdcc78c6594ad690a1d3a76b129665198fd142e487a1eb64bf43163a1b949f`.
The evaluator verified that Minimal instantiated no Mamba module and did not
import `mamba_ssm`.

## 4. Gate-A Oracle long-term ceiling

Each old candidate was injected as its exact accepted token K/V alongside the
unchanged recent four chunks. Oracle Top-1 exhaustively selected the best old
singleton. Oracle Top-2 greedily retained Top-1 and exhaustively selected the
second old chunk.

Across all 36 cases:

```text
policy                 mean MSE    paired relative gain vs Minimal
Minimal recent-only    1.1187693   --
random old             1.1202705   -0.2378%
latest evicted         1.1180072   -0.0630% mean relative gain
Oracle Top-1           1.1154292    0.3974%
Oracle Top-2           1.1144264    0.4358%
```

Oracle Top-1 improved 29/36 cases. Its trajectory-bootstrap absolute-gain 95%
CI was `[0.001512, 0.005263]`. Oracle Top-2 improved 26/36 cases, with CI
`[0.001373, 0.007343]`. These non-zero effects are too small for the gate: the
task explicitly requires practical value, not merely statistical difference.

Full result:

```text
/workspace/ardy_stage06s/gate_a_oracle.json
SHA-256 77da291e5c74d8e12e38a7acd0ad52b17f2e72f798a84a7e3808c56c26b7b65d
```

## 5. Horizon-stratified Oracle gain

```text
horizon  Top-1 relative gain  Top-2 relative gain
8        0.3709%              0.1144%
16       0.6031%              0.9203%
32       0.2182%              0.2727%
```

The effect did not grow with horizon. Top-2 peaked just below 1% at horizon 16
and fell sharply at horizon 32. Overall Top-1/Top-2 also failed to exceed the
approximately 1% recent-window routing ceiling measured in Stage 06R.

Top-1 selected event distances ranged from 5 to 31 chunks, so the experiment
did exercise genuinely distant content; distance counts were
`{5:5, 6:2, 7:20, 15:2, 18:1, 21:2, 22:2, 31:2}`.

## 6. Gate-A decision and stopping reason

`LONG_TERM_ORACLE_FAIL`. Exact accepted out-of-window K/V produced only
0.397% Top-1 and 0.436% Top-2 overall relative MSE gain, failed to exceed the
Stage-06R approximately 1% ceiling, and did not improve with horizon. Gate A
therefore failed the required practical-meaningfulness condition.

Per the hard-stop rule, no larger dataset, learned retrieval, or generator
training was used to compensate for this result.

## 7. Unreached conditional evidence

The following were deliberately not run because Gate A failed:

```text
Gate B existing ContentValueEncoder/memory_proj payload       NOT REACHED
Gate C state retrieval                                        NOT REACHED
State-only vs state+delta                                     NOT REACHED
Token-only retrieval control                                  NOT REACHED
Gate D frozen denoising gain                                  NOT REACHED
Naive Hybrid comparison                                       NOT REACHED
FIFO vs Participative                                         NOT REACHED
Real C1 checkpoint                                            NOT CREATED
C1 4/8/16/32 generation                                       NOT RUN
```

No conclusion about trained Mamba retrieval quality is drawn. The result is
narrower: on these real frozen-Minimal denoising cases, even perfect selection
of raw old accepted K/V has an insufficient ceiling to justify training it.

## 8. Long-range diagnostic and runtime diagnostics

The horizon-stratified last-chunk task is the bounded long-range diagnostic.
It guarantees histories longer than `W=4` and reports the selected event ages.
A separate semantic dependency benchmark was not created after the oracle
ceiling failed.

Gate-A runtime for all 36 cases was 69.7131 seconds on one A800 in bf16 with
SDPA. Peak allocated GPU memory was 2201.84 MiB. There is no production latency
or memory overhead to report because no C1 module was integrated.

## 9. Reproduction commands

From `/root/workspace/mamba_transformer_keneral` on the A800 host:

```bash
CUDA_VISIBLE_DEVICES=2 /dev/shm/ardy_env/bin/python examples/build_ardy_long_horizon_cache.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --teacher-revision abe6c43beb28c867c950acb824b9c4ef3d63fb76 \
  --text-feature-manifest /workspace/ardy_stage05/cache/minimal_seq/manifest_eval.json \
  --output-dir /workspace/ardy_stage06s/long_eval \
  --horizons 8 16 32 --max-prompts 4 \
  --capture-diffusion-indices 0 5 9 --device cuda:0

CUDA_VISIBLE_DEVICES=2 /dev/shm/ardy_env/bin/python examples/evaluate_ardy_long_term_oracle.py \
  --ardy-root /root/workspace/ardy \
  --checkpoints-dir /root/workspace/ardy_checkpoints \
  --minimal-checkpoint /workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt \
  --manifest /workspace/ardy_stage06s/long_eval/manifest.json \
  --output /workspace/ardy_stage06s/gate_a_oracle.json \
  --device cuda:0 --precision bf16
```

## 10. Boundaries preserved

Stage-05 Minimal and all model source code remain unchanged. Stage-06 and
Stage-06R negative evidence is preserved. No direct state conditioning,
recent-KV routing, full MotionMamba training, custom/fused kernel, FA4 fusion,
Deep Forcing, simulation, or robot work was performed.

Regression verification used the existing `mamba_op` environment with CUDA
hidden (so incompatible local Triton/Mamba kernels are skipped):

```text
61 passed, 15 skipped in 3.95 s
```

The first CUDA-visible attempt also completed the same 61 non-Mamba tests, but
10 Mamba CUDA cases failed because that local environment's Triton lacks
`triton.language.make_tensor_descriptor`; this is an environment mismatch and
not a regression caused by Stage 06S. The actual Gate-A A800 run completed
successfully. The parent source revision is
`47030b79ec99128e3c143223e4e71e9d0931e0e8`; the experiment/closeout revision
is the commit containing this document.

C1_LONG_TERM_MEMORY_STOP
