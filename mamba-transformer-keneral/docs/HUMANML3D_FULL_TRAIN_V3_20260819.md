# HumanML3D Full Train v3

Date: 2026-08-19

## Decision

The original 50k runs are retained as reference artifacts. The v3 runs start
from fresh Minimal and Hybrid weights after correcting three training/inference
contract errors:

- direct constraint and FK losses now supervise generation frames only;
- foot-skate loss now supervises valid generation velocity pairs only;
- root-velocity loss includes the history-to-generation window boundary.

Future constraints remain model inputs, but are not treated as output targets
because the ARDY raw denoiser deliberately zeroes future outputs.

Teacher-forced persistent-cache pilots were finite and reduced fixed validation
loss, but worsened generated root continuity. They are therefore a documented
negative result and `sequential_cache_probability=0` for v3. Persistent memory
remains enabled in generation and is evaluated separately against explicit
history.

## Preflight

```text
Full tests:                 86 passed, 6 skipped
HumanML3D train/val/test:   21096 / 1322 / 3972
Text backend:               real cached LLM2Vec, 4096-D
DDP scaling:                Minimal and Hybrid 4-GPU PASS
FA2:                        unavailable in active environment
Attention backend:          SDPA correctness reference
```

The best 200-step geometry probes reduced the fixed eight-sample Table 5 proxy
as follows. These are internal proxy values, not paper-comparable metrics.

| Mode | In error | Out error | In max root speed | Out max root speed |
|---|---:|---:|---:|---:|
| Minimal | 55.26 cm | 88.97 cm | 7.72 m/s | 6.70 m/s |
| Hybrid | 58.61 cm | 127.99 cm | 3.21 m/s | 3.03 m/s |

The repository still lacks the official HumanML3D R-Precision/FID evaluator,
so those metrics remain `N/A` until the evaluator assets are supplied.

## Frozen v3 configuration

```text
GPUs per mode:               4
per-rank / global batch:     8 / 32
steps:                       50000
learning rate:               5e-5
frames / horizon:            80 / 40
history probability:         0.5
sequential cache training:   0.0
boundary weight:             10
root velocity weight:        50
explicit constraint weight:  1
FK constraint weight:        1
foot skate weight:           50
eval interval:               1000
checkpoint interval:         5000
retained resume checkpoints: 2
```

Output directories:

```text
/data3/zts/results/humanml3d_ardy_full_v3_minimal_20260819
/data3/zts/results/humanml3d_ardy_full_v3_hybrid_20260819
```

## v4 preflight addendum

The first v3 launch overlapped with the final CFG pilot and was stopped before
effective training. It is not a result. The audit then identified that
separated CFG requires independently sampled zero-text and zero-constraint
training examples. The dataset now zeros both the caption and cached LLM2Vec
embedding for text dropout, with a regression test.

The final full configuration adds:

```text
text dropout:        0.1
constraint dropout:  0.1
```

The 1k checkpoint gates reduced fixed eval loss from `0.81344` to `0.76094`
for Minimal and from `1.01017` to `0.84700` for Hybrid; all gradients were
finite. The v4 output directories are:

```text
/data3/zts/results/humanml3d_ardy_full_v4_minimal_20260819
/data3/zts/results/humanml3d_ardy_full_v4_hybrid_20260819
```

The fixed eight-sample proxy currently favors CFG `(text=1, constraint=2)` for
constraint tracking while `(1,1)` is retained as the lower-guidance stability
reference. CFG is an evaluation setting, not a training architecture change.
