# Training Audit 4090

> The initial smoke-only gate is preserved below as history. Current data,
> control, tiny/pilot, and DDP results are in
> [`HUMANML3D_CONTROL_TRAINING_REPORT.md`](HUMANML3D_CONTROL_TRAINING_REPORT.md).

Date: 2026-08-17
Commit: `91090b986c99e8ff574e4fa2c96f822ddac3b18b`
Branch: `BaselineAndC1`
Hardware: 8 x RTX 4090, 24 GB each, CUDA 12.4, PyTorch 2.6.0+cu124

## Executive result

The Minimal and Hybrid reference paths are **WORK for correctness smoke**.
They are ready for the next MD gates: HumanML3D adapter, tiny overfit, and 1k
pilot. The inference Control Pipeline Audit is complete, but control training
and the HumanML3D conversion remain unimplemented. Do not start Full Train yet.

## Commands and results

| Check | Result |
|---|---|
| Full repository test suite | PASS: 77 passed, 6 skipped |
| Minimal standalone cache example | PASS: SDPA, 8 chunks, bounded 3-chunk history, 2 sinks |
| Real Mamba smoke suite | PASS: 10 passed |
| Real Mamba Hybrid/C1 CUDA suite | PASS: 3 passed |
| Minimal manual optimizer + checkpoint reload | PASS |
| Hybrid state-query manual smoke: state/Delta gradients + optimizer + reload | PASS |

The per-row reset regression was corrected. The previous test compared a reset
row in a batch of three with a fresh single-row bf16 Triton execution. That
comparison has batch-shape numerical variation (`0.0039-0.0078`) despite a
correct reset. The repaired test compares with a fresh state at the same batch
shape and verifies both reset rows; the resulting difference is exactly zero.

The Minimal import-isolation test was also made order-independent: it now
asserts that constructing a Minimal/DiT adapter imports no *new* Mamba module,
rather than assuming a prior real-Mamba test has not imported one into the
shared pytest process.

## Training-interface audit

Present:

- cached teacher-call Dataset/DataLoader training for DiT;
- sequential teacher-cache Minimal trainer with `--resume`;
- C1/Hybrid pilot trainer and checkpoint writing;
- explicit Minimal vs Hybrid sequence configurations;
- attention KV/sink and Mamba recurrent state implementations.

Not present or not auditable in this checkout:

- HumanML3D Dataset Adapter wired to the model;
- `torchrun`/DDP/DistributedSampler training entrypoint;
- control-training manifests, sampling policy, and in/out-of-horizon metrics;
- FA2 package in the active environment (`flash_attn` import fails; smoke used
  SDPA instead);
- a current `TRAINING_AUDIT_4090.md` artifact before this run.

The existing cached trainers operate on ARDY teacher-generated traces, not on
HumanML3D ground-truth motion. They are valid engineering/distillation smoke
tests, but they are not the formal full-dataset training pipeline.

## Gate decision

```text
Minimal smoke                  PASS
Hybrid basic/full smoke        PASS
Inference control audit        PASS
HumanML3D adapter              BLOCKED: retarget specification absent
Control training audit         NOT READY
DDP/full-train readiness       NOT READY
```

Next required actions are to define and validate the HumanML3D-to-ARDY
retargeting contract, then implement the adapter, control-training manifests,
tiny overfit, 1k pilot, DDP scaling, and Full Train.
