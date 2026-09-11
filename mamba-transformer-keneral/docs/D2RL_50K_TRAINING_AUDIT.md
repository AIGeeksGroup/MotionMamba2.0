# D2R-L 50k Training Audit

Date: 2026-09-06

## Objective

Train the frozen D2R-L definition on BONES-SEED to total optimizer step 50,000:
Native Hybrid backbone + `state_query` + K=2 text-conditioned persistent
content memory. No architecture, loss, attention window, or checkpoint weights
are changed by this launch.

## Initialization and budget

The available parent is D2R-full checkpoint step 250, not a Native Hybrid
parent. The existing continuation reached total step 5,000, so the correct
description is **D2R-L continuation from D2R step 250 to total step 50,000**;
it must not be described as a fresh D2R-L 50k run. Parent SHA-256:
`27aec5040485b65fd58ad4983925de1918f144f2fb4b62692665997ccc5221cd`.

The preserved step-5,000 checkpoint SHA-256 is
`081ed55c08b39c685cbd3d2c63de7ae89999134798266022985e4c54b6952eee`.
The new launch resumes from that checkpoint and targets total step 50,000,
adding 45,000 optimizer steps.

## Fixed protocol

| Field | Value |
|---|---|
| Dataset | BONES-SEED-v004 train / validation manifests |
| mode / model | `hybrid` / `g1` |
| train scope | `state_memory` |
| world / batch | 4 ranks, per-rank 1, global 4 |
| optimizer / LR | AdamW, `1e-4` (resume LR explicitly overridden) |
| frames / history | 312 / history probability 0 |
| rollout | six windows, `d2_post_eviction_v1`, generated history enabled |
| attention | W4 chunks, SDPA |
| state | `state_query`, summary 128, delta enabled, native gated add |
| content memory | enabled, K=2, text conditioned, normalized delta/slots, max write rate 0.25 |
| seed | 9032026 |
| checkpoint cadence | every 5000 total steps, keep last 2 |
| training GPUs | CUDA 0,1,2,3 |

## Safety monitoring

The launcher writes `train.jsonl`, checkpoint files, and a safety audit at each
5k boundary. It checks finite gradients/diagnostics, update fractions for the
decoder/write/read parameters, write-gate cap, memory/proposal norms, and read
scale. The existing step-5k run measured ~9.77 s/step (0.409 samples/s), with
finite gradients, gate max 0.210, memory norm 17.79 and read scale -0.0216.

## Runtime estimate

At the measured throughput, 45,000 additional steps require approximately
122.1 hours (5.1 days), excluding restart overhead. A conservative planning
range is **5-6 days**. Each 32-sample 20s evaluation previously required about
6 minutes on one RTX 4090; a 16-sample 40s evaluation is roughly 3-5 minutes.
Table 4/5 panels are expected to take a few minutes each when run in parallel
on separate GPUs. Evaluation will be launched only after the final checkpoint
passes the training safety audit.

## Restrictions

No `locked_final` actors are read. No D3b/D3c donor experiment, no additional
training variant, and no checkpoint overwrite is permitted during this run.
