# D2R-L 50k Training Status

Updated: 2026-09-06 15:10 Asia/Shanghai

## Active tmux jobs

| Job | tmux session | State |
|---|---|---|
| D2R-L continuation | `d2rl50k` | running |
| final selection evaluator | `d2rl_post` | waiting for `TRAINING_COMPLETE` |
| structural ablations | `d2rl_ablate` | waiting for final selection marker |

The training output is `/data4/zts/result/stage_d2rl_50k_20260906`. The
post-training jobs do not consume GPUs while waiting and will not overwrite
the training directory.

## Progress and throughput

- Initialization: D2R step-5000 checkpoint, continuing to total optimizer step
  50,000; this is not a fresh 50k run.
- Latest observed log step: 5,160.
- Stable measured window: 10 optimizer steps in 96 seconds, approximately
  9.6 s/step.
- Equivalent throughput: approximately 0.409 global samples/s and 127.7
  frames/s under the four-rank configuration.
- Remaining work at step 5,160: 44,840 steps, approximately 119.6 hours at
  the measured rate.
- Planning estimate: about 5 days of uninterrupted runtime; reserve 5--5.5
  days for checkpoint and restart overhead.

## Health snapshot

The latest records contain finite loss, gradients, memory norms, write gates,
and read scales. The most recent diagnostic values were approximately:

| Quantity | Value |
|---|---:|
| memory slot norm | 17.55 |
| write gate max | 0.214 |
| read residual scale | -0.022 |

The loss and gradient norm vary substantially between random minibatches. No
NaN/Inf or persistent gate-cap violation has been observed, so the registered
configuration remains unchanged.

## Evaluation timing estimate

After the final checkpoint is written, the selection evaluator runs in
`d2rl_post`. Prior matched runs took roughly 3--6 minutes per 16/32-sample
20/40-second panel; the four-model Table 4/9-second/20-second/40-second
selection wave is expected to take tens of minutes when parallelized across
GPUs. The final write/read intervention wave adds roughly 10--20 minutes.

The structural 2x2 ablation job is intentionally separate. Each 50k
two-rank training variant is expected to require several days and is not
counted in the main D2R-L completion estimate.
