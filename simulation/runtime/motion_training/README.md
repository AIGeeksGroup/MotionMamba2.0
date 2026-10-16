# Unified G1 training

Latest evidence (2026-10-04):
`/data3/zts/result/g1_training_framework_execution_20261003/CONTINUATION_20261004.md`.
A complete81-label/31-stage current-B8 pool was admitted and used in actual
four-rank8GT+2teacher updates. The new branch stopped at8 under protectedP0
retention; its paired fitting improved but it is not a qualified model.
The earlier statement that no complete pool exists is now historical.

The current status command, which does not restart the stopped run, is:

```bash
P=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
R=/data3/zts/result/g1_master_experiment_roadmap_20261003/execution_v1_1/R2
bash "$P/examples/train_motion.sh" pipeline --config "$R/closed_loop500.yaml" --output "$R/closed_loop" --pool "$R/w1_completion/POOL.json" --training-ranks 4 --status-only
```

All recipes use `engine.py`, the same schema, loss registry, FP32 master AdamW,
accepted-history reconstruction and full-state checkpoint code. Imports resolve
to this package and its pinned `_runtime`; historical directories provide data,
text and checkpoints, not runtime Python main programs. ARDY's private trainer
is unavailable. Loss reductions, beta, ratios and optimizer schedule are project
choices. Production serving has not been replaced.

Results: `/data3/zts/result/g1_training_framework_execution_20261003/REPORT.md`.

## Commands that actually ran

From `/data3/zts`, set shell variables once:

```bash
P=/data3/zts/MotionMamba2.0/mamba-transformer-keneral
F=/data3/zts/result/g1_training_framework_execution_20261003
B=/data3/zts/result/g1_master_experiment_roadmap_20261003/R2_P0_B
```

These historical commands succeeded; do **not** rerun the stopped B branch:

```bash
CUDA_VISIBLE_DEVICES=4 bash "$P/examples/train_motion.sh" train --config "$P/configs/training/foundation_gt_v1.yaml" --output "$B" --until 8
CUDA_VISIBLE_DEVICES=4 bash "$P/examples/train_motion.sh" train --config "$P/configs/training/foundation_gt_v1.yaml" --output "$B" --until 9 --resume "$B/checkpoint8.pt"
CUDA_VISIBLE_DEVICES=4 bash "$P/examples/train_motion.sh" evaluate --config "$P/configs/training/foundation_gt_v1.yaml" --checkpoint "$B/checkpoint128.pt" --output "$F/B/eval128"
```

The output directory is unfortunately named R2; its 128 updates are R1 B and
are counted once. It is now `STOPPED_FOR_HARM` and its trainer refuses resume.

Current no-training continuation/status command (returns the actual four
missing qualified stages instead of silently starting a substitute experiment):

```bash
bash "$P/examples/train_motion.sh" pipeline --config "$P/configs/training/closed_loop_v1.yaml" --output "$F/C" --pool "$F/C/pool0_supplement/POOL.json" --status-only
```

After a fully qualified pool is available, the same pipeline without
`--status-only` runs the fixed save/evaluate/stop/refresh sequence. An explicit
`--parent` selects the reviewed safe state; the existing P0 pool is not proof
of B8-generated histories. `--training-ranks 4` applies only to training, keeping
teacher collection and evaluation single-process. Budgets never reset.

## Four/eight GPUs

The actual four-GPU shared-engine check and its separate-process resume were:

```bash
CUDA_VISIBLE_DEVICES=0,4,5,7 MOTION_NPROC=4 bash "$P/examples/train_motion.sh" verify-worker --config "$P/configs/training/foundation_gt_v1.yaml" --output "$F/distributed/four_gpu" --mode dispatch_reference --steps 1
CUDA_VISIBLE_DEVICES=0,4,5,7 MOTION_NPROC=4 bash "$P/examples/train_motion.sh" verify-worker --config "$P/configs/training/foundation_gt_v1.yaml" --output "$F/distributed/four_gpu" --mode dispatch_reference --steps 2 --resume "$F/distributed/four_gpu/checkpoint1.pt"
```

These are evidence commands, not instructions to spend the same check budget
again. The global batch is **10**, distributed as 3/3/2/2 on four ranks, with
summed gradients; GPU hours include all ranks. Same-stage resume requires the
saved world size and every rank's RNG. `MOTION_NPROC=8` is implemented, but eight
GPU execution/throughput is unverified; do not claim a linear speedup.

In current measurements, tiny batch and gradient communication limit four-GPU
benefits. Do not multiply effective batch by ranks without a new registered
recipe. The pipeline preserves stop gates even when all GPUs are available.

## Inference kernel versus training

No-grad collection/evaluation selects `optimized` after the gate in the budget
root. Training has a differentiable fallback; kernel on/off loss, gradients,
FP32 master update and optimizer were exactly equal. The new inference kernel
has no validated backward acceleration. Update receipts separate build,
forward/loss, backward, gradient reduction and optimizer time for each rank.

## Foundation 50k plan

`configs/training/foundation_full50k_plan.yaml` is explicitly `PLAN_ONLY`.
Running it refuses a launch until a separately authorized execution recipe and
resource limits are frozen. Old 50k and stopped 128-step updates do not count as
new-contract 50k. Architecture-controlled fresh initialization is registered
separately from engineering warm-start and is not already trained.
