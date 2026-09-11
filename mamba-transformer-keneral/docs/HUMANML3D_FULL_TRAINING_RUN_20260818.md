# HumanML3D Full Training Run

Date: 2026-08-18

## Preflight

- Train/validation/test manifests: `21096 / 1322 / 3972` records.
- Train-only dual-root statistics: 334-D, finite.
- Real LLM2Vec cache: `22418 x 4096`, FP16 on disk, all finite.
- Full repository tests: `84 passed, 6 skipped`.
- DDP exact resume after stochastic augmentation: maximum parameter difference `0.0`.
- 1k-step stochastic-augmentation DDP pilots: Minimal and Hybrid both passed
  loss/evaluation/finite-gradient gates.
- No network dependency: text embeddings are read from the local cache and
  `LOCAL_CACHE=true` is set in both tmux jobs.

The 1k checkpoints are engineering pilots, not final quality results. Their
long-horizon geometry is retained as a post-training benchmark artifact and is
not used to claim final Table 4/5 quality.

## Frozen training configuration

Both variants use the same dataset, split, text cache, 80-frame window,
40-frame generation horizon, history probability `0.5`, BF16, SDPA, global
batch `32`, AdamW, learning rate `5e-5`, boundary continuity weight `10`, and
seed `123`. Only the sequence mixer (`minimal` vs `hybrid`) changes.

Measured scaling selected two GPUs per variant because the fixed-global-batch
benchmark was faster on two GPUs than four GPUs on this PCIe host. The jobs run
concurrently in independent tmux sessions:

```text
mm2_hml3d_full_minimal_20260818  CUDA_VISIBLE_DEVICES=0,1
mm2_hml3d_full_hybrid_20260818   CUDA_VISIBLE_DEVICES=2,3
```

Each run is configured for 50,000 optimizer steps, evaluation every 1,000
steps, and resumable checkpoints every 5,000 steps. Logs and checkpoints are
under `/data3/zts/results/humanml3d_ardy_full_*_20260818`.

Both jobs started successfully at approximately `2026-08-18 15:14 CST` and
were confirmed to advance with finite losses and active utilization on all four
assigned GPUs. Based on the final 1k pilots, the estimated wall times are about
3.8 hours for Minimal and 5.2 hours for Hybrid when run concurrently.

## Post-training gates

After both runs finish, use the final checkpoints for Table 4-style offline
control, Table 5-style in/out-of-horizon autoregressive control, and the
kernel-vs-reference efficiency comparison. Interactive/Rigplay claims remain
out of scope for this HumanML3D run; the separate Bones Seed checkpoint is the
planned simulation/real-robot demo artifact.
