# D2R-L 5k Freeze Audit

Date: 2026-09-06

The 50k continuation was stopped at the user's request. No checkpoint from
that interrupted continuation is used below. The frozen main-model candidate
is the complete pre-existing 5k artifact.

## Frozen artifacts

| Item | Path | SHA-256 |
|---|---|---|
| Native Hybrid parent (50k) | `/data4/zts/result/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt` | `cfa9d44d9a804e3f71dbfde98cbad7faceed98bca83b5417c3afb5bca6071a4a` |
| Frozen D2R-L candidate | `/data4/zts/result/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt` | `f197e8d7c12fca6c82db0d36a1e31ccf34163074ec8ec9aa847f0a25ca696020` |
| Frozen D2R-L resumable checkpoint | `/data4/zts/result/stage_d2rl_bones_long_20260904/step5000/full/checkpoint_step_005000.pt` | `081ed55c08b39c685cbd3d2c63de7ae89999134798266022985e4c54b6952eee` |

The frozen D2R-L artifact and the Native Hybrid parent are distinct files; the
resumable checkpoint is the authoritative training-state file for provenance
checks.

## Provenance clarification

The historical step-5000 run was initialized from the D2R step-250 full
checkpoint, whose lineage begins at the D2R step-100 run. It therefore cannot
be honestly described as a fresh D2R-L run initialized directly from the
Native Hybrid 50k parent. The defensible paper wording is:

> D2R-L uses the Native Hybrid-style backbone with `state_query` and
> persistent content memory and is evaluated at the existing approximately
> 5k-step continuation checkpoint; its recorded continuation lineage is
> D2R step-250 to total step 5,000.

The new structural ablations are launched separately from the exact Native
Hybrid 50k parent, with the same data split, optimizer, learning rate, global
batch, rollout protocol, seed, and 5,000-step budget.

## Frozen configuration

- BONES-SEED-v004 train/validation manifests
- Hybrid g1, W4 attention, six-window `d2_post_eviction_v1` rollout
- `state_query`, summary dimension 128, state delta enabled
- K=2 text-conditioned persistent memory, normalized slots/delta, max write
  rate 0.25
- AdamW, learning rate `1e-4`, global batch 4, seed `9032026`
- generated-history BPTT windows 4, generated-history diffusion steps 10

## Interrupted continuation

The later 50k continuation output remains at
`/data4/zts/result/stage_d2rl_50k_20260906` for audit purposes. It was stopped
without replacing the frozen 5k checkpoint. Its latest log step was recorded
as 5,680; no claim or result is based on those unfinished weights.

Git worktree was dirty before this audit; the active source revision is the
working-tree revision at execution time.
