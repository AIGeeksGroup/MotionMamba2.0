# D3 Memory Specificity Pre-audit

Date: 2026-09-05

Scope: BONES-SEED selection actors, 40 s in-horizon, Attention W4, D2R-L-full step 5000.

## Locked inputs

- Primary checkpoint: `/data4/zts/result/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt`
- Primary checkpoint SHA-256: `f197e8d7c12fca6c82db0d36a1e31ccf34163074ec8ec9aa847f0a25ca696020`
- Panel: `/data4/zts/result/bones_seed_long_memory_panel_v2_20260831/selection/40s_in_horizon.jsonl`
- Panel SHA-256: `41909e76aa2a0011bb76cddec77048dd4bdb9aaa4cb8feac78a42e57dfa4579a`
- Panel size: 16 motions / 7 actors. The selection validation manifest has no additional eligible 40 s rows, so the requested 32-motion/10-actor target is not attainable without changing the locked selection source.
- No locked-final actors are read.

## Lifecycle findings

1. A generated window is accepted only from the clean post-recenter hook in `ArdyDiffusionMemoryController._post_window_hook`. The hook calls `canonical_history_prime`, which executes one clean read/write pass and commits exactly one chunk. No noisy diffusion forward commits state.
2. The prior `permute_memory` intervention runs in `_maybe_permute_content_memory` when `completed_windows >= 4`, immediately before the next denoising call. It directly reorders each layer's `layer.persistent_memory`, so it changes durable memory, runs once, and leaves later accepted writes enabled. This permits later windows to repair the swapped content.
3. The new `persistent_read_permutation` intervention is a read-view only. At every post-W4 forward it builds a fixed derangement and passes `persistent_memory.index_select(0, permutation)` through the stack. The durable tensor is never assigned, and accepted clean commits use `read_view=None` so their own write proposal remains based on the trajectory's durable memory.
4. `freeze_write_after_w4` disables content writes only after the fourth generated window. `one_shot_permute_plus_freeze_write` preserves the old one-shot durable swap but disables all later writes. Their matched controls are therefore normal, one-shot+normal-write, freeze-only, and one-shot+freeze.
5. The persistent-memory object is stored independently per DiT layer in `DiTSequenceState.layers[*].persistent_memory`, with shape `[batch, K=2, d_model]`. Attention K/V live in the separate `attention_cache`; Mamba recurrent state and projected `state_summary/state_delta` are separate fields.

## Intervention invariants

The controller records durable-memory checksums before/after, read-view checksum, Attention K/V checksum, Mamba-state checksum, and the fixed permutation for every post-W4 read window. The read-view path does not alter text, noise, CFG, constraints, Attention W4 K/V, Mamba state, state summary, state delta, or write proposal. Batch size one is rejected for all permutation interventions.

## Existing artifact limitation

The historical step-5000 `permute_memory` artifact is a one-shot durable mutation. It is retained as the B condition, but it cannot answer whether later writes repaired the swap. D3 therefore adds the persistent-read and freeze controls before any mechanism claim.
