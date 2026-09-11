# D3b Pre-implementation Audit

Date: 2026-09-06

## Scope and locked inputs

This audit covers the read-only D3b cross-run memory transplant on the BONES-SEED
selection protocol. No parameters, checkpoint files, text-conditioning code,
loss, optimizer, or training schedule are changed. The primary checkpoint is
`/data4/zts/result/stage_d2rl_bones_long_20260904/step5000/full/hybrid_humanml3d_ddp.pt`
with SHA-256 `f197e8d7c12fca6c82db0d36a1e31ccf34163074ec8ec9aa847f0a25ca696020`.
The secondary no-delta checkpoint is conditionally reserved and is not run
unless the full checkpoint passes the D3b specificity gate.

The only permitted evaluation data are the BONES-SEED validation/selection
records. The `locked_final` directory is excluded. The original 40 s panel is
16 motions / 7 actors. The validation manifest contains 146 records with at
least 500 frames (28 actors), so D3b's 20 s panel uses all such records with a
fixed deterministic crop and actor-balanced ordering. The requested 32-motion
/ 10-actor minimum is therefore exceeded for the 20 s eligibility panel, while
the original 40 s panel remains unchanged.

## Existing implementation contract

`ArdyDiffusionMemoryController` commits only the clean, recentered accepted
window. Durable content memory is stored independently for every DiT layer,
root/body stream, with K=2 slots. The existing `content_memory_read_view` is a
read-only override: it is consumed by the attention content read and is not the
source of the write proposal. Attention W4 K/V, Mamba state, state summary, and
state delta are separate state fields.

D3b adds an opt-in external bank read. A bank key is
`(donor_id, completed_window, block, layer)` and stores the committed `[K,D]`
slots. For recipient window `t`, the evaluator injects donor slots recorded at
the same accepted-window index `t`; there is no time shift. Root and body use
the same donor identity and every layer in a stream uses that identity. The
recipient durable tensor remains untouched and subsequent accepted writes still
use the recipient's own state summary/delta and durable memory.

## Plumbing gate

Before donor comparisons, an offline-own replay is run with the same checkpoint,
panel, captions, noise seeds, CFG, diffusion steps, and window schedule as the
normal run. The gate requires finite raw output, identical durable-memory
checksums before/after each external read, and unchanged Attention K/V and
Mamba-state checksums. A failed parity check invalidates all transplant claims
and requires stopping for plumbing repair.

## Statistical preregistration

Donors are selected only from pre-inference caption embeddings: own,
semantic-nearest, seeded random, and semantic-farthest, with self mappings
forbidden and different actors preferred. Primary endpoints are Root and
Constraint post-W5 raw AUC within each horizon. Paired comparisons use 20,000
actor-cluster bootstrap resamples with fixed seeds; normalized AUC per
post-W5-window is reported across horizons. No donor is selected from motion
error or any post-inference result.

## Decomposition and diagnostics

The donor bank also supports actor-balanced leave-one-actor-out common memory
`G`, residual reconstruction `R_i`, `G+R_i`, `G`, `G+R_j`, and `R_i` in raw and
recipient-norm-matched forms. These averaged-vector interventions are
diagnostic only and cannot alone establish Stage E eligibility. Layerwise
cosine, variance, temporal stability, slot-collapse, and donor-distance
statistics are likewise diagnostic.

## Stop conditions

No Stage E is started in this run, even if the specificity gate passes. If the
20 s panel cannot be evaluated, the report must retain the 40 s direction-only
confirmation and classify the result as insufficient power rather than changing
the panel after seeing errors.
