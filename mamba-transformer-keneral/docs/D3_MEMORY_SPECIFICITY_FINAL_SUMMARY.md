# D3 Memory Specificity Audit: Final Results

Date: 2026-09-05
Decision: **STOP**
Primary mechanism classification: **Case D - insufficient statistical power**
`ELIGIBLE_FOR_STAGE_E = false`

## Outcome

Persistent-memory write/read remains causally important, but this D3 run does not establish that the 5k memory content is trajectory-specific. Persistent read permutation worsened the mean Root and Constraint post-W5 AUC, but both actor-cluster 95% CIs cross zero. The required dual-AUC gate therefore fails.

| Intervention minus normal (lower is better) |          Root post-W5 AUC |    Constraint post-W5 AUC | Confirmatory conclusion        |
| ------------------------------------------- | ------------------------: | ------------------------: | ------------------------------ |
| One-shot durable permutation                |    +0.05 [-38.37, +25.31] |    +0.02 [-36.06, +24.29] | Null; reproduces 5k ambiguity  |
| Persistent read permutation                 |   +17.86 [-23.37, +58.55] |   +17.51 [-18.65, +54.47] | Positive mean, CI crosses zero |
| Freeze write after W4                       |    +3.29 [-10.36, +17.35] |     +8.74 [-4.14, +23.46] | Null control                   |
| One-shot permutation + freeze               |   +22.54 [-20.97, +70.43] |   +25.31 [-13.79, +68.97] | Positive mean, CI crosses zero |
| Write off                                   | +120.55 [+53.79, +214.38] | +117.93 [+44.98, +218.68] | Significant harm               |
| Readout off after W4                        |  +69.33 [+25.27, +123.59] |  +69.08 [+22.05, +129.75] | Significant harm               |

The identity-isolating `D-C` comparison (permuted frozen minus correct frozen) was Root +19.25 [-26.09, +64.80] and Constraint +16.57 [-21.94, +58.03]. Both CIs cross zero, so self-recovery is plausible but not proven.

## Checkpoint trajectory

| Total step | Persistent permutation Root AUC gap |      Constraint AUC gap |
| ---------: | ----------------------------------: | ----------------------: |
|        250 |                +3.79 [+0.24, +7.98] |    +3.94 [-0.12, +9.09] |
|        500 |               +8.33 [-1.57, +22.68] |   +8.93 [-0.75, +24.41] |
|       1000 |               +9.86 [-3.02, +27.06] |   +2.35 [-7.15, +18.68] |
|       2000 |              +15.37 [-1.65, +37.10] |   +5.00 [-6.82, +25.66] |
|       5000 |             +17.86 [-23.37, +58.55] | +17.51 [-18.65, +54.47] |

The mean specificity gap tends to increase with training, but no checkpoint has both primary AUC CIs above zero. This pattern supports Case D more than Case C: it is not evidence of homogeneous memory, because effects are consistently in the harmful direction and persistent permutation significantly worsened Root slope at 5k. It is still insufficient for the trajectory-specific claim.

Step 250 is the verified D2R parent checkpoint; steps 500-5000 are D2R-L continuation checkpoints. The requested no-delta replication was conditionally gated on full first establishing specificity. Because that condition failed, no-delta was not run and no delta claim is made.

## Protocol and implementation audit

- Checkpoint SHA-256 was recomputed as `f197e8d7c12fca6c82db0d36a1e31ccf34163074ec8ec9aa847f0a25ca696020`.
- Fixed BONES-SEED selection panel: 16 motions / 7 actors, SHA-256 `41909e76aa2a0011bb76cddec77048dd4bdb9aaa4cb8feac78a42e57dfa4579a`.
- The selection manifest cannot supply the requested 32 motions / 10 actors under the exact 40s span. No locked-final actor was read.
- Persistent permutation applies at every post-W4 read with one fixed derangement. Durable memory checksums are identical before/after each read; Attention K/V and Mamba-state checksums are recorded separately.
- Semantic mismatch could not become stronger than random pairing because the locked protocol uses batch size two: only one derangement exists per batch. Its result is identical and is not presented as independent confirmation.
- Exact K=2 tensor traces show one-shot recovery distance peaking at 7.87 after W5 and declining to 2.74 by W19. Across-layer averages are: trajectory cosine 0.80, temporal cosine 0.99, slot-collapse cosine 0.11, and per-element trajectory variance 0.088. Slots do not collapse, and the decay is supporting evidence for gradual self-recovery, but the confirmatory AUC tests remain underpowered.

## Scientific boundary

Currently established: **Persistent memory write/read improves long-horizon BONES selection performance.**

D3 sought to establish: **Using another trajectory's persistent memory causally harms future generation after W4 eviction.** The mean effect supports this direction, but the two primary CIs cross zero, so it is not established.

Still not established: **The 5k memory necessarily stores unique long-term history for each trajectory.**

No training was run, no weights were modified, and Stage E was not started. The correct next action is to obtain a larger actor-diverse selection protocol or an independently preregistered public-data panel, then rerun persistent/semantic permutation with batch-level mappings that allow genuinely different semantic pairings.

Raw evaluation and tensor artifacts are under `/data4/zts/result/stage_d3_memory_specificity_20260905`; large tensors are intentionally not copied into Git.
