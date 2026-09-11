0

# HumanML3D Unified Table 4/5 Evaluation

## Status

This matched evaluation compares a newly trained HumanML3D ARDY baseline, the
old 2026-08-19 `v5r` checkpoints, and the 2026-08-20 cache-fix checkpoints
under one protocol. All five checkpoints were evaluated with the same
HumanML3D test manifest, the same seed
(`20260818`), the same 64 selected test samples, batch size 4, 10 diffusion
steps, CFG text/constraint weights 2/2, SDPA attention, and the same Table 4/5
proxy implementation. Minimal/Hybrid use their persistent adapter memory;
original ARDY uses its native explicit autoregressive history path.

The HumanML3D ARDY baseline preserves ARDY's native two-stage Transformer
architecture but rebuilds both native Transformer mixers with fresh weights.
It copies only the same compatible frontend/projection initialization retained
by Minimal/Hybrid, thereby excluding the Released ARDY Bones/Rigplay mixer
weights. It was trained for 50,000 steps with the same HumanML3D train/dev
manifests, motion representation, global batch 32, learning rate `5e-5`,
dropout, geometry losses, and checkpoint budget. It used 4 GPUs for 3.96 h;
training status is `PASS`, all gradients are finite, and fixed eval loss
decreased from `11.7444` to `0.8153`. Original ARDY has no
adapter persistent KV/Mamba state, so its sequential-cache probability is
necessarily zero and its autoregressive path carries explicit generated
history. This is now the fair `HumanML3D-trained ARDY baseline`; it is not the
Bones/Rigplay `Released ARDY native checkpoint`.

A separate released-mixer-initialized HumanML3D fine-tune was also completed
during the initialization audit. It is deliberately excluded from the main
tables because it retains the Bones/Rigplay Transformer mixer prior and is not
a fair architecture baseline.

The 2026-08-19 report used only an eight-sample smoke panel. Its apparent
Table 4 Hybrid advantage therefore does not survive this fixed 64-sample
paired evaluation. R-Precision and FID remain `N/A`: generated ARDY motion is
still CoreSkeleton27/330-D, while the HumanML3D evaluator requires
22-joint/263-D conversion. The reported error, skate, and latency are unified
protocol proxy metrics, not official paper metrics.

## Table 4 proxy: offline text and joint-position control

| Method                                    | R-Prec. Top-3 | FID |       Skate (%) |      Error (cm) |      Latency (s) |
| HumanML3D-trained ARDY, fresh mixer (50k) |           N/A | N/A |           34.09 |           57.07 | **0.0292** |
| HumanML3D v5r Minimal (50k)               |           N/A | N/A |           33.81 |          39.58 |           0.1144 |
| HumanML3D v5r Hybrid (50k)                |           N/A | N/A | **29.81** | 47.90 |           0.3352 |
| HumanML3D cache-fix Minimal (50k)         |           N/A | N/A |           34.44 |           44.02 |           0.1201 |
| HumanML3D cache-fix Hybrid (50k)          |           N/A | N/A |           36.73 |           62.19 |           0.3298 |

On the same 64 samples, both Minimal checkpoints and v5r Hybrid have lower
Table 4 error than fresh-mixer ARDY. Cache-fix Hybrid is worse. ARDY remains
the fastest reference implementation but does not have the best quality.

## Table 5: Long-Horizon Generation

| Method                                                    | R-Prec. Top-3 | FID |       Skate (%) |      Error (cm) |      Latency (s) |
| --------------------------------------------------------- | ------------: | --: | --------------: | --------------: | ---------------: |
| HumanML3D-trained ARDY, fresh mixer (out-of-horizon, 50k) |           N/A | N/A |           33.75 |           62.89 | **0.0559** |
| v5r Minimal (out-of-horizon, 50k)                         |           N/A | N/A |           26.48 |           59.87 |           0.2403 |
| v5r Hybrid (out-of-horizon, 50k)                          |           N/A | N/A | **22.15** | **48.57** |           0.6582 |
| cache-fix Minimal (out-of-horizon, 50k)                   |           N/A | N/A |           28.59 |           69.35 |           0.2545 |
| cache-fix Hybrid (out-of-horizon, 50k)                    |           N/A | N/A |           28.65 |           56.21 |           0.6618 |

Window-boundary maximum root speed (m/s): v5r Minimal/Hybrid are
`6.002/2.246` in-horizon and `5.669/2.295` out-of-horizon; cache-fix
Minimal/Hybrid are `4.380/4.217` in-horizon and `4.047/2.869`
out-of-horizon. Fresh-mixer HumanML3D-trained ARDY is `3.780` in-horizon and
`4.044` out-of-horizon; its corresponding mean boundary speeds are
`1.090/0.939 m/s`.

## Interpretation and Remaining Work

The earlier 8.19 Table 4 values (`52.21 cm` Minimal versus `30.12 cm`
Hybrid) came from an eight-sample smoke panel. On the fixed 64-sample panel,
the same v5r checkpoints give `39.58 cm` versus `47.90 cm`; the earlier
reversal was therefore sample-sensitive and must not be used as evidence of a
stable Hybrid Table 4 advantage.

The cache-fix change did not merely reformat the experiment. It changed both
the checkpoint and the training/inference contract: v5r used
`sequential_cache_probability=0.0`, whereas cache-fix used `0.5` and deferred
accepted cache commit until after ARDY history recentering. This explicitly
trained and corrected the persistent cross-window path. The new cache-fix
Hybrid is worse than cache-fix Minimal on Table 4 and Table 5 in-horizon, but
improves out-of-horizon error by about 19% (`69.35 -> 56.21 cm`) and reduces
the out-of-horizon boundary maximum (`4.047 -> 2.869 m/s`).

The fair same-data fresh-mixer ARDY baseline changes the architecture
conclusion relative to the released-mixer fine-tune. Fresh ARDY reaches
`57.07/44.69/62.89 cm` on Table 4/Table 5 in/Table 5 out. Minimal is better on
Table 4 (`39.58 cm` v5r; `44.02 cm` cache-fix), and cache-fix Minimal is also
better in-horizon (`41.84 cm`). Most importantly, both Hybrid checkpoints are
better than fresh ARDY out-of-horizon (`48.57 cm` v5r and `56.21 cm`
cache-fix), while also reducing skate proxy. This supports the intended
long-horizon architecture complementarity, but not a universal Hybrid win:
cache-fix Hybrid remains worse on Table 4 and in-horizon. ARDY is substantially
faster because it is smaller and has no persistent adapter memory.

Latency is included for protocol completeness, but these jobs were run in
parallel on separate GPUs and the exact seconds are not a final kernel-speed
benchmark. A final efficiency comparison must use a serialized warm-up and
timing protocol.

Remaining work:

1. Validate Core27/330-D -> HumanML3D 22-joint/263-D conversion and populate
   R-Precision/FID for all models on the same generated samples.
2. Use checkpoint curves and matched Table 4/5 results to decide whether
   100k/150k training is justified.
3. Report robust boundary statistics in addition to the maximum and run
   multi-seed/full-test replications before a paper-level architecture claim.

## Provenance

- v5r Minimal JSON: `/data4/zts/result/humanml3d_matched_v5r_cachefix_eval_20260821/v5r_minimal_table4.json`
- v5r Hybrid JSON: `/data4/zts/result/humanml3d_matched_v5r_cachefix_eval_20260821/v5r_hybrid_table4.json`
- cache-fix JSONs: `/data4/zts/result/humanml3d_matched_v5r_cachefix_eval_20260821/`
- HumanML3D ARDY checkpoint: `/data4/zts/result/humanml3d_ardy_full_fresh_baseline_20260821_original/original_humanml3d_ddp.pt`
- HumanML3D ARDY checkpoint SHA256: `ce47eadac8a22c67178dce5a392f661df65899249f2ac6fcb9268c546e975ecd`
- HumanML3D ARDY metrics: `/data4/zts/result/humanml3d_ardy_full_fresh_baseline_20260821_original/metrics.json`
- HumanML3D ARDY evaluation JSONs: `/data4/zts/result/humanml3d_matched_v5r_cachefix_eval_20260821/humanml3d_ardy_fresh_table4.json` and matching Table 5 files
