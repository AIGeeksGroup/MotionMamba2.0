# BONES-SEED Contract-Fix Table 4/5 Evaluation

## Status

This report evaluates three 50k denoiser checkpoints on the
actor-disjoint BONES-SEED v004 G1 split under the `canonical_history_v2` contract.
The released ARDY-G1 representation frontend and tokenizer/decoder are shared;
released Rigplay mixer weights are excluded from all three architecture rows.
The original ARDY run was invalidated after its root output branch collapsed. The
replacement resumes the stable fresh-mixer 5k checkpoint at learning rate `1e-5`,
continues to 50k total steps, and must pass fixed-dev-loss and root-output-norm gates.

| Method                  | Checkpoint                                                                                      | Training status                     |
| ----------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------- |
| ARDY stabilized retrain | `/data4/zts/result/bones_seed_ardy_retrain_lr1e5_20260826_original/original_humanml3d_ddp.pt` | PASS, 50k, finite gradients, 4.14 h |
| Minimal contract-fix    | `/data4/zts/result/bones_seed_contract_full50k_20260824_minimal/minimal_humanml3d_ddp.pt`     | PASS, 50k, finite gradients, 6.53 h |
| Hybrid contract-fix     | `/data4/zts/result/bones_seed_contract_full50k_20260824_hybrid/hybrid_humanml3d_ddp.pt`       | PASS, 50k, finite gradients, 8.39 h |

All nine evaluation jobs completed with `status=PASS`. Every model used the same
test manifest, seed `20260824`, 64 samples per protocol, batch size 4, 10 diffusion
steps, CFG text/constraint weights 2/2, SDPA attention, and identical sample IDs
within each protocol. Table 4 uses a 52-frame panel. Table 5 follows ARDY Section 6.3:
a nine-second, 225-frame sequence with a tokenizer-aligned 24-frame (0.96 s) GT prefix
and 201 generated frames. Table 5 in/out use the same motion IDs.

| Protocol           | Fixed panel SHA-256                                                  |
| ------------------ | -------------------------------------------------------------------- |
| Table 4            | `fe9d8c63b8c7f8cc45e0d4f43874cd065b29ca462a2dc119718adad3f2e0e8bd` |
| Table 5 in         | `7c7c980fc18fa80d080d9edf5fab554a22497be7fc43f272ae6bcea804da505e` |
| Table 5 out        | `7c7c980fc18fa80d080d9edf5fab554a22497be7fc43f272ae6bcea804da505e` |
| 20/40 s supplement | `86b0f2f94f2b6317a6b2e53d42dffe588b5857c55921d88e8120c08e561c6db2` |

For Table 5, Minimal and Hybrid prime the GT prefix once and then use persistent
cache-only rollout (`protocol_guard=gt_prefix_primed_cache_only`). ARDY has no
adapter-owned persistent state and uses native explicit autoregressive history.

## Dataset and Split

| Split | Assigned motions | Eligible motions (>=52 frames) | Actors | Invalid CSVs |
| ----- | ---------------: | -----------------------------: | -----: | -----------: |
| Train |          125,258 |                        123,651 |    463 |            0 |
| Val   |            7,578 |                          7,450 |     31 |            0 |
| Test  |            9,384 |                          9,222 |     28 |            0 |

The SHA-256 actor split has zero actor overlap. All eligible rows have captions.
Because several categories have too few contributing actors, the test split contains no eligible samples for: `Environments`, `Household`, `Looking and Pointing`, `Martial Arts`. This category-coverage limitation must be considered when interpreting aggregate results.

The model-native source is the 120 fps G1 CSV archive, deterministically resampled to 25 fps
and represented as 414-D explicit / 418-D dual-root statistics. All three downloaded
archives passed exact-size and gzip validation; SOMA is retained as dataset material but is
not silently retargeted into the G1 experiment.

## Official BONES-SEED Track

BONES-SEED has no locally available calibrated retrieval/FID or MoBench evaluator. These
values remain `N/A`; HumanML3D embeddings are not substituted as official BONES-SEED scores.

| Method                  | R-Precision Top-3 | FID | FVMD | TM-Sim |
| ----------------------- | ----------------: | --: | ---: | -----: |
| ARDY stabilized retrain |               N/A | N/A |  N/A |    N/A |
| Minimal contract-fix    |               N/A | N/A |  N/A |    N/A |
| Hybrid contract-fix     |               N/A | N/A |  N/A |    N/A |

## Table 4 proxy: Offline Text and Joint-Position Control

This protocol uses the 52-frame native G1 in-horizon generation panel. Lower is better for
every reported metric.

| Method                        | Constraint Error (cm) |   Root ADE (cm) |   Root FDE (cm) |   PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
| ----------------------------- | --------------------: | --------------: | --------------: | --------------: | ----------------: | --------------: | --------------: | -----------------: |
| ARDY stabilized retrain (50k) |                 52.91 |           55.39 |           65.89 |           18.02 | **0.02200** |           15.32 | **38.47** |   **0.0276** |
| Minimal contract-fix (50k)    |                 41.69 |           46.88 |           54.30 | **16.50** |           0.02904 | **14.89** |           49.52 |             0.1188 |
| Hybrid contract-fix (50k)     |       **39.79** | **40.44** | **48.38** |           16.88 |           0.02927 |           15.76 |           47.09 |             0.3281 |

## Table 5: ARDY-Paper-Aligned 9-Second Autoregressive Control

In-horizon places one sampled pelvis/wrist/foot position goal at the end of every
generation window. Out-of-horizon places one goal only at the final frame, beyond the
initial generation window. Metrics exclude the GT prefix and cover generated frames only.

| Method                        | Protocol       | Constraint Error (cm) |   Root ADE (cm) |   Root FDE (cm) |   PA-MPJPE (cm) | Jitter (m/frame2) | Angle MAE (deg) | Skate proxy (%) | Latency (s/sample) |
| ----------------------------- | -------------- | --------------------: | --------------: | --------------: | --------------: | ----------------: | --------------: | --------------: | -----------------: |
| ARDY stabilized retrain (50k) | in-horizon     |       **32.50** | **24.11** | **39.00** |           18.76 | **0.01832** |           14.27 |           40.96 |   **0.1865** |
| Minimal contract-fix (50k)    | in-horizon     |                 39.00 |           30.59 |           46.28 |           20.33 |           0.02568 | **13.76** |           51.70 |             1.0705 |
| Hybrid contract-fix (50k)     | in-horizon     |                 35.71 |           26.89 |           45.51 | **16.56** |           0.02226 |           13.84 | **40.23** |             2.2883 |
| ARDY stabilized retrain (50k) | out-of-horizon |                 55.13 |           35.17 |           54.20 |           22.08 | **0.01671** |           13.96 | **37.06** |   **0.1810** |
| Minimal contract-fix (50k)    | out-of-horizon |                 61.37 |           37.63 |           58.80 |           23.14 |           0.02326 | **13.30** |           50.36 |             1.0823 |
| Hybrid contract-fix (50k)     | out-of-horizon |       **51.65** | **29.26** | **48.69** | **17.60** |           0.02100 |           14.07 |           38.87 |             2.3362 |

## 20/40-Second Long-Sequence Supplement

The supplement uses the same 20 test motions that are each at least 40 seconds long.
The 20-second crop and 40-second crop share the same motion ID and source start, and all
three models receive identical text, goals, diffusion noise, and CFG settings. Window 5
is the first generation window that reads memory after the four-chunk attention cache has
evicted the GT-prefix chunk. Slopes are least-squares growth rates from windows 6 onward;
lower values indicate slower post-window-5 degradation.

| Duration | Method                        | In constraint (cm) | Out constraint (cm) | In Root ADE (cm) | Out Root ADE (cm) | Post-W5 in Root slope (cm/window) | Post-W5 in constraint slope (cm/window) |
| -------: | ----------------------------- | -----------------: | ------------------: | ---------------: | ----------------: | --------------------------------: | --------------------------------------: |
|     20 s | ARDY stabilized retrain (50k) |             155.37 |              884.35 |           118.16 |            337.19 |                            68.641 |                                  73.944 |
|     20 s | Minimal contract-fix (50k)    |             134.76 |              366.72 |           125.93 |            217.43 |                             6.002 |                                   3.256 |
|     20 s | Hybrid contract-fix (50k)     |             138.46 |              315.38 |           128.67 |            215.60 |                            -1.903 |                                  -5.942 |
|     40 s | ARDY stabilized retrain (50k) |             511.49 |             3624.43 |           480.05 |           1144.74 |                           132.124 |                                 132.820 |
|     40 s | Minimal contract-fix (50k)    |             155.87 |              535.11 |           156.72 |            348.69 |                            -1.130 |                                  -1.770 |
|     40 s | Hybrid contract-fix (50k)     |             166.89 |              635.33 |           160.81 |            391.10 |                             3.769 |                                   3.481 |

| Duration | First post-W5 window where Hybrid Root ADE is worse | Post-W5 Hybrid-Minimal Root-gap slope (cm/window) | Final-window Root gap (cm) |
| -------: | --------------------------------------------------: | ------------------------------------------------: | -------------------------: |
|     20 s |                                                   6 |                                            -7.905 |                      -8.91 |
|     40 s |                                                   8 |                                             4.899 |                      62.28 |

### 40-Second Window Trace: Hybrid vs Minimal

| Window | Minimal Root ADE (cm) | Hybrid Root ADE (cm) | Minimal constraint (cm) | Hybrid constraint (cm) |
| -----: | --------------------: | -------------------: | ----------------------: | ---------------------: |
|      1 |                 49.39 |                34.14 |                   82.77 |                  62.52 |
|      2 |                 93.45 |                65.75 |                  102.93 |                  80.47 |
|      3 |                116.10 |                80.03 |                  123.98 |                  92.38 |
|      4 |                128.60 |               105.36 |                  134.40 |                 114.17 |
|      5 |                152.92 |               129.15 |                  152.87 |                 137.60 |
|      6 |                168.98 |               152.87 |                  167.72 |                 165.07 |
|      7 |                181.44 |               178.08 |                  188.39 |                 192.03 |
|      8 |                193.55 |               194.68 |                  191.46 |                 191.39 |
|      9 |                197.40 |               185.79 |                  188.40 |                 175.58 |
|     10 |                173.34 |               172.29 |                  164.99 |                 171.92 |
|     11 |                166.06 |               172.82 |                  153.44 |                 166.37 |
|     12 |                152.21 |               172.34 |                  148.51 |                 181.37 |
|     13 |                169.55 |               185.02 |                  173.80 |                 193.22 |
|     14 |                175.67 |               197.59 |                  171.34 |                 201.57 |
|     15 |                172.56 |               194.02 |                  164.40 |                 191.20 |
|     16 |                177.92 |               201.43 |                  176.52 |                 204.17 |
|     17 |                181.98 |               211.07 |                  172.64 |                 214.12 |
|     18 |                174.82 |               222.28 |                  157.81 |                 220.88 |
|     19 |                150.29 |               212.57 |                  145.24 |                 214.92 |

At 40 seconds, Hybrid first exceeds Minimal Root ADE at window 8; its post-window-5 Hybrid-Minimal gap grows by 4.899 cm per window and reaches 62.28 cm in the final window. The current checkpoint therefore does not establish a 40-second Hybrid advantage. Training exposes the recurrent state to only one teacher-history commit followed by one generated window, so repeated-state rollout is an out-of-distribution regime and can accumulate Mamba-state bias. This negative result is kept separate from the healthy nine-second result.

## Boundary and Interpretation

| Method                        | Table 5 in max boundary speed (m/s) | Table 5 out max boundary speed (m/s) | Eval peak memory (MiB, max) | Active mixer parameters |
| ----------------------------- | ----------------------------------: | -----------------------------------: | --------------------------: | ----------------------: |
| ARDY stabilized retrain (50k) |                                4.46 |                                 5.29 |                      1540.7 |             147,150,996 |
| Minimal contract-fix (50k)    |                                6.44 |                                 3.47 |                      2305.9 |             247,914,644 |
| Hybrid contract-fix (50k)     |                                4.92 |                                 2.08 |                      2971.5 |             352,578,724 |

On the fixed panels, Hybrid contract-fix has the lowest Table 4 constraint error; ARDY stabilized retrain and Hybrid contract-fix have the lowest Table 5 in-horizon and out-of-horizon constraint errors respectively. Bold values identify the global minimum
within each protocol and metric. These paired geometry and physical proxies support only
this fixed BONES-SEED protocol; they do not establish an official leaderboard result.
Hybrid improves Table 5 constraint error over Minimal by 8.45% in-horizon and 15.84%
out-of-horizon. It also has the lowest Root ADE/FDE, PA-MPJPE, jitter, and skate proxy
in both protocols; Minimal retains the lowest Angle MAE. This supports a broad nine-second
long-rollout advantage on BONES-SEED while keeping the metric-specific exception explicit.

Latency was measured in parallel on separate GPUs and is suitable for within-run comparison
only. A systems claim requires serialized warm-up and identical GPU/timing isolation.

## Provenance

- Dataset protocol: `docs/BONES_SEED_TRAINING_PROTOCOL_20260824.md`
- Trainer: `examples/train_humanml3d_ardy_ddp.py`
- ARDY retraining launcher: `examples/run_bones_seed_ardy_retrain_20260826.sh`
- Table 4 evaluator: `examples/evaluate_humanml3d_table45.py`
- Table 5 evaluator: `examples/evaluate_ardy_official_table5.py`
- Corrected and long evaluation launcher: `examples/run_bones_seed_ardy_retrained_eval_20260826.sh`
- Table 4 JSON and NPZ artifacts: `/data4/zts/result/bones_seed_ardy_retrained_dual_eval_20260826/`
- Table 5 JSON and NPZ artifacts: `/data4/zts/result/bones_seed_ardy_retrained_official_table5_20260826/`
- 20/40 s JSON and NPZ artifacts: `/data4/zts/result/bones_seed_ardy_retrained_long_table5_20260826/`
- Stabilized ARDY training metadata: `/data4/zts/result/bones_seed_ardy_retrain_lr1e5_20260826_original/metrics.json`
- Archive validation: `/data4/zts/datasets/bones_seed/archive_validation.json`
