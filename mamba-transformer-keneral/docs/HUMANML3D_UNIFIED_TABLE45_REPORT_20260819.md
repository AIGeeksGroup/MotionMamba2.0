# HumanML3D Unified Table 4/5 Evaluation

## Status

The frozen original HumanML3D evaluator is now loaded from the provenance-
tracked Hugging Face repackaging of the official `finest.tar` (movement, text,
and motion encoders; 263-D HumanML3D statistics). The implementation uses the
original GloVe/POS vectorizer, 40 <= length < 200 filtering, random caption and
crop selection, 32-way retrieval candidates, and 20 replications.

The evaluator calibration passed. On 4,640 eligible test samples per
replication:

| Dataset calibration | Value |
|---|---:|
| R-Precision Top-1 | 0.4592 +/- 0.0023 |
| R-Precision Top-2 | 0.6570 +/- 0.0018 |
| R-Precision Top-3 | 0.7633 +/- 0.0019 |
| FID | 0 (GT reference to itself) |

The Top-3 value has the expected scale but differs from the reported Dataset
anchor (approximately 0.739) by more than the measured confidence interval.
Therefore this report is labelled **Our unified HumanML3D protocol**, not an
exact paper reproduction. It remains suitable for controlled architecture
comparisons because every model uses the same evaluator and sample protocol.
The FID value above is necessarily zero because this row is the ground-truth
reference distribution; it is not a generative-model FID.

## Table 4: In-Horizon Generation

| Method | R-Prec. Top-3 | FID | Skate (%) | Error (cm) | Latency (s) |
|---|---:|---:|---:|---:|---:|
| Dataset (GT calibration) | 0.7633 +/- 0.0019 | 0 (reference) | N/A | N/A | N/A |
| Released ARDY | **pending unified generated-motion conversion** | **pending** | 2.64 | 10.60 | 0.0643 |
| Minimal v5r (50k) | **pending unified generated-motion conversion** | **pending** | 33.89 | 52.21 | 0.1978 |
| Hybrid v5r (50k) | **pending unified generated-motion conversion** | **pending** | 33.41 | 30.12 | 0.8607 |

## Table 5: Long-Horizon Generation

| Method | R-Prec. Top-3 | FID | Skate (%) | Error (cm) | Latency (s) |
|---|---:|---:|---:|---:|---:|
| Dataset (GT calibration) | 0.7633 +/- 0.0019 | 0 (reference) | N/A | N/A | N/A |
| Released ARDY (in-horizon) | **pending unified generated-motion conversion** | **pending** | 2.29 | 28.45 | 0.0941 |
| Minimal v5r (in-horizon) | **pending unified generated-motion conversion** | **pending** | 50.95 | 51.70 | 0.3873 |
| Hybrid v5r (in-horizon) | **pending unified generated-motion conversion** | **pending** | 33.86 | 68.44 | 1.6488 |
| Released ARDY (out-of-horizon) | **pending unified generated-motion conversion** | **pending** | 2.53 | 32.78 | 0.0923 |
| Minimal v5r (out-of-horizon) | **pending unified generated-motion conversion** | **pending** | 43.79 | 47.46 | 0.2629 |
| Hybrid v5r (out-of-horizon) | **pending unified generated-motion conversion** | **pending** | 35.96 | 147.00 | 1.6445 |

## Interpretation and Remaining Work

The learned evaluator and Dataset calibration are complete and validated. The
remaining blocker is representational, not evaluator-related: ARDY emits
CoreSkeleton27/330-D motion, while the original evaluator consumes
HumanML3D's 22-joint/263-D representation. A single shared Core27 -> SMPL22 ->
`process_file()` conversion must be applied to the same generated samples for
released ARDY, Minimal, and Hybrid before R-Precision/FID can be populated.
The geometry/latency values above are a deterministic eight-sample smoke using
the same sample IDs, seed, ten diffusion steps, CFG weights, batch size, and
GPU for all three methods. They prove that the generation paths execute, but
the panel is too small for paper claims. Until the conversion is run on the
full panel, these rows must not be presented as final unified paper metrics,
and no claim of exact ARDY Table 4/5 reproduction is justified.

Evaluator assets are under `/data4/zts/humanml3d_evaluator_assets/official`;
the complete calibration JSON is
`/data4/zts/result/humanml3d_unified_eval/dataset_calibration.json`.
