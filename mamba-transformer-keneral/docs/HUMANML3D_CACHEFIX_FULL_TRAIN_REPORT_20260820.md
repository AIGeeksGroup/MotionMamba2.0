# HumanML3D Unified Table 4/5 Evaluation

## Status

This is the cache-fix follow-up to the HumanML3D architecture experiments.
Minimal and Root-Mamba/Body-Attention Hybrid were both trained from scratch
for 50,000 steps on the same HumanML3D train/dev manifests and evaluated on
the same HumanML3D test manifest. The jobs used 4 GPUs, global batch size 32,
`sequential_cache_probability=0.5`, 10 diffusion steps, SDPA attention,
`llm2vec_cache` text features, and persistent memory at generation time.

Both training `metrics.json` files report `status=PASS` and finite gradients.
The Minimal run took 6.36 h and reached final eval loss 0.3135; the Hybrid run
took 8.45 h and reached 0.2997. These are formal training-engineering results,
not evidence that 50k is the final converged budget. Extension to 100k/150k
must be decided from loss curves, long-horizon stability, and these tables.

The frozen original HumanML3D evaluator is available from the
provenance-tracked official `finest.tar` repackaging (GloVe/POS vectorizer,
40 <= length < 200 filtering, 32-way retrieval candidates, and 20
replications). However, generated ARDY output is still in CoreSkeleton27/330-D
while the evaluator consumes HumanML3D 22-joint/263-D motion. Until the shared
Core27 -> SMPL22 -> `process_file()` conversion is implemented and validated,
R-Precision and FID remain `N/A`. The current geometry, skate, and latency
numbers are therefore **unified-protocol proxy results**, not official paper
metrics.

This report does not use a `Released ARDY native checkpoint` as a formal row.
That checkpoint belongs to the official ARDY training/data setting and is not
a HumanML3D-trained ARDY baseline. A fair ARDY architecture row requires a
separately trained HumanML3D ARDY baseline under the same pipeline.

## Table 4: In-Horizon Generation

| Method | R-Prec. Top-3 | FID | Skate (%) | Error (cm) | Latency (s) |
|---|---:|---:|---:|---:|---:|
| HumanML3D-trained Minimal (cache-fix, 50k) | N/A | N/A | 34.44 | 44.02 | 0.1652 |
| HumanML3D-trained Hybrid (cache-fix, 50k) | N/A | N/A | 36.73 | 62.19 | 0.3710 |

Evaluation panel: 64 fixed test samples, 40 frames, batch size 4,
10 diffusion steps, CFG text/constraint weights 2/2, and persistent memory.

## Table 5: Long-Horizon Generation

| Method | R-Prec. Top-3 | FID | Skate (%) | Error (cm) | Latency (s) |
|---|---:|---:|---:|---:|---:|
| HumanML3D-trained Minimal (in-horizon, 50k) | N/A | N/A | 27.69 | 41.84 | 0.2513 |
| HumanML3D-trained Hybrid (in-horizon, 50k) | N/A | N/A | 26.97 | 50.09 | 0.6584 |
| HumanML3D-trained Minimal (out-of-horizon, 50k) | N/A | N/A | 28.59 | 69.35 | 0.2697 |
| HumanML3D-trained Hybrid (out-of-horizon, 50k) | N/A | N/A | 28.55 | 56.10 | 0.6506 |

The maximum window-boundary root speeds are 4.380/4.217 m/s for
Minimal/Hybrid in-horizon and 4.047/2.772 m/s out-of-horizon.

## Interpretation and Remaining Work

The cache lifecycle correction improved persistent-memory stability relative to
the pre-fix diagnostic, but the current result supports only a local
long-horizon complementarity claim:

- Table 4 error is 44.02 cm for Minimal versus 62.19 cm for Hybrid; Hybrid is
  also about 2.25x slower.
- Table 5 in-horizon error is 41.84 cm versus 50.09 cm; Hybrid has a slightly
  lower skate proxy but higher error and latency.
- Table 5 out-of-horizon error is 69.35 cm versus 56.10 cm, an approximately
  19% reduction for Hybrid. Its maximum boundary root speed also falls from
  4.047 to 2.772 m/s.

Thus the present data suggest that recurrent Mamba memory plus
attention-window/sink handling helps out-of-horizon behavior, but do not show
a universal Hybrid advantage and do not establish superiority over ARDY. The
attention sink used here must not be described as a validated compressed
settled-memory mechanism without dedicated ablations.

The 2026-08-19 report is not numerically interchangeable with this report: it
records an older `v5r` checkpoint and an eight-sample smoke panel, whereas this
cache-fix report uses new checkpoints and 64 samples. Future reports will use
this same structure and will state checkpoint, sample count, protocol, and
metric scope in the Status section before presenting Table 4/5.

Remaining work:

1. Train a HumanML3D ARDY baseline with the same data, representation,
   optimizer, budget, sampling protocol, and evaluator pipeline.
2. Validate the Core27/330-D -> HumanML3D 22-joint/263-D conversion, then run
   R-Precision/FID on identical generated samples for ARDY, Minimal, and
   Hybrid.
3. Decide 100k/150k continuation from checkpoint curves and Table 4/5 rather
   than assuming 50k convergence.
4. Freeze the fair baseline and architecture ablations before reporting kernel
   speed, memory, active parameters, and quality changes.

Evaluator calibration remains appendix-only evidence. Its dataset-reference
values must not be inserted as a model row in the final Table 4/5 tables.

## Provenance

- Minimal checkpoint: `/data4/zts/result/humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt`
- Hybrid checkpoint: `/data4/zts/result/humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt`
- Evaluation JSONs: `/data4/zts/result/humanml3d_full_cachefix_eval_20260820/`
- Evaluator assets: `/data4/zts/humanml3d_evaluator_assets/official/`

