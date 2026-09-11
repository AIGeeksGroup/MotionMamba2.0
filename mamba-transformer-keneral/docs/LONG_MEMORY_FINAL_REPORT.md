# Long-Memory Final Report

Date: 2026-09-02

## What is established

The repository now contains a gated A-I execution record. Stage B locks the
Native Hybrid 50k reference. The deterministic causal bookkeeping benchmark,
Stage C failure record, and Stage D persistent-content implementation are
reproducible. The D implementation reuses the existing
`StateProjector -> state_summary -> state_delta -> state_to_address` path,
defines `delta = P(S_t) - P(S_(t-1))`, and exposes fixed-capacity content slots
to future Attention only through accepted-window commits.

## What is publishable

Use the baseline lock and the C failure as an ablation/method-motivation table.
The D contract tests (58/58) can appear as implementation validation or
supplementary material. Do not present them as trained motion quality.

## What is not established

The official HumanML3D learned-metric branch is blocked by the bridge bias.
No real-motion D-full vs D-no-delta causal result exists. E-I were correctly
not run, so there is no evidence for a trajectory-specific long-memory claim,
3-seed final model, or fused-kernel quality result.

The exact numerical records are in `A_HML_BRIDGE_RESULTS.json`,
`B_BASELINE_MANIFEST.json`, `STAGE_C_RESULTS.json`, and `STAGE_D_RESULTS.json`.
