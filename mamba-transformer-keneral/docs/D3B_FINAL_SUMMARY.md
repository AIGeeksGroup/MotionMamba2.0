# D3b Cross-Run Memory Transplant: Final Summary

Date: 2026-09-06

Decision: **STOP**

Classification: **Case E - intervention plumbing failure**

`ELIGIBLE_FOR_STAGE_E = false`

## Completed artifacts

- The primary 5k D2R-L-full checkpoint was kept read-only and matched the locked SHA-256.
- A deterministic 20 s panel was enumerated from validation records only: **32 motions / 28 actors**. The original 40 s panel remains **16 motions / 7 actors**. No `locked_final` file was read.
- Own/nearest/random/farthest donor mappings were pre-registered from caption embeddings before inference.
- The donor bank contains **5,632** complete keys: 32 trajectories x 11 accepted-window indices x 2 streams x 8 layers, with K=2 slots of width 1024. Window and root/body/layer alignment checks passed.
- The external read-only hook, bank builder, pairing builder, power-analysis script, and focused tests were added. The focused D3 tests pass (`12 passed` after the commit-view fix; the prior D3/content/protocol set passed `35 passed`).

## Mandatory parity result

The exact normal configuration was run twice. The two normal runs matched every per-sample metric exactly, establishing deterministic baseline behavior. Offline own-memory replay did not match normal:

| Condition | Root post-W5 AUC mean gap | Constraint post-W5 AUC mean gap | Root max abs gap | Constraint max abs gap |
|---|---:|---:|---:|---:|
| Own replay v1 | +0.141 | +0.089 | 1.919 | 2.382 |
| Own replay v2 (clear donor view before clean commit) | +0.198 | +0.167 | 3.194 | 3.906 |

Because the own replay is not equivalent to normal, the intervention plumbing is not trustworthy. The donor conditions were therefore **not run**. The common/specific generation decomposition, layerwise intervention, 40 s confirmation, and conditional no-delta replication were also not run.

## Scientific boundary

This run provides no new evidence for or against trajectory-specific memory. It is invalid to interpret the incomplete donor results as a semantic mismatch effect. The correct next action is to debug the bank capture/read alignment, especially the exact accepted-commit state used by the next noisy window, then repeat the offline-own parity gate before any donor comparison.

No checkpoint was modified, no training was started, no Stage E was started, and no `ELIGIBLE_FOR_STAGE_E=true` decision is permitted.
