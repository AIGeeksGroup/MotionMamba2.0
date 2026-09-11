# HumanML3D Cache/Window Diagnosis (2026-08-20)

## Finding

The 50k v5r runs used `sequential_cache_probability=0.0`. They optimized
single-window denoising and history-conditioned inputs, but never trained an
accepted-history commit followed by a second-window read. Inference did enable
that persistent memory path.

On the fixed 64-sample panel before the lifecycle fix:

| Mode | Memory | Error (cm) | Boundary max root speed (m/s) |
|---|---|---:|---:|
| Minimal | persistent | 46.48 | 14.86 |
| Minimal | explicit history | 46.48 | 4.82 |
| Hybrid | persistent | 46.01 | 7.67 |
| Hybrid | explicit history | 46.00 | 1.80 |

This isolated the primary failure to cross-window state/coordinate handling,
not to inability of the denoiser to fit the local target.

## Fix

`ArdyDiffusionMemoryController` now defers a t=0 commit until ARDY's
`_recenter_history` has completed. It then replays only the newly generated
40-frame chunk with generation-only masks, so the memory state is stored in the
same recentered coordinate frame consumed by the next window. Test doubles
without ARDY's recenter lifecycle retain the old immediate-commit behavior.

The focused adapter suite passes 7/7 tests. On an 8-sample regression panel
after the fix:

| Mode | Persistent error (cm) | Persistent boundary max (m/s) |
|---|---:|---:|
| Minimal | 51.70 | 1.83 |
| Hybrid | 69.01 | 3.18 |

The boundary spike is substantially reduced, but quality is not yet a final
result. A short cache-aware training gate is required before the next 50k full
run.

## Training decision

The corrected full run must use the same HumanML3D manifests and checkpoints,
with `sequential_cache_probability=0.5`, `history_probability=0.5`, and global
batch 32. This makes accepted-history training a first-class part of the
objective instead of an inference-only path. The long runs are launched in
detached tmux sessions and their logs/checkpoints are written under
`/data4/zts/result`.
