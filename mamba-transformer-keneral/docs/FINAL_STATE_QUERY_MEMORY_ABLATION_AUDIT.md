# Final structural-ablation audit

## Completed training artifacts

| Variant | Valid directory | Steps | World size | Global batch | Gradients finite | Final eval loss |
|---|---|---:|---:|---:|---|---:|
| state_query only | `/data4/zts/result/stage_d2rl_fair_ablations_5k_20260906/state_query_only` | 5000 | 4 | 4 | True | 0.453359 |
| persistent memory only | `/data4/zts/result/stage_d2rl_fair_ablations_5k_20260906/persistent_memory_only_v3` | 5000 | 4 | 4 | True | 0.452564 |

Both used AdamW at learning rate 1e-4, seed 9032026, W4 attention, six-window generated-history rollout, four accepted-history BPTT windows, and the exact Native Hybrid 50k parent. No variant received result-dependent tuning or extra steps.

The trainer records `status=FAIL` because its random-minibatch loss-trend gate is informational for fine-tuning. This is not hidden: both runs nevertheless reached exactly 5,000 steps with finite gradients. Persistent-memory parameters received nonzero gradients on 4,999/5,000 (decoder/write gate) and 5,000/5,000 (read scale) updates. Failed startup directories `persistent_memory_only` and `persistent_memory_only_v2` are excluded; only `persistent_memory_only_v3` is used.

The frozen D2R-L checkpoint is not falsely described as a direct continuation from that parent. Its historical lineage is D2R step250 to total step5000; see `D2RL_5K_FREEZE_AUDIT.md`.
