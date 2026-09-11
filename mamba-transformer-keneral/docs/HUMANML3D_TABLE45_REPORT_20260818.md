# HumanML3D Table 4/5 Evaluation Report

Date: 2026-08-18

## Scope and protocol

The final 50k Minimal and Hybrid checkpoints were evaluated on the same 64
fixed HumanML3D test samples with real cached LLM2Vec features, 10 diffusion
steps, SDPA, batch size 4, and no post-processing. Table 4 proxy uses one
40-frame window with a joint-position goal. Table 5 proxy uses 80-frame
autoregressive generation with the goal respectively inside or beyond the
first 40-frame horizon.

This repository does not contain ARDY's released HumanML3D retrieval/FID
evaluator or the exact paper Table 4/5 evaluation script. Consequently,
R-Precision and FID are reported as `N/A`; `Skate` is an explicitly labelled
contact/velocity proxy, latency is batch-4 amortized, and the resulting rows
must not be presented as an exact reproduction of the paper protocol.

## Table 4 proxy: offline text and joint-position control

| Method | R-Prec. ↑ | FID ↓ | Skate (%) ↓ | Error (cm) ↓ | Latency (s) ↓ |
|---|---:|---:|---:|---:|---:|
| ARDY paper, no optimization | 0.729 | 0.044 | 6.28 | 4.15 | 0.15 |
| Minimal 50k, internal proxy | N/A | N/A | 57.62* | 90.49* | 0.214* |
| Hybrid 50k, internal proxy | N/A | N/A | 51.86* | 77.46* | 1.364* |

Hybrid lowers the internal constraint error by `14.4%` and the skate proxy by
`10.0%` relative to Minimal, but is `6.4x` slower in this unoptimized reference
implementation. The proxy errors are far above the ARDY paper row.

## Table 5 proxy: autoregressive text and joint-position control

### In-horizon goals

| Method | R-Prec. ↑ | FID ↓ | Skate (%) ↓ | Error (cm) ↓ | Latency (s) ↓ |
|---|---:|---:|---:|---:|---:|
| ARDY paper | 0.690 | 0.092 | 7.07 | 2.48 | 0.15 |
| Minimal 50k, internal proxy | N/A | N/A | 52.83* | 116.34* | 0.902* |
| Hybrid 50k, internal proxy | N/A | N/A | 46.84* | 82.53* | 1.972* |

### Out-of-horizon goals

| Method | R-Prec. ↑ | FID ↓ | Skate (%) ↓ | Error (cm) ↓ | Latency (s) ↓ |
|---|---:|---:|---:|---:|---:|
| ARDY paper | 0.684 | 0.100 | 7.63 | 2.92 | 0.15 |
| Minimal 50k, internal proxy | N/A | N/A | 52.71* | 228.82* | 0.823* |
| Hybrid 50k, internal proxy | N/A | N/A | 46.44* | 176.45* | 1.681* |

Hybrid reduces the internal constraint error by `29.1%` for in-horizon goals
and `22.9%` for out-of-horizon goals. It nevertheless remains far from the
ARDY paper error and latency, so the present checkpoints do not pass a Table 5
quality claim.

`*` Internal proxy definition/protocol; not directly comparable to the paper
without ARDY's evaluator checkpoint and exact preprocessing.

## Geometry

| Protocol | Minimal mean / max root speed (m/s) | Hybrid mean / max root speed (m/s) |
|---|---:|---:|
| Table 4 proxy | 1.46 / 46.32 | 1.12 / 32.57 |
| Table 5 in-horizon | 1.71 / 157.30 | 1.33 / 108.57 |
| Table 5 out-of-horizon | 1.81 / 157.37 | 1.44 / 106.41 |

All tensors and rotations are finite, but the root-speed maxima confirm severe
internal/chunk discontinuities. Hybrid is consistently better than Minimal but
neither final checkpoint passes the existing `5.42 m/s` geometry threshold.

## Reference efficiency and future kernel comparison

| Implementation | Active params | Train samples/s | Train peak memory | T4 latency | T5-in latency | T5-out latency |
|---|---:|---:|---:|---:|---:|---:|
| Minimal reference | 247.91M | 109.45 | 5.91 GiB | 0.214 s | 0.902 s | 0.823 s |
| Hybrid reference | 352.58M | 88.07 | 7.26 GiB | 1.364 s | 1.972 s | 1.681 s |
| Hybrid + custom kernel | N/A | N/A | N/A | N/A | N/A | N/A |

No custom kernel has been implemented or benchmarked. The current reference
shows the optimization target: Hybrid has `42.2%` more active parameters,
`19.5%` lower training throughput, and `22.9%` higher training peak memory than
Minimal. A future kernel result must use the same checkpoints, sample IDs,
noise seed, batch size, and protocol, and must report numerical/quality
equivalence in addition to speed and memory.

## Conclusion

本轮 50k HumanML3D 全量训练在数值训练层面成功，但 Table 4/5 proxy 评测表明模型质量尚未达到 ARDY：Hybrid 相比 Minimal 在 offline、in-horizon 和 out-of-horizon 三种设置下分别将约束误差降低约 14.4%、29.1% 和 22.9%，同时降低 foot-skate proxy 与 root-speed 峰值，说明 Attention+Mamba 的结构互补确实存在；但两者的约束误差、足滑和长序列根轨迹跳变仍明显高于 ARDY，且未优化 Hybrid 延迟更高。当前可把这轮结果作为 kernel 前的 reference baseline，不能作为已复现 ARDY Table 4/5 的最终结论；下一步应先补齐官方 R-Precision/FID evaluator 与精确协议，再修复 root continuity/control objective，最后在同一评测集上加入 kernel 前后吞吐、显存、延迟和质量等价性对比。
