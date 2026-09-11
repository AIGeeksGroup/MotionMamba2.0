# D2 Learned Causal-Memory Benchmark

Date: 2026-09-03  
Status: PASS  
Evidence scope: learned mechanism gate, not real-motion quality

## Protocol

The benchmark uses the production `SequenceKernel`, `StateProjector`, official
`Mamba3ChunkWrapper`, `PersistentContentMemory`, and W4 sliding Attention KV.
Each accepted window contains 16 tokens, which satisfies the official Mamba3
Triton kernel's chunk requirement.

```text
W1: balanced binary cue A/B
W2-W7: cue-independent matched random distractors
W8: classify the W1 cue
```

Only W1 may write persistent content. Direct Mamba output and the state-address
scale are fixed at zero, so a W8 success cannot come from direct Mamba readout
or state-query addressing. At W8, the W4 KV cache contains only W4-W7 and W1
has been evicted. The same trained model is evaluated under every intervention.

Registered run: seed `20260903`, 800 updates, batch 64, 512 paired evaluation
cases, AdamW LR `3e-3`, K=2 persistent slots.

## Results

| Intervention | Accuracy |
|---|---:|
| correct | 1.0000 |
| zero delta at W1 | 1.0000 |
| permute delta at W1 | 0.5000 |
| write off at W1 | 0.5000 |
| readout off after W4 | 0.5000 |
| permute memory once after W4 | 0.0000 |
| direct Mamba readout off after W4 | 1.0000 |
| state address off after W4 | 1.0000 |
| Minimal-W4 control | 0.5039 |

The correct-minus-best core causal ablation gap is `0.50`, exceeding the
registered `0.25` threshold. The memory model received nonzero gradients on
799/800 updates for the content decoder, write gate, StateProjector, and Mamba;
the read scale received gradients on 800/800 updates. Its first-20 mean loss
was `0.695431`, and its last-20 mean loss was `1.8808e-6`.

The fixed K=2 memory norm at the last training forward was `21.9724`. Cache
occupancy followed `[1, 2, 3, 4, 4, 4, 4]`, confirming W4-bounded recent KV.

## Gate and limitation

All preregistered mechanism criteria pass. Write-off, readout-off, and a
trajectory permutation each destroy or reverse correct behavior after W1 KV
eviction. This establishes learned causal use of fixed-capacity content in the
mechanism benchmark.

`zero_delta_at_W1` remains perfect, while `permute_delta_at_W1` fails. That is
not sufficient to claim an independent delta contribution because the
absolute state remains available. The BONES D2-full versus D2-no-delta gate is
still required.

The initial registered run is retained at
`/data4/zts/result/stage_d2_learned_causal_memory_20260903`. It failed because
the implementation repeatedly permuted rows after W4; four cyclic
permutations restored the original row order. The corrected protocol applies
exactly one permutation after W4 and uses a new v2 result directory.

Raw result:
`/data4/zts/result/stage_d2_learned_causal_memory_v2_20260903/results.json`
(`SHA-256 0947b382f2e7c2550267dc99ca7c41028b480609f281a2fd93170145fb5e3524`).

