# Causal Memory Benchmark

Date: 2026-09-02

This deterministic harness validates the cue-eviction intervention bookkeeping
requested by 9.2-revise.MD. It is not a trained BONES-SEED or HumanML3D result.

## Protocol

- W1 emits a binary cue with two equally sized classes.
- W2-W8 emit distractor windows and do not repeat the cue.
- The recent Attention cache is W4, so W1 is absent at the W8 decision.
- correct carries the cue state; zero removes it; permute swaps states across
  trajectories; fresh resets state; minimal has only recent W4.
- Seed 20260902 and 256 paired samples are fixed.

## Result

Machine output: CAUSAL_MEMORY_BENCHMARK_RESULTS.json.

| Mode | Cue accuracy | Final-goal success | Root/constraint proxy |
|---|---:|---:|---:|
| Correct carry | 1.000 | 1.000 | 0.000 |
| Minimal-W4 | 0.500 | 0.500 | 0.500 |
| Zero state | 0.500 | 0.500 | 0.500 |
| Permuted state | 0.000 | 0.000 | 1.000 |
| Fresh state | 0.500 | 0.500 | 0.500 |

The harness passes its deterministic bookkeeping gate. It does not show that a
trained Mamba state in real motion data stores a useful cue; that claim requires
the real-model C/D/F gates and actor-cluster bootstrap.

## Reproduction

    python examples/causal_memory_benchmark.py \\
      --seed 20260902 --samples 256 \\
      --output /data4/zts/result/causal_memory_benchmark_20260902/results.json
