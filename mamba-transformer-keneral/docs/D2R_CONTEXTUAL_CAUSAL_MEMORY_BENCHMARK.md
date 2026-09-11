# D2R Contextual Causal-Memory Benchmark

Date: 2026-09-03
Status: **PASS (synthetic mechanism evidence only)**

## Purpose

The benchmark tests the user's semantic hypothesis directly: the same state
transition has a different meaning under different text context. The W1 target
is cue XOR context, W2-W7 are distractors, W1 leaves the W4 Attention cache,
and W8 must recover the target through fixed-capacity content memory.

## Results

| Condition | Accuracy |
|---|---:|
| Correct contextual memory | 1.0000 |
| Zero delta at W1 | 0.5000 |
| Permute delta at W1 | 0.0000 |
| Write off at W1 | 0.5000 |
| Readout off after W4 | 0.5000 |
| Permute memory after W4 | 0.0000 |
| Permute text context at W1 | 0.0000 |
| Minimal-W4 control | 0.5098 |

The content decoder, write gate, StateProjector, and Mamba each received
nonzero task gradients on 799/800 updates; the read scale did so on 800/800.
Memory capacity stayed fixed at K=2 after W1 eviction.

## Boundary

This result proves that the implemented SequenceKernel can learn a
text-conditioned delta code and use it causally after recent KV eviction. It
does not prove that BONES motion delta has independent value. That question is
answered only by the real-motion paired interventions in
`D2R_STEP250_CAUSAL_INTERVENTION_RESULTS.json`.

Raw artifact:
`/data4/zts/result/stage_d2r_contextual_cue_v3_20260903/results.json`.
