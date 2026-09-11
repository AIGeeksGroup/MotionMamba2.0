# Baseline + C1 handoff

## Completed baseline

The repository preserves four CLI modes: `--original` (legacy ARDY), `--dit`
(DiT modernization), `--minimal` (recommended attention-only baseline), and
`--hybrid` (the existing experimental Attention-Mamba integration scaffold).

Minimal is deliberately a standard baseline:

```text
AR diffusion + DiT + native attention backend
+ chunk-level sliding-window KV cache + attention sinks
+ no Mamba
```

Its noisy denoising calls only read memory. Each accepted clean chunk commits
exactly once, with root/body rollback on partial failure.

## Current C1 evidence and status

Attention-Mamba complementarity is supported as architecture-specific on the
Stage-08 development protocol: Minimal+Mamba averaging improved 29.12%, while
Minimal+Minimal improved only 0.31%. Root strongly favors Mamba. Body is
Attention-dominant but retains a lightweight Mamba complement.

The current research candidate is a motion-structured Attention-Mamba hybrid:
Mamba-dominant root and Attention-dominant body with a lightweight Mamba
complement. The final paper architecture is **not yet fully frozen**, pending
full-dataset training and a standardized benchmark. In particular, Stage 09
did not select a final C1: its corrected H0 missed the required gain gate and
the aggregate structured direction was not robust across trajectories. The
existing `--hybrid` remains an integration scaffold, not a finalized paper
operator.

Tested negative evidence is retained in the Stage-06--09 logs: temporal state
delta, direct state conditioning, recent-KV state routing, out-of-window
EventMemory, and explicit token-state delta/delta-rule correction were not
selected.

## Next phase

The handoff boundary is full-dataset training, standardized benchmarking,
long-horizon evaluation, simulation, and real-robot evaluation. Contribution
2 (an integrated/custom kernel for the eventual finalized hybrid mechanism)
is separate collaborator-owned future work and is not included here.
