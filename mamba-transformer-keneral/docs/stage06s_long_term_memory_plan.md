# Stage 06S Plan — Dual-Timescale Long-Term Event Memory

Date: 2026-08-15
Status: Gate A in progress

Stage 06S does not reopen the failed direct state-conditioning or recent-KV
routing directions. It tests a distinct question: whether accepted token
content older than the frozen Minimal window has useful predictive value.

## Frozen short-term baseline

Stage-05 Minimal remains unchanged:

```text
ARDY autoregressive diffusion
+ DiT
+ four-chunk Sliding-KV
+ four attention sinks
+ no Mamba
```

Parent checkpoint:
`/workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt`, SHA-256
`acbdcc78c6594ad690a1d3a76b129665198fd142e487a1eb64bf43163a1b949f`.

## Reused long-term machinery

The existing `StateTransitionMemoryOperator` already composes
`Mamba3ChunkWrapper`, `StateProjector`, `StateQKVEncoder`,
`ContentValueEncoder`, and `EventMemory`. Its temporal contract is suitable:
query from committed pre-current state, token-derived payload, read-only noisy
calls, and explicit accepted writes. It will not be rewritten. If later gates
are reached, only a thin ARDY/Minimal adapter is permitted.

## Ownership rule

For current chunk `t` and short-term window `W=4`:

```text
Sliding-KV owns:  C_(t-W) ... C_(t-1)
EventMemory owns: C_0 ... C_(t-W-1)
```

Recent chunks must not be simultaneously active in EventMemory retrieval.
The preferred implementation may write at accepted commit and mask eligibility
until age exceeds W, provided the observable ownership is identical to an
eviction handoff.

## Long-horizon evaluation cache

The existing four-chunk cache has no out-of-window event. Build a bounded real
teacher set at horizons 8, 16, and 32 using the existing ARDY autoregressive
generation path and exact Stage-05 text features. To bound disk and replay
cost, store accepted motion and only the last current chunk's first, middle,
and final denoising calls. Those calls contain the full accepted-history
root/body inputs needed to reconstruct every historical chunk. Replay parity
must remain exact before Gate A.

## Gate A — raw long-term oracle ceiling

Freeze Minimal. For each last-current case, preserve four recent chunks and
the sinks/current K/V. Evaluate:

- recent-only Minimal;
- a deterministic random old chunk;
- most recently evicted old chunk;
- every old singleton and choose Oracle Top-1;
- greedy Oracle Top-2: keep Oracle Top-1 and test every second old chunk.

Raw old content is injected as its exact accepted token K/V. This is a ceiling
test, not the final EventMemory representation. A diagnostic cache with larger
physical capacity is populated once, cloned, and filtered by chunk ID for each
condition; the readable set always obeys the ownership rule. No model weight
changes.

Report mean/median MSE, paired absolute/relative gain, trajectory bootstrap
confidence interval, fraction improved, selected ages, and horizon strata.
Gate A passes only if the old-memory ceiling is practically meaningful,
preferably grows with horizon, and clearly exceeds the approximately 1%
recent-routing oracle ceiling from Stage 06R.

## Conditional Gates B--D

Only if Gate A passes:

1. Gate B compares raw oracle content with the existing
   `ContentValueEncoder -> memory_proj` injection path. It must retain a
   substantial fraction of the raw ceiling.
2. Gate C trains/evaluates the minimum state-addressing modules against oracle
   old-event labels, with Recent-old and token-only controls. Frozen random
   Mamba is diagnostic only; a small purpose-specific state training step is
   allowed only after a strong Gate-A ceiling and its own tiny-overfit gate.
3. Gate D injects learned state-indexed EventMemory into frozen Minimal and
   must improve denoising reproducibly, preferably over naive gated-add.

State+delta is an ablation and earns no claim unless it beats state-only.
FIFO/Participative comparison, real C1 training, and 4/8/16/32 generation are
opened only after Gates A--D.

## Hard stop

Failure at any gate stops the later gates. No full MotionMamba training,
custom kernel, FA4 fusion, Deep Forcing, simulation, or robot work is in scope.
