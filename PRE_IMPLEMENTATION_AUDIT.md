# Pre-Implementation Audit (2026-09-02)

This audit is the required precondition for the A-I long-memory work described
in 9.2-revise.MD. It records existing implementation and bounded planned
changes before any new training run.

## Existing capabilities

- HumanML3D conversion and evaluator modules exist in
  mamba-transformer-keneral/src/humanml3d_conversion.py and
  src/humanml3d_official_evaluator.py. Existing reports document the
  Core27-to-263D round-trip bias and official-metric block.
- SequenceKernel provides Minimal and Hybrid modes, bounded sliding KV,
  Mamba3 chunk execution, read-only forward, and accepted-window commit().
- TokenStateInteraction implements naive_gated_add, state_query, and
  state_kv_modulation using the existing projector, summary, delta, and
  address path.
- StateProjector and project_transition define s=P(S) and
  delta=P(S_after)-P(S_before).
- Mamba3ChunkWrapper exposes recurrent carry/reset, while diffusion_memory.py
  and history_commit.py enforce accepted clean, recentered window commits.
- Differentiable generated multi-window rollout, carry/fresh/zero/freeze/
  permutation/readout-off interventions, actor-cluster bootstrap, long-horizon
  metrics, and W4/W8/W16/ParamMatch controls already exist.
- Existing BONES-SEED and HumanML3D reports contain locked panels, 50k
  checkpoints, 9s/20s/40s metrics, timing, and persistent-state accounting.
- StateTransitionMemoryOperator and EventMemory provide a standalone
  fixed-capacity state/content prototype with explicit read-before-write
  ordering, but are not wired into the ARDY Attention content path.

## Reuse without redesign

- Reuse StateProjector, StateTransition, state_summary, state_delta, and
  state_to_address; do not create a second state representation.
- Reuse the current Mamba wrapper, cache lifecycle, accepted-window
  transaction, reset/permutation helpers, protocol lock, paired metrics, and
  bootstrap code.
- Reuse neutral interaction initialization and checkpoint loading conventions.
  New content-memory gates must be identity-safe at initialization.
- Reuse existing W4/W8/W16/ParamMatch definitions and locked long-sequence
  panels rather than changing panel semantics mid-experiment.

## New implementation required

1. Rotation/semantics-preserving HumanML3D bridge audit and regression
   calibration; if it cannot be repaired, publish a clearly labelled
   Core27-native evaluator fallback.
2. B_BASELINE_MANIFEST.json for the healthy Native Hybrid 50k checkpoint,
   including config, split/panel hashes, checkpoint SHA/path, metrics, and
   commands.
3. A deterministic BONES-SEED cue benchmark where a W1 cue is absent from the
   W4 cache by W8, with correct/zero/permute/fresh Hybrid and Minimal controls.
4. Small 100/250-step C gates for existing state_query and
   state_kv_modulation.
5. D content decoder and fixed-capacity persistent content slots layered on the
   existing [s, delta] -> state_to_address branch, with explicit write/read
   interventions and contract tests.
6. Only if C/D gates pass, generated-history 5-8 then 8-16 differentiable
   training, formal state permutation, fairness controls, and multi-seed
   confirmation. Fused kernels remain last.

## Historical bugs/contracts to preserve

- Never commit Mamba state, summary, delta, or content memory during noisy
  diffusion forwards; one accepted clean/recentered window commits once.
- Do not use GT history in evaluation or mix teacher-history and
  generated-history protocols.
- Do not treat P(S_after-S_before) as the delta; the defined delta is
  P(S_after)-P(S_before).
- Do not inject evicted content into C; only D may expose persistent content to
  future Attention.
- Do not claim official HumanML3D learned metrics while the Core27 bridge has
  material round-trip bias. BONES-SEED learned metrics remain unavailable.
- Do not infer long-memory superiority from denoising loss, one checkpoint,
  unpaired panels, or a failed/selection-only gate.
- Do not compare Hybrid only against Minimal-W4; retain ParamMatch and W8/W16.
- Do not start H or I before earlier gates pass.

## Planned files

- New audit/results: A_HML_BRIDGE_AUDIT.md, B_BASELINE_MANIFEST.json,
  CAUSAL_MEMORY_BENCHMARK.md, and stage C/D/E/F/G summaries and machine
  results.
- Existing conversion/evaluator modules and tests, only where required by the
  bridge audit and regression tests.
- New benchmark/config/launcher modules under examples/, configs/, and tests/.
- New content-memory module under src/state_memory_operator/, integrated via an
  opt-in config path in the Hybrid sequence operator.
- No checkpoint, dataset, generated-motion, or external result directory is
  committed.

## Gate policy

Each stage must emit a summary, machine-readable result, command/config
snapshot, raw per-window metrics, bootstrap summary, abnormal-case audit, and a
PASS or FAIL decision. A failed gate is retained and stops that branch; it does
not authorize silently scaling the training budget.
