# Mamba Integration Contract

> 2026-08-31 historical-status addendum: references below to an unimplemented
> cross-window interface are historical. Accepted-window lifecycle, bounded
> Attention/Mamba state, differentiable rollout, and causal state interventions
> now exist. This does not imply a positive result: the preregistered 100/250
> gate failed. See `MAMBA_LONG_SEQUENCE_EVIDENCE_AUDIT_20260831.md`.

Last updated: 2026-07-20.

Audience: collaborators connecting the Mamba-side operator to downstream
training, simulation, or server-side experiments.

For ARDY-specific replacement details, also read:

```text
docs/ardy_mamba_replacement_plan.md
```

## 1. Responsibility Split

The Mamba side owns the **sequence operator and memory interface**.
Downstream motion / ARDY / G1 code owns **data, task loss, simulator runtime,
and evaluation**.

Boundary:

```text
downstream motion system
  -> prepares chunks, conditions, masks, episode resets
  -> calls Mamba operator
  <- receives chunk outputs, next recurrent state, memory diagnostics
```

The Mamba-side code should not own simulator stepping, robot deployment,
dataset preprocessing, or benchmark reporting. The downstream side should not
write into Mamba internal state tensors directly.

## 2. Current Technical Target

Active target: **State-Transition Memory Operator**.

Instead of using a plain transformer-style token KV cache, the external memory
is built from Mamba recurrent state transitions:

```text
raw Mamba recurrent state
  -> StateProjector
  -> s_before, s_after, delta
  -> state-based Q / K / V
  -> fixed-capacity Event Memory
  -> memory-conditioned next chunk
```

Stage 1 is **SISO Mamba-3 only**. MIMO, CUTE step decode, learned memory
lifecycle, compressed memory, and downstream motion integration are later.

## 3. Master Checklist And Interface Maturity

Do not skip ahead.

The Mamba side has three interface levels:

```text
Level 0: planned interface, not implemented yet
Level 1: chunk wrapper implemented and Phase-0 verified
Level 2: memory-augmented operator implemented and verified
```

**Current state is Level 2** (corrected 2026-07-25 -- this section had gone
stale since 2026-07-20 and understated actual progress; the code below now
exists in this repo at `src/state_memory_operator/`, ported and cleaned up
from the `mamba_algorithm` research repo where it was validated).

Master checklist:

- [x] **Audit Mamba-3 state/chunk API.**
  Acceptance: identify the recurrent state tensor to expose, identify the
  kernel path for multi-token chunk continuation, and document why downstream
  code must not call Mamba internals directly.
- [x] **Verify SISO Mamba-3 kernel can run in isolation.**
  Acceptance: basic forward/backward smoke test runs outside ARDY.
- [x] **Implement `Mamba3ChunkWrapper.forward_chunk`.**
  Acceptance: exposes the planned `forward_chunk` API and returns an opaque
  `MambaState`. Done: `src/state_memory_operator/mamba3_chunk_wrapper.py`.
- [x] **Pass Phase-0 wrapper tests.**
  Acceptance: full-vs-chunk consistency, reset, determinism, and basic
  forward/backward tests all pass. Done: `tests/test_smoke.py` (9/9 passing,
  includes a per-row async reset test beyond the original scope -- see
  Section 6).
- [x] **Implement `StateProjector`.**
  Acceptance: projects `ssm_state` into stable per-chunk state summaries
  `s_before`, `s_after`, and `delta`. Done:
  `src/state_memory_operator/state_projector.py`.
- [x] **Pass state probe.**
  Acceptance: state summaries predict basic synthetic event attributes above
  random baseline. Done in the research repo (Stage-1 program); also found
  (2026-07-21/22) that state is a reliable INDEX but not a reliable PAYLOAD
  at realistic vocab/class-count scale -- see `state_qkv.py`'s docstring for
  why V is content-derived, not state-derived, as a result.
- [x] **Implement fixed-capacity oracle Event Memory.**
  Acceptance: oracle retrieval improves long-horizon synthetic tasks. Done
  and gone further: FIFO vs. Participative-eviction comparison validated
  end-to-end on a real (if synthetic) recall task (2026-07-25) -- FIFO shows
  a hard capacity cliff, Participative eliminates it. Done:
  `src/state_memory_operator/event_memory.py` (also adds per-row async
  reset and compaction, needed for real use but not exercised by the
  synthetic tasks -- see module docstring).
- [x] **Implement learned State-QKV retrieval.**
  Acceptance: learned retrieval beats random and approaches oracle behavior.
  Done: `src/state_memory_operator/state_qkv.py`, retrieval Hit@1 ~99.8% in
  the research repo's validated configuration.
- [x] **Implement ARDY-compatible `ArdyMambaBlock`.**
  Acceptance: same call signature as ARDY `TransformerEncoderBlock`; output
  shapes match `root_model` and `body_model` requirements. Done (2026-07-27):
  `src/ardy_adapter/ardy_mamba_block.py`.
- [x] **Pass ARDY drop-in smoke tests** (partial -- see caveats).
  Acceptance: single denoising step, one autoregressive window, short sample,
  and G1 export path run with the Mamba block. Single denoising step and one
  autoregressive window PASS on the real "core" checkpoint (real
  `model.__call__`, real weights, both `root_model`/`body_model` swapped):
  `SINGLE_WINDOW_SMOKE_OK`, output `(1, 40, 330)`, finite, verified to
  genuinely differ from the original transformer's output
  (`max_abs_diff=10.75`) -- a first version of this test swapped through
  the CFG wrapper instead of the raw denoiser and produced a false
  -positive no-op "pass"; fixed and re-verified, see
  `/data3/zts/log/2026-07-27_ardy_mamba_block_milestone_b_correction.md`.
  Short-sample NPZ export and the G1 MuJoCo `.csv` path were NOT exercised
  with the Mamba block (only with the original transformer block, during
  Milestone A). Full record:
  `/data3/zts/log/2026-07-27_ardy_mamba_block_milestone_b.md`.
- [ ] **Add cross-window state/memory for ARDY.**
  Acceptance: memory is read during denoising and committed only after the
  final clean/generated window. NOT started -- `write_mask` on
  `EventMemory.write` and the `update_memory` flag on
  `forward_chunk_with_memory` are already there to support "commit only on
  accept," but no ARDY-side wiring exists yet.
- [ ] **Run final comparison.**
  Acceptance: compare ARDY baseline, Mamba-denoiser ARDY, and
  memory-augmented Mamba on agreed latency, memory, and motion metrics. NOT
  started -- blocked on the three items above.

## 4. Current Mamba-Side Progress

Current status checklist:

- [x] Mamba-3 state/chunk API audit is complete.
- [x] SISO kernel smoke and basic forward/backward memory checks are complete.
- [x] `Mamba3ChunkWrapper` is implemented and Phase-0-verified (`src/state_memory_operator/`).
- [x] `StateProjector` is implemented.
- [x] Event Memory is implemented (FIFO + Participative, unified, with
  per-row async reset and compaction).
- [x] Learned State-QKV retrieval is implemented and validated.
- [x] `ArdyMambaBlock` exists (`src/ardy_adapter/`) and passes shape/mask/
  backward/single-window smoke tests on the real "core" checkpoint on the
  server GPU.
- [ ] Cross-window Event Memory is NOT wired into ARDY's generation loop yet
  (Milestone D) -- this smoke test used the stateless single-window path.
- [ ] Training (Milestone C) has not started -- current `ArdyMambaBlock`
  weights are random init; motion quality is not yet meaningful.

What collaborators can rely on now:

- `from state_memory_operator import StateTransitionMemoryOperator, StateMemoryOperatorConfig`
  is real, importable, tested code -- see `examples/quickstart.py`.
- the validated defaults in `StateMemoryOperatorConfig`'s docstrings (e.g.
  prefer `topk >= 2`, prefer `memory_backend="participative"` if content
  gets re-queried, avoid memory-dropout unless you've checked your task's
  query density tolerates it).
- the replacement point for ARDY (Section 3, `docs/ardy_mamba_replacement_plan.md`).

What collaborators should NOT assume yet:

- that `ArdyMambaBlock` exists, or that ANY ARDY-side code has been touched;
- that the operator has been validated on anything but synthetic
  associative-recall tasks -- motion-specific content/event-label design
  (Section 9) is unstarted;
- that per-row async reset / write_mask / compaction (new in this repo,
  beyond what the research repo needed) have been exercised at real motion
  scale -- only unit-tested at small synthetic scale so far.

## 5. Level 0 Interface: SUPERSEDED -- see Level 1 below

This section originally planned the API before any code existed. It's kept
only for history; the interface that actually got implemented (Section 6)
differs in small ways from what was planned here (no `memory_condition` on
the memory-free call -- memory moved entirely to the Level 2 call instead;
`aux` is always returned rather than gated by a `return_aux` flag). Read
Section 6, not this one, for the real signature.

## 6. Level 1 Interface: Implemented (`state_memory_operator.StateTransitionMemoryOperator.forward_chunk`)

Level 1 status:

- [x] `Mamba3ChunkWrapper.forward_chunk` implemented
  (`src/state_memory_operator/mamba3_chunk_wrapper.py`).
- [x] Full-sequence vs chunked consistency test passed.
- [x] State reset test passed -- and made PER-ROW (an addition beyond the
  original plan: different rows in a batch can reset at different times,
  which real motion training needs and the original single-shared-pointer
  design didn't support).
- [x] Determinism test passed.
- [x] Basic forward/backward smoke test passed.
  (all five: `tests/test_smoke.py`)

The actual interface:

```python
outputs, next_state, aux = operator.forward_chunk(
    x=chunk,             # [batch, chunk_len, d_model]
    state=prev_state,    # Mamba3State or None
    reset_mask=reset_mask,  # [batch] bool or None, True = reset this row
)
```

`aux` keys actually returned: `state_shape`, `chunk_len`, `state_was_reset`,
`peak_gpu_memory` (matches what was planned).

Capability confirmed:

- [x] Chunked Mamba sequence modeling.
- [x] Recurrent state carried across chunks.
- [x] Per-row state reset between episodes.
- [x] `peak_gpu_memory` measurement (via `torch.cuda.max_memory_allocated()`).
- [ ] Latency benchmarking against a transformer baseline -- NOT done (no
  head-to-head timing comparison exists yet; do this as part of Milestone
  B/C in `ardy_mamba_replacement_plan.md`, not before).
- [x] Pure-Mamba baseline training support (validated in the research repo
  on synthetic tasks, prior to this repo's existence).

ARDY still does not depend on Mamba for production generation at this
level -- `ArdyMambaBlock` doesn't exist yet (Section 3).

## 7. Level 2 Interface: Implemented (`...forward_chunk_with_memory`)

Level 2 status:

- [x] `StateProjector` implemented.
- [x] State probes passed (research repo, Stage-1 program).
- [x] Oracle Event Memory improves long-horizon synthetic tasks -- and the
  FIFO-vs-Participative-eviction comparison, not originally scoped this
  early, is ALSO done (2026-07-25): FIFO shows a hard capacity cliff,
  Participative eliminates it.
- [x] Learned State-QKV implemented, validated after oracle memory showed
  a gain (order respected).

The actual interface (slightly reshaped from the original plan -- state and
the two state-summary scalars needed for querying/writing are bundled into
one `OperatorState` object so callers manage a single opaque handle instead
of three separate pieces):

```python
outputs, next_op_state, memory, aux = operator.forward_chunk_with_memory(
    x=chunk,                    # [batch, chunk_len, d_model]
    op_state=prev_op_state,     # from init_operator_state(), or previous call's next_op_state
    memory=prev_memory,         # from init_memory(), or previous call's memory
    reset_mask=reset_mask,      # [batch] bool or None
    do_read=True,               # query memory before running this chunk
    update_memory=True,         # write this chunk into memory after running it
    read_content=read_content,  # [batch, d_model], required if config.use_content_query
    write_content=write_content,  # [batch, d_model], required if update_memory
    write_mask=write_mask,      # [batch] bool or None -- which rows actually commit a
                                 # write this call (for ARDY: commit only on accepted
                                 # clean windows, not noisy denoising intermediates)
    keep_mask=keep_mask,        # [batch] bool or None -- memory-dropout mask; leave
                                 # None unless you've confirmed your task tolerates it
                                 # (see config.py docstring -- a 0.7 default caused a
                                 # real, initially-misdiagnosed training failure)
)
```

`aux` keys actually returned: `state_before_summary`, `state_after_summary`,
`state_delta`, `retrieval_topk_idx`, `retrieval_weights`, `retrieval_sims`,
`retrieval_entropy`, `injection_delta_norm`, `memory_usage_norm`,
`peak_gpu_memory`. (`retrieval_hit_at_k`/`retrieval_rank` from the original
plan aren't produced automatically -- computing them needs a ground-truth
target slot, which is task-specific; derive them from `retrieval_topk_idx`
the same way `mamba_algorithm`'s synthetic-task scripts do if your task has
one.)

Capability confirmed:

- [x] Compressed recurrent state preserved across chunks.
- [x] Explicit long-term event memory retrieval.
- [x] Long-horizon recall decay eliminated in synthetic tasks (Participative
  backend) / hard-cliffed as expected (FIFO backend) -- both validated.
- [x] Diagnostics (`aux`) showing whether memory is used.
- [ ] Any of this validated on real motion data or motion-specific content
  representations -- NOT done (Section 9 below is still unstarted).

## 8. Final ARDY-Facing Interface

The final ARDY-facing block should preserve ARDY's transformer-block signature
and hide Mamba internals behind optional state/memory arguments:

```python
class ArdyMambaBlock(nn.Module):
    def forward(
        self,
        x,
        x_pad_mask,
        text_feat,
        text_feat_pad_mask,
        timesteps,
        first_heading_angle=None,
        token_index=None,
        state=None,
        memory=None,
        reset_mask=None,
        update_memory=False,
        return_aux=False,
    ):
        ...
```

Return modes:

```python
# drop-in ARDY mode
output = block(...)

# memory-aware mode
output, next_state, next_memory, aux = block(..., return_aux=True)
```

Final capability checklist:

- [ ] Replace ARDY `root_model` and `body_model`.
- [ ] Keep ARDY diffusion, autoencoder, motion representation, and export code
  unchanged;
- [ ] Run autoregressive diffusion with Mamba sequence mixing.
- [ ] Optionally carry Mamba state / Event Memory across AR windows.

## 9. Downstream Inputs

The downstream side provides:

```text
chunk tensor x
condition tensor, if used
target tensor and task loss
episode_id or reset_mask
sequence position / chunk id
event labels for synthetic or motion-event retrieval probes
evaluation metrics
```

For motion tasks, useful event labels include:

```text
contact onset / release
foot support switch
root velocity transition
action phase transition
repeated pose motif
object-interaction transition
```

## 10. Training Modes

Expose these modes as configs:

```text
baseline_mamba3:
  pure Mamba-3, no external memory

oracle_event_memory:
  ground-truth event id retrieves memory value

state_qkv_memory:
  learned State-QKV retrieval

injection_ablation:
  output injection / input injection / state correction
```

Recommended curriculum:

- [ ] Train pure Mamba-3 baseline.
- [ ] Freeze backbone; train projector / memory / injection modules.
- [ ] Joint finetune with smaller backbone LR.

## 11. Minimal Integration Checklist

Before downstream work depends on this operator, verify (all done as of
2026-07-25, at synthetic/unit-test scale -- re-verify at real motion scale
before depending on this in production, per the caveats in Section 4):

- [x] `forward_chunk` returns the same outputs as full-sequence execution within
  tolerance. (`tests/test_smoke.py::test_chunk_consistency_matches_full_sequence`)
- [x] State reset prevents leakage across episodes, PER ROW.
  (`test_reset_mask_is_per_row_and_isolates_rows`)
- [x] Carrying state without reset changes the second episode, proving state is
  active. (implied by the same test: the non-reset row's output differs from
  a fresh-state run)
- [x] Determinism holds under the same seed and input. (`test_determinism`)
- [x] Baseline training loss decreases on a small synthetic task. (research
  repo, Stage-1 program, prior to this repo's existence)
- [x] State projector features predict basic event attributes better than
  random. (research repo, Stage-1 program)
- [x] Oracle memory improves long-horizon performance before learned QKV is
  built. (research repo, order respected: oracle memory tested and shown to
  help BEFORE learned State-QKV work started)
