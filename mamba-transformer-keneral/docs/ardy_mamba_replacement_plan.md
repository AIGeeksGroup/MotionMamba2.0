# ARDY to Mamba Replacement Plan

> 2026-08-31 historical-status addendum: unchecked cross-window items below no
> longer describe the current implementation. Persistent KV/Mamba carry,
> differentiable multiwindow rollout, and state interventions are implemented
> and tested. The evidence gate is nevertheless `NOT SUPPORTED`; see
> `MAMBA_LONG_SEQUENCE_EVIDENCE_AUDIT_20260831.md`. The original checklist is
> preserved as historical planning context.

Last updated: 2026-07-20.

Post-Stage-05 scope note (2026-08-14): this document began as an engineering
replacement plan. Its paper-level contribution mapping is superseded by
`docs/paper_architecture_scope.md`. Original, DiT, and Minimal are standard
baselines; MotionMamba 2.0 begins only with Mamba-specific token/state
interaction. The current generic Hybrid path is scaffolding, not the final
paper method.

Purpose: define exactly how the Mamba-side work connects to ARDY-style
humanoid motion generation/control.

This document is for collaborators using
`https://github.com/neptune-T/mamba_transformer_keneral` as the server-side
handoff repo. It is not an experiment log.

## 1. Paper-Level Goal

Our humanoid-control paper has three contribution tracks:

1. **Real-time sequence modeling.** Replace transformer-style sequence mixing
   with Mamba-3 / linear-time kernels where possible. The goal is
   hardware-friendly inference with linear sequence complexity.
2. **Long-horizon autoregressive diffusion.** Generate/control motion by
   diffusion inside each chunk and autoregression across chunks:

   ```text
   chunk 1: cond -> diffusion -> output_1
   chunk 2: memory/cache + cond + output_1 -> diffusion -> output_2
   chunk k: memory/cache + cond + output_{k-1} -> diffusion -> output_k
   ```

   The key problem is error accumulation across chunks. Our Mamba-side answer
   is state-transition memory first, then compressed memory / sink-style memory
   after the minimal mechanism works.
3. **Humanoid benchmark and G1 evaluation.** Compare on HumanML3D, KITML,
   BABEL, ARDY-style motion generation, and eventually G1 simulation / real
   robot evaluation.

## 2. What ARDY Already Provides

ARDY already provides:

- autoregressive diffusion over motion windows;
- online text prompting;
- kinematic constraints through `motion_mask` and `observed_motion`;
- G1 skeleton assets and MuJoCo qpos export;
- interactive generation through `Ardy.autoregressive_step()`;
- non-interactive generation through `scripts/generate.py`;
- motion representation, autoencoder, hybrid root/body token conversion, and
  post-processing.

We should not rewrite those first.

The Mamba replacement should integrate at the model interface level: keep
ARDY's generation loop and motion I/O stable, and replace only the denoiser
sequence mixer first.

## 3. Actual Replacement Point

The first Mamba replacement target is the denoiser backbone, not the whole
ARDY system.

ARDY's denoiser is two-stage:

```text
AutoLatentTwostageDenoiser.forward(...)
  -> root_model: TransformerEncoderBlock
       predicts global root motion
  -> convert predicted global root to local root
  -> body_model: TransformerEncoderBlock
       predicts latent body motion conditioned on local root
  -> concatenate root prediction + body prediction
```

The current ARDY transformer block:

```text
TransformerEncoderBlock.forward(
    x,
    x_pad_mask,
    text_feat,
    text_feat_pad_mask,
    timesteps,
    first_heading_angle,
    token_index,
) -> output
```

Inside it:

```text
motion tokens x
  -> input projection
text tokens + diffusion timestep + optional heading
  -> prefix tokens
prefix tokens + motion tokens
  -> positional encoding
  -> SDPATransformerEncoder
  -> keep only motion-token outputs
  -> output projection
```

Mamba should first replace this block while preserving the same call signature.

## 4. What Mamba Replaces vs Does Not Replace

Mamba replaces:

```text
root_model sequence mixer
body_model sequence mixer
later: cross-window history/memory mechanism
```

Mamba does not replace in the first integration:

```text
Ardy.autoregressive_step
Ardy._generate_window
Diffusion / DDIMSampler
AutoLatentClassifierFreeGuidedModel CFG wrapper
HybridMotionConverter
motion_rep
autoencoder / FSQ VAE
G1 skeleton and MuJoCo export
post_process_motion
interactive demo UI
```

This keeps the ARDY runtime and downstream APIs stable while we test whether
Mamba improves speed / memory / horizon.

## 5. Drop-In Block API

This interface is the target, not current implemented code. The Mamba block
should expose the ARDY transformer block interface:

```python
class ArdyMambaBlock(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: torch.Tensor,
        first_heading_angle: torch.Tensor | None = None,
        token_index: torch.Tensor | None = None,
        state: "MambaState | None" = None,
        reset_mask: torch.Tensor | None = None,
        memory_condition: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        ...
```

Required tensor convention:

```text
x:                  [batch, num_tokens, input_dim]
x_pad_mask:         [batch, num_tokens], True = valid token
text_feat:          [batch, max_text_len, llm_dim]
text_feat_pad_mask: [batch, max_text_len], True = valid token
timesteps:          [batch]
first_heading_angle:[batch] or None
token_index:        [batch, num_tokens] or None
output:             [batch, num_tokens, output_dim]
```

For strict ARDY compatibility, `state`, `reset_mask`, `memory_condition`, and
`return_aux` can be ignored in the first drop-in version. They become active
when cross-window memory is added.

## 6. What We Do Now

**Corrected 2026-07-25** -- this section had gone stale since 2026-07-20 and
understated actual progress. Current status checklist:

- [x] Mamba-3 SISO kernel can run in isolation.
- [x] `Mamba3ChunkWrapper.forward_chunk` implemented and Phase-0-verified.
- [x] `StateProjector`, Event Memory (FIFO + Participative), and learned
  State-QKV retrieval are ALL implemented and validated -- see
  `src/state_memory_operator/` and `docs/mamba_integration_contract.md`
  Sections 3/6/7 for the full status.
- [ ] No ARDY Mamba replacement code is implemented yet -- `ArdyMambaBlock`
  does not exist. This is the actual next step now, not the chunk wrapper.

Current interface status:

- [x] `forward_chunk`: implemented, real, tested
  (`state_memory_operator.StateTransitionMemoryOperator.forward_chunk`).
- [x] `forward_chunk_with_memory`: implemented, real, tested (same class).
- [ ] `ArdyMambaBlock`: planned API, not implemented yet -- this is the
  actual immediate work now.

The immediate work is still not to touch ARDY's own code -- but it's no
longer "make the chunk interface real," since that's done. It's building
`ArdyMambaBlock` around the now-real `state_memory_operator` package.

## 7. The Chunk Interface Is Verified -- What That Unlocks

`Mamba3ChunkWrapper.forward_chunk` passed Phase-0 tests; the interface is
usable for non-ARDY synthetic tests (and has been used for exactly that --
see the research repo's whole Stage-1 program):

```python
outputs, next_state, aux = operator.forward_chunk(
    x=chunk,
    state=prev_state,
    reset_mask=reset_mask,
)
```

Implemented and verified:

- [x] Chunked SISO Mamba inference/training.
- [x] State carry across chunks.
- [x] Episode reset -- and made PER-ROW, which the original plan didn't
  call for but real motion training (asynchronous per-sequence resets in a
  batch) needs.
- [x] Peak-memory measurement.
- [ ] Latency measurement against a transformer baseline -- NOT done; do
  this as part of Milestone B/C below, since it needs an actual ARDY
  -shaped comparison to mean anything.
- [x] Synthetic baseline training.

Also implemented, going beyond what this section originally scoped for
"after one step is verified" (validated in order: oracle memory shown to
help BEFORE learned QKV work started, per Section 12's stage order):

- [x] Learned State-QKV retrieval (Hit@1 ~99.8% on the research repo's
  validated synthetic task).
- [x] Long-horizon memory correction -- FIFO vs. Participative-eviction
  comparison validated (2026-07-25): FIFO shows a hard capacity cliff,
  Participative eliminates it out to 3.6x raw capacity.

Still NOT implemented (unchanged from the original plan -- these are
genuinely ARDY-side, not Mamba-side, work):

- [ ] ARDY root/body replacement.
- [ ] Cross-window Event Memory wired into ARDY's actual generation loop
  (the Mamba-side primitives for this -- `write_mask`, `update_memory`,
  per-row `reset_mask` -- exist and are unit-tested, but nothing calls them
  from ARDY yet).

## 8. First ARDY Implementation Mode -- DONE (2026-07-27)

First version: **stateless denoiser-block replacement**. Implemented:
`src/ardy_adapter/ardy_mamba_block.py` in this repo.

Keep the ARDY windowing and diffusion loop unchanged. Replaced:

```python
self.root_model = TransformerEncoderBlock(...)
self.body_model = TransformerEncoderBlock(...)
```

with:

```python
self.root_model = ArdyMambaBlock(...)
self.body_model = ArdyMambaBlock(...)
```

The first Mamba block:

- [x] Builds the same prefix tokens as `TransformerEncoderBlock` (imports
  and reuses its own `embed_text`/`embed_timestep`/positional-encoding
  components rather than reimplementing them).
- [x] Concatenates prefix + motion tokens.
- [x] Runs a stack of causal Mamba-3 layers (via
  `state_memory_operator.Mamba3ChunkWrapper`), each wrapped in residual +
  LayerNorm.
- [x] Returns only the motion-token outputs.
- [x] Projects to `output_dim`.

Validated on the server (GPU, real "core" checkpoint, real
`ardy.model.load_model` + real generation call path, not a hand-rolled
approximation):

- [x] Shape parity (root + body stage, real denoiser config values).
- [x] Mask parity (padded tokens don't crash).
- [x] Backward pass (gradients finite, reach input and first Mamba layer).
- [x] Single denoising step + one autoregressive window with BOTH
  `root_model` and `body_model` swapped: `SINGLE_WINDOW_SMOKE_OK`, output
  `(1, 40, 330)`, all finite, and (after a correction -- see below)
  verified to genuinely differ from the original transformer's output
  (`max_abs_diff=10.75`). Random-init weights, so motion quality is
  expected garbage -- this validates the structural claim only (see
  Section 14 Milestone B below for the full record).

  **Correction (same day)**: the first version of this test swapped
  `model.denoiser.root_model` where `model.denoiser` is the CFG wrapper
  (`AutoLatentClassifierFreeGuidedModelSeparated`), not the raw denoiser.
  The wrapper's `__getattr__` delegates READS to the raw denoiser
  (`self.model`), making the swap LOOK like it worked, but the assignment
  itself created a disconnected attribute on the wrapper -- `forward()`
  still called the original, untouched transformer the whole time. Fixed
  to swap through `model.denoiser.model` instead, and added an identity
  assert + a baseline-vs-Mamba output-difference check so a future no-op
  swap can't pass silently again. Full note:
  `/data3/zts/log/2026-07-27_ardy_mamba_block_milestone_b_correction.md`.

This version answers:

```text
Can Mamba replace ARDY's transformer denoiser block without breaking generation?
    YES, structurally -- confirmed 2026-07-27 (see Section 14, Milestone B).
Does it reduce latency or memory for the same ARDY window?
    NOT YET MEASURED -- no head-to-head timing/memory comparison exists yet.
Can it train on the same denoising target?
    NOT YET TESTED -- Milestone C, next.
```

**Known first-version simplifications, not yet resolved** (see the module
docstring in `ardy_mamba_block.py` for full detail): the mixer is causal,
not bidirectional, within a window (a real representational difference
from the original self-attention, which the empirical test above is
exactly meant to probe); padding is handled by zeroing content rather than
true attention-style masking (checked to not crash, not checked against
ARDY's real batch-padding convention at scale); no checkpoint weight
sharing with the original block (fresh random init -- needs its own
training, Milestone C).

It does not yet solve long-horizon memory (Milestone D, separate).

## 9. After ARDY Drop-In Block Is Verified

After `ArdyMambaBlock` exists and passes shape / denoising / one-window tests,
the ARDY-facing interface is:

```python
output = block(
    x,
    x_pad_mask,
    text_feat,
    text_feat_pad_mask,
    timesteps,
    first_heading_angle,
    token_index,
)
```

This can implement after verification:

- [ ] Drop-in replacement of `root_model`.
- [ ] Drop-in replacement of `body_model`.
- [ ] Single denoising step.
- [ ] One autoregressive generation window.
- [ ] Short ARDY generation sample.
- [ ] Latency / memory comparison against transformer block.

It still cannot implement:

- [ ] Cross-window memory.
- [ ] Event retrieval.
- [ ] State-based long-horizon correction.

## 10. Final Work After Memory Is Done

After StateProjector + Event Memory + State-QKV pass the required gates, the
final interface is:

```python
output, next_state, next_memory, aux = block(
    x,
    x_pad_mask,
    text_feat,
    text_feat_pad_mask,
    timesteps,
    first_heading_angle,
    token_index,
    state=prev_state,
    memory=prev_memory,
    reset_mask=reset_mask,
    update_memory=commit_clean_window,
    return_aux=True,
)
```

This can implement after final memory work:

- [ ] Mamba state carry across autoregressive windows.
- [ ] Event Memory retrieval across windows.
- [ ] Memory read during diffusion.
- [ ] Memory commit after final clean/generated window.
- [ ] Long-horizon drift / recall correction.
- [ ] Diagnostics for retrieval and memory usage.

## 11. Long-Horizon Upgrade

ARDY currently manages long generation by cropping history to a trained window.
In `scripts/generate.py`, `--history_frames` caps visible history; in
`Ardy.__call__`, each autoregressive step calls `_generate_window(...)` with
`history_start_frame` and `history_end_frame`.

This is practical for real-time interaction, but it means the denoiser only
sees a bounded history window.

The Mamba upgrade should add cross-window memory:

```python
outputs, next_state, next_memory, aux = model.forward_chunk_with_memory(
    x=chunk,
    state=prev_state,
    memory=prev_memory,
    reset_mask=reset_mask,
    update_memory=True,
)
```

For ARDY integration, the memory should be updated only after a clean generated
window is accepted, not at every noisy denoising intermediate. The diffusion
loop should treat memory as read-only during denoising, then commit memory from
the final clean/generated chunk.

This preserves the rule:

```text
inner chunk = diffusion
inter chunk = autoregressive memory update
```

## 12. State and Memory Semantics

Mamba recurrent state is continuous compressed history. Event Memory is
explicit addressable history.

Stage order:

```text
SISO Mamba block
  -> Mamba3ChunkWrapper.forward_chunk
  -> StateProjector(ssm_state)
  -> oracle Event Memory
  -> learned State-QKV
  -> compressed/sink memory after oracle memory works
```

Do not train learned QKV or compressed memory before the oracle-memory gate
shows a clear gain on long-horizon synthetic tasks.

## 13. Current Mamba-Side Progress

**Corrected 2026-07-25.** Current status:

```text
done: Mamba-3 state/chunk API audit
done: SISO kernel smoke and basic forward/backward memory checks
done: Mamba3ChunkWrapper.forward_chunk (Phase-0 tests passing)
done: StateProjector
done: Event Memory (FIFO + Participative, unified, validated:
      FIFO hard capacity cliff vs. Participative flat accuracy)
done: learned State-QKV retrieval (~99.8% Hit@1 on validated synthetic task)
not yet: ArdyMambaBlock
not yet: ARDY root/body replacement
not yet: cross-window Event Memory integration INTO ARDY
```

Code lives in `src/state_memory_operator/` in this repo, importable as
`from state_memory_operator import StateTransitionMemoryOperator,
StateMemoryOperatorConfig` (see `examples/quickstart.py`,
`tests/test_smoke.py`).

What this means for ARDY collaborators:

- The interface contract is ready AND the Mamba-side operator it describes
  is now real, tested code -- not just a plan.
- The first ARDY replacement target is clear (Section 3).
- The drop-in `ArdyMambaBlock` is NOT implemented yet -- this is the actual
  next step, not a formality; nothing in this repo has touched ARDY.
- ARDY-side data, diffusion, motion representation, and export code should
  remain unchanged until `ArdyMambaBlock` passes its own drop-in tests
  (Milestone B below).
- Content/event-label design for motion (Section 9) is unstarted --
  everything validated so far used discrete synthetic-vocabulary tasks, not
  continuous motion latents. Don't assume the memory mechanism transfers to
  motion content without that design work.

## 14. ARDY Integration Milestones

### Milestone A: Read-only ARDY baseline -- DONE (2026-07-27)

Run ARDY generation without modifying ARDY:

```text
scripts/generate.py
scripts/interactive_demo/generation.py
```

Checklist:

- [x] Run unmodified ARDY generation (Core and G1 models).
- [x] Record model name (`core` / `g1`, released
  `ARDY-Core-RP-20FPS-Horizon40` / `ARDY-G1-RP-25FPS-Horizon52`).
- [x] Record duration (1.0s smoke; up to 4.0s for later action sets).
- [x] Record diffusion steps (1 for initial smoke; 4/8/10 for later sets).
- [ ] Record history frames explicitly (not isolated as its own number).
- [ ] Record generation time (console progress observed, not instrumented
  as a clean wall-clock number -- do a timed run before quoting speed).
- [ ] Record peak GPU memory (not captured in this pass).
- [x] Record output `.npz` / `.csv` path -- see
  `/data3/zts/results/ardy_smoke/` and `/data3/zts/results/ardy_smoke/actions_g1/`
  on the server (multiple prompts, both Core and G1, G1 CSVs are the MuJoCo
  qpos export). Full detail: `mamba_project/logs/daily/2026-07-27.md`.

### Milestone B: Drop-in Mamba denoiser block -- DONE (2026-07-27)

Implemented `ArdyMambaBlock` with the same forward signature as
`TransformerEncoderBlock` (`src/ardy_adapter/ardy_mamba_block.py`).

Validation checklist:

- [x] Shape parity: root output and body output match ARDY expected shapes
  (real denoiser config: `latent_dim=1024`, `num_layers=8`).
- [x] Mask parity: valid/padded tokens do not crash.
- [x] Single denoising step runs (real "core" checkpoint, real
  `model.__call__` path, not simulated).
- [x] One autoregressive window runs (40 frames = `gen_horizon_len`,
  output `(1, 40, 330)`, all finite).
- [ ] Short generation sample writes `.npz` -- the smoke test used the raw
  tensor output directly and did not run it through
  `model.motion_rep.inverse` / NPZ export; do this next if a saved sample
  is needed (motion will be low-quality garbage regardless, given random
  Mamba init -- not useful evidence until Milestone C training happens).
- [ ] G1 sample writes MuJoCo `.csv` -- NOT tested; only the "core" model
  was used for this milestone.

Full record: `/data3/zts/log/2026-07-27_ardy_mamba_block_milestone_b.md`
(server) and `mamba_project/logs/daily/2026-07-27.md` (control repo).

### Milestone C: Basic training -- FIRST RUN DONE (2026-07-27), needs more rigor

Train the Mamba-denoiser version on the same ARDY-style target.

**No real ARDY training data/loop exists (confirmed -- ARDY's public
release is inference-only).** First version uses distillation instead: a
small pseudo-dataset generated by the frozen pretrained teacher itself
(16 prompts, full denoising schedule), then standard diffusion
x0-prediction training against those cached samples, freezing everything
except the two Mamba blocks.
(`examples/train_ardy_mamba_distill.py`.)

Checklist:

- [x] Record loss curve: 1.164 (step 0) -> 0.017 (step 1999), wandb run
  https://wandb.ai/2980976495-peking-university/ardy-mamba-distill/runs/8yjvnr43
- [ ] Record peak GPU memory (not captured in this pass).
- [ ] Record tokens/sec or windows/sec (not captured in this pass).
- [x] Record short qualitative generation with the TRAINED checkpoint
  through the real multi-step `model.__call__` path (`examples/ardy_mamba_eval_and_render.py`):
  generated + rendered (custom matplotlib/imageio stick-figure MP4s, since
  ARDY's own `scripts/visualize.py` is an interactive viser viewer, not a
  file exporter) on one TRAINING prompt ("A person jumps.") and one
  HELD-OUT prompt not in the distillation set ("A person dances happily."),
  for both the teacher and the trained Mamba blocks.
- [x] Compare to transformer denoiser under same settings (qualitative,
  spot-checked frames only -- not a rigorous quantitative comparison yet):
  teacher looks coherent on both prompts (expected). Trained Mamba looks
  structurally plausible on BOTH prompts at the frames checked (correct
  proportions, no NaN/degenerate joints) -- one mid-sequence frame on the
  held-out prompt showed crossed legs, but start/end frames of that same
  clip are clean upright poses, consistent with a normal stepping/turning
  transition rather than a failure. This is NOT strong evidence of
  generalization (3 spot-checked frames, one held-out prompt, no
  quantitative metric) -- it only rules out gross breakage. A real
  generalization claim needs the items below.
  Videos: `results/videos/2026-07-28_ardy_mamba_eval/` (control repo,
  gitignored -- not committed, per the "no large media in git" rule).

Full record:
`/data3/zts/log/2026-07-27_ardy_mamba_distill_milestone_c_first_run.md`.

**Remaining for a rigorous Milestone C**: quantitative joint-position
error / motion-quality metric against the teacher on a larger held-out
prompt set (not just one), peak-memory and throughput numbers, and a
larger/more diverse pseudo-dataset to reduce the memorization concern
before treating any generalization claim as established.

### Milestone D: Cross-window state/memory

Add state handoff and Event Memory across AR windows.

Validation checklist:

- [ ] Reset prevents episode leakage.
- [ ] State carry changes later windows.
- [ ] Oracle memory improves long-horizon synthetic recall.
- [ ] Motion drift reduces at longer horizons.

## 15. Final contribution boundary

Standard baselines are Original ARDY, DiT, and Minimal. Minimal is AR diffusion
+ DiT + native FlashAttention + sliding-window K/V + attention sinks, with no
Mamba and no novelty claim for those components.

Contribution 1 begins above Minimal:

```text
token-level representation from Attention
+ state-level representation/propagation from Mamba
-> model their interaction or delta
-> Hybrid Attention–Mamba modeling
```

The final mathematical interaction is deliberately unresolved. The existing
parallel attention/Mamba paths with scalar gated addition are an integration
scaffold and must not be relabeled as the final contribution.

Contribution 2 follows only after Contribution 1 is finalized:

```text
final token/state interaction
-> corresponding efficient integrated/custom kernel
```

It is not Minimal's attention backend, and it is not simply FA4. Custom
Attention–Mamba CUDA/Triton fusion, fused state/K/V update, and paged-K/V work
are separate future work. Motion generation, long-horizon evaluation,
benchmarks, simulation, and real-robot evaluation form the evaluation program,
not additional definitions of the Minimal baseline.

## 16. Non-Goals For First Server Handoff

Do not start with:

- [ ] Rewriting ARDY diffusion sampler.
- [ ] Rewriting the autoencoder.
- [ ] Rewriting `motion_rep`.
- [ ] Rewriting MuJoCo export.
- [ ] Training MIMO/CUTE path.
- [ ] Full HumanML3D/KITML/BABEL benchmark.
- [ ] Real robot deployment.

The clean first target is the root/body denoiser block replacement.
