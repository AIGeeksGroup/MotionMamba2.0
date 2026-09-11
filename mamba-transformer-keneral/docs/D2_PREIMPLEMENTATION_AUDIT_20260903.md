# D2 Preimplementation Audit

Date: 2026-09-03  
Audited baseline: `f4cea2ab8652edb7c5d1e3dc0a269434883d99e2`  
Scope: D2 trainability and Native-preserving migration only  
Status: PREIMPLEMENTATION; no D2 mechanism or real-motion claim

## Evidence boundary

This audit was performed before changing the D implementation. It covers the
current content-memory module, `SequenceKernel`, ARDY integration and accepted
history lifecycle, and the tiny/DDP training paths. Historical Stage C and D
results are retained as historical results. In particular, the deterministic
causal-memory program remains an intervention-bookkeeping test, not evidence
that a trained Mamba state carries useful long-term motion content.

The locked-final BONES actors have not been read. The healthy parent named by
`docs/B_BASELINE_MANIFEST.json` exists and its SHA-256 was independently
verified as:

```text
cfa9d44d9a804e3f71dbfde98cbad7faceed98bca83b5417c3afb5bca6071a4a
```

## Risk 1: default initialization can dead-start the content branch

Static finding: **risk is present; gradient magnitude requires the registered
W1 -> W2 -> W3 regression before it is called experimentally confirmed.**

`PersistentContentMemory.__init__()` zero-initializes both decoder tensors and
the write-gate weight, sets the write-gate bias to `-12`, and initializes the
read gate to `-12` (`src/state_memory_operator/content_memory.py:19-32`). Thus
the initial proposal is exactly zero, the initial slots are exactly zero, and
both read and write sigmoids are approximately `6.14e-6`.

`SequenceKernel.forward_chunk()` reads those slots and then uses
`bool(torch.count_nonzero(candidate_memory_content))` to decide whether an
extra Attention content path exists (`src/sequence_kernel/operator.py:327-340`).
If slots remain exactly zero, task output has no dependency on the augmented
Attention branch. This hard, data-dependent Python branch is therefore a
possible closed gradient cycle:

```text
zero slots -> no extra-content Attention -> no task dependency on memory
           -> no decoder/write/read task gradient -> zero slots
```

Required measurement before implementation: accepted W1, accepted W2, loss on
W3 under `detach=True` and `detach=False`, with per-parameter gradient norms.

## Risk 2: the write gate is proposal-aware, not previous-memory-aware

Static finding: **confirmed.**

`propose()` constructs `context = [s_t, delta_t]`, decodes `proposal`, and
feeds `[context, flatten(proposal)]` to `write_gate`
(`src/state_memory_operator/content_memory.py:37-43`). The previous memory is
used only in the convex update. It is not an input to the gate. Therefore the
implemented gate is not the registered
`f_g([s_t, delta_t, flatten(M_(t-1))])` mechanism.

D2 must add an explicit `previous_memory`/`proposal` configuration. The D2 main
variant will use `previous_memory`; `proposal` will remain only as an ablation.

## Risk 3: detached cache training cannot train cross-window writes

Static finding: **confirmed by autograd ownership; the detach control still
requires the registered numeric regression.**

`SequenceKernel.commit()` detaches Mamba state, projected summary/delta, and
persistent memory whenever `detach=True`
(`src/sequence_kernel/operator.py:526-544`). `canonical_history_prime()` uses
`torch.no_grad()` for that same mode and only retains a graph for
`detach=False` (`src/ardy_adapter/history_commit.py:100-154`). A later loss can
train a read-side parameter from a detached slot value, but cannot propagate
through that value to the earlier `content_decoder`, `write_gate`,
`StateProjector`, or Mamba transition.

The existing generated-history path already provides the correct bounded D2
primitive. `_prepare_generated_history()` samples accepted history under
`no_grad`, while `_prime_generated_history_bptt()` restores a prefix and
replays the newest accepted chunks through
`canonical_history_prime(..., detach=False)`
(`examples/train_humanml3d_ardy_tiny.py:779-837`). `_loss()` invokes that replay
before the future supervised target (`examples/train_humanml3d_ardy_tiny.py:1170-1190`).
D2 should reuse this path rather than inventing a second lifecycle.

One limitation is explicit: `_prime_generated_history_bptt()` always commits
the seed with `detach=True`. D2 training must retain at least one generated
accepted write and one future supervised target; it must not claim that a
detached teacher-only prime trained the write mechanism.

## Risk 4: state-aware output drops the trained Native Hybrid readout

Static finding: **confirmed.**

Native Hybrid constructs `fusion_gate_logit` only for
`interaction=naive_gated_add` (`src/sequence_kernel/operator.py:106-113`) and
returns `Attention + sigmoid(fusion_gate_logit) * Mamba`
(`src/sequence_kernel/operator.py:426-450`). In contrast, the state-query and
state-KV branches unconditionally return only `attention_output`. They do not
register or use the Native fusion gate.

Consequently, replacing the healthy Native Hybrid 50k blocks with the current
state-aware D implementation removes a trained direct Mamba contribution at
step 0. This is a structural change, not a neutral extension, and prevents the
requested strict migration comparison.

D2 must add an opt-in `state_aware_output_mode=native_gated_add` while retaining
`attention_only` as the historical C default. The new mode must register the
same `fusion_gate_logit` key and compute:

```text
state-aware Attention + sigmoid(native fusion gate) * Mamba output
```

With state-conditioning scale and content read residual scale both zero, a
same-weight D2 model must match Native Hybrid output, pending Attention K/V,
and Mamba transition within the registered dtype tolerance.

## Training-path gaps

The DDP trainer currently supports `mixer`, `denoiser`, and `recurrent`
(`examples/train_humanml3d_ardy_ddp.py:51-55`). `recurrent` trains Mamba and the
native fusion gate, while `mixer`/`denoiser` are much broader
(`examples/train_humanml3d_ardy_ddp.py:798-820`). No current scope freezes the
healthy base while training exactly `state_projector`,
`token_state_interaction`, and `content_memory`.

The trainer logs only an aggregate clipped gradient norm. It does not record
the D2 per-group gradient counts or the required memory/read/write diagnostics.
These omissions must be fixed before a BONES 100-step gate is auditable.

## Required implementation order

1. Measure and preserve the old default-init W1/W2/W3 gradient result.
2. Add Native-preserving state-aware output without changing historical C defaults.
3. Replace the content read with a zero-initialized signed augmented-Attention residual.
4. Use small nonzero reproducible decoder initialization and an open trainable write gate.
5. Remove tensor-value-dependent branch creation and make the gate previous-memory-aware.
6. Add `state_memory` train scope and complete gradient/memory diagnostics.
7. Pass unit, contract, full-suite, compile, diff, and real CUDA gradient tests.
8. Run the learned actual-`SequenceKernel` delayed-cue gate.

No BONES D2 training is authorized until steps 1-8 pass. No Stage E-I work is
authorized by this audit.
