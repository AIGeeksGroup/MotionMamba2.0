# Stage 06 Contribution-1 Plan — Token–State Delta Modeling

Date: 2026-08-14
Status: **IMPLEMENTED; STOPPED AT FAILED TINY-OVERFIT GATE**

This stage does not reopen Original, DiT, or Minimal. Minimal remains the
frozen standard attention-only baseline. The existing gated-add Hybrid remains
a control. MotionMamba 2.0 Contribution 1 is accepted only if a precisely
defined token/state interaction beats that naive control with real-motion and
long-horizon evidence.

## 1. Existing components and C1 roles

```text
MinimalAttentionKernel / token Q,K,V
    -> standard token-level attention and content interaction

ChunkSlidingWindowKVCache
    -> the one explicit token/content memory; retain chunk boundaries

attention sinks
    -> standard persistent token-memory anchors; unchanged

SequenceKernel minimal mode
    -> frozen Minimal control A

SequenceKernel hybrid gated-add mode
    -> naive Attention + Mamba control B; not Contribution 1

Mamba3ChunkWrapper / Mamba3State
    -> compressed recurrent dynamics and accepted state handoff

StateProjector
    -> project raw SSM state S_t to compact summary s_t

StateTransition / delta
    -> carry delta s_t = s_t - s_{t-1} to the next chunk

StateQKVEncoder
    -> validated prototype that state/transition can produce addresses

ContentValueEncoder
    -> mechanism evidence that payload must stay token/content-derived

EventMemory
    -> separate-memory ablation and historical mechanism evidence only

ARDY diffusion controller
    -> noisy reads, exactly one accepted clean root/body commit, rollback

Stage-05 cache/train/generation bridge
    -> identical real-motion data, initialization, metrics, and horizons
```

Historical evidence is retained: learned state-based retrieval reached about
99.8% Hit@1 in the validated synthetic configuration, while fine-grained
state-payload probes remained below 1% at realistic class counts. Therefore C1
uses Mamba state for addressing/dynamics and existing token V for content.

## 2. Temporal semantics and notation

At layer `l`, before chunk `t`, durable state contains only accepted history:

```text
Z_t^l                  current token representation
S_{t-1}^l              committed raw Mamba state
s_{t-1}^l=P_l(S_{t-1}^l)
d_{t-1}^l=s_{t-1}^l-s_{t-2}^l
M_{t-1}^l={sink K/V, accepted K/V chunks t-W...t-1}
```

Retrieval for chunk `t` may use only `(s_{t-1}, d_{t-1}, M_{t-1}, Z_t)`.
Mamba may process `Z_t` in the same call to produce candidate `S_t`, but
`S_t`, `s_t`, and `d_t` cannot affect chunk `t` output. They become readable
only after the clean chunk is accepted and committed. Noisy diffusion calls
discard candidates and leave all durable state unchanged.

## 3. Bounded variants and controls

### A — frozen Minimal

Standard attention over sinks + sliding K/V + current tokens. No Mamba.
Weights and behavior remain identical to Stage 05.

### B — naive Hybrid control

The existing scaffold:

```text
h = h_attn + sigmoid(gamma) h_mamba
```

This answers whether adding an ordinary Mamba feature branch is sufficient.
It is not presented as C1.

### C — state-conditioned token query

Using only committed prior state:

```text
c_{t-1} = [s_{t-1}; d_{t-1}]
q_token = W_q Z_t
q_state = W_s c_{t-1}
q = split_heads(q_token + tanh(beta) q_state)
```

The state-only ablation sets `d_{t-1}=0`; the delta version uses both terms.
`beta=0` at initialization, so C(step 0) equals Minimal. Keys and values remain
token-derived. Mamba output is not injected as a content payload; it is used
only to form the candidate next recurrent state.

### D — state-conditioned historical-key modulation

For each existing cached token chunk `j`:

```text
u = normalize(heads(W_s [s_{t-1}; d_{t-1}]))
kbar_j = normalize(mean_tokens(K_j))
r_j = mean_heads(<u, kbar_j>)
a_j = 1 + tanh(beta) tanh(r_j)
K'_j = a_j K_j
V'_j = V_j
```

Standard token attention then uses the modulated historical keys and unchanged
token values. Modulation is differentiable, preserves chunk boundaries, uses
no hard pruning, and equals Minimal when `beta=0`. The same state-only versus
state+delta ablation is available.

No additional architecture variant will be added before these controls are
evaluated. A separate EventMemory path is an ablation, not a preferred C1
candidate.

## 4. State, commit, and initialization contract

- Each DiT layer carries attention cache, raw Mamba state, projected summary,
  and previous delta as distinct fields.
- A forward returns candidate K/V, raw state, summary, and delta without
  mutating durable state.
- One accepted commit advances all candidates atomically at the existing ARDY
  root/body boundary; reset clears per-row K/V and all recurrent summaries.
- C1 loads every compatible parameter from
  `/workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt`.
- Newly introduced Mamba/state modules are explicit missing keys. Interaction
  coefficients initialize to zero, making the C1 output exactly/within numeric
  tolerance of Minimal before training.
- During training, accepted teacher-forced history is committed with
  `detach=False` inside one trajectory so gradients can reach Mamba/state
  parameters. All cache/state is discarded after `optimizer.step()`.

## 5. Required gates

1. CPU injected-wrapper tests: finite forward/backward, state/KV carry, full
   and per-row reset, no future-state leakage, zero-init Minimal equivalence.
2. Real Mamba CUDA: wrapper/state projector/C1 gradients and unchanged naive
   Hybrid smoke.
3. Historical state-addressing regression where practical; no state-derived
   value payload is introduced.
4. Tiny real-cache overfit for B/C/D: falling finite loss, gradients to new
   state/Mamba and intended attention parameters, coefficient moves, save/load.
5. Equal-budget pilots from the same Minimal checkpoint/data/seed/update count:
   Minimal, naive Hybrid, query state-only, query state+delta, modulation
   state+delta, and separate EventMemory ablation.
6. Candidate selection must beat/preserve short-horizon quality, beat naive
   Hybrid, demonstrate delta value if claimed, and degrade more slowly across
   1/2/4/8/16 chunks. Top candidate receives a small seed replication.
7. Only then train a real C1 checkpoint, freeze `c1_operator_contract.md`, and
   hand Contribution 2 a stable API. No fused-kernel work occurs in Stage 06.

If evidence does not distinguish the proposed interaction from naive Hybrid,
or delta from state-only, Stage 06 closes as `CONTRIBUTION1_NOT_CLOSED`.

Execution results and the stop decision are recorded in
`docs/stage06_contribution1_closeout.md`. No equal-budget varied-data pilot,
operator selection, horizon study, or final C1 checkpoint is authorized until
the failed tiny-overfit gate is diagnosed.
