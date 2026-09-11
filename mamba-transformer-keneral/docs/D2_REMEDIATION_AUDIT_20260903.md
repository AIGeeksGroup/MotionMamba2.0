# D2 Remediation Audit

Date: 2026-09-03
Parent for any new run: healthy Native Hybrid 50k, never D2 step 250

## Why D2 did not establish the intended mechanism

1. Training used three windows and `generated_history_bptt_windows=1`. The
   supervised target was W3, before a W4 cache can evict W1. The seed W1 write
   was detached. The optimizer therefore never learned from a post-eviction
   target how W1 should be written.
2. At step 250, the aggregate projected-state norm was about 12.01 while the
   delta norm was only 0.99. Decoder weight norms do not compensate for this
   roughly 12x input-scale gap, especially in root layers. Absolute state can
   consequently dominate `[s, delta]`.
3. The decoder receives only `[s, delta]`. Text affects the Mamba state
   indirectly through DiT modulation, but no explicit nonlinear
   `text x delta` interaction exists. A linear decoder over concatenated state
   features cannot reliably assign different meanings to the same transition
   under different prompts.
4. The final write-gate mean is about 0.713. Old content is therefore retained
   by only about `0.287^4 = 0.0068` after four later writes. This is inconsistent
   with durable W1 content and explains why a one-time post-W4 memory
   permutation has little lasting effect.
5. Memory/proposal norms grew to 35.92/38.13 and augmented-minus-base Attention
   reached 121.75. The learned read scale stayed small, so D2 can act as a
   generic signed correction while remaining weakly sensitive to memory row
   identity.
6. The 40-second selection panel contains only 16 motions and seven actors,
   with several adjacent same-action/mirrored pairs. Its permutation and
   bootstrap tests have limited semantic contrast and wide confidence
   intervals. This is an evaluation-power limitation, not evidence of success.

## Registered corrective experiment

- Start again from the locked Native Hybrid checkpoint.
- Preserve exact step-0 output with zero content read residual scale.
- Condition content writes explicitly on pooled text.
- Interpret normalized delta through a text-conditioned FiLM transform before
  decoding content; keep the exact raw delta definition unchanged.
- Normalize proposals/slots and cap per-window write rate at 0.25 so early
  content cannot be erased in a few windows.
- Use a W6 target with generated W2-W5 replay. Attach the W1 seed write and
  retain gradients through the four later accepted writes. This is finite
  accepted-history replay, not Stage E end-to-end differentiable sampling.
- Use a lower learning rate and keep the Native Attention, Mamba, DiT/MLP,
  frontend, and fusion gate frozen.
- Run preflight and short safety gates before any 100-step experiment. Never
  overwrite or resume the failed D2 artifacts.

This design tests the user's semantic hypothesis directly: delta remains
`P(S_t)-P(S_(t-1))`, while its decoded meaning is conditioned on text rather
than assuming the same transition has one context-free meaning.
