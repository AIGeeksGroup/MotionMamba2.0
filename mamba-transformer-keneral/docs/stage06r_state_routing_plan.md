# Stage 06R Plan — State-Guided Routing over Attention KV

Date: 2026-08-14
Status: implemented; stopped at failed Gate C

Stage 06R is a gated rescue experiment. It does not scale the failed Stage-06
state-query or state-KV coupling variants. Stage-05 Minimal remains frozen and
provides the token/content data plane: attention sinks, accepted clean
sliding-window K/V, and current-chunk K/V. Mamba is considered only as a
possible control signal for choosing among already committed historical
chunks.

## Frozen assets

- Minimal checkpoint:
  `/workspace/ardy_stage05/outputs/minimal_real_v1/sequence_minimal.pt`
- Sequential eval cache:
  `/workspace/ardy_stage05/cache/minimal_seq/manifest_eval.json`
- ARDY teacher revision: `abe6c43beb28c867c950acb824b9c4ef3d63fb76`
- Cache replay verification: maximum and mean absolute error `0.0`
- Existing Mamba3 implementation, DiT, diffusion lifecycle, sinks, K/V cache,
  and accepted-clean commit semantics are unchanged.

The eval cache contains 16 real trajectories, four accepted chunks per
trajectory, and ten diffusion calls per chunk. Gate A evaluates the first,
middle, and final diffusion calls for each current chunk `t=1..3`, yielding
144 sequential cases.

## Gate A — oracle history selection

For each current case, only accepted clean chunks before `t` are eligible.
Attention sinks and current K/V are always present and are never routing
candidates. The frozen Minimal model is evaluated with:

1. no historical chunk;
2. most-recent chunk only;
3. oldest available chunk only;
4. the full available sliding window;
5. every singleton chunk, with Oracle Top-1 defined by minimum root+body MSE;
6. every two-chunk subset where at least two chunks exist, with Oracle Top-2
   defined by minimum root+body MSE.

The evaluator records total/root/body MSE, per-case gains, fractions beating
Full and Recent, selected chunk IDs, selected temporal distances, runtime, and
peak allocated GPU memory. Oracle labels are derived only from prediction loss;
no Mamba state or future chunk is used.

Gate A passes only if Oracle Top-k improves over Full by a reproducible margin
and/or selects non-recent history often enough to establish meaningful
selection value. If Oracle is effectively Full or almost always Recent, this
direction stops without training a router.

## Gate B — frozen supervised router (conditional on Gate A)

If Gate A passes, freeze Minimal, Mamba, and attention projections. Build
labels from the oracle and train only small router projections. Compare:

- Recent heuristic;
- token-only query versus pooled attention-side chunk summaries;
- committed `s_(t-1)` query versus the same chunk summaries;
- committed `[s_(t-1), delta s_(t-1)]` as an explicit ablation.

Candidate summaries will initially be mean-pooled existing K or token
representations. No second memory bank is introduced. Required metrics are
Hit@1, Hit@2, MRR, and distance-stratified accuracy. A 16–64-example overfit
must collapse classification loss and reach high Hit@1 before held-out
evaluation. State must beat Recent and token-only to justify Mamba routing.

## Gate C — frozen routed denoising (conditional on Gate B)

If Gate B passes, plug the frozen router into frozen Minimal and compare Full
W, Recent, Oracle Top-1/Top-2, token-only Top-k, state Top-k, and state+delta
Top-k on identical cached calls. Hard chunk-level Top-k is tested first.
Quality evidence is lower denoising MSE or preserved quality with materially
fewer attended K/V tokens. Record K/V token counts, analytical attention-FLOP
ratio, latency, and peak memory. No generator training or fused kernel work is
authorized in this stage.

## Interpretation boundary

The current experiment tests Mamba state as a history router. It does not
equate the mentor phrase “state and token delta” with any one operator. Three
distinct later meanings remain documented but unimplemented:

- temporal state delta: `s_t - s_(t-1)`;
- temporal token delta: `z_t - z_(t-1)`;
- token-state discrepancy: `phi(z_t) - psi(s_t)`.

Only the first is an optional Gate-B ablation. The other two are out of scope.

## Stop rule

Failure of Gate A stops before router implementation. Failure of Gate B stops
before routed attention. Failure of Gate C stops before any MotionMamba
training. A final C1 operator and Contribution-2 contract are created only in
a later stage after all three gates pass.

## Outcome

Gate A passed narrowly for Oracle Top-2, and Gate B passed only for nontrivial
Top-2 recall. The frozen learned state Top-2 policy then failed Gate C: it used
20 instead of 30 historical K/V tokens at `t=3`, but MSE was `0.1104880`
versus `0.1101033` for Full. No denoising improvement or demonstrated net
runtime/memory advantage remained after accounting for the required Mamba
control path. Execution therefore stopped before generator training or a C1
operator freeze. Exact evidence is in `stage06r_state_routing_closeout.md`.
