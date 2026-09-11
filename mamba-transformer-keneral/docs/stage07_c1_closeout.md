# Stage 07 — Contribution-1 bounded-search closeout

## Decision

Stage 07 established a meaningful real-motion Mamba control and strong static
Minimal/Mamba output complementarity. It did **not** establish a reproducible
token-state interaction that passes the mandatory tiny-overfit gate. No final
C1 architecture is frozen, H3 is unjustified, and no Contribution-2 operator
contract is issued.

## Real Mamba

The standalone Mamba checkpoint is
`/workspace/ardy_stage07/mamba_real_v2_lr2e5/sequence_mamba_c1_base.pt`.
Its 32-call tiny gate reduced mean loss by 82.4%. The 1500-update checkpoint
reduced development MSE from 1.57488466 to 0.34594337, maintained finite and
nontrivial recurrent state/carry statistics, reloaded correctly, and generated
finite non-collapsed 1/2/4-chunk motion. It is therefore a valid scientific
control rather than a random-state proxy.

## Complementarity

On the development-only calibration/evaluation partition, Minimal/Mamba MSE
was 0.24398084/0.30456382. A plain average improved Minimal by 19.77%, affine
fusion by 19.44%, and a small diagnostic gate by 33.65%. Error correlation was
0.4291 and each model won a substantial fraction of examples. This supports
model-level complementarity, but not the proposed internal delta mechanism.

## Bounded candidates

- **H0 parallel control:** exact Minimal initialization and finite gradients;
  repeated tiny runs plateaued near 0.66735 from 0.69605 and did not improve
  development loss. Rejected at Gate 2.
- **H1 token-state innovation:** exact Minimal initialization and nonzero
  branch gradients/scales, but repeated runs reduced tiny loss only about 4%.
  The equation-correct final refinement (`g*W(delta)`) ended at 0.66823018
  from 0.69604731 and worsened dev MSE to 0.79526359. Rejected at Gate 2.
- **H2 delta-rule correction:** isolated out-of-place state adapter passed
  lifecycle/shape/dtype/gradient safety. Tiny loss fell only 4.12%, while dev
  MSE worsened by 26.5%. Rejected at Gate 2.
- **H3:** not reached because H1 and H2 were not individually positive.

No candidate reached equal-budget development pilots, short candidate
generation, multi-seed selection, final-heldout evaluation, long-horizon
comparison, or final ablations. Those omissions are deliberate consequences
of the mandatory cheap-gate policy, not positive evidence.

## Representation evidence

Static prediction fusion shows clear complementary errors. It does not show
that `LN(attention)-LN(mamba)` or the H2 matrix-state innovation is a useful
correction feature. Because all delta candidates failed before a meaningful
trained representation existed, P0--P4 and per-layer CKA would not support
candidate selection and are reported as **not reached**, not silently treated
as successful mechanistic evidence.

## Artifacts

- Frozen splits: `/workspace/ardy_stage07/splits/`
- Mamba tiny/full: `/workspace/ardy_stage07/mamba_tiny_v1/` and
  `/workspace/ardy_stage07/mamba_real_v2_lr2e5/`
- Mamba generation: `/workspace/ardy_stage07/mamba_generation/`
- Complementarity: `/workspace/ardy_stage07/complementarity/`
- Failed H0/H1/H2 attempts: `/workspace/ardy_stage07/h0_parallel_*`,
  `/workspace/ardy_stage07/h1_innovation_*`, and
  `/workspace/ardy_stage07/h2_delta_rule_*`

The final-heldout manifest was created and hashed but never evaluated. There
is no selected architecture, so opening it would violate the frozen protocol.

## Remaining limitation

The central question remains unresolved: output predictions are complementary,
yet none of the bounded internal interactions could exploit that difference
under the required learning gate. The next action is a precise mathematical
discussion with the mentor, not another autonomous architecture family.

C1_TOKEN_STATE_STOP
