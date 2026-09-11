# Stage 08 C1 closeout

## Decision

Attention-Mamba complementarity is real and architecture-specific, but the
tested explicit token-state innovation parameterization is not the best
compression. A generic late fusion wins consistently. Therefore Stage 08
does not close a delta-specific MotionMamba Contribution 1.

## Evidence summary

1. Minimal/Mamba fixed averaging reproduced a 29.12% DEVELOPMENT gain.
2. Minimal-A/Minimal-B averaging gained only 0.31%; ordinary seed ensembling
   does not explain the result.
3. Root is strongly Mamba-favored; body retains substantial same-target
   complementarity. Gains vary sharply by chunk and reverse at chunk 3.
4. The output bridge reproduced fixed averaging exactly and overfit the fixed
   case by 86.92%, ruling out general fusion-plumbing failure.
5. State prediction selected actual recurrent state for root (R² 0.512) but
   Mamba hidden for body (R² 0.740). The paper-level recurrent-state claim is
   therefore not uniformly supported.
6. Matched P0-P5 probes found only a root-local delta signal. Body and
   aggregate Gate B did not pass.
7. Frozen late bridges were trainable. Three-seed mean gains were G0 11.52%,
   I1 5.02%, and I2 7.09%. G0 beat both innovation variants on every seed,
   failing Gate D.

## Stopping boundary

I3 was not reached because I1/I2 failed delta-specific Gate D. No joint
fine-tuning, layer expansion, long-horizon generation, final ablation,
FINAL-heldout evaluation, or `docs/c1_operator_contract.md` was produced.
This is required by the Stage-08 hard-stop rules, not missing execution.
FINAL-heldout remains unread.

The correct scientific conclusion is: current evidence supports generic
hybrid complementarity, not the claim that Attention should specifically
model `z - R(s)`.

## Artifacts

- A0: `/workspace/ardy_stage08/complementarity_train_dev_v3/complementarity.json`
- A1: `/workspace/ardy_stage08/same_architecture_control/same_architecture.json`
- output bridge: `/workspace/ardy_stage08/output_bridge/output_bridge.json`
- representations: `/workspace/ardy_stage08/representations_mid_final_v2/representations.pt`
- probes: `/workspace/ardy_stage08/token_state_probe_v2/probe_results.json`
- late bridges: `/workspace/ardy_stage08/late_bridges_v3/` and
  `/workspace/ardy_stage08/late_bridges_seed{223,227}/`

## Regression status

The focused local suite reported 27 passed and 5 skipped; one real-Mamba CUDA
test could not compile because the local Triton lacks
`make_tensor_descriptor`, an environment mismatch unrelated to Stage-08
changes. Earlier focused instrumentation tests were 20 passed, 5 skipped.
The A800 extraction and all real-Mamba diagnostics executed successfully with
the validated Triton overlay. Model prediction paths are unchanged unless
diagnostic capture is explicitly enabled.

Final status: `C1_DELTA_NOT_SUPPORTED`.
