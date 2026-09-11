# Stage 09 motion-structured C1 plan

## Frozen provenance

- Branch: `codex/stage09-motion-structured-c1`
- Base commit: `57b267ddc92a7abcaf85beb4b01d0ba576b3c17c`
- Minimal SHA-256: `acbdcc78c6594ad690a1d3a76b129665198fd142e487a1eb64bf43163a1b949f`
- Mamba SHA-256: `cdcc3a658f221843048889a07d20329718fea963d5bf0eacddd016538fd198c1`
- Teacher: `Ardy-Core-RP-20FPS-Horizon40`, revision
  `abe6c43beb28c867c950acb824b9c4ef3d63fb76`
- TRAIN/DEV/FINAL-heldout hashes:
  `10f9dc0010d08bcf1558612fb7b0d81cc28cc326700dca5c65f65fc9cdbf8067`,
  `94588595c106df0b1fb6b1c641e8d6d855db5c748bd1c7da894fccbe1e944fa6`,
  `ac189a9f1b0aae151ad873b55101d1e5e65b74e62b1a8201db6b283d7859f88b`.
  FINAL-heldout remains unread.
- Environment: NVIDIA A800-SXM4-80GB, PyTorch 2.4.1+cu121, CUDA 12.1,
  Triton 3.5.0 overlay.

The pre-existing `src/dit_sequence_kernel/stack.py` docstring modification is
preserved separately as stash `pre-stage09 user docstring change`; it is not
part of this branch's experiments.

## Existing-component map

| Existing component | Stage-09 role |
|---|---|
| frozen Stage-05 `ArdyDiTSequenceBlock` | Attention root/body branches |
| frozen Stage-07 `ArdyMambaBlock` | Mamba root/body branches |
| ARDY `AutoLatentTwostageDenoiser.forward` | authoritative root-to-local-root-to-body coupling |
| cached `outer_inputs` and full teacher output | coupled replay and target |
| cached root/body block inputs and outputs | isolated-stage diagnostic |
| Stage-08 G0 `MLP([z,state])` correction | unchanged generic fusion family for H0/S2 |
| Stage-08 representation capture | final/middle branch features without prediction changes |
| Minimal explicit history commit | teacher-forced sliding-KV history for active Attention branches |

No DiT, Minimal, Mamba, cache, or ARDY diffusion-loop redesign is planned.
Stage-specific selection is a thin composition/wrapper layer.

## Canonical evaluator

The evaluator uses the frozen DEVELOPMENT manifest, diffusion indices
`{0, 5, 9}`, all four chunk indices, bf16 autocast, SDPA for Minimal, and the
validated Mamba Triton environment. Every candidate receives the identical
cached `outer_inputs`.

Coupled evaluation invokes the complete ARDY raw denoiser. Therefore the
selected root prediction is converted by ARDY's own
`global_root_to_local_root` and used to construct the body input. This is the
primary metric. Isolated evaluation invokes root/body blocks on the cached
teacher-provided block inputs and is diagnostic only.

Metrics are fixed before candidate selection:

- root MSE: per-element squared error of the root block output;
- body MSE: per-element squared error of the body block output;
- total MSE: coupled root MSE plus coupled body MSE, matching the repository's
  explicit cached-training objective.

Complete coupled-output per-element MSE is retained as a diagnostic because
it includes mask/history positions and is not the repository training loss.
The first evaluator smoke incorrectly promoted that diagnostic to primary;
it was corrected before any gate decision. Results include
paired bootstrap 95% confidence intervals, call win rates, diffusion/chunk
breakdowns, latency, peak allocated memory, and active/loaded parameters.

## Execution gates

1. Reproduce B0, B1, Minimal+Mamba, Minimal+Minimal, and canonical H0.
2. If Gate 0 passes, evaluate the frozen B0/B1/S1/D0 cross matrix in isolated
   and coupled modes.
3. Implement S2 only if Gate A confirms directional root/body asymmetry.
4. Train H0/S2 frozen-parent G0 modules with seeds 211/223/227 and equal
   budgets. S2 starts from S1 through zero correction.
5. Run generation and FINAL-heldout only after the corresponding development
   gates pass and a candidate is frozen.

## Known minimal integration gap

Stage-08 G0 was validated as an offline frozen-representation bridge and did
not save an ARDY-coupled model checkpoint. Stage 09 must re-fit the exact G0
family under canonical coupled conditioning and save its normalization and
weights. A thin dual-branch root/body wrapper is sufficient: it executes both
parents on the same stage input, captures the validated features, and applies
the G0 correction. S1/D0 bypass this wrapper and execute only their selected
branch. This is integration work, not a new fusion operator.
