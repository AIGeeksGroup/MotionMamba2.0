# Stage 09 motion-structured closeout

## Outcome

The canonical evaluator confirms strong output-level Attention-Mamba
diversity, but the two required prerequisites for the final structured C1 do
not hold together:

1. corrected three-seed H0 improves B0 by only 2.69%, below the mandatory 5%
   Gate-0 threshold in all three seeds;
2. aggregate S1 beats D0, but only three of eight trajectories favor S1, so
   the directionality is concentrated and fails Gate A's robustness clause.

Stage 09 therefore stops before a selectable S2, generation, or held-out
freeze. The final autonomous C1 search is closed without a contribution.

## Required report

1. Base commit: `57b267ddc92a7abcaf85beb4b01d0ba576b3c17c`; clean Stage-09 branch, user
   docstring edit preserved separately.
2. Canonical total: coupled root MSE + coupled body MSE, matching training;
   bf16, SDPA Minimal, diffusion indices 0/5/9, all four chunks.
3. Minimal+Mamba output average: 31.54% gain.
4. Minimal+Minimal average: 0.62% gain.
5. H0: `0.410792 ± 0.000333`, 2.69% gain; Gate 0 fail.
6. Isolated root: Attention 0.156476, Mamba 0.066048.
7. Isolated body: Attention 0.260077, Mamba 0.279898.
8. Coupled B0/B1/S1/D0: 0.422135/0.417296/0.413828/0.439944.
9. Gate A: aggregate direction positive, trajectory robustness fail (3/8 wins).
10. S2 wrapper was implemented as thin composition; parents unchanged.
11. S2 body tiny overfit: 55.97%, finite gradients, save/reload pass.
12. No valid S2 three-seed selection result: corrected jobs stopped at Gate 0.
13. Canonical root/body/total metrics are recorded in the search log.
14. Efficiency diagnostics: B0 0.0538 s/call and 247.9M active parameters;
    S1 0.0334 s/call and 182.7M active parameters. These do not override
    failed statistical gates. H0 corrected runtime was about 0.069 s/call.
15. Conditioned gate not reached.
16. Generation not reached.
17. FINAL-heldout remained unread.
18. No C1 architecture selected.
19. Required directionality ablation is the B0/B1/S1/D0 matrix; it fails
    trajectory robustness. Deeper ablations were not opened.
20. No `c1_operator_contract.md`, because C1 did not close.
21. Git commit is recorded after final verification.
22. Exact stop: H0 failed the 5% Gate-0 requirement in 0/3 seeds; Gate A was
    additionally concentrated in two high-gain trajectories.

## Preserved boundaries

No delta, routing, EventMemory, new memory mechanism, custom kernel, FA4,
simulation, robotics, or FINAL-heldout work was performed.

Final status: `C1_COMPLEMENTARITY_NOT_REPRODUCED`.
