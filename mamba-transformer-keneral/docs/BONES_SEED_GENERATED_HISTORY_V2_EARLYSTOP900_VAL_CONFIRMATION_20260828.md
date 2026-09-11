# BONES-SEED Step-900 Validation Confirmation

This confirmation uses the actor-disjoint validation split, seed 20260829, all 16 clips with at least 1000 frames for 20/40-second evaluation, and 64 samples for Table 4/9-second retention. It was not used by the checkpoint sweep, but the checkpoints' informational dev loss was previously computed on this split.

**FAIL**

| Criterion | Passed |
|---|---|
| short retention | True |
| hybrid 20 40 means and slopes better | False |
| state finite | True |
| no state norm explosion | True |

| Duration | Constraint mean gap | Root ADE gap | Root slope gap | Constraint slope gap | Paired Root-slope 95% CI |
|---:|---:|---:|---:|---:|---:|
| 20 s | -9.51 | -7.67 | 4.019 | 3.773 | [2.417, 5.373] |
| 40 s | -13.87 | -13.83 | 3.146 | 3.669 | [1.404, 4.606] |

Negative gaps favor Hybrid.
