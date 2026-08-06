# Bootstrap-stabilized SDSF — real-artifact replay

Decision: **DOES_NOT_MEET_REAL_CONTRIBUTION_GATE**.

Feature schema: `confidence-orientation-v1`; candidate: bootstrap rho shrinkage with tau=0.5, 10 row bootstraps, leading two coordinates preserved.

This is a retrospective replay on the same 24-cell artifact used to diagnose SDSF. Labels were read only after each method's score was frozen.

## Matched contrasts

| reference -> candidate | mean [cell-bootstrap 95% CI] | QA / math | family macro | W/L | worst | <=-5pp |
|---|---:|---:|---:|---:|---:|---:|
| `su_pcr` -> `stable_sdsf` | -2.91 [-4.10, -1.84] | -2.68 / -3.05 | -1.70 | 2/22 | -11.14 | 3 |
| `current_sdsf` -> `stable_sdsf` | +1.80 [+0.99, +3.16] | +2.45 / +1.41 | +1.31 | 23/1 | -0.03 | 0 |

## Frozen gates

| gate | observed | rule | result |
|---|---:|---:|:---:|
| `mean_vs_su_min_pp` | -2.907 | >= +1.000 | **FAIL** |
| `cell_ci_low_vs_su_min_pp` | -4.104 | >= +0.000 | **FAIL** |
| `qa_vs_su_min_pp` | -2.676 | >= -0.500 | **FAIL** |
| `math_vs_su_min_pp` | -3.046 | >= -0.500 | **FAIL** |
| `family_macro_vs_su_min_pp` | -1.696 | >= +0.000 | **FAIL** |
| `mean_vs_current_sdsf_min_pp` | +1.803 | >= +0.000 | **PASS** |
| `cell_ci_low_vs_current_sdsf_min_pp` | +0.988 | >= +0.000 | **PASS** |

## Interpretation

The candidate fails at least one predeclared real-data contribution condition. It does provide strong evidence that bootstrap reliability shrinkage improves the current SDSF implementation, but stable SDSF remains materially below SU-PCR and therefore cannot replace the leading method. The failed gates identify the next hypothesis; they must not be repaired by tuning tau on these labels.
