# HARP-inspired IU contribution-subspace PoC

**Status:** retrospective supervised feasibility study; not a label-free method.

## Main result

The primary anchored family head (`prior=0.3`) reached **0.7778** cell-macro AUROC, a **+0.800pp** change from IU-PCR (21W/2L; worst -0.238pp). Its equal-family interval is [+0.309, +1.108]pp.

The unrestricted family-space ridge changed IU-PCR by +0.867pp. Full-feature ridge changed it by +1.388pp; this control tests whether labels alone explain the teacher's gain.

## Method boundary

For every sample, ordinary IU-PCR contributions are summed inside the frozen probability-telemetry provenance families. These family contributions reconstruct the IU score exactly. They are residualized against that score on the training partition. The supervised head learns only a small residual correction; zero correction returns the IU ranking exactly.

No new feature, generation, hidden state, model weight, attention map, or white-box quantity enters the method. Correctness labels train the proof-of-concept teacher, so the result is supervised.

## Results by label budget

| budget | method | AUROC | delta vs IU | equal-family 95% interval | W/L | worst |
|---:|---|---:|---:|---:|---:|---:|
| 20 | `family_ridge` | 0.6790 | -9.076pp | [-9.443, -7.222] | 0/23 | -19.606pp |
| 20 | `full_ridge` | 0.7229 | -4.683pp | [-6.948, -3.453] | 0/23 | -11.338pp |
| 20 | `anchored_0.1` | 0.7303 | -3.950pp | [-4.243, -2.872] | 0/23 | -8.494pp |
| 20 | `anchored_0.3` | 0.7531 | -1.669pp | [-1.790, -1.131] | 1/22 | -4.195pp |
| 20 | `anchored_1` | 0.7671 | -0.271pp | [-0.351, +0.026] | 9/14 | -1.419pp |
| 20 | `anchored_3` | 0.7704 | +0.062pp | [-0.031, +0.214] | 14/9 | -0.321pp |
| 20 | `anchored_10` | 0.7704 | +0.060pp | [+0.022, +0.114] | 18/5 | -0.104pp |
| 20 | `anchored_30` | 0.7701 | +0.029pp | [+0.008, +0.047] | 18/5 | -0.028pp |
| 40 | `family_ridge` | 0.7179 | -5.183pp | [-5.795, -3.532] | 0/23 | -13.574pp |
| 40 | `full_ridge` | 0.7414 | -2.836pp | [-4.217, -2.145] | 1/22 | -7.257pp |
| 40 | `anchored_0.1` | 0.7528 | -1.696pp | [-2.207, -0.871] | 2/21 | -4.884pp |
| 40 | `anchored_0.3` | 0.7650 | -0.480pp | [-0.770, +0.069] | 9/14 | -2.690pp |
| 40 | `anchored_1` | 0.7709 | +0.114pp | [-0.023, +0.411] | 14/9 | -0.896pp |
| 40 | `anchored_3` | 0.7712 | +0.143pp | [+0.027, +0.273] | 18/5 | -0.163pp |
| 40 | `anchored_10` | 0.7706 | +0.078pp | [+0.055, +0.116] | 20/3 | -0.046pp |
| 40 | `anchored_30` | 0.7701 | +0.033pp | [+0.020, +0.048] | 20/3 | -0.010pp |
| 80 | `family_ridge` | 0.7463 | -2.345pp | [-3.356, -1.761] | 2/21 | -7.131pp |
| 80 | `full_ridge` | 0.7508 | -1.895pp | [-3.815, -1.238] | 6/17 | -6.422pp |
| 80 | `anchored_0.1` | 0.7652 | -0.459pp | [-0.749, -0.067] | 8/15 | -3.020pp |
| 80 | `anchored_0.3` | 0.7710 | +0.118pp | [-0.101, +0.485] | 13/10 | -1.003pp |
| 80 | `anchored_1` | 0.7726 | +0.281pp | [+0.114, +0.528] | 16/7 | -0.202pp |
| 80 | `anchored_3` | 0.7716 | +0.187pp | [+0.092, +0.304] | 20/3 | -0.087pp |
| 80 | `anchored_10` | 0.7706 | +0.080pp | [+0.048, +0.118] | 21/2 | -0.033pp |
| 80 | `anchored_30` | 0.7701 | +0.032pp | [+0.016, +0.045] | 22/1 | -0.015pp |
| all | `family_ridge` | 0.7784 | +0.867pp | [-0.628, +1.562] | 16/7 | -2.239pp |
| all | `full_ridge` | 0.7837 | +1.388pp | [-0.295, +2.434] | 17/6 | -2.784pp |
| all | `anchored_0.1` | 0.7793 | +0.955pp | [+0.140, +1.355] | 20/3 | -1.013pp |
| all | `anchored_0.3` | 0.7778 | +0.800pp | [+0.309, +1.108] | 21/2 | -0.238pp |
| all | `anchored_1` | 0.7745 | +0.472pp | [+0.223, +0.652] | 21/2 | -0.050pp |
| all | `anchored_3` | 0.7719 | +0.216pp | [+0.115, +0.299] | 22/1 | -0.031pp |
| all | `anchored_10` | 0.7706 | +0.080pp | [+0.046, +0.105] | 21/2 | -0.009pp |
| all | `anchored_30` | 0.7701 | +0.031pp | [+0.021, +0.040] | 22/0 | +0.000pp |

## Interpretation

A low-dimensional supervised correction exists in IU-PCR's own contribution space and generalizes to held-out samples from the same cell. This does not establish transfer to unseen dataset families. The next premise test must ask whether unlabeled, cell-local statistics predict the supervised correction under leave-one-family-out evaluation. A fixed cross-family correction and another graph regularizer are not justified by the existing evidence.

The 20/40/80 budgets use a label-aware prevalence-preserving acquisition diagnostic. They are optimistic and may not be described as deployable active or semi-supervised results.

Across the primary cross-fitted teachers, the median per-cell cosine to that cell's mean correction was 0.967. This is a within-cell stability diagnostic, not evidence that one correction transfers across cells.

## Exclusions and audit

Cells below 20 positive examples were excluded from the PoC: `spilled_triviaqa_llama8b`.
All 23 evaluated cells reconstructed their IU score from family contributions; maximum reconstruction error was 8.882e-16.

Primary artifacts: `replicates.csv`, `cell_means.csv`, `summary.csv`, `teacher_targets.csv`, and `RUN_DEFINITION.json`.
