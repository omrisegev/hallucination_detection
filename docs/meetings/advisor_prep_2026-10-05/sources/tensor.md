# Negative result — tensor_mom_v1 (third-moment method of moments as the label-free channel estimator)

**Hypothesis (one sentence):** the Jaffe-Nadler-Kluger (AISTATS 2015) third-moment method-of-moments estimator recovers each channel's sensitivity, specificity and the error prevalence from the binary top-20% marks more accurately than the Dawid-Skene EM / HEM / SML estimators of stage A (Step 450).
**Date / step:** 2026-09-30, Step 464
**Benchmark / population:** stage A population of expectation_realization_v1 (PRMBench steps of the three fit folds per eval fold, 13 channels, label release v3, 5 folds; inputs hash-identical to `results/expectation_realization_v1/run_20260927/INPUT_MANIFEST.json`, truth replayed from its `STAGE_A_CHANNELS.csv` to 1e-12 on all 5 folds). Labels enter only the truth side.

This is an estimation experiment, not a scoring experiment: no PRMBench within-AUC or ProcessBench F1 is produced, so the arm table carries the stage-A estimation metrics instead (mean over the 5 folds, `run_20260930/SUMMARY.json`, `compare_mean_over_folds`).

## Arms compared
| arm | what it is | prevalence error | MAE sensitivity | MAE specificity | Spearman of balanced accuracy vs truth | folds passing the stage-A bar |
|---|---|---|---|---|---|---|
| reference: Dawid-Skene EM (stage A, read from its outputs) | EM on the binary marks | 0.1375 | 0.1178 | 0.0816 | 0.7516 | 0/5 |
| HEM (stage A, read from its outputs) | hierarchical EM | 0.1435 | 0.1101 | 0.0717 | 0.7791 | 0/5 |
| candidate: tensor MoM | paper's Algorithm 1 (second + third moments) | **0.0920** | 0.1785 | 0.0768 | 0.7516 | 0/5 |
| SML with MoM's imbalance b | stage-A SML, b_hat from MoM instead of DS | — (uses MoM's b) | — | — | 0.7472–0.7582 | 0/5 |

Truth prevalence 0.137–0.141; MoM 0.231–0.232 on every fold; DS 0.134–0.141 error means about 0.28 estimated (`run_20260930/CHANNELS.csv`, `SUMMARY.json` per_fold).

**Win/loss record:** candidate vs reference on the fold-level bar (`bar_MoM.passes`): 0-5 for both arms; the bar is not met by any estimator. On the prevalence number alone MoM is closer on 5/5 folds (0.09 vs 0.14 error); on per-channel sensitivity it is worse on 5/5 folds (0.175–0.182 vs 0.118). No paired CI: this is a deterministic estimator comparison on identical rows, and the run does not bootstrap. The channels kept by MoM (pi_hat > 1/2) equal the channels kept by DS on 5/5 folds (`per_fold[k].kept_MoM == kept_DS`), so the frozen candidate (DS filter + plain average, Step 457) would be unchanged.

## Three plausible reasons it failed
1. The latent class the moment structure identifies is not the error label. All three estimators rank the channels the same way (Spearman 0.75–0.78) and all inflate sensitivity the same way (`q15_H1`: true 0.456, MoM 0.818, DS 0.757; `CHANNELS.csv` fold 0), which is what a shared latent factor would produce; Steps 460–461 found the label-free class shares the content channels with step position.
2. The estimator assumes conditional independence of the channels given the class; the channels are grouped (level sub-groups, Step 451) and the estimate's bias is nearly constant across folds (0.231–0.232), i.e. systematic, matching the synthetic within-group-dependence check of Step 463 (true 0.15, MoM 0.26–0.27).
3. The marks are a uniform top-20% inside every answer, including clean answers, so the vote matrix carries a fixed share of false marks by construction; the pooled prevalence 0.14 is not a quantity the marks can express.

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open
Third-moment MoM on these marks is closed as an estimator: it does not meet the bar, does not change the filter, and its better prevalence number does not carry over to the per-channel estimates; label-free estimation of channel quality on marks that are not conditionally independent stays open.

## What would reopen it
Votes whose dependence is removed before estimation (position-residualized channels, or one vote per declared block), or a marking rule that is not a uniform per-answer quantile; evaluated against the same stage-A truth on the same folds.

**Source files:** `results/tensor_mom_v1/run_20260930/SUMMARY.json`, `CHANNELS.csv`, `run_20260930.log`; command `python scripts/experiments/tensor_mom_stage_a_run.py run_20260930` from `.worktrees/tensor-mom-v1` (branch `claude/estimator-provenance-collection-2026-09-29`, tests `tests/test_er_stage_a.py` 10/10); 5 of 5 folds, 13 of 13 channels checked.
