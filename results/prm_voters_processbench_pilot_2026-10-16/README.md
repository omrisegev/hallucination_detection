# Pilot — PRMs/judges as label-free voters on ProcessBench (2026-10-16)

Script: `scripts/run_prm_voters_processbench_pilot.py` (CPU, ~3 min). Raw numbers: `RESULTS.json`.

## Setup
- 3,392 ProcessBench answers (25,618 steps), joined on `id`; 8 critic records unusable (truncated/no prediction).
- Voters: `prm7b` Qwen2.5-Math-PRM-7B · `j8b` Qwen3-8B judge (uPRM baseline) · `j14b` Qwen2.5-14B uPRM judge (L1) · `c72b` Qwen2.5-72B critic.
- Votes = each voter's **native** first-error prediction (reconstruction matches the saved `prediction` 3400/3400 for prm7b/j8b/j14b).
- Step target is cumulative: step k is "erroneous" iff the gold first error is at or before k (fully defined by ProcessBench labels).
- MoM = `spectral_utils.error_count.fit_mom` (strict, no clipping). L-SML = `fusion_utils.lsml_continuous` on z-scored continuous views (prm7b: 1 − prefix-min reward; judges: CDF of first-error distribution; critic: binary). Orientation is semantic, no labels used to fit.
- Bootstrap: 2,000 answer-level resamples of AUC(L-SML) − AUC(best single), scores held fixed (no refit).

## Pooled results
| | prm7b | j8b | j14b | c72b | MoM | L-SML (4) | mean-z (4) |
|---|---|---|---|---|---|---|---|
| answer AUC | **0.944** | 0.544 | 0.730 | 0.821 | 0.917 | 0.913 | 0.899 |
| step AUC | **0.947** | 0.515 | 0.712 | 0.783 | 0.902 | 0.905 | 0.880 |
| ProcessBench F1 | **0.713** | 0.084 | 0.186 | 0.565 | 0.591 | 0.542* | — |

\* L-SML thresholded at the MoM prevalence quota (label-free).

L-SML − prm7b: answer −3.1pp [−3.9, −2.2], step −4.2pp [−5.0, −3.5]. Same sign in all 4 subsets (omnimath answer CI crosses 0).

### Error-count estimation
| | truth | MoM | prm7b vote rate |
|---|---|---|---|
| erroneous answers (fraction) | 0.652 | 0.521 | 0.601 |
| erroneous prefix-steps (fraction) | 0.442 | 0.326 | — |
| per-answer count MAE | — | 1.30 | **0.95** |

MoM underestimates prevalence by ~13pp in every subset, and fails (`INFEASIBLE_EMISSIONS`) at step level on omnimath.

## Diagnosis
1. **One strong voter, three weak.** j8b is at chance (AUC 0.51–0.55). MoM detects this without labels (estimated sens ≈ 1 − spec ≈ 0.52). Fusion can only dilute prm7b.
2. **Conditional dependence.** prm7b and c72b correlate 0.30–0.34 *within* each class. MoM reads this shared signal as higher accuracy: it estimates prm7b/c72b sensitivity at 0.94/0.93 when the truth is 0.86/0.74, and that pulls prevalence down.
3. L-SML puts j8b in the same group as prm7b and c72b (groups `[0,0,1,0]`), so the group detection does not isolate the noise voter at m = 4.

## Verdict
Negative pilot for this voter set. The idea is not refuted. The pool is too unbalanced: one PRM-trained scorer, two generic LLM judges, one critic. The real test needs several **comparably strong, independently trained PRMs** on PRMBench (per-step labels), as planned in the adapter work in `prm-boosting-generalization-v1`.

## Follow-up: per-answer counts and PRMScore
Script: `scripts/run_prm_voters_processbench_counts_prmscore.py` → `COUNTS_PRMSCORE.json`.

**Why ProcessBench and not PRMBench.** Of the four voters, only PRM-7B has been run on PRMBench. j8b, j14b and c72b exist only on ProcessBench. ProcessBench labels only the first error, so:
- "count" here is the cumulative count: steps from the first error to the end, 0 for correct answers;
- PRMScore = (F1_err + F1_valid)/2 on the ProcessBench step domain: all steps of correct answers, plus steps up to and including the gold first error.

These numbers are not comparable to PRMBench's official PRMScore.

### Mean count per answer (cumulative)
| | all | erroneous answers | correct answers | MAE vs truth | bias |
|---|---|---|---|---|---|
| truth | 3.34 | 5.12 | 0 | — | — |
| prm7b | 3.01 | 4.37 | 0.46 | **0.95** | −0.33 |
| j8b | 3.87 | 4.07 | 3.49 | 3.83 | +0.53 |
| j14b | 1.52 | 1.90 | 0.81 | 2.88 | −1.82 |
| c72b | 2.22 | 3.22 | 0.35 | 1.49 | −1.11 |
| MoM (posterior sum) | 2.46 | 3.56 | 0.38 | 1.28 | −0.88 |

### PRMScore (pooled, 3,392 answers)
| scorer | native | Q80 | MoM count | MoM quota |
|---|---|---|---|---|
| prm7b | 0.795 | 0.614 | **0.797** | 0.793 |
| j8b | 0.412 | 0.504 | 0.721 | 0.465 |
| j14b | 0.545 | 0.510 | 0.721 | 0.546 |
| c72b | 0.733 | 0.683 | 0.697 | 0.733 |
| L-SML (4) | — | 0.555 | 0.721 | 0.699 |
| MoM posterior | 0.742 | 0.669 | 0.792 | 0.742 |

- Q80 = canonical ANSWER_Z_Q80: within-answer z-score, global 80th percentile.
- MoM count = per-answer round(sum of posterior), then the top-k steps of that answer.
- MoM quota = the global top π̂ fraction of steps.

L-SML − prm7b (paired answer bootstrap, 1,000 draws, scores held fixed):
- Q80: −5.9pp [−6.8, −5.0]
- MoM count: −7.6pp [−8.6, −6.7]
- MoM quota: −9.4pp [−11.0, −7.9]

**Caveat on the MoM-count column.** j8b, j14b and L-SML tie exactly at 0.721. Their cumulative scores rise strictly within each answer, so "top k" always means "the last k steps", and the arm measures only the quality of MoM's count. prm7b differs because its prefix-min score has flat runs, and stable tie-breaking then picks the step where the error starts.
