# Independent retrospective audit: short cycles 1–3

Date: 2026-09-07. Original scores and reports were preserved. No fusion model
was refitted, and no new localization candidate was evaluated.

The original AUROC values reproduce. The evidence does not establish a clear
localization winner, and it does not close Joint/graph research on a different
representation. The current user mandate reopens that work while preserving
the negative results for the tested configurations.

## Findings, with their practical consequences

1. **The pilots borrow feature orientation.** The actual imported cycle-1
   module is in `local_cache/short_cycle01_code`, and it calls
   `confidence_sign_vector`. Its metadata explicitly records borrowed
   calibration. The follow-up cycles inherit this convention. The accurate
   description is answer-fitted fusion with historical feature signs. Keep
   this as a compatibility lane; a strict all-components-answer-only lane
   must state how orientation is obtained without those fitted signs.
2. **Answers are not all independent source groups.** The 30 answers contain
   29 registered source groups. The common 24 Joint/IU answers contain 23.
   Previous code sampled whole answers but called the unit "answer/source
   group". The revised intervals below resample source groups, including all
   retained answers of a sampled group together. They use the existing
   `source_idx` contract; possible broader problem duplication remains open.
3. **Convergence and multistart agreement are separate checks.** The original
   OK filter requires convergence but not the recorded multistart PASS flag.
   In these stored results, all 24 OK internal fits and all 27 OK fixed-group
   fits also have PASS, so this omission does not change those counts. This
   audit does not establish Jacobian identifiability; future fitting must
   persist and inspect that diagnostic.
4. **Source lineage is now explicit.** All eight capsule upstream hashes
   match their manifest. The actual runtime imports include both capsule and
   main-workspace modules. Their paths/hashes and the inspected score-array
   hashes are in `AUDIT.json`. These are present-day verification, not proof
   of every historical executed dependency. Lambda-zero replay matches at
   window, token and step levels with maximum absolute error zero.
5. **Raw-weight cosine can mislead.** Raw coefficients divide by feature
   scales, which can be tiny. A large or sign-changing raw coefficient is not
   sufficient evidence that predictions reverse. Future stability selection
   should inspect aligned predictions and standardized coordinates as well
   as fit diagnostics.
6. **The target is only partly tested.** The pilots are long PRMB answers,
   scored with pooled step AUROC. They do not test ProcessBench first-error
   plus no-error decisions. Per-answer standardization also removes some
   absolute between-answer information. Treat within-answer ranking and
   clean-answer detection as distinct questions with predeclared metrics.

## Matched comparisons with corrected source-group uncertainty

All differences are left minus right. Intervals are descriptive percentile
bootstrap intervals from 2,000 source-group resamples, seed 2026090704. Each
row uses that pair's common OK fits; different rows may have different IDs.
The altered random seed also differs from the original bootstrap, so interval
changes are not attributable solely to correcting the grouping unit.

| Comparison | Answers / groups | Left / right pooled AUROC | Difference, 95% interval |
|---|---:|---:|---:|
| IU − equal | 30 / 29 | 0.70070 / 0.69648 | +0.00422 [−0.01877, +0.02648] |
| Internal Joint lambda 0 − IU | 24 / 23 | 0.69148 / 0.72072 | −0.02923 [−0.04730, −0.00854] |
| Fixed-group Joint − IU | 27 / 27 | 0.62577 / 0.67159 | −0.04582 [−0.07273, −0.01972] |
| Fixed-group continuous L-SML − IU | 30 / 29 | 0.68916 / 0.70070 | −0.01154 [−0.02815, +0.00263] |
| Joint graph 0.1 − lambda 0 | 24 / 23 | 0.69188 / 0.69148 | +0.00040 [−0.00573, +0.00867] |
| Joint graph 0.1 − permuted graph | 24 / 23 | 0.69188 / 0.69578 | −0.00390 [−0.00993, +0.00265] |
| Joint graph 0.1 − IU | 24 / 23 | 0.69188 / 0.72072 | −0.02884 [−0.04781, −0.00677] |

## Additional within-answer diagnostic

This metric was added after inspecting the original experiment. It cannot
replace the original endpoint or justify a promotion. It averages per-answer
AUROC only for answers containing both clean and erroneous steps; single-class
answers are excluded. It does not measure no-error detection.

On the 18 mixed-label answers among the common 24 Joint/IU answers:

| Method | Within-answer mean AUROC |
|---|---:|
| IU | 0.74237 |
| Joint graph 0.1 | 0.73951 |
| Permuted graph | 0.73821 |
| Joint lambda 0 | 0.72802 |

Graph minus IU is −0.00286 [−0.03096, +0.02560]. Graph minus lambda zero is
+0.01149 [−0.00025, +0.02509], and graph minus permutation is +0.00130
[−0.00458, +0.00713]. None establishes an attributable graph advantage.
IU minus equal across their 24 mixed-label answers is +0.01124
[−0.00798, +0.03123]. The within-answer and pooled metrics answer different
questions and weight the data differently; the smaller gap is a reason for
a better specified next experiment, not evidence of successful recovery.

## Claude v2: what the inspected artifacts establish

`evaluation/report_contrasts.json`, written at 01:00 on September 7, contains
six PRMB contrasts and an empty ProcessBench object. The existing report has
pending/empty sections. PID 145884 was still present at inspection and was
previously identified as `report_v2.py --stage contrasts`. This is consistent
with report work still running even if Claude's conversational work stopped.
No new target-condition sweep was established by the inspected evidence.

The completed PRMB contrast entries include:

- Model-inverse lambda zero minus tuned IU: +0.00691 [0.00374, 0.01003].
- Permuted graph minus tuned IU: +0.00667 [0.00356, 0.00972].
- Meaningful graph 0.1 minus lambda zero: −0.00102 [−0.00151, −0.00052].

These support investigating the model-inverse mechanism; they do not support
attributing the gain to meaningful graph edges. Their pooled-across-answer
fitting contract differs from the Codex pilots, so their absolute AUROCs are
context, not paired comparisons to the table above.

The v2 tuned headline selects arms using inner-validation labels:
`evaluate_v2.py::_inner_select` evaluates step error flags and takes the
maximum mean metric. PB also fits correctness thresholds; Module-B blend
alpha uses labels. Label-free fold assignment and label-free base-weight
fitting do not make the selected pipeline fully unsupervised. Keep this as a
label-selected/calibrated comparison lane. No leakage allegation is implied
by properly nested tuning; it simply differs from the primary research scope.

Claude's two-fold K diagnostic observes inadmissible groups of size two for
larger K. It does not prove that every dataset has the same partition or that
three groups mean three features. The implemented Joint covariance includes
a nonnegative diagonal residual in addition to global and block components;
it is not necessarily rank K+1. Any proposal to permit pair groups needs a
separate identifiability analysis and estimator contract, not a silent change
to the existing minimum-group-size rule.

## How to continue

1. Freeze one continuing two-benchmark contract, retain existing incumbent
   and simple controls, and record a bridge to these historical pilots.
2. Reopen Joint with feature feasibility, nonredundant representation and
   within-answer stability diagnostics before increasing lambda or K. Keep
   lambda-zero and permuted-graph controls. Test IU on the same representation.
3. Separate feature fusion, chronological readout and observation sampling.
   HMM/BOCPD and graph token/window selection remain explicit research tracks.
   The sampling comparison needs full-grid and uniform controls and retained
   coordinates; selecting points must not hide unscored official steps.
4. Freeze a selection procedure using no task labels in the primary lane.
   Keep all already inspected data marked development. Verify a genuinely
   untouched confirmation population before making publication claims.

Execution/memory ledger:
`docs/experiments/LOCALIZATION_RESEARCH_MANDATE_20260907.md`.

Reproduce this audit with
`python scripts/audit_localization_short_cycles.py`.
It writes only this audit namespace. The JSON records the code hash,
environment, source identities, fit status counts and all numerical results.
