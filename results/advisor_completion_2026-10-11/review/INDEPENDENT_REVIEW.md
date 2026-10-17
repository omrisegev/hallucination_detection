# Independent advisor v11 review

Status: **PASS — no unresolved major implementation, population, numerical or training-leakage defects found in the reviewed delivery.**

This is a scientific and implementation review of the CPU completion package,
not a claim that the research method or a new variant has been validated on
untouched data. The final HTML, complete task inventory, metric conventions and
replay wrapper have now been checked. Publication remains separate from local
completion, and the excluded GPU experiment remains outside this delivery.

## Independently reproduced evidence

- Canonical membership and labels: 6,969 answers, 94,203 steps, 707 source groups;
  6,211 noncontrols / 83,371 steps; 758 synthetic clean controls; 160 MS answers.
  Five source folds agree with the authenticated group map and do not overlap.
- Direct error-versus-valid pair comparisons reproduce N16 AUC and top1 for
  L-SML, IU-PCR, plain average, DS-filter and the aligned PRM cache. This avoids
  relying on the production rank-sum helper. The distinction between 6,030
  AUC-eligible answers and 6,035 error-bearing top1 answers is correct.
- Q80, raw-survivor R2 counts and corrected-entropy gate flags reproduce exactly.
  Every subtype contingency table and the PS/DR/MS Sensitivity mean match.
- The masked telemetry SHA256 matches the MASK output receipt. All 6,969 raw
  answer-mean entropies and all 94,203 step lengths agree with derived inputs.
  Independently recomputed repetition values on every 97th answer plus the first
  three have no mismatches.
- Same-position/length matched AUC reproduces with its own eligible population:
  3,021 answers overall, 465 NR, 315 NCL. The 5,000 canonical-group bootstrap
  reproduces the NR L-SML-minus-PRM interval `[0.158891, 0.236413]`; the NCL Q80
  PRMScore interval reproduces as `[0.113474, 0.146403]`.
- Direct log-space Bayes computation reproduces the MoM posterior and rounded
  answer quotas. The tested quota loses overall PRMScore; that is an informative
  negative result, not an implementation failure.
- An independent sklearn refit reproduces N16 outer-fold-0 full LR. An additional
  independent nested probe repeats the five inner C=1 selectors and 35 candidate
  classifiers for LR on its selected half: masks match; balanced losses differ
  by at most `2.22e-16`; the winning C is 10.
- All 1,069 saved successful models reconstruct their held-out affine scores
  exactly. All 50 outer masks and every model's saved training mean/SD agree with
  the correct training partition. Maximum SD replay discrepancy is `1.81e-14`.
- The expected N11 / outer-fold-4 L-SML-on-LR-half arm fails with K=2. It is
  recorded explicitly rather than replaced. The displayed full-population
  diagnostic omits its partial numerical result and states the failure.
- The 353/354-group split and all leave-one-type-out partitions are disjoint;
  LOTO fitting excludes both the held type and the entire held source fold.
- Across all ten banks and 14 defined score exports, plain argmax and the
  canonical earliest-tie policy within eight floating-point epsilons agree.

Receipts and independent verification scripts are saved beside this report.

## Interpretation and limits

The code does not choose score signs from evaluation labels or rank concatenated
OOF probabilities globally. LR feature selection, standardization and C tuning
are confined to their relevant training splits, including inner validation.
Label-free grouping, weights and IU global statistics are also fitted on the
training partition for the new OOF diagnostics. Native transductive score replay
is distinguished from those new group-strict fits and from historical LR.

The research conclusions need to remain narrow: mask1 provides no demonstrated
general performance gain; NR remains strong under the registered coarse
controls, while NCL's matched interval crosses zero and there is no overall
controlled win. Conditional correlations are not an independence proof. High
reliability rank correlation does not establish calibrated sensitivity,
specificity or prevalence. Large cancelling coefficients in correlated digit
channels are not causal feature importance. LR-selected label-free fusers remain
supervised-selected diagnostics. New splits are internal checks on previously
inspected data; bootstrap intervals are pointwise and conditional on the fitted
scores and development selection, not a whole-learning-procedure guarantee.

The review did not refit every nested model, rerun GPU inference, independently
validate the canonical grouping algorithm itself, establish historical LR split
identity, or establish private Q80 registration time. Publication and Claude
access must be reported separately from local computation.

## Reporting issues identified during assembly

1. **Resolved:** the quota table used `answer_falseflag_rate` for error-bearing
   populations. It is now `answer_flag_rate`; `controls_falseflag_rate` exists
   only for genuine clean controls. The underlying flags and metrics are unchanged.
2. **Resolved:** the final task table includes previous state and action completed
   alongside status, finding and evidence for all 16 explicit assignment IDs.
3. **Resolved:** all six methods/baselines in the displayed nuisance-control table
   use the same position-and-length matched pairs and eligible denominators;
   every displayed point agrees with the persisted CSV.
4. **Resolved (P2 replay/provenance):** `--report` and `--all` initially verified
   only inputs/code while consuming stored output/checkpoints and refreshing
   hashes. Every replay mode now verifies all manifest-listed identities and recorded
   Python/numerical dependency versions before proceeding. An independent temporary fixture confirms that changing a saved
   output CSV is rejected by the actual manifest validator. Changed Python and dependency identities are also rejected. No delivery results
   were changed to run these probes.

## Final presentation and explicit assignment coverage

All IDs `1`, `2`, `3`, `4`, `5`, `5א`, `6`, `7`, `8`, `9`, `9א`, `9ב`, `10`, `11`,
`12`, `13` appear exactly once in the final task inventory. Required evidence paths
resolve, numerical stage statuses are complete, and the presentation distinguishes
local completion from unverified Claude publication. Task 7 preserves the single
undefined experimental arm rather than marking its numerical score complete.

The final HTML uses a common zero-based category scale for L-SML, IU-PCR and PRM,
keeps MS valid-step F1 and full PS/DR/MS Sensitivity consistent, and displays the
corrected AUC/top1/PRMScore headline values. It contains the source-group uncertainty,
coarse matching coverage, historical rule provenance, feature diagnostics and
negative quota result. LR-selected masks are labelled supervised diagnostics;
new split results are labelled internal validation on previously inspected data.
There are no duplicate section/SVG IDs, broken local evidence links or external
runtime dependencies. The independent parser review checks structure and values;
it does not substitute for browser screenshot/pixel review.

A further covariance-by-sums replay reproduces all feature/channel/group conditional
correlations and their differences (maximum discrepancy `7.71e-13`). Independent
vote contingency calculations reproduce sensitivity, specificity, reliability
ranks and calibration errors for both DS/MoM scopes. Low conditional correlations
remain a diagnostic, not a claim of independence.

The top-level manifest archives the declared runtime helpers, frozen inputs,
canonical folds, protocol and source snapshots, saved results and replay code.
The replay wrapper validates required task evidence and stage completion, rejects
changed identities, logs each subprocess and stops on errors. It resumes/replays
saved LR/checkpoints rather than promising a fresh fit of every expensive model.
The final inspected HTML SHA256 is `02cf84ae30cb87fe9446f5db084279e2b809abe2e06bb8e8cd84769ff237315d`
and is recorded in `HTML_RECEIPT.json` and `REVIEW_RECEIPT.json`; later rebuilds should preserve scientific content and be
checked against these receipts.

No unresolved major implementation or numerical defect was found. This PASS
approves the reviewed CPU delivery and its claim scope; it does not convert
negative/uncertain scientific findings into validated improvements.
