# Answer-localization representation pilot v1

Registered 2026-09-07 before producing or evaluating these candidates.
Status: development pilot, not untouched confirmation and not a new SOTA claim.
This implements stages B/C of `LOCALIZATION_RESEARCH_MANDATE_20260907.md`.

## Decision question

Does a local measurement bank, with enough windows in typical short answers,
improve answer-fitted IU and Joint relative to the original full-trace feature
definitions applied inside 32-token windows? Does the meaningful Joint graph
improve on its lambda-zero and node-permuted controls under that representation?

The two changes (measurement bank and window width) are separated by retaining
the new bank at width 32 as well as width 8. This is not an isolated test of a
single feature, and the 8-token arm does not compute unsupported short FFTs.

## Stable data and development cohort

Release ID: `localization-cached-v1-20260907`. Source: the nine frozen v2 NPZ
telemetry/label pairs in Claude's worktree. Save complete row/source-group IDs,
token/step geometry, source file hashes and label file hashes. This release is
explicitly previously exposed development data. It is not a confirmation pool.

Initial pilot: PRMBench Qwen3-8B and all four ProcessBench Qwen3-8B subsets.
Within each cell select four distinct source groups in each token-length bin
64–255, 256–1023 and 1024–2048, ordered by SHA256 of release/cell/group ID.
Take all if fewer than four are available. Keep one deterministically selected
answer per group; do not inspect labels for selection. Qwen3-4B is registered
for later replication on shared question IDs, not silently pooled as fresh
questions. Save the complete and selected ID manifests before scoring.

All candidates use exactly these IDs and official step spans. Failed fits
remain visible. No label balance is enforced; sparse or missing label classes
must be reported rather than repairing the cohort after looking at labels.

## Representations and frozen methods

| Representation | Window | Definitions | Orientation |
|---|---:|---|---|
| `legacy_fixed32` | 32 | Existing 30 window features | Original borrowed feature-sign contract; compatibility row |
| `global30_local32` | 32 | Same 30 definitions | Signs estimated from this answer's leading off-diagonal covariance loading |
| `moments27_local32` | 32 | Mean, standard deviation and least-squares endpoint trend of every primitive stream below | Same answer-local loading convention |
| `moments27_local8` | 8 | Same 27 local measurements | Same answer-local loading convention |

Primitive streams: entropy, sampled-token spilled energy, energy, top-1
log-probability, log-probability margin, top-k entropy, varentropy, Renyi-2
entropy, and top-k tail mass. Each gets the same three measurements; no
correctness-label feature selection. Standard deviation uses population
normalization; slope regresses on positions from −0.5 to +0.5, so its units
are the estimated change over a window. No features are duplicated to make
groups admissible. In the new local arms remove constant/nonfinite columns
and exact affine duplicates on fitting windows, retaining schema order.

Fit windows are nonoverlapping. An additional full-width end-anchored window
provides final-token scoring support but never increases the fitting count.
Minimum fit count is eight, an engineering floor, not proof of reliable
covariance estimation. Center/scale only within the answer. The local sign
routine is the existing `raw_orientation_cell`, with an explicit negative
entropy-level global anchor. This is a fixed physical orientation assumption,
not a sign table learned from other answers or a claim of anchor-free learning.
All within-answer signs are estimated again for each answer.

Each representation includes equal fusion, canonical IU-PCR and Joint
model-inverse lambda zero. The three local-sign representations also include
Joint LIU lambda 0.1 and its node-permuted graph control. No lambda grid or
label-selected feature/parameter search is run in this pilot.

Joint: four contiguous resampling blocks, the existing consensus procedure,
K candidates 3/4/6/8, minimum group size three, held admissibility 0.95, seed
2026090601, five starts, 5,000 maximum sweeps, target condition 1,000. New-arm
validity requires convergence, multistart PASS, full profiled global Jacobian
rank and finite Jacobian condition at most 1e8. Preserve the original admission
rule for the explicitly historical compatibility row. Persist every diagnosis
and distinguish numerical admission from useful prediction. No pooled or
fixed-group fallback is used.

Graph: existing adapted-DUFS soft gates, seeds 0/1/2, 120 epochs, k=7,
normalized-Laplacian roughness and trace matching as in short cycle 3.
Permutation seed derives from release/cell/row/representation identity. Use
the same fit and gates for meaningful and permuted graphs. No sampling-point
selector is introduced here; that remains a separate authorized stage.

Include raw entropy window mean at width 8 as a strong simple control. It is
a control, not an implicitly selected contribution.

## Score mapping and explicit no-error baseline

Risk is the negative of the normalized confidence fusion score, with unit
standard deviation on fitting windows. Map windows to tokens by mean of
covering windows, then to official steps by maximum risk, preserving the
previous ranking readout. Every token and official step must be scored.

For PB first-error/no-error decisions, use the same fixed, answer-only
mixture readout for every method. Fit one and two one-dimensional Gaussian
components to its nonoverlapping window risks (three starts, seed 2026090705,
max 300 iterations, variance regularization 1e-4). A non-converged fit is an
unavailable readout. For converged fits select two components only if their
BIC is smaller; otherwise predict no error. When two components are selected,
use the midpoint of their means as
the risk threshold and return the earliest official step whose maximum risk
exceeds it. It is a simple label-free baseline for a later chronological
HMM/BOCPD comparison, not a calibrated correctness probability. Mixture shape
can reflect benign reasoning changes; that assumption is being tested.

Fitting/readout failures count as misses on either PB target class in the
all-population metric. They must not earn a clean-answer success by default.
Report available/common valid predictions separately. Shortfall of eight
windows is an explicit unavailable fit, not a hidden no-error decision.

## Evaluation and stopping rule

Produce and hash all score/decision artifacts before opening evaluation
labels. Preserve a separate scoring process with no label loading API; hash
label files only as opaque bytes during release preparation.

- PRMB: pooled official-step AUROC, plus within-answer mean AUROC on mixed
  answers, with those excluded counts explicit. Use raw orientation; never
  flip a score using the evaluation AUROC.
- PB: per-subset error-side exact-first-error accuracy and clean-side
  accuracy, their harmonic mean, and equal-subset macro. Save complete
  per-answer predictions. If either target class is absent, mark that cell's
  metric undefined; do not silently treat its denominator as one.
- Every row: feature/window counts, fit coverage, failures, runtime and
  grouping/Jacobian diagnostics. Paired source-group uncertainty where the
  cohort supports it (1,000 draws, seed 2026090706); all pilot comparisons are
  exploratory and unadjusted. PRMB paired intervals use common valid fits;
  PB paired intervals use all selected answers and count failures as misses.
- Report width-32 bank comparison and moments-bank width comparison, IU/equal
  controls, and graph versus lambda-zero/permutation on common valid IDs.

This pilot can justify a bounded follow-up or a revised hypothesis. It cannot
promote a publication winner. No primary metric, cutoff, feature bank or
readout is changed using these results. Future changes get a new experiment
version within the same continuing release. Historical scalar-final-answer,
pooled/calibrated localization and this answer-only development lane stay
visible but are not combined into an unmatched leaderboard.

Compute cap: the above fixed cohort and methods, at most three CPU workers,
cached telemetry only. Persist each answer independently and resume only
verified missing artifacts. No inference, cluster GPU job, or mutation of
Claude's worktree is required.
