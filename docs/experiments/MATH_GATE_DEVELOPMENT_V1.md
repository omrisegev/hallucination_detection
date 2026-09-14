# Math-panel gate development v1

Status before execution: **PROTOCOL FROZEN / NOT YET SCORED**

Frozen on: 2026-09-14

Post-execution decision note: the numerical three-feature winner is preserved
as a completed ablation but is not promoted because its improvement over the
best single was too small for the added complexity. The active simplified
decision is recorded in `SIMPLE_GATE_CHOICE_V1.md` and
`../../results/simple_gate_choice_v1/REPORT.md`.

Purpose: select one benchmark-independent detector of whether an answer contains
an error, using the 15 math/reasoning cells from the historical 24-cell panel,
then transfer that detector unchanged to the ProcessBench localization study.

## Question

Can the gate be improved by selecting both its token uncertainty signal and its
whole-answer temporal readout, and can several complementary candidates improve
over the best single detector?

This is retrospective development. All 15 math cells and their labels may be
used for selection. ProcessBench labels are not used to select the signal,
readout, feature set, fusion weights, or quantile. A later new-model dataset is
still required for external confirmation.

## Development population

The population is the complete math subset in `scripts/inscope_cells.py`:

- 10 GSM8K cells and 5 MATH500 cells;
- 18,614 generated answers in total;
- 11,013 correct and 7,601 erroneous answers;
- all cells contain raw top-50 log-probability telemetry.

The raw payloads must match the committed Git-LFS pointers in both byte size and
SHA-256 before feature extraction. Labels use the existing candidate-level
grader field: `error = 1 - label`.

## Candidate token signals

The signal roster is shared with `spectral_utils/gate_feature_readout.py` and is
computed without labels. Higher values always mean more evidence of error:

1. native stored top-15 Shannon entropy (renormalized head);
2. q15 `H0lim`;
3. q15 `VE0`;
4. q15 `VE0.75`;
5. q15 `VE1`;
6. q15 Shannon entropy `H1`;
7. q15 min-entropy `Hinf`;
8. raw top-1 surprisal `-log p1`;
9. raw probability mass missing outside top-15;
10. raw probability mass missing outside top-50;
11. arithmetic mean of the four frozen q15 finalist views.

The raw tail signals are deliberately not the historical
`topk_tail_mass`: that older feature renormalized the saved top-50 support and
measured mass outside top-5. Here `tail15_mass = 1-sum(exp(lp[:15]))`, preserving
the missing probability mass whose loss under normalization is being tested.

`entropy_native` is not full-vocabulary entropy in these caches: generation
computed it by renormalizing the top-15 probabilities. Therefore q15 `H1` is an
explicit reconstruction/control of the current entropy gate and is expected to
be a numerical duplicate. Its inclusion tests that identity rather than adding
an independent view. Likewise q15 `Hinf` differs from raw `-log p1` exactly by
the head-mass normalization term; retaining both directly tests whether that
normalization removes useful answer-level information.

## Candidate temporal readouts

Every token signal is reduced to an answer score by the same eleven readouts:

- mean over all tokens;
- mean of the ten largest token values (`Top10 mean`);
- token quantiles q75, q90, and q95;
- maximum rolling mean with window 8 (or the full trace when shorter);
- mean in each of four normalized-position quarters, separately;
- least-squares slope against normalized token position.

These are 121 single candidates. Step spans are intentionally excluded: the
same token-only definition must work on the historical math cells and on every
ProcessBench cell. No feature or readout may vary by benchmark, dataset, model,
or cell.

## Comparable scale and q

For evaluation and multi-feature fusion, each answer-level candidate is mapped
to its empirical percentile within its cell. This transformation is label-free,
preserves ranking, and prevents model-specific natural units from acting as
hidden fusion weights. It is a batch/transductive calibration and will be used
identically on the transferred ProcessBench cells.

The gate predicts error when its percentile score is at least `q`. The common
q is swept over `{.05, .10, ..., .95}` on the math panel. Exactly one q is
selected and frozen for all cells and for ProcessBench. In particular, q=.30 is
only a candidate; it is not retained by assumption.

## Evaluation and selection

The primary score is binary clean/error macro-F1, first averaged equally across
cells inside each dataset family and then equally across GSM8K and MATH500.
Secondary reports are family-macro AUROC, family-macro AUPRC, cell-macro scores,
the worst family and worst cell, accuracy, sensitivity, and specificity.

This family-first macro prevents the ten GSM8K cells from receiving twice the
weight of the five MATH500 cells. A single method and q are selected globally;
per-cell or per-benchmark winners are diagnostics only.

Learned fusion scores are leave-one-cell-out cross-fitted: each held-out cell is
scored by weights fitted on the other 14 cells. Candidate-set selection may use
all development cells, so the reported result is a development estimate, not an
untouched confirmation estimate.

Exact selection ties prefer, in order: higher family-macro AUROC, higher worst-
family F1, higher worst-cell F1, fewer features, then lexical method name.

## Fusion arms

After single-candidate screening, near-duplicates with absolute Spearman
correlation at least .995 are removed, keeping the better-ranked candidate.
Greedy forward selection may retain at most four candidates and stops when no
candidate increases the primary development score.

The following arms are compared:

1. best single candidate;
2. equal mean of the forward-selected percentile features;
3. non-negative simplex fusion whose weights sum to one;
4. L1-sparse logistic fusion.

For the simplex arm, fitted weights below epsilon=.05 are set exactly to zero
and the remaining weights are renormalized to sum to one. Logistic
regularization is selected from `C in {.01, .1, 1, 10}` using only training
cells within each cross-fit. Final weights are refit on all 15 development cells
after the winning arm is selected.

## Frozen transfer to ProcessBench

Only after the math result, selected candidate set, fitted fusion, and q have
been written and hashed may ProcessBench be evaluated. The transfer keeps:

- the signal definitions;
- temporal readouts;
- within-cell percentile calibration;
- selected feature set and fusion weights;
- common q;
- frozen q15 temporal locator and earliest-argmax rule.

The transferred gate first decides clean versus erroneous answer. If it opens,
the q15 locator supplies the predicted first-error step. The comparison includes
the original entropy-mean/q=.30 gate and the previously integrated
tail15-mean/q=.15 gate. No ProcessBench outcome may change the transferred gate.

## Required artifacts and gates

- raw-source audit with local and Drive SHA-256 values;
- deterministic unit tests for every readout and fusion constraint;
- complete 15-cell feature archive with identities and labels;
- all 121 single-candidate results and q curves;
- fusion-selection trace and leave-one-cell-out predictions;
- selected frozen gate, weights, q, and hashes written before ProcessBench;
- ProcessBench transfer table and error decomposition;
- explicit distinction between retrospective development and later external
  generalization.
