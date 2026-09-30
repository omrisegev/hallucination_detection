# Corrected historical fusion refits v3, first comparator panel

This is a full-development benchmark component, not a new algorithm sweep.
Small preflight runs establish code fidelity only. No performance-based
selection occurs in preflight or while the full run is incomplete.

The primary research method still fits each answer alone. These five controls
instead fit normalization and fusion weights using other training answers.
They use the historical active-23 telemetry contract, fixed feature signs and
the exact v2 donor normalization/orientation implementation from the read-only
`hd_jlsml_v2_wt` checkout. The manifest hashes every imported reference source.
No new inference, Claude file changes, or relabeling are involved.

| Historical arm | Configuration / purpose |
|---|---|
| iu_c2_s25_l2_exoff | Deployed two-component L2 IU-PCR, scale .25, no exclusion |
| iu_c2_s25_l2_exon | Corresponding deployed U-PCR port, exclusion and recomputation |
| equal_all23 | Equal-weight control, same normalization/orientation |
| fixed_family_cont_unguarded | Historical fixed-family continuous L-SML |
| prov5_cont | Same provenance families, v2 small-m guard enabled |

These are the first controls, not the complete historical roster. Corrected
Joint model-inverse lambda0/LIU/permutation, DUFS-LIU, token-IU29, Unified28,
dedicated localizers and declared-access external comparators remain required
in `localization_full_benchmark_v3/METHOD_REGISTRY.json`. No family is closed
because it is not implemented in this panel.

## Population and folds

Use all 13,769 model-answer rows from the reviewed full-anchor manifest:
6,969 PRMBench and 3,400 ProcessBench answers for each of Qwen3-4B and Qwen3-8B.
Use v3 PRMB labels and the canonical v2 source-question outer/inner fold maps.
The canonical group in the manifest overrides historical telemetry group IDs.
Fit separately per cell. For every outer fold, fit on its four training folds
and evaluate all held-out rows. Keep short answers and numerical failures.
Fit-token cap remains the exact historical deterministic 60,000-token cap;
full evaluation does not mean changing the historical estimator's fit budget.

For PB only, five additional inner fits per outer fold produce calibration
scores using only outer-training groups. Every calibration row is held out
from the fit that scores it. The outer test groups enter neither the
preprocessing/fusion fits nor the threshold calibration. This needs 245 fits:
40 PB outer + 200 PB inner + five PRMB outer; each fits the five fixed arms.
No feature, graph, or hyperparameter selection from correctness labels.

## Readout and explicit label access

Preserve v2 token risk, top-10 token-risk mean within each reasoning step,
step maximum, and maximum over the full answer. PB's locator is argmax of the
step top-10 means; PRMB uses step maxima. Store float32 readouts as v2 did.
Historical preparation uses training medians to impute fit values, and its
training means to impute scoring values; preserve this exact behavior.

The PB decision is a **label-calibrated comparator**, not an end-to-end
unsupervised method. For each outer fold/arm, concatenate that fold's five
inner held-out calibration predictions across the eight PB cells. Search
the historical 99 detector quantiles (.01 through .99), maximize eight-cell
macro harmonic mean of clean/error exact accuracy, break ties by first/lower
threshold. Apply that one threshold to the outer fit's test predictions.
Inner donor scores share the declared fit-SD normalization; calibration
transfer across inner/outer fits is an explicit assumption, not guaranteed.
This nested calibration corrects the old panel's use of other outer folds'
OOF predictions, whose fusers could include the current outer test features.
Therefore old published development points are historical context, not a
claim of bit-identical corrected performance. Every failed PB output counts
as incorrect; it is never silently removed or called no error.

## Evaluation and review

Compare to the 19 full answer-only anchors on identical evaluation rows,
labels and outer folds, while labelling their different fitting/calibration
access. Primary PRMB ranking is the mean of five held-out-fold AUCs; include
within-answer AUC. Do not compare pooled OOF AUC scales. Report PB cell
metrics, Q4/Q8/eight-cell macro, clean accuracy, first-error accuracy, raw peak
accuracy and coverage/failures. These end-to-end comparisons also differ in
representation/readout; they do not isolate the fusion formula's contribution.

Use 1,000 joint canonical-source-group bootstrap draws for fixed-prediction
paired differences against answer-only IU and the equal control. PRMB paired
contrasts use each pair's common valid rows (and common mixed-label rows for
within-answer AUC), while absolute tables retain coverage on the full cohort.
Every PB failure remains in its full-population denominator. These
intervals condition on fitted weights/thresholds and do not quantify refit or
method-selection uncertainty. Require complete outputs before reporting;
no candidate promotion from partial folds. Review old GSM8K-Q8 fold-0 replay
of all five weights/readouts, row/group separation on every fit, source/label
hashes, coverage, and independent small-array metric checks. Save status,
fit metadata, score arrays, metrics, a simple English HTML report and review.
Full cached data remain exposed development data. Untouched confirmation is
separate and still required after method lock.
