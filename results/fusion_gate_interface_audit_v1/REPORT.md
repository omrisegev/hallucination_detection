# Our fusion needs both a gate and a useful locator

Completed diagnostic, 2026-09-07. Same 58 development answers: 12 PRMBench and 46 ProcessBench. IU-PCR / Joint L-SML remain the method. This audit changes no deployed prediction and promotes no candidate.

The next stage should address fusion representation and the location readout as well as the no-error decision. Fixing only the gate cannot solve the observed failures. Restoring a constant score offset cannot repair the current free-mean GMM gate.

## 1. What the current pipeline keeps and discards

Within each answer: form N windows by P features; subtract each feature mean and divide by its standard deviation; fit IU or Joint; normalize the fused score to unit standard deviation; use a GMM to decide whether two levels exist; return the highest-risk step if the gate opens. The feature signs have the declared negative-entropy anchor.

The normalization removes per-feature level and scale from the matrix used for fusion. All 58 coordinate-counterfactual checks reproduce the normalized matrix. All 361 valid core/answer checks have score mean zero and standard deviation one. Shifting the score and its step values by +10 leaves every GMM decision unchanged.

This is a verified invariance of this pipeline, not a proof that absolute telemetry is useless, that every gray-box method is impossible, or that centering caused all failures. The counterfactual transforms feature coordinates; it is not an alternate generated answer.

## 2. A perfect gate would still leave localization errors

PB macro-F1 (%), always on all 46 answers. Every oracle column uses labels and is diagnostic only. Invalid fits remain failures, including in the last column. These are ceilings conditional on the component held fixed, not attainable forecasts.

| Core | Actual | Perfect binary gate; same peak | Same gate; perfect locator | Both perfect on valid fits |
|---|---|---|---|---|
| Equal fusion | 17.76 | 44.92 | 54.13 | 100.00 |
| IU-PCR | 17.71 | 41.89 | 55.18 | 100.00 |
| Joint L-SML, lambda 0 | 12.50 | 44.33 | 23.33 | 75.73 |
| Joint graph, lambda 0.1 | 8.33 | 41.50 | 28.24 | 75.73 |
| Joint permuted graph | 4.17 | 41.50 | 20.83 | 75.73 |
| Entropy control | 19.85 | 33.39 | 37.56 | 100.00 |
| IU + graph correction | 14.45 | 40.06 | 44.15 | 100.00 |

The raw IU peak hits the first error in 7 of 25 erroneous answers; Joint lambda-zero in 9, graph 0.1 in 8, equal fusion in 8. Joint has four invalid erroneous answers. These are small development counts, not evidence of a reliable Joint win.

| Core | Peak before first error | Peak exact | Peak after first error | Invalid error fits | Exact peaks hidden by gate |
|---|---|---|---|---|---|
| Equal fusion | 9 | 8 | 8 | 0 | 4 |
| IU-PCR | 8 | 7 | 10 | 0 | 3 |
| Joint L-SML, lambda 0 | 6 | 9 | 6 | 4 | 3 |
| Joint graph, lambda 0.1 | 8 | 8 | 5 | 4 | 5 |
| Joint permuted graph | 7 | 8 | 6 | 4 | 4 |
| Entropy control | 9 | 5 | 11 | 0 | 0 |
| IU + graph correction | 10 | 6 | 9 | 0 | 4 |

## 3. Mixture states are not correctness labels

A two-component GMM models two distributions of score values. The code treats that as evidence of an error, but a clean answer can have two uncertainty levels and a wrong answer can have one. The observations below measure the mismatch; they do not invalidate mixture modelling as a possible supporting tool.

| Core | Clean: gate closed / 21 | Clean: gate open | Clean: invalid | Error: gate open / 25 | Error: gate closed | Error: invalid |
|---|---|---|---|---|---|---|
| Equal fusion | 12 | 9 | 0 | 17 | 8 | 0 |
| IU-PCR | 12 | 9 | 0 | 18 | 7 | 0 |
| Joint L-SML, lambda 0 | 4 | 11 | 6 | 16 | 5 | 4 |
| Joint graph, lambda 0.1 | 5 | 10 | 6 | 14 | 7 | 4 |
| Joint permuted graph | 4 | 11 | 6 | 15 | 6 | 4 |
| Entropy control | 6 | 15 | 0 | 19 | 6 | 0 |
| IU + graph correction | 9 | 12 | 0 | 17 | 8 | 0 |

The IU graph correction illustrates component interaction: its own gate gives PB 14.45%; reusing the original IU gate gives 24.64%. That frozen diagnostic was already in the preceding experiment. It does not establish a graph gain or improvement on both tasks.

## 4. A higher pooled PRMB AUC can leave every local ordering unchanged

For IU, 4,109 of 4,600 positive-negative step pairs (89.33%) compare different answers. The pooled benchmark score therefore measures both within-answer ranking and alignment of score levels across answers. Both matter for that metric, but they are different achievements.

The diagnostic origin projection removes the centering offset while retaining the answer-fitted scales and weights. For entropy, it retains entropy divided by its own standard deviation. The offset has no established calibration meaning, and feature units/origins matter. It changes cross-answer ordering only. This is not a new selected candidate.

| Core | Valid PRMB answers | Original pooled AUC | Origin projection pooled AUC | Mean within-answer AUC, both | Cross-answer pair share |
|---|---|---|---|---|---|
| Equal fusion | 12 | 0.61717 | 0.68891 | 0.66742 | 89.33% |
| IU-PCR | 12 | 0.62261 | 0.68174 | 0.67567 | 89.33% |
| Joint L-SML, lambda 0 | 7 | 0.66171 | 0.71112 | 0.69200 | 81.08% |
| Joint graph, lambda 0.1 | 7 | 0.66255 | 0.73723 | 0.68745 | 81.08% |
| Joint permuted graph | 7 | 0.66115 | 0.69876 | 0.69131 | 81.08% |
| Entropy control | 12 | 0.62587 | 0.67413 | 0.61303 | 89.33% |
| IU + graph correction | 12 | 0.61141 | 0.68902 | 0.67128 | 89.33% |

Keep the registered pooled endpoint unchanged for continuity; this is our project evaluation contract, not a claim about every metric in the PRMBench paper. Also report within-answer ranking and exact first-error localization. Different PRMB coverage prevents reading this table as a matched ranking between Joint and IU. The report preserves both the unweighted mean per-answer AUC and the pair-weighted decomposition in EVALUATION.json.

## 5. Neither absolute uncertainty nor more rows is an automatic fix

Descriptive binary-error AUROCs for fixed-direction probes, on the IU-valid population (all 46 PB answers). No threshold or sign is fitted from these labels. The small length-stratified development sample does not establish significance or transfer.

| PB subset | Clean / error | GMM BIC gain | Raw entropy mean | Raw spilled-energy mean |
|---|---|---|---|---|
| gsm8k | 6 / 4 | 0.667 | 0.500 | 0.417 |
| math | 5 / 7 | 0.686 | 0.429 | 0.429 |
| olympiadbench | 6 / 6 | 0.639 | 0.556 | 0.389 |
| omnimath | 4 / 8 | 0.625 | 0.969 | 0.625 |

Absolute entropy has a strong observed separation only in Omnimath here; it does not supply a demonstrated common gate. BIC gain has some observed binary ranking in each subset, but the current zero cutoff and model-fit assumptions are not calibrated correctness evidence.

BIC also uses the nominal row count. An analytic stress test duplicates every observation while keeping the fitted parameters fixed. It adds no information, yet the number of IU PB answers favoring two components rises from 27 to 35. This is not a refit result or a proposed correction. Non-overlapping windows can still be temporally dependent; the saved lag-one correlations are diagnostics, not validated effective sample sizes.

## 6. Continue development of our fusion

Next bounded stage: revisit the window feature bank and its Joint grouping using an explicitly local, multiscale representation, with IU and equal fusion controls. First audit the existing fast/slow, innovation and persistence mechanisms in spectral_utils/unified_causal_iu.py and unified_causal_subset_search.py. Their existing fitted pipeline and subset search explicitly use supervised development, so reusing fitted signs, rosters or references would require a borrowed/hybrid label. A new answer-only adaptation must fit its quantities from the current answer and disclose fixed engineering choices. These mechanisms are not newly invented here.

Evaluate the same frozen no-error gate as a diagnostic alongside each native gate so a feature/ranking change can be separated from a gate change. Check fit coverage and incremental information; correlated extra coordinates must not be counted as independent views. Freeze one bounded feature-bank comparison before looking at its new scores.

Preserve absolute feature summaries for a separate future gate experiment, rather than assuming that adding an arbitrary offset fixes GMM. Any pooled unlabeled normalization or gate is explicitly hybrid; retain the strict one-answer reference. Freeze that experiment before evaluation. Neither the current GMM zero threshold nor a label-picked replacement is a final correctness rule.

Joint feature/group discovery, learned temporal support, task-aware sampling, the full comparator registry/replay, untouched confirmation on both tasks and historical 24-cell transfer remain open. No method is declared a publication winner.

## Verification and historical bridge

Five scientific identity tests passed. Independent review rejoined 58 labels, reconstructed 116 raw absolute levels, checked 361 frozen score/gate records, independently counted 1,610 oracle outcomes, verified 35 PB macro values, 14 AUC decompositions and 84 descriptive probe AUROCs. Fourteen original task endpoints replay exactly.

Unlabeled audit time: 27.2 seconds on one CPU process; evaluation and review are additional. Maximum affine-normalization discrepancy: 3.55e-15; shifted-GMM BIC discrepancy: 4.66e-12. All bound source and result hashes match.

The historical 30-long-answer IU AUC 0.70070 and Claude pooled-fit experiments have different populations/fitting contracts. They remain context; this diagnostic replays the current 58-answer release exactly. No model inference or Claude worktree edit was needed.
