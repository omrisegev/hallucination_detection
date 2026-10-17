# Rényi position-temporal fusion v1

Status: frozen implementation contract before benchmark execution. Authorized
by Omri on 2026-09-14. This experiment combines the reviewed Stage-3b
Rényi/escort-varentropy representation line with the reviewed whole-answer
position-IU line. It is development evidence, not untouched confirmation.

## Question

Can a label-free fusion map that changes with normalized position inside the
complete answer recover part of the observed complementarity between low-alpha
and high-alpha token views, while preserving the frozen token Top10 readout?

The prior early/late analysis used labels and is hypothesis-generating only.
Neither training nor inference receives the true error position, step labels or
answer correctness.

## Frozen representation

For every token, use exactly four top-15-head views:

- `H0lim`: mean log q, the alpha-to-zero Rényi ordering.
- `VE_0`: uniform-escort surprisal variance, anchor-oriented.
- `VE_0.75`: the largest ProcessBench development point in Stage 3b.
- `VE_1`: frozen varentropy15 and the label-free orientation anchor.

The four fusion columns are oriented within each answer to its own `VE_1`
stream and z-scored within the answer. Single-view references retain the exact
Stage-3b orientation. No alpha is tuned in this experiment.

## Frozen fusion roster

1. Four single-view references and oriented equal fusion.
2. Answer-local static IU-PCR.
3. Other-answer stationary IU-PCR.
4. Other-answer IU-PCR with coefficients varying over 16 fractional
   whole-answer position regions.
5. Stationary IU coefficients with only the matched position-dependent mean.
6. Position-IU with deterministically shuffled position assignments.
7. Answer-local Shrinkage-IU with an external pooled covariance prior.
8. Answer-local Shrinkage-IU with an external position covariance prior.
9. The same position prior restricted to scaling the static local-IU direction.
10. The same position prior with shuffled assignments.

Other-answer moments are weighted equally by canonical source group and then by
answer. For an outer held fold, all its source groups are excluded. PRMScore
calibration uses the established nested two-fold exclusion. Borrowing strength
is fixed at 0.25 and the region count at 16, inherited from the completed
conditional-IU experiment; neither is swept.

Primary contrasts:

- other-answer position IU minus stationary-IU plus position mean;
- local Shrinkage-IU plus position prior minus its scale-only control.

Secondary controls compare each candidate with its static counterpart,
shuffled position, the four single views and the frozen external references.

## Evaluation contract

- Full cached development population: 13,769 answers.
- v3 labels and v2 canonical source-group folds.
- Complete-answer normalized position; steps never reset the position clock.
- Token score, then Top10 token mean per official step.
- Frozen mean-entropy q=0.3 no-error gate and earliest argmax tie rule.
- PRMBench within-answer AUC, pooled AUC and held-fold-blind PRMScore q=0.8.
- ProcessBench first-error macro F1 over all eight cells.
- 10,000 paired canonical-source-group bootstrap draws; primary family-wise
  intervals are 98.333%, secondary intervals 95%.
- Full population and explicit failures in denominators; no silent fallback.

Smoke is a 27-answer mechanics/exclusion check and cannot be used for method
selection. A full result is reportable only after score replay, source-group
firewall, calibration, coverage and independent endpoint reviews pass.
