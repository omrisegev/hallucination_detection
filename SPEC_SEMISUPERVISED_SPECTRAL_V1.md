# Semi-supervised spectral fusion v1

## Question

Can a small trusted correctness set repair the low-dimensional part of U-PCR
that unlabeled covariance does not identify reliably, while using fewer labels
than a supervised model over every feature?

This is a label-efficiency experiment, not a new hallucination-detection result.
The real replay is retrospective on the existing 24-cell derived-feature bundle.
Any promoted method still requires confirmation on a new dataset/model family.

## Frozen hypothesis

Let `w_U` be the U-PCR weight vector estimated from all *unlabeled training
rows*.  Construct a covariance-orthonormal score basis whose first direction is
the U-PCR score and whose remaining directions are the leading feature-
covariance directions after removing their component along earlier scores.
Estimate only the coefficients in this small basis from trusted labels, with a
quadratic prior centred on the U-PCR coefficients.

The primary hypothesis is:

> At 20 trusted labels, a six-direction U-PCR-anchored head improves held-out
> AUROC over U-PCR and over an equally labelled ridge-logistic head on all
> features.  Its advantage should be larger in planted conditional-dependency
> worlds than in an independent-error world.

## Inputs and leakage boundary

- Synthetic data: disjoint deterministic seed namespace
  `semi-supervised-spectral-v1-2026-08-06`; four worlds (`independent`,
  `grouped`, `sparse_pairs`, `correlated_weak_block`).
- Real replay: `results/dependency_fusion_raw/cells.npz`, reconstructed under
  `confidence-orientation-v1` with the four quarantined raw views excluded.
- Each repetition makes a stratified 60/40 train/test split.
- Standardisation, U-PCR, covariance eigenvectors, pseudo-labels, and all model
  fits use the training partition only.
- The test labels are read only by AUROC after every score is frozen.
- The trusted subset is sampled without replacement from the training
  partition, approximately preserving training prevalence and forcing both
  classes when the cell contains both.  This controlled stratification is an
  optimistic acquisition policy and must be named as such.

## Label budgets and repetitions

- Budgets: `0, 5, 10, 20, 40, 80` trusted labels.
- Confirmatory run: 40 repetitions per synthetic world and 30 repetitions per
  real cell.
- Quick runs are implementation checks only and cannot trigger a decision.

## Frozen methods

1. `upcr`: incumbent U-PCR configuration, fixed confidence orientation.
2. `platt_upcr`: positive one-dimensional calibration control.  It must have
   the same AUROC as `upcr`; otherwise the harness is wrong.
3. `gold_pcr2`: ridge-logistic regression on the top two unlabeled PCs.
4. `gold_pcr6`: ridge-logistic regression on the top six unlabeled PCs.
5. `gold_ridge_all`: ridge-logistic regression over all stable features.
6. `anchored_pcr2`: U-PCR score plus the minimum additional covariance
   direction, with prior strength 10 centred on U-PCR.
7. `anchored_pcr6`: U-PCR score plus up to five covariance corrections, with
   the same prior.
8. `pseudo_gold_pcr6`: the six-direction head trained on trusted labels plus
   U-PCR soft pseudo-labels whose *total* loss weight equals ten trusted
   examples.  This tests whether self-training adds more than the explicit
   U-PCR-centred prior.

All labelled heads use a fixed L2 strength; no hyperparameter is selected with
test labels.  Budget zero maps the anchored and pseudo-label heads exactly to
U-PCR and leaves purely supervised heads undefined.

## Primary real-data gates

All deltas are paired cell means after averaging repetitions; uncertainty is a
10,000-draw cell bootstrap.

At 20 labels, `anchored_pcr6` must satisfy all of:

1. mean AUROC delta versus `upcr` >= +1.00 percentage point;
2. bootstrap 95% lower bound versus `upcr` > 0;
3. mean AUROC delta versus `gold_ridge_all` >= 0;
4. QA and math domain deltas versus `upcr` each >= -0.50 points;
5. no more than two cells lose at least 5 points versus `upcr`.

These gates promote only the tested semi-supervised head.  Failure does not
imply that labels have no value; it distinguishes whether *this spectral prior*
improves label efficiency.

## Synthetic mechanism checks

At 20 labels:

- `anchored_pcr6 - upcr` must be positive in `grouped` and
  `correlated_weak_block`;
- `anchored_pcr6 - gold_ridge_all` must be positive in at least three of four
  worlds;
- mean harm versus U-PCR in `independent` must be no worse than -0.50 points.

## Falsification and interpretation rules

- If `platt_upcr` changes AUROC by more than `1e-10`, stop: a monotone
  one-dimensional transformation cannot change ranking.
- If `pseudo_gold_pcr6` tracks U-PCR and not the trusted-label methods, the
  pseudo-labels recycle the teacher; do not describe this as semi-supervised
  improvement.
- If `gold_ridge_all` wins at 5--20 labels, the unlabeled spectral structure is
  not buying label efficiency.
- If a gain appears only in one cell or only under controlled stratification,
  it is not evidence for a general method.
- Results on the existing real bundle are retrospective.  Passing permits a
  prospective replay; it does not establish a final contribution.

