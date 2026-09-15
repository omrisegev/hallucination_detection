# Step395: seven alternatives, reliability-error complementarity and shrinkage

User authorized evaluating ALL seven views Claude screened and mapping the
places IU-PCR/shrinkage could be used. The aim is complementary errors about
the same target, not merely low correlation between features. Full13769-answer
development population; no subset-based quality claim.

## Fixed raw streams

Provided-token surprisal (cached -log p), capped zero-based rank in top50
(50 means absent), cumulative mass preceding its saved rank (sum of all50 if
absent), log-probability gap top1 minus provided, corrected log tail15,
corrected log tail50, digit disagreement. Tail = log(max(1-sum(exp(lp)),1e-12));
lp is already normalized. Retain raw tail15/50 as two singleton controls since
the deployed gate uses raw mass. Neither variant subtracts logsumexp again.
Digit predicate and all raw sources must replay Step393. Provided probabilities
must agree with saved surprisal when the token appears in the top50.

## Fixed scoring roster

Nine singles (seven plus two raw-tail controls), each as:
- raw Top10 step diagnostic;
- base + .25*sd(base)*z(Top10 auxiliary), the common bounded correction.

Two banks: new7 and augmented12 (innovation5 + new7). Each bank has five heads:
equal; family_equal; native IU2PC; IU on diagonal-shrunk covariance; IU on
family-block-shrunk covariance. Families: existing shape5 (augmented only),
provided-token4, tail2, digit1. Groups are engineering provenance categories,
NOT assumed independent-error groups. Family equal allocates equal total weight
to each nonempty family. No group discovery or labelled subset search.

Within each answer: standardize token columns, drop constant columns, fit full
centered token covariance, calculate separate Top10 for each standardized stream,
then weighted sum. Global sign aligns native scores with equal risk evidence.
The final fused auxiliary is standardized over answer steps and added ONCE to
the unchanged innovation5 base at amplitude .25, for all heads. This makes
each head directly comparable to singleton corrections, including digit025.
Do not confuse this with Step393 direct bank replacement or token-fuse-then-Top10.

## Shrinkage rule

Estimate variability of covariance products using contiguous blocks of up to16
tokens over the full answer, crossing step boundaries. Weighted block means
reproduce the full covariance. For target T, alpha =
clip(sum weighted block variance-of-mean on changed off-diagonals /
sum(C-T)^2 on those entries,0,1). One block -> alpha1 if target differs.
This is a label-free block-variability heuristic, not an IID/LW optimality claim;
adjacent blocks may remain dependent. Diagonal target keeps only diagonals.
Block target keeps within-family entries and sets cross-family entries to zero;
both targets are PSD. Full IU re-estimates rho/subspace/solve on shrunk C.
Alpha0 must reproduce native IU. Constant/too-few-view/rank<2 failure explicitly
uses the equal head, counts reported. No signed-weight clipping or gamma search.

## Targets, evaluation and access

No labels enter extraction, normalization, grouping, shrinkage or fitting.
All operators in this new roster are answer-local; cached TCN is a comparator.
Fixed tail15 gate>=.33, same steps, labels and source folds. PRMScore uses
other-source-fold unlabeled quantiles q=.8; no predictor in these new heads,
so no nested model refits are required. Old TCN/digit references retain their
already nested thresholds. Complete system remains offline and gate-transductive.
Keep all Step39330 methods as explicit historical/current comparators.

Eight primary pairs x2 endpoints;10000 source-group bootstrap draws,99.6875% CI:
per bank, IU-equal; diag-IU; block-IU; family_equal-equal.
Secondary95%: each nine single corrections versus base and digit025; each of
ten bank heads versus digit025; log-vs-raw tail pairs. Report without adaptive
winner correction claims. Selection/existing bank exposure remains development.

## Error complementarity diagnostics

On4442 PB erroneous answers: raw peak successes (gate-independent), final
successes, joint raw failures, phi correlation of binary raw-failure events,
unique successes and gains/losses. Also stratify common failures by cell and
early/middle/late first-error position where sample sizes permit.
For PRMB, compare the nine standalone views on the SAME positive/negative step
pairs: pairwise misordering loss1, tie.5, success0; report answer-balanced
marginal losses and joint products. Labels enter only this diagnostic/evaluation.
These operational errors are NOT calibrated f_i-y residuals, and dependence of
their products is not a proof of the U-PCR latent-error assumption. No token
correctness labels inferred; PB post-first-error steps are not relabelled wrong.
Fixed885 raw/707 open historical miss cohorts remain descriptive.

## Outputs and limits

Source/formula audit, exact digit/base anchor replay, canonical IU sample audit,
five new numerical tests, full metrics/paired uncertainty/coverage, weights and
alpha diagnostics, per-example error ledger, plots of operational failure
overlap, and an end-to-end insertion map with historical leaders and current
evidence. Do not start gate learning, new predictors, final-answer transfer,
operator views or FM/DiFlo training in this stage. One hour execution cap;
checkpoint per-answer extraction and scored arrays. Review before any next stage.
