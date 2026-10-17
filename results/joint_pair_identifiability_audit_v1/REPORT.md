# Allowing feature pairs inside Joint L-SML

2026-09-07. Reviewed mathematical and unlabeled structural audit.


## The useful result

A careful extension can admit pairs of features without replacing our fusion. Joint fit coverage rises from 78 to 106 of 110 answers for the moment bank, and from 102 to 108 for context. At least one bank has a valid Joint fit for every answer, versus 107 previously. This earns a localization experiment; it does not yet show higher AUC or better error detection. No error labels were read in this stage.


## Where this fits in our method

The input is still N windows from one answer by P feature measurements. Joint estimates a global factor shared across features and a local dependence factor inside each feature group. Its native fusion weights solve a regularized covariance system. We change how a two-feature group is represented inside that same covariance model.


## One equation cannot identify two loadings

After removing the global contribution v_i*v_j from the covariance between two features, Joint fits the residual r = u_i*u_j. A residual of 0.20 can come from (0.5, 0.4), (1.0, 0.2), or many other pairs. The product is determined, but the individual loadings are not. Choosing equal-magnitude loadings is a convention. Claude's report and R1 amendment describe this as an exactly determined system; that explanation is too strong.


## Why this matters for the code

An arbitrary loading ratio is harmless to the native fusion if its covariance stays unchanged. However, the old constructor computes diagonal noise as observed variance minus factor variance, then clips negative values to zero. A rescaled pair can exceed the available variance, inflate the covariance diagonal and change the final weights while keeping the same off-diagonal fitting objective. The existing global-loading Jacobian still passes in our counterexample. That test checks global loadings after removing nuisance directions; it is not a full covariance/head invariance test. The frozen minimum-three experiment is preserved; the counterexample tests what would happen if pairs were admitted naively.


## A numerical counterexample you can inspect

This is a synthetic covariance with six features in three pairs, not a benchmark answer. All rows below have identical off-diagonal covariances and pass the global Jacobian check. The old clipped weights vary; the new pair covariance and its weights stay fixed. The full off-diagonal Jacobian has rank 9 for 12 latent loadings: three pair-scale directions remain undetermined.

| Scale t | Old variance 1 | Old variance 2 | Old weight 1 | Old weight 2 | New weight 1 | New weight 2 |
|---|---|---|---|---|---|---|
| 0.03 | 1.00000 | 69.56694 | 0.15629 | 0.00250 | 0.15586 | 0.18515 |
| 0.2 | 1.00000 | 1.68500 | 0.15604 | 0.10703 | 0.15586 | 0.18515 |
| 1.0 | 1.00000 | 1.00000 | 0.15586 | 0.18515 | 0.15586 | 0.18515 |
| 4.0 | 1.00000 | 1.00000 | 0.15586 | 0.18515 | 0.15586 | 0.18515 |
| 10.0 | 4.09000 | 1.00000 | 0.03679 | 0.18543 | 0.15586 | 0.18515 |
| 30.0 | 36.09000 | 1.00000 | 0.00413 | 0.18551 | 0.15586 | 0.18515 |


## The pair representation we implemented

Let b_i = S_ii - v_i^2 be the variance left after the global factor, and similarly b_j. The pair is feasible exactly when both budgets are nonnegative and |r| <= sqrt(b_i*b_j). We allocate the same fraction of each budget to the group factor. This preserves the fitted product and the observed diagonal. It is a declared representative of equivalent loadings, not recovered latent truth. The invariant statement is conditional on the same global loadings, pair product and other groups. Near-roundoff adjustments are recorded; substantive infeasibility is rejected.


## A second review finding: zero pair products

At u_i=u_j=0, the old first-order nuisance derivative vanishes. That can make its global-identification check pass even though a different global loading can be absorbed by a changed pair product. The amended check represents each pair by its product directly, including at zero. A synthetic counterexample passes the old check and correctly fails the amended one. This was added transparently during review, after the 110 fits completed; the original frozen prototype and fitted arrays are preserved. All 219 fitted records were checked again, with an independent product-profile reconstruction on 69 pair fits. No current eligibility status changes, and none of those pair products is exactly zero. Three additional tests pass. Future experiments must call fit_joint_pairs_checked from joint_pair_jacobian.py, rather than the earlier prototype directly.


## What was held fixed

The 110-answer cohort, both 27-coordinate feature banks, width-eight windows, answer-only normalization and fixed negative-entropy anchor, four chronological blocks, seed, K={3,4,6,8}, held-block admissibility fraction 0.95, five optimizer starts, 5,000-sweep cap and inverse condition target 1000 were retained. We lowered the group minimum to two and added the pair covariance/validity treatment. We did not simultaneously widen K or run a condition-number sweep. The lower minimum also applies to held-block partitions: a final partition without a pair can become admissible because one held-block partition contains a pair. No inference, graph-quality or hierarchical-head experiment was run.


## Matched structural comparison on the same 110 answers

Higher coverage is useful, but not a correctness metric. Changing the selected partition can lose an old successful fit, so rescues and losses are both shown. The union across banks is 110/110; this is a potential route, not a new measured localization score. The previous quality results and all benchmark targets remain unchanged.

| Bank | Old valid | New valid | Rescued | Lost | Valid fits with a pair | Same selected partition |
|---|---|---|---|---|---|---|
| moment | 78 | 106 | 31 | 3 | 55 | 32 |
| context | 102 | 108 | 7 | 1 | 12 | 87 |


## More groups now become possible

For moment features, 60 of 110 selected partitions now use six or eight groups; previously every admissible partition had three or four. There are 55 valid moment fits and 12 valid context fits whose final partition actually includes a pair. Context still selects only three or four groups. A larger K is not itself evidence of better fusion.

| Bank | K | Old selected | New selected | New valid |
|---|---|---|---|---|
| moment | 3 | 53 | 24 | 23 |
| moment | 4 | 26 | 26 | 26 |
| moment | 6 | 0 | 51 | 49 |
| moment | 8 | 0 | 9 | 8 |
| context | 3 | 77 | 64 | 63 |
| context | 4 | 26 | 46 | 45 |
| context | 6 | 0 | 0 | 0 |
| context | 8 | 0 | 0 | 0 |


## Coverage by benchmark cell

All counts use the same fixed source questions as the preceding replication. The current cache is development data already evaluated elsewhere in the project; the lack of label reads in this audit does not make it untouched confirmation.

| Cell | Answers | Moment: old -> new | Context: old -> new |
|---|---|---|---|
| prmbench_qwen3_8b | 24 | 17 -> 23 | 21 -> 23 |
| pb_gsm8k_q8 | 16 | 14 -> 14 | 14 -> 15 |
| pb_math_q8 | 24 | 16 -> 24 | 23 -> 24 |
| pb_olympiadbench_q8 | 22 | 14 -> 21 | 21 -> 22 |
| pb_omnimath_q8 | 24 | 17 -> 24 | 23 -> 24 |


## The remaining failures are real

Five fits fail convergence/multistart guards: three moment and two context. One additional moment fit has an infeasible pair. Its variance budgets are 0.0299446 and 0.00212325, permitting a residual product magnitude at most 0.00797371; the fitted product is 0.0145291. The extension rejects it instead of inflating a diagonal. We do not pick another partition after seeing this outcome in this stage. This feasibility guard deliberately changes how a pair-related diagonal conflict is handled; it is not a general repair of all historical diagonal clipping.


## Review and limits

Eight original scientific tests and three review-amendment tests pass, including a real pair fit, exact minimum-three replay, scaling invariance, signed and zero products, infeasible variances, permutation/sign equivariance and a finite-difference Jacobian. Independent review reconstructs 220 normalized feature covariances, 880 candidate-group guards/ARI summaries, all 220 selections, 219 native inverse maps, 107 pair products/variance allocations, the one infeasibility, and 117 unchanged-partition parent maps. Nine representative pair refits match. The original clustering labels and optimizer are reused; this is not an independent clustering/optimizer implementation. The mathematical example demonstrates a possible failure of naive pair admission, not that every unconstrained pair fit in practice would fail.

| Check | Evidence |
|---|---|
| Scoring wall time, three CPU workers | 130.96 s |
| Independent review | 8.44 s |
| Maximum native-weight reconstruction difference | 1.2717604747081168e-13 |
| Benchmark error-label reads in this stage | None |
| Browser visual inspection | Not run; structural/link checks only |


## Next: measure the contribution to localization

Freeze one pair-enabled native Joint recipe with lambda zero, graph lambda 0.1 and permuted-graph controls. Compare it on shared development answers with the frozen minimum-three versions, moment/dual IU, both equal banks and matched routing controls. Preserve native no-error gates and fixed-IU diagnostics separately. Report failures and common-ID ranking alongside full-population decisions. Only those results can tell us whether the added fitting capacity helps. Hierarchical heads, wider hyperparameter work, supporting temporal/geometry/sampling ideas, corrected-fold multi-answer contenders, full comparator coverage, untouched confirmation and historical 24-cell transfer remain open.


## Evidence and code

- [Structural results](SUMMARY.json)
- [Counterexample arrays and weights](ALGEBRA.json)
- [Independent review](REVIEW.json)
- [Eight original scientific tests](TESTS.txt)
- [Three zero-product review tests](JACOBIAN_TESTS.txt)
- [Post-fit review amendment and source hashes](JACOBIAN_AMENDMENT.json)
- [Frozen inputs and scientific source hashes](MANIFEST.json)
- [Previous quality results and historical comparisons](../fusion_replication_v1/REPORT.html)
- [Pair covariance construction and optimizer wrapper](../../spectral_utils/joint_pair_extension.py)
- [Checked entry point for future pair-fusion experiments](../../spectral_utils/joint_pair_jacobian.py)
- [Frozen audit protocol](../../docs/experiments/JOINT_PAIR_IDENTIFIABILITY_AUDIT_V1.md)
- [Visual guide to the existing fusion method](../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html)
