# Independent review receipt

Reviewer: `/root/family_subset_review`, separate from implementing agent `/root/family_subset_run`.

Decision: **ACCEPT the executed complete comparison.**

The reviewer independently reproduced all 40 canonical fits (full/subset × two methods × two banks × five folds), with zero maximum score differences. It recomputed every within-answer AUC using scikit-learn on the 6,030 eligible answers; the eight means match exactly. Official evaluator Overall and pooled noncontrol macro-F1 differ by no more than 1.11e-16. Donor Q80 thresholds and test flags match exactly. All eight AUC/PRMScore confidence intervals and four paired contrasts reproduce within 1.11e-16. Prediction seal, code/input hashes and source-disjoint identities pass.

The subset L-SML has seven automatic clusters in every N56 fold and four in every N16 fold. Effective coefficient vectors are not equal weights. Reviewer measured cosine to equal coefficients around .704–.706 (N56 subset) and .966–.968 (N16 subset); this is a diagnostic of unequal weighting, not a performance comparison with a mean arm.

Minor receipt correction: AUDIT.json initially wrote nominal zero for official equivalence after asserting a 1e-12 tolerance. It now records the actual float64 difference. Scores, metrics, fitting code and the prediction seal were not changed.

Scope limitation: generic all-fold failure would require an unavailable-results display instead of meaningful metrics. No failure occurred in any of the 40 executed fits, so this does not affect the current complete comparison. No variant search, sign correction, imposed family clustering or fallback was used.

Additional implementer acceptance evidence is in VALIDATION.json: 20 fresh subset refits, 40 threshold/flag reconstructions, exact metadata-to-label reconstruction, unchanged frozen package inputs and full-bank flag parity with the prior clustered comparison.
