# Joint L-SML existing-data evaluation amendment R2

Registered status: `REGISTERED_POSTFREEZE_EVALUATOR_SERIALIZATION_AMENDMENT_R2`

R1 completed its registered 2,000-draw computation in memory but stopped before
writing an artifact because the `multi_solutions` PRMBench family has no
positive steps. Its family AUROC, AUPRC, and normalized AP are undefined, and
strict canonical JSON correctly rejects non-finite floats.

R2 changes only evaluation engineering:

1. Overall paired resampling retains the exact R1 group roster, nine strata,
   seed, 2,000 draws, tie-aware AUROC/AUPRC definitions, and contrast rules.
2. The implementation uses the maintained tie-block sufficient-statistic form
   from the historical H3 PRMBench evaluator. It generates each draw's
   stratified group counts in the exact same RNG call order as
   `paired_grouped_bootstrap`, then evaluates draws in vectorized blocks.
3. Registration requires numerical equality to the registered generic
   evaluator on real-data probe draws.
4. Undefined metrics in a single-class family are serialized as `null` with
   `metric_status=SINGLE_CLASS_NO_POSITIVE`; they do not enter the overall
   statistic or the decision state.

No score, method, feature, group, fusion weight, reducer, cohort, seed,
bootstrap draw, or decision rule changes. ProcessBench remains
`STRUCTURAL_NO_SCORE`, and its labels remain outside this evaluation path.

