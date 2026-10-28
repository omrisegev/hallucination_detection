# Negative result: family-representative refactor of N56 (2026-10-20)

**Hypothesis.** A label-free transformation T of N56 that removes copies and merges redundant blocks
(and, in one arm, removes step position/length) yields a feature set that better satisfies L-SML's
assumptions, and L-SML on it scores higher than L-SML on N56 (spec: docs/experiments/FAMILY_REPRESENTATIVES_V1_HE.md, 10-12).

**Arms (frozen in FREEZE.json before labels).** A = drop |r|>=.999 copies + PC1 of |r|>=.8 blocks (56 -> 39).
B = A + cubic-spline residualization on relative step position and log step length. Reference: N56.
Fusion: continuous L-SML (primary), IU-PCR, plain average, L-SML with Qwen2.5-Math-PRM-7B as an extra view.
PRMBench thinkfix mask1, 6,969 answers / 94,203 steps, 5 source-group folds, T and fusion fitted on training folds.

**Win/loss.** Loss on every fusion rule and both endpoints. L-SML within-answer AUC: N56 .7704, A .7490, B .7392;
PRMScore .6410 / .6282 / .6245. N56 reproduces the earlier saved result exactly.

**Effect with CI (paired, 707 source groups, 2,000 draws).** A - N56 L-SML: -.021 AUC [-.028, -.016], PRMScore -.013 [-.018, -.008].
B - N56 L-SML: -.031 [-.035, -.028], PRMScore -.016 [-.020, -.013]. All 8 AUC contrasts and all 8 PRMScore contrasts have intervals below 0.

**Three reasons.**
1. Weighting, not information (A): the uncertainty-level block (9 near-copies, AUC .73-.75) held 25.6% of the L-SML weight
   in N56 and 7.1% after merging; supervised LR barely moves (.821 -> .814). The copy count acted as a correct weight prior that
   L-SML's shrink-to-equal estimator cannot recreate for one merged feature.
2. Signal removal (B): position/length residualization drops the supervised ceiling .821 -> .780 and energy_level .759 -> .702.
3. The structural gain was real (S1g .52 -> .40 / .23; between-group |r| given the truth -8 to -40%) but is not the binding
   constraint on quality here.

**Direction vs implementation.** The direction "make the bank satisfy L-SML" is not refuted, but this implementation shows that
removing redundancy without carrying its weight hurts. Not tested: block-mass-weighted merged features, L-SML+HEM, other banks,
external data. PRM weight share stayed 1-3% (hypothesis of a larger PRM weight not supported).

Files: evaluation/RESULTS.json, evaluation/DIAGNOSE.json, METRICS_A_STEPS.json, FREEZE.json, evaluation/PREDICTION_SEAL.json.
Presentation: https://claude.ai/artifact/VwkriRnvj4ashuxf7mmj24
