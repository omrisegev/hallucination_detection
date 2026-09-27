# Two complementary digit-family features, fixed before outcomes

2026-09-27. User requested two additional members of the new decoding-independent
digit family. Continue the authorized dedicated worktree/branch; preserve v1.

## Features

From original next-token top50 probabilities and ten verified ASCII digit IDs:
1. Existing second-digit probability, zero if fewer than two saved digits.
2. Digit spread: `M * H(p_digit / M) / log(10)`, M = total SAVED digit mass;
   zero for M=0. No full-vocabulary entropy or selected-token conditioning.
3. Second-digit probability minus its strictly previous all-token mean, token0
   inactive. No future information in the token feature; within-answer step
   normalization remains offline and noncausal, as in the baseline pipeline.

All three use Top2 mean per step, then per-answer step z-score. Innovation excludes
the first token during readout, and no-active-token steps map to neutral zero
after masked standardization. Top50 censoring remains explicit; only observed
digit probabilities enter these operational definitions. No inference or sweep.

## Full matched experiment

Same 13,769 source answers, 145,597 steps, raw hashes and span audit as v1. Replay
existing feature1 scores exactly before fitting. Two added features are compared
individually; one frozen family mean is a diagnostic, not the proposed final model.

Use the current bank11 L-SML implementation and source 3 fit / 1 calibration /
1 evaluation fold protocol. Bank11 base is the previously saved 11-column level
matrix, standardized per answer. The original entropy orientation anchor remains
unchanged. No chosen-digit anchor. No labels enter grouping, weights or calibration.
The historical bank itself was development-selected: this is not untouched research.

L-SML arms: bank11 baseline; bank11+one digit; bank11+three distinct digits;
bank11+three exact copies of feature1 (duplication diagnostic). Automatic grouping
remains automatic: three columns are not forced to have a protected family weight.
Report grouping and signed/absolute numeric weights in each fold.
Matched equal fusion on each bank is control-only. The duplicate arm isolates the
possibility of extra weight from counting a view repeatedly. No protected grouping.

Calibrate risk threshold to q80 on the held calibration fold using the same
answer-standardized score each method evaluates. Report official PRMScore on
6,211 noncontrol PRMB answers, within-answer AUC on 6,030 mixed answers, and PB
macro exact-first-error localization on 4,442 erroneous answer/model records.
PB endpoint is not F1/no-error detection. Reproduce frozen bank11 scores and
decisions across all rows before evaluating additions.

Four primary PRMScore contrasts: bank11+3 L-SML minus base, +1, duplicates, and
same-bank equal. 5,000 source-group bootstrap draws, Bonferroni family4. Other
endpoints and single features are diagnostic. No score-based variant selection.

## Error-dependence question

The six stable bank11 groups from historical source fits define comparison families.
Use per-answer standardized means within each group (diagnostic readouts), and the
mean of the three standardized digit features. Report score correlations separately
from errors. Each diagnostic family uses its own unlabeled held-calibration q80
threshold. On noncontrol PRMB steps, report decision-error phi correlations and
joint-error excess over marginal-product rates, separately conditional on correct
versus erroneous gold steps. Labels enter only this post-prediction diagnostic.
Also show PB missed-first-error correlations by cell to avoid pooling difficulty.
These are descriptive pairwise checks, not proof of conditional independence given
the unobserved L-SML latent variable; low correlation cannot certify independence.
Within-digit-family correlations and errors must be reported, since transforms of
one stream can remain almost redundant.

## Bounded execution and acceptance

CPU only, complete source population, no external-data access; 30-minute run cap.
Save inputs/code hashes and feature/prediction seals before opening quality labels.
Raw telemetry also carries labels; extraction code must not read them. Original
v1 quality is already known; new quality is only opened after freeze.
Mechanism tests cover digit mass weighting, multi-alternative sensitivity, causal
prefix response, token-choice invariance and missing positions. Scalar entropy
replay checks all raw tokens; previous feature and baseline score/prediction replay
checks all records. Official PRMScore replay required for all methods.
Promote only on demonstrated incremental gain, not three-member family size.
