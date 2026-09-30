# Independent review — group confidence v1

Three independent reviewers followed `.claude/commands/red-team.md`, reading raw
artifacts and code, without original metric summaries or one another's outputs.

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| Full source evaluation is complete and correctly separated | confirmed | Population reviewer:13,769/13,769 IDs,145,597/145,597 steps,3,483 source groups,5 folds,10 arms. Every step written once. All50 calibration thresholds and decisions replay. Independent model reconstruction on cal/test differs by at most3.11e-15. `AUDIT_POPULATION.json`. |
| Candidate PRMScore is0.631878 and below same-bank means | confirmed | Independent scorer:6,211 noncontrol answers/83,371 steps; candidate0.631878315807,raw28mean0.640876303490,family15mean0.643322526912. All10 rows match the main scorer. `AUDIT_RECOMPUTE.json`. |
| Candidate source AUC/PB are0.757599/0.348029 | confirmed | Independent scorer:6,030 eligible PRMB answers,4,442 PB error answers across8cells. PB uses explicit first_error targets, not sentinel step labels. `AUDIT_RECOMPUTE.json`. |
| Frozen bank11 and CT7 controls are matched | confirmed | All13,769 rows/all145,597 steps; zero decision differences for all3 anchors. Maximum numerical score error2.11e-15. `AUDIT_POPULATION.json`. |
| Exact duplicate removal and complementary evidence work in the simulated model | confirmed, scoped | Synthetic30,000 held-out rows: one member log loss0.154357; three complementary members0.136218; averaging their evidence0.145559. Exact within-family copies cause zero score change. This does not establish robustness to all near-duplicates or across-group copies. `AUDIT_MATH_NULL.json`. |
| The implementation's binary likelihood and EM algebra are consistent | confirmed | Exact latent enumeration, MAP updates, sign gauges, gradient, binary-continuous identity and5/5 converged fits. `AUDIT_MATH_NULL.json`. |
| Source quality survives random labels | refuted, as desired for null | 20 within-answer label shuffles over6,030 answers: candidate AUC mean0.500292,SD0.002808,range0.494590–0.504355; control differences centered near0. `AUDIT_MATH_NULL.json`. |
| This is exact published L-SML / model-estimated accuracy is measured gold accuracy | refuted / not a supported claim | Fixed-family latent-tree EM is declared adaptation; continuous extrapolation is unproved; theta is relative to alpha, and a is relative to inferred Y. Off-block covariance residual0.407–0.419 indicates approximate model fit. `AUDIT_MATH_NULL.json`. |
| Sum-vs-average isolates a pure feature-count effect | weakened | Dividing entire log evidence changes its slope AND intercept (temperature). It tests this precise accumulation rule, not every possible use of group size. `AUDIT_MATH_NULL.json`. |

Additional intervention: permuting entropy within every answer, using frozen fits
and thresholds, changed1,408 candidate decisions over13,769 answers/145,597 steps.
Within-AUC changed+0.000030;27 other features remain. This is a sensitivity check,
not a null requiring the entire predictor to fall to chance.

The reviewers found no scoring or calibration defect requiring refitting. They
restricted the theoretical interpretation. All source data are development data;
the bank's historical selection used labels. No independent confirmation claim.

The population reviewer verified11 input hashes,5 frozen code/protocol hashes,
5 model hashes and all prediction/calibration seals. Audit commands and their
own input hashes are recorded in the three JSON outputs.
