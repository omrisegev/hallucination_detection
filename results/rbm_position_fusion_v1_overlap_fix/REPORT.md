# Position-conditioned RBM fusion: full result

The tested correction is not adopted: it lowers ProcessBench and within-answer PRMBench against its fixed-weight baseline. No next model has started.

Full corrected v3:13,769 answers,145,597 steps,6,030 eligible mixed-label PRMB answers. Bank12, same-answer unlabeled fitting, unchanged externally calibrated entropy q=.3 gate, top10 token mean and original argmax. The new arms never use first_near_max. The32 prior arms remain unchanged archived comparators.

| Method | PB macro % | PRMB within AUC | PRMB pooled AUC | PRMScore |
|---|---:|---:|---:|---:|
| RBM12, chronological half weights | 35.5837 | 0.742490 | 0.706144 | 0.622079 |
| RBM12, shared weight update | 36.2712 | 0.745204 | 0.705849 | 0.622215 |
| RBM12, shuffled step groups | 35.6343 | 0.743963 | 0.704691 | 0.621543 |
| RBM12, fixed weights, Logit | 36.2712 | 0.745204 | 0.705849 | 0.622215 |
| RBM12, fixed weights, Posterior | 36.3750 | 0.738702 | 0.710417 | 0.629276 |
| RBM6, fixed weights, Posterior | 36.2017 | 0.735982 | 0.708110 | 0.630749 |
| Varentropy15 contributions + IU-PCR | 35.3498 | 0.746824 | 0.710261 | 0.622689 |
| Varentropy15 | 35.9610 | 0.737786 | 0.710138 | 0.625781 |
| Varentropy50 | 35.6755 | 0.742465 | 0.715783 | 0.632777 |
| Entropy | 35.4444 | 0.730111 | 0.702664 | 0.625426 |

Model: freeze the saved shared visible mean, hidden bias, weights, normalization and orientation. Learn one correction d from exact conditional Gaussian-RBM likelihood: early weights w-d, late weights w+d. Early means first ceil(number of steps/2) steps. Ridge lambda=.1+P/min(early tokens,late tokens), no tuning. The shared and shuffled controls use the same parameter count, ridge and optimization budget. This tests a constrained weight adaptation, not every possible conditional RBM.

Primary paired position-minus-fixed and position-minus-shared contrasts coincide: PB -0.6876pp,97.5%CI[-1.2541,-.1256]; within-answer AUC -.002714,CI[-.003756,-.001684].10,000 source-group bootstrap draws. Other comparisons are descriptive95% intervals. PRMScore/pooled AUC are point estimates. PRMScore q=.8 is recalculated on other folds. See METRICS.json and PB_CELLS.csv for all thresholds and cell denominators.

PB Q4:36.9818 ->36.4582; Q8:35.5606 ->34.7092. Six cells decline, one improves, one ties. Position gains65 gated exact successes and loses104;63 losses are earlier,41 later. No gate change and no invalid scores. Both true-early and true-late error groups lose net successes (32/7). Against the shuffled control, gains81/losses81 and PB contrast includes zero; within-answer AUC is lower by.001473 (95%CI[-.002501,-.000467]).

All41,307 saved arm states pass separate score and normalized-mixture likelihood replay. One single-step answer is an explicit identity in each arm, so41,304 optimizations actually run; all converge. Median position correction norm / base norm is4.747%, with481 PB peak changes. Shared correction is about1e-6 relative to base and changes no PB peaks. Median penalized objective improvement is.01852; on gained cases .01383 and lost cases .01633. Improving the density objective does not select task-beneficial changes in this experiment.

On the same1,672 PRMB answers eligible for the prior early/late marginal AUC comparison, the selected-surprisal-minus-entropy coefficient difference increases late in only19.56% of cases (median early-to-late change-.1451). This is descriptive, not a feature-importance proof: the powers are correlated, and marginal AUC is not the coefficient a joint model should necessarily assign. The mismatch motivates testing JOINT predictive value before redesigning the unlabeled objective.

Integrity amendment: the first run, freeze b5614775e, stopped on three original PRMB records with shared boundary tokens. It retains8,112 completed rows in results/rbm_position_fusion_v1 and produced no benchmark metrics. The corrected run, freeze258393249, restarted from zero in this directory. Context ownership follows the latest step start; original scoring spans and labels are unchanged. All81,120 saved arrays from the8,112 previously completed answers replay exactly.30-case/90-state feasibility audit passed; full35-arm metric review and41,307-state independent arithmetic audit passed. These are separate implementations within the same session, not external scientific replication.

Next: retain static references. Discuss a small, clearly labeled supervised DIAGNOSTIC of whether position adds joint predictive information under source-group-disjoint evaluation; it would not replace or validate the answer-only method. No such diagnostic or new model has been launched. Do not interpret this negative constrained implementation as rejecting every conditional-fusion family. All findings remain development evidence. No HTML.
