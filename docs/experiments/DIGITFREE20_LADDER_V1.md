# Digit-free 20-stream ladder v1 (Step 413 [Claude], 2026-09-17)

Development only, frozen 13,769-answer population (v3 labels / v2 source groups), no untouched
confirmation. Runner `scripts/run_digitfree20_ladder_v1.py`; results `results/digitfree20_ladder_v1/RUN.json`
(OOF scores and per-fold weights/partitions in the ignored npz/json files, regenerable in ~1 minute).

## Question

After the Step 399 addendum showed that the digit-era leading Joint arm reduces to "auto K=3 partition +
block averaging", Omri asked whether the partition carries anything once the digit streams are gone. Codex
(other machine) proposed a 20-stream digit-free bank and the recipe "auto clustering, average within
cluster, Continuous L-SML across clusters, cluster-equal as mandatory control". This ladder runs that
recipe and its alternatives at EVERY admissible K under BOTH label-free grouping rules, so that nothing
is hidden behind a K-selection tie-break.

## Contract (identical to `results/digitfree_broad50_v1`)

* Bank: Codex's 20 streams (`rank_1_risk, rank_3_risk, surprisal, top1_loggap, censored_rank50,
  mass_above, top2_ratio, H0lim, Renyi a0.25, H1, VE0, VE0.75, VE1, four prefix innovations
  (H0lim, a0.25, VE0.75, VE1), top15_turnover, top50_truncated_js, BOCPD residual`). Values come from
  Codex's digit-free broad50 extraction plus the pure BOCPD residual channel of
  `joint_feature_selection_bocpd_v1`. Per-stream Top10 step readout, answer-local standardization.
* Five outer source folds; every weight, group and Joint fit uses the pooled training steps of the four
  other folds; scores are out-of-fold. Hybrid pooled fitting, not answer-only.
* Gate: frozen non-digit tail15 answer prominence, within-cell midrank >= .33. Sign anchor: H1 (probability
  Shannon entropy), Spearman orientation of the fused score. No digit stream, gate or anchor anywhere.
* Metrics: ProcessBench eight-cell macro-F1 (gated) and PRMBench mean within-answer AUROC (ungated);
  paired source-group bootstrap, 10,000 draws, 95% intervals.

## Arms

Two grouping rules, each at K = 3..8 with no selection: `affinity` (Joint's residual-affinity
|C - vv^T| spectral rule; the stability tie-break's own choice is reported) and `eq15` (Continuous
L-SML's score-matrix rule; its Eq.14 residual choice is reported). On every partition:

| readout | within group | across groups |
|---|---|---|
| ceq | mean of standardized streams | equal weight per standardized virtual |
| csml | mean | SML eigen-solve at every K (real solve at K=3) |
| csml_guard | mean | SML with the Step-205 guard (Codex's proposal verbatim) |
| lsml | SML eigenvector | SML (Continuous L-SML on the given groups) |
| jrel | mean | Joint factor model: v_g / (u_g^2 + s_g^2/n_g) from the fitted loadings |

Global rows: `equal20`, `H1` singleton, `iu` (IU-PCR, frozen defaults), `continuous_auto` (Continuous
L-SML with its own K). Comparators under the same gate: Codex's broad50 arms and the historical
innovation5 / BOCPD-corrected innovation5 locators.

## Result

Selected K: the stability rule picks K=3 in all five folds (every K from 3 to 8 has ARI 1.0 under block
deletion, so the smaller-K tie-break decides again); the Eq.14 residual rule picks K=6 in all five folds.
At K=6 and K=7 the two rules return the identical partition. Partitions are clean stream families
(probability-rank block, surprisal block, entropy-shape block, varentropy block, innovation pair(s),
turnover/JS/BOCPD trio).

PB macro-F1 % / PRMB within-AUC, out of fold:

| arm | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 |
|---|---|---|---|---|---|---|
| affinity ceq | 39.93 / .7494 | 39.94 / .7540 | **40.04 / .7550** | 39.86 / .7562 | 39.67 / .7570 | 39.58 / .7570 |
| affinity csml | 39.42 / .7536 | 39.68 / .7562 | 39.82 / .7557 | 39.60 / .7574 | 39.77 / .7573 | 39.58 / .7574 |
| affinity lsml | 39.95 / .7492 | 39.37 / .7560 | 39.28 / .7555 | 39.21 / .7550 | 39.27 / .7552 | 39.33 / .7565 |
| affinity jrel | 39.29 / .7516 | 38.94 / .7500 | 39.74 / .7566 | 39.49 / .7569 | 39.02 / .7570 | 39.12 / .7555 |
| eq15 ceq | 36.82 / .7431 | 39.03 / .7484 | 39.46 / .7534 | 39.86 / .7562 | 39.67 / .7570 | 39.53 / .7571 |
| eq15 csml | 36.29 / .7473 | 38.93 / .7505 | 39.59 / .7552 | 39.60 / .7574 | 39.77 / .7573 | 39.54 / .7582 |
| eq15 lsml | 36.69 / .7427 | 38.95 / .7537 | 39.23 / .7550 | 39.21 / .7550 | 39.27 / .7552 | 39.45 / .7561 |
| eq15 jrel | 39.17 / .7537 | 39.32 / .7574 | 39.65 / .7572 | 39.49 / .7569 | 39.02 / .7570 | 39.53 / .7571 (fallback) |

| global row | PB | within |
|---|---|---|
| equal20 (no partition) | 39.19 | .7558 |
| continuous_auto (K=6) | 39.21 | .7550 |
| iu | 39.47 | .7555 |
| H1 alone | 36.35 | .7301 |
| Codex broad50: equal50 / continuous / joint / joint_balanced | 37.90 / 37.76 / 38.05 / 37.99 | .7489 / .7479 / .7461 / .7460 |
| Codex broad50+BOCPD: joint_auto / equal / continuous | 38.60 / 37.82 / 38.01 | .7491 / .7496 / .7482 |
| historical innovation5 (single stream) | 39.83 | .7603 |
| historical BOCPD-corrected innovation5 | **40.37** | **.7632** |

`csml_guard` equals `ceq` at K=3 and `csml` at K>=4 by construction. jrel fell back to ceq (declared)
in the 5 folds where eq15's K=8 partition has a singleton and in 2 folds of affinity K=8.

Paired contrasts (PB pp [95%], within [95%]); * = interval excludes zero:

* Partition + block-equal minus equal20: affinity K5 +0.84 [+0.27, +1.44]* / -.0008 [-.0023, +.0007];
  K6 +0.66 [+0.19, +1.15]* / +.0004; K3 +0.74 [+0.01, +1.48]* / **-.0064 [-.0086, -.0043]***;
  K7 +0.48 [-0.03, +0.99] / +.0012*; eq15 K3 -2.37* / -.0127*. continuous_auto +0.02 / -.0008; iu +0.28 / -.0003.
* Cross-group SML minus block-equal on the same partition: never positive on PB with an interval
  excluding zero at any K under either rule; at K=3 the real eigen-solve costs -0.51 [-1.04, -0.01]*
  (affinity) and -0.54 [-0.93, -0.16]* (eq15). Within gains are +.001 to +.004.
* Continuous L-SML (within SML too) minus block-equal: -0.76 [-1.40, -0.14]* at affinity K5,
  -0.64 [-1.19, -0.11]* at K6; elsewhere intervals include zero, points negative.
* Joint reliability weights (jrel) minus block-equal: negative or zero on every affinity partition
  (K4 -1.00 [-2.15, +0.17], within -.0040*); the only positive contrast is on eq15's K=3 partition
  (+2.35 [+0.77, +3.93]*), where jrel repairs a bad 13-stream blob to 39.17, still below affinity K3's
  block-equal 39.93. The model helps only where the partition is poor, and not beyond a good partition.
* Rule against rule at fixed K (block-equal): affinity beats eq15 at K3 (+3.11*) and K4 (+0.91*),
  identical from K6.

## Reading

1. The entire ladder sits in a 1.1-point band (39.0-40.0 PB, .749-.758 within). Removing digit
   removed the thing the K=3 partition was paying for: the best partition now buys +0.8pp over the
   plain 20-stream mean on PB and nothing on within-answer ranking.
2. No cross-group weighting rule (SML at any K, Continuous L-SML, Joint reliability) beats equal
   weight per block on this bank. Codex's proposed recipe (mean within, Continuous L-SML across,
   guard on) is, at its own selected K=6, 39.60 versus 39.86 for cluster-equal.
3. Both K-selection rules are again non-choices: stability is perfectly flat (ARI 1.0 for every K), and
   the Eq.14 residual curve is nearly flat (131.6-136.3), landing on K=6 where both rules coincide.
4. The 20-stream bank at equal weight (39.19) already beats every Codex broad50 arm (<= 38.60) under
   the same gate, so the bank composition matters more than any fusion rule. But nothing in this
   ladder reaches the historical single-stream innovation5 with the BOCPD correction (40.37 / .7632),
   and every ladder arm is below plain innovation5 on within-answer AUROC (.7603).
5. Consequence for the method: on a digit-free bank the "fusion core" (partition, SML, Joint model) has
   no measurable advantage over averaging 20 standardized streams, and averaging is not a method
   (Omri). The loss is not in the weights; the remaining lever is the streams and the readout
   (innovation5 + BOCPD is one stream plus one predictor), or a different fusion objective altogether.

No arm is promoted. Development evidence, hybrid pooled fitting, gate developed on these data.
