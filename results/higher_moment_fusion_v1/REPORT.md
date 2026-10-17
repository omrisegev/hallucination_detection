# Compact moment orders 3 through 6

Full matched localization development: 13,769 model-answer rows. Code freeze e17ada312. Metric arithmetic review PASS; saved-state audit PASS (220,304 model records; independent replay on27 real answers). No new HTML.

Each answer supplies T token rows. Original bank is H15, V15, m3, a, a^2, a^3. Append m4/a^4, m5/a^5, m6/a^6: 6/8/10/12 columns. Here mr=sum(q*(-log(q+1e-12))^r), using frozen top15-normalized probabilities; a is selected-token surprisal. V is retained variance, not raw second moment. Both input families change together.

All orders keep the original six-column mean orientation, answer-only standardization and fitting, external entropy q0.3 gate, Top10 token mean, v3 labels and canonical folds. RBM uses original exact unregularized likelihood with one hidden unit. IU-PCR is the historical unchanged implementation. No shrinkage, feature selection, new window or graph.

| Order | Model | PB macro % | PRMB within AUC | PRMScore |
|---|---|---:|---:|---:|
| 3 | Equal mean | 21.7316 | 0.706096 | 0.581753 |
| 3 | IU-PCR | 22.3788 | 0.708585 | 0.581905 |
| 3 | RBM before learning | 35.9662 | 0.744068 | 0.613085 |
| 3 | Trained RBM | 36.2017 | 0.735982 | 0.630749 |
| 4 | Equal mean | 20.9435 | 0.700362 | 0.577791 |
| 4 | IU-PCR | 21.6282 | 0.704043 | 0.583652 |
| 4 | RBM before learning | 36.0658 | 0.743549 | 0.612477 |
| 4 | Trained RBM | 36.1734 | 0.734940 | 0.628874 |
| 5 | Equal mean | 20.5919 | 0.695234 | 0.573074 |
| 5 | IU-PCR | 20.7651 | 0.698812 | 0.579859 |
| 5 | RBM before learning | 36.3194 | 0.744350 | 0.610077 |
| 5 | Trained RBM | 36.1553 | 0.734705 | 0.628851 |
| 6 | Equal mean | 20.1812 | 0.689104 | 0.567796 |
| 6 | IU-PCR | 20.2046 | 0.690603 | 0.570694 |
| 6 | RBM before learning | 36.2946 | 0.746325 | 0.605313 |
| 6 | Trained RBM | 36.3750 | 0.738702 | 0.629276 |

Primary paired contrasts: 10,000 canonical-source bootstrap draws, 97.5% intervals.

- RBM order6 minus order3: PB +0.1734 pp [-0.6183, +1.0193]; within AUC +0.002720 [-0.000118, +0.005683].
- IU order6 minus order3: PB -2.1742 pp [-3.1026, -1.3234]; within AUC -0.017981 [-0.020235, -0.015709].

RBM degree6 has slightly higher PB and within-AUC points, but neither primary interval excludes zero. IU-PCR declines on both primary endpoints. RBM degree6 still ranks worse than its untrained same-bank control (within delta -0.007623, exploratory95% interval [-0.009755,-0.005421]); PB delta +0.0805 pp is inconclusive. The new representation does not establish a gain from learning. Cross-solver comparisons also include sigmoid-before-Top10 for RBM, so they do not isolate the weight-learning rule.

| Frozen reference | PB macro % | Within AUC | PRMScore |
|---|---:|---:|---:|
| Entropy | 35.4444 | 0.730111 | 0.625426 |
| Varentropy15 | 35.9610 | 0.737786 | 0.625781 |
| Varentropy50 | 35.6755 | 0.742465 | 0.632777 |
| Varentropy15 contribution mean | 35.5980 | 0.746980 | 0.612812 |
| Longest step + entropy gate | 33.6944 | 0.618136 | 0.527878 |
| Random step + entropy gate | 20.2750 | 0.498480 | 0.497941 |

All16 arms have full coverage and no failed/collapsed outputs. Original order3 RBM has one finite nonconverged fit retained; all higher-order RBM fits converge. All4 order3 methods replay historical step arrays exactly. All37 method/reference metrics pass separate arithmetic verification, including held-group PRMScore thresholds and full PB failure denominators.

Interpretation: retain degree3 as the compact reference; degree6 is not a validated new winner. Orders4/5 and comparator contrasts are exploratory. This is seen development data, no untouched confirmation or global24 result. Prior Mind-the-Gap and historical-reference scope remain recorded in METRICS.json; this experiment does not finish missing historical refits. No follow-up sweep launched.

Per-cell results: SUMMARY.csv. Full intervals: METRICS.json. Prediction changes: ERROR_CASES.json. Saved states: CHECKPOINT.sqlite. Scores: SCORES.npz.
