# Frozen RBM12 hierarchical time fusion

Full development population: 13,769 answers. Fixed RBM12 coefficients and entropy gate. No first_near_max.
Other-answer time weights use source-group exclusion; PRMScore uses nested calibration. No untouched confirmation.

| Method | ProcessBench % | PRMB within AUC | PRMScore | Valid answers |
|---|---:|---:|---:|---:|
| Frozen RBM12: Top10 mean | 36.271 | 0.74520 | 0.62222 | 13769 |
| Frozen RBM12: all-token mean | 29.429 | 0.66050 | 0.56348 | 13769 |
| Frozen RBM12: best consecutive 10 | 31.653 | 0.70476 | 0.60425 | 13769 |
| Time fusion: current answer | 28.142 | 0.64238 | 0.57764 | 13769 |
| Time fusion: other answers | 29.728 | 0.66991 | 0.57134 | 13769 |
| Time fusion: shared + local | 29.385 | 0.66454 | 0.57272 | 13769 |
| Time fusion: shuffled tokens | 29.190 | 0.65985 | 0.56285 | 13769 |
| Supervised time weights | 30.242 | 0.68101 | 0.59483 | 13769 |
| Top10: remove first token | 36.360 | 0.74387 | 0.62747 | 13769 |
| Hierarchy: remove first token | 29.546 | 0.66687 | 0.57507 | 13769 |

Primary contrasts (paired source groups; 10,000 draws; 97.5% intervals):

- Time fusion: other answers minus Time fusion: current answer: PB delta 1.586 pp, CI [0.044, 3.115]; within-AUC delta 0.02753, CI [0.022897578675740767, 0.032038523722473365].
- Time fusion: shared + local minus Time fusion: other answers: PB delta -0.343 pp, CI [-1.26, 0.557]; within-AUC delta -0.00537, CI [-0.00739746925516824, -0.0033008456508337248].

See COMPARISON.csv for historical references and per-cell metrics in METRICS.json.
WEIGHT_SUMMARY.json records temporal weights, adaptation distance, covariance misfit and conditioning.
ANSWER_DIAGNOSTICS.csv and ERROR_TRANSITIONS.json retain exact peaks, shifts and exclusive losses.
FLAGGED_FITS.json lists numerical failures and finite nonconverged fits. No method is silently substituted.

## Interpretation

Primary, 10,000 paired source-group draws, 97.5% CI: shared-local PB +1.586 pp [0.044, 3.115], within-AUC +0.02753 [0.02290, 0.03204], but PRMScore -0.00631 [-0.01013, -0.00251]. Hierarchical-shared PB -0.343 pp [-1.260, 0.557], within-AUC -0.00537 [-0.00740, -0.00330]; PRMScore +0.00138 has an interval including zero. Sharing data helps relative to local covariance fitting on two endpoints, not across every metric. Local adaptation supplies no overall gain.

The hierarchy loses to original Top10 in every PB cell: it gains 281 exact successes and loses 608 (269 early, 339 late; zero gate or computation losses). Top10-to-all-mean alone loses 6.842 PB points before fitting any time weights. This isolates a substantial loss from replacing score-adaptive selection with averaging. Supervised fixed positive time weights do not recover it; this diagnostic is not a supervised ceiling for other architectures.

Average shared mass over successive step quarters: 12.6%, 21.5%, 29.7%, 36.2% (uniform: 25% each). Hierarchy: 12.7%, 21.9%, 29.4%, 36.1%. The median answer-level L1 change from shared weights is 0.258, despite similar aggregate profiles. Shuffling yields nearly uniform quarters. Descriptive original-order versus shuffled hierarchy within-AUC +0.00470 [0.00238, 0.00700], PB +0.195 pp [-0.972, 1.348]: evidence for modest within-answer ordering information, not a PB gain.

Boundary diagnostics are not promoted: Top10 without the first token gives PB 36.360% (difference not clear), within-AUC 0.74387 (slightly worse), PRMScore 0.62747 (better). Removing the first token from hierarchy gives 29.546% PB; it does not close the gap. Regions do not create observations: 22,333 of 145,597 steps have fewer than 16 tokens; 91 have one token. Median step length is 31 tokens; median answer has eight steps and alpha=0.304.

Decision: retain original RBM12 Logit + Top10 as this temporal experiment's reference. Do not adopt the tested positive fixed-position averaging hierarchy. More data stabilizes temporal weights but cannot by itself recover information discarded by the readout. A next family should preserve score-adaptive high-risk token selection while isolating added temporal/context information. No tensor, conditional RBM, convolution or new feature experiment was started; discussion is required before the next family. Findings remain development evidence, not untouched confirmation. Historical low-correlation RBM6 and Varentropy50 remain explicit competing references, not replaced by this run.

