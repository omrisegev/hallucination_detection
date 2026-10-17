# Direct probability-rank fusion v1

Gray-box only. K=15 vocabulary ranks; top-10 is the separate token readout.

## One-answer localization

| method | PB all8 | PB Q4 | PB Q8 | PRMB within | PRMScore | fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Token Entropy | 35.44% | 36.40% | 34.49% | 0.7301 | 0.6254 | 0 |
| Direct Probability Fusion - Equal Weights | 34.82% | 35.60% | 34.04% | 0.7333 | 0.6195 | 0 |
| Direct Probability Fusion - IU-PCR | 34.75% | 35.48% | 34.02% | 0.7327 | 0.6202 | 0 |
| Direct Probability Fusion - Joint Shrinkage | 34.71% | 35.52% | 33.90% | 0.7324 | 0.6215 | 0 |
| Mind the Gap Locator - Common Gate | 26.54% | 27.37% | 25.72% | n/a | n/a | 0 |

Primary paired contrasts (97.5% intervals):

- `rank_iu_minus_entropy`: PB -0.69 pp [-1.62, +0.26]; PRMB within +0.0026 [-0.0014, +0.0065].
- `rank_iu_minus_mindgap_paper_locator_common_gate`: PB +8.21 pp [+6.25, +10.17].
- `rank_joint_lw_minus_entropy`: PB -0.73 pp [-1.66, +0.21]; PRMB within +0.0022 [-0.0020, +0.0064].
- `rank_joint_lw_minus_mindgap_paper_locator_common_gate`: PB +8.17 pp [+6.20, +10.14].

## Complete-answer detection: historical 24 cells

| method | all24 macro AUROC | QA9 | math15 |
|---|---:|---:|---:|
| Token Entropy | 0.7717 | 0.7519 | 0.7836 |
| Direct Probability Fusion - Equal Weights | 0.7772 | 0.7562 | 0.7898 |
| Direct Probability Fusion - IU-PCR | 0.7631 | 0.7438 | 0.7747 |
| Direct Probability Fusion - Joint Shrinkage | 0.7307 | 0.6913 | 0.7544 |
| Token Varentropy | 0.7689 | 0.7504 | 0.7801 |
| Historical IU-PCR | 0.7761 | 0.7597 | 0.7859 |

Paired cell-level contrasts:

- `rank_joint_lw_minus_entropy`: -0.0410 [-0.0701, -0.0184] (paired-cell CI 95%), 4W/0T/20L.
- `rank_iu_minus_entropy`: -0.0086 [-0.0214, +0.0010] (paired-cell CI 95%), 7W/0T/17L.
- `rank_joint_lw_minus_rank_iu`: -0.0324 [-0.0520, -0.0163] (paired-cell CI 95%), 3W/0T/21L.
- `rank_iu_minus_historical_iu_pcr`: -0.0130 [-0.0314, +0.0002] (hierarchical group CI 97.5%), 9W/0T/15L.
- `rank_joint_lw_minus_historical_iu_pcr`: -0.0453 [-0.0742, -0.0227] (paired-cell CI 95%), 3W/0T/21L.

## Interpretation boundary

DEEM is not part of this primary run. Its soft input is a probability over the target class from each base learner, while these columns are alternative vocabulary ranks. A direct-rank DEEM adapter is a separate nonlinear experiment and should be attempted only after this run establishes that the rank representation itself carries useful signal.
