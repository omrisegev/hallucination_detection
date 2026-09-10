# Selected-token and tail probability fusion v2

Gray-box only. 15 probability ranks + selected-token surprisal + residual tail; top-10 is the separate token readout.

No automatic result threshold is applied. This run measures whether the two
added inputs carry useful signal.

## One-answer localization

| method | PB all8 | PB Q4 | PB Q8 | PRMB within | PRMScore | fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Token Entropy | 35.44% | 36.40% | 34.49% | 0.7301 | 0.6254 | 0 |
| Selected + Tail Probability Fusion - Equal Weights | 34.56% | 35.32% | 33.81% | 0.7339 | 0.6207 | 0 |
| Selected + Tail Probability Fusion - IU-PCR | 34.50% | 35.31% | 33.69% | 0.7328 | 0.6205 | 0 |
| Selected + Tail Probability Fusion - Joint Shrinkage | 34.59% | 35.18% | 34.01% | 0.7297 | 0.6198 | 0 |
| Mind the Gap Locator - Common Gate | 26.54% | 27.37% | 25.72% | n/a | n/a | 0 |

Primary paired contrasts (97.5% intervals):

- `augmented_iu_minus_entropy`: PB -0.94 pp [-1.94, +0.03]; PRMB within +0.0026 [-0.0012, +0.0065].
- `augmented_iu_minus_mindgap_paper_locator_common_gate`: PB +7.96 pp [+5.99, +9.97].
- `augmented_joint_lw_minus_entropy`: PB -0.85 pp [-1.79, +0.06]; PRMB within -0.0004 [-0.0040, +0.0032].
- `augmented_joint_lw_minus_mindgap_paper_locator_common_gate`: PB +8.05 pp [+6.11, +10.01].

## Complete-answer detection: historical 24 cells

| method | all24 macro AUROC | QA9 | math15 |
|---|---:|---:|---:|
| Token Entropy | 0.7717 | 0.7519 | 0.7836 |
| Selected + Tail Probability Fusion - Equal Weights | 0.7781 | 0.7570 | 0.7908 |
| Selected + Tail Probability Fusion - IU-PCR | 0.7688 | 0.7509 | 0.7796 |
| Selected + Tail Probability Fusion - Joint Shrinkage | 0.7285 | 0.6821 | 0.7564 |
| Token Varentropy | 0.7689 | 0.7504 | 0.7801 |
| Historical IU-PCR | 0.7761 | 0.7597 | 0.7859 |

Paired cell-level contrasts:

- `augmented_joint_lw_minus_entropy`: -0.0432 [-0.0839, -0.0126] (paired-cell CI 95%), 6W/0T/18L.
- `augmented_iu_minus_entropy`: -0.0029 [-0.0120, +0.0046] (paired-cell CI 95%), 14W/0T/10L.
- `augmented_joint_lw_minus_augmented_iu`: -0.0403 [-0.0763, -0.0148] (paired-cell CI 95%), 6W/0T/18L.
- `augmented_iu_minus_historical_iu_pcr`: -0.0073 [-0.0213, +0.0036] (hierarchical group CI 97.5%), 11W/0T/13L.
- `augmented_joint_lw_minus_historical_iu_pcr`: -0.0475 [-0.0885, -0.0170] (paired-cell CI 95%), 6W/0T/18L.

Same-method v2 versus v1 comparisons (97.5% intervals):

- IU-PCR: +0.0057 [+0.0015, +0.0118], 20W/0T/4L.
- Equal Weights: +0.0009 [+0.0002, +0.0017], 16W/0T/8L.
- Joint Shrinkage: -0.0022 [-0.0269, +0.0128], 17W/1T/6L.

Exploratory comparisons added after viewing the full table:

- Equal Weights versus Historical IU-PCR: +0.0020 [-0.0031, +0.0073], 14W/0T/10L.
- Equal Weights versus Token Entropy: +0.0064 [+0.0026, +0.0107], 18W/0T/6L.

The added inputs do not improve one-answer localization. They do carry a small
complete-answer signal, but the simple Equal Weights route uses it better than
the current IU-PCR and Joint Shrinkage routes. The combined run cannot attribute
the signal separately to selected-token surprisal or residual tail mass.

## Interpretation boundary

DEEM is not part of this primary run. Its soft input is a probability over the
target class from each base learner, while these columns are vocabulary-rank and
token-probability coordinates. The small answer-level signal first justifies a
selected-only versus tail-only ablation under Equal Weights. A direct-rank DEEM
adapter remains a later nonlinear experiment.
