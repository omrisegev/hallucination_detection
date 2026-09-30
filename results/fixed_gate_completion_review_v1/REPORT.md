# Independent review of the fixed entropy gate

**PASS.** All6,800 PB model-answer rows, six evaluated arms, all entropy thresholds and metrics replay.

| IU gate | PB all8 |
|---|---:|
| gmm_bic_saved | 20.0040% |
| entropy_mean / nested_labels | 31.3129% |
| entropy_mean / quantile_0.3 | 31.1580% |

Quantile0.3 minus GMM: **11.1540 pp**, 95% CI [9.2911, 13.0157] pp, 10,000 canonical-group draws.

Intervals condition on fixed predictions. They exclude calibration refits and detector/q selection uncertainty.

| Transfer | transferred | nested, same destination |
|---|---:|---:|
| entropy_mean / transfer_easy_to_hard | 28.9864% | 29.0339% |
| entropy_mean / transfer_hard_to_easy | 32.8907% | 33.5920% |

Quantile calibration uses other answers. Fusion remains answer-local. This is development evidence, not untouched confirmation.

Source reports are preserved. Detailed source hashes, threshold ranges, cell denominators and audit checks are in REVIEW.json.
