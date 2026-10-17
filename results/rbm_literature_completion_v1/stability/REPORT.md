# stability: full development result

All 13,769 answers scored; saved-state and separate metric review PASS.
Fusion fits each answer without labels. The entropy gate and PRMScore calibration use external folds.
Posterior and logit scores use the same Top10/argmax; no first_near_max.

| Method | PB macro % (full population) | PB macro % (covered answers) | PRMB within AUC | PRMScore | PRMScore conditional | Valid answers |
|---|---:|---:|---:|---:|---:|---:|
| RBM, 1 hidden unit, best density fit of 3 starts, 12 features, logit | 36.232532 | 36.232532 | 0.745024 | 0.622175 | 0.622175 | 13769 |
| RBM, 1 hidden unit, best density fit of 3 starts, 12 features, posterior | 36.344519 | 36.344519 | 0.738478 | 0.629354 | 0.629354 | 13769 |
| RBM, 4 hidden units, best density fit of 3 starts, 12 features, logit | 27.326199 | 27.326199 | 0.721817 | 0.612483 | 0.612483 | 13769 |
| RBM, 4 hidden units, best density fit of 3 starts, 12 features, posterior | 26.653991 | 26.653991 | 0.714019 | 0.604969 | 0.604969 | 13769 |
| RBM, 1 hidden unit, exact training, 12 features, logit | 36.271231 | 36.271231 | 0.745204 | 0.622215 | 0.622215 | 13769 |
| RBM, 1 hidden unit, exact training, 12 features, posterior | 36.375047 | 36.375047 | 0.738702 | 0.629276 | 0.629276 | 13769 |
| RBM, 4 hidden units, exact training, 12 features, logit | 28.655321 | 28.655321 | 0.721693 | 0.601725 | 0.601725 | 13769 |
| RBM, 4 hidden units, exact training, 12 features, posterior | 27.836583 | 27.836583 | 0.715044 | 0.588855 | 0.588855 | 13769 |
| RBM, 1 hidden unit, best density fit of 3 starts, 6 features, logit | 34.970784 | 34.970784 | 0.735636 | 0.622273 | 0.622273 | 13769 |
| RBM, 1 hidden unit, best density fit of 3 starts, 6 features, posterior | 36.201684 | 36.201684 | 0.735954 | 0.630789 | 0.630789 | 13769 |
| RBM, 4 hidden units, best density fit of 3 starts, 6 features, logit | 24.201531 | 24.201531 | 0.711629 | 0.616449 | 0.616449 | 13769 |
| RBM, 4 hidden units, best density fit of 3 starts, 6 features, posterior | 24.706066 | 24.706066 | 0.708662 | 0.618526 | 0.618526 | 13769 |
| RBM, 1 hidden unit, exact training, 6 features, logit | 34.970784 | 34.970784 | 0.735636 | 0.622233 | 0.622233 | 13769 |
| RBM, 1 hidden unit, exact training, 6 features, posterior | 36.201684 | 36.201684 | 0.735982 | 0.630749 | 0.630749 | 13769 |
| RBM, 4 hidden units, exact training, 6 features, logit | 24.646095 | 24.646095 | 0.712608 | 0.614358 | 0.614358 | 13769 |
| RBM, 4 hidden units, exact training, 6 features, posterior | 25.320459 | 25.320459 | 0.710100 | 0.616757 | 0.616757 | 13769 |
| RBM6 with shared diagonal variance, previous experiment | 35.806949 | 35.806949 | 0.733045 | 0.625301 | 0.625301 | 13769 |
| Token entropy, saved reference | 35.444377 | 35.444377 | 0.730111 | 0.625426 | 0.625426 | 13769 |
| RBM12, before training, posterior | 36.294594 | 36.294594 | 0.746325 | 0.605313 | 0.605313 | 13769 |
| RBM6, before training, posterior | 35.966210 | 35.966210 | 0.744068 | 0.613085 | 0.613085 | 13769 |
| RBM12, trained, Logit | 36.271231 | 36.271231 | 0.745204 | 0.622215 | 0.622215 | 13769 |
| RBM12, trained, posterior | 36.375047 | 36.375047 | 0.738702 | 0.629276 | 0.629276 | 13769 |
| RBM6, trained, Logit | 34.970784 | 34.970784 | 0.735636 | 0.622233 | 0.622233 | 13769 |
| RBM6, trained, posterior | 36.201684 | 36.201684 | 0.735982 | 0.630749 | 0.630749 | 13769 |
| RBM6 with weight shrinkage, posterior | 35.871143 | 35.871143 | 0.738630 | 0.629544 | 0.629544 | 13769 |
| Varentropy, top 15 probabilities | 35.960987 | 35.960987 | 0.737786 | 0.625781 | 0.625781 | 13769 |
| Varentropy15 contributions, equal fusion | 35.598044 | 35.598044 | 0.746980 | 0.612812 | 0.612812 | 13769 |
| Varentropy15 contributions, IU-PCR fusion | 35.349836 | 35.349836 | 0.746824 | 0.622689 | 0.622689 | 13769 |
| Varentropy, top 50 probabilities | 35.675524 | 35.675524 | 0.742465 | 0.632777 | 0.632777 | 13769 |

Primary planned contrasts (97.5% source-group intervals; 10,000 draws):
- RBM, 4 hidden units, best density fit of 3 starts, 12 features, logit minus RBM, 4 hidden units, exact training, 12 features, logit: PB -1.329122 pp, CI [-2.1655121496878396, -0.5144902124786155]; within 0.00012370725073840262, CI [-0.0016237603076788775, 0.001934042843527759]; gained/lost 79/133.
- RBM, 4 hidden units, best density fit of 3 starts, 6 features, posterior minus RBM, 4 hidden units, exact training, 6 features, posterior: PB -0.614393 pp, CI [-1.322951953751537, 0.07836518612743675]; within -0.0014378590904460583, CI [-0.0027385497016206625, -0.00015997113287633926]; gained/lost 44/71.

No automatic promotion. Other comparisons are descriptive95%; intervals do not cover all prior research choices.
See FIT_HEALTH.json for convergence, failures and method details; PB_CELLS.csv and CHANGED_SUCCESSES.csv preserve case-level differences.
