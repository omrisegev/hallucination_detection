# depth_amended: full development result

All 13,769 answers scored; saved-state and separate metric review PASS.
Fusion fits each answer without labels. The entropy gate and PRMScore calibration use external folds.
Posterior and logit scores use the same Top10/argmax; no first_near_max.

| Method | PB macro % (full population) | PB macro % (covered answers) | PRMB within AUC | PRMScore | PRMScore conditional | Valid answers |
|---|---:|---:|---:|---:|---:|---:|
| RBM, 1 hidden unit, exact training, 12 features, logit | 36.271231 | 36.271231 | 0.745204 | 0.622215 | 0.622215 | 13769 |
| RBM, 1 hidden unit, exact training, 12 features, posterior | 36.375047 | 36.375047 | 0.738702 | 0.629276 | 0.629276 | 13769 |
| RBM, 4 hidden units, exact training, 12 features, logit | 28.655321 | 28.655321 | 0.721693 | 0.601725 | 0.601725 | 13769 |
| RBM, 4 hidden units, exact training, 12 features, posterior | 27.836583 | 27.836583 | 0.715044 | 0.588855 | 0.588855 | 13769 |
| Stacked RBMs, 4 to 1 hidden units, CD-10 second layer, 12 features, logit | 20.307947 | 21.126738 | 0.656764 | NA | 0.536128 | 13416 |
| Stacked RBMs, 4 to 1 hidden units, CD-10 second layer, 12 features, posterior | 29.143856 | 30.352457 | 0.722602 | NA | 0.576025 | 13416 |
| Stacked RBMs, 4 to 1 hidden units, exact second layer, 12 features, logit | 22.096522 | 22.964542 | 0.579489 | NA | 0.486925 | 13416 |
| Stacked RBMs, 4 to 1 hidden units, exact second layer, 12 features, posterior | 22.782261 | 23.676280 | 0.572059 | NA | 0.486190 | 13416 |
| Stacked RBMs, 4 to 1 hidden units, CD-10 second layer on hidden logits (amendment), 12 features, logit | 22.974005 | 22.974005 | 0.673483 | 0.543016 | 0.543016 | 13769 |
| Stacked RBMs, 4 to 1 hidden units, CD-10 second layer on hidden logits (amendment), 12 features, posterior | 34.681214 | 34.681214 | 0.734350 | 0.587323 | 0.587323 | 13769 |
| Stacked RBMs, 4 to 1 hidden units, exact second layer on hidden logits (amendment), 12 features, logit | 32.776625 | 32.776625 | 0.715080 | 0.595517 | 0.595517 | 13769 |
| Stacked RBMs, 4 to 1 hidden units, exact second layer on hidden logits (amendment), 12 features, posterior | 34.500548 | 34.500548 | 0.708580 | 0.598028 | 0.598028 | 13769 |
| RBM, 1 hidden unit, exact training, 6 features, logit | 34.970784 | 34.970784 | 0.735636 | 0.622233 | 0.622233 | 13769 |
| RBM, 1 hidden unit, exact training, 6 features, posterior | 36.201684 | 36.201684 | 0.735982 | 0.630749 | 0.630749 | 13769 |
| RBM, 4 hidden units, exact training, 6 features, logit | 24.646095 | 24.646095 | 0.712608 | 0.614358 | 0.614358 | 13769 |
| RBM, 4 hidden units, exact training, 6 features, posterior | 25.320459 | 25.320459 | 0.710100 | 0.616757 | 0.616757 | 13769 |
| Stacked RBMs, 4 to 1 hidden units, CD-10 second layer, 6 features, logit | 18.884452 | 19.252861 | 0.651676 | NA | 0.552084 | 13629 |
| Stacked RBMs, 4 to 1 hidden units, CD-10 second layer, 6 features, posterior | 28.157072 | 28.762341 | 0.726867 | NA | 0.599428 | 13629 |
| Stacked RBMs, 4 to 1 hidden units, exact second layer, 6 features, logit | 18.977294 | 19.344716 | 0.634470 | NA | 0.479050 | 13629 |
| Stacked RBMs, 4 to 1 hidden units, exact second layer, 6 features, posterior | 19.270553 | 19.642289 | 0.621533 | NA | 0.467270 | 13629 |
| Stacked RBMs, 4 to 1 hidden units, CD-10 second layer on hidden logits (amendment), 6 features, logit | 19.602355 | 19.602355 | 0.671648 | 0.569531 | 0.569531 | 13769 |
| Stacked RBMs, 4 to 1 hidden units, CD-10 second layer on hidden logits (amendment), 6 features, posterior | 32.320864 | 32.320864 | 0.737389 | 0.610108 | 0.610108 | 13769 |
| Stacked RBMs, 4 to 1 hidden units, exact second layer on hidden logits (amendment), 6 features, logit | 29.771502 | 29.771502 | 0.698674 | 0.568395 | 0.568395 | 13769 |
| Stacked RBMs, 4 to 1 hidden units, exact second layer on hidden logits (amendment), 6 features, posterior | 31.693279 | 31.693279 | 0.689489 | 0.569463 | 0.569463 | 13769 |
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
- Stacked RBMs, 4 to 1 hidden units, exact second layer, 12 features, logit minus RBM, 4 hidden units, exact training, 12 features, logit: PB -6.558799 pp, CI [-7.947308946768319, -5.255692918123089]; within -0.14247904159381589, CI [-0.153648627834933, -0.13156190167496634]; gained/lost 50/249.
- Stacked RBMs, 4 to 1 hidden units, exact second layer, 6 features, posterior minus RBM, 4 hidden units, exact training, 6 features, posterior: PB -6.049906 pp, CI [-7.4185886370758345, -4.717214714464349]; within -0.08894369288481196, CI [-0.09715906517853311, -0.0808314121631887]; gained/lost 42/229.

No automatic promotion. Other comparisons are descriptive95%; intervals do not cover all prior research choices.
See FIT_HEALTH.json for convergence, failures and method details; PB_CELLS.csv and CHANGED_SUCCESSES.csv preserve case-level differences.
