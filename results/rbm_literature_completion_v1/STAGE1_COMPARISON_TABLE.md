| Row | Class | PB all-8 % (full population) | PB all-8 % (covered) | Coverage | PRMB within | PRMB pooled | PRMScore | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Token entropy (retained reference) | simple reference | 35.4444 | 35.4444 | 1.0000 | 0.730111 | 0.702664 | 0.625426 | 13769 |
| Varentropy15 raw | simple reference | 35.9610 | 35.9610 | 1.0000 | 0.737786 | 0.710138 | 0.625781 | 13769 |
| Varentropy50 raw | simple reference | 35.6755 | 35.6755 | 1.0000 | 0.742465 | 0.715783 | 0.632777 | 13769 |
| Varentropy15 contributions, equal | fixed fusion | 35.5980 | 35.5980 | 1.0000 | 0.746980 | 0.699988 | 0.612812 | 13769 |
| Varentropy15 contributions, IU-PCR | learned fusion | 35.3498 | 35.3498 | 1.0000 | 0.746824 | 0.710261 | 0.622689 | 13769 |
| Longest-step control | simple control | 33.6944 | 33.6944 | 1.0000 | 0.618136 | 0.586199 | 0.527878 | 13769 |
| Random-step control | simple control | 20.2750 | 20.2750 | 1.0000 | 0.498480 | 0.501535 | 0.497941 | 13769 |
| RBM6 (order-3 bank), posterior | learned fusion | 36.2017 | 36.2017 | 1.0000 | 0.735982 | 0.708110 | 0.630749 | 13769 |
| RBM6 before training | fixed fusion | 35.9662 | 35.9662 | 1.0000 | 0.744068 | 0.695046 | 0.613085 | 13769 |
| RBM12 (order-6 bank), logit | learned fusion | 36.2712 | 36.2712 | 1.0000 | 0.745204 | 0.705849 | 0.622215 | 13769 |
| RBM12 (order-6 bank), posterior | learned fusion | 36.3750 | 36.3750 | 1.0000 | 0.738702 | 0.710417 | 0.629276 | 13769 |
| RBM12 before training | fixed fusion | 36.2946 | 36.2946 | 1.0000 | 0.746325 | 0.690110 | 0.605313 | 13769 |
| RBM6 weight shrinkage | learned fusion | 35.8711 | 35.8711 | 1.0000 | 0.738630 | 0.707997 | 0.629544 | 13769 |
| RBM6 shared diagonal variance | learned fusion | 35.8069 | 35.8069 | 1.0000 | 0.733045 | 0.704727 | 0.625301 | 13769 |
| Two-state shared variance, bank12, posterior | learned fusion | 36.8106 | 36.8106 | 1.0000 | 0.739357 | 0.710522 | 0.628349 | 13769 |
| Two-state separate variance, bank12, logit | learned fusion | 21.0920 | 21.0920 | 1.0000 | 0.698596 | 0.677533 | 0.589463 | 13769 |
| CD-10 H1, bank12, posterior | learned fusion (fixed epoch budget) | 36.2766 | 36.2766 | 1.0000 | 0.743705 | 0.651979 | 0.558246 | 13769 |
| Exact H4, bank12, logit (maxiter 100) | learned fusion (iteration cap) | 28.6553 | 28.6553 | 1.0000 | 0.721693 | 0.676472 | 0.601725 | 13769 |
| Exact H4, bank6, posterior (maxiter 100) | learned fusion (iteration cap) | 25.3205 | 25.3205 | 1.0000 | 0.710100 | 0.687584 | 0.616757 | 13769 |
| Token-chain Markov, bank12, logit | learned fusion | 35.8688 | 35.8688 | 1.0000 | 0.742263 | 0.703860 | 0.618424 | 13769 |
| Token-chain shuffled control, bank12, logit | control | 36.1991 | 36.1991 | 1.0000 | 0.745076 | 0.706140 | 0.622643 | 13769 |
| Position-conditioned RBM12 | learned fusion | 35.5837 | 35.5837 | 1.0000 | 0.742490 | 0.706144 | 0.622079 | 13769 |
| DUFS-selected 6 of 12, RBM | learned selection + fusion | 36.1235 | 36.1235 | 1.0000 | 0.740974 | 0.709615 | 0.626847 | 13769 |
| Low-correlation 6 of 12, RBM | selection control + fusion | 36.9930 | 36.9930 | 1.0000 | 0.740490 | 0.708511 | 0.630604 | 13769 |
| 48 raw rank-power columns, RBM before training | fixed fusion | 36.3835 | 36.3835 | 1.0000 | 0.742119 | 0.640875 | 0.558595 | 13769 |
| 48 raw rank-power columns, RBM trained | learned fusion | 19.4417 | 19.4417 | 1.0000 | 0.593472 | 0.589022 | 0.376400 | 13769 |
| Supervised step-BCE correction of RBM12 (labels, other answers) | supervised diagnostic | 37.2042 | 37.2042 | 1.0000 | 0.747301 | n/a | 0.599189 | 13769 |
| Stability: best-of-3 exact H1, bank6, posterior | learned fusion | 36.2017 | 36.2017 | 1.0000 | 0.735954 | 0.708111 | 0.630789 | 13769 |
| Stability: best-of-3 exact H4, bank6, posterior | learned fusion | 24.7061 | 24.7061 | 1.0000 | 0.708662 | 0.689888 | 0.618526 | 13769 |
| Stability: best-of-3 exact H1, bank12, logit | learned fusion | 36.2325 | 36.2325 | 1.0000 | 0.745024 | 0.705626 | 0.622175 | 13769 |
| Stability: best-of-3 exact H4, bank12, logit | learned fusion | 27.3262 | 27.3262 | 1.0000 | 0.721817 | 0.685433 | 0.612483 | 13769 |
| Depth: exact second layer on H4 posteriors, bank6 | learned fusion | 19.2706 | 19.6423 | 0.9898 | 0.621533 | 0.511115 | 0.467270 (conditional) | 13629 |
| Depth: CD-10 second layer on H4 posteriors, bank6 | learned fusion | 28.1571 | 28.7623 | 0.9898 | 0.726867 | 0.682365 | 0.599428 (conditional) | 13629 |
| Depth: exact second layer on H4 logits, bank6 (amendment) | learned fusion | 31.6933 | 31.6933 | 1.0000 | 0.689489 | 0.591206 | 0.569463 | 13769 |
| Depth: CD-10 second layer on H4 logits, bank6 (amendment) | learned fusion | 32.3209 | 32.3209 | 1.0000 | 0.737389 | 0.687362 | 0.610108 | 13769 |
| Depth: exact second layer on H4 posteriors, bank12 | learned fusion | 22.0965 | 22.9645 | 0.9744 | 0.579489 | 0.512072 | 0.486925 (conditional) | 13416 |
| Depth: CD-10 second layer on H4 posteriors, bank12 | learned fusion | 20.3079 | 21.1267 | 0.9744 | 0.656764 | 0.623718 | 0.536128 (conditional) | 13416 |
| Depth: exact second layer on H4 logits, bank12 (amendment) | learned fusion | 32.7766 | 32.7766 | 1.0000 | 0.715080 | 0.606142 | 0.595517 | 13769 |
| Depth: CD-10 second layer on H4 logits, bank12 (amendment) | learned fusion | 22.9740 | 22.9740 | 1.0000 | 0.673483 | 0.626183 | 0.543016 | 13769 |
