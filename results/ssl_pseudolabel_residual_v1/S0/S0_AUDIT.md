# S0 audit — Claude Step 432 (A1), read-only replay

Source: `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1`. Recomputed with `scripts/experiments/ssl_s0_audit.py` in 4008s. No fit, no inference.

## Exit condition

**VALIDATED** — jobs 70 (30 outer / 40 inner), stale jobs 0, input hashes today all match: True, groups crossing folds 0, folds match FOLDS_V2: True, max |replay - report| over 57 methods x 7 metrics = 8.33e-17.

## Population / contract

answers 13769 (PB 6800: 4442 erroneous + 2358 clean; PRMB 6969, two-class 6030), steps 145597. CT7 gate opens on 4556 PB answers and 0 PRMB answers. Non-finite score methods: ['dual__iu', 'dual__joint0', 'dual__equal', 'context__iu', 'single__joint0']; PB predictions missing: {'dual__iu': 4, 'dual__joint0': 4, 'dual__equal': 4, 'context__iu': 4, 'single__joint0': 4}.

## Provenance deviations

- PRMB pseudo rule(s): ['every training answer (amendment A1: the CT7 gate never opens on PRMBench)']; PB rule: seed argmax of gate-open training answers. Min pseudo-positive mass PRMB 4138, PB 1802.
- RUN_FREEZE `step_evidence_v1.py` = module hash; driver hash NOT frozen (basename collision). Driver today: `1411462fabfe`, module: `2860a10a4eb8`.
- Seed parity: raw replay reproduces the stored seed exactly = True; the protocol-described step-z seed changes 1508 PB argmaxes. Two method IDs are kept (see SEED_PARITY.csv).

## Seed parity

| seed                                                                           |   max_abs_score_diff_vs_stored |   pb_argmax_diff_vs_stored |   pb_sla_macro8 |   prm_within_auc |   top_tie_rate |
|:-------------------------------------------------------------------------------|-------------------------------:|---------------------------:|----------------:|-----------------:|---------------:|
| stored evidence__all__seed__equal (top5, actual run)                           |                         0.0000 |                          0 |          0.2964 |           0.6761 |         0.0000 |
| replay: softmax over steps of raw top5 profile, mean over channels (no step-z) |                         0.0000 |                          0 |          0.2964 |           0.6761 |         0.0000 |
| protocol-described: step-z per channel, then softmax, mean over channels       |                         0.2186 |                       1508 |          0.3249 |           0.7261 |         0.0005 |

## Scoreboard replay (key arms)

| method                      |   pb_sla_macro8 |   pb_f1_ct7_gate |   within_auc |   prmscore_q80 |   prmscore_inner |   pb_covered |   prm_covered |
|:----------------------------|----------------:|-----------------:|-------------:|---------------:|-----------------:|-------------:|--------------:|
| CT7 (frozen anchor)         |          0.3989 |           0.4119 |       0.7724 |         0.6425 |           0.6471 |    6800.0000 |     6969.0000 |
| token L-SML                 |          0.3592 |           0.3732 |       0.7532 |         0.6304 |           0.6299 |    6800.0000 |     6969.0000 |
| token equal                 |          0.3259 |           0.3488 |       0.7311 |         0.6169 |           0.6183 |    6800.0000 |     6969.0000 |
| top5 seed (teacher)         |          0.2964 |           0.3266 |       0.6761 |         0.6078 |           0.6122 |    6800.0000 |     6969.0000 |
| top5 plain evidence         |          0.2904 |           0.3251 |       0.7589 |         0.5832 |           0.5831 |    6800.0000 |     6969.0000 |
| top5 position evidence      |          0.2952 |           0.3300 |       0.7674 |         0.5783 |           0.5783 |    6800.0000 |     6969.0000 |
| top5 plain, iter 2          |          0.2797 |           0.3140 |       0.7619 |         0.5910 |           0.5913 |    6800.0000 |     6969.0000 |
| top5 position, iter 2       |          0.2806 |           0.3148 |       0.7696 |         0.5884 |           0.5882 |    6800.0000 |     6969.0000 |
| top5 random-seed null       |          0.1151 |           0.1624 |       0.5347 |         0.5226 |           0.5256 |    6800.0000 |     6969.0000 |
| top5 random-seed + position |          0.1455 |           0.2041 |       0.5673 |         0.5045 |           0.5064 |    6800.0000 |     6969.0000 |
| top5 prior only             |          0.0580 |           0.0778 |       0.3390 |         0.4071 |           0.4262 |    6800.0000 |     6969.0000 |
| top5 TRUE-label ceiling     |          0.2795 |           0.3155 |       0.7659 |         0.6184 |           0.6221 |    6800.0000 |     6969.0000 |
| top5 plain + L-SML          |          0.2537 |           0.2955 |       0.7619 |         0.5845 |           0.5853 |    6800.0000 |     6969.0000 |
| top30 seed                  |          0.3327 |           0.3622 |       0.6400 |         0.6044 |           0.6042 |    6800.0000 |     6969.0000 |
| top30 plain evidence        |          0.3495 |           0.3696 |       0.7118 |         0.5589 |           0.5590 |    6800.0000 |     6969.0000 |
| top30 position evidence     |          0.3265 |           0.3447 |       0.7174 |         0.5561 |           0.5560 |    6800.0000 |     6969.0000 |

Max replay error per metric: {"err_pb_sla_macro8": 7.63e-17, "err_pb_f1_ct7_gate": 8.33e-17, "err_pb_sla_q4": 8.33e-17, "err_pb_sla_q8": 8.33e-17, "err_within_auc": 5.55e-17, "err_eligible": 0.0, "err_prmscore_q80": 0.0, "err_prmscore_inner": 0.0}

## Planned contrasts by question

| question                                                         | endpoint       | a                                        | b                                  |   delta |   ci_lo |   ci_hi |   p_holm |
|:-----------------------------------------------------------------|:---------------|:-----------------------------------------|:-----------------------------------|--------:|--------:|--------:|---------:|
| Q1 labels contribute? (primary - random-seed null)               | pb_sla         | top5 plain evidence                      | top5 random-seed null              |  0.1753 |  0.1534 |  0.1979 |   0.0312 |
| Q1 labels contribute? (primary - random-seed null)               | prm_within_auc | top5 plain evidence                      | top5 random-seed null              |  0.2242 |  0.2140 |  0.2347 |   0.0312 |
| Q1 labels contribute? (primary - random-seed null)               | pb_sla         | top5 position evidence                   | top5 random-seed + position        |  0.1497 |  0.1277 |  0.1717 |   0.0312 |
| Q1 labels contribute? (primary - random-seed null)               | prm_within_auc | top5 position evidence                   | top5 random-seed + position        |  0.2001 |  0.1900 |  0.2102 |   0.0312 |
| Q1 labels contribute? (primary - random-seed null)               | pb_sla         | top30 plain evidence                     | evidence30__all__randomseed__equal |  0.1791 |  0.1549 |  0.2036 |   0.0312 |
| Q1 labels contribute? (primary - random-seed null)               | prm_within_auc | top30 plain evidence                     | evidence30__all__randomseed__equal |  0.2613 |  0.2487 |  0.2741 |   0.0312 |
| Q1b evidence vs its own seed                                     | pb_sla         | top5 plain evidence                      | top5 seed (teacher)                | -0.0060 | -0.0235 |  0.0112 |   1.0000 |
| Q1b evidence vs its own seed                                     | prm_within_auc | top5 plain evidence                      | top5 seed (teacher)                |  0.0828 |  0.0774 |  0.0883 |   0.0312 |
| Q1b evidence vs its own seed                                     | pb_sla         | top30 plain evidence                     | top30 seed                         |  0.0168 |  0.0007 |  0.0328 |   1.0000 |
| Q1b evidence vs its own seed                                     | prm_within_auc | top30 plain evidence                     | top30 seed                         |  0.0718 |  0.0657 |  0.0778 |   0.0312 |
| Q2 position conditioning? (position - plain; prior-only - plain) | pb_sla         | top5 position evidence                   | top5 plain evidence                |  0.0048 | -0.0082 |  0.0176 |   1.0000 |
| Q2 position conditioning? (position - plain; prior-only - plain) | prm_within_auc | top5 position evidence                   | top5 plain evidence                |  0.0085 |  0.0063 |  0.0107 |   0.0312 |
| Q2 position conditioning? (position - plain; prior-only - plain) | pb_sla         | top30 position evidence                  | top30 plain evidence               | -0.0230 | -0.0338 | -0.0121 |   0.0312 |
| Q2 position conditioning? (position - plain; prior-only - plain) | prm_within_auc | top30 position evidence                  | top30 plain evidence               |  0.0057 |  0.0039 |  0.0074 |   0.0312 |
| Q2 position conditioning? (position - plain; prior-only - plain) | pb_sla         | top5 prior only                          | top5 plain evidence                | -0.2324 | -0.2526 | -0.2113 |   0.0312 |
| Q2 position conditioning? (position - plain; prior-only - plain) | prm_within_auc | top5 prior only                          | top5 plain evidence                | -0.4199 | -0.4315 | -0.4085 |   0.0312 |
| Q3 iteration 2?                                                  | pb_sla         | top5 plain, iter 2                       | top5 plain evidence                | -0.0107 | -0.0199 | -0.0017 |   1.0000 |
| Q3 iteration 2?                                                  | prm_within_auc | top5 plain, iter 2                       | top5 plain evidence                |  0.0030 |  0.0015 |  0.0046 |   0.0312 |
| Q3 iteration 2?                                                  | pb_sla         | top5 position, iter 2                    | top5 position evidence             | -0.0146 | -0.0238 | -0.0052 |   0.1890 |
| Q3 iteration 2?                                                  | prm_within_auc | top5 position, iter 2                    | top5 position evidence             |  0.0021 |  0.0003 |  0.0040 |   1.0000 |
| Q4 learned fusion?                                               | pb_sla         | top5 plain + L-SML                       | top5 plain evidence                | -0.0368 | -0.0501 | -0.0236 |   0.0312 |
| Q4 learned fusion?                                               | prm_within_auc | top5 plain + L-SML                       | top5 plain evidence                |  0.0030 |  0.0015 |  0.0045 |   0.0312 |
| Q4 learned fusion?                                               | pb_sla         | evidence__all__plain__spectral           | top5 plain evidence                | -0.0192 | -0.0319 | -0.0069 |   0.1792 |
| Q4 learned fusion?                                               | prm_within_auc | evidence__all__plain__spectral           | top5 plain evidence                |  0.0024 |  0.0012 |  0.0036 |   0.0312 |
| Q4 learned fusion?                                               | pb_sla         | evidence__all__position__continuous_lsml | top5 position evidence             | -0.0650 | -0.0785 | -0.0520 |   0.0312 |
| Q4 learned fusion?                                               | prm_within_auc | evidence__all__position__continuous_lsml | top5 position evidence             | -0.0010 | -0.0022 |  0.0002 |   1.0000 |
| Q5 vs CT7                                                        | pb_sla         | top5 plain evidence                      | CT7 (frozen anchor)                | -0.1085 | -0.1285 | -0.0884 |   0.0312 |
| Q5 vs CT7                                                        | prm_within_auc | top5 plain evidence                      | CT7 (frozen anchor)                | -0.0135 | -0.0179 | -0.0092 |   0.0312 |
| Q5 vs CT7                                                        | pb_sla         | top5 position evidence                   | CT7 (frozen anchor)                | -0.1037 | -0.1248 | -0.0826 |   0.0312 |
| Q5 vs CT7                                                        | prm_within_auc | top5 position evidence                   | CT7 (frozen anchor)                | -0.0050 | -0.0092 | -0.0009 |   1.0000 |
| Q5 vs CT7                                                        | pb_sla         | top5 position, iter 2                    | CT7 (frozen anchor)                | -0.1183 | -0.1393 | -0.0978 |   0.0312 |
| Q5 vs CT7                                                        | prm_within_auc | top5 position, iter 2                    | CT7 (frozen anchor)                | -0.0028 | -0.0073 |  0.0015 |   1.0000 |
| Q5 vs CT7                                                        | pb_sla         | top30 plain evidence                     | CT7 (frozen anchor)                | -0.0494 | -0.0670 | -0.0321 |   0.0312 |
| Q5 vs CT7                                                        | prm_within_auc | top30 plain evidence                     | CT7 (frozen anchor)                | -0.0606 | -0.0662 | -0.0552 |   0.0312 |
| Q6 label ceiling - primary                                       | pb_sla         | top5 TRUE-label ceiling                  | top5 plain evidence                | -0.0109 | -0.0269 |  0.0045 |   1.0000 |
| Q6 label ceiling - primary                                       | prm_within_auc | top5 TRUE-label ceiling                  | top5 plain evidence                |  0.0070 |  0.0036 |  0.0103 |   0.0312 |
| Q6 label ceiling - primary                                       | pb_sla         | evidence__all__ceiling_position__equal   | top5 position evidence             | -0.0251 | -0.0396 | -0.0102 |   0.0603 |
| Q6 label ceiling - primary                                       | prm_within_auc | evidence__all__ceiling_position__equal   | top5 position evidence             | -0.0036 | -0.0079 |  0.0006 |   1.0000 |

Family: 312 contrasts, 10000 draws, unit = source_question_shared_across_scorers (3483 groups). Tail resolution 1.0e-04 so the Holm floor is 0.0312; 245 contrasts sit at that floor.

## Decision changes on ProcessBench (erroneous answers)

| candidate               | reference              |   agreement |   wrong_to_correct |   correct_to_wrong |   net_gain |   moved_earlier |   moved_later |   cand_early_miss |   cand_late_miss |   long11_wrong_to_correct |   long11_correct_to_wrong |
|:------------------------|:-----------------------|------------:|-------------------:|-------------------:|-----------:|----------------:|--------------:|------------------:|-----------------:|--------------------------:|--------------------------:|
| top5 plain evidence     | top5 seed (teacher)    |       0.517 |                426 |                465 |        -39 |             699 |          1448 |             0.272 |            0.460 |                        60 |                        66 |
| top5 position evidence  | top5 seed (teacher)    |       0.446 |                531 |                537 |         -6 |             712 |          1750 |             0.233 |            0.492 |                        76 |                        75 |
| top5 position evidence  | top5 plain evidence    |       0.689 |                318 |                285 |         33 |             511 |           869 |             0.233 |            0.492 |                        48 |                        41 |
| top5 plain, iter 2      | top5 plain evidence    |       0.811 |                149 |                196 |        -47 |             363 |           477 |             0.266 |            0.477 |                        20 |                        25 |
| top5 position, iter 2   | top5 position evidence |       0.803 |                156 |                228 |        -72 |             364 |           512 |             0.225 |            0.516 |                        20 |                        32 |
| top5 TRUE-label ceiling | top5 plain evidence    |       0.496 |                438 |                490 |        -52 |             985 |          1252 |             0.239 |            0.505 |                        60 |                        68 |
| top5 plain + L-SML      | top5 plain evidence    |       0.625 |                256 |                412 |       -156 |            1170 |           496 |             0.385 |            0.382 |                        31 |                        56 |
| top5 random-seed null   | top5 plain evidence    |       0.117 |                336 |               1057 |       -721 |            1789 |          2133 |             0.332 |            0.563 |                        24 |                       113 |
| top30 plain evidence    | top30 seed             |       0.605 |                439 |                382 |         57 |             566 |          1190 |             0.215 |            0.460 |                        72 |                        66 |
| top30 position evidence | top30 plain evidence   |       0.760 |                168 |                286 |       -118 |             156 |           908 |             0.156 |            0.545 |                        34 |                        43 |
| top5 plain evidence     | CT7 (frozen anchor)    |       0.427 |                395 |                906 |       -511 |            1003 |          1542 |             0.272 |            0.460 |                        48 |                       179 |
| top5 position evidence  | CT7 (frozen anchor)    |       0.414 |                426 |                904 |       -478 |             877 |          1727 |             0.233 |            0.492 |                        53 |                       177 |
| top30 plain evidence    | CT7 (frozen anchor)    |       0.551 |                384 |                636 |       -252 |             550 |          1444 |             0.215 |            0.460 |                        55 |                       144 |
| token L-SML             | CT7 (frozen anchor)    |       0.598 |                333 |                560 |       -227 |             550 |          1234 |             0.225 |            0.443 |                        39 |                       114 |

## Paired per-answer changes on PRMBench (6,030 two-class answers)

| candidate               | reference              |   mean_delta_auc |   answers_improved |   answers_worsened |   answers_unchanged |   pairs_corrected |   pairs_destroyed |   net_pairs |
|:------------------------|:-----------------------|-----------------:|-------------------:|-------------------:|--------------------:|------------------:|------------------:|------------:|
| top5 plain evidence     | top5 seed (teacher)    |           0.0828 |               3486 |               1041 |                1503 |             21165 |             10034 |       11131 |
| top5 position evidence  | top5 seed (teacher)    |           0.0913 |               3581 |               1004 |                1445 |             22116 |             10262 |       11854 |
| top5 position evidence  | top5 plain evidence    |           0.0085 |               1678 |               1109 |                3243 |              4337 |              3614 |         723 |
| top5 plain, iter 2      | top5 plain evidence    |           0.0030 |               1317 |               1003 |                3710 |              3112 |              2491 |         621 |
| top5 position, iter 2   | top5 position evidence |           0.0021 |               1316 |               1114 |                3600 |              3335 |              2853 |         482 |
| top5 TRUE-label ceiling | top5 plain evidence    |           0.0070 |               1926 |               1420 |                2684 |              7214 |              6215 |         999 |
| top5 plain + L-SML      | top5 plain evidence    |           0.0030 |               1200 |               1033 |                3797 |              2772 |              2346 |         426 |
| top5 random-seed null   | top5 plain evidence    |          -0.2242 |               1219 |               4265 |                 546 |             20252 |             48576 |      -28324 |
| top30 plain evidence    | top30 seed             |           0.0718 |               3262 |               1467 |                1301 |             23794 |             13654 |       10140 |
| top30 position evidence | top30 plain evidence   |           0.0057 |               1436 |               1039 |                3555 |              3163 |              2549 |         614 |
| top5 plain evidence     | CT7 (frozen anchor)    |          -0.0135 |               1835 |               2243 |                1952 |             10390 |             11513 |       -1123 |
| top5 position evidence  | CT7 (frozen anchor)    |          -0.0050 |               1986 |               2042 |                2002 |             10858 |             11258 |        -400 |
| top30 plain evidence    | CT7 (frozen anchor)    |          -0.0606 |               1430 |               3191 |                1409 |             11368 |             18807 |       -7439 |
| token L-SML             | CT7 (frozen anchor)    |          -0.0192 |               1638 |               2176 |                2216 |              8091 |             10181 |       -2090 |

## PRMBench within-AUC by answer kind

|                             |   all_error |   clean |   multi |   single |
|:----------------------------|------------:|--------:|--------:|---------:|
| CT7 (frozen anchor)         |         nan |     nan |  0.7339 |   0.8521 |
| token L-SML                 |         nan |     nan |  0.7206 |   0.8206 |
| token equal                 |         nan |     nan |  0.6964 |   0.8029 |
| top5 seed (teacher)         |         nan |     nan |  0.6258 |   0.7802 |
| top5 plain evidence         |         nan |     nan |  0.7238 |   0.8316 |
| top5 position evidence      |         nan |     nan |  0.7337 |   0.8372 |
| top5 plain, iter 2          |         nan |     nan |  0.7309 |   0.8260 |
| top5 position, iter 2       |         nan |     nan |  0.7381 |   0.8348 |
| top5 random-seed null       |         nan |     nan |  0.5323 |   0.5397 |
| top5 random-seed + position |         nan |     nan |  0.5689 |   0.5640 |
| top5 prior only             |         nan |     nan |  0.3339 |   0.3494 |
| top5 TRUE-label ceiling     |         nan |     nan |  0.7402 |   0.8190 |
| top5 plain + L-SML          |         nan |     nan |  0.7267 |   0.8348 |
| top30 seed                  |         nan |     nan |  0.5972 |   0.7286 |
| top30 plain evidence        |         nan |     nan |  0.6803 |   0.7769 |
| top30 position evidence     |         nan |     nan |  0.6869 |   0.7807 |

## ProcessBench by depth

| method                      | depth_bin   |    n |   sla |   hit3 |   hit5 |   mean_rank |   median_margin |   early |   late |   tie_at_top |
|:----------------------------|:------------|-----:|------:|-------:|-------:|------------:|----------------:|--------:|-------:|-------------:|
| CT7 (frozen anchor)         | 11+         |  822 | 0.299 |  0.584 |  0.754 |       4.090 |          -0.337 |   0.281 |  0.420 |        0.000 |
| CT7 (frozen anchor)         | 2-5         | 1312 | 0.476 |  0.914 |  1.000 |       1.870 |          -0.037 |   0.245 |  0.279 |        0.005 |
| CT7 (frozen anchor)         | 6-10        | 2308 | 0.359 |  0.726 |  0.914 |       2.650 |          -0.201 |   0.288 |  0.353 |        0.000 |
| top30 plain evidence        | 11+         |  822 | 0.191 |  0.485 |  0.659 |       4.940 |          -2.867 |   0.242 |  0.567 |        0.000 |
| top30 plain evidence        | 2-5         | 1312 | 0.447 |  0.902 |  1.000 |       1.944 |          -0.437 |   0.202 |  0.351 |        0.000 |
| top30 plain evidence        | 6-10        | 2308 | 0.305 |  0.672 |  0.881 |       2.938 |          -1.501 |   0.212 |  0.483 |        0.002 |
| top30 position evidence     | 11+         |  822 | 0.180 |  0.487 |  0.661 |       4.928 |          -4.074 |   0.153 |  0.667 |        0.000 |
| top30 position evidence     | 2-5         | 1312 | 0.399 |  0.902 |  1.000 |       2.016 |          -1.057 |   0.150 |  0.450 |        0.000 |
| top30 position evidence     | 6-10        | 2308 | 0.285 |  0.662 |  0.878 |       3.009 |          -2.275 |   0.159 |  0.556 |        0.000 |
| top30 seed                  | 11+         |  822 | 0.184 |  0.425 |  0.630 |       5.299 |          -0.129 |   0.350 |  0.466 |        0.000 |
| top30 seed                  | 2-5         | 1312 | 0.417 |  0.866 |  1.000 |       2.075 |          -0.041 |   0.282 |  0.301 |        0.000 |
| top30 seed                  | 6-10        | 2308 | 0.300 |  0.641 |  0.855 |       3.088 |          -0.086 |   0.289 |  0.411 |        0.000 |
| top5 TRUE-label ceiling     | 11+         |  822 | 0.130 |  0.365 |  0.544 |       5.999 |          -0.832 |   0.273 |  0.597 |        0.001 |
| top5 TRUE-label ceiling     | 2-5         | 1312 | 0.370 |  0.859 |  1.000 |       2.160 |          -0.246 |   0.206 |  0.424 |        0.000 |
| top5 TRUE-label ceiling     | 6-10        | 2308 | 0.235 |  0.585 |  0.830 |       3.358 |          -0.542 |   0.246 |  0.519 |        0.000 |
| top5 plain, iter 2          | 11+         |  822 | 0.134 |  0.372 |  0.563 |       5.714 |          -1.843 |   0.319 |  0.547 |        0.000 |
| top5 plain, iter 2          | 2-5         | 1312 | 0.367 |  0.857 |  1.000 |       2.178 |          -0.739 |   0.214 |  0.418 |        0.000 |
| top5 plain, iter 2          | 6-10        | 2308 | 0.238 |  0.591 |  0.822 |       3.367 |          -1.354 |   0.276 |  0.486 |        0.000 |
| top5 plain + L-SML          | 11+         |  822 | 0.109 |  0.314 |  0.512 |       6.382 |          -0.842 |   0.405 |  0.485 |        0.000 |
| top5 plain + L-SML          | 2-5         | 1312 | 0.355 |  0.840 |  1.000 |       2.207 |          -0.347 |   0.342 |  0.303 |        0.000 |
| top5 plain + L-SML          | 6-10        | 2308 | 0.206 |  0.546 |  0.797 |       3.578 |          -0.633 |   0.403 |  0.391 |        0.000 |
| top5 plain evidence         | 11+         |  822 | 0.140 |  0.382 |  0.582 |       5.539 |          -1.288 |   0.311 |  0.549 |        0.000 |
| top5 plain evidence         | 2-5         | 1312 | 0.383 |  0.866 |  1.000 |       2.123 |          -0.511 |   0.223 |  0.395 |        0.000 |
| top5 plain evidence         | 6-10        | 2308 | 0.247 |  0.604 |  0.841 |       3.274 |          -0.931 |   0.287 |  0.466 |        0.000 |
| top5 position, iter 2       | 11+         |  822 | 0.134 |  0.376 |  0.571 |       5.734 |          -2.093 |   0.251 |  0.616 |        0.000 |
| top5 position, iter 2       | 2-5         | 1312 | 0.361 |  0.851 |  1.000 |       2.170 |          -0.761 |   0.176 |  0.463 |        0.000 |
| top5 position, iter 2       | 6-10        | 2308 | 0.245 |  0.606 |  0.837 |       3.277 |          -1.511 |   0.244 |  0.512 |        0.000 |
| top5 position evidence      | 11+         |  822 | 0.148 |  0.420 |  0.590 |       5.470 |          -1.542 |   0.224 |  0.628 |        0.000 |
| top5 position evidence      | 2-5         | 1312 | 0.378 |  0.868 |  1.000 |       2.133 |          -0.509 |   0.198 |  0.424 |        0.000 |
| top5 position evidence      | 6-10        | 2308 | 0.261 |  0.630 |  0.854 |       3.155 |          -1.034 |   0.256 |  0.483 |        0.000 |
| top5 prior only             | 11+         |  822 | 0.051 |  0.122 |  0.288 |       8.080 |          -0.528 |   0.029 |  0.920 |        1.000 |
| top5 prior only             | 2-5         | 1312 | 0.079 |  0.546 |  1.000 |       3.227 |          -0.549 |   0.000 |  0.921 |        0.000 |
| top5 prior only             | 6-10        | 2308 | 0.038 |  0.269 |  0.579 |       5.051 |          -0.547 |   0.003 |  0.959 |        0.289 |
| top5 random-seed null       | 11+         |  822 | 0.032 |  0.122 |  0.218 |       9.353 |          -1.306 |   0.366 |  0.602 |        0.000 |
| top5 random-seed null       | 2-5         | 1312 | 0.169 |  0.636 |  1.000 |       2.944 |          -0.572 |   0.300 |  0.531 |        0.000 |
| top5 random-seed null       | 6-10        | 2308 | 0.095 |  0.300 |  0.588 |       4.876 |          -0.814 |   0.339 |  0.566 |        0.000 |
| top5 random-seed + position | 11+         |  822 | 0.035 |  0.156 |  0.282 |       8.691 |          -1.876 |   0.439 |  0.526 |        0.000 |
| top5 random-seed + position | 2-5         | 1312 | 0.208 |  0.724 |  1.000 |       2.687 |          -0.793 |   0.281 |  0.511 |        0.000 |
| top5 random-seed + position | 6-10        | 2308 | 0.130 |  0.378 |  0.682 |       4.362 |          -1.158 |   0.383 |  0.487 |        0.000 |
| top5 seed (teacher)         | 11+         |  822 | 0.147 |  0.354 |  0.569 |       5.667 |          -0.112 |   0.409 |  0.444 |        0.000 |
| top5 seed (teacher)         | 2-5         | 1312 | 0.389 |  0.829 |  1.000 |       2.205 |          -0.045 |   0.319 |  0.292 |        0.000 |
| top5 seed (teacher)         | 6-10        | 2308 | 0.258 |  0.552 |  0.829 |       3.379 |          -0.091 |   0.396 |  0.347 |        0.000 |
| token equal                 | 11+         |  822 | 0.178 |  0.437 |  0.635 |       5.263 |          -0.176 |   0.218 |  0.605 |        0.000 |
| token equal                 | 2-5         | 1312 | 0.399 |  0.864 |  1.000 |       2.104 |          -0.042 |   0.179 |  0.422 |        0.000 |
| token equal                 | 6-10        | 2308 | 0.295 |  0.652 |  0.857 |       3.053 |          -0.100 |   0.179 |  0.526 |        0.000 |
| token L-SML                 | 11+         |  822 | 0.208 |  0.471 |  0.651 |       5.011 |          -0.598 |   0.255 |  0.536 |        0.000 |
| token L-SML                 | 2-5         | 1312 | 0.432 |  0.892 |  1.000 |       1.995 |          -0.125 |   0.205 |  0.363 |        0.000 |
| token L-SML                 | 6-10        | 2308 | 0.318 |  0.675 |  0.878 |       2.908 |          -0.290 |   0.226 |  0.456 |        0.000 |

## ProcessBench by relative error position

| method                      | rel_bin   |    n |   sla |   hit3 |   hit5 |   mean_rank |   median_margin |   early |   late |   tie_at_top |
|:----------------------------|:----------|-----:|------:|-------:|-------:|------------:|----------------:|--------:|-------:|-------------:|
| CT7 (frozen anchor)         | [.2,.4)   | 1204 | 0.378 |  0.759 |  0.923 |       2.625 |          -0.186 |   0.221 |  0.401 |        0.000 |
| CT7 (frozen anchor)         | [.4,.6)   |  906 | 0.391 |  0.756 |  0.913 |       2.684 |          -0.164 |   0.322 |  0.287 |        0.000 |
| CT7 (frozen anchor)         | [.6,.8)   |  742 | 0.375 |  0.747 |  0.904 |       2.687 |          -0.211 |   0.445 |  0.181 |        0.000 |
| CT7 (frozen anchor)         | [.8,1]    |  518 | 0.384 |  0.743 |  0.880 |       2.807 |          -0.221 |   0.556 |  0.060 |        0.000 |
| CT7 (frozen anchor)         | [0,.2)    | 1072 | 0.385 |  0.761 |  0.910 |       2.697 |          -0.159 |   0.038 |  0.576 |        0.006 |
| top30 plain evidence        | [.2,.4)   | 1204 | 0.262 |  0.681 |  0.873 |       3.120 |          -1.789 |   0.155 |  0.583 |        0.000 |
| top30 plain evidence        | [.4,.6)   |  906 | 0.370 |  0.769 |  0.908 |       2.675 |          -0.863 |   0.223 |  0.407 |        0.002 |
| top30 plain evidence        | [.6,.8)   |  742 | 0.410 |  0.757 |  0.904 |       2.674 |          -0.885 |   0.348 |  0.243 |        0.001 |
| top30 plain evidence        | [.8,1]    |  518 | 0.384 |  0.736 |  0.900 |       2.695 |          -0.852 |   0.550 |  0.066 |        0.004 |
| top30 plain evidence        | [0,.2)    | 1072 | 0.274 |  0.628 |  0.818 |       3.574 |          -2.156 |   0.020 |  0.706 |        0.000 |
| top30 position evidence     | [.2,.4)   | 1204 | 0.233 |  0.661 |  0.868 |       3.213 |          -2.706 |   0.110 |  0.657 |        0.000 |
| top30 position evidence     | [.4,.6)   |  906 | 0.323 |  0.769 |  0.915 |       2.746 |          -1.682 |   0.158 |  0.519 |        0.000 |
| top30 position evidence     | [.6,.8)   |  742 | 0.352 |  0.767 |  0.908 |       2.686 |          -1.585 |   0.279 |  0.369 |        0.000 |
| top30 position evidence     | [.8,1]    |  518 | 0.490 |  0.809 |  0.913 |       2.396 |          -0.105 |   0.384 |  0.125 |        0.000 |
| top30 position evidence     | [0,.2)    | 1072 | 0.225 |  0.589 |  0.803 |       3.778 |          -3.229 |   0.008 |  0.767 |        0.000 |
| top30 seed                  | [.2,.4)   | 1204 | 0.248 |  0.609 |  0.838 |       3.463 |          -0.111 |   0.279 |  0.473 |        0.000 |
| top30 seed                  | [.4,.6)   |  906 | 0.313 |  0.657 |  0.877 |       3.083 |          -0.090 |   0.331 |  0.355 |        0.000 |
| top30 seed                  | [.6,.8)   |  742 | 0.319 |  0.690 |  0.875 |       3.075 |          -0.095 |   0.404 |  0.276 |        0.000 |
| top30 seed                  | [.8,1]    |  518 | 0.326 |  0.691 |  0.880 |       3.000 |          -0.090 |   0.600 |  0.073 |        0.000 |
| top30 seed                  | [0,.2)    | 1072 | 0.374 |  0.715 |  0.834 |       3.176 |          -0.056 |   0.073 |  0.553 |        0.000 |
| top5 TRUE-label ceiling     | [.2,.4)   | 1204 | 0.246 |  0.598 |  0.821 |       3.501 |          -0.597 |   0.133 |  0.621 |        0.001 |
| top5 TRUE-label ceiling     | [.4,.6)   |  906 | 0.283 |  0.678 |  0.876 |       3.124 |          -0.424 |   0.268 |  0.449 |        0.000 |
| top5 TRUE-label ceiling     | [.6,.8)   |  742 | 0.310 |  0.721 |  0.888 |       2.889 |          -0.312 |   0.418 |  0.272 |        0.000 |
| top5 TRUE-label ceiling     | [.8,1]    |  518 | 0.299 |  0.668 |  0.851 |       3.220 |          -0.393 |   0.625 |  0.075 |        0.000 |
| top5 TRUE-label ceiling     | [0,.2)    | 1072 | 0.186 |  0.525 |  0.739 |       4.345 |          -0.711 |   0.022 |  0.792 |        0.000 |
| top5 plain, iter 2          | [.2,.4)   | 1204 | 0.204 |  0.584 |  0.816 |       3.603 |          -1.570 |   0.208 |  0.588 |        0.000 |
| top5 plain, iter 2          | [.4,.6)   |  906 | 0.259 |  0.650 |  0.849 |       3.315 |          -1.214 |   0.290 |  0.450 |        0.000 |
| top5 plain, iter 2          | [.6,.8)   |  742 | 0.325 |  0.726 |  0.879 |       2.889 |          -0.819 |   0.399 |  0.276 |        0.000 |
| top5 plain, iter 2          | [.8,1]    |  518 | 0.317 |  0.695 |  0.869 |       2.992 |          -0.861 |   0.606 |  0.077 |        0.000 |
| top5 plain, iter 2          | [0,.2)    | 1072 | 0.238 |  0.562 |  0.763 |       4.002 |          -1.677 |   0.054 |  0.708 |        0.000 |
| top5 plain + L-SML          | [.2,.4)   | 1204 | 0.198 |  0.589 |  0.816 |       3.645 |          -0.630 |   0.320 |  0.483 |        0.000 |
| top5 plain + L-SML          | [.4,.6)   |  906 | 0.214 |  0.583 |  0.812 |       3.640 |          -0.610 |   0.456 |  0.330 |        0.000 |
| top5 plain + L-SML          | [.6,.8)   |  742 | 0.226 |  0.604 |  0.814 |       3.600 |          -0.619 |   0.569 |  0.205 |        0.000 |
| top5 plain + L-SML          | [.8,1]    |  518 | 0.224 |  0.554 |  0.759 |       3.971 |          -0.655 |   0.716 |  0.060 |        0.000 |
| top5 plain + L-SML          | [0,.2)    | 1072 | 0.295 |  0.605 |  0.799 |       3.718 |          -0.481 |   0.113 |  0.592 |        0.000 |
| top5 plain evidence         | [.2,.4)   | 1204 | 0.209 |  0.602 |  0.829 |       3.528 |          -1.073 |   0.233 |  0.558 |        0.000 |
| top5 plain evidence         | [.4,.6)   |  906 | 0.255 |  0.647 |  0.864 |       3.280 |          -0.910 |   0.312 |  0.433 |        0.000 |
| top5 plain evidence         | [.6,.8)   |  742 | 0.321 |  0.710 |  0.881 |       2.920 |          -0.674 |   0.392 |  0.287 |        0.000 |
| top5 plain evidence         | [.8,1]    |  518 | 0.344 |  0.707 |  0.875 |       2.890 |          -0.576 |   0.566 |  0.091 |        0.000 |
| top5 plain evidence         | [0,.2)    | 1072 | 0.270 |  0.598 |  0.787 |       3.743 |          -1.032 |   0.059 |  0.672 |        0.000 |
| top5 position, iter 2       | [.2,.4)   | 1204 | 0.203 |  0.607 |  0.834 |       3.509 |          -1.648 |   0.158 |  0.639 |        0.000 |
| top5 position, iter 2       | [.4,.6)   |  906 | 0.276 |  0.656 |  0.855 |       3.213 |          -1.403 |   0.247 |  0.477 |        0.000 |
| top5 position, iter 2       | [.6,.8)   |  742 | 0.301 |  0.726 |  0.892 |       2.841 |          -0.887 |   0.376 |  0.323 |        0.000 |
| top5 position, iter 2       | [.8,1]    |  518 | 0.376 |  0.728 |  0.882 |       2.830 |          -0.596 |   0.523 |  0.100 |        0.000 |
| top5 position, iter 2       | [0,.2)    | 1072 | 0.220 |  0.544 |  0.760 |       4.117 |          -2.076 |   0.033 |  0.747 |        0.000 |
| top5 position evidence      | [.2,.4)   | 1204 | 0.217 |  0.645 |  0.836 |       3.378 |          -1.146 |   0.184 |  0.600 |        0.000 |
| top5 position evidence      | [.4,.6)   |  906 | 0.268 |  0.667 |  0.876 |       3.149 |          -0.979 |   0.270 |  0.461 |        0.000 |
| top5 position evidence      | [.6,.8)   |  742 | 0.295 |  0.704 |  0.895 |       2.892 |          -0.881 |   0.388 |  0.317 |        0.000 |
| top5 position evidence      | [.8,1]    |  518 | 0.419 |  0.743 |  0.876 |       2.757 |          -0.369 |   0.479 |  0.102 |        0.000 |
| top5 position evidence      | [0,.2)    | 1072 | 0.262 |  0.606 |  0.793 |       3.808 |          -1.254 |   0.030 |  0.708 |        0.000 |
| top5 prior only             | [.2,.4)   | 1204 | 0.000 |  0.000 |  0.488 |       6.570 |          -0.574 |   0.000 |  1.000 |        0.342 |
| top5 prior only             | [.4,.6)   |  906 | 0.000 |  0.336 |  0.646 |       5.396 |          -0.559 |   0.000 |  1.000 |        0.274 |
| top5 prior only             | [.6,.8)   |  742 | 0.000 |  0.326 |  0.538 |       5.341 |          -0.554 |   0.000 |  1.000 |        0.326 |
| top5 prior only             | [.8,1]    |  518 | 0.452 |  0.598 |  0.869 |       3.137 |           0.000 |   0.058 |  0.490 |        0.320 |
| top5 prior only             | [0,.2)    | 1072 | 0.000 |  0.542 |  0.805 |       3.868 |          -0.369 |   0.000 |  1.000 |        0.392 |
| top5 random-seed null       | [.2,.4)   | 1204 | 0.102 |  0.388 |  0.699 |       4.757 |          -0.729 |   0.198 |  0.700 |        0.000 |
| top5 random-seed null       | [.4,.6)   |  906 | 0.123 |  0.386 |  0.660 |       4.752 |          -0.756 |   0.390 |  0.488 |        0.000 |
| top5 random-seed null       | [.6,.8)   |  742 | 0.109 |  0.379 |  0.663 |       4.860 |          -0.771 |   0.580 |  0.311 |        0.000 |
| top5 random-seed null       | [.8,1]    |  518 | 0.089 |  0.311 |  0.589 |       5.647 |          -0.928 |   0.801 |  0.110 |        0.000 |
| top5 random-seed null       | [0,.2)    | 1072 | 0.099 |  0.344 |  0.570 |       5.822 |          -0.850 |   0.037 |  0.864 |        0.000 |
| top5 random-seed + position | [.2,.4)   | 1204 | 0.164 |  0.499 |  0.756 |       4.174 |          -1.008 |   0.232 |  0.605 |        0.000 |
| top5 random-seed + position | [.4,.6)   |  906 | 0.114 |  0.444 |  0.735 |       4.325 |          -1.122 |   0.454 |  0.433 |        0.000 |
| top5 random-seed + position | [.6,.8)   |  742 | 0.089 |  0.414 |  0.708 |       4.608 |          -1.263 |   0.644 |  0.267 |        0.000 |
| top5 random-seed + position | [.8,1]    |  518 | 0.120 |  0.394 |  0.674 |       4.973 |          -1.290 |   0.793 |  0.087 |        0.000 |
| top5 random-seed + position | [0,.2)    | 1072 | 0.161 |  0.407 |  0.623 |       5.410 |          -1.312 |   0.034 |  0.805 |        0.000 |
| top5 seed (teacher)         | [.2,.4)   | 1204 | 0.206 |  0.496 |  0.787 |       3.866 |          -0.109 |   0.365 |  0.429 |        0.000 |
| top5 seed (teacher)         | [.4,.6)   |  906 | 0.270 |  0.592 |  0.857 |       3.365 |          -0.099 |   0.424 |  0.306 |        0.000 |
| top5 seed (teacher)         | [.6,.8)   |  742 | 0.260 |  0.593 |  0.833 |       3.441 |          -0.099 |   0.487 |  0.253 |        0.000 |
| top5 seed (teacher)         | [.8,1]    |  518 | 0.286 |  0.627 |  0.867 |       3.222 |          -0.090 |   0.633 |  0.081 |        0.000 |
| top5 seed (teacher)         | [0,.2)    | 1072 | 0.367 |  0.705 |  0.842 |       3.194 |          -0.039 |   0.144 |  0.490 |        0.000 |
| token equal                 | [.2,.4)   | 1204 | 0.257 |  0.641 |  0.853 |       3.269 |          -0.109 |   0.126 |  0.617 |        0.000 |
| token equal                 | [.4,.6)   |  906 | 0.347 |  0.736 |  0.912 |       2.832 |          -0.079 |   0.180 |  0.474 |        0.000 |
| token equal                 | [.6,.8)   |  742 | 0.379 |  0.757 |  0.902 |       2.694 |          -0.056 |   0.317 |  0.305 |        0.000 |
| token equal                 | [.8,1]    |  518 | 0.425 |  0.803 |  0.913 |       2.471 |          -0.038 |   0.492 |  0.083 |        0.000 |
| token equal                 | [0,.2)    | 1072 | 0.210 |  0.541 |  0.763 |       4.060 |          -0.145 |   0.021 |  0.769 |        0.000 |
| token L-SML                 | [.2,.4)   | 1204 | 0.290 |  0.679 |  0.873 |       3.110 |          -0.301 |   0.180 |  0.530 |        0.000 |
| token L-SML                 | [.4,.6)   |  906 | 0.360 |  0.734 |  0.896 |       2.837 |          -0.265 |   0.241 |  0.400 |        0.000 |
| token L-SML                 | [.6,.8)   |  742 | 0.367 |  0.753 |  0.904 |       2.736 |          -0.252 |   0.342 |  0.291 |        0.000 |
| token L-SML                 | [.8,1]    |  518 | 0.403 |  0.772 |  0.900 |       2.593 |          -0.159 |   0.529 |  0.068 |        0.000 |
| token L-SML                 | [0,.2)    | 1072 | 0.295 |  0.629 |  0.814 |       3.507 |          -0.378 |   0.035 |  0.670 |        0.000 |

## Competition oracle (S>=4; keep truth + 3 uniformly chosen steps; label-using diagnostic)

| method                      |   n_S4 |   exact_top1_S4 |   competition_oracle_3 |   hit1 |   hit3 |   hit5 |   mean_rank |   median_margin |
|:----------------------------|-------:|----------------:|-----------------------:|-------:|-------:|-------:|------------:|----------------:|
| CT7 (frozen anchor)         |   4242 |           0.374 |                  0.525 |  0.374 |  0.744 |  0.905 |       2.738 |          -0.189 |
| top30 plain evidence        |   4242 |           0.317 |                  0.466 |  0.317 |  0.691 |  0.869 |       3.080 |          -1.485 |
| top30 position evidence     |   4242 |           0.291 |                  0.444 |  0.291 |  0.686 |  0.868 |       3.135 |          -2.261 |
| top30 seed                  |   4242 |           0.306 |                  0.442 |  0.306 |  0.652 |  0.849 |       3.266 |          -0.090 |
| top5 TRUE-label ceiling     |   4242 |           0.247 |                  0.384 |  0.247 |  0.608 |  0.819 |       3.576 |          -0.536 |
| top5 plain, iter 2          |   4242 |           0.248 |                  0.388 |  0.248 |  0.612 |  0.818 |       3.530 |          -1.310 |
| top5 plain + L-SML          |   4242 |           0.221 |                  0.356 |  0.221 |  0.571 |  0.795 |       3.785 |          -0.615 |
| top5 plain evidence         |   4242 |           0.260 |                  0.404 |  0.260 |  0.624 |  0.833 |       3.428 |          -0.914 |
| top5 position, iter 2       |   4242 |           0.251 |                  0.394 |  0.251 |  0.619 |  0.828 |       3.482 |          -1.463 |
| top5 position evidence      |   4242 |           0.267 |                  0.415 |  0.267 |  0.645 |  0.841 |       3.353 |          -1.030 |
| top5 prior only             |   4242 |           0.048 |                  0.127 |  0.048 |  0.292 |  0.633 |       5.205 |          -0.547 |
| top5 random-seed null       |   4242 |           0.094 |                  0.170 |  0.094 |  0.337 |  0.624 |       5.282 |          -0.821 |
| top5 random-seed + position |   4242 |           0.125 |                  0.219 |  0.125 |  0.413 |  0.688 |       4.799 |          -1.194 |
| top5 seed (teacher)         |   4242 |           0.269 |                  0.396 |  0.269 |  0.579 |  0.824 |       3.534 |          -0.094 |
| token equal                 |   4242 |           0.298 |                  0.441 |  0.298 |  0.659 |  0.852 |       3.248 |          -0.098 |
| token L-SML                 |   4242 |           0.326 |                  0.472 |  0.326 |  0.687 |  0.866 |       3.087 |          -0.291 |
