# Prediction views inside IU/Joint — Step312

Decision: retain original IU/Joint. All21 augmented recipes are below both original dual IU and graph100 Joint on both primary point metrics. No consistent winner.

| Method | PRMB AUROC | PB macro F1 |
|---|---:|---:|
| dual__equal | 0.62649 | 25.55% |
| dual__iu | 0.63797 | 30.16% |
| dual__cond100 | 0.63296 | 25.47% |
| dual__cond100_graph010 | 0.63847 | 30.22% |
| dual__cond100_graph_perm | 0.63504 | 25.47% |
| dual__equal_graph010 | 0.62993 | 26.62% |
| dual__equal_graph_perm | 0.62170 | 31.32% |
| ar1__equal | 0.60579 | 22.74% |
| ar1__iu | 0.61334 | 23.50% |
| ar1__joint0 | 0.61835 | 15.55% |
| ar1__graph010 | 0.62355 | 18.01% |
| ar1__graph_perm | 0.62043 | 15.55% |
| ar1__equal_graph010 | 0.61112 | 26.41% |
| ar1__equal_graph_perm | 0.59494 | 27.06% |
| last__equal | 0.62220 | 27.87% |
| last__iu | 0.62491 | 29.74% |
| last__joint0 | 0.61840 | 17.81% |
| last__graph010 | 0.62392 | 17.55% |
| last__graph_perm | 0.61967 | 17.89% |
| last__equal_graph010 | 0.62414 | 26.84% |
| last__equal_graph_perm | 0.62473 | 25.45% |
| ema32__equal | 0.61008 | 24.99% |
| ema32__iu | 0.62351 | 25.05% |
| ema32__joint0 | 0.61415 | 25.61% |
| ema32__graph010 | 0.62066 | 28.39% |
| ema32__graph_perm | 0.61551 | 25.61% |
| ema32__equal_graph010 | 0.61953 | 24.98% |
| ema32__equal_graph_perm | 0.60357 | 26.59% |

All330 native Joint fits pass; original selected-bank native coverage107/110. Fixed81moment/29context banks; no fallback used. AR has K3/4/6/8 in34/32/39/5 answers. Better fitting did not give better localization.

AR+IU minus original IU: PRMB -0.02464 CI[-0.05027,-0.00477]; PB -6.663pp CI[-16.424,+1.426]. AR+Joint graph minus original graph100: PRMB -0.01492 CI[-0.03538,+0.00334]; PB -12.208pp CI[-23.824,-0.764]. Conditional107 fit-valid PB difference remains negative. All intervals exploratory/unadjusted for74 comparisons.

Original27 raw columns replay exactly, but refitting changes their signs/weights/groups. Post-evaluation sign-change/trajectory-correlation/weight-share and hit diagnostics are descriptive, not causal explanation or new candidate.

Review PASS: {'direct_label_group_joins': 110, 'exact_parent_method_rows': 8470, 'exact_augmented_matrix_replays': 330, 'normalization_orientation_replays': 330, 'independent_covariance_constructions': 330, 'source_jacobian_replays': 330, 'native_fit_guards': 330, 'source_iu_refits': 330, 'source_graph_replays': 330, 'independent_laplacians': 660, 'independent_weight_projections': 2310, 'independent_step_reductions': 2310, 'independent_native_decisions': 2310, 'fixed_bank_fallback_and_row_checks': 2310, 'metric_bundles': 98, 'parent_metric_replays': 77, 'paired_point_scope_bundles': 74}. Max risk difference 2.265e-14. Scoring 359.84s; contrasts 40.89s; review 106.96s. Three tests; six explicit bootstraps. Scientific kernels reused for specified replays; independent algebra/metrics in same session. No new model inference.

Review harness fixes: recursive grouping-record comparison; established1e-12 independent projection tolerance for direct overlap sums versus difference-array cumsum (first mismatch4.44e-16). Frozen scientific code/predictions unchanged.

Next bounded work: actual benchmark span/text and frozen-score failure analysis (first-error peaks, near ties and clean decisions). Avoid another residual-dose sweep before identifying a mechanism. Keep fusion central and all wider authorized tracks pending.

Full historical98 table and all74 uncertainty/scope comparisons appear in [REPORT.html](REPORT.html); original58 and cross-answer results remain separate context.

| Comparison | Scope (N) | PRMB delta / CI | PB delta / CI in pp |
|---|---|---:|---:|
| ar1__equal minus dual__equal | all (110) | -0.02071 [-0.0431, -0.0019] | -2.81% [-10.1143, +3.9484] |
| ar1__iu minus dual__iu | all (110) | -0.02464 [-0.0503, -0.0048] | -6.66% [-16.4240, +1.4261] |
| ar1__joint0 minus dual__cond100 | all (110) | -0.01460 [-0.0378, +0.0044] | -9.92% [-21.8456, +1.7734] |
| ar1__graph010 minus dual__cond100_graph010 | all (110) | -0.01492 [-0.0354, +0.0033] | -12.21% [-23.8244, -0.7638] |
| ar1__graph_perm minus dual__cond100_graph_perm | all (110) | -0.01460 [-0.0369, +0.0034] | -9.92% [-21.8456, +1.7734] |
| ar1__equal_graph010 minus dual__equal_graph010 | all (110) | -0.01881 [-0.0397, -0.0013] | -0.21% [-8.2467, +8.2747] |
| ar1__equal_graph_perm minus dual__equal_graph_perm | all (110) | -0.02676 [-0.0495, -0.0086] | -4.26% [-17.9630, +9.2532] |
| ar1__iu minus ar1__equal | all (110) | 0.00755 [-0.0119, +0.0207] | 0.76% [-9.4483, +9.5798] |
| ar1__joint0 minus ar1__iu | all (110) | 0.00502 [-0.0175, +0.0205] | -7.95% [-18.5534, +2.7030] |
| ar1__joint0 minus ar1__equal | all (110) | 0.01257 [-0.0255, +0.0388] | -7.19% [-19.1524, +5.6032] |
| ar1__graph010 minus ar1__joint0 | all (110) | 0.00520 [-0.0020, +0.0146] | 2.47% [-2.5531, +8.7632] |
| ar1__graph010 minus ar1__graph_perm | all (110) | 0.00312 [-0.0037, +0.0134] | 2.47% [-2.5531, +8.7632] |
| ar1__graph010 minus ar1__iu | all (110) | 0.01022 [-0.0084, +0.0253] | -5.48% [-16.7333, +5.5075] |
| ar1__graph010 minus ar1__equal_graph010 | all (110) | 0.01243 [-0.0222, +0.0365] | -8.40% [-21.5708, +3.8579] |
| ar1__equal_graph010 minus ar1__equal | all (110) | 0.00533 [-0.0033, +0.0146] | 3.68% [-3.2423, +11.6536] |
| ar1__equal_graph010 minus ar1__equal_graph_perm | all (110) | 0.01618 [-0.0052, +0.0373] | -0.65% [-11.2386, +10.2563] |
| last__equal minus dual__equal | all (110) | -0.00429 [-0.0297, +0.0149] | 2.32% [-5.2009, +10.2998] |
| last__iu minus dual__iu | all (110) | -0.01307 [-0.0368, +0.0087] | -0.42% [-8.4720, +7.0356] |
| last__joint0 minus dual__cond100 | all (110) | -0.01456 [-0.0408, +0.0128] | -7.66% [-15.6410, +1.3359] |
| last__graph010 minus dual__cond100_graph010 | all (110) | -0.01456 [-0.0394, +0.0112] | -12.68% [-19.2450, -3.6331] |
| last__graph_perm minus dual__cond100_graph_perm | all (110) | -0.01537 [-0.0414, +0.0123] | -7.58% [-15.8820, +1.5797] |
| last__equal_graph010 minus dual__equal_graph010 | all (110) | -0.00579 [-0.0296, +0.0130] | 0.22% [-8.7771, +9.1106] |
| last__equal_graph_perm minus dual__equal_graph_perm | all (110) | 0.00303 [-0.0241, +0.0262] | -5.87% [-17.6141, +3.9478] |
| last__iu minus last__equal | all (110) | 0.00271 [-0.0120, +0.0188] | 1.86% [-2.4558, +6.7417] |
| last__joint0 minus last__iu | all (110) | -0.00651 [-0.0337, +0.0133] | -11.93% [-18.1141, -3.4090] |
| last__joint0 minus last__equal | all (110) | -0.00380 [-0.0326, +0.0188] | -10.06% [-17.2051, -0.8358] |
| last__graph010 minus last__joint0 | all (110) | 0.00552 [-0.0019, +0.0143] | -0.26% [-2.6640, +0.0000] |
| last__graph010 minus last__graph_perm | all (110) | 0.00425 [-0.0022, +0.0126] | -0.35% [-3.3016, +3.9937] |
| last__graph010 minus last__iu | all (110) | -0.00099 [-0.0237, +0.0157] | -12.19% [-18.7980, -3.8929] |
| last__graph010 minus last__equal_graph010 | all (110) | -0.00023 [-0.0305, +0.0244] | -9.29% [-15.5959, -1.1122] |
| last__equal_graph010 minus last__equal | all (110) | 0.00194 [-0.0077, +0.0134] | -1.03% [-6.5456, +4.5811] |
| last__equal_graph010 minus last__equal_graph_perm | all (110) | -0.00059 [-0.0268, +0.0212] | 1.39% [-6.2521, +8.2385] |
| ema32__equal minus dual__equal | all (110) | -0.01641 [-0.0333, +0.0028] | -0.56% [-6.5887, +5.8739] |
| ema32__iu minus dual__iu | all (110) | -0.01447 [-0.0341, +0.0015] | -5.11% [-11.4708, +0.0061] |
| ema32__joint0 minus dual__cond100 | all (110) | -0.01881 [-0.0456, +0.0031] | 0.14% [-9.3015, +7.1157] |
| ema32__graph010 minus dual__cond100_graph010 | all (110) | -0.01781 [-0.0443, +0.0031] | -1.84% [-7.7701, +3.0806] |
| ema32__graph_perm minus dual__cond100_graph_perm | all (110) | -0.01953 [-0.0451, +0.0013] | 0.14% [-9.3015, +7.1157] |
| ema32__equal_graph010 minus dual__equal_graph010 | all (110) | -0.01040 [-0.0271, +0.0078] | -1.64% [-9.0269, +6.8403] |
| ema32__equal_graph_perm minus dual__equal_graph_perm | all (110) | -0.01813 [-0.0397, +0.0050] | -4.73% [-17.3699, +8.5452] |
| ema32__iu minus ema32__equal | all (110) | 0.01343 [-0.0045, +0.0257] | 0.06% [-6.8958, +6.7852] |
| ema32__joint0 minus ema32__iu | all (110) | -0.00936 [-0.0403, +0.0151] | 0.56% [-8.5509, +9.1375] |
| ema32__joint0 minus ema32__equal | all (110) | 0.00407 [-0.0414, +0.0358] | 0.63% [-9.1495, +8.8053] |
| ema32__graph010 minus ema32__joint0 | all (110) | 0.00651 [-0.0025, +0.0176] | 2.77% [+0.0000, +7.9212] |
| ema32__graph010 minus ema32__graph_perm | all (110) | 0.00515 [-0.0036, +0.0161] | 2.77% [+0.0000, +7.9212] |
| ema32__graph010 minus ema32__iu | all (110) | -0.00285 [-0.0295, +0.0177] | 3.34% [-5.9417, +12.0703] |
| ema32__graph010 minus ema32__equal_graph010 | all (110) | 0.00113 [-0.0403, +0.0277] | 3.40% [-5.8550, +11.8922] |
| ema32__equal_graph010 minus ema32__equal | all (110) | 0.00945 [-0.0018, +0.0210] | -0.00% [-2.1655, +2.9718] |
| ema32__equal_graph010 minus ema32__equal_graph_perm | all (110) | 0.01596 [-0.0024, +0.0331] | -1.61% [-10.0681, +6.5263] |
| ar1__equal minus last__equal | all (110) | -0.01641 [-0.0306, -0.0017] | -5.14% [-13.3693, +1.8473] |
| ar1__iu minus last__iu | all (110) | -0.01157 [-0.0234, -0.0035] | -6.24% [-16.9596, +2.5808] |
| ar1__joint0 minus last__joint0 | all (110) | -0.00005 [-0.0182, +0.0153] | -2.26% [-14.1595, +8.0378] |
| ar1__graph010 minus last__graph010 | all (110) | -0.00036 [-0.0177, +0.0145] | 0.47% [-11.3718, +10.4178] |
| ar1__graph_perm minus last__graph_perm | all (110) | 0.00077 [-0.0178, +0.0153] | -2.34% [-13.7134, +8.5857] |
| ar1__equal_graph010 minus last__equal_graph010 | all (110) | -0.01302 [-0.0249, -0.0011] | -0.43% [-8.0060, +7.9268] |
| ar1__equal_graph_perm minus last__equal_graph_perm | all (110) | -0.02979 [-0.0563, -0.0013] | 1.61% [-7.5563, +11.1906] |
| ar1__equal minus ema32__equal | all (110) | -0.00429 [-0.0325, +0.0170] | -2.25% [-8.2598, +3.0419] |
| ar1__iu minus ema32__iu | all (110) | -0.01017 [-0.0306, +0.0071] | -1.55% [-12.4221, +6.4741] |
| ar1__joint0 minus ema32__joint0 | all (110) | 0.00420 [-0.0168, +0.0195] | -10.06% [-19.9142, +0.7321] |
| ar1__graph010 minus ema32__graph010 | all (110) | 0.00289 [-0.0162, +0.0204] | -10.37% [-20.0664, -0.3466] |
| ar1__graph_perm minus ema32__graph_perm | all (110) | 0.00493 [-0.0163, +0.0228] | -10.06% [-19.9142, +0.7321] |
| ar1__equal_graph010 minus ema32__equal_graph010 | all (110) | -0.00841 [-0.0347, +0.0100] | 1.43% [-2.5746, +5.7921] |
| ar1__equal_graph_perm minus ema32__equal_graph_perm | all (110) | -0.00863 [-0.0390, +0.0158] | 0.47% [-5.4274, +5.7143] |
| ar1__graph010 minus ar1__iu | native:ar1 (110) | 0.01022 [-0.0084, +0.0253] | -5.48% [-16.7333, +5.5075] |
| ar1__graph010 minus ar1__equal_graph010 | native:ar1 (110) | 0.01243 [-0.0222, +0.0365] | -8.40% [-21.5708, +3.8579] |
| ar1__graph010 minus dual__cond100_graph010 | native_and_original:ar1 (107) | -0.01122 [-0.0320, +0.0067] | -12.55% [-24.4660, -1.0723] |
| ar1__joint0 minus dual__cond100 | native_and_original:ar1 (107) | -0.01056 [-0.0318, +0.0094] | -10.71% [-22.5425, +0.5711] |
| last__graph010 minus last__iu | native:last (110) | -0.00099 [-0.0237, +0.0157] | -12.19% [-18.7980, -3.8929] |
| last__graph010 minus last__equal_graph010 | native:last (110) | -0.00023 [-0.0305, +0.0244] | -9.29% [-15.5959, -1.1122] |
| last__graph010 minus dual__cond100_graph010 | native_and_original:last (107) | -0.01061 [-0.0358, +0.0172] | -12.99% [-20.2509, -3.3391] |
| last__joint0 minus dual__cond100 | native_and_original:last (107) | -0.01163 [-0.0401, +0.0184] | -7.94% [-15.2497, +1.4063] |
| ema32__graph010 minus ema32__iu | native:ema32 (110) | -0.00285 [-0.0295, +0.0177] | 3.34% [-5.9417, +12.0703] |
| ema32__graph010 minus ema32__equal_graph010 | native:ema32 (110) | 0.00113 [-0.0403, +0.0277] | 3.40% [-5.8550, +11.8922] |
| ema32__graph010 minus dual__cond100_graph010 | native_and_original:ema32 (107) | -0.01041 [-0.0307, +0.0078] | -1.62% [-7.3461, +3.2548] |
| ema32__joint0 minus dual__cond100 | native_and_original:ema32 (107) | -0.01153 [-0.0319, +0.0070] | 0.08% [-8.4726, +6.8375] |
