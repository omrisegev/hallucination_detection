# S1 Local feature and locator screen

**Verdict: `PARITY_WITH_DIRECT_COMPETITOR`.**

Direct Tier-A reference on the same development rows: `step272_twohead_replay`.
Frozen S1 selection: `l_family6__level__step_top5mean`.

| method | F1 | tier | delta vs direct bar | 95% CI |
|---|---:|---|---:|---|
| qwen_prm | 0.8100 | B | — | — |
| qwen72b_critic | 0.6847 | B | — | — |
| l_family6__level_innovation_shortlong__peak | 0.3548 | A | +0.0044 | [-0.0776, +0.0848] |
| l_family6__level__step_top5mean | 0.3517 | A | +0.0014 | [-0.0708, +0.0705] |
| step272_twohead_replay | 0.3503 | A | — | — |
| l_raw9__level__persistent_q90_3 | 0.3468 | A | -0.0035 | [-0.0793, +0.0665] |
| l_raw9__level_innovation_shortlong__step_top5mean | 0.3466 | A | -0.0038 | [-0.0809, +0.0757] |
| max_entropy__step_top5mean | 0.3456 | A | -0.0047 | [-0.0695, +0.0639] |
| l_raw9__level__step_top5mean | 0.3445 | A | -0.0058 | [-0.0820, +0.0672] |
| max_entropy__persistent_q90_3 | 0.3404 | A | -0.0099 | [-0.0857, +0.0672] |
| l_raw9__level__peak | 0.3389 | A | -0.0114 | [-0.0398, +0.0131] |
| l_family6__level_shortlong__peak | 0.3370 | A | -0.0133 | [-0.0956, +0.0643] |
| l_family6__level_innovation__step_top5mean | 0.3344 | A | -0.0160 | [-0.0998, +0.0678] |
| l_raw7_opened_drop__level__step_top5mean | 0.3304 | A | -0.0199 | [-0.0952, +0.0495] |
| l_broad28__level__peak | 0.3235 | A | -0.0269 | [-0.0922, +0.0406] |
| l_family6__level__persistent_q90_3 | 0.3223 | A | -0.0280 | [-0.1131, +0.0579] |
| l_raw9__innovation__step_top5mean | 0.3177 | A | -0.0327 | [-0.1123, +0.0501] |
| l_raw9__level_innovation__persistent_q90_3 | 0.3135 | A | -0.0368 | [-0.1237, +0.0471] |
| l_family6__shortlong__step_top5mean | 0.3105 | A | -0.0398 | [-0.1253, +0.0415] |
| l_raw9__level_innovation__step_top5mean | 0.3103 | A | -0.0401 | [-0.1157, +0.0385] |
| l_family6__level_innovation_shortlong__step_top5mean | 0.3096 | A | -0.0407 | [-0.1221, +0.0391] |
| l_raw7_opened_drop__level__persistent_q90_3 | 0.3060 | A | -0.0444 | [-0.1231, +0.0299] |
| l_family6__level__peak | 0.3057 | A | -0.0447 | [-0.1053, +0.0227] |
| l_family6__innovation__peak | 0.3056 | A | -0.0448 | [-0.1386, +0.0462] |
| l_raw9__level_shortlong__step_top5mean | 0.3055 | A | -0.0449 | [-0.1207, +0.0289] |
| l_family6__level_innovation__peak | 0.3040 | A | -0.0463 | [-0.1310, +0.0342] |
| l_family6__level_shortlong__step_top5mean | 0.3038 | A | -0.0466 | [-0.1233, +0.0296] |
| l_raw7_opened_drop__level__peak | 0.3027 | A | -0.0477 | [-0.1032, +0.0048] |
| l_family6__shortlong__peak | 0.3022 | A | -0.0481 | [-0.1343, +0.0395] |
| l_broad28__level__step_top5mean | 0.3001 | A | -0.0502 | [-0.1236, +0.0180] |
| l_raw9__level_innovation__peak | 0.2986 | A | -0.0517 | [-0.1370, +0.0275] |
| max_entropy__peak | 0.2985 | A | -0.0518 | [-0.1064, -0.0001] |
| gl_liu_v1_replay | 0.2939 | A | -0.0564 | [-0.1251, +0.0055] |
| l_raw9__level_shortlong__peak | 0.2904 | A | -0.0599 | [-0.1372, +0.0181] |
| l_family6__innovation__step_top5mean | 0.2904 | A | -0.0600 | [-0.1502, +0.0282] |
| l_broad28__level_shortlong__step_top5mean | 0.2899 | A | -0.0604 | [-0.1429, +0.0137] |
| l_broad28__level_innovation__persistent_q90_3 | 0.2879 | A | -0.0624 | [-0.1573, +0.0330] |
| l_broad28__level_innovation__step_top5mean | 0.2877 | A | -0.0627 | [-0.1434, +0.0197] |
| l_raw9__shortlong__peak | 0.2861 | A | -0.0642 | [-0.1517, +0.0239] |
| mind_the_gap | 0.2810 | A | -0.0693 | [-0.1351, -0.0040] |
| l_broad28__level_innovation__peak | 0.2807 | A | -0.0697 | [-0.1528, +0.0179] |
| l_broad28__level_shortlong__peak | 0.2804 | A | -0.0699 | [-0.1495, +0.0077] |
| l_raw9__shortlong__step_top5mean | 0.2783 | A | -0.0720 | [-0.1550, +0.0124] |
| l_core5__persistent_q90_3 | 0.2781 | A | -0.0723 | [-0.1709, +0.0286] |
| l_family6__level_shortlong__persistent_q90_3 | 0.2774 | A | -0.0729 | [-0.1555, +0.0146] |
| l_raw9__level_shortlong__persistent_q90_3 | 0.2763 | A | -0.0740 | [-0.1689, +0.0165] |
| l_broad28__innovation__persistent_q90_3 | 0.2749 | A | -0.0754 | [-0.1627, +0.0151] |
| l_raw9__level_innovation_shortlong__peak | 0.2729 | A | -0.0774 | [-0.1616, +0.0057] |
| l_broad28__shortlong__peak | 0.2721 | A | -0.0783 | [-0.1657, +0.0056] |
| l_broad28__innovation__peak | 0.2687 | A | -0.0817 | [-0.1699, -0.0010] |
| l_broad28__level_innovation_shortlong__peak | 0.2661 | A | -0.0843 | [-0.1625, -0.0123] |
| l_broad28__innovation__step_top5mean | 0.2657 | A | -0.0847 | [-0.1759, +0.0065] |
| l_broad28__shortlong__step_top5mean | 0.2633 | A | -0.0871 | [-0.1717, -0.0040] |
| l_raw9__innovation__peak | 0.2632 | A | -0.0871 | [-0.1722, -0.0051] |
| l_family6__level_innovation__persistent_q90_3 | 0.2629 | A | -0.0874 | [-0.1806, +0.0081] |
| l_core5__step_top5mean | 0.2612 | A | -0.0892 | [-0.1734, -0.0073] |
| l_broad28__level_innovation_shortlong__step_top5mean | 0.2574 | A | -0.0930 | [-0.1852, -0.0051] |
| l_broad28__level_shortlong__persistent_q90_3 | 0.2513 | A | -0.0991 | [-0.1868, -0.0144] |
| l_raw9__innovation__persistent_q90_3 | 0.2464 | A | -0.1039 | [-0.1990, -0.0139] |
| l_family6__innovation__persistent_q90_3 | 0.2428 | A | -0.1075 | [-0.1969, -0.0149] |
| l_core5__peak | 0.2416 | A | -0.1087 | [-0.1895, -0.0274] |
| l_family6__level_innovation_shortlong__persistent_q90_3 | 0.2351 | A | -0.1152 | [-0.2142, -0.0188] |
| l_raw9__level_innovation_shortlong__persistent_q90_3 | 0.2345 | A | -0.1158 | [-0.2117, -0.0168] |
| l_family6__shortlong__persistent_q90_3 | 0.2247 | A | -0.1256 | [-0.2106, -0.0374] |
| l_broad28__shortlong__persistent_q90_3 | 0.2245 | A | -0.1258 | [-0.2107, -0.0357] |
| l_broad28__level__persistent_q90_3 | 0.2133 | A | -0.1371 | [-0.2381, -0.0264] |
| l_broad28__level_innovation_shortlong__persistent_q90_3 | 0.2132 | A | -0.1372 | [-0.2276, -0.0435] |
| l_raw9__shortlong__persistent_q90_3 | 0.1950 | A | -0.1553 | [-0.2454, -0.0620] |
| qwen3_judge_control | 0.1118 | B | — | — |

Tier-B critic/PRM rows are same-row compute ceilings and are not used as same-access deltas. The full report keeps every candidate; selection does not hide losing variants.
