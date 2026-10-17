# Red team: er_generality_v1 run_20260927 (commit 2b3a8e862)

- **A** independent recomputation: fresh code, its own Dawid-Skene EM (6 starts, all at one optimum) and a second tie key.
- **B** coverage: STEP_SCORES / FIT_MANIFEST / PARTITIONS, per fold, class, length tertile, concentration, ProcessBench cells.
- **C** nulls and mechanisms: 500-permutation within-answer shuffle and same-length swap nulls (own code, seeds 101/102),
  including the secondary bank B51.

Scripts in the session scratchpad (`rtg_A`, `rtg_B`, `rtg_C`).

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| 1. The DS filter drops only channels with true balanced accuracy < 0.52 and every channel < 0.475, on B20/B32/B51, in every fold | **confirmed, thin margin** | **A**: B20 exactly the 2 reversed channels; B32 the 4 strongly reversed always, the 0.48 one in 4/5 folds (3/5 with another tie key); B51 the claimed 8, highest dropped 0.51987 (margin 0.00013). Protocol P2 (drop every channel <= 0.5) fails on B32 in 2 folds. Prevalence 0.28-0.295 vs 0.137-0.141 |
| 2. The simple filter misfires on larger banks because the level family dominates the mean | **numbers confirmed, mechanism refuted, wording weakened** | **A**: on B32 it drops 4 channels (one reversed 0.48, the good 0.56 / AUC 0.640, two near chance) and misses the 4 strongest reversed; B51 misses all. **C**: the level family is 16% of the B32 sum variance; the decisions follow the non-level majority (Pearson 0.92 vs 0.66); the statistic is informative (Spearman 0.70/0.79 with single AUC) but misfires near zero. DS > simple filter +0.0152 (B32), +0.0103 (B51, ~46% positional), but -0.0018 on B20 |
| 3. DS filter gain over all-channel averaging: none on B20, +0.0035 on B32 (not positional), +0.0128 on B51 | **B20 confirmed (no effect); B32 confirmed but concentrated; B51 mostly positional** | **B**: B32 5/5 folds, 6/8 classes, 3/3 tertiles, but 86% of the gain from 61 answers (+0.0005 without them; 5%-trimmed +0.0026); B51 5/5, 7/8, 3/3, broad (5%-trimmed +0.0112). **C**: B32 swap-null mean -0.0116 (content +0.015; position-adjusted +0.0095); B51 swap-null mean +0.0089, so ~70% positional, content ~+0.0039 (position-adjusted +0.0058). ProcessBench mixed: B32 macro -0.40 points, B51 +0.63 (2 of 4 datasets; q4/q8 cells share answers) |
| 4. Binary partition beats continuous at equal group weights on every bank; both lose to plain averaging after the filter | **confirmed** | **B**: binary > continuous 5/5 folds on B13/B51, 4/5 on B20/B32 (B32 fold 3 identical partitions); grouping loss robust on B20/B51, fragile on B32 (+0.0003 after dropping the top 1%). **C**: not "fewer groups" (binary has more on B20/B51); grouping is a pure reweighting (fixed average weights reproduce every arm within 0.002); weight on the 10 strongest channels plain > binary > continuous on every bank; ~35% of the B32 binary-vs-continuous gain positional |
| 5. Partitions are identical across folds on B13 only | **confirmed** | **B**: B20 binary 7 groups in 4 folds, 4 in fold 1 (ARI 0.40); B32 differences come from the survivor set (fold 3); B51 binary 2 distinct partitions (one channel moves), continuous 4 distinct |
| 6. L-SML collapses on B32; the filter partly rescues it; it stays below averaging | **confirmed** | **C**: no global flip, no dominant group; 4 wrong-signed channels (incl. energy_innovation, top50_js) carry 16.6% of weight, near-chance groups most of the rest (true signs with fitted magnitudes 0.7209, plain average 0.7440); the filter removes the two reversed channels; the remaining DSF_lsml deficit (-0.029) is mostly positional (swap -0.0235) and magnitude on near-chance STFT channels |

## What the author's preliminary report got wrong
- It called the B51 gain "content" without a null (the runner computed nulls for the primary banks only); it is ~70% positional.
- It proposed a mechanism for the simple filter's failure (level-dominated mean) that the data refute.
- It said the simple filter "missed all four reversed channels" on B32; it caught the weakest one.
- It presented the B32 filter gain without its concentration (86% from 1% of answers).
