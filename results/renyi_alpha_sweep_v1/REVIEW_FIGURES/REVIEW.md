# Renyi ALPHA SWEEP v1 - results review (Stage 3b)

Reviewed 2026-09-13 from `results/renyi_alpha_sweep_v1/{METRICS.json, COMPARISON.csv, SUMMARY.csv, PB_CELLS.csv, DIAGNOSTICS.json, FIT_HEALTH.json, SELECTION.json}` only (json/csv; no pickle, no sqlite, no score arrays). Scope: 31 single-view arms (20 Renyi H_alpha incl. the alpha->0 limit, 11 escort varentropy VE_alpha) + 4 frozen references; 13,769 answers, 145,597 steps; full cached development data. No pre-registered primary; all 144 contrasts at 95 %, 10,000 paired source-group bootstrap draws, 6,030 common PRMB answers. Definitions: `spectral_utils/renyi_alpha_sweep.py`.

**Verdict: PASS WITH CAVEATS.** Checks (a)-(g), (j), (k) and the SUMMARY/PB_CELLS cross-checks pass; (h)-(i) are readings. Caveats at the end.

## Files

| file | content |
|---|---|
| `fig1_renyi_curve.png` | H family: PB all-8, within, pooled, PRMScore vs alpha (log-x; limit / inf at the ends; alpha=1 marked; varentropy15 line) |
| `fig2_ve_curve.png` | VE family: same four panels (0 at left) + flip fraction per alpha; ve0.5 boundary marked |
| `fig3_contrasts.png` | forest plot: 13 named contrasts + all 19 H-adjacent + all 10 VE-adjacent pairs (PB pp / within), diamond = CI excludes 0 |
| `fig4_pb_cells.png` | per-cell PB F1 heatmap, 10 arms x 8 cells |
| `fig5_selection_stability.png` | fold-wise within argmax, cell-wise PB argmax, cross-fitted vs oracle per fold |
| `CHECKS.json` | machine-readable checks, per-contrast/per-arm detail, disclosures |

## Curves (SUMMARY.csv; PB %, within / pooled AUC, PRMScore)

| H alpha | 0 lim | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 1 (=entropy) | 2 | 4 | 8 | inf | ref varentropy15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PB | 35.53 | 35.53 | 35.50 | 35.53 | 35.37 | 35.52 | 35.55 | 35.42 | 35.44 | 35.59 | 35.80 | 35.88 | 35.77 | 35.96 |
| within | .7440 | .7440 | .7438 | .7433 | .7425 | .7414 | .7371 | .7330 | .7301 | .7242 | .7207 | .7192 | .7178 | .7378 |
| pooled | .7162 | .7162 | .7161 | .7158 | .7152 | .7131 | .7088 | .7053 | .7027 | .6972 | .6942 | .6930 | .6924 | .7101 |
| PRMScore | .6334 | .6334 | .6334 | .6334 | .6331 | .6322 | .6296 | .6268 | .6254 | .6213 | .6177 | .6160 | .6158 | .6258 |

| VE alpha | 0 | 0.1 | 0.25 | 0.5 | 0.75 | 1 (=varentropy15) | 1.5 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PB | 35.57 | 35.77 | 34.89 | 36.20 | 36.76 | 35.96 | 35.46 | 35.01 | 34.98 | 34.06 | 34.17 |
| within | .7534 | .7482 | .7352 | .6788 | .7323 | .7378 | .7333 | .7259 | .7136 | .7061 | .6824 |
| pooled | .7231 | .7201 | .7089 | .6341 | .7043 | .7101 | .7049 | .6986 | .6876 | .6790 | .6455 |
| flip % | 100 | 100 | 99.98 | 1.66 | 0 | 0 | 0 | 0 | 0.11 | 0.25 | 1.38 |

## Checks

| check | result | detail |
|---|---|---|
| (a) pb_delta = pb_all8[a] - pb_all8[b] | PASS | 144 contrasts, max abs diff 0.0 (tol 1e-12); COMPARISON.csv row set equals METRICS, csv vs METRICS max diff 0.0 on delta/CI (pp and within) |
| (b) point delta inside own CI | PASS | 144 x 2 endpoints, 0 violations |
| (c) pb_all8 = mean of 8 cells; q4/q8 = mean of 4 | PASS | 35 arms, max diff 0.0 all-8 / q4 / q8 (tol 1e-12); PB_CELLS.csv matches to 0.0 pp |
| (d) coverage 1.0, n_failed 0 | PASS | 31 scored arms in FIT_HEALTH, all coverage 1.0, n_failed 0, n_fitted 13,769, telemetry failures empty; valid_answers 13,769 for all 35 arms; fit 3.8-5.7 s per arm |
| (e) H1 = entropy; ve1 = ref__varentropy15 | PASS | H1-entropy: pb_all8 0, within 0, PRMScore 0, all 8 cells 0; pooled 8.44e-09 (as recorded in METRICS). ve1-ref__varentropy15: 0 on pb_all8, within, pooled, PRMScore and all 8 cells (tol 1e-9) |
| (f) H-family within monotone non-increasing (incl. limit) | PASS | strictly decreasing: .74397 (lim) > .74396 > .74382 > .74368 > .74326 > .74252 > .74214 > .74214 > .74139 > .74080 > .73943 > .73711 > .73298 > .73011 (H1) > .72651 > .72416 > .72194 > .72070 > .71917 > .71784 (inf). Pooled: strictly decreasing. PRMScore: NOT monotone (a0.001 .63337 < a0.01 .63341 < a0.02 .63343; 1e-5 scale). PB: not monotone (12 adjacent rises) |
| (g) ci_level 0.95, 10,000 draws | PASS | all 144: ci_level 0.95, bootstrap_draws 10000, prm_valid_bootstrap_draws 10000, primary False; METRICS.primary = [] |
| (h) reading: H vs H1 | READ | within excludes 0 for ALL 19 contrasts: positive for every alpha < 1 (lim +0.01386 [+0.01139, +0.01639] down to a0.75 +0.00287 [+0.00195, +0.00379]); negative for every alpha > 1 (a1.5 -0.00360 ... Hinf -0.01227 [-0.01446, -0.01007]). PB: NO contrast excludes 0 (range -0.07 pp (a0.1) to +0.44 pp (a8, CI [-0.27, +1.13])). Adjacent-H pairs: within excludes 0 for 16/19 (not lim-a0.001, a0.001-a0.01, a0.15-a0.2); PB excludes 0 only for a0.05-a0.1 (+0.16 pp [+0.01, +0.40]) |
| (i) reading: VE family | READ | vs ve1 - ve0: PB -0.40 [-1.51, +0.69] incl 0, within +0.01562 [+0.01214, +0.01914] pos; ve0.1: PB -0.19 incl 0, within +0.01039 pos; ve0.75: PB +0.80 [-0.20, +1.81] incl 0, within -0.00548 [-0.00892, -0.00196] neg. vs H1 - ve0: PB +0.12 incl 0, within +0.02329 [+0.01985, +0.02683] pos; ve0.1: PB +0.33 incl 0, within +0.01807 pos; ve0.75: PB +1.32 [-0.05, +2.71] incl 0, within +0.00219 [-0.00338, +0.00786] incl 0. ve0-ve0.1: PB -0.21 incl 0, within +0.00523 pos. Only ve4 and ve8 have PB CIs excluding 0 (negative vs both ve1 and H1). **ve0.5 is the orientation boundary**: flip 1.66 %, within .6788 (lowest of the family), pooled .6341, PB 36.20; mean anchor Pearson +0.45, mean Spearman with H1 +0.56 / with limit +0.67 (vs -0.90 / -0.92 for ve0.1 and +0.85 / +0.90 for ve0.75); ve0.25-ve0.5 within +0.058 and ve0.5-ve0.75 within -0.054, both CIs exclude 0. Adjacent-VE pairs: within excludes 0 for all 10; PB excludes 0 for ve0.1-ve0.25 (+0.88 pp [+0.22, +1.56]) and ve3-ve4 (+0.92 pp [+0.21, +1.65]) only |
| (j) selection stability | PASS | Argmax recomputed from METRICS agrees with SELECTION.json for both families (H: PB a8, within/pooled H0lim, PRMScore a0.02; VE: PB ve0.75, within/pooled/PRMScore ve0). H fold-wise within argmax: [a0.001, H0lim, H0lim, H0lim, a0.001] (folds 0-4; n 1404/1378/1356/1424/1407); cell-wise PB argmax: [a0.05, a0.15, Hinf, a8, a8, Hinf, a8, a0.15] (margins over a8: 1.90, 0.77, 0.16, 0, 0, 0.13, 0, 0.92 pp). VE fold-wise: [ve0 x5]; cell-wise: [ve1, ve0.1, ve4, ve0.5, ve0.75, ve0.75, ve0.25, ve0.1] (margins over ve0.75: 0.37, 1.01, 0.31, 0.38, 0, 0, 2.77, 1.11 pp). Cross-fitted within: H 0.743876 vs oracle 0.743934 (NOT equal; gap 5.7e-5, choices [H0lim, H0lim, a0.001, a0.001, H0lim]); VE 0.753363 = oracle exactly (ve0 chosen on every fold). Curve, cell F1 and global-best F1 entries all match METRICS |
| (k) flip fractions | PASS | 100 %: ve0, ve0.1 (a fixed negative sign is equivalent). 0 %: all 20 H arms, ve0.75, ve1, ve1.5, ve2. Between: ve0.25 99.98 %, ve0.5 1.66 %, ve3 0.11 %, ve4 0.25 %, ve8 1.38 % |
| (extra) SUMMARY.csv = METRICS | PASS | 35 rows, max abs diff 0.0; same arm set |

## Disclosures (DIAGNOSTICS.json / METRICS.json)

| item | value |
|---|---|
| mean within-answer Spearman with the limit view | a0.001 .99995, a0.1 .9988, a0.5 .9680, H1 .9391, Hinf .9180; fraction of answers > 0.99: a0.1 1.00, a0.2 .947, a0.3 .279, a0.5 .005, H1 0 |
| mean within-answer Spearman with H1 | H0lim .939, a0.5 .993, a0.75 .9988, a1.5 .9988, Hinf .994; ve0 -.765, ve0.1 -.901, ve0.25 -.634, ve0.5 +.565, ve0.75 +.847, ve1 +.924 |
| mean anchor Pearson (raw column vs own varentropy15) | H0lim .788, a0.3 .887 (max), H1 .729, Hinf .522; ve0 -.652, ve0.25 -.788, ve0.5 +.452, ve0.75 +.920, ve8 +.189 |
| mean per-answer std of raw column | a0.001 0.005, a0.1 0.45, H1 0.50, Hinf 0.29; ve0.1 34.4, ve0 9.8, ve1 0.45, ve8 0.002; near-constant fraction 0 everywhere |
| ref__varentropy15 vs historical `token_varentropy` row in METRICS | differs: PB 0.29 pp, within .0047, pooled .0056, PRMScore .0070 (two different frozen reference bundles; not a check target) |
| fast_pass_scores_sha256 | null |

## Caveats

- This is a label-guided one-parameter search on development data (module docstring). Curves and argmaxes are development evidence; no arm is promoted here and nothing is an untouched confirmation.
- No H-family contrast against H1 excludes zero on PB; every within-AUC gain at alpha < 1 comes with a PB point change inside [-0.8, +0.8] pp. The PB curve is non-monotone and its per-cell argmax scatters across a0.05 ... Hinf.
- The near-limit views (a0.001 ... a0.1) are rank-equivalent to the limit on essentially every answer (Spearman > 0.99 on 100 %), so their contrasts to each other are not independent evidence; a0.001 has mean raw std 0.005 (harmless for a single ranked view, relevant if these columns are ever fused unstandardized).
- VE family: ve0/ve0.1 rely on a 100 % anchor flip (equivalently a fixed negative sign); ve0.25 flips 99.98 %; ve0.5 sits at the boundary with a 1.66 % flip and the family's lowest within/pooled/PRMScore while having a high PB point (36.20). PB argmax ve0.75 has within below ve1 (CI excludes 0) and a PB CI vs both ve1 and H1 that includes 0.
- PRMScore is not strictly monotone in the H family (two 1e-5-scale rises between a0.001 and a0.02); within and pooled are strictly decreasing.
- H-family cross-fitted within (0.743876) is 5.7e-5 below the per-fold oracle (0.743934); VE cross-fitted equals its oracle. Fold-wise/cross-fitted values are taken from SELECTION.json and were verified for internal consistency only; per-fold recomputation would need the score arrays, which this review did not read.
- H1 pooled AUC differs from the frozen entropy row by 8.44e-09 (recorded in METRICS); all other identity metrics are bit-identical.

Mismatches found: none. This review checks arithmetic, consistency, monotonicity and disclosures; it draws no research conclusion beyond the readings in (h)-(k).
