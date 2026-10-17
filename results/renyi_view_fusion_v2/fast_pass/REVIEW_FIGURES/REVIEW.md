# Renyi-view fusion v2 - FAST PASS review (Stage 3)

Reviewed 2026-09-13 from `results/renyi_view_fusion_v2/fast_pass/{METRICS.json, COMPARISON.csv, SUMMARY.csv, PB_CELLS.csv, DIAGNOSTICS.json, FIT_HEALTH.json}` only (json/csv; no pickle, no sqlite, no score arrays). Scope: 12 scored arms + 4 frozen references, 13769 answers, 145597 steps, development data. Protocol: `docs/experiments/RENYI_VIEW_FUSION_V2.md`.

**Verdict: PASS WITH CAVEATS.** All arithmetic/consistency checks (a)-(g), (j) and the SUMMARY.csv cross-check pass; (h)-(i) are readings. Caveats are listed at the end.

## Files

| file | content |
|---|---|
| `fig1_single_views_by_alpha.png` | PB all-8, within-AUC, pooled AUC, PRMScore of the six Renyi views vs alpha; varentropy15 reference line |
| `fig2_fused_vs_single.png` | forest plot of the 51 contrasts with a fused left side (PB pp / within-AUC), primaries as diamonds |
| `fig3_pb_cells.png` | per-cell PB F1 heatmap, 16 arms x 8 cells |
| `fig4_redundancy.png` | 12x12 fraction of answers with token-level \|Pearson\| > 0.95 |
| `fig5_weights.png` | mean standardized coefficients per column, R6__iu / R6_sel__iu / R6_sel__shrink / R6_sel__equal |
| `CHECKS.json` | machine-readable checks and disclosures |

## Point table (SUMMARY.csv)

| arm | PB all-8 % | PB Q4 | PB Q8 | within AUC | pooled AUC | PRMScore |
|---|---|---|---|---|---|---|
| view__H0.1 | 35.37 | 36.02 | 34.72 | 0.7425 | 0.7152 | 0.6331 |
| view__H0.25 | 35.52 | 36.34 | 34.70 | 0.7414 | 0.7131 | 0.6322 |
| view__H0.5 | 35.55 | 36.25 | 34.85 | 0.7371 | 0.7088 | 0.6296 |
| view__H1 | 35.44 | 36.40 | 34.49 | 0.7301 | 0.7027 | 0.6254 |
| view__H2 | 35.59 | 36.53 | 34.65 | 0.7242 | 0.6972 | 0.6213 |
| view__Hinf | 35.77 | 36.51 | 35.04 | 0.7178 | 0.6924 | 0.6158 |
| view__sel1 | 22.84 | 23.06 | 22.63 | 0.6975 | 0.7121 | 0.6119 |
| R6__equal | 35.62 | 36.58 | 34.67 | 0.7325 | 0.6776 | 0.5968 |
| R6__iu | 35.62 | 36.59 | 34.65 | 0.7310 | 0.6777 | 0.5975 |
| R6_sel__equal | 35.61 | 36.39 | 34.82 | 0.7352 | 0.6756 | 0.5949 |
| R6_sel__iu | 35.78 | 36.43 | 35.14 | 0.7338 | 0.6820 | 0.6028 |
| R6_sel__shrink | 35.75 | 36.41 | 35.09 | 0.7338 | 0.6814 | 0.6024 |
| entropy | 35.44 | 36.40 | 34.49 | 0.7301 | 0.7027 | 0.6254 |
| direct_iu | 34.50 | 35.31 | 33.69 | 0.7328 | 0.7038 | 0.6205 |
| ref__varentropy15 | 35.96 | 36.64 | 35.29 | 0.7378 | 0.7101 | 0.6258 |
| ref__varentropy15_iu | 35.35 | 35.90 | 34.80 | 0.7468 | 0.7103 | 0.6227 |

## Primary contrasts (97.5% paired source-group bootstrap, 10,000 draws, 6,030 common PRMB answers)

| contrast | PB delta pp [CI] | CI excl. 0 | within delta [CI] | CI excl. 0 | gained/lost |
|---|---|---|---|---|---|
| R6__iu - view__H1 | +0.178 [-0.278, +0.645] | no | +0.00089 [-0.00022, +0.00202] | no | 63/55 |
| R6__equal - view__H1 | +0.180 [-0.235, +0.603] | no | +0.00238 [+0.00123, +0.00357] | yes | 51/45 |
| R6_sel__iu - view__H1 | +0.339 [-0.139, +0.824] | no | +0.00368 [+0.00230, +0.00512] | yes | 68/53 |
| R6__iu - R6__equal | -0.002 [-0.205, +0.195] | no | -0.00149 [-0.00226, -0.00071] | yes | 13/11 |

## Checks

| check | result | detail |
|---|---|---|
| (a) contrast pb_delta = pb_all8[a] - pb_all8[b] | PASS | 57 contrasts, max abs diff 0.0e+00 (tol 1e-12); COMPARISON.csv equals METRICS to 0.0e+00 |
| (b) point delta inside its own CI | PASS | 57 contrasts x 2 endpoints, 0 violations |
| (c) pb_all8 / q4 / q8 = cell means | PASS | 16 arms, max abs diff all-8 0.0e+00, q4 0.0e+00, q8 0.0e+00 (tol 1e-12); PB_CELLS.csv matches to 0.0e+00 pp |
| (d) fused arms coverage 1.0, n_failed 0 | PASS | 5 arms, FIT_HEALTH + DIAGNOSTICS + telemetry agree; 13,769 fitted each; fit s: R6__equal 32, R6__iu 97, R6_sel__equal 19, R6_sel__iu 99, R6_sel__shrink 102 |
| (e) view__H1 = frozen entropy | PASS | pb_all8 diff 0, within diff 0, all 8 cells identical, PRMScore diff 0; pooled diff 8.44e-09 (tol 1e-8; recorded 8.4445e-09) |
| (f) single-view within-AUC monotone decreasing in alpha | PASS | 0.7425 > 0.7414 > 0.7371 > 0.7301 > 0.7242 > 0.7178 (alpha 0.1 -> inf); pooled AUC and PRMScore also strictly decreasing; PB is not monotone |
| (g) ci_level 0.975 primaries / 0.95 secondaries | PASS | 4 primaries match METRICS.primary, 53 secondaries, 0 violations; all 10,000 draws |
| (h) primary readings | READ | see table above: no primary has a PB interval excluding zero; within intervals exclude zero for R6__equal-H1 (+), R6_sel__iu-H1 (+) and R6__iu-R6__equal (-); R6__iu-H1 includes zero on both |
| (i) fused vs best single view (view__H0.1, within 0.7425) | READ | no fused arm exceeds it (max fused within 0.7352, R6_sel__equal); all five within CIs vs H0.1 are negative and exclude zero (-0.0074 to -0.0115); PB deltas vs H0.1 are +0.24 to +0.41 pp with all CIs including zero |
| (j) frozen references reproduce | PASS | ref__varentropy15 35.9610 / 0.737786 / 0.625781; ref__varentropy15_iu 35.3498 / 0.746824 / 0.622689 (tol 1e-4 pp / 1e-6) |
| (extra) SUMMARY.csv = METRICS | PASS | 16 arms, max abs diff 0.0e+00 |

## Disclosures read from DIAGNOSTICS.json / FIT_HEALTH.json

| item | value |
|---|---|
| \|Pearson\| > 0.95 fraction, adjacent orders | H0.1-H0.25 0.797, H0.25-H0.5 0.821, H0.5-H1 0.9985, H1-H2 1.000, H2-Hinf 1.000, H1-Hinf 0.619; H0.1-H0.5 0.0001, H0.1-H1 0.000 |
| \|Pearson\| > 0.99 fraction, adjacent orders | H0.1-H0.25 0.006, H0.25-H0.5 0.000, H0.5-H1 0.000, H1-H2 0.011, H2-Hinf 0.042, H1-Hinf 0.000 |
| H2 (K=15) vs frozen Renyi-2 (K=50) Pearson | mean 0.9993, min 0.9865 |
| condition number R6 / R6_sel / tail block (median, max) | 14316, 67823 / 14952, 125810 / 441, 5171; fraction R6_sel > 1e6: 0 |
| anchor correlation mean (raw column vs own varentropy15) | H0.1 0.806, H0.25 0.879, H0.5 0.860, H1 0.729, H2 0.607, Hinf 0.522, sel1 0.284, sel2 0.034, sel3 -0.025 |
| near-constant / dropped-by-zscore fractions | all zero |
| Hartley H0 | constant 2.7081 on every answer (max std 1.8e-15), diagnostics only |
| shrink alpha (R6_sel__shrink) | at 1.0 on 16.43% of answers; mean 0.382, median 0.235, min 0.031 |
| IU g2_hat | R6__iu min 0.250000000000000 max 0.250000000000000 (0.25 on every answer); R6_sel__iu mean 0.2480 min 0.0351; shrink mean 0.2489 min 0.0602 |
| orientation | global flip fraction: R6__equal 0.00000, R6__iu 0.00051, R6_sel__equal 0.00000, R6_sel__iu 0.00000, R6_sel__shrink 0.00000; mean column flips per answer: R6 arms 0.0014, R6_sel arms 1.065 |
| mean standardized coefficients (fig5) | R6__iu rises with alpha: 0.060/0.081/0.106/0.122/0.128/0.129; R6_sel__iu Renyi block 0.088-0.100, SEL 0.039 / 0.012 / 0.005; R6_sel__equal sel3 mean -0.085 (oriented negative on most answers) |
| not applicable / absent | R6__joint (two groups only), R6__shrink (== R6__iu) declared not applicable; R6_sel__joint not in this fast pass |
| runtime (fit s, contended machine) | R6__equal 32, R6__iu 97, R6_sel__equal 19, R6_sel__iu 99, R6_sel__shrink 102, view__H0.1 22, view__H0.25 3, view__H0.5 3, view__H1 3, view__H2 3, view__Hinf 3, view__sel1 3 |

## Caveats

- R6_sel__joint (the sixth fused arm in the protocol) is absent from this fast pass; every Joint contrast and Joint disclosure is not reviewed here. R6__joint and R6__shrink are declared not applicable.
- view__H1 pooled AUC differs from the frozen entropy row by 8.44e-09 (recorded in METRICS.view_h1_minus_frozen_entropy); all other H1/entropy metrics are bit-identical.
- Six views, two effective directions on most answers: H1-H2 and H2-Hinf are |Pearson| > 0.95 on 100% of answers, H0.5-H1 on 99.85%, H0.1-H0.25 on 79.7%; R6 condition number median 1.43e4 (max 6.8e4). The bank is not pruned by protocol; the fits are on near-collinear columns.
- R6_sel__shrink: Ledoit-Wolf alpha at 1.0 on 16.43% of answers (mean 0.382, median 0.235). IU g2_hat is 0.25 on every answer for R6__iu (min 0.24999..., max 0.25000...) and on the median answer for the R6_sel arms.
- Anchor orientation flips on average 1.07 columns per answer in the R6_sel arms (SEL block; R6_sel__equal mean coefficient on sel3 is -0.085) and 0.0014 in the R6 arms; global flips 0.05% (R6__iu) or 0.
- Every fused arm lowers PRMB pooled AUC (0.6756-0.6820) and PRMScore (0.5949-0.6028) below the frozen entropy row (0.7027 / 0.6254); these endpoints are secondary and carry no interval in COMPARISON.csv.
- All numbers are full cached development data (13,769 answers, 145,597 steps); nothing here is an untouched confirmation. fast_pass_scores_sha256 is null in METRICS.json.

Mismatches found: none. This review checks arithmetic, consistency and disclosures only; it draws no research conclusion beyond the readings in (h) and (i).
