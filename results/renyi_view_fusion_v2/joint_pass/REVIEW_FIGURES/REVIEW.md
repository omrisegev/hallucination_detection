# Renyi-view fusion v2 - JOINT PASS review (Stage 3)

Reviewed 2026-09-14 from `results/renyi_view_fusion_v2/joint_pass/{METRICS.json, COMPARISON.csv, SUMMARY.csv, PB_CELLS.csv, DIAGNOSTICS.json, FIT_HEALTH.json, FEASIBILITY.json, RUN_STATE.json}` and `results/renyi_view_fusion_v2/fast_pass/METRICS.json` (json/csv only; the fast-pass `SCORES.npz` was hashed as bytes, never loaded; no pickle, no sqlite, no score arrays). Independent of the replay reviewer. Scope: one newly scored arm, `R6_sel__joint` = Joint L-SML, lambda 0, groups tail {H0.1, H0.25, H0.5} / head {H1, H2, Hinf} / selected {sel1, sel2, sel3}, plus the 12 fast-pass arms and 4 frozen references appended at evaluation; 13,769 answers, 145,597 steps; full cached development data under the frozen external gate. Protocol: `docs/experiments/RENYI_VIEW_FUSION_V2.md`. Pattern followed: `results/renyi_view_fusion_v2/fast_pass/REVIEW_FIGURES/REVIEW.md`.

**Verdict: PASS WITH CAVEATS.** All five hard checks (a)-(e) pass at machine precision; (f)-(h) are readings. The caveats (end of file) concern fit health - convergence 68.6% - the absence of any pre-registered contrast for the Joint arm, and the two secondary endpoints, not arithmetic.

## Files

| file | content |
|---|---|
| `fig1_joint_vs_others.png` | PB all-8 %, within-AUC, pooled AUC, PRMScore as bars for R6_sel__joint, R6_sel__iu, R6_sel__shrink, R6_sel__equal, R6__iu, view__H0.1, view__H1, ref__varentropy15, ref__varentropy15_iu (y-axes truncated; dotted line = Joint value) |
| `fig2_joint_contrasts.png` | forest plot of the 12 registered contrasts with R6_sel__joint on the left: PB delta pp with 95% CI and within-AUC delta with 95% CI, answers gained / lost; sel1 row off scale on PB |
| `fig3_joint_health.png` | summary fractions (converged, multistart pass, joint misfit <= hard misfit, condition > 1e12, map ridge positive, Jacobian full rank); mean standardized coefficient per column; saved quantiles of the joint vs hard-L-SML off-diagonal misfit (per-answer values are not in the JSON) |
| `CHECKS.json` | machine-readable output of every check below, including per-arm and per-contrast rows |

Palette: blue / orange / aqua / violet validated (adjacent CVD Delta E 9.2, normal-vision 27.6; aqua sits at 2.74:1 on the surface, relieved by direct value labels on every bar); blue / red for the signed coefficient bars (Delta E 21.6 CVD). Every bar and forest row is direct-labelled with its value, so identity never rests on colour alone.

## Point table (SUMMARY.csv, equal to METRICS at 0.0)

| arm | PB all-8 % | PB Q4 | PB Q8 | within AUC | pooled AUC | PRMScore |
|---|---|---|---|---|---|---|
| **R6_sel__joint** (this pass) | **35.49** | **36.30** | **34.68** | **0.7380** | **0.6761** | **0.5915** |
| R6_sel__iu | 35.78 | 36.43 | 35.14 | 0.7338 | 0.6820 | 0.6028 |
| R6_sel__shrink | 35.75 | 36.41 | 35.09 | 0.7338 | 0.6814 | 0.6024 |
| R6_sel__equal | 35.61 | 36.39 | 34.82 | 0.7352 | 0.6756 | 0.5949 |
| R6__iu | 35.62 | 36.59 | 34.65 | 0.7310 | 0.6777 | 0.5975 |
| R6__equal | 35.62 | 36.58 | 34.67 | 0.7325 | 0.6776 | 0.5968 |
| view__H0.1 | 35.37 | 36.02 | 34.72 | 0.7425 | 0.7152 | 0.6331 |
| view__H0.25 | 35.52 | 36.34 | 34.70 | 0.7414 | 0.7131 | 0.6322 |
| view__H0.5 | 35.55 | 36.25 | 34.85 | 0.7371 | 0.7088 | 0.6296 |
| view__H1 (= entropy) | 35.44 | 36.40 | 34.49 | 0.7301 | 0.7027 | 0.6254 |
| view__H2 | 35.59 | 36.53 | 34.65 | 0.7242 | 0.6972 | 0.6213 |
| view__Hinf | 35.77 | 36.51 | 35.04 | 0.7178 | 0.6924 | 0.6158 |
| view__sel1 | 22.84 | 23.06 | 22.63 | 0.6975 | 0.7121 | 0.6119 |
| direct_iu (frozen) | 34.50 | 35.31 | 33.69 | 0.7328 | 0.7038 | 0.6205 |
| ref__varentropy15 (frozen) | 35.96 | 36.64 | 35.29 | 0.7378 | 0.7101 | 0.6258 |
| ref__varentropy15_iu (frozen) | 35.35 | 35.90 | 34.80 | 0.7468 | 0.7103 | 0.6227 |

ProcessBench decomposition for the Joint arm (METRICS): raw exact 1,414 / 4,442 erroneous answers (0.3183), 246 correct peaks suppressed by the shared gate, 1,093 early / 1,935 late misses; clean accuracy 0.5008 (identical on every arm, shared external gate). R6_sel__iu: 1,409 raw exact, 225 suppressed, 1,174 early / 1,859 late. Per cell (Joint vs IU, F1 %): gsm8k_q4 44.16 / 44.91, gsm8k_q8 40.50 / 40.76, math_q4 34.93 / 34.45, math_q8 33.79 / 33.92, olympiadbench_q4 31.20 / 30.59, olympiadbench_q8 30.45 / 31.67, omnimath_q4 34.89 / 35.78, omnimath_q8 33.99 / 34.19 - Joint above IU on 2 of 8 cells.

## Checks

| check | result | detail |
|---|---|---|
| (a) every fast-pass arm's metrics identical in the joint-pass METRICS | PASS | 16 arms (12 scored + 4 frozen references), pb_all8 / prm_within / prm_pooled / prmscore_q08 plus pb_q4, pb_q8, valid_answers, prm_within_n, pb_raw_exact, pb_exact_count and all 8 cell F1: max abs diff 0.0 (tol 1e-12); none missing; the 57 contrasts shared with the fast pass also equal at 0.0 on delta, CI, level, gained, lost, primary flag |
| (b) contrast pb_delta = pb_all8[a] - pb_all8[b]; point deltas inside their own CI | PASS | 69 contrasts, max abs diff 0.0; 0 of 138 (PB, within) point deltas outside their CI; COMPARISON.csv equals METRICS to 1.4e-17 with the same 69-row set; 12 contrasts have R6_sel__joint on the left, all `primary = False`, all 95% level, 10,000 draws, 6,030 common PRMB answers |
| (c) pb_all8 / q4 / q8 = mean of cell F1 | PASS | 17 arms, max abs diff 5.6e-17 (tol 1e-12); PB_CELLS.csv (17 rows) and SUMMARY.csv (17 rows) equal METRICS to 5.6e-17 with the same arm set |
| (d) R6_sel__joint coverage 1.0, n_failed 0 | PASS | FIT_HEALTH coverage 1.0, n_failed 0, n_fitted 13,769, failure_reasons []; DIAGNOSTICS n_failed 0, n_fitted 13,769; telemetry failures []; METRICS valid_answers 13,769, pb_invalid 0, prm_valid_answers 6,969, prm_within_n 6,030; RUN_STATE COMPLETE 13,769 / 13,769 |
| (e) fast_pass_scores_sha256 non-null, roster == 'joint' | PASS | sha `232e3f99...4faf1` recorded, and it equals the actual sha256 of `fast_pass/SCORES.npz` computed here; roster `joint`; scored_methods `['R6_sel__joint']`; schema `renyi-view-fusion-v2`; METRICS.primary lists the four fast-pass pairs only - no Joint contrast is pre-registered primary |
| (f) readings | READ | below |
| (g) joint fit health | READ | below |
| (h) runtime | READ | below |

Mismatches found: none.

## (f) Readings - R6_sel__joint versus the named arms (95% paired source-group bootstrap; pooled / PRMScore are point differences without intervals)

| contrast | PB delta pp [CI] | sign / CI excl. 0 | within delta [CI] | sign / CI excl. 0 | gained / lost | pooled (pt) | PRMScore (pt) |
|---|---|---|---|---|---|---|---|
| Joint - R6_sel__iu | -0.29 [-0.95, +0.35] | - / no | +0.0042 [+0.0026, +0.0059] | + / yes | 109 / 125 | -0.0058 | -0.0114 |
| Joint - R6_sel__shrink | -0.26 [-0.93, +0.39] | - / no | +0.0042 [+0.0026, +0.0058] | + / yes | 111 / 126 | -0.0053 | -0.0109 |
| Joint - R6_sel__equal | -0.12 [-0.81, +0.60] | - / no | +0.0029 [+0.0011, +0.0048] | + / yes | 127 / 140 | +0.0005 | -0.0035 |
| Joint - view__H1 (entropy) | +0.04 [-0.52, +0.62] | + / no | +0.0079 [+0.0063, +0.0096] | + / yes | 79 / 80 | -0.0265 | -0.0340 |
| Joint - view__H0.1 | +0.12 [-0.34, +0.56] | + / no | -0.0045 [-0.0061, -0.0029] | - / yes | 58 / 53 | -0.0391 | -0.0417 |
| Joint - ref__varentropy15 | -0.47 [-1.62, +0.68] | - / no | +0.0003 [-0.0032, +0.0037] | + / no | 353 / 374 | -0.0340 | -0.0343 |
| Joint - ref__varentropy15_iu | +0.14 [-0.83, +1.11] | + / no | -0.0088 [-0.0114, -0.0061] | - / yes | 264 / 261 | -0.0341 | -0.0312 |

In words:

- **vs R6_sel__iu (same bank, IU-PCR weights):** the Joint arm is 0.29 pp lower on ProcessBench at the point level with an interval that includes zero, and 0.0042 higher on within-answer AUC with an interval that excludes zero. It gains 109 and loses 125 exact PB answers. On the two secondary endpoints without intervals it is lower (pooled -0.006, PRMScore -0.011). The same pattern holds against R6_sel__shrink. This is the same signature the varentropy-expansion Joint pass showed against IU (within interval excluding zero, PB inconclusive), except that there the PB point was positive.
- **vs R6_sel__equal (same bank, oriented equal weights):** PB -0.12 pp, interval includes zero; within +0.0029, interval excludes zero; pooled essentially equal (+0.0005), PRMScore -0.0035. The learned Joint weighting adds a within-answer margin over simple aggregation on this bank; it adds nothing measurable on ProcessBench.
- **vs view__H1 (Shannon entropy, the frozen entropy row):** PB +0.04 pp, interval includes zero (79 gained / 80 lost); within +0.0079, interval excludes zero. Pooled AUC and PRMScore are lower by 0.027 / 0.034 at the point level.
- **vs view__H0.1 (best single view on within-AUC):** PB +0.12 pp, interval includes zero; within -0.0045, interval excludes zero on the negative side. The Joint arm does not reach the best single Renyi view on within-answer ranking, as no fused arm in the fast pass did.
- **vs ref__varentropy15 (frozen Step-339 raw varentropy):** PB -0.47 pp, interval [-1.62, +0.68] includes zero (353 gained / 374 lost); within +0.0003, interval includes zero. Inconclusive on both primary endpoints; lower on pooled and PRMScore by 0.034 at the point level.
- **vs ref__varentropy15_iu:** PB +0.14 pp, interval includes zero; within -0.0088, interval excludes zero on the negative side.

No contrast with the Joint arm on the left has a ProcessBench interval excluding zero except the sel1 control (+12.65 pp [+10.43, +14.93], expected). None of these contrasts is pre-registered primary; they are all 95% secondary contrasts and are read as such.

## (g) Joint fit-health fractions (DIAGNOSTICS.json `fused_arms.R6_sel__joint`, FIT_HEALTH.json)

| item | value |
|---|---|
| fitted / failed | 13,769 / 0; coverage 1.0; active columns 9 on every answer |
| fraction converged (sweep cap not hit) | **0.6859** (about 4,325 answers did not converge; they were scored, not withheld) |
| fraction multistart audit pass | **0.6859** (identical to the converged fraction) |
| fraction joint misfit <= hard-L-SML misfit | 0.8908 |
| fraction model-covariance condition > 1e12 | 0.0630 (median 972, q90 6.95e4, max 1.28e20) |
| map condition after ridge | median 972, q90 1000, max 1000 (cap) |
| fraction map ridge positive | 1.0 |
| fraction Jacobian full rank | 1.0 |
| global loading cosine (min) | 1.0 on every answer where saved; **n = 9,522 only** (4,247 answers absent from this summary; the JSON does not say why) |
| orientation | global flip fraction 0.0; mean column flips per answer 1.065 (SEL block, as in the fast-pass R6_sel arms) |
| anchor column | H0.25 on 12,318 answers, H0.5 on 1,405, H0.1 on 46; never H1 / H2 / Hinf / sel |
| joint relative off-diagonal misfit | min 0.0053, q10 0.0091, median 0.0134, mean 0.0151, q90 0.0225, max 0.1312 |
| hard relative off-diagonal misfit | min 0.0053, q10 0.0092, median 0.0135, mean 0.0153, q90 0.0230, max 0.1239 |
| mean standardized coefficients | H0.1 -0.030, H0.25 +0.171, **H0.5 +0.703**, H1 +0.153, H2 -0.021, Hinf -0.027, sel1 +0.003, sel2 -0.001, sel3 -0.001 |
| mean absolute share | H0.5 0.626, H0.25 0.206, H1 0.100, H0.1 0.029, H2 0.020, Hinf 0.017, sel1 0.002, sel2 0.001, sel3 0.001 |
| mean negative share | 0.067 |

The joint and hard misfit summaries are nearly identical at every saved quantile; the joint fit is at or below the hard fit on 89% of answers. The coefficient mass sits on H0.5 and H0.25 (83% of absolute share); the head block H2 / Hinf and the three SEL columns carry almost nothing, and H0.1, H2, Hinf have negative mean coefficients.

## (h) Runtime

| item | value |
|---|---|
| fit_seconds (FIT_HEALTH = telemetry) | 87,857.7 s = **24.40 h** for 13,769 answers, 6.38 s per answer |
| smoke projection (FEASIBILITY.json, 27 answers, 1 worker) | 78,462.9 s projected, 5.70 s per answer; actual / projected 1.12 |
| fast-pass fit seconds on the same machine | R6_sel__iu 98.7 s, R6_sel__shrink 102.2 s, R6_sel__equal 19.1 s, R6__iu 97.1 s, single views 2.6-21.7 s |
| Joint / R6_sel__iu | about 890x |

## Caveats

- **Convergence 68.6%.** About 4,325 of 13,769 Joint fits hit the 5,000-sweep cap, and the multistart audit passes on exactly the same fraction. The varentropy-expansion Joint pass on its two banks converged on 99.5-99.6% of answers, so this bank is markedly harder for the solver (six near-collinear Renyi columns; the fast-pass review recorded H1-H2 and H2-Hinf at |Pearson| > 0.95 on 100% of answers). Non-converged fits were scored and flagged, not withheld, and no per-answer convergence flag is in the JSON, so this review cannot say whether the within-AUC margin over IU comes from converged answers. The loading-cosine summary covers only 9,522 answers, also unexplained in the JSON.
- **No pre-registered primary contrast involves the Joint arm.** The 12 Joint contrasts are 95% secondary contrasts; the four primaries are the fast-pass pairs. Any reading of the Joint arm is therefore a secondary reading.
- **Two endpoints move in opposite directions.** Against IU, shrink and equal on the same bank the within-AUC interval excludes zero on the positive side while the ProcessBench point is negative with an interval including zero, and pooled AUC / PRMScore (no intervals) are lower. The Joint arm is above IU on 2 of 8 PB cells. Nothing here is a consistent two-task gain.
- **Best single view not reached on within-AUC**: view__H0.1 (0.7425) and ref__varentropy15_iu (0.7468) are above the Joint arm (0.7380) with intervals excluding zero; vs ref__varentropy15 both primary endpoints are inconclusive.
- **Secondary endpoints below the frozen entropy row**: pooled AUC 0.6761 and PRMScore 0.5915 are the lowest of all 17 arms except R6_sel__equal on pooled (0.6756); the entropy row has 0.7027 / 0.6254.
- **Runtime**: 24.4 h against under two minutes for the IU arm on the same bank, 12% over the smoke projection.
- **Weight concentration**: 83% of absolute coefficient share on H0.5 + H0.25, SEL columns near zero, negative mean coefficients on H0.1 / H2 / Hinf; the anchor is H0.25 on 89% of answers. These are descriptive disclosures, not a diagnosed cause.
- All numbers are full cached development data (13,769 answers, 145,597 steps) under a frozen external gate; nothing here is an untouched confirmation.

This review checks arithmetic, consistency and disclosures only; it draws no research conclusion beyond the readings in (f)-(h).
