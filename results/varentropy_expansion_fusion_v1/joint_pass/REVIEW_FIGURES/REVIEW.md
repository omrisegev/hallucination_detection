# Varentropy-expansion (cross-rank) fusion v1 - STAGE 2 review (fast pass + Joint pass + corrected supervised diagnostic)

Reviewed 2026-09-13 from saved, already-reviewed outputs only:
`joint_pass/{METRICS.json, COMPARISON.csv, PB_CELLS.csv, FIT_HEALTH.json, RESULT_REVIEW.json, RUN_STATE.json, SCORES.npz}`,
`fast_pass/{METRICS.json, SCORES.npz (hash only), REVIEW_FIGURES/CHECKS.json}` (consistency checks only),
`supervised/{METRICS.json, CONTRASTS.json, correction_20260913/{METRICS_CORRECTED.json, THRESHOLDS.json, PROVENANCE.json, FIT_HEALTH_SUPERVISED.json}}`,
and the frozen step labels / offsets in `results/localization_full_benchmark_v3/evaluation/JOINED.{json,npz}` (main repo) for the within-AUC
recomputation. No benchmark pickle and no sqlite checkpoint was opened; one process; peak memory well under 1.5 GB (SCORES.npz 31 MB + JOINED 10 MB).
Scope: 21 unsupervised arms (19 fast-pass + 2 Joint) on 13,769 answers / 145,597 steps, 0 declared failures, plus the 2 supervised arms.
**The two Joint arms on the non-selected banks (`B2__joint`, `B2d__joint`) remain PENDING** (deferred for compute, not dropped) and appear on
the figures only as "deferred" slots. All numbers are cached development data; nothing here is an untouched confirmation.
The note under review is `docs/research_notes/VARENTROPY_EXPANSION_STAGE2_INTERIM_2026-09-12.md` (Joint, fast-pass, supervised and correction sections).

Files in this directory:

| file | content |
|---|---|
| `fig_architecture_complete.png` | same bank across the five solvers (identity-sign equal / oriented equal / IU / shrink / Joint) for B2d_sel and B2_sel (secondary B2d / B2 in lighter marks), PB all-8 and PRMB within-AUC, entropy / varentropy15 / longest-step lines, Joint on B2 / B2d marked deferred |
| `fig_joint_contrasts_forest.png` | forest: primary (97.5%), Joint bank contrast, Joint - IU per bank, Joint - shrink per bank (point only; not a registered contrast), Joint arms vs entropy and vs varentropy15; PB pp and within deltas with CIs, answers gained / lost |
| `fig_joint_fit_health.png` | converged / multistart / condition > 1e12 / n < p / orientation-flip rates, anchor-correlation and fit-time summaries, other per-answer diagnostics (aggregates from FIT_HEALTH.json) |
| `fig_supervised_calibration.png` | per bank, per fold original vs corrected q = 0.8 thresholds; original vs corrected PRMScore; supervised vs unsupervised (IU / Joint on the same banks) PB / within points |
| `fig_stage2_summary.png` | every Stage-2 arm (19 fast + 2 Joint + 2 supervised) as PB vs within; hue = solver family, marker = bank, star = supervised, references as crosses / lines |
| `CHECKS.json` | machine-readable output of every check below |
| `REVIEW.md` | this file |

Palette: banks blue (B2d_sel) / orange (B2_sel) - validated all-pairs (worst CVD Delta E 24.7, normal 33.6). Solver families red / aqua / green /
yellow / violet - passes the light-mode normal-vision floor all-pairs (worst 15.6) with the CVD all-pairs check in the 6-8 warn band (6.9, red vs aqua),
covered by the mandatory secondary encoding (marker shape per bank, direct labels on the named points, tables in this file). Yellow / aqua sit below
3:1 on the surface; every value they carry is also in the tables here.

## Readings per figure

### fig_architecture_complete.png (same bank, five solvers)

With the Joint slots filled, the ordering within a bank is unchanged on within-AUC and on PB the Joint arm is now the highest solver on both
primary banks: B2d_sel 35.15 PB / 0.7371 within (IU 34.75 / 0.7336, shrink 34.66 / 0.7314, oriented equal 34.73 / 0.7344, identity-sign equal
32.80 / 0.7321) and B2_sel 34.37 / 0.7293 (IU 33.95 / 0.7270, shrink 34.06 / 0.7274, oriented equal 34.38 / 0.7291, identity-sign equal 34.08 /
0.7241). The bank effect is still larger than the solver effect: every B2_sel solver sits 0.3-0.8 pp and 0.003-0.008 within below its B2d_sel
counterpart (identity-sign equal excepted on PB). Neither Joint arm reaches entropy (35.44 / 0.7301) on PB or varentropy15 (35.96 / 0.7378) on
either endpoint at the point level; B2d_sel Joint is 0.29 pp below entropy and 0.81 pp below varentropy15 on PB, and 0.0006 below varentropy15
on within. The Joint slots for B2 and B2d are empty (deferred), so the secondary-bank ladder is complete only for the four non-Joint solvers.

### fig_joint_contrasts_forest.png (registered Joint-pass contrasts)

Bank: adding the 105 P_ij products under Joint gives PB -0.78 pp [-1.29, -0.30] (48 answers gained, 82 lost) and within -0.0079 [-0.0100, -0.0058],
the same size and sign as the pre-registered primary under IU (-0.80 pp [-1.31, -0.30] at 97.5%; -0.0066 [-0.0089, -0.0043]). Architecture:
Joint - IU is +0.40 pp [-0.00, +0.81] / +0.0036 [+0.0022, +0.0049] on B2d_sel (67 gained / 46 lost) and +0.42 pp [+0.05, +0.79] / +0.0023
[+0.0013, +0.0034] on B2_sel (52 / 31); the within intervals exclude zero on both banks, the PB interval excludes zero on B2_sel and touches zero
(lower bound -6.8e-6 pp) on B2d_sel. Joint - shrink is NOT a registered contrast: the point differences are +0.49 pp / +0.0058 (B2d_sel) and
+0.32 pp / +0.0019 (B2_sel), drawn hollow without intervals. References: B2d_sel Joint vs entropy is inconclusive on PB (-0.29 pp [-1.09, +0.50])
and higher on within (+0.0070 [+0.0038, +0.0103]); vs varentropy15 both endpoints are inconclusive (-0.81 pp [-2.00, +0.36]; -0.0006 [-0.0051,
+0.0037]). B2_sel Joint is below entropy on PB (-1.07 pp [-1.94, -0.21]; within inconclusive -0.0008 [-0.0048, +0.0031]) and below varentropy15
on both endpoints (-1.59 pp [-2.85, -0.33]; -0.0085 [-0.0137, -0.0034]).

### fig_joint_fit_health.png (Joint disclosures)

All 13,769 answers fitted on both banks with 0 declared failures; convergence (5,000-sweep cap not hit) 99.56% (B2d_sel) / 99.46% (B2_sel),
multistart audit pass 99.48% / 99.43%; non-converged fits were scored and flagged, not withheld. The model-covariance condition exceeds 1e12
on 1.63% / 5.62% of answers (median 4.1e3 / 4.8e4, max 8e19 / 2e20) and is absorbed by the analytic ridge to a post-ridge map condition of
1e3 (median map ridge 0.016 / 0.106). n < p fits: 1 (B2d_sel) / 705 = 5.12% (B2_sel, the same 705 as B2_sel IU). Orientation flips 0.01% /
0.09%. The fused score's anchor correlation falls from 0.806 (B1_hist IU, median) to 0.589 (B2d_sel Joint) to 0.403 (B2_sel Joint), the
same ladder the IU arms showed. Fit time per answer: median 0.79 s / 3.67 s, mean 1.05 / 5.00 s, max 59 s / 210 s; totals 4.0 h / 19.1 h
(= mean x 13,769 exactly), i.e. two to three orders of magnitude above the IU arms (0.01 s per answer). The joint misfit is at or below the
hard-LSML misfit on 99.95% / 100% of answers; the Jacobian is full global rank on every answer (condition 1.58-1.81). Only min / median /
mean / max are saved per arm, so no distribution shape is drawn.

### fig_supervised_calibration.png (correction and supervised vs unsupervised)

The corrected held-fold-blind thresholds move in both directions: B2_sel folds 0-4 by +0.0017, +0.0217, -0.0090, +0.0024, -0.0147; B2d_sel by
+0.0037, +0.0164, -0.0096, -0.0000, -0.0116 (largest rise in fold 1 and largest fall in fold 4 on both banks; the +0.022 / -0.015 magnitudes quoted
in the note are the B2_sel values). The net PRMScore change is tiny: B2_sel 0.629387 -> 0.629332 (-0.000055), B2d_sel 0.629090 -> 0.629366
(+0.000277); both remain above every unsupervised row (entropy 0.6254, varentropy15 0.6258, B2d_sel Joint 0.6202, B2d_sel IU 0.6177). PB and
within-AUC are byte-identical under the original and corrected evaluations (asserted in METRICS_CORRECTED and re-checked here). On the two
endpoints the supervised arms (36.03 / 0.7530 and 36.30 / 0.7531) sit above IU and Joint on the same banks (+1.3 to +2.4 pp; +0.016 to +0.026
within) and above entropy / varentropy15 on within only; this is other-answer labelled access and is a matched diagnostic, not a ceiling.
All 40 inner calibration fits converged; the perturbation test (B2d_sel, fold 0, 19,415 held-fold labels flipped) reproduced theta and q bit-identically.

### fig_stage2_summary.png (all Stage-2 arms)

The plane separates into three bands: the supervised stars (36.0-36.3 / 0.753), the Step-339 contribution arms and the identity-sign diagonal
arm without SEL (35.4-36.1 / 0.746-0.747), and everything with an expanded bank carrying the selected block (32.8-35.2 / 0.724-0.737). Within
the last band the two Joint arms are the top-right members of their banks, B2d_sel Joint alone crossing entropy on within (0.7371 vs 0.7301)
while staying left of entropy on PB. Every B2_sel-family point lies below the entropy line on both endpoints. The collapsed identity-sign
B2d_sel arm (32.80 / 0.7321) and the identity-sign B2d arm without SEL (36.06 / 0.7464) remain the two outliers of the equal-weight family; the
latter contains no cross-rank product and does not bear on the cross-rank claim.

## Checks

| check | result | detail |
|---|---|---|
| Macro PB = mean of the 8 cell F1 values (PB_CELLS.csv) vs `PB_macro_percent` vs `METRICS.metrics[arm].pb_all8` vs mean of `pb_cells[*].f1` | MATCHED, 28/28 rows (21 arms + 7 references) | max abs difference 7.1e-15 pp |
| Coverage | MATCHED, 21/21 arms | `valid_answers` 13,769 (coverage 1.000), `pb_valid` 6,800, `prm_valid` 6,969, `prm_within_n` 6,030 on every arm; FIT_HEALTH `fits` 13,769 for the 5 arms fitted in this pass |
| Failures | CONFIRMED 0 | `coverage[*].failures = 0`, `undeclared_failures = 0`, `METRICS.failures[arm] = []` for all 21 arms; FIT_HEALTH failures 0; RESULT_REVIEW.json PASS (13,769 answers replayed, 68,845 arm checks, 69 hashes, 5 methods replayed); RUN_STATE `JOINT_PASS_COMPLETE`, 13,769 / 13,769 |
| The 16 appended fast-pass arms identical to `fast_pass/METRICS.json` | MATCHED to 1e-12 | every numeric leaf of `metrics`, `weights` and `coverage` equal with max abs difference 0.0, no missing keys, no non-numeric mismatch; the 3 re-fitted B1_hist arms and the 7 frozen references also equal at 0.0; the fast-pass METRICS / SCORES sha256 recorded in the joint METRICS equal the files' actual hashes |
| Primary pair identical between passes | MATCHED | `B2_sel__iu_minus_B2d_sel__iu`: every leaf equal (0.0) between fast_pass and joint_pass METRICS; `primary = [["B2_sel__iu","B2d_sel__iu"]]` in both; COMPARISON.csv equal to METRICS at 0.0; PB -0.7985 pp [-1.3146, -0.3040], within -0.006612 [-0.008918, -0.004350], 97.5%, 26 gained / 60 lost; equal to the values plotted in the fast-pass review |
| All 8 joint-pass contrasts: COMPARISON.csv vs METRICS; point deltas vs arm metrics | MATCHED | max abs difference 0.0 on delta / CI / level / gained / lost; PB, PRMScore and all-answer within point deltas reproduce the contrast deltas at 0.0; no Joint - shrink contrast is registered (figure marks it point-only) |
| Within-AUC recomputed from SCORES.npz + frozen JOINED labels (labels >= 0 inside each PRMB answer, answers with both classes) | MATCHED | B2d_sel__joint 0.7371422 (n 6,030), B2_sel__joint 0.7292724, B2d_sel__iu 0.7335809, B2_sel__iu 0.7269694, entropy 0.7301114, B1_hist__raw 0.7377863 - all equal to METRICS at <= 3.4e-16; common-answer deltas Joint - IU +0.0035613 (B2d_sel; 1,185 answers up / 872 down / 3,973 tied) and +0.0023030 (B2_sel; 827 / 660 / 4,543), bank Joint -0.0078698 (1,104 / 1,550 / 3,376), each equal to METRICS at 0.0; JOINED offsets equal the per-record step counts (145,597 steps) |
| ProcessBench cell arithmetic for the Joint arms | CONFIRMED | cell F1 = harmonic mean of clean accuracy and gated exact accuracy on all 8 cells; gated exact count from the cells = raw exact - gate-suppressed correct peaks (B2d_sel Joint 1,374 - 223 = 1,151; B2_sel Joint 1,331 - 214 = 1,117; IU 1,353 - 223 = 1,130 and 1,311 - 215 = 1,096); `pb_raw_exact` = raw exact / 4,442; clean accuracy 0.5008 identical on every arm (shared external gate), so all PB differences arise on the 4,442 erroneous answers |
| Corrected PRMScore vs METRICS_CORRECTED vs the note | MATCHED | B2_sel original 0.6293866605 / corrected 0.6293316962 / delta -0.0000549644; B2d_sel 0.6290897697 / 0.6293663729 / +0.0002766032; originals equal `supervised/METRICS.json`, corrected equal `METRICS_CORRECTED.metrics[*].prmscore_q08`, deltas are arithmetic; the note's 0.629387 / 0.629332 / -0.000055 and 0.629090 / 0.629366 / +0.000277 match at 6 decimals; PB all-8, Q4, Q8, per-cell F1 and within-AUC identical (0.0) between the original and corrected evaluations |
| Thresholds (THRESHOLDS.json) | MATCHED, with one wording precision item | 10 fold thresholds: original values replay from saved scores and equal `supervised/METRICS.json`; corrected values equal `METRICS_CORRECTED`; deltas arithmetic; 4 inner models per fold, all converged, none trained on the held fold; held fold scored by the reused outer fit. Extremes: B2_sel +0.0217 (fold 1) / -0.0147 (fold 4); B2d_sel +0.0164 (fold 1) / -0.0116 (fold 4). The note's "up to +0.022 (fold 1) and -0.015 (fold 4) in both banks" is exact for B2_sel and gives the right folds and directions for B2d_sel, whose magnitudes are +0.016 / -0.012 |
| Supervised numbers in the note vs `supervised/METRICS.json` and `CONTRASTS.json` | MATCHED | 36.03 / 0.7530 and 36.30 / 0.7531; B2_sel - B2d_sel +0.27 pp [-0.53, +1.08], within +0.0001 [-0.0009, +0.0010] (97.5%); vs B2_sel IU +2.35 [+1.17, +3.52], +0.026 [+0.023, +0.030]; vs entropy +0.85 [-0.21, +1.94], +0.023 [+0.019, +0.027]; vs varentropy15 +0.34 [-0.80, +1.48], +0.015 [+0.011, +0.019]; fit health 90 fits, 67 converged, 23 at the iteration limit (all B2_sel ProcessBench folds), 0 stalled; 40 inner fits converged; perturbation q and theta bit-identical |
| The note's Joint numbers vs METRICS / FIT_HEALTH | MATCHED | table rows (PB 2 dp, within 4 dp, PRMScore 3 dp) for the 4 IU / Joint rows and the 14 fast-pass rows all match; the 6 quoted contrasts match at the quoted precision (including "-0.00" for the B2d_sel Joint - IU lower bound of -6.8e-6 pp); converged 99.6% / 99.5%, condition > 1e12 1.6% / 5.6%, n < p 1 / 705, median fit 0.8 s / 3.7 s all match |
| Reference rows named in the brief | MATCHED | entropy 35.4444 / 0.730111 / 0.625426; varentropy15 = B1_hist__raw = ref__k15__raw 35.9610 / 0.737786 / 0.625781 (the two rows equal at 0.0); longest-step control 33.6944 used as a context line only |
| Pending arms | CONFIRMED | METRICS / RUN_STATE / RESULT_REVIEW all list exactly `B2__joint`, `B2d__joint` as pending, "deferred for compute, not dropped" |

Mismatches found: none. One wording precision item (threshold magnitudes attributed to "both banks") noted above.

## Reading

Numerical identity: the 16 appended fast-pass arms, the three re-fitted B1_hist arms, the seven frozen references and the pre-registered
primary contrast are identical between the fast-pass and joint-pass METRICS (0.0), and the supervised PB / within-AUC values are identical
under the original and corrected calibration. Interval-supported differences: the Joint L-SML arms improve on IU-PCR on the same bank by a
small within-answer AUC margin whose 95% interval excludes zero on both primary banks (+0.0036 [+0.0022, +0.0049] on B2d_sel, +0.0023 [+0.0013,
+0.0034] on B2_sel) and by a ProcessBench point gain of about +0.4 pp that sits at the edge of its interval (B2_sel +0.42 [+0.05, +0.79];
B2d_sel +0.40 [-0.00, +0.81], lower bound at zero); the cross-rank products hurt under all four registered solvers on both endpoints, and under
Joint the bank contrast is -0.78 pp [-1.29, -0.30] / -0.0079 [-0.0100, -0.0058], the same size as under IU. Joint - shrink is a point difference
only (no registered bootstrap). Inconclusive differences: B2d_sel Joint vs entropy on ProcessBench (-0.29 pp [-1.09, +0.50]) and vs varentropy15
on both endpoints (-0.81 pp [-2.00, +0.36]; -0.0006 [-0.0051, +0.0037]); the supervised B2_sel - B2d_sel pair (+0.27 pp [-0.53, +1.08]; +0.0001
[-0.0009, +0.0010]) is an inconclusive difference, not a demonstrated zero. No expansion arm carrying the selected block reaches token entropy or
varentropy15 on ProcessBench at the point level, and B2_sel Joint is below both with intervals excluding zero on PB; the only expanded arm above
varentropy15 on points remains the identity-sign diagonal arm without the selected block, which contains no cross-rank product. The supervised
rows are other-answer labelled access and bound neither the unsupervised methods nor the quadratic feature class. The two deferred Joint arms
(B2, B2d) remain pending; the secondary-bank ladder and any Joint statement about the non-selected banks wait on them. Everything here is cached
development data under a frozen external gate and calibration; no localization improvement or untouched confirmation is claimed.
