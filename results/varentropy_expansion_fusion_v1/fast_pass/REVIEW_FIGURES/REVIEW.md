# Varentropy-expansion (cross-rank) fusion v1 - FAST PASS review

Reviewed 2026-09-12 from the saved fast-pass outputs only
(`results/varentropy_expansion_fusion_v1/fast_pass/{METRICS.json, COMPARISON.csv, PB_CELLS.csv, SUMMARY.csv, FIT_HEALTH.json, FEASIBILITY.json, RESULT_REVIEW.json, SCORES.npz}`)
plus the frozen step labels in
`results/localization_full_benchmark_v3/evaluation/JOINED.{npz,json}` (main repo) for the one recomputation.
No benchmark pickle or sqlite checkpoint was opened. Scope: 19 non-Joint arms, 13,769 answers, 145,597 steps,
0 declared failures. **The four Joint L-SML arms (`B2d_sel__joint`, `B2_sel__joint`, `B2d__joint`, `B2__joint`)
are PENDING and appear on no figure; every pre-registered contrast involving them is absent.** All numbers are
development data (cached, previously inspected); nothing here is an untouched confirmation.

Files in this directory:

| file | content |
|---|---|
| `fig_bank_ladder.png` | feature-bank comparison: same solver across B1_hist -> B2d -> B2d_sel -> B2 -> B2_sel, PB all-8 and PRMB within-AUC, entropy / varentropy15 / longest-step reference lines |
| `fig_architecture.png` | architecture comparison: same bank, solvers side by side (dot plot on truncated axes), Joint slots marked pending |
| `fig_primary_and_secondary_contrasts.png` | forest plot: primary (97.5%) on top, then bank, selected-block, historical-IU, architecture, and reference contrasts (95%) for PB delta (pp) and within-AUC delta |
| `fig_pb_cells_heatmap.png` | 8 ProcessBench cell F1 for entropy, B1_hist raw/equal/iu, B2d_sel x 4 solvers, B2_sel x 4 solvers, with the macro column |
| `fig_weights.png` | per-column mean standardized coefficient by term type for B2_sel__iu, B2_sel__shrink, B2d_sel__iu (+ term |w| shares) |
| `fig_fit_health.png` | shrink alpha clipping, n<p fits per bank, orientation flip rate per arm, anchor-correlation summaries |
| `CHECKS.json` | machine-readable output of the checks below |
| `REVIEW.md` | this file |

## Readings per figure

### fig_bank_ladder.png (feature bank, same solver)

Under every solver the diagonal banks (B2d, B2d_sel; 30/33 columns) sit at 34.6-34.8% PB / 0.731-0.734 within,
and adding the 105 cross-rank products P_ij (B2, B2_sel; 135/138 columns) moves the same solver DOWN on both
endpoints: IU-PCR 34.75 -> 33.95 PB and 0.7336 -> 0.7270 within; shrinkage 34.66 -> 34.06 and 0.7314 -> 0.7274;
anchor-sign equal 34.73 -> 34.38 and 0.7344 -> 0.7291. The only line that rises on PB when P_ij is added is the
anchor-oriented identity-sign equal arm (32.80 -> 34.08), and that is a recovery from a collapsed B2d_sel point,
not a gain over the diagonal banks under the other solvers; on within it also falls (0.7321 -> 0.7241). No
expanded bank reaches entropy (35.44 PB) or varentropy15 (35.96 PB / 0.7378 within) with the learned solvers;
the diagonal-bank solvers do clear entropy on within only (0.7314-0.7344 vs 0.7301). The B1_hist points (Step 339
EQUAL 35.60 / 0.7470, IU 35.35 / 0.7468) remain the highest within-AUC in the whole ladder. The one expanded arm
above varentropy15 at the point level is `B2d__equal_identity` (36.06 PB, 0.7464 within), a fixed-sign arm on
the diagonal bank without the selected block (see the contrasts figure for its interval).

### fig_architecture.png (architecture, same bank)

Within a bank the four solvers are much closer to each other than the banks are to each other: on B2d_sel the
spread is 34.66-34.75 PB (excluding the collapsed identity arm) and 0.7314-0.7344 within; on B2_sel it is
33.95-34.38 PB and 0.7241-0.7291 within. Shrinkage IU is numerically indistinguishable from IU-PCR on the pair
banks (alpha clips at 1.0 on 98.9% of answers, so it is IU on the rank-1-completed cross-group target) and
slightly below IU on the diagonal banks (0.7314 vs 0.7336 within). The anchor-sign equal arm is the best or tied
solver on every expanded bank on within and on both pair banks on PB, i.e. the learned IU/shrink weights do not
beat simple oriented aggregation in this representation. The anchor-oriented identity-sign equal arm behaves
inconsistently: 36.06 PB on B2d, 32.80 on B2d_sel (adding the three SEL columns with +1 identity sign costs 3.3 pp
under equal weights, whereas under IU the SEL block is inert, +0.01 pp), 34.07-34.08 on the pair banks.

### fig_primary_and_secondary_contrasts.png (forest plot)

The pre-registered primary `B2_sel__iu - B2d_sel__iu` is negative on both endpoints with 97.5% intervals excluding
zero: PB -0.80 pp [-1.31, -0.30] (26 answers gained, 60 lost), within-AUC -0.0066 [-0.0089, -0.0043] on the 6,030
common PRMB answers; pooled AUC -0.0016, PRMScore -0.0021. Adding P_ij under the other solvers is also negative
with intervals excluding zero (anchor-sign equal -0.35 pp / -0.0053; shrink -0.60 pp / -0.0040; IU without the
selected block -0.76 pp / -0.0062), except identity-sign equal on PB (+1.27 pp [-0.35, +2.92], but within -0.0080
[-0.0137, -0.0024]). The selected block is inert under IU (|delta| <= 0.02 pp and <= 0.0004 within, intervals
include zero). Against the historical IU bank, B2_sel__iu loses on both endpoints (-1.40 pp [-2.47, -0.34];
-0.0199 [-0.0237, -0.0161]) and B2d_sel__iu loses on within (-0.0132 [-0.0161, -0.0104]) with a PB interval that
includes zero (-0.60 pp [-1.56, +0.38]). Against entropy every expanded B2_sel/B2d_sel arm is at or below on PB
(B2_sel arms exclude zero; B2d_sel iu/oriented/shrink include zero), while B2d_sel anchor-sign equal and IU are above
entropy on within (+0.0042 [+0.0015, +0.0071] and +0.0035 [+0.0005, +0.0065]). Against varentropy15 every expanded
B2_sel/B2d_sel arm is below on PB with intervals excluding zero, and below or tied on within.

### fig_pb_cells_heatmap.png (8 ProcessBench cells)

The macro deficits of the expanded banks are concentrated in the olympiadbench and omnimath cells: B1_hist raw
holds 35.3 / 33.2 on olympiadbench (4B / 8B) where the expanded arms reach 29.3-31.0 / 27.7-30.7, and 33.1-33.8 on
omnimath where B2_sel arms reach 33.0-33.6 / 32.1-32.4. On gsm8k-4B the expanded IU/shrink arms lose 2.3-2.7 pp to
B1_hist raw (45.3) and on gsm8k-8B 1.4-3.9 pp. The expanded arms are at or slightly above the references on the two
math cells (33.6-33.8 vs 32.8-33.4 for B2d_sel; entropy holds 34.8 on math-4B). The collapsed `B2d_sel__equal_identity`
row is a broad loss on olympiadbench/omnimath (27.2-29.3) while it keeps gsm8k-4B at 44.2 and math-4B at 35.3.

### fig_weights.png (fitted weights)

METRICS['weights'] stores the per-column MEAN standardized coefficient over the 13,769 answers and the term-level
absolute-weight share; no per-column median is stored, so the figure shows the means (a "median per column group"
cannot be produced from the saved aggregates). Sign convention: coefficients are as saved after `_orient`, i.e.
higher fused score = more risk relative to the answer's own raw top-15 varentropy; the orientation flip rate on
these three arms is 0.03-0.12%, so the means are effectively the un-flipped fits. On `B2_sel__iu` the 105 P_ij
columns carry 82% of the absolute weight (D 8%, P_ii 10%, SEL 0.1%); the D and P_ii means rise monotonically with
rank (deep ranks 8-15 weighted 3-6x rank 1-3), and all P_ij block-pair means are positive, largest for pairs
involving ranks 6-15. Note the fitted risk-oriented signs on P_ii and P_ij are POSITIVE, whereas the varentropy
identity gives them -1 / -2: the fit on standardized columns does not reproduce the identity direction on these
terms. `B2_sel__shrink` is flatter across ranks (0.0026-0.0056 on D and P_ii), gives the r1-5 x r1-5 pair block a
negative mean (-0.009) and negative SEL a^2/a^3. `B2d_sel__iu` splits the weight evenly between D (51%) and P_ii
(48%), again increasing with rank, with SEL at 1%.

### fig_fit_health.png (disclosures)

Shrinkage alpha clips at 1.0 on 86.73% (B2d), 86.64% (B2d_sel), 98.88% (B2), 98.88% (B2_sel) of answers (alpha
means 0.957 / 0.958 / 0.997 / 0.997; minima 0.114 / 0.123 / 0.294 / 0.307). n < p fits (tokens fewer than active
columns): 1 answer on B2d and B2d_sel, 648 (4.71%) on B2, 705 (5.12%) on B2_sel; not reported for B1_hist
(`null`). Orientation flips: identity-sign equal on the pair banks is flipped by `_orient` on 99.86% of answers
(pre-orientation anchor correlation mean -0.365, range -0.726 to +0.264), so the scored arm is the anchor-oriented
identity-sign equal fusion; IU/shrink flip on 0.12% / 0.14% (pair banks) and 0.03% / 0.03-0.04% (diagonal banks);
anchor-sign equal and the B1_hist arms never flip. Anchor correlation of the fused score falls from 0.80 (B1_hist
IU) to 0.57-0.58 (B2d/B2d_sel IU) to 0.39-0.40 (B2/B2_sel IU). Only min/median/mean/max are stored per arm, so
no full distribution is drawn. Tiny-scale (< 1e-6) kept columns average 0.11 per answer on the pair banks
(max 64) and 0.03 on the diagonal banks.

## Checks

| check | result | detail |
|---|---|---|
| Macro PB = mean of the 8 cell F1 values (PB_CELLS.csv) vs `PB_macro_percent` and vs `METRICS.metrics[arm].pb_all8` | MATCHED, 26/26 arms | max abs difference 1.4e-14 pp |
| Coverage = valid_answers / 13,769 | MATCHED, 19/19 arms at 1.000 | pb_valid 6,800, prm_valid 6,969, prm_within_n 6,030 on every arm |
| Failures | CONFIRMED 0 | `coverage[*].failures = 0`, `undeclared_failures = 0`, `METRICS.failures[arm] = []` for all 19 arms; RESULT_REVIEW.json PASS (13,769 answers replayed, 261,611 arm checks, 69 hashes) |
| Primary pair and CI: METRICS.json vs COMPARISON.csv vs what is plotted | MATCHED | `primary = [["B2_sel__iu","B2d_sel__iu"]]`, ci_level 0.975; PB delta -0.7985 pp, CI [-1.3146, -0.3040]; within delta -0.006612 (common), CI [-0.008918, -0.004350]; identical in all three places; SUMMARY.csv point differences reproduce the deltas to 1e-14 |
| Within-AUC delta for the primary recomputed from SCORES.npz + frozen JOINED labels | RECOVERABLE and MATCHED | per-answer within = ROC-AUC of the step scores against step labels >= 0 inside each PRMB answer, averaged over answers with both classes: B2_sel__iu 0.7269694 (n 6,030), B2d_sel__iu 0.7335809 (n 6,030), entropy 0.7301114, all equal to METRICS at 1e-15; common-answer delta -0.0066115 = METRICS -0.0066115; 1,113 answers move up, 1,463 move down, 3,454 tie |
| Approximate paired source-group bootstrap of that delta (2,000 draws over the 707 groups of the common answers, seed 20260912) | CONSISTENT | 97.5% CI [-0.0088, -0.0045] vs METRICS [-0.0089, -0.0043]; not the same draw count or seed, so a cross-check, not a replay |
| Step layout alignment | CONFIRMED | `JOINED.offsets` diffs equal the per-record `steps`; 145,597 steps = METRICS `n_steps`; the exact reproduction of three METRICS within values establishes that SCORES.npz uses the same step order (the JOINED `entropy_parent` column is a different frozen arm and was not used) |
| Reference rows named in the brief | MATCHED | entropy 35.4444 / 0.730111 / PRMScore 0.625426; B1_hist__raw 35.9610 / 0.737786 / 0.625781; B1_hist__equal 35.5980 / 0.746980; B1_hist__iu 35.3498 / 0.746824 |
| Pending arms | CONFIRMED | METRICS/RUN_STATE/RESULT_REVIEW list the four Joint arms as pending; `FAST_PASS_COMPLETE` |

Mismatches found: none.

Not recoverable from the saved aggregates: per-column median weights (only means and shares are stored), full
anchor-correlation distributions (only min/median/mean/max), and any Joint statistic (arms pending).

## Reading

Under IU-PCR, the pre-registered primary solver, adding the 105 cross-rank products P_ij to the diagonal bank
hurts localization on both benchmarks: -0.80 pp ProcessBench macro F1 (97.5% CI [-1.31, -0.30]; 26 answers gained,
60 lost) and -0.0066 PRMBench within-answer AUC ([-0.0089, -0.0043]) on the same 6,030 answers. The same direction
holds under joint-target shrinkage (-0.60 pp / -0.0040), under anchor-sign equal weights (-0.35 pp / -0.0053), and
under IU without the selected block (-0.76 pp / -0.0062), each with 95% intervals excluding zero. The one solver
where the pair terms raise the PB point is the anchor-oriented identity-sign equal arm (+1.27 pp, interval
including zero), and it pays on within-AUC (-0.0080, excluding zero); that arm is also the least stable one across
banks (36.06 -> 32.80 -> 34.08 PB), so its PB rise reads as a recovery from a collapsed B2d_sel point rather than as
evidence that the products carry localization signal. The fitted weights show where the capacity went: on B2_sel
the P_ij columns absorb 82% of the absolute IU weight while the fused score's correlation with the varentropy
anchor drops from 0.80 (B1_hist IU) to 0.40, and the fitted P_ii/P_ij signs are positive where the varentropy
identity puts -1/-2, so the expanded fit is a different combination, not a re-weighting of varentropy. The
selected-surprisal block [a, a^2, a^3] is inert under IU (|delta| <= 0.02 pp) and destructive only under the
identity-sign equal weights on the diagonal bank.

No expanded bank reaches the references with the learned solvers. On ProcessBench, every B2_sel / B2d_sel arm is
below entropy (35.44) and below varentropy15 (35.96); the B2_sel arms are below entropy with intervals excluding
zero (-1.07 to -1.49 pp), the B2d_sel IU / anchor-sign / shrink arms are 0.69-0.79 pp below entropy with intervals
that include zero, and all eight are 1.2-3.2 pp below varentropy15 with intervals excluding zero. On PRMBench
within-AUC the diagonal-bank solvers do beat entropy (B2d_sel anchor-sign +0.0042 [+0.0015, +0.0071], IU +0.0035
[+0.0005, +0.0065]), but none reaches varentropy15 (0.7378) or the Step 339 contribution arms (0.7468-0.7470); the
B2d_sel IU arm is 0.0132 below B1_hist IU with an interval excluding zero. The highest expanded point is
`B2d__equal_identity` (36.06 PB / 0.7464 within, a fixed-sign, label-free arm on the diagonal bank without the
selected block): +0.10 pp vs varentropy15 on PB with an interval including zero ([-0.74, +0.96]) and +0.0086 on
within ([+0.0060, +0.0112]); it is a secondary arm, it was not pre-registered as a candidate, and it is essentially
the Step 339 EQUAL arm's level (35.60 / 0.7470) on a re-expressed bank, so it is an observation to carry forward,
not a promotion.

Within a bank the choice of solver moves the endpoints far less than the choice of bank: on B2d_sel the learned
IU and shrink arms are within 0.1 pp / 0.003 within-AUC of the anchor-sign equal arm, and shrinkage is
indistinguishable from IU on the pair banks (+0.11 pp [-0.05, +0.28]; +0.0004) because alpha clips at 1.0 on
98.9% of answers, i.e. the shrink arm is IU on the rank-1-completed cross-group target. IU beats the anchor-oriented
identity-sign equal arm on B2d_sel PB (+1.95 pp [+0.37, +3.55]) and on B2_sel within (+0.0029 [+0.0019, +0.0039]),
and not elsewhere. The architecture comparison is incomplete: the four Joint L-SML arms are pending, so the
pre-registered Joint-vs-IU and Joint-vs-shrink contrasts and the Joint disclosures (convergence, multistart,
model-covariance condition) are not available here.

Disclosures required by the protocol: shrinkage alpha clipped at 1.0 on 86.73% (B2d), 86.64% (B2d_sel), 98.88%
(B2) and 98.88% (B2_sel) of answers (alpha minima 0.11-0.31); n < p fits on 1 answer for B2d and B2d_sel, 648
(4.71%) for B2 and 705 (5.12%) for B2_sel (not reported for B1_hist); the identity-sign equal arm on the pair banks
is flipped by `_orient` on 99.86% of answers (mean pre-orientation anchor correlation -0.365) and must be named
"identity-sign equal weights, anchor-oriented"; the B1_hist EQUAL arm is the unoriented Step 339 arm; IU/shrink
flip on 0.03-0.14% of answers; tiny-scale kept columns average 0.11 per answer on the pair banks (max 64); fit
runtime 118 s (B1_hist IU), 128 s (B2d_sel IU), 169 s (B2_sel IU), 207 s (B2_sel shrink) for 13,769 answers.
All arms have full coverage (13,769 / 13,769) and zero declared or undeclared failures. Fusion is answer-local;
the ProcessBench gate and PRMScore calibration are external and frozen; all cached data are development data,
not an untouched test, and no localization improvement is claimed from this pass.
