# Stage 1 review figures — independent visual review of the completed RBM literature-completion suites

Reviewer: Claude (results reviewer), 2026-09-12. Scope: read-only over every saved artifact; the only
outputs are the seven PNGs, `CHECKS.json` and this file in `STAGE1_REVIEW_FIGURES/`. No experiment was
launched, no benchmark pickle or sqlite checkpoint was opened, one process, peak memory well under 2 GB
(the 48 MB `stability/FIT_HEALTH.json` is decoded one entry at a time). Script:
`make_stage1_review_figures.py` (same directory) regenerates everything.

Account under review: `docs/research_notes/RBM_PROGRAM_STAGE1_ACCOUNT_2026-09-12.md` (with
`docs/experiments/RBM_DEPTH_AMENDMENT_20260912.md`, `capacity/CAPACITY_INTERPRETATION.md`,
`STAGE1_COMPARISON_TABLE.{md,csv}`, `STAGE1_CONTRASTS.md`, `RBM_FUSION_COMPARISON.csv`, per-suite
`METRICS.json` / `COMPARISON.csv` / `PB_CELLS.csv`, `capacity/CAPACITY_CONVERGENCE.{csv,json}`,
`capacity/CAPACITY_MAXITER_PROBE.json`, `depth/SMOKE_DIAGNOSIS.json`, `depth*/SMOKE.json`,
`EXPERIMENT_LEDGER.json`, `variance/VARIANCE_LOSS_DECOMPOSITION.json`).

All numbers are development evidence on the same 13,769 cached answers (v3 labels, v2 source groups);
nothing here is untouched confirmation. Framing kept throughout: learned fusion has not shown a
consistent overall advantage, and representation, optimization, normalization and readout remain partly
entangled where that is what the data show.

## Figures

### fig_pb_vs_within_scatter.png
Every answer-local row of `RBM_FUSION_COMPARISON.csv` (110 rows, 91 distinct score sets because the H1
references are re-listed by each suite), the two simple controls and the three other-answer diagnostics,
as ProcessBench macro F1 (%) against PRMBench within-answer AUC; colour = experiment family, hollow
squares = depth rows with declared failures (coverage < 1), star = other-answer diagnostics, horizontal
lines = longest-step control 33.69 and random-step control 20.28, thin lines = token entropy 35.44 /
0.7301. The right panel zooms on the reference band. Reading: every learned single-unit RBM row sits
inside a band of about 34.9-37.0 PB and 0.732-0.747 within-AUC that also contains varentropy15
(35.96 / 0.7378), the before-training RBMs (35.97-36.38 / 0.742-0.746) and the equal-weight varentropy
contributions (35.60 / 0.7470, the highest within-AUC of any answer-local row). The rows that leave the
band all leave it downward: separate-variance logit (18.76 / 21.09), exact and best-of-3 H4 (24.2-28.7),
CD-10 logit readouts (27.4-30.5), the 48-column trained RBM (19.44 / 0.593), the moment-bank IU-PCR and
equal rows (20.2-22.4), and every depth row (18.9-34.7). The only points above the band are the
low-correlation-6 RBM (36.99), the shared-variance bank12 posterior (36.81) and the supervised
step-BCE diagnostic (37.20), each a point estimate without a registered interval in these inputs.

### fig_primary_contrasts_forest.png
The twelve pre-registered primary contrasts (ten full-population primaries from the five suites plus the
two depth conditional primaries on common covered answers), PB delta in pp (left) and within-AUC delta
(right) with 97.5% paired source-group bootstrap intervals, gained/lost exact-success counts in the row
label. All 72 numbers and the primary flags were re-read from the suites' `METRICS.json` and match
`STAGE1_CONTRASTS.md` exactly. Reading: no primary interval lies entirely above zero on ProcessBench.
Seven of twelve PB intervals are entirely below zero (separate-variance bank12, both capacity H4
contrasts, best-of-3 H4 bank12, both depth primaries and both depth conditionals); the remaining five
include zero (separate-variance bank6, both temporal contrasts, best-of-3 H4 bank6). On within-AUC one
interval is entirely above zero (separate-variance bank6 posterior, +0.005 [+0.003, +0.008]) while its PB
interval includes zero; the temporal contrasts have PB intervals including zero but within-AUC intervals
entirely below zero (-0.003, -0.004). The suites therefore established negative results and null
results, not a candidate.

### fig_capacity_convergence.png
From `CAPACITY_CONVERGENCE.csv/.json`: (a) final max |gradient| of every exact-H4 fit on a log axis,
medians 0.031 (bank6) and 0.072 (bank12) against gtol 1e-6; 13,768 / 13,769 and 13,769 / 13,769 fits
stopped at the maxiter-100 cap. (b) NLL gain of H4 over H1 per token, medians 0.982 / 1.779, minimum
0.019 / 0.090, so H4 fits the density better on every answer. (c) PB exact successes gained / lost versus
exact H1 by gradient quartile: losses exceed gains in every quartile (bank6 q1 24/48 to q4 114/333; bank12
q1 35/91 to q4 97/250); as loss rates over the PB answers in each quartile, bank6 10.4% (q1) to 13.2%
(q4) and bank12 5.2% (q1) to 14.1% (q4). (d) surviving varying posterior views per fit: 140 (bank6) and 353
(bank12) answers have only two, which is exactly the depth failure count. Reading: the H4 deficit is
conditional on an optimization budget that no fit reached, but it is not confined to the worst-optimized
fits; the data do not separate capacity from optimization.

### fig_capacity_probe.png
`CAPACITY_MAXITER_PROBE.json`, 54 exact-H4 refits (27 smoke answers x 2 banks) at maxiter 1000 against
the registered maxiter-100 NLL, x = y line, hollow diamonds where the top-10 peak moved in either readout,
crosses where the fit is still unconverged at 1000. 48 / 54 converge, median further NLL decrease 0.473,
peak moved in 7 (logit) / 8 (posterior) of 54, longest fit 1.39 s. Reading: bank12 fits sit far below
the diagonal (the registered budget leaves a large NLL gap) and the peak moves in about one fit in
seven, so a converged-H4 evaluation could change the capacity conclusion in either direction; the figure
is feasibility only and supports no benchmark statement.

### fig_stability_starts.png
Streamed from `stability/FIT_HEALTH.json` (13,769 entries, four models each). (a) H1: NLL spread across
the three exact starts is at machine precision (median 1.3e-10 / 3.3e-10; p95 6.6e-10 for bank6 and
1.02e-2 for bank12), and the top-10 peak is identical in 13,768 / 13,705 answers. (b) H4 at maxiter 100:
median spread 0.056 / 0.233 nats per token, p95 0.195 / 0.730. (c) distinct peaks across starts: H4
disagrees in 18.0% (bank6: 2,385 two-peak + 99 three-peak) and 32.8% (bank12: 4,119 + 399) of answers.
(d) the selected lowest-NLL H4 start improves on the capacity start by a median 0.009 / 0.070, and the
chosen start is spread almost uniformly over the three starts (4580/4576/4613; 4615/4556/4598).
Reading: restarts are not a lever for H1; for H4 the min-NLL selector picks a different fit in two
thirds of answers, and the primaries show that this lowers ProcessBench (bank12 -1.33 pp [-2.17, -0.51])
while leaving within-AUC flat, so better density fit at this budget is not selecting task-useful
solutions.

### fig_depth_coverage_and_performance.png
`depth_amended/COMPARISON.csv`, retained readouts (bank6 posterior, bank12 logit): full-population PB,
covered-answer PB, coverage with valid-answer counts, PRMBench within-AUC with n and pooled AUC, beside
the exact H1 / H4 references. The original posterior-input second layers have coverage 0.9898 (13,629)
and 0.9744 (13,416) and are the lowest rows on every endpoint (exact L2: 19.27 / 22.10 PB, 0.6215 /
0.5795 within); the covered-answer PB differs from the full-population PB by only 0.37-0.87 pp, so the
deficit is not a coverage artefact. The logit-input amendment restores full coverage and raises the exact
second layer by +12.42 / +10.68 pp, but leaves it at 31.69 / 32.78 PB, below the H1 first layer (36.20 /
36.27) and below the longest-step control on bank6. The bank6 CD-10 logit-input row reaches within-AUC
0.7374, which is 0.0014 above its H1 reference (0.7360) while being 3.88 pp lower on PB and lower on
pooled AUC and PRMScore.

### fig_pb_cells_heatmap.png
Per-cell PB F1 (%) from the suites' `PB_CELLS.csv` for the requested configurations, with the macro
recomputed as the mean of the eight cells (it equals `METRICS.pb_all8` for all 141 method rows).
Reading: the reference-band rows differ from token entropy by cell-level swaps of one to four points in
both directions (for example RBM6 gains on omnimath_q4 36.3 vs 35.4 and olympiadbench 33.0 / 31.9 vs 31.2
/ 30.4 but loses on gsm8k_q8 39.2 vs 39.7), while the H4, separate-variance and depth rows lose most
heavily on the olympiadbench and omnimath cells (11.8-22.9 against 30-35 for the references). The
longest-step control has no per-cell values in these suites' files; only its macro 33.69 is listed.

## Checks

`CHECKS.json` holds every recomputation. Summary: **260 checks matched, 4 did not match** (after
correcting a defect in this reviewer's first JSON streamer, which had read only the first 303 of the
13,769 stability entries; the figure and every stability number below are from the full file).

Matched (all numbers recomputed from CSV/JSON, never copied from prose):

- Macro PB equals the mean of the 8 cell F1 values in `PB_CELLS.csv` for all 141 method rows across the
  five suites (max |difference| < 1e-9), including the depth rows whose failures are counted as missed
  decisions (64 cell rows with `valid_decisions < answers`).
- `COMPARISON.csv` pb_all8 / prm_within equal `METRICS.json` for all 141 rows; all 55 `__old` reference
  rows listed inside the suites equal the source values in `RBM_FUSION_COMPARISON.csv`; the four exact-H1
  capacity rows are bit-identical to the historical RBM6 / RBM12 rows.
- All 39 rows of `STAGE1_COMPARISON_TABLE.csv` are found by value in `RBM_FUSION_COMPARISON.csv`; all
  39 coverage entries equal valid_answers / 13,769.
- All 12 rows of `STAGE1_CONTRASTS.md` (PB delta, PB CI, within delta, within CI, primary flag at 97.5%)
  match `METRICS.json` contrasts / conditional_contrasts; gained/lost 283/934, 191/175, 299/701,
  285/802, 75/108, 85/119, 79/133, 44/71, 50/249, 42/229 match.
- Reference rows: token entropy 35.4444 / 0.730111 / 0.625426, varentropy15 35.9610 / 0.737786,
  longest-step 33.6944, random 20.2750; 130 rows, 110 answer-local.
- Depth failures: 140 (bank6) / 353 (bank12) in `METRICS.failures` (count and named_collapse, no other
  kind), in `EXPERIMENT_LEDGER.json`, in `SMOKE_DIAGNOSIS.json`, as the two-view counts and
  `depth_would_fail` sums of `CAPACITY_CONVERGENCE.csv`, and as the expected coverage 0.989832 /
  0.974363 = (13,769 - 140) / 13,769 and (13,769 - 353) / 13,769. Original smoke: 14 failed records on
  6 answers; amended smoke 188 exact replays = (108 - 14 records) x 2 readouts. Within-AUC n = 6,022 /
  5,914 for the original second layers.
- Depth amendment gains: +12.42 pp (bank6 exact, posterior) and +10.68 pp (bank12 exact, logit) over
  the posterior-input originals (account: +12.4 / +10.7). 15 of the 16 amendment-vs-H1 endpoint deltas
  are negative (see mismatch 2 for the sixteenth).
- Capacity: 13,768 / 13,769 exact-H4 fits at the cap, exact-H1 non-converged 1 / 0, median gradient
  0.0310 / 0.0715, median NLL gain 0.982 / 1.779, one saturated unit in 9.93% / 16.57% of fits, two in
  1.02% / 2.56%, duplicate units in 0.43% / 0.36% (account "0.4%"), zero dead units, gained/lost
  285/802 and 299/701 with quartile counts 24/48 ... 114/333 and 35/91 ... 97/250; losses with no
  saturated unit 721/802 (89.9%) and 578/701 (82.5%). Probe: 54 fits, 48 converged, median decrease
  0.473, peaks moved 7 / 8, median iterations 410.5, longest fit 1.388 s.
- Stability: 13,769 entries; H1 same peak in 13,768 / 13,705 answers; H1 median spread 0; H1 restart
  convergence 99.989% / 100%; H4 restart convergence 0.0036% / 0.0%; H4 median spread 0.0560 / 0.2325;
  median improvement over the capacity start 0.0094 / 0.0697; peak disagreement 18.04% / 32.81%;
  bank12 lost successes 114 early / 19 late of 133.
- Variance decomposition: quadratic term reverses the linear term in 922 of 934 lost cases; lost
  925 early / 9 late. Temporal within deltas -0.00281 / -0.00356.
- Account section 5 deltas: DUFS minus low-correlation -0.8695 pp; position-conditioned minus shared
  -0.6876 pp; supervised minus unsupervised update +0.9330 pp; 48-column trained minus initial -16.94
  pp; varentropy15 IU-PCR minus raw +0.00904 within / -0.611 pp PB; separate-variance bank12 logit
  21.09; shared-variance bank12 posterior 36.81; low-correlation-6 36.99; equal contributions within
  0.7470.

Not matched:

1. "H1 ... same NLL (spread 0 at the median, <= 0.01 at p95)": bank12 H1 p95 spread is 1.02e-2
   (bank6 6.6e-10). Rounding-level; the median is exactly 0 on both banks.
2. "the stacked model still sits below the single-unit first layer on every endpoint" (account section
   3): the bank6 CD-10 logit-input second layer has within-AUC 0.737389 against 0.735982 for exact H1
   bank6 posterior (+0.001407). It is below on PB (32.32 vs 36.20), pooled AUC (0.687 vs 0.708) and
   PRMScore (0.610 vs 0.631). The other 15 endpoint comparisons are below H1.
3. "Posterior versus logit readout of the same weights changes PB by up to 1.5 pp and within-AUC by
   0.01" (account section 5, attributed to rbm-logit-readout-v1): 14 of the 34 same-weight
   posterior/logit pairs in the record exceed 1.5 pp; the maximum is 17.33 pp (bank6 separate variance:
   36.08 vs 18.76). Inside rbm-logit-readout-v1 itself the before-training rows differ by 14.23 pp
   (RBM6 initial: 35.97 vs 21.73) and 16.11 pp (RBM12 initial: 36.29 vs 20.18), with within-AUC
   differences of 0.038 and 0.057. The 1.5 pp / 0.01 figure describes only the trained RBM6 / RBM12 pairs
   (1.23 and 0.10 pp; 0.0003 and 0.0065).
4. Same statement restricted to that suite's rows: not supported for the reason in 3.

Not checkable from the provided inputs: the section-5 statement that "every primary interval against a
matched reference includes zero" for the single-unit RBM rows against the raw varentropy references (those
contrasts live in the earlier worktrees' METRICS files, which were not among the inputs); the 1e-12
reproduction of the 13 reference rows (only equality of the saved values was checked, which holds).

## Concerns

Statements in the account that the data do not support as written, with the values:

1. **Readout is a larger confound than the account states, and it enters the primaries.** The
   separate-variance bank12 primary (-15.09 pp [-17.47, -12.77]) is measured with the logit readout. The
   same fitted separate-variance models read out as posteriors give 35.44 (bank12) and 36.08 (bank6),
   i.e. -1.37 pp and +0.28 pp against the shared-variance posterior rows (36.81 / 35.81). The account's
   classification of this row as a "real learned-model failure" therefore describes a model x readout
   interaction: the fitted density is the same, the readout decides whether the row loses 15 pp or 1 pp.
   The same holds for CD-10 (posterior 35.99 / 36.28 vs logit 27.78 / 30.54 for H1) and for the depth
   amendment CD-10 bank12 row, where the retained logit readout gives 22.97 and the posterior readout of
   the identical second layer gives 34.68 / 0.7344 (account: "its CD variant on bank12 is far below").
   The account's own section-3 table omits the four posterior-readout amendment rows on bank12 (34.50,
   34.68) that are present in `depth_amended/COMPARISON.csv`.

2. **"Below the single-unit first layer on every endpoint" is false for one endpoint** (mismatch 2:
   bank6 CD-10 logit-input within 0.7374 > 0.7360). The correct statement is: below on ProcessBench,
   pooled AUC and PRMScore for all four retained-readout amendment rows; on within-AUC three of four are
   below and one is 0.0014 above.

3. **The learned rows are not consistently above their own initialization on within-AUC.** RBM12 initial
   posterior 0.7463 vs trained posterior 0.7387 and trained logit 0.7452; RBM6 initial 0.7441 vs trained
   0.7360; DUFS-6 initial 0.7442 vs trained 0.7410; low-correlation-6 initial 0.7453 vs trained 0.7405;
   48-column initial 0.7421 vs trained 0.5935. On ProcessBench the trained rows are +0.08 to +1.46 pp
   above their initializations except the 48-column bank (-16.94 pp). The highest within-AUC among
   answer-local rows remains the equal-weight varentropy contributions (0.7470, fixed formula). This is
   the evidence behind the framing that learned fusion has not shown a consistent overall advantage;
   the account's section-5 first row says this correctly, but section 2 of `STAGE1_COMPARISON_TABLE.md`
   classifies "RBM6 before training" and "RBM12 before training" as "fixed fusion" without noting that
   they match or exceed the trained rows on within-AUC.

4. **The section-5 readout line understates the entanglement** (mismatch 3). Any statement that the
   readout confound is "up to 1.5 pp" should be restricted to the trained single-unit RBM6 / RBM12 rows;
   over the whole record it is up to 17.3 pp on PB and 0.079 on within-AUC (34 pairs, values in
   `CHECKS.json` under `readout_pairs`).

5. **Saturated fits are not where the H4 losses concentrate, in rate terms.** The account says "fits with
   no saturated unit account for most losses" (true: 89.9% / 82.5%) but those fits are also 89.1% / 80.9%
   of the population. Loss rates per PB answer are 12.8% (0 saturated) / 7.6% (1) / 0.8% (2) on bank6 and
   10.8% / 9.6% / 2.5% on bank12, so the saturated fits lose less often, not more. This is descriptive
   (`pb_vs_exact1.by_unit_condition`), not a causal claim, but it means saturation cannot be offered as
   the mechanism of the capacity deficit.

6. **Section-1 wording**: "188 exact replays of the 21 non-failing original records" mixes units; 21 is
   the number of non-failing smoke answers, the non-failing records are 94 (108 - 14) and 188 = 94 x 2
   readouts.

7. **Point gains without intervals.** The three rows above the reference band (low-correlation-6 RBM
   36.99, shared-variance bank12 posterior 36.81, supervised step-BCE diagnostic 37.20) carry no
   registered primary interval against a raw reference in these inputs; low-correlation-6 is a selection
   control, not a registered primary, and the supervised row uses labels from other answers. The account
   handles these correctly as point estimates; any reader summary should keep it that way.

8. **Row duplication in the comparison file.** 110 answer-local rows contain 91 distinct score sets
   (`rbm6__old` alone appears seven times under different experiment names). The visual density of the
   reference band in the scatter partly reflects re-listing, and any count of "rows in the band" must be
   made over distinct score sets.

Nothing in the data contradicts the account's overall classification (optimization limitation for H4,
representation property for the depth collapse, negative results at the registered budget, no
promotion). The corrections above concern the size of the readout confound, one "every endpoint" claim,
and the wording of two descriptive statements.

## Files

- `fig_pb_vs_within_scatter.png`
- `fig_primary_contrasts_forest.png`
- `fig_capacity_convergence.png`
- `fig_capacity_probe.png`
- `fig_stability_starts.png`
- `fig_depth_coverage_and_performance.png`
- `fig_pb_cells_heatmap.png`
- `CHECKS.json` (every recomputed number, matched and mismatched, plus the 34 readout pairs and the 16
  depth-amendment-vs-H1 deltas)
- `make_stage1_review_figures.py` (regenerates all of the above; read-only over inputs)
