# Label-free readout calibration toward the SML assumptions (2026-11-12)

Question (Omri): can a label-free criterion choose the step readout of each TELE16 token channel (today: top-10 mean
for all 11) so that the 16 columns better meet the SML assumptions? Pipeline is fixed: GSP-T (no PRM) + GAP count,
standalone base-model telemetry. Code: `calibrate.py` (arms), `qwen3_8b_pb.py`, `boot.py`, `figures.py`.

## Arms

| Arm | Readout choice (no labels) |
|---|---|
| A0 | top-10 for all channels (baseline, parity with gsp_ablation_baseline_v1 GSP-T + GAP) |
| A1 | per channel, the readout with least correlation to relative position and log token count |
| A2 | coordinate descent (2 sweeps) that minimizes the rank-1 misfit of the 16 x 16 correlation matrix |
| A3 | residualize all 16 columns on position and log token count, then A2 |
| A4 | residualize all 16 columns, top-10 readout |

Grid per token channel: top10, mean, top20pct, p90, median, frac_above, top10_drop1. The 5 step-statistic channels
have no grid. Qwen3-8B (PB, Mind the Gap scorer) has no readout grid and no token count: only A0 and A4 (position only).

## Results (change vs A0, 95% bootstrap CI, `BOOT_*.csv`)

| Benchmark | Arm | 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|---|
| PRMBench PRMScore | A2 | +0.7 [0.4, 1.0] | +0.7 [0.3, 1.0] | -0.9 [-1.3, -0.5] | +0.1 [-0.2, 0.3] |
| PRMBench PRMScore | A3 | -2.4 | -1.5 | -1.1 | -1.1 |
| PRMBench PRMScore | A4 | -1.3 | -2.3 | -1.7 | -0.9 |
| Socratic PRMScore | A2 | +1.3 [0.7, 2.0] | -0.8 [-1.2, -0.4] | +0.9 [0.5, 1.2] | +0.9 [0.4, 1.3] |
| Socratic PRMScore | A3 | -6.8 | -4.8 | -2.7 | -3.4 |
| Socratic PRMScore | A4 | -3.1 | -3.0 | +0.7 | -0.7 |
| Hard2Verify F1 | A2 | +3.2 [0.4, 6.0] | +0.3 [-0.5, 1.4] | -0.2 [-2.4, 2.2] | -0.4 [-3.4, 2.4] |
| ProcessBench macro SLA | A2 | +3.1 [1.0, 5.3] | +7.2 [5.4, 8.9] | -1.2 [-2.3, -0.2] | +1.0 [-0.8, 2.9] |
| ProcessBench macro SLA | A4 | -3.6 [-5.9, -1.3] | +6.8 [4.9, 8.7] | +2.9 [0.9, 4.9] | +0.2 [-2.0, 2.4] |

- A2: point gain in 11 of 16 cells; CI above 0 in 8 of 16, below 0 in 3 of 16 (PRMBench PURE, PB PURE, Socratic 72B).
- A3 (the planned main arm) and A4 lose on PRMBench in 4 of 4 cells. Removing position also removes error signal there.
- Qwen3-8B PB: A4 macro SLA 22.1 -> 36.6 (GAP); psi/eta out of range 2 -> 0.
- Success criterion set before the run in the chat with Omri, not in a repo file (PB SLA >= equal mean + GAP 38.4/38.3/33.6/35.3/36.9, PRMBench loss <= 1): not met.
  A2 PB = 38.0/36.2/28.3/33.9; A4 Qwen3 = 36.6.

## SML diagnostics (`RESULTS_*.csv`)

- A2 lowers the label-free rank-1 misfit in 16 of 16 cells, but conditional |corr| given the label goes down only a
  little and not in all cells: 11/16 for y=0 (mean -0.014), 9/16 for y=1 (mean -0.018); largest in Socratic
  (-0.06 / -0.09). PRMBench y=0: .281/.292/.284/.314 -> .280/.273/.286/.312. A2 gains come with more L-SML groups
  (A0 K = 3..4, A2 K = 4..7).
- A1 and A4 lower conditional |corr| (A4 16/16, A1 16/16 for y=0 and 15/16 for y=1) (for example PRMBench y=0 A4 .256/.250/.249/.297), but performance goes down.
  More conditional independence did not give better ranking here.
- A2 choices follow one pattern in most cells (`FIT_*.json`): level channels take top10_drop1 (drop the largest token); change channels
  energy_innovation (12/16) and top50_js (11/16) take mean / median / p90 / frac_above (smoother readouts);
  top15_turnover (6/16) and dominant_freq16 (6/16) do not follow this; in Hard2Verify almost all channels take top10_drop1.
- Position-matched PB FE-AUC (new): A0 .722/.720/.711/.713; A2 .727/.739/.711/.710.

## Follow-up 2026-11-13: joint readout + grouping + vote threshold (`joint.py`)

Label-free criterion: the misfit of the hierarchical (bifactor) correlation model that GSP assumes (one error factor
across groups, one extra factor inside each L-SML group). Alternation: fix groups -> coordinate descent on readouts ->
recompute L-SML groups -> repeat (max 4 rounds). The regrouping step is L-SML, not misfit minimization, so the
alternation does not always lower the misfit (it can cycle), and the output is the last round, not the best one.
B1 = readout + grouping on the continuous columns. B2 = B1 + a vote rule per column (GAP, top-1, top-2, quantile .8,
.9; votes with error share < 2% or > 98% not eligible), misfit on the +-1 votes, and GSP-T uses these votes.

**Correction 2026-11-13 (code review).** The first B2 run oriented the votes by the fixed channel mean of B1, while
GSP-T orients the continuous columns by the L-SML score; in 10 of 16 cells at least one column had its votes and its
score on opposite sides. B2 now orients by the L-SML score of each round, and at scoring uses the votes that match the
sign GSP-T uses. The table below is the corrected run. B1 and A0-A4 did not change.

| Benchmark | Arm | 7B | 72B | PURE | RLHFlow |
|---|---|---|---|---|---|
| PRMBench PRMScore | B1 | -0.5 [-0.9, -0.1] | +0.0 [-0.3, 0.3] | -2.2 [-2.5, -1.8] | -0.2 [-0.5, 0.1] |
| PRMBench PRMScore | B2 | -6.6 [-7.1, -6.0] | +0.4 [0.1, 0.7] | -1.4 [-1.8, -1.0] | -0.2 [-0.5, 0.1] |
| Socratic PRMScore | B1 | +2.5 [1.8, 3.2] | +0.7 [0.3, 1.1] | +0.5 [0.1, 0.8] | +1.0 [0.5, 1.4] |
| Socratic PRMScore | B2 | +2.9 [2.3, 3.5] | -0.1 [-0.6, 0.3] | +1.0 [0.6, 1.4] | +0.8 [0.5, 1.1] |
| Hard2Verify F1 | B1 | +2.8 [0.2, 5.5] | -0.4 [-1.2, 0.2] | -14.1 [-20.0, -8.4] | -1.2 [-4.3, 1.9] |
| Hard2Verify F1 | B2 | +2.8 [0.3, 5.4] | -0.8 [-2.9, 1.2] | -26.2 [-31.7, -20.2] | +0.6 [-1.6, 2.8] |
| ProcessBench SLA | B1 | +2.2 [0.4, 4.0] | +6.1 [4.5, 7.7] | +0.6 [-0.5, 1.6] | +1.4 [-0.4, 3.3] |
| ProcessBench SLA | B2 | +1.5 [-0.4, 3.5] | +7.6 [5.4, 10.0] | +0.7 [-0.3, 1.7] | +1.5 [-0.3, 3.2] |

- Point gain (> 0.05): A2 11/16, B1 9/16, B2 10/16. CI above 0: A2 8, B1 7, B2 6. CI below 0: 3 for each arm.
- New B2 loss: PRMBench 7B -6.6 (was -1.6 in the first run).
- Conditional dependence of the votes given the label: B2 lowers mean |phi corr| in 15/16 cells (y=0) and 16/16 (y=1).
  But phi shrinks when votes fire less often, and B2 often picks rare rules (q90, top1/top2). With Yule's Q, which does
  not depend on how often votes fire, B2 lowers it only in 12/16 (y=0) and 13/16 (y=1); PRMBench 7B y=0 goes up
  .356 -> .502. Not a clear gain in conditional independence. A1/A4 already lowered conditional |corr| of the
  continuous columns (section above), so B2 is not the first label-free arm to do so.
- PB SLA B1 37.1/35.1/30.0/34.3, B2 36.4/36.6/30.1/34.4: all 8 above A0, all below equal mean + GAP (38.4/38.3/33.6/35.3).
- PURE fails: Hard2Verify B1 -14.1 / B2 -26.2 and PRMBench B1 -2.2 / B2 -1.4.
- Convergence: 6 of 32 alternations reach the 4-round cap (B1 3, B2 3); after the 4th round the grouping is not checked.
  GSP-T then runs L-SML again on the final columns, so the scored groups can differ from the groups the readouts were
  tuned for.
- Caveats (results audit): bootstrap CIs keep each arm's choices fixed (the label-free search is not refit inside the
  bootstrap), so they understate uncertainty for A2/B1/B2; the 16 cells share telemetry within a benchmark and are not
  16 independent tests. Shuffled-label null (Hard2Verify 7B, first B2 run): about 1.2 of B2's F1 gain survives
  shuffling, so part of it comes from how many steps are flagged, not which. top1/top2 rules use the earliest step on
  ties; top15_turnover and dominant_freq16 tie at the maximum in most answers, so for them these rules become
  position votes (label-free, but a position signal).
