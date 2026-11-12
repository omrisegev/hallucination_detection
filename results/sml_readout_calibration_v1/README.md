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
- Pre-registered success criterion (PB SLA >= equal mean + GAP 38.4/38.3/33.6/35.3/36.9, PRMBench loss <= 1): not met.
  A2 PB = 38.0/36.2/28.3/33.9; A4 Qwen3 = 36.6.

## SML diagnostics (`RESULTS_*.csv`)

- A2 lowers the label-free rank-1 misfit in 16 of 16 cells, but conditional |corr| given the label does not go down
  (PRMBench y=0: .281/.292/.284/.314 -> .280/.273/.286/.312). The label-free criterion does not give conditional
  independence. A2 gains come with more L-SML groups (K 4 -> 4..7).
- A1 and A4 lower conditional |corr| (for example PRMBench y=0 A4 .256/.250/.249/.297), but performance goes down.
  More conditional independence did not give better ranking here.
- A2 choices follow one pattern in most cells (`FIT_*.json`): level channels take top10_drop1 (drop the largest token); change channels
  (energy_innovation, top50_js, top15_turnover, dominant_freq16) take mean / median / p90 (smoother readouts).
- Position-matched PB FE-AUC (new): A0 .722/.720/.711/.713; A2 .727/.739/.711/.710.
