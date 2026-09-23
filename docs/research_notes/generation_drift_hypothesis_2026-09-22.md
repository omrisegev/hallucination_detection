# The generation-drift hypothesis, stated and measured (Step 430, 2026-09-22)

Data: `results/competition_diagnostics_v1/A1_*.csv`, protocol
`docs/experiments/COMPETITION_DIAGNOSTICS_V1.md`. Development population (6,800 ProcessBench
answers), labels used only to define the diagnostic populations. Everything below is in
within-answer standard-deviation units of the step score (mean 0, SD 1 per answer).

## The hypothesis as it was stated

The token statistics the bank reads (entropy, varentropy, surprisal, margin, tail mass,
energy innovation, turnover, divergence, change-point mass) are not stationary along the
generation: their distribution drifts as the model moves away from the question. If the
drift is towards higher scores, an error-free late step competes with the first-error step
on drift alone, the argmax loses more often on long chains, and the fix is a null
distribution that depends on position, estimated from unlabelled answers.

## What the frozen population shows

**There is a strong drift, and it runs the other way.** On answers the frozen gate closes
(the label-free reference population; the label-clean population gives the same curve):

| series | slope over the answer (SD / unit position) | step 0 | step 1 | step 2 | steps 4-7 |
|---|---:|---:|---:|---:|---:|
| CT7 | -1.17 [-1.25, -1.09] | +0.67 | +0.43 | +0.02 | -0.27 to -0.40 |
| top5 pmf equal (bank) | -1.06 [-1.13, -0.99] | | | | |
| token L-SML | -0.80 [-0.88, -0.71] | | | | |
| entropy q15_H1, top5 | -1.13 | | | | |
| varentropy q15_VE1, top5 | -1.14 | | | | |
| chosen_surprisal, top5 | -1.63 | | | | |
| logprob_margin, top5 | -0.98 | | | | |
| energy_innovation, top5 | -1.06 | | | | |
| true_tail50, top5 | -0.42 | | | | |
| bocpd_p0, top5 | **+0.39** | | | | |
| energy_level, top5 | +0.06 (flat) | | | | |

The shape is not a ramp: it is a **start-of-answer excess**. The first step sits two thirds
of a standard deviation above the answer mean, the second still +0.4, and from the third
step on the null is flat at about -0.3 SD. The same excess exists at token level (entropy
-0.29 SD per unit relative position, surprisal -0.38, with the first tokens highest). Only
the change-point channel rises along the answer; the raw energy level is flat.

Consequences that the numbers confirm:

- On gate-closed answers the argmax of every fused locator falls in the last fifth of the
  answer in 13-18 % of cases against 26 % for a uniform draw, and at step 0 far above its
  uniform share: the competing peak that the drift manufactures is **early**, not late.
- **Early misses** of CT7 (27 % of erroneous answers) predict relative position 0.19 while the
  target sits at 0.58; the gate-closed null is +0.48 SD higher at the predicted position than
  at the target. These are the drift's misses.
- **Late misses** (34 %, growing to 42 % on 11+ steps) predict 0.67 against a target at 0.27,
  and the null there is **lower** by 0.46 SD [-0.48, -0.44]. Those competing peaks are
  genuine excursions above a lower null, not drift.
- On the pre-error steps of erroneous answers the drift is weaker (CT7 -0.58 [-0.95, -0.20];
  the bank fusion -0.24 [-0.64, +0.19]): erroneous answers start less "surprised" than clean
  ones, consistent with the gate reading the same start-of-answer excess.

## Verdict

The hypothesis survives as a **stationarity failure**, falls as an **explanation of the
long-chain deficit**. The channels do drift, strongly and consistently across channels and
depths, and a position-conditional null is the right label-free correction for the misses
that the drift causes. But those are the early misses; the late misses that dominate long
chains sit where the null is lowest, so a position-conditional null cannot rescue them and
may hand more decisions to them. Step 432 tests exactly that, with the plain and the
position-conditional null side by side and the position prior alone as a control.

What would revive the late-miss part of the hypothesis: a channel whose late excursions on
clean answers are as large as the first-error excursion (none of the eleven is; see the
A1 curves), or evidence that the late competing peaks are concentrated on a specific step
type that a different null could absorb.
