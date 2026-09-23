# Results — change-point readouts as the first-error rule

Claude, 2026-09-19. Branch `claude/token-probability-fusion-v1`.
Pre-registration: `CHANGEPOINT_STEP_READOUT_V1_PREREGISTRATION.md`, written before any
number below existed. Development-only. Artefacts:
`results/token_probability_fusion_v1/{CHANGEPOINT_READOUT.json, CHANGEPOINT_WITHIN_ARM.json,
CHANGEPOINT_PEAKS.npz, IMPULSE_SHAPE_CONTROL.json, EVIDENCE_DROP.npz,
figures/changepoint_readout.*, figures/impulse_shape_control.*}`.

Anchor replayed exactly in every script before any new number was printed: the published
C1 token L-SML arm, mean gate-free SLA **35.92**. Its step readout re-derived from the
newly cached token fusion matches the frozen arm to **0.000e+00**.

---

## 1. The headline, in one table

Gate-free SLA (Mind-the-Gap protocol: exact first-error hit on erroneous ProcessBench
answers only, per cell, then the mean over eight cells). 4,442 erroneous answers.
Paired source-group bootstrap, 10,000 draws. Chance is 16.6.

| series | ~obs / answer | argmax | first crossing q=.9 | CUSUM alarm | CUSUM onset | BOCPD rise | BOCPD reset | best of ~130 settings |
|---|---|---|---|---|---|---|---|---|
| **our level readout, per step** (published) | 8 | **35.92** | 35.81 | 30.00 | 34.66 | 34.88 | 16.56 | 35.92 (argmax) |
| our level fusion, per token | 713 | 20.39 | 14.38 | 24.16 | 23.70 | 19.59 | 20.40 | 28.82 |
| their evidence drop, per step (mean of 5 worst) | 8 | 33.23 | 33.08 | 28.20 | 31.63 | 33.67 | 17.43 | 33.67 |
| their evidence drop, per step (single worst) | 8 | 22.65 | 23.04 | 20.63 | 22.95 | 22.84 | 15.35 | 25.30 |
| their evidence drop, per token | 713 | 22.67 | 13.48 | 25.24 | 26.34 | 22.47 | 24.16 | 28.40 |

**Nothing in the grid reaches 35.92** — not a pre-registered default, and not the
label-selected best of roughly 130 settings per arm, which is a ceiling and not a
candidate. The best non-incumbent cell in the whole table is 33.67.

---

## 2. Every pre-registered prediction, with its verdict

**P1 — the pairing hypothesis. FALSIFIED.** The missing 2x2 cell — a derivative statistic
read out by a first-crossing rule, the one that is actually theirs — scores **33.08** at
step granularity and **13.48** at token granularity against the anchor's 35.92, with
paired intervals of −2.84 [−4.58, −1.14] and −22.45 [−25.12, −19.74]. Both exclude zero
in the wrong direction. Its arm's label-selected ceiling, 33.67, also loses.

The handoff committed to the consequence in advance: *"If it does not, the pairing
hypothesis is wrong and this line is finished."* It does not. The line is finished. All
four cells of the 2x2 are now measured and the incumbent wins every one of them.

**P2 — the length hypothesis. Condition met, but it does not mean what it was written to
mean.** The drop arms do have a positive long-minus-short interaction against the anchor,
interval excluding zero: +3.68 [+0.27, +7.06] for the per-step drop, +5.21 [+0.78, +9.61]
per token. Ratio to chance by subset makes the shape plain:

| series / rule | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---|---|---|---|
| our level readout, argmax (anchor) | 2.34 | 1.85 | 2.27 | 2.22 |
| their evidence drop per step, argmax | 1.87 | 1.89 | 2.09 | **2.27** |
| their evidence drop per token, CUSUM onset | 1.60 | 1.51 | 1.59 | 1.68 |

So the qualitative pattern in their Table 3 **does** reproduce: relative to chance, the
drop statistic climbs with chain difficulty while ours is flat. But it climbs *from
below* — the interaction is bought by losing 0.47 of ratio on GSM8K to gain 0.05 on
Omni-MATH, not by winning anywhere. I pre-registered P2 as a one-sided test and it
passes; it is reported as met, and it is **not** evidence of an advantage. A future
version of this test should require the interaction *and* non-inferiority of the mean.

**P3 — Omri's change-point hypothesis, on the published series. FALSIFIED.** On the
~8-step series every change-point rule at its pre-registered default is at or below
argmax: CUSUM alarm −5.92 [−7.62, −4.25]\*, CUSUM onset −1.25 [−2.82, +0.29], BOCPD rise
−1.03 [−2.41, +0.30], BOCPD reset −19.35 [−21.67, −17.02]\*. Nothing beats it, and the
two rules that come closest have intervals covering zero.

**P4 — the granularity hypothesis. SUPPORTED, and it is the one real finding here.** See
section 3.

---

## 3. What Omri's idea actually does, once the statistic is held fixed

The anchor comparison confounds two axes: moving from the step readout to the token series
costs far more than any decision rule can return, so every token row loses regardless of
its rule. Re-referencing each rule to **its own series' argmax** isolates the question that
was asked.

| series | ~obs | CUSUM alarm | CUSUM onset | BOCPD rise | BOCPD reset | first crossing |
|---|---|---|---|---|---|---|
| our level readout, per step | 8 | −5.92 [−7.62, −4.25]\* | −1.25 [−2.82, +0.29] | −1.03 [−2.41, +0.30] | −19.35\* | −0.11 |
| **our level fusion, per token** | 713 | **+3.76 [+1.42, +6.12]\*** | **+3.30 [+0.95, +5.73]\*** | −0.80 | +0.02 | −6.01\* |
| their drop, per step (5 worst) | 8 | −5.02\* | −1.60 [−3.12, −0.07]\* | +0.44 [−0.86, +1.75] | −15.81\* | −0.16 |
| their drop, per step (worst) | 8 | −2.01\* | +0.31 [−0.86, +1.52] | +0.21 | −7.29\* | +0.39 |
| **their drop, per token** | 713 | **+2.57 [+0.46, +4.63]\*** | **+3.67 [+1.54, +5.73]\*** | −0.20 | +1.50 [−0.02, +3.02] | −9.19\* |

\* interval excludes zero.

**The mechanism is real, and it appears exactly where the theory says it should.** On a
long series, CUSUM beats argmax on **both** statistics, by 2.6 to 3.8 pp at the
pre-registered constants and by +8.43 [+6.36, +10.49] / +5.74 [+3.56, +7.90] at the
arm's label-selected ceiling. On the ~8-step series it loses on both. That split is the
prediction P4 made, and it is the direct answer to the question: a sequential detector
does add something an argmax cannot see — it needs a sequence to see it in.

**And it does not help us,** because the top-10 step readout is worth more than the rule
can recover. Our own token series read out by its best change-point rule reaches 28.82;
the same fusion read out by a Top-10 step mean and an argmax reaches 35.92. The readout
aggregation is a 7-to-15 pp effect and the decision rule a 3-to-8 pp one, in that order.

BOCPD is the weaker of the two throughout. Its `rise` curve never separates from argmax
anywhere (every interval covers zero), and `reset` is catastrophic on short series
(−19.35 pp), where its t=0 hazard convention and a ~8-point run-length posterior leave
it with almost no evidence. Note this is the **verified** reset-before-observation filter,
validated here against brute-force enumeration of every partition of every prefix to
1e-10 — the failure is the model's fit to this problem, not the old mixed-convention bug.

---

## 4. Why the step series defeats sequential detection: the shape

Measured before the grid was run, and recorded in `IMPULSE_SHAPE_CONTROL.json` and
`figures/impulse_shape_control.*`. The other working line reached the impulse finding
independently and first; this adds its controls.

**The error step is an isolated impulse.** On the published arm the step score at the
true first error is +0.65 SD with both neighbours at or below zero, and the error step is
a local maximum in 51.7–61.4% of answers against a **within-answer** null of 33.9–37.1%
— an excess of +17.8 to +24.2 pp. There is nothing for an accumulating statistic to
accumulate, which is why CUSUM loses on eight points and wins on seven hundred.

**Their statistic has the same shape.** Rebuilt exactly — `−H` of the renormalized
**top-20**, EMA span **5**, first difference, their M = 5 collapse — the profile is
+0.61 / +0.52 / +0.63 / +0.64 SD at the error, superimposable on ours. And the series
itself correlates **0.9996** with the bank's own `q15_H1` channel. The statistic axis
between the two arms is thinner than the handoff assumed: it is the derivative and the
smoothing, not the top-K.

**One claim in the handoff amendment reverses under control.** The amendment reports that
the mean after the error is 0.19–0.35 SD below the mean before it, read as "the model
becomes more confident after it errs". Both quantities are measured at the same index in
the same answer, and this readout family has a strong positional trend: the score
collapses by 0.4–0.6 SD in the last two deciles of **every** answer, error or not. The
mean first error sits at relative position 0.408, so "after" always contains the collapsed
tail and "before" never does. Compared against every admissible index in the same answer,
the sign flips — the excess is **+0.15 / +0.07 / +0.10 / +0.20** across the four subsets:
the true error's neighbourhood declines *less* than an arbitrary index does.

This does not rescue CUSUM — the impulse is what argues against accumulation, and the
impulse survives its control. It does void the specific reason given for BOCPD aiming one
step off, and it explains why the amendment's shape rules lost: `v[t] − mean(v[t+1:])`
mostly subtracts the positional tail collapse, which carries no information about where
the error is.

---

## 5. What is closed, and what is not

**Closed.** The decision rule as a route to improvement *on the step series*. Four
independent attempts now agree: first-crossing on the level series (handoff section 1),
the derivative under argmax (27.96), the five shape rules (amendment), and the full
change-point family here. For an isolated impulse in noise the argmax is close to the
matched-filter optimum, and on a ~8-point series there is no ordering evidence to exploit.
The missing 2x2 cell is measured and the pairing hypothesis is falsified.

**Closed.** The Mind-the-Gap statistic as a drop-in replacement for our level readout, at
either granularity, under any of the six rules.

**Not closed, and not tested here.** Whether a sequential change-point readout helps a
*better* token-level series. The finding in section 3 is that the rule needs a long series
and that our token fusion is a poor one (20.39 by argmax against the step readout's 35.92).
Those two facts together say the experiment worth running next is not another rule — it is
a token-level series good enough to deserve one. Nothing here bears on that.

**Not touched.** CT7, the gate, and every frozen release. No arm here is a candidate for
the method of record, and the token→step collapses `step_m5` / `step_worst` are our
pre-registered choices for a parameter the paper leaves undefined — this is an adaptation
and carries no author's name.
