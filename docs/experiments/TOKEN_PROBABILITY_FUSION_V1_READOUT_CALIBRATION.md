# Calibrating the step readout — the first positive result in this line

Runs `scripts/diagnostics/readout_calibration_c1_v1.py` (C1, with L-SML) and
`readout_calibration_v1.py` (CT7-style, fit-free). Outputs
`READOUT_CALIBRATION_C1.json` and `READOUT_CALIBRATION.json`. Development-only.

![Readout calibration](../../results/token_probability_fusion_v1/figures/readout_calibration.svg)

## The question

The feature-behaviour analysis showed the long-chain deficit is a **selection** problem,
not a feature problem: the per-step evidence is as strong on OlympiadBench as on GSM8K,
but argmax has to beat more competitors. The one component never examined is the
aggregation that turns tokens into a step value — `Top-K mean` with K fixed at 10.

Two questions, deliberately separated:

1. What K is best? — a **label-using** answer, a ceiling.
2. **Can a label-free criterion pick it?** — the question that decides whether this is a
   method or an oracle.

**An architecture correction from Omri, applied here.** The first run used CT7's
architecture, which contains **no L-SML at all** (CT7 is an equal-weight mean of seven
views, no fitted parameter). Since Stage B showed the L-SML advantage exists only when
fusion happens at token level *before* the readout, the grid belongs in C1. It is run in
both; C1 is the primary. The L-SML fit is over token rows and never sees a step, so it
does not depend on K: the fit is done once and every readout is a sweep over the same
fused series.

## Result — C1 architecture, with L-SML

| readout | L-SML | equal | L-SML−equal | GSM8K | OlympiadBench | Omni-MATH | repro (label-free) |
|---|---|---|---|---|---|---|---|
| K=1 | 20.42 | 18.15 | +2.28 | 28.99 | 15.73 | 16.27 | 17.51 |
| K=5 | 31.93 | 28.88 | +3.05 | 43.48 | 26.85 | 27.34 | 25.14 |
| **K=10 (today)** | **35.92** | 32.59 | +3.33 | **48.79** | **30.79** | **30.70** | 40.48 |
| **K=20** | **37.01** | 33.83 | +3.18 | 49.52 | 32.75 | 32.28 | 56.89 |
| **K=40** | 36.74 | 33.83 | +2.92 | 48.07 | **33.89** | 32.28 | **67.90** |
| K=80 | 34.18 | 31.04 | +3.13 | 40.82 | 31.77 | 32.21 | 67.83 |
| K=160 | 27.13 | 24.02 | +3.11 | 33.09 | 26.48 | 24.70 | 61.61 |
| K=all (step mean) | 23.69 | 20.98 | +2.71 | 31.64 | 22.31 | 20.09 | 59.89 |
| q=0.1 | 20.65 | 15.36 | +5.29 | 26.09 | 18.46 | 19.96 | 25.66 |
| q=0.4 | 25.54 | 20.49 | +5.05 | 35.51 | 23.90 | 21.81 | 48.27 |

K=10 replays the published 35.92 exactly, so the grid sits on a verified anchor.

**A wider readout is better, and the gain is concentrated where we were losing.**
Against today's K=10: K=20 gives **+1.96 pp on OlympiadBench and +1.58 on Omni-MATH**
while also gaining on GSM8K; K=40 gives **+3.10 on OlympiadBench** at the cost of 0.72
on GSM8K. The long subsets peak at a wider readout than the short ones — which is exactly
the length-conditional behaviour that both earlier attacks failed to find.

**My scale-free hypothesis is falsified.** I argued that a fixed quantile should beat a
fixed count because `Top-10` is the top 17.6% of a GSM8K step and the top 10.1% of an
Omni-MATH step. Every quantile variant is far worse (20.7–25.5 against 35.9), and so are
the hybrids. Evidence of an error is apparently concentrated in an **absolute** number
of tokens, not a fraction of the step — which is why making the statistic scale-free
dilutes it.

The L-SML advantage over equal weighting survives across the whole grid (+2.7 to +3.3 pp
for count rules), so this is not a readout change that quietly removes the fusion's role.

## Can a label-free criterion find it?

| criterion | picks | its SLA | cost vs the ceiling |
|---|---|---|---|
| **1. split-half reproducibility** | **K=40** | **36.74** | **0.27 pp** |
| 2. top1−top2 margin | q=0.05 | 17.55 | 19.47 pp |
| 3. length decoupling | K=max(5,0.2n) | 24.58 | 12.43 pp |

**Criterion 1 works.** Split the tokens of each step odd/even, apply the readout to each
half, and ask whether the two halves choose the same step. The K that maximises that
agreement is K=40, which costs **0.27 pp** against the label-using optimum of K=20 — and
is **+0.82 pp better than the K=10 we use today**, picked with no labels at all.

Criterion 3 fails as predicted before the run: Step 420 already established that
minimising length coupling costs accuracy because length is real evidence. Criterion 2
fails on its own, which is why it was never allowed to stand alone.

### The falsification test that mattered

Reproducibility rose monotonically with K across the first grid (43% at K=1 to 71% at
K=40). A criterion that only ever increases is not selecting anything — it would name the
largest K in whatever grid it was handed, and its apparent success would be an artefact
of where I stopped. The grid was therefore extended to K=80, K=160 and the full step
mean.

**It is not monotone.** Reproducibility peaks at K=40 and falls away: 67.90 → 67.83 →
61.61 → 59.89, while accuracy turns over in the same region. The criterion has a genuine
interior optimum and tracks the accuracy curve's shape. The same holds in the
CT7-architecture run (70.80 at K=40, then 68.50, 53.83, 48.99).

## What this is, and what it is not

**It is** the first thing in this line that improves the long subsets, and it comes with
a label-free rule that finds it. Both architectures agree on the shape and on the pick.

**It is not** a confirmed candidate. Three limits, stated plainly:

- The criterion was evaluated on the same development population it selected on. A real
  claim needs the rule frozen and re-run on data untouched by this session.
- K=20 versus K=40 is inside the criterion's resolution; it distinguishes "around 20–40"
  from "10" or "160", not 20 from 40.
- No paired interval is computed here. The +1.09 pp mean gain and the +1.96/+3.10 pp
  long-chain gains are point estimates on a grid, and a grid maximum is biased upward.
  Intervals against K=10 on the frozen pick are the next step, not an afterthought.

Two follow-ups this suggests: **freeze `K` chosen by split-half reproducibility and test
it with paired intervals**, and check whether the criterion also picks a **per-subset K**
correctly — cell identity is available at scoring time, and the long subsets clearly peak
wider than GSM8K.

---

## Correction — 2026-09-18, after the paired intervals were computed

This document called the readout result "the first positive result in this line."
**That was too strong, and the intervals say so.** They were listed above as a missing
next step; they have now been run, on the same paired source-group bootstrap used
everywhere else in this line:

| contrast (C1, gate-free mean SLA) | difference | 95% CI | verdict |
|---|---|---|---|
| K=20 − K=10 | +1.09 pp | [−0.01, +2.23] | **includes zero** |
| K=40 − K=10 | +0.93 pp | [−0.56, +2.43] | includes zero |

So **no readout width is established as better than the K=10 we already use.** The
point estimates still favour a wider readout, and the per-subset pattern — the gain
landing on OlympiadBench and Omni-MATH — is still the right shape. But a point estimate
on a grid, with the maximum selected from that same grid, is exactly the situation where
an interval is required before the word "improvement" is used, and this one does not
clear it.

What survives unchanged: the falsification test on criterion 1 (reproducibility is not
monotone, it peaks at K=40 and falls), the failure of the quantile family, and the
failure of criterion 3 as predicted. What does not survive: the claim of a gain.
