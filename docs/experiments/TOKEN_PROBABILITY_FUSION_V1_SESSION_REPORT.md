# The token-probability line — what was tested and what was found

Branch `claude/token-probability-fusion-v1`, 2026-09-18. Written for someone who did
not watch the run. Every number traces to a JSON under
`results/token_probability_fusion_v1/`; the figures regenerate from those files via
`scripts/diagnostics/build_session_report_v1.py`.

**Everything here is development evidence.** The fold structure is source-disjoint, but
this population has been inspected many times before. Nothing below is a publication
confirmation, and no arm is proposed as a candidate.

---

## 1. In one paragraph

We asked whether the token-level fusion arm was really behind CT7 or only looked
behind because the two were scored under different no-error gates. It is really
behind: CT7 leads by ~4 pp with the gate removed and under either gate held fixed.
We then ran the pre-registered 2×2 that was supposed to explain the arm's one
genuinely positive result — L-SML beating equal weighting — and it did explain it, in
the deflationary direction: **the advantage exists only when fusion happens before the
step readout, and vanishes once the readout averages first.** Separately we attacked
the long-chain deficit against the published literature from two directions. Both
attacks failed, and the second failed for a reason that reframes the problem: the
length coupling is not in the level-versus-derivative choice, it is in the top-M
aggregation that both use. Finally, the noise floor — computed rather than assumed —
showed that a fully shuffled pipeline still localizes at 25–30% against a 16.6% chance
floor, which puts every absolute number in this report in a harsher light.

---

## 2. Stage A — is the gate hiding the result?

Step 422 reported the token arm at 34.08 ProcessBench macro-F1 against CT7's 41.19 and
warned the two were not comparable, because CT7 runs under its frozen non-digit tail15
gate and the token arm under LOCO-5 at 0.33. That warning was right. The conclusion
drawn from it — that the gate was *hiding* the token result — was too strong.

CT7's scores were restored from the Drive backup of 2026-09-17 rather than re-derived;
its frozen-candidate JSON is byte-identical to master's. The scorer is imported from the
Step-422 diagnostic rather than rewritten, and reproduces every number that step
published, including its paired interval to the reported digits.

### Gate-free, every locator on one protocol

Per-subset Step-level Localization Accuracy on erroneous answers, no gate at all.

| subset | CT7 | token L-SML | token equal | chance | Chen Shannon-Drop |
|---|---|---|---|---|---|
| GSM8K 4B | **48.31** | 47.83 | 44.93 | 20.84 | 43.42 |
| GSM8K 8B | 47.34 | **49.76** | 41.06 | 20.84 | 46.11 |
| MATH 4B | **35.69** | 34.85 | 30.30 | 18.10 | 32.03 |
| MATH 8B | **36.87** | 31.99 | 29.80 | 18.10 | 32.90 |
| OlympiadBench 4B | 39.03 | 30.56 | 27.53 | 13.55 | **43.06** |
| OlympiadBench 8B | 37.67 | 31.01 | 29.80 | 13.55 | **41.52** |
| Omni-MATH 4B | 37.15 | 30.96 | 29.51 | 13.84 | **38.04** |
| Omni-MATH 8B | 37.02 | 30.43 | 27.80 | 13.84 | **37.04** |
| **mean** | **39.89** | 35.92 | 32.59 | 16.58 | 39.27 |

### The gate held fixed, both directions

| gate | answers opened | CT7 | token L-SML | token equal |
|---|---|---|---|---|
| CT7 frozen tail15 | 4,556 | **41.19** | 37.32 | 34.88 |
| LOCO-5 @ 0.33 | 5,391 | **36.98** | 34.08 | 32.34 |

The 7.11 pp original gap decomposes additively and exactly, by either route:
**3.86 locator + 3.25 gate = 7.11**, and **4.21 gate + 2.90 locator = 7.11**. About
half was the locator and half the gate; the ranking was never in question.

| contrast | difference | 95% CI |
|---|---|---|
| gate-free mean SLA, CT7 − token L-SML | **+3.96 pp** | [+2.28, +5.59] |
| macro-F1 under tail15, CT7 − token L-SML | +3.86 pp | [+2.57, +5.15] |
| macro-F1 under LOCO-5, CT7 − token L-SML | +2.90 pp | [+1.95, +3.85] |

### Three things that fell out

**LOCO-5 is the worse gate, not a mistuned one.** tail15 beats it for *both* locators,
and LOCO-5's own label-selected optimum at 0.41 (36.84) is still **0.48 pp below**
tail15 at its registered setting (37.32). The 0.41 "headroom" is worth close to nothing,
and a label-free gate rule rises in priority.

**CT7 ties to slightly exceeds a published ICML method on exactly our population** —
39.89 against Chen et al.'s Shannon Drop 39.27, as a *level* signal against a
*derivative*. The project has never had a comparison of this kind. It carries an explicit
caveat: **their SLA takes the first step crossing a threshold, not an argmax, and they
run at K=20.** It is the closest public comparison that exists, not a controlled
experiment. Per cell CT7 wins GSM8K and MATH and still loses OlympiadBench.

**The length asymmetry survives in CT7, and is much smaller there.** Worst long-chain
deficit against Chen: CT7 −4.03 pp, token arm −12.50 pp; mean over the four long cells
−2.20 against −9.17, while both beat Chen on the four short cells (+3.44 / +2.49). So
the asymmetry is a property of level readouts generally, not of the token representation.

---

## 3. The length axis — two attacks, both closed

![Both attacks on the length axis](../../results/token_probability_fusion_v1/figures/length_axis.svg)

### Attack 1 — subtract the length prior out of the level statistic

Step 420 built a length-calibrated readout and measured that neutralising the length
coupling costs 8.16 PB points, with only 3.35 recovered by adding an explicit log-length
view. Both were **macro** numbers. The decomposition by subset had never been computed,
and the decision rule was fixed before running it: uniform cost means the tool is dead;
negative-on-short and positive-on-long means it is a length-*conditional* mechanism.

The rebuild is gated — it refuses to print anything new unless all five Step-420 macro
numbers reproduce. They do, within 0.0354 pp.

| | short (GSM8K, MATH) | long (Olympiad, Omni) | **long − short** |
|---|---|---|---|
| LX7 − CT7 | −10.85 [−14.24, −7.54] | −10.42 [−12.83, −8.03] | **+0.43 [−3.68, +4.60]** |
| LX8 − CT7 | −6.41 [−9.41, −3.52] | −6.35 [−8.40, −4.32] | **+0.06 [−3.57, +3.65]** |
| CT7+LEN − CT7 | +0.74 [−0.62, +2.10] | −0.00 [−1.06, +1.06] | −0.74 [−2.44, +0.94] |

**Uniform.** Calibration hurts OlympiadBench and Omni-MATH by the same amount it hurts
GSM8K and MATH. What rules the mechanism out is the *location* of the interaction, not
the width of its interval: a readout that selectively rescued long chains would show a
large positive interaction, and the estimates are +0.43 and +0.06, centred on zero.

Worth keeping: **`LEN` alone** — pure step length — scores 41.06 on GSM8K and 37.11
averaged over the short subsets against 26.44 on the long ones. Step 420's "length is
evidence, not only bias" caveat shows up clearly per subset.

### Attack 2 — use a derivative instead of a level

Built from the mechanism rather than the recipe: EMA smoothing, first differences, keep
the rises, mean of the sharpest M inside each step. The step-level aggregation is ours
(the published score is per answer), it runs on all eleven oriented channels rather than
entropy alone, and both constants are the bank's own. Nothing is labelled with an
author's name.

| arm | short | long | mean | corr with log tokens-per-step |
|---|---|---|---|---|
| CT7 | 42.05 | 37.72 | **39.89** | **+0.477** |
| token LEVEL (equal) | 39.40 | 30.28 | 34.84 | +0.665 |
| token DERIVATIVE (equal) | 34.89 | 21.03 | 27.96 | **+0.583** |
| LEVEL + DERIVATIVE (equal) | 38.89 | 27.34 | 33.11 | +0.674 |

| contrast | short | long | **long − short** |
|---|---|---|---|
| DERIVATIVE − CT7 | −7.16 | −16.69 | **−9.53 [−13.97, −5.28]** |
| DERIVATIVE − LEVEL | −4.50 | −9.26 | −4.76 [−8.77, −0.88] |
| LEVEL+DERIVATIVE − LEVEL | −0.51 | −2.95 | −0.44 [−2.50, +1.63] |

**The derivative is worse, and worse specifically on long chains** — the opposite of the
prediction. Adding it to the level bank buys nothing.

**And the premise was false.** The argument for attacking from this side was that a
derivative "uses a statistic that does not carry the length prior in the first place."
Measured with Step 420's own diagnostic, the derivative's within-answer correlation with
log tokens-per-step is **+0.583** — barely below the level readout's +0.665, and *higher*
than CT7's +0.477. The module's docstring recorded this as an open question rather than
an assumption, and the measurement settled it against the premise: **"mean of the M
largest rises in a step" is itself an order statistic, so it reintroduces the same
length coupling.** The length prior lives in the top-M aggregation shared by both, not in
the level-versus-derivative choice.

---

## 4. Stage B — the 2×2, and every prediction against what happened

![Stage B interaction](../../results/token_probability_fusion_v1/figures/stage_b_interaction.svg)

Four cells over one bank, one fold set, one code path; cells differ only in which matrix
enters and whether it is standardized within the answer. **C1 replays the published arm
exactly** — max absolute difference 0.0 on both fusion rules, peak agreement 1.000.

| cell | standardization | fusion | L-SML | equal | L-SML − equal | vs C1 |
|---|---|---|---|---|---|---|
| **C1** | pooled | before | **35.92** | 32.59 | **+3.32 [+1.90, +4.75]** | — |
| **C2** | answer-local | before | 33.14 | 31.92 | +1.22 [+0.36, +2.11] | −2.79 [−4.07, −1.53] |
| **C3** | pooled | after | 33.64 | 33.41 | +0.22 [−0.42, +0.86] | −2.28 [−3.89, −0.66] |
| **C4** | answer-local | after | 34.95 | 34.84 | +0.10 [−0.73, +0.91] | −0.97 [−2.27, +0.30] |

C4 is CT7's architecture applied to this bank, which is what makes this table a bridge
between the two lines rather than four arbitrary arms.

### Predictions versus outcomes

| # | prediction, recorded before the run | outcome | verdict |
|---|---|---|---|
| **P2** | the L-SML advantage shrinks substantially or vanishes once fusion moves after the readout | +3.32 → +0.22 (pooled), +1.22 → +0.10 (answer-local); both post-readout intervals include zero | **CONFIRMED** |
| **C2 level** | above 35.92 | 33.14 | **FALSIFIED** |
| **C3 level** | at or above 35.92 | 33.64 | **FALSIFIED** |
| **C4 level** | highest of the four | 34.95, second to C1 | **FALSIFIED** |
| **C2 gain** | smaller than C1 | +1.22 vs +3.32 | CONFIRMED |
| **C3 gain** | much smaller, may include zero | +0.22, includes zero | CONFIRMED |
| **C4 gain** | smallest, closest to zero | +0.10, smallest | CONFIRMED |
| **sign** | `chosen_surprisal` non-negative in a majority of folds in C2 **and** C4 | C2 0/5, C4 5/5 | **HALF FALSIFIED** |
| **P1a** | conditional PR is 2–4 in every cell | 4.46 – 5.01, all above the band | **FALSIFIED** |
| **P1a** | lower in the answer-local cells | C2 4.46 < C1 5.01; C4 4.57 < C3 5.01 | CONFIRMED |
| **P1b** | weight IPR stays above 8 in all four | 9.69 / 10.14 / 10.40 / 9.77 | **CONFIRMED** |

### What the falsifications mean

**Correcting the "protocol deviation" made things worse.** C1 — the arm fitted on a
pooled between-answer matrix and read out within the answer — is the best of the four.
Aligning the fit axis with the readout axis costs 2.79 pp, and fusing after the readout
costs 2.28 pp. Fitting between answers and applying within them is doing work, not
merely producing an artefact. That was the opposite of the pre-registered expectation.

**The `chosen_surprisal` sign flip has nothing to do with the standardization axis.**

| | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| weight, all five folds | −0.233 | −0.250 | **+0.225** | **+0.331** |
| non-negative in | 0/5 | 0/5 | **5/5** | **5/5** |

Answer-local standardization was predicted to remove the flip, on the theory that
between answers surprisal tracks difficulty while within an answer it tracks error. It
does not (C2, 0/5). The axis that controls the sign is the **fusion stage**: both
post-readout cells flip it positive. The pre-registration named this exact outcome as
the condition falsifying its mechanism story, and it is falsified.

**A structural note.** C1 and C3 return identical conditional PR (5.01) because a Top-10
mean is affine-equivariant: their view matrices coincide up to a per-channel affine map,
which leaves correlations unchanged. The participation ratio simply cannot separate those
two cells; only the fusion stage can.

---

## 5. The noise floor — computed, not inferred

![PR against its computed floor](../../results/token_probability_fusion_v1/figures/pr_vs_noise_floor.svg)

The floor is a separate computation, as required: the *entire* pipeline — shuffle,
standardize, fit L-SML on donor folds, score, build views — re-run on tokens permuted
within each answer independently per channel. That keeps each answer's per-channel
marginal exactly and destroys token order and cross-channel alignment.

| cell | conditional PR | its null | weight IPR | its null | **null's own SLA** |
|---|---|---|---|---|---|
| C1 | 5.01 | 4.16 | 9.69 | 8.25 | **25.12** |
| C2 | 4.46 | 3.29 | 10.14 | 8.26 | **25.77** |
| C3 | 5.01 | 4.16 | 10.40 | **10.62** | **29.28** |
| C4 | 4.57 | 3.29 | 9.77 | **10.97** | **30.01** |

**The null does not collapse.** The pre-registered reading applies: the PR contrast
therefore says little about dimensionality. Shuffled data retains a conditional PR of
3.3–4.2 against a measured 4.5–5.0, and in both post-readout cells the shuffled *weight*
IPR is **higher** than the real one. Most of what looked like spread across many
independent channels is reproduced by independent noise with the same marginals.

**The sharper number is the last column.** A fully shuffled pipeline — no token order,
no cross-channel alignment — still localizes at **25–30%** against a 16.58% chance floor.
C4 sits only **4.9 pp above its own null**. Much of every absolute SLA in this report is
reachable from step length and per-answer marginals alone. This is consistent with the
length result in §3, where pure step length scores 31.78 mean gate-free SLA by itself,
and it is the single most sobering measurement of the session.

---

## 6. The per-cell screen

Per-cell adaptation means fitting covariance and L-SML separately per cell, label-free,
using only cell identity — available at scoring time. Before building it, the screen
asks whether the nine cells have different covariance at all. The object compared is the
**marginal** correlation matrix, which is the point rather than a compromise: L-SML never
sees labels, so the marginal covariance is exactly what a per-cell fit would fit.

| comparison | n pairs | median distance | vs split-half baseline |
|---|---|---|---|
| ProcessBench vs ProcessBench | 28 | 0.0565 | **3.04×** |
| ProcessBench vs PRMBench | 8 | 0.1278 | ~7× |

By the screen's own rule the idea is not closed. But the structure is **misaligned with
the deficit**: the smallest distance anywhere in ProcessBench is **MATH vs OlympiadBench
at 0.0234**, barely above their 0.013–0.018 baselines — nearly the same covariance to
fit, in the two cells whose performance diverges most, and OlympiadBench is where we lose
12.5 pp to Chen. The most *distinctive* cell is GSM8K, where we are already strongest.

The larger finding was incidental: **PRMBench sits 0.0401 from the pooled matrix while
every ProcessBench cell sits 0.084–0.106 from it.** The shared standardizer is closer to
PRMBench than to anything it scores. With 37% of the tokens and the opposite shape (13.5
steps per answer at 27 tokens per step, against ProcessBench's 5–9 steps at 55–93), it
drags the fit into a regime no ProcessBench cell occupies. That is fixable under the
same access contract by not fitting on it, and is cheaper than any per-cell variant.

---

## 7. What is closed, what is open

**Closed this session.** The gate explanation for the token arm's deficit. The
length-calibrated readout (attack 1). The derivative step channel as built (attack 2).
The claim that the token arm's L-SML gain is a fusion gain rather than a readout
artefact. The reading of 9.69 as evidence of nine independent directions.

**Open, in the order the evidence argues for.**

1. **A label-free gate rule.** tail15 beats LOCO-5 for both locators and beats its
   label-selected optimum. This is the cheapest remaining gain and does not depend on
   anything above.
2. **Excluding PRMBench from the ProcessBench fit.** One-line change, same access
   contract, directly motivated by §6.
3. **The top-M aggregation itself.** §3 relocates the length prior from the choice of
   statistic to the aggregation shared by every arm we have. That is the real target, and
   neither attack touched it.
4. **Per-cell fitting** stays unbuilt: permitted by its screen, but the structure it
   would exploit is in the wrong cells.

**What should not be inferred.** None of this shows the token representation is
worthless — it beats chance everywhere and beats CT7 on GSM8K-8B. It shows that its one
positive result was a readout artefact, that it trails CT7 under every gate condition
tested, and that a large part of what every arm here achieves is available from step
length alone.
