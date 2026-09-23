# Corrections after independent audit — Steps 424–426

Claude, 2026-09-19. Two independent audits were commissioned by Omri on the session's own
work: one on implementation correctness, one on whether the conclusions follow from the
numbers. Both found real defects. This document records every correction, with the
superseded claim quoted, so that nothing is quietly rewritten.

**Nothing in Steps 424–426 is deleted.** The frozen artefacts stay. The claims below are
marked superseded and replaced.

---

## 1. Code defects, found and fixed

### D1 — the whitener's padding manufactured a step-0 bias (REAL, changes numbers)

`spectral_utils/step_measure_v1.py:whiten` padded the pre-sample history with `x[0]`.
Because `x[0]` is systematically the lowest-risk token of an answer (≈ −1.65 SD), this
crushed `e[0]` to `0.42·x[0]` and injected a **+1.06 offset into `e[1]` of every answer**.
Measured consequence: the whitened arms predicted **step 0 for 22.0–24.2%** of erroneous
answers against a true base rate of **12.25%** and against raw's well-calibrated 12.1–13.0%.

My docstring justified the padding by saying that dropping the first samples "would make
the first step unscoreable, and 12.25% of first errors are at step 0". That reasoning was
exactly backwards: **the padding is what distorted step 0.**

Fixed with the statistically correct treatment rather than another guess: `yule_walker`
now returns the **ladder** of every lower-order predictor from the same autocorrelation,
and `whiten` uses the order-*t* predictor at position *t < order*. At *t = 0* there is no
history, so the predictor is the unconditional mean and `e[0] = x[0]`.

| arm | as published | corrected |
|---|---:|---:|
| whitened Top-10 | 33.68 | **33.98** |
| whitened Top-20 | 35.42 | **35.59** |
| whitened Top-40 | 35.63 | **35.80** |
| P1 ceiling contrast (best whitened − best raw) | −1.38 [−3.08, +0.27] | **−1.21 [−2.88, +0.43]** |
| P1 at matched width K=10 | −2.24 [−3.91, −0.57]\* | **−1.94 [−3.56, −0.28]\*** |

P1 is still falsified — the optimum still moves the wrong way (20 → 40) and the peak still
falls (37.01 → 35.80) — but every whitened number was depressed by 0.2–0.3 pp by my bug,
and three new tests now pin the first-`order` moments to the steady state.

### D2 — the "label-free" selection criterion was label-conditioned (REAL, changes a conclusion)

`step_measure_eval_v1.py` computed split-half reproducibility **only on rows with
`target >= 0`** — i.e. the population of a supposedly label-free selector was chosen by
the label — and aggregated it as a flat mean over rows rather than as the per-cell macro
the endpoint uses.

That is also the true cause of the K=80 disagreement with the established run.
`STEP_MEASURE_V1_RESULTS.md` §4 guessed it was "how each caps K against a half-step
shorter than K". **That guess was wrong**; the two K-capping implementations are
equivalent. The audit decomposed it exactly: micro/erroneous 0.7267 → macro/erroneous
0.7096 → micro/all 0.6985 → macro/all 0.6783. GSM8K's reproducibility collapses at K=80
(0.564 on all PB answers), and the flat row mean gives GSM8K 9.3% weight where the macro
gives it 25%.

Fixed: reproducibility is now computed over **all 6,800 ProcessBench answers** and
aggregated as a per-cell macro over the eight cells.

| | as published | corrected |
|---|---|---|
| label-free pick, raw family | K=80 (rep 0.727), SLA 34.18 | **K=40 (rep 0.6790), SLA 36.74** |
| agreement with the established run | disagreed at K=80 | **matches (0.679 vs 0.679)** |

So the criterion does select a sensible width after all, and the discrepancy I left open
is closed. The corrected label-free pick is **K=40, +0.82 [−0.73, +2.34]** over the
deployed K=10 — interval covering zero.

### D3 — the P2 "key contrast" crossed two axes (REAL, mislabelled)

The −6.65 pp quoted as *the contiguity effect* was `white_pooled__win16 − raw__topk20`,
which is whitening **and** contiguity. Fixed; both are now reported and labelled. The
matched-axis contrast is **`raw__win32 − raw__topk20` = −7.09 [−9.05, −5.15]\***, so P2's
conclusion is unchanged and in fact slightly stronger once the axes are separated.

### D4 — inconsistent `within1` aggregation (REAL, made two tables incomparable)

`changepoint_readout_eval_v1.py` used a per-cell macro over 8 cells; `step_measure_eval_v1.py`
used a per-subset macro over 4 subsets, next to a `mean_sla` that used 8 cells. Fixed to
8-cell macro in both.

### Defects found and judged immaterial (verified, left alone)

The `r_max` fold-in understates the folded run's variance (argmax agrees 40/40, max |rise|
difference 7.8e-4); `page_cusum_readout(estimator="onset")` can return an out-of-range
index on a length-1 series (roster minimum is 2 steps; 0/295 arms affected); empty
half-steps score a hard 0.0 (2 of 4,442 scored answers); the bootstrap's absent-subset
handling differs from its absent-cell handling (unreachable at these group counts); the
`interval` helper reports the bootstrap mean rather than the plug-in difference (1.096 vs
1.089). Each is recorded here rather than silently fixed, because fixing them would change
frozen numbers by less than their rounding.

### Checks that passed

Fold discipline is clean in all three fitting scripts, including that the per-model AR fit
is restricted to both the training folds and the right model. `subset_of` is correct (the
recorded `cell[3:-3]` bug is not present). Span indexing is correct everywhere. No missing
`dtype=bool`. The CUSUM closed form and its onset match an independent recursion on 3,000
series including ties. The BOCPD brute-force enumeration is itself correct and the filter
matches it to 1e-10; truncation keeps the posterior normalized to 4.4e-16. **The
evidence-drop orientation is correct and not flipped** (Δ₀ is exactly −0.0 at all 13,769
answer starts; risk = −Δ = ΔH). `best_window_steps` matches a direct recomputation to
exactly 0.0. The vectorized bootstrap equals `sla_gate_free` exactly. All 295 grid values
replay from the saved peaks. Anchors hold everywhere.

---

## 2. The finding that reframes the whole line: the step-length prior

**This is the most serious correction and it was not a coding error — it was a missing
control.** The project's own standing note says every ProcessBench step readout inherits a
step-length prior and that no length control had ever been run. I did not run one either.

Two controls are now standing rows in the sweep:

| | gate-free SLA |
|---|---:|
| chance | 16.58 |
| **name the LONGEST step, read no score at all** | **31.78** |
| incumbent raw Top-10 | 35.92 |
| raw Top-20 (grid maximum) | 37.01 |
| every contiguous-window arm | 28.6 – 30.4 |

**The entire window family scores below a readout that ignores the scores.** And
residualising each step score on log(step length) *within the answer* before the argmax:

| K | raw | length-residualised | cost |
|---:|---:|---:|---:|
| 1 | 20.42 | 18.06 | −2.37 |
| 5 | 31.93 | 23.34 | −8.59 |
| 10 | 35.92 | 24.99 | −10.94 |
| 20 | 37.01 | 25.23 | −11.78 |
| 40 | 36.74 | 26.52 | −10.23 |
| 80 | 34.18 | **27.95** | −6.22 |

### What this changes

**Superseded:** *"its width alone moves gate-free SLA from 20.42 at K=1 to 37.01 at K=20 —
a 16.6 pp range … the evidence for an error is diffuse across the tokens of a step."*

**Replacement:** the 16.6 pp range is roughly two-thirds step length. The honest measure of
what the locator extracts from the token scores is its margin over a no-score length
heuristic: **37.01 − 31.78 = 5.2 pp** for the grid maximum, **4.1 pp** for the deployed
arm. (For context, CT7's 39.89 is +8.1 over the same baseline.) The claim that the evidence
is "diffuse across a step's tokens" is *not* established by the width sweep and is
separately contradicted: a contiguous window centred on the step's own peak — which would
capture diffuse evidence if it were there — loses 14.5 pp.

**What survives, and is in fact strengthened:** after the control the sweep *still* rises
with width, and it now rises **monotonically to K=80** (18.06 → 24.99 → 26.52 → 27.95)
where the raw curve turns over at 20–40. So the raw curve's optimum at K=20 is itself a
length artefact, and in the length-free component wider aggregation keeps helping. "More
aggregation, not sharpening" survives as a direction; its size and its stated mechanism do
not.

---

## 3. Claims withdrawn or weakened

### W1 — "the mechanism is real" for change-point detection. WITHDRAWN.

**Superseded** (`CHANGEPOINT_STEP_READOUT_V1_RESULTS.md` §3, HISTORY 424): *"The mechanism
is real, and it appears exactly where the theory says it should … a sequential detector
does add something an argmax cannot see — it needs a sequence to see it in."*

I compared CUSUM against a **single-token argmax**, the weakest possible baseline on a
713-point series, and ran no aggregation control. The audit ran one:

| token-series readout | SLA |
|---|---:|
| argmax (the baseline I used) | 20.39 |
| CUSUM onset, default | 23.70 |
| CUSUM alarm, default | 24.16 |
| **causal boxcar mean, w=8** | **28.08** |
| CUSUM's label-selected ceiling | 28.82 |
| **centred boxcar mean, w=64** | **30.08** |

A plain boxcar with no sequential machinery beats both default CUSUM arms by **+3.9 and
+4.4 pp**, intervals excluding zero, and matches CUSUM's own ceiling. **Replacement:** the
token arm shows that aggregation over a long series beats picking one token. It does not
isolate a sequential mechanism. This is the same conclusion Step 426 reached independently,
and 426 retrospectively explains away 424's headline.

**Multiplicity, not acknowledged anywhere:** 28 within-arm contrasts, 39 intervals against
the anchor, 39 interaction intervals, 295 point settings, no correction. Of the four
headline CUSUM wins, two fail a Bonferroni adjustment across the 28.

### W2 — P2's length interaction. WITHDRAWN.

**Superseded:** *"the qualitative pattern in their Table 3 does reproduce."*

The test has no discriminating power. Against the same anchor: a uniform random step scores
an interaction of **+3.15**, a constant "always the middle step" rule **+4.35**, both at or
above the per-step drop's +3.68; the two largest interactions in the whole table belong to
`bocpd_reset`, a near-chance rule. Any arm uniformly worse than the anchor shows a positive
interaction, because the anchor's edge over chance is concentrated on the short subsets.
The verdict also flips on an undefined word: reading "best" as the highest-SLA drop cell
gives +2.74 [−0.73, +6.25] and P2 fails. **Nothing shows Chen et al.'s Table 3 shape
reproducing.** Both the pre-registration's wording and the report are at fault.

### W3 — "the error step is an isolated impulse". WEAKENED.

**Superseded:** *"+0.65 SD with both neighbours at or below zero … there is nothing for an
accumulating statistic to accumulate."*

"Both neighbours at or below zero" is false in three of four subsets, and zero is the wrong
reference: scores are answer-standardized, so non-error points are *forced* to average
below zero. Against the rest of the same answer the neighbours are **elevated by +0.10 to
+0.29 SD** — a three-step elevated neighbourhood, not an isolated impulse. The
local-maximum rate is also computed only on answers whose first error is interior (74–89%
of them), so step-0 errors are structurally excluded from a statistic reported as being
about "answers".

**Replacement:** the error step carries +0.53 to +0.74 SD, its neighbourhood is elevated
about three steps wide, and it is a strict local maximum in 51.7–61.4% of interior-error
answers against a 33.9–37.1% null. That is a modest mean shift in noise. It is a reason to
*doubt* that accumulation has much to gain, not a demonstration that there is nothing to
accumulate — and the session's own table contains a counterexample it did not flag:
`drop_worst__step@@first_q0.7`, an ordering rule on the 8-point series, beats its own
argmax by **+2.66 [+0.99, +4.30]**, one of the few contrasts surviving Bonferroni.

The impulse *itself* survives its controls: the local-maximum excess is +0.189 under my
control and +0.192 under a position-matched one, and the position-control reversal of the
other line's "more confident after it errs" claim holds with an interval I had not
computed: **+0.132 [+0.104, +0.161]** pooled. Under a stricter position-matched permutation
control the excess halves to +0.060 [+0.031, +0.088] and nearly vanishes on MATH — so the
reversal is real but smaller than the headline number suggested.

### W4 — the whitening conclusion. WEAKENED.

**Superseded:** *"The autocorrelation is therefore not nuisance — it is carrying signal …
whitening also lowers split-half reproducibility … exactly what happens when a transform
destroys signal rather than noise."*

Two problems. The contrast the script itself calls decisive (ceiling against ceiling) is
−1.21 [−2.88, +0.43] and **covers zero**, so the size of the cost is not established; I
headlined the matched-width contrast instead. And the reproducibility argument is circular:
split-half reproducibility over odd/even tokens is mechanically high on an autocorrelated
series and mechanically low after *any* whitener, signal or not — the established run
already carries a `reproducibility_is_monotone_in_K` flag because this was recognised.

**Replacement:** P1 is falsified as a directional prediction. What is established is that
linear causal prewhitening does not improve this readout. That some of the autocorrelation
is signal rather than nuisance is the most natural reading, not a measured result. (One
rival was checked and ruled out: whitening does not lose by shedding the length coupling —
whitened Top-40 agrees with "longest step" 62.7% of the time against raw Top-20's 43.4%,
and still loses.)

### W5 — "raw Top-20, the largest unclaimed number". WITHDRAWN as a candidate.

**Superseded:** *"measured twice on the same population by two independent scripts … the
largest unclaimed number on the locator."*

It is the **label-using maximum of its own grid** — `READOUT_CALIBRATION_C1.json` literally
records `"label_using_best": "K=20"` — and my own pre-registration promised to report such
a number as a ceiling and never as a candidate. I broke that rule in the document that
stated it. The two scripts are not independent (same C1 fusion via the same code path, same
rows, same labels), and re-measuring the same rows cannot narrow a sampling interval
anyway. Under the length control the gap is **+0.24 [−0.89, +1.39]**.

**Replacement:** the grid's label-using maximum is K=20, +1.10 [−0.03, +2.22] over the
deployed K=10 — a ceiling with an interval covering zero. The corrected label-free
criterion picks **K=40**, +0.82 [−0.73, +2.34]. Neither is a demonstrated gain.

### W6 — P4's verdict. WITHDRAWN as pre-registered.

P4 read: *"the token-granularity arm of each change-point rule beats its own
step-granularity arm."* It does not: CUSUM alarm 24.16 vs 30.00, CUSUM onset 23.70 vs
34.66, BOCPD rise 19.59 vs 34.88 — all **below**; only `bocpd_reset` is above, the rule the
report calls catastrophic. Calling P4 "SUPPORTED, and the one real finding here" was not
faithful to what was registered. The within-arm analysis in §3 is a legitimate different
question that was **not** pre-registered, and it is separately withdrawn by W1.

### Smaller factual corrections

- **"roughly 130 settings per arm"** → **59 per arm, 295 in total.**
- **"correlates 0.9996 with `q15_H1`"** → that is the top-20 entropy **before** the EMA and
  the difference; the drop statistic itself is not 0.9996-redundant with an existing channel.
- **"collapses by 0.4–0.6 SD in the last two deciles of every answer, error or not"** → the
  fall is 0.23–0.47 SD in the ninth and tenth deciles, GSM8K rebounds in the last decile,
  and only erroneous answers were profiled; clean answers were never measured.
- **"`energy_level` is free to drop"** → true for the L-SML arm (−0.03 [−0.79, +0.74]),
  **false under equal weighting** (−3.75 [−5.32, −2.21], the largest drop in that table).
- **"two thirds of the margin was a redundancy artefact"** → the shrinkage is −2.10
  [−3.42, −0.83], which I had not computed; but the repair uses the very direction L-SML
  converges on, so it is not an independent repair, and "artefact" overstates a margin that
  stays positive and excludes zero.
- **"nothing in the grid reaches 35.92"** → nothing *exceeds* it; four `cusum_alarm_series`
  settings tie it exactly, because they reduce to argmax.
- **"on the ~8-step series it loses on both"** → it loses in five of six step cells, two
  with intervals covering zero, and `drop_worst__step cusum_onset` is +0.31 [−0.86, +1.52].
- **A pre-registered secondary endpoint was omitted from the report.** On `within1`,
  `cusum_onset` **beats** argmax on the step series (61.14 vs 59.48). P3's "every
  change-point rule is at or below argmax" is true only on SLA.
- **P3 is near-vacuous by construction**: 6,969 of the 10,369 q8 training answers are
  PRMBench, so the "per-model q8" whitener is nearly the pooled one and the contrast rides
  on ~1,700 q4 rows. The gloss "the two backbones' token dynamics are nearly identical"
  overstates what was measured.
- **The ±offset profile averages over different populations at each offset** — only 26 of
  414 GSM8K erroneous answers have full ±3 support. The impulse survives a common-support
  restriction on Omni-MATH, but the GSM8K row of that table mixes populations.

---

## 4. What still stands

- Anchor discipline: 35.92 replayed before every new number, `step_replay_max_abs_diff = 0.0`.
- **P1 of the change-point grid is properly falsified**, and the handoff's advance
  commitment to close that line on this outcome was honoured.
- The position-control reversal of the other line's "more confident after it errs" claim,
  now with an interval: +0.132 [+0.104, +0.161], halving to +0.060 [+0.031, +0.088] under a
  stricter position-matched control.
- Contiguity is decisively worse, and more robustly than the report showed: a window
  centred on the step's own peak loses 14.52 pp [−16.93, −12.09].
- `energy_level` drops free from the L-SML arm.
- Everything is labelled development-only; no confirmation is claimed anywhere.
