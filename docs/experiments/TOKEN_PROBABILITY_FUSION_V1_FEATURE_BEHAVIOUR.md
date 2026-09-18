# Feature behaviour: a cell we do well on versus one we do badly on

Runs `scripts/diagnostics/feature_behaviour_good_vs_bad_cell_v1.py`; output
`results/token_probability_fusion_v1/FEATURE_BEHAVIOUR.json` and two figures.

**Label-using descriptive diagnostic.** It looks at the true first-error step. Nothing
here selects a feature, a sign, or a threshold; no arm is proposed.

GSM8K (we score 47.8–49.8) against Omni-MATH and OlympiadBench (30.4–31.0), split by
erroneous versus error-free answers, for all twelve channels.

![Per-channel trajectories](../../results/token_probability_fusion_v1/figures/feature_trajectories.svg)

![The mechanism](../../results/token_probability_fusion_v1/figures/feature_mechanism.svg)

---

## The headline: the channels do not weaken on long chains

Lift at the true error step, in units of the answer's own standard deviation — i.e. how
far above that answer's typical step the channel sits at the real error:

| channel | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---|---|---|---|
| q15_VE1 | 0.725 | 0.555 | **0.718** | 0.652 |
| q15_H1 | 0.668 | 0.522 | **0.691** | 0.628 |
| energy_level | 0.616 | 0.361 | 0.561 | 0.452 |
| logprob_margin | 0.540 | 0.493 | **0.550** | 0.540 |
| true_tail50 | 0.525 | 0.362 | **0.629** | 0.484 |
| bocpd_p0 | 0.464 | 0.366 | 0.370 | 0.413 |
| chosen_surprisal | 0.429 | 0.429 | 0.375 | 0.413 |
| sw_var16 | 0.403 | 0.407 | **0.486** | 0.434 |
| top15_turnover | 0.365 | 0.220 | 0.207 | 0.198 |
| dominant_freq16 | 0.280 | 0.195 | 0.237 | 0.212 |
| top50_js | 0.271 | 0.255 | 0.193 | 0.271 |
| energy_innovation | 0.217 | 0.219 | 0.015 | 0.122 |

**The two best channels mark the error just as strongly on OlympiadBench as on GSM8K**
(0.718 vs 0.725; 0.691 vs 0.668), and five channels have their *highest* lift on
OlympiadBench. The subset where the channels are genuinely weakest is **MATH**, not the
long ones. The evidence per step does not degrade with chain length.

## So why do we score 30 there and 48 on GSM8K?

Because argmax has to beat more competitors, and the strongest competitor grows.

| subset | steps | value at the true error | strongest wrong step | margin |
|---|---|---|---|---|
| GSM8K | 5.3 | 0.464 | 0.578 | −0.115 |
| MATH | 6.8 | 0.362 | 0.624 | −0.263 |
| OlympiadBench | 8.9 | 0.413 | **0.727** | −0.314 |
| Omni-MATH | 8.6 | 0.399 | 0.680 | −0.282 |

The true error step stays flat at 0.36–0.46 SD across all four. The strongest *wrong*
step climbs from 0.578 to 0.727 — a plain order-statistic effect, the maximum of more
draws being larger. The gap the locator must close nearly triples.

The cleanest way to see that this is difficulty and not a broken detector is the ratio
to chance:

| subset | ours | chance | **ours / chance** | Chen et al. | **theirs / chance** |
|---|---|---|---|---|---|
| GSM8K | 48.80 | 20.84 | **2.34** | 44.77 | 2.15 |
| MATH | 33.42 | 18.10 | **1.85** | 32.47 | 1.79 |
| OlympiadBench | 30.79 | 13.55 | **2.27** | 42.29 | **3.12** |
| Omni-MATH | 30.70 | 13.84 | **2.22** | 37.54 | **2.71** |

**Our ratio is flat** at 2.2–2.3 (MATH aside). Relative to how hard each subset is, the
detector is equally good everywhere — the absolute number falls only because there are
more steps to choose among.

**Their ratio rises** on the long subsets, from 2.15 to 3.12. *That* is the asymmetry
worth explaining: not that we get worse, but that they get better. Caveat that their
SLA takes the first step crossing a threshold rather than an argmax, and runs at K=20,
so part of the difference may be the readout rather than the signal.

## What genuinely does differ: three channels go flat

Percentage of steps tied at the answer's own minimum value:

| channel | GSM8K | Omni-MATH |
|---|---|---|
| top15_turnover | 36.0% | **53.6%** |
| dominant_freq16 | 34.6% | **40.5%** |
| top50_js | 24.8% | 29.9% |
| *(every other channel)* | ~20.8% | ~13.8% |

The ~20.8% / ~13.8% baseline is just `1/steps` — the single minimum step. These three are
far above it and **get worse as chains grow**: on a long answer `top15_turnover` is
constant over more than half the steps, so it cannot discriminate among them at all.
That is a genuine length-dependent degradation, and it is the one thing in this analysis
that is specific to the long subsets.

## Two readings I tested and had to discard

**"Three channels are inverted."** Their mean percentile rank for the true error step is
far *below* chance — `top15_turnover` 0.127 on Omni-MATH, `dominant_freq16` 0.166. That
looks like a sign error. It is not: flipping them collapses their standalone SLA (17.12
→ 8.45, 17.40 → 8.49), and flipping all three in the equal fusion costs 1.21 pp. The low
ranks are an artefact of the ties above — with half the steps tied at the minimum, a
strictly-less rank is deflated. The declared orientations are right.

**"The channels mostly encode position."** The trajectories all show a strong
start-to-end trend, so I checked how much variance relative position alone explains:
0.05–0.21 for every channel except `chosen_surprisal` (0.49–0.60, which is its step-0
spike — exactly what CT7's de-spiking removes). Removing the position profile leaves the
error lift almost intact (q15_H1 0.668 → 0.602). The signal is error-driven, not
positional.

## What this changes

The long-chain deficit is **not** a feature problem. The per-step evidence is as strong
there as anywhere. It is a **selection** problem: one argmax over 8.9 candidates with a
margin of −0.31 SD.

That reframes what is worth trying. Anything that improves the per-step evidence is
attacking a part that is not broken — which is consistent with both length attacks
failing, with `sw_var` failing, and with every narrower fitting scope failing. What has
never been touched is the **decision rule**: we commit to a single argmax. The published
comparator does not — it thresholds and aggregates, and its advantage over chance
*grows* exactly where ours is flat.

The two concrete follow-ups this suggests, neither yet run:

1. **Stop using a bare argmax.** Their ratio-to-chance rising with length while ours is
   flat points at the readout, not the bank. A rule that accumulates evidence over the
   chain rather than picking one maximum is the untested axis.
2. **The three flat channels.** `top15_turnover` is constant across half a long answer.
   Either it needs a formulation that stays informative at that scale, or it is dead
   weight on exactly the subsets we care about.
