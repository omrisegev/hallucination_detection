# The readout width: the pattern is real, exploiting it is not — and what that leaves

Runs `readout_shape_and_neff_v1.py` and `per_cell_k_honesty_v1.py`; outputs
`READOUT_SHAPE_AND_NEFF.json` and `PER_CELL_K_HONESTY.json`. Development-only.

Three questions from Omri, answered separately because they have different answers.

## 1. Is there a maximum between K=10 and K=40? Yes — I never sampled it

The earlier grid ran 1, 3, 5, 10, 20, 40, 80, 160: between 10 and 40 only K=20 was ever
measured. Sampling every integer K is free (one descending sort per step yields the
running mean, hence the whole ladder), and the per-cell optima are:

| cell | SLA at K=10 | argmax K | SLA there |
|---|---|---|---|
| GSM8K 4B | 47.8 | **15** | 51.7 |
| GSM8K 8B | 49.8 | **12** | 50.7 |
| MATH 4B | 34.8 | 9 | 34.8 |
| MATH 8B | 32.0 | **16** | 34.5 |
| OlympiadBench 4B | 30.6 | **30** | 33.3 |
| OlympiadBench 8B | 31.0 | **37** | 35.1 |
| Omni-MATH 4B | 31.0 | **28** | 34.3 |
| Omni-MATH 8B | 30.4 | **56** | 32.7 |

## 2. Is the pattern real, and does a label-free statistic derive it?

**The pattern is real.** I expected the per-cell argmax to be an artefact of maximising a
noisy curve over 64 correlated values on 200–760 answers. It is not: across 400
source-group half splits the selected K is stable — GSM8K-4B 15 [15,17], MATH-8B 18
[16,18], Omni-MATH-4B 28 [27,29], OlympiadBench-8B 35 [32,37]. The optimum location is
reproducible and rises with chain length.

**The derivation I proposed does not work.** The hypothesis was that K should scale with
the *effective* number of independent tokens, `n_eff = n / (1 + 2Σρ)` — the quantity
sitting between a fixed count (which partly worked) and a fixed fraction (which failed
badly). Its rank correlation with the optimum is +0.857, which looks convincing, and it
is misleading:

| cell | tokens/step | n_eff | n_eff/n | best K | **K / n_eff** |
|---|---|---|---|---|---|
| GSM8K 4B | 55.0 | 22.1 | 0.460 | 15 | 0.68 |
| MATH 4B | 80.6 | 23.9 | 0.358 | 9 | 0.38 |
| OlympiadBench 8B | 88.9 | 27.1 | 0.361 | 37 | 1.36 |
| Omni-MATH 8B | 93.5 | 28.7 | 0.360 | 56 | 1.95 |

`n_eff` is nearly **constant** across cells (22.1 → 28.7, +30%) while the optimum K
triples. Under the hypothesis K should have been roughly constant. `K/n_eff` spans
0.38–1.95, a five-fold range, so there is no proportional rule. **n_eff explains the
ordering and not the magnitude, which means it is not the mechanism.**

One real by-product: `n_eff/n` falls from 0.46–0.51 on GSM8K to 0.35–0.36 on the long
cells. Longer steps are relatively *more* autocorrelated, so each extra token there
carries less new information.

## 3. Does any of it pay? No

400 source-group half splits: K chosen on one half, scored on the other.

| arm | out-of-sample SLA |
|---|---|
| K=10 (today) | 35.92 |
| global fixed K, chosen | 36.82 |
| per-cell K, chosen | 37.08 |
| *oracle per-cell (label-using)* | *38.97* |

| contrast | difference | 95% CI |
|---|---|---|
| **per-cell tuned − global fixed** | **+0.26 pp** | **[−1.18, +1.29] includes zero** |
| global fixed − K=10 | +0.90 pp | [−0.47, +2.12] includes zero |
| per-cell tuned − K=10 | +1.16 pp | [−0.41, +2.27] includes zero |
| oracle − per-cell tuned | +1.89 pp | [+0.94, +3.24] excludes zero |

**Per-cell K buys nothing over one global K.** The reason is visible in the curve: it is
broad around its optimum, so a stably-identified-but-flat maximum is worth little. The
location is real; the height is not.

The oracle row is an inflated ceiling, not available headroom: it picks the argmax **on
the evaluation half itself**, so it is guaranteed to lead there by construction and
carries the same maximisation bias this whole section is about.

## What this closes, and what it leaves

**Closed — the readout, thoroughly.** Global width (~1 pp, no interval excluding zero,
flat optimum); per-answer selection (0.70 pp ceiling on the binary case, every spectral
and L-SML descriptor at chance); per-cell selection (+0.26 pp, includes zero).

**What is left is not the aggregation.** From the feature-behaviour analysis, the mean
margin between the true error step and the strongest wrong step is **negative** — −0.11
SD on GSM8K to −0.31 on OlympiadBench. On average the best distractor *beats* the true
error step. No choice of K inside a step repairs that.

And the unexplained asymmetry points the same way: Chen et al.'s ratio to chance **rises**
with chain length (2.15 → 3.12) while ours is flat (2.34 → 2.27). Their readout is a
first threshold crossing over a smoothed derivative; ours commits to a single argmax.
**The decision rule is the one axis never touched**, and it is the only one that would
explain their length behaviour.
