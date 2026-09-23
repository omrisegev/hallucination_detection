# The adaptive step readout — the mechanism works, the accuracy does not follow

Module `spectral_utils/adaptive_step_readout_v1.py`, evaluation
`scripts/diagnostics/adaptive_readout_eval_v1.py`, output `ADAPTIVE_READOUT.json`.
C1 architecture throughout (token-level L-SML, fitted once, then the readout).
Development-only.

## What it does

Instead of one fixed `K` for everything, each answer picks its own `K*` from its own
tokens, with no labels, by a variance decomposition:

    total(K)  = variance of the step scores across this answer's steps
    noise(K)  = sampling variance of the statistic, from repeated random half-splits
    signal(K) = max(total − noise, 0)
    K*        = argmax  signal(K) / noise(K)

Small `K` inflates the denominator; large `K` collapses the numerator as every step
converges on its own mean. The interior optimum is a property of the decomposition, not
of where a grid stops.

Two design points worth stating. Per-answer adaptation is **admissible here specifically
because the locator argmaxes inside the answer** — the readout never has to be comparable
between answers, so a different K per answer costs nothing. (A gate that ranks answers
against each other would not have that freedom.) And the whole K ladder comes from one
sort per block: sort descending, take the running mean, read off every K at once —
verified exact against a direct top-K mean to 8.9e-16.

## The mechanism check — this is the part that worked

The question that decides whether this is intelligent or just another parameter: does
`K*`, chosen with no labels, come out larger on the long subsets?

| subset | tokens/step | **mean K\*** | median K\* |
|---|---|---|---|
| GSM8K | 56.8 | **28.03** | 23 |
| MATH | 81.9 | **38.97** | 38 |
| OlympiadBench | 96.8 | **42.97** | 42 |
| Omni-MATH | 99.3 | **45.08** | 47 |

**`K*` rises monotonically with step length across all four subsets, discovered from the
data with no labels and no subset identity.** The algorithm recovers the length
dependence that the fixed grid had to be told about. That is a real result about the
mechanism.

## The accuracy — it does not follow

| arm | mean | GSM8K | MATH | Olympiad | Omni-MATH | vs K=10 |
|---|---|---|---|---|---|---|
| K=10 (today) | 35.92 | 48.79 | 33.42 | 30.79 | 30.70 | reference |
| K=20 | 37.01 | 49.52 | 33.50 | 32.75 | 32.28 | +1.09 [−0.01, +2.23] |
| K=40 | 36.86 | 47.83 | 33.08 | **34.04** | 32.48 | +0.93 [−0.56, +2.43] |
| **adaptive** | 36.84 | 48.79 | 33.25 | 32.38 | **32.94** | **+0.92 [−0.40, +2.21]** |
| *oracle K (label-using)* | *58.13* | *68.60* | *55.98* | *53.78* | *54.15* | *+22.21 [+20.63, +23.82]* |

| contrast | difference | 95% CI |
|---|---|---|
| adaptive − K=20 | −0.17 pp | [−1.15, +0.75] |
| adaptive − K=40 | −0.01 pp | [−1.08, +1.07] |

**The adaptive readout is indistinguishable from a good fixed K, and its gain over
today's K=10 does not exclude zero.** Picking K per answer, correctly tracking chain
length, buys nothing measurable over picking one K globally.

## The ceiling, and what it says

The oracle row is **label-using and a very loose ceiling**: it asks whether *any* K in
the ladder puts the argmax on the true error step. At 58.13 against 35.92 it says
per-answer K selection has around **+22 pp** of headroom in principle — and the label-free
criterion captures about **4%** of it.

A caveat on the oracle's own K statistic, which I will not report as meaningful: the
implementation takes the *first* K in the ladder that hits, so its mean "oracle K" of
4.3–4.9 is an artefact of that convention, not evidence that small K is right.

## Reading

The criterion is measuring something real — it finds the length dependence unaided — but
what it optimises is not what the locator needs. It maximises the discriminability of the
step ranking against sampling noise; the locator needs the *true* step to win, and the
gap between those two is where the missing 21 pp of the ceiling sits.

Consistent with that, the synthetic check built into the module's development showed the
criterion choosing a median K of 47 when the planted evidence occupied 12 tokens — it
systematically prefers a wider readout than the concentration of the signal, because with
only 5–9 steps the between-step variance estimate is itself noisy and wider K stabilises
it. The criterion is partly optimising its own estimator rather than the task.

**What this closes:** per-answer K selection by a stability/SNR criterion. Both the
global version (previous document, corrected) and this per-answer version leave the gain
inside the noise.

**What it leaves open:** the ceiling is large and real. Something about *which* K works
for a given answer is learnable in principle; a criterion built on within-answer variance
is not the thing that learns it.
