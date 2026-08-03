# Where is there room in U-PCR? — the ceiling measurements

24 in-scope test sets. The arm of record throughout: U-PCR with each feature's direction taken
from the sign of its estimated correlation with correctness, and the one global direction from
the test set's anchor (macro 0.7741). The hand-picked 6-feature reference sits at 0.7733.

Scripts: `scripts/upcr_study/exp10_channel_ceilings.py` (the pre-registered ceilings),
`scripts/upcr_study/exp11_posthoc_controls.py` (the controls added afterwards),
`scripts/action_items_jul2026/sign_identifiability.py` (the direction diagnostic).
Data: `results/upcr_study/{10_channel_ceilings,11_posthoc_controls}/`,
`results/action_items_jul2026/_data/`.

---

## The question

U-PCR takes ~30 candidate features per test set and returns one fused score. On the way it
decides **which features to throw away**, **how to weight the survivors** (a blend of the top
two principal directions of the feature covariance, with one free number in it), **one scalar**
for the shared signal strength, and **which direction each feature reads in**.

For each of those we asked: *if we used the labels to do this step perfectly, how much better
would the detector get?* That is a ceiling — nothing built later can beat it. A step that
prices near zero can be closed with a number instead of a failed attempt.

## The answer

| what could be improved | best possible gain | 95% interval |
|---|---:|---|
| **which features get kept** | **+1.48pp** | [+0.97, +2.03] |
| how the two directions blend | +0.19pp | [−0.08, +0.51] |
| the three hand-tuned constants | +0.19pp | not measured out-of-sample |
| the label-variance guess | ~0 | three separate ways |
| every feature's direction correct | −0.06pp | [−0.29, +0.08] |

**Feature selection is the only step with room in it. Everything else in U-PCR is already
performing as well as it can.**

Two facts about the feature selection that matter more than the headline:

- The good feature sets are **half the size** of what we keep now — about **10** features
  instead of **21**, and smaller on **every one of the 24 test sets**.
- But you cannot get there by keeping fewer of the *same* ranking. The good features overlap
  U-PCR's own ranking at **0.340**, against a random baseline of **0.360** — **at chance.**
  Cutting deeper by that ranking loses at every size: −1.49pp at 6 features, −2.11pp at 8,
  −1.60pp at 10, −0.28pp at 16.

---

## What was measured, one by one

### Are the thrown-away features worthless?

Scored every discarded feature on its own, letting the labels choose its direction so it reads
as favourably as possible. **181 of 682** features get dropped; they average **0.584** AUROC
against **0.708** for the survivors, and across all 24 test sets exactly **one** dropped feature
beat the best surviving feature in its own test set.

So the current rule discards weak features, not good ones. The original motivation for a
cluster-aware version of it — "it keeps good features out" — is not supported.

### How much better could the feature selection be?

From the current keep-set, try adding each excluded feature and removing each kept one; take
the best single change; repeat until nothing helps. Selection on half the rows, scoring on the
other half, five splits, halves standardised independently and feature directions re-derived on
the selection half only. The search converged on all 24 test sets.

| | macro | vs current |
|---|---:|---:|
| U-PCR as deployed | 0.7741 | — |
| same-rows search (optimistic) | 0.8052 | +3.09pp |
| **held-out search** | **0.7852** | **+1.48pp**, 21W/3L |
| random subsets of the same size | 0.7555 | −1.55pp, 3W/21L |
| shallow label-guided search (best of 20) | — | +0.69pp, 18W/6L |

Roughly half the optimistic gain is the search fitting its own noise. Random subsets of the
*same size* lose 1.55pp, so the gain is about **which** features, not **how many**. A shallow
search recovers about half of it, so the rest needs depth.

**And it does not transfer.** A keep-set built from the other 23 test sets' choices scores
−0.81pp at matched size and −2.37pp at ten features. The gain is per-test-set.

### How much better could the weighting be?

The blend of the top two principal directions has one degree of freedom — an angle. Swept 721
values covering every possible blend. In-sample the best angle is worth +0.49pp; **chosen on
half the rows and scored on the other half it is +0.19pp, 11W/13L, p = 0.57.** Nothing.

This is the ceiling for every idea that improves how U-PCR estimates its weights, **including
clustering the surviving features** — the placement originally proposed.

### Do the three hand-tuned constants matter?

All 125 combinations on all 24 test sets. The absolute drop threshold **never binds** at its
deployed value (0.0, 0.01 and 0.05 give identical results). The relative threshold does bind;
loosening it helps slightly. The label-variance guess has an interior optimum exactly at the
hand-picked 0.25. Best single setting for all test sets: **+0.19pp**, and picking the best of
125 on the same 24 test sets is itself a small oracle.

### Is the label-variance guess wrong?

The shared-signal scalar is searched in a box ending at 0.25. Correctness is 0/1, so its
variance is at most 0.25 and usually much less. Substituting the true value gives −0.23pp;
substituting the value the model's own assumptions imply gives −0.10pp (p = 0.83); deriving it
from the model's internal consistency condition gives −0.06pp (p = 0.93). **The axis is inert.**

A caution for anyone revisiting this: only the *ratio* of that constant to the average feature
variance is identifiable, and because features are z-scored, matching moments to a binary label
is exactly a ratio of 1.0 — not p(1−p). Substituting p(1−p) is a one-sided nudge bounded below
the deployed setting by construction, so a negative result there is guaranteed and means nothing.

### Does U-PCR's own model-selection criterion rank feature sets?

If it did, a differentiable gate could descend it. Correlating the criterion against actual
performance across the 125 feature sets, the sign flips depending on what you control for
(−0.13 controlling feature count, +0.15 controlling the variance guess), every magnitude is
under 0.16, and 14 of 24 test sets point the "right" way. **No reliable ordering.** There is
nothing for a gate to descend.

### What if every feature's direction were correct?

**This is the measurement the plan never asked for, and it is the one that mattered most.**

Orienting all ~30 features by the direction the labels say is right: **0.7741 → 0.7735,
−0.06pp, 17 of 24 test sets change by exactly 0.0000, p = 1.00.**

Perfect direction recovery is worth nothing. Every proposed sign fix was competing for ≤ 0.00pp.

The diagnostic underneath it is still interesting and still true — direction errors concentrate
on features where correctness first rises then falls, so a straight-line correlation is near
zero (65.8% correct on those, 93.4% elsewhere). But those features are also the weakest ones
(mean best-possible AUROC 0.5417 vs 0.6022), and 12 of the 13 wrongly-signed ones sit at 0.557
or below — at chance, where the "correct" direction is itself a coin flip and getting it wrong
costs nothing. That is why the diagnosis is real and the channel is still empty.

---

## Honest labelling of what came from where

**Pre-registered:** the feature-selection ceiling, the weighting ceiling, the constant sweep,
the label-variance test, the criterion-ranking test, the direction diagnostic.

**Added after seeing the data** (`exp11_posthoc_controls.py`, all reproducible from the repo):
the oracle direction ceiling, the held-out weighting ceiling, the random-subset floor, the
shallow-search comparison, the between-test-set transfer test, and the overlap with U-PCR's own
ranking. These have a different evidential status and are marked as such wherever quoted. The
direction ceiling in particular was run *after* the direction section had been written, and it
overturned it.

**Which instrument resolved the "is there room in feature selection" question.** The
pre-registered search was the same-rows version, at +3.09pp. The held-out version at +1.48pp is
the number worth building against, and its interval is [+0.97, +2.03] — a real effect, but not
a precise one. Anything that recovers half of it is inside the noise of the ceiling itself.

**Reviews run:** a code-correctness pass (16 findings, all fixed — including a search that was
stopping early on 17 of 24 test sets), an adversarial results pass (11 findings, including the
missing direction ceiling), and a pre-registration compliance pass. Corrections that came out
of them are folded into the text above rather than appended.

---

## What this means

Three of the four places a clustering stage could have gone are worth nothing: clustering the
survivors to remix the weights (+0.19pp, p = 0.57), clustering to compute direction confidence
(−0.06pp, p = 1.00), and routing through U-PCR's own criterion (no reliable ordering). The
fourth — a cluster-aware keep rule — sits in front of real room, but the version specified for
it ranks features by the estimated correlation, and that ranking is at chance for this target.

**The clustering line closes on ceilings.** The live question this phase actually opened is a
**feature-selection** question: what *does* separate the good features, if not their estimated
correlation with correctness?
