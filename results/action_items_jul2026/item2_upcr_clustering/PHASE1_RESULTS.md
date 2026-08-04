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
  U-PCR's own ranking at **0.340**, against a random baseline of **0.360**. Cutting deeper by
  that ranking loses at every size: −1.49pp at 6 features, −2.11pp at 8, −1.60pp at 10,
  −0.28pp at 16.

  > **Corrected at Step 221 — this is worse than "at chance".** That 0.360 baseline draws
  > uniformly from the whole pool, but 98.3% of the top-k by `rho_hat` sits inside U-PCR's own
  > keep set, which is only 73.5% of the pool — so the baseline is too easy for it. Against a
  > null matched on that composition, U-PCR's ranking lands **below** chance: −0.05, 5W/19L,
  > p = 0.016. It does not merely fail to find the good features, it systematically avoids
  > them. See `results/upcr_study/13_incumbent_anchored_ranking/summary.json`.

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

Every delta below is a **paired mean over the 24 test sets, against the deployed pool measured
on the same rows and the same splits as the arm it is compared to.** The two scripts drew
different splits, so they have different half-B baselines; the deltas are correct within each
script and the macros are **not** subtractable across rows. (Corrected at Step 221 — this table
previously printed one macro column headed by the full-data 0.7741, which no delta uses.)

| | macro | its own baseline | delta |
|---|---:|---:|---:|
| U-PCR as deployed, all rows (the registered macro) | 0.7741 | — | — |
| same-rows search (optimistic) | 0.8052 | in-sample deployed | +3.09pp |
| **held-out search** | **0.7852** | 0.7704 (half B) | **+1.48pp**, 21W/3L |
| random subsets of the same size | 0.7598 | 0.7753 (half B) | −1.55pp, 3W/21L |
| shallow label-guided search (best of 20) | 0.7823 | 0.7753 (half B) | +0.69pp, 18W/6L |

Roughly half the optimistic gain is the search fitting its own noise. Random subsets of the
*same size* lose 1.55pp, so the gain is about **which** features, not **how many**. A shallow
search recovers about half of it, so the rest needs depth.

> **The floor moved at Step 221.** Those random subsets were built from nothing, while the
> search *starts from* the deployed keep set and trims it. Against a floor that also starts
> from the keep set and trims at random, the floor is **−0.84pp**, not −1.55pp — 0.69pp of it
> was the rebuild, not the feature choice — and the search is worth **+2.25pp, CI
> [+1.53, +3.04], 23W/1L** against it. **Score any new selector against −0.84pp / +2.25pp**,
> not against the numbers in the table above. The shallow-search +0.69pp is also due a re-read
> against the matched floor.

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

---

## Step 221 — the answer to that question

**What separates the good features is not their correlation with correctness. But the
correlation does know which they are.** Those two facts are both solid and they sit in
different places, which is the whole result.

Method: `scripts/upcr_study/exp12_what_separates_good_features.py` computes each feature's
*true* correlation with correctness on the selection half — labels used deliberately, so it is
a ceiling on any estimator, not a method — takes the good set's own size, and scores held out.
`scripts/upcr_study/exp13_incumbent_anchored_ranking.py` re-runs it on the identical splits
after the review found the comparison confounded.

### It identifies the good features

| overlap of the ranking's keep-set with the good set | excess over null | W/L | p |
|---|---:|---:|---:|
| true correlation, vs a uniform-over-pool null | +0.15 | 22/2 | <1e-4 |
| **true correlation, vs a null matched on keep-set composition** | **+0.11** | **20/4** | **<1e-4** |
| estimated correlation, vs a uniform null | +0.00 | 12/12 | 0.83 |
| **estimated correlation, vs the matched null** | **−0.05** | **5/19** | **0.016** |

The matched null exists because both rankings live almost entirely inside U-PCR's keep set
(94.5% and 98.3%) while the keep set is 73.5% of the pool, so a uniform null is too easy. The
true correlation survives the correction; `rho_hat` goes from at-chance to below chance.

### And none of it converts into performance

| held out on half B, vs the deployed pool | delta | CI | W/L | p |
|---|---:|---|---:|---:|
| the good set (the ceiling) | +1.41pp | [+0.80, +2.07] | 22/2 | 4e-5 |
| true correlation, rebuilt from scratch at size k | −0.66pp | [−1.57, +0.12] | 11/13 | 0.34 |
| true correlation, **pruning** the deployed keep set | −0.77pp | [−1.68, +0.01] | 9/15 | 0.18 |
| random pruning, same size (**the matched floor**) | −0.84pp | [−1.06, −0.63] | 1/23 | <1e-4 |
| **true correlation pruning vs that floor** | **+0.08pp** | [−0.78, +0.87] | 11/13 | **0.62** |

The pruning arm exists because the good set is a search that *starts from* the deployed keep
set and trims it, while the original ranking arm rebuilt from nothing — so a loss could have
meant "wrong quantity" or just "discarded a good starting point". It did not: pruning is
0.10pp *worse* than rebuilding (p = 0.13). What the fix removed was 0.69pp of flattery from
the floor.

Size sweep, ruling out a size artefact: −1.56pp at 6 features through −0.31pp at 16, every
size negative, best at the largest tested. The trend toward zero is the arm converging on the
do-nothing baseline (the deployed keep set averages 20.9 features), not the ranking improving.

### The number that closes Bracha's second proposal

**A perfect estimate of the correlation, spent on selection, is worth +0.34pp, CI
[−0.47, +1.30], p = 0.88.** With the weighting blend at +0.19pp (p = 0.57) and polarity at
−0.06pp (p = 1.00), **every channel a better `rho_hat` feeds is now priced, and all three are
worth nothing.** Improving U-PCR's estimation cannot pay through any of them.

### Corrected from the sections above

- The claim that the good features are made of individually **weaker** views is withdrawn. It
  compared the good set's mean |correlation| against the *top-k* by that same statistic, which
  is by construction the maximum over all size-k subsets — true 24/24 by arithmetic. The
  informative comparison is against the whole-pool mean, which is exactly the random-subset
  expectation: good set **0.2932** vs pool **0.2563**, higher on 22/24, p = 4e-6. The good
  features are **above** average in marginal strength, about a third of the way from random to
  the maximum. They are simply not the strongest.
- The floor and the room: **−0.84pp / +2.25pp**, not −1.55pp / +1.48pp. See the note under the
  feature-selection table.

### Gates and checks

Both anchor gates passed on both scripts (GOOD_6 0.7733, U-PCR + sign(rho) 0.7741). Ceiling
reproduction gate: +1.41pp inside Step 220's [+0.97, +2.03]. The deciding test re-runs
byte-identical, and zero U-PCR fits failed across the whole run, so nothing was dropped from
any floor. Dropping the six splits at k ≤ 4 moves the ceiling to +1.38pp; dropping the three
test sets containing them, +1.28pp [+0.65, +1.98].

**Reviews run:** a code-correctness pass (12 findings; the protocol itself passed line-for-line
against Step 220's, no leakage) and a results-vs-pre-registration pass (13 findings; every
summary number reproduced from the CSVs). The confound, the too-easy null and the tautological
diagnostic all came from them and are corrected above rather than appended.

**Open, not blocking:** the shallow-search +0.69pp is still measured against the rebuilt floor
and is due a re-read against the matched one; neither new script carries a permutation null
(Step 220's shows the ceiling clears its null by +7.94pp, 23W/1L).

---

## Step 222 — the whole label-free menu, priced in the same channel

Step 221 asked what separates the good features and answered with an oracle statistic. Step 222
asks whether any **label-free** statistic can be used as a ranker in that channel. Eight arms,
pre-registered with their directions in the module docstring of
`scripts/upcr_study/exp14_ranker_menu.py` before any scoring, on exp12's own splits. Each is
scored twice: held-out AUROC of the pruned keep set against the matched pruning floor (primary,
Holm–Bonferroni over the six label-free arms), and overlap with the good set against a
composition-matched null (secondary).

The room and the floor re-derive exactly on these rows — **+2.25pp, CI [+1.53, +3.04], 23W/1L**,
floor **−0.84pp** against the deployed pool — so the bar is unchanged.

### None of them clears the floor

| arm | statistic (all computed on half A) | vs the matched floor | Holm p | overlap vs null |
|---|---|---|---|---|
| cluster round-robin *(set-level)* | one feature per L-SML group in rotation, within-group by DUFS gate | +0.23pp [−0.09, +0.55] | 0.53 | −0.00, p=0.92 |
| additive pair-fit residual | mean \|C_ij − (ρ_i + ρ_j)\| from U-PCR's own Eq. 15 solve | −0.09pp [−0.73, +0.56] | 0.66 | −0.02, p=0.08 |
| **DUFS gate value** | seed-averaged stochastic gate, Eq. 7 (parameter-free) | −0.70pp [−1.45, −0.03] | 0.36 | −0.01, p=0.32 |
| principal-direction leverage | eigenvalue-weighted loading on the top 2 eigenvectors | −0.92pp [−1.78, −0.19] | 0.36 | −0.03, p=0.08 |
| L-SML cluster size | size of the dependent group it lands in | **−1.61pp** [−2.60, −0.71] | **0.008** | +0.02, p=0.49 |
| redundancy to the pool | mean \|correlation\| to every other feature | **−3.13pp** [−4.80, −1.62] | **0.002** | **+0.04** [+0.001, +0.071] |
| *estimated correlation* — control, known below chance | U-PCR's `rho_hat_full` | −0.26pp | — | **−0.05**, p=0.023 |
| *true correlation* — control, known at the floor | uses labels | +0.08pp | — | **+0.09**, p=0.0002 |

### The finding is the redundancy arm, not the losing ones

Redundancy to the rest of the pool is the label-free statistic that **most identifies** the good
features and the **worst performer** of the eight. Identification without conversion — Step 221's
two-sided result, reproduced with a label-free statistic instead of an oracle one. Since the true
correlation already puts the ceiling of the marginal family on the floor, and the label-free
members not only fail to reach it but two of them fall significantly below random pruning, the
family closes: **the +2.25pp is not reachable by scoring features one at a time.**

### The set-level arm did not escape the shape

Cluster round-robin is the only arm whose score for a feature depends on the other features, and
the only one on the positive side of the floor. It is also the arm **closest to the null** on the
overlap test (−0.00, p=0.92), and across the six label-free arms |overlap excess| against
performance has Spearman −0.71: the nearer an arm is to random, the better it scores against a
floor that *is* random. Exploratory, outside the multiplicity family — at the other two L-SML
loading scales it is +0.04pp (`eigen`) and −0.02pp (`complete`), so it does not survive its own
scale choice. The floor-crossing arms do (cluster size −1.48 to −2.12pp, significant at all three).

### Caveats that belong with these numbers

- **The DUFS number is a three-cell effect.** 9 of 24 cells are positive; 80% of the −0.70pp comes
  from `internalstates_gsm8k_qwen25_7b`, `ars_gsm8k_r1distill8b`, `se_squad_v2_llama8b`. The two
  arms with consistent sign across the grid are exactly the two significant after Holm
  (redundancy 19 of 24 negative; cluster size 20 of 24).
- **DUFS is used outside its operating range on 16 of 120 splits**, where it opens fewer gates
  than the target size k, so the top-k must admit rejected (negative-µ) gates ranked by how
  strongly they were rejected.
- **Cluster size is largely a coin flip.** It takes 4.75 distinct values on average over a pool of
  28.4, so with ~21 candidates and ~11.75 kept the cut falls inside a tied block of 4–6 ordered by
  the random tie-break. Its number prices "a coarse partition plus a tie-break", not a ranking.
  The round-robin shares that stream.
- **The room's denominator is not construction-matched.** The good set lives 81.3% inside the
  deployed keep set while every pruning arm is confined to it (99.85%), so about a fifth of the
  target is unreachable by any arm here. "Recovers X% of the room" should carry that.
- **Under pruning the conditional null collapses to one composition** on all 120 splits, so the
  secondary is a common-null comparison across arms rather than eight separately matched ones.
- **The controls were re-measured, not copied.** Step 221 reported overlap for the *rebuilt* top-k
  against a rebuilt-composition null (+0.11 / −0.05); Step 222 reports it for the *pruned* set
  against a pruned-composition null (+0.09 / −0.05). Both directions and both significances hold.
  The two figures are two measurements of related quantities, not a discrepancy.

### The negative result is measured, not forced by the design

The arms remain genuinely different objects: mean pairwise Jaccard among the six label-free
selections is 0.36 on the 95 splits with ≥5 features to drop (0.45 overall), only 1 of 120 splits
has all six choosing the same set, and the median split has 9 features to drop from a keep set of
~21. Sixteen splits are near-degenerate (≤2 to drop, Jaccard 0.84 there) and two have k > keep set.

### Gates

Four, all passed before any number was read: DUFS gate extraction **exact on 24/24** cells against
the published selector bench; GOOD_6 = 0.7733 and U-PCR + sign(ρ) = 0.7741 checked **by value**;
every Step 221 arm reproduced **per split to exactly 0.0** on all 120 splits (deployed, greedy,
floor, both control arms, both control overlaps, k, m, keep-set size); and DUFS's gates verified
**invariant to per-column sign flips at exactly 0.000e+00**, which is what licenses feeding it the
derived-polarity matrix rather than the hand-oriented one the bench used. No floor fit failed and
L-SML never degenerated. The 6 splits at k ≤ 4 (where L-SML is undetermined, Step 205) change
nothing: dropping them leaves the room at +2.17pp and every arm's ordering intact.
