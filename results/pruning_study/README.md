# Trimming study — can measurements be picked without answer keys?

Open **[index.html](index.html)** for the charts. This file is the text version.

Run date: 2026-07-26. All numbers computed on the current 25 in-scope test sets.
Every experiment re-checks the standing anchor (the six hand-picked measurements
must reproduce **0.7594**) before reporting anything.

---

## The question

The detector takes ~30 measurements from a language model while it writes an
answer and combines them into one number: how likely the answer is wrong. Two
things are open — **which** measurements to keep, and **how much to trust each
one**. This study tests both, with no answer keys used at any point.

## Reference points

| | Accuracy (AUROC) |
|---|---|
| Ceiling: all measurements, trust levels learned from answer keys | 0.7809 |
| Six hand-picked measurements (chosen using answer keys) | 0.7594 |
| Best automatic picker so far | 0.7524 |
| All measurements, our detector | 0.7457 |
| Six measurements picked at random | 0.7360 |

The gap that matters: same measurements, same directions, **0.7664** averaged
equally vs **0.7809** with learned trust levels. That **1.45 points** is the
whole value of knowing how much to trust each measurement.

---

## Headline findings

### 1. The fit score the trimming algorithm steers by has its sign backwards

The algorithm removes whichever measurement most *improves* how well a "these
are all noisy readings of one hidden thing" model explains the data. Two
independent experiments say that direction is wrong:

- **Experiment 3** — among combinations of equal size, those fitting the model
  *worse* are *more* accurate: correlation **+0.223**, positive in **24 of 25**
  test sets.
- **Experiment 2** — repairing the worst-fitting group scores **0.7080**, while
  repairing a **randomly chosen** group scores **0.7302**. The localizer is
  **2.2 points worse than random**.

Same explanation both times: the strongest measurements (average uncertainty per
token, average surprise per token, confidence in the chosen word) are
near-duplicates of one another, and that duplication is exactly the extra shared
structure a single-factor model cannot absorb. **Poor fit marks where the signal
is concentrated, not where the junk is.**

The constructive reading: the criterion is not useless, it is *inverted*. A rule
that preserves badly-fitting structure — or trims to *increase* misfit — is a
different algorithm, and it has not been tested.

### 2. Trimming has a high ceiling and a poor average

A typical smaller combination is worse than keeping everything, in **25 of 25**
test sets — there is no middle size that is good on average, so a "stop where the
curve turns" rule has no turn to find. But the *best* combination found at each
size runs several points above a typical one, and the best ones are small (3–11
measurements). All the value is in choosing well; none in being small.

### 3. Nothing tested closes the weight-estimation gap

All 16 weighting pipelines land between **0.7434 and 0.7555** — a 1.2-point
spread, every one below the six hand-picked measurements. Main effects per slot:

| Slot | Option | Accuracy |
|---|---|---|
| Loading estimator | triplet method-of-moments | 0.7548 |
| | low-rank plus sparse split | 0.7538 |
| | leading eigenvector (current) | 0.7527 |
| | robust re-weighted fit | 0.7494 |
| Conditioning | random-matrix cleaning | 0.7534 |
| | none | 0.7520 |
| Weighting | by signal (current) | 0.7533 |
| | by signal over noise ("precision") | 0.7520 |

Precision weighting — predicted beforehand to be worth +0.5 to +1.2 points —
measures **−0.13 points** as a main effect. No option is adopted: the differences
are inside the noise for 25 test sets.

### 4. The grouping step does not earn its keep

Switching off the detector's internal "group similar measurements" stage is
better at **every size tested** (13/13, 12 individually significant). On the full
pool: **0.7457 → 0.7533** (p = 0.024, better on 17/25). The effect is small, so
this says the stage is not doing its job — not that removing it is a real gain.
It was built to handle near-duplicate measurements, and Experiment 4 confirms
those are present in every test set.

### 5. About a million cached scored combinations are stale

An exhaustive sweep of ~1.03M combinations sits in `results/subset_sweep/`. Only
**5 of 19** test sets still reproduce; disagreements reach **0.37 AUROC**. The
test sets were re-graded after the sweep ran. Everything in this study was
recomputed. **Any future analysis reaching for that cache needs this check
first.**

> An earlier pass of this study used that cache and reported the fit-score
> correlation as ≈ −0.02 with inconsistent sign. That reading is superseded by
> the +0.223 above.

---

## Two things Omri's proposal got right

- **The prototype that supposedly refuted this idea was broken.** Its fit score
  was `‖Cov·v₁ − λ₁·v₁‖`, which is **zero by definition** (measured ~2×10⁻¹⁵). It
  ranked candidate removals by floating-point rounding error, so the 0.7004 in
  the record is the score of a coin flip. The idea had never been tested.
- **The tie-breaker instinct was right.** Near-ties dominate: about **11 of 18**
  removal steps had a runner-up within 10% of the best candidate, so whatever
  breaks ties makes most of the decisions. The Laplacian-smoothness variants
  differ too little to separate (0.7063 coin-flip vs 0.7112 best), and the
  question is moot once the localizer is shown to point the wrong way — but the
  diagnosis of *where* the algorithm's decisions actually get made was correct.

---

## Experiments

| Folder | What it tests |
|---|---|
| `01_grouping/` | Does the detector's internal grouping stage help, at every subset size |
| `02_cluster_localized/` | Omri's algorithm: find the worst-fitting group, repair it. Controls for group choice and tie-break |
| `03_size_and_criterion/` | Is there a best number of measurements; does the fit score rank combinations |
| `04_weight_diagnostic/` | Where the label-free trust levels differ from learned ones |
| `05_weighting/` | 16 weighting pipelines as a three-slot factorial |

Each folder holds `index.html` (charts + explanation) and the CSVs behind every
figure. Raw per-combination data is in `.npz`/`.csv` so any result can be
re-derived or re-tested.

## Reproducing

```
python scripts/pruning_study/exp01_grouping.py
python scripts/pruning_study/exp02_cluster_localized.py
python scripts/pruning_study/exp03_preflight.py
python scripts/pruning_study/exp04_weight_diagnostic.py
python scripts/pruning_study/exp05_weighting_factorial.py
python scripts/pruning_study/render_reports.py      # rebuild pages from saved CSVs
python scripts/pruning_study/build_index.py
```

## Speed changes made to shared code

Three avoidable costs in `spectral_utils/fusion_utils.py`, all verified to leave
output bit-identical:

- `_score_matrix_lsml` was a pure-Python quadruple loop — vectorised, **34×**
  faster (482 ms → 14 ms at 30 measurements), max difference 8×10⁻¹⁶.
- `_residual_lsml` and `_estimate_von_voff` had O(m²) Python loops — vectorised.
- `lsml_continuous` computed that O(m⁴) matrix **even when nothing reads it**
  (the `groups=` path). New `compute_score_matrix=False` flag skips it: **103×**
  faster, default unchanged.

Regression checked: group count, misfit and group sizes identical on a reference
test set (K=4, misfit 88.455, sizes [5,7,7,11]), and the 0.7594 anchor holds.

## Caveats

- The number of available measurements varies by test set (27–30), so "size 30"
  only covers the 6 test sets holding all 30. Full-pool comparisons use each test
  set's own complete set and cover all 25.
- 25 test sets resolve differences of roughly a point. Nothing here is adopted on
  a smaller difference, and win/loss counts are reported beside every average.
- Experiment 5's random-matrix cleaning is eigenvalue clipping at the
  Marchenko–Pastur edge (Laloux et al.), not the Ledoit–Wolf 2020 analytical
  non-linear shrinkage. Labelled as implemented, not as cited.
