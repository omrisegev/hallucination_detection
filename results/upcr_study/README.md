# `results/upcr_study/` — reader's guide

**Written for a session that has this repo and nothing else** — no Google Drive, no inference
`.pkl`s, no cluster. That is the expected situation for whoever picks this up next.

Every headline number in Steps 210–224 can be **recomputed from the CSVs in this directory
alone**, because each run writes one row per (cell, split, arm) with the held-out AUROC *and*
its matched floor already in the row. You do not need the source data to re-derive, re-test,
or re-aggregate any published claim. You need the source data only to run a *new* arm.

---

## What you cannot do without the dataset

`spectral_utils/../scripts/upcr_study/common.py:load()` reads per-cell feature matrices that
live outside the repo (`*.pkl`, gitignored — see the "Raw inference data" section of
`CLAUDE.md`). Without them you cannot:

- run a new selector, fusion, or estimator arm;
- re-fit U-PCR, or recompute a view's `ρ̂`;
- reproduce a number from scratch rather than from the saved rows.

You **can** do all of the following from the CSVs: re-aggregate per-split → per-cell → paired
statistic, change the multiplicity correction, change the contrast (vs floor / vs deployed),
subset to a family of cells, re-bootstrap, or check any claim in `HISTORY.md`.

---

## Vocabulary

| term | meaning |
|---|---|
| **cell** | one (dataset × model) pair; there are **24** in this study |
| **split** | one random half-split of a cell; **5 per cell = 120 splits** |
| **half A / half B** | selection happens on A with **no labels**; scoring happens on B |
| **arena `full`** | the arm may choose from the whole pool — the real swap, and the **primary** |
| **arena `keep`** | the arm may only prune U-PCR's deployed keep set — where room/floor are defined |
| **the room** | **+2.25pp**, CI [+1.52, +3.05], 23W/1L — a label-*guided* greedy, i.e. the search handed the answers |
| **the floor** | **−0.84pp**, CI [−1.07, −0.63], 1W/23L — trimming the deployed keep set at random |

The metric everywhere is **difference in AUROC in percentage points, paired over the 24
cells**. CIs are 95% bootstrap over cells; *p* is Wilcoxon signed-rank.

**Void-run check.** Every script re-derives the room and the floor and asserts exp13's
reference arms per split to 1e-9. If a `summary.json` does not show `room.mean_delta ≈ 2.2498`
and `floor_vs_deployed.mean_delta ≈ −0.8446`, **discard every other number in that file.**

---

## The directories

| dir | step | script | question it answered |
|---|---|---|---|
| `00_reproduction_audit/` | 200 | `exp00_*` | does the deployed pipeline reproduce its published macro (0.7741)? |
| `01_g2_criterion/` … `09_add_test/` | 201–212 | `exp01`–`exp09` | U-PCR's own knobs: orientation, exclusion threshold, pool composition (leave-one-view-out, add-test) |
| `10_channel_ceilings/` | 213 | `exp10_channel_ceilings.py` | **where the room is** — selection vs weights vs orientation vs encoding. Defines `fit_cols`, `derive_cell`, `derived_arm_gate` that every later script imports |
| `11_posthoc_controls/` | 214 | `exp11_*` | controls for the ceiling measurement |
| `12_what_separates_good_features/` | 220 | `exp12_*` | what the label-guided good sets have in common. **Its splits are the canonical splits** every later script reuses |
| `13_incumbent_anchored_ranking/` | 221 | `exp13_*` | does true correlation with correctness buy anything? (**+0.08pp, p=0.62** — it identifies the good features and pays nothing) |
| `14_ranker_menu/` | 222 | `exp14_*` | eight label-free per-feature rankers — none clears the floor |
| `15_composite_reliability/` | 223 | `exp15_composite_reliability.py` | set-level covariance functionals over `C_S = λλᵀ + Ψ + Δ` — best **+0.08pp, Holm 0.72** |
| `15_l0cca/` | 223 | `exp15_l0cca.py` | ℓ0-penalised CCA between the entropy-trace and energy channels — **−0.12pp / −0.47pp** |
| `15_l0cca_partial/` | 223 | same, `--no-cca` | **the structural dry run. Read the next section — it is the most reusable thing in this directory.** |
| `16_paper_conditions/` | 224 | `exp16_paper_conditions.py` | the published FS literature as U-PCR keep rules — round 1, 8 conditions |
| `16_paper_conditions_dpp/` | 224 | same, `--only dpp` | DPP MAP alone (**−8.08pp at k=4, 0W/24L**) |
| `16_paper_conditions_round2/` | 224 | same, `--only lscae,mmdufs,rfae,scfs` | the four newly implemented conditions |

Two scripts share the `exp15_` prefix because they were written in parallel sessions. They are
different experiments and both are real: the **composite-reliability arm** and the **ℓ0-CCA
arm** of Step 223. Names kept as-is rather than silently renamed.

---

## `15_l0cca_partial/` — why a dry run is kept as a result

`--no-cca` makes every CCA score `NaN`. The arms still run, because `by_score` maps non-finite
scores to `−inf` and falls back to the tiebreak order. So this directory is a **run of the
harness with no signal in it at all** — and that is exactly what makes it valuable.

**What it caught, before any real number existed:**

| arm in the dry run | vs the pruning floor | what it means |
|---|---|---|
| every all-NaN arm (`cca_leverage`, `cca_gates`, `cca_init_r50`, all six λ arms) | **+0.086pp**, 16W/8L, identical to 15 decimal places | the harness's own tiebreak order, not a method |
| **`cca_leverage_rr` / `cca_gates_rr`** (channel round-robin) | **+0.320pp, CI [+0.034, +0.583], 18W/6L, p = 0.019** | **a significant "win" from zero signal** |

The round-robin arms take one view per channel in rotation. With no scores to rotate by, that
is nothing but a **channel-balance prior** — and it pays, because the label-guided good sets
are **51% spectral** while the marginal rankings pick only **32–34%**.

Consequence, and the reason this directory is committed: **a round-robin arm must be scored
against `chan_rr_random`** — same rotation, random order within each channel — and never
against the pruning floor, or the balance prior is silently credited to the method. In the full
run (`15_l0cca/`), scored correctly, the round-robin arms are **−0.05pp** and **−0.40pp**.

The general lesson generalises past this arm: **any arm carrying a structural prior needs a
floor that carries the same prior.** A same-size random subset is not that floor.

`summary.json` here carries an explicit `WARNING` and `is_partial_run: true`. Its one slightly
misleading phrase — "the ℓ0-CCA arms are NaN" — refers to the *scores*, not the AUROCs; the
AUROC columns are populated with the tiebreak fallback, which is the whole point.

---

## CSV schemas

**`16_paper_conditions*/splits_long.csv`** — one row per (cell, split, arena, variant):

```
cell, rep, arena, family, variant, size, auroc, floor, overlap, overlap_null,
fallback, seconds, error
```

- `auroc` is on **half B**; `floor` is the **size- and population-matched** random subset for
  that exact (cell, split, arena, size) — drawn from a substream keyed by that tuple, so a
  single-arm `--only` run reproduces the full sweep's floors exactly.
- `fallback = 1` means the selector failed and **returned the whole pool**. Its floor then
  draws the whole population too, so `auroc − floor ≡ 0` and the split silently shrinks any
  real effect toward zero. **Filter these out and report the rate.**
- `overlap` / `overlap_null` are Jaccard against exp12's good set. **Do not use them as
  evidence** — the good set is only Jaccard 0.524 stable against a rerun of the same cell (see
  `HANDOFF_FEATURE_SELECTION_AND_FUSE.md` §3.2).

**`16_paper_conditions*/splits_ref.csv`** — one row per (cell, split), the reference arms:

```
cell, rep, m, k, n_keep_deployed_halfA, auroc_deployed, auroc_greedy,
auroc_keepset, auroc_random_prune, floor_keep_at_k, k_exceeds_keepset
```

Join on `(cell, rep)` to get the **vs-deployed** contrast: `auroc − auroc_deployed`. The room
is `auroc_greedy − auroc_deployed`; the floor is `auroc_random_prune − auroc_deployed`.

**`15_*/splits.csv`** — wide format, one row per (cell, split), one column group per arm
(`auroc_<arm>_prune`, `<arm>_prune_cols`, `overlap_<arm>_prune`, `cond_null_<arm>_prune`).
`per_cell.csv` is the same aggregated over the 5 splits, minus the `_cols` strings.

**`summary.json`** — the aggregated paired statistics, plus a `PROVENANCE` string recording
that the arm family was pre-registered in the script's module docstring *before* the run. Check
`is_partial_run` first.

---

## Re-deriving a headline number without the dataset

```python
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

L = pd.read_csv("results/upcr_study/16_paper_conditions/splits_long.csv")
R = pd.read_csv("results/upcr_study/16_paper_conditions/splits_ref.csv")
d = L.merge(R[["cell", "rep", "auroc_deployed"]], on=["cell", "rep"])
d = d[(d.arena == "full") & (d.fallback == 0) & d.auroc.notna()]

arm = d[d.variant == "a2.dufs_pf"]
per_cell = (arm.auroc - arm.auroc_deployed).groupby(arm.cell).mean() * 100
print(per_cell.mean(), wilcoxon(per_cell).pvalue, (per_cell > 0).sum(), (per_cell < 0).sum())
# -> -0.9569 pp, p = 0.007189, 6W/18L — the DUFS Eq.(7) row of Step 224
```

That `p` is the **raw** Wilcoxon value. It happens to equal the published Holm-adjusted one
(`arms.full.primary_holm_adjusted_p_vs_deployed`) only because DUFS carries the *largest* p in
its family, and Holm's step-down leaves the last one unmultiplied. For every other arm the two
differ — read the adjusted value out of `summary.json`, and if you re-run the correction
yourself, **keep the family size fixed at its registered value** (5 primaries here) even when
an arm is missing or NaN.

Aggregate **per split → per cell → across cells**, in that order. Aggregating splits and cells
together weights cells by how many splits survived the fallback filter and will not reproduce
the published numbers.
