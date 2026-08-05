# Handoff — the feature-selection channel is closed; the weights channel is open

**Written 2026-08-05, after Steps 223–224.** For whoever picks this up next, including a
different tool or a cold session. Read this before proposing anything in the U-PCR
feature-selection channel — a large amount of it is already closed with numbers, and the
most common failure mode in this line of work has been re-proposing a family that was
already measured.

Companion files: `HISTORY.md` Steps 223–224 (the narrative), `PROGRESS.md` (the current
headline), `results/action_items_jul2026/item2_upcr_clustering/PLAN_NEXT.md` (the older
plan, now banner-flagged as executed).

Shareable report of Steps 223–224:
https://claude.ai/code/artifact/a4d307aa-3053-4e52-83df-8c2c917967f5

---

## 0. The standing instruction — tailor, do not transplant

**Omri, 2026-08-05, after reading the Step-224 results. This governs everything below and it
is the single most important paragraph in this file.**

> A published metric is **inspiration, not a specification.** Take the *concept*, then think
> about it and develop it into the form our problem actually needs. This is true of every
> algorithm and metric we try, not just the one that prompted it. If a variant needs its own
> discussion before it is built, then it gets its own discussion.

Step 224 ran 21 published conditions **faithfully** — a fidelity reviewer cut the primary
family from eight to five precisely for not being faithful enough — and every one of the 111
variants lost to the deployed rule. That result is real and it stands. But read what it is
evidence *for*: **transplanting a published keep rule into this channel does not work.** It is
not evidence that the ideas inside those papers are exhausted, and the next round should not
be another faithfulness exercise.

Practically, this changes three things:

1. **Fidelity to the paper stops being the acceptance criterion for a new arm.** It remains the
   criterion for anything *labelled* with the paper's name — do not report a tailored variant
   under an author's name (trap §7.3, and see `a9_dpp.py`'s docstring for the convention:
   describe the mechanism, cite the idea, do not claim the method).
2. **A published statistic is a starting point to be reshaped**, and the reshaping is where the
   work is. §4.1 below works one example through in detail.
3. **Each variant is discussed before it is built.** Omri's words: *"If we need to run a
   discussion on each variant — so be it."* Do not batch-build a family and report the table.

---

## 1. The two numbers everything is measured against

| | value | how it is defined |
|---|---|---|
| **The room** | **+2.25pp**, CI [+1.52, +3.05], 23W/1L | a label-guided greedy search over subsets, i.e. the same search handed the answers, at k ≈ 11.75 |
| **The floor** | **−0.84pp**, CI [−1.07, −0.63], 1W/23L | trimming the deployed keep set at random |

Both re-derived exactly in every run since Step 213. **If a new run does not reproduce
these two numbers, the run is void** — do not read any other number from it.

Metric throughout: **difference in AUROC, in percentage points, paired over 24 cells**
(one cell = one dataset×model pair). Select on half A with no labels, score on half B.
CIs are 95% bootstrap over cells; *p* is Wilcoxon signed-rank; multiplicity is Holm with
the family size **fixed at its registered value**.

---

## 2. What is CLOSED in the feature-selection channel — do not re-propose these

### 2.1 Per-feature ranking (Steps 221–222)
- The **true correlation with correctness** — an oracle statistic — identifies the good
  features (+0.11 overlap, p<1e-4) and **buys +0.08pp, p=0.62**.
- A **perfect** estimate of `ρ̂` is priced at +0.34pp with a CI crossing zero.
- **Eight label-free per-feature rankers**: none clears the floor; two are significantly
  *worse* than random pruning.
- The arm that **most** identifies the good features is the **worst** performer.

### 2.2 Set-level covariance functionals (Step 223)
Five arms over `C_S = λλᵀ + Ψ + Δ` with sparse Δ — McDonald's ω, `Σ|Δ_ij|`, effective
number of independent views, within-set cohesion, total loading. **Best is +0.08pp,
Holm 0.72.**

### 2.3 The published unsupervised feature-selection literature (Step 224)
21 conditions, 111 variants, 24 cells × 5 splits × 2 arenas.
**Every single variant is negative against the deployed rule.** Pre-registered primaries:
DUFS Eq.(7) −0.96pp (Holm 0.0072), Concrete Autoencoder −2.74pp, Laplacian Score and SPEC
−3.77pp, Eq-14 residual −5.89pp. Nothing clears a same-size random floor after Holm.

Newly implemented and scored, all with planted-world known-answer tests:
`spectral_utils/selectors/a8_lscae.py` (LS-CAE), `a9_dpp.py` (DPP MAP),
`a10_mmdufs.py` (mmDUFS), `a11_rfae_scfs.py` (RFAE + SCFS).

**Scope this claim precisely (see §0).** What is closed is **transplanting a published keep
rule into this channel** — the arms were run faithfully, and a fidelity review cut the primary
family from eight to five for not being faithful enough. It is *not* a finding that the ideas
in those papers are exhausted in a **tailored** form. §2.4 is the one sub-result that survives
reformulation, because it is about the *direction* of the criterion rather than its algebra.

### 2.4 Anti-redundancy / diversity criteria — **harmful, not merely useless**
Measured three independent ways, three different formalisms:

| condition | mechanism | vs same-size random subset | W/L |
|---|---|---:|---|
| `cohesion_set` | greedy on mean \|corr\|, set-level | −0.75pp | last of 5 |
| `decorr_s5` | greedy min pairwise \|ρ\| | −5.98pp | 1W/23L |
| **`dpp.k4`** | pivoted-Cholesky log-det volume | **−8.08pp** | **0W/24L** |

The damage is **dose-dependent**: DPP's own data-driven stop declines to prune (size 21.9)
and is neutral at −0.47pp, 12W/12L. **Do not propose another diversity criterion.** This
covers the entire "redundancy reduction" reading list — Barlow Twins, VICReg, DPPs,
manifold k-medoids, minimum-redundancy subspace learning are the same condition in
different clothing.

### 2.5 Ruled out by reading the paper, not by experiment
Record these so they are not re-proposed as label-free arms:
- **SEFS** (Lee, Imrie, van der Schaar ICLR 2022) — π is fixed and *equal across all
  features* throughout the self-supervised phase; it becomes feature-specific only under
  `ℓY(y,·)`. **Not label-free at selection.** *(But see §4 — a pseudo-label makes it
  runnable.)*
- **Few-Sample FS via Feature Manifold Learning** — **Cohen, Shnitzer, Kluger, Talmon,
  ICML 2023**, *not* Bracha Laufer-Goldshtein as an earlier reading list claimed. It is
  "few-sample **supervised** FS" and learns the manifold of *each class*. *(Same caveat —
  see §4.)*
- **VICReg** — its variance hinge `max(0, γ − √(Var+ε))` is **identically zero** on our
  data because views are z-scored. Verified across **6,820 views**: max |std−1| = 1.3e−14,
  zero non-zero hinges. Its invariance term needs two augmented views we do not have.
  What remains is the covariance term = Barlow Twins = already measured.
- **Graph Information Bottleneck** — needs a labelled graph, not a feature matrix.

### 2.6 Other closed channels (Steps 206, 218, 220)
Pool composition (removing views −0.50pp; adding the strongest unused views negative on
all 6 pre-registered variants); non-monotone reshaping (+0.05pp conditional); sign and
orientation (−0.06pp); U-PCR's own exclusion criterion (all magnitudes < 0.16).

---

## 3. Three facts that constrain anything proposed next

1. **The search is not the bottleneck.** Handed half-A labels, the *same* greedy takes
   **+1.88pp — 84% of the room.** Whatever fails, it is the objective, not the optimiser.
2. **The good set is not a stable target.** Two random half-splits of the *same* cell,
   both using that cell's own labels, produce good sets agreeing at Jaccard **0.524**
   (across cells 0.303). Many different subsets are good.
   **⇒ Overlap-with-the-good-set is no longer admissible evidence for any arm.** Every
   `overlap_*` secondary in Steps 213–223 was scoring reproduction of a set a rerun would
   only half reproduce.
3. **Structural fit points the wrong way.** The good subsets fit the 1-factor model
   *worse* than matched random subsets: additive misfit +0.131 (23W/1L, p=6e-7),
   multiplicative rank-1 misfit +0.100 (21W/3L, p=2.5e-5). Any criterion that rewards
   better conditional-independence fit is aimed away from the target.

---

## 4. THE LIVE LINE — triplet consistency, as a concept to develop

**This is the next piece of work, and it is Omri's idea.** Read §0 first: FUSE's
triplet-consistency machinery is **not to be transplanted as published**. The concept — *a
triplet of views carries a checkable constraint under conditional independence* — is the
input. What we build out of it is ours to design.

There are **two distinct uses**, and earlier sessions confused them. Both are live:

| | what it scores | which channel | status |
|---|---|---|---|
| **§4.2** | **the views themselves** — keep the views that sit in many well-behaved triplets | feature selection | Omri's original framing. Naive version measured and weak; **the tailoring is the work** |
| **§4.3–4.4** | **each sample** — the triplet-averaged posterior as a pseudo-label | weights / aggregation | FUSE's own Steps 4–5; targets the +1.24pp that is measured and unclaimed |

### 4.1 What FUSE actually does
`papers/FUSE - Ensembling Verifiers with Zero Labeled Data.pdf` — Lee, Ma, Zhao, Nair,
Spector, Cohen, **Candès**, arXiv:2604.18547. Algorithm 1:

```
1. tau* = argmin_tau  S_hat( g_tau(V) )        # S_hat = empirical TCI-violation statistic
2. V~   = g_tau*(V)  in {+-1}^{N x m}          # binarised at the chosen threshold
3. apply Jaffe et al. (2015) spectral algorithm to V~  ->  (psi_hat, eta_hat, b_hat)
4. p_hat(r) = 1/C(m,3) * SUM_{j1<j2<j3} p_hat_{j1,j2,j3}(r)      # Eq. (9) PSEUDO-LABEL
5. theta*  = argmax_theta  Acc_hat(theta)      # optimise ANY parametrised ensemble rule
```

Three levers, and note which channel each targets:

| FUSE step | what it is | our channel | status |
|---|---|---|---|
| Step 1–2 | choose the **score transformation** (binarisation threshold) to *minimise* triplet-CI violation | encoding | **untested in this form.** Steps 134–136 found encoding is the dominant lever; `binarize_classifiers` exists from Step 105, but the threshold has never been chosen by a TCI-violation criterion |
| Step 4 | **triplet-averaged per-sample posterior** as a pseudo-label | — | **new here in this form** |
| Step 5 | pseudo-label trains an arbitrary aggregation rule (e.g. LR) | **weights** | **OPEN, with measured room** |

### 4.2 Use one — scoring the VIEWS by triplet consistency, and how to tailor it

**Omri's framing, in his words**: *"using this metric of triplets of features/views to score
the views themselves and choose those who are doing well… we can improve this by thinking of it
and developing it."*

**Start from the algebra, not from FUSE's statistic.** Under conditional independence the
off-diagonal covariance of oriented views is rank-1: `C_ij = v_i v_j`. For a triplet that is 3
equations in 3 unknowns and solves in closed form:

```
v_i^2 = C_ij * C_ik / C_jk
```

**At m = 3 the system is exactly determined, so there is no residual** — every triplet fits by
construction. Only *admissibility* is testable: `SIGN: C_ij·C_ik·C_jk > 0` and
`MAGNITUDE: v_i² ≤ 1`. The naive score is then "fraction of a view's triplets that pass",
and §5 records what that is worth: **Spearman +0.0386, CI [−0.0073, +0.0856]** against the
good sets — weak, and the magnitude bound makes it worse.

**Do not read that as the idea failing.** It is the as-published admissibility test, which is
exactly the thing §0 says not to transplant. Six concrete directions for developing it, roughly
in order of how much new information each adds:

1. **Go to quadruplets — this is the sharpest one.** At m = 4 the rank-1 model has 6 equations
   and 4 unknowns: **2 spare, so a genuine residual exists.** m = 4 is the *first* order at
   which "how well does this set fit" is even a defined question. Everything at m = 3 is
   pass/fail admissibility; everything at m ≥ 4 is a continuous misfit. Never run.
2. **Score the *stability* of `v̂_i`, not the pass rate.** Each of view *i*'s
   `(m−1)(m−2)/2` triplets yields its own estimate `v̂_i`. A well-behaved view should give a
   *consistent* estimate across them. The **variance (or IQR) of `v̂_i` across its triplets** is
   a completely different statistic from the pass rate, and it is much closer to what the
   estimator actually consumes. This is the cheapest one to try and it reuses the existing
   probe code almost unchanged.
3. **Make the score continuous and reliability-weighted.** A pass rate discards *how badly* a
   triplet violates. A product `C_ij·C_ik·C_jk` that is barely negative flips sign on noise; one
   that is strongly negative is a real violation. Weight each triplet by its own precision
   (Fisher-z standard errors on the three correlations, or simply `min|C|`) before averaging.
4. **Check the sign of the objective against §3.3 before trusting it.** Good subsets fit the
   1-factor model **worse** than matched random ones (misfit +0.131, 23W/1L). So "keep the views
   in many *passing* triplets" may be pointed the wrong way, and the tailored rule might keep
   views in many *violating* triplets — violation as an **interaction detector**, not as a
   defect. Measure the sign first; do not assume it.
5. **Use it to select triplets, not views.** FUSE's Eq. (9) averages over **all** `C(m,3)`
   triplets. A tailored Eq. (9) averages only over admissible ones, or weights each triplet by
   its consistency. That is a change to the pseudo-label itself and it composes directly with
   §4.3 rather than competing with it.
6. **Condition on the deployed keep set.** Everything above can run over the full pool or over
   U-PCR's kept views. Omri's stated intent is **on top of** the existing selection, so the
   default is the kept views, with full-pool as the sensitivity.

**Honest prior, stated once and then set aside**: this use lands in the feature-selection
channel, which has now absorbed per-feature rankers, set-level covariance functionals, and the
published literature (§2) — so the base rate is poor. Directions 1 and 2 are the ones that are
genuinely *new objects* rather than restatements, and they are where the prior is weakest.
That is an argument about ordering, not about whether to run it.

### 4.3 Why the weights channel is the right target
Step 220 measured the weights ceiling but only swept `span(v₁, v₂)`. Outside that span:

> **supervised linear − best-in-span = +1.24pp, CI [+0.17, +2.29], p = 0.016**

That is unexploited, and it is exactly the quantity FUSE's Step 5 optimises. The
feature-selection channel absorbed three families and the whole published literature; the
weights channel has a positive, significant, unclaimed number sitting in it.

### 4.4 The forks — each gets a discussion with Omri before it is built
Per §0, these are not a checklist to resolve in one pass; each variant is discussed on its own.
They are genuine forks, not rhetorical:
- **The TCI-violation statistic `Ŝ`.** FUSE's own definition (read their §2 — extract at
  `papers/extracted/fuse-ensembling-verifiers-with-zero-labeled-data.md`), our existing
  admissibility test (`C_ij·C_ik·C_jk > 0`, 83.4% pass — see §5), or one of the §4.2
  developments. **§0 says the default answer is the tailored one**, and FUSE's `Ŝ` is the
  reference point it is measured against, not the thing to ship.
- **Binarise or not.** FUSE binarises to `{±1}`. Our pipeline is continuous and Step 134
  established continuous (CONT) as the main config. Do we binarise (following FUSE) and
  sweep τ, or keep continuous and skip Step 1?
- **What the pseudo-label trains.** LR on the views (FUSE's example), or the U-PCR weight
  vector directly, or a reweighting on top of the existing `w`?
- **Where it sits relative to U-PCR.** Omri's stated intent is **on top of** the existing
  U-PCR selection — U-PCR keeps its views, then the pseudo-label improves the aggregation.
  That must be the default arm, with "replace" as a sensitivity.
- **Guard against the Step-199 trap.** A pseudo-label seeded from a fusion reproduced that
  fusion on 25/25 cells. The triplet-averaged posterior must be shown to carry information
  the deployed `w @ F` score does not — measure their correlation *before* building
  anything on top.

### 4.5 What this reopens
Because a pseudo-label is a form of labels, **the two conditions ruled out in §2.5 for
needing labels become runnable**: SEFS's supervision phase, and Feature Manifold Learning's
per-class kernels. Both should be revisited once a validated pseudo-label exists. They are
*not* closed — they were deferred for want of a label.

---

## 5. The triplet probe already run, and what it does and does not show

An earlier probe in this session scored features by the fraction of their triplets passing
the admissibility test derived from `v_i² = C_ij·C_ik / C_jk`:

- `SIGN: C_ij·C_ik·C_jk > 0` and `MAGNITUDE: v_i² ≤ 1`
- At m=3 the system is exactly determined, so **there is no residual test** — only
  admissibility.
- **The test is not degenerate**: sign passes on **83.4%** of triplets (range 0.66–0.98),
  and 0 of 119 splits have all triplets passing. 17% violation is an independent
  corroboration of the misfit in §3.3.
- As a **per-feature ranking score** it barely points at the good features:
  Spearman +0.0386, CI [−0.0073, +0.0856], 60+/59− across splits; adding the magnitude
  bound makes it worse (−0.0084). Good-minus-non-good pass rate +0.0127 [+0.0021, +0.0235].

**Read this correctly — it bounds one thing and nothing else.**

- It **is** a first measurement of §4.2 (scoring the views), in its most naive form: binary
  pass/fail, unweighted, at m = 3 where only admissibility is testable. Treat it as the
  baseline the §4.2 developments have to beat, **not** as a verdict on the idea. Directions 1
  and 2 in §4.2 are different statistics, not tunings of this one.
- It says **nothing** about §4.3–4.4, which is a *per-sample* pseudo-label feeding the
  *weights*. Do not cite this probe there.
- The 17% violation rate is worth keeping for its own sake: it is an independent corroboration
  of the misfit in §3.3, measured a completely different way.

Script: `scripts/upcr_study/probe_triplet_consistency.py` — **exploratory, not pre-registered,
not quotable in a results table.** Committed so the next session can extend it rather than
rewrite it. It needs the dataset (see §9).

---

## 6. Omri's modelling idea — the sparse-Δ relaxation, still live

**The idea**: U-PCR assumes `E[h_i h_j] = 0` (uncorrelated errors), which is what makes the
system solvable. That assumption is false on our data. Model it instead as
`C_S = λλᵀ + Ψ + Δ` with `Δ` the residual dependence, and solve under a weaker assumption.

**What was established (Step 223)**:
- The assumption is genuinely broken: normalised additive misfit **0.464** on the full pool.
- The violation is **sparse, not low-rank**: top decile of pairs carries **44%** of the
  residual mass (uniform would be 10%); leading eigenvalue share only **0.33**.
- **Degrees of freedom work out**: `m(m−1)/2` pair equations against `m+1` unknowns leaves
  **360 spare equations at m≈28** and 50 at m≈12. A *full* Δ adds exactly `m(m−1)/2`
  unknowns and saturates the system; a **sparse** Δ fits comfortably.

**What was tested and failed**: using the fitted Δ as a *feature-selection* criterion
(`resid_dep = Σ|Δ_ij|`, and the ω / m_eff / loading arms built on the same fit) — none
clears the floor.

**What has NOT been tested**: using the sparse-Δ model **in the estimator itself** — i.e.
solving for `ρ̂` and the weights under `C_ij = ρ_i + ρ_j − g² + Δ_ij` with sparse Δ,
instead of assuming Δ = 0. That is a change to the *fit*, not to the *selection*, and it
lands in the same open channel as §4 (weights/estimation). **This is the second live line
and it composes with FUSE**: FUSE's Step 1 exists precisely to make the CI assumption more
nearly true by transforming the scores; the sparse-Δ route instead relaxes the assumption
in the model. They are two attacks on the same weakness and could be compared directly.

**Implementation already exists**: `spectral_utils/composite_reliability.py`
(`fit_sparse_factor`, `delta_structure`).

**It is tested** — `scripts/test_composite_reliability.py`, CPU-only, no dataset, runs in
seconds and passes: planted-Δ support recovery, the sparse-vs-low-rank separation, greedy
landing on exact fixed *k*, the `MIN_SET=3` floor, and all five objectives oriented the same
way. The duplicate-degeneracy check is the load-bearing one: it asserts the degeneracy is
*real* under a plain factor fit and that the sparsity prior is what removes it, so if it ever
starts passing trivially the objective has stopped doing what it claims. **It was untracked
until Step 225** — written during Step 223 and never `git add`ed, which is why an earlier
draft of this file called the module untested. Run it before touching the module.

One real gap remains: it is a standalone script, not wired into
`scripts/smoke_selectors.py`, so the accumulating gate does not cover it. Wiring it in is a
small job worth doing.

**Provenance for the numbers in §3.3 and §4.3**, both committed as exploratory scripts
(not pre-registered, not quotable in a results table, both need the dataset):
- `scripts/upcr_study/probe_delta_violation.py` — the 0.464 misfit, the 44%-in-top-decile
  sparsity, and the good-sets-fit-worse sign.
- `scripts/upcr_study/probe_delta_followups.py` — follow-up (c) is where the **+1.24pp
  outside `span(v₁,v₂)`** comes from. **Re-run and confirm that number before building on
  it**: it is exploratory, its CI clears zero but not by much, and it is the single
  load-bearing result for choosing the weights channel over the closed selection channel.

---

## 7. Practical traps this project has already hit — dodge these

1. **Ask which method to evaluate. Never infer it.** Several methods have more than one
   entry point and the obvious one is often the legacy path. U-PCR's maintained arm is
   `spectral_utils.upcr.upcr_fit` over the full pool with `sign(ρ̂)` polarity
   (`scripts/labelfree_standing_report.py:upcr_rho_oriented`) — **not**
   `fusion_utils.upcr_pipeline` and **not** `eval_subset_flex(fusion='upcr')`.
2. **Hand-picked subsets are reference rows, never the contribution.** `GOOD_5`, `GOOD_6`,
   `LOCO_5`, `STABLE_H9` and anything seeded on the `epr` anchor carry a prior. Label them.
   Note `a5.mrmr_*` acquires the `epr` prior *through `cell.anchor`* — a name-based
   prior check will not catch it.
3. **A selector's published objective is not its published keep rule.** The Step-224
   fidelity review cut the primary family from eight to five: GroupFS's group-gate readout
   had been replaced by DUFS gates; `a1.upcrres_greedy` ran our own residual, not
   Jaffe/Nadler/Kluger Eq. 14; `mcfs_adapt` carried an undocumented re-weighting. **Read
   the readout, not just the loss.**
4. **DUFS gates are SIGNED.** `mu` starts at 0.5 (or 0 with a +0.5 offset) and Adam drives
   rejected features *negative*, so ranking by `|mu|` promotes the most strongly rejected
   views. On a probed cell the two rules disagree on 4 of 13 kept views.
5. **Holm's family size must be fixed at its registered value.** If a missing or NaN arm
   shrinks the family, every survivor looks more significant than the pre-registration
   allows.
6. **A selector that falls back returns the WHOLE POOL.** Scored naively, its size-matched
   floor draws the whole population every time, so floor ≡ arm and the split contributes an
   exact zero — shrinking a real effect toward the floor in proportion to how often the
   method crashed. Exclude fallback splits and report the rate.
7. **Checkpoint every cell.** A 24-cell × 2-arena sweep is hours; a run with no incremental
   saving loses everything on any interruption. `exp16_paper_conditions.py` has
   `--resume`, and its floors/nulls are drawn from substreams keyed by
   `(cell, split, arena, size)` so a single-arm run reproduces the full sweep exactly —
   which is what makes `--only` legitimate.
8. **Gemini's literature backfills have fabricated attribution three times** (Steps 176,
   179, and Step 224's reading list, which misattributed the Feature Manifold paper to
   Bracha and called it unsupervised, and attributed the subspace-clustering slot to Ma et
   al. when the PDF is Parsa, Zare & Ghatee). **Always check `papers/extracted/` before
   trusting an author, venue, or supervision claim.**
9. **The `keep` arena cannot reach the room.** 19% of the good set lies outside the
   deployed keep set, so "fraction of room recovered" is undefined there. Use the `full`
   arena as primary.
9b. **An arm that carries a structural prior needs a floor that carries the same prior.** The
   ℓ0-CCA **dry run** (`results/upcr_study/15_l0cca_partial/`, every score NaN) scored its
   channel round-robin arms at **+0.32pp, p = 0.019** against the pruning floor **with no
   signal in them at all** — because taking one view per channel in rotation is a
   channel-*balance* prior that pays by itself (the good sets are 51% spectral; the marginal
   rankings pick 32–34%). Scored against `chan_rr_random` instead, the same arms are −0.05pp
   and −0.40pp. **Run the structural dry run before the real one** — it is nearly free and it
   catches this class of error before a number exists to be attached to.
10. **Both contrasts, always.** A same-size random floor is a *low bar at small sizes* — a
    3-view rule can clear it by 3.6pp and still lose to the 21-view incumbent by 2.3pp.
    Report vs-floor *and* vs-deployed.

---

## 8. Suggested order of work

1. **Read FUSE properly** — §2 for the TCI-violation statistic `Ŝ` and Theorem 2.3, §4 and
   Appendix E for the pseudo-label variants. The current digest
   (`papers/digests/fuse-ensembling-verifiers-with-zero-labeled-data.md`) is from the
   2026-07-13 batch, is thin, and **does not mention triplets or pseudo-labels at all** —
   re-digest it. Read it as the source of a *concept* to develop (§0), not a recipe.
2. **Discuss §4.4's forks with Omri, one variant at a time.** Do not batch-build.
3. **Cheapest first, from §4.2**: the variance of `v̂_i` across a view's triplets (direction 2)
   reuses `probe_triplet_consistency.py` almost unchanged and is a genuinely different
   statistic from the pass rate already measured.
4. **Then the quadruplet residual (§4.2 direction 1)** — the first order at which the model
   has spare equations and "fit" is defined at all.
5. **Then the pseudo-label**, with the Step-199 guard (correlation against the deployed
   `w @ F` score) run *before* anything is built on top of it.
6. **Price the weights channel** against its own ceiling: the target is the +1.24pp outside
   `span(v₁,v₂)`.
7. Only then revisit SEFS and Feature Manifold Learning (§4.5), and the sparse-Δ estimator
   (§6).

Whatever the arm, run the **structural dry run** first (trap §7.9b) and check the
**void-run condition** (§1) before reading any number.

---

## 9. If you have the repo and not the data

This is the expected situation for a session on a different machine. The per-cell feature
`.pkl`s are gitignored and live on Drive / the cluster; `scripts/upcr_study/common.py:load()`
will not find them.

**What travels with the repo, deliberately:**

- **All 66 research PDFs** in `papers/` plus the root `Tenzer2022_*.pdf`, un-ignored in
  `.gitignore` on 2026-08-05 specifically so the literature migrates with the work. With
  `papers/extracted/` (63 files) and `papers/digests/` (52), the whole reading pipeline is
  self-contained — check `papers/index.md` before re-extracting anything.
- **Every result CSV and `summary.json`** under `results/upcr_study/`, including the ℓ0-CCA
  **dry run**. Each row carries the held-out AUROC *and* its matched floor, so **every headline
  number in Steps 210–224 can be re-derived, re-tested and re-aggregated from the CSVs alone.**
  `results/upcr_study/README.md` is the reader's guide: directory map, CSV schemas, the
  aggregation order that reproduces the published numbers, and a worked example.
- **All selector implementations** (`spectral_utils/selectors/a1`–`a11`), each with a
  planted-world `smoke()` known-answer test that runs **without the dataset** —
  `python scripts/smoke_selectors.py`. That is how to verify a new or modified selector on a
  machine with no cells.

**What you cannot do**: run a new arm, re-fit U-PCR, or recompute any `ρ̂`. Those need the
cells. Ask Omri for a data drop before planning anything that requires them.
