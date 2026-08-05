# The plan from here — after the feature-selection question was answered

> ## STATUS AT STEP 224 (2026-08-05) — READ THIS BEFORE THE REST OF THE FILE
>
> **The feature-selection channel has now absorbed three whole families and the published
> literature.** Per-feature rankers (Step 222), set-level covariance functionals and ℓ0-CCA
> (Step 223), and 21 published unsupervised conditions run as keep rules (Step 224). **Every one
> of Step 224's 111 variants is negative against the deployed rule**, and nothing in the
> pre-registered family clears a same-size random floor after Holm.
>
> **Everything below this banner about "the Step 223 joint-selection library" has been executed.**
> It is a record, not a plan. The four papers it recommended were either run (LS-CAE, mmDUFS,
> RFAE, SCFS, DPP MAP) or ruled out on inspection with a stated reason (SEFS is not label-free at
> selection; Feature Manifold Learning is supervised and is not Bracha's paper; VICReg's variance
> term is identically zero on z-scored views; Graph Information Bottleneck needs a labelled graph).
>
> **Three findings constrain whatever comes next:**
> 1. **Anti-redundancy is harmful here**, measured three independent ways, worst being DPP MAP at
>    −8.08pp vs a same-size random subset, 0W/24L. Do not propose another diversity criterion.
> 2. **The search is not the bottleneck.** Handed half-A labels, the same greedy takes **84% of
>    the room** (+1.88pp). The objective is what fails.
> 3. **The good set is not a stable target** (within-cell Jaccard 0.524). Many different subsets
>    are good, so **overlap-with-the-good-set is no longer admissible evidence** for any arm.
>
> Report: https://claude.ai/code/artifact/a4d307aa-3053-4e52-83df-8c2c917967f5
>
> **The forward plan now lives in `HANDOFF_FEATURE_SELECTION_AND_FUSE.md` (repo root)**, not in
> this file. Two live lines: FUSE-style triplet pseudo-labels aimed at the **weights** channel
> (+1.24pp measured outside `span(v₁,v₂)`), and the sparse-Δ relaxation applied to the
> **estimator** rather than to selection.


Updated at **Step 222**, extended **2026-08-04** with the Step 223 design. The July clustering
plan closed on ceilings at Step 220; this file replaced it. Step 221 closed one of Bracha's two
proposals with a number, and Step 222 priced the other one and with it the whole per-feature
family. Read `PHASE1_RESULTS.md` for the measurements.

> **The forward plan is the section "Step 223 — the joint-selection library" below.** A 12-paper
> deep dive on joint / redundancy-aware feature selection was folded in on 2026-08-04. Its
> recommendations are adopted as *objectives to price*, not as methods to build — see that section
> for why, and for the one idea in the library that inverts our worst measured result instead of
> repeating it.

> **Step 222 outcome, in one line: the ranker menu landed on the floor, which is the branch this
> file pre-registered as the *stronger* deliverable.** No label-free per-feature statistic clears
> the matched floor; two are significantly worse than random pruning; and the statistic that most
> identifies the good features is the worst performer of the eight. Combined with the true
> correlation already sitting on the floor, **the +2.25pp is not reachable by scoring features one
> at a time.** The section "The next step — the ranker search" below is now a record of what was
> run, not a plan; "If the whole menu lands on the floor" is what happened. The live question is
> the follow-on it names: **what property of the *set* the search exploits, measured directly.**

---

## Settled. Do not re-open any of these.

Each was measured, not argued. If a future idea lands on one of them, the answer is a number.

| | why it is closed |
|---|---|
| **Improving how U-PCR estimates the correlation** — Bracha's second proposal, the differentiable pair reweighting | **New at Step 221.** A *perfect* estimate, spent on feature selection, is worth **+0.34pp, CI [−0.47, +1.30], p = 0.88**. With the weighting blend at +0.19pp and polarity at −0.06pp, **all three channels a better `rho_hat` can reach are priced at zero.** No estimator, however built, can pay through any of them. |
| **Ranking features by their correlation with correctness, in any form** | The *true* correlation, given the good set's own size and the deployed keep set to prune, is worth **+0.08pp against a matched floor, p = 0.62**. The quantity itself is the dead end, not our estimate of it. |
| **Clustering the surviving features to improve the weights** | The best possible blend of the two principal directions is worth **+0.19pp held out, p = 0.57**. That is the ceiling for every "estimate the weights better" idea. |
| **All sign / direction work** | Giving every feature its *correct* direction is worth **−0.06pp**, 17 of 24 test sets unchanged to the digit. |
| **Anything that descends U-PCR's own model-selection criterion** | The criterion does not rank feature sets by performance — the correlation's sign flips depending on what you control for and every magnitude is under 0.16. |
| **Tuning the label-variance constant** | Inert three separate ways. Only the *ratio* to the average feature variance is identifiable, and for z-scored features the assumption-faithful ratio is 1.0, not p(1−p). |
| **Keeping fewer features by U-PCR's own ranking** | Loses at every size. And against a null matched on its own keep-set composition, that ranking is **below** chance at finding the good features (−0.05, 5W/19L, p = 0.016) — it avoids them. |
| **A fixed feature set shared across test sets** | The good sets do not transfer: −0.81pp at matched size, −2.37pp at ten features. |
| **Ranking features by ANY label-free per-feature statistic** — Bracha's first proposal, DUFS supplying the ranking | **New at Step 222.** Eight arms, pre-registered with directions, scored on the same splits. None clears the matched floor: DUFS gate value **−0.70pp** (Holm 0.36, a three-cell effect), principal-direction leverage −0.92pp, additive pair-fit residual −0.09pp, cluster round-robin +0.23pp (Holm 0.53, and gone at two of three loading scales). Two are significantly **worse** than random pruning: redundancy to the pool −3.13pp (Holm 0.002) and L-SML cluster size −1.61pp (Holm 0.008). |

Earlier closures that still stand: pool composition (Step 206), non-monotone reshaping as a
deployable gain (Step 218), synchronising the global sign (Step 213), and the clustered
pair-fit variant (Step 204).

---

## Where the room is, and what the bar is now

**About +2.25pp sits in which features get kept** — held out, CI [+1.53, +3.04], 23W/1L,
against a floor that trims the deployed keep set at random.

**Use these numbers, not the older ones.** The floor of record is **−0.84pp** and the ceiling
**+2.25pp**. The previously quoted −1.55pp floor built its random subsets from nothing while
the search trims the deployed keep set, and 0.69pp of the apparent gap was that difference in
construction rather than in feature choice. Anything scored against −1.55pp will look better
than it is.

Two cautions carried forward, both still live:

- It is about *which* features, not *how many* — a method that just prunes harder loses.
- A shallow label-guided search already recovers +0.69pp of the room (measured against the old
  floor and due a re-read against the matched one), so the remainder needs depth. Anything
  recovering half the ceiling is inside its own error bar.

And one new caution, which is the main lesson of Step 221:

- **What failed is the *shape* "score each feature on its own, keep the top k".** The true
  correlation has that shape and lands on the floor. Any candidate with the same shape inherits
  the risk, including DUFS's gates. It does not condemn them — DUFS's gates are trained jointly
  on the sample graph and can express redundancy in a way a marginal statistic cannot — but a
  per-feature score is not automatically a different kind of object just because it is computed
  differently.

---

## What Step 221 actually found, in one paragraph

The correlation with correctness **identifies** the good features — its keep-set overlaps them
+0.11 above a null matched on keep-set composition, 20W/4L, p < 1e-4 — and **none of that
converts into performance**: +0.08pp over a matched floor, p = 0.62. So the good features are
not distinguished by being individually well-correlated with correctness (they are somewhat
above average, about a third of the way from random to the maximum, but that is not what makes
the set good). Whatever makes them work is a property of the **set**, and it survives being
partially recognisable one feature at a time without being reachable that way.

---

## The next step — the ranker search

Bracha's **first** proposal is what survives, and the pre-registered form of it is unchanged
apart from the bar. Treat the held-out good feature sets as a target and score a menu of
**label-free per-feature statistics** as rankers of it:

- the **DUFS gate value** (Bracha's first proposal, directly) — already implemented in
  `spectral_utils/selectors/a2_groupfs.py`, standalone entry `scripts/nonmono_v2/dufs_pf.py`,
  already benchmarked at 0.7687
- leverage on the top principal directions of the feature covariance
- mean absolute correlation to the rest of the pool
- L-SML cluster membership and cluster size
- residual under the additive pair fit
- the estimated correlation itself, as the known-**below**-chance control
- the true correlation, as the known-at-floor control — it is a ceiling on the whole marginal
  family, and Step 221 puts that ceiling on the floor

**Pre-register the menu before scoring.** A best-of-seven search over rankers is exactly the
researcher-degrees-of-freedom problem that has bitten this project before.

**Score each one twice**, because Step 221 showed the two can disagree completely:
1. **held-out performance**, against the matched floor **−0.84pp** and ceiling **+2.25pp**,
   induced by pruning the deployed keep set to the good set's own size
2. **overlap with the good sets**, against a null matched on keep-set composition — not a
   uniform-over-pool null, which is too easy for anything that lives inside the keep set

A ranker that only clears the overlap test has reproduced Step 221's outcome, not beaten it.

`scripts/upcr_study/exp13_incumbent_anchored_ranking.py` is the harness: it already replays
exp12's splits exactly, prunes the incumbent by an arbitrary per-feature statistic
(`prune_incumbent`), computes the matched floor, and computes the conditional overlap null
(`cond_overlap_null`). Adding a ranker is one entry in the `for name, stat in (...)` loop.

**Before building anything on top of a ranker, price it.** The standing rule applies to DUFS
as much as it applied to the sign work: run the gate ranking through the harness first and see
what it is worth, then decide whether to build a U-PCR-aligned version of it.

### If the whole menu lands on the floor — THIS IS WHAT HAPPENED (Step 222)

That is a publishable outcome, not a failure. It would mean the room is **not reachable by any
per-feature ranking** — the true correlation already establishes that the strongest label-based
marginal statistic cannot reach it — and the item closes with a ceiling *and* an impossibility
statement, which is a stronger deliverable than a variant that merely loses. The natural
follow-on would then be set-level rather than per-feature: what property of the *set* the
search is exploiting, measured directly.

**Step 222 result, and the four things a follow-on must carry.**

1. **The sharpest evidence is the redundancy arm**, not the losing ones. It is the label-free
   statistic that *most* identifies the good features (overlap +0.036 over the null, 17W/7L,
   bootstrap CI [+0.001, +0.071]) and the *worst* performer of the eight (−3.13pp, Holm 0.002,
   19 of 24 cells negative). Identification without conversion, now with a label-free statistic
   instead of an oracle one. Quote this, not "the menu lost".
2. **The set-level arm did not escape the shape.** Cluster round-robin is the only non-marginal
   arm and the only one above the floor (+0.23pp), but it is the arm *closest to the null* on the
   overlap test (−0.00, p=0.92), and across the six label-free arms |overlap excess| against
   performance has Spearman −0.71 — the nearer an arm is to random, the better it scores against
   a floor that *is* random. A future set-level method must beat the floor **while** separating
   from the null, or it is reproducing this.
3. **The room's denominator is not construction-matched.** The good set lives 81.3% inside the
   deployed keep set; every pruning arm is confined to it (99.85%). About a fifth of the target
   is unreachable by a pruning arm at all. Either the follow-on is allowed to reach outside the
   keep set, or "% of the room" needs restating against a reachable target.
4. **Two arms are partly a coin flip.** L-SML cluster size takes only 4.75 distinct values over a
   pool of 28.4, so the cut falls inside a tied block of 4–6 ordered by the random tie-break; the
   round-robin shares that stream. Their numbers price "a coarse partition plus a tie-break".

The pre-registration is the module docstring of `scripts/upcr_study/exp14_ranker_menu.py`;
results in `results/upcr_study/14_ranker_menu/`.

---

## Step 223 — the joint-selection library, and why the step is a screen and not a build

A 12-paper deep dive (2026-08-04, agent-run) proposed four theoretical directions to escape the
marginal trap, with three top recommendations: **DUFS-CAE** (Lindenbaum et al. 2022, nuisance +
correlated), **SEFS** (Lee et al. 2022, correlated gates), **VICReg** (Bardes et al. 2022,
covariance term). PDFs are in `papers/`. The plan below adopts the *reading* and rejects the
*sequencing*: none of the three gets built until its objective is priced.

### The three top recommendations are one family, and it is the family our data ranks last

DUFS-CAE's correlation penalty, SEFS's correlated gates, VICReg's covariance term and Barlow
Twins' redundancy reduction are four spellings of one criterion: **do not co-select correlated
features.** Step 222 measured the marginal version of exactly that criterion — mean |correlation|
to the rest of the pool — and it is **−3.13pp, the worst of eight arms, Holm p = 0.002, negative
on 19 of 24 cells**, while simultaneously being the statistic that *most* identifies the good
features. That is a strong negative prior on the whole family and it must be carried into any
build decision.

It does not close the family. The Step-222 arm scored each feature against the *whole fixed pool*;
the papers penalise correlation *within the selected set*, which is a genuine set function. But
"we implemented it jointly" is not a reason to expect the sign to flip, and this project's
standing rule is to price the channel first.

**The mechanistic reason to expect the sign we measured.** U-PCR is a **consensus** estimator: it
infers each view's reliability from the *inter-view covariance* under conditional independence
given the latent. The correlation between views **is** the estimator's signal channel.
Decorrelation-based feature selection was designed for reconstruction and representation
objectives, where correlated inputs are wasted capacity. Applied to a consensus estimator it
removes the structure the estimator reads. This predicts the sign of the redundancy result, and it
predicts that every covariance-penalty method underperforms here **unless the penalty is applied
to the correlation the consensus factor does not explain**.

**That last clause is the one genuinely new idea to take from the library, and it comes from the
advisors' own paper.** Lindenbaum et al. 2022 separates *nuisance* from *correlated*. For us the
nuisance is correlation **orthogonal to the U-PCR factor** — shared variation that is not the
latent the fusion is reading. So the repaired objective is the off-diagonal mass of
`Corr(residual after projecting out U-PCR's fitted factor)`, not of `Corr(V_S)`. Same family,
different matrix, opposite prediction. It is registered as an arm below.

### ℓ0-CCA is the one recommendation outside that family — and it should be re-framed

`papers/digests/l0-based-sparse-canonical-correlation-analysis.md` (advisors' own: Lindenbaum,
Salhov, **Averbuch**, Kluger). Its criterion is **shared structure across two measurement
channels**, not decorrelation and not correlation-with-correctness. That is *aligned* with the
consensus mechanism rather than against it, which is why it survives the Step 221/222 closures
where the redundancy family does not. Channel split for our pool, pre-registered: **X = the 16
entropy-trace views** (`epr` … `cusum_shift_idx`), **Y = the 14 spilled-energy + token-logprob
views** (`epr_spilled` … `topk_tail_mass`). **Split by feature NAME, never by index** — pool size
varies across the grid (19 / 27 / 28 / 29 / 30 features on the 24 in-scope cells), so a positional
split silently mis-assigns channels on every short cell. Measured on the 24 in-scope cells the
split runs X = 14–16 / Y = 13–14, **except `seiclr_triviaqa_opt30b` at X = 5 / Y = 14** — the
3-token-answer cell whose spectral views do not exist (Step 216). That cell reports the objective
as NaN and is excluded from this arm's pairing, counted rather than nan-meaned away.

**One trap registered now, before the run.** The total correlation
`trace(Ĉy^{-1/2} Ĉyx Ĉx^{-1} Ĉxy Ĉy^{-1/2})` is the sum of squared canonical correlations, so its
maximum is `min(|S∩X|, |S∩Y|)`. Candidates that happen to split their k features evenly across the
two channels score higher **for that reason alone**, and the arm would then be measuring channel
balance rather than shared structure. The registered form is therefore the **mean** squared
canonical correlation — trace divided by `min(|S∩X|, |S∩Y|)` — and the raw trace is reported
beside it as a diagnostic, together with Spearman(`J`, held-out AUROC) partialled on
`min(|S∩X|, |S∩Y|)`. If the two forms disagree, the normalised one is the registered answer.

Its readout, though, is still `return the s features with largest µ_i` — a per-feature top-k, the
shape Step 222 closed. Its escape is in how the statistic is *computed*, not how it is read.

### D = 30 inverts the regime every one of the 12 papers was built for

All 12 solve **optimisation** in D ≫ N or D in the thousands, where the subset space cannot be
searched and a differentiable relaxation is the only way in. We have **D ≤ 30, an incumbent keep
set of ~21, and a target size k ≈ 11.8** → C(21,12) ≈ 2.9e5 subsets, and a `fit_cols` U-PCR fit
costs **~10 ms** (measured, 24 cells). The subset space is enumerable in under an hour and
samplable in seconds.

**So we do not need stochastic gates, concrete relaxations, or DPP MAP inference.** The binding
constraint is not the optimiser. It is that **we have no label-free set-level objective known to
rank feature subsets by held-out AUROC** — Step 220 already established that U-PCR's own
model-selection criterion is not one. Every paper in the library supplies an optimiser for an
objective; none supplies an objective validated against our target.

### THE STEP: price the objectives, not the optimisers

One experiment prices all 12 papers, because each reduces to a **set-level scalar `J(S)`**
computable on half A with no labels and no training. Build the optimiser only for an objective
that survives.

`scripts/upcr_study/exp15_objective_screen.py`, pre-registration in its module docstring.
Splits are exp12's, replayed through exp13's machinery exactly as exp14 did (same per-cell crc32,
same replayed consumption, all new randomness on a separate generator, per-split reproduction
asserted to 1e-9 against `13_.../splits.csv`).

**Per split**: take the incumbent keep set `start_a` (~21) and the good set's own size k. Draw a
**candidate population** of size-k subsets and score each one twice — held-out AUROC on half B
(`fit_cols(cb, S, exclusion=False)`), and every objective on half A alone.

Population, fixed in advance — 500 per split:
- **300 drawn uniformly from the keep set** — matched to every Step-222 arm's construction
- **200 drawn from the full pool** — so the 18.7% of the good set that lives *outside* the keep
  set is reachable, which is Step 222's caveat 3 fixed rather than restated
- plus the six Step-222 arm selections and the good set itself, so the old arms and the target sit
  inside the same population as anchors

Every candidate has the same size k within a split, so **size is controlled by construction** —
the Step-220 r5 lesson (partial out `n_keep` or measure nothing) is handled by the design, not by
a partial correlation.

**Pre-registered objective menu, directions fixed here.** `V_S` = half-A z-scored derived-polarity
columns of S.

| arm | `J(S)` | direction | source |
|---|---|---|---|
| `vicreg_cov` | mean squared off-diagonal of `Corr(V_S)` | LOWER | VICReg covariance term / Barlow Twins / min-redundant subspace |
| `nuisance_cov` | same, on the residual after projecting out U-PCR's fitted factor | LOWER | **the repair** — Lindenbaum 2022 nuisance-vs-correlated, adapted |
| `l0cca_totalcorr` | `trace(Ĉy^{-1/2} Ĉyx Ĉx^{-1} Ĉxy Ĉy^{-1/2})` over S∩X vs S∩Y, ridge γI | HIGHER | ℓ0-CCA's own loss, no gates trained |
| `recon_pool` | ridge reconstruction error of the full pool from `V_S` | LOWER | SEFS / concrete AE / fractal AE |
| `dpp_logdet` | `log det L_S`, `L = diag(q) Corr diag(q)`, q = DUFS gate | HIGHER | DPP / nsDPP quality × diversity |
| `manifold_kmedoids` | k-medoids cost of S on the feature diffusion manifold | LOWER | Laufer-Goldshtein & Talmon few-sample feature manifold |
| `eff_rank` | participation ratio of `Corr(V_S)` eigenvalues | HIGHER | VICReg variance term, non-degenerate form |
| `upcr_res` | Eq. 20 residual | HIGHER | **control**, known dead (Step 220) |
| `mean_rho_hat` | mean \|ρ̂\| over S | HIGHER | **control**, known below chance as a ranker |
| `random` | i.i.d. noise per candidate | — | **null**, gives the empirical Spearman floor at this population size |
| `truecorr_mean` | mean \|corr(f, y)\| on half A | HIGHER | **oracle**, the marginal ceiling as a set score |
| `insample_auroc` | AUROC of the fused set on half A | HIGHER | **oracle, and the power gate** |

**Endpoints and decision rule, fixed before the run:**

- **PRIMARY** — Spearman(`J`, held-out AUROC) over the candidate population, per split, averaged
  to a per-cell value, then **paired over the 24 cells** with a bootstrap CI. Holm–Bonferroni over
  the **seven** label-free arms. Controls, oracles and the null are outside the family.
- **POWER GATE, checked first** — `insample_auroc` must reach mean Spearman **≥ +0.20 with a CI
  excluding zero**, or the run is declared underpowered and **no negative conclusion is drawn**.
  If half-A labels cannot rank half-B performance, nothing label-free can, and the screen is
  uninformative rather than decisive. This is the premise-test rule from Step 213.
- **PASS** — mean Spearman ≥ +0.20, CI excluding zero, Holm p < 0.05. The threshold is not
  arbitrary: Step 220 found **every magnitude under 0.16** for U-PCR's own model-selection
  criterion against performance, and called that not-a-ranking. An arm has to clear the thing we
  already declared dead. (Different population — Step 220 swept constants, this sweeps subsets —
  so 0.16 is the reference point, not a re-used measurement.)
- **SECONDARY** — for every arm, the held-out AUROC of the subset it actually **selects**
  (argmax/argmin over the population), reported against the matched floor **−0.84pp** and the room
  **+2.25pp**. A passing objective converts to a deployable number in the same run; a failing one
  is priced on the same scale as Steps 221–222.
- **RESTRICTED SECONDARY** — Spearman recomputed on the top quartile of candidates by held-out
  AUROC. An objective can order random subsets well and fail to discriminate among good ones; that
  is the discrimination that matters and it is reported separately, not blended in.
- **READING, fixed now so it cannot be renegotiated**: if every label-free objective fails while
  the power gate passes, the impossibility statement extends **from per-feature to set-level**, and
  the item closes having priced a 12-paper library with one number. That is the stronger
  deliverable, exactly as Step 222's was.

**Cost**: 24 cells × 5 splits × 500 candidates × ~10 ms ≈ **10 minutes of fits** plus objective
evaluation. One local run, no cluster.

### What gets built after the screen, conditional on it

- **`l0cca_totalcorr` passes** → build ℓ0-CCA properly: STG gates on the total-correlation loss,
  reusing `_train_dufs`'s gate code in [a2_groupfs.py:303](../../../spectral_utils/selectors/a2_groupfs.py#L303)
  with the loss swapped. λ tuned label-free by held-out total correlation, per the paper's own
  procedure; σ = 0.25 per the paper, not our DUFS σ = 0.5. Score through exp14's harness on both
  endpoints like every other arm.
- **`nuisance_cov` passes** → no network needed. Greedy or exhaustive descent on `J` at D = 30,
  and the Lindenbaum-2022 paper becomes the citation for the objective rather than the source of
  the optimiser.
- **`vicreg_cov` / `recon_pool` / `dpp_logdet` pass** → same: descend them directly. The
  differentiable machinery in those papers exists to make D = 10⁴ tractable and buys us nothing.
- **Nothing passes** → close the item, and the follow-on is not another selector. It is the
  question of whether the good set is characterisable at all from half A, which the power gate
  will already have answered on the label-*ful* side.

### Carried into the screen from Step 222 — do not drop these

The four things a follow-on must carry are honoured as follows: caveat 1 (identification without
conversion) is why the primary is performance and not overlap; caveat 2 (an arm near the null
scores well against a floor that *is* random) is why the null arm is in the menu and why the
restricted secondary exists; caveat 3 (the unreachable fifth of the target) is why 200 of the 500
candidates are drawn from the full pool; caveat 4 (tied blocks and coin-flip tie-breaks) does not
apply — every objective here is continuous over subsets.

---

## Preserved — the re-run rule

**This must not be lost.**

Step 218 pre-registered a mapping from selector quality to which feature *transform* is
correct: `mode_centre` is right at today's selector precision of 0.562, `squared` overtakes it
above **0.654**, `abs_rank` above 0.715, `dist_median` above 0.549.

So **if the feature selection improves past that line, the transform of record changes** — and
the reshaped feature matrix changes with it, which makes every performance number computed on
the old matrix stale.

**The rule: one re-derivation round, round two only, never iterate to convergence.** Which runs
repeat is decided by which upstream quantity moved, not by which result disappointed. The
round-two number is the reported number — never the better of the two. If a third round would
be triggered, **stop and report the oscillation**. `scripts/nonmono_v2/repick_transforms.py`
re-derives the picks without repeating the full sweep.

| when this moves | what must be re-run |
|---|---|
| selector precision | the transform of record → the reshaped matrix → every performance number on it |
| the keep rule | the surviving feature set → anything measured on it |
| the pseudo-label | the estimated base rate, and the registered selector → the benchmark |

---

## Standing rules that earned their place

**Price the channel before building in it.** Two proposals have now been closed by pricing
rather than by attempting them. Before any new line of work: *what is the oracle worth here?*

**Check the null before trusting that clearing it means something — and check that it is the
right null.** Step 220's null (selection on shuffled labels) turned out to be *easier* than
chance. Step 221's overlap null drew uniformly from the pool while both rankings lived inside
U-PCR's keep set, which flattered them; the correction turned "at chance" into "below chance".
A null must match the arm's own construction, not just its size.

**Match the floor to the arm's construction.** 0.69pp of the reported headroom was the
difference between rebuilding a set and trimming the deployed one, not between good and random
features.

**Sensitivity is not headroom.** That the pipeline *reacts* to an input says nothing about
whether the current value of that input is leaving anything on the table.

**A review pass before every checkpoint.** Two agents in parallel on different objects — one
reads the diff against the code, one reads the results against the pre-registration. Between
them across Steps 220–221 they have found a search stopping early on 17 of 24 test sets, a null
easier than chance, a missing ceiling that overturned a section, a confounded comparison, a
too-easy null, and a headline diagnostic that was true by arithmetic. Every finding must carry
a `file:line` or a CSV cell; unreferenced findings are discarded. **The agent flags, it never
decides.**

**Put an interval on every ceiling.** **Paired statistics over test sets, never pooled.**

---

## Checkpoints

Stop and discuss after the **ranker menu is scored** (done — Step 222), after the **objective
screen is scored** (Step 223), and again before anything is adopted into the deployed path. A
review pass runs before each.

**Standing rule this plan adds, earned from the 12-paper dive**: when a literature review
recommends a *method*, extract its **objective** and price that first. A method is an optimiser
plus an objective plus a regime; ours is the regime the optimiser was not built for (D = 30,
N ≫ D), so the objective is the only part that transfers. Two of the three top recommendations
would have cost weeks to build and are three spellings of a criterion our own data already ranks
last.
