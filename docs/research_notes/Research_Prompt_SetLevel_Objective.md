# Prompt for Gemini — find us a label-free *set-level objective*, not another selector

**Paste everything below the line into Gemini. Use its sub-agents for parallel web search.**

---

## Read this first: your last recommendation was aimed at the wrong problem

You ran a deep dive on unsupervised/self-supervised feature selection and came back with four
directions and three top picks: **DUFS-CAE** (Lindenbaum et al. 2022, nuisance + correlated),
**SEFS** (Lee et al. 2022, correlated gates), **VICReg** (Bardes et al. 2022, covariance term).

Two corrections, both backed by measurements we already have:

**1. All three are one criterion — "do not co-select correlated features" — and that is the
criterion our data ranks dead last.** DUFS-CAE's correlation penalty, SEFS's correlated gates,
VICReg's covariance term and Barlow Twins' redundancy reduction are four spellings of the same
thing. We measured the marginal version of exactly that criterion on our 24 test sets: ranking
features by mean |correlation| to the rest of the pool and keeping the least redundant scores
**−3.13pp against a matched random-pruning floor** — the worst of eight pre-registered arms,
Holm-adjusted p = 0.002, negative on 19 of 24 test sets.

There is a mechanistic reason to expect that sign, and you need to internalise it before searching:
**our fusion is a consensus estimator.** It infers each view's reliability from the *inter-view
covariance matrix* under a conditional-independence-given-the-latent assumption (Spectral Meta-Learner
/ Dawid-Skene family). The correlation between views **is** the estimator's signal channel.
Decorrelation-based feature selection was designed for reconstruction and representation objectives,
where correlated inputs are wasted capacity. Applied to a consensus estimator it deletes the
structure the estimator reads. **Do not bring us more anti-redundancy methods.**

**2. You recommended optimisers, and optimisation is not our bottleneck.** Every paper you named
solves the *search* problem at D ≫ N or D in the thousands, where the subset space is intractable
and a differentiable relaxation is the only way in. Our regime is inverted: **D ≤ 30 features,
N = 200–8,500 rows, target subset size ≈ 12, and one full model fit costs ~10 ms.** The candidate
space is C(21,12) ≈ 2.9e5 — enumerable in under an hour, samplable in seconds. Stochastic gates,
concrete relaxations and DPP MAP inference buy us nothing.

**What we lack is an objective, not a way to descend one.**

---

## What our problem actually is

**Setup.** We detect LLM hallucinations without labels. From a single generation we compute ~30
scalar "views" of the answer (spectral statistics of the token-entropy trace, spilled-energy
statistics, token-logprob statistics). Each view is a weak, noisy, unsigned predictor of whether the
answer is correct. We fuse them with an unsupervised consensus estimator in the **Spectral
Meta-Learner / unsupervised-ensemble** family — it estimates each view's reliability from the
off-diagonal structure of the view-by-view covariance matrix, then takes a reliability-weighted
combination. Output is one score per answer; we measure **AUROC against correctness** on 24
(dataset × model) test sets. Everything must be **label-free at fit time**.

**Where the headroom is.** We priced every channel inside the estimator by letting labels do it
perfectly. All but one are empty: the weighting blend is worth +0.19pp, getting every view's sign
right is worth −0.06pp, the constants are inert, the estimator's own model-selection criterion does
not rank feature sets at all. **The only channel with room is which features get kept: +2.25pp
held out, CI [+1.53, +3.04], winning on 23 of 24 test sets**, against a floor that prunes the
deployed keep-set at random (−0.84pp).

**What is closed, with numbers. Do not propose anything that reduces to these.**

| closed | the number that closed it |
|---|---|
| Ranking features by correlation with correctness, in **any** form | The *true* correlation, given the good set's own size, is worth **+0.08pp, p = 0.62**. The quantity is the dead end, not our estimate of it. |
| Improving how we estimate that correlation | A *perfect* estimate spent on selection is worth **+0.34pp, CI [−0.47, +1.30], p = 0.88**. |
| Ranking by **any** label-free per-feature statistic | Eight pre-registered arms, none clears the floor: DUFS gate value −0.70pp, principal-direction leverage −0.92pp, additive pair-fit residual −0.09pp, cluster round-robin +0.23pp (n.s.), L-SML cluster size −1.61pp, redundancy-to-pool −3.13pp. |
| Anti-redundancy / diversity selection | see above, −3.13pp, the worst arm, and the one that *best* identifies the good features. |
| Changing which views are in the pool | measured, negative. |
| All sign/orientation work | −0.06pp; 17 of 24 test sets unchanged to the digit. |
| Descending the estimator's own criterion | it does not rank feature sets by performance — every correlation magnitude under 0.16, sign flips with what you control for. |

**The shape that failed is "score each feature on its own, keep the top k".** The good feature sets
are good because of a property of the **set**. That property survives being partly recognisable one
feature at a time — the redundancy statistic *identifies* the good features better than anything else
we tried — without being reachable that way.

---

## What we are asking you to find

**One thing: papers that supply a label-free, set-level scalar `J(S)` that plausibly ranks feature
subsets by the held-out accuracy of an unsupervised consensus / latent-variable fusion.**

Not a selector. Not an optimiser. Not a relaxation. **An objective**, that we can compute on a
candidate subset and correlate against held-out AUROC. We already have the harness to score any
`J(S)` you find, on 24 real test sets, in ten minutes.

`J(S)` must be:
- computable from an `n × |S|` real matrix of view values, **with no labels**
- cheap, or at worst cheap to train — `|S| ≈ 12`, `D ≤ 30`, `n` from 200 to 8,500
- **not** a decorrelation/diversity criterion, and **not** a per-feature score summed over `S`
- ideally motivated by *whether the latent-variable / conditional-independence model that the fusion
  assumes actually holds on `S`* — that is our best guess at what makes a set good, and it is the
  one thing nobody has measured for us

### Literature to search, in priority order

These are leads, not answers. Search each; report what exists.

1. **Unsupervised ensemble learning / crowdsourcing** — Dawid-Skene, Spectral Meta-Learner, Parisi et
   al., Jaffe/Nadler, restricted-likelihood and tensor-decomposition variants. Specifically: **model
   selection and goodness-of-fit within this family.** Which classifiers to include, how to detect
   violations of conditional independence, how to choose the number of latent factors, how to test
   whether a given subset of predictors is well-explained by a single latent. This is the closest
   literature to our problem and the least likely to have been mined.
2. **Pure measurement models in SEM / causal discovery** — vanishing **tetrad constraints**,
   BuildPureClusters, FindOneFactorClusters, Silva & Scheines, Kummerfeld & Ramsey. The question
   *"which subset of measured variables are pure indicators of a single latent factor"* is
   **literally our question**, asked in a different field, and it is answered by a label-free
   rank/tetrad test on the covariance matrix. Search this hard.
3. **Unsupervised evaluation / model selection without labels** — "estimating classifier accuracy
   without labels", agreement-based validation, Platanios et al., Steinhardt & Liang, Jaffe et al.
   Any scalar that scores an ensemble's internal consistency.
4. **Weak supervision / data programming** — Snorkel, FlyingSquid, Ratner et al., dependency-structure
   learning among labelling functions. They have a genuine *"which labelling functions to include"*
   problem and answer it with generative-model likelihood, not with diversity.
5. **Factor-analytic variable selection** — which variables load cleanly on the common factor,
   parallel analysis, minimum-rank factor analysis, Ledermann bound, "sufficient" indicator sets.
6. **Latent tree / latent graphical model structure learning** — which observed variables attach to
   which latent, and goodness-of-fit tests for that attachment.

### Rules for what you return

- **No fabrication.** For every paper: give the exact title, authors, venue, year, and a **verbatim
  quote** of the sentence or equation that defines the objective, with its section or equation
  number. If you cannot quote it, mark the row `UNVERIFIED` and say so. Two of your previous
  literature backfills for this project contained invented authors, venues, datasets and scores;
  every row will be spot-checked.
- **No re-recommendations of the closed families above.** If a paper's core criterion is
  decorrelation, diversity, redundancy-minimisation, or a per-feature score, do not include it —
  unless you can state precisely why it is not one of those, in which case say so explicitly.
- **Skip papers whose contribution is scalability or a differentiable relaxation.** We have D = 30.
- Prefer papers where the objective can be lifted out and computed standalone.

### Output format

One table, ranked by how directly the objective addresses "is this subset well-explained by a single
latent":

| paper | venue / year | the objective `J(S)`, as a formula | inputs it needs | label-free? | why it is not one of our closed families | why it might rank subsets by consensus-fusion accuracy | verbatim quote + location |

Then a short section: **the three you would actually have us compute first, and why** — judged on
whether the objective is a *different kind of quantity* from what we have already priced at zero, not
on how well the paper performed on its own benchmarks.

Then: **anything in this space that says our goal is impossible.** A theorem that a set's fusion
accuracy is not identifiable from unlabelled data would be as valuable to us as a method, and we
would rather learn it from the literature than from another six weeks of experiments.
