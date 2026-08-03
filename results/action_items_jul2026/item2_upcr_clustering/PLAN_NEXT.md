# The plan from here — pruned after the ceiling measurements

This replaces the July clustering plan. That plan had four places a clustering stage could go;
the ceiling measurements priced three of them at zero. What is left is not a clustering
question — it is a **feature-selection** question. Read `PHASE1_RESULTS.md` first.

---

## Settled. Do not re-open any of these.

Each of these was measured, not argued. If a future idea lands on one of them, the answer is
already a number.

| | why it is closed |
|---|---|
| **Clustering the surviving features to improve the weights** | The best possible blend of the two principal directions is worth **+0.19pp held out, p = 0.57**. That is the ceiling for every "estimate the weights better" idea. |
| **All sign / direction work** | Giving every feature its *correct* direction is worth **−0.06pp**, with 17 of 24 test sets unchanged to the digit. The channel is empty. The diagnostic (direction errors concentrate on non-monotone features) is true and does not matter. |
| **Anything that descends U-PCR's own model-selection criterion** | The criterion does not rank feature sets by performance — the correlation's sign flips depending on what you control for and every magnitude is under 0.16. |
| **Tuning the label-variance constant** | Inert three separate ways. Note for anyone tempted: only the *ratio* to the average feature variance is identifiable, and for z-scored features the assumption-faithful ratio is 1.0, not p(1−p). |
| **Keeping fewer features by U-PCR's own ranking** | Loses at every size, −1.49pp at 6 features through −0.28pp at 16. |
| **A fixed feature set shared across test sets** | The good sets do not transfer: −0.81pp at matched size, −2.37pp at ten features. |

Earlier closures that still stand: pool composition (Step 206), non-monotone reshaping as a
deployable gain (Step 218), synchronising the global sign (Step 213), and the clustered
pair-fit variant (Step 204).

---

## The one live question

**About +1.5 points sits in which features get kept** — held out, interval [+0.97, +2.03].
The good sets are half the size of what we keep now, smaller on every test set, and **at chance
with respect to U-PCR's own ranking** (overlap 0.340 against a random baseline of 0.360).

So: **what separates the good features, if not their estimated correlation with correctness?**

Two cautions to carry forward. Random subsets of the same size lose 1.55pp, so this is about
*which* features, not *how many* — a method that just prunes harder will lose. And a shallow
label-guided search already recovers +0.69pp of the +1.48pp, so the remainder needs depth;
anything recovering half the ceiling is inside its own error bar.

---

## Bracha's suggestion is the spine of what comes next

From the July meeting, verbatim:

> *"I also wonder whether the DUFS feature selection method could be integrated more naturally.
> Since both U-PCR and DUFS are spectral approaches, it may be possible to align the DUFS
> objective with the U-PCR formulation for feature importance or maybe use the differential
> learning mechanism of DUFS to improve the parameter estimation of U-PCR (not necessarily with
> respect to feature selection)."*

That is two distinct proposals, and the measurements above make them testable against each
other rather than a matter of taste.

**Her first proposal — DUFS decides which features matter.** We need a feature ranking that is
not the estimated correlation, and DUFS supplies exactly that: continuous gates trained on the
sample-graph geometry rather than on agreement with correctness. It is already implemented
(`spectral_utils/selectors/a2_groupfs.py`) and already benchmarked (0.7687).

**Her second proposal — DUFS's gradient machinery fixes U-PCR's estimation.** U-PCR estimates
each feature's correlation with correctness by solving one equation per *pair* of features,
assuming every pair obeys its model. Our features are all computed from the same token trace,
so many pairs violate it badly and unevenly. Her idea reads as: learn weights on the pairs by
gradient descent, down-weight the violating ones, and solve a corrected system. Nobody has
tried this. The one hard version we tried — deleting same-cluster pairs outright — lost 4.46pp
(Step 204); a smooth learned version is a genuinely different object.

### Step 1 — the test that decides which of the two to build

**Rank each test set's features by their *true* covariance with correctness, take the good
set's own size, score held-out. Does that ranking reproduce the good feature set?**

- **If yes** → the correlation is the right quantity and we merely estimate it badly. Her
  *second* proposal is then the whole game: fix the estimation, and the better feature set
  follows for free.
- **If no** → the correlation is the wrong quantity, no estimator of it can help however good,
  and her *first* proposal is the answer.

Cheap, decisive, and it uses labels so it is a ceiling and not a method. Run it before
building either.

### Step 2a — if the correlation is the wrong quantity: the ranker search

Treat the held-out good feature sets as a target (682 features, ~243 chosen). Pre-register a
menu of **label-free per-feature statistics** and score each as a ranker of that target:

- the **DUFS gate value** (Bracha's first proposal, directly)
- leverage on the top principal directions of the feature covariance
- mean absolute correlation to the rest of the pool
- L-SML cluster membership and cluster size
- residual under the additive pair fit
- the estimated correlation itself, as the known-at-chance control

Then induce each statistic's keep-set at the good set's own size and score it held-out, against
a floor of **−1.55pp** (random subsets of the same size) and a ceiling of **+1.48pp**.

Pre-register the menu before scoring — a best-of-seven search over rankers is exactly the
researcher-degrees-of-freedom problem that has bitten this project before.

### Step 2b — if the correlation is the right quantity: the differentiable pair reweighting

Replace the plain least-squares solve over all feature pairs with a learned weighting. Gate it
on Step 1, because if the correlation is the wrong quantity this cannot pay however well built.

**Both outcomes of Step 1 are publishable.** If nothing in the ranker menu beats random
overlap, the room is not reachable without labels, and the item closes with a ceiling *and* an
impossibility statement — a stronger deliverable than any variant that merely loses.

---

## Preserved from the old plan — the re-run rule

**This is the thing we said we would come back to, and it must not be lost.**

Step 218 pre-registered a mapping from selector quality to which feature *transform* is
correct: `mode_centre` is right at today's selector precision of 0.562, `squared` overtakes it
above **0.654**, `abs_rank` above 0.715, `dist_median` above 0.549.

So **if the feature selection improves past that line, the transform of record changes** — and
the reshaped feature matrix changes with it, which makes every performance number computed on
the old matrix stale. That is not a judgement call; the mapping is already written down and
just needs evaluating.

**The rule: one re-derivation round, round two only, never iterate to convergence.** Which
runs repeat is decided by which upstream quantity moved, not by which result disappointed. The
round-two number is the reported number — never the better of the two. If a third round would
be triggered, **stop and report the oscillation**; a pipeline that does not settle after one
pass is itself a finding. `scripts/nonmono_v2/repick_transforms.py` exists specifically to
re-derive the picks without repeating the full sweep.

What goes stale when what moves:

| when this moves | what must be re-run |
|---|---|
| selector precision | the transform of record → the reshaped matrix → every performance number on it |
| the keep rule | the surviving feature set → anything measured on it |
| the pseudo-label | the estimated base rate, and the registered selector → the benchmark |

---

## Standing rules that earned their place

**Price the channel before building in it.** Every hour of the sign work was spent on a channel
worth −0.06pp, and one script would have shown that up front. The old plan applied this
rigorously to three channels and simply never asked about the fourth. **Before any new line of
work: what is the oracle worth here?**

**Sensitivity is not headroom.** The sign line was justified by showing the pipeline *reacts*
to input directions. It does. That says nothing about whether the current directions are
leaving anything on the table.

**A review pass before every checkpoint.** Two agents, run in parallel on different objects —
one reads the diff and asks whether the code does what the write-up says, one reads the results
and the pre-registration and asks whether the conclusions follow. They found a search that was
stopping early on 17 of 24 test sets, two successive confounds in one correlation, a null that
was easier than chance, and a missing ceiling that overturned a whole section. Every finding
must carry a file:line or a CSV cell; unreferenced findings are discarded. The agent flags, it
never decides.

**Put an interval on every ceiling.** The headline here was reported for two turns without one.

**Paired statistics over test sets, never pooled**, and a null that is genuinely harder than
chance — check the null's own number against a random baseline before trusting that clearing it
means anything.

---

## Checkpoints

Stop and discuss after **Step 1** (it decides which of Bracha's two proposals to build), and
again after **Step 2**, before anything is adopted into the deployed path. A review pass runs
before each.
