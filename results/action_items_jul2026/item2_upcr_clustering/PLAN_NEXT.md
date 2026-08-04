# The plan from here — after the feature-selection question was answered

Updated at **Step 222**. The July clustering plan closed on ceilings at Step 220; this file
replaced it. Step 221 closed one of Bracha's two proposals with a number, and Step 222 priced
the other one and with it the whole per-feature family. Read `PHASE1_RESULTS.md` for the
measurements.

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

Stop and discuss after the **ranker menu is scored**, and again before anything is adopted into
the deployed path. A review pass runs before each.
