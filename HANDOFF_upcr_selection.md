# Handoff — U-PCR feature selection, picking up from Step 220

Written 2026-08-03. Everything below is committed (`207172d` on `master`). Nothing is in flight.

## Read these three, in this order, and nothing else first

1. `results/action_items_jul2026/item2_upcr_clustering/PLAN_NEXT.md` — what to do and why
2. `results/action_items_jul2026/item2_upcr_clustering/PHASE1_RESULTS.md` — the measurements
3. `HISTORY.md` Step 220

Do **not** read the July clustering plan (`~/.claude/plans/i-wanted-to-work-tidy-wind.md`). It is
947 lines and most of it is settled; `PLAN_NEXT.md` is the pruned version and supersedes it.

## The one-paragraph state

We were going to build a clustering stage inside U-PCR, as the July meeting asked. Before
building, we priced every channel by letting the labels do that step perfectly. **Three of the
four places a clustering stage could go are worth zero.** The fourth — which features get kept —
has about **+1.5 points** in it, but the good feature sets are **half the size** of what we keep
now and are **at chance** with respect to the ranking U-PCR uses. So the clustering line closes on
ceilings, and the live question is: *what separates the good features, if not their estimated
correlation with correctness?*

## The next thing to run

**The true-correlation test.** Rank each test set's features by their *actual* covariance with
correctness (using labels), take the good set's own size, score held out. Does it reproduce the
good feature set?

- **yes** → the correlation is the right quantity, we just estimate it badly → build Bracha's
  *second* proposal (learned per-pair weights on U-PCR's estimation system)
- **no** → the correlation is the wrong quantity entirely → build Bracha's *first* proposal
  (DUFS supplies the ranking)

It is cheap and it decides the branch. `PLAN_NEXT.md` has both branches written out.

Machinery to reuse, not rebuild:
- `scripts/upcr_study/exp11_posthoc_controls.py` already has the held-out scoring protocol, the
  random floor, and the ranking-overlap computation. `run_rho_ranking()` is the closest template.
- `results/upcr_study/10_channel_ceilings/r2_oracle_mask.csv` column `oracle_cols` holds the good
  feature sets per test set — that is the target to rank against. **Use the held-out ones**, not
  the same-rows ones; the same-rows sets are half search-noise and ranking against them would
  teach a statistic to predict overfitting.

## Things that will waste your time if you don't know them

- **The arm of record is `sign(rho)`-oriented, not hand-oriented.** `prepare_cell` has *already*
  applied `ALL_SIGNS`, so fitting `cell["V"]` directly gives the wrong arm and a plausible-looking
  0.75713 instead of 0.7741. Use `derive_cell()` from `exp10_channel_ceilings.py`, and let
  `derived_arm_gate()` assert it. This cost half a day.
- **Step numbering**: 219 belongs to the localization / Extension-F work on the
  `experiment/step-localization` worktree. Master is at 220. Check both before numbering.
- **Two anchor gates must pass on every script**: GOOD_6 at 0.7733 and U-PCR at 0.7741. Check the
  *return value*, not its truthiness — `assert_good6` returns a tuple.
- `scripts/prior_free_bench.py:72` still carries a stale `GOOD6_EXPECTED = 0.7594`. It is not in
  this work's path (nothing here imports it) but fix it if you touch that file.

## Standing rules that earned their place this step

- **Price the channel before building in it.** Every hour of the sign work went into a channel
  worth −0.06pp, and one script would have shown that up front. Before any new line: *what is the
  oracle worth here?*
- **Sensitivity is not headroom.** The sign line was justified by showing the pipeline reacts to
  input directions. It does. That says nothing about whether the current directions cost anything.
- **Put an interval on every ceiling.** The headline was reported for two turns without one.
- **Check a null against a random baseline before trusting it.** Ours (selection on shuffled
  labels) turned out to be *easier* than chance, so clearing it meant nothing.
- **Run a review pass before each checkpoint** — two agents in parallel, one on the diff, one on
  the results plus the pre-registration. Between them this step they found a search stopping early
  on 17 of 24 test sets, a p-value of 5.7e-50 on a statistic that cannot cross its own bound, and
  a missing measurement that overturned an entire section. Every finding must carry a `file:line`
  or a CSV cell; the agent flags, it never decides.

## Do not re-open

Clustering the survivors to remix weights; all sign/direction work; anything descending U-PCR's
model-selection criterion; tuning the label-variance constant; keeping fewer features by U-PCR's
own ranking; a fixed feature set shared across test sets. Each has a number in
`PHASE1_RESULTS.md`. Earlier closures that still stand: pool composition (206), non-monotone
reshaping as a deployable gain (218), global sign synchronisation (213), the clustered pair-fit
(204).

## The thing we said we'd come back to

**If the feature selection improves, the feature *transform* has to be re-derived.** Step 218
pre-registered the mapping: `mode_centre` is right at today's selector precision of 0.562,
`squared` overtakes above 0.654. So a better selector makes the reshaped feature matrix stale, and
every number computed on it with it. **One re-derivation round, round two only, never iterate to
convergence** — if a third round would be triggered, report the oscillation instead.
`scripts/nonmono_v2/repick_transforms.py` re-derives the picks without repeating the full sweep.

## Open item

One Phase-2 number got produced early and ungated: sign-instability as a headroom detector, 0.500
against an incumbent of 0.562. It was scored against a **hard-coded** incumbent without the
reproduction gate that requires reproducing 0.562 / +0.309 / 13-of-61 first. **Not quotable** until
that gate runs. It is in `sign_identifiability.py` under `s3_instability_detector`.

---

### Paste-ready prompt for the next session

> Read `PROGRESS.md`, then
> `results/action_items_jul2026/item2_upcr_clustering/PLAN_NEXT.md` and `PHASE1_RESULTS.md`, then
> `HANDOFF_upcr_selection.md`. Do not read the July clustering plan — it is superseded.
>
> We closed the U-PCR clustering line on ceilings at Step 220. Feature selection is the only
> channel with room (~+1.5pp held out) and the good feature sets are at chance with respect to
> U-PCR's own ranking. The next thing to run is the true-correlation test described in
> `PLAN_NEXT.md` Step 1, which decides which of Bracha's two DUFS proposals to build.
>
> Explain what you are going to run before running it. Do not use code labels like R2 or T4 —
> name each experiment by what it does and what it found. Lead with what matters, not with
> whether a number clears a threshold we picked ourselves.
