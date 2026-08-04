# Handoff — U-PCR feature selection, picking up from Step 221

Written 2026-08-04. **Not committed yet** — two new scripts and two results directories are
untracked; see "State of the tree" at the bottom.

## Read these, in this order, and nothing else first

1. `results/action_items_jul2026/item2_upcr_clustering/PLAN_NEXT.md` — what to do and why
2. `results/action_items_jul2026/item2_upcr_clustering/PHASE1_RESULTS.md` — the measurements
3. `HISTORY.md` Steps 220 and 221

Do **not** read the July clustering plan (`~/.claude/plans/i-wanted-to-work-tidy-wind.md`). It
is 947 lines, settled, and superseded by `PLAN_NEXT.md`.

## The one-paragraph state

We priced every step inside U-PCR by letting the labels do it perfectly. Only **which features
get kept** has room — about **+2.25pp** held out. Step 221 then asked what separates the good
features, and got a two-sided answer: their correlation with correctness **identifies** them
(overlap +0.11 above a matched null, p < 1e-4) and **buys nothing** (+0.08pp over a matched
floor, p = 0.62). Since a *perfect* estimate of that correlation is worth +0.34pp with an
interval crossing zero, and the other two places `rho_hat` feeds were already priced at +0.19pp
and −0.06pp, **improving U-PCR's estimation cannot pay anywhere** — Bracha's second proposal is
closed. Her first, DUFS supplying the ranking, is the live one.

## The ranker menu has been RUN — Step 222. Do not re-run it.

`scripts/upcr_study/exp14_ranker_menu.py` (pre-registration in its module docstring, results in
`results/upcr_study/14_ranker_menu/`). Eight arms, and **none of the six label-free ones clears
the matched floor**. DUFS's gate value — Bracha's first proposal — is −0.70pp, not separable from
the floor after Holm (0.36) and a three-cell effect. Two arms are significantly *worse* than
random pruning: redundancy to the pool −3.13pp (Holm 0.002) and L-SML cluster size −1.61pp
(Holm 0.008). The set-level cluster round-robin is the only arm above the floor (+0.23pp,
Holm 0.53) and is also the arm closest to the overlap null (p=0.92) — across the six,
|overlap excess| vs performance has Spearman −0.71, so being nearest to random is what wins
against a random floor.

**The statement this closes on**: the redundancy statistic *most* identifies the good features
and performs *worst*, which with the true correlation already on the floor means **the +2.25pp is
not reachable by scoring features one at a time.**

## The next thing to run

The follow-on `PLAN_NEXT.md` names: **what property of the *set* the search exploits, measured
directly** — set-level, not another per-feature score. Read the four constraints under
"If the whole menu lands on the floor" in `PLAN_NEXT.md` before designing it; the third one
(the room's denominator is not construction-matched — the good set is only 81.3% inside the keep
set while every pruning arm is confined to it) changes what a follow-on is even allowed to
measure against.

The separately pre-registered line is **ℓ0-CCA** (`papers/l0-based Sparse Canonical Correlation
Analysis.pdf`, Lindenbaum–Salhov–**Averbuch**–Kluger). It is not closed by Step 222: its criterion
is shared structure across measurement channels, not correlation with correctness, and its gates
are trained jointly against a set-level objective. The pre-registered split is the pool's own
construction — X = the 16 entropy-trace spectral views, Y = the 14 spilled-energy + token-logprob
views. Price the closed-form cross-modality leverage (the paper's own initialization, one SVD, no
training) through the Step 222 harness first, and label it a **probe, not a ceiling**. The paper
is not yet cached under `papers/` — do that first per `skills/paper-digest/SKILL.md`.

`scripts/upcr_study/exp14_ranker_menu.py` is the harness now: adding an arm is one entry in the
`orders` dict inside `run_cell`, plus a name in `LABEL_FREE`. It gates against exp13 per split.
DUFS entry points are `spectral_utils.selectors.a2_groupfs.dufs_pf_gates` /
`dufs_pf_cell_rng` (moved there at Step 222 — `scripts/nonmono_v2/dufs_pf.py` cannot be imported
from `scripts/upcr_study/` because both directories have a `common.py`).

## Numbers to use, and the stale ones they replace

| use this | not this | why |
|---|---|---|
| floor **−0.84pp** | −1.55pp | the old floor rebuilt sets from nothing; the search *trims* the deployed keep set. 0.69pp of the gap was construction, not feature choice. |
| room **+2.25pp**, CI [+1.53, +3.04] | +1.48pp | that is the same measurement against the matched floor rather than the deployed pool. Both are correct; quote the matched one when judging a selector. |
| rho ranking **below** chance (−0.05, p = 0.016) | "at chance" (0.340 vs 0.360) | the old null drew uniformly from the pool while 98.3% of the ranking sits inside a keep set that is 73.5% of it. |
| good features are **above**-average marginal strength (0.2932 vs pool 0.2563) | "individually weaker" | the old comparison was against the top-k by the same statistic, which is the maximum by construction — true 24/24 by arithmetic. |

## Things that will waste your time if you don't know them

- **The arm of record is `sign(rho)`-oriented, not hand-oriented.** `prepare_cell` has *already*
  applied `ALL_SIGNS`, so fitting `cell["V"]` directly gives a plausible-looking 0.75713 instead
  of 0.7741. Use `derive_cell()` from `exp10_channel_ceilings.py` and let `derived_arm_gate()`
  assert it.
- **Two anchor gates must pass on every script**: GOOD_6 at 0.7733 and U-PCR at 0.7741. Check
  the *return value*, not its truthiness — `assert_good6` returns a tuple.
- **`r2_oracle_mask.csv`'s `oracle_cols` is the IN-SAMPLE greedy**, written at
  `exp10_channel_ceilings.py:443` from `cols_in`; the split-half greedy at line 405 is scored
  and discarded. The held-out good sets you actually want are persisted as
  `results/upcr_study/12_what_separates_good_features/splits.csv` → `greedy_cols`.
- **Reproducing exp12's splits from a new script**: same per-cell seed
  (`np.random.default_rng(zlib.crc32(cell_key) % 2**32)`) **and** replay its random consumption
  — 2 × 2000 overlap draws + 25 floor draws per completed split — or every later split diverges.
  `exp13` does this and asserts the recovered deployed AUROC matches per split. Use a separate
  generator for anything new.
- **Step numbering**: 219 belongs to the localization / Extension-F work on the
  `experiment/step-localization` worktree. Master is at 221. Check both before numbering.
- `scripts/prior_free_bench.py:72` still carries a stale `GOOD6_EXPECTED = 0.7594`. Nothing in
  this work's path imports it; fix it if you touch that file.

## Standing rules that earned their place

- **Price the channel before building in it.** Two proposals have now been closed by pricing
  rather than by attempting them. Before any new line: *what is the oracle worth here?*
- **Match the floor to the arm's construction.** A floor that builds sets differently from the
  arm is not a floor, it is a second experiment.
- **Check that the null is the right null, not just that one exists.** Step 220's was easier
  than chance; Step 221's overlap null was too easy for anything living inside the keep set.
- **Sensitivity is not headroom.**
- **A review pass before every checkpoint** — two agents in parallel, one on the diff, one on
  the results plus the pre-registration. Across Steps 220–221 they found a search stopping early
  on 17 of 24 test sets, a null easier than chance, a missing ceiling that overturned a section,
  a confounded comparison, a too-easy null, and a headline diagnostic true by arithmetic. Every
  finding must carry a `file:line` or a CSV cell; the agent flags, it never decides.
- **Put an interval on every ceiling. Paired statistics over test sets, never pooled.**

## Do not re-open

Improving U-PCR's estimation of the correlation (any form); ranking features by correlation with
correctness (any estimator — the true value is on the floor); **ranking features by any label-free
per-feature statistic, DUFS gates included (Step 222 — the whole menu is on the floor or below
it)**; clustering the survivors to remix weights; all sign/direction work; anything descending
U-PCR's model-selection criterion; tuning the label-variance constant; keeping fewer features by
U-PCR's own ranking; a fixed feature set shared across test sets. Each has a number in
`PHASE1_RESULTS.md`. Earlier closures that still stand: pool composition (206), non-monotone
reshaping as a deployable gain (218), global sign synchronisation (213), the clustered
pair-fit (204).

## The thing we said we'd come back to

**If the feature selection improves, the feature *transform* has to be re-derived.** Step 218
pre-registered the mapping: `mode_centre` is right at today's selector precision of 0.562,
`squared` overtakes above 0.654. A better selector makes the reshaped feature matrix stale, and
every number computed on it. **One re-derivation round, round two only, never iterate to
convergence** — if a third round would be triggered, report the oscillation instead.
`scripts/nonmono_v2/repick_transforms.py` re-derives the picks without repeating the full sweep.

## Open items

- **Not quotable**: sign-instability as a headroom detector, 0.500 against an incumbent of
  0.562, in `sign_identifiability.py` under `s3_instability_detector`. It was scored against a
  **hard-coded** incumbent without the reproduction gate that requires reproducing 0.562 /
  +0.309 / 13-of-61 first.
- The shallow label-guided search (+0.69pp) is measured against the **old** floor and is due a
  re-read against the matched one before it is used to judge "how much depth is needed".
- Neither Step 221 script carries a permutation null. Step 220's shows the ceiling clears its
  null by +7.94pp, 23W/1L, so this is a completeness gap rather than a doubt.

## State of the tree

Step 221's artifacts are committed (`c98f294`). Step 222 adds
`scripts/upcr_study/exp14_ranker_menu.py`, `results/upcr_study/14_ranker_menu/`, and the
`dufs_pf_gates` / `dufs_pf_cell_rng` move into `spectral_utils/selectors/a2_groupfs.py`.
Nothing is in flight; no jobs running.

Unrelated to this line, the tree also carries a large set of pre-existing uncommitted
modifications from earlier sessions (selector-bench CSVs with the `seiclr_triviaqa_opt30b` rows
removed, advisor HTML, several docs). They were left untouched — do not sweep them into a commit
without checking whose they are.

---

### Paste-ready prompt for the next session

> Read `PROGRESS.md`, then
> `results/action_items_jul2026/item2_upcr_clustering/PLAN_NEXT.md`, then `PHASE1_RESULTS.md`,
> then `HANDOFF_upcr_selection.md`. Do **not** read the July clustering plan
> (`~/.claude/plans/i-wanted-to-work-tidy-wind.md`) — 947 lines, settled, superseded.
>
> **Where we are.** Only one step inside U-PCR has room in it: which features get kept, worth
> about +2.25pp held out. Step 221 asked what separates the good features and got a two-sided
> answer — their correlation with correctness *identifies* them (overlap +0.11 above a matched
> null, p<1e-4) and *buys nothing* (+0.08pp over a matched floor, p=0.62). Since a perfect
> estimate of that correlation is worth +0.34pp with an interval crossing zero, and the other
> two channels it feeds were already +0.19pp and −0.06pp, **Bracha's second proposal (fix
> U-PCR's estimation) is closed before being built. Her first — DUFS supplies the ranking — is
> the live one.**
>
> **What to run.** The ranker menu in `PLAN_NEXT.md`. Pre-register the list before scoring, then
> score every candidate **twice**: held-out performance against the matched floor (−0.84pp) and
> ceiling (+2.25pp), and overlap with the good sets against the composition-matched null. Those
> two disagreed completely at Step 221, so a candidate that clears only the overlap test has
> reproduced that outcome, not beaten it. `scripts/upcr_study/exp13_incumbent_anchored_ranking.py`
> is the harness — adding a ranker is one entry in the `for name, stat in (...)` loop. Price the
> DUFS gate ranking on its own first, before building anything U-PCR-aligned on top of it.
>
> **Use the corrected numbers**: floor −0.84pp (not −1.55pp), room +2.25pp (not +1.48pp),
> and U-PCR's own ranking is *below* chance at finding the good features (not at chance). The
> stale-numbers table in the handoff says why each changed.
>
> **Two traps.** `prepare_cell` has already applied `ALL_SIGNS`, so fitting `cell["V"]` directly
> gives the hand-oriented arm (0.75713), not the registered one (0.7741) — use `derive_cell()`
> and let `derived_arm_gate()` assert it. And Step 219 belongs to the localization work on the
> `experiment/step-localization` worktree; master is at 221, so check both before numbering.
>
> **How to work.** Explain what you are about to run, and why, before you run it. Never use code
> labels — no R2, T4, S1, phase letters, reviewer names; name each experiment by what it does
> and what it found. Lead with what matters — whether a number clears a threshold we picked
> ourselves is a footnote, not a headline. Price the channel before building in it: ask what a
> perfect oracle is worth there first, and put a confidence interval on every ceiling. Match the
> floor to the arm's own construction, and check that a null is the *right* null before trusting
> that clearing it means anything. Run a review pass before each checkpoint — two agents in
> parallel, one reading the diff against the code, one reading the results against what was
> pre-registered; every finding must carry a file:line or a CSV cell, and the agent flags, it
> never decides. Be concise.
