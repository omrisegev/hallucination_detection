# HANDOFF — Step 205: reproduction audit + the L-SML small-subset instability

> **SUPERSEDED 2026-07-28 by HISTORY.md Step 205 (commit `d0bcea7`). Read that instead.**
> This file was verified item by item rather than inherited. Most of it held; four things
> did not, and are corrected in HISTORY Step 205:
>
> 1. **§1b mislabels 0.6833.** It is the AUROC of the **K=2** route `[0,1,1,0]` (residual
>    0.5257), not of a K=3 partition. The mechanism sentence is right; the table caption is not.
> 2. **§1b's loop-vs-vectorised framing is dead on current code** — both give the same K=3
>    partition at m=4. The real trigger is smaller: `np.cov` on a non-contiguous column slice
>    vs a contiguous copy of the same numbers differs by **5.55e-17** and flips the partition.
>    Its jitter counts (2 partitions at 1e-16, 4 at 1e-8) are seed-dependent — not statistics.
> 3. **§5 item 1 was already done.** `stability_audit.csv` was 18:04, not 17:14 — the run
>    finished after the session ended. And the `a4.intrinsic_k_ah` concern was moot: it scores
>    0.733009 on either K path.
> 4. **§3's Spearman is −0.492, not −0.499** (the latter came from a superseded file).
>
> **§4b is resolved**: both measurements were right and fit different feature sets
> (pre- vs post-exclusion). Step 204's B1 is narrowed, not retracted. The defect is now
> **fixed** — exact partition enumeration at m ≤ 4 — and the instability is gone: rows moving
> <0.1pp under jitter went 134/165 → **165/165**.

**Written**: 2026-07-27, end of a long session. **Status**: findings complete and
cross-checked; documentation and one final rebuild remain.

**Why this file exists**: Omri asked for a handoff so a fresh agent can verify this
work rather than inherit it on trust. **Verify before you build on any of it.** Every
claim below has a command next to it.

---

## 0. Where this came from

Omri asked whether every number on `results/upcr_study/comparison.html` was current
("any leftovers?"), then — because the page is going to advisors — asked for all of
them to be *faithful*. Rather than reason about which numbers might be stale, I built
an instrument that replays every published row through today's code. It found two
defects, one of which is a property of the method rather than a bug.

Mid-session Omri asked the sharp question this handoff must not lose:

> "I am trying to see if the fixes actually improve the algorithms or we just found
> numerical instability of our algorithm that makes it impossible to reproduce the
> numbers?"

**The answer is: the fixes improved nothing, and we found real instability — but it
is sharply localised, not global.** Section 3 has the evidence. Do not let this get
softened in the write-up.

---

## 1. What was found

### 1a. A real bug: Eq.15 is identically zero at m < 4 (FIXED)

`_score_matrix_lsml` computes `s_ij = Σ_{k≠i,j} Σ_{l≠i,j,k} |r_ij·r_kl − r_il·r_kj|`.
At **m < 4 that double sum is empty** — {i,j} takes two of at most three indices, k is
forced to the third, no l remains. The original quadruple loop returned exactly `0.0`.
Step 203's vectorisation (commit `a7e8741`, documented as "identical output") computes
it as a difference of large partial sums and returned **2.8e-17** of cancellation noise.

Spectral clustering of an all-zero similarity is pure tie-break, so that noise flipped
assignments. Bisected on `epr_triviaqa_mistral24b`, cols 3/11/21:

| code | assignment | Eq.14 residual | AUROC |
|---|---|---|---|
| pre-Step-203 loop = the published CSV | `[0,0,1]` | 0.0029711 | 0.6023 |
| Step-203 vectorised | `[1,0,1]` | 0.0528410 | 0.5891 |

**Fix**: `_score_matrix_lsml` short-circuits `m < 4` to exact zeros. This restores the
published values bit for bit. Regression gate **U0** added to
`scripts/verify_residual_scaling.py`, checking the vectorised form against a literal
transcription of the paper's sum at m = 2..9.

> Verify: `python scripts/verify_residual_scaling.py` — U0 must PASS, and U1/U2/U3/R1/R2
> must still pass (R1 is the GOOD_6 = 0.7594 anchor).

**This does not make L-SML meaningful at m=3.** Zero is correct, and it means Eq.15
carries no structural information at three features at all. Size-3 rows measure a
degenerate case of the model, not the model.

### 1b. Not a bug: at small m the group assignment is numerically undetermined

Chasing the *size-4* drifts produced the more important finding. On
`lapeigvals_gsm8k_phi35` with `ref.consensus_4`'s subset:

```
score-matrix magnitude          0.156
max |vectorised − loop| at m=4  2.5e-16     (pure float rounding)
K=3 partition, loop             [0,1,2,0] → Eq.14 residual 0.6018
K=3 partition, vectorised       [0,1,0,2] → Eq.14 residual 0.3927
→ residual grid picks a different K → AUROC 0.6833 vs 0.7802  (9.7pp)
```

Perturbing the score matrix by a *relative* 1e-16 already yields two different K=3
partitions across random draws; by 1e-8, four. Step 203 did not cause this — it
perturbed at 1e-16 and that was enough to expose it. **Reverting would not fix it; it
would re-pick one arbitrary side of the tie.**

> Verify: the scratch script that produced this is gone with the session, but it is ~20
> lines — build `R = np.cov(V[:,cols].T)` for that cell/subset, compute the score matrix
> with both the vectorised `_score_matrix_lsml` and a literal quadruple loop, cluster
> each at K=3, and compare `_residual_lsml`. Re-deriving it independently is the point.

### 1c. Every drift is now accounted for

`reproduction_audit.py` replays all 169 (variant, pool) rows; `stability_audit.py`
measures how far each row moves under a 1e-10 relative jitter of the feature matrix
(5 seeds). Cross-tabulating the two closes the question:

Counts below are from the FINAL reproduction audit
(`results/upcr_study/00_reproduction_audit/summary.json`, written 2026-07-27 17:57,
after the `is_k_override()` fix described in the note):

| | count |
|---|---|
| reproduce to the last decimal | **75** |
| drift **less than their own numerical noise** (a coin flip, not staleness) | ~78 |
| drift for a named reason (Step-189 K clamp; h16 rows whose published number came from the Step-153 **lookup table**, not a live fusion) | ~12 |
| **unexplained** | **0** |
| not replayable (a2's two `groups` arms — assignment not stored in the CSV) | 4 |

Totals: 169 pairs, 75 exact / 90 drifting / 4 not replayable.

**Both audits completed and the page was rebuilt on both.** Final verdict tally across
the 187 table rows (`build_comparison.py` prints it, and now hard-asserts the last line):

```
  77  within its own noise
  68  verified
  16  re-run today                      (the a1 family + its +scale_complete arm)
  10  lookup table vs live re-fusion
   9  re-run Step 204
   4  not replayable
   2  not audited                       (the a7 anchor-ablation pair)
   1  code fix: Step-189 K clamp
   0  UNEXPLAINED
```

And: **every row that reproduces exactly is also numerically stable (<0.1pp).**
Perfect concordance — reproducibility and determinacy are the same property here.

> Note: an earlier pass reported 74 exact and one "unexplained" row. That row was a
> replay-fidelity bug of mine, not a data problem: `a4.intrinsic_k_ah` emits an explicit
> K but is not named `*+K_*`, so the replay let the default search pick K instead of
> re-applying the stored one. Fixed by sharing `is_k_override()` across the three
> scripts; it now reproduces at 0.0pp drift on both pools.

---

## 2. Answering Omri's original question (L2) — the corrected criterion

`a1_residual` is the **only** family whose selection objective calls L-SML, so it is the
only one where the loading-scale finding can change *which features get picked*. Every
other family's objective is scale-free and its published subsets stand as selected.

Re-ran the family end to end at both scales (`results/upcr_study/07_a1_rerun/`):

| variant | unit | complete | Δ | W/L | mean size u→c | subsets changed |
|---|---|---|---|---|---|---|
| a1.relres_greedy | 0.6952 | 0.6960 | +0.08pp | 10/15 | 3.1 → 3.0 | 25/25 |
| a1.minres+K_ah | 0.6952 | 0.6960 | +0.08pp | 10/15 | 3.1 → 3.0 | 25/25 |
| a1.router@good5 | 0.7494 | 0.7519 | +0.25pp | 11/13 | 5.0 → 5.0 | 0/25 |
| a1.router@minres | 0.6851 | 0.6960 | +1.09pp | 10/15 | 3.1 → 3.0 | 25/25 |
| a1.router@loco5 | 0.7584 | 0.7705 | +1.22pp | 14/9 | 5.0 → 5.0 | 0/24 |
| a1.upcrres_greedy | 0.7225 | 0.7225 | 0.00pp | — | 3.7 → 3.7 | 0/25 |

**Reading**: the corrected criterion picks *entirely different* features (25/25 cells)
and gains **+0.08pp with more losses than wins**. The residual criterion was never
handicapped by the mis-scaling — it is simply not a good selection criterion. It also
drives every cell to exactly 3 features, i.e. straight into the regime where the model
has no information.

### The router is dead, and for a clean reason

`a1.router@*` compares `lsml_rel_residual` against `upcr_k1_residual` — **two
quantities in different units**. Changing L-SML's scaling convention moves one side and
not the other:

| scale | lsml_rel_residual (median) | upcr_k1_residual (median) | routes to L-SML |
|---|---|---|---|
| unit | 0.7501 | 0.5576 | **1/25** |
| complete | 0.0317 | **0.5576 (identical)** | **25/25** |

So the router's "+1.22pp improvement" is the router *ceasing to route*:
`a1.router@loco5` at `complete` is **bit-identical to `ref.LOCO_5` on 24/24 shared
cells**. It is not selecting; the fixed subset is showing through.

> Verify: `results/upcr_study/07_a1_rerun/*.csv`, `diag_json` column, key `route`.

---

## 3. The answer to "did the fixes help, or is it just instability?"

**Neither fix improved any algorithm.** The m<4 fix restores old numbers exactly (that
is its purpose). The corrected loading scale gives +0.08pp, 10W/15L.

**But "impossible to reproduce" is too strong.** The instability is localised:

| mean subset size | rows | median macro spread under 1e-10 jitter | rows ≥0.5pp |
|---|---|---|---|
| 3 (degenerate) | 12 | 0.000pp | 1 |
| **4** | **36** | **0.439pp** | **18** |
| 5–6 | 87 | 0.000pp | 3 |
| 7–10 | 14 | 0.000pp | 0 |
| 11+ | 16 | 0.000pp | 0 |

Spearman(mean size, macro spread) = **−0.499**. 134 of 165 rows move < 0.1pp.

Note the shape: **size 3 is degenerate but deterministic** (Eq.15 exactly zero → a
constant tie-break), **size 4 is meaningful but undetermined** (Eq.15 has exactly two
terms and K ∈ {2,3} is decided in the last bits). Both are bad, for opposite reasons.

**Every row anyone cares about is stable at 0.00pp**: `ref.LOCO_5`, `ref.GOOD_6`,
`ref.GOOD_5`, `a6.pl_dufs` (20 feats), `a2.dufs` (19), `a4.anchor_adapt` (8),
`a1.router@good5`. The leaderboard's top is unaffected; what is undermined is the
size-4 band, which is mostly the *poorly*-scoring end.

---

## 4. Code changed this session (all uncommitted)

| file | change |
|---|---|
| `spectral_utils/fusion_utils.py` | `_score_matrix_lsml` m<4 short-circuit (+ long comment). Earlier in the session: `LOADING_SCALES`, `_rank1_masked`, `loading_scale` threaded through `_estimate_von_voff` / `_residual_lsml` / `detect_dependent_groups` / `lsml_fuse` / `lsml_continuous` — all defaulting to `'unit'`, verified bit-identical (U4: max \|Δw\| = 3.3e-16) |
| `spectral_utils/selector_bench.py` | `bench_selector(..., sel_kwargs=None)` — forwarded to the selector; **evaluation deliberately stays on the canonical path** |
| `scripts/run_selector_bench.py` | `--sel-kwargs` JSON flag |
| `spectral_utils/selectors/a1_residual.py` | `loading_scale` param threaded into `_eq14_residual` / `_lsml_rel_residual` / `a1_residual`; default `'unit'`. The exhaustive branch reads the Step-153 npz `residual` (unit-scale) and cannot honour it — noted in-code |
| `spectral_utils/glossary.py` | `+scale_{S}` suffix in `SUFFIX_NOTES` + `_SCALE_SUFFIX_RE` in `resolve()`; legacy-U-PCR caveat on `a1.upcrres_greedy` |
| `scripts/verify_residual_scaling.py` | gate **U0** |
| `scripts/upcr_study/reproduction_audit.py` | NEW |
| `scripts/upcr_study/stability_audit.py` | NEW |
| `scripts/upcr_study/build_comparison.py` | verification + stability + scale columns, verdict taxonomy, provenance block, two new filters |

Outputs: `results/upcr_study/00_reproduction_audit/`, `results/upcr_study/07_a1_rerun/`.

---

## 4b. OPEN CONTRADICTION — resolve this first (raised by Omri, 2026-07-27)

Omri's reading at the end of the session:

> "It was never 'we used the wrong Var(Y)'. It's that we compare var_y against a
> covariance matrix whose diagonal is exactly 1.0, so the g2 search only ever explores
> the bottom quarter of its meaningful range. Since g2 is the dial between one and two
> eigenvectors, we've been permanently pinned toward the 2-eigenvector end — which is
> where the redundant second factor gets in."

Checking it produced **a direct contradiction between two of our own measurements**,
which I did not have time to resolve. Do not quote either number until you have.

**What is confirmed true:**
- The g2 grid is literally `np.linspace(0.0, var_y_b, 300)` (`upcr.py`, `_fit_block`),
  and `var_y = scale_ratio * mean(diag C)` with `scale_ratio=0.25`. The features are
  z-scored so `mean(diag C) ≈ 1.0`. The grid therefore spans **[0, 0.25]**. Omri's
  description of the range is exactly right.
- We ARE pinned at the 2-eigenvector end: `auto_components` picks **2 components in
  24/25 cells**, and Step 204's factorial measured that as costing **−3.67pp mean /
  −2.36pp median, 3W/21L, p=9.1e-05**. So the *conclusion* — the redundant second
  factor is getting in and it hurts — is correct and already measured.

**What is NOT the mechanism:** g2 does not gate the component count. `upcr.py:253-254`:
```python
if auto_components:
    n_components = 2 if (k_probe >= 2 and lambda2_frac > lambda2_threshold) else 1
```
The 1-vs-2 decision is made by `lambda2_frac > lambda2_threshold` (λ₂'s share of
trace(C) vs **0.1**), computed *before* g2 is fitted and independently of it. Measured:
`lambda2_frac` median **0.1435**, min 0.0942, max 0.2328 — above 0.1 in **24/25** cells.
So the real dial is `lambda2_threshold=0.1`, sitting just below a tightly-clustered
distribution. That is a much better lead than the g2 range, and it is one cheap sweep.

**THE CONTRADICTION:** fitting all 25 cells at the deployed `scale_ratio=0.25` gives
`g2_hat / var_y` median **1.0000**, and `g2_at_ceiling` true in **24/25 cells** — i.e.
g2 IS clipped at the top of its grid, which supports Omri's concern. But
`results/upcr_study/01_g2_criterion/summary.json` reports
`n_legacy_pinned_at_ceiling: 0` and `n_argmin_moved_when_range_widened: 0`, and its
verdict says *"THE RANGE WAS NEVER BINDING"*. **Both cannot be right.** Likely they
measure different things (exp01 sweeps `scale_ratio` and may evaluate the ceiling flag
at the swept q rather than at the legacy 0.25, or on the legacy `upcr_proj_residual`
path rather than `upcr_fit`) — but that is a hypothesis, not a finding. Read
`scripts/upcr_study/exp01_g2_criterion.py` and settle it.

> Reproduce the new measurement:
> ```python
> from spectral_utils.upcr import upcr_fit
> r = upcr_fit(cell['V'].T, scale_ratio=0.25)
> r.g2_at_ceiling, r.g2_frac_of_var_y, r.lambda2_frac, r.n_components_used
> ```

If exp01's verdict does not survive, **Step 204's section-C bullet B1 ("the g2 search
range never binds") is wrong and must be retracted in HISTORY.md.** I predicted the
ceiling would bind, exp01 refuted me, and I recorded the refutation; this new evidence
points back the other way. Treat all three of us — me, exp01, and this paragraph — as
unverified.

---

## 5. What is LEFT to do

0. **Resolve §4b before anything else.** It may retract a published Step-204 claim, and
   `lambda2_threshold` is the most promising open lead in the U-PCR line.
1. ~~Audits~~ **DONE.** Both completed post-fix (`reproduction_audit.csv` 17:57,
   `stability_audit.csv` 18:04) and `comparison.html` was rebuilt on both.
2. ~~Rebuild the page~~ **DONE**, gates green: GOOD_6 = 0.7594, glossary gaps = 0,
   unexplained rows = 0 (now a hard `assert`, not a printed warning). Re-run
   `python scripts/upcr_study/build_comparison.py` after any further change.
3. **`results/pruning_study/06_scale_vs_criterion` is STALE** — its size grid starts at 3,
   so it was computed with the noisy score matrix. It is the source of Step 204's P2
   headline (Spearman(misfit, AUROC): unit +0.223 → complete −0.006, shift −0.228,
   p=0.0015). **Re-run `python scripts/pruning_study/exp06_scale_vs_criterion.py`
   (~75 min) before quoting that number again.** The Step-204 U-PCR experiments
   (`results/upcr_study/0[1-6]_*`) are NOT affected — they run on the full pool or GOOD_6.
4. **HISTORY.md Step 205** — a draft exists in this session's scratchpad but is not
   written to the repo; re-derive it from this handoff, whose numbers are the checked ones.
5. **PROGRESS.md** — new headline; note that Step 203's own conclusion depends on item 3.
6. Consider whether **size-4 subsets should be excluded** from future selector searches,
   or whether `lsml_continuous` should refuse m < 5. Not decided — Omri's call.

---

## 6. Paste-ready prompt for the new session

```
Read HANDOFF_step205_reproduction_audit.md in the repo root, then CLAUDE.md and PROGRESS.md.

Before continuing, VERIFY the previous agent's work — do not take it on trust:
  1. Resolve the contradiction in §4b. Two of our own measurements disagree about
     whether the g2 grid ceiling binds, and one of them is a published Step-204 claim.
     Read scripts/upcr_study/exp01_g2_criterion.py and settle which is measuring what.
     This is the highest-value item in the file.
  2. python scripts/verify_residual_scaling.py       (U0 must pass; R1 GOOD_6 = 0.7594)
  3. Independently re-derive the m=4 knife-edge claim in §1b — build the score matrix
     for that cell/subset both ways, cluster at K=3, compare the residuals. If it does
     not reproduce, say so; the whole framing depends on it.
  4. Spot-check two rows of results/upcr_study/00_reproduction_audit/stability_audit.csv
     by hand.

Then finish items 1-6 in §5 of the handoff. Note items 1 and 4 are long background runs
(~30 min each and ~75 min) — start them early and write documentation while they run.

Context: the deliverable is results/upcr_study/comparison.html, which Omri is sending to
advisors to show the last two weeks of runs. Every number on it has to be one today's
code actually produces, or be labelled as not.
```
