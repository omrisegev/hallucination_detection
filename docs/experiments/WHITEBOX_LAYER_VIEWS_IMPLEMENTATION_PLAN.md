# White-box depth channel on the localization population — implementation plan

Branch `claude/whitebox-layer-views-v1`, worktree `.worktrees/whitebox-layer-views-v1` (sparse).
Proposal of record: `docs/experiments/WHITEBOX_LAYER_VIEWS_LOCALIZATION_V1.md` (v4).

---

## Context

Step 421 extracted, for the entire localization population (13,769 answers / 145,597 steps /
6,968,779 tokens / 9 cells, teacher-forced, zero gate failures), a per-layer logit-lens field.
It is the **first measurement channel in this project that is not a transform of the output
distribution of one greedy pass**. Step 414 had shown that the ~2.46 conditionally independent
signals available from that distribution cannot be increased by adding more transforms of it,
and named three ways out — internal layer states, multiple samples, a second model. This is the
first of the three, and whether it delivers is unmeasured.

The honest prior is unfavourable. Per-layer lens fusion has already been tried here (Steps
243–245b, final-answer detection): `lens-96` + DUFS-LIU scored **0.7253** macro AUROC against
**0.7298 for final-layer target NLL alone**; after a 245-summary screen the best white-box fusion
only *tied* the best label-selected atomic view (+0.000426, CI crossing zero); and on 31,440
matched rows white ≈ gray (−0.0013) at Spearman 0.8677. Different task, so not closing — but it
fixes the null:

> **H0: the depth field is a noisy reconstruction of the final-layer signal.**

Every stage below is built to reject H0, cheapest test first, with pre-declared kill rules.
Nothing has been measured yet.

---

## The spine: four questions every stage must answer

Each stage's report carries this table, with explicit "not applicable" where a stage cannot speak
to an axis. Estimators are existing project quantities, not new inventions.

| # | axis | estimator | where it already exists |
|---|---|---|---|
| **1** | **A significant feature** | single-feature AUROC / SLA against mandatory baselines, never only inside a fusion | — |
| **2** | **Number of views** | `view_pr = (Σλ)²/Σλ²` over **family virtuals**, conditional (within label class), all three tags; plus the **depth-decay curve** | `A6/scripts/diagnostics/evidence_domain_independence_v1.py` |
| **3** | **Conditional independence of errors given the answer** | **failure phi** between binary "this view's raw peak missed the first error" events, + unique/lost hits, stratified by cell and early/middle/late first-error position | `A6/scripts/evaluate_alternative_views_fusion.py :: failure_table()` |
| **4** | **Order of aggregation and fusion** | fuse-before-readout vs fuse-after-readout, same bank/folds; plus which of the five registered insertion points the channel enters | `A6/docs/reviews/fusion_insertion_map_2026-09-16.md` |

**Axis 3 has a calibrated scale already** — phi .981 (surprisal/gap, near-total redundancy),
.665 (logtail15/base), **.103 (digit/base, the most complementary pair ever found here)**. If the
depth arm lands near .9, H0 is confirmed. It must be **pre-registered**: CLAUDE.md records the
previous use of this diagnostic as *"explicitly post-evaluation"*.

Project caveat to carry: this measures complementary **operational mistakes**, not independence,
and does not test U-PCR's latent residual assumption.

---

## Stage 0 — get the data, and the entry gates. No metric, no claim.

**Zero bytes of the layer-view field are on this machine.** It exists only on the cluster
(`$SHARED/results/{pb_layer_views_qwen3_4b,pb_layer_views_qwen3_8b,prmbench_layer_views_qwen3_8b}`)
and on `gdrive:hallucination_detection/cluster_results/`. 42.5 GiB free locally.

### S0.1 Cluster-side reduction (new script)

`cluster/reduce_layer_views_answer_level.py`, run via the existing CPU harness:

```
ssh aircc 'export SLURM_CONF_SERVER=controller-primary; cd $S/code && \
  sbatch -p power-gpu --qos=owner_880 cluster/cpu_job.sbatch \
    cluster/reduce_layer_views_answer_level.py --out $S/results/layer_views_answer_level_v1'
```

Reads all 13,769 `rows/<row_id>.npz`, emits **one stacked npz**:

| array | shape | dtype | bytes |
|---|---|---|---:|
| `cov_eigs` | `[13769, 36, 32]` | f32 (as captured) | 63.4 MB |
| `hid_proj` | `[13769, 36, 256]` | **f16 as captured — cast to f32 at load** | 253.8 MB |
| `resid_norm_mean` | `[13769, 36]` | f32 | 2.0 MB |
| `lens_anchor` | `[13769]` | f32 — `lens_logp_tgt[resid, -1, :].mean()` | — |
| `depth_decay_corr` | `[13769, 36]` | f32 — per-layer corr with `final_lens_H` | 2.0 MB |
| `identity_check` | `[13769]` | f32 — max abs diff for the S0.3 gate | — |
| `row_id`, `cell`, `gate_flag` | `[13769]` | str | — |
| | | **total** | **≈ 320 MB** |

**Correction to an earlier estimate.** Writing `hid_proj` as f32 would make it
`13,769 × 36 × 256 × 4 ≈ 508 MB` and the bundle ≈ 575 MB, not 320 MB. Store it **f16, exactly as
captured**, and cast to f32 **at load**. That is correct on both counts: the float16 hazard is in
*accumulation* (`np.linalg.norm`, `np.dot`, `np.mean`), not in storage, and casting at load
removes it while halving the transfer.

**Why this shape:** `extract_geometry()` consumes `resid_norm` only as a per-layer token-mean, so
reducing it cluster-side turns 502 MB into 2 MB. Result: **~320 MB in one file** versus 5.5 GB in
13,769 files — one `scp` instead of a VPN file-count problem. The token-level 5.5 GB stays on the
cluster; Stage 3 needs it there anyway.

Hard asserts: 13,769 in → 13,769 out; `gate_flag` propagated, never dropped.
Land with `scp`, then `scripts/land_layer_views_answer_level.py` validates shapes/counts/hashes.

### S0.2 The join — one real trap

ProcessBench npz filenames are **bare** (`gsm8k-0.npz`) while `JOINED.json` uses
`gsm8k::gsm8k-0` → **strip the `<subset>::` prefix**. The *same* filename exists in both the 4B
and 8B trees, so **directory separation is the only cell discriminator — never flatten**.
PRMBench `row_id` matches exactly.

### S0.3 Entry gate — an **identity** check, not a correlation (blocks everything downstream)

`final_lens_H_fullvocab` is stored precisely because it is **the same statistic** as
`lens_H[resid, 35, :]`. So this is not a correlation question with a vague "≈ 1.0 modulo the
top-15 difference" — that would be an unfalsifiable gate with no threshold. It is an **identity**
check with a float16-vs-float32 tolerance:

```
max |lens_H[resid, 35, :].astype(f32) - final_lens_H_fullvocab| <= 1e-2   (f16 ulp at H≈10)
```

Binary, sharp, and far stronger. If it fails there is a token-alignment or axis-order bug and
**no stage proceeds**.

**Separately** — and this is a scientific quantity, not a gate — the decay curve of
`corr(lens_H[resid, l, :], final_lens_H)` against depth, using the **top-15** `final_lens_H`.
It needs its own declared decision rule or it is an illustration rather than a test of H0:

> **H0 decision rule:** H0 survives unless the curve falls below **|corr| = 0.5** at some layer
> `l* <= 30`, i.e. unless there exists a non-trivial depth range whose lens is substantially
> decorrelated from the final layer. Declared before measurement.

### S0.4 Anchors and inputs

- PR anchors pinned: 2.75 marginal / **2.83** conditional (13 virtuals) / **2.46** (12
  within-answer, `n_steps` dropped) / 1.80 (CT7's 7 views) — all (step, conditional, virtuals) on
  the 94,203 labelled PRMBench steps.
- **`9.69` is not a participation ratio** — it is a weight-vector IPR, sign-blind, not comparable
  to any of the above. Naming: `weight_ipr` / `view_pr` / `spectral_effective_rank`.
- Missing inputs are **not a blocker**: regenerate `digitfree_broad50_v1/extracted/*.npz` (9 of 13
  families) via `A6/scripts/run_digitfree_broad50_v1.py::extract()` — all raw pickles are local.
  **Do not regenerate BOCPD**; compute an own 11-family baseline instead (see Stage 2).

---

## Stage 1 — **two** kill tests, one per sub-channel. Run first.

The two sub-channels are structurally different and **a failure of one says nothing about the
other**. The geometry arrays are answer-constant, hence gate-only arithmetically; the lens field
is per-token and is the locator candidate. A single merged kill rule would re-fuse exactly the
split that §2.1 of the proposal established.

| # | sub-channel | test | incumbent to beat | what its failure kills |
|---|---|---|---|---|
| **1a** | geometry (`cov_eigs`, `hid_proj`), answer-constant | answer-level "contains an error" **AUROC** | `n_steps` **and** locator-max | the **gate** only |
| **1b** | lens field, per-token | **gate-free SLA** of a single depth summary | **`final_lens_H` alone** | the **locator** only |

1b follows directly from the strongest historical finding: depth fusion already lost to the
final-layer statistic (0.7253 vs 0.7298). Making `final_lens_H` a mandatory table row is not
enough — it has to be a kill test in its own right. **If no single depth summary beats
`final_lens_H` on gate-free SLA, the locator is dead even if the geometry survives 1a.** Both
tests are cheap and both run before anything else.

Both are CPU-only, minutes, once S0 has landed. Kill rules below are stated on the
**per-cell stratified** statistic — see "the pooled trap".

### The label — the one that would have silently broken this

`target >= 0` is **wrong**: all 6,969 PRMBench rows carry `target == -2` (a "not applicable"
sentinel), which would score every PRMBench answer negative.

```python
pb      = np.char.startswith(cells, 'pb_')
any_err = np.add.reduceat((labels == 1).astype(np.int64), offsets[:-1]) > 0
y       = np.where(pb, target >= 0, any_err)     # 10,477 / 13,769 positives, verified
```

**Label-contract assert (cheap, catches a whole bug class before the experiment).** The two
sources cannot be cross-checked against each other — every ProcessBench step carries
`labels == -2`, so `any_err` is identically `False` on all 6,800 PB answers and an
"agreement" assert would fail on all 4,442 PB error rows by construction. What *is* free is
asserting the **partition** the contract depends on:

```python
assert (labels[pb_steps] == -2).all()          # PB has no per-step labels at all
assert (labels[prm_steps] != -2).all()         # PRMBench has them for every step
assert (target[~pb] == -2).all()               # no PB-style target leaked into PRMBench
assert ((target[pb] >= -1) & (target[pb] < n_steps[pb])).all()
```

If any fails, the label contract has drifted — the exact failure mode that shifted every
PRMBench annotation by one step in Step 313 and was missed because reviews validated against the
derived NPZ rather than the raw source.

### Arms and baselines

**1a arm:** rotation-invariant geometry only — no lens channels, no fusion, no learned weights.
Baselines: **`n_steps`**, **max of the locator score** (the true incumbent: both current gates
are threshold functions of it), **CT7's gate score**
(`TPF/results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz`, `step_scores [145597]` +
`gate [13769]` — CT7 needs no re-derivation).

**1b arm:** one depth summary at a time from the per-token lens field, reduced to steps with the
adopted top-10 token-mean (`A6/spectral_utils/chosen_token_calibration.py:72
step_top_readout(x, spans, k=10)`). Baselines: **`final_lens_H`** (the kill incumbent),
**`argmax(step length)`**, **`random_step`**. No fusion at this stage — single summaries only, so
that a survivor is attributable to depth rather than to combination.

`n_steps`' known **.667 is a pooled *step-level* AUC** and is **not transferable** to an
answer-level question — measure it, never quote it. Use `np.diff(offsets)`.

### Statistics

`auc_plan` + `weighted_auc` from `spectral_utils/historical_fusion_evaluation.py` — what the
frozen v3 benchmark used, exactly tie-corrected, accepts group multiplicities. **Not `boot_auc`**
(hardcoded seed 42, cannot express grouping).

**Do not import it from the root checkout.** The root tree sits on a different branch at a
different commit, so importing across trees is silent version drift — the exact bug class that is
hardest to see. Instead **vendor the 13 lines** into
`spectral_utils/whitebox_layer_views.py` with a provenance comment naming the source file and
commit, plus a test asserting equality with `sklearn.roc_auc_score` to 1e-14 on a fixture. The
repo already carries three byte-identical copies of these functions, so a fourth is in keeping
with its own convention, and it pins the code to this branch. (Widening the sparse cone is the
alternative, but `git sparse-checkout add`/`reapply` is the operation that deleted local-only
results on 2026-09-17 and is barred in this worktree.)

One **shared** weight matrix across arms, cells and pooled — this is what makes it paired:
`rng.multinomial(ng, np.full(ng,1/ng), size=DRAWS)` with **ng = 3,483** (not 1,979, which is
PB-*error* groups). 66 group ids span both benchmarks, so resample the **union**. Filter
non-finite draws before quantiling.

### The pooled trap — why the kill rule is stratified

The corrected label produces base rates that differ by benchmark:

| | answers | positives | rate |
|---|---:|---:|---:|
| ProcessBench (8 cells) | 6,800 | 4,442 | 65.3% |
| **PRMBench (1 cell)** | 6,969 | 6,035 | **86.6%** |

PRMBench is a **single cell**, more than half the population, with a base rate 21 points higher.
So **any feature that merely separates PRMBench from ProcessBench earns free pooled AUROC with
zero within-cell signal** — and the geometry will separate, since it differs structurally between
Qwen3-4B and 8B and between GSM8K and OlympiadBench. `n_steps` separates too, by a different
amount. The three arms are contaminated at *different* rates, so a pooled comparison between them
is not trustworthy. This is the same class of error as the documented pooled-vs-within AUC issue
(89.33% of pooled PRMBench comparisons cross answers).

**Therefore the kill rules are written on the per-cell stratified statistic. Pooled AUROC is
reported as descriptive only and decides nothing.**

### Pre-declared kill rules

> **1a (gate):** if invariant geometry does not beat **both** `n_steps` **and** the locator
> maximum on the **per-cell stratified** AUROC, with a paired source-group interval excluding
> zero, **the gate branch ends**. The locator branch is unaffected.
>
> **1b (locator):** if no single depth summary beats **`final_lens_H` alone** on **per-cell
> gate-free SLA**, with an interval excluding zero, **the locator branch ends**. The gate branch
> is unaffected.
>
> If **both** fail, the line ends, and that is the result — H0 confirmed at the lowest available
> cost.

| axis | 1a returns | 1b returns |
|---|---|---|
| 1 significant feature | per-cell AUROC vs three baselines | per-cell SLA vs `final_lens_H`, `length`, `random_step` |
| 2 views | weak — does the family contribute before counting | the depth-decay curve's decision rule (S0.3) |
| 3 error independence | conditional correlation with locator-max, within answer-label class | **failure phi** vs `final_lens_H`, on the 4,442 PB error answers, pre-registered |
| 4 fusion order | not applicable | not applicable (single summaries, no fusion yet) |

---

## Stage 2 — the gate (conditional on 1a)

Largest measured deficit: **3.20–5.31 pp in every one of the 8 cells**, clean accuracy
27.46–55.60%. ProcessBench-only (6,800 answers: 4,442 erroneous / 2,358 clean).

- **Input: `extract_geometry()`'s contract, via its primitives.** Reuse `_cosine_distance`,
  `_normalized_distance`, `_covariance_summaries` — **not** `validate_and_join`, which hard-
  requires all four `(3,36,T)` lens tensors and would force pulling the full 5.5 GB. 283 nominal
  columns at L=36. **Cast `hid_proj`/`resid_norm` to float32** — the originals are read uncast and
  float16 would accumulate norms in float16. Never `extract_haloscope_projection` (raw `hid_proj`
  coordinates = basis-dependent, the inadmissible pattern). Second admissible set available:
  `extract_ghost_geometry` (turbulence / stubbornness).
- **Incumbent: tail15-Top10 @ q=.33**, not LOCO-5 (Step 423: LOCO-5 is the *worse* gate, not a
  mistuned one). Gate must be fed a cross-answer-comparable **raw** statistic — answer-
  standardized fused step risks are at chance (AUC .43–.52) while raw telemetry is .74–.78.
- **Condition number reported before any fit**; centering and scaling decided separately
  (precedent: a collapse traced to condition 65.8 → 23,334.6, centering the culprit).
- **Deciding endpoint: exact localizations gained vs lost at a matched opened fraction.** Clean
  accuracy + error detection alone is *insufficient* — gates that improved separability a lot
  (macro-F1 .65→.69, AUROC .74→.79) still cut end-to-end localization 36.62→35.53, losing 338
  exact localizations against 85 gained.
- Locator held fixed. Length controlled first. Access level declared (the registered midrank rule
  is **transductive**).

---

## Stage 3 — how many views depth adds (conditional)

- **Estimator is virtual-level**, reusing `evidence_domain_independence_v1.py`. Reference bank:
  **broad-50 primary**, CT7 secondary (coverage, not candidate status — CT7 is the densest bank we
  own at 1.80 of 7).
- **Answer-constant arrays excluded** from the number compared to 2.46; with/without pair
  reported. Including them would inflate the count exactly as `n_steps` inflated 2.46 → 2.83.
- **Self-computed 11-family baseline**, not a regenerated BOCPD. Landing in 2.4–2.9 is free
  validation of the reimplementation; landing elsewhere surfaces a bug before the science.
- Two distinct references: the measured **2.7–2.9 level** is a *comparator* for "more of the
  same" (output-distribution transforms only); the **within-answer token shuffle** is the *null*.
  Both needed. PR as a **curve** vs number of depth virtuals, never one scalar.
- Shuffle null is defined at token level ⇒ **this reduction runs on the cluster**.
- **No authority over the line** (Step 415: +0.32 effective signals, −0.73 pp quality).

---

## Stage 4 — locator variant and fusion order (conditional on **1b** and Stage 3)

One variant, derived from Stage 3's geometry, discussed before building. Constraints fixed now:
a handful of channels not 432; fusion core with and without the component on matched IDs; learned
fusion vs equal averaging under it; **the pre-registered fuse-before vs fuse-after 2×2**; gate
held fixed; effective count beside every fusion claim; no digit channels; CT7 frozen.

---

## Code layout

| path | purpose |
|---|---|
| `cluster/reduce_layer_views_answer_level.py` | S0.1 cluster reduction |
| `scripts/land_layer_views_answer_level.py` | validated landing |
| `spectral_utils/whitebox_layer_views.py` | npz loader + geometry summaries via the three primitives + `answer_error_label()` |
| `scripts/run_whitebox_kill_test_v1.py` | Stage 1a |
| `results/whitebox_layer_views_localization_v1/` | all outputs (already in the sparse cone) |

`tests/` is **outside** the sparse cone — stage tests go under `scripts/` or are staged with
`git add --sparse`. Per CLAUDE.md, no helper is inlined in a driver; everything reusable lands in
`spectral_utils/`.

---

## Reporting contract

1. Numbers and comparisons in **tables**, not prose.
2. **Every feature and method explained in words**, not only by code identifier.
3. **Effective views reported beside every fusion claim**, with all three tags.
4. Every quality table carries the **simple-average control** and the **existing methods**.
5. Mandatory rows: **CT7 gate-free SLA 39.89** (the anchor — *not* macro-F1 41.19), token-level
   L-SML 35.92, token equal mean 32.59, Chen Shannon Drop 39.27 / Avg 25.34, chance 16.58, plus
   **`final_lens_H`**, **`argmax(step length)`** and **`random_step`**.
6. Three lanes never merged: raw SLA · gated exact-error accuracy · end-to-end macro-F1.
7. Per-cell before pooled; **leave-subset-out and leave-model-out separately** (the 8 PB cells are
   4 subsets × 2 models — the first fully crossed roster this project has had).
8. Report format follows `A6/scripts/report_alternative_views_fusion.py` (Hebrew RTL HTML +
   METRICS.csv/json, ERROR_COMPLEMENTARITY.json, AUDIT.json, MANIFEST.json).

---

## Verification

| step | check |
|---|---|
| reducer | local smoke on 3 synthetic npz → N=30 pilot on one cell → full. 13,769 in = 13,769 out asserted. |
| landing | shapes, dtypes, per-cell counts against `JOINED.json` (6969/1000×6/400×2), `gate_flag` distribution |
| **identity entry gate** | `max abs(lens_H[resid,35,:] − final_lens_H_fullvocab) <= 1e-2` — binary, blocks all downstream |
| label contract | the four partition asserts in Stage 1 |
| label derivation | reproduce 10,477/13,769 positives and the per-cell table exactly |
| AUROC | `weighted_auc(plan, ones)` equals `sklearn.roc_auc_score` to ~1e-14 on a fixture |
| bootstrap | ng = 3,483; one shared weight matrix; degenerate draws filtered |
| geometry | column count 283 at L=36; re-derive 2–3 columns by hand against the primitives |
| **pre-run** | a verification subagent checks implementation against this plan **and** this plan against the history lessons (§13 of the proposal) — required by Omri before any run |

Kill rules are pre-declared and honoured: **1a can end the gate branch, 1b can end the locator
branch, and both failing ends the line** — a publishable result, not a failure.

---

## One item to settle across sessions before running

The two working lines currently **disagree about what `9.69` is**. This plan holds that it is an
inverse participation ratio of the *weight vector* — `1/Σ(|w_i|/Σ|w_j|)²`, equal to 11 at uniform
weights, **sign-blind**, and therefore never a dimension count at all. The token-probability line
still reports it as "effective ≈ 9.7". If this reading is right it is a stronger correction than
"measured under different tags": the number was never a participation ratio.

This must be decided **in one place and propagated**, not settled twice. The naming convention
should be adopted by both lines: **`weight_ipr` / `view_pr` / `spectral_effective_rank`**.
