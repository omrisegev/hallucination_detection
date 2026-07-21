# HANDOFF — In-scope competitor grid + 4 method variants (post-Step-192)

Planning doc for the next session. Three items Omri raised after the Step-192 in-scope
evaluation (a fourth, an SML-vs-L-SML pilot, was considered and CUT). **Do not start until this is turned into a plan and approved.** Scope is the
25 in-scope cells (10 QA + 15 math); RAG/GPQA remain out of scope. GOOD_6 is the leading
detector (0.7587 macro); the label-free selection prize is small and label-gated (honest
split-half ceiling +1.7pp, math-only). Read HISTORY Step 192 + PROGRESS first.

---

## Item 1 — Per-cell competitor comparison grid (the advisor deliverable)

**Goal**: for every in-scope cell that came from a paper with a competitor method, one table row
comparing: **GOOD_5, GOOD_6, top_macro_5, the paper's competitor method, LR@16, LR@30, GroupFS@16,
GroupFS@30** — AUROC each, plus **the features GroupFS selected** (@16 and @30), plus a per-cell
diagnosis of *why* GroupFS's subset is non-optimal.

What already exists (reuse, don't rebuild):
- **Competitor numbers**: `results/repgrid/published_baselines.csv` — per-cell paper baselines already
  curated. Cells with a paper competitor: `losnet_hotpotqa_mistral7b` (LOS-Net 72.92, SemEntropy 67.66,
  p(True) 54.0…), `epr_triviaqa_mistral24b` (EPR 74.6, SelfCheckGPT 79.0), `sciq_llama8b` (HCPD 86.04,
  SAPLMA 85.63…), plus `se_nq_open_llama8b`, `se_squad_v2_llama8b`, `spilled_triviaqa_llama8b`,
  `seiclr_triviaqa_opt30b`, `semenergy_triviaqa_qwen3_8b`, `truthfulqa_llama8b`, `inside_coqa_llama7b`,
  and the LapEigvals GSM8K cells. Confirm each cell's competitor row before building.
- **GOOD_5 / GOOD_6 / top_macro_5**: `results/selector_bench/comparison_inscope.csv` (macro) +
  `results/repgrid/scores_lsml_upcr.csv` (per-cell).
- **GroupFS@30 + chosen features**: `results/selector_bench/a2_groupfs__c46.csv` — the `chosen` and
  `size` columns are already populated (`a2.select`). DONE this session for all 25 cells.
- **GroupFS chosen-set report**: `scripts/selector_chosen_sets_report.py` already renders per-cell chosen
  features + a chosen-frequency table — extend it, don't start fresh.
- **LR@16**: `scripts/logistic_oracle.py` — `FEATURE_SETS = {'5':GOOD_5,'9':STABLE_H9,'16':ALL_H16}`,
  grouped CV via `problem_id` for K=10 cells (Step 182). Already runs 16-feature LR.

What must be BUILT (small):
- **LR@30**: extend `logistic_oracle.py` `FEATURE_SETS` with a `'30'`/`'c46'` = `CANONICAL_POOL`
  entry, **availability-aware per cell** (drop views a cell lacks — same pattern as `prepare_cell`).
  Keep the balanced class weights + grouped-CV discipline from `SUPERVISED_ORACLE_CORRECTION.md`
  (never the `cross_val_predict` calibration pitfall). This is the supervised ceiling with the full
  pool — expect it to beat everything (Step 182: ALL_H16→LR was +8.4pp over GOOD_5 unsup on 19 cells).
- **GroupFS@16**: run `run_selector_bench.py --selector a2_groupfs --pool h16 --cells <6 new>` — only
  the 6 new cluster cells are missing (19 grid cells already have h16 GroupFS rows). Then the
  GroupFS@16-vs-@30 comparison is complete.
- **The grid renderer**: one new HTML table (extend `inscope_report.py` or a sibling) that joins all
  8 columns per competitor-cell, with the GroupFS chosen-feature chips inline.

**Why GroupFS is non-optimal — the diagnosis panel (the interesting part)**:
The mechanism is already known and now reproduced on in-scope cells — surface it explicitly:
- **Gate saturation**: GroupFS's DUFS STG gates saturate OPEN on imbalanced / many-anti-oriented cells
  and dump ~the whole pool. Confirmed this session: `math500_qwenmath7b` → GroupFS selected **size 28
  of 30**; the Step-189 autopsy of `inside_coqa_llama7b` (pos_rate 0.147, 7/23 anti-oriented) found the
  same — selecting ~100% overwhelms L-SML's own K-clustering, which then can't isolate the few bad views.
- **Contrast with GOOD_5's edge**: GOOD_5 wins those cells by *clean isolation* of one bad feature in
  five; a 28-feature dump gives L-SML no such leverage.
- Deliverable: per-cell `chosen size / pool size`, `n_anti_oriented in chosen`, and `pos_rate`, next to
  the AUROC gap vs GOOD_5 — so the "why" reads directly off the table. Cross-link to the Step-192
  orientation audit (`inscope_feature_orientation.csv`).

Risk: LOW (mostly plumbing + one supervised extension). Value: HIGH (advisor-facing, answers "is our
curated subset competitive with the papers AND with supervised / selection ceilings?").

---

## Item 2 — Ofir Lindenbaum's trace-based (Gated-Laplacian) selector

**What Omri means** ("another method by Ofir Lindenbaum … with the trace"): from the Step-185 memo,
Ofir's "trace of a sub-matrix" reference was identified as the **Gated-Laplacian objective
`Tr[X̃ᵀ L_X̃ X̃]`** (Differentiable Unsupervised Feature Selection line, NeurIPS 2021 arXiv:2007.04728).
This is DISTINCT from what we already have:
- `classical_fs.py` has plain **Laplacian Score** (unlearned, per-feature).
- `a2_groupfs.py` is GroupFS (arXiv:2511.09166, 2025) — a *later* Lindenbaum paper using DUFS STG gates.
- The trace/Gated-Laplacian is the *learned, gated* version of the Laplacian-score idea — a different
  objective from GroupFS's group-structured gates.

**Plan**: implement it as a new selector family (`a6_gated_laplacian` or `classical_fs` extension) in the
existing bench harness — same `UnlabeledCell` → `chosen subset` → same L-SML fusion → same npz/flex
scoring. Follow the A2/A3 worktree convention (new-file-only branch, pre-stubbed registry, `smoke()`
with a planted-cluster known-answer test). **Verify the identification against `papers/extracted/` and
the memo before coding** (the memo left it as "candidate identification", not confirmed — re-read the
arXiv:2007.04728 extraction; digest the paper via `/paper-digest` if not cached).

**Why the Gated-Laplacian specifically (the feature-independence angle)**: across the FS research,
"feature independence" appeared in TWO roles, and the Gated-Laplacian is the strongest label-free
selector that operates in role A while respecting the lesson from role B — that's exactly why it's the
one to build.
- **Role A — selectors that pick a NON-REDUNDANT subset** (enforce independence *between chosen
  features*): our A5 **mRMR** (`a5_mrmr.py`, relevance − redundancy, `alpha` knob — helped only on the
  wide pool, not H16), the `decorr` simple-stats floor (greedy low-|ρ|), **LSCAE** (Shaham/Lindenbaum
  NN 2022, arXiv:2110.05306 — reconstruction + Laplacian, "drops redundant/correlated features"), and
  **the Gated-Laplacian itself**. The Gated-Laplacian is the subset-LEVEL member of this family: it
  scores the gated *sub-matrix's* Laplacian trace `Tr[X̃ᵀL_X̃X̃]`, so redundant columns are penalized by
  construction rather than filtered pairwise or ranked one-at-a-time.
- **Role B — the fusion models whose ASSUMPTION is conditional independence of the classifiers given Y**:
  SML (pairwise CI), our **L-SML** (block/latent-group CI — the assumption our band-power features fit
  best, Step 136), U-PCR (uncorrelated errors, Eq. 13, "may be strongly violated"), FUSE (triplet CI).
- **The catch that shapes the design** (memo §1.5, `results/subset_sweep/rho_check.csv`): the naive
  Nadler-lineage proxy for role B — the **ρ≥0.75 correlation filter** — is **empirically REFUTED for
  continuous L-SML** (subsets that violate it score *higher* AUROC). So a good independence-aware
  selector must NOT be a hard pairwise-correlation filter; it must model subset-level structure the way
  L-SML's latent groups already do. The Gated-Laplacian's soft, subset-level, graph-smoothness objective
  is the right shape for this (and is a cleaner formulation than GroupFS's DUFS gates, which saturate —
  Item 1); it is the natural "enforce in selection the independence L-SML assumes in fusion" experiment.

**Manage expectations**: Step 189/192 established the label-free selection prize is small (~ties GOOD_5,
honest ceiling +1.7pp math-only). This is another *label-free* selector and cannot exceed that ceiling.
BUT it's **Ofir's own method** → high advisor relevance; it's the strongest independence-aware label-free
selector we haven't built; and it may dodge GroupFS's gate-saturation failure mode (Item 1) — a clean
scientific result even at parity macro. Risk: MEDIUM (new learned selector + paper re-grounding). Value:
MEDIUM-HIGH (advisor buy-in; principled independence handling; better failure mode). Register it in the
same bench harness as a new family (`a6_gated_laplacian`), new-file-only worktree branch with a
planted-cluster `smoke()`, per the A2/A3/A4 convention.

---

## Item 3 — Use a left-out H16 feature as the orientation anchor

**Idea (Omri)**: the anchor only fixes the label-free global SIGN (it is not itself fused). Currently
`ANCHOR_PRIORITY = ['epr','low_band_power','spectral_entropy','cusum_max']` — all GOOD_5 members. Try
using one of the ~11 H16 features NOT in GOOD_5/GOOD_6 as the anchor direction.

**Why it's cheap**: `score_subset` / the fusion pipeline already take an `anchor` param; Step 184 ran a
`cusum_max` alt-anchor pass, and the anchor-orient memo (`project_anchor_orient_verdict`) exists. So this
is an ablation sweep, not new machinery: for each candidate anchor feature, score GOOD_6 across the 25
cells and compare macro + per-cell sign-flip agreement with the epr anchor.

**What the data already says (temper expectations, but the sweep is still worth it)**:
- The anchor needs a **stable oriented sign**, not high AUROC. From the Step-192 orientation audit
  (`inscope_feature_orientation_summary.csv`): epr is the strongest, most stable (mean 0.734, 0/25 anti).
  Most *left-out* H16 features are WEAK or **anti-oriented** in-scope (per Step 187/192: `hl_ratio`,
  `dominant_freq`, `spectral_centroid`, `high_band_power`, `hurst_exponent`, `cusum_shift_idx`,
  `pe_mean`, `stft_spectral_entropy` all mean-AUROC < 0.5) — a poor anchor flips the whole fusion.
  Candidates that MIGHT anchor (need per-cell sign-stability check, not just mean): `rpdi`,
  `stft_max_high_power`, `sw_var_peak` (in GOOD_5), `low_band_power` (in GOOD_5).
- `project_anchor_orient_verdict`: single-epr anchor already beat multi-feature-average; Step 184 found
  cusum_max agreed with epr on 18/19 cells. So epr is probably near-optimal — but a systematic
  per-feature anchor sweep is a few CPU-minutes and could either (a) confirm epr as a robustness result,
  or (b) surface a left-out feature that anchors the QA cells (where epr is weakest, min 0.56 on losnet)
  better than epr does. The QA cells are the place to look.

Risk: LOW (ablation only). Value: LOW-MEDIUM (likely confirms epr; small chance of a QA-specific win).
Sequence it as the cheap first experiment of the session — it also informs Items 1–3 (whichever anchor
is best should be used consistently).

---

## Suggested sequencing (fastest-payoff first)

*(SML-vs-L-SML pilot CUT per Omri — the extra-clustering-hurts hypothesis is dropped for now.)*

1. **Item 3** anchor sweep (minutes; sets the anchor for everything else).
2. **Item 1** competitor grid (mostly plumbing; the advisor deliverable) — LR@30 extension + GroupFS@16
   6-cell run + the join/render + the GroupFS-non-optimality diagnosis panel.
3. **Item 2** gated-Laplacian selector (largest build; re-ground the paper first).

All must pass `python scripts/smoke_selectors.py` after any bench/selector change and follow the
`SUPERVISED_ORACLE_CORRECTION.md` rules for the LR arms. Keep in-scope artifacts as separate files;
do not regenerate the canonical all-cell reports (they'd re-mix RAG/GPQA).

---

### Paste-ready next-session prompt

> Read HISTORY Step 192, PROGRESS, and HANDOFF_inscope_competitor_and_variants.md. Plan (don't execute
> yet) the three post-Step-192 items on the 25 in-scope cells: (1) a per-cell competitor grid comparing
> GOOD_5 / GOOD_6 / top_macro_5 / paper-competitor / LR@16 / LR@30 / GroupFS@16 / GroupFS@30 with
> GroupFS's chosen features and a why-it's-non-optimal (gate-saturation) diagnosis; (2) implementing Ofir
> Lindenbaum's trace-based Gated-Laplacian selector as a new bench family; (3) a left-out-H16-feature
> anchor sweep. (The SML-vs-L-SML pilot is CUT.) Reuse published_baselines.csv, logistic_oracle.py,
> a2_groupfs__c46.csv (chosen features), selector_chosen_sets_report.py. Sequence anchor-sweep →
> competitor-grid → gated-Laplacian.
