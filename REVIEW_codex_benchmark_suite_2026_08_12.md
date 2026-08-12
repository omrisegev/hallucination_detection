# REVIEW — Codex's paper-aligned benchmark suite (commit 00f3ec2)

**Date**: 2026-08-12
**Reviewer**: Claude (session following `HANDOFF_codex_localization_report_review.md`)
**Target**: `results/paper_aligned_benchmark_suite_2026_08_11/` + `scripts/paper_aligned_benchmark_suite.py`
+ `spectral_utils/paper_benchmark_suite.py` + `scripts/test_paper_aligned_benchmark_suite.py`
**Verification method**: every claim below was checked against the canonical CSVs
(`results/hard_filter_dufs_liu_24cell/per_cell_metrics.csv`, `results/repgrid/headline_X_vs_Y.csv`,
`results/rag_ec_v1/full_test_split_result.json`) and against the **raw caches themselves** — I loaded
and counted `refchecker_claim_telemetry.pkl` (21,466 entries), `prmbench_telemetry.pkl` (6,969 rows,
per-classification span counts), `gasp_exact.pkl` (2,508 items / 400 responses / 2,714 sentences),
and `pb_critic_qwen72b_full/manifest.json`. Nothing in the suite was modified by this review.

---

## 1. What Codex got right

- **Detection numbers are an exact, traceable read of the canonical bench** — all 24 cells, zero
  mismatches: Deployed U-PCR = `fixed_stable_v1` + hard-filter rows; IU-PCR / DUFS-LIU-PCR =
  `mixed_v2` full pool from `results/hard_filter_dufs_liu_24cell/per_cell_metrics.csv`. No re-fit,
  no invented numbers.
- **Label isolation is real**: no label argument exists anywhere in the fitting path
  (`spectral_utils/paper_benchmark_suite.py:104`), the λ=0 ≡ IU-PCR identity is asserted at runtime
  (line 135), and the frozen settings (2 components, k=7, λ=0.1, DUFS seeds 11/23/37, 80 epochs)
  are exactly the handoff §4.2 requirements.
- **The three worst known traps were all avoided**:
  - PRMBench gold comes from one-based `error_steps`; the PRM's thresholded `labels` field is
    explicitly *not* used as gold (`scripts/paper_aligned_benchmark_suite.py:473-475`), and PRM
    rewards are negated for risk orientation.
  - ProcessBench competitor F1s are divided by 100 at ingestion (line 591) — no percent/fraction
    mixing anywhere.
  - The 3 registered PRMBench alignment-defect IDs are excluded and asserted
    (`BAD_PRM_IDS`, acceptance check in `score_all`).
- **Ground-truth diffs pass**: LettuceDetect 0.792899 vs published 0.7922 ✓; GASP target 0.673
  (not the 0.67 cross-scorer average) with an explicit "do not subtract" caveat ✓; RefChecker
  macro-F1s exact to all digits ✓; PRM ceiling F1s and the official valid-step F1 0.9156 ✓;
  the Qwen3-8B judge correctly named "control, not uPRM" ✓; HLE excluded ✓; N=30 pilot numbers
  never quoted ✓; no cross-task macro anywhere ✓; grouped bootstrap by `source_id` / complete
  problem ✓.
- **GASP input verified complete**: 400 responses all with full+noctx, 2,714 sentences — the
  suite's n matches the authoritative cache exactly.
- Charts are rendered only from machine-readable rows (a test asserts this), and published
  references with mismatched protocol signatures are structurally prevented from sharing a chart
  with local rows.

---

## 2. FATAL findings (would put a wrong number/claim in front of an advisor)

### Fatal 1 — RefChecker "ours" panel is computed on corrupted data
The adapter groups telemetry by `(example_id, claim_index)`, but the pkl key is
`setting|generator|example_id|claim_index::condition`. Dropping the **generator** field collapses
10,733 claims (from ~7 generator models: alpaca_7B, chatgpt, claude2, davinci001, …) into 3,468
dict-overwrite survivors. Verified directly on the pkl: it is complete (21,466 entries =
10,733 × 2 conditions); distinct `(example_id, claim_index, condition)` triples = 6,936, with up
to 7 generators colliding per key. Every "ours" AUROC/AUPRC row on that page (0.788–0.798 "all"
and all per-setting rows) is computed on an arbitrary ~32% sample whose composition depends on
dict iteration order.
- **Where**: `scripts/paper_aligned_benchmark_suite.py:388-390`
- **Fix**: key by the full pkl key prefix (or `(setting, generator, example_id, claim_index)`)
  and rerun only the `refchecker_claims` stage — inputs are local, CPU-only.

### Fatal 2 — The Qwen2.5-72B critic is excluded on a false premise
The page limitations, protocol registry, per-row caveat, and `REVIEW_GUIDE.md` all state the
critic's "four-subset package is incomplete." The authoritative local
`dataset_cache/four_localization/pb_critic_qwen72b_full/manifest.json` (job 177775) is **complete**:
all 4 subsets, F1 74.82 / 60.70 / 50.57 / 51.52, macro 59.40 — the exact handoff §5.7 ground truth.
Codex's machine evidently fetched a stale Drive copy (see §5 below). Result: a named competitor
row is missing and our own data is publicly described as broken.
- **Where**: `scripts/paper_aligned_benchmark_suite.py:616-622`, `LOCALIZATION_META` line 740,
  index-page "Claim boundary" bullet, `REVIEW_GUIDE.md`
- **Fix**: add the critic rows from the complete manifest ("ProcessBench protocol reproduction
  with a different critic model"); delete the "incomplete" claim in all four places.

### Fatal 3 — `baseline_max_entropy` is absent from the ProcessBench panel
The only peer shown is Mind the Gap (macro 0.2496), so the page's visual story is "ours +7pp over
the peer." Our own external-family test (`results/gl_liu_external_v1/`, handoff §5.8) showed we do
**not** clearly beat plain max token entropy (31.71 vs 31.50; the margin flips sign per subset;
0.21pp is noise at ~850 rows/subset). The handoff marks a claimed win over max-entropy as fatal —
omitting the one transparent baseline that erases the margin is the same overclaim by omission.
- **Fix**: add the max-entropy row from the frozen results (no GPU needed) with the honest
  "margin is noise-level" note.

### Fatal 4 — The supervised PRM ceiling is tagged `role="ours"` in the PRMBench AUROC rows
`spectral["qwen_prm"] = …` is injected into the ours score-map before `_method_rows`
(default role `"ours"`), so the supervised ceiling charts inside the same "matched comparison" as
the label-free methods, and the auto-generated conclusion box literally reads: *"Within the local
auprc rows, Qwen2.5-Math-PRM-7B is highest at 0.4454."* A supervised ceiling presented as the best
of our methods is checklist §4.1's exact prohibition. The same mechanism tags the GASP-threshold
reproduction `ours` on the GASP page (had the metric sort landed on AUROC, that page's conclusion
would have named GASP as our best method).
- **Where**: `scripts/paper_aligned_benchmark_suite.py:484` (qwen_prm), line 344 (gasp)
- **Fix**: carry per-method roles into `_method_rows`; regenerate pages (conclusions fix
  themselves).

---

## 3. MATERIAL findings

1. **The RAG-detection quadrant is missing entirely.** No page covers the RAGTruth
   evidence-contrast response-level result (`ec_dufs_liu_evidence_graph` 0.7536; the confirmed
   intervention-design finding; the preregistered novelty test at +2.51pp with CI [−0.58, +5.72],
   P(Δ≤0)=0.066 — crosses zero). The handoff scope is a 2×2 (detection × localization,
   reasoning × RAG); this is a whole quadrant, and it is the project's most current result.
   **Fix**: build the page from `results/rag_ec_v1/full_test_split_result.json`, framing verbatim:
   design confirmed, fusion novelty NOT confirmed.
2. **PRMBench "all" pools the constructed control class and `multi_solutions`.** Verified counts:
   `correct` 758 rows / 10,832 spans + `multi_solutions` 160 rows / 2,241 spans = 13,073 of the
   94,112 "all" spans (14%), against the handoff rule ("scored, not pooled into totals"). Both
   classes then vanish from the subgroup list (single-class skip), so a reader cannot see they are
   inside the totals. **Fix**: report "all (9 error categories)" excluding controls; show the
   control class as its own labelled row.
3. **The step-length availability constraint is not stated** on the PRMBench page (71.0% of steps
   < 32 tokens, median 24). The adapter sidesteps it by aggregating token-level views per step —
   which also means the panel does **not** test the thesis spectral trace-feature pool, only a
   reduced task adapter. Neither fact appears in the limitations card
   (`LOCALIZATION_META`, lines 716-728). **Fix**: add both statements.
4. **ProcessBench "ours" rows are a new two-stage pairing, not frozen GL-LIU v1.**
   `dufs_liu_pcr` = `answer_dufs_liu_mixed` detector + `token_dufs_liu_l0p1` locator
   (macro 0.3191). Frozen GL-LIU v1 (DUFS global + temporal locator, 31.36) and unified core-five
   (31.72) appear nowhere, and the external-family Llama-3.1-8B confirmation — our only
   independent-family evidence — is absent. Presenting a differently-paired system as "ours" on a
   measurement panel is the method-selection-inside-measurement pattern the spec forbids.
   **Fix**: lead with the frozen v1 pairing; label the new pairing explicitly if kept; add the
   external-family cells.
5. **The two detection scoreboards now disagree in front of advisors.** The suite's full-pool
   label-free arms vs `results/repgrid/headline_X_vs_Y.csv` per-cell best-subset arms tell
   different stories on some cells. Worst case `seiclr_triviaqa_opt30b`: documented clear loss
   (0.6197/0.6304, n=4,993) vs the suite's 0.812–0.827 (n=5,000) near-tie vs the published 0.83.
   The n mismatch (4,993 vs 5,000) suggests the repgrid row is additionally stale (the known
   re-grade staleness-carrier failure mode). Both numbers are "real" but unreconciled they look
   like cherry-picking. **Fix**: Omri decides which arm leads (standing rule: never infer the
   leading method); regenerate the repgrid scoreboard against staleness; add a one-line
   reconciliation note wherever both exist.
6. **`diagnostics/` and `results/data_readiness_2026_08_11/dataset_registry.json` are not in the
   commit**, though `REVIEW_GUIDE.md` promises both. Every score hash, input SHA-256, DUFS gate
   record, and the record of which caches were read lives there — the "state exactly what it ran"
   requirement is currently unverifiable, and it is also the only artifact that could
   prove/disprove which Drive copies Codex read. **Fix**: Codex commits both directories from its
   machine.
7. **INSIDE and HARP pages have no ours rows at all** (n=0 in the index). The INSIDE/CoQA cell is
   a documented clear loss (−0.12) that simply vanishes from the suite — a suite whose loss cells
   go missing is the "page that shows only wins" finding. Cells with multiple paper references
   (spilled_triviaqa → HCPD + HARP + ALS) place ours rows on one page only, leaving the other
   pages context-free. **Fix**: add the CoQA ours row (exists in repgrid results); cross-link ours
   values onto every page that references the same cell.
8. **Duplicate published rows**: `published:HCPD` appears twice under two names, `published:EPR`
   twice, `published:LOS-Net` twice — double bars in advisor-facing charts.
   **Fix**: de-duplicate in `build_detection_rows` (lines 185-201).
9. **The SemGrad registry branch is dead code**: `detection-semgrad-*` matches the generic
   `detection-` branch first (line 757 shadows line 796), so those pages carry wrong metadata
   ("cell-specific frozen correctness grader" instead of BEM; sample_count doubled to 2000/1634;
   empty dataset/model fields). Separately, SemGrad was designated background, not a detection
   panel — Omri's call whether it stays. **Fix**: reorder the branch; ask Omri about panel status.

## 4. COSMETIC findings

- `forbid_cross_task_macro` checks a `scope` field that no generated row ever has — the acceptance
  gate is vacuous (`spectral_utils/paper_benchmark_suite.py:376-381`). The real protection is
  structural, but the manifest's `cross_task_macro: False` claim rests on a no-op check.
- RefChecker shows a pooled "all" row first for both ours and the NLI competitor; the three
  settings are three different tasks (macro F1 differs by 0.26) — demote "all" below the
  per-setting rows.
- SemGrad rows have an empty `cell` field in the CSV; HTML tables render n as "400.0000".

---

## 5. The Drive duplicate-folder question — verified

Exactly **one confirmed casualty**: `pb_critic_qwen72b_full`. Codex's machine saw a copy with
missing subsets and no final manifest; the authoritative copy (local and on Drive per Step 242's
byte-verified rclone) is complete. Consistent with the critic's history: it died at its 8 h wall
and was completed by a resume chain, so an early upload could have left a stale same-named folder
on Drive (Drive permits duplicate folder names).

Everything else Codex read matches the authoritative data exactly: GASP counts, RefChecker pkl
completeness (that panel's defect is Codex's grouping code, not missing data), PRMBench
span arithmetic to the row, the LettuceDetect gate number, and the detection CSVs.

**Unverifiable from the repo**: Codex's copy of `ragtruth_ec_test.pkl` (it read
`local_cache/ragtruth_ec/test/` on its machine; the diagnostics that would show `n_responses`
are not committed — see Material 6).

**Cleanup**: search Drive for duplicate-named folders under
`hallucination_detection/cluster_results/`, delete stale ones, and have Codex re-fetch the critic
directory.

---

## 6. Advisor-eyes strategic assessment

1. **The algorithm story is now consistent across five new tasks, and it is not the Laplacian.**
   IU-PCR ≥ Deployed U-PCR on most panels, and DUFS-LIU-PCR ≈ IU-PCR everywhere (differences
   ~10⁻³ with overlapping CIs) — matching the 24-cell +0.0005 result. Codex's own REVIEW_GUIDE
   concedes this. Expect the direct question: *why carry the DUFS/Laplacian machinery at all?*
   The defensible answer today: "current implementation standard, no proven gain" — and the one
   place a real DUFS-LIU gain might still exist (the RAGTruth evidence-graph arm, +2.51pp,
   P=0.066) has a preregistered replication resource sitting unused (the never-scored dev slice,
   `dataset_cache/ragtruth_ec_full/dev/`). **That replication is the single highest-value
   experiment in the project right now.**
2. **Where we are relevant**: answer-level label-free detection (same-model roster wins on the
   GSM8K cells: Mistral-Small-24B +0.28, Nemo +0.16–0.17, Phi-3.5 +0.14–0.15; strong QA cells
   with coverage caveats) and **evidence-contrast RAG detection**, where the *intervention design*
   is the confirmed contribution (full-context-only and likelihood-drop are significantly worse
   than contrast averaging). The same evidence-contrast mechanism powers the claim-checking panel
   (once Fatal 1 is fixed) — a coherent thesis line worth leading with.
3. **Where we are not**: step-level localization. Against a supervised PRM at 0.80–0.92 AUROC per
   category (ours 0.50–0.66, literally chance on `deception`) and a 72B critic at 59.4 macro F1
   (ours ~31.9), plus no clear win over max token entropy in-family — the localization panels are
   *context that shows the gap*, not a claim. The suite's role/access framing (supervised ceiling
   vs label-free, 1 pass vs generation) is the right way to present this; no page may imply
   competitiveness there.
4. **Advisor-report style gap**: the suite never computes a delta even on SAME-MODEL cells, so the
   published-roster headline the advisors expect (one grid, columns per method, winner marked)
   does not exist in it — all published values are reference rows. Honest, but it un-tells the one
   story where we win. The repgrid scoreboard and this suite must be reconciled into a single
   leading grid; which arm leads is Omri's decision (Material 5).

---

## 7. Suggested next actions

1. Fix Fatal 1 (one-line grouping key) and rerun the `refchecker_claims` stage locally —
   all inputs local, CPU-only.
2. Add critic + max-entropy + frozen GL-LIU v1 + RAG-detection rows — every number already exists
   in frozen local artifacts; no GPU, no cluster.
3. Codex: commit `diagnostics/` and `results/data_readiness_2026_08_11/`; re-fetch
   `pb_critic_qwen72b_full` after de-duplicating the Drive folders.
4. Omri decides: which detection arm heads the advisor pages (full-pool label-free arms per the
   prior-free thesis line, vs the repgrid best-subset scoreboard the advisors have already seen),
   and whether SemGrad stays a panel or returns to background.
5. Rerun `score` / `report` and re-review the regenerated pages against handoff §5.
6. Run the RAGTruth evidence-contrast dev-slice replication (see §6.1) before any advisor-facing
   claim about the evidence-graph arm.
