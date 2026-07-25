# HANDOFF — Advisor action-items HTML report (plan mid-execution)

**For the next agent.** Read this + the approved plan file FIRST, then continue. Do not re-plan.

- **Plan file (read it in full — it is the spec)**: `C:\Users\DELL\.claude\plans\i-want-to-adress-enumerated-dahl.md`
- **Goal**: generate 9 advisor-facing HTML pages under `results/action_items/` (index, items 1–6,
  per_domain_breakdown, advisor_scrutiny) via a new `scripts/action_items_report.py`, plus the
  supporting analyses. Everything advisor-facing: NEVER the words "recommended", "MV_EPR", or bare
  "Nadler" (only "Jaffe-Fetaya-Nadler" lineage name allowed; say "L-SML"). `guardrail_scan` from
  `scripts/advisor_report.py` enforces this — run it on ALL 9 pages.
- **User instructions in force**: use the fable model (user set `/model claude-fable-5`); user
  approved the "simplified path" = don't over-debug side quests, prioritize the 9 HTML pages;
  do NOT commit anything (repo files land uncommitted, "await Omri" convention).
- Cluster context: numbers reflect tip commit `5af2931`. 3 cluster cells still running (A2
  `ars_gsm8k_qwen3_8b`, A3 `ars_math500_qwen3_8b` ETA ~2026-07-12 evening, C1 `inside_coqa` +
  chained judge-regrade). Pages must carry an "as of commit 5af2931" note; regenerating after
  those land is a stated follow-up, not part of this session.

## State: DONE (verified)

1. **0a — stale doc tables fixed**: `PROGRESS.md` MEETING ACTION ITEMS rows 3/4/6 and
   `Research_Directions.md` summary rows updated (Item 6 → "✅ Complete (Step 158)…", Items 3/4 →
   "In progress — actively running (Steps 160–169)"). Verify with `git diff PROGRESS.md Research_Directions.md`.
2. **0e — ubaseline_scores.csv recovered + regression fixed**: `results/repgrid/ubaseline_scores.csv`
   now 22 data rows (was silently truncated to 6 by an unconditional overwrite; recovered via
   `scripts/recover_ubaseline_csv.py` merging `git show 4df18aa:` version). All 11 named lost cells
   back. `scripts/score_ubaselines.py` now has merge-on-write (ported from `score_repgrid.py:170-181`)
   so `--cells` runs never drop other cells' rows again.
3. **0b — `scripts/test_multi_anchor_orient.py` REWRITTEN correctly and RUNNING** (see In-flight).
4. **Script 1 — `scripts/compute_legacy_upcr.py` REWRITTEN correctly and RUNNING** (see In-flight).

## Critical API facts (all verified by reading source — do NOT re-derive, do NOT "fix" back)

The previous session's failures were ALL from these, now fixed:

- `boot_auc(y, scores, n=1000)` — **labels FIRST, scores second** (`spectral_utils/fusion_utils.py:52`).
  Reversed args → `ValueError: continuous format is not supported`.
- `upcr_pipeline(feats_dict, feat_names, signs)` returns a **4-tuple** `(score, w, rho_hat, g2_hat)`
  (fusion_utils.py:1208). Unpack `score, *_ = ...`.
- `lsml_continuous_pipeline(feats_dict, feat_names, signs)` returns `(fused_scores, meta_dict)`.
- `anchor_orient(scores, anchor)` → `(oriented_scores, flipped_bool)` — lives in
  `spectral_utils/streaming_utils.py:193`, NOT fusion_utils.
- **Canonical scoring recipe** (must match for comparability — it's what produced
  `scores_lsml_upcr.csv`): see `spectral_utils/repgrid_scoring.py:136-171` `score_subset`:
  valid rows = all subset features finite, ≥20 valid, both classes, ≥3 feats → pipeline with
  `ALL_SIGNS` (from `repgrid_scoring`, = FEATURE_SIGNS + energy/logprob signs) → `anchor_orient`
  vs z-scored oriented `epr` (first subset feature if epr absent) → `boot_auc(y, score)`. Raw
  AUROC, never max(a,1-a).
- Correct subset defs (`scripts/score_repgrid.py:33-45`): `consensus_4` = [spectral_entropy,
  sw_var_peak, cusum_max, cusum_shift_idx]; `GOOD_5` = [epr, low_band_power, sw_var_peak,
  cusum_max, spectral_entropy]; `STABLE_H9` = [epr, low_band_power, high_band_power, hl_ratio,
  spectral_centroid, sw_var_peak, rpdi, pe_mean, cusum_max]; `ALL_H16` = FEAT_NAMES[:16].
- `subset_sweep.iter_cells(data_dir)` yields `(domain, cell_key, feats_dict, labels)` from
  `local_cache/{gsm8k,gpqa,math500,rag,qa}_res.pkl`; labels already int, single-class cells skipped.

## In-flight: two background scripts (this session's processes — they may finish after session end)

Both were mid-run and **producing verified-correct output** (MATH-500/Qwen-Math-7B lsml GOOD_5 =
0.9444 = the known 94.4 headline; R1-Distill 0.8439 = the known 84.4; so the recipe reproduces
canon exactly).

1. `scripts/compute_legacy_upcr.py` — **COMPLETED, exit 0. Wrote 192 rows (24 non-GPQA legacy
   cells × 4 subsets × {lsml, upcr}) → `results/subset_sweep/upcr_legacy.csv`. Do not re-run.**
   The lsml rows are a cross-check vs `sweep_summary.csv` (spot-checked: Qwen-Math-7B GOOD_5
   lsml 0.9444 = canon 94.4; R1-Distill 0.8439 = canon 84.4); report uses `method=='upcr'` rows.
2. `scripts/test_multi_anchor_orient.py` — **COMPLETED, exit 0. Final result (0b — do not re-run):**
   - **GOOD_5** (29 cells): macro epr-anchor **0.6360** vs multi-anchor **0.6388** (delta +0.28pp),
     win/tie/loss 1/26/2, sign disagreements 3/29. Disagreement cells: gpqa Llama-8B (epr right,
     0.5327 vs 0.4673), gpqa R1-Distill (epr right, 0.5525 vs 0.4475), rag
     Mistral-24B/natural-questions (**multi right** — epr anchor leaves it below chance at 0.3746,
     multi puts it at 0.6254).
   - **ALL_H16** (29 cells): macro epr **0.6233** vs multi **0.5467** (delta **−7.66pp** — multi
     anchor loses badly), 6/29 disagreements incl. catastrophic wrong flips of the strongest math
     cells (Qwen-Math-7B 0.9419 → 0.0581, Qwen-1.5B 0.8667 → 0.1333, gsm8k Llama-8B 0.7382 →
     0.2618). With 16 features the equal-weight average dilutes into noise; epr alone is the
     stronger reference on math.
   - **Verdict for scrutiny point 4 and for 0c**: the multi-feature-average anchor does NOT beat
     the single-epr anchor — essentially a tie on GOOD_5, a clear loss on ALL_H16. **Keep the epr
     anchor** (use it in refix_phase12_signs.py). Honest nuance to include: the one RAG
     disagreement shows the epr anchor CAN misfire on a RAG cell (consistent with Step-158's
     "weak anchor" caveat), but it is net-better everywhere else.
   - Sanity: GOOD_5 epr-anchor macro 0.6360 vs the ~0.653 reference — individual cells match canon
     exactly (0.9444 Qwen-Math-7B, 0.8439 R1-Distill, 0.7563 gsm8k Llama-8B), so the gap is battery
     composition (29 scorable cells here vs the sweep's 32-row set), not a recipe bug.
   - Full stdout: `C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\65c08fba-8b62-445b-8625-d23960a83737\tasks\b3yux1a78.output`
     (upcr job stdout: `...\brjuxu2jd.output`).
   - This result belongs in the new HISTORY.md step (per plan: the sign investigation is a
     HISTORY-worthy completed experiment).

## Remaining work, in order

1. **Collect both background results** (or re-run). For 0b, note macro AUROCs + disagreement count.
2. **Replace placeholders 0c/0d with real ready-to-run implementations** (currently stubs that
   just print WAITING — the plan requires real code that executes the moment Omri drops the pkls):
   - `scripts/refix_phase12_signs.py`: load `local_cache/phase12_corrected_results.pkl`, re-apply
     `anchor_orient` (epr anchor per 0b result) to the 3 Phase-12-Corrected analysis cells
     (MATH-500 / GSM8K / GPQA fusion scores), print corrected AUROCs (`boot_auc(y, score)`).
     MATH-500 flip-corrected should land near ~0.94 (historical ref), not ~0.23.
   - `scripts/rescore_phase15_selfconsistency.py`: load `local_cache/phase15_results.pkl` (5 cached
     T=1.0 MATH-500 passes), extract final answers per pass, compute answer-agreement
     self-consistency score, fuse with single-pass L-SML GOOD_5 (z-score both, average, or L-SML if
     ≥3 views), re-check Item-5 gate (ρ < 0.75 AND fused > max(single) + 1pp).
   - Neither pkl exists locally yet — scripts must exit gracefully with the WAITING message when
     absent (keep that behavior), but the full analysis code must be behind it.
3. **Write `scripts/action_items_report.py`** — THE main deliverable. Import
   `esc, pct, read_csv, CSS, guardrail_scan, BANNED` from `scripts.advisor_report` (add
   `sys.path` insert for repo root; advisor_report has no module-level side effects except
   constants — safe to import). Generates all 9 pages under `results/action_items/`. Full
   per-page content spec is in the plan file ("Content plan per page" section) — follow it
   closely; it embeds all the numbers narrative sections need (LR-oracle table, Step-152 fusion
   table, Phase-15 numbers). Data sources + verified schemas:
   - `results/reasoning_benchmark.csv`: dataset, model, method, is_ours, supervision, category,
     auroc (percent-scale), ci_lo, ci_hi, compute, citable, source, note. 56 rows.
   - `results/repgrid/scores_lsml_upcr.csv`: cell, model, dataset, n_problems, acc, subset, method
     (lsml|upcr), n_feats, auroc_X, lo, hi, n_rows, valid_rate, published_Y, Y_method,
     delta_X_minus_Y, head_to_head, flipped. 304 rows, 19 cells (list below).
   - `results/repgrid/ubaseline_scores.csv`: cell, …, ppl_auroc, seqlp_auroc, nent_auroc (+_lo/_hi/
     _lex variants), k, lnpe_q_auroc, pe_q_auroc, lsml_good5_auroc. 22 rows.
   - `results/repgrid/headline_X_vs_Y.csv`: cell, model, dataset, method, best_subset, X, Y, delta,
     head_to_head, n, valid_rate.
   - `results/subset_sweep/sweep_summary.csv`: domain, cell_key, n, …, good5_auroc, all16_auroc,
     epr_auroc, avg_auroc (legacy L-SML; fraction-scale).
   - `results/subset_sweep/upcr_legacy.csv` (new): domain, cell_key, n, subset, method, n_feats,
     auroc, lo, hi, flipped.
   - AIRCC cells: ars_gsm8k_r1distill8b, epr_triviaqa_mistral24b, inside_coqa_llama7b,
     internalstates_gsm8k_qwen25_7b, lapeigvals_gsm8k_{llama3b,llama8b,mistral24b,nemo,phi35},
     losnet_hotpotqa_mistral7b, noise_gsm8k_{mistral7b,phi3mini}, sciq_llama8b, se_nq_open_llama8b,
     se_squad_v2_llama8b, seiclr_triviaqa_opt30b, semenergy_triviaqa_qwen3_8b,
     spilled_triviaqa_llama8b, truthfulqa_llama8b.
   - **Computed (not hand-typed) checks the script must do**: (a) CI-overlap scan — for each
     reasoning_benchmark (dataset, model) group: our row's [ci_lo, ci_hi] vs each competitor's
     point auroc; CI-contained ⇒ "numerically ahead, CI-overlapping", NOT "beats" (known: R1-Distill
     75.0 CI [70.4,79.7] contains ARS 74.72; internalstates U-PCR 69.08 CI [64.0,73.8] contains
     SelfCheckGPT 67.98; the two LapEigvals wins llama8b/phi35 are CI-clear). (b) GOOD_5-vs-seqlp
     tally: join ubaseline_scores (seqlp_auroc; use lsml_good5_auroc when present, else look up
     scores_lsml_upcr GOOD_5 lsml auroc_X by cell), delta per cell, win/loss table — plan predicts
     ≈8 GOOD_5 wins / 11 seqlp wins-or-ties; RE-COMPUTE, don't trust.
   - Item-6 numbers (Phase-15/Step 158): single 0.851 / same-T 5× 0.912 (+6.1pp) / diverse-T 0.859
     (−5.3pp, 95% CI [−10.3, −1.1]); acc collapse 80→4% at T=2.0; mechanism same-T ρ≈+0.45 vs
     multi-T ρ≈+0.01. Item-5 (Step 152): LW-SE K=10 0.614, SelfCheckGPT 0.701, L-SML 1-pass 0.754,
     fused 0.758 (+0.4pp, gate FAIL); MATH-500 row invalid (sign flip); GPQA +2.0 but LW-SE at chance.
   - Item-2 (LR oracle): L-SML 64.2/62.9/64.1 vs LR-CV 68.9/66.8/67.8 (gap +4.7/+3.8/+3.6),
     ceilings 70.5/73.7/79.3; per-domain gap ≈0 reasoning / +4.9 GPQA / +5.8 RAG+QA; weight
     Spearman 0.1–0.2.
   - **U-PCR phrasing trap**: describe as "Tenzer et al. 2022 (AISTATS), continuous-input successor
     in the Jaffe-Fetaya-Nadler line" — the phrase "Nadler's own follow-up" trips the guardrail.
   - Style: reuse advisor_report.py CSS + section-card/badge/table classes; no Chart.js needed
     (tables suffice; pages must be self-contained, no CDN). Relative links between pages
     (they all live in the same dir). Status badges: COMPLETE / IN PROGRESS / COMPLETE-BUT-FLAGGED.
     Item 5/6 pages show "pending local data drop" badges for the 0c/0d rescores (pkls not local).
4. **Run it + verify** (plan "Verification" section, 11 points): guardrail scan 0 hits on all 9
   pages; every index link resolves; spot-check ~10 numbers vs CSVs (94.4 headline, 75.0 vs 74.72
   CI overlap, LR table, Phase-15 numbers); upcr_legacy.csv finite; item3 gate check (SQuAD v2
   ~79.8 ✓, SciQ ~73.8 ✓, TruthfulQA ~66.0 ✓ vs ≥65 bar — verify from CSV, cite as GOOD_5/best
   subset with exact values from scores_lsml_upcr).
5. **HISTORY.md step** — draft ONE new step (next number after current tip; check `git log` /
   HISTORY tail): the report build + the 0b anchor result + the 0e data-loss recovery. Follow
   the What/Why/Result format. Do not commit unless Omri asks.
6. **Final message to Omri**: file list, what ran vs what waits on the two Drive pkls
   (`local_cache/phase12_corrected_results.pkl`, `local_cache/phase15_results.pkl` — he said he'd
   copy them), "as of 5af2931 / 3 cells still running" caveat, and that rerunning
   `action_items_report.py` after A2/A3/C1 land refreshes the pages.

## Gotchas

- Windows env; use `PYTHONPATH=.` via the Bash tool (Git Bash) — that's how everything above ran.
- Background scoring >100 MB pkls: always `run_in_background: true`, generous timeout (CLAUDE.md).
- `results/subset_sweep/*.npz` etc. are uncommitted local artifacts — leave them.
- Do not touch `results/Advisors_Action_Items_Report.html` (the existing advisor report) — the new
  pages are a separate deliverable under `results/action_items/`.
- Concurrent-session rule (memory): if a merge/rebase is somehow in progress, resolve+commit in one
  atomic Bash call; but again — no commits unless Omri asks.
