# HANDOFF — Advisor-report verification, correction, and rewrite

**Created**: 2026-07-15 (end of Step 182 session)
**For**: the next agent, whose sole job is to make every advisor-facing HTML deliverable
**provably correct** and **self-explanatory**, then rewrite them to actually teach what the
experiments did.

This session found a headline-level mislabel (the MATH-500 "94.4" reasoning number is a **T=1.5 /
28%-accuracy** operating point, not the **T=1.0** result it is presented as). That one bug is the
reason for this whole task: if one headline number was mislabeled and shipped to advisors, others
may be too. The next agent must verify all of them, not just this one.

**Ground rules that never change** (from CLAUDE.md + saved memory):
- **Read `PROGRESS.md` first** (the Step-182 header has the full current state), then HISTORY.md
  Steps 152–182 (the "past ~30 sessions" the user means), then this file.
- **Do NOT commit.** Leave everything staged; Omri reviews and commits.
- **Terminology bans (advisor-facing, strict)**: never write "Nadler" (the method is **L-SML** /
  continuous L-SML), never "MV_EPR", never the word "recommended" in advisor mail/HTML, never
  compare against the supervised `best_nadler_on` (that was a bug). `advisor_report.py` has a
  built-in guardrail scan — it must pass clean on every regenerated page.
- **Anti-hallucination is the whole point**: if a number cannot be traced to a source, mark it
  **UNVERIFIED** and surface it. Never invent, round-to-plausible, or "reconstruct from memory" a
  value to fill a gap. A missing number is a finding, not a problem to paper over.

---

## The mission (5 requirements, verbatim intent from Omri)

1. **Every number and result in the HTML must be verified** so it is clear no mislabel like the
   94.4/T=1.0 case slipped through.
2. **Every relevant result from the past ~30 sessions (Steps ~152–182) must be recorded correctly**
   in these files — no hallucinated results, no label mistakes (temperature, accuracy, N, K,
   judge-vs-lexical label protocol, supervised-vs-unsupervised, citable-vs-not).
3. **The HTML may be rewritten.** It should *explain what the experiments did*: prose describing the
   setup, plots, and at least one **worked question→answer example** per domain showing the actual
   question, the model's answer, its correct/incorrect label, and **what our features/entropy-trace
   looked like on that example** — make the abstract spectral features legible to a reader.
4. **Any old number/table/plot already sent to the advisors must now show the corrected value.**
   (The 94.4 case is the known one; find the rest.)
5. **Every action item must be closed rigorously**, with the conclusion stated at 100% confidence
   *and the evidence for why we claim it*. No "suspected / probably / unresolved" left dangling in
   an advisor deliverable unless it is explicitly labeled an open question with a plan.

---

## The confirmed correction you MUST propagate (the 94.4 case)

**Claim (proven this session, evidence in `results/phase1/math500_discrepancy.json`)**: the
MATH-500 / Qwen2.5-Math-7B **94.4** AUROC that appears as the citable "reasoning headline" is a
**T=1.5, accuracy 0.28** operating point — NOT a T=1.0 result.

**The proof, two independent checks** (re-run `python scripts/phase1_math500_discrepancy.py` to
reproduce):
- The four legacy `math500_res.pkl` cells keyed `_T1.0` reproduce the **Phase-4 T=1.5** table's
  accuracy AND epr-AUROC to 3 decimals (Qwen-Math-7B: acc 0.280, epr 0.966 — the T=1.5 row exactly).
  Accuracy is temperature-sensitive, so a cell carrying the T=1.5 accuracy was generated at T=1.5.
- A fresh, genuine T=1.0 generation of the same (model, dataset) gives **acc 0.705, GOOD_5 0.851
  [0.777, 0.918]** — a model does not drop 70%→28% accuracy at one temperature; the 28% is T=1.5.

**The honest reconciled MATH-500 / Qwen-Math-7B picture** (both real, different temperatures):

| temperature | accuracy | GOOD_5 1-pass AUROC | note |
|---|---|---|---|
| **T=1.0** (real operating point) | ~0.70 | **0.851 [0.777,0.918]** | the citable headline; a ~90 *fused* T=1.0 number is also referenced (Phase 5) — **locate + verify its exact value/source before using it** |
| **T=1.5** (harder task, easier detection) | 0.28 | **0.944 [0.901,0.977]** | valid but must be labeled T=1.5 / acc 0.28, not the headline |

**Files that currently carry 94.4 as an unqualified/T=1.0 headline** (grep `94.4` / `0.944`):
`results/reasoning_benchmark.csv` (the source row: `citable=yes`, note "Our reasoning headline"),
`results/Advisors_Action_Items_Report.html`, `results/action_items/{advisor_scrutiny,
item4_benchmarking,item5_sampling_fusion,per_domain_breakdown}.html`,
`results/method_comparison_report.html`, `results/method_comparison_table1.csv`,
`results/Spectral_LSML_Report.html`. Several also carry a now-**stale** caveat calling this "the
known Step-152 P2 trace-regime discrepancy" — A1 **resolves** that open item; update the wording.

**Caution on the comparison, not just the number**: in `reasoning_benchmark.csv` the 94.4 is
compared to Semantic Entropy 87.7 / Self-Consistency 87.2 that come from a *different* source
("Phase-12 old-cache") and are **not** temperature-pinned, and are already `citable=no` with an
NLI-truncation caveat (Step-152 P1). Do not just swap the number — decide, and state, whether that
head-to-head is even apples-to-apples, or flag it.

**Presentation decision is Omri's — ASK before mass-editing**: the recommended fix is to make the
citable MATH-500 headline the **T=1.0 ~85/90** number and demote 94.4 to a clearly-labeled *T=1.5,
acc 0.28* operating-point row. But confirm with Omri whether to (a) relabel to the T=1.0 headline,
or (b) keep 94.4 visible but annotate it "T=1.5, acc 0.28" alongside the T=1.0 number. `CLAUDE.md`'s
"Best results" table already does this correctly (separate T=1.0 90.0% and T=1.5 88.3% rows) — mirror
that convention.

---

## Verification protocol (do this systematically, per file)

**A. Number-provenance audit.** For every numeric claim in every advisor HTML, trace it to a source
CSV cell, and trace that CSV cell to a raw scored result. Produce a provenance table:
`file → claim → source CSV:row → raw pkl / scorer → re-derived value → MATCH / MISMATCH / UNVERIFIED`.
Where feasible, **re-derive the AUROC from the raw cache** with the canonical scorer
(`spectral_utils/repgrid_scoring.score_subset`; `boot_auc(labels, scores)` — labels first) and
confirm it equals the CSV. `scripts/build_repgrid_featcache.py`'s validation gate (Δ=0.0000 on all
19 repgrid cells this session) is the template for "re-derive and compare".

**B. Label-integrity audit (this is where 94.4 hid).** For every (dataset, model) cell, verify the
labels on the row are correct and consistent: **temperature, accuracy, N, K, decoding, label
protocol (judge vs lexical/ROUGE — dual-label swings reach 35pp), supervision (unsup/semi/sup),
head-to-head model match, citable flag, and any CEILING/FLOOR/REJECT gate flag**. A number can be
arithmetically correct and still be a lie if its temperature/accuracy label is wrong.

**C. Published-baseline audit (Gemini-fabrication risk).** Every competitor/published number
(`results/repgrid/published_baselines.csv`, the "vs published Y" columns, paper citations) must be
checked against `papers/extracted/<slug>.md` (verbatim source). Saved memory records that Gemini's
paper backfills fabricated authors/venues/datasets/scores twice even after a grounding fix — trust
nothing Gemini-generated without spot-checking the extraction.

**D. Cross-file consistency.** The same (dataset, model) result must show the same number everywhere
it appears (CSV, item pages, advisor report, method-comparison, explainers). Diff them.

**E. Terminology + guardrail scan** on every regenerated page (see ground rules).

---

## How to make edits (regen chain — do NOT hand-edit generated HTML)

Most advisor HTML is **generated** from CSVs; fix the **source CSV or the generator**, then
regenerate. Hand-editing generated HTML gets silently overwritten on the next build.

- **Generated chain** (memory `project_report_regen_chain`): raw scored →
  `scripts/score_repgrid.py` → `results/repgrid/*.csv` (incl. `headline_X_vs_Y.csv`) →
  `scripts/repgrid_report.py` → `scripts/advisor_report.py` **and**
  `scripts/action_items_report.py` (which builds `results/action_items/*.html` +
  `Advisors_Action_Items_Report.html`); figures via `scripts/report_figs.py`. A new/changed cell
  needs its exact model string in the CSV **and** an entry in the generator's order-list, or it
  silently drops.
- **`reasoning_benchmark.csv`** feeds the reasoning tables — the 94.4 row lives here; fix it here.
- **Hand-written explainers** (edit directly, but still source-verify every number):
  `Spectral_LSML_Report.html`, `Replication_Grid_Report.html`, `Subset_Sweep_Report.html`
  (this one is regenerated by `scripts/subset_sweep_report.py` — regenerate, don't hand-edit),
  `Phase12_Corrected_Explainer.html`, `Streaming_Pilot_Explainer.html`, `method_comparison_report.html`.
- After any CSV/generator fix: rerun the relevant generator, then re-run the guardrail scan.

**Inventory of advisor-facing HTML** (verify/rewrite these):
`results/Advisors_Action_Items_Report.html`, `results/action_items/{index,advisor_scrutiny,
item1_literature_search,item2_lr_oracle,item3_qa_evaluation,item4_benchmarking,
item5_sampling_fusion,item6_temperature_variation,per_domain_breakdown}.html`,
`results/Spectral_LSML_Report.html`, `results/Replication_Grid_Report.html`,
`results/Subset_Sweep_Report.html`, `results/method_comparison_report.html`,
`results/Phase12_Corrected_Explainer.html`, `results/Streaming_Pilot_Explainer.html`.
(Also note any `claude.ai/code/artifact/...` URLs already shared with advisors — listed in
PROGRESS.md history; those are snapshots and cannot be edited, but the local source must be correct
so a re-share is right. Flag any shared artifact whose numbers are now known-wrong.)

---

## Rewrite requirement (requirement 3 — make the reports teach)

Each experiment's page should let an advisor understand *what was actually done* without the code:
- **Prose**: the question the experiment asked, the setup (dataset, model, N, K, temperature,
  labels), and the conclusion at 100% confidence + the evidence for it.
- **A worked example per domain**: a real question, the model's generated answer, the
  correct/incorrect label, and a plot of the **entropy trace H(n)** for that answer with the GOOD_5
  feature values annotated — ideally one *correct* and one *hallucinated* example side by side, so a
  reader sees why the spectral features separate them. Pull real examples from the raw caches
  (`cache/repgrid/<cell>/raw_*.pkl` has `question`, `full_text`, `token_entropies`, `label`).
- **Plots**: per-feature AUROC bars, the fusion landscape, and the H(n) trajectories above.
  Follow the `dataviz` skill; keep them theme-aware and self-contained if published as artifacts.

---

## Known result families to verify (the "past 30 sessions", with known risk areas)

- **Phase-12 Corrected (Step 152)**: multi-pass baselines + fusion gate. **Known caveat**: fresh-cache
  SE/SC baselines collapsed vs the old Phase-12 table (NLI-truncation suspected, Step-152 P1) — the
  old table is flagged "no longer citable until reconciled". Verify no report still cites the old
  SE/SC numbers as live.
- **Replication grid / benchmarking (Steps 160–181)**: 19–20 cells vs published Y. **Known churn**:
  concurrent sessions overwrote CSV cells twice (Steps 167, 173 added merge-on-write + stripped
  401→320 rows). Re-verify CSV integrity against the raw caches. Gate flags/REJECTs
  (`gate_flag()`/`REJECT_REGISTRY` in `report_figs.py`) must render correctly.
- **Item 4/5/6** (benchmarking / sampling fusion / temperature): item5's fusion-gate PASS (95.2,
  Step 174) and item6's temperature story both depend on Phase-15 caches — verify the numbers and
  that item6 correctly frames the AUROC-vs-T inverted-U as accuracy-confounded (the same T-vs-acc
  effect that produced the 94.4 mislabel).
- **Subset sweep (Steps 154, 182)**: `Subset_Sweep_Report.html` was regenerated this session (19
  repgrid cells added). GOOD_5 beats honest LOCO by +1.09pp; **`cusum_max_spilled` does NOT
  replicate** (Step-181's "gate pass" is retired — make sure no report still promotes it);
  **`varentropy`** is a new +1.12pp candidate. Verify these render correctly.
- **LR oracle (Steps 147, 182)**: `item2_lr_oracle.html` + `oracle_repgrid.csv`. GOOD_5 supervised
  headroom is +2.4pp (near-ceiling); wider sets +5.9/+8.4pp.
- **Step 182 itself**: A1 (this correction), A2 streaming +5.6pp, A3 RAG-SCGPT NOT-CITABLE,
  B temperature (spectral_entropy sign), C1–C3 staged. Ensure any report touching these is current.

---

## Deliverables for the next session

1. A **verification report** (`results/report_verification_audit.md` or an HTML): the provenance
   table, every MISMATCH/UNVERIFIED found, and what was corrected.
2. **Corrected CSVs + regenerated HTML** (guardrail-clean), with the 94.4 fix applied per Omri's
   chosen presentation, and every other mislabel found in the audit fixed.
3. **Rewritten explanatory HTML** meeting requirement 3 (prose + plots + worked Q→A examples).
4. **HISTORY.md Step 183 + PROGRESS.md** update. **Do not commit.**

Start by confirming with Omri the 94.4 presentation choice (relabel vs annotate), then run the
number-provenance audit before editing anything.

---

## PASTE-READY PROMPT (copy the block below to the next agent)

```
Read PROGRESS.md (Step-182 header) and HISTORY.md Steps 152–182, then read
HANDOFF_report_verification.md IN FULL — it is your task spec.

Your job: make every advisor-facing HTML deliverable provably correct and self-explanatory,
then rewrite them to teach what the experiments did. This is a rigorous verification +
correction + rewrite task, not a cosmetic one. Anti-hallucination is the whole point: any
number that cannot be traced to a source cache/CSV is marked UNVERIFIED and surfaced, never
invented or reconstructed from memory.

Do it in this order:
1. Confirm with me how to present the MATH-500 "94.4" number: it is PROVEN (results/phase1/
   math500_discrepancy.json) to be a T=1.5 / acc-0.28 operating point, not the T=1.0 result it
   is currently shipped as. Options: (a) relabel the citable headline to the true T=1.0 number
   (GOOD_5 85.1 / ~90 fused, acc 0.70) and demote 94.4 to a labeled T=1.5 row, or (b) keep 94.4
   visible but annotate it "T=1.5, acc 0.28" alongside the T=1.0 number. Recommend (a). Mirror
   CLAUDE.md's "Best results" table convention.
2. Run the full number-provenance + label-integrity + published-baseline audit described in the
   handoff, across all advisor HTML (results/*.html + results/action_items/*.html). Re-derive
   AUROCs from the raw caches with the canonical scorer (repgrid_scoring.score_subset;
   boot_auc(labels, scores)) where feasible. Produce a provenance/audit table.
3. Fix at the SOURCE (CSV or generator) and regenerate — never hand-edit generated HTML. Apply
   the 94.4 fix and every other mislabel the audit finds. Keep the advisor_report guardrail scan
   clean; obey the terminology bans (no "Nadler"/"MV_EPR"/"recommended"; method = L-SML).
4. Rewrite the explanatory pages so each experiment is understandable from prose + plots + at
   least one worked question→answer example per domain (real question, model answer, label, and
   the H(n) entropy-trace/GOOD_5 feature values for a correct vs a hallucinated example, pulled
   from cache/repgrid/<cell>/raw_*.pkl). Use the dataviz skill for plots.
5. Write the verification audit to results/, update HISTORY.md (Step 183) + PROGRESS.md. DO NOT
   COMMIT — I review and commit.

Every action item must end with a 100%-confidence conclusion and the evidence for it. If you
find a number that can't be verified, stop and tell me — do not guess.
```
