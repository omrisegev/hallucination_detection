# Handoff: mapping the 9 new papers (Step 179 batch) to our benchmarking desk

**Date**: 2026-07-13
**Purpose**: gap analysis only — no cluster runs, presets, or files were created as a result of
this analysis. This is input for a future planning pass. Source papers are the 9 digested in
`papers/digests/` under HISTORY.md Step 179 (corrected — see
`memory/project_paper_digest_skill.md` for the fabrication history on this batch; every claim
below was re-verified against the corrected digest files, not the original Gemini output).

Two of the 9 are **not new comparisons** — they're already our anchors: ARS
(`arXiv:2601.17467`, → `ars_gsm8k_qwen3_8b`/`ars_math500_qwen3_8b`/`ars_gsm8k_r1distill8b`) and
Noise Injection (`arXiv:2502.03799`, → `noise_gsm8k_{phi3mini,mistral7b,gemma2b,llama3b}`). The
7 genuinely new papers are analyzed below.

---

## 1. Reusable now — same model+dataset cells already scored, no new runs needed

### HCPD (`zero-source-llm-hallucination-detection-with-human-like-crit.md`, arXiv:2606.12900, ICML 2026)
Datasets: TriviaQA, SciQ, NQ Open, CoQA. Models: **Llama-3.1-8B and Qwen-3-8B**.
Published Table 2 (AUROC %, avg on Llama): Perplexity 71.52, SAPLMA(sup) 77.99, Semantic
Entropy 73.21, TSV(sup) 74.82, **HCPD 88.19**.

Our matching cells (all `meta-llama/Llama-3.1-8B-Instruct`, GOOD_5 subset, from
`results/repgrid/scores_lsml_upcr.csv`):
| Dataset | Our cell | lsml | upcr | valid_rate | caveat |
|---|---|---|---|---|---|
| TriviaQA | `spilled_triviaqa_llama8b` | 0.934 | 0.914 | 0.512 | only 256/500 rows had usable energy features |
| SciQ | `sciq_llama8b` | 0.738 | 0.744 | 0.198 | very low valid_rate (ceiling accuracy 0.877) |
| NQ Open | `se_nq_open_llama8b` | 0.718 | 0.732 | 0.846 | good valid_rate |
| CoQA | `inside_coqa_llama7b` | 0.684 | 0.608 | 0.901 | **model mismatch**: `huggyllama/llama-7b` BASE, not Llama-3.1-8B-Instruct |

3 of 4 datasets already match HCPD's exact model. Our numbers are directly citable as-is
against their Table 2 (none currently beat 88.19 avg, but SciQ/NQ-Open are close-ish and worth
a real side-by-side write-up). CoQA needs a re-run on the correct model before it's a fair
comparison (see gaps below).

### HalluGuard (`halluguard-...md`, arXiv:2601.18753, ICLR 2026)
Datasets: RAGTruth, **NQ-Open**, **SQuAD**, **MATH-500**, TruthfulQA. Method is an NTK-based
"Hallucination Risk Bound," not spectral — only loosely related to our features, but the
dataset overlap is real.

Our matching cells: `se_nq_open_llama8b` (NQ Open, GOOD_5 lsml 0.718/upcr 0.732),
`se_squad_v2_llama8b` (SQuAD v2, GOOD_5 lsml 0.798/upcr 0.792 — note theirs is plain SQuAD,
ours is SQuAD v2 with unanswerables; not identical), `truthfulqa_llama8b` (GOOD_5 lsml
0.660/upcr 0.673 vs their own baseline TSV published 0.842 — an honest loss), plus our existing
MATH-500 headline results (Qwen2.5-Math-7B etc., see `results/reasoning_benchmark.csv`).
4 of 5 datasets already covered; RAGTruth is the one gap.

### Automatic Layer Selection (`automatic-layer-selection-for-hallucination-detection.md`, arXiv:2605.26366, ICML 2026)
Datasets: TriviaQA, CoQA, SQuAD. Models: "open-weight transformers" — **not specified in the
digest**, so same-model status is unconfirmed. Digest's own results table is non-quantitative
("Matches or exceeds exhaustive grid-search layer selection" — no numbers extracted). Dataset
overlap with our `spilled_triviaqa_llama8b` / `inside_coqa_llama7b` / `se_squad_v2_llama8b` is
real, but **there's nothing numeric yet to compare against** — would need a closer read of the
source PDF's actual results table before this is a citable comparison, independent of any new
cluster run.

### HARP (`harp-...md`, arXiv:2509.11536, unconfirmed venue — likely just an arXiv preprint)
Only one concrete number in the digest: 92.8% AUROC on TriviaQA (model unspecified). We have
TriviaQA cells on four different models already:
`spilled_triviaqa_llama8b` (Llama-3.1-8B, 0.934 lsml), `epr_triviaqa_mistral24b`
(Mistral-Small-24B, GOOD_5 upcr 0.728 vs its own EPR-paper baseline 0.746),
`semenergy_triviaqa_qwen3_8b` (Qwen3-8B, GOOD_5 lsml **0.8006** vs Semantic-Energy-paper
baseline 0.748 — a win), `seiclr_triviaqa_opt30b` (OPT-30B, GOOD_5 lsml 0.595 vs SE-ICLR'23
baseline 0.83 — an honest loss, different K=10 sampling regime). None of our models match
HARP's unspecified one exactly, but the TriviaQA-vs-92.8% comparison is still usable as a
rough anchor across model scales.

---

## 2. Needs new cluster runs / new infra — real gaps

- **CoQA on Llama-3.1-8B-Instruct** — we only have CoQA on `llama-7b` *base* (INSIDE protocol).
  HCPD and Automatic Layer Selection both use the instruct 3.1-8B. This is the single most
  valuable gap to close since it would make HCPD's full 4-dataset grid a same-model comparison.
- **Qwen-3-8B on SciQ / NQ-Open / CoQA** — we have Qwen3-8B only on TriviaQA/GSM8K/MATH-500.
  HCPD's other model arm (Qwen-3-8B) has no coverage on its other 3 datasets.
- **RAGTruth** — HalluGuard's one dataset we've never touched; span-level grounded-hallucination
  labels, different loader shape than our existing QA presets.
- **Summarization + Machine Translation task types** — RAUQ (`efficient-hallucination-detection-
  for-llms-using-uncertainty.md`, arXiv:2505.20045, ICML 2026 — NOT ICLR, corrected) evaluates
  across QA + Summ + MT, 12 datasets, 9 models (incl. Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B,
  Falcon-3-10B). Its Table 1 headline: Perplexity .357, MSP .318, Semantic Entropy .240,
  EigenScore .199, **RAUQ .384** (Mean PRR). We have QA coverage overlap (Llama-3.1-8B across
  several datasets) but zero summarization/MT infrastructure — this needs new loaders, not just
  a new preset.
- **Gemma-2-9B, Falcon-3-10B** (RAUQ) and **Falcon3, Gemma-3, SmolLM3, Qwen2.5-1.5B/3B**
  (Grad-Detect) — model families we've never loaded.
- **PopQA** — Grad-Detect (`grad-detect-gradient-based-hallucination-detection-in-llms.md`,
  arXiv:2606.24790, ICML 2026 *workshop* paper, not main track) uses TriviaQA/SciQ/PopQA/
  TruthfulQA across Qwen2.5(1.5B-7B)/Falcon3(1B-10B)/Gemma-3(1B-12B)/SmolLM3(3B). Its Table 3
  headline (TriviaQA AUC): Semantic Entropy .76-.81, **Grad-Detect .81-.86** across model sizes.
  PopQA has no loader yet; none of its 11 models match ours exactly.
- **Quantum Tensor Network** (`semantic-uncertainty-quantification-of-hallucinations-in-llm.md`,
  arXiv:2601.20026, ICLR 2026) — digest is too vague to plan against ("open-domain QA and
  long-form generation benchmarks," no dataset/model names, no numeric results table). Needs a
  closer read of the source PDF/extracted text before any run could even be targeted — this is
  a digest-quality gap, not a benchmarking gap yet.

---

## 3. RAG / GPQA scope check — answer: no overlap

**GPQA**: none of the 9 papers use it (confirmed against each digest's dataset list). Our own
GPQA result (`reasoning_benchmark.csv`: Qwen2.5-7B, L-SML continuous 5-feat, AUROC 52.3,
"out-of-regime loss... reported honestly") is unrelated to this batch either way.

**RAG**: only HalluGuard touches anything RAG-shaped, via **RAGTruth** — a span-level grounded-
hallucination benchmark, *not* the same thing as our old multi-hop RAG quartet (HotpotQA,
2WikiMultihopQA, NarrativeQA, Natural-Questions-with-context) whose only results live in
untracked `results/subset_sweep/rag__*.npz` files, outside the active repgrid pipeline. No
paper in this batch asks us to revisit that quartet.

---

## Reference: where the numbers above came from

- `results/repgrid/scores_lsml_upcr.csv` — all our own AUROC/CI/valid_rate numbers (columns:
  `cell,model,dataset,n_problems,acc,subset,method,n_feats,auroc_X,lo,hi,n_rows,valid_rate,
  published_Y,Y_method,delta_X_minus_Y,head_to_head,flipped`)
- `results/repgrid/published_baselines.csv` — LOS-Net/HotpotQA and EPR/TriviaQA baseline tables
- `results/reasoning_benchmark.csv` — curated headline table (MATH-500/GSM8K rows + the one
  GPQA row)
- `cache/repgrid/` — local cache dirs for every cell named above (one subdir per cell)
- `papers/digests/*.md` — the 7 new-paper digests, all re-verified line-by-line against
  `papers/extracted/*.md` in the Step-179 audit before this file was written

---

## Prompt for the planning agent

```
You are planning the next phase of cluster benchmarking work for the MV_EPR spectral
hallucination-detection project. Read HANDOFF_new_papers_benchmark_gaps.md at the repo root
first — it contains a verified gap analysis between 9 newly-digested papers
(papers/digests/*.md) and our existing cluster results (results/repgrid/scores_lsml_upcr.csv,
results/reasoning_benchmark.csv, cache/repgrid/).

Your job: turn the "Needs new cluster runs / new infra" section of that handoff into a
prioritized, concrete execution plan. For each gap, decide and justify:

1. Is this worth doing at all? Weigh it against Research_Directions.md's priority order and
   the project's established thesis scope (spectral H(n) features work on reasoning-heavy
   domains — math, science MCQ; NOT short factual QA, per CLAUDE.md's "Best results" table and
   thesis-scope note). Some gaps (e.g. PopQA, a short-answer factual QA set) may be low-value
   even if easy to run.
2. If worth doing, what's the smallest new preset(s) needed in cluster/presets.py? Reuse
   existing loaders (spectral_utils/data_loaders.py) wherever possible — flag which gaps need
   a genuinely new loader (RAGTruth, summarization, machine translation) vs. which just need a
   new (model, dataset) pair on an existing loader (CoQA on Llama-3.1-8B-Instruct, Qwen-3-8B on
   SciQ/NQ-Open/CoQA).
3. Sequence the work: which single new run would close the most valuable gap first? (The
   handoff flags CoQA-on-Llama-3.1-8B-Instruct as the highest-value single gap — it would
   complete HCPD's full 4-dataset same-model grid. Sanity-check that recommendation, don't just
   repeat it.)
4. For the two "digest-quality gap" items (Automatic Layer Selection's missing results table,
   Quantum Tensor Network's fully vague digest) — recommend whether to re-extract/re-dig into
   the source PDF text before committing any cluster compute, since a run can't be well-targeted
   without knowing the paper's actual model/dataset specifics.

Constraints: follow every rule in CLAUDE.md (spectral_utils non-negotiable rule, smoke_preset.py
gate before any cluster submission, Nadler/L-SML fusion invariants, gate/REJECT policy — every
cell ends scored-in-CSV or documented-REJECT). Do not submit any cluster jobs yourself — this is
a planning pass only. Output a prioritized punch list (not a narrative), each item scoped to
one preset or one investigation step.
```

---

## Outcome of the planning pass (2026-07-13, same day)

The planning agent turned the punch list above into a plan
(`C:\Users\DELL\.claude\plans\you-are-planning-the-abundant-quiche.md`), Omri approved it, and
Tier 0 + Tier 1 were executed the same session. This section is the resolution record; the
sections above are left as the original gap analysis (historical — some claims below were
corrected during execution, noted inline where relevant).

**Tier 0 — done:**
- **Automatic Layer Selection**: re-digested from the existing extraction (no PDF re-read
  needed). The original digest's "model unspecified, nothing numeric" claim was wrong — the
  extraction has full Table 2/3 numeric grids and explicitly names LLaMA-3.1-8B-Instruct +
  Mistral-7B-Instruct-v0.3 (same-model overlap on TriviaQA/SQuAD-v1/CoQA). Corrected digest:
  `papers/digests/automatic-layer-selection-for-hallucination-detection.md`. Their own method
  (FEPoID) is a **supervised** MLP-probe ceiling (9k-example trained probe per dataset) —
  flagged as such everywhere it's cited, not used as a fair unsupervised Y.
- **Quantum Tensor Network**: re-digested; also corrected (datasets/8 models ARE named in the
  abstract — the "too vague" framing above was imprecise) but confirmed **documented-REJECT for
  benchmarking**: zero model-roster overlap (Llama-2-7b/13b(-chat), Mistral-7B(-v0.1/-instruct),
  Falcon-RW-1B, LLaMA-3.2-1B — none in our roster) AND zero extractable numeric result anywhere
  in the paper (every result is a pairwise win-rate matrix or RAC-curve figure, verified by
  direct search of the full extraction, main text + appendix). Either reason alone would reject
  it; both hold. See `papers/digests/semantic-uncertainty-quantification-of-hallucinations-in-llm.md`.
- **Anchors wired into the report chain**: HCPD's own Table 2 numbers (the paper's headline
  method, unsupervised zero-source probing — not a baseline) are now the primary `published_Y`
  anchor on `sciq_llama8b` (86.04) and `se_nq_open_llama8b` (90.38), set via each preset's
  `published={...}` block in `cluster/presets.py` (the actual source of truth — `manifest.json`
  is frozen at cluster-submission time and doesn't auto-update from presets.py, so the two
  already-cached manifests were hand-patched to match, then `scripts/score_repgrid.py --cells
  sciq_llama8b,se_nq_open_llama8b,spilled_triviaqa_llama8b` was re-run locally, CPU-only, to
  regenerate exactly those 3 cells' CSV rows — merge-on-write left the other 272 rows untouched).
  Result: **`spilled_triviaqa_llama8b` (Llama-3.1-8B, GOOD_5 lsml 0.934) beats HCPD's own 0.8625
  by +7.1pp** — a new, CI-checkable, citable win. SciQ (-12.2pp) and NQ-Open (-18.6pp) are honest
  losses against HCPD's headline, now with an exact number instead of "close-ish."
  Full per-cell baseline tables (HCPD's own Perplexity/SemanticEntropy/SAPLMA(sup)/TSV(sup);
  ALS's Pred.Entropy/SemanticEntropy/LexicalSimilarity/FEPoID(sup ceiling); HARP's cross-model
  92.8) went into `results/repgrid/published_baselines.csv`. `se_squad_v2_llama8b` deliberately
  got ALS baseline rows only (no `published_Y`) — ALS's own SQuAD method is a supervised
  ceiling and their SQuAD is v1 (ours is v2 w/ unanswerables), so no single number there is a
  safe "the" anchor.
  **Code change beyond the original file list**: `scripts/report_figs.py`'s
  `fig_qa_extension_forest()` only rendered a hardcoded `method == "Semantic Entropy"` pb row as
  a secondary anchor (a narrow patch written for the one pre-existing LOS-Net/HotpotQA case) —
  broadened to `method.startswith("Semantic Entropy")` (still narrow — NOT "all pb rows for the
  cell", which would've flooded the HotpotQA row with LOS-Net's 11 other ablation baselines) so
  the new ALS/HCPD Semantic-Entropy rows render too. Verified end-to-end: `python
  scripts/action_items_report.py` regenerates all 9 pages, guardrail scan clean, HCPD/ALS values
  spot-checked present in the rendered `item3_qa_evaluation.html`.
- **Digest-quality gap recommendation (question 4 in the original prompt)**: re-extracting from
  the PDF was never needed for either paper — both already had full `papers/extracted/*.md`
  text; the gap was in the *digest*, not the extraction. Re-digesting from the existing
  extraction (a local, zero-GPU, ~minutes-long step) was sufficient and should be the default
  move before assuming a PDF re-read is needed.

**Tier 1 — staged, not submitted:**
- New preset `hcpd_coqa_llama8b` in `cluster/presets.py`: Llama-3.1-8B-Instruct, CoQA
  validation, N=500, K=1 greedy (T=0.0) — HCPD's extraction (line ~920) confirms plain greedy
  decoding for the Table-2 detection eval; a separate "greedy decoding with 5 beam search"
  mention (line ~2214) is a different section entirely (RL training-data construction, citing
  Kuhn et al. 2023's protocol for a canonical reference answer) and does not apply here —
  checked directly before writing the preset, not assumed. `judge="Qwen/Qwen2.5-7B-Instruct"`
  (uniform-judge deviation, same pattern as the EPR/SemEnergy cells); `is_correct_coqa`
  (ROUGE-L>0.3) still runs at inference as the lexical label. Reuses the existing `coqa` loader
  — no new loader, per the spectral_utils non-negotiable rule.
  `inside_coqa_llama7b` (the llama-7b-BASE cell) is left in place, unchanged — it's the correct
  cell for the INSIDE-paper comparison it was built for; this new preset is the separate
  SAME-MODEL cell for HCPD/Automatic-Layer-Selection.
  **Gate passed**: `python scripts/smoke_preset.py hcpd_coqa_llama8b` — all HARD checks pass.
  Found and closed a real pre-existing gap while doing this: `coqa` had **no grader fixture at
  all** in `scripts/smoke_preset.py` (silently `[skip]`, not tested — affected
  `inside_coqa_llama7b` too, since it was created). Added a `coqa_family` fixture (5 cases,
  hand-verified against the exact ROUGE-L/LCS formula in `spectral_utils/data_loaders.py`,
  including the "full sentence vs single-word gold" case that exact-match alone wouldn't catch).
  `python scripts/smoke_preset.py --all` — **28/28 pass, 0 fail** after this change.
  **Not submitted** — next step is Omri running `/aircc-submit` for the N=30 pilot (watch the
  accuracy band; `inside_coqa_llama7b` floored at acc=0.132 on the wrong model, the instruct
  model is expected to land in-band).

**Tier 2 — deferred (confirmed with Omri, 2026-07-13):** the Qwen-3-8B arm of HCPD's grid
(SciQ/NQ-Open/CoQA, 3 more presets on existing loaders) waits until `hcpd_coqa_llama8b` is
scored and the Llama-arm HCPD table proves advisor-worthy. If activated: mirror
`semenergy_triviaqa_qwen3_8b`'s `prompt_suffix=" /no_think"` pattern, and first check HCPD's
extraction for whether their own Qwen-3-8B arm ran thinking on/off (their Table 2 doesn't
currently have a Qwen column in our digest — would need pulling before staging those presets).

**Tier 3 — documented skips (closed, no cluster run planned):**
- **RAGTruth** (HalluGuard's one uncovered dataset, confirmed with Omri): span-labeled corpus of
  responses **pre-generated by other models** (GPT-4, Llama-2-chat family, Mistral-7B) — our
  method needs its own-generation entropy traces, so using RAGTruth would require a
  teacher-forcing protocol decision (forced-decoding entropies ≠ generation entropies), which is
  a research-design question, not a loader-building task. HalluGuard is already covered on 4 of
  its 5 datasets without this one.
- **RAUQ summarization + machine translation**: out of thesis scope (spectral H(n) features are
  established on reasoning-heavy domains, not summarization/MT), needs new loaders AND a new
  metric (Mean PRR, not AUROC) — the numbers wouldn't sit next to ours even with the infra
  built. RAUQ's QA overlap is already served by our existing Llama-3.1-8B QA cells; cite as
  related work only.
- **PopQA + Grad-Detect's model families** (Falcon3/Gemma-3/SmolLM3/Qwen2.5-1.5B/3B): workshop
  paper (not ICML main track), zero same-model overlap with our roster, short factual QA (out
  of regime — see CLAUDE.md's thesis-scope note). Our existing 4-model TriviaQA set is already a
  rough cross-model anchor against their 0.81-0.86 range.
- **Gemma-2-9B / Falcon-3-10B** (RAUQ's models): new model families with no reasoning-domain
  payoff on their own — collecting them wouldn't serve the thesis without also solving the
  summarization/MT scope problem above, so skipped together with RAUQ.
