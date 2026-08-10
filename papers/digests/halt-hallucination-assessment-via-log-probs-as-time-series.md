---
slug: halt-hallucination-assessment-via-log-probs-as-time-series
title: "HALT: Hallucination Assessment via Log-probs as Time series"
authors: "Ahmad Shapiro, Karan Taneja, Ashok Goel"
arxiv_id: "arXiv:2602.02888"
venue: "arXiv:2602.02888"
year: 2026
source_pdf: papers/HALT Hallucination Assessment via Log-probs as Time series.pdf
extracted_text: papers/extracted/halt-hallucination-assessment-via-log-probs-as-time-series.md
last_digested: 2026-07-13
---

## Summary

**Corrected 2026-08-09** — the prior version of this digest (TriviaQA/CoQA/SQuAD, single GRU
method, no named benchmark) did not match the paper's actual content; re-verified against
`papers/extracted/halt-hallucination-assessment-via-log-probs-as-time-series.md` (2252 lines,
confirmed present in the one and only arXiv version, v1). Likely a bad backfill pass — see
project convention to spot-check backfilled digests before trusting them.

Presents HALT, a lightweight hallucination detector (F=25 per-token feature vector fed to a GRU)
that treats the top-k token log-probabilities from LLM generations as a temporal time series,
combined with entropy features, to flag hallucinated responses using only output log-probs (no
hidden states/attention maps). Releases two trained variants: **HALT-L** (trained on
LLaMA-3.1-8B log-probs) and **HALT-Q** (trained on Qwen-2.5-7B log-probs; hyperparameters
transferred from HALT-L without re-tuning, which partly explains its lower performance).

The paper also introduces **HUB (Hallucination detection Unified Benchmark)**: ten LLM-capability
clusters (reasoning: Algorithmic, Commonsense, Mathematical, Symbolic, Code Generation;
general-purpose: Chat, Data-to-Text, QA, Summarization, World Knowledge), built by **relabeling
and aggregating prior annotated corpora** — CriticBench (reasoning clusters, original annotations
reused as-is after a manual review confirming their failure taxonomy maps to HALT's
factual/logical hallucination framework), FAVA Annotations, HaluEval (balanced 500-example test
subset), and RAGTruth (test portion). Training is restricted to Chat/Data-to-Text/QA; other
capabilities are held out for out-of-distribution evaluation. **HUB does not use a single
controlled fresh-generation protocol of its own** — because the underlying responses come from
several prior benchmarks' own (heterogeneous, unspecified-per-item) generating models, HALT
instead computes log-probabilities for the existing response text by **teacher-forcing** it
through LLaMA-3.1-8B (HALT-L) / Qwen-2.5-7B (HALT-Q).

**No code or data release is mentioned anywhere in the paper** (no GitHub/HuggingFace link, no
license statement found in the extracted text or the arXiv abstract page).

## Datasets & models used

HUB's ten capability clusters, sourced from CriticBench, FAVA Annotations, HaluEval, and
RAGTruth (train/val/test composition detailed in Table 10 / Appendix E). Log-probability
features recomputed via teacher-forcing through LLaMA-3.1-8B and Qwen-2.5-7B — HALT does not
generate any responses itself.

## Methods it compared itself against

Aggregated log-prob/entropy statistics baselines, the span-based Lettuce detector, and other
prior single-response confidence baselines, evaluated per HUB cluster (Table 2) and for
cross-model transferability (Table 4).

## Experiments — methodology & scores

Macro-F1 and AUROC across the ten HUB test clusters; HALT beats the aggregated-statistics
baselines on 7/10 clusters. Feature-ablation study (Appendix C.3) across all ten HUB clusters
(Tables 5–7).

## Connection to our pipeline

Directly parallels our EPR and WEPR temporal trace work, confirming that modeling log-probabilities
as a time series improves hallucination detection — but note HALT's log-probs come from
teacher-forcing pre-existing text, not on-policy generation, which is exactly the reproduction
gap flagged in `docs/research_notes/external_data_collection_plan_2026.md`'s HUB feasibility
gate (no single generating-model/decoding protocol to reproduce; block fresh on-policy
regeneration of HUB itself unless the authors are contacted for missing provenance, per that
plan's Outcome 3).

## Notes / open questions

2026 preprint (arXiv:2602.02888v1, Shapiro/Taneja/Goel — single version, no revisions).
HUB's four source corpora (CriticBench, FAVA, HaluEval, RAGTruth) are each independently
accessible with their own prompts/labels, so a **new, clearly-named HUB-style on-policy
generation** (Outcome 2 in the data-collection plan) using one of those four source datasets'
own released prompts remains open — just not a reproduction of HUB itself.
