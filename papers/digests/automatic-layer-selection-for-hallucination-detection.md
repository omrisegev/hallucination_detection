---
slug: automatic-layer-selection-for-hallucination-detection
title: "Automatic Layer Selection for Hallucination Detection"
authors: "Xinpeng Wang, William X. Cao, Andrew Gordon Wilson, Zhe Zeng"
arxiv_id: "arXiv:2605.26366"
venue: "ICML 2026 (PMLR 306)"
year: 2026
source_pdf: papers/Automatic Layer Selection for Hallucination Detection.pdf
extracted_text: papers/extracted/automatic-layer-selection-for-hallucination-detection.md
last_digested: 2026-07-13
---

## Summary

Proposes First Effective Peak of Intrinsic Dimension (FEPoID), a training-free criterion
for picking which intermediate transformer layer to probe for hallucination detection —
picks the first local peak in the layer-wise intrinsic-dimension (TwoNN) curve, filtered
against a forward-horizon window so it doesn't fire on spurious early bumps. Also proposes
First-Sentence Truncation (FST): extract the hidden state at the last token of the first
generated sentence rather than the last token of the whole generation, to avoid
end-of-sequence noise (degenerate repetition, semantic drift, inconsistent continuation).
Both plug into a standard hidden-state-probing pipeline (frozen LLM, MLP classifier trained
per-layer on labeled data).

## Datasets & models used

**QA** (context-aware: CoQA, SQuAD, PsiLoQA include the passage; question-only: HotpotQA,
TriviaQA): CoQA (Reddy et al. 2019), **SQuAD — v1** (cited as Rajpurkar et al. 2016, i.e.
SQuAD1.1, not v2/unanswerable), HotpotQA, TriviaQA, PsiLoQA (Rykov et al. 2025). Per-dataset
split: 9,000 train / 1,000 val / test = 7,983 (CoQA), 10,000 (SQuAD), 7,405 (HotpotQA),
10,000 (TriviaQA), 8,103 (PsiLoQA) (extraction lines 1552-1556).

**Summarization**: HaluEval, CNN/DailyMail (7,200/800/2,000 and 9,000/1,000/10,000
train/val/test respectively).

**Models** — main tables (Table 2/3): **LLaMA-3.1-8B-Instruct**, **Mistral-7B-Instruct-v0.3**.
Generalization tables (11/12, base-model + scale sweep): LLaMA-3.1-8B (base), LLaMA-3.2-3B,
LLaMA-3.2-1B.

**Same-model overlap with our roster**: LLaMA-3.1-8B-Instruct on CoQA/SQuAD/TriviaQA is the
exact model in our `spilled_triviaqa_llama8b` / `se_squad_v2_llama8b` cells (SQuAD version
differs — see caveat below) and the planned `hcpd_coqa_llama8b` preset. Mistral-7B-Instruct-v0.3
is NOT the same as our roster's `Mistral-Small-24B` — cross-scale only, not same-model.

## Methods it compared itself against

Two families, evaluated identically on AUROC:
- **Unsupervised / zero-shot** (label-free, directly comparable to our L-SML): Predictive
  Entropy, Length-Normalized Predictive Entropy, Semantic Entropy (Farquhar et al. 2024),
  Lexical Similarity (ROUGE-L to reference).
- **Representation-based, layer-selection required**: EigenScore (fixed middle-layer),
  Local Intrinsic Dimension (LID, probes layer after max-LID layer).
- **Their own criterion family, all downstream of a SUPERVISED MLP probe** trained per-layer
  on the 9,000-example labeled training set (the criteria only differ in *which layer* feeds
  the MLP): RankMe, Curvature, Val Loss, RGN, SNR, ID, **FEPoID**.

**Important for fair comparison**: FEPoID itself is "training-free" only in the sense that
*selecting the layer* needs no labels — but the hidden-state probe it feeds is a supervised
MLP trained on 9k labeled examples per dataset. Treat FEPoID/ID/RankMe/Curvature/ValLoss/
RGN/SNR numbers as a **supervised ceiling**, same policy as SAPLMA(sup)/TSV(sup) in the HCPD
digest — not a head-to-head against our unsupervised L-SML fusion. The fair unsupervised
anchors are Predictive Entropy / Semantic Entropy / Lexical Similarity.

## Experiments — methodology & scores

Labels: QA — exact string match against reference; LLM-as-judge fallback for non-exact
matches (Orgad et al. 2025 protocol). Summarization — TrueTeacher (Gekhman et al. 2023).
Metric: AUROC on the held-out test split, `w=7` forward-horizon for FEPoID (main tables).

**Table 2 (QA, AUROC), our same-model overlap only — LLaMA-3.1-8B-Instruct:**

| Method | CoQA | SQuAD (v1) | TriviaQA | Supervision |
|---|---|---|---|---|
| Pred. Entropy | 0.5833 | 0.5703 | 0.6859 | unsupervised |
| Semantic Entropy | 0.5003 | 0.5518 | 0.5505 | unsupervised |
| Lexical Similarity | 0.6780 | 0.5988 | 0.6838 | unsupervised |
| EigenScore | 0.5247 | 0.5300 | 0.5882 | unsupervised (fixed mid-layer) |
| **FEPoID (ceiling)** | **0.6705** | **0.6377** | **0.7516** | supervised MLP probe |
| Avg (Llama, all 5 datasets) — FEPoID | 0.7253 | | | supervised |

Mistral-7B-Instruct-v0.3 (cross-scale reference only, not same-model): Semantic Entropy
avg 0.6560, Lexical Similarity avg 0.6903, FEPoID avg 0.8531 (Mistral-7B scores noticeably
higher than Llama-3.1-8B across the board on this paper's own eval — a model-behavior
finding, see Figure 3: Mistral-Instruct "consistently produces concise and well-terminated
responses" vs LLaMA-Instruct's inconsistent continuation / semantic drift / degenerate
repetition).

**Table 3 (summarization, AUROC, no FST)** — no dataset overlap with our roster (HaluEval,
CNN/DM); not used for benchmarking anchors.

**Table 4** — FEPoID is ~3-6x cheaper than RGN/SNR/Curvature (10.14s vs 27-58s avg per
dataset on LLaMA-3.1-8B-Instruct, all 32 layers) — an efficiency result, not an AUROC one.

## Connection to our pipeline

No architectural overlap — FEPoID/FST operate on frozen hidden-state representations fed to
a trained MLP; our spectral features operate on generation-time token entropy traces H(n),
no hidden states, no training. The layer-selection contribution doesn't transfer to our
method at all.

What *is* useful: the unsupervised-baseline numbers (Pred. Entropy / Semantic Entropy /
Lexical Similarity) are a same-model (LLaMA-3.1-8B-Instruct), same-family-of-datasets
(CoQA/SQuAD/TriviaQA) reference point for our QA cells once `hcpd_coqa_llama8b` lands —
use alongside the HCPD Table 2 anchors, not instead of them (this paper's numbers are lower
across the board than HCPD's own baselines run on the same datasets/model, e.g. their
Semantic Entropy CoQA 0.5003 vs HCPD's Semantic Entropy CoQA-arm-of-avg 75.26 — different
papers, different prompt/label protocols, so don't average them together; cite both,
separately, as independent literature points).

## Notes / open questions

- **SQuAD version mismatch**: this paper's SQuAD is v1 (answerable-only, Rajpurkar et al.
  2016), context-aware. Our `se_squad_v2_llama8b` cell is SQuAD v2 (includes unanswerables).
  Same caveat already applied to HalluGuard's SQuAD row — annotate, don't treat as identical.
- Corrected 2026-07-13: original digest claimed "TriviaQA, CoQA, SQuAD evaluated on
  open-weight transformers" with model unspecified and "nothing numeric yet to compare
  against." Both were wrong — extraction has full numeric tables (Table 2/3, 664 numeric
  hits) and explicitly names LLaMA-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 (p.1 Figure
  1 subplot captions, confirmed again in Table 2 headers and the "Model" paragraph of
  Section 4.1).
