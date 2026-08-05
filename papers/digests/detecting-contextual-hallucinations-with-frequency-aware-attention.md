---
slug: detecting-contextual-hallucinations-with-frequency-aware-attention
title: "Detecting Contextual Hallucinations in Large Language Models with Frequency-Aware Attention"
authors: "Siya Qi, Yudong Chen, Runcong Zhao, Qinglin Zhu, Zhanghao Hu, Wei Liu, Yulan He, Zheng Yuan, Lin Gui — King's College London, University of Warwick, The Alan Turing Institute, University of Sheffield"
arxiv_id: "2602.18145"
venue: "Preprint, 23 February 2026 (not a conference paper)"
year: 2026
source_pdf: papers/Frequency_Aware_Attention.pdf
extracted_text: papers/extracted/detecting-contextual-hallucinations-with-frequency-aware-attention.md
last_digested: 2026-07-28
---

> **Re-digested 2026-07-28 from the real PDF.** The previous version of this card was written
> without a source and got six things wrong. See "Corrections to the previous digest" at the bottom.

## Summary

At each generation step the paper takes the attention distribution from every layer and head, treats it
as a one-dimensional discrete signal **indexed by token position**, and applies a high-pass operator
(DFT, DWT, or a Laplacian) to isolate its high-frequency component. Two signals are kept per (layer,
head): *context-directed* attention (current token to input context) and *generated-token* attention
(current token to previously generated tokens). The energy of the high-frequency component is the
feature. Hallucinated tokens carry more high-frequency attention energy, which the authors read as
fragmented, unstable grounding. The features from all heads are concatenated and fed to a **supervised
single-layer logistic regression classifier**.

## Datasets & models used

- **Datasets:** RAGTruth (three tasks: QA, Data-to-Text, Summarization) and HalluRAG.
- **Models:** LLaMA-7B-Chat, LLaMA-13B-Chat, Mistral-7B-Instruct.

## Methods it compared itself against

SelfCheckGPT, RefChecker, EigenScore, ReDeEP, Lookback-Lens, and an attention-variance baseline. The
paper groups these as verification-based, internal-representation-based, and attention-based. Its own
claimed difference is that prior attention methods use coarse summaries (mass, variance, entropy) that
miss fine-grained instability, which the high-pass filter recovers.

## Experiments — methodology & scores

Detector trained per domain and evaluated on held-out test sets; primary metric AUROC (also reports F).
A cross-domain table trains on one task and tests on another.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| RAGTruth-QA, LLaMA-7B | AUROC | 0.8482 (Lookback-Lens) | Strongest baseline in the LLaMA-7B group |
| RAGTruth-Summ, LLaMA-13B | AUROC gain | **+6.6% over Lookback-Lens** (Fourier-high) | The headline improvement they quote |
| RAGTruth-Summ, LLaMA-7B | AUROC gain | +2.7% over Lookback-Lens | |
| Top-k heads only, LLaMA-7B | AUROC | recovers >95% of full performance | Signal is concentrated in few heads |

## Connection to our pipeline

- **Overlap, and it is the closest methodological neighbour we have:** both of us apply a frequency
  transform to an internal signal from a single forward pass. We FFT the per-token entropy trace H(n)
  over decoding steps; they FFT the attention distribution over token positions, per layer and head.
- **Difference 1, access class:** they are **white-box** (need attention maps from every layer and
  head). We are **grey-box** (logprobs only). Different cost class in our benchmark taxonomy.
- **Difference 2, supervision:** their detector is **supervised** (logistic regression trained on
  hallucination labels). Ours is label-free at runtime. They belong in Bar D of
  `results/BENCHMARK_STANDING.md` (best method in paper, including supervised probes), not Bar A/B.
- **Not a head-to-head competitor on our cells:** they evaluate on RAGTruth and HalluRAG, both RAG
  benchmarks, and RAG has been out of thesis scope since Step 191.
- **Reusable idea:** the axis of the transform. Their signal is indexed by token position within one
  attention row; ours is indexed by decoding step. An attention trace across decoding steps is a third
  option neither of us has measured.

## Notes / open questions

- Would attention-derived views add anything to our 30-view pool? Step 206 closed pool composition in
  both directions, but every view tested there came from the same logprob signal. A genuinely different
  signal is the one case that negative does not cover. Would need a cluster re-run with attention hooks;
  `spectral_utils/model_utils.py:524-540` already sets `output_attentions=True` but reduces on-GPU to
  LapEigvals features rather than saving a trace.
- Their Top-k head result (>95% of AUROC from a few heads) is a feature-selection finding in a different
  domain, and it is the kind of sparsity claim our selector bench could be pointed at.

## Corrections to the previous digest

The 2026-07-15 card was written with `source_pdf: not downloaded` and `extracted_text: not extracted`.
Six claims in it are contradicted by the PDF:

| Previous claim | Actual |
|---|---|
| authors "Siya Qi et al., **Harbin Institute of Technology**" | King's College London, Warwick, Alan Turing Institute, Sheffield |
| arxiv_id **2604.18647** | **2602.18145** |
| venue **ICML 2026** | Preprint, 23 February 2026 |
| "**training-free** detector" | Supervised single-layer logistic regression trained on labels |
| "attention weights **across decoding steps**" | Attention signals indexed by **token position**, per layer and head |
| "fragmented and unstable **visual**/textual grounding" | No vision component anywhere in the paper |
| models "LLaMA-3.1-8B, Qwen-2.5-7B, etc." | LLaMA-7B-Chat, LLaMA-13B-Chat, Mistral-7B-Instruct |
| "Competitor: Yes, on RAGTruth (we score 87.7% on Llama-8B)" | Our 87.7% is L-CiteEval/hotpotqa, not RAGTruth. Not a valid comparison. |
