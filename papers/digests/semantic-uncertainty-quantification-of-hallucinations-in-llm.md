---
slug: semantic-uncertainty-quantification-of-hallucinations-in-llm
title: "Semantic Uncertainty Quantification of Hallucinations in LLMs: A Quantum Tensor Network Based Method"
authors: "Pragatheeswaran Vipulanandan, Kamal Premaratne, Dilip Sarkar (University of Miami)"
arxiv_id: "arXiv:2601.20026"
venue: "ICLR 2026"
year: 2026
source_pdf: papers/Semantic Uncertainty Quantification of Hallucinations in LLMs A Quantum Tensor Network Based Method.pdf
extracted_text: papers/extracted/semantic-uncertainty-quantification-of-hallucinations-in-llm.md
last_digested: 2026-07-13
---

## Summary

Treats the kernel mean embedding of a question's token-sequence (TS) probability
distribution as the wave function of a quantum tensor network (QTN), and applies
first-order perturbation theory to the QTN's Hamiltonian to quantify the *aleatoric
uncertainty of the TS probabilities themselves* (not just their entropy). Uses this
per-sample uncertainty signal to recalibrate the probabilities via an entropy-maximization
step (balancing Rényi-entropy maximization against a KL penalty scaled inversely by the
uncertainty), then computes semantic Rényi entropy over the recalibrated probabilities —
the final detector, "SRE-UQ" (denoted SE_R^+). Requires R repeated samples per question
(worked example uses R=10) clustered by bidirectional entailment (DeBERTa-large/MNLI),
same sampling requirement as Semantic Entropy (Farquhar et al. 2024).

## Datasets & models used

**Datasets**: TriviaQA, SQuAD 1.1 (Stanford), NQ-Open, SVAMP (math word problems — distinct
from our MATH-500/GSM8K).

**Models** (8, all via HuggingFace): Falcon-RW-1B, LLaMA-3.2-1B, LLaMA-2-7B-chat, LLaMA-2-7B,
LLaMA-2-13B-chat, LLaMA-2-13B, Mistral-7B-Instruct-v0.3, Mistral-7B-v0.1. Includes both
base and instruction-tuned variants, deliberately favoring small/quantizable models (16/8/4-bit
sweep) for resource-constrained-deployment analysis.

**Same-model overlap with our roster: zero.** None of the 8 models match anything in our
cluster roster (Llama-3.1-8B-Instruct, Qwen-7B/Qwen3-8B/Qwen2.5-72B-Instruct-AWQ,
Mistral-Small-24B, DeepSeek-R1-Distill-Llama-8B, Qwen2.5-Math variants). Their
Mistral-7B-Instruct-v0.3 is the same checkpoint used by the Automatic Layer Selection paper
(see `[[project_paper_digest_skill]]`-adjacent digest
`automatic-layer-selection-for-hallucination-detection.md`) but still not our Mistral-Small-24B.

## Methods it compared itself against

Naive Entropy (NE), Semantic Entropy (SES) and Discrete Semantic Entropy (DSES) — all
Farquhar et al. (2024); Embedding Regression (ER, supervised — logistic regression on final
hidden states, P(IK)-style); p(True) (Kadavath et al. 2022, few-shot prompted self-assessment).

## Experiments — methodology & scores

**No numeric AUROC/AURAC/RAC results table exists anywhere in the extracted text — main
body or appendix.** Verified by direct search: the only literal "Table" other than Table 1
is a single-question worked toy example (10 repeated generations of one TriviaQA-style
question, illustrating the entropy calculation mechanics — not a benchmark result). All 116
main-paper experiments (and the further ~150+ quantization/length-ablation experiments in
Appendix C) are reported exclusively as:
- Pairwise **win-rate matrices** (Figs. 2, 3) — "probability that the row method outperforms
  the column method," no absolute AUROC/AURAC values given in text.
- **RAC curves** (Figs. 15–23) and AUROC bar figures (Figs. 6–14) across quantization levels
  (16/8/4-bit) × generation length (phrase/sentence) × instruct/non-instruct models — described
  only qualitatively ("SRE-UQ is in par or even higher than SOTA methods").

This is a genuine property of the paper's presentation (results are plotted, not tabulated),
not an extraction gap — a PDF text extractor cannot recover bar heights or matrix cell values
from vector graphics. Re-reading the source PDF as images (out of scope here) would be needed
to pull any number, and even then there would be no same-model anchor to use it for (see
below).

**Qualitative findings** (verified in text, not numeric): SRE-UQ wins most pairwise
comparisons against all baselines without using labels/supervision (ER and p(True) use
supervision/few-shot labels); robust across 16/8/4-bit quantization with only minor
degradation; larger models (LLaMA-2-13B) score higher than smaller ones (LLaMA-3.2-1B,
Falcon-RW-1B); NE degrades most under quantization, especially on NQ/SQuAD.

## Connection to our pipeline

**No architectural overlap and no usable benchmarking anchor.** Requires R repeated samples
per question for entailment-based clustering (same sampling cost as Semantic Entropy) —
our spectral H(n) features work from a single generation's entropy trace. The QTN/
perturbation-theory machinery operates on the *distribution* of TS probabilities across
repeated samples, not on a single trace's spectral structure — genuinely different signal.

Conceptually adjacent as related work: like our L-SML fusion, SRE-UQ's headline claim is
beating supervised baselines (ER, p(True)) without using labels — a useful "unsupervised
beats supervised" citation if the thesis wants one, but not a comparable number (no shared
model, no extractable digit).

## Notes / open questions

- **Documented-REJECT for benchmarking** (2026-07-13, Tier-0 gap-analysis pass, see
  `HANDOFF_new_papers_benchmark_gaps.md`): not pursued as a cluster-run target. Two
  independent, sufficient reasons: (1) zero model-roster overlap (all 8 models are outside
  our roster), (2) zero extractable numeric result anywhere in the source text to even use as
  a rough cross-model anchor (everything is figure-only win-rate matrices / RAC curves).
  Either reason alone would be enough; both hold.
- **Corrected 2026-07-13**: original digest said "open-domain QA and long-form generation
  benchmarks" (no names) and "no numeric results table" framed as an extraction gap. Datasets
  and all 8 models are in fact explicitly named (abstract + Section 3) — the *no numeric
  results* observation was correct, but for a different reason than assumed: the paper simply
  never tabulates AUROC anywhere, in the main text or appendix, rather than the digest having
  missed a table that exists.
