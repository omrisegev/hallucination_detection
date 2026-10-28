# How each competitor shows generalization, beside ours (2026-10-28)

Sources: `docs/research_notes/competitor_registry_2026-10-21/REGISTRY.md`, `docs/research_notes/competitor_primer_2026-10-22/`
(read from the papers' PDFs) and `papers/digests/crop-conformal-clean-prefix.md`. Counts come from those notes and were
not re-checked against the PDFs in this session.

| Method | Benchmarks | Models / PRMs | Labels | How generalization is shown |
|---|---|---|---|---|
| **PRISM** (arXiv 2606.09078) | PRMBench (Tables 2, 7), ProcessBench (Table 6); downstream: MATH-500 (100-problem subsample), AIME-24 and LiveCodeBench for Best-of-N and guided decoding; GRPO with Qwen2.5-1.5B on six math sets | 1 PRM retrained (Qwen2.5-Math-PRM-7B), plus ReasonEval-7B in an appendix | PRM800K step pairs | A second benchmark plus downstream use. On ProcessBench, lowering FPR also lowers TPR, so the gain does not clearly transfer (its F1 is flat at 77). No code or checkpoint. |
| **CROP** (arXiv 2605.30085, EMNLP 2026) | 6 sets: CRV-Arith, CRV-GSM8K, ProcessBench, Math-Shepherd, PRMBench, PRM800K | 1 PRM (Qwen2.5-Math-PRM-7B) and 1 telemetry model (Llama-3.1-8B-Instruct), 5 risk proxies | Every threshold calibrated with labels; the LR proxies trained on 60% of each set | Calibrated and tested within each dataset. Cross-domain (Table 17, CRV-Arith ↔ CRV-GSM8K) is shown as a FAILURE: 14.8% risk against a 5% target. Each new domain needs recalibration. |
| **SLIDER / Step-RRI** (arXiv 2610.00571, ICLR 2026 workshop) | PRMBench redundancy class only (~750 problems), plus synthetic injected-noise checks | MiniLM embeddings, GPT-4o-mini for the variants | Threshold from a labelled held-out set | None beyond one class of one benchmark. No code. |
| **ARES** (arXiv 2507.12948, EMNLP 2025) | PRMBench (unstated subset), DeltaBench, ClaimTrees (synthetic), CaptainCook recipes | Entailment by GPT-4o-mini or Qwen3-4B; a Qwen-PRM-7B variant in an appendix | Threshold tuned by 5-fold CV on labels | 4 datasets, 2 entailment models. Its own appendix: the PRM alone beats ARES on PRMBench (.749 vs .736). |
| **Mind the Gap** (ICML 2026) | GSM8K and MATH (final-answer correctness); ProcessBench (exact first-error localization, not official F1) | Qwen3-4B and Qwen3-8B | 50/50 calibration split for the thresholds | 2 tasks, 2 model sizes of one family. No PRMBench. The released code covers final answers only. |
| **BR2-PRM** (ACL 2026) | PRMBench-V (multimodal, their own benchmark) and Best-of-N on MMK12 | 7 MLLM judges; policies Gemma-3-12B and LLaVA-1.5-7B | Supervised naive Bayes on the benchmark's labels | One benchmark plus one downstream task. No code found. |
| **R-PRM** (EMNLP 2025) | ProcessBench, PRMBench; Best-of-8 and guided search | 1 generative PRM (from Qwen2.5-Math-7B-Instruct) | PRM800K plus distillation from a 70B teacher | 2 benchmarks plus search. The ACL version drops the stronger Qwen-PRM baseline. |
| **Ours: TELE16 / AUTO / CHAIN** | 8 benchmarks: PRMBench, Socratic-PRMBench, Hard2Verify, ProcessBench, CRV-Arith, CRV-GSM8K, PRM800K, DeltaBench | 5 PRMs (Qwen-7B, Qwen-72B, PURE, RLHFlow, Math-Shepherd; 4 on the CROP/ARES sets) × 2 telemetry sources (PRM backbone, standalone base model) | None in the fusion; the PRMs were trained by their authors | One recipe fixed on PRMBench and applied unchanged everywhere else, under each competitor's own protocol, with paired bootstrap CIs. **Result:** strong on PRMBench (in sample); out of domain it helps weak PRMs and loses to strong ones. |

**What this means for a "generalization" claim.**
- **Breadth:** our evaluation is the broadest here, with more benchmarks × models than any competitor. Only CROP comes close (6 sets, 1 PRM).
- **No competitor shows a method that beats a strong PRM across benchmarks without per-dataset labels:**
  - CROP recalibrates on every dataset with labels and fails across domains.
  - PRISM's gain does not carry to ProcessBench.
  - ARES loses to the PRM in its own appendix.
- **Our honest statement:** one label-free recipe, applied unchanged, gives consistent gains on PRMBench and gains with weak PRMs elsewhere, but not with strong PRMs out of domain. The supervised LR in this directory shows that labels from another dataset do not fix that either.
