# State of the Art in LLM Hallucination Detection for Reasoning Tasks (as of July 2026): A Benchmarking Guide for Unsupervised Gray-Box Methods

## TL;DR
- **The field is dominated by unsupervised gray-box and sampling-consistency methods (semantic entropy and its descendants, INSIDE/EigenScore, and self-consistency variants), but almost none of the canonical papers report AUROC directly on MATH/GSM8K — those math-reasoning numbers exist only in recent (2025–2026) re-evaluations, which is precisely the gap your method can own.** The strongest directly-comparable unsupervised gray-box baseline on GSM8K today is Noise Injection (arXiv:2502.03799), which lifts AUROC on GSM8K from 76.53 to 82.70 on Llama-3.2-3B-Instruct; EigenScore reaches ~63.4 (GSM8K) and ~81.4 (MATH-500) on Qwen3-8B in re-evaluation.
- **For a fair related-work table, group methods into four categories:** (1) unsupervised gray-box/black-box uncertainty (semantic entropy, LN-entropy, KLE, SelfCheckGPT, perplexity/P(True), Noise Injection); (2) white-box internal-state methods (INSIDE/EigenScore, LLM-Check, SAPLMA, MIND, Lookback Lens, spectral/attention methods, ReDeEP); (3) supervised/semi-supervised probes (HaloScope, Semantic Entropy Probes, TSV); (4) reasoning-specific detectors and process-error benchmarks (RACE, FG-PRM, ProcessBench, MR-GSM8K, ReasonEval, PRMBench).
- **The most useful benchmarks for a MATH-reasoning-focused unsupervised gray-box detector are GSM8K/MATH themselves (with correctness-derived hallucination labels), plus dedicated math-error benchmarks ProcessBench, MR-GSM8K, PRMBench, and FG-PRM's synthetic set; for general hallucination-detection evaluation, use HaluEval, TruthfulQA, HaluLens, RAGTruth, LLM-Uncertainty-Bench, and the HELM benchmark from MIND.** Be aware of a major 2025 caveat: ROUGE-based correctness labels dramatically overstate detector AUROC versus human/LLM-judge labels.

## Key Findings

1. **"Gray-box" is now a well-established category** distinct from black-box (output text only) and white-box (hidden states/attention/full internals). Gray-box specifically means access to output token probabilities/logits but not necessarily deeper internals. Your method, using token probabilities/logits/entropy without labeled detector training, sits squarely in the unsupervised gray-box category alongside semantic entropy, LN-entropy, perplexity, and P(True).

2. **Semantic entropy (Farquhar et al., Nature 2024; Kuhn et al., ICLR 2023) is the reference unsupervised gray-box method.** It clusters semantically-equivalent sampled generations (via NLI bidirectional entailment) and computes entropy over meaning clusters. It requires ~10 generations per prompt (a 5–10× inference overhead); the K=10 sampling protocol from Farquhar et al. (Nature 630, 625–630, 2024) is now standard and is replicated by downstream work such as Noise Injection ("We generate K=10 samples for each question and compute answer entropy"). Farquhar et al. report that "Semantic entropy outperforms leading baselines and naive entropy" using AUROC that "measures how well methods predict LLM mistakes" (Fig. 2), averaged over five datasets; on TriviaQA it reaches AUROC ~0.83 versus ~0.80 for length-normalized entropy and much higher than P(True).

3. **Most canonical methods were NOT evaluated on math reasoning.** Semantic entropy, INSIDE/EigenScore, HaloScope, KLE, SEPs, and SelfCheckGPT were evaluated on QA datasets (TriviaQA, SQuAD, NQ, CoQA, TruthfulQA, BioASQ) — not GSM8K/MATH. This is a genuine gap: math-reasoning hallucination detection with unsupervised gray-box signals is under-explored, and the few numbers that exist come from 2025–2026 re-evaluations (Noise Injection; arXiv:2601.17467; arXiv:2510.11529).

4. **A key 2025 methodological warning:** Janiak et al., "The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs" (EMNLP 2025 main, pp. 34716–34733; arXiv:2508.08285), show that established detectors collapse under human-aligned labels: "Perplexity sees its AUROC score plummet by as much as 45.9% for the MISTRAL model on NQ-Open. Similarly, Eigenscore's performance erodes by 19.0% and 30.4% for LLAMA and MISTRAL… Even eRank… experiences a sharp decline of 30.6% and 36.4% under the LLM-as-Judge paradigm." Simple length heuristics can rival semantic entropy under ROUGE labels. Report both label schemes if possible.

## Details

### (a) Method comparison table

Categories: **UGB** = unsupervised gray-box (logits/entropy/consistency, no detector training); **BB** = black-box (text only); **WB** = white-box (hidden states/attention); **SUP** = supervised/semi-supervised (trained probe/classifier); **PB** = prompting-based.

| Method (link) | Category | Evaluation setup (model + dataset + #samples) | AUROC (or other metric) |
|---|---|---|---|
| **Semantic Entropy** — Farquhar et al., Nature 2024 (nature.com/articles/s41586-024-07421-0); Kuhn et al. ICLR 2023 (arXiv:2302.09664) | UGB (sampling + NLI clustering) | LLaMA-2-70B, Falcon, Mistral; TriviaQA, CoQA, SQuAD, NQ, SVAMP, others; K=10 generations/prompt | TriviaQA AUROC ~0.83 (vs ~0.80 LN-entropy); "outperforms leading baselines and naive entropy" averaged over 5 datasets; consistently > P(True) |
| **LN-Entropy (length-normalized predictive entropy)** — Malinin & Gales 2021; baseline in most papers | UGB | Same QA datasets as above | Typically 0.75–0.80 on TriviaQA (baseline) |
| **Perplexity / sequence log-prob** — standard baseline | UGB | Ubiquitous baseline | ~0.62 on clinical QA vs 0.76 for SE; AUROC drops up to 45.9% (Mistral, NQ-Open) under LLM-as-judge labels; weak on math |
| **P(True) self-evaluation** — Kadavath et al. 2022 (arXiv:2207.05221) | PB / UGB | Anthropic LMs; MC + open-ended (incl. math, code) | AUROC 0.51–0.61 in recent re-tests (near-random); scales with model size |
| **Verbalized confidence** — Lin et al. 2022; Xiong et al. 2024 | PB | GPT-4 and others; general tasks | ~62.7% avg AUROC for GPT-4, "close to random guess" on hard tasks |
| **SelfCheckGPT** — Manakul et al., EMNLP 2023 (arXiv:2303.08896) | BB (sampling consistency) | GPT-3; WikiBio; also re-eval on GSM8K/MathQA/MATH | GSM8K (Qwen2.5-7B re-eval) ~67.98 AUROC; original WikiBio uses AUC-PR |
| **SelfCheck (Miao et al. 2023)** — zero-shot step-check (arXiv:2308.00436) | BB / PB | GPT-3.5, GPT-4; GSM8K (1319), MathQA (2985), MATH (subset) | Reports final-answer accuracy improvement, not AUROC |
| **Kernel Language Entropy (KLE)** — Nikitin et al., NeurIPS 2024 (arXiv:2405.20003) | UGB / BB (von Neumann entropy of semantic kernel) | Multiple LLMs; NLG datasets | Improves over SE across datasets (AUROC); no GSM8K/MATH |
| **INSIDE / EigenScore** — Chen et al., ICLR 2024 (arXiv:2402.03744) | WB (covariance eigenvalues of internal embeddings) | LLaMA-7B, OPT; CoQA, SQuAD, NQ, TriviaQA; multiple samplings | AUC per dataset; re-eval on GSM8K ~63.4, MATH-500 ~81.4 (Qwen3-8B); AUROC erodes 19–30% under LLM-as-judge |
| **HaloScope** — Du et al., NeurIPS 2024 spotlight (arXiv:2409.17504) | SUP (trains MLP on unlabeled-mixture membership; no manual labels) | LLaMA-2-7B/13B-chat, OPT-6.7B/13B; TruthfulQA, TriviaQA, CoQA, TydiQA | TruthfulQA AUROC 78.64% (in-domain), 76.26% (transfer from TriviaQA); "favorably matches the supervised oracle (AUROC: 81.04%)"; a 10.69% increase over baselines on TruthfulQA (LLaMA-2-7b-chat) |
| **Semantic Entropy Probes (SEPs)** — Kossen et al. 2024 (arXiv:2406.15927) | SUP (linear probe on hidden states, approximates SE) | LLaMA-2-7B, Mistral-7B, Phi-3; TriviaQA, SQuAD, BioASQ, NQ Open | AUROC 0.7–0.95 depending on scenario; ~0.83 vs 0.65 for token entropy (TriviaQA) |
| **SAPLMA** — Azaria & Mitchell 2023 (arXiv:2304.13734) | SUP + WB (probe on hidden layer) | facts datasets | Truthfulness classification accuracy; ~0.7 AUROC in re-tests |
| **MIND** — Su et al., ACL Findings 2024 (arXiv:2403.06448) | WB, unsupervised training (auto-generated data, no manual labels) | Multiple LLMs; HELM benchmark (introduced) | Up to 98.55% AUROC on HaluEval/TriviaQA (English) per follow-ups; outperforms SOTA |
| **LLM-Check** — Sriramanan et al., NeurIPS 2024 (OpenReview LYx4w3CAgy) | WB/UGB (eigenvalue analysis of attention/hidden + output token uncertainty; single forward pass) | LLaMA-2; FAVA (zero-resource), SelfCheck, RAGTruth settings | AUROC across settings; strong zero-resource single-pass performance, no train/inference overhead |
| **Lookback Lens** — Chuang et al., EMNLP 2024 (arXiv:2407.07071) | WB/SUP (logistic regression on attention lookback ratio) | LLaMA-2-7B/13B-chat; NQ (QA), CNN/DM (Summarization) | AUROC on contextual hallucination; transfers 7B→13B without retraining |
| **Spectral Attention (LapEigvals)** — Binkowski et al., EMNLP 2025 (arXiv:2502.17598) | WB/SUP (Laplacian eigenvalues of attention maps + probe) | LLaMA-3.1-8B and others | SOTA vs tested attention-based probes (ROC-AUC, 5-fold CV) |
| **ReDeEP** — Sun et al., ICLR 2025 spotlight (arXiv:2410.11414) | WB (mechanistic: external context vs parametric knowledge) | LLaMA-2-7B/13B, LLaMA-3-8B; RAGTruth, Dolly(AC) | Best on RAGTruth; unsupervised, single forward pass |
| **Noise Injection** — Liu et al. 2025 (arXiv:2502.03799) | UGB (training-free epistemic-uncertainty via activation perturbation, uniform noise U(0,0.07) into MLP activations of layers 20–32; unsupervised, question-level labels by majority-vote correctness) | Gemma-2B, Llama-3.2-3B-Inst, Phi-3-mini, Mistral-7B, Llama-2-7B-chat; **GSM8K**, CSQA, TriviaQA; K=10 | **GSM8K: Llama-3.2-3B 76.53→82.70; Gemma-2B 51.36→57.11; Phi-3-mini 65.86→72.51; Mistral-7B 75.85** |
| **RACE** — Wang et al., AAAI 2026 (arXiv:2506.04832) | BB/gray-box, reasoning+answer consistency (LRM-focused) | DeepSeek-R1-Distill-Qwen/Llama 7–14B, Qwen3-14B, QwQ-32B, DeepSeek-R1, Qwen2.5-14B; TriviaQA, SQuAD, NQ-Open, HotpotQA (**no GSM8K/MATH**) | Reports AUROC (Table 1); best overall vs LNPE/SE/SINdex/SelfCheckGPT/P(True); exact cells to verify in PDF |
| **TSV (Steer LLM Latents)** — Park et al., ICML 2025 (arXiv:2503.01917) | SUP/semi-sup (32 labeled exemplars + unlabeled OT pseudo-labels; latent steering) | LLaMA-3.1-8B/70B, Qwen-2.5-7B/14B; TruthfulQA, SciQ | TruthfulQA 84.2 (vs 85.5 fully-sup upper bound); SciQ 89.7 (Qwen2.5-14B) vs 82.0 (7B); +12.8% over SOTA |
| **FG-PRM** — Li et al. 2024 (arXiv:2410.06304) | SUP (process reward model on synthetic fine-grained hallucinations) | fine-tuned PRM; **GSM8K, MATH**; 700 train/100 test augmented to 12K | Beats GPT-3.5/Claude-3 by >5% F1 on fine-grained detection; F1 metric (not AUROC) |

### (b) Benchmarks and datasets for evaluating a MATH-reasoning unsupervised gray-box detector

**Math/reasoning-specific (most relevant to your method):**
- **GSM8K** (Cobbe et al. 2021; huggingface.co/datasets/gsm8k) — 8.5K grade-school word problems (7.5K train / 1,319 test). The de facto math-reasoning testbed; hallucination labels derived from answer correctness. Used directly by Noise Injection and math re-evaluations.
- **MATH / MATH-500** (Hendrycks et al. 2021) — competition-level problems with step-by-step solutions; MATH-500 is the common 500-problem eval subset. Used in EigenScore re-eval (~81.4 AUROC, Qwen3-8B).
- **ProcessBench** (Zheng et al. 2024, arXiv:2412.06559; github.com/QwenLM/ProcessBench) — "The resulting PROCESSBENCH has four subsets, consisting of 3,400 test cases in total" across GSM8K, MATH, OlympiadBench, and Omni-MATH, with "200 each for GSM8K and 500 each for other subsets," balanced correct/incorrect, expert-annotated first-error-step labels, scored by F1.
- **MR-GSM8K** (Zeng et al. 2023, arXiv:2312.17080) — 3,000 instances; meta-reasoning task (predict solution correctness, first error step, error reason); MR-Score metric.
- **PRMBench** (Song et al. 2025, arXiv:2501.03124) — fine-grained process-error benchmark with error types across simplicity/soundness/sensitivity dimensions.
- **ReasonEval / MR-MATH** (Xia et al., AAAI 2025 oral, arXiv:2404.05692; github.com/GAIR-NLP/ReasonEval) — evaluates reasoning-step validity and redundancy; ReasonEval-7B/34B evaluators.
- **FG-PRM synthetic set** (arXiv:2410.06304) — GSM8K/MATH augmented to 12K instances with six hallucination types.
- **MathCheck-GSM** — LLM-synthesized erroneous steps from GSM8K for step verification.

**General hallucination-detection benchmarks (for breadth):**
- **HaluEval** (Li et al., EMNLP 2023) — 35K samples (QA/dialogue/summarization + general); widely used, includes HaluEval-QA (10K).
- **TruthfulQA** (Lin et al. 2022) — 817 questions targeting common misconceptions; the standard hallucination-detection AUROC benchmark.
- **HaluLens** (Bang et al., ACL 2025, arXiv:2504.17550) — Meta FAIR; extrinsic (LongWiki, PreciseQA, Nonsense) + intrinsic tasks with dynamic test-set regeneration to prevent leakage. Codebase released.
- **RAGTruth** (Niu et al., ACL 2024) — human-annotated RAG hallucination corpus (QA, data-to-text, summarization); LLaMA-2/Mistral/GPT generations.
- **LLM-Uncertainty-Bench** (Ye et al. 2024, arXiv:2401.12794; github.com/smartyfh/LLM-Uncertainty-Bench) — benchmarks LLMs via conformal prediction across QA/reading comprehension/commonsense/summarization; metrics include coverage rate, set size, UAcc.
- **HELM** (introduced with MIND, arXiv:2403.06448) — hallucination-detection benchmark providing LLM outputs AND internal states across multiple LLMs — uniquely useful for comparing gray-box vs white-box.
- **FELM, FActScore, HaluBench (15K)** — factuality benchmarks for long-form/atomic-fact evaluation.
- **LM-Polygraph** (Fadeeva et al.) — open-source UQ benchmarking toolkit implementing many gray-box baselines; useful for reproducible comparison.

## Recommendations

1. **Anchor your related-work table on the four categories above and explicitly claim the math-reasoning gap.** Since semantic entropy, INSIDE, HaloScope, KLE, and SEPs did not originally report GSM8K/MATH, your unsupervised gray-box method that is best on MATH occupies genuinely open territory. State this explicitly and cite the 2025–2026 re-evaluations (Noise Injection; arXiv:2601.17467; arXiv:2510.11529) for the few existing math AUROC numbers.

2. **Benchmark primarily on GSM8K and MATH-500 with correctness-derived labels, reporting AUROC**, and directly compare against: (i) semantic entropy, (ii) LN-entropy, (iii) perplexity, (iv) P(True) — all unsupervised gray-box; plus (v) EigenScore and (vi) SelfCheckGPT as white/black-box references. These are the standard, reproducible baselines and several have public code.

3. **Add a process-level benchmark (ProcessBench or MR-GSM8K) as a secondary evaluation** to demonstrate step-level localization, which strengthens a reasoning-focused paper beyond final-answer detection.

4. **Report robustness to the label-scheme issue.** Given the EMNLP 2025 finding (Janiak et al.) that ROUGE overstates AUROC by up to 45.9% (Perplexity, Mistral, NQ-Open) and 19–36% for EigenScore/eRank, evaluate with an LLM-as-judge or exact-match correctness label and note the difference; this preempts a common reviewer criticism.

5. **Use HELM (from MIND) or LM-Polygraph for a controlled gray-box-vs-white-box comparison** if you want to argue your gray-box method approaches white-box performance without internal access.

**Thresholds that would change these recommendations:** If your method requires multiple samples (like semantic entropy), emphasize efficiency comparisons (SEPs, LLM-Check, Noise Injection all address cost). If it can run single-pass, position against SEPs/LLM-Check directly. If you gain access to hidden states, you shift to the white-box category and should compare against INSIDE, MIND, and spectral-attention methods.

## Caveats
- **RACE's exact per-cell AUROC values could not be independently verified** from the accessible sources, and RACE evaluates on factual QA (TriviaQA/SQuAD/NQ/HotpotQA), NOT GSM8K/MATH — do not cite it as a math-reasoning detector, though it is highly relevant as a reasoning-consistency framework for large reasoning models.
- **TSV is not fully unsupervised** — it uses 32 labeled exemplars plus unlabeled data (semi-supervised); classify it accordingly.
- **HaloScope and MIND are "unsupervised" in the sense of no manual labels, but they train a classifier/probe** — they are not training-free like semantic entropy or your gray-box method. Label them precisely.
- **The original INSIDE/EigenScore and semantic-entropy papers did not report GSM8K/MATH AUROC**; those math numbers come from later re-evaluation papers with different setups, so cross-setup comparisons are approximate.
- **Metrics vary:** many process/reasoning papers (FG-PRM, ProcessBench, ReasonEval, MR-GSM8K) report F1/MR-Score/accuracy rather than AUROC because they do error-step localization, not binary hallucination scoring. Match the metric to the task.
- Several 2026 arXiv IDs surfaced in citations (e.g., 2601.xxxxx, 2604.xxxxx, 2606.xxxxx) reflect very recent preprints whose numbers may not be peer-reviewed yet; treat as provisional and verify before citing in a camera-ready paper.