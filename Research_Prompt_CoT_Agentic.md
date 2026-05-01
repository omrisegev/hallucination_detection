# Research Prompt: SOTA for CoT and Agentic Hallucination Detection

---

## Context (read before answering)

I am writing an MSc thesis on hallucination detection in LLMs. The core framework is **Nadler spectral fusion**: an unsupervised algorithm that aggregates multiple uncertainty "views" of the same question by estimating each view's reliability from the covariance structure of the signals alone — no ground-truth labels needed at inference time. Nadler requires two conditions to work: (1) views must make errors on different questions (low pairwise correlation), and (2) all views must predict the same underlying correctness signal.

My working constraints are:
- **Gray-box access only**: I can read log-probabilities, first-token distributions, and generated text — but no weight access, no fine-tuning, no hidden state hooks (i.e., must work through an API or standard `generate()` interface)
- **Single pretrained model** — primarily Falcon-3-10B; also tested Ministral-8B and Phi-4
- **Unsupervised or self-supervised** — no ground-truth correctness labels at inference time
- **Practical**: must run on a single A100 in ~10s per forward pass

The uncertainty measures I have already implemented and tested:
- **EPR (Entropy Production Rate)**: mean token-level entropy of a single greedy generation (artefactual library, K=15 log-probs, T=1.0 to T=2.0)
- **Semantic Entropy**: sampling-based entropy over NLI-clustered answer distributions (Farquhar et al., Nature 2024)
- **Verification/Skeptic signals**: first-token P(Yes) from reflective prompts ("Is this answer correct?" / "Does this answer contain errors?")
- **Temperature-varied EPR**: same question at T=0.3 / 1.0 / 1.5 / 2.0 as four decorrelated views

Key finding so far: the best-performing configuration (4 temperature-varied EPR views + Verify + Skeptic behavioral views fused with Nadler) achieves +7.8% AUC over single-view EPR on WebQuestions. All prompt-template variation approaches failed because they violate Nadler's conditional independence requirement (pairwise Spearman ρ > 0.8). Multi-model ensemble failed because it violates the common-target requirement.

The **next experimental direction** is applying this multiview fusion framework to **Chain-of-Thought (CoT) reasoning traces and agentic workflows**, where the model generates intermediate reasoning steps before the final answer. The intuition: EPR computed on the reasoning trace may be decorrelated from EPR on the final answer, providing a new dimension of uncertainty signal. An agentic pipeline (multi-step tool use, reflection, self-critique) exposes even more structural signals at each step.

---

## Research Questions

Please survey the literature (2021–2025) and answer the following, with specific paper citations:

### 1. CoT-specific hallucination detection

- What are the current SOTA methods for detecting hallucinations specifically in **chain-of-thought reasoning traces** (not just the final answer)?
- Do any methods compute uncertainty or confidence **per reasoning step** rather than per output? How do they aggregate step-level uncertainty into a final hallucination score?
- Is there evidence that reasoning-trace uncertainty (e.g. entropy of intermediate steps) is informative beyond final-answer uncertainty? Are they empirically decorrelated?
- What benchmarks are used to evaluate CoT hallucination detection specifically?

### 2. Agentic flow validation

- What methods exist for validating the reliability of **agentic LLM pipelines** (multi-step, tool-using agents)?
- How do current approaches handle **error propagation** in agentic flows — where a hallucination in step 2 corrupts all downstream steps?
- Are there methods that assign a **per-step reliability or uncertainty score** to agentic traces, and aggregate these into an overall confidence? How do they do the aggregation?
- What are the specific failure modes of agentic systems beyond factual hallucination — e.g., goal drift, premature stopping, incorrect tool calls — and are there uncertainty-based detectors for these?

### 3. Uncertainty signals from CoT traces

In the gray-box setting (log-probabilities + generated text, no weight access), what signals from a CoT trace have been shown to be informative for hallucination detection?

Specifically, I am interested in:
- **Token-level entropy over the reasoning trace** vs over the final answer — are they correlated or orthogonal?
- **Step-boundary uncertainty** — is entropy at transition points between reasoning steps (e.g. "Therefore...", "So the answer is...") a useful signal?
- **Self-consistency of reasoning** — e.g., does the chain support the final answer? Methods that verify internal logical coherence without external knowledge.
- **Length and verbosity** — do longer/shorter chains correlate with confidence?
- **Contradiction detection within a trace** — identifying when a model contradicts itself mid-chain.

### 4. Multiview / ensemble approaches on CoT

- Are there existing methods that apply **multi-view or ensemble uncertainty aggregation** to CoT traces?
- Has any work applied uncertainty fusion (similar to Nadler, DPP-based selection, or other aggregation) to signals extracted from different parts of a CoT trace (e.g., early steps vs late steps, multiple sampled traces)?
- Do any papers compare entropy signals from the **reasoning trace vs the final answer** as complementary views? What fusion method do they use?

### 5. Practical recommendations for my setting

Given my constraints (gray-box, single model, unsupervised, API-compatible):

- What is the **most promising underexplored signal** from CoT traces that I could add as a new Nadler view alongside my existing temperature-varied EPR and behavioral views?
- Are there any methods that extract meaningful **scalar uncertainty signals from a CoT trace** using only log-probabilities from the standard `generate()` call?
- What is the theoretical argument for why CoT trace uncertainty should be decorrelated from final-answer EPR? Under what conditions does this hold or break down?
- Are there papers that specifically address the **gray-box, single-model, no-fine-tuning** hallucination detection setting — and what do they recommend as best practice?

---

## Output format requested

Please structure your response as follows:

1. **Key papers to read** — up to 10, each with a 2-3 sentence summary of what they contribute and why it's relevant to my setting
2. **Signal inventory** — a table of CoT/agentic uncertainty signals that are extractable in the gray-box setting, with: signal name, how to compute it, approximate compute cost, and whether there is evidence it is decorrelated from token-level entropy
3. **Recommended approach** — given my thesis framework (Nadler fusion, existing views, gray-box constraint), what is the single most promising direction for a new CoT-based view? Be specific about the signal, how to extract it, and what prior work supports it
4. **Gaps** — what is missing from the literature that my thesis could potentially fill?
