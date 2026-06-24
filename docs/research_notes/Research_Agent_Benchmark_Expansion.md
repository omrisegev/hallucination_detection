# Research Brief: Expanding Agentic Hallucination Detection Beyond RAG

**For**: Gemini Deep Research  
**From**: Omri Segev (TAU MSc Thesis)  
**Date**: 2026-05-15  
**Purpose**: Identify the best agentic benchmarks and tasks to extend our hallucination detection experiments, with a focus on SOTA comparability.

---

## What We Have Built

We detect hallucinations in LLM-based agents using **spectral features of the per-token entropy trajectory H(n)**. During inference, we capture the model's raw logit distribution at every generated token (via HuggingFace `model.generate(output_scores=True)`). From these logits we compute per-token Shannon entropy, producing a 1D signal H(1), H(2), …, H(N). We then extract 16 spectral and time-domain features from this signal (FFT energy bands, STFT variance, CUSUM regime shifts, Hurst exponent, permutation entropy, etc.) and fuse them using **Nadler spectral fusion** — an unsupervised covariance-weighted leading-eigenvector method that requires no labels at test time.

**Critical constraint**: We must run inference ourselves using a locally-loaded open-source model. We cannot use pre-collected trajectory datasets (like AgentHallu) because those contain no log probabilities. We cannot use closed API models (GPT-4o, Gemini) unless they expose full per-token logprobs, which most do not. Our models run on an NVIDIA A100 GPU on Google Colab.

### Current Agentic Experiment (Phase 11a)

We implement a **3-step ReAct loop** on multi-hop QA (HotpotQA, 2WikiMultiHopQA). At each step the model emits:

```
Thought: <reasoning>
Action: search("<query>") or finish("<answer>")
Confidence: <float in [0,1]>  
Concern: <risk description>
```

The "tool" is a **simulated retriever**: word-overlap scoring over the gold context paragraphs provided with each question. No external API. No internet access. The label is whether the final answer matches the gold answer (substring match).

Per step we extract the full entropy trace H(n), compute 16 spectral features + `branching_entropy` (first-3-token entropy, a step-boundary signal), and aggregate across steps using min/avg/last. We compare against the **AUQ baseline** (Zhang et al. 2026, arXiv:2601.15703): verbalized confidence aggregated the same way.

**Current SOTA we are trying to beat**:
- AUQ Φ_min AUROC = **0.791** on ALFWorld (Salesforce AI, 2026)
- AUQ Φ_min AUROC = **0.755** on WebShop
- LOS-Net AUC = **72.92%** on HotpotQA (supervised, arXiv:2503.14043)

---

## What We Want to Extend To

### The Core Thesis Extension

Our current experiment uses one type of tool: **document retrieval**. The agent reads text and cites/answers from it. This is essentially a structured RAG loop. We want to test whether spectral entropy features generalize to **qualitatively different tool-use environments** — specifically:

1. **Code execution** — the agent writes and runs code; the tool is a Python interpreter
2. **Structured environment interaction** — the agent navigates a state-machine world (household tasks, web navigation, software workspace)
3. **Multi-turn tool APIs** — the agent calls external tools with structured arguments across many turns

The hypothesis is that **hallucination always manifests as uncertainty in the entropy trajectory**, regardless of the domain. A model that fabricates a Python function name, invents a non-existent bash command, or hallucinates a product ID should show a measurable entropy signature at the moment of fabrication — just as a model citing a non-existent paper does in the RAG setting.

---

## The SOTA Comparison Requirement (Critical)

**This is the most important constraint for benchmark selection.**

For a result to be publishable and presentable to advisors, we must be able to say: *"On benchmark X, our method achieves Y%, compared to SOTA method Z which achieves W%."*

This means we need benchmarks where:
1. **SOTA hallucination detection / uncertainty quantification results already exist** with reported AUROC, accuracy, or step-localization metrics
2. **The benchmark is publicly available** and we can run our own inference on it (not just download pre-collected trajectories)
3. **The environment is simulatable locally** — no paid APIs, no Docker requiring terabytes of data, no browser automation that breaks on Colab

Without a SOTA comparison target, a result like "our method achieves 74% on benchmark X" has no meaning because we don't know if 74% is good or trivial.

---

## Research Questions for Gemini

### Question 1: Agentic Benchmarks With Reported Uncertainty / Hallucination Detection SOTA

For each benchmark below (and any others you identify), we need to know:
- Is there a published AUROC or accuracy for hallucination/uncertainty detection on this benchmark?
- Which paper(s) report it, and what is the metric and score?
- Is the benchmark runnable locally (open-source simulator, no paid API)?
- What does the agent's action space look like? (retrieval, code execution, tool calls, navigation...)

**Known candidates to investigate:**

| Benchmark | Known result | Notes |
|---|---|---|
| **ALFWorld** | AUQ Φ_min AUROC=0.791 | Household tasks, text-based simulator, `pip install alfworld` |
| **WebShop** | AUQ Φ_min AUROC=0.755 | E-commerce, open-source product catalog |
| **HumanEval / HumanEval+** | ? | Python code generation, unit test grading |
| **InterCode** | ? | Interactive code execution (Python, bash, SQL) |
| **MBPP / MBPP+** | ? | Python code generation |
| **SWE-bench Lite** | ? | GitHub issues, requires Docker |
| **τ-bench** | ? | Multi-turn tool use |
| **AgentBench** | ? | 8 environments including OS, DB, web |
| **TheAgentCompany** | ? | Virtual software-company tasks |
| **AppWorld** | ? | 750 tasks across 9 simulated apps |
| **MuSiQue** | ? | 4-hop multi-hop QA, same format as HotpotQA |
| **GAIA / GAIA2** | ? | General AI assistants tasks |
| **Mind2Web** | ? | Real-world web navigation |

### Question 2: Hallucination Detection Methods in Agentic Settings (2024–2026)

What methods have been published for detecting hallucinations or calibrating uncertainty in agentic / multi-step LLM systems? For each:
- Method name, paper, arXiv ID
- What signal they use (internal representations, verbalized confidence, consistency sampling, entropy-based, other)
- What benchmark(s) they report on and the metric/score
- Whether it requires training/fine-tuning or is training-free
- Whether it requires black-box access (output only) or gray-box (logprobs)

Known methods to cross-check against:
- AUQ (arXiv:2601.15703) — verbalized confidence, training-free, gray-box
- LOS-Net (arXiv:2503.14043) — supervised, HotpotQA
- SAUP (arXiv:2412.01033) — step-aware uncertainty propagation
- SIA (arXiv:2604.06192) — conditional answer entropy
- MARCH (arXiv:2603.24579) — multi-agent pipeline
- UProp (arXiv:2506.17419) — uncertainty propagation
- Streaming Prefix-Level (arXiv:2601.02170)

### Question 3: Code Execution Agents — Hallucination and Uncertainty

Code generation is a natural extension: the model "hallucinates" by writing code that references non-existent functions, wrong APIs, or incorrect logic. The label is automatic (unit test pass/fail). 

- What benchmarks exist for **iterative code-fix agents** (model writes code, sees error, fixes it — multi-step)?
- Is there published uncertainty quantification or hallucination detection on these benchmarks?
- What is the action space (Python only? Bash? Multi-language?)?
- Can the execution environment be set up in a Google Colab Python runtime without Docker?
- Which specific datasets have HuggingFace entries and can be loaded with `datasets.load_dataset()`?

### Question 4: Tool-Call Hallucination — Structured APIs

Some recent benchmarks test whether agents call tools with correct arguments (function name, parameter values). Hallucination = fabricating a tool name or argument that doesn't exist.

- What benchmarks exist for **tool-call hallucination** specifically?
- Is there a hallucination detection SOTA on these?
- Do any provide a local simulator (no external API)?
- Notable: ToolBench/ToolEval, BFCL (Berkeley Function-Calling Leaderboard), Gorilla

### Question 5: Feasibility Triage

Given our constraints (A100 Colab, HuggingFace open-source models 7B–8B, local environment, must run own inference), rank the top 5 candidate benchmarks by:

1. **SOTA comparability** — existing published detection results we can beat
2. **Setup effort** — how hard to get running on Colab (pip install? Docker? Paid API?)
3. **Scientific novelty** — does it add a qualitatively new tool-use modality beyond retrieval?
4. **Thesis coherence** — does it fit a "spectral features generalize across agent types" narrative?

---

## What a Good Answer Looks Like

For each recommended benchmark, we need:

```
Benchmark: <name>
Task type: <retrieval / code-execution / navigation / tool-call / ...>
HuggingFace dataset: <path or "not on HF">
Local simulator: <pip package / GitHub repo / Docker required>
Action space: <what actions the agent takes>
Label: <how correctness is determined>
SOTA hallucination detection: <paper, metric, score — or "none found">
Setup effort on Colab: <low / medium / high>
Why it fits our method: <1-2 sentences>
```

---

## Context: Our Existing Results

For reference, what we have shown so far:

| Domain | Best AUC | Model | Notes |
|---|---|---|---|
| MATH-500 (reasoning) | **90.0%** | Qwen2.5-7B | Spectral Nadler |
| GSM8K (arithmetic) | **76.0%** | Llama-3.1-8B | vs LapEigvals 72.0% |
| GPQA Diamond (science MCQ) | **69.0%** | Qwen2.5-72B-AWQ | Limited by MCQ structure |
| L-CiteEval RAG (HotpotQA) | **79.5%** | Qwen2.5-7B | Citation grounding |
| L-CiteEval RAG (2WikiMultiHop) | **80.5%** | Qwen2.5-7B | Citation grounding |
| Agentic ReAct (Phase 11a) | TBD — running now | Qwen2.5-7B + DeepSeek-R1-7B | vs AUQ 0.791 |

The narrative thread: spectral features of H(n) detect hallucination across domains (math, science, RAG, agentic). Each new domain that works strengthens the generalization claim.
