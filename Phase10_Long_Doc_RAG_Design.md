# Phase 10 — Long-Document RAG-Agentic Design Notes

**Date**: 2026-05-07
**Status**: Pre-research design discussion. No experiment committed yet.
**Related**: Direction 2 (RAG), Direction 4 / Phase 10 (Agentic), HISTORY Steps 78 & 82, `Spectral_Analysis_Phase6.ipynb`

This document captures the design discussion held before launching `/research` on the unified RAG + agentic question. It feeds into Research_Directions.md once a plan is committed.

---

## Context — where the spectral chapter stands after Phase 6 + 9

| Domain | Result | Status |
|---|---|---|
| Math (MATH-500, GSM8K) | 76–96.6% AUC | ✓ strong |
| Science MCQ at scale (GPQA, Qwen-72B-AWQ) | 69.0% (+3.6pp over 7B) | ✓ marginal — G1 caveat (acc 40.4%) |
| Single-hop factual QA (TriviaQA, WebQ CoT) | 53.6% / 61.9% | ✗ clean negative |
| Multi-hop QA (HotpotQA, Mistral-7B) | 59.5%, 6/7 gates failed | ✗ clean negative |

**Scope claim is hardened**: spectral features need *long structured reasoning traces* to be informative. This design doc asks whether long-document multi-iteration RAG / agentic settings restore those conditions, opening a path to extend the thesis beyond reasoning-only domains *without contradicting* Phases 6 and 9.

---

## The hypothesis

In a long-doc agentic RAG setup (e.g., 10K-token document + N-iteration ReAct loop + multi-claim output):

1. **Trace length is restored.** Output is 500–2000 tokens, comparable to MATH-500 (478–1151) where spectral features hit 96.6%. The structural precondition that killed factual QA in Phase 9 (median valid trace = 50 tok) is gone.
2. **Grounded↔parametric transitions create local entropy bursts.** When the model paraphrases from the doc, the next-token distribution is sharp and entropy is low and stable. When it fabricates from parametric memory, the distribution flattens and entropy spikes locally. That is exactly the local-burst pattern `sw_var_peak` was designed to catch — and the same pattern that made it work at step boundaries in math.
3. **Iterations create natural step structure.** read → reason → extract → verify, repeated. Closer to math step-boundary periodicity than to single-shot QA.

**Sharp research question**: does the entropy spectrum of a long-doc agent's output carry *claim-level* hallucination signal — can it localize *where* in the output the model ungrounded?

---

## Why this is a better RAG framing than the original Direction 2

Direction 2 originally proposed TriviaQA + Wikipedia passage + EPR(with context) vs EPR(without context). That's a 1-step task. The trace from a single TriviaQA answer is too short to have meaningful spectral content (Phase 9 confirmed this empirically). The contrast signal in Direction 2 is interesting on its own, but it does not exercise the spectral pipeline.

The long-doc agentic version uses the spectral pipeline *as designed* — long traces, multi-step structure, segment-level labels.

---

## How this relates to existing directions

| Direction | Original framing | Under this proposal |
|---|---|---|
| Direction 2 (RAG) | 1-step TriviaQA contrast | Becomes the 1-iteration ablation of long-doc RAG |
| Direction 4 / Phase 10 (Agentic, Step 78) | GPQA Diamond ReAct loop, no retrieval | Pivots to long-doc RAG-agentic with retrieval-as-action |
| Result | Two separate thesis chapters | One unified chapter with a trajectory-length axis (1 → N iterations) |

---

## Risks

1. **Confidence masking.** LLMs often hallucinate *confidently* when the doc is related but unsupportive. Entropy could stay flat across grounded and fabricated tokens. This is the main empirical question and the dominant failure mode on TriviaQA.
2. **Document-token noise.** If the trace includes the doc-reading phase, lexical entropy variation drowns the grounding signal. Mitigation: compute spectral features on *generation tokens only*, not on the prompt prefix.
3. **Claim-level ground truth is scarce.** Most QA benchmarks score the whole answer. Datasets with span/claim-level grounding labels (FActScore, ALCE) exist but are smaller and more expensive to evaluate.
4. **Long-context model behavior is not characterized.** Does Falcon-3-10B's long-context attention actually produce the entropy pattern we hypothesize? Open question for the literature search.

---

## AgentHallu — retrieval is a first-class category

The AgentHallu paper (HISTORY Step 78) annotates step-level hallucinations with 5 categories: **Planning, Retrieval, Reasoning, Human-Interaction, Tool-Use**. Retrieval is a first-class category. The agentic-hallucination literature already treats RAG-style failures as agent-step failures.

The benchmark itself is unusable for us (text-only outputs, GPT-4.1-generated, no logprobs), but:
- The annotation schema is a useful blueprint for our own labeling.
- The categorization confirms the unification angle: RAG hallucination ⊂ agentic hallucination.
- Their reported SOTA (Gemini 2.5 Pro = 41.1% step localization) shows this is an open problem.

---

## Three options for relating RAG and Phase 10

| Plan | Structure | Pro | Con |
|---|---|---|---|
| A. Two separate chapters | RAG = Direction 2 standalone (TriviaQA static). Phase 10 = GPQA agentic. | Cleanest, each self-contained | Two distinct contributions; thesis feels like two papers stapled |
| B. Unified | Pivot Phase 10 to long-doc RAG-agentic. RAG = 1-step ablation. | Most thesis-coherent. Single retrieval pipeline, single EPR(grounded)/EPR(parametric) decomposition, 1→N scaling | New infrastructure track; HotpotQA already failed at single-shot |
| C. Sequential | Run a cheap RAG pilot first. Decide based on results. | Lowest risk; pilot is half a day | Splits planning across sessions |

**Current preference**: C (pilot first), then likely B (unify) if pilot is positive.

---

## Pilot experiment proposal (if research findings support it)

- **Dataset**: FActScore biographies (atomic-fact decomposition with claim-level grounding labels) — *to be validated by /research*
- **Model**: Falcon-3-10B (infra already validated for QA)
- **Setup**: 1-iteration RAG (give bio, ask about subject), no agentic loop yet
- **Sample size**: 50 samples
- **Test**: do `sw_var_peak` and `spectral_entropy` separate grounded vs ungrounded claims at the claim-span level? Bootstrap AUC at the claim level.
- **Decision gate**: AUC > 60% on at least one feature → build out into full Phase 10 design. AUC near 50% → confidence masking dominates, abandon and stay with Plan A.
- **Cost**: half a day on Colab.

---

## What `/research` should investigate before we commit

1. **Long-doc faithfulness benchmarks** with claim/span-level labels — survey FActScore, ALCE, RAGAS, NIAH/RULER, ExpertQA, others. Pick 2–3 most viable for a 50–200 sample pilot. Report dataset size, labeling granularity, license, code availability.
2. **Entropy/logit-based hallucination detectors on long-doc RAG** — has anyone tested per-token entropy or its frequency content for *span-level* faithfulness detection? This is the most important pre-emption check.
3. **Agentic hallucination detection methods beyond AUQ + AgentHallu** — what's SOTA on step-level localization? What signals are used? Any methods beyond verbalized confidence?
4. **Empirical evidence on long-context entropy behavior** — does entropy actually track grounding state in long-doc generations? Any prior empirical studies?

---

## Decision points after `/research` returns

1. Which 1–2 benchmarks for the pilot.
2. Which 3–5 baselines to compare against.
3. Whether anyone has pre-empted the contribution.
4. Commit to Plan B (unify) or stay with Plan A/C.
5. Whether to follow up with `/research-deep` on one specific sub-question.
