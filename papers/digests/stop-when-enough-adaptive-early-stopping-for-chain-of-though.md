---
slug: stop-when-enough-adaptive-early-stopping-for-chain-of-though
title: "Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning"
authors: "Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, Wei Wang — UCLA / NEC Labs America / Arizona State University"
arxiv_id: "ACL Anthology 2026.acl-long.1256"
venue: "ACL 2026 Long Papers"
year: 2026
source_pdf: "papers/Stop When Enough Adaptive Early-Stopping for Chain-of-Thought Reasoning (ACL 2026).pdf"
extracted_text: papers/extracted/stop-when-enough-adaptive-early-stopping-for-chain-of-though.md
last_digested: 2026-08-16
---

## Summary

REFRAIN is a training-free, one-trace-per-question stopping method with a controller that
adapts **across questions**. It stops only after a provisional answer has appeared and the
current blank-line-delimited step both contains a reflection cue and is semantically redundant
with an earlier step. A sliding-window UCB bandit selects the similarity threshold using an
answer-likelihood-minus-length reward.

## Datasets & models used

- Models: Qwen3-8B in official thinking mode and gpt-oss-20B.
- Benchmarks: GSM8K 1,319, MATH-500 500, CommonsenseQA validation 1,221, and
  GPQA-Diamond 198.
- Primary Qwen3 decoding: temperature 0.6, top-p 0.95, top-k 20; maximum 16,384 tokens;
  seed 42. gpt-oss uses temperature 1.0, top-p 1.0, top-k 50.

## Methods it compared itself against

Vanilla, No-thinking, fixed budgets, DEER, HALT-CoT, AlphaOne, CoT-Valve, and an adapted
single-trace DeepConf. CoT-Valve requires fine-tuning and DeepConf needs additional trace
generation, so those are not equal-cost training-free controls.

## Experiments — methodology & scores

Use prompt P0 verbatim, split steps on blank lines, and use the Appendix-A trigger vocabulary.
Embed the current step with `all-MiniLM-L6-v2`; stop when a prior provisional-answer cue is
present, a reflection cue occurs in the current step, and maximum cosine similarity to a prior
step exceeds the chosen threshold. Force the boxed answer, compute its length-normalized
geometric-mean likelihood, and update SW-UCB reward `score - lambda * L/mean(L)`.

Freeze dataset order because the bandit state crosses questions: threshold grid
`{0.60,0.65,0.70,0.75,0.80}`, `C=1`, `W=100`, `lambda=0.2`, and first-round cold-start
penalty `0.0001*L`.

| Model / benchmark | Vanilla accuracy / tokens | REFRAIN accuracy / tokens | Token change |
|---|---:|---:|---:|
| Qwen3-8B / GSM8K | 94.24 / 2.62M | 94.54 / 1.68M | -35.9% |
| Qwen3-8B / MATH-500 | 91.40 / 2.64M | 91.20 / 1.61M | -39.0% |
| Qwen3-8B / CSQA | 83.13 / 1.66M | 84.03 / 0.76M | -54.2% |
| Qwen3-8B / GPQA-D | 53.54 / 1.81M | 60.10 / 1.42M | -21.5% |
| gpt-oss-20B / MATH-500 | 80.80 / 1.07M | 84.20 / 0.69M | -35.5% |

## Connection to our pipeline

This is the primary paper-specified single-trace stopping comparison. Run its native policy and
our frozen causal method on the same Qwen3-8B/MATH-500 generation contract, but preserve
REFRAIN's native closure in the native table. A common-closure experiment is a separate adapted
policy. Compare pass@1 and total generated tokens/latency, not our prefix AUROC against its token
savings.

## Notes / open questions

- The official repository was a README-only release placeholder when audited, so the PDF-based
  implementation is `paper-specified`, not `official-exact`.
- Because SW-UCB persists across questions, freeze row order, reset scope, cold-start arm order,
  tie-breaking, running-length update, reward timing, model/tokenizer revision, and chat template.
- A 30-question pilot validates implementation only; after five arm initializations it has at
  most 25 adaptive rounds. Only the full 500-question run is the paper-specified table attempt.
