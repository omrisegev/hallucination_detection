---
slug: streaming-hallucination-detection-in-long-chain-of-thought-r
title: "Streaming Hallucination Detection in Long Chain-of-Thought Reasoning"
authors: "Haolang Lu, Minghui Pan, Ripeng Li, Guoshun Nan, Jialin Zhuang, Zijie Zhao, Zhongxiang Sun, Kun Wang, Yang Liu"
arxiv_id: "arXiv:2601.02170v1"
venue: "arXiv preprint"
year: 2026
source_pdf: "papers/Streaming Hallucination Detection in Long Chain-of-Thought Reasoning (arXiv 2601.02170v1).pdf"
extracted_text: papers/extracted/streaming-hallucination-detection-in-long-chain-of-thought-r.md
last_digested: 2026-08-16
---

## Summary

This is the closest conceptual paper to causal prefix hallucination detection, but it is a
**supervised hidden-state probe**, not a label-free logit method or an early-stopping policy.
It predicts both whether the current sentence introduces an error and whether the prefix has
entered a hallucinated state. The core representation is an L2-normalized, exponentially
time-weighted average of token hidden states within the current sentence.

## Datasets & models used

- Questions originate from BBH and MuSiQue.
- Llama-3.1-8B-Instruct: 3,400 generated questions, about 2,500 usable traces, 58,619 steps.
- Qwen2.5-7B-Instruct: 3,000 generated, about 2,900 usable, 53,728 steps.
- DeepSeek-R1-Distill-Llama-8B: 3,500 generated, about 2,800 usable, 89,918 steps.
- Claude Sonnet 4.5 supplies sentence-level and prefix-level labels and semantically judges
  final-answer correctness. Logical transition filters remove inconsistent labels; experts
  audit a random 5% and the paper reports greater than 96% agreement.

## Methods it compared itself against

- Step-level: TTPD, SAPLMA, global-mean hidden-state aggregation, and several alternative
  pooling/probability-statistics representations.
- Prefix-level: ICR, LLM-Check, and global-mean aggregation, all trained with the same
  prefix supervision in the comparison.
- All reported methods are not in our one-trace, label-free, logprob-only access tier.

## Experiments — methodology & scores

Treat each sentence as a step. At layer `l`, weight token `j` of an `L`-token step by the
softmax of `(j-1)/(L-1)`, sum only within the current step, L2-normalize, and train the
step probe on the step labels. The prefix probe receives the same representation plus the
step score and uses terminal-anchor BCE and an alarm-synchronization loss. Report AUC, ACC,
and F1; do not mix `Local` (average over prefixes) with `Final` (last-prefix prediction).

| Model | Step AUC | Prefix Local AUC | Prefix Final AUC |
|---|---:|---:|---:|
| Llama-3.1-8B-Instruct | 87.83 | 87.30 | 72.69 |
| Qwen2.5-7B-Instruct | 86.70 | 88.02 | 81.05 |
| DeepSeek-R1-Distill-8B | 93.27 | 87.98 | 92.18 |

The appendix sweeps even layers 2-30. Peak step AUC is near layers 16-20 (Llama 88.04 at
layer 16, Qwen 86.66 at layer 20, DeepSeek 93.27 at layer 18).

## Connection to our pipeline

Use this as a supervised white-box ceiling for two separate panels: sentence-local error AUC
and prefix-state AUC. A number from our final-answer correctness task is not directly
comparable. A valid shared experiment requires the paper's same trajectories, labels, splits,
and hidden-state layer; only then may our label-free telemetry be scored on the same rows.

## Notes / open questions

- The PDF says code is at an anonymous link, but the audited endpoint was unreachable.
- The PDF does not expose enough exact split, probe architecture/optimizer, loss coefficients,
  prompt/decoding, or checkpoint detail to recreate the published table from scratch.
- Therefore reproduction is **asset-gated**. If official trajectories, labels, split files,
  and checkpoints remain inaccessible, mark the row `blocked-assets`; do not invent a split or
  re-label 10,000 traces and call it exact.
