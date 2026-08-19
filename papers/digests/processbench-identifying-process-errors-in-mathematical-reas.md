---
slug: processbench-identifying-process-errors-in-mathematical-reas
title: "PROCESSBENCH: Identifying Process Errors in Mathematical Reasoning"
authors: "Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin — Qwen Team, Alibaba"
arxiv_id: "arXiv:2412.06559v4"
venue: "arXiv preprint"
year: 2025
source_pdf: "papers/ProcessBench Identifying Process Errors in Mathematical Reasoning (arXiv 2412.06559v4).pdf"
extracted_text: papers/extracted/processbench-identifying-process-errors-in-mathematical-reas.md
last_digested: 2026-08-16
---

## Summary

PROCESSBENCH is the canonical first-error localization benchmark used by this project. Each
record contains a math problem, a paragraph/step-separated solution, and the zero-based index
of the earliest erroneous step, or `-1` when all steps are correct. Its primary metric is the
harmonic mean of accuracy on erroneous traces and accuracy on clean traces, called F1 by the
paper.

## Datasets & models used

- Fixed test set: **3,400** expert-annotated cases: GSM8K 400, MATH 1,000,
  OlympiadBench 1,000, and Omni-MATH 1,000.
- Erroneous/clean counts are respectively 207/193, 594/406, 661/339, and 759/241.
- Three doctoral-level annotators label each case until three agree; up to five annotators
  are used, and unresolved cases are discarded (about 30%).
- The paper evaluates released 1.5B-8B PRMs, its own Qwen2.5-Math-7B PRM, prompted open
  critics from 7B to 72B, QwQ-32B-Preview, GPT-4o, and o1-mini.

## Methods it compared itself against

- Process reward models: Math-Shepherd, two RLHFlow PRMs, two Skywork PRMs, and the authors'
  Qwen2.5-Math-7B-PRM800K.
- Prompted critic models: Llama 3/3.1/3.3, Qwen2/2.5/2.5-Math/2.5-Coder, QwQ-32B,
  GPT-4o, and o1-mini.
- Scalar PRM thresholds are selected to maximize F1 on the GSM8K subset. Open critics use
  majority vote over eight samples in the primary table; greedy results are reported
  separately.

## Experiments — methodology & scores

For critics, use the exact Appendix-E prompt, preserve the paper's paragraph indices, parse
the boxed integer, and majority-vote over eight samples for open models. Qwen2.5-Math critics
use `temperature=0.7`, `top_p=0.8`, `top_k=20`; other open critics use `top_p=0.9` as reported.
PRMs return the earliest predicted-incorrect step. Report per-subset error accuracy, clean
accuracy, their harmonic mean, and the macro mean over the four subset F1 values.

| Setup | Metric | Score | Notes |
|---|---|---:|---|
| Qwen2.5-Math-7B-PRM800K | macro ProcessBench F1 | 56.5 | 68.2/62.6/50.7/44.3 by subset |
| Skywork-PRM-7B | macro ProcessBench F1 | 42.1 | best released PRM in the original table |
| Qwen2.5-72B-Instruct critic | macro ProcessBench F1 | 61.2 | eight-sample majority vote |
| QwQ-32B-Preview critic | macro ProcessBench F1 | 71.5 | strongest open critic in the table |
| GPT-4o-0806 critic | macro ProcessBench F1 | 61.9 | greedy |
| o1-mini critic | macro ProcessBench F1 | 87.9 | single API sample |

## Connection to our pipeline

This paper defines the native metric and clean-trace handling for every ProcessBench claim.
Mind the Gap's SLA on erroneous traces is not a substitute for ProcessBench F1. Our label-free
locator, maximum-entropy control, Mind-the-Gap reproduction, uPRM/LLM-judge arm, PRMs, and
critics must all be reduced to the same earliest-step-or-`-1` prediction before sharing a table.
The PRM and critic rows are separate access tiers, not same-cost competitors.

## Notes / open questions

- The benchmark is a test set, not a tuning set. Threshold selection on its labels must be
  identified as retrospective or nested; it cannot support a fresh confirmation claim.
- The authors' PRM training description specifies the reward head, line-break positions,
  PRM800K labels, contamination removal, and 8xA100-80GB, but omits enough optimizer detail
  that retraining from the PDF alone is not code-exact. Prefer the released checkpoint.
- Record dataset revision/hash and official evaluator commit in every cluster manifest.
