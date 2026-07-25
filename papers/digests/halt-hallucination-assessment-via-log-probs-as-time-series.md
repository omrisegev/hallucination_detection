---
slug: halt-hallucination-assessment-via-log-probs-as-time-series
title: "HALT: Hallucination Assessment via Log-probs as Time series"
authors: "Ahmad Shapiro, Karan Taneja, Ashok Goel"
arxiv_id: "arXiv:2602.02888"
venue: "arXiv:2602.02888"
year: 2026
source_pdf: papers/HALT Hallucination Assessment via Log-probs as Time series.pdf
extracted_text: papers/extracted/halt-hallucination-assessment-via-log-probs-as-time-series.md
last_digested: 2026-07-13
---

## Summary

Presents HALT, a lightweight hallucination detector that treats the top-20 token log-probabilities from LLM generations as a temporal time series. Uses a recurrent GRU combined with entropy features to assess model calibration without needing hidden states or attention maps.

## Datasets & models used

TriviaQA, CoQA, and SQuAD across open-weight LLMs.

## Methods it compared itself against

Static mean log-probability, minimum token probability, and white-box activation classifiers.

## Experiments — methodology & scores

Evaluates ROC-AUC for hallucination detection across QA benchmarks.

| Setup | Method | ROC-AUC | Notes |
|---|---|---|---|
| Temporal Log-Prob Trajectory | HALT | Consistently higher than static scalar averages | Operates purely on top-K output log-probabilities |

## Connection to our pipeline

Directly parallels our EPR and WEPR temporal trace work, confirming that modeling log-probabilities as a time series improves hallucination detection.

## Notes / open questions

2026 preprint.
