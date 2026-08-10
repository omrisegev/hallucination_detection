# SemGrad BEM regrading

## Purpose

The cluster generation jobs used ROUGE-L as a temporary correctness proxy.
This run replaced that proxy for evaluation with the BEM answer-equivalence
grader used by SemGrad. The original cache files and their proxy labels were
not changed.

BEM returns a score between zero and one. For questions with several accepted
answers, every accepted answer is scored and the maximum score is retained.
The registered decision threshold for these cluster runs is 0.8.

## Implementation

- Model: `https://tfhub.dev/google/answer_equivalence/bem/1`
- SemGrad reference implementation commit:
  `118b6949f9641df3872caa7ad65a797f4ae28d63`
- Vocabulary SHA-256:
  `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`
- Device: CPU with TensorFlow 2.20 and XLA
- Batch size: 25
- Script: `scripts/bem_regrade.py`

The Apple `tensorflow-metal` 1.2.0 plugin was tested, but it was incompatible
with the resolved TensorFlow 2.20 package. CPU inference was therefore used.
This changes the execution device, not the BEM model or tokenizer.

## Results

| Dataset | Examples | Candidate-reference pairs | ROUGE-L proxy accuracy | BEM accuracy at 0.8 | Proxy/BEM agreement |
|---|---:|---:|---:|---:|---:|
| SciQ | 1,000 | 1,000 | 0.6480 | 0.6120 | 0.8680 |
| TruthfulQA | 817 | 2,600 | 0.3084 | 0.3856 | 0.7931 |

At threshold 0.7, used only as a sensitivity check, BEM accuracy is 0.6260 on
SciQ and 0.4541 on TruthfulQA. The registered result remains threshold 0.8.

The disagreements are:

| Dataset | Proxy correct, BEM incorrect | Proxy incorrect, BEM correct |
|---|---:|---:|
| SciQ | 84 | 48 |
| TruthfulQA | 53 | 116 |

The TruthfulQA difference is material. Downstream evaluation should use the
BEM labels rather than the temporary ROUGE-L labels. BEM is still an automatic
grader, so a blinded human or LLM audit of a stratified disagreement sample is
recommended before treating every BEM decision as ground truth.

## Integrity

| Dataset | Input SHA-256 | Scored output SHA-256 |
|---|---|---|
| SciQ | `2c4dc700ab6f0c478f295ae8afe16be106daffc724a2f9b011c8447e38b6f5ac` | `4607f2379c85baaa7369bb05605d2008b0df234c37195fdb1216855b6d33a057` |
| TruthfulQA | `4de81f95c8778caec087096eb6be1a190d3d8be30efdbc57ccfa8ee1b8776aef` | `b9036923f874badc0e8345d30fdb8db7c9bc226a1c710f063bb42fed94d774cc` |

The scored caches, manifests, downloaded model, and pinned vocabulary are kept
locally under `local_cache/` and are intentionally excluded from Git.
