# RAGTruth Evidence-Contrast Experiment: Runbook

This file describes the exact execution order for version 1. The order matters:
scores must be written and hashed before labels are opened.

## Required local inputs

The large inputs are intentionally outside Git.

```text
local_cache/ragtruth_ec/dev/ragtruth_ec_train.pkl
local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl
local_cache/RAGTruth_official/dataset/response.jsonl
local_cache/qwen25_15b_tokenizer/
```

Run commands from the repository root with the existing `.venv`. Set a
headless matplotlib cache so report generation also works on a cluster node.

## 1. Unit tests

```bash
.venv/bin/python scripts/test_ragtruth_evidence_contrast.py
```

The tests cover safe pickle loading, adapter label isolation, token alignment,
feature formulas, approximate JSD, exact lambda-zero equality, grouped
bootstrap integrity, row-order-invariant evaluation, and the final decision
rule.

The `validate-input` command may be run independently to compare the cache with
its sidecar manifest without touching a frozen score file:

```bash
.venv/bin/python scripts/ragtruth_ec_experiment.py validate-input \
  --split test \
  --cache local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl \
  --official-responses local_cache/RAGTruth_official/dataset/response.jsonl \
  --tokenizer local_cache/qwen25_15b_tokenizer \
  --out results/ragtruth_evidence_contrast_v1
```

## 2. Freeze development scores

```bash
.venv/bin/python scripts/ragtruth_ec_experiment.py score \
  --split dev \
  --cache local_cache/ragtruth_ec/dev/ragtruth_ec_train.pkl \
  --official-responses local_cache/RAGTruth_official/dataset/response.jsonl \
  --tokenizer local_cache/qwen25_15b_tokenizer \
  --out results/ragtruth_evidence_contrast_v1
```

This command discards the isolated label object before building features. It
writes `scores_dev.npz` and `scores_dev.sha256`.

## 3. Open development labels once

```bash
.venv/bin/python scripts/ragtruth_ec_experiment.py evaluate \
  --split dev \
  --cache local_cache/ragtruth_ec/dev/ragtruth_ec_train.pkl \
  --official-responses local_cache/RAGTruth_official/dataset/response.jsonl \
  --tokenizer local_cache/qwen25_15b_tokenizer \
  --out results/ragtruth_evidence_contrast_v1 \
  --bootstrap 1000
```

Read `development_gate.json`. Stop if `passed` is false. Do not revise the
method and reuse the same development evaluation under version 1.

## 4. Freeze test scores

Run this step only after the development gate passes.

```bash
.venv/bin/python scripts/ragtruth_ec_experiment.py score \
  --split test \
  --cache local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl \
  --official-responses local_cache/RAGTruth_official/dataset/response.jsonl \
  --tokenizer local_cache/qwen25_15b_tokenizer \
  --out results/ragtruth_evidence_contrast_v1
```

Record `scores_test.sha256` before continuing.

## 5. Open test labels and apply the registered rule

```bash
.venv/bin/python scripts/ragtruth_ec_experiment.py evaluate \
  --split test \
  --cache local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl \
  --official-responses local_cache/RAGTruth_official/dataset/response.jsonl \
  --tokenizer local_cache/qwen25_15b_tokenizer \
  --out results/ragtruth_evidence_contrast_v1 \
  --bootstrap 1000
```

The command verifies the score hash before reading labels. It writes summary,
paired-comparison, confound, hallucination-type and source-family CSV files,
plus `final_decision.json`.

## 6. Optional intrinsic mixed-v2 audit

This baseline was added after version-1 test labels had already been opened.
It is reproducible, but it is post-hoc and must not enter the registered
decision.

```bash
.venv/bin/python scripts/ragtruth_ec_experiment.py intrinsic-score \
  --split test \
  --cache local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl \
  --official-responses local_cache/RAGTruth_official/dataset/response.jsonl \
  --tokenizer local_cache/qwen25_15b_tokenizer \
  --out results/ragtruth_evidence_contrast_v1

.venv/bin/python scripts/ragtruth_ec_experiment.py intrinsic-evaluate \
  --split test \
  --cache local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl \
  --official-responses local_cache/RAGTruth_official/dataset/response.jsonl \
  --tokenizer local_cache/qwen25_15b_tokenizer \
  --out results/ragtruth_evidence_contrast_v1 \
  --bootstrap 1000
```

## 7. Generate the reports

```bash
MPLCONFIGDIR=/tmp/ragtruth-mpl \
  .venv/bin/python scripts/ragtruth_ec_experiment.py report \
  --out results/ragtruth_evidence_contrast_v1
```

The report command reads the CSV and JSON artifacts. It does not type headline
numbers into the report manually. It writes the self-contained `REPORT.html`,
the concise `REPORT.md`, `experiment_manifest.json`, and every PNG in
`figures/`.

## Version-1 outcome

The registered development gate passed. The final test decision is not a full
success: Evidence-Contrast plus IU-PCR improves over approximate GASP, but the
DUFS-gated Laplacian does not improve over IU-PCR. Do not change this conclusion
by selecting another lambda, graph or feature contract on the opened test set.
