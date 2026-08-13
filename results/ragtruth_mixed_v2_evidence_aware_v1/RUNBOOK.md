# Runbook: RAGTruth original-30 evidence-aware experiment

Run every command from the repository root. These paths match the current
local data layout. The scoring commands do not use labels for fitting.

```bash
OUT=results/ragtruth_mixed_v2_evidence_aware_v1
TOKENIZER=local_cache/qwen25_15b_tokenizer
OFFICIAL=local_cache/RAGTruth_official/dataset/response.jsonl
REFERENCE=results/ragtruth_evidence_contrast_v1_top50_correction
```

## 1. Write label-free development scores

```bash
.venv/bin/python scripts/ragtruth_mixed_v2_evidence_experiment.py score \
  --split dev \
  --cache local_cache/ragtruth_ec/dev/ragtruth_ec_train.pkl \
  --official-responses "$OFFICIAL" \
  --tokenizer "$TOKENIZER" \
  --reference-scores "$REFERENCE/scores_dev.npz" \
  --out "$OUT"
```

## 2. Evaluate development

```bash
.venv/bin/python scripts/ragtruth_mixed_v2_evidence_experiment.py evaluate \
  --split dev \
  --cache local_cache/ragtruth_ec/dev/ragtruth_ec_train.pkl \
  --official-responses "$OFFICIAL" \
  --tokenizer "$TOKENIZER" \
  --bootstrap 1000 \
  --out "$OUT"
```

Do not modify a variant after reading this output. This experiment is already
exploratory because RAGTruth labels were opened in earlier work.

## 3. Write label-free test scores

```bash
.venv/bin/python scripts/ragtruth_mixed_v2_evidence_experiment.py score \
  --split test \
  --cache local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl \
  --official-responses "$OFFICIAL" \
  --tokenizer "$TOKENIZER" \
  --reference-scores "$REFERENCE/scores_test.npz" \
  --out "$OUT"
```

## 4. Evaluate test

```bash
.venv/bin/python scripts/ragtruth_mixed_v2_evidence_experiment.py evaluate \
  --split test \
  --cache local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl \
  --official-responses "$OFFICIAL" \
  --tokenizer "$TOKENIZER" \
  --bootstrap 1000 \
  --out "$OUT"
```

## 5. Build the reports

First verify that the full-only arm reproduces the earlier mixed-v2 audit:

```bash
.venv/bin/python scripts/ragtruth_mixed_v2_evidence_experiment.py \
  reproduction-audit --split dev \
  --old-scores results/ragtruth_evidence_contrast_v1/scores_intrinsic_mixed_v2_posthoc_dev.npz \
  --out "$OUT"

.venv/bin/python scripts/ragtruth_mixed_v2_evidence_experiment.py \
  reproduction-audit --split test \
  --old-scores results/ragtruth_evidence_contrast_v1/scores_intrinsic_mixed_v2_posthoc_test.npz \
  --out "$OUT"
```

Then build the reports:

```bash
MPLBACKEND=Agg .venv/bin/python \
  scripts/ragtruth_mixed_v2_evidence_experiment.py report --out "$OUT"
```

## 6. Verification

```bash
.venv/bin/python -m unittest scripts/test_ragtruth_mixed_v2_evidence.py
.venv/bin/python -m unittest scripts/test_ragtruth_evidence_contrast.py
.venv/bin/python -m py_compile \
  spectral_utils/ragtruth_mixed_v2_evidence.py \
  scripts/ragtruth_mixed_v2_evidence_experiment.py
git diff --check
git status --short
```

The final report must state that the comparison is exploratory and that a new
benchmark or scorer is needed for confirmation.
