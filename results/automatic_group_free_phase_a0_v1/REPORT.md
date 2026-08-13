# Automatic group-free IU — Phase A0 audit

- Version: `automatic-group-free-iu-a0-v1-2026-08-13`
- Correctness labels accessed: **no**
- Source environments: **23**
- Canonical mixed-v2 features: **30**
- Features present in every source environment: **17**
- Minimum / maximum feature-pair source coverage: **8 / 23**
- Cells whose valid mixed-v2 rows are fewer than manifest attempts: **6** (minimum retention 19.8%)
- Exact ProcessBench cross-model pairs: **3400** across Qwen-3 4B, Qwen-3 8B, and Llama-3.1 8B
- Fully exact ProcessBench subsets: **4 / 4**
- Simulator crossed dimensions: **5 channels x 6 operators**, 8 environments
- Simulator duplicate error: **0**
- Reserved confirmation: `semgrad-triviaqa-qwen3-4b-confirmation-v1` (RESERVED_REQUIRES_COLLECTION)

## Decision

A0 passes. The source roster has a fully auditable missingness and pair-coverage
boundary, and exact cross-model pairing exists for 3,400 fixed reasoning traces.
This supports A1/A2 structural work and gives A4 an exact paired-view calibration
surface without semantic matching. The confirmation cell is reserved but must be
collected only after a finalist and all target-selection rules are frozen.

The feature DAG records source streams and computational operators from the
extractor-owned registries and function signatures. It does not import or reproduce
the manual `FEATURE_TO_VIEW` partition.

Manifest observation counts describe attempted/generated candidates, whereas the
bundle contains the valid rows admitted to the frozen mixed-v2 comparison. A1/A2
must preserve the bundle population and equal-environment weighting; they may not
silently restore filtered rows or weight an environment by its candidate count.
