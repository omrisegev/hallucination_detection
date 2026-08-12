# Paper-Aligned Benchmark Suite — Review Guide

This directory is generated from machine-readable inputs. Each HTML page is
one protocol; incompatible tasks and metrics are never averaged.

## Reproduce

Run from the repository root:

```bash
.venv/bin/python scripts/paper_aligned_benchmark_suite.py all
.venv/bin/python scripts/test_paper_aligned_benchmark_suite.py
```

The generator writes `benchmark_scores.csv`, `protocol_registry.json`, one
self-contained page per protocol, stage diagnostics, and `suite_manifest.json`.

## Review priorities after the 2026-08-12 audit

1. RefChecker must key claims by `(setting, generator, example_id,
   claim_index)`. The diagnostics must report 10,733 claims, and the three
   context settings must remain separate in both fitting and evaluation.
2. Supervised Qwen PRM rows use `published_ceiling`; GASP reproduction uses
   `protocol_reproduction`. Neither may be called one of our methods.
3. PRMBench's headline excludes the constructed `correct` controls. The
   `multi_solutions` paper class remains pooled but has no standalone binary
   AUROC because it contains no annotated error step. The local adapter is ten
   summaries from five token-resolved views, not the original 30-feature
   response contract. The 71% shorter-than-32-token constraint must be visible.
4. ProcessBench includes all four Qwen2.5-72B critic subsets and the macro,
   labelled as a protocol reproduction with a different critic model.
5. The ProcessBench page leads with frozen GL-LIU v1 and also shows the newer
   exploratory solver pairings. The independent Llama-3.1-8B panel must show
   GL-LIU beside maximum token entropy and describe the margin as noise-level.
6. The RAGTruth response-detection page must state that evidence interventions
   helped, while the evidence-graph fusion novelty interval crosses zero.
7. Detection references are de-duplicated; cells supporting multiple paper
   pages carry the same local rows to each page; the legacy INSIDE/CoQA loss is
   visible and explicitly outside the current 24-cell full-pool suite.
8. SemGrad is background-only and has its BEM metadata. HLE remains absent
   from paper-faithful headlines until the original GPT-4o judge is available.
9. `diagnostics/` and the data-readiness registry must be included when this
   correction is eventually committed so cache hashes and score provenance
   are reviewable.
10. The detection pages use the frozen full-pool core methods. The older
    repgrid used per-cell selected subsets; its values are compatibility
    evidence, not a second estimate of the same arm.

## Current scientific reading

IU-PCR often improves over deployed U-PCR. DUFS-LIU-PCR is generally tied
with IU-PCR, so the suite does not establish a general Laplacian gain.
Evidence interventions are useful in RAG, but the extra evidence-graph fusion
gain over naive contrast averaging is promising rather than confirmed.
Task-native supervised models remain much stronger for step localization.
