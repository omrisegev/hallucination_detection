# Paper-Aligned Benchmark Suite — Review Guide

This directory contains the generated advisor-facing benchmark suite. Each
HTML page represents one published protocol or one explicitly marked internal
transfer appendix. The index does not average incompatible tasks or metrics.

## Source files

- `scripts/paper_aligned_benchmark_suite.py` builds task adapters, runs the
  frozen solvers, writes machine-readable scores, and renders the HTML pages.
- `spectral_utils/paper_benchmark_suite.py` contains shared label-free fitting,
  protocol-signature validation, grouped bootstrap, hashing, and HTML chart
  helpers.
- `scripts/test_paper_aligned_benchmark_suite.py` contains the acceptance tests.

The reporting work does not change the existing U-PCR, IU-PCR, or DUFS-LIU-PCR
implementations. It calls their existing repository APIs.

## Reproduction

From the repository root, using the project virtual environment:

```bash
.venv/bin/python scripts/paper_aligned_benchmark_suite.py score
.venv/bin/python scripts/paper_aligned_benchmark_suite.py report
.venv/bin/python scripts/test_paper_aligned_benchmark_suite.py
```

The combined command is:

```bash
.venv/bin/python scripts/paper_aligned_benchmark_suite.py all
```

The scoring command checkpoints after each protocol in `score_progress.json`.
It expects the validated local benchmark caches described in
`results/data_readiness_2026_08_11/dataset_registry.json`. Those large caches
are intentionally not copied into this report directory.

## Generated artifacts

- `index.html`: suite entry point.
- `*.html`: one self-contained page per protocol.
- `benchmark_scores.csv`: every plotted and tabulated value.
- `protocol_registry.json`: paper, dataset, model, unit, grader, fidelity, and
  limitation metadata.
- `diagnostics/*.json`: label-free fit diagnostics, gates, graph properties,
  score hashes, exclusions, and input hashes.
- `suite_manifest.json`: generated-file hashes and suite-level acceptance
  declarations.

No chart value is typed into HTML. Reports are regenerated from the CSV/JSON
artifacts.

## Review priorities

1. Check every proposed head-to-head comparison against its full protocol
   signature: dataset, model, split, prediction unit, metric, and grader.
2. Confirm that published references with incomplete or different signatures
   remain visually separate from matched local comparisons.
3. Check that the task adapters create a meaningful local feature matrix while
   keeping the three spectral solvers unchanged.
4. Confirm that labels enter only evaluation and grouped uncertainty, not
   feature construction, DUFS gates, graphs, or fusion weights.
5. Inspect the RAGTruth sentence mapping limitation: the exact GASP cache did
   not store scorer-token character offsets, so sentence spans use a declared
   character-mass projection and are not labelled an exact reproduction.
6. Confirm that PRMBench gold error labels come from the official one-based
   `error_steps` field. The cached `labels` field is the PRM's thresholded
   prediction and is deliberately not used as gold.
7. Confirm that the three registered PRMBench alignment defects are excluded
   from every compared method.
8. Confirm that HLE and the incomplete ProcessBench 72B critic do not enter the
   publishable suite.

## Current high-level result

Across the new localization protocols, IU-PCR usually improves over deployed
U-PCR. DUFS-LIU-PCR remains nearly tied with IU-PCR, so these reports do not
support a strong additional DUFS/Laplacian contribution. Supervised task-native
models remain substantially stronger on PRMBench and ProcessBench.
