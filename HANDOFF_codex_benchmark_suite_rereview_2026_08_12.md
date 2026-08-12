# Handoff — re-review the corrected paper-aligned benchmark suite

**Date:** 2026-08-12

**Original review:** `REVIEW_codex_benchmark_suite_2026_08_12.md`

**Scope:** the paper-aligned benchmark generator, generated suite, diagnostics,
and data-readiness provenance. No fusion algorithm was changed.

## What to review

Please review the corrected implementation and regenerated artifacts:

- `scripts/paper_aligned_benchmark_suite.py`
- `spectral_utils/paper_benchmark_suite.py`
- `scripts/test_paper_aligned_benchmark_suite.py`
- `results/paper_aligned_benchmark_suite_2026_08_11/`
- `results/data_readiness_2026_08_11/dataset_registry.json`

The main entry point is:

- `results/paper_aligned_benchmark_suite_2026_08_11/index.html`

## Accepted findings and implemented corrections

### RefChecker identity and task separation

- Claims are now keyed by `(setting, generator, example_id, claim_index)`.
- The adapter retains all **10,733** claims rather than 3,468 overwrite
  survivors.
- `accurate_context`, `noisy_context`, and `zero_context` are fitted and
  evaluated separately. There is no pooled RefChecker fit or headline.
- Bootstrap groups are source `example_id` values within each setting.
- Per-setting score hashes and fit diagnostics are recorded.

### Roles and visual claims

- Qwen2.5-Math-PRM is `published_ceiling`, not `ours`.
- GASP is `protocol_reproduction`, not `ours`.
- The HTML conclusion generator considers only frozen/local method roles, so a
  supervised ceiling or protocol reproduction cannot be called our best
  method.
- Ceiling styling is checked before the more general published-role styling.

### ProcessBench

- The complete Qwen2.5-72B critic package is included: four subsets and its
  **59.4003% macro F1**. It is described as a protocol reproduction using a
  different critic model.
- Frozen Qwen3-8B GL-LIU v1 is shown and distinguished from newer exploratory
  solver pairings.
- The independent Llama-3.1-8B results show GL-LIU v1 beside maximum token
  entropy. Their macro values are **31.7091%** and **31.5012%**, respectively,
  and the page describes this as a noise-level margin whose sign changes by
  subset.
- The report discloses the development-label component selection and
  calibration-half threshold used by the frozen GL-LIU system.

### PRMBench

- The constructed `correct` control is excluded from the headline fit and
  pooled evaluation.
- The headline is explicitly named **all nine paper classes (constructed
  control excluded)**.
- The three registered alignment-defect rows remain excluded from all methods.
- The page states that the task adapter contains ten aggregates of five
  token-resolved views, not the original 30 response features.
- The page states that 71% of steps have fewer than 32 tokens.

### Detection reporting

- Duplicate EPR, HCPD, and LOS-Net published bars are removed.
- A frozen local cell is linked to every verified paper page that cites that
  cell, rather than only one page.
- The legacy INSIDE/CoQA loss is visible and clearly marked as outside the
  current 24-cell full-pool core suite.
- The pages state that the current suite uses frozen full-pool methods while
  the older repgrid uses per-cell selected-subset compatibility arms. Those are
  not presented as two estimates of the same arm.
- SemGrad metadata now uses BEM, the correct sample counts, and
  `BACKGROUND_ONLY` readiness.

### Missing RAG-detection quadrant

- A RAGTruth response-detection page was generated from the frozen Step-239
  artifact.
- It distinguishes the supported intervention-design result from the
  unconfirmed fusion novelty result: evidence graph AUROC **0.753642** versus
  naive aggregation **0.728988**, delta **+0.02505**, 95% interval
  **[-0.00584, 0.05721]**, `p=0.066`.

### Auditability and guards

- `suite_manifest.json` now hashes the score table, registry, progress file,
  HTML pages, review guide, and all seven diagnostic JSON files.
- The complete data-readiness registry is available for inclusion with the
  correction.
- The macro guard now rejects explicit suite/cross-task macros and rejects a
  ProcessBench four-subset macro outside the ProcessBench protocol.
- Generated count columns render as integer counts.
- Nine suite tests and eight data-readiness tests pass.

## One review recommendation that was not followed literally

The original review said that both `correct` and `multi_solutions` should be
excluded from PRMBench pooled totals. I could verify the first part but not the
second:

- `correct` is the constructed control and is now excluded.
- `multi_solutions` is one of the paper's nine named evaluation classes in the
  repository's official evaluator. It contains no annotated error step, so it
  cannot produce a standalone binary AUROC, but removing it from the pooled
  nine-class headline would no longer reproduce the official paper-class
  population.

The correction therefore keeps `multi_solutions` in the explicitly named
nine-class pooled result and documents why it has no standalone binary row.
Please re-check this against the original PRMBench protocol and identify a
specific official equation, table, or evaluator line if it should instead be
excluded.

## Decisions made explicitly rather than hidden

1. The **full-pool frozen core methods** lead the 24-cell detection suite. The
   old repgrid selected-subset results remain compatibility evidence. Please
   check that no page conflates these contracts.
2. SemGrad remains visible for auditability but is marked `BACKGROUND_ONLY`.
3. New ProcessBench pairings remain visible only as `ours_exploratory`; frozen
   GL-LIU v1 is the principal local system.
4. The RAG evidence-graph result is promising, not confirmed, because its
   paired interval includes zero.

## Reproduce and verify

From the repository root:

```bash
.venv/bin/python scripts/paper_aligned_benchmark_suite.py all
.venv/bin/python scripts/test_paper_aligned_benchmark_suite.py
.venv/bin/python scripts/test_data_readiness.py
git diff --check
```

Expected suite summary:

- 20 protocol pages
- 338 machine-readable score rows
- 7 diagnostic JSON files
- 9/9 benchmark-suite tests passing
- 8/8 data-readiness tests passing

## Requested re-review output

Please report only:

1. remaining incorrect numbers or protocol mismatches;
2. any remaining role that overstates the evidence;
3. any generated artifact promised by the report but absent from Git;
4. whether the PRMBench `multi_solutions` treatment above matches the official
   protocol;
5. whether the detection full-pool/legacy-repgrid separation is sufficiently
   clear for advisor presentation.

Please distinguish fatal, material, and cosmetic findings, and verify every
claimed error against the machine-readable artifact or original evaluator.
