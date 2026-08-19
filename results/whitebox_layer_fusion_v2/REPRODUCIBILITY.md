# Reproducibility and Artifact Inventory

## Isolation and sources

- Worktree: `/Users/osegev/Desktop/hallucination_detection_whitebox_layer_fusion`
- Branch: `codex/whitebox-layer-fusion`
- No push or pull request was performed.
- Google Drive was accessed read-only under `gdrive:hallucination_detection/`.
- Download cache: ignored `dataset_cache/whitebox_layer_fusion_v1/` inside the dedicated worktree.
- Results: `results/whitebox_layer_fusion_v2/`.

The run is split into processes so fitting cannot inherit loaded labels:

```bash
PYTHONPATH=/private/tmp/whitebox_test_deps:. PYTHONDONTWRITEBYTECODE=1 \
  python3 -B scripts/whitebox_layer_fusion_experiment.py --phase prepare
PYTHONPATH=/private/tmp/whitebox_test_deps:. PYTHONDONTWRITEBYTECODE=1 \
  python3 -B scripts/whitebox_layer_fusion_experiment.py --phase fit
PYTHONPATH=/private/tmp/whitebox_test_deps:. PYTHONDONTWRITEBYTECODE=1 \
  python3 -B scripts/whitebox_layer_fusion_experiment.py --phase evaluate
PYTHONPATH=/private/tmp/whitebox_test_deps:. PYTHONDONTWRITEBYTECODE=1 \
  python3 -B scripts/whitebox_layer_fusion_experiment.py --phase report
```

## Freeze chain

- `RUN_DEFINITION.json`: registered cells, contracts, DUFS settings, bootstrap, orientation, and protocol signature.
- `SOURCE_FREEZE_MANIFEST.json`: 28 remote/local data hashes plus registered source-code hashes.
- `PREPARED_FEATURE_MANIFEST.json`: 155 label-free NPZ hashes and field rosters.
- `FIT_COMPLETE.json`: 14 frozen score bundles and diagnostics; `labels_seen_during_fit=false`.
- `SCORE_FREEZE_MANIFEST.json`: verifies all score hashes before labels and attests `scores_frozen_before_labels=true`.
- `bootstrap_draw_manifest.json`: actual per-cell seeds and identical-draw hashes.
- `REPORT_MANIFEST.json`: hashes the self-contained HTML and seven separate SVG figures.

## Evaluation artifacts

- `per_cell_metrics.csv`: every candidate-level method/cell AUROC and AUPRC.
- `headline_summary.csv`: 13-cell equal-cell macro estimates and bootstrap intervals.
- `cohort_summary.csv`: original-six, seven-model GSM8K, primary-13, and descriptive-14 macros.
- `paired_comparisons.csv`: all registered paired deltas and supporting statistics.
- `supervised_grouped_cv_diagnostics.json`: fold-level metrics and overlap checks.
- `layer_diagnostics.csv`: evaluation-only layer×module×metric curves.
- `dependence_diagnostics.csv`: layer correlation matrices, distance curves, and effective rank.
- `weights_diagnostics.csv`: fusion weights, DUFS gates, graph health, and convergence data.
- `data_audit.json` and `data_coverage.csv`: complete row/tensor/provenance audit.
- `comparator_fidelity.csv`: literature method reproduction boundary.
- `REPORT.html`: portable, theme-aware report with no external assets.

## Tests run

- `scripts/test_whitebox_layer_fusion.py`: 15 synthetic contract/solver tests.
- `scripts/test_whitebox_layer_fusion_experiment.py`: four phase/freeze/bootstrap/CV tests.
- `scripts/test_whitebox_layer_fusion_full_data.py`: four exact 14-cell acceptance tests.
- `scripts/test_whitebox_layer_fusion_report.py`: five sentinel/hash/semantic/no-network tests.
- `scripts/test_layer_views_reference.py`: reference hook/order/logit/KL/Gate-B fixture suite.

The report was also rendered in the in-app browser at 1440×900 and 390×844. Both had zero body overflow, zero clipped top-level elements, zero broken images, zero external resources, no `<pre>` report body, and no browser console warnings/errors. Wide semantic tables scroll inside their wrappers on mobile.

## Known blockers

The report remains **PRELIMINARY / VALIDATION BLOCKED** until:

1. corrected live Gate B is rerun with the nested-candidate loader over all 14 cells;
2. the independent two-cell architecture-fidelity pilot passes;
3. covariance geometry is either recaptured without float16 overflow or remains explicitly omitted.

These blockers do not invalidate the offline scores; they prevent promotion of a capture-fidelity or robust-improvement claim.
