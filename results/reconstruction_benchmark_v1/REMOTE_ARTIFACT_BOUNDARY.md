# Remote artifact boundary

This directory is a compact, reviewable Git snapshot of the reconstruction benchmark. It is not the complete local artifact store.

Included in Git:

- A/B verification certificates and build manifests;
- aggregate evaluation tables and reporting manifests;
- the canonical advisor-facing HTML report and its small plot-data files;
- reviewed derived winner/contrast artifacts and external audit records;
- advisor-update documents that explain the methods, evidence, and claim boundaries.

Intentionally excluded from Git:

- `source_overlays/` and copied raw input assets;
- `private_control/`, private labels, and other label-bearing control files;
- per-example predictions, bootstrap arrays, fitted-input snapshots, and large databases;
- large graph-diagnostic payloads and duplicate smoke/rebuild releases.

The complete local results tree is approximately 22 GB. Large source and experiment artifacts remain in the local/Google Drive artifact workflow described by the manifests. The committed certificates and manifests preserve the hashes needed to identify those omitted artifacts without making GitHub the raw-data store.

Canonical entry points:

- Main visual report: `releases/2026-08-24_frozen24_v1/reporting_v2/2026-08-24_frozen24_v1/07_reports/REPORT.html`
- Aggregate metrics: `releases/2026-08-24_frozen24_v1/reporting_v2/2026-08-24_frozen24_v1/05_evaluation/metrics_long.csv`
- Aggregate contrasts: `releases/2026-08-24_frozen24_v1/reporting_v2/2026-08-24_frozen24_v1/05_evaluation/contrasts_long.csv`
- Reconstruction evaluation summary: `releases/2026-08-24_frozen24_v1/evaluation/EVALUATION.json`
- Reviewed derived comparisons: `derived/`
