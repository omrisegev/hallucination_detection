# Reconstruction benchmark v1

## Frozen-24 scientific sequence

Run this only from the committed, clean reconstruction branch. Replace
`<release_id>` once and keep that immutable ID throughout:

```bash
python scripts/reconstruction_benchmark/prepare_24cell_inputs.py \
  --release-id <release_id> --build both

python scripts/reconstruction_benchmark/build_24cell_group_sidecars.py \
  --release-root results/reconstruction_benchmark_v1/releases/<release_id> \
  --raw-root <verified-raw-cache-root>

python scripts/reconstruction_benchmark/run_24cell_methods.py \
  --release-id <release_id> --build A --jobs 4
python scripts/reconstruction_benchmark/run_24cell_methods.py \
  --release-id <release_id> --build B --jobs 4

python scripts/reconstruction_benchmark/verify_24cell_fits.py \
  --release-id <release_id>

python scripts/reconstruction_benchmark/evaluate_24cell_release.py \
  --release-id <release_id> \
  --group-manifest results/reconstruction_benchmark_v1/releases/<release_id>/group_sidecars/GROUP_SIDECARS.json

python scripts/reconstruction_benchmark/build_24cell_graph_diagnostics.py \
  --release-id <release_id>
```

The fit commands refuse a dirty worktree. Evaluation refuses to open labels
until the complete 24×13 A/B certificate and all 24 source-group sidecars pass.

## External final-answer sequence

The external registry covers compatible response-level populations and keeps
blocked, incompatible, protocol-failed, and quarantined rows explicit. First
materialize hash-frozen assets under one source root, then audit before fitting:

```bash
python scripts/reconstruction_benchmark/audit_external_final_answer.py \
  --source-root <materialized-source-root> --deep \
  --output <scientific-release>/external_final_answer/APPLICABILITY.json

python scripts/reconstruction_benchmark/prepare_external_final_answer.py \
  --release-id <release_id> --build A --source-root <materialized-source-root>
python scripts/reconstruction_benchmark/prepare_external_final_answer.py \
  --release-id <release_id> --build B --source-root <materialized-source-root>

python scripts/reconstruction_benchmark/run_external_final_answer_methods.py \
  --release-id <release_id> --build A
python scripts/reconstruction_benchmark/run_external_final_answer_methods.py \
  --release-id <release_id> --build B

python scripts/reconstruction_benchmark/verify_external_final_answer_ab.py \
  --release-id <release_id>

python scripts/reconstruction_benchmark/evaluate_external_final_answer.py \
  --release-id <release_id> --build A \
  --source-root <materialized-source-root>
```

The external evaluator opens labels only after independently rechecking the
full A/B certificate. Population-level estimates use the registry's linked
source groups and aggregation rule; per-cell rows remain available separately.

External fitting never receives historical source IDs. A controller-only,
gitignored key derives row IDs with a domain-separated HMAC; the fit-visible
contract contains only the row binding and a keyed commitment to private group
metadata. Rows are sorted by opaque row ID before the single mixed-v2 pass, so
a semantic raw-key order cannot become a graph tie breaker. Group membership,
class counts, raw paths, labels, and the key are reconstructed only after the
scores are frozen. The worker runs from an exact fit-only code capsule and can
read only the prepared matrices, fit-safe registries, trusted runtime, and its
output directory. This audit-hook tier protects the frozen first-party worker
from accidental target access; it is not described as a sandbox for hostile
native code. Any input tree made under the former raw-ID contract is audit-only
and is actively refused; create a new release ID after a contract change.

## EDIS/AIME multi-sample sequence

EDIS is a separate multi-pass cost tier and never enters the one-pass response
leaderboard. The registered sources are AIME24 full (5,760 traces), AMC23 full
(3,840), GSM8K pilot3 (720), and MATH-500 pilot3 (720). Every stored acquisition
cell failed at least one registered gate, so the resulting metrics are
descriptive stress evidence only.

```bash
python scripts/reconstruction_benchmark/prepare_edis.py \
  --release-id <edis_release_id> --build A \
  --source-root local_cache/reconstruction_edis_stage
python scripts/reconstruction_benchmark/prepare_edis.py \
  --release-id <edis_release_id> --build B \
  --source-root local_cache/reconstruction_edis_stage

python scripts/reconstruction_benchmark/run_edis_methods.py \
  --release-id <edis_release_id> --build A \
  --source-root local_cache/reconstruction_edis_stage
python scripts/reconstruction_benchmark/run_edis_methods.py \
  --release-id <edis_release_id> --build B \
  --source-root local_cache/reconstruction_edis_stage

python scripts/reconstruction_benchmark/verify_edis_ab.py \
  --release-id <edis_release_id>

python scripts/reconstruction_benchmark/evaluate_edis.py \
  --release-id <edis_release_id> --build A \
  --source-root local_cache/reconstruction_edis_stage
python scripts/reconstruction_benchmark/evaluate_edis.py \
  --release-id <edis_release_id> --build B \
  --source-root local_cache/reconstruction_edis_stage

python scripts/reconstruction_benchmark/verify_edis_evaluation_ab.py \
  --release-id <edis_release_id>
```

The two evaluations use the same 20,000 paired source-question bootstrap draws.
Question-content commitments prove the cross-temperature linkage before any
linked resampling, and the final verifier compares the post-label A/B tables as
well as the prepared matrices and score artifacts.

## Reporting layer

This directory contains only the final, data-facing layer of the reconstruction
benchmark. It does not run a detector or compute experiment scores.

## Inputs

`build_reporting_release.py` requires one validated research registry and five
producer-owned long-form tables:

- predictions (`.json`, `.jsonl`, `.csv`, or schema-tagged `.parquet`);
- metrics;
- paired contrasts;
- coverage/status rows;
- graph diagnostics.

The exact fields and controlled vocabularies live in
`spectral_utils/reconstruction_reporting/schemas.py`. The registry separates
`method_id`, `method_version_id`, `adapter_id`, and `system_id`, and separately
registers task, dataset, population, cell, slice, feature/access/evaluator
contracts, and aggregation rules.

Every expected system × cell × slice combination must have one coverage row.
A failure is a named status such as `BLOCKED_ASSET`, `ADAPTER_MISSING`, or
`METRIC_UNDEFINED_SINGLE_CLASS`; it is never an omitted row or a numeric zero.

## Build

Install the two output-format dependencies in the intended clean environment:

```bash
python -m pip install -r scripts/reconstruction_benchmark/requirements-reporting.txt
```

Validate all dependency-free scientific contracts without writing anything:

```bash
python scripts/reconstruction_benchmark/build_24cell_reporting_inputs.py \
  --release-id <release_id> \
  --evaluation-dir results/reconstruction_benchmark_v1/releases/<release_id>/evaluation \
  --graph-diagnostics-dir results/reconstruction_benchmark_v1/releases/<release_id>/graph_diagnostics \
  --published-comparator-registry configs/reconstruction_benchmark_v1/frozen24_published_comparator_registry_v1.json \
  --output-dir results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs

python scripts/reconstruction_benchmark/build_reporting_release.py \
  --release-root results/reconstruction_benchmark_v1/releases/<release_id>/reporting/<release_id> \
  --bridge-manifest results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs/BRIDGE_MANIFEST.json \
  --registry results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs/research_registry.json \
  --predictions results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs/predictions.jsonl \
  --metrics results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs/metrics_long.csv \
  --contrasts results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs/contrasts_long.csv \
  --coverage results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs/coverage_long.csv \
  --graph-diagnostics results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs/graph_diagnostics_long.csv \
  --graph-examples results/reconstruction_benchmark_v1/releases/<release_id>/reporting_inputs/graph_examples_long.csv \
  --validate-only
```

Remove `--validate-only` to publish an immutable release directory. The command
refuses to overwrite an existing release. It writes deterministic CSV/JSON,
schema-tagged Parquet, `benchmark.duckdb`, a content-addressed plot manifest,
one CSV per static plot, five ready-to-use leaderboard CSVs under
`05_evaluation/leaderboards/`, and a self-contained `REPORT.html` that opens
through `file://` without network resources. The leaderboard files correspond
exactly to the validated cell, dataset, domain/model-family slice, task, and
release views; they contain only rankable `OK`/`OK_FALLBACK` rows with non-null
metric values. Published paper values are copied separately as
`01_registries/published_comparators.json`; they never enter a leaderboard,
plot-data table, or delta.

## Query

For example, to retrieve every compatible system for a ProcessBench
localization cell:

```bash
python scripts/reconstruction_benchmark/query_results.py \
  results/reconstruction_benchmark_v1/releases/<release_id>/reporting/<release_id>/05_evaluation/benchmark.duckdb \
  --view v_processbench_localization \
  --dataset-id processbench \
  --cell-id <cell_id>
```

The database exposes:

- `v_atomic_leaderboard`;
- `v_dataset_leaderboard`;
- `v_task_leaderboard`;
- `v_slice_leaderboard`;
- `v_release_leaderboard`;
- `v_processbench_localization`;
- `v_prmbench_error_class`;
- `v_prefix_by_budget`;
- `v_graph_assumption_checks`.

Rows are ranked only within one exact `comparison_group_id`. `point_leader`
means the best point estimate. `uncertainty_tie` is a descriptive marginal-CI
overlap set; paired contrasts remain the inferential source of truth.
