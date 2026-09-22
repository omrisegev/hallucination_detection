#!/usr/bin/env python3
"""Build the tidy/query/report layer of one immutable benchmark release.

This command consumes producer-owned prediction and evaluation records.  It
does not run a detector, compute a metric, or copy a historical experiment
score.  A release is published only after every input passes the shared schema,
registry-reference, coverage, comparison-group, and plot-data gates.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.reconstruction_reporting.io import (  # noqa: E402
    ReleaseLayout,
    build_reporting_manifest,
    load_records,
    materialize_plot_data,
    sha256_file,
    write_canonical_json,
    write_parquet,
    write_tidy_csv,
    validate_plot_data_sources,
)
from spectral_utils.reconstruction_reporting.query import (  # noqa: E402
    build_duckdb,
    export_leaderboard_csvs,
)
from spectral_utils.reconstruction_reporting.published_context import (  # noqa: E402
    frozen24_cell_auroc_logical_sha256,
    load_published_context_projection,
    validate_published_context_projection,
)
from spectral_utils.reconstruction_reporting.registry import (  # noqa: E402
    expected_coverage_rows,
    validate_registry,
    validate_result_references,
)
from spectral_utils.reconstruction_reporting.report import (  # noqa: E402
    default_plot_manifest,
    write_report,
)
from spectral_utils.reconstruction_reporting.schemas import (  # noqa: E402
    SchemaError,
    canonical_sha256,
    validate_comparison_groups,
    validate_equal_unit_aggregates,
    validate_expected_coverage,
    validate_plot_manifest,
)
from spectral_utils.reconstruction_benchmark.reporting_bridge import BRIDGE_SCHEMA  # noqa: E402


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument(
        "--bridge-manifest",
        type=Path,
        required=True,
        help="Signed BRIDGE_MANIFEST.json produced by build_24cell_reporting_inputs.py.",
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--graph-diagnostics", type=Path, required=True)
    parser.add_argument("--graph-examples", type=Path, required=True)
    parser.add_argument("--plot-manifest", type=Path)
    parser.add_argument("--title", default="Reconstruction benchmark explorer")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run every dependency-free scientific gate without writing a release.",
    )
    return parser.parse_args(argv)


def _read_registry(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read registry {path}: {exc}") from exc
    return validate_registry(value)


def _read_plot_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read plot manifest {path}: {exc}") from exc
    return validate_plot_manifest(value)


def _verify_scientific_bridge(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.bridge_manifest.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read bridge manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != BRIDGE_SCHEMA:
        raise SchemaError("bridge manifest schema drift")
    payload_sha256 = manifest.get("payload_sha256")
    body = dict(manifest)
    body.pop("payload_sha256", None)
    if payload_sha256 != canonical_sha256(body):
        raise SchemaError("bridge manifest payload hash mismatch")
    if manifest.get("scientific_publication_eligible") is not True:
        raise SchemaError("bridge manifest is explicitly ineligible for scientific publication")
    if manifest.get("graph_diagnostics_status") != "VERIFIED_SIGNED_SOURCE_CONVERTED":
        raise SchemaError("bridge manifest lacks a verified signed graph package")
    if manifest.get("published_context_status") != "VERIFIED_SEPARATE_REPORT_ONLY_ARTIFACT":
        raise SchemaError("bridge manifest lacks a verified published-context artifact")

    bridge_root = manifest_path.parent.resolve()
    input_paths = {
        "research_registry.json": args.registry,
        "published_comparators.json": bridge_root / "published_comparators.json",
        "predictions.jsonl": args.predictions,
        "metrics_long.csv": args.metrics,
        "contrasts_long.csv": args.contrasts,
        "coverage_long.csv": args.coverage,
        "graph_diagnostics_long.csv": args.graph_diagnostics,
        "graph_examples_long.csv": args.graph_examples,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SchemaError("bridge manifest artifacts are missing")
    by_path: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise SchemaError("bridge manifest contains an invalid artifact record")
        relative_path = str(record["path"])
        if relative_path in by_path:
            raise SchemaError(
                f"bridge manifest contains duplicate artifact path: {relative_path}"
            )
        by_path[relative_path] = record
    for relative_path, supplied_path in input_paths.items():
        resolved = supplied_path.resolve()
        if resolved != bridge_root / relative_path:
            raise SchemaError(
                f"report input {relative_path} must come from the signed bridge directory"
            )
        record = by_path.get(relative_path)
        if record is None:
            raise SchemaError(f"bridge manifest does not bind {relative_path}")
        if record.get("file_sha256") != sha256_file(resolved):
            raise SchemaError(f"bridge artifact hash mismatch: {relative_path}")
    return manifest


def load_and_validate(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    bridge_manifest = _verify_scientific_bridge(args)
    registry = _read_registry(args.registry)
    rows = {
        "predictions": load_records(args.predictions, "predictions"),
        "metrics": load_records(args.metrics, "metrics"),
        "contrasts": load_records(args.contrasts, "contrasts"),
        "coverage": load_records(args.coverage, "coverage"),
        "graph_diagnostics": load_records(args.graph_diagnostics, "graph_diagnostics"),
        "graph_examples": load_records(args.graph_examples, "graph_examples"),
    }
    cell_ids = sorted(
        {
            str(row["cell_id"])
            for row in rows["metrics"]
            if row.get("aggregation_level") == "cell"
        },
        key=lambda value: value.encode("utf-8"),
    )
    published_context = validate_published_context_projection(
        load_published_context_projection(
            args.bridge_manifest.resolve().parent / "published_comparators.json"
        ),
        release_id=str(registry["release_id"]),
        expected_cell_ids=cell_ids,
        metrics_cell_auroc_logical_sha256=frozen24_cell_auroc_logical_sha256(
            rows["metrics"],
            expected_cell_ids=cell_ids,
            expected_method_ids=[
                str(method["method_id"]) for method in registry["methods"]
            ],
        ),
    )
    if bridge_manifest.get("published_context_status_counts") != published_context[
        "status_counts"
    ]:
        raise SchemaError("bridge manifest published-context counts drift")
    source_hashes = bridge_manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise SchemaError("bridge manifest source hashes are missing")
    if source_hashes.get("published_context_logical_sha256") != canonical_sha256(
        published_context
    ):
        raise SchemaError("bridge manifest published-context logical hash mismatch")
    if bridge_manifest.get("release_id") != registry["release_id"]:
        raise SchemaError("bridge manifest release_id does not match registry")
    expected_counts = bridge_manifest.get("row_counts")
    if not isinstance(expected_counts, dict):
        raise SchemaError("bridge manifest row counts are missing")
    observed_counts = {table: len(table_rows) for table, table_rows in rows.items()}
    if expected_counts != observed_counts:
        raise SchemaError("bridge manifest row counts disagree with report inputs")
    validate_result_references(registry, rows)
    validate_comparison_groups(rows["metrics"])
    validate_expected_coverage(expected_coverage_rows(registry), rows["coverage"])
    validate_equal_unit_aggregates(rows["metrics"], registry["aggregations"])
    plot_manifest = (
        _read_plot_manifest(args.plot_manifest)
        if args.plot_manifest is not None
        else default_plot_manifest(registry["release_id"], rows)
    )
    if plot_manifest["release_id"] != registry["release_id"]:
        raise SchemaError("plot manifest release_id does not match registry")
    validate_plot_data_sources(plot_manifest, rows)
    return registry, rows, plot_manifest, bridge_manifest, published_context


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ("duckdb", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _artifact_record(layout: ReleaseLayout, path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(layout.root).as_posix(),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        **extra,
    }


def build_release(
    release_root: Path,
    *,
    registry: Mapping[str, Any],
    rows: Mapping[str, list[dict[str, Any]]],
    plot_manifest: Mapping[str, Any],
    title: str,
    bridge_manifest: Mapping[str, Any],
    published_context: Mapping[str, Any] | None = None,
) -> Path:
    release_root = release_root.resolve()
    if release_root.exists():
        raise FileExistsError(
            f"release roots are immutable; choose a new release_id/path instead of overwriting {release_root}"
        )
    if release_root.name != registry["release_id"]:
        raise SchemaError(
            f"release directory name {release_root.name!r} must equal registry release_id {registry['release_id']!r}"
        )
    release_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_root.name}.building-", dir=release_root.parent))
    layout = ReleaseLayout.from_root(staging)
    artifacts: list[dict[str, Any]] = []
    duckdb_source_sha256: dict[str, str] = {}
    try:
        layout.create_directories()
        write_canonical_json(layout.registry_json, registry, atomic=False)
        registry_record = _artifact_record(layout, layout.registry_json, kind="registry")
        artifacts.append(registry_record)
        duckdb_source_sha256["registry"] = registry_record["file_sha256"]
        write_canonical_json(
            layout.bridge_manifest,
            bridge_manifest,
            atomic=False,
        )
        bridge_record = _artifact_record(
            layout,
            layout.bridge_manifest,
            kind="scientific_bridge_attestation",
        )
        artifacts.append(bridge_record)
        bridge_attestation = {
            "schema": str(bridge_manifest["schema"]),
            "payload_sha256": str(bridge_manifest["payload_sha256"]),
            "file_sha256": bridge_record["file_sha256"],
            "scientific_publication_eligible": bool(
                bridge_manifest["scientific_publication_eligible"]
            ),
            "graph_diagnostics_status": str(
                bridge_manifest["graph_diagnostics_status"]
            ),
            "relative_path": bridge_record["relative_path"],
        }

        if published_context is not None:
            cell_ids = sorted(
                {
                    str(row["cell_id"])
                    for row in rows["metrics"]
                    if row.get("aggregation_level") == "cell"
                },
                key=lambda value: value.encode("utf-8"),
            )
            published_context = validate_published_context_projection(
                published_context,
                release_id=str(registry["release_id"]),
                expected_cell_ids=cell_ids,
                metrics_cell_auroc_logical_sha256=(
                    frozen24_cell_auroc_logical_sha256(
                        rows["metrics"],
                        expected_cell_ids=cell_ids,
                        expected_method_ids=[
                            str(method["method_id"])
                            for method in registry["methods"]
                        ],
                    )
                ),
            )
            write_canonical_json(
                layout.published_comparators_json,
                published_context,
                atomic=False,
            )
            published_record = _artifact_record(
                layout,
                layout.published_comparators_json,
                kind="published_context_report_only",
                logical_sha256=canonical_sha256(published_context),
                row_count=len(published_context["rows"]),
            )
            artifacts.append(published_record)
            bridge_attestation.update(
                {
                    "published_context_relative_path": published_record[
                        "relative_path"
                    ],
                    "published_context_file_sha256": published_record[
                        "file_sha256"
                    ],
                    "published_context_logical_sha256": published_record[
                        "logical_sha256"
                    ],
                    "published_context_status_counts": dict(
                        published_context["status_counts"]
                    ),
                }
            )

        predictions_record = {
            **write_parquet(
                layout.predictions_parquet,
                "predictions",
                rows["predictions"],
                atomic=False,
            ),
            "relative_path": layout.predictions_parquet.relative_to(layout.root).as_posix(),
            "kind": "predictions",
        }
        artifacts.append(predictions_record)
        duckdb_source_sha256["predictions"] = predictions_record["file_sha256"]
        table_targets = {
            "metrics": (layout.metrics_csv, layout.metrics_parquet),
            "contrasts": (layout.contrasts_csv, layout.contrasts_parquet),
            "coverage": (layout.coverage_csv, layout.coverage_parquet),
            "graph_diagnostics": (layout.graph_diagnostics_csv, layout.graph_diagnostics_parquet),
            "graph_examples": (layout.graph_examples_csv, layout.graph_examples_parquet),
        }
        for table, (csv_path, parquet_path) in table_targets.items():
            csv_record = write_tidy_csv(
                csv_path,
                table,
                rows[table],
                atomic=False,
            )
            csv_record.update(
                relative_path=csv_path.relative_to(layout.root).as_posix(),
                kind="tidy_csv",
            )
            artifacts.append(csv_record)
            duckdb_source_sha256[table] = csv_record["file_sha256"]
            parquet_record = write_parquet(
                parquet_path,
                table,
                rows[table],
                atomic=False,
            )
            parquet_record.update(
                relative_path=parquet_path.relative_to(layout.root).as_posix(),
                kind="tidy_parquet",
            )
            artifacts.append(parquet_record)

        write_canonical_json(layout.plot_manifest, plot_manifest, atomic=False)
        artifacts.append(_artifact_record(layout, layout.plot_manifest, kind="plot_manifest"))
        artifacts.extend(
            materialize_plot_data(
                layout,
                plot_manifest,
                rows,
                atomic=False,
            )
        )
        report_record = write_report(
            layout.report_html,
            registry=registry,
            rows_by_table=rows,
            plot_manifest=plot_manifest,
            title=title,
            published_context=published_context,
            atomic=False,
        )
        report_record.update(
            relative_path=layout.report_html.relative_to(layout.root).as_posix(),
            kind="self_contained_html",
        )
        artifacts.append(report_record)

        database_record = build_duckdb(
            layout.root,
            atomic=False,
            source_sha256=duckdb_source_sha256,
        )
        database_record.update(
            path=layout.database.name,
            relative_path=layout.database.relative_to(layout.root).as_posix(),
            kind="duckdb",
        )
        artifacts.append(database_record)
        leaderboard_records = export_leaderboard_csvs(
            layout.root,
            atomic=False,
        )
        for record in leaderboard_records:
            record["source_database_logical_sha256"] = database_record["logical_sha256"]
        artifacts.extend(leaderboard_records)

        manifest = build_reporting_manifest(
            layout,
            release_id=registry["release_id"],
            registry=registry,
            artifact_records=artifacts,
            optional_dependencies=_dependency_versions(),
            bridge_attestation=bridge_attestation,
        )
        write_canonical_json(layout.reporting_manifest, manifest, atomic=False)
        os.replace(staging, release_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return release_root


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    (
        registry,
        rows,
        plot_manifest,
        bridge_manifest,
        published_context,
    ) = load_and_validate(args)
    summary = {
        "release_id": registry["release_id"],
        "registry_sha256": registry["registry_sha256"],
        "row_counts": {table: len(values) for table, values in rows.items()},
        "plot_count": len(plot_manifest["plots"]),
    }
    if args.validate_only:
        print(json.dumps({"status": "VALID", **summary}, sort_keys=True))
        return 0
    output = build_release(
        args.release_root,
        registry=registry,
        rows=rows,
        plot_manifest=plot_manifest,
        title=args.title,
        bridge_manifest=bridge_manifest,
        published_context=published_context,
    )
    print(json.dumps({"status": "BUILT", "release_root": str(output), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
