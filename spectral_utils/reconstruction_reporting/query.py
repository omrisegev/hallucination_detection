"""DuckDB build and safe query helpers for reconstruction benchmark releases."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .io import ReleaseLayout, read_tidy_csv, sha256_file
from .registry import validate_registry
from .schemas import (
    BOOLEAN_FIELDS,
    FLOAT_FIELDS,
    INTEGER_FIELDS,
    JSON_FIELDS,
    MissingOptionalDependency,
    SchemaError,
    TABLE_FIELDS,
    canonical_json_bytes,
    canonical_sha256,
)


VIEW_NAMES = (
    "v_atomic_leaderboard",
    "v_dataset_leaderboard",
    "v_task_leaderboard",
    "v_release_leaderboard",
    "v_processbench_localization",
    "v_prmbench_error_class",
    "v_prefix_by_budget",
    "v_graph_assumption_checks",
    "v_graph_examples",
)

QUERY_FILTER_COLUMNS = frozenset(
    (
        "lane_id",
        "task_id",
        "dataset_id",
        "population_id",
        "cell_id",
        "slice_id",
        "cohort_id",
        "method_id",
        "method_version_id",
        "adapter_id",
        "system_id",
        "comparison_group_id",
        "aggregation_id",
        "aggregation_level",
        "metric_id",
        "status",
        "evidence_grade",
        "fidelity",
        "access_contract_id",
        "feature_contract_id",
        "evaluator_id",
    )
)


def _duckdb_module() -> Any:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise MissingOptionalDependency(
            "The query database requires DuckDB. Install the reporting dependencies "
            "from scripts/reconstruction_benchmark/requirements-reporting.txt."
        ) from exc
    return duckdb


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_type(field: str) -> str:
    if field in BOOLEAN_FIELDS:
        return "BOOLEAN"
    if field in INTEGER_FIELDS:
        return "BIGINT"
    if field in FLOAT_FIELDS:
        return "DOUBLE"
    return "VARCHAR"


def _sql_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field in JSON_FIELDS:
        return canonical_json_bytes(value).decode("utf-8")
    return value


def _create_tidy_table(connection: Any, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = TABLE_FIELDS[table]
    columns = ",".join(f"{_quote_identifier(field)} {_sql_type(field)}" for field in fields)
    connection.execute(f"CREATE TABLE {_quote_identifier(table)} ({columns})")
    if not rows:
        return
    placeholders = ",".join("?" for _ in fields)
    query = f"INSERT INTO {_quote_identifier(table)} VALUES ({placeholders})"
    values = [tuple(_sql_value(field, row.get(field)) for field in fields) for row in rows]
    connection.executemany(query, values)


def _create_dimension_table(
    connection: Any,
    table: str,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    columns = ",".join(f"{_quote_identifier(field)} VARCHAR" for field in fields)
    connection.execute(f"CREATE TABLE {_quote_identifier(table)} ({columns})")
    values = []
    for row in rows:
        values.append(
            tuple(
                canonical_json_bytes(row.get(field)).decode("utf-8")
                if isinstance(row.get(field), (Mapping, list, tuple))
                else (None if row.get(field) is None else str(row.get(field)))
                for field in fields
            )
        )
    if values:
        placeholders = ",".join("?" for _ in fields)
        connection.executemany(
            f"INSERT INTO {_quote_identifier(table)} VALUES ({placeholders})",
            values,
        )


def _leader_view_sql(view_name: str, level: str) -> str:
    """SQL for a point leaderboard with a clearly labelled CI-overlap set."""

    return f"""
CREATE VIEW {view_name} AS
WITH base AS (
    SELECT *
    FROM metrics
    WHERE aggregation_level = '{level}'
      AND status IN ('OK', 'OK_FALLBACK')
      AND value IS NOT NULL
), ranked AS (
    SELECT
        base.*,
        DENSE_RANK() OVER (
            PARTITION BY comparison_group_id
            ORDER BY
                CASE WHEN better_direction = 'higher' THEN value END DESC NULLS LAST,
                CASE WHEN better_direction = 'lower' THEN value END ASC NULLS LAST
        ) AS point_rank
    FROM base
)
SELECT
    ranked.*,
    ranked.point_rank = 1 AS point_leader,
    CASE
        WHEN ranked.point_rank = 1 THEN TRUE
        WHEN ranked.ci_low IS NULL OR ranked.ci_high IS NULL THEN FALSE
        ELSE EXISTS (
            SELECT 1
            FROM ranked AS leader
            WHERE leader.comparison_group_id = ranked.comparison_group_id
              AND leader.point_rank = 1
              AND leader.ci_low IS NOT NULL
              AND leader.ci_high IS NOT NULL
              AND NOT (
                  ranked.ci_high < leader.ci_low
                  OR ranked.ci_low > leader.ci_high
              )
        )
    END AS uncertainty_tie,
    '95% marginal CI overlaps point leader; paired contrasts remain inferential' AS uncertainty_tie_rule
FROM ranked
""".strip()


def query_view_sql() -> list[str]:
    """Return the deterministic SQL view definitions for audit and tests."""

    return [
        _leader_view_sql("v_atomic_leaderboard", "cell"),
        _leader_view_sql("v_dataset_leaderboard", "dataset"),
        _leader_view_sql("v_task_leaderboard", "task"),
        _leader_view_sql("v_release_leaderboard", "release"),
        """
CREATE VIEW v_processbench_localization AS
SELECT leaderboard.*, cells.generation_model_id, cells.scorer_model_id,
       cells.dataset_family, slices.slice_dimension, slices.slice_value
FROM v_atomic_leaderboard AS leaderboard
JOIN cells USING (cell_id)
JOIN slices USING (slice_id)
WHERE leaderboard.task_id = 'localization'
  AND leaderboard.dataset_id = 'processbench'
""".strip(),
        """
CREATE VIEW v_prmbench_error_class AS
SELECT leaderboard.*, slices.slice_dimension, slices.slice_value
FROM v_atomic_leaderboard AS leaderboard
JOIN slices USING (slice_id)
WHERE leaderboard.dataset_id = 'prmbench'
  AND slices.slice_dimension = 'error_class'
""".strip(),
        """
CREATE VIEW v_prefix_by_budget AS
SELECT leaderboard.*, cells.generation_model_id, cells.scorer_model_id,
       slices.slice_dimension, slices.slice_value
FROM v_atomic_leaderboard AS leaderboard
JOIN cells USING (cell_id)
JOIN slices USING (slice_id)
WHERE leaderboard.task_id = 'early_detection'
  AND slices.slice_dimension = 'budget_tokens'
""".strip(),
        """
CREATE VIEW v_graph_assumption_checks AS
SELECT diagnostics.*, methods.display_name AS method_display_name,
       methods.plain_summary AS method_plain_summary,
       cells.dataset_family, cells.generation_model_id, cells.scorer_model_id
FROM graph_diagnostics AS diagnostics
JOIN methods USING (method_id)
JOIN cells USING (cell_id)
""".strip(),
        """
CREATE VIEW v_graph_examples AS
SELECT examples.*, methods.display_name AS method_display_name,
       cells.dataset_family, cells.generation_model_id, cells.scorer_model_id
FROM graph_examples AS examples
JOIN methods USING (method_id)
JOIN cells USING (cell_id)
""".strip(),
    ]


def build_duckdb(
    release_root: os.PathLike[str] | str,
    *,
    database_path: Optional[os.PathLike[str] | str] = None,
    overwrite: bool = False,
    atomic: bool = True,
) -> dict[str, Any]:
    """Build a portable analytical database from validated release artifacts.

    ``atomic=False`` is reserved for a unique unpublished staging tree whose
    directory rename is the publication boundary.
    """

    duckdb = _duckdb_module()
    layout = ReleaseLayout.from_root(release_root)
    target = Path(database_path).resolve() if database_path is not None else layout.database
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing DuckDB file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    destination = (
        target.with_name(f".{target.name}.tmp-{os.getpid()}")
        if atomic
        else target
    )
    if destination.exists():
        destination.unlink()

    registry = validate_registry(json.loads(layout.registry_json.read_text(encoding="utf-8")))
    table_rows = {
        "metrics": read_tidy_csv(layout.metrics_csv, "metrics"),
        "contrasts": read_tidy_csv(layout.contrasts_csv, "contrasts"),
        "coverage": read_tidy_csv(layout.coverage_csv, "coverage"),
        "graph_diagnostics": read_tidy_csv(layout.graph_diagnostics_csv, "graph_diagnostics"),
        "graph_examples": read_tidy_csv(layout.graph_examples_csv, "graph_examples"),
    }
    source_paths = {
        "registry": layout.registry_json,
        "predictions": layout.predictions_parquet,
        "metrics": layout.metrics_csv,
        "contrasts": layout.contrasts_csv,
        "coverage": layout.coverage_csv,
        "graph_diagnostics": layout.graph_diagnostics_csv,
        "graph_examples": layout.graph_examples_csv,
    }
    source_sha256 = {
        name: sha256_file(path)
        for name, path in sorted(source_paths.items())
        if path.exists()
    }
    logical_sha256 = canonical_sha256(
        {
            "schema": "reconstruction_duckdb_logical_v1",
            "release_id": registry["release_id"],
            "registry_sha256": registry["registry_sha256"],
            "source_sha256": source_sha256,
            "view_sql": query_view_sql(),
        }
    )
    connection = duckdb.connect(str(destination))
    try:
        connection.execute("BEGIN TRANSACTION")
        if layout.predictions_parquet.exists():
            connection.execute(
                "CREATE TABLE predictions AS SELECT * FROM read_parquet(?)",
                [str(layout.predictions_parquet)],
            )
        else:
            _create_tidy_table(connection, "predictions", [])
        for table, rows in table_rows.items():
            _create_tidy_table(connection, table, rows)

        _create_dimension_table(
            connection,
            "methods",
            (
                "method_id",
                "display_name",
                "family_id",
                "plain_summary",
                "input_operation_output",
                "formula",
                "access_tier",
                "supervision",
                "donor_regime",
                "role",
                "research_stage",
                "style",
            ),
            registry["methods"],
        )
        _create_dimension_table(
            connection,
            "datasets",
            (
                "dataset_id",
                "task_id",
                "display_name",
                "description",
                "prediction_unit",
                "label_definition",
                "positive_class",
                "dataset_family",
                "revision",
            ),
            registry["datasets"],
        )
        _create_dimension_table(
            connection,
            "cells",
            (
                "cell_id",
                "population_id",
                "task_id",
                "dataset_id",
                "generation_model_id",
                "scorer_model_id",
                "split_id",
                "decoding_id",
                "dataset_family",
                "expected_n",
                "status",
            ),
            registry["cells"],
        )
        _create_dimension_table(
            connection,
            "slices",
            (
                "slice_id",
                "population_id",
                "cell_id",
                "slice_dimension",
                "slice_value",
                "display_name",
                "expected_n",
            ),
            registry["slices"],
        )
        _create_dimension_table(
            connection,
            "systems",
            (
                "system_id",
                "method_version_id",
                "adapter_id",
                "access_contract_id",
                "display_name",
                "enabled",
            ),
            registry["systems"],
        )
        for statement in query_view_sql():
            connection.execute(statement)
        connection.execute("CREATE TABLE reporting_metadata (key VARCHAR PRIMARY KEY, value VARCHAR)")
        metadata = {
            "schema": "reconstruction_duckdb_v1",
            "release_id": registry["release_id"],
            "registry_sha256": registry["registry_sha256"],
            "logical_sha256": logical_sha256,
            "source_sha256": json.dumps(source_sha256, sort_keys=True, separators=(",", ":")),
            "view_names": json.dumps(VIEW_NAMES, separators=(",", ":")),
        }
        connection.executemany(
            "INSERT INTO reporting_metadata VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        connection.close()
    if atomic:
        os.replace(destination, target)
    return {
        "schema": "reconstruction_duckdb_v1",
        "path": target.as_posix(),
        "release_id": registry["release_id"],
        "logical_sha256": logical_sha256,
        "source_sha256": source_sha256,
        "physical_bytes_canonical": False,
        "views": list(VIEW_NAMES),
    }


def _validate_filters(filters: Mapping[str, Any]) -> None:
    unknown = sorted(set(filters) - QUERY_FILTER_COLUMNS)
    if unknown:
        raise SchemaError(f"unsupported query filters: {unknown}")
    for field, value in filters.items():
        if not isinstance(value, str) or not value:
            raise SchemaError(f"query filter {field!r} must be a non-empty string")


def query_results(
    database_path: os.PathLike[str] | str,
    *,
    view: str = "v_atomic_leaderboard",
    filters: Optional[Mapping[str, str]] = None,
    limit: Optional[int] = None,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Run a parameter-bound drill-down query; identifiers come from allowlists."""

    duckdb = _duckdb_module()
    if view not in VIEW_NAMES:
        raise SchemaError(f"view must be one of {VIEW_NAMES!r}")
    filters = dict(filters or {})
    _validate_filters(filters)
    clauses = []
    parameters = []
    for field in sorted(filters):
        clauses.append(f"{_quote_identifier(field)} = ?")
        parameters.append(filters[field])
    query = f"SELECT * FROM {_quote_identifier(view)}"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    order = ["task_id", "dataset_id", "cell_id", "slice_id"]
    if view not in {"v_graph_assumption_checks", "v_graph_examples"}:
        order.append("point_rank")
    order.append("system_id")
    query += " ORDER BY " + ", ".join(_quote_identifier(field) for field in order)
    if limit is not None:
        if type(limit) is bool or not isinstance(limit, int) or limit <= 0:
            raise SchemaError("query limit must be a positive integer")
        query += " LIMIT ?"
        parameters.append(limit)
    connection = duckdb.connect(str(Path(database_path).resolve()), read_only=True)
    try:
        cursor = connection.execute(query, parameters)
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
    finally:
        connection.close()
    return columns, rows


def write_query_csv(
    path: os.PathLike[str] | str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
