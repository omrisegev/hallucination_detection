"""Deterministic I/O for reconstruction-reporting release artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .registry import validate_registry
from .schemas import (
    BOOLEAN_FIELDS,
    FLOAT_FIELDS,
    INTEGER_FIELDS,
    JSON_FIELDS,
    MissingOptionalDependency,
    SCHEMA_REVISION,
    SchemaError,
    TABLE_FIELDS,
    canonical_json_bytes,
    record_sort_key,
    table_sha256,
    validate_plot_manifest,
    validate_records,
)


def sha256_file(path: os.PathLike[str] | str, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_canonical_json(path: os.PathLike[str] | str, value: Any) -> None:
    _atomic_write_bytes(Path(path), canonical_json_bytes(value) + b"\n")


def read_canonical_json(path: os.PathLike[str] | str) -> Any:
    source = Path(path)
    raw = source.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(value) + b"\n":
        raise SchemaError(f"{source} is not canonical JSON with one trailing newline")
    return value


def _encode_csv_value(field: str, value: Any) -> str:
    if value is None:
        return ""
    if field in BOOLEAN_FIELDS:
        return "true" if bool(value) else "false"
    if field in JSON_FIELDS:
        return canonical_json_bytes(value).decode("utf-8")
    if field in FLOAT_FIELDS:
        return format(float(value), ".17g")
    if field in INTEGER_FIELDS:
        return str(int(value))
    return str(value)


def _decode_csv_value(field: str, value: str) -> Any:
    if value == "":
        return None if field in FLOAT_FIELDS | BOOLEAN_FIELDS else ([] if field in JSON_FIELDS else "")
    if field in BOOLEAN_FIELDS:
        if value not in ("true", "false"):
            raise SchemaError(f"CSV boolean field {field!r} has invalid value {value!r}")
        return value == "true"
    if field in INTEGER_FIELDS:
        try:
            return int(value)
        except ValueError as exc:
            raise SchemaError(f"CSV integer field {field!r} has invalid value {value!r}") from exc
    if field in FLOAT_FIELDS:
        try:
            return float(value)
        except ValueError as exc:
            raise SchemaError(f"CSV numeric field {field!r} has invalid value {value!r}") from exc
    if field in JSON_FIELDS:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"CSV JSON field {field!r} is invalid") from exc
        return parsed
    return value


def write_tidy_csv(
    path: os.PathLike[str] | str,
    table: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    atomic: bool = True,
) -> dict[str, Any]:
    """Validate, sort, and write one long-form CSV deterministically.

    The default is a standalone atomic file replacement.  A caller that already
    owns a unique unpublished staging tree may set ``atomic=False``: the outer
    directory rename then remains the sole atomic publication boundary, and an
    exception still discards the entire staging tree.
    """

    if table not in TABLE_FIELDS:
        raise SchemaError(f"unknown table {table!r}")
    normalized = validate_records(table, rows)
    fields = TABLE_FIELDS[table]
    for index, row in enumerate(normalized):
        unknown = sorted(set(row) - set(fields))
        if unknown:
            raise SchemaError(f"{table} row {index} contains unregistered fields: {unknown}")
    ordered = sorted(normalized, key=lambda row: record_sort_key(table, row))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not atomic and target.exists():
        raise FileExistsError(f"staged CSV target already exists: {target}")
    destination = (
        target.with_name(f".{target.name}.tmp-{os.getpid()}")
        if atomic
        else target
    )
    try:
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in ordered:
                writer.writerow({field: _encode_csv_value(field, row.get(field)) for field in fields})
        if atomic:
            os.replace(destination, target)
    finally:
        if atomic and destination.exists():
            destination.unlink()
    return {
        "table": table,
        "schema": SCHEMA_REVISION,
        "path": target.name,
        "row_count": len(ordered),
        "logical_sha256": table_sha256(table, ordered),
        "file_sha256": sha256_file(target),
    }


def read_tidy_csv(path: os.PathLike[str] | str, table: str) -> list[dict[str, Any]]:
    if table not in TABLE_FIELDS:
        raise SchemaError(f"unknown table {table!r}")
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = list(TABLE_FIELDS[table])
        if reader.fieldnames != expected:
            raise SchemaError(
                f"{source} columns differ from {table} contract: expected={expected}, "
                f"observed={reader.fieldnames}"
            )
        rows = [
            {field: _decode_csv_value(field, raw[field]) for field in expected}
            for raw in reader
        ]
    return validate_records(table, rows)


def read_jsonl(path: os.PathLike[str] | str, table: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{line_number} is not valid JSON") from exc
    return validate_records(table, rows)


def load_records(path: os.PathLike[str] | str, table: str) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return read_tidy_csv(source, table)
    if suffix in (".jsonl", ".ndjson"):
        return read_jsonl(source, table)
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and "rows" in value:
            value = value["rows"]
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise SchemaError(f"{source} must contain a list of {table} rows")
        return validate_records(table, value)
    if suffix == ".parquet":
        return read_parquet(source, table)
    raise SchemaError(f"unsupported {table} input format: {source.suffix}")


def _pyarrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise MissingOptionalDependency(
            "Parquet output requires PyArrow. Install the reporting dependencies "
            "from scripts/reconstruction_benchmark/requirements-reporting.txt."
        ) from exc
    return pa, pq


def _arrow_schema(table: str) -> Any:
    pa, _ = _pyarrow_modules()
    fields = []
    for field in TABLE_FIELDS[table]:
        if field in JSON_FIELDS:
            data_type = pa.list_(pa.string())
        elif field in BOOLEAN_FIELDS:
            data_type = pa.bool_()
        elif field in INTEGER_FIELDS:
            data_type = pa.int64()
        elif field in FLOAT_FIELDS:
            data_type = pa.float64()
        else:
            data_type = pa.string()
        nullable = field not in (
            "release_id",
            "run_id",
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
            "feature_contract_id",
            "access_contract_id",
            "evaluator_id",
            "evidence_grade",
            "status",
            "status_detail",
        )
        fields.append(pa.field(field, data_type, nullable=nullable))
    metadata = {
        b"reporting_schema": SCHEMA_REVISION.encode("utf-8"),
        b"table": table.encode("utf-8"),
    }
    return pa.schema(fields, metadata=metadata)


def write_parquet(
    path: os.PathLike[str] | str,
    table: str,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    pa, pq = _pyarrow_modules()
    normalized = validate_records(table, rows)
    ordered = sorted(normalized, key=lambda row: record_sort_key(table, row))
    fields = TABLE_FIELDS[table]
    arrays = {field: [row.get(field) for row in ordered] for field in fields}
    arrow_table = pa.Table.from_pydict(arrays, schema=_arrow_schema(table))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        pq.write_table(
            arrow_table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            data_page_version="1.0",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "table": table,
        "schema": SCHEMA_REVISION,
        "path": target.name,
        "row_count": len(ordered),
        "logical_sha256": table_sha256(table, ordered),
        "file_sha256": sha256_file(target),
    }


def read_parquet(path: os.PathLike[str] | str, table: str) -> list[dict[str, Any]]:
    _, pq = _pyarrow_modules()
    source = Path(path)
    arrow_table = pq.read_table(source)
    if arrow_table.column_names != list(TABLE_FIELDS[table]):
        raise SchemaError(f"{source} Parquet columns do not match the {table} contract")
    metadata = arrow_table.schema.metadata or {}
    if metadata.get(b"reporting_schema") != SCHEMA_REVISION.encode("utf-8"):
        raise SchemaError(f"{source} does not declare reporting schema {SCHEMA_REVISION}")
    if metadata.get(b"table") != table.encode("utf-8"):
        raise SchemaError(f"{source} table metadata is not {table!r}")
    return validate_records(table, arrow_table.to_pylist())


@dataclass(frozen=True)
class ReleaseLayout:
    root: Path

    @classmethod
    def from_root(cls, root: os.PathLike[str] | str) -> "ReleaseLayout":
        return cls(Path(root).resolve())

    @property
    def registries(self) -> Path:
        return self.root / "01_registries"

    @property
    def evaluation(self) -> Path:
        return self.root / "05_evaluation"

    @property
    def diagnostics(self) -> Path:
        return self.root / "06_diagnostics"

    @property
    def reports(self) -> Path:
        return self.root / "07_reports"

    @property
    def registry_json(self) -> Path:
        return self.registries / "research_registry.json"

    @property
    def predictions_parquet(self) -> Path:
        return self.evaluation / "predictions.parquet"

    @property
    def metrics_csv(self) -> Path:
        return self.evaluation / "metrics_long.csv"

    @property
    def metrics_parquet(self) -> Path:
        return self.evaluation / "metrics_long.parquet"

    @property
    def contrasts_csv(self) -> Path:
        return self.evaluation / "contrasts_long.csv"

    @property
    def contrasts_parquet(self) -> Path:
        return self.evaluation / "contrasts_long.parquet"

    @property
    def coverage_csv(self) -> Path:
        return self.evaluation / "coverage_long.csv"

    @property
    def coverage_parquet(self) -> Path:
        return self.evaluation / "coverage_long.parquet"

    @property
    def graph_diagnostics_csv(self) -> Path:
        return self.diagnostics / "graph_diagnostics_long.csv"

    @property
    def graph_diagnostics_parquet(self) -> Path:
        return self.diagnostics / "graph_diagnostics_long.parquet"

    @property
    def graph_examples_csv(self) -> Path:
        return self.diagnostics / "graph_examples_long.csv"

    @property
    def graph_examples_parquet(self) -> Path:
        return self.diagnostics / "graph_examples_long.parquet"

    @property
    def database(self) -> Path:
        return self.evaluation / "benchmark.duckdb"

    @property
    def report_html(self) -> Path:
        return self.reports / "REPORT.html"

    @property
    def plot_manifest(self) -> Path:
        return self.reports / "plot_manifest.json"

    @property
    def plot_data(self) -> Path:
        return self.reports / "plot_data"

    @property
    def reporting_manifest(self) -> Path:
        return self.root / "REPORTING_MANIFEST.json"

    def create_directories(self) -> None:
        for directory in (self.registries, self.evaluation, self.diagnostics, self.reports, self.plot_data):
            directory.mkdir(parents=True, exist_ok=True)


def _match_filters(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for field, expected in filters.items():
        if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
            if row.get(field) not in expected:
                return False
        elif row.get(field) != expected:
            return False
    return True


def materialize_plot_data(
    layout: ReleaseLayout,
    plot_manifest: Mapping[str, Any],
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Write the exact CSV behind each static plot contract."""

    manifest, selected_by_plot = validate_plot_data_sources(plot_manifest, rows_by_table)
    outputs = []
    for plot in manifest["plots"]:
        table = plot["source_table"]
        selected = selected_by_plot[plot["plot_id"]]
        target = layout.plot_data / f"{plot['plot_id']}.csv"
        record = write_tidy_csv(target, table, selected)
        record["plot_id"] = plot["plot_id"]
        record["relative_path"] = target.relative_to(layout.root).as_posix()
        outputs.append(record)
    return outputs


def validate_plot_data_sources(
    plot_manifest: Mapping[str, Any],
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, list[Mapping[str, Any]]]]:
    """Validate plot hashes/counts without writing, including in validate-only runs."""

    manifest = validate_plot_manifest(plot_manifest)
    selected_by_plot: dict[str, list[Mapping[str, Any]]] = {}
    for plot in manifest["plots"]:
        table = plot["source_table"]
        source_rows = validate_records(table, rows_by_table.get(table, []))
        selected = sorted(
            (row for row in source_rows if _match_filters(row, plot["filters"])),
            key=lambda row: record_sort_key(table, row),
        )
        observed_hash = table_sha256(table, selected)
        if observed_hash != plot["data_sha256"] or len(selected) != plot["n_source_rows"]:
            raise SchemaError(
                f"plot {plot['plot_id']!r} source rows do not match its registered hash/count"
            )
        selected_by_plot[plot["plot_id"]] = selected
    return manifest, selected_by_plot


def copy_validated_parquet(
    source: os.PathLike[str] | str,
    target: os.PathLike[str] | str,
    table: str,
) -> dict[str, Any]:
    """Copy a producer-owned Parquet file only after full schema validation."""

    rows = read_parquet(source, table)
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "table": table,
        "schema": SCHEMA_REVISION,
        "path": destination.name,
        "row_count": len(rows),
        "logical_sha256": table_sha256(table, rows),
        "file_sha256": sha256_file(destination),
    }


def build_reporting_manifest(
    layout: ReleaseLayout,
    *,
    release_id: str,
    registry: Mapping[str, Any],
    artifact_records: Iterable[Mapping[str, Any]],
    optional_dependencies: Mapping[str, str],
) -> dict[str, Any]:
    validated_registry = validate_registry(registry)
    if validated_registry["release_id"] != release_id:
        raise SchemaError("reporting manifest release_id disagrees with registry")
    records = sorted(
        [dict(record) for record in artifact_records],
        key=lambda record: (str(record.get("relative_path", "")), str(record.get("path", ""))),
    )
    value = {
        "schema": "reconstruction_reporting_manifest_v1",
        "release_id": release_id,
        "registry_sha256": validated_registry["registry_sha256"],
        "artifacts": records,
        "optional_dependencies": dict(sorted(optional_dependencies.items())),
    }
    value["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value
