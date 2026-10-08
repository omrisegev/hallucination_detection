"""Reporting interfaces for the reconstruction benchmark release."""

from .io import (
    ReleaseLayout,
    load_records,
    materialize_plot_data,
    validate_plot_data_sources,
    read_tidy_csv,
    write_canonical_json,
    write_parquet,
    write_tidy_csv,
)
from .query import (
    LEADERBOARD_EXPORT_SPECS,
    VIEW_NAMES,
    build_duckdb,
    export_leaderboard_csvs,
    query_results,
)
from .registry import (
    REGISTRY_SCHEMA,
    build_registry,
    expected_coverage_rows,
    make_system_id,
    validate_registry,
    validate_result_references,
)
from .report import default_plot_manifest, render_report, write_report
from .schemas import (
    SCHEMA_REVISION,
    STATUS_VALUES,
    MissingOptionalDependency,
    SchemaError,
    derive_aggregate_cohort_id,
    derive_cohort_id,
    derive_comparison_group_id,
    rank_metric_group,
    validate_comparison_groups,
    validate_expected_coverage,
    validate_plot_manifest,
    validate_records,
)

PACKAGE_REVISION = "reconstruction_reporting_v1.2.0"

__all__ = [
    "PACKAGE_REVISION",
    "REGISTRY_SCHEMA",
    "SCHEMA_REVISION",
    "STATUS_VALUES",
    "MissingOptionalDependency",
    "LEADERBOARD_EXPORT_SPECS",
    "ReleaseLayout",
    "SchemaError",
    "VIEW_NAMES",
    "build_duckdb",
    "build_registry",
    "default_plot_manifest",
    "derive_aggregate_cohort_id",
    "derive_cohort_id",
    "derive_comparison_group_id",
    "expected_coverage_rows",
    "export_leaderboard_csvs",
    "load_records",
    "make_system_id",
    "materialize_plot_data",
    "validate_plot_data_sources",
    "query_results",
    "rank_metric_group",
    "read_tidy_csv",
    "render_report",
    "validate_comparison_groups",
    "validate_expected_coverage",
    "validate_plot_manifest",
    "validate_records",
    "validate_registry",
    "validate_result_references",
    "write_canonical_json",
    "write_parquet",
    "write_report",
    "write_tidy_csv",
]
