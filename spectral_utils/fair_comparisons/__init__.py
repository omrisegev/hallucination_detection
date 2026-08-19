"""Frozen interfaces for Fair Paper-Exact Comparison Package v1.

The package deliberately keeps Global detection, ProcessBench first-error
Localization, causal Prefix detection, and Stopping/adaptive compute in separate
modules.  Importing this package never starts inference, downloads artifacts, or
touches Google Drive.
"""

from .evaluator import EVALUATOR_REVISION
from .folds import FOLD_REVISION
from .processbench import PROCESSBENCH_ADAPTER_REVISION
from .registry import (
    ASSET_REGISTRY_SCHEMA,
    COMPARISON_SCHEMA,
    METHOD_SCHEMA,
    POPULATION_SCHEMA,
)

PACKAGE_REVISION = "fair_paper_exact_comparisons_v1.0.0"

__all__ = [
    "ASSET_REGISTRY_SCHEMA",
    "COMPARISON_SCHEMA",
    "EVALUATOR_REVISION",
    "FOLD_REVISION",
    "METHOD_SCHEMA",
    "PACKAGE_REVISION",
    "POPULATION_SCHEMA",
    "PROCESSBENCH_ADAPTER_REVISION",
]
