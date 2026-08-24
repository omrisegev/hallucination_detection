"""Fail-closed contracts for the ProcessBench causal-prefix reconstruction lane."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import ctypes
import errno
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

from .io import canonical_json_bytes, sha256_bytes, sha256_file


REGISTRY_SCHEMA = "reconstruction-prefix-registry-v1"
PREPARATION_SCHEMA = "reconstruction-prefix-preparation-v1"
PRIVATE_LABEL_SCHEMA = "reconstruction-prefix-private-labels-v1"
FIT_INPUT_SCHEMA = "reconstruction-prefix-fit-input-v1"
PREPARATION_AB_SCHEMA = "reconstruction-prefix-preparation-ab-v1"
SCORE_FREEZE_SCHEMA = "reconstruction-prefix-score-freeze-v1"
SCORE_AB_SCHEMA = "reconstruction-prefix-score-ab-v1"
EVALUATION_SCHEMA = "reconstruction-prefix-evaluation-v1"
EVALUATION_AB_SCHEMA = "reconstruction-prefix-evaluation-ab-v1"

METHOD_IDS = (
    "unified28",
    "iu28_no_length",
    "step272_two_head_global_local_w0p50_peak",
)
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
BUDGETS = (16, 32, 64, 128, 256, 512)
FIT_ALLOWED_FIELDS = frozenset(
    {
        "row_id",
        "source_question_id",
        "family",
        "partition",
        "token_entropies",
        "token_spilled_energies",
        "token_logsumexp",
        "top_k_logprobs",
    }
)
FIT_FORBIDDEN_FIELDS = frozenset(
    {
        "label",
        "correct",
        "final_answer_correct",
        "target",
        "first_error",
        "steps",
        "response",
        "problem",
        "answer",
    }
)


class PrefixContractError(RuntimeError):
    """The prefix registry or one frozen artifact violated its contract."""


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing a raced target."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        operation = getattr(libc, "renamex_np", None)
        if operation is None:
            raise RuntimeError("atomic no-replace prefix publication is unavailable")
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise RuntimeError("atomic no-replace prefix publication is unavailable")
        operation.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(-100, source_bytes, -100, target_bytes, 1)  # RENAME_NOREPLACE
    else:
        raise RuntimeError(
            f"atomic no-replace prefix publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"prefix output directory already exists: {target}")
    raise OSError(error_number, os.strerror(error_number), os.fspath(target))


def write_json_noreplace(path: str | Path, value: Any) -> str:
    """Atomically publish canonical JSON while preserving any raced target."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = target.parent.resolve(strict=True)
    target = parent / target.name
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"prefix certificate already exists: {target}") from error
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return sha256_bytes(payload)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class AtomicPrefixDirectory:
    """Build one prefix artifact tree privately, then publish it by rename.

    The staging directory is created beside the final directory so publication
    stays on one filesystem.  Callers decide when the fully written tree is
    ready to commit; cleanup never removes a committed tree.
    """

    def __init__(self, final_path: str | Path) -> None:
        self.final_path = Path(final_path)
        if self.final_path.exists() or self.final_path.is_symlink():
            raise FileExistsError(
                f"prefix output directory already exists: {self.final_path}"
            )
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(
            tempfile.mkdtemp(
                prefix=f".{self.final_path.name}.prefix-staging-",
                dir=self.final_path.parent,
            )
        )
        self.committed = False

    def commit(self) -> None:
        if self.committed:
            raise RuntimeError("prefix artifact stage was already committed")
        if self.final_path.exists() or self.final_path.is_symlink():
            raise FileExistsError(
                f"prefix output directory already exists: {self.final_path}"
            )
        _rename_directory_noreplace(self.path, self.final_path)
        self.committed = True

    def rollback(self) -> None:
        """Return a just-published tree to staging for paired-tree rollback."""

        if not self.committed:
            return
        if self.path.exists() or self.path.is_symlink():
            raise RuntimeError("prefix rollback staging path unexpectedly exists")
        _rename_directory_noreplace(self.final_path, self.path)
        self.committed = False

    def cleanup(self) -> None:
        if not self.committed and self.path.exists():
            shutil.rmtree(self.path)


def payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def add_payload_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output["payload_sha256"] = payload_sha256(output)
    return output


def verify_payload(value: Mapping[str, Any], *, name: str) -> None:
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != payload_sha256(payload):
        raise PrefixContractError(f"{name} payload SHA-256 failed")


def _require_exact_keys(value: Mapping[str, Any], expected: Sequence[str], *, name: str) -> None:
    if set(value) != set(expected):
        raise PrefixContractError(
            f"{name} keys drifted: expected={sorted(expected)}, observed={sorted(value)}"
        )


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        value = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrefixContractError(f"invalid prefix registry: {registry_path}") from error
    if value.get("schema_version") != REGISTRY_SCHEMA:
        raise PrefixContractError("unexpected prefix registry schema")
    population = value.get("population")
    if not isinstance(population, Mapping):
        raise PrefixContractError("prefix registry has no population contract")
    if tuple(population.get("subsets", ())) != SUBSETS:
        raise PrefixContractError("prefix subset roster drifted")
    if tuple(population.get("budgets", ())) != BUDGETS:
        raise PrefixContractError("prefix budget roster drifted")
    methods = value.get("method_roster")
    if not isinstance(methods, list) or tuple(row.get("method_id") for row in methods) != METHOD_IDS:
        raise PrefixContractError("prefix method roster/order drifted")
    if any("SIGNED_HISTORICAL_SCORE_REBIND" in str(row.get("execution_mode")) for row in methods):
        raise PrefixContractError("score rebinding cannot be registered as a prefix rerun")
    anchor = value.get("score_anchor")
    if (
        not isinstance(anchor, Mapping)
        or set(anchor) != {
            "absolute_tolerance", "required_status",
            "signed_historical_score_rebind_status",
        }
        or float(anchor.get("absolute_tolerance", 0.0)) != 2e-12
        or anchor.get("required_status") != "CPU_RECOMPUTED_AND_ANCHOR_VERIFIED"
        or anchor.get("signed_historical_score_rebind_status") != "FORBIDDEN_AS_RERUN"
    ):
        raise PrefixContractError("prefix recomputation/anchor contract drifted")
    visibility = value.get("fit_visibility", {})
    if set(visibility.get("allowed_fields", ())) != FIT_ALLOWED_FIELDS:
        raise PrefixContractError("fit-visible prefix field roster drifted")
    if set(visibility.get("forbidden_fields", ())) != FIT_FORBIDDEN_FIELDS:
        raise PrefixContractError("fit-forbidden prefix field roster drifted")
    evaluation = value.get("evaluation", {})
    bootstrap = evaluation.get("bootstrap", {})
    if (
        evaluation.get("aggregation") != "equal-subset macro separately at each budget"
        or evaluation.get("missing_subset_policy")
        != "MACRO_UNDEFINED_IF_ANY_REGISTERED_SUBSET_METRIC_UNDEFINED"
        or evaluation.get("cross_budget_macro") != "FORBIDDEN"
        or int(bootstrap.get("draws", 0)) != 2000
        or bootstrap.get("unit") != "source question"
        or bootstrap.get("stratification") != "within subset"
        or bootstrap.get("paired_across_methods_and_budgets") is not True
    ):
        raise PrefixContractError("prefix evaluation/bootstrap contract drifted")
    contrasts = evaluation.get("contrasts")
    expected_pairs = {
        frozenset((METHOD_IDS[0], METHOD_IDS[1])),
        frozenset((METHOD_IDS[0], METHOD_IDS[2])),
        frozenset((METHOD_IDS[1], METHOD_IDS[2])),
    }
    if not isinstance(contrasts, list) or {
        frozenset(map(str, pair)) for pair in contrasts
    } != expected_pairs:
        raise PrefixContractError("prefix contrast roster is not the complete three-method set")
    return value


def resolve_source_asset(source_root: str | Path, item: Mapping[str, Any], *, name: str) -> Path:
    root = Path(source_root).resolve()
    relative = str(item.get("path", ""))
    if not relative or Path(relative).is_absolute():
        raise PrefixContractError(f"{name} source path must be non-empty and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PrefixContractError(f"{name} source path escapes --source-root") from error
    if not path.is_file():
        raise PrefixContractError(f"{name} source is missing: {path}")
    observed = sha256_file(path)
    if observed != item.get("sha256"):
        raise PrefixContractError(
            f"{name} source SHA-256 drifted: expected={item.get('sha256')}, observed={observed}"
        )
    return path


def validate_sanitized_row(row: Mapping[str, Any]) -> None:
    unknown = set(row).difference(FIT_ALLOWED_FIELDS)
    if unknown:
        raise PrefixContractError(f"fit-visible prefix row has non-contract fields: {sorted(unknown)}")
    forbidden = set(row).intersection(FIT_FORBIDDEN_FIELDS)
    if forbidden:
        raise PrefixContractError(f"fit-visible prefix row leaked targets: {sorted(forbidden)}")
    required = {
        "row_id",
        "source_question_id",
        "family",
        "partition",
        "token_entropies",
        "token_logsumexp",
        "top_k_logprobs",
    }
    missing = required.difference(row)
    if missing:
        raise PrefixContractError(f"fit-visible prefix row is incomplete: {sorted(missing)}")
    if row["family"] not in SUBSETS or row["partition"] not in {"calibration", "evaluation"}:
        raise PrefixContractError("fit-visible prefix family/partition is invalid")
    entropy = np.asarray(row["token_entropies"], dtype=float).reshape(-1)
    if len(entropy) <= min(BUDGETS) or not np.isfinite(entropy).all():
        raise PrefixContractError("fit-visible entropy trace is too short or non-finite")
    n = len(entropy)
    for name in ("token_spilled_energies", "token_logsumexp"):
        if name not in row:
            continue
        values = np.asarray(row[name], dtype=float).reshape(-1)
        if len(values) != n or not np.isfinite(values).all():
            raise PrefixContractError(f"fit-visible {name} is unaligned or non-finite")
    topk = row["top_k_logprobs"]
    if not isinstance(topk, Mapping) or set(topk) != {"ids", "logprobs"}:
        raise PrefixContractError("fit-visible top-k telemetry is missing")
    ids = np.asarray(topk["ids"])
    logprobs = np.asarray(topk["logprobs"], dtype=float)
    if (
        ids.ndim != 2
        or logprobs.ndim != 2
        or ids.shape != logprobs.shape
        or ids.shape[0] != n
        or ids.shape[1] < 1
        or not np.issubdtype(ids.dtype, np.number)
        or not np.isfinite(ids.astype(float)).all()
        or not np.isfinite(logprobs).all()
    ):
        raise PrefixContractError("fit-visible top-k telemetry is invalid")


def validate_observation_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    registry: Mapping[str, Any],
    include_scores: bool,
) -> None:
    required = {"row_id", "family", "budget"}
    if include_scores:
        required.update(METHOD_IDS)
    if set(arrays) != required:
        raise PrefixContractError(
            f"prefix observation members drifted: expected={sorted(required)}, observed={sorted(arrays)}"
        )
    sizes = {len(np.asarray(value)) for value in arrays.values()}
    expected_n = int(registry["population"]["expected_prefix_observations"])
    if sizes != {expected_n}:
        raise PrefixContractError(f"prefix observation count drifted: {sizes} != {{{expected_n}}}")
    row_ids = np.asarray(arrays["row_id"]).astype(str)
    families = np.asarray(arrays["family"]).astype(str)
    budgets = np.asarray(arrays["budget"], dtype=int)
    if any(not value for value in row_ids) or not set(families) <= set(SUBSETS):
        raise PrefixContractError("prefix observation identity/family is invalid")
    if not set(budgets) <= set(BUDGETS):
        raise PrefixContractError("prefix observation budget is outside the frozen grid")
    keys = list(zip(row_ids.tolist(), budgets.tolist(), strict=True))
    if len(set(keys)) != expected_n:
        raise PrefixContractError("prefix observation row-budget keys are not unique")
    family_by_row: dict[str, str] = {}
    budgets_by_row: dict[str, set[int]] = {}
    for row_id, family, budget in zip(row_ids, families, budgets, strict=True):
        if row_id in family_by_row and family_by_row[row_id] != family:
            raise PrefixContractError(f"prefix trace changes subset across budgets: {row_id}")
        family_by_row[row_id] = family
        budgets_by_row.setdefault(row_id, set()).add(int(budget))
    population = registry["population"]
    expected_traces = int(population["expected_evaluation_traces"])
    if len(family_by_row) != expected_traces:
        raise PrefixContractError(
            f"prefix trace-union count drifted: {len(family_by_row)} != {expected_traces}"
        )
    expected_family = Counter(
        {
            str(key): int(value)
            for key, value in population["expected_evaluation_traces_by_subset"].items()
        }
    )
    if Counter(family_by_row.values()) != expected_family:
        raise PrefixContractError("prefix trace-union subset counts drifted")
    complete = sum(set(BUDGETS) <= observed for observed in budgets_by_row.values())
    if complete != int(population["expected_complete_all_budgets"]):
        raise PrefixContractError("prefix complete-six-budget trace count drifted")
    expected_budget = {
        int(key): int(value)
        for key, value in population["expected_prefix_observations_by_budget"].items()
    }
    if Counter(map(int, budgets)) != Counter(expected_budget):
        raise PrefixContractError("prefix observation budget counts drifted")
    if include_scores:
        for method_id in METHOD_IDS:
            values = np.asarray(arrays[method_id], dtype=float)
            if values.shape != (expected_n,) or not np.isfinite(values).all():
                raise PrefixContractError(f"{method_id} prefix scores are invalid")


def finite_metric(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


__all__ = [
    "AtomicPrefixDirectory",
    "BUDGETS",
    "EVALUATION_AB_SCHEMA",
    "EVALUATION_SCHEMA",
    "FIT_ALLOWED_FIELDS",
    "FIT_FORBIDDEN_FIELDS",
    "FIT_INPUT_SCHEMA",
    "METHOD_IDS",
    "PREPARATION_AB_SCHEMA",
    "PREPARATION_SCHEMA",
    "PRIVATE_LABEL_SCHEMA",
    "PrefixContractError",
    "REGISTRY_SCHEMA",
    "SCORE_AB_SCHEMA",
    "SCORE_FREEZE_SCHEMA",
    "SUBSETS",
    "add_payload_sha256",
    "finite_metric",
    "load_registry",
    "payload_sha256",
    "resolve_source_asset",
    "validate_observation_arrays",
    "validate_sanitized_row",
    "verify_payload",
    "write_json_noreplace",
]
