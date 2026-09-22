"""Prepare independent target-isolated builds for causal prefix recomputation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from spectral_utils.fair_comparisons.prefix import (
    _reconstruct_frozen_prefix_incumbent,
)
from spectral_utils.fair_comparisons.processbench import canonical_processbench_id
from spectral_utils.fair_comparisons.twentyfour import (
    load_unified28_model,
    unified28_parameter_sha256,
)
from spectral_utils.multitask_trajectory import stable_partition

from .io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    canonical_tree_manifest,
    deterministic_npz_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .prefix_contract import (
    AtomicPrefixDirectory,
    BUDGETS,
    FIT_INPUT_SCHEMA,
    METHOD_IDS,
    PREPARATION_SCHEMA,
    PRIVATE_LABEL_SCHEMA,
    SUBSETS,
    PrefixContractError,
    add_payload_sha256,
    load_registry,
    payload_sha256,
    resolve_source_asset,
    validate_observation_arrays,
    validate_sanitized_row,
    verify_payload,
)


FIT_INPUT_FILENAME = "FIT_INPUT.pkl"
EXPECTED_SCORE_FILENAME = "EXPECTED_SCORES.npz"
PRIVATE_LABEL_FILENAME = "LABELS.json"
PREPARATION_MANIFEST_FILENAME = "PREPARATION_MANIFEST.json"


def _registered_source_asset_specs(
    registry: Mapping[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    """Return the complete ordered source roster encoded by the registry."""

    contract = registry.get("source_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "raw_root",
        "raw_manifest",
        "raw_files",
        "unified28",
        "step272",
        "iu28",
        "signed_prefix_package",
    }:
        raise PrefixContractError("prefix registered source-asset roster drifted")
    if set(contract.get("raw_files", {})) != set(SUBSETS):
        raise PrefixContractError("prefix raw source-asset roster drifted")
    if set(contract.get("unified28", {})) != {
        "model", "run_definition", "score_anchor"
    }:
        raise PrefixContractError("prefix Unified-28 source-asset roster drifted")
    if set(contract.get("step272", {})) != {
        "head_selection", "architecture_selection", "score_anchor", "cache_inventory"
    }:
        raise PrefixContractError("prefix Step272 source-asset roster drifted")
    iu = contract.get("iu28")
    if (
        not isinstance(iu, Mapping)
        or set(iu) != {"historical_results_root", "fit_source_files", "anchor_files"}
        or set(iu.get("fit_source_files", {})) != set(SUBSETS)
        or set(iu.get("anchor_files", {})) != set(SUBSETS)
    ):
        raise PrefixContractError("prefix IU28 source-asset roster drifted")
    signed = contract.get("signed_prefix_package")
    if not isinstance(signed, Mapping) or set(signed) != {"manifest", "score_ledger", "use"}:
        raise PrefixContractError("prefix signed-anchor source-asset roster drifted")

    specs: list[tuple[str, str, str]] = []

    def append(asset_id: str, item: Any) -> None:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise PrefixContractError(f"prefix source asset is malformed: {asset_id}")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64:
            raise PrefixContractError(f"prefix source asset path/hash is malformed: {asset_id}")
        specs.append((asset_id, path, digest))

    append("raw_manifest", contract["raw_manifest"])
    for family in SUBSETS:
        append(f"raw::{family}", contract["raw_files"][family])
    for component in ("model", "run_definition", "score_anchor"):
        append(f"unified28::{component}", contract["unified28"][component])
    for component in (
        "head_selection", "architecture_selection", "score_anchor", "cache_inventory"
    ):
        append(f"step272::{component}", contract["step272"][component])
    historical_root = str(iu["historical_results_root"])
    if not historical_root or Path(historical_root).is_absolute():
        raise PrefixContractError("prefix IU28 historical result root is malformed")
    for family in SUBSETS:
        append(f"iu28::fit_source::{family}", iu["fit_source_files"][family])
        anchors = iu["anchor_files"][family]
        if not isinstance(anchors, Mapping) or tuple(anchors) != (
            "result.json", "scores_calibration.csv", "scores_evaluation.csv"
        ):
            raise PrefixContractError(f"prefix IU28 {family} anchor roster drifted")
        for filename, digest in anchors.items():
            append(
                f"iu28::anchor::{family}::{filename}",
                {
                    "path": (
                        f"{historical_root}/processbench_{family}__llama31_8b/{filename}"
                    ),
                    "sha256": digest,
                },
            )
    append("signed_prefix_package::manifest", signed["manifest"])
    append("signed_prefix_package::score_ledger", signed["score_ledger"])
    return tuple(specs)


def _bind_registered_source_assets(
    *,
    source_root: Path,
    repo: Path,
    registry_path: Path,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    assets = []
    for asset_id, relative, expected_sha in _registered_source_asset_specs(registry):
        path = resolve_source_asset(
            source_root,
            {"path": relative, "sha256": expected_sha},
            name=asset_id,
        )
        assets.append(
            {
                "asset_id": asset_id,
                "path": relative,
                "sha256": expected_sha,
                "size_bytes": path.stat().st_size,
            }
        )
    registry_resolved = registry_path.resolve()
    try:
        registry_relative = registry_resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        registry_relative = str(registry_resolved)
    return {
        "source_root": str(source_root.resolve()),
        "registry": {
            "path": registry_relative,
            "sha256": sha256_file(registry_resolved),
        },
        "assets": assets,
        "asset_roster_sha256": payload_sha256(assets),
    }


def _truth(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise PrefixContractError(f"{name} must be binary")


def _load_cache(path: Path) -> Mapping[Any, Any] | Sequence[Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (str, bytes)):
        raise PrefixContractError(f"unsupported prefix cache object: {path}")
    return value


def _ordered_cache_rows(value: Mapping[Any, Any] | Sequence[Any]) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        candidates = [value[key] for key in sorted(value, key=str)]
    else:
        candidates = list(value)
    rows = []
    for row in candidates:
        if not isinstance(row, Mapping):
            raise PrefixContractError("prefix cache contains a non-mapping row")
        if row.get("align_diag", {}).get("problems"):
            continue
        rows.append(row)
    return rows


def sanitize_source_row(row: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    """Copy only causal telemetry; all outcome/text/localization fields are dropped."""

    source_question_id = str(row.get("id", ""))
    if not source_question_id:
        raise PrefixContractError(f"{family} prefix row has no official ID")
    output: dict[str, Any] = {
        "row_id": canonical_processbench_id(family, source_question_id),
        "source_question_id": source_question_id,
        "family": family,
        "partition": stable_partition(source_question_id),
        "token_entropies": np.asarray(row.get("token_entropies"), dtype=float),
        "token_logsumexp": np.asarray(row.get("token_logsumexp"), dtype=float),
    }
    if row.get("token_spilled_energies") is not None:
        output["token_spilled_energies"] = np.asarray(
            row["token_spilled_energies"], dtype=float
        )
    topk = row.get("top_k_logprobs")
    if not isinstance(topk, Mapping):
        raise PrefixContractError(f"{family}/{source_question_id} lacks top-k telemetry")
    output["top_k_logprobs"] = {
        "ids": np.asarray(topk.get("ids")),
        "logprobs": np.asarray(topk.get("logprobs"), dtype=float),
    }
    validate_sanitized_row(output)
    return output


def _verify_signed_ledger(
    source_root: Path, registry: Mapping[str, Any]
) -> Path:
    package = registry["source_contract"]["signed_prefix_package"]
    manifest_path = resolve_source_asset(
        source_root, package["manifest"], name="signed prefix package manifest"
    )
    ledger_path = resolve_source_asset(
        source_root, package["score_ledger"], name="signed prefix score ledger"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise PrefixContractError("signed prefix package has no file manifest")
    relative = package["score_ledger"]["path"].split(
        "results/fair_paper_exact_comparisons_v1/", 1
    )[-1]
    matches = [item for item in files if item.get("path") == relative]
    if len(matches) != 1 or matches[0].get("sha256") != package["score_ledger"]["sha256"]:
        raise PrefixContractError("signed package does not bind the prefix score ledger")
    return ledger_path


def _load_expected_scores(
    ledger_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "budget",
            "continuous_score",
            "direct_eligible",
            "family",
            "final_length",
            "group_id",
            "label",
            "method_id",
            "row_id",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise PrefixContractError(f"signed prefix ledger lacks {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            method_id = str(row["method_id"])
            if method_id not in METHOD_IDS or row["budget"] == "final":
                continue
            if str(row["direct_eligible"]).lower() != "true":
                raise PrefixContractError(
                    f"{ledger_path}:{line_number}: registered prefix row is not direct-eligible"
                )
            budget = int(row["budget"])
            if budget not in BUDGETS:
                raise PrefixContractError(f"unexpected prefix budget {budget}")
            row_id = str(row["row_id"])
            key = (row_id, budget)
            if method_id in by_key[key]:
                raise PrefixContractError(f"duplicate signed prefix score {key}/{method_id}")
            score = float(row["continuous_score"])
            if not np.isfinite(score):
                raise PrefixContractError(f"non-finite signed prefix score {key}/{method_id}")
            by_key[key][method_id] = score
            current = {
                "row_id": row_id,
                "family": str(row["family"]),
                "group_id": str(row["group_id"]),
                "label": int(row["label"]),
                "final_length": int(row["final_length"]),
            }
            if key in metadata and metadata[key] != current:
                raise PrefixContractError(f"signed prefix method metadata disagree for {key}")
            metadata[key] = current
    incomplete = [key for key, scores in by_key.items() if set(scores) != set(METHOD_IDS)]
    if incomplete:
        raise PrefixContractError(f"signed prefix score roster is incomplete: {incomplete[:3]}")
    ordered = sorted(by_key, key=lambda key: (metadata[key]["family"], key[0], key[1]))
    arrays: dict[str, np.ndarray] = {
        "row_id": np.asarray([key[0] for key in ordered]),
        "family": np.asarray([metadata[key]["family"] for key in ordered]),
        "budget": np.asarray([key[1] for key in ordered], dtype=np.int16),
    }
    for method_id in METHOD_IDS:
        arrays[method_id] = np.asarray([by_key[key][method_id] for key in ordered], dtype=np.float64)
    trace_metadata: dict[str, dict[str, Any]] = {}
    for key in ordered:
        item = metadata[key]
        row_id = item["row_id"]
        trace = {
            "row_id": row_id,
            "family": item["family"],
            "group_id": item["group_id"],
            "label": item["label"],
            "final_length": item["final_length"],
        }
        if row_id in trace_metadata and trace_metadata[row_id] != trace:
            raise PrefixContractError(f"signed prefix budgets disagree for {row_id}")
        trace_metadata[row_id] = trace
    return arrays, trace_metadata


def _validate_source_selections(source_root: Path, registry: Mapping[str, Any]) -> None:
    contract = registry["source_contract"]
    for component in ("model", "run_definition", "score_anchor"):
        resolve_source_asset(
            source_root, contract["unified28"][component], name=f"Unified-28 {component}"
        )
    for component in (
        "head_selection",
        "architecture_selection",
        "score_anchor",
        "cache_inventory",
    ):
        resolve_source_asset(
            source_root, contract["step272"][component], name=f"Step272 {component}"
        )
    head_path = source_root / contract["step272"]["head_selection"]["path"]
    architecture_path = source_root / contract["step272"]["architecture_selection"]["path"]
    head = json.loads(head_path.read_text(encoding="utf-8"))
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    expected_method = next(row for row in registry["method_roster"] if row["method_id"] == METHOD_IDS[2])
    if head.get("selected") != expected_method["frozen_heads"]:
        raise PrefixContractError("Step272 frozen head selection drifted")
    if architecture.get("selected", {}).get("architecture") != expected_method["frozen_architecture"]:
        raise PrefixContractError("Step272 frozen architecture selection drifted")


def _verify_iu_assets(source_root: Path, registry: Mapping[str, Any]) -> None:
    iu = registry["source_contract"]["iu28"]
    for family in SUBSETS:
        resolve_source_asset(
            source_root, iu["fit_source_files"][family], name=f"IU28 {family} fit source"
        )
        cell = source_root / iu["historical_results_root"] / f"processbench_{family}__llama31_8b"
        for filename, expected in iu["anchor_files"][family].items():
            path = cell / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise PrefixContractError(f"IU28 {family} anchor drifted: {filename}")
        metadata = json.loads((cell / "result.json").read_text(encoding="utf-8"))["metadata"]
        fit_source = Path(str(metadata.get("source_path", ""))).resolve()
        expected_fit_source = (
            source_root / iu["fit_source_files"][family]["path"]
        ).resolve()
        if fit_source != expected_fit_source:
            raise PrefixContractError(
                f"IU28 {family} historical fit source does not equal its registered asset: "
                f"{fit_source} != {expected_fit_source}"
            )


def _load_and_sanitize_sources(
    source_root: Path,
    registry: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    raw_contract = registry["source_contract"]["raw_files"]
    rows_by_family: dict[str, list[dict[str, Any]]] = {}
    source_by_id: dict[str, Mapping[str, Any]] = {}
    bindings = []
    for family in SUBSETS:
        path = resolve_source_asset(source_root, raw_contract[family], name=f"raw {family}")
        source = _load_cache(path)
        raw_rows = _ordered_cache_rows(source)
        sanitized = [sanitize_source_row(row, family=family) for row in raw_rows]
        if len({row["row_id"] for row in sanitized}) != len(sanitized):
            raise PrefixContractError(f"raw {family} official IDs are not unique")
        rows_by_family[family] = sanitized
        for raw, clean in zip(raw_rows, sanitized, strict=True):
            source_by_id[clean["row_id"]] = raw
        bindings.append(
            {
                "family": family,
                "path": raw_contract[family]["path"],
                "sha256": raw_contract[family]["sha256"],
                "size_bytes": path.stat().st_size,
                "rows": len(sanitized),
            }
        )
    expected = int(registry["population"]["expected_source_rows"])
    if len(source_by_id) != expected:
        raise PrefixContractError(f"raw ProcessBench count drifted: {len(source_by_id)} != {expected}")
    return rows_by_family, source_by_id, bindings


def _private_labels(
    *,
    trace_metadata: Mapping[str, Mapping[str, Any]],
    source_by_id: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    labels = []
    for row_id in sorted(trace_metadata, key=lambda value: (trace_metadata[value]["family"], value)):
        expected = trace_metadata[row_id]
        raw = source_by_id.get(row_id)
        if raw is None:
            raise PrefixContractError(f"signed prefix row is absent from raw telemetry: {row_id}")
        family = str(expected["family"])
        source_question_id = str(raw.get("id", ""))
        if stable_partition(source_question_id) != "evaluation":
            raise PrefixContractError(f"signed prefix row is not in the frozen evaluation split: {row_id}")
        label = 1 - int(_truth(raw.get("final_answer_correct"), name="final_answer_correct"))
        final_length = len(np.asarray(raw.get("token_entropies")))
        if (
            label != int(expected["label"])
            or final_length != int(expected["final_length"])
            or expected["group_id"] != row_id
            or canonical_processbench_id(family, source_question_id) != row_id
        ):
            raise PrefixContractError(f"raw/signed prefix identity or outcome mismatch: {row_id}")
        labels.append(
            {
                "row_id": row_id,
                "group_id": row_id,
                "family": family,
                "label": label,
                "final_length": final_length,
            }
        )
    population = registry["population"]
    if (
        len(labels) != int(population["expected_evaluation_traces"])
        or Counter(row["label"] for row in labels)
        != Counter({0: int(population["expected_correct"]), 1: int(population["expected_incorrect"])})
        or Counter(row["family"] for row in labels)
        != Counter(population["expected_evaluation_traces_by_subset"])
    ):
        raise PrefixContractError("prefix evaluation population count/balance drifted")
    return add_payload_sha256(
        {
            "schema_version": PRIVATE_LABEL_SCHEMA,
            "population_id": population["population_id"],
            "positive_class": registry["evaluation"]["positive_class"],
            "grouping_unit": "source question",
            "rows": labels,
        }
    )


def _fit_models(source_root: Path, registry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    unified_path = source_root / registry["source_contract"]["unified28"]["model"]["path"]
    unified, unified_artifact_hash = load_unified28_model(source_root, path=unified_path)
    historical_root = source_root / registry["source_contract"]["iu28"]["historical_results_root"]
    iu_models: dict[str, Any] = {}
    iu_audits: dict[str, Any] = {}
    for family in SUBSETS:
        model, audit, _ = _reconstruct_frozen_prefix_incumbent(family, historical_root)
        if audit.get("labels_used_for_score_parameter_fit") is not False:
            raise PrefixContractError(f"IU28 {family} fit is not target-free")
        iu_models[family] = model
        iu_audits[family] = audit
    models = {"unified28": unified, "iu28_no_length": iu_models}
    audit = {
        "unified28": {
            "artifact_sha256": unified_artifact_hash,
            "parameter_sha256": unified28_parameter_sha256(unified),
        },
        "iu28_no_length": iu_audits,
    }
    return models, audit


def reconstruct_prefix_preparation(
    *,
    repo: str | Path,
    registry_path: str | Path,
    source_root: str | Path,
) -> dict[str, Any]:
    """Rebuild every prepared payload from the exact registered source assets.

    This pure reconstruction is shared by the producer and the independent A/B
    verifier.  The verifier compares the resulting bytes, not self-attested
    hashes or manifests supplied by either build.
    """

    repo_path = Path(repo).resolve()
    registry_file = Path(registry_path).resolve()
    source = Path(source_root).resolve()
    canonical_registry = (
        repo_path / "configs/reconstruction_benchmark_v1/prefix.json"
    ).resolve()
    if registry_file != canonical_registry:
        raise PrefixContractError(
            "prefix source reconstruction requires the canonical frozen registry"
        )
    registry = load_registry(registry_file)
    source_binding = _bind_registered_source_assets(
        source_root=source,
        repo=repo_path,
        registry_path=registry_file,
        registry=registry,
    )
    _validate_source_selections(source, registry)
    _verify_iu_assets(source, registry)
    ledger_path = _verify_signed_ledger(source, registry)
    expected_scores, trace_metadata = _load_expected_scores(ledger_path)
    validate_observation_arrays(expected_scores, registry=registry, include_scores=True)
    rows_by_family, source_by_id, raw_bindings = _load_and_sanitize_sources(source, registry)
    expected_raw_assets = [
        row for row in source_binding["assets"] if str(row["asset_id"]).startswith("raw::")
    ]
    if [
        {
            "family": str(row["asset_id"]).split("::", 1)[1],
            "path": row["path"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "rows": next(
                item["rows"]
                for item in raw_bindings
                if item["family"] == str(row["asset_id"]).split("::", 1)[1]
            ),
        }
        for row in expected_raw_assets
    ] != raw_bindings:
        raise PrefixContractError("prefix raw source binding differs from full asset roster")
    labels = _private_labels(
        trace_metadata=trace_metadata,
        source_by_id=source_by_id,
        registry=registry,
    )
    models, model_audit = _fit_models(source, registry)
    fit_input = {
        "schema_version": FIT_INPUT_SCHEMA,
        "lane_id": registry["lane_id"],
        "task_id": registry["task_id"],
        "population_id": registry["population"]["population_id"],
        "budgets": BUDGETS,
        "method_ids": METHOD_IDS,
        "rows_by_family": rows_by_family,
        "frozen_models": models,
        "model_audit": model_audit,
        "target_fields_present": False,
        "claim_boundary": registry["claim_boundary"],
    }
    fit_payload = pickle.dumps(fit_input, protocol=5)
    expected_payload = deterministic_npz_bytes(expected_scores)
    label_payload = canonical_json_bytes(labels) + b"\n"
    return {
        "registry": registry,
        "source_binding": source_binding,
        "fit_input": fit_input,
        "fit_input_bytes": fit_payload,
        "fit_input_sha256": sha256_bytes(fit_payload),
        "expected_scores": expected_scores,
        "expected_scores_bytes": expected_payload,
        "expected_scores_sha256": sha256_bytes(expected_payload),
        "private_labels": labels,
        "private_labels_bytes": label_payload,
        "private_labels_sha256": sha256_bytes(label_payload),
        "model_audit_sha256": payload_sha256(model_audit),
    }


def prepare_prefix_build(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    build_id: str,
    source_root: str | Path,
    scientific_full: bool,
) -> dict[str, Any]:
    if build_id not in {"A", "B"}:
        raise PrefixContractError("prefix build must be A or B")
    reconstruction = reconstruct_prefix_preparation(
        repo=repo,
        registry_path=registry_path,
        source_root=source_root,
    )
    registry = reconstruction["registry"]
    public = Path(release_root) / release_id / "prefix" / build_id
    private = Path(private_root) / release_id / "prefix" / build_id
    if public.exists() or private.exists():
        raise FileExistsError(f"prefix build already exists: {public} or {private}")

    expected_scores = reconstruction["expected_scores"]
    labels = reconstruction["private_labels"]
    fit_payload = reconstruction["fit_input_bytes"]

    public_stage = AtomicPrefixDirectory(public)
    try:
        private_stage = AtomicPrefixDirectory(private)
    except BaseException:
        public_stage.cleanup()
        raise
    try:
        inputs = public_stage.path / "inputs"
        inputs.mkdir()
        fit_sha = atomic_write_bytes(inputs / FIT_INPUT_FILENAME, fit_payload)
        expected_sha = atomic_write_npz(inputs / EXPECTED_SCORE_FILENAME, expected_scores)
        labels_sha = atomic_write_json(
            private_stage.path / PRIVATE_LABEL_FILENAME, labels
        )
        source_binding = reconstruction["source_binding"]
        manifest = add_payload_sha256(
            {
                "schema_version": PREPARATION_SCHEMA,
                "release_id": release_id,
                "build_id": build_id,
                "scientific_full_build": bool(scientific_full),
                "lane_id": registry["lane_id"],
                "task_id": registry["task_id"],
                "population_id": registry["population"]["population_id"],
                "source_binding": source_binding,
                "source_binding_sha256": payload_sha256(source_binding),
                "fit_input": {
                    "path": f"inputs/{FIT_INPUT_FILENAME}",
                    "sha256": fit_sha,
                    "size_bytes": len(fit_payload),
                    "target_fields_present": False,
                },
                "expected_scores": {
                    "path": f"inputs/{EXPECTED_SCORE_FILENAME}",
                    "sha256": expected_sha,
                    "observations": int(len(expected_scores["row_id"])),
                    "labels_present": False,
                    "use": "post-recomputation score anchor only",
                },
                "private_labels": {
                    "path": str(private / PRIVATE_LABEL_FILENAME),
                    "sha256": labels_sha,
                    "rows": len(labels["rows"]),
                    "fit_visible": False,
                },
                "fit_model_audit_sha256": reconstruction["model_audit_sha256"],
                "execution_modes": {
                    row["method_id"]: row["execution_mode"]
                    for row in registry["method_roster"]
                },
                "labels_opened_by_preparation_controller": True,
                "labels_exposed_to_fit": False,
                "historical_scores_are_execution_substitute": False,
                "claim_boundary": registry["claim_boundary"],
            }
        )
        atomic_write_json(
            public_stage.path / PREPARATION_MANIFEST_FILENAME, manifest
        )
        tree = canonical_tree_manifest(public_stage.path)
        atomic_write_json(public_stage.path / "TREE_MANIFEST.json", tree)
        private_stage.commit()
        try:
            public_stage.commit()
        except BaseException:
            private_stage.rollback()
            raise
        return manifest
    finally:
        public_stage.cleanup()
        private_stage.cleanup()


def load_preparation_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_payload(value, name="prefix preparation manifest")
    expected_fields = {
        "schema_version",
        "release_id",
        "build_id",
        "scientific_full_build",
        "lane_id",
        "task_id",
        "population_id",
        "source_binding",
        "source_binding_sha256",
        "fit_input",
        "expected_scores",
        "private_labels",
        "fit_model_audit_sha256",
        "execution_modes",
        "labels_opened_by_preparation_controller",
        "labels_exposed_to_fit",
        "historical_scores_are_execution_substitute",
        "claim_boundary",
        "payload_sha256",
    }
    if set(value) != expected_fields:
        raise PrefixContractError("prefix preparation manifest field roster drifted")
    if value.get("schema_version") != PREPARATION_SCHEMA:
        raise PrefixContractError("unexpected prefix preparation schema")
    if (
        value.get("build_id") not in {"A", "B"}
        or type(value.get("scientific_full_build")) is not bool
        or value.get("labels_opened_by_preparation_controller") is not True
        or value.get("labels_exposed_to_fit") is not False
        or value.get("historical_scores_are_execution_substitute") is not False
        or value.get("fit_input", {}).get("target_fields_present") is not False
        or value.get("expected_scores", {}).get("labels_present") is not False
        or value.get("private_labels", {}).get("fit_visible") is not False
        or value.get("source_binding_sha256")
        != payload_sha256(value.get("source_binding", {}))
    ):
        raise PrefixContractError("prefix preparation target/replay boundary failed")
    return value


def load_fit_input(
    path: str | Path, *, registry: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError, TypeError) as error:
        raise PrefixContractError("prefix fit input is not a valid pickle artifact") from error
    if not isinstance(value, dict) or value.get("schema_version") != FIT_INPUT_SCHEMA:
        raise PrefixContractError("unexpected prefix fit-input schema")
    expected_fields = {
        "schema_version",
        "lane_id",
        "task_id",
        "population_id",
        "budgets",
        "method_ids",
        "rows_by_family",
        "frozen_models",
        "model_audit",
        "target_fields_present",
        "claim_boundary",
    }
    if set(value) != expected_fields:
        raise PrefixContractError("prefix fit-input field roster drifted")
    if value.get("target_fields_present") is not False:
        raise PrefixContractError("prefix fit input does not attest target isolation")
    if tuple(value.get("method_ids", ())) != METHOD_IDS or tuple(value.get("budgets", ())) != BUDGETS:
        raise PrefixContractError("prefix fit input roster drifted")
    rows_by_family = value.get("rows_by_family")
    if not isinstance(rows_by_family, Mapping) or tuple(rows_by_family) != SUBSETS:
        raise PrefixContractError("prefix fit input family roster drifted")
    observed_ids: set[str] = set()
    for family, rows in rows_by_family.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise PrefixContractError(f"prefix fit input {family} rows are malformed")
        for row in rows:
            if not isinstance(row, Mapping) or row.get("family") != family:
                raise PrefixContractError("prefix fit input row/family binding drifted")
            validate_sanitized_row(row)
            row_id = str(row["row_id"])
            if row_id in observed_ids:
                raise PrefixContractError(f"duplicate prefix fit-input row ID: {row_id}")
            observed_ids.add(row_id)
    frozen = value.get("frozen_models")
    if not isinstance(frozen, Mapping) or set(frozen) != {"unified28", "iu28_no_length"}:
        raise PrefixContractError("prefix frozen-model roster drifted")
    if not isinstance(value.get("model_audit"), Mapping):
        raise PrefixContractError("prefix fit model audit is malformed")
    if registry is not None:
        if (
            value.get("lane_id") != registry["lane_id"]
            or value.get("task_id") != registry["task_id"]
            or value.get("population_id") != registry["population"]["population_id"]
            or value.get("claim_boundary") != registry["claim_boundary"]
            or len(observed_ids) != int(registry["population"]["expected_source_rows"])
        ):
            raise PrefixContractError("prefix fit input registry/population binding drifted")
    return value


def load_private_labels(
    path: str | Path, *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_payload(value, name="prefix private labels")
    if set(value) != {
        "schema_version",
        "population_id",
        "positive_class",
        "grouping_unit",
        "rows",
        "payload_sha256",
    }:
        raise PrefixContractError("prefix private-label field roster drifted")
    if (
        value.get("schema_version") != PRIVATE_LABEL_SCHEMA
        or value.get("population_id") != registry["population"]["population_id"]
        or value.get("positive_class") != registry["evaluation"]["positive_class"]
        or value.get("grouping_unit") != "source question"
    ):
        raise PrefixContractError("unexpected prefix private-label schema")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise PrefixContractError("prefix private-label rows are malformed")
    row_ids: set[str] = set()
    group_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "row_id", "group_id", "family", "label", "final_length"
        }:
            raise PrefixContractError("prefix private-label row fields drifted")
        row_id = row["row_id"]
        group_id = row["group_id"]
        family = row["family"]
        label = row["label"]
        final_length = row["final_length"]
        if (
            not isinstance(row_id, str)
            or not row_id
            or group_id != row_id
            or family not in SUBSETS
            or type(label) is not int
            or label not in {0, 1}
            or type(final_length) is not int
            or final_length <= min(BUDGETS)
            or row_id in row_ids
            or group_id in group_ids
        ):
            raise PrefixContractError("prefix private-label row semantics drifted")
        row_ids.add(row_id)
        group_ids.add(group_id)
        family_counts[str(family)] += 1
        label_counts[int(label)] += 1
    population = registry["population"]
    if (
        len(rows) != int(population["expected_evaluation_traces"])
        or family_counts
        != Counter({
            str(key): int(count)
            for key, count in population["expected_evaluation_traces_by_subset"].items()
        })
        or label_counts
        != Counter({
            0: int(population["expected_correct"]),
            1: int(population["expected_incorrect"]),
        })
    ):
        raise PrefixContractError("prefix private-label population contract drifted")
    return value


__all__ = [
    "EXPECTED_SCORE_FILENAME",
    "FIT_INPUT_FILENAME",
    "PREPARATION_MANIFEST_FILENAME",
    "PRIVATE_LABEL_FILENAME",
    "load_fit_input",
    "load_preparation_manifest",
    "load_private_labels",
    "prepare_prefix_build",
    "reconstruct_prefix_preparation",
    "sanitize_source_row",
]
