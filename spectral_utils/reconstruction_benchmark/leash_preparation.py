"""Source-bound, target-isolated preparation for the LEASH stopping lane."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
from io import BytesIO
import math
from pathlib import Path, PurePosixPath
import pickle
import tempfile
from typing import Any

from spectral_utils.fair_comparisons.stopping import canonical_s2_group_id, canonical_s2_id
from spectral_utils.paper_exact.leash import LeashConfig, LeashStopper
from spectral_utils.paper_exact.manifest import sha256_order

from .io import (
    canonical_json_bytes,
)
from .leash_contract import (
    ARMS,
    AtomicLeashDirectory,
    BLOCKED_STATUS,
    FIDELITY,
    PREPARATION_SCHEMA,
    PREPARATION_AB_SCHEMA,
    PRIVATE_OUTCOME_SCHEMA,
    READY_STATUS,
    LeashContractError,
    add_payload_sha256,
    assert_no_forbidden_keys,
    assert_no_symlinks,
    bound_json_sha256,
    bound_tree_manifest,
    canonical_jsonl_bytes,
    load_registry,
    leash_tree_load_json,
    leash_tree_manifest,
    leash_tree_write_bytes,
    leash_tree_write_json,
    payload_sha256,
    parse_json_bytes,
    parse_jsonl_bytes,
    read_bound_bytes,
    read_authenticated_source_guard_code,
    resolve_source_path,
    source_guard_closure_sha256,
    validate_fit_row,
    verify_blocked_run,
    verify_ready_tree,
    load_json,
)


FIT_INPUT_FILENAME = "FIT_INPUT.jsonl"
COVERAGE_FILENAME = "COVERAGE.json"
SOURCE_SNAPSHOT_FILENAME = "SOURCE_SNAPSHOT.json"
PREPARATION_MANIFEST_FILENAME = "PREPARATION_MANIFEST.json"
OUTCOMES_FILENAME = "OUTCOMES.jsonl"
OUTCOME_MANIFEST_FILENAME = "OUTCOME_MANIFEST.json"


def _frozen_leash_config(registry: Mapping[str, Any]) -> LeashConfig:
    published = registry["policy_contract"]["published_constants"]
    declared = registry["policy_contract"]["declared_not_paper_specified"]
    return LeashConfig(
        k=int(published["k"]),
        L=int(published["L"]),
        eps_H=float(published["eps_H"]),
        delta_M=float(published["delta_M"]),
        m=int(published["m"]),
        M=int(published["M"]),
        B=float(declared["B"]),
        tau_p=float(declared["tau_p"]),
        w=int(declared["w"]),
        gamma=float(declared["gamma"]),
        setting_label="central",
    )


def _token_and_policy_evidence(
    raw: Mapping[str, Any], *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove closure realization and replay every LEASH callback from raw channels."""

    trace_key = str(raw.get("trace_key", ""))
    arm = str(raw.get("arm", ""))
    gen_tokens = raw.get("gen_token_ids")
    answer_tokens = raw.get("answer_token_ids")
    if not isinstance(gen_tokens, list) or not isinstance(answer_tokens, list):
        raise LeashContractError(f"{trace_key} lacks generation/answer token evidence")
    reasoning = raw.get("n_reasoning_tokens")
    closure = raw.get("n_closure_tokens")
    total = raw.get("n_total_tokens")
    if (
        isinstance(reasoning, bool)
        or not isinstance(reasoning, int)
        or isinstance(closure, bool)
        or not isinstance(closure, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or len(gen_tokens) != reasoning
        or len(answer_tokens) != closure
        or total != len(gen_tokens) + len(answer_tokens)
    ):
        raise LeashContractError(f"{trace_key} token arrays/counts do not agree")
    if raw.get("closure_generated") is not True or closure <= 0:
        raise LeashContractError(f"{trace_key} lacks a realized non-empty closure")
    if not isinstance(raw.get("answer_text"), str):
        raise LeashContractError(f"{trace_key} lacks generated closure text")

    if arm != "leash":
        if raw.get("leash") is not None:
            raise LeashContractError(
                f"control trace unexpectedly carries LEASH diagnostics: {trace_key}"
            )
        return {
            "actual_policy_execution_observed": False,
            "policy_replay_verified": False,
            "policy_replay_fired": None,
            "policy_replay_stop_index": None,
            "closure_evidence_verified": True,
        }

    channels = raw.get("channels")
    if not isinstance(channels, Mapping):
        raise LeashContractError(f"LEASH trace lacks raw callback channels: {trace_key}")
    series: list[list[float]] = []
    for name in ("raw_entropy", "raw_margin", "raw_pmax"):
        values = channels.get(name)
        if not isinstance(values, list) or len(values) != reasoning:
            raise LeashContractError(f"LEASH trace lacks complete {name}: {trace_key}")
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError) as error:
            raise LeashContractError(f"LEASH trace has invalid {name}: {trace_key}") from error
        if any(not math.isfinite(value) for value in vector):
            raise LeashContractError(f"LEASH trace has non-finite {name}: {trace_key}")
        series.append(vector)
    if any(value < 0.0 or value > 1.0 for value in series[2]):
        raise LeashContractError(f"LEASH trace has invalid raw_pmax: {trace_key}")

    cfg = _frozen_leash_config(registry)
    stopper = LeashStopper(cfg)
    first_fire: int | None = None
    for index, (entropy, margin, pmax) in enumerate(zip(*series, strict=True), start=1):
        if stopper.push(entropy, margin, pmax) and first_fire is None:
            first_fire = index
    replay_fired = first_fire is not None
    expected_reason = "policy" if replay_fired else "length"
    if replay_fired != (raw.get("stopped_early") is True) or raw.get("stop_reason") != expected_reason:
        raise LeashContractError(f"LEASH callback replay disagrees with stop fields: {trace_key}")
    if replay_fired and first_fire != reasoning:
        raise LeashContractError(f"LEASH trace continued after replayed policy fire: {trace_key}")
    if not replay_fired and reasoning != cfg.M:
        raise LeashContractError(f"LEASH non-fired rationale did not reach frozen cap: {trace_key}")

    diagnostics = raw.get("leash")
    if not isinstance(diagnostics, Mapping):
        raise LeashContractError(f"LEASH trace lacks stopper diagnostics: {trace_key}")
    fired_diag = diagnostics.get("fired")
    diag_stop = None if fired_diag is None else fired_diag.get("t")
    if (
        int(diagnostics.get("n_steps", -1)) != reasoning
        or diagnostics.get("config") != cfg.as_manifest()
        or diag_stop != first_fire
        or (fired_diag is None) != (first_fire is None)
    ):
        raise LeashContractError(f"LEASH replay disagrees with stored diagnostics: {trace_key}")
    if stopper.diagnostics().get("fired") != fired_diag:
        raise LeashContractError(f"LEASH fired diagnostics failed exact replay: {trace_key}")
    return {
        "actual_policy_execution_observed": True,
        "policy_replay_verified": True,
        "policy_replay_fired": replay_fired,
        "policy_replay_stop_index": first_fire,
        "closure_evidence_verified": True,
    }


def _raw_outcomes(
    run_dir: Path,
    *,
    registry: Mapping[str, Any],
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_tree: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    outcomes: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []
    tree_files = {item["path"]: item for item in expected_tree["files"]}
    index_file = tree_files.get("INDEX.jsonl", {})
    index_bytes = read_bound_bytes(
        run_dir / "INDEX.jsonl",
        name=f"LEASH shard index {spec['run_id']}",
        expected_bytes=index_file.get("bytes"),
        expected_sha256=str(spec["index_sha256"]),
    )
    entries = parse_jsonl_bytes(index_bytes, name=f"LEASH shard index {spec['run_id']}")
    expected_ids = set(str(value) for value in manifest["dataset_example_ids"])
    seen_shards: set[int] = set()
    seen_paths: set[str] = set()
    seen_keys: set[str] = set()
    indexed_bytes = 0
    for entry_number, entry in enumerate(entries):
        required = {"shard", "path", "n_traces", "bytes", "sha256", "keys", "question_ids"}
        if not required.issubset(entry):
            raise LeashContractError(
                f"LEASH shard index entry {entry_number} lacks required fields"
            )
        shard_number = entry["shard"]
        if (
            isinstance(shard_number, bool)
            or not isinstance(shard_number, int)
            or shard_number in seen_shards
        ):
            raise LeashContractError(f"invalid/duplicate LEASH shard number: {shard_number!r}")
        seen_shards.add(shard_number)
        relative = str(entry.get("path", ""))
        posix = PurePosixPath(relative)
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or not relative.startswith("shards/")
            or relative in seen_paths
        ):
            raise LeashContractError(f"unsafe shard path {relative!r}")
        seen_paths.add(relative)
        path = run_dir.joinpath(*posix.parts)
        registered_file = tree_files.get(relative, {})
        expected_size = int(entry["bytes"])
        if registered_file.get("bytes") != expected_size:
            raise LeashContractError(f"raw outcome shard size binding failed: {path}")
        payload = read_bound_bytes(
            path,
            name=f"raw LEASH outcome shard {relative}",
            expected_bytes=expected_size,
            expected_sha256=str(entry["sha256"]),
        )
        if registered_file.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise LeashContractError(f"raw outcome shard tree binding failed: {path}")
        indexed_bytes += len(payload)
        try:
            stream = BytesIO(payload)
            records = pickle.load(stream)
            if stream.tell() != len(payload):
                raise LeashContractError(f"raw outcome shard has trailing pickle bytes: {path}")
        except Exception as error:  # noqa: BLE001 - any decode failure is integrity failure
            raise LeashContractError(f"cannot decode raw outcome shard: {path}") from error
        if not isinstance(records, list) or any(not isinstance(record, Mapping) for record in records):
            raise LeashContractError(f"raw outcome shard is not a record list: {path}")
        keys = entry["keys"]
        if (
            not isinstance(keys, list)
            or len(keys) != int(entry["n_traces"])
            or len(records) != int(entry["n_traces"])
            or [record.get("trace_key") for record in records] != keys
        ):
            raise LeashContractError(f"raw outcome shard key/count binding failed: {path}")
        if seen_keys.intersection(str(value) for value in keys):
            raise LeashContractError(f"duplicate raw trace key across shards: {path}")
        seen_keys.update(str(value) for value in keys)
        question_ids = sorted({str(record.get("question_id")) for record in records})
        if question_ids != entry["question_ids"]:
            raise LeashContractError(f"raw outcome shard question coverage failed: {path}")
        for raw in records:
            if not isinstance(raw, Mapping):
                raise LeashContractError(f"raw outcome shard contains a non-object: {path}")
            trace_key = str(raw.get("trace_key", ""))
            if not trace_key or trace_key in outcomes:
                raise LeashContractError(f"missing/duplicate raw trace key {trace_key!r}")
            arm = str(raw.get("arm", ""))
            question_id = str(raw.get("question_id", ""))
            if (
                arm not in ARMS
                or question_id not in expected_ids
                or raw.get("setting_label") != "central"
                or trace_key != f"{arm}:central:{question_id}"
                or not isinstance(raw.get("stopped_early"), bool)
                or raw.get("closure_generated") is not True
            ):
                raise LeashContractError(f"raw LEASH trace identity/setting drifted: {trace_key}")
            try:
                wall_s = float(raw.get("wall_s"))
            except (TypeError, ValueError) as error:
                raise LeashContractError(f"raw LEASH wall time is invalid: {trace_key}") from error
            if not math.isfinite(wall_s) or wall_s < 0:
                raise LeashContractError(f"raw LEASH wall time is invalid: {trace_key}")
            evidence[trace_key] = _token_and_policy_evidence(raw, registry=registry)
            revision = str(manifest["dataset_revision"])
            dataset = str(spec["dataset"])
            model = str(manifest["model_id"])
            source_records.append(
                {
                    "row_id": canonical_s2_id(revision, dataset, question_id, model, arm),
                    "group_id": canonical_s2_group_id(revision, dataset, question_id),
                    "cell_id": f"s2::{dataset}::{model}",
                    "population_id": f"s2_stopping::{revision}::{dataset}::{model}",
                    "dataset_revision": revision,
                    "dataset": dataset,
                    "question_id": question_id,
                    "model": model,
                    "model_revision": str(manifest["model_revision"]),
                    "arm": arm,
                    "method_id": f"{arm}|central",
                    "trace_key": trace_key,
                    "source_artifact_sha256": str(entry["sha256"]),
                    "n_reasoning_tokens": raw.get("n_reasoning_tokens"),
                    "n_closure_tokens": raw.get("n_closure_tokens"),
                    "n_total_tokens": raw.get("n_total_tokens"),
                    "wall_s": wall_s,
                    "stopped_early": raw.get("stopped_early"),
                    "closure_generated": raw.get("closure_generated"),
                    "stop_reason": raw.get("stop_reason"),
                    "fidelity": str(manifest["fidelity"]),
                }
            )
            outcomes[trace_key] = {
                "trace_key": trace_key,
                "answer_text": raw.get("answer_text"),
                "gold_answer": raw.get("gold_answer"),
                "stored_correct": bool(raw.get("correct")),
                "stored_prediction": raw.get("pred_answer"),
                "stored_parse_status": raw.get("parse_status"),
                "source_artifact_sha256": str(entry["sha256"]),
            }
    expected_keys = {
        f"{arm}:central:{question_id}" for arm in ARMS for question_id in expected_ids
    }
    if seen_keys != expected_keys:
        raise LeashContractError(f"raw LEASH trace coverage drifted under {run_dir}")
    leash_rows = [value for value in evidence.values() if value["policy_replay_verified"]]
    n_stops = sum(bool(value["policy_replay_fired"]) for value in leash_rows)
    if n_stops != int(spec["expected_leash_policy_stops"]):
        raise LeashContractError(
            f"LEASH source stop count drifted under {run_dir}: "
            f"expected {spec['expected_leash_policy_stops']}, observed {n_stops}"
        )
    return outcomes, evidence, source_records, {
        "n_leash_rows_replayed": len(leash_rows),
        "n_leash_policy_stops": n_stops,
        "n_replay_mismatches": 0,
        "n_rows_with_token_count_and_closure_evidence": len(evidence),
        "n_shards": len(entries),
        "bytes_total": indexed_bytes,
        "frozen_config": _frozen_leash_config(registry).as_manifest(),
    }


def _fit_row(record: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "row_id": str(record["row_id"]),
        "group_id": str(record["group_id"]),
        "cell_id": str(record["cell_id"]),
        "population_id": str(record["population_id"]),
        "dataset_revision": str(record["dataset_revision"]),
        "dataset": str(record["dataset"]),
        "question_id": str(record["question_id"]),
        "model": str(record["model"]),
        "model_revision": str(record["model_revision"]),
        "arm": str(record["arm"]),
        "method_id": str(record["method_id"]),
        "trace_key": str(record["trace_key"]),
        "source_artifact_sha256": str(record["source_artifact_sha256"]),
        "n_reasoning_tokens": int(record["n_reasoning_tokens"]),
        "n_closure_tokens": int(record["n_closure_tokens"]),
        "n_total_tokens": int(record["n_total_tokens"]),
        "wall_s": float(record["wall_s"]),
        "stopped_early": bool(record["stopped_early"]),
        "closure_generated": bool(record["closure_generated"]),
        "stop_reason": record.get("stop_reason"),
        "setting_label": "central",
        "fidelity": str(record["fidelity"]),
        "actual_policy_execution_observed": evidence["actual_policy_execution_observed"],
        "policy_replay_verified": evidence["policy_replay_verified"],
        "policy_replay_fired": evidence["policy_replay_fired"],
        "policy_replay_stop_index": evidence["policy_replay_stop_index"],
        "closure_evidence_verified": evidence["closure_evidence_verified"],
    }
    validate_fit_row(row)
    return row


def _validate_rectangular(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_cell: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        key = (str(row["dataset"]), str(row["model"]))
        arm = str(row["arm"])
        question_id = str(row["question_id"])
        if question_id in by_cell[key][arm]:
            raise LeashContractError(f"duplicate LEASH cell/arm/question row: {key}/{arm}/{question_id}")
        by_cell[key][arm].add(question_id)
    if len(by_cell) != 6:
        raise LeashContractError(f"LEASH ready population has {len(by_cell)} cells, expected six")
    dataset_questions: dict[str, set[str]] = {}
    for key, arm_sets in by_cell.items():
        if set(arm_sets) != set(ARMS):
            raise LeashContractError(f"LEASH cell {key} lacks the exact arm roster")
        baseline = arm_sets[ARMS[0]]
        if any(arm_sets[arm] != baseline for arm in ARMS[1:]):
            raise LeashContractError(f"LEASH cell {key} is not paired across arms")
        dataset = key[0]
        if dataset in dataset_questions and dataset_questions[dataset] != baseline:
            raise LeashContractError(f"LEASH dataset {dataset} is not paired across models")
        dataset_questions[dataset] = baseline
    return {
        "identical_question_ids_across_arms": True,
        "identical_question_ids_across_model_copies": True,
        "n_cells": 6,
        "n_rows": len(rows),
        "question_counts": {dataset: len(ids) for dataset, ids in sorted(dataset_questions.items())},
    }


def derive_leash_preparation(
    *, source_root: str | Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently derive sanitized fit rows and private outcomes from raw shards."""

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    replay_audits: list[dict[str, Any]] = []

    implementation_files = []
    for name, spec in sorted(registry["source_contract"]["implementation_files"].items()):
        relative = str(spec["path"])
        candidate = resolve_source_path(
            source_root, relative, name=f"LEASH implementation::{name}"
        )
        payload = read_bound_bytes(
            candidate,
            name=f"LEASH implementation source {name}",
            expected_sha256=str(spec["sha256"]),
        )
        implementation_files.append(
            {"asset_id": name, "path": relative, "bytes": len(payload), "sha256": spec["sha256"]}
        )

    code_root = Path(__file__).resolve().parents[2]
    guard_payloads = read_authenticated_source_guard_code(code_root, registry)
    source_guard_code_files = [
        {
            "asset_id": name,
            "path": str(spec["path"]),
            "bytes": len(guard_payloads[str(spec["path"])]),
            "sha256": str(spec["sha256"]),
        }
        for name, spec in sorted(
            registry["source_contract"]["source_guard_code_files"].items()
        )
    ]

    for spec in registry["source_contract"]["ready_runs"]:
        run_dir, tree = verify_ready_tree(source_root, spec)
        tree_files = {item["path"]: item for item in tree["files"]}
        control_payloads: dict[str, bytes] = {}
        for filename, digest_key in (
            ("RUN_MANIFEST.json", "manifest_sha256"),
            ("STATUS.json", "status_sha256"),
            ("SUMMARY.json", "summary_sha256"),
        ):
            registered = tree_files.get(filename, {})
            control_payloads[filename] = read_bound_bytes(
                run_dir / filename,
                name=f"ready LEASH source {spec['run_id']}/{filename}",
                expected_bytes=registered.get("bytes"),
                expected_sha256=str(spec[digest_key]),
            )
        manifest = parse_json_bytes(
            control_payloads["RUN_MANIFEST.json"], name="ready LEASH RUN_MANIFEST"
        )
        status = parse_json_bytes(
            control_payloads["STATUS.json"], name="ready LEASH STATUS"
        )
        parse_json_bytes(
            control_payloads["SUMMARY.json"], name="ready LEASH SUMMARY"
        )
        question_ids = manifest.get("dataset_example_ids")
        leash_config = manifest.get("extra", {}).get("leash_config", {})
        if (
            manifest.get("run_id") != spec["run_id"]
            or manifest.get("model_id") != spec["model"]
            or not isinstance(question_ids, list)
            or len(question_ids) != int(spec["expected_questions"])
            or len(question_ids) != len(set(question_ids))
            or manifest.get("dataset_order_sha256") != sha256_order(question_ids)
            or manifest.get("dataset_source")
            != {"aqua": "deepmind/aqua_rat", "gsm8k": "openai/gsm8k"}[spec["dataset"]]
            or int(manifest.get("expected_traces", -1)) != int(spec["expected_traces"])
            or status.get("complete") is not True
            or int(status.get("n_expected", -1)) != int(spec["expected_traces"])
            or int(status.get("n_finished", -1)) != int(spec["expected_traces"])
            or int(status.get("n_failed", -1)) != 0
            or bool(status.get("failures"))
            or not isinstance(manifest.get("extra", {}).get("arms"), list)
            or set(manifest.get("extra", {}).get("arms", ())) != set(ARMS)
            or len(manifest.get("extra", {}).get("arms", ())) != len(ARMS)
            or manifest.get("extra", {}).get("sweep") is not False
            or manifest.get("fidelity") != FIDELITY
            or leash_config.get("published") != registry["policy_contract"]["published_constants"]
            or leash_config.get("declared_by_us") != registry["policy_contract"]["declared_not_paper_specified"]
            or leash_config.get("setting_label") != "central"
            or int(leash_config.get("t_min", -1)) != int(registry["policy_contract"]["t_min"])
            or leash_config.get("fidelity") != FIDELITY
            or manifest.get("repo_commit") != registry["policy_contract"]["acquisition_repo_commit"]
            or manifest.get("repo_dirty") is not registry["policy_contract"]["acquisition_repo_dirty"]
            or manifest.get("stop_behavior", {}).get("rationale_cap")
            != registry["policy_contract"]["published_constants"]["M"]
            or manifest.get("stop_behavior", {}).get("policy") != "LEASH Alg. 1 (leash arm only)"
            or not isinstance(manifest.get("stop_behavior", {}).get("second_stage"), str)
            or not manifest.get("stop_behavior", {}).get("second_stage")
        ):
            raise LeashContractError(f"ready source semantics drifted for {spec['run_id']}")
        raw, evidence, source_records, replay_audit = _raw_outcomes(
            run_dir,
            registry=registry,
            spec=spec,
            manifest=manifest,
            expected_tree=tree,
        )
        if (
            len(raw) != int(spec["expected_traces"])
            or len(source_records) != int(spec["expected_traces"])
            or int(status.get("n_shards", -1)) != int(replay_audit["n_shards"])
            or int(status.get("bytes_total", -1)) != int(replay_audit["bytes_total"])
        ):
            raise LeashContractError(f"raw/private outcome count drifted for {spec['run_id']}")
        for record in source_records:
            trace_evidence = evidence.get(str(record["trace_key"]))
            if trace_evidence is None:
                raise LeashContractError(
                    f"missing execution replay evidence for {record['trace_key']}"
                )
            row = _fit_row(record, trace_evidence)
            if row["row_id"] in seen_row_ids:
                raise LeashContractError(f"duplicate canonical LEASH row ID {row['row_id']}")
            seen_row_ids.add(row["row_id"])
            outcome = raw.get(row["trace_key"])
            if outcome is None or outcome["source_artifact_sha256"] != row["source_artifact_sha256"]:
                raise LeashContractError(f"private/public source join failed for {row['row_id']}")
            public_rows.append(row)
            private_rows.append({"row_id": row["row_id"], **outcome})
        if bound_tree_manifest(
            run_dir, name=f"ready LEASH source {spec['run_id']} final binding"
        ) != tree:
            raise LeashContractError(
                f"ready LEASH source changed during derivation: {spec['run_id']}"
            )
        source_runs.append(
            {
                "run_id": spec["run_id"],
                "registered_path": spec["path"],
                "tree": tree,
                "coverage_status": READY_STATUS,
                "policy_replay_audit": replay_audit,
            }
        )
        replay_audits.append({"run_id": spec["run_id"], **replay_audit})
        coverage.append(
            {
                "run_id": spec["run_id"],
                "dataset": spec["dataset"],
                "model": spec["model"],
                "coverage_status": READY_STATUS,
                "n_expected": int(spec["expected_traces"]),
                "n_finished": int(spec["expected_traces"]),
                "n_failed": 0,
                "actual_policy_execution_observed": True,
                "n_leash_rows_replayed": replay_audit["n_leash_rows_replayed"],
                "n_leash_policy_stops": replay_audit["n_leash_policy_stops"],
                "n_policy_replay_mismatches": 0,
                "actual_stopping_claim_eligible": False,
                "usable_for_evaluation": True,
                "fidelity": FIDELITY,
            }
        )

    for spec in registry["source_contract"]["blocked_runs"]:
        _, blocked_coverage = verify_blocked_run(source_root, spec)
        coverage.append(blocked_coverage)
        source_runs.append(
            {
                "run_id": spec["run_id"],
                "registered_path": spec["path"],
                "registered_files": [
                    {"path": name, "sha256": digest}
                    for name, digest in sorted(spec["files"].items())
                ],
                "coverage_status": BLOCKED_STATUS,
            }
        )

    public_rows.sort(key=lambda row: row["row_id"])
    private_rows.sort(key=lambda row: row["row_id"])
    coverage.sort(key=lambda row: row["run_id"])
    source_runs.sort(key=lambda row: row["run_id"])
    if len(public_rows) != int(registry["population"]["expected_ready_traces"]):
        raise LeashContractError("prepared LEASH trace count does not match registry")
    if [row["row_id"] for row in public_rows] != [row["row_id"] for row in private_rows]:
        raise LeashContractError("public/private LEASH row order drifted")
    for row in public_rows:
        validate_fit_row(row)
    assert_no_forbidden_keys(public_rows)
    pairing = _validate_rectangular(public_rows)
    return {
        "fit_rows": public_rows,
        "outcome_rows": private_rows,
        "coverage": coverage,
        "source_snapshot": add_payload_sha256(
            {
                "schema_version": "reconstruction-leash-source-snapshot-v1",
                "fidelity": FIDELITY,
                "implementation_files": implementation_files,
                "source_guard_code_files": source_guard_code_files,
                "source_guard_code_closure_sha256": source_guard_closure_sha256(
                    registry
                ),
                "runs": source_runs,
            }
        ),
        "pairing": pairing,
        "policy_replay_audits": sorted(replay_audits, key=lambda row: row["run_id"]),
    }


def audit_leash_sources(
    *, source_root: str | Path, registry_path: str | Path
) -> dict[str, Any]:
    """Read-only real/synthetic preflight; creates no release artifact."""

    registry = load_registry(registry_path)
    derived = derive_leash_preparation(source_root=source_root, registry=registry)
    return {
        "status": "PASS",
        "mode": "READ_ONLY_SOURCE_PREFLIGHT",
        "lane_id": registry["lane_id"],
        "fidelity": FIDELITY,
        "n_ready_rows": len(derived["fit_rows"]),
        "coverage": derived["coverage"],
        "pairing": derived["pairing"],
        "policy_replay_audits": derived["policy_replay_audits"],
        "n_leash_rows_replayed": sum(
            row["n_leash_rows_replayed"] for row in derived["policy_replay_audits"]
        ),
        "n_policy_replay_mismatches": 0,
        "source_snapshot_sha256": derived["source_snapshot"]["payload_sha256"],
        "paper_exact_claim": False,
        "conceptual_objective_reproduced_as_equation": False,
        "science_results_published": False,
        "actual_stopping_claim_eligible": False,
        "next_required_gate": "independent preparation A/B then target-blind fit A/B policy validation",
    }


def _write_preparation_trees(
    *,
    public_stage: AtomicLeashDirectory,
    private_stage: AtomicLeashDirectory,
    derived: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_path: str | Path,
) -> None:
    fit_bytes = canonical_jsonl_bytes(derived["fit_rows"])
    outcome_bytes = canonical_jsonl_bytes(derived["outcome_rows"])
    fit_sha = leash_tree_write_bytes(public_stage, FIT_INPUT_FILENAME, fit_bytes)
    coverage_bundle = add_payload_sha256(
        {
            "schema_version": "reconstruction-leash-coverage-v1",
            "lane_id": registry["lane_id"],
            "rows": derived["coverage"],
        }
    )
    coverage_sha = leash_tree_write_json(public_stage, COVERAGE_FILENAME, coverage_bundle)
    snapshot_sha = leash_tree_write_json(
        public_stage, SOURCE_SNAPSHOT_FILENAME, derived["source_snapshot"]
    )
    manifest = add_payload_sha256(
        {
            "schema_version": PREPARATION_SCHEMA,
            "lane_id": registry["lane_id"],
            "fidelity": FIDELITY,
            "registry_sha256": bound_json_sha256(
                registry_path, registry, name="LEASH registry"
            ),
            "fit_visible_targets": False,
            "outcomes_stored_in_private_tree": True,
            "actual_policy_execution_observed": True,
            "actual_stopping_claim_eligible": False,
            "proxy_stopping": False,
            "paper_exact_claim": False,
            "conceptual_objective_reproduced_as_equation": False,
            "n_rows": len(derived["fit_rows"]),
            "n_groups": len({row["group_id"] for row in derived["fit_rows"]}),
            "row_order_sha256": payload_sha256([row["row_id"] for row in derived["fit_rows"]]),
            "pairing": derived["pairing"],
            "policy_replay_audits": derived["policy_replay_audits"],
            "files": {
                FIT_INPUT_FILENAME: fit_sha,
                COVERAGE_FILENAME: coverage_sha,
                SOURCE_SNAPSHOT_FILENAME: snapshot_sha,
            },
        }
    )
    leash_tree_write_json(public_stage, PREPARATION_MANIFEST_FILENAME, manifest)

    outcome_sha = leash_tree_write_bytes(private_stage, OUTCOMES_FILENAME, outcome_bytes)
    private_manifest = add_payload_sha256(
        {
            "schema_version": PRIVATE_OUTCOME_SCHEMA,
            "lane_id": registry["lane_id"],
            "fidelity": FIDELITY,
            "n_rows": len(derived["outcome_rows"]),
            "row_order_sha256": payload_sha256([row["row_id"] for row in derived["outcome_rows"]]),
            "public_fit_input_sha256": fit_sha,
            "files": {OUTCOMES_FILENAME: outcome_sha},
        }
    )
    leash_tree_write_json(private_stage, OUTCOME_MANIFEST_FILENAME, private_manifest)


def derive_source_bound_preparation_contract(
    *, source_root: str | Path, registry_path: str | Path
) -> dict[str, Any]:
    """Rederive the exact preparation trees and canonical A/B certificate from source.

    This is the trust anchor used by every downstream gate.  In particular, a
    self-consistent pair of edited preparation trees and a refreshed self-hash
    certificate cannot replace this source-derived expectation.
    """

    registry = load_registry(registry_path)
    derived = derive_leash_preparation(source_root=source_root, registry=registry)
    with tempfile.TemporaryDirectory(
        prefix="leash-source-prep-contract-",
        dir=Path(tempfile.gettempdir()).resolve(strict=True),
    ) as temporary:
        public = AtomicLeashDirectory(Path(temporary) / "public")
        private = AtomicLeashDirectory(Path(temporary) / "private")
        try:
            _write_preparation_trees(
                public_stage=public,
                private_stage=private,
                derived=derived,
                registry=registry,
                registry_path=registry_path,
            )
            public_tree = leash_tree_manifest(public)
            private_tree = leash_tree_manifest(private)
            preparation_manifest = leash_tree_load_json(
                public,
                PREPARATION_MANIFEST_FILENAME,
                name="source-rederived LEASH preparation manifest",
            )
        finally:
            public.cleanup()
            private.cleanup()
    certificate = add_payload_sha256(
        {
            "schema_version": PREPARATION_AB_SCHEMA,
            "status": "PASS",
            "lane_id": registry["lane_id"],
            "registry_sha256": bound_json_sha256(
                registry_path, registry, name="LEASH registry"
            ),
            "n_rows": len(derived["fit_rows"]),
            "source_snapshot_payload_sha256": derived["source_snapshot"]["payload_sha256"],
            "public_tree_sha256": {
                "A": public_tree["tree_sha256"],
                "B": public_tree["tree_sha256"],
            },
            "private_tree_sha256": {
                "A": private_tree["tree_sha256"],
                "B": private_tree["tree_sha256"],
            },
            "rederived_public_tree_sha256": public_tree["tree_sha256"],
            "rederived_private_tree_sha256": private_tree["tree_sha256"],
            "byte_identical": True,
            "source_rebound": True,
            "label_firewall_rechecked": True,
            "transitive_rederivation": True,
        }
    )
    return {
        "registry": registry,
        "fit_rows": derived["fit_rows"],
        "public_tree": public_tree,
        "private_tree": private_tree,
        "preparation_manifest": preparation_manifest,
        "policy_replay_audits": derived["policy_replay_audits"],
        "certificate": certificate,
    }


def require_source_bound_preparation(
    *,
    source_root: str | Path,
    registry_path: str | Path,
    preparation_dir: str | Path,
    preparation_ab_certificate: str | Path,
) -> dict[str, Any]:
    """Authenticate one public prep copy and its certificate against current source."""

    contract = derive_source_bound_preparation_contract(
        source_root=source_root, registry_path=registry_path
    )
    assert_no_symlinks(preparation_dir, name="LEASH public preparation")
    if Path(preparation_ab_certificate).is_symlink():
        raise LeashContractError("LEASH preparation certificate is a symlink")
    observed_tree = bound_tree_manifest(
        preparation_dir, name="LEASH public preparation"
    )
    if observed_tree != contract["public_tree"]:
        raise LeashContractError(
            "LEASH preparation differs from exact current-source rederivation"
        )
    expected_certificate = contract["certificate"]
    expected_certificate_bytes = canonical_json_bytes(expected_certificate) + b"\n"
    if read_bound_bytes(
        preparation_ab_certificate, name="LEASH preparation A/B certificate"
    ) != expected_certificate_bytes:
        raise LeashContractError("LEASH preparation certificate is not canonical source-derived bytes")
    return contract


def prepare_leash_build(
    *,
    source_root: str | Path,
    registry_path: str | Path,
    public_output: str | Path,
    private_output: str | Path,
) -> dict[str, Any]:
    """Build one preparation copy with paired public/private no-clobber publication."""

    registry = load_registry(registry_path)
    derived = derive_leash_preparation(source_root=source_root, registry=registry)
    public_stage = AtomicLeashDirectory(public_output)
    private_stage = AtomicLeashDirectory(private_output)
    try:
        _write_preparation_trees(
            public_stage=public_stage,
            private_stage=private_stage,
            derived=derived,
            registry=registry,
            registry_path=registry_path,
        )
        public_tree = leash_tree_manifest(public_stage)
        private_tree = leash_tree_manifest(private_stage)
        public_stage.commit()
        try:
            private_stage.commit()
        except BaseException:
            public_stage.rollback()
            raise
    finally:
        public_stage.cleanup()
        private_stage.cleanup()
    return {
        "status": "PASS",
        "public_output": str(public_stage.final_path),
        "private_output": str(private_stage.final_path),
        "public_tree": public_tree,
        "private_tree": private_tree,
        "n_rows": len(derived["fit_rows"]),
    }


__all__ = [
    "COVERAGE_FILENAME", "FIT_INPUT_FILENAME", "OUTCOMES_FILENAME", "OUTCOME_MANIFEST_FILENAME",
    "PREPARATION_MANIFEST_FILENAME", "SOURCE_SNAPSHOT_FILENAME", "audit_leash_sources",
    "derive_leash_preparation", "derive_source_bound_preparation_contract",
    "prepare_leash_build", "require_source_bound_preparation",
]
