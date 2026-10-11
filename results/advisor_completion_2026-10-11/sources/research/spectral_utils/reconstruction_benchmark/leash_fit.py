"""Target-blind policy-execution freeze for the LEASH stopping lane."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import os
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess
import sys
import tempfile
from typing import Any

from .io import (
    canonical_json_bytes,
)
from .leash_contract import (
    AtomicLeashDirectory,
    FIDELITY,
    FIT_SCHEMA,
    FIT_AB_SCHEMA,
    PREPARATION_AB_SCHEMA,
    PREPARATION_SCHEMA,
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
    source_guard_closure_sha256,
    validate_fit_row,
    verify_payload,
)
from .leash_preparation import (
    FIT_INPUT_FILENAME,
    PREPARATION_MANIFEST_FILENAME,
    derive_source_bound_preparation_contract,
)


POLICY_LEDGER_FILENAME = "POLICY_EXECUTION.jsonl"
FIT_MANIFEST_FILENAME = "FIT_MANIFEST.json"
_MAX_SOURCE_GUARD_BYTES = 128 * 1024 * 1024


def _source_guard_attestation(registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "process_isolated": True,
        "target_fields_returned": False,
        "private_outcome_rows_returned": False,
        "authenticated_code_capsule": True,
        "controller_independent_rederivation": True,
        "code_closure_sha256": source_guard_closure_sha256(registry),
    }


def _controller_source_preparation_bytes(
    *, source_root: str | Path, registry_path: str | Path
) -> bytes:
    """Independently derive expected guard bytes in a label-isolated fork."""

    if not hasattr(os, "fork"):
        raise LeashContractError(
            "LEASH source guard requires fork isolation for controller rederivation"
        )
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertions happen in the controller
        os.close(reader)
        try:
            contract = derive_source_bound_preparation_contract(
                source_root=source_root, registry_path=registry_path
            )
            contract["source_guard"] = _source_guard_attestation(contract["registry"])
            payload = canonical_json_bytes(contract) + b"\n"
            if len(payload) > _MAX_SOURCE_GUARD_BYTES:
                os._exit(72)
            offset = 0
            while offset < len(payload):
                offset += os.write(writer, payload[offset:])
            os.close(writer)
            os._exit(0)
        except BaseException:  # noqa: BLE001 - any child failure fails the gate
            try:
                os.close(writer)
            finally:
                os._exit(71)

    os.close(writer)
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            block = os.read(reader, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > _MAX_SOURCE_GUARD_BYTES:
                raise LeashContractError(
                    "controller source rederivation exceeded the guard byte limit"
                )
            chunks.append(block)
    finally:
        os.close(reader)
    _, status = os.waitpid(pid, 0)
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise LeashContractError("independent controller source rederivation failed")
    return b"".join(chunks)


def _write_authenticated_guard_capsule(
    capsule: Path, code: Mapping[str, bytes]
) -> Path:
    """Materialize only authenticated closure bytes inside a private capsule."""

    os.chmod(capsule, 0o700)
    worker: Path | None = None
    for relative, payload in sorted(code.items()):
        parts = PurePosixPath(relative).parts
        if not parts or ".." in parts:
            raise LeashContractError("unsafe LEASH source-guard capsule path")
        target = capsule.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(target, flags, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        rebound = read_bound_bytes(
            target,
            name=f"LEASH authenticated capsule {relative}",
            expected_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
        if rebound != payload:
            raise LeashContractError("LEASH capsule bytes changed after creation")
        if relative == "scripts/reconstruction_benchmark/leash_source_guard_worker.py":
            worker = target
    if worker is None:
        raise LeashContractError("authenticated LEASH capsule lacks its worker")
    return worker


def _isolated_source_preparation_contract(
    *, source_root: str | Path, registry_path: str | Path
) -> dict[str, Any]:
    """Run privileged raw rederivation outside the target-blind fit process."""

    expected_registry = load_registry(registry_path)
    repo = Path(__file__).resolve().parents[2]
    authenticated_code = read_authenticated_source_guard_code(repo, expected_registry)
    expected_bytes = _controller_source_preparation_bytes(
        source_root=source_root, registry_path=registry_path
    )
    with tempfile.TemporaryDirectory(
        prefix="leash-authenticated-source-guard-",
        dir=Path(tempfile.gettempdir()).resolve(strict=True),
    ) as temporary:
        capsule = Path(temporary)
        worker = _write_authenticated_guard_capsule(capsule, authenticated_code)
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                str(worker),
                "--source-root", str(Path(source_root).resolve(strict=True)),
                "--registry", str(Path(registry_path).resolve(strict=True)),
            ],
            cwd=capsule,
            check=False,
            capture_output=True,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        detail = stderr.splitlines()[-1] if stderr else "unknown guard failure"
        raise LeashContractError(f"isolated current-source guard failed: {detail}")
    if completed.stdout != expected_bytes:
        raise LeashContractError(
            "isolated source guard differs from independent controller rederivation"
        )
    try:
        decoded = completed.stdout.decode("utf-8")
        contract = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeashContractError(
            "isolated current-source guard returned invalid JSON bytes"
        ) from exc
    if not isinstance(contract, dict):
        raise LeashContractError("isolated current-source guard returned a non-object")
    if completed.stdout != canonical_json_bytes(contract) + b"\n":
        raise LeashContractError(
            "isolated current-source guard returned non-canonical JSON bytes"
        )
    expected_fields = {
        "registry", "fit_rows", "public_tree", "private_tree",
        "preparation_manifest", "policy_replay_audits", "certificate",
        "source_guard",
    }
    if set(contract) != expected_fields:
        raise LeashContractError("isolated current-source guard field roster drifted")
    guard = contract.get("source_guard")
    if guard != _source_guard_attestation(expected_registry):
        raise LeashContractError("isolated LEASH source guard attestation drifted")
    if contract.get("registry") != expected_registry:
        raise LeashContractError("isolated LEASH source guard registry binding drifted")
    fit_rows = contract.get("fit_rows")
    if not isinstance(fit_rows, list) or not fit_rows:
        raise LeashContractError("isolated LEASH source guard returned no fit rows")
    for row in fit_rows:
        if not isinstance(row, Mapping):
            raise LeashContractError("isolated LEASH source guard returned a non-object fit row")
        validate_fit_row(row)
    assert_no_forbidden_keys(fit_rows)

    manifest = contract.get("preparation_manifest")
    certificate = contract.get("certificate")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != PREPARATION_SCHEMA:
        raise LeashContractError("isolated LEASH preparation manifest schema drifted")
    if not isinstance(certificate, Mapping) or certificate.get("schema_version") != PREPARATION_AB_SCHEMA:
        raise LeashContractError("isolated LEASH preparation certificate schema drifted")
    verify_payload(manifest, name="isolated LEASH preparation manifest")
    verify_payload(certificate, name="isolated LEASH preparation certificate")
    registry_sha256 = bound_json_sha256(
        registry_path, expected_registry, name="LEASH registry"
    )
    row_order_sha256 = payload_sha256([row["row_id"] for row in fit_rows])
    if (
        manifest.get("lane_id") != expected_registry["lane_id"]
        or manifest.get("registry_sha256") != registry_sha256
        or manifest.get("n_rows") != len(fit_rows)
        or manifest.get("row_order_sha256") != row_order_sha256
        or certificate.get("lane_id") != expected_registry["lane_id"]
        or certificate.get("registry_sha256") != registry_sha256
        or certificate.get("n_rows") != len(fit_rows)
    ):
        raise LeashContractError("isolated LEASH source contract binding drifted")

    public_tree = contract.get("public_tree")
    private_tree = contract.get("private_tree")
    for name, tree in (("public", public_tree), ("private", private_tree)):
        if (
            not isinstance(tree, Mapping)
            or tree.get("schema_version") != "canonical-tree-manifest-v1"
            or not isinstance(tree.get("files"), list)
            or tree.get("tree_sha256") != payload_sha256(tree["files"])
        ):
            raise LeashContractError(f"isolated LEASH {name} tree manifest drifted")
    if (
        certificate.get("rederived_public_tree_sha256") != public_tree["tree_sha256"]
        or certificate.get("rederived_private_tree_sha256") != private_tree["tree_sha256"]
        or certificate.get("public_tree_sha256")
        != {"A": public_tree["tree_sha256"], "B": public_tree["tree_sha256"]}
        or certificate.get("private_tree_sha256")
        != {"A": private_tree["tree_sha256"], "B": private_tree["tree_sha256"]}
    ):
        raise LeashContractError("isolated LEASH source tree/certificate binding drifted")

    fit_sha256 = hashlib.sha256(canonical_jsonl_bytes(fit_rows)).hexdigest()
    public_files = {
        item.get("path"): item
        for item in public_tree["files"]
        if isinstance(item, Mapping)
    }
    fit_file = public_files.get(FIT_INPUT_FILENAME)
    if (
        not isinstance(fit_file, Mapping)
        or fit_file.get("sha256") != fit_sha256
        or manifest.get("files", {}).get(FIT_INPUT_FILENAME) != fit_sha256
    ):
        raise LeashContractError("isolated LEASH fit-row bytes/hash binding drifted")
    preparation_manifest_file = public_files.get(PREPARATION_MANIFEST_FILENAME)
    if (
        not isinstance(preparation_manifest_file, Mapping)
        or preparation_manifest_file.get("sha256")
        != hashlib.sha256(canonical_json_bytes(dict(manifest)) + b"\n").hexdigest()
    ):
        raise LeashContractError("isolated LEASH preparation manifest hash binding drifted")
    if contract.get("policy_replay_audits") != manifest.get("policy_replay_audits"):
        raise LeashContractError("isolated LEASH policy replay audit binding drifted")
    return contract


def load_verified_fit_input(
    preparation_dir: str | Path,
    *,
    registry: Mapping[str, Any],
    expected_tree: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(preparation_dir)
    expected_files = {item["path"]: item for item in expected_tree["files"]}
    expected_manifest_file = expected_files.get(PREPARATION_MANIFEST_FILENAME, {})
    manifest_bytes = read_bound_bytes(
        root / PREPARATION_MANIFEST_FILENAME,
        name="LEASH preparation manifest",
        expected_bytes=expected_manifest_file.get("bytes"),
        expected_sha256=expected_manifest_file.get("sha256"),
    )
    manifest = parse_json_bytes(manifest_bytes, name="LEASH preparation manifest")
    if manifest != expected_manifest:
        raise LeashContractError("LEASH preparation manifest changed after source verification")
    if manifest.get("schema_version") != PREPARATION_SCHEMA:
        raise LeashContractError("unexpected LEASH preparation manifest schema")
    verify_payload(manifest, name="LEASH preparation manifest")
    if (
        manifest.get("lane_id") != registry["lane_id"]
        or manifest.get("fidelity") != FIDELITY
        or manifest.get("fit_visible_targets") is not False
        or manifest.get("actual_policy_execution_observed") is not True
        or manifest.get("proxy_stopping") is not False
        or manifest.get("paper_exact_claim") is not False
        or manifest.get("conceptual_objective_reproduced_as_equation") is not False
    ):
        raise LeashContractError("LEASH preparation claim/firewall manifest drifted")
    fit_path = root / FIT_INPUT_FILENAME
    expected_fit_file = expected_files.get(FIT_INPUT_FILENAME, {})
    fit_bytes = read_bound_bytes(
        fit_path,
        name="LEASH fit input",
        expected_bytes=expected_fit_file.get("bytes"),
        expected_sha256=expected_fit_file.get("sha256"),
    )
    if hashlib.sha256(fit_bytes).hexdigest() != manifest.get("files", {}).get(FIT_INPUT_FILENAME):
        raise LeashContractError("LEASH fit input hash failed")
    rows = parse_jsonl_bytes(fit_bytes, name="LEASH fit input")
    if len(rows) != int(manifest.get("n_rows", -1)):
        raise LeashContractError("LEASH fit input row count drifted")
    if payload_sha256([row["row_id"] for row in rows]) != manifest.get("row_order_sha256"):
        raise LeashContractError("LEASH fit row order hash failed")
    for row in rows:
        validate_fit_row(row)
    assert_no_forbidden_keys(rows)
    return rows, manifest


def derive_policy_ledger(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate real callback/closure semantics without consulting any outcome."""

    if not rows:
        raise LeashContractError("cannot freeze an empty LEASH fit input")
    ledger: list[dict[str, Any]] = []
    per_cell_stops: Counter[str] = Counter()
    arm_counts: Counter[str] = Counter()
    seen: set[str] = set()
    by_cell_question: dict[tuple[str, str], set[str]] = defaultdict(set)
    for original in rows:
        row = dict(original)
        validate_fit_row(row)
        row_id = str(row["row_id"])
        if row_id in seen:
            raise LeashContractError(f"duplicate LEASH fit row {row_id}")
        seen.add(row_id)
        arm = str(row["arm"])
        stopped = bool(row["stopped_early"])
        closure = bool(row["closure_generated"])
        reason = row.get("stop_reason")
        if arm == "leash":
            if stopped and reason != "policy":
                raise LeashContractError(f"LEASH stopped row lacks policy reason: {row_id}")
            if not stopped and reason != "length":
                raise LeashContractError(f"LEASH non-stopped row lacks length reason: {row_id}")
            if stopped and not closure:
                raise LeashContractError(f"LEASH policy stop lacks realized closure: {row_id}")
            if stopped:
                per_cell_stops[str(row["cell_id"])] += 1
        elif arm == "cot":
            if stopped or reason != "length":
                raise LeashContractError(f"CoT row falsely presents a policy stop: {row_id}")
        elif arm == "nocot":
            if stopped or reason != "n/a":
                raise LeashContractError(f"noCoT row falsely presents a policy stop: {row_id}")
        else:  # validate_fit_row already rejects this, retained for fail-closed readability
            raise LeashContractError(f"unknown LEASH arm {arm!r}")
        if not closure:
            raise LeashContractError(f"evaluated answer stage was not generated: {row_id}")
        forced_closure = stopped and closure
        ledger_row = {
            **row,
            "forced_closure": forced_closure,
            "policy_event_verified": arm == "leash" and stopped and reason == "policy",
            "actual_stopping_claim_eligible": False,
            "proxy_stopping": False,
        }
        ledger.append(ledger_row)
        arm_counts[arm] += 1
        by_cell_question[(str(row["cell_id"]), str(row["question_id"]))].add(arm)

    if any(arms != {"cot", "leash", "nocot"} for arms in by_cell_question.values()):
        raise LeashContractError("policy ledger lost within-question arm pairing")
    ready_cells = {str(row["cell_id"]) for row in rows}
    missing_stop_cells = sorted(ready_cells - set(per_cell_stops))
    if missing_stop_cells:
        raise LeashContractError(
            f"actual stopping was not observed in ready cells: {missing_stop_cells}"
        )
    ledger.sort(key=lambda row: row["row_id"])
    audit = {
        "n_rows": len(ledger),
        "n_cells": len(ready_cells),
        "n_groups": len({row["group_id"] for row in ledger}),
        "arm_counts": dict(sorted(arm_counts.items())),
        "leash_policy_stops_by_cell": dict(sorted(per_cell_stops.items())),
        "all_policy_stops_have_realized_closure": True,
        "all_evaluated_rows_have_generated_answer_stage": True,
        "policy_execution_evaluated": True,
        "actual_stopping_claim_eligible_for_ready_leash_cells": False,
        "next_required_gate": "independent current-source FIT_AB verification",
        "proxy_stopping": False,
    }
    return ledger, audit


def _write_fit_tree(
    *,
    stage_path: AtomicLeashDirectory,
    registry: Mapping[str, Any],
    preparation_manifest: Mapping[str, Any],
    certificate: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> None:
    ledger_sha = leash_tree_write_bytes(
        stage_path, POLICY_LEDGER_FILENAME, canonical_jsonl_bytes(ledger)
    )
    manifest = add_payload_sha256(
        {
            "schema_version": FIT_SCHEMA,
            "lane_id": registry["lane_id"],
            "fidelity": FIDELITY,
            "execution_mode": "TARGET_BLIND_POLICY_EXECUTION_FREEZE",
            "fit_visible_targets": False,
            "policy_execution_evaluated": True,
            "actual_stopping_claim_eligible_for_ready_leash_cells": False,
            "next_required_gate": "independent current-source FIT_AB verification",
            "proxy_stopping": False,
            "paper_exact_claim": False,
            "conceptual_objective_reproduced_as_equation": False,
            "matched_accuracy_claim": False,
            "preparation_manifest_payload_sha256": preparation_manifest["payload_sha256"],
            "preparation_ab_certificate_sha256": hashlib.sha256(
                canonical_json_bytes(dict(certificate)) + b"\n"
            ).hexdigest(),
            "preparation_ab_payload_sha256": certificate["payload_sha256"],
            "n_rows": len(ledger),
            "row_order_sha256": payload_sha256([row["row_id"] for row in ledger]),
            "audit": dict(audit),
            "files": {POLICY_LEDGER_FILENAME: ledger_sha},
        }
    )
    leash_tree_write_json(stage_path, FIT_MANIFEST_FILENAME, manifest)


def derive_source_bound_fit_contract(
    *,
    source_root: str | Path,
    registry_path: str | Path,
    preparation_ab_certificate: str | Path,
) -> dict[str, Any]:
    """Rederive the exact fit tree/certificate from current raw sources."""

    preparation = _isolated_source_preparation_contract(
        source_root=source_root, registry_path=registry_path
    )
    if Path(preparation_ab_certificate).is_symlink():
        raise LeashContractError("LEASH preparation A/B certificate is a symlink")
    canonical_certificate = canonical_json_bytes(preparation["certificate"]) + b"\n"
    observed_certificate_bytes = read_bound_bytes(
        preparation_ab_certificate, name="LEASH preparation A/B certificate"
    )
    if observed_certificate_bytes != canonical_certificate:
        raise LeashContractError(
            "LEASH preparation certificate differs from exact current-source "
            "rederivation/canonical bytes"
        )
    ledger, audit = derive_policy_ledger(preparation["fit_rows"])
    with tempfile.TemporaryDirectory(
        prefix="leash-source-fit-contract-",
        dir=Path(tempfile.gettempdir()).resolve(strict=True),
    ) as temporary:
        fit_root = AtomicLeashDirectory(Path(temporary) / "fit")
        try:
            _write_fit_tree(
                stage_path=fit_root,
                registry=preparation["registry"],
                preparation_manifest=preparation["preparation_manifest"],
                certificate=preparation["certificate"],
                ledger=ledger,
                audit=audit,
            )
            fit_tree = leash_tree_manifest(fit_root)
            fit_manifest = leash_tree_load_json(
                fit_root,
                FIT_MANIFEST_FILENAME,
                name="source-rederived LEASH fit manifest",
            )
        finally:
            fit_root.cleanup()
    fit_certificate = add_payload_sha256(
        {
            "schema_version": FIT_AB_SCHEMA,
            "status": "PASS",
            "lane_id": preparation["registry"]["lane_id"],
            "registry_sha256": bound_json_sha256(
                registry_path, preparation["registry"], name="LEASH registry"
            ),
            "preparation_ab_certificate_sha256": hashlib.sha256(
                canonical_json_bytes(preparation["certificate"]) + b"\n"
            ).hexdigest(),
            "preparation_ab_payload_sha256": preparation["certificate"]["payload_sha256"],
            "fit_tree_sha256": {"A": fit_tree["tree_sha256"], "B": fit_tree["tree_sha256"]},
            "rederived_fit_tree_sha256": fit_tree["tree_sha256"],
            "n_rows": audit["n_rows"],
            "policy_execution_evaluated": True,
            "all_policy_stops_have_realized_closure": True,
            "actual_stopping_claim_eligible_after_fit_ab": True,
            "fit_visible_targets": False,
            "byte_identical": True,
            "transitive_rederivation": True,
        }
    )
    return {
        "registry": preparation["registry"],
        "preparation": preparation,
        "ledger": ledger,
        "audit": audit,
        "fit_tree": fit_tree,
        "fit_manifest": fit_manifest,
        "certificate": fit_certificate,
    }


def run_leash_fit(
    *,
    preparation_dir: str | Path,
    preparation_ab_certificate: str | Path,
    source_root: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish one target-blind policy ledger after a passing preparation A/B gate."""

    expected = derive_source_bound_fit_contract(
        source_root=source_root,
        registry_path=registry_path,
        preparation_ab_certificate=preparation_ab_certificate,
    )
    source_contract = expected["preparation"]
    assert_no_symlinks(preparation_dir, name="LEASH public preparation")
    if bound_tree_manifest(
        preparation_dir, name="LEASH public preparation"
    ) != source_contract["public_tree"]:
        raise LeashContractError("LEASH preparation differs from exact current-source rederivation")
    certificate = source_contract["certificate"]
    registry = source_contract["registry"]
    rows, preparation_manifest = load_verified_fit_input(
        preparation_dir,
        registry=registry,
        expected_tree=source_contract["public_tree"],
        expected_manifest=source_contract["preparation_manifest"],
    )
    if rows != source_contract["fit_rows"] or preparation_manifest != source_contract["preparation_manifest"]:
        raise LeashContractError("LEASH fit input changed after current-source verification")
    if bound_tree_manifest(
        preparation_dir, name="LEASH public preparation final binding"
    ) != source_contract["public_tree"]:
        raise LeashContractError("LEASH preparation changed during fit input loading")
    ledger, audit = derive_policy_ledger(rows)
    assert_no_forbidden_keys(ledger)
    stage = AtomicLeashDirectory(output_dir)
    try:
        _write_fit_tree(
            stage_path=stage,
            registry=registry,
            preparation_manifest=preparation_manifest,
            certificate=certificate,
            ledger=ledger,
            audit=audit,
        )
        tree = leash_tree_manifest(stage)
        if tree != expected["fit_tree"]:
            raise LeashContractError("LEASH fit output differs from current-source rederivation")
        stage.commit()
    finally:
        stage.cleanup()
    return {
        "status": "PASS",
        "output_dir": str(stage.final_path),
        "tree": tree,
        "audit": audit,
    }


__all__ = [
    "FIT_MANIFEST_FILENAME", "POLICY_LEDGER_FILENAME", "derive_policy_ledger",
    "derive_source_bound_fit_contract", "load_verified_fit_input", "run_leash_fit",
]
