#!/usr/bin/env python3
"""Prepare and execute the frozen A6-S0b shortcut/matching audit.

S0b reads only the public mechanically verified S0a quartet records and a
prompt-only Pythia model.  It never generates a response, extracts A6 telemetry,
opens a correctness sidecar, reads PopQA content, or opens an S1 seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VERSION = "automatic-group-free-iu-a6-s0b-v1-2026-08-15"
STATUS = "FROZEN_BEFORE_A6_S0B_EXECUTION"
DEFAULT_OUT = REPO / "results" / "automatic_group_free_phase_a6_s0b_v1"
DEFAULT_S0A = REPO / "results" / "automatic_group_free_phase_a6_s0a_v1"
EXECUTION_CONTRACT = (
    REPO / "docs" / "experiments"
    / "AUTOMATIC_GROUP_FREE_IU_PHASE_A6_S0_S1_EXECUTION_V1.md"
)
PARENT_PROTOCOL = (
    REPO / "docs" / "experiments" / "AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md"
)
STATIC_SOURCE_FILES = (
    "docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md",
    "docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_S0_S1_EXECUTION_V1.md",
    "scripts/automatic_group_free_phase_a6_s0a.py",
    "scripts/automatic_group_free_phase_a6_s0b.py",
    "scripts/test_a6_s0b.py",
    "scripts/test_a6_s0b_input.py",
    "scripts/test_automatic_group_free_phase_a6_s0b.py",
)
RUNTIME_PACKAGES = (
    "transformers", "tokenizers", "numpy", "scipy", "scikit-learn", "torch",
    "huggingface-hub", "safetensors",
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
)
BOUNDARY_REPORT_BYTES = (
    "# A6-S0b frozen source/runtime/Pythia boundary\n\n"
    "No response telemetry, natural response, correctness sidecar, PopQA "
    "content, or sealed S1 seed was opened. S0b has not run.\n"
).encode("utf-8")
# The unique sealed S0a artifact identity (HISTORY Step 268). Full prior
# verification requires byte identity with this tree, never merely a
# self-consistent substitute.
S0A_SEALED_BOUNDARY_SHA256 = \
    "698261d467a3f0a394ef244dafcac67d1cf8a69a9cf2de8888f0ff54678c545e"
S0A_SEALED_AGGREGATE_SHA256 = \
    "2a11b37c4fd649490675e8da4d826084c137a2a072c77ab2fdd5efcad8e8685a"


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


def source_files() -> tuple[str, ...]:
    package = tuple(
        str(path.relative_to(REPO))
        for path in sorted((REPO / "spectral_utils").glob("*.py"))
    )
    return STATIC_SOURCE_FILES + package


def _source_hashes() -> dict[str, str]:
    output = {}
    for name in source_files():
        path = REPO / name
        if not path.is_file():
            raise RuntimeError(f"A6-S0b source closure is missing: {name}")
        output[name] = sha256_file(path)
    return output


def _runtime_versions() -> dict[str, Any]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    versions["thread_environment"] = {
        name: os.environ.get(name) for name in THREAD_ENVIRONMENT
    }
    return versions


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_real_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} is missing") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(f"{label} must be a real directory")


def _require_real_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} is missing") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError(f"{label} must be a real regular file")


def _require_real_containment(root: Path, path: Path) -> None:
    _require_real_directory(root, "artifact output root")
    try:
        relative_parent = path.parent.relative_to(root)
    except ValueError as error:
        raise RuntimeError("artifact path escapes output root") from error
    current = root
    for component in relative_parent.parts:
        current = current / component
        if _lexists(current):
            _require_real_directory(current, "artifact parent")


def _exclusive_bytes(path: Path, data: bytes, *, root: Path) -> None:
    _require_real_containment(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_real_containment(root, path)
    temporary = path.with_name(path.name + ".tmp")
    if _lexists(path):
        _require_real_file(path, "immutable artifact")
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    if _lexists(temporary):
        _require_real_file(temporary, "interrupted artifact temporary")
        if temporary.read_bytes() != data:
            temporary.unlink()
        else:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
            return
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _exclusive_json(path: Path, value: Any, *, root: Path) -> None:
    _exclusive_bytes(path, canonical_json_bytes(value), root=root)


def _emit_json(path: Path, value: Any, *, root: Path, replay: bool) -> None:
    """Write once during execution; compare canonical bytes during verification."""
    data = canonical_json_bytes(value)
    if replay:
        _require_real_file(path, "required replay artifact")
        if path.read_bytes() != data:
            raise RuntimeError(f"A6-S0b replay mismatch: {path}")
        return
    _exclusive_bytes(path, data, root=root)


def _load_json(path: Path) -> Any:
    _require_real_file(path, "JSON artifact")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(value):
        raise RuntimeError(f"JSON artifact is noncanonical: {path}")
    return value


def _assert_known_output_paths(out: Path, boundary: dict[str, Any]) -> None:
    """Reject every unregistered S0b file, directory, symlink, or payload."""
    _require_real_directory(out, "A6-S0b output root")
    allowed_top_files = {
        "A6_S0B_BOUNDARY.json", "BOUNDARY_REPORT.md", "PYTHIA_OFFICIAL_TREE.json",
        "PYTHIA_NLL_COMPLETE.json", "SHORTCUT_TABLE_AUDIT.json",
        "SHORTCUT_OOF.json", "SHORTCUT_AUDIT.json", "MATCHING_GRAPH.json",
        "S0B_AGGREGATE.json", "S0B_COMPLETE.json", "S0B_CLOSED.json",
    }
    allowed_top_dirs = {"inputs", "checkpoints"}
    for path in out.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"A6-S0b output contains symlink: {path.name}")
        if path.is_file() and path.name not in allowed_top_files:
            raise RuntimeError(f"A6-S0b output contains unmanifested file: {path.name}")
        if path.is_dir() and path.name not in allowed_top_dirs:
            raise RuntimeError(f"A6-S0b output contains unmanifested directory: {path.name}")
        if not path.is_file() and not path.is_dir():
            raise RuntimeError(f"A6-S0b output contains non-regular path: {path.name}")
    snapshot = out / boundary["pythia_input"]["relative_directory"]
    if (out / "inputs").exists():
        _require_real_directory(out / "inputs", "A6-S0b inputs root")
        if {path for path in (out / "inputs").iterdir()} != {snapshot}:
            raise RuntimeError("A6-S0b inputs root contains unmanifested paths")
        _require_real_directory(snapshot, "A6-S0b Pythia snapshot")
        expected_files = {row["path"] for row in boundary["pythia_input"]["files"]}
        if {path.name for path in snapshot.iterdir()} != expected_files:
            raise RuntimeError("A6-S0b Pythia snapshot namespace changed")
        for path in snapshot.iterdir():
            _require_real_file(path, "A6-S0b Pythia input file")
    checkpoint_root = out / "checkpoints"
    if not checkpoint_root.exists():
        return
    _require_real_directory(checkpoint_root, "A6-S0b checkpoint root")
    if {path.name for path in checkpoint_root.iterdir()} - {"pythia", "bootstrap", "control"}:
        raise RuntimeError("A6-S0b checkpoint root contains an unknown family")
    for family in checkpoint_root.iterdir():
        _require_real_directory(family, "A6-S0b checkpoint family")
        if family.name == "pythia":
            allowed = {f"{index:05d}.json" for index in range(14_400)}
            children = list(family.iterdir())
            if any(path.name not in allowed for path in children):
                raise RuntimeError("A6-S0b Pythia checkpoint namespace changed")
            for path in children:
                _require_real_file(path, "A6-S0b Pythia checkpoint")
        elif family.name == "bootstrap":
            if {path.name for path in family.iterdir()} - {"qwen-source", "llama-audit"}:
                raise RuntimeError("A6-S0b bootstrap population namespace changed")
            for population in family.iterdir():
                _require_real_directory(population, "A6-S0b bootstrap population")
                allowed = {f"{index:02d}.json" for index in range(19)}
                if any(path.name not in allowed for path in population.iterdir()):
                    raise RuntimeError("A6-S0b bootstrap checkpoint namespace changed")
                for path in population.iterdir():
                    _require_real_file(path, "A6-S0b bootstrap checkpoint")
        else:
            if {path.name for path in family.iterdir()} - {"2", "3"}:
                raise RuntimeError("A6-S0b control family namespace changed")
            for control in family.iterdir():
                _require_real_directory(control, "A6-S0b control family")
                allowed = {f"{index:03d}.json" for index in range(200)}
                if any(path.name not in allowed for path in control.iterdir()):
                    raise RuntimeError("A6-S0b control checkpoint namespace changed")
                for path in control.iterdir():
                    _require_real_file(path, "A6-S0b control checkpoint")


def _load_input_spec_stdlib():
    path = REPO / "spectral_utils" / "a6_s0b_input.py"
    spec = importlib.util.spec_from_file_location("_a6_s0b_input_boundary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load A6-S0b input verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prior_s0a_provenance(prior: Path, *, full_verify: bool) -> dict[str, Any]:
    """Authenticate the sealed S0a artifact tree.

    ``full_verify`` authenticates every byte S0b will consume against the
    S0a-recorded manifest chain: completion pins the boundary and aggregate
    hashes, the aggregate pins all 7,800 checkpoints and the four result
    files by path/size/SHA-256.  It deliberately does NOT call
    ``s0a.verify(replay=True)``: the S0a freeze is environment-locked (exact
    ba983aa source tree, git HEAD, and macOS runtime recorded in its
    boundary), so its own authoritative replay can only ever pass in the
    original frozen session — where it did pass, `PASS_S0A_VERIFIED`,
    Step 268.  A later stage on a later commit or another machine can verify
    exactly what the sealed artifacts themselves prove, and nothing less.
    """
    _require_real_directory(prior, "prior A6-S0a root")
    required = ("A6_S0A_BOUNDARY.json", "S0A_AGGREGATE.json", "S0A_COMPLETE.json")
    for name in required:
        _require_real_file(prior / name, f"prior {name}")
    completion = _load_json(prior / "S0A_COMPLETE.json")
    if completion.get("verdict") != "PASS_S0A":
        raise RuntimeError("prior A6-S0a completion verdict changed")
    if full_verify:
        # S0a is sealed and unique, so full verification demands identity with
        # the Step-268 artifact set, not merely a self-consistent tree.
        if sha256_file(prior / "A6_S0A_BOUNDARY.json") != S0A_SEALED_BOUNDARY_SHA256 \
                or sha256_file(prior / "S0A_AGGREGATE.json") != S0A_SEALED_AGGREGATE_SHA256:
            raise RuntimeError("prior A6-S0a tree is not the sealed Step-268 artifact set")
        if completion.get("boundary_sha256") != S0A_SEALED_BOUNDARY_SHA256 \
                or completion.get("aggregate_sha256") != S0A_SEALED_AGGREGATE_SHA256:
            raise RuntimeError("prior A6-S0a completion hashes changed")
        aggregate = _load_json(prior / "S0A_AGGREGATE.json")
        if aggregate.get("verdict") != "PASS_S0A" \
                or aggregate.get("boundary_sha256") != completion["boundary_sha256"] \
                or aggregate.get("n_quartets") != 1_800 \
                or aggregate.get("n_natural_prompts") != 6_000 \
                or aggregate.get("n_inner_fold_assignments") != 7_200 \
                or aggregate.get("n_null_cells") != 36 \
                or aggregate.get("response_telemetry_accessed") is not False \
                or aggregate.get("natural_response_accessed") is not False \
                or aggregate.get("correctness_sidecar_created") is not False \
                or aggregate.get("popqa_content_accessed") is not False \
                or aggregate.get("sealed_s1_seed_opened") is not False:
            raise RuntimeError("prior A6-S0a aggregate provenance changed")
        result_hashes = aggregate.get("result_file_sha256")
        if not isinstance(result_hashes, dict) \
                or {"INNER_FOLDS.json", "NULL_STRATA.json"} - set(result_hashes):
            raise RuntimeError("prior A6-S0a result-file manifest changed")
        for name, expected_sha in sorted(result_hashes.items()):
            if not isinstance(name, str) or "/" in name or "\\" in name or name in {".", ".."}:
                raise RuntimeError("prior A6-S0a result-file path invalid")
            path = prior / name
            _require_real_file(path, f"prior A6-S0a result file {name}")
            if sha256_file(path) != expected_sha:
                raise RuntimeError(f"prior A6-S0a result file changed: {name}")
        manifest = aggregate.get("checkpoint_manifest")
        if not isinstance(manifest, list) or len(manifest) != 7_800:
            raise RuntimeError("prior A6-S0a checkpoint manifest changed")
        seen_paths = set()
        for row in manifest:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise RuntimeError("prior A6-S0a checkpoint manifest row invalid")
            pure = PurePosixPath(row["path"])
            if pure.is_absolute() or ".." in pure.parts or "\\" in row["path"] \
                    or row["path"] in seen_paths \
                    or pure.parts[:1] != ("checkpoints",):
                raise RuntimeError("prior A6-S0a checkpoint path invalid")
            seen_paths.add(row["path"])
            path = prior.joinpath(*pure.parts)
            _require_real_file(path, "prior A6-S0a checkpoint")
            if path.stat().st_size != row.get("size") \
                    or sha256_file(path) != row.get("sha256"):
                raise RuntimeError(f"prior A6-S0a checkpoint changed: {row['path']}")
    return {
        "relative_path": os.path.relpath(prior, REPO),
        "boundary_sha256": sha256_file(prior / "A6_S0A_BOUNDARY.json"),
        "aggregate_sha256": sha256_file(prior / "S0A_AGGREGATE.json"),
        "completion_sha256": sha256_file(prior / "S0A_COMPLETE.json"),
    }


def _prepare_pythia_input(
    out: Path, source: Path, official_tree_path: Path,
) -> dict[str, Any]:
    input_spec = _load_input_spec_stdlib()
    _require_real_directory(source, "Pythia source root")
    _require_real_file(official_tree_path, "Pythia official-tree evidence")
    official_raw = official_tree_path.read_bytes()
    projection = input_spec.validate_official_tree(official_raw)
    file_rows = []
    payloads: dict[str, bytes] = {}
    for item in input_spec.SELECTED_FILES:
        path = source / item.path
        _require_real_file(path, f"Pythia source {item.path}")
        payload = path.read_bytes()
        file_rows.append(input_spec.verify_selected_bytes(item, payload))
        payloads[item.path] = payload
    content_sha = hashlib.sha256(canonical_json_bytes(file_rows)).hexdigest()
    relative_root = f"inputs/pythia-{content_sha}"
    destination = out / relative_root
    destination.mkdir(parents=True, exist_ok=True)
    for item in input_spec.SELECTED_FILES:
        _exclusive_bytes(destination / item.path, payloads[item.path], root=out)
    evidence_path = out / "PYTHIA_OFFICIAL_TREE.json"
    _exclusive_bytes(evidence_path, official_raw, root=out)
    return {
        "schema_version": input_spec.SCHEMA_VERSION,
        "repository": input_spec.REPOSITORY,
        "revision": input_spec.REVISION,
        "official_api_url": input_spec.OFFICIAL_API_URL,
        "official_raw_sha256": hashlib.sha256(official_raw).hexdigest(),
        "official_projection_sha256": hashlib.sha256(
            canonical_json_bytes(projection)
        ).hexdigest(),
        "relative_directory": relative_root,
        "content_sha256": content_sha,
        "files": file_rows,
    }


def _configure_numerical_runtime() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"


def _load_pythia(snapshot: Path):
    _configure_numerical_runtime()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float32,
        device_map=None, low_cpu_mem_usage=False,
    )
    model.to("cpu")
    model.eval()
    if next(model.parameters()).dtype != torch.float32 \
            or next(model.parameters()).device.type != "cpu" \
            or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 \
            or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Pythia numerical runtime contract is invalid")
    return model, tokenizer


def _pythia_runtime_audit(model: Any, tokenizer: Any) -> dict[str, Any]:
    import torch
    return {
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
    }


def prepare(
    out: str | Path, *, pythia_source: str | Path,
    pythia_official_tree: str | Path, prior_s0a: str | Path = DEFAULT_S0A,
) -> dict[str, Any]:
    _configure_numerical_runtime()
    out = Path(out)
    if _lexists(out):
        _require_real_directory(out, "A6-S0b output root")
        allowed_partial = {"inputs", "PYTHIA_OFFICIAL_TREE.json"}
        unexpected = {path.name for path in out.iterdir()} - allowed_partial
        if unexpected:
            raise RuntimeError(f"A6-S0b prepare found unregistered partial paths: {sorted(unexpected)}")
    else:
        _require_real_directory(out.parent, "A6-S0b output parent")
        out.mkdir(parents=False, exist_ok=False)
    _require_real_directory(out, "A6-S0b output root")
    pythia = _prepare_pythia_input(
        out, Path(pythia_source), Path(pythia_official_tree),
    )
    # Only after all Pythia bytes authenticate may package/model imports occur.
    prior = _prior_s0a_provenance(Path(prior_s0a), full_verify=True)
    snapshot = out / pythia["relative_directory"]
    model, tokenizer = _load_pythia(snapshot)
    model_audit = _pythia_runtime_audit(model, tokenizer)
    del model, tokenizer
    boundary = {
        "version": VERSION, "status": STATUS,
        "execution_contract_sha256": sha256_file(EXECUTION_CONTRACT),
        "parent_protocol_sha256": sha256_file(PARENT_PROTOCOL),
        "source_sha256": _source_hashes(),
        "runtime_versions": _runtime_versions(),
        "git_head": _git_head(),
        "prior_s0a": prior,
        "pythia_input": pythia,
        "pythia_runtime_audit": model_audit,
        "configuration": {
            "pythia_prompt_checkpoint_count": 14_400,
            "shortcut_bootstrap_draws": 20_000,
            "control_draws_per_family": 200,
            "response_telemetry_accessed": False,
            "natural_response_accessed": False,
            "correctness_sidecar_created": False,
            "popqa_content_accessed": False,
            "sealed_s1_seed_opened": False,
        },
    }
    _exclusive_json(out / "A6_S0B_BOUNDARY.json", boundary, root=out)
    _exclusive_bytes(out / "BOUNDARY_REPORT.md", BOUNDARY_REPORT_BYTES, root=out)
    _assert_known_output_paths(out, boundary)
    return boundary


def _verify_pythia_input(out: Path, row: dict[str, Any]) -> Path:
    input_spec = _load_input_spec_stdlib()
    expected_keys = {
        "schema_version", "repository", "revision", "official_api_url",
        "official_raw_sha256", "official_projection_sha256",
        "relative_directory", "content_sha256", "files",
    }
    if set(row) != expected_keys or row["schema_version"] != input_spec.SCHEMA_VERSION \
            or row["repository"] != input_spec.REPOSITORY \
            or row["revision"] != input_spec.REVISION \
            or row["official_api_url"] != input_spec.OFFICIAL_API_URL:
        raise RuntimeError("A6-S0b Pythia input schema/identity mismatch")
    evidence = out / "PYTHIA_OFFICIAL_TREE.json"
    _require_real_file(evidence, "Pythia official-tree evidence")
    raw = evidence.read_bytes()
    projection = input_spec.validate_official_tree(raw)
    if hashlib.sha256(raw).hexdigest() != row["official_raw_sha256"] \
            or hashlib.sha256(canonical_json_bytes(projection)).hexdigest() \
            != row["official_projection_sha256"]:
        raise RuntimeError("A6-S0b Pythia official-tree evidence changed")
    snapshot = out / row["relative_directory"]
    _require_real_directory(snapshot, "Pythia materialized snapshot")
    expected_names = {item.path for item in input_spec.SELECTED_FILES}
    if {path.name for path in snapshot.iterdir()} != expected_names:
        raise RuntimeError("A6-S0b Pythia snapshot has extra/missing paths")
    file_rows = []
    for item in input_spec.SELECTED_FILES:
        path = snapshot / item.path
        _require_real_file(path, f"Pythia materialized {item.path}")
        file_rows.append(input_spec.verify_selected_bytes(item, path.read_bytes()))
    content_sha = hashlib.sha256(canonical_json_bytes(file_rows)).hexdigest()
    if file_rows != row["files"] or content_sha != row["content_sha256"] \
            or row["relative_directory"] != f"inputs/pythia-{content_sha}":
        raise RuntimeError("A6-S0b Pythia materialized manifest changed")
    return snapshot


def load_and_verify_boundary(
    out: str | Path, *, load_model: bool = False, verify_prior: bool = True,
):
    _configure_numerical_runtime()
    out = Path(out)
    _require_real_directory(out, "A6-S0b output root")
    boundary = _load_json(out / "A6_S0B_BOUNDARY.json")
    if set(boundary) != {
        "version", "status", "execution_contract_sha256", "parent_protocol_sha256",
        "source_sha256", "runtime_versions", "git_head", "prior_s0a",
        "pythia_input", "pythia_runtime_audit", "configuration",
    } or boundary["version"] != VERSION or boundary["status"] != STATUS:
        raise RuntimeError("A6-S0b boundary schema/status mismatch")
    if boundary["execution_contract_sha256"] != sha256_file(EXECUTION_CONTRACT) \
            or boundary["parent_protocol_sha256"] != sha256_file(PARENT_PROTOCOL):
        raise RuntimeError("A6-S0b protocol changed after freeze")
    if boundary["source_sha256"] != _source_hashes() \
            or boundary["runtime_versions"] != _runtime_versions() \
            or boundary["git_head"] != _git_head():
        raise RuntimeError("A6-S0b source/runtime/commit changed after freeze")
    report = out / "BOUNDARY_REPORT.md"
    _require_real_file(report, "A6-S0b boundary report")
    if report.read_bytes() != BOUNDARY_REPORT_BYTES:
        raise RuntimeError("A6-S0b boundary report changed")
    prior_path = REPO / boundary["prior_s0a"]["relative_path"]
    if _prior_s0a_provenance(prior_path, full_verify=verify_prior) != boundary["prior_s0a"]:
        raise RuntimeError("A6-S0b prior S0a provenance changed")
    snapshot = _verify_pythia_input(out, boundary["pythia_input"])
    _assert_known_output_paths(out, boundary)
    if not load_model:
        return boundary, None, None
    model, tokenizer = _load_pythia(snapshot)
    if _pythia_runtime_audit(model, tokenizer) != boundary["pythia_runtime_audit"]:
        raise RuntimeError("A6-S0b Pythia runtime audit changed")
    return boundary, model, tokenizer


def _load_quartet_payloads(prior_s0a: Path) -> list[dict[str, Any]]:
    checkpoint_root = prior_s0a / "checkpoints" / "quartet"
    _require_real_directory(checkpoint_root, "prior S0a quartet checkpoints")
    payloads = []
    for path in sorted(checkpoint_root.iterdir(), key=lambda value: value.name.encode("utf-8")):
        _require_real_file(path, "prior S0a quartet checkpoint")
        wrapper = _load_json(path)
        payload = wrapper.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError("prior S0a quartet payload is invalid")
        payloads.append(payload)
    if len(payloads) != 1_800:
        raise RuntimeError("prior S0a quartet checkpoint count changed")
    return payloads


def _prompt_schedule(payloads: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    prompts: dict[str, str] = {}
    for payload in payloads:
        group = payload["group"]
        context = payload["contextual_evidence"]
        if set(context) != {"qwen3-4b", "qwen3-8b", "llama31-8b"}:
            raise RuntimeError("S0a contextual scorer roster changed")
        for world, prompt_rows in (("A", group["prompts_a"]), ("B", group["prompts_b"])):
            for render_index, prompt in enumerate(prompt_rows):
                base = (0 if world == "A" else 4) + render_index
                hashes = set()
                for scorer_rows in context.values():
                    # Response A and response B must expose the same prompt hash.
                    first = scorer_rows[base]["prompt_sha256"]
                    second = scorer_rows[base + 8]["prompt_sha256"]
                    if first != second:
                        raise RuntimeError("S0a prompt hash depends on response world")
                    hashes.add(first)
                if len(hashes) != 1:
                    raise RuntimeError("S0a prompt hash differs across scorers")
                prompt_hash = next(iter(hashes))
                if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_hash:
                    raise RuntimeError("S0a prompt content hash changed")
                previous = prompts.setdefault(prompt_hash, prompt)
                if previous != prompt:
                    raise RuntimeError("S0a prompt hash collision")
    if len(prompts) != 14_400:
        raise RuntimeError("A6-S0b Pythia prompt schedule must contain 14,400 prompts")
    return tuple(sorted(prompts.items(), key=lambda item: item[0].encode("utf-8")))


def run_pythia(
    out: str | Path, *, verify_prior: bool = True,
) -> dict[str, Any]:
    out = Path(out)
    boundary, model, tokenizer = load_and_verify_boundary(
        out, load_model=True, verify_prior=verify_prior,
    )
    prior = REPO / boundary["prior_s0a"]["relative_path"]
    schedule = _prompt_schedule(_load_quartet_payloads(prior))
    boundary_sha = sha256_file(out / "A6_S0B_BOUNDARY.json")
    checkpoint_root = out / "checkpoints" / "pythia"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    from spectral_utils.a6_s0b import pythia_prompt_mean_nll

    for index, (prompt_sha, prompt) in enumerate(schedule):
        path = checkpoint_root / f"{index:05d}.json"
        base = {
            "version": VERSION, "boundary_sha256": boundary_sha,
            "index": index, "prompt_sha256": prompt_sha,
        }
        if path.exists():
            value = _load_json(path)
            if set(value) != set(base) | {"mean_nll"} \
                    or any(value[key] != expected for key, expected in base.items()) \
                    or not isinstance(value["mean_nll"], float) \
                    or not math.isfinite(value["mean_nll"]):
                raise RuntimeError(f"A6-S0b Pythia checkpoint mismatch: {path}")
            continue
        value = base | {"mean_nll": float(pythia_prompt_mean_nll(model, tokenizer, prompt))}
        _exclusive_json(path, value, root=out)
    manifest = [
        {"path": path.relative_to(out).as_posix(), "size": path.stat().st_size,
         "sha256": sha256_file(path)}
        for path in sorted(checkpoint_root.iterdir(), key=lambda value: value.name.encode("utf-8"))
    ]
    if len(manifest) != 14_400:
        raise RuntimeError("A6-S0b Pythia checkpoint schedule is incomplete")
    aggregate = {
        "version": VERSION, "status": "PYTHIA_PROMPT_NLL_COMPLETE",
        "boundary_sha256": boundary_sha, "checkpoint_manifest": manifest,
        "schedule_sha256": hashlib.sha256(canonical_json_bytes(schedule)).hexdigest(),
    }
    _exclusive_json(out / "PYTHIA_NLL_COMPLETE.json", aggregate, root=out)
    _assert_known_output_paths(out, boundary)
    return aggregate


def _load_pythia_nll(
    out: Path, boundary: dict[str, Any], schedule: tuple[tuple[str, str], ...],
) -> dict[str, float]:
    completion = _load_json(out / "PYTHIA_NLL_COMPLETE.json")
    boundary_sha = sha256_file(out / "A6_S0B_BOUNDARY.json")
    if completion.get("version") != VERSION \
            or completion.get("status") != "PYTHIA_PROMPT_NLL_COMPLETE" \
            or completion.get("boundary_sha256") != boundary_sha \
            or completion.get("schedule_sha256") != hashlib.sha256(
                canonical_json_bytes(schedule)
            ).hexdigest():
        raise RuntimeError("A6-S0b Pythia completion provenance mismatch")
    checkpoint_root = out / "checkpoints" / "pythia"
    _require_real_directory(checkpoint_root, "Pythia checkpoint root")
    paths = sorted(checkpoint_root.iterdir(), key=lambda value: value.name.encode("utf-8"))
    if [path.name for path in paths] != [f"{index:05d}.json" for index in range(14_400)]:
        raise RuntimeError("A6-S0b Pythia checkpoint namespace changed")
    manifest = []
    values: dict[str, float] = {}
    for index, ((prompt_sha, _), path) in enumerate(zip(schedule, paths)):
        _require_real_file(path, "Pythia checkpoint")
        row = _load_json(path)
        if set(row) != {
            "version", "boundary_sha256", "index", "prompt_sha256", "mean_nll",
        } or row["version"] != VERSION or row["boundary_sha256"] != boundary_sha \
                or row["index"] != index or row["prompt_sha256"] != prompt_sha \
                or not isinstance(row["mean_nll"], float) \
                or not math.isfinite(row["mean_nll"]):
            raise RuntimeError("A6-S0b Pythia checkpoint changed")
        values[prompt_sha] = row["mean_nll"]
        manifest.append({
            "path": path.relative_to(out).as_posix(), "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if completion.get("checkpoint_manifest") != manifest or len(values) != 14_400:
        raise RuntimeError("A6-S0b Pythia checkpoint manifest changed")
    return values


def _recompute_pythia_nll(
    out: Path, boundary: dict[str, Any], model: Any, tokenizer: Any,
    schedule: tuple[tuple[str, str], ...],
) -> str:
    """Authoritatively replay every prompt score without writing or repairing."""
    stored = _load_pythia_nll(out, boundary, schedule)
    from spectral_utils.a6_s0b import pythia_prompt_mean_nll

    replay_rows = []
    for prompt_sha, prompt in schedule:
        observed = float(pythia_prompt_mean_nll(model, tokenizer, prompt))
        if observed != stored[prompt_sha]:
            raise RuntimeError(f"A6-S0b Pythia semantic replay mismatch: {prompt_sha}")
        replay_rows.append([prompt_sha, observed])
    return hashlib.sha256(canonical_json_bytes(replay_rows)).hexdigest()


def _serialize_oof_bundle(bundle: Any) -> dict[str, Any]:
    return {
        "population_id": bundle.population_id,
        "ridge": bundle.ridge,
        "scores": list(bundle.scores),
        "fold_auc": list(bundle.fold_auc),
        "macro_auc": bundle.macro_auc,
        "fits": [asdict(fit) for fit in bundle.fits],
    }


def _bootstrap_checkpoint(
    out: Path, population: str, gate_index: int, result: Any,
    boundary_sha: str, *, replay: bool,
) -> dict[str, Any]:
    draws = list(result.bootstrap_max_macro_auc)
    payload = {
        "version": VERSION, "boundary_sha256": boundary_sha,
        "population_id": population, "gate_name": result.gate_name,
        "observed_max_macro_auc": result.observed_max_macro_auc,
        "selected_ridge": result.selected_ridge, "upper_95": result.upper_95,
        "gate_pass": result.gate_pass,
        "draw_count": len(draws),
        "draw_sha256": hashlib.sha256(canonical_json_bytes(draws)).hexdigest(),
        "draw_min": min(draws), "draw_max": max(draws),
        "draw_unique_count": len(set(draws)),
        "draws": draws,
    }
    path = out / "checkpoints" / "bootstrap" / population / f"{gate_index:02d}.json"
    _emit_json(path, payload, root=out, replay=replay)
    return payload


def _load_bootstrap_checkpoint(
    out: Path, population: str, gate_index: int, gate_name: str,
    rows: tuple[Any, ...], bundles: tuple[Any, ...], boundary_sha: str,
) -> dict[str, Any]:
    """Validate a completed immutable draw without regenerating 20k samples."""
    from spectral_utils import a6_s0b as core

    path = out / "checkpoints" / "bootstrap" / population / f"{gate_index:02d}.json"
    value = _load_json(path)
    expected_keys = {
        "version", "boundary_sha256", "population_id", "gate_name",
        "observed_max_macro_auc", "selected_ridge", "upper_95", "gate_pass",
        "draw_count", "draw_sha256", "draw_min", "draw_max",
        "draw_unique_count", "draws",
    }
    if set(value) != expected_keys or value["version"] != VERSION \
            or value["boundary_sha256"] != boundary_sha \
            or value["population_id"] != population or value["gate_name"] != gate_name:
        raise RuntimeError(f"A6-S0b bootstrap checkpoint identity changed: {path}")
    draws = value["draws"]
    if not isinstance(draws, list) or len(draws) != 20_000 \
            or any(not isinstance(item, float) or not math.isfinite(item) for item in draws):
        raise RuntimeError(f"A6-S0b bootstrap draw vector changed: {path}")
    observed = [
        (core.gate_macro_auc(rows, bundle.scores, gate_name), bundle.ridge)
        for bundle in bundles
    ]
    observed_max, selected_ridge = max(observed, key=lambda item: (item[0], item[1]))
    # Mirror of the core order statistic: method="higher" at 0.975.
    upper = sorted(draws)[math.ceil(0.975 * (len(draws) - 1))]
    expected = {
        "observed_max_macro_auc": observed_max,
        "selected_ridge": selected_ridge,
        "upper_95": upper,
        "gate_pass": upper <= 0.60,
        "draw_count": len(draws),
        "draw_sha256": hashlib.sha256(canonical_json_bytes(draws)).hexdigest(),
        "draw_min": min(draws),
        "draw_max": max(draws),
        "draw_unique_count": len(set(draws)),
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"A6-S0b bootstrap checkpoint summary changed: {path}")
    return value


def _index_schedule_payload(schedule: Any, group_ids: tuple[str, ...]) -> list[Any]:
    index = {group_id: position for position, group_id in enumerate(group_ids)}
    return [
        [partition_id, [[index[left], index[right]] for left, right in mapping]]
        for partition_id, mapping in schedule.assignments
    ]


def _schedule_checkpoint(
    out: Path, schedule: Any, group_ids: tuple[str, ...], boundary_sha: str,
    *, replay: bool,
) -> dict[str, Any]:
    assignment_indices = _index_schedule_payload(schedule, group_ids)
    payload = {
        "version": VERSION, "boundary_sha256": boundary_sha,
        "family": schedule.family, "draw": schedule.draw,
        "seed_u64": schedule.seed_u64,
        "schedule_sha256": schedule.schedule_sha256,
        "outer_held_sha256": schedule.outer_held_sha256,
        "assignments": assignment_indices,
    }
    path = out / "checkpoints" / "control" / str(schedule.family) \
        / f"{schedule.draw:03d}.json"
    _emit_json(path, payload, root=out, replay=replay)
    return payload


def _load_schedule_checkpoint(
    out: Path, family: int, draw: int, group_ids: tuple[str, ...],
    partitions: tuple[tuple[str, tuple[str, ...]], ...], records: tuple[Any, ...],
    eligible_edges: tuple[tuple[str, str], ...], boundary_sha: str,
) -> dict[str, Any]:
    """Validate a frozen control bijection structurally without rerunning Hungarian."""
    from spectral_utils import a6_s0b as core

    path = out / "checkpoints" / "control" / str(family) / f"{draw:03d}.json"
    value = _load_json(path)
    if set(value) != {
        "version", "boundary_sha256", "family", "draw", "seed_u64",
        "schedule_sha256", "outer_held_sha256", "assignments",
    } or value["version"] != VERSION or value["boundary_sha256"] != boundary_sha \
            or value["family"] != family or value["draw"] != draw:
        raise RuntimeError(f"A6-S0b control checkpoint identity changed: {path}")
    seed_u64, _ = core.control_seed(family, draw)
    if value["seed_u64"] != seed_u64 or not isinstance(value["assignments"], list):
        raise RuntimeError(f"A6-S0b control checkpoint seed changed: {path}")
    index = {position: group_id for position, group_id in enumerate(group_ids)}
    partition_map = dict(partitions)
    stratum = {record.group_id: record.null_stratum_id for record in records}
    eligible = set(eligible_edges)
    decoded = []
    for row in value["assignments"]:
        if not isinstance(row, list) or len(row) != 2 or row[0] not in partition_map \
                or not isinstance(row[1], list):
            raise RuntimeError(f"A6-S0b control assignment schema changed: {path}")
        pairs = []
        for pair in row[1]:
            if not isinstance(pair, list) or len(pair) != 2 \
                    or any(type(item) is not int or item not in index for item in pair):
                raise RuntimeError(f"A6-S0b control assignment index changed: {path}")
            pairs.append((index[pair[0]], index[pair[1]]))
        roster = set(partition_map[row[0]])
        if {left for left, _ in pairs} != roster or {right for _, right in pairs} != roster \
                or any(left == right for left, right in pairs):
            raise RuntimeError(f"A6-S0b control assignment is not a derangement: {path}")
        if family == 2 and any(stratum[left] != stratum[right] for left, right in pairs):
            raise RuntimeError(f"A6-S0b Control-2 crossed a frozen stratum: {path}")
        if family == 3 and any((left, right) not in eligible for left, right in pairs):
            raise RuntimeError(f"A6-S0b Control-3 crossed an ineligible edge: {path}")
        decoded.append((row[0], tuple(pairs)))
    if tuple(partition_id for partition_id, _ in decoded) != tuple(
        partition_id for partition_id, _ in partitions
    ):
        raise RuntimeError(f"A6-S0b control partition order changed: {path}")
    schedule_bytes = canonical_json_bytes([
        [partition_id, [list(pair) for pair in mapping]]
        for partition_id, mapping in decoded
    ])
    outer_bytes = canonical_json_bytes([
        [partition_id, [list(pair) for pair in mapping]]
        for partition_id, mapping in decoded if partition_id.endswith(":held")
    ])
    if value["schedule_sha256"] != hashlib.sha256(schedule_bytes).hexdigest() \
            or value["outer_held_sha256"] != hashlib.sha256(outer_bytes).hexdigest():
        raise RuntimeError(f"A6-S0b control schedule hash changed: {path}")
    return value


def run_analysis(
    out: str | Path, *, verify_prior: bool = True, replay: bool = False,
) -> dict[str, Any]:
    """Run the fixed shortcut audit and, only after PASS, freeze matching schedules."""
    out = Path(out)
    boundary, _, _ = load_and_verify_boundary(
        out, load_model=False, verify_prior=verify_prior,
    )
    if not replay and (
        (out / "S0B_COMPLETE.json").exists() or (out / "S0B_CLOSED.json").exists()
    ):
        raise RuntimeError("A6-S0b already has an immutable terminal artifact")
    prior = REPO / boundary["prior_s0a"]["relative_path"]
    payloads = _load_quartet_payloads(prior)
    schedule = _prompt_schedule(payloads)
    nll = _load_pythia_nll(out, boundary, schedule)
    boundary_sha = sha256_file(out / "A6_S0B_BOUNDARY.json")
    from spectral_utils import a6_s0b as core

    rows = core.build_shortcut_rows(payloads, nll)
    qwen_rows = tuple(row for row in rows if row.population_id == "qwen-source")
    llama_rows = tuple(row for row in rows if row.population_id == "llama-audit")
    vocabulary = core.freeze_vocabulary(qwen_rows)
    table_audit = {
        "version": VERSION, "boundary_sha256": boundary_sha,
        "continuous_columns": list(core.CONTINUOUS_COLUMNS),
        "categorical_columns": list(core.CATEGORICAL_COLUMNS),
        "vocabulary": [[name, list(values)] for name, values in vocabulary.values],
        "population_rows": {"qwen-source": len(qwen_rows), "llama-audit": len(llama_rows)},
        "marginal_prevalence": {
            "qwen-source": core.marginal_prevalence_audit(qwen_rows),
            "llama-audit": core.marginal_prevalence_audit(llama_rows),
        },
    }
    _emit_json(
        out / "SHORTCUT_TABLE_AUDIT.json", table_audit, root=out, replay=replay,
    )
    try:
        qwen_bundles = core.fit_oof_bundles(qwen_rows, vocabulary)
        llama_bundles = core.fit_oof_bundles(llama_rows, vocabulary)
    except RuntimeError as error:
        if "shortcut logistic is unusable" not in str(error):
            raise
        closure = {
            "version": VERSION,
            "verdict": "CLOSE_S0B_NUMERICAL_NONCONVERGENCE",
            "reason": str(error), "boundary_sha256": boundary_sha,
            "shortcut_table_sha256": sha256_file(out / "SHORTCUT_TABLE_AUDIT.json"),
            "authorizes_s1": False,
        }
        _emit_json(out / "S0B_CLOSED.json", closure, root=out, replay=replay)
        _assert_known_output_paths(out, boundary)
        return closure
    oof_artifact = {
        "version": VERSION, "boundary_sha256": boundary_sha,
        "qwen-source": [_serialize_oof_bundle(bundle) for bundle in qwen_bundles],
        "llama-audit": [_serialize_oof_bundle(bundle) for bundle in llama_bundles],
    }
    _emit_json(out / "SHORTCUT_OOF.json", oof_artifact, root=out, replay=replay)
    bootstrap_rows = []
    for population, population_rows, bundles in (
        ("qwen-source", qwen_rows, qwen_bundles),
        ("llama-audit", llama_rows, llama_bundles),
    ):
        for gate_index, gate_name in enumerate(core.gate_names()):
            path = out / "checkpoints" / "bootstrap" / population \
                / f"{gate_index:02d}.json"
            if not replay and path.exists():
                bootstrap_rows.append(_load_bootstrap_checkpoint(
                    out, population, gate_index, gate_name,
                    population_rows, bundles, boundary_sha,
                ))
            else:
                result = core.shortcut_gate_bootstrap(
                    population_rows, bundles, gate_name, n_draws=20_000,
                )
                bootstrap_rows.append(_bootstrap_checkpoint(
                    out, population, gate_index, result, boundary_sha, replay=replay,
                ))
    shortcut_pass = all(row["gate_pass"] for row in bootstrap_rows) \
        and all(value["pass"] for value in table_audit["marginal_prevalence"].values())
    shortcut_summary = {
        "version": VERSION, "boundary_sha256": boundary_sha,
        "verdict": "PASS_SHORTCUT_AUDIT" if shortcut_pass
        else "CLOSE_S0B_SHORTCUT_CONFOUNDING",
        "gates": bootstrap_rows,
    }
    _emit_json(
        out / "SHORTCUT_AUDIT.json", shortcut_summary, root=out, replay=replay,
    )
    if not shortcut_pass:
        closure = {
            "version": VERSION, "verdict": "CLOSE_S0B_SHORTCUT_CONFOUNDING",
            "boundary_sha256": boundary_sha,
            "shortcut_audit_sha256": sha256_file(out / "SHORTCUT_AUDIT.json"),
            "authorizes_s1": False,
        }
        _emit_json(out / "S0B_CLOSED.json", closure, root=out, replay=replay)
        _assert_known_output_paths(out, boundary)
        return closure

    null_strata = _load_json(prior / "NULL_STRATA.json")
    inner_folds = _load_json(prior / "INNER_FOLDS.json")
    records = core.group_matching_records(payloads, null_strata)
    matching = core.freeze_matching_graph(qwen_rows, vocabulary, records)
    partitions = core.canonical_partition_memberships(records, inner_folds)
    group_index = {group_id: index for index, group_id in enumerate(matching.group_ids)}
    matching_artifact = {
        "version": VERSION, "boundary_sha256": boundary_sha,
        "group_ids": list(matching.group_ids),
        "group_records": [asdict(record) for record in records],
        "vector_sha256": matching.vector_sha256,
        "caliper": matching.caliper,
        "unordered_pool_size": matching.unordered_pool_size,
        "directed_eligible_edges": [
            [group_index[left], group_index[right]]
            for left, right in matching.directed_eligible_edges
        ],
        "partitions": [
            [partition_id, [group_index[group_id] for group_id in group_ids]]
            for partition_id, group_ids in partitions
        ],
    }
    _emit_json(out / "MATCHING_GRAPH.json", matching_artifact, root=out, replay=replay)
    schedule_rows = []
    try:
        for family in (2, 3):
            for draw in range(200):
                path = out / "checkpoints" / "control" / str(family) \
                    / f"{draw:03d}.json"
                if not replay and path.exists():
                    schedule_rows.append(_load_schedule_checkpoint(
                        out, family, draw, matching.group_ids, partitions, records,
                        matching.directed_eligible_edges, boundary_sha,
                    ))
                else:
                    control = core.materialize_control_schedule(
                        family, draw, partitions, records,
                        matching.directed_eligible_edges,
                    )
                    schedule_rows.append(_schedule_checkpoint(
                        out, control, matching.group_ids, boundary_sha, replay=replay,
                    ))
    except RuntimeError as error:
        if "CLOSE_S0B_CONTROL" not in str(error):
            raise
        closure = {
            "version": VERSION, "verdict": "CLOSE_S0B_MATCHING_PREMISE",
            "reason": str(error), "boundary_sha256": boundary_sha,
            "shortcut_audit_sha256": sha256_file(out / "SHORTCUT_AUDIT.json"),
            "matching_graph_sha256": sha256_file(out / "MATCHING_GRAPH.json"),
            "authorizes_s1": False,
        }
        _emit_json(out / "S0B_CLOSED.json", closure, root=out, replay=replay)
        _assert_known_output_paths(out, boundary)
        return closure
    by_family = {
        family: [row for row in schedule_rows if row["family"] == family]
        for family in (2, 3)
    }
    if any(len(rows_) != 200 for rows_ in by_family.values()) \
            or any(len({row["schedule_sha256"] for row in rows_}) != 200
                   for rows_ in by_family.values()) \
            or any(len({row["outer_held_sha256"] for row in rows_}) != 200
                   for rows_ in by_family.values()):
        closure = {
            "version": VERSION, "verdict": "CLOSE_S0B_MATCHING_HASH_SUPPORT",
            "boundary_sha256": boundary_sha, "authorizes_s1": False,
        }
        _emit_json(out / "S0B_CLOSED.json", closure, root=out, replay=replay)
        _assert_known_output_paths(out, boundary)
        return closure
    aggregate = {
        "version": VERSION, "verdict": "PASS_S0B",
        "boundary_sha256": boundary_sha,
        "pythia_completion_sha256": sha256_file(out / "PYTHIA_NLL_COMPLETE.json"),
        "shortcut_table_sha256": sha256_file(out / "SHORTCUT_TABLE_AUDIT.json"),
        "shortcut_oof_sha256": sha256_file(out / "SHORTCUT_OOF.json"),
        "shortcut_audit_sha256": sha256_file(out / "SHORTCUT_AUDIT.json"),
        "matching_graph_sha256": sha256_file(out / "MATCHING_GRAPH.json"),
        "control_checkpoint_count": len(schedule_rows),
        "authorizes_s1": True,
    }
    _emit_json(out / "S0B_AGGREGATE.json", aggregate, root=out, replay=replay)
    completion = {
        "version": VERSION, "verdict": "PASS_S0B",
        "boundary_sha256": boundary_sha,
        "aggregate_sha256": sha256_file(out / "S0B_AGGREGATE.json"),
        "authorizes_s1": True,
    }
    _emit_json(out / "S0B_COMPLETE.json", completion, root=out, replay=replay)
    _assert_known_output_paths(out, boundary)
    return completion


def verify(out: str | Path, *, verify_prior: bool = True) -> dict[str, Any]:
    """Full no-write replay of the Pythia scores and the complete S0b decision."""
    out = Path(out)
    boundary, model, tokenizer = load_and_verify_boundary(
        out, load_model=True, verify_prior=verify_prior,
    )
    prior = REPO / boundary["prior_s0a"]["relative_path"]
    schedule = _prompt_schedule(_load_quartet_payloads(prior))
    pythia_replay_sha = _recompute_pythia_nll(
        out, boundary, model, tokenizer, schedule,
    )
    del model, tokenizer
    result = run_analysis(out, verify_prior=verify_prior, replay=True)
    terminal_names = {
        name for name in ("S0B_COMPLETE.json", "S0B_CLOSED.json")
        if (out / name).exists()
    }
    expected_terminal = "S0B_COMPLETE.json" if result.get("verdict") == "PASS_S0B" \
        else "S0B_CLOSED.json"
    if terminal_names != {expected_terminal}:
        raise RuntimeError("A6-S0b terminal artifact set is inconsistent")
    _assert_known_output_paths(out, boundary)
    return {
        "status": "S0B_VERIFIED",
        "verdict": result["verdict"],
        "boundary_sha256": sha256_file(out / "A6_S0B_BOUNDARY.json"),
        "pythia_replay_sha256": pythia_replay_sha,
        "terminal_sha256": sha256_file(out / expected_terminal),
        # Record whether the prior S0a manifest chain was fully authenticated
        # (--skip-prior-replay downgrades it); the sealed record must be able
        # to prove which mode produced it.
        "prior_s0a_full_verification": bool(verify_prior),
        "authorizes_s1": bool(result.get("authorizes_s1", False)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=(
            "prepare", "verify-boundary", "run-pythia", "run-analysis", "verify",
        ),
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--prior-s0a", default=str(DEFAULT_S0A))
    parser.add_argument("--pythia-source")
    parser.add_argument("--pythia-official-tree")
    parser.add_argument("--skip-prior-replay", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        if not args.pythia_source or not args.pythia_official_tree:
            raise SystemExit("prepare requires --pythia-source and --pythia-official-tree")
        result = prepare(
            args.out, pythia_source=args.pythia_source,
            pythia_official_tree=args.pythia_official_tree,
            prior_s0a=args.prior_s0a,
        )
    elif args.command == "verify-boundary":
        result = load_and_verify_boundary(
            args.out, load_model=True, verify_prior=not args.skip_prior_replay,
        )[0]
    elif args.command == "run-pythia":
        result = run_pythia(args.out, verify_prior=not args.skip_prior_replay)
    elif args.command == "run-analysis":
        result = run_analysis(args.out, verify_prior=not args.skip_prior_replay)
    else:
        result = verify(args.out, verify_prior=not args.skip_prior_replay)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
