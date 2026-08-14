#!/usr/bin/env python3
"""Prepare, run, and verify the frozen A6-S0a mechanical boundary.

No command in this runner generates a model response, extracts telemetry,
constructs a correctness sidecar, reads PopQA content, or opens an S1 seed.
``prepare`` freezes source/runtime/tokenizer bytes only.  ``run`` constructs
mechanical quartets, prompt-only natural manifests, folds, and null strata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VERSION = "automatic-group-free-iu-a6-s0a-v1-2026-08-14"
STATUS = "FROZEN_BEFORE_A6_S0A_EXECUTION"
DEFAULT_OUT = REPO / "results" / "automatic_group_free_phase_a6_s0a_v1"
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
    "scripts/test_automatic_group_free_phase_a6_s0a.py",
    "scripts/test_a6_s0a.py",
    "scripts/test_a6_interventions.py",
    "scripts/test_a6_s0_population.py",
)
RUNTIME_PACKAGES = (
    "transformers", "tokenizers", "numpy", "scipy", "scikit-learn", "torch",
    "huggingface-hub",
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS",
)
TOKENIZER_ARGUMENTS = (
    ("qwen3-4b", "qwen4_source"),
    ("qwen3-8b", "qwen8_source"),
    ("llama31-8b", "llama_source"),
)
IDENTITIES = {
    "qwen3-4b": {
        "role": "qwen_source_1", "repository": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    },
    "qwen3-8b": {
        "role": "qwen_source_2", "repository": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    },
    "llama31-8b": {
        "role": "held_llama", "repository": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
}
TOKENIZER_LITERAL_FILES = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "tokenizer.model",
}
FORBIDDEN_LLAMA_PATHS = (
    "llama_responses", "llama_features", "llama_correctness", "llama_sidecars",
)
BOUNDARY_REPORT_BYTES = (
    "# A6-S0a frozen source/runtime/input boundary\n\n"
    "No response telemetry, natural response, correctness sidecar, PopQA "
    "content, or sealed S1 seed was opened. S0a has not run.\n"
).encode("utf-8")


def _core():
    from spectral_utils import a6_s0a
    return a6_s0a


def source_files() -> tuple[str, ...]:
    package = tuple(
        str(path.relative_to(REPO))
        for path in sorted((REPO / "spectral_utils").glob("*.py"))
    )
    return STATIC_SOURCE_FILES + package


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _json_native(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPO / name) for name in source_files()}


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


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


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
        raise RuntimeError("artifact path escapes its output root") from error
    current = root
    for component in relative_parent.parts:
        current = current / component
        if _lexists(current):
            _require_real_directory(current, "artifact parent")


def _exclusive_bytes(path: Path, data: bytes, *, root: Path | None = None) -> None:
    root = path.parent if root is None else root
    _require_real_containment(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_real_containment(root, path)
    temporary = path.with_name(path.name + ".tmp")
    if _lexists(path):
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    if _lexists(temporary):
        _require_real_file(temporary, "interrupted artifact temporary")
        if temporary.read_bytes() == data:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
            return
        # A partial fsyncless write has no authority.  It is the only mutable
        # object in the append-only protocol and is safely regenerated.
        temporary.unlink()
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


def _exclusive_json(path: Path, value: Any) -> None:
    _exclusive_bytes(path, canonical_json_bytes(value))


def _load_json(path: Path) -> Any:
    _require_real_file(path, "JSON artifact")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _identity_by_scorer():
    return {value.scorer_id: value for value in _core().TOKENIZER_IDENTITIES}


def _selected_tokenizer_file(path: Path) -> bool:
    name = path.name
    return (
        name in TOKENIZER_LITERAL_FILES
        or (name.startswith("sentencepiece") and name.endswith(".model"))
        or (name.startswith("chat_template") and name.endswith(".jinja"))
    )


def _prepare_snapshot_stdlib(
    source: Path, destination_root: Path, scorer_id: str,
) -> tuple[Path, dict[str, Any]]:
    identity = IDENTITIES[scorer_id]
    tree = tuple(
        path.relative_to(source).as_posix()
        for path in sorted(source.rglob("*")) if path.is_file() or path.is_symlink()
    )
    selected = []
    for path in sorted(source.rglob("*")):
        if path.is_dir() or not _selected_tokenizer_file(path):
            continue
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            target = Path(os.readlink(path))
            target = target if target.is_absolute() else path.parent / target
            if target.is_symlink():
                raise RuntimeError(f"tokenizer source has a multi-hop link: {relative}")
            resolved = target.resolve(strict=True)
        else:
            resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f"tokenizer source is not a regular file: {relative}")
        selected.append((relative, resolved))
    if not selected:
        raise RuntimeError(f"BLOCKED_TOKENIZER_ACCESS:{scorer_id}:empty_snapshot")
    files = [
        {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for relative, path in selected
    ]
    content_sha = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    destination = destination_root / f"{scorer_id}-{content_sha}"
    if destination.exists():
        raise FileExistsError(f"refusing to reuse snapshot destination: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for (relative, source_path), expected in zip(selected, files):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_path.open("rb") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            if target.stat().st_size != expected["size"] or sha256_file(target) != expected["sha256"]:
                raise RuntimeError("copied tokenizer bytes differ from source manifest")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    manifest = {
        "role": identity["role"], "scorer_id": scorer_id,
        "repository": identity["repository"], "revision": identity["revision"],
        "content_sha256": content_sha, "repository_tree": list(tree), "files": files,
    }
    _verify_snapshot_stdlib(destination, manifest)
    return destination, manifest


def _verify_snapshot_stdlib(snapshot: Path, manifest: dict[str, Any]) -> None:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError("content-addressed tokenizer snapshot root is invalid")
    required_manifest_keys = {
        "role", "scorer_id", "repository", "revision", "content_sha256",
        "repository_tree", "files",
    }
    if set(manifest) != required_manifest_keys:
        raise RuntimeError("tokenizer snapshot manifest schema mismatch")
    if not isinstance(manifest["content_sha256"], str) \
            or len(manifest["content_sha256"]) != 64:
        raise RuntimeError("tokenizer snapshot content hash is invalid")
    actual_paths = []
    actual_directories = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            actual_directories.append(path.relative_to(snapshot).as_posix())
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("content-addressed tokenizer input is not regular")
        actual_paths.append(path.relative_to(snapshot).as_posix())
    expected_paths = [row["path"] for row in manifest["files"]]
    if expected_paths != sorted(set(expected_paths)):
        raise RuntimeError("tokenizer snapshot paths are not sorted and unique")
    for row, relative in zip(manifest["files"], expected_paths):
        if set(row) != {"path", "size", "sha256"} \
                or isinstance(row["size"], bool) or not isinstance(row["size"], int) \
                or row["size"] < 0 or not isinstance(row["sha256"], str) \
                or len(row["sha256"]) != 64:
            raise RuntimeError("tokenizer snapshot file manifest is invalid")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not _selected_tokenizer_file(Path(relative)):
            raise RuntimeError("tokenizer snapshot contains an invalid relative path")
    allowed_directories = sorted({
        parent.as_posix()
        for relative in expected_paths for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    })
    if actual_directories != allowed_directories:
        raise RuntimeError("tokenizer snapshot directory set changed")
    if actual_paths != expected_paths:
        raise RuntimeError("tokenizer snapshot path set changed")
    actual_files = []
    for expected in manifest["files"]:
        path = snapshot / expected["path"]
        row = {
            "path": expected["path"], "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if row != expected:
            raise RuntimeError(f"tokenizer snapshot changed: {expected['path']}")
        actual_files.append(row)
    if hashlib.sha256(canonical_json_bytes(actual_files)).hexdigest() != manifest["content_sha256"]:
        raise RuntimeError("tokenizer snapshot aggregate hash mismatch")


def _require_resolved_revision_directory(source: Path, scorer_id: str) -> None:
    identity = IDENTITIES[scorer_id]
    expected = identity["revision"]
    resolved = source.resolve(strict=True)
    repo_directory = "models--" + identity["repository"].replace("/", "--")
    if tuple(resolved.parts[-3:]) != (repo_directory, "snapshots", expected):
        raise RuntimeError(
            "BLOCKED_TOKENIZER_ACCESS: source path is not the frozen repo snapshot"
        )
    cache_root = resolved.parents[2]
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir(cache_root)
    except Exception as error:
        raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: Hugging Face cache audit failed") from error
    matches = [
        revision for repository in cache.repos
        if repository.repo_id == identity["repository"]
        for revision in repository.revisions
        if revision.commit_hash == expected
        and Path(revision.snapshot_path).resolve() == resolved
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "BLOCKED_TOKENIZER_ACCESS: cache metadata does not bind repo and revision"
        )


def _config_eos_ids(path: Path) -> list[int]:
    if not path.is_file() or path.is_symlink():
        return []
    value = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
    if value is None:
        return []
    values = [value] if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(values, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in values
    ):
        raise RuntimeError(f"invalid eos_token_id in {path.name}")
    return list(dict.fromkeys(values))


def _config_pad_id(path: Path) -> int | None:
    if not path.is_file() or path.is_symlink():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("pad_token_id")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"invalid pad_token_id in {path.name}")
    return value


def _tokenizer_boundary_audit(tokenizer, identity, snapshot: Path) -> dict[str, Any]:
    core = _core()
    try:
        template = tokenizer.get_chat_template()
    except AttributeError:
        template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise RuntimeError("tokenizer has no resolved chat-template text")
    audit = core.build_natural_tokenizer_evidence(
        tokenizer, identity, "A6 tokenizer boundary audit.", 0,
    )
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and not isinstance(eos, bool):
        tokenizer_eos = [eos]
    elif isinstance(eos, (tuple, list)) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in eos
    ):
        tokenizer_eos = list(dict.fromkeys(eos))
    else:
        raise RuntimeError("tokenizer eos_token_id is missing")
    model_eos = _config_eos_ids(snapshot / "config.json")
    generation_eos = _config_eos_ids(snapshot / "generation_config.json")
    effective_eos = generation_eos or model_eos or tokenizer_eos
    if not effective_eos:
        raise RuntimeError("effective generation EOS set is empty")
    setattr(tokenizer, "_a6_effective_generation_eos_token_ids", tuple(effective_eos))
    tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
    if tokenizer_pad is not None and (
        isinstance(tokenizer_pad, bool) or not isinstance(tokenizer_pad, int)
        or tokenizer_pad < 0
    ):
        raise RuntimeError("tokenizer pad_token_id is invalid")
    model_pad = _config_pad_id(snapshot / "config.json")
    generation_pad = _config_pad_id(snapshot / "generation_config.json")
    effective_pad = (
        generation_pad if generation_pad is not None
        else model_pad if model_pad is not None else tokenizer_pad
    )
    setattr(tokenizer, "_a6_effective_generation_pad_token_id", effective_pad)
    return _json_native({
        "scorer_id": identity.scorer_id,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "chat_template": template,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "tokenizer_eos_token_ids": tokenizer_eos,
        "model_config_eos_token_ids": model_eos,
        "generation_config_eos_token_ids": generation_eos,
        "effective_generation_eos_token_ids": effective_eos,
        "tokenizer_pad_token_id": tokenizer_pad,
        "model_config_pad_token_id": model_pad,
        "generation_config_pad_token_id": generation_pad,
        "effective_generation_pad_token_id": effective_pad,
        "audit_prompt_evidence": {
            key: value for key, value in vars(audit).items()
        },
    })


def _snapshot_manifest_from_json(core, value):
    files = tuple(core.SnapshotFile(**row) for row in value["files"])
    return core.SnapshotManifest(
        value["role"], value["scorer_id"], value["repository"], value["revision"],
        value["content_sha256"], tuple(value["repository_tree"]), files,
    )


def _validate_tokenizer_audits(audits: Any) -> None:
    if not isinstance(audits, dict) or set(audits) != set(IDENTITIES):
        raise RuntimeError("A6-S0a tokenizer-audit roster mismatch")
    audit_fields = {
        "scorer_id", "is_fast", "chat_template", "chat_template_sha256",
        "tokenizer_eos_token_ids", "model_config_eos_token_ids",
        "generation_config_eos_token_ids", "effective_generation_eos_token_ids",
        "tokenizer_pad_token_id", "model_config_pad_token_id",
        "generation_config_pad_token_id", "effective_generation_pad_token_id",
        "audit_prompt_evidence",
    }
    prompt_fields = {
        "scorer_id", "repository", "revision", "prefix_text_sha256", "input_ids",
        "input_ids_sha256", "input_length", "attention_mask_sha256",
        "generation_seed", "generation_parameters",
    }
    for scorer_id, audit in audits.items():
        if not isinstance(audit, dict) or set(audit) != audit_fields \
                or audit.get("scorer_id") != scorer_id or audit.get("is_fast") is not True:
            raise RuntimeError("A6-S0a tokenizer-audit schema mismatch")
        template = audit.get("chat_template")
        if not isinstance(template, str) or not template \
                or audit.get("chat_template_sha256") != hashlib.sha256(
                    template.encode("utf-8")
                ).hexdigest():
            raise RuntimeError("A6-S0a tokenizer chat-template audit is invalid")
        for field in (
            "tokenizer_eos_token_ids", "model_config_eos_token_ids",
            "generation_config_eos_token_ids", "effective_generation_eos_token_ids",
        ):
            values = audit.get(field)
            if not isinstance(values, list) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            ) or len(values) != len(set(values)):
                raise RuntimeError("A6-S0a tokenizer EOS manifest is invalid")
        effective = (
            audit["generation_config_eos_token_ids"]
            or audit["model_config_eos_token_ids"]
            or audit["tokenizer_eos_token_ids"]
        )
        if not effective or audit["effective_generation_eos_token_ids"] != effective:
            raise RuntimeError("A6-S0a effective EOS set is invalid")
        for field in (
            "tokenizer_pad_token_id", "model_config_pad_token_id",
            "generation_config_pad_token_id", "effective_generation_pad_token_id",
        ):
            pad = audit.get(field)
            if pad is not None and (
                isinstance(pad, bool) or not isinstance(pad, int) or pad < 0
            ):
                raise RuntimeError("A6-S0a tokenizer pad ID is invalid")
        effective_pad = (
            audit["generation_config_pad_token_id"]
            if audit["generation_config_pad_token_id"] is not None
            else audit["model_config_pad_token_id"]
            if audit["model_config_pad_token_id"] is not None
            else audit["tokenizer_pad_token_id"]
        )
        if audit["effective_generation_pad_token_id"] != effective_pad:
            raise RuntimeError("A6-S0a effective pad ID is invalid")
        prompt = audit.get("audit_prompt_evidence")
        identity = IDENTITIES[scorer_id]
        if not isinstance(prompt, dict) or set(prompt) != prompt_fields \
                or prompt.get("scorer_id") != scorer_id \
                or prompt.get("repository") != identity["repository"] \
                or prompt.get("revision") != identity["revision"]:
            raise RuntimeError("A6-S0a tokenizer prompt-audit schema mismatch")
        ids = prompt.get("input_ids")
        if not isinstance(ids, list) or not ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ) or prompt.get("input_length") != len(ids):
            raise RuntimeError("A6-S0a tokenizer prompt IDs are invalid")
        ids_sha = hashlib.sha256(canonical_json_bytes(ids)).hexdigest()
        ones_sha = hashlib.sha256(canonical_json_bytes([1] * len(ids))).hexdigest()
        if prompt.get("input_ids_sha256") != ids_sha \
                or prompt.get("attention_mask_sha256") != ones_sha:
            raise RuntimeError("A6-S0a tokenizer prompt-ID hashes are invalid")
        for name in ("prefix_text_sha256", "input_ids_sha256", "attention_mask_sha256"):
            value = prompt.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise RuntimeError("A6-S0a tokenizer prompt hash is invalid")


def prepare(
    out: str | Path,
    *,
    qwen4_source: str | Path,
    qwen8_source: str | Path,
    llama_source: str | Path,
) -> dict[str, Any]:
    out = Path(out)
    if _lexists(out):
        _require_real_directory(out, "A6-S0a output root")
        if any(out.iterdir()):
            raise RuntimeError("A6-S0a prepare requires a new or empty output directory")
    else:
        _require_real_directory(out.parent, "A6-S0a output parent")
        out.mkdir(parents=True, exist_ok=False)
    _require_real_directory(out, "A6-S0a output root")
    sources = {
        "qwen3-4b": Path(qwen4_source), "qwen3-8b": Path(qwen8_source),
        "llama31-8b": Path(llama_source),
    }
    for scorer_id in IDENTITIES:
        source = sources[scorer_id]
        if not source.exists():
            raise RuntimeError(f"BLOCKED_TOKENIZER_ACCESS:{scorer_id}")
        _require_resolved_revision_directory(source, scorer_id)
    snapshot_rows, tokenizer_audits = {}, {}
    for scorer_id in IDENTITIES:
        source = sources[scorer_id]
        destination, manifest_json = _prepare_snapshot_stdlib(
            source, out / "inputs", scorer_id,
        )
        snapshot_rows[scorer_id] = {
            "relative_directory": destination.relative_to(out).as_posix(),
            "manifest": manifest_json,
        }
    # Only now, after all three input byte manifests verify, may package and
    # Transformers imports occur.
    core = _core()
    identities = _identity_by_scorer()
    for scorer_id in core.SCORER_IDS:
        row = snapshot_rows[scorer_id]
        manifest = _snapshot_manifest_from_json(core, row["manifest"])
        destination = out / row["relative_directory"]
        tokenizer = core.load_verified_fast_tokenizer(destination, manifest)
        tokenizer_audits[scorer_id] = _tokenizer_boundary_audit(
            tokenizer, identities[scorer_id], destination,
        )
    boundary = {
        "version": VERSION,
        "status": STATUS,
        "execution_contract_sha256": sha256_file(EXECUTION_CONTRACT),
        "parent_protocol_sha256": sha256_file(PARENT_PROTOCOL),
        "source_sha256": _source_hashes(),
        "runtime_versions": _runtime_versions(),
        "git_head": _git_head(),
        "tokenizer_snapshots": snapshot_rows,
        "tokenizer_audits": tokenizer_audits,
        "configuration": {
            "quartet_slots": 1_800,
            "natural_slots": 6_000,
            "popqa_reserved_rows": 14_267,
            "max_attempts_per_slot": 10_000,
            "response_telemetry_accessed": False,
            "natural_response_accessed": False,
            "correctness_sidecar_created": False,
            "popqa_content_accessed": False,
            "sealed_s1_seed_opened": False,
        },
    }
    _exclusive_json(out / "A6_S0A_BOUNDARY.json", boundary)
    _exclusive_bytes(out / "BOUNDARY_REPORT.md", BOUNDARY_REPORT_BYTES)
    return boundary


def load_and_verify_boundary(out: str | Path, *, load_tokenizers: bool = False):
    out = Path(out)
    _require_real_directory(out, "A6-S0a output root")
    boundary_path = out / "A6_S0A_BOUNDARY.json"
    boundary = _load_json(boundary_path)
    if set(boundary) != {
        "version", "status", "execution_contract_sha256", "parent_protocol_sha256",
        "source_sha256", "runtime_versions", "git_head", "tokenizer_snapshots",
        "tokenizer_audits", "configuration",
    }:
        raise RuntimeError("A6-S0a boundary schema mismatch")
    if boundary.get("version") != VERSION or boundary.get("status") != STATUS:
        raise RuntimeError("A6-S0a boundary version/status mismatch")
    report = out / "BOUNDARY_REPORT.md"
    if not report.is_file() or report.is_symlink() \
            or report.read_bytes() != BOUNDARY_REPORT_BYTES:
        raise RuntimeError("A6-S0a boundary report changed")
    if boundary.get("execution_contract_sha256") != sha256_file(EXECUTION_CONTRACT):
        raise RuntimeError("A6-S0a execution contract changed after freeze")
    if boundary.get("parent_protocol_sha256") != sha256_file(PARENT_PROTOCOL):
        raise RuntimeError("A6 parent protocol changed after freeze")
    if boundary.get("source_sha256") != _source_hashes():
        raise RuntimeError("A6-S0a source closure changed after freeze")
    if boundary.get("runtime_versions") != _runtime_versions():
        raise RuntimeError("A6-S0a runtime changed after freeze")
    if boundary.get("git_head") != _git_head():
        raise RuntimeError("A6-S0a git commit changed after freeze")
    expected_configuration = {
        "quartet_slots": 1_800, "natural_slots": 6_000,
        "popqa_reserved_rows": 14_267, "max_attempts_per_slot": 10_000,
        "response_telemetry_accessed": False, "natural_response_accessed": False,
        "correctness_sidecar_created": False, "popqa_content_accessed": False,
        "sealed_s1_seed_opened": False,
    }
    if boundary.get("configuration") != expected_configuration:
        raise RuntimeError("A6-S0a boundary configuration mismatch")
    tokenizers = {}
    snapshots = boundary.get("tokenizer_snapshots", {})
    if set(snapshots) != set(IDENTITIES):
        raise RuntimeError("A6-S0a tokenizer snapshot roster mismatch")
    audits = boundary.get("tokenizer_audits", {})
    _validate_tokenizer_audits(audits)
    inputs = out / "inputs"
    if not inputs.is_dir() or inputs.is_symlink():
        raise RuntimeError("A6-S0a inputs root is invalid")
    declared_children = {
        PurePosixPath(row["relative_directory"]).name for row in snapshots.values()
    }
    actual_children = {path.name for path in inputs.iterdir()}
    if actual_children != declared_children or any(
        not path.is_dir() or path.is_symlink() for path in inputs.iterdir()
    ):
        raise RuntimeError("A6-S0a inputs namespace contains unmanifested paths")
    # Verify every content-addressed path with stdlib only, before importing
    # the project package (whose legacy initializer imports Transformers).
    for scorer_id in IDENTITIES:
        row = snapshots[scorer_id]
        manifest_json = row["manifest"]
        identity_json = IDENTITIES[scorer_id]
        if (
            manifest_json["scorer_id"] != scorer_id
            or manifest_json["repository"] != identity_json["repository"]
            or manifest_json["revision"] != identity_json["revision"]
        ):
            raise RuntimeError("A6-S0a tokenizer identity mismatch")
        expected_relative = f"inputs/{scorer_id}-{manifest_json['content_sha256']}"
        if set(row) != {"relative_directory", "manifest"} \
                or row["relative_directory"] != expected_relative:
            raise RuntimeError("A6-S0a tokenizer input path mismatch")
        snapshot = out / expected_relative
        _verify_snapshot_stdlib(snapshot, manifest_json)
    core = _core()
    identities = _identity_by_scorer()
    for scorer_id in core.SCORER_IDS:
        row = snapshots[scorer_id]
        manifest = _snapshot_manifest_from_json(core, row["manifest"])
        identity = identities[scorer_id]
        snapshot = out / row["relative_directory"]
        core.verify_content_addressed_snapshot(snapshot, manifest)
        if load_tokenizers:
            tokenizer = core.load_verified_fast_tokenizer(snapshot, manifest)
            audit = _tokenizer_boundary_audit(tokenizer, identity, snapshot)
            if audit != boundary["tokenizer_audits"][scorer_id]:
                raise RuntimeError("A6-S0a tokenizer/chat audit changed")
            tokenizers[scorer_id] = tokenizer
    return boundary, tokenizers


def _checkpoint(path: Path, payload: dict[str, Any], boundary_sha256: str) -> None:
    wrapped = {"boundary_sha256": boundary_sha256, "payload": payload}
    data = canonical_json_bytes(wrapped)
    if _lexists(path):
        _require_real_file(path, "checkpoint")
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable checkpoint mismatch: {path}")
        return
    _exclusive_bytes(path, data, root=path.parents[2])


def _checkpoint_manifest(root: Path) -> list[dict[str, Any]]:
    _require_real_directory(root, "A6-S0a output root")
    checkpoint_root = root / "checkpoints"
    if not _lexists(checkpoint_root):
        return []
    _require_real_directory(checkpoint_root, "checkpoint root")
    files = []
    for family in checkpoint_root.iterdir():
        _require_real_directory(family, "checkpoint family")
        for path in family.iterdir():
            _require_real_file(path, "checkpoint file")
            if path.name.endswith(".json"):
                files.append(path)
            elif not path.name.endswith(".json.tmp"):
                raise RuntimeError("checkpoint tree contains an unregistered file")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]


def _assert_known_output_paths(
    out: Path, *, completed: bool, boundary: dict[str, Any] | None = None,
) -> None:
    _require_real_directory(out, "A6-S0a output root")
    allowed_top = {
        "A6_S0A_BOUNDARY.json", "BOUNDARY_REPORT.md", "inputs", "checkpoints",
        "INNER_FOLDS.json", "NULL_STRATA.json", "POPQA_RESERVATION.json",
        "LLAMA_FUTURE_SCHEMA.json", "S0A_AGGREGATE.json", "S0A_COMPLETE.json",
        "S0A_CLOSED.json",
    }
    allowed_temps = {name + ".tmp" for name in allowed_top if "." in name}
    unknown = sorted(
        path.name for path in out.iterdir()
        if path.name not in allowed_top
        and not (not completed and path.name in allowed_temps)
    )
    if unknown:
        raise RuntimeError(f"unmanifested A6-S0a output paths: {unknown}")
    for path in out.iterdir():
        if path.name in {"inputs", "checkpoints"}:
            _require_real_directory(path, f"A6-S0a {path.name} root")
        else:
            _require_real_file(path, f"A6-S0a artifact {path.name}")
        if completed and path.name.endswith(".tmp"):
            raise RuntimeError("completed A6-S0a output contains a temporary file")
    if boundary is not None:
        snapshots = boundary["tokenizer_snapshots"]
        expected = {
            PurePosixPath(row["relative_directory"]).name for row in snapshots.values()
        }
        inputs = out / "inputs"
        if not inputs.is_dir() or inputs.is_symlink() \
                or {path.name for path in inputs.iterdir()} != expected \
                or any(not path.is_dir() or path.is_symlink() for path in inputs.iterdir()):
            raise RuntimeError("unmanifested A6-S0a inputs path")
    checkpoint_root = out / "checkpoints"
    if _lexists(checkpoint_root):
        _require_real_directory(checkpoint_root, "checkpoint root")
        if any(
            path.name not in {"quartet", "natural"}
            for path in checkpoint_root.iterdir()
        ):
            raise RuntimeError("unregistered A6-S0a checkpoint family")
        for family, count in (("quartet", 1_800), ("natural", 6_000)):
            directory = checkpoint_root / family
            if not _lexists(directory):
                if completed:
                    raise RuntimeError(f"missing completed checkpoint family: {family}")
                continue
            _require_real_directory(directory, f"{family} checkpoint family")
            names = sorted(path.name for path in directory.iterdir())
            for path in directory.iterdir():
                _require_real_file(path, "checkpoint-family member")
            allowed_names = {f"{index:04d}.json" for index in range(count)}
            allowed_temp_names = {name + ".tmp" for name in allowed_names}
            permitted = allowed_names if completed else allowed_names | allowed_temp_names
            if not set(names) <= permitted:
                raise RuntimeError(f"unregistered {family} checkpoint name")
            final_names = {name for name in names if name.endswith(".json")}
            if completed and final_names != allowed_names:
                raise RuntimeError(f"completed {family} checkpoint schedule is incomplete")


def _write_or_compare_json(path: Path, payload: Any) -> None:
    data = canonical_json_bytes(payload)
    if _lexists(path):
        _require_real_file(path, "frozen S0a artifact")
        if path.read_bytes() != data:
            raise RuntimeError(f"frozen S0a artifact mismatch: {path.name}")
    else:
        _exclusive_bytes(path, data)


def _load_checkpoint_prefix(
    out: Path, family: str, count: int, boundary_sha256: str, decoder,
) -> tuple[Any, ...]:
    directory = out / "checkpoints" / family
    if not _lexists(directory):
        return ()
    _require_real_directory(directory, f"{family} checkpoint family")
    for temporary in sorted(directory.glob("*.json.tmp")):
        _require_real_file(temporary, "checkpoint temporary")
        target = temporary.with_name(temporary.name[:-4])
        if _lexists(target):
            _require_real_file(target, "checkpoint")
            temporary.unlink()
    names = sorted(
        path.name for path in directory.iterdir() if path.name.endswith(".json")
    )
    indices = [int(name[:-5]) for name in names]
    if indices != list(range(len(indices))) or len(indices) > count:
        raise RuntimeError(f"{family} checkpoints are not one contiguous prefix")
    records = []
    for index, name in enumerate(names):
        path = directory / name
        _require_real_file(path, "checkpoint")
        raw = path.read_bytes()
        wrapped = json.loads(raw)
        if raw != canonical_json_bytes(wrapped):
            raise RuntimeError(f"{family} checkpoint is not canonical JSON")
        if set(wrapped) != {"boundary_sha256", "payload"} \
                or wrapped["boundary_sha256"] != boundary_sha256:
            raise RuntimeError(f"{family} checkpoint boundary mismatch")
        records.append(decoder(wrapped["payload"]))
    return tuple(records)


def run(out: str | Path, *, resume: bool = True) -> dict[str, Any]:
    out = Path(out)
    boundary, tokenizers = load_and_verify_boundary(out, load_tokenizers=True)
    boundary_sha = sha256_file(out / "A6_S0A_BOUNDARY.json")
    for relative in FORBIDDEN_LLAMA_PATHS:
        if (out / relative).exists():
            raise RuntimeError(f"forbidden A6 Llama payload namespace exists: {relative}")
    _assert_known_output_paths(out, completed=False, boundary=boundary)
    core = _core()
    existing_quartets = existing_natural = ()
    if resume:
        existing_quartets = _load_checkpoint_prefix(
            out, "quartet", 1_800, boundary_sha, core.quartet_record_from_public,
        )
        existing_natural = _load_checkpoint_prefix(
            out, "natural", 6_000, boundary_sha, core.natural_record_from_public,
        )
        if existing_natural and len(existing_quartets) != 1_800:
            raise RuntimeError("natural checkpoints precede the complete quartet schedule")
    def save_quartet(index, record):
        _checkpoint(
            out / "checkpoints" / "quartet" / f"{index:04d}.json",
            core.public_quartet_record(record), boundary_sha,
        )

    def save_natural(index, record):
        _checkpoint(
            out / "checkpoints" / "natural" / f"{index:04d}.json",
            core.public_natural_prompt_record(record), boundary_sha,
        )
    try:
        quartets, natural = core.build_full_s0a_population(
            tokenizers, on_quartet=save_quartet, on_natural=save_natural,
            existing_quartets=existing_quartets,
            existing_natural=existing_natural,
        )
        inner = core.inner_fold_manifest(quartets)
        strata = core.build_null_strata(quartets, inner)
    except RuntimeError as error:
        reason = str(error)
        if not reason.startswith("CLOSE_INVALID_INTERVENTION_BOUNDARY:"):
            raise
        closure = {
            "version": VERSION,
            "verdict": "CLOSE_INVALID_INTERVENTION_BOUNDARY",
            "reason": reason,
            "boundary_sha256": boundary_sha,
            "checkpoint_manifest": _checkpoint_manifest(out),
            "response_telemetry_accessed": False,
            "natural_response_accessed": False,
            "correctness_sidecar_created": False,
            "popqa_content_accessed": False,
            "sealed_s1_seed_opened": False,
        }
        _write_or_compare_json(out / "S0A_CLOSED.json", closure)
        _assert_known_output_paths(out, completed=False, boundary=boundary)
        return closure
    if (out / "S0A_CLOSED.json").exists():
        raise RuntimeError("frozen S0a closure no longer reproduces")
    artifacts = {
        "INNER_FOLDS.json": [list(value) for value in inner],
        "NULL_STRATA.json": [vars(value) | {
            "merges": [vars(item) for item in value.merges]
        } for value in strata],
        "POPQA_RESERVATION.json": core.popqa_opaque_reservation(),
        "LLAMA_FUTURE_SCHEMA.json": vars(core.future_llama_sidecar_schema()),
    }
    for name, payload in artifacts.items():
        _write_or_compare_json(out / name, payload)
    checkpoint_manifest = _checkpoint_manifest(out)
    if len(checkpoint_manifest) != 7_800:
        raise RuntimeError("A6-S0a checkpoint schedule is incomplete")
    result_files = {
        name: sha256_file(out / name) for name in sorted(artifacts)
    }
    aggregate = {
        "version": VERSION,
        "verdict": "PASS_S0A",
        "boundary_sha256": boundary_sha,
        "n_quartets": len(quartets),
        "n_natural_prompts": len(natural),
        "n_inner_fold_assignments": len(inner),
        "n_null_cells": len(strata),
        "checkpoint_manifest": checkpoint_manifest,
        "result_file_sha256": result_files,
        "response_telemetry_accessed": False,
        "natural_response_accessed": False,
        "correctness_sidecar_created": False,
        "popqa_content_accessed": False,
        "sealed_s1_seed_opened": False,
    }
    _write_or_compare_json(out / "S0A_AGGREGATE.json", aggregate)
    completion = {
        "version": VERSION, "verdict": "PASS_S0A",
        "boundary_sha256": boundary_sha,
        "aggregate_sha256": sha256_file(out / "S0A_AGGREGATE.json"),
    }
    _write_or_compare_json(out / "S0A_COMPLETE.json", completion)
    _assert_known_output_paths(out, completed=True, boundary=boundary)
    return completion


def verify(out: str | Path, *, replay: bool = True) -> dict[str, Any]:
    out = Path(out)
    # Even a boundary-only verification must reconstruct the frozen tokenizer
    # objects and contextual chat audit.  File hashes alone do not authorize a
    # later stage because template resolution is result-relevant.
    boundary, _ = load_and_verify_boundary(out, load_tokenizers=True)
    closure_path = out / "S0A_CLOSED.json"
    completion_path = out / "S0A_COMPLETE.json"
    if closure_path.exists():
        if completion_path.exists():
            raise RuntimeError("A6-S0a has both closure and completion artifacts")
        _assert_known_output_paths(out, completed=False, boundary=boundary)
        closure = _load_json(closure_path)
        expected = {
            "version": VERSION,
            "verdict": "CLOSE_INVALID_INTERVENTION_BOUNDARY",
            "reason": closure.get("reason"),
            "boundary_sha256": sha256_file(out / "A6_S0A_BOUNDARY.json"),
            "checkpoint_manifest": _checkpoint_manifest(out),
            "response_telemetry_accessed": False,
            "natural_response_accessed": False,
            "correctness_sidecar_created": False,
            "popqa_content_accessed": False,
            "sealed_s1_seed_opened": False,
        }
        if (
            closure != expected
            or not isinstance(closure.get("reason"), str)
            or not closure["reason"].startswith(
                "CLOSE_INVALID_INTERVENTION_BOUNDARY:"
            )
        ):
            raise RuntimeError("A6-S0a closure provenance mismatch")
        if replay:
            if run(out, resume=False) != closure:
                raise RuntimeError("A6-S0a closure replay mismatch")
            return {"status": "CLOSE_S0A_VERIFIED", "closure": closure}
        return {
            "status": "HASH_ONLY_DIAGNOSTIC_NOT_AUTHORIZING_PASS",
            "authorizes_next_stage": False,
            "closure": closure,
        }
    if not completion_path.exists():
        _assert_known_output_paths(out, completed=False, boundary=boundary)
        return {"status": "BOUNDARY_ONLY", "boundary": boundary}
    _assert_known_output_paths(out, completed=True, boundary=boundary)
    completion = _load_json(completion_path)
    boundary_sha = sha256_file(out / "A6_S0A_BOUNDARY.json")
    if completion != {
        "version": VERSION, "verdict": "PASS_S0A",
        "boundary_sha256": boundary_sha,
        "aggregate_sha256": sha256_file(out / "S0A_AGGREGATE.json"),
    }:
        raise RuntimeError("A6-S0a completion provenance mismatch")
    aggregate = _load_json(out / "S0A_AGGREGATE.json")
    if aggregate.get("boundary_sha256") != boundary_sha:
        raise RuntimeError("A6-S0a aggregate boundary mismatch")
    if aggregate.get("checkpoint_manifest") != _checkpoint_manifest(out):
        raise RuntimeError("A6-S0a checkpoint manifest mismatch")
    for name, expected in aggregate.get("result_file_sha256", {}).items():
        if sha256_file(out / name) != expected:
            raise RuntimeError(f"A6-S0a result artifact changed: {name}")
    if replay:
        # ``run`` is append-only and compares every reconstructed checkpoint and
        # aggregate byte-for-byte, so it is also the complete semantic verifier.
        if run(out, resume=False) != completion:
            raise RuntimeError("A6-S0a semantic replay mismatch")
        return {"status": "PASS_S0A_VERIFIED", "completion": completion}
    return {
        "status": "HASH_ONLY_DIAGNOSTIC_NOT_AUTHORIZING_PASS",
        "authorizes_next_stage": False,
        "completion": completion,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--qwen4-source")
    parser.add_argument("--qwen8-source")
    parser.add_argument("--llama-source")
    parser.add_argument("--hash-only-diagnostic", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        missing = [
            name for name in ("qwen4_source", "qwen8_source", "llama_source")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"prepare requires: {', '.join(missing)}")
        result = prepare(
            args.out, qwen4_source=args.qwen4_source,
            qwen8_source=args.qwen8_source, llama_source=args.llama_source,
        )
    elif args.command == "run":
        result = run(args.out)
    else:
        result = verify(args.out, replay=not args.hash_only_diagnostic)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
