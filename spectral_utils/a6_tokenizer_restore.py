"""Authenticated, all-three tokenizer materialization for A6 S0a.

Google Drive is treated only as an untrusted byte transport.  Repository
identity and allowlist completeness come from the exact Hugging Face revision
API; every selected byte string must match its official Git blob or LFS digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.request
from typing import Any


SCHEMA_VERSION = "a6-tokenizer-restore-v1"
ROLE_ORDER = ("qwen3-4b", "qwen3-8b", "llama31-8b")
REPO = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "spectral_utils/a6_tokenizer_restore.py",
    "scripts/automatic_group_free_phase_a6_tokenizer_restore.py",
)
ALLOWLIST_NAMES = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "tokenizer.model",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: sha256_bytes((REPO / name).read_bytes()) for name in SOURCE_PATHS}


def git_blob_sha1(value: bytes) -> str:
    header = b"blob " + str(len(value)).encode("ascii") + b"\0"
    return hashlib.sha1(header + value).hexdigest()


def _utf8_sorted(values):
    return tuple(sorted(values, key=lambda item: item.encode("utf-8")))


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str) and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class SelectedFile:
    path: str
    blob_id: str
    size: int
    sha256: str
    git_blob_sha1: str
    lfs_sha256: str | None = None
    source_remote: str | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class RoleSpec:
    role: str
    repository: str
    revision: str
    gated: bool | str
    all_paths: tuple[str, ...]
    selected: tuple[SelectedFile, ...]


QWEN_SHARED = {
    "generation_config.json": SelectedFile(
        "generation_config.json", "20a8a9156fc8c3f25295ca067f61fdf120d517c5",
        239, "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "20a8a9156fc8c3f25295ca067f61fdf120d517c5",
        source_remote="gdrive:hf_cache/hub/models--Qwen--Qwen3-8B/blobs/20a8a9156fc8c3f25295ca067f61fdf120d517c5",
    ),
    "merges.txt": SelectedFile(
        "merges.txt", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",
        1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",
        source_remote="gdrive:hf_cache/hub/models--Qwen--Qwen3-8B/blobs/31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",
    ),
    "tokenizer.json": SelectedFile(
        "tokenizer.json", "cd71f61a15a522601badb3dc960d800d9cb3766c",
        11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "cd71f61a15a522601badb3dc960d800d9cb3766c",
        lfs_sha256="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        source_remote="gdrive:hf_cache/hub/models--Qwen--Qwen3-8B/blobs/aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    "tokenizer_config.json": SelectedFile(
        "tokenizer_config.json", "417d038a63fa3de29cfde265caedae14d1a58d92",
        9_732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "417d038a63fa3de29cfde265caedae14d1a58d92",
        source_remote="gdrive:hf_cache/hub/models--Qwen--Qwen3-8B/blobs/417d038a63fa3de29cfde265caedae14d1a58d92",
    ),
    "vocab.json": SelectedFile(
        "vocab.json", "4783fe10ac3adce15ac8f358ef5462739852c569",
        2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "4783fe10ac3adce15ac8f358ef5462739852c569",
        source_remote="gdrive:hf_cache/hub/models--Qwen--Qwen3-8B/blobs/4783fe10ac3adce15ac8f358ef5462739852c569",
    ),
}


QWEN4_PATHS = (
    ".gitattributes", "LICENSE", "README.md", "config.json",
    "generation_config.json", "merges.txt", "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json",
)
QWEN8_PATHS = (
    ".gitattributes", "LICENSE", "README.md", "config.json",
    "generation_config.json", "merges.txt", "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors", "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors", "model-00005-of-00005.safetensors",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json",
)
LLAMA_PATHS = (
    ".gitattributes", "LICENSE", "README.md", "USE_POLICY.md", "config.json",
    "generation_config.json", "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors", "model.safetensors.index.json",
    "original/consolidated.00.pth", "original/params.json",
    "original/tokenizer.model", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json",
)


ROLE_SPECS = {
    "qwen3-4b": RoleSpec(
        "qwen3-4b", "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c",
        False, QWEN4_PATHS,
        (
            SelectedFile(
                "config.json", "e49eccdc32f36da9c09cfa0e737084f9e0105e5e", 726,
                "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
                "e49eccdc32f36da9c09cfa0e737084f9e0105e5e",
                source_url="https://huggingface.co/Qwen/Qwen3-4B/raw/1cfa9a7208912126459214e8b04321603b3df60c/config.json",
            ),
            *(QWEN_SHARED[name] for name in (
                "generation_config.json", "merges.txt", "tokenizer.json",
                "tokenizer_config.json", "vocab.json",
            )),
        ),
    ),
    "qwen3-8b": RoleSpec(
        "qwen3-8b", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
        False, QWEN8_PATHS,
        (
            SelectedFile(
                "config.json", "d46195ac87f837ad233d02b2f80f148bf7c005e0", 728,
                "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
                "d46195ac87f837ad233d02b2f80f148bf7c005e0",
                source_remote="gdrive:hf_cache/hub/models--Qwen--Qwen3-8B/blobs/d46195ac87f837ad233d02b2f80f148bf7c005e0",
            ),
            *(QWEN_SHARED[name] for name in (
                "generation_config.json", "merges.txt", "tokenizer.json",
                "tokenizer_config.json", "vocab.json",
            )),
        ),
    ),
    "llama31-8b": RoleSpec(
        "llama31-8b", "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659", "manual", LLAMA_PATHS,
        (
            SelectedFile(
                "config.json", "0bb6fd75b3ad2fe988565929f329945262c2814e", 855,
                "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
                "0bb6fd75b3ad2fe988565929f329945262c2814e",
                source_remote="gdrive:hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/blobs/0bb6fd75b3ad2fe988565929f329945262c2814e",
            ),
            SelectedFile(
                "generation_config.json", "cc7276afd599de091142c6ed3005faf8a74aa257", 184,
                "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
                "cc7276afd599de091142c6ed3005faf8a74aa257",
                source_remote="gdrive:hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/blobs/cc7276afd599de091142c6ed3005faf8a74aa257",
            ),
            SelectedFile(
                "original/tokenizer.model", "a097ce5a06fce0fa3d685a8cfb175cef243dfde9",
                2_183_982, "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
                "a097ce5a06fce0fa3d685a8cfb175cef243dfde9",
                lfs_sha256="82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
                source_remote="gdrive:hf_cache_flat/meta-llama__Llama-3.1-8B-Instruct/original/tokenizer.model",
            ),
            SelectedFile(
                "special_tokens_map.json", "02ee80b6196926a5ad790a004d9efd6ab1ba6542", 296,
                "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
                "02ee80b6196926a5ad790a004d9efd6ab1ba6542",
                source_remote="gdrive:hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/blobs/02ee80b6196926a5ad790a004d9efd6ab1ba6542",
            ),
            SelectedFile(
                "tokenizer.json", "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8", 9_085_657,
                "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
                "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
                source_remote="gdrive:hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/blobs/5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
            ),
            SelectedFile(
                "tokenizer_config.json", "db88166e2bc4c799fd5d1ae643b75e84d03ee70e", 55_351,
                "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
                "db88166e2bc4c799fd5d1ae643b75e84d03ee70e",
                source_remote="gdrive:hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/blobs/db88166e2bc4c799fd5d1ae643b75e84d03ee70e",
            ),
        ),
    ),
}


def selected_allowlist(paths: tuple[str, ...]) -> tuple[str, ...]:
    values = []
    for value in paths:
        name = PurePosixPath(value).name
        if name in ALLOWLIST_NAMES or (
            name.startswith("sentencepiece") and name.endswith(".model")
        ) or (name.startswith("chat_template") and name.endswith(".jinja")):
            values.append(value)
    return _utf8_sorted(values)


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _official_api_url(spec: RoleSpec) -> str:
    return (
        f"https://huggingface.co/api/models/{spec.repository}/revision/"
        f"{spec.revision}?blobs=true"
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def https_get(url: str, *, accept: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "a6-tokenizer-restore-v1"},
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"BLOCKED_TOKENIZER_ACCESS: HTTP {error.code}: {url}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"BLOCKED_TOKENIZER_ACCESS: HTTP transport failed: {url}") from error
    with response:
        if response.status != 200 or response.geturl() != url:
            raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: unexpected HTTP response")
        body = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
    return body, headers


def validate_official_tree(spec: RoleSpec, raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except Exception as error:
        raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: invalid official JSON") from error
    if value.get("id") != spec.repository or value.get("sha") != spec.revision \
            or value.get("gated") != spec.gated:
        raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: official identity mismatch")
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: official tree missing")
    paths = []
    by_path = {}
    for row in siblings:
        if not isinstance(row, dict) or not isinstance(row.get("rfilename"), str):
            raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: official tree row invalid")
        path = unicodedata.normalize("NFC", row["rfilename"])
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or path in by_path:
            raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: official path invalid")
        paths.append(path)
        by_path[path] = row
    if _utf8_sorted(paths) != _utf8_sorted(spec.all_paths):
        raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: official tree changed")
    expected_selected = _utf8_sorted(item.path for item in spec.selected)
    if selected_allowlist(spec.all_paths) != expected_selected:
        raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: official allowlist changed")
    for expected in spec.selected:
        row = by_path[expected.path]
        if row.get("blobId") != expected.blob_id or row.get("size") != expected.size:
            raise RuntimeError(f"official selected object changed: {expected.path}")
        lfs = row.get("lfs")
        if expected.lfs_sha256 is None:
            if lfs is not None:
                raise RuntimeError(f"unexpected LFS object: {expected.path}")
        elif not isinstance(lfs, dict) or lfs.get("sha256") != expected.lfs_sha256 \
                or lfs.get("size") != expected.size:
            raise RuntimeError(f"official LFS object changed: {expected.path}")
    projection = {
        "id": spec.repository, "sha": spec.revision, "gated": spec.gated,
        "siblings": [
            {
                key: row[key] for key in ("rfilename", "blobId", "size", "lfs")
                if key in row
            }
            for row in sorted(siblings, key=lambda item: item["rfilename"].encode("utf-8"))
        ],
    }
    return projection


def verify_selected_bytes(spec: SelectedFile, payload: bytes) -> None:
    if len(payload) != spec.size or sha256_bytes(payload) != spec.sha256:
        raise RuntimeError(f"selected payload hash mismatch: {spec.path}")
    if spec.lfs_sha256 is None and git_blob_sha1(payload) != spec.git_blob_sha1:
        raise RuntimeError(f"selected Git blob mismatch: {spec.path}")
    if spec.lfs_sha256 is not None:
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{spec.lfs_sha256}\nsize {spec.size}\n"
        ).encode("ascii")
        if git_blob_sha1(pointer) != spec.git_blob_sha1:
            raise RuntimeError(f"selected LFS pointer mismatch: {spec.path}")


def rclone_cat(remote: str, *, rclone: str = "rclone") -> bytes:
    result = subprocess.run(
        [rclone, "cat", remote], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "BLOCKED_TOKENIZER_ACCESS: rclone cat failed: "
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise


def _assert_safe_tree_root(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("restore root must be a real directory")


def _publish_stage(stage: Path, out: Path) -> None:
    """Serialize publication and refuse replacement of an existing tree."""
    lock = out.with_name("." + out.name + ".publish.lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, canonical_json_bytes({"pid": os.getpid()}))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if out.exists() or out.is_symlink():
        raise FileExistsError("restore destination appeared during publication")
    os.rename(stage, out)
    lock.unlink()


def _tree_records(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("materialized subtree is not a real directory")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("materialized tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("materialized tree contains a non-regular file")
        payload = path.read_bytes()
        records.append({
            "path": path.relative_to(root).as_posix(), "size": len(payload),
            "sha256": sha256_bytes(payload),
        })
    return records


def verify_materialized(root: Path, manifest: dict[str, Any]) -> None:
    _assert_safe_tree_root(root)
    _assert_safe_tree_root(root / "materialized")
    _assert_safe_tree_root(root / "evidence")
    if set(manifest) != {
        "schema_version", "status", "roles", "cross_role_equalities",
        "materialized_sha256", "evidence_files", "evidence_sha256",
        "source_sha256", "runtime",
    }:
        raise RuntimeError("restore manifest schema mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION \
            or manifest.get("status") != "AUTHENTICATED_COMPLETE_ALL_THREE":
        raise RuntimeError("restore manifest identity mismatch")
    if manifest.get("source_sha256") != _source_hashes():
        raise RuntimeError("restore source changed")
    if not isinstance(manifest.get("roles"), dict) \
            or set(manifest["roles"]) != set(ROLE_ORDER):
        raise RuntimeError("restore manifest role roster mismatch")
    aggregate = []
    for role in ROLE_ORDER:
        role_root = root / "materialized" / role
        records = _tree_records(role_root)
        role_row = manifest["roles"][role]
        if set(role_row) != {
            "repository", "revision", "official_api_url",
            "official_raw_sha256", "official_tree_sha256", "transport", "files",
        }:
            raise RuntimeError(f"restore role schema changed: {role}")
        spec = ROLE_SPECS[role]
        if role_row["repository"] != spec.repository \
                or role_row["revision"] != spec.revision \
                or role_row["official_api_url"] != _official_api_url(spec) \
                or not _is_lower_hex(role_row["official_raw_sha256"], 64) \
                or not _is_lower_hex(role_row["official_tree_sha256"], 64):
            raise RuntimeError(f"restore role identity changed: {role}")
        expected = role_row["files"]
        if records != expected:
            raise RuntimeError(f"materialized role tree changed: {role}")
        expected_files = [
            {"path": row.path, "size": row.size, "sha256": row.sha256}
            for row in sorted(spec.selected, key=lambda row: row.path.encode("utf-8"))
        ]
        if records != expected_files:
            raise RuntimeError(f"materialized role differs from frozen objects: {role}")
        expected_transport = [
            {
                "path": row.path,
                "kind": "rclone_drive" if row.source_remote is not None else "official_https",
                "locator": row.source_remote or row.source_url,
            }
            for row in sorted(spec.selected, key=lambda row: row.path.encode("utf-8"))
        ]
        if role_row["transport"] != expected_transport:
            raise RuntimeError(f"restore transport record changed: {role}")
        for selected in spec.selected:
            verify_selected_bytes(selected, (role_root / selected.path).read_bytes())
        raw_official = (root / "evidence" / "official" / f"{role}.json").read_bytes()
        if sha256_bytes(raw_official) != role_row["official_raw_sha256"]:
            raise RuntimeError(f"official response changed: {role}")
        projection = validate_official_tree(spec, raw_official)
        if sha256_bytes(canonical_json_bytes(projection)) \
                != role_row["official_tree_sha256"]:
            raise RuntimeError(f"official projection changed: {role}")
        aggregate.append({
            "role": role,
            "tree_sha256": sha256_bytes(canonical_json_bytes(records)),
        })
    if sha256_bytes(canonical_json_bytes(aggregate)) != manifest["materialized_sha256"]:
        raise RuntimeError("materialized aggregate changed")
    evidence_records = _tree_records(root / "evidence")
    if evidence_records != manifest["evidence_files"] \
            or sha256_bytes(canonical_json_bytes(evidence_records)) \
            != manifest["evidence_sha256"]:
        raise RuntimeError("restore evidence tree changed")
    expected_equalities = [
        "generation_config.json", "merges.txt", "tokenizer.json",
        "tokenizer_config.json", "vocab.json",
    ]
    if manifest["cross_role_equalities"] != expected_equalities \
            or not isinstance(manifest["runtime"], dict):
        raise RuntimeError("restore aggregate metadata changed")
    for relative in expected_equalities:
        left = (root / "materialized" / "qwen3-4b" / relative).read_bytes()
        right = (root / "materialized" / "qwen3-8b" / relative).read_bytes()
        if left != right:
            raise RuntimeError(f"cross-Qwen payload changed: {relative}")
    allowed_top = {"materialized", "evidence", "CACHE_RESTORE_PROVENANCE.json"}
    if {path.name for path in root.iterdir()} != allowed_top:
        raise RuntimeError("restore root contains an unmanifested path")


def restore_all_three(out: str | Path, *, rclone: str = "rclone") -> dict[str, Any]:
    out = Path(out)
    stage = out.with_name("." + out.name + ".staging")
    if out.exists() or stage.exists():
        raise FileExistsError("restore requires absent final and staging roots")
    if out.parent.is_symlink() or not out.parent.is_dir():
        raise RuntimeError("restore parent must be a real existing directory")
    stage.mkdir(parents=True, exist_ok=False)
    official = {}
    for role in ROLE_ORDER:
        spec = ROLE_SPECS[role]
        raw, headers = https_get(_official_api_url(spec), accept="application/json")
        if headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: official API content type changed")
        official[role] = {
            "raw_sha256": sha256_bytes(raw),
            "projection": validate_official_tree(spec, raw),
            "content_type": headers.get("content-type"),
        }
        _write_exclusive(stage / "evidence" / "official" / f"{role}.json", raw)
    role_rows = {}
    payload_cache: dict[str, bytes] = {}
    for role in ROLE_ORDER:
        spec = ROLE_SPECS[role]
        files = []
        for selected in spec.selected:
            source_key = selected.source_remote or selected.source_url
            if source_key in payload_cache:
                payload = payload_cache[source_key]
            elif selected.source_remote is not None:
                payload = rclone_cat(selected.source_remote, rclone=rclone)
                payload_cache[source_key] = payload
            elif selected.source_url is not None:
                payload, headers = https_get(selected.source_url, accept="text/plain")
                if headers.get("etag") != '"e49eccdc32f36da9c09cfa0e737084f9e0105e5e"' \
                        or headers.get("x-repo-commit") != ROLE_SPECS["qwen3-4b"].revision \
                        or headers.get("content-type", "").split(";", 1)[0] != "text/plain" \
                        or headers.get("content-length") != str(selected.size):
                    raise RuntimeError("Qwen3-4B config HTTP provenance mismatch")
                _write_exclusive(
                    stage / "evidence" / "official" / "qwen3-4b-config-headers.json",
                    canonical_json_bytes({
                        "url": selected.source_url,
                        "content_length": headers["content-length"],
                        "content_type": headers["content-type"],
                        "etag": headers["etag"],
                        "x_repo_commit": headers["x-repo-commit"],
                    }),
                )
                payload_cache[source_key] = payload
            else:
                raise RuntimeError("selected file has no byte source")
            verify_selected_bytes(selected, payload)
            target = stage / "materialized" / role / selected.path
            _write_exclusive(target, payload)
            files.append({
                "path": selected.path, "size": len(payload),
                "sha256": sha256_bytes(payload),
            })
        files.sort(key=lambda item: item["path"].encode("utf-8"))
        role_rows[role] = {
            "repository": spec.repository, "revision": spec.revision,
            "official_api_url": _official_api_url(spec),
            "official_raw_sha256": official[role]["raw_sha256"],
            "official_tree_sha256": sha256_bytes(
                canonical_json_bytes(official[role]["projection"])
            ),
            "transport": [
                {
                    "path": row.path,
                    "kind": (
                        "rclone_drive" if row.source_remote is not None
                        else "official_https"
                    ),
                    "locator": row.source_remote or row.source_url,
                }
                for row in sorted(
                    spec.selected, key=lambda row: row.path.encode("utf-8")
                )
            ],
            "files": files,
        }
    equal_paths = (
        "generation_config.json", "merges.txt", "tokenizer.json",
        "tokenizer_config.json", "vocab.json",
    )
    for path in equal_paths:
        left = (stage / "materialized" / "qwen3-4b" / path).read_bytes()
        right = (stage / "materialized" / "qwen3-8b" / path).read_bytes()
        if left != right:
            raise RuntimeError(f"cross-Qwen payload differs: {path}")
    aggregate = [
        {
            "role": role,
            "tree_sha256": sha256_bytes(canonical_json_bytes(role_rows[role]["files"])),
        }
        for role in ROLE_ORDER
    ]
    evidence_files = _tree_records(stage / "evidence")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "AUTHENTICATED_COMPLETE_ALL_THREE",
        "roles": role_rows,
        "cross_role_equalities": list(equal_paths),
        "materialized_sha256": sha256_bytes(canonical_json_bytes(aggregate)),
        "evidence_files": evidence_files,
        "evidence_sha256": sha256_bytes(canonical_json_bytes(evidence_files)),
        "source_sha256": _source_hashes(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    _write_exclusive(stage / "CACHE_RESTORE_PROVENANCE.json", canonical_json_bytes(manifest))
    verify_materialized(stage, manifest)
    _publish_stage(stage, out)
    return manifest


def load_and_verify_restore(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    path = root / "CACHE_RESTORE_PROVENANCE.json"
    if root.is_symlink() or not root.is_dir() or path.is_symlink() or not path.is_file():
        raise RuntimeError("BLOCKED_TOKENIZER_ACCESS: authenticated restore is missing")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_object)
    if path.read_bytes() != canonical_json_bytes(value):
        raise RuntimeError("restore manifest is not canonical JSON")
    verify_materialized(root, value)
    return value


def remove_failed_stage(out: str | Path) -> None:
    """Remove only this restorer's explicit failed stage after inspection."""
    out = Path(out)
    stage = out.with_name("." + out.name + ".staging")
    if out.exists() or out.is_symlink() or not stage.is_dir() or stage.is_symlink():
        raise RuntimeError("refusing ambiguous failed-stage cleanup")
    shutil.rmtree(stage)
