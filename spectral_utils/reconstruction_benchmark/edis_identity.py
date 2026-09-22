"""Thin EDIS adapter over the shared external keyed-HMAC identity API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .external_final_answer import (
    ID_CONTRACT_VERSION,
    ID_DIGEST_ALGORITHM,
    IDENTITY_KEY_BYTES,
    IDENTITY_KEY_CONTRACT_VERSION,
    OPAQUE_GROUP_ID_PREFIX,
    OPAQUE_ROW_ID_PREFIX,
    identity_key_id,
    keyed_opaque_external_id,
    load_identity_key,
)
from .io import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True)
class SharedEdisIdentityController:
    """Controller holding the sealed key; only its binding crosses into fit."""

    identity_key: bytes

    def __post_init__(self) -> None:
        if len(bytes(self.identity_key)) != IDENTITY_KEY_BYTES:
            raise ValueError("EDIS identity key length disagrees with the shared contract")

    @property
    def public_binding(self) -> Mapping[str, Any]:
        """Fit-safe row-only binding; it deliberately says nothing about groups."""

        binding: dict[str, Any] = {
            "contract_version": ID_CONTRACT_VERSION,
            "digest_algorithm": ID_DIGEST_ALGORITHM,
            "identity_key_contract_version": IDENTITY_KEY_CONTRACT_VERSION,
            "identity_key_bytes": IDENTITY_KEY_BYTES,
            "opaque_row_id_prefix": OPAQUE_ROW_ID_PREFIX,
            "canonical_row_order": "lexicographic_opaque_row_id",
            "row_namespace_scope": "dataset_temperature_cell",
            "key_id": identity_key_id(self.identity_key),
        }
        binding["contract_sha256"] = sha256_bytes(canonical_json_bytes(binding))
        return binding

    @property
    def private_identity_binding(self) -> Mapping[str, Any]:
        """Controller-only full binding used by post-freeze group reconstruction."""

        binding: dict[str, Any] = {
            **dict(self.public_binding),
            "opaque_group_id_prefix": OPAQUE_GROUP_ID_PREFIX,
            "group_namespace_scope": "dataset_question_content_postfreeze_only",
            "group_raw_identity": "dataset_id_plus_sha256_of_saved_question_text",
        }
        binding.pop("contract_sha256")
        binding["contract_sha256"] = sha256_bytes(canonical_json_bytes(binding))
        return binding

    @property
    def private_identity_commitment_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.private_identity_binding))

    def row_id(self, *, namespace: Mapping[str, str], raw_identity: str) -> str:
        return keyed_opaque_external_id(
            identity_key=self.identity_key,
            kind="row",
            namespace=namespace,
            raw=raw_identity,
        )

    def group_id(self, *, namespace: Mapping[str, str], raw_identity: str) -> str:
        return keyed_opaque_external_id(
            identity_key=self.identity_key,
            kind="group",
            namespace=namespace,
            raw=raw_identity,
        )


def controller_key_path(*, private_control_root: str | Path, release_id: str) -> Path:
    return Path(private_control_root) / release_id / "external-id-v2.key"


def load_edis_identity_controller(
    *,
    private_control_root: str | Path,
    release_id: str,
    create: bool,
    release_root: str | Path | None = None,
    repo: str | Path | None = None,
) -> SharedEdisIdentityController:
    """Load/create the shared-format key while enforcing controller isolation."""

    key_path = controller_key_path(
        private_control_root=private_control_root, release_id=release_id
    ).resolve()
    if release_root is not None:
        releases = Path(release_root).resolve()
        try:
            key_path.relative_to(releases)
        except ValueError:
            pass
        else:
            raise ValueError("EDIS identity key must be outside every release/fit tree")
    if repo is not None:
        repository = Path(repo).resolve()
        try:
            key_path.relative_to(repository)
        except ValueError:
            pass
        else:
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", str(key_path)],
                cwd=repository,
                check=False,
            )
            if ignored.returncode != 0:
                raise RuntimeError("in-repository EDIS controller key path is not git-ignored")
    return SharedEdisIdentityController(load_identity_key(key_path, create=create))


__all__ = [
    "SharedEdisIdentityController",
    "controller_key_path",
    "load_edis_identity_controller",
]
