"""Independent A/B and transitive rederivation gates for LEASH prep and fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .leash_contract import (
    LeashContractError,
    assert_no_symlinks,
    bound_tree_manifest,
    require_physically_disjoint_trees,
    write_json_noreplace,
)
from .leash_fit import derive_source_bound_fit_contract
from .leash_preparation import derive_source_bound_preparation_contract


def _require_exact_tree(actual: str | Path, expected: str | Path, *, name: str) -> dict[str, Any]:
    observed = bound_tree_manifest(actual, name=f"{name} observed tree")
    rederived = bound_tree_manifest(expected, name=f"{name} rederived tree")
    if observed != rederived:
        observed_files = {item["path"]: item for item in observed["files"]}
        expected_files = {item["path"]: item for item in rederived["files"]}
        drift = sorted(
            path for path in set(observed_files) | set(expected_files)
            if observed_files.get(path) != expected_files.get(path)
        )
        raise LeashContractError(f"{name} differs from independent rederivation: {drift[:8]}")
    return observed


def verify_leash_preparation_ab(
    *,
    source_root: str | Path,
    registry_path: str | Path,
    public_a: str | Path,
    private_a: str | Path,
    public_b: str | Path,
    private_b: str | Path,
    certificate_path: str | Path,
) -> dict[str, Any]:
    """Rebuild raw-derived prep state and require both copies to match exactly."""

    contract = derive_source_bound_preparation_contract(
        source_root=source_root, registry_path=registry_path
    )
    registry = contract["registry"]
    for name, path in (
        ("LEASH preparation A public", public_a),
        ("LEASH preparation B public", public_b),
        ("LEASH preparation A private", private_a),
        ("LEASH preparation B private", private_b),
    ):
        assert_no_symlinks(path, name=name)
    physical = require_physically_disjoint_trees(
        {
            "LEASH preparation A public": public_a,
            "LEASH preparation B public": public_b,
            "LEASH preparation A private": private_a,
            "LEASH preparation B private": private_b,
        }
    )
    a_public_tree = physical["LEASH preparation A public"]
    b_public_tree = physical["LEASH preparation B public"]
    a_private_tree = physical["LEASH preparation A private"]
    b_private_tree = physical["LEASH preparation B private"]
    for name, observed, expected in (
        ("LEASH preparation A public", a_public_tree, contract["public_tree"]),
        ("LEASH preparation B public", b_public_tree, contract["public_tree"]),
        ("LEASH preparation A private", a_private_tree, contract["private_tree"]),
        ("LEASH preparation B private", b_private_tree, contract["private_tree"]),
    ):
        if observed != expected:
            raise LeashContractError(f"{name} differs from independent current-source rederivation")
    if a_public_tree != b_public_tree or a_private_tree != b_private_tree:
        raise LeashContractError("LEASH preparation A/B trees are not byte-identical")
    certificate = contract["certificate"]
    write_json_noreplace(certificate_path, certificate)
    return certificate


def verify_leash_fit_ab(
    *,
    source_root: str | Path,
    registry_path: str | Path,
    preparation_a: str | Path,
    preparation_b: str | Path,
    preparation_ab_certificate: str | Path,
    fit_a: str | Path,
    fit_b: str | Path,
    certificate_path: str | Path,
) -> dict[str, Any]:
    """Rederive target-blind policy ledgers from both certified prep copies."""

    contract = derive_source_bound_fit_contract(
        source_root=source_root,
        registry_path=registry_path,
        preparation_ab_certificate=preparation_ab_certificate,
    )
    roots = {
        "LEASH preparation A": preparation_a,
        "LEASH preparation B": preparation_b,
        "LEASH fit A": fit_a,
        "LEASH fit B": fit_b,
    }
    physical = require_physically_disjoint_trees(roots)
    for name, path, expected in (
        ("LEASH preparation A", preparation_a, contract["preparation"]["public_tree"]),
        ("LEASH preparation B", preparation_b, contract["preparation"]["public_tree"]),
        ("LEASH fit A", fit_a, contract["fit_tree"]),
        ("LEASH fit B", fit_b, contract["fit_tree"]),
    ):
        assert_no_symlinks(path, name=name)
        if physical[name] != expected:
            raise LeashContractError(
                f"{name} differs from independent current-source rederivation"
            )
    certificate = contract["certificate"]
    write_json_noreplace(certificate_path, certificate)
    return certificate


__all__ = ["verify_leash_fit_ab", "verify_leash_preparation_ab"]
