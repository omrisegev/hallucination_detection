#!/usr/bin/env python3
"""Privileged source guard: return only target-free LEASH contract material."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types


CAPSULE = Path(__file__).resolve().parents[2]


def _package(name: str, relative: str, **attributes: object) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(CAPSULE / relative)]
    module.__package__ = name
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _module(name: str, relative: str) -> types.ModuleType:
    path = CAPSULE / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load authenticated LEASH capsule module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_authenticated_capsule() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Load only the controller-copied, hash-authenticated source-guard closure."""

    _package("spectral_utils", "spectral_utils")
    _package(
        "spectral_utils.paper_exact",
        "spectral_utils/paper_exact",
        FIDELITY_LABELS=(
            "official-exact", "paper-specified", "paper-specified-partial",
            "adapted-common-protocol", "published-context-only", "blocked-assets",
        ),
        SCHEMA_VERSION="paper_exact_acquisition_v1",
    )
    _package("spectral_utils.fair_comparisons", "spectral_utils/fair_comparisons")
    _package(
        "spectral_utils.reconstruction_benchmark",
        "spectral_utils/reconstruction_benchmark",
    )
    io_module = _module(
        "spectral_utils.reconstruction_benchmark.io",
        "spectral_utils/reconstruction_benchmark/io.py",
    )
    _module(
        "spectral_utils.paper_exact.evaluator",
        "spectral_utils/paper_exact/evaluator.py",
    )
    _module(
        "spectral_utils.paper_exact.leash", "spectral_utils/paper_exact/leash.py"
    )
    _module(
        "spectral_utils.paper_exact.manifest",
        "spectral_utils/paper_exact/manifest.py",
    )
    _module(
        "spectral_utils.fair_comparisons.registry",
        "spectral_utils/fair_comparisons/registry.py",
    )
    _module(
        "spectral_utils.fair_comparisons.stopping",
        "spectral_utils/fair_comparisons/stopping.py",
    )
    contract_module = _module(
        "spectral_utils.reconstruction_benchmark.leash_contract",
        "spectral_utils/reconstruction_benchmark/leash_contract.py",
    )
    preparation_module = _module(
        "spectral_utils.reconstruction_benchmark.leash_preparation",
        "spectral_utils/reconstruction_benchmark/leash_preparation.py",
    )
    return io_module, contract_module, preparation_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    io_module, contract_module, preparation_module = _load_authenticated_capsule()
    contract = preparation_module.derive_source_bound_preparation_contract(
        source_root=args.source_root, registry_path=args.registry
    )
    if "outcome_rows" in contract or "answer_text" in contract:
        raise RuntimeError("privileged LEASH source guard attempted to return outcomes")
    contract_module.assert_no_forbidden_keys(contract["fit_rows"])
    contract["source_guard"] = {
        "process_isolated": True,
        "target_fields_returned": False,
        "private_outcome_rows_returned": False,
        "authenticated_code_capsule": True,
        "controller_independent_rederivation": True,
        "code_closure_sha256": contract_module.source_guard_closure_sha256(
            contract["registry"]
        ),
    }
    # This process is the privileged current-source trust anchor.  Return the
    # contract only through the controller-owned stdout pipe: a pathname output
    # could be exchanged after process completion and before the controller
    # opens it.  The controller also verifies these exact canonical bytes.
    sys.stdout.buffer.write(io_module.canonical_json_bytes(contract) + b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
