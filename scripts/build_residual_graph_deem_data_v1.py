#!/usr/bin/env python3
"""Build target-free bundles or, after score freeze, evaluation sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256  # noqa: E402
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    build_target_free_cell,
    load_registry,
    load_target_free_bundle,
    write_target_free_bundle,
)


DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"


def selected_cells(registry, requested: str | None) -> list[str]:
    roster = [cell["cell_id"] for cell in registry["cells"]]
    if not requested:
        return roster
    values = [value.strip() for value in requested.split(",") if value.strip()]
    unknown = sorted(set(values) - set(roster))
    if unknown:
        raise SystemExit("unknown cells: " + ", ".join(unknown))
    return values


def build_bundles(args) -> None:
    registry = load_registry(args.registry)
    cells = selected_cells(registry, args.cells)
    out = Path(args.out_dir).resolve()
    manifests = []
    for cell_id in cells:
        bundle, _ = build_target_free_cell(
            args.repo_root, registry, cell_id, source_root=args.source_root
        )
        path = out / "bundles" / f"{cell_id}.npz"
        manifest = write_target_free_bundle(path, bundle)
        manifests.append(manifest)
        print(f"{cell_id}: rows={len(bundle.row_ids)} p={len(bundle.feature_names)} {path}", flush=True)
    aggregate = {
        "schema": "residual_graph_deem_bundle_set_v1",
        "status": "complete" if len(cells) == 24 else "partial",
        "labels_accessed": False,
        "cells": cells,
        "n_rows": sum(item["n_rows"] for item in manifests),
        "bundle_manifests": manifests,
        "registry_content_sha256": registry["registry_content_sha256"],
    }
    aggregate["content_sha256"] = canonical_sha256(aggregate)
    atomic_write_json(out / "TARGET_FREE_BUNDLES.json", aggregate)


def build_sidecars(args) -> None:
    # Evaluation-only import: Stage-A code never imports this module.
    from spectral_utils.residual_graph_deem_labels import (
        build_label_sidecar,
        require_complete_score_freeze,
        write_label_sidecar,
    )

    registry = load_registry(args.registry)
    cells = selected_cells(registry, args.cells)
    require_complete_score_freeze(args.score_freeze_manifest, cells)
    bundle_root = Path(args.bundle_dir).resolve()
    out = Path(args.out_dir).resolve()
    manifests = []
    for cell_id in cells:
        frozen = load_target_free_bundle(bundle_root / f"{cell_id}.npz")
        rebuilt, identities = build_target_free_cell(
            args.repo_root, registry, cell_id, source_root=args.source_root
        )
        if (
            frozen.row_ids != rebuilt.row_ids
            or frozen.inventory_sha256 != rebuilt.inventory_sha256
            or frozen.source_sha256 != rebuilt.source_sha256
        ):
            raise RuntimeError(f"sidecar rebuild differs from Stage-A bundle: {cell_id}")
        sidecar = build_label_sidecar(frozen, identities)
        path = out / f"{cell_id}.npz"
        manifests.append(write_label_sidecar(path, sidecar))
        print(f"{cell_id}: labels={len(sidecar.row_ids)} {path}", flush=True)
    aggregate = {
        "schema": "residual_graph_deem_label_sidecar_set_v1",
        "status": "complete" if len(cells) == 24 else "partial",
        "cells": cells,
        "n_rows": sum(item["n_rows"] for item in manifests),
        "sidecar_manifests": manifests,
        "score_freeze_manifest": str(Path(args.score_freeze_manifest).resolve()),
    }
    aggregate["content_sha256"] = canonical_sha256(aggregate)
    atomic_write_json(out / "LABEL_SIDECARS.json", aggregate)


def finalize_bundles(args) -> None:
    registry = load_registry(args.registry)
    cells = [cell["cell_id"] for cell in registry["cells"]]
    bundle_root = Path(args.bundle_dir).resolve()
    manifests = []
    for registered in registry["cells"]:
        cell_id = registered["cell_id"]
        path = bundle_root / f"{cell_id}.npz"
        bundle = load_target_free_bundle(path)
        if (
            bundle.cell_id != cell_id
            or len(bundle.row_ids) != int(registered["n_rows"])
            or bundle.inventory_sha256 != registered["inventory_sha256"]
            or bundle.source_sha256 != registered["source"]["source_sha256"]
        ):
            raise RuntimeError(f"bundle/registry mismatch during finalization: {cell_id}")
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bundle_sha256") != bundle.bundle_sha256:
            raise RuntimeError(f"bundle manifest hash mismatch: {cell_id}")
        manifests.append(manifest)
    aggregate = {
        "schema": "residual_graph_deem_bundle_set_v1",
        "status": "complete", "labels_accessed": False, "cells": cells,
        "n_rows": sum(item["n_rows"] for item in manifests),
        "bundle_manifests": manifests,
        "registry_content_sha256": registry["registry_content_sha256"],
    }
    aggregate["content_sha256"] = canonical_sha256(aggregate)
    atomic_write_json(Path(args.out).resolve(), aggregate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    common.add_argument("--repo-root", type=Path, default=ROOT)
    common.add_argument("--source-root", type=Path)
    common.add_argument("--cells", help="comma-separated subset; omission means all 24")

    bundles = subparsers.add_parser("bundles", parents=[common])
    bundles.add_argument("--out-dir", type=Path, required=True)
    bundles.set_defaults(func=build_bundles)

    sidecars = subparsers.add_parser("sidecars", parents=[common])
    sidecars.add_argument("--bundle-dir", type=Path, required=True)
    sidecars.add_argument("--score-freeze-manifest", type=Path, required=True)
    sidecars.add_argument("--out-dir", type=Path, required=True)
    sidecars.set_defaults(func=build_sidecars)

    finalize = subparsers.add_parser("finalize-bundles")
    finalize.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    finalize.add_argument("--bundle-dir", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)
    finalize.set_defaults(func=finalize_bundles)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
