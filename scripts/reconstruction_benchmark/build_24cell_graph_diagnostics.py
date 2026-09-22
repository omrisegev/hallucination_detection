#!/usr/bin/env python3
"""Build the audited frozen-24 graph-assumption diagnostic package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.graph_diagnostics import (  # noqa: E402
    DIAGNOSTIC_VERSION,
    MANIFEST_SCHEMA_VERSION,
    NODE_PERMUTATION_COUNT,
    assert_source_environment_snapshot_unchanged,
    build_graph_diagnostics,
    capture_source_environment_snapshot,
    verify_diagnostic_release,
)
import spectral_utils.reconstruction_benchmark.graph_diagnostics as diagnostic_module  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    canonical_tree_manifest,
    sha256_bytes,
    sha256_file,
)


DEFAULT_RELEASE_ROOT = REPO / "results" / "reconstruction_benchmark_v1" / "releases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <release>/graph_diagnostics; must not already exist",
    )
    return parser.parse_args()


PRODUCER_RELATIVE_PATH = Path(__file__).relative_to(REPO).as_posix()


def publish(
    output_dir: Path,
    verified,
    diagnostics,
    plot_arrays,
    example_arrays,
    source_snapshot,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"graph-diagnostic output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        diagnostics_path = temporary / "GRAPH_DIAGNOSTICS.json"
        plot_path = temporary / "PLOT_DATA.npz"
        examples_path = temporary / "EXAMPLE_GRAPH_DATA.npz"
        diagnostics_sha = atomic_write_json(diagnostics_path, diagnostics)
        plot_sha = atomic_write_npz(plot_path, plot_arrays)
        examples_sha = atomic_write_npz(examples_path, example_arrays)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "release_id": verified.release_id,
            "status": diagnostics["status"],
            "n_records": len(diagnostics["records"]),
            "n_source_bindings": len(diagnostics["source_bindings"]),
            "node_permutation_draws_per_cell_method": NODE_PERMUTATION_COUNT,
            "diagnostics_path": diagnostics_path.name,
            "diagnostics_sha256": diagnostics_sha,
            "diagnostics_payload_sha256": diagnostics["payload_sha256"],
            "plot_data_path": plot_path.name,
            "plot_data_sha256": plot_sha,
            "example_graph_data_path": examples_path.name,
            "example_graph_data_sha256": examples_sha,
            "selected_examples": diagnostics["example_selection"]["selected_cell_by_method"],
            "source_provenance": dict(verified.provenance),
            "source_environment_snapshot": dict(source_snapshot),
            "source_environment_snapshot_sha256": source_snapshot["snapshot_sha256"],
            "producer_path": PRODUCER_RELATIVE_PATH,
            "producer_sha256": sha256_file(Path(__file__)),
            "diagnostics_module_path": Path(diagnostic_module.__file__).relative_to(REPO).as_posix(),
            "diagnostics_module_sha256": sha256_file(Path(diagnostic_module.__file__)),
        }
        manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        atomic_write_json(temporary / "GRAPH_DIAGNOSTICS_MANIFEST.json", manifest)

        # Re-hash immediately before the atomic directory publication.
        if sha256_file(diagnostics_path) != diagnostics_sha:
            raise RuntimeError("diagnostics changed during publication")
        if sha256_file(plot_path) != plot_sha:
            raise RuntimeError("plot data changed during publication")
        if sha256_file(examples_path) != examples_sha:
            raise RuntimeError("example graph data changed during publication")
        tree = canonical_tree_manifest(temporary)
        atomic_write_json(temporary / "TREE_MANIFEST.json", tree)
        assert_source_environment_snapshot_unchanged(
            REPO,
            source_snapshot,
            extra_source_paths=(PRODUCER_RELATIVE_PATH,),
        )
        os.replace(temporary, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    source_snapshot = capture_source_environment_snapshot(
        REPO,
        extra_source_paths=(PRODUCER_RELATIVE_PATH,),
    )
    release = (args.release_root / args.release_id).resolve()
    output = (args.output_dir or release / "graph_diagnostics").resolve()
    if output.parent != release:
        raise RuntimeError("graph diagnostics must be an immediate subdirectory of the scientific release")
    verified = verify_diagnostic_release(release)
    diagnostics, plot_arrays, example_arrays = build_graph_diagnostics(
        verified,
        producer_snapshot=source_snapshot,
    )
    manifest = publish(
        output,
        verified,
        diagnostics,
        plot_arrays,
        example_arrays,
        source_snapshot,
    )
    print(json.dumps({
        "status": manifest["status"],
        "n_records": manifest["n_records"],
        "n_source_bindings": manifest["n_source_bindings"],
        "node_permutation_draws_per_cell_method": manifest["node_permutation_draws_per_cell_method"],
        "selected_examples": manifest["selected_examples"],
        "output_dir": str(output),
        "manifest_payload_sha256": manifest["payload_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
