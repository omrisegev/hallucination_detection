#!/usr/bin/env python3
"""Fit fixed mechanism controls after the development finalist is selected."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.family_residual_graph_liu_fit import (  # noqa: E402
    DEFAULT_OUT,
    VERSION,
    canonical_hash,
    graph_pairs,
    sha256_file,
    write_json,
)
from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    DUFS_EPOCHS,
    DUFS_SEEDS,
    family,
    load_contract,
    validate_bundle_without_labels,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    ContributionSpace,
    cardinality_balanced_contribution_score,
    fit_contribution_transform,
)
from spectral_utils.family_residual_graph import (  # noqa: E402
    build_family_graphs,
    contribution_laplacian_path,
    diffuse_score_path,
    fit_family_residual_state,
    graphs_from_coordinates,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    dufs_soft_gates,
    laplacian_iu_path,
    permute_graph,
)


ARMS = (
    "iu", "cardinality", "selected", "same_graph_u2", "same_graph_cs",
    "dufs_same_readout", "contribution_graph", "row_permuted_residual_graph",
    "node_permuted_graph", "random_family_graph", "length_only_graph",
    "baseline_only_graph", "residual_only_graph", "score_diffusion",
    "mutual_graph",
)


def verify_selection(path):
    selection = json.loads(Path(path).read_text())
    payload = dict(selection)
    recorded = payload.pop("selection_hash")
    if canonical_hash(payload) != recorded:
        raise RuntimeError("frozen selection is not self-consistent")
    return selection


def verify_development_freeze(fit_dir, bundle):
    # Local import avoids a module cycle: the report imports fit constants,
    # while the fixed-control phase runs only after report selection exists.
    from scripts.family_residual_graph_liu_report import verify_and_freeze

    definition = json.loads((fit_dir / "RUN_DEFINITION.json").read_text())
    if Path(definition["bundle"]).resolve() != Path(bundle).resolve():
        raise RuntimeError("control bundle path differs from development bundle")
    verify_and_freeze(fit_dir, bundle)


def seed_for(cell, suffix):
    payload = hashlib.sha256(f"{cell}:{suffix}".encode()).digest()
    return int.from_bytes(payload[:8], "little")


def random_family_state(F, state, rng):
    sizes = [
        len(state.contribution_space.members[name])
        for name in state.contribution_space.families
    ]
    order = rng.permutation(F.shape[0])
    groups, start = {}, 0
    for name, size in zip(state.contribution_space.families, sizes):
        groups[name] = np.sort(order[start:start + size])
        start += size
    contributions = np.column_stack([
        state.baseline_fit.w[index] @ F[index] for index in groups.values()
    ])
    space = ContributionSpace(
        families=state.contribution_space.families,
        members=groups,
        baseline_score=state.contribution_space.baseline_score,
        contributions=contributions,
        diagnostics={"control": "cardinality_matched_random_partition"},
    )
    transform = fit_contribution_transform(space, np.arange(F.shape[1]))
    _, residuals = transform.apply(space.baseline_score, space.contributions)
    return replace(state, residuals=residuals)


def readout(F, state, W, config, kind=None):
    kind = config["readout"] if kind is None else kind
    lambda_ = float(config["lambda"])
    if kind == "u2":
        result = laplacian_iu_path(F, (lambda_,), graph=W)[lambda_]
        raw = np.asarray(result.w @ F, dtype=float)
        return (
            raw - state.transform.baseline_mean
        ) / state.transform.baseline_scale
    if kind == "cs":
        factor = float(config.get("trust_factor", 1.0))
        cap = factor / state.residuals.shape[1]
        return contribution_laplacian_path(
            state.baseline, state.residuals, W, (lambda_,), trust_caps=(cap,)
        )[(lambda_, cap)].score
    raise ValueError(kind)


def fit_controls(data, cell, config):
    F, names = load_contract(data, cell, "mixed_v2")
    state = fit_family_residual_state(F, names)
    gates, _ = dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
    graph_args = dict(
        eta=float(config["eta"]), beta=float(config["beta"]),
        ks=(int(config["k"]),),
        topology=str(config["topology"]),
        scale_seed=1729,
    )
    selected = build_family_graphs(F, gates, state, **graph_args)[config["k"]].graph
    contribution = build_family_graphs(
        F, gates, state, family_mode="contribution", **graph_args
    )[config["k"]].graph
    rng = np.random.default_rng(seed_for(cell, "controls"))
    permuted_R = state.residuals.copy()
    for column in range(permuted_R.shape[1]):
        permuted_R[:, column] = permuted_R[
            rng.permutation(len(permuted_R)), column
        ]
    row_permuted = build_family_graphs(
        F, gates, replace(state, residuals=permuted_R), **graph_args
    )[config["k"]].graph
    random_state = random_family_state(F, state, rng)
    random_family = build_family_graphs(
        F, gates, random_state, **graph_args
    )[config["k"]].graph
    mutual = build_family_graphs(
        F, gates, state,
        eta=float(config["eta"]), beta=float(config["beta"]),
        ks=(int(config["k"]),), topology="mutual",
        scale_seed=graph_args["scale_seed"],
    )[config["k"]].graph
    node_permuted = permute_graph(selected, rng.permutation(F.shape[1]))
    dufs = build_family_graphs(
        F, gates, state, eta=0.0, beta=0.5, ks=(config["k"],),
        topology=config["topology"],
    )[config["k"]].graph
    baseline_only = build_family_graphs(
        F, gates, state, eta=1.0, beta=1.0, ks=(config["k"],),
        topology=config["topology"],
    )[config["k"]].graph
    residual_only = build_family_graphs(
        F, gates, state, eta=1.0, beta=0.0, ks=(config["k"],),
        topology=config["topology"],
    )[config["k"]].graph
    length_rows = [i for i, name in enumerate(names) if name == "trace_length"]
    if len(length_rows) > 1:
        raise RuntimeError(f"trace_length duplicated in {cell}")
    length_graph = (
        graphs_from_coordinates(
            F[length_rows].T, (config["k"],), topology=config["topology"]
        )[config["k"]]
        if length_rows else None
    )
    cardinality = cardinality_balanced_contribution_score(
        state.contribution_space, state.baseline_fit.w
    )
    scores = {
        "iu": state.baseline,
        "cardinality": cardinality.score,
        "selected": readout(F, state, selected, config),
        "same_graph_u2": readout(F, state, selected, config, "u2"),
        "same_graph_cs": readout(F, state, selected, config, "cs"),
        "dufs_same_readout": readout(F, state, dufs, config),
        "contribution_graph": readout(F, state, contribution, config),
        "row_permuted_residual_graph": readout(F, state, row_permuted, config),
        "node_permuted_graph": readout(F, state, node_permuted, config),
        "random_family_graph": readout(F, state, random_family, config),
        "length_only_graph": (
            readout(F, state, length_graph, config)
            if length_graph is not None else np.full(F.shape[1], np.nan)
        ),
        "baseline_only_graph": readout(F, state, baseline_only, config),
        "residual_only_graph": readout(F, state, residual_only, config),
        "score_diffusion": diffuse_score_path(
            state.baseline, selected, (float(config["lambda"]),)
        )[float(config["lambda"])],
        "mutual_graph": readout(F, state, mutual, config),
        "sample_index": np.arange(F.shape[1]),
    }
    if set(scores) != set(ARMS) | {"sample_index"}:
        raise RuntimeError("control score registry mismatch")
    return scores


def fit_phase(args, selection):
    out = args.fit_dir / "controls"
    out.mkdir(exist_ok=True)
    score_dir = out / "scores"
    score_dir.mkdir(exist_ok=True)
    if (out / "FIT_MANIFEST.json").exists() or any(score_dir.glob("*.npz")):
        raise FileExistsError(
            "fixed control artifacts already exist; do not overwrite the lineage"
        )
    hashes = {}
    reconstruction_errors = {}
    with np.load(args.bundle, allow_pickle=True) as data:
        validate_bundle_without_labels(data)
        for index, cell in enumerate(INSCOPE, 1):
            print(f"[{index}/{len(INSCOPE)}] controls {cell}", flush=True)
            path = score_dir / f"{cell}.npz"
            scores = fit_controls(data, cell, selection["selected_config"])
            with np.load(args.fit_dir / "scores" / f"{cell}.npz") as frozen:
                frozen_selected = np.asarray(
                    frozen[selection["selected_key"]], dtype=float
                )
                error = float(np.max(np.abs(
                    scores["selected"] - frozen_selected
                )))
            if error > 1e-10:
                raise RuntimeError(
                    f"selected-score reconstruction failed for {cell}: {error:.3e}"
                )
            reconstruction_errors[cell] = error
            # The primary arm is the byte-frozen development score.  Rebuilt
            # same-graph arms remain separate controls; tiny kNN tie changes
            # must not silently redefine the selected estimator.
            scores["selected"] = frozen_selected
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **scores)
            temporary.replace(path)
            hashes[cell] = sha256_file(path)
    manifest = {
        "version": VERSION,
        "phase": "fixed_control_fit_without_labels",
        "selection_hash": selection["selection_hash"],
        "bundle_sha256": sha256_file(args.bundle),
        "selection_file_sha256": sha256_file(
            args.fit_dir / "FROZEN_SELECTION.json"
        ),
        "controls_script_sha256": sha256_file(Path(__file__)),
        "arms": list(ARMS),
        "score_hashes": hashes,
        "selected_reconstruction_errors": reconstruction_errors,
        "labels_used": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(out / "FIT_MANIFEST.json", manifest)


def report_phase(args, selection):
    out = args.fit_dir / "controls"
    manifest = json.loads((out / "FIT_MANIFEST.json").read_text())
    manifest_payload = dict(manifest)
    recorded_manifest_hash = manifest_payload.pop("manifest_hash")
    if canonical_hash(manifest_payload) != recorded_manifest_hash:
        raise RuntimeError("control fit manifest is not self-consistent")
    if manifest.get("version") != VERSION or manifest.get("labels_used") is not False:
        raise RuntimeError("control fit manifest version/label boundary failed")
    if manifest["selection_hash"] != selection["selection_hash"]:
        raise RuntimeError("control selection mismatch")
    if manifest["bundle_sha256"] != sha256_file(args.bundle):
        raise RuntimeError("control bundle changed")
    if manifest["selection_file_sha256"] != sha256_file(
        args.fit_dir / "FROZEN_SELECTION.json"
    ):
        raise RuntimeError("control selection file changed")
    if manifest["controls_script_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("control script changed after fit")
    for cell, expected in manifest["score_hashes"].items():
        if sha256_file(out / "scores" / f"{cell}.npz") != expected:
            raise RuntimeError(f"control score changed: {cell}")
    by_arm = {arm: {} for arm in ARMS}
    cell_rows = []
    with np.load(args.bundle, allow_pickle=True) as data:
        for cell in INSCOPE:
            labels = np.asarray(data[f"{cell}__labels"], dtype=int)
            if int(labels.sum()) < 20:
                continue
            with np.load(out / "scores" / f"{cell}.npz") as scores:
                iu_auc = roc_auc_score(labels, scores["iu"])
                row = {"cell": cell, "dataset_family": family(cell)}
                for arm in ARMS:
                    arm_score = np.asarray(scores[arm], dtype=float)
                    if not np.isfinite(arm_score).all():
                        row[f"{arm}_auroc"] = ""
                        row[f"{arm}_delta_pp"] = ""
                        continue
                    value = float(roc_auc_score(labels, arm_score))
                    row[f"{arm}_auroc"] = value
                    row[f"{arm}_delta_pp"] = 100 * (value - iu_auc)
                    by_arm[arm].setdefault(family(cell), []).append(value - iu_auc)
                cell_rows.append(row)
    summary = {}
    for arm in ARMS:
        family_values = [np.mean(values) for values in by_arm[arm].values()]
        summary[arm] = {
            "equal_family_delta_pp": 100 * float(np.mean(family_values)),
            "positive_families": int(np.sum(np.asarray(family_values) > 0)),
            "worst_family_pp": 100 * float(np.min(family_values)),
            "evaluated_families": len(family_values),
            "evaluated_cells": int(sum(len(values) for values in by_arm[arm].values())),
        }
    write_json(out / "RESULT.json", {
        "version": VERSION, "selection_hash": selection["selection_hash"],
        "arms": summary,
    })
    fields = list(cell_rows[0])
    with (out / "cell_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(cell_rows)
    lines = ["# Family-residual graph LIU — fixed controls", ""]
    for arm, result in summary.items():
        lines.append(
            f"- `{arm}`: {result['equal_family_delta_pp']:+.3f}pp; "
            f"{result['positive_families']}/8 positive; worst "
            f"{result['worst_family_pp']:+.3f}pp"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "report", "both"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    verify_development_freeze(args.fit_dir, args.bundle)
    selection = verify_selection(args.fit_dir / "FROZEN_SELECTION.json")
    if args.phase in {"fit", "both"}:
        fit_phase(args, selection)
    if args.phase in {"report", "both"}:
        report_phase(args, selection)


if __name__ == "__main__":
    main()
