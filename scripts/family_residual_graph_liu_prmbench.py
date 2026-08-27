#!/usr/bin/env python3
"""Frozen external-to-development PRMBench audit for graph LIU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.family_residual_graph_liu_controls import readout  # noqa: E402
from scripts.family_residual_graph_liu_fit import (  # noqa: E402
    DEFAULT_OUT as DEVELOPMENT_OUT,
    VERSION,
    canonical_hash,
    graph_pairs,
    sha256_file,
    write_json,
)
from scripts.family_residual_graph_liu_report import (  # noqa: E402
    DEFAULT_KEY,
    verify_and_freeze,
)
from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DUFS_EPOCHS,
    DUFS_SEEDS,
)
from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
)
from scripts.neutral_residual_mode_prmbench_confirmation import (  # noqa: E402
    DEFAULT_RAW,
    ordered_eligible_rows,
    telemetry_payload,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_contribution_score,
)
from spectral_utils.family_residual_graph import (  # noqa: E402
    build_family_graphs,
    fit_family_residual_state,
)
from spectral_utils.laplacian_upcr import dufs_soft_gates  # noqa: E402


TRANSFER_VERSION = "family-residual-graph-liu-prmbench-v3-2026-08-23"
DEFAULT_OUT = REPO / "results" / "family_residual_graph_liu_prmbench_v3"
DEFAULT_NRM = (
    REPO / "results" / "neutral_residual_mode_prmbench_v1"
    / "FROZEN_SCORES.npz"
)
BOOTSTRAPS = 5000


def verify_development(development):
    definition = json.loads((development / "RUN_DEFINITION.json").read_text())
    _, freeze = verify_and_freeze(development, Path(definition["bundle"]))
    selection_path = development / "FROZEN_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    payload = dict(selection)
    recorded = payload.pop("selection_hash")
    if canonical_hash(payload) != recorded:
        raise RuntimeError("development selection is not self-consistent")
    if selection["score_freeze_hash"] != freeze["freeze_hash"]:
        raise RuntimeError("development selection/freeze mismatch")
    return selection, selection_path


def candidate_score(F, names, state, gates, config):
    pair = (float(config["eta"]), float(config["beta"]))
    W = build_family_graphs(
        F, gates, state, eta=pair[0], beta=pair[1],
        ks=(int(config["k"]),),
        topology=str(config["topology"]),
        scale_seed=1729,
    )[int(config["k"])].graph
    return readout(F, state, W, config)


def score_phase(args):
    args.out.mkdir(parents=True, exist_ok=False)
    selection, selection_path = verify_development(args.development)
    config_path = args.development / "CONFIG_INDEX.json"
    configs = json.loads(config_path.read_text())
    with args.raw.open("rb") as handle:
        cache = pickle.load(handle)
    selected = ordered_eligible_rows(cache)
    telemetry = [telemetry_payload(row) for _, _, _, row in selected]
    F, names, availability, contract = mixed_v2_matrix(telemetry)
    state = fit_family_residual_state(F, names)
    gates, _ = dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
    final_config = selection["selected_config"]
    default_config = configs[DEFAULT_KEY]
    final = candidate_score(F, names, state, gates, final_config)
    default = candidate_score(F, names, state, gates, default_config)
    cardinality = cardinality_balanced_contribution_score(
        state.contribution_space, state.baseline_fit.w
    ).score
    scores = {
        "row_keys": np.asarray([row[0] for row in selected]),
        "row_ids": np.asarray([row[1] for row in selected]),
        "source_ids": np.asarray([row[2] for row in selected]),
        "iu": state.baseline,
        "finalist": final,
        "fixed_default": default,
        "cardinality": cardinality,
    }
    score_path = args.out / "FROZEN_SCORES.npz"
    np.savez_compressed(score_path, **scores)
    manifest = {
        "version": TRANSFER_VERSION,
        "development_version": VERSION,
        "phase": "telemetry_only_frozen_transfer_fit",
        "selection_hash": selection["selection_hash"],
        "selected_key": selection["selected_key"],
        "fixed_default_key": DEFAULT_KEY,
        "n": len(selected),
        "feature_names": list(names),
        "availability": availability,
        "contract": contract,
        "labels_used": False,
        "target_fields_received_by_fusion": [],
        "hashes": {
            "raw": sha256_file(args.raw),
            "selection": sha256_file(selection_path),
            "config_index": sha256_file(config_path),
            "scores": sha256_file(score_path),
            "nrm_comparator": sha256_file(args.nrm),
            "transfer_script": sha256_file(Path(__file__)),
        },
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(args.out / "FIT_MANIFEST.json", manifest)
    print(json.dumps({
        "phase": manifest["phase"], "n": manifest["n"],
        "labels_used": False, "manifest_hash": manifest["manifest_hash"],
    }, indent=2))


def bootstrap(y, scores, source_ids):
    unique, inverse = np.unique(source_ids, return_inverse=True)
    rng = np.random.default_rng(20260823)
    draws = {name: np.empty(BOOTSTRAPS) for name in scores if name != "iu"}
    probability = np.full(len(unique), 1 / len(unique))
    for draw in range(BOOTSTRAPS):
        counts = rng.multinomial(len(unique), probability)
        weights = counts[inverse]
        base = roc_auc_score(y, scores["iu"], sample_weight=weights)
        for name in draws:
            draws[name][draw] = (
                roc_auc_score(y, scores[name], sample_weight=weights) - base
            )
    return {
        name: {
            "low_pp": 100 * float(np.quantile(values, .025)),
            "median_pp": 100 * float(np.quantile(values, .5)),
            "high_pp": 100 * float(np.quantile(values, .975)),
            "probability_positive": float(np.mean(values > 0)),
        }
        for name, values in draws.items()
    }, draws


def report_phase(args):
    manifest = json.loads((args.out / "FIT_MANIFEST.json").read_text())
    payload = dict(manifest)
    recorded_manifest_hash = payload.pop("manifest_hash")
    if canonical_hash(payload) != recorded_manifest_hash:
        raise RuntimeError("PRMBench transfer manifest is not self-consistent")
    selection, selection_path = verify_development(args.development)
    current_hashes = {
        "raw": sha256_file(args.raw),
        "selection": sha256_file(selection_path),
        "config_index": sha256_file(args.development / "CONFIG_INDEX.json"),
        "scores": sha256_file(args.out / "FROZEN_SCORES.npz"),
        "nrm_comparator": sha256_file(args.nrm),
        "transfer_script": sha256_file(Path(__file__)),
    }
    if current_hashes != manifest["hashes"]:
        raise RuntimeError("PRMBench transfer input/source hash changed")
    if selection["selection_hash"] != manifest["selection_hash"]:
        raise RuntimeError("PRMBench development selection changed")
    with np.load(args.out / "FROZEN_SCORES.npz") as stored:
        row_keys = stored["row_keys"].astype(int)
        row_ids = stored["row_ids"].astype(str)
        source_ids = stored["source_ids"].astype(str)
        scores = {
            name: stored[name].astype(float)
            for name in ("iu", "finalist", "fixed_default", "cardinality")
        }
    with np.load(args.nrm) as stored:
        if not np.array_equal(stored["row_keys"].astype(int), row_keys):
            raise RuntimeError("NRM/graph-LIU PRMBench row mismatch")
        scores["family_nrm"] = stored["nrm_correctness_score"].astype(float)
    with args.raw.open("rb") as handle:
        cache = pickle.load(handle)
    selected = ordered_eligible_rows(cache)
    if [row[1] for row in selected] != row_ids.tolist():
        raise RuntimeError("PRMBench IDs do not align")
    if [row[2] for row in selected] != source_ids.tolist():
        raise RuntimeError("PRMBench source IDs do not align")
    labels = np.asarray([
        str(row[3]["classification"]) == "correct" for row in selected
    ], dtype=int)
    metrics = {
        name: {
            "auroc": float(roc_auc_score(labels, value)),
            "auprc": float(average_precision_score(labels, value)),
        }
        for name, value in scores.items()
    }
    intervals, draws = bootstrap(labels, scores, source_ids)
    delta = metrics["finalist"]["auroc"] - metrics["iu"]["auroc"]
    nrm_delta = metrics["family_nrm"]["auroc"] - metrics["iu"]["auroc"]
    d50 = draws["finalist"] - .5 * draws["family_nrm"]
    result = {
        "version": TRANSFER_VERSION,
        "status": "PASS" if intervals["finalist"]["low_pp"] > 0 else "FAIL",
        "scope": "post-audit retrospective bug-repair sensitivity; outcome known before v3",
        "n": len(labels),
        "n_correct": int(labels.sum()),
        "metrics": metrics,
        "delta_vs_iu_pp": 100 * delta,
        "family_nrm_delta_pp": 100 * nrm_delta,
        "nrm_recovery_fraction": delta / nrm_delta,
        "bootstrap": intervals,
        "d50_pp": 100 * float(np.mean(d50)),
        "d50_ci_pp": [
            100 * float(np.quantile(d50, .025)),
            100 * float(np.quantile(d50, .975)),
        ],
    }
    write_json(args.out / "RESULT.json", result)
    lines = [
        "# Family-residual graph LIU v3 bug-repair sensitivity — PRMBench", "",
        f"**{result['status']}**: finalist vs IU {result['delta_vs_iu_pp']:+.3f}pp "
        f"(source-group 95% CI [{intervals['finalist']['low_pp']:+.3f}, "
        f"{intervals['finalist']['high_pp']:+.3f}]pp).", "",
        f"Family-NRM changed AUROC by {result['family_nrm_delta_pp']:+.3f}pp; "
        f"the finalist recovered {100 * result['nrm_recovery_fraction']:.1f}% "
        f"of that point gain. `D_0.5`={result['d50_pp']:+.3f}pp.", "",
        "| method | AUROC | AUPRC | delta vs IU |", "|---|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        change = 100 * (row["auroc"] - metrics["iu"]["auroc"])
        lines.append(
            f"| `{name}` | {row['auroc']:.6f} | {row['auprc']:.6f} "
            f"| {change:+.3f}pp |"
        )
    lines += ["", "This outcome was known before v3 was specified. It is a "
              "retrospective bug-repair sensitivity, not transfer confirmation.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("score", "report", "both"))
    parser.add_argument("--development", type=Path, default=DEVELOPMENT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--nrm", type=Path, default=DEFAULT_NRM)
    args = parser.parse_args()
    if args.phase in {"score", "both"}:
        score_phase(args)
    if args.phase in {"report", "both"}:
        report_phase(args)


if __name__ == "__main__":
    main()
