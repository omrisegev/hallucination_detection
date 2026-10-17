#!/usr/bin/env python3
"""EXPLORATORY: run the archived graph arms G2/G3/G4 on the real 24 cells.

Commissioned by Omri (2026-08-24) after the deem_vs_iupcr_24cell_v1 decision,
to see how the graph arms would have performed relative to B3.  This run is
explicitly OUTSIDE every frozen protocol:

- Phase 0 stopped the graph experiment (`stop_before_natural_targets`): the
  target mechanism showed effect WITHOUT specificity (it also improves worlds
  where the graph is length artifact or pure noise), so any gain measured here
  is NOT attributable to the residual-graph mechanism.  The registered graph
  closure is unchanged by anything this script produces.
- The 24-cell label sidecars were already opened by the v1 evaluation, and no
  lambda was ever nominated for the target mechanism, so numbers produced here
  are post-hoc exploration on open labels across a lambda grid.  They can
  never support a registered claim; report the full grid, never a best-lambda
  headline.

What it reuses from the frozen v1 run (results/deem_vs_iupcr_24cell_v1):
the target-free bundles, the label sidecars, and the frozen B0/B3 Stage-A
fits as the comparison reference.  Graph construction mirrors the archived
`run_residual_graph_deem_24cell_v1.stage_a_cell` exactly:

  G2 = residual graph, uniform metric, target-smoothness mechanism
  G3 = residual graph, DUFS metric,   target-smoothness mechanism
  G4 = residual graph, DUFS metric,   explicit nuisance latent

The `fit` subcommand touches no label data (same firewall discipline as
Stage A); `evaluate` is a separate process that opens the sidecars.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import (  # noqa: E402
    SEEDS,
    ContinuousDeemConfig,
    DufsConfig,
    GraphDeemConfig,
    atomic_save_npz,
    atomic_write_json,
    build_inventory_graph,
    cross_view_dufs,
    crossfit_continuous_deem,
    donor_risk_matrix,
    fit_continuous_deem,
    graph_health,
    jsonable,
    symmetric_normalized_laplacian,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
)

# (arm, mechanism, graph kind).  Graph kinds: "uniform" and "dufs" are both
# built on the cross-fitted residuals, exactly as in the archived Stage A.
GRAPH_ARMS = (
    ("G2", "target", "uniform"),
    ("G3", "target", "dufs"),
    ("G4", "nuisance", "dufs"),
)
# The archived LAMBDA_GRID minus 0.0 (lambda=0 is definitionally B3, which is
# already frozen in the v1 run).
EXPLORE_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0)


def lambda_token(value: float) -> str:
    return str(float(value)).replace(".", "p")


def stem_name(arm: str, lambda_: float, seed: int) -> str:
    return f"{arm}__lambda{lambda_token(lambda_)}__seed{seed}"


def stem_state(fits_dir: Path, cell_id: str, stem: str) -> str | None:
    """Return 'complete'/'failed' if this stem already ran, else None."""
    meta = fits_dir / cell_id / f"{stem}.json"
    if not meta.is_file():
        return None
    try:
        status = json.loads(meta.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return None
    if status == "complete" and (fits_dir / cell_id / f"{stem}.npz").is_file():
        return "complete"
    if status == "failed":
        # Fits are deterministic: a recorded failure will fail identically on
        # every retry, so a resume skips it instead of looping.
        return "failed"
    return None


def fit_cell(bundle_dir: Path, fits_dir: Path, cell_id: str,
             config: ContinuousDeemConfig, lambdas, seeds) -> dict:
    bundle = load_target_free_bundle(bundle_dir / f"{cell_id}.npz")
    X_risk, _, _ = donor_risk_matrix(bundle.X_raw, bundle.X_raw, bundle.feature_names)
    dufs_config = DufsConfig()
    counts = {"complete": 0, "failed": 0, "skipped": 0}
    for seed in seeds:
        wanted = [
            (arm, mechanism, kind, lambda_)
            for arm, mechanism, kind in GRAPH_ARMS
            for lambda_ in lambdas
            if stem_state(fits_dir, cell_id, stem_name(arm, lambda_, seed)) is None
        ]
        if not wanted:
            counts["skipped"] += len(GRAPH_ARMS) * len(lambdas)
            continue
        # Residuals, gates, and graphs are shared by every arm/lambda of this
        # seed; build them once, exactly as the archived stage_a_cell does.
        crossfit = crossfit_continuous_deem(
            bundle.X_raw, bundle.feature_names, bundle.confidence_signs,
            bundle.group_ids, bundle.raw_trace_length, seed=seed, config=config,
        )
        graphs = {
            "uniform": build_inventory_graph(
                crossfit.residuals, bundle.feature_names, bundle.row_ids, k=7
            ),
        }
        gates, gate_diag = cross_view_dufs(
            crossfit.residuals, bundle.feature_names, crossfit.folds,
            bundle.row_ids, config=dufs_config,
        )
        graphs["dufs"] = build_inventory_graph(
            crossfit.residuals, bundle.feature_names, bundle.row_ids,
            k=7, gates=gates,
        )
        laplacians = {kind: symmetric_normalized_laplacian(graph)
                      for kind, graph in graphs.items()}
        health_by_kind = {kind: graph_health(graph) for kind, graph in graphs.items()}
        for arm, mechanism, kind, lambda_ in wanted:
            stem = stem_name(arm, lambda_, seed)
            base = {
                "schema": "explore_graph_arms_24cell_v1_fit",
                "cell_id": cell_id, "arm": arm, "mechanism": mechanism,
                "graph_kind": kind, "lambda": float(lambda_), "seed": int(seed),
                "bundle_sha256": bundle.bundle_sha256,
                "graph_health": jsonable(health_by_kind[kind]),
                "exploratory": True,
            }
            started = time.perf_counter()
            try:
                result = fit_continuous_deem(
                    X_risk, bundle.feature_names, seed=seed, config=config,
                    graph_config=GraphDeemConfig(lambda_=float(lambda_), mechanism=mechanism),
                    laplacian=laplacians[kind],
                )
                atomic_save_npz(fits_dir / cell_id / f"{stem}.npz",
                                score=np.asarray(result.score, dtype=np.float64))
                atomic_write_json(fits_dir / cell_id / f"{stem}.json", {
                    **base, "status": "complete",
                    "health": jsonable(result.health),
                    "runtime_seconds": time.perf_counter() - started,
                })
                counts["complete"] += 1
            except Exception as exc:  # record-not-block: exploratory analogue of A1
                atomic_write_json(fits_dir / cell_id / f"{stem}.json", {
                    **base, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                    "runtime_seconds": time.perf_counter() - started,
                })
                counts["failed"] += 1
    return {"cell_id": cell_id, **counts}


def run_fit(args) -> None:
    registry = load_registry(args.registry)
    cells = [row["cell_id"] for row in registry["cells"]]
    if args.cells:
        cells = [cell for cell in cells if cell in set(args.cells)]
    config = ContinuousDeemConfig()
    fits_dir = args.out_dir / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fit_cell, args.bundle_dir, fits_dir, cell,
                            config, EXPLORE_LAMBDAS, SEEDS): cell
                for cell in cells
            }
            for future, cell in futures.items():
                try:
                    reports.append(future.result())
                except Exception as exc:
                    reports.append({"cell_id": cell, "cell_error": f"{type(exc).__name__}: {exc}"})
                print(f"[fit] {reports[-1]}", flush=True)
    else:
        for cell in cells:
            try:
                reports.append(fit_cell(args.bundle_dir, fits_dir, cell,
                                        config, EXPLORE_LAMBDAS, SEEDS))
            except Exception as exc:
                reports.append({"cell_id": cell, "cell_error": f"{type(exc).__name__}: {exc}"})
            print(f"[fit] {reports[-1]}", flush=True)
    atomic_write_json(args.out_dir / "FIT_SUMMARY.json", {
        "schema": "explore_graph_arms_24cell_v1_fit_summary",
        "exploratory": True,
        "arms": [list(arm) for arm in GRAPH_ARMS],
        "lambdas": list(EXPLORE_LAMBDAS),
        "seeds": list(SEEDS),
        "cells": reports,
    })
    print(f"[fit] summary written to {args.out_dir / 'FIT_SUMMARY.json'}", flush=True)


def ensemble_scores(fits_dir: Path, cell_id: str, arm: str, lambda_: float,
                    min_seeds: int = 3):
    scores, used = [], []
    for seed in SEEDS:
        stem = stem_name(arm, lambda_, seed)
        if stem_state(fits_dir, cell_id, stem) != "complete":
            continue
        with np.load(fits_dir / cell_id / f"{stem}.npz", allow_pickle=False) as data:
            values = np.asarray(data["score"], dtype=np.float64)
        if np.isfinite(values).all():
            scores.append(values)
            used.append(seed)
    if len(scores) < min_seeds:
        return None, used
    return np.mean(np.stack(scores), axis=0), used


def run_evaluate(args) -> None:
    from spectral_utils.residual_graph_deem_labels import (  # label access here only
        join_labels_by_id, load_label_sidecar,
    )
    from scripts.evaluate_deem_vs_iupcr_24cell_v1 import metrics
    from scripts.evaluate_residual_graph_deem_24cell_v1 import ensemble as frozen_ensemble

    registry = load_registry(args.registry)
    cells = [row["cell_id"] for row in registry["cells"]]
    fits_dir = args.out_dir / "fits"
    rows = []
    family_of = {}
    for cell in cells:
        bundle = load_target_free_bundle(args.bundle_dir / f"{cell}.npz")
        sidecar = load_label_sidecar(args.sidecar_dir / f"{cell}.npz")
        y = join_labels_by_id(bundle, sidecar)
        family_of[cell] = bundle.dataset_family
        for arm in ("B0", "B3"):
            score, _ = frozen_ensemble(args.frozen_stage_a, cell, arm)
            rows.append({"cell_id": cell, "dataset_family": bundle.dataset_family,
                         "arm": arm, "lambda": "", "n_seeds": len(SEEDS),
                         **metrics(y, score)})
        for arm, _, _ in GRAPH_ARMS:
            for lambda_ in EXPLORE_LAMBDAS:
                score, used = ensemble_scores(fits_dir, cell, arm, lambda_)
                if score is None:
                    rows.append({"cell_id": cell, "dataset_family": bundle.dataset_family,
                                 "arm": arm, "lambda": float(lambda_),
                                 "n_seeds": len(used), "auroc": "", "auprc": ""})
                    continue
                rows.append({"cell_id": cell, "dataset_family": bundle.dataset_family,
                             "arm": arm, "lambda": float(lambda_), "n_seeds": len(used),
                             **metrics(y, score)})

    out_eval = args.out_dir / "evaluation"
    out_eval.mkdir(parents=True, exist_ok=True)
    fields = ["cell_id", "dataset_family", "arm", "lambda", "n_seeds", "auroc", "auprc"]
    with open(out_eval / "PER_CELL.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def macro(arm: str, lambda_) -> dict:
        by_family = defaultdict(list)
        missing = 0
        for row in rows:
            if row["arm"] == arm and row["lambda"] == lambda_:
                if row["auroc"] == "":
                    missing += 1
                else:
                    by_family[row["dataset_family"]].append(row)
        if not by_family:
            return {"missing_cells": missing}
        return {
            "equal_family_auroc": float(np.mean([
                np.mean([row["auroc"] for row in group]) for group in by_family.values()
            ])),
            "equal_family_auprc": float(np.mean([
                np.mean([row["auprc"] for row in group]) for group in by_family.values()
            ])),
            "cells": int(sum(len(group) for group in by_family.values())),
            "missing_cells": missing,
        }

    b3_per_cell = {row["cell_id"]: row["auroc"] for row in rows if row["arm"] == "B3"}
    summary = {
        "schema": "explore_graph_arms_24cell_v1_summary",
        "exploratory": True,
        "caveats": [
            "post-hoc exploration on open labels across a lambda grid",
            "phase0 found effect-without-specificity for the target mechanism: "
            "gains here are not attributable to the residual-graph mechanism",
            "the registered graph closure and the v1 decision are unchanged",
        ],
        "reference": {arm: macro(arm, "") for arm in ("B0", "B3")},
        "arms": {},
    }
    for arm, mechanism, kind in GRAPH_ARMS:
        summary["arms"][arm] = {"mechanism": mechanism, "graph": kind, "by_lambda": {}}
        for lambda_ in EXPLORE_LAMBDAS:
            value = macro(arm, float(lambda_))
            wins = ties = losses = 0
            for row in rows:
                if row["arm"] == arm and row["lambda"] == float(lambda_) and row["auroc"] != "":
                    delta = row["auroc"] - b3_per_cell[row["cell_id"]]
                    if abs(delta) <= 0.0005:
                        ties += 1
                    elif delta > 0:
                        wins += 1
                    else:
                        losses += 1
            value["vs_B3_wins_ties_losses"] = [wins, ties, losses]
            if "equal_family_auroc" in value and "equal_family_auroc" in summary["reference"]["B3"]:
                value["delta_vs_B3_equal_family_auroc"] = (
                    value["equal_family_auroc"] - summary["reference"]["B3"]["equal_family_auroc"]
                )
            summary["arms"][arm]["by_lambda"][lambda_token(lambda_)] = value
    atomic_write_json(out_eval / "SUMMARY.json", summary)
    b3 = summary["reference"]["B3"].get("equal_family_auroc")
    b0 = summary["reference"]["B0"].get("equal_family_auroc")
    print(f"[evaluate] reference recomputed: B0 {b0:.4f} B3 {b3:.4f} "
          f"(frozen v1: B0 0.7428 B3 0.7485 -- must match)", flush=True)
    print(f"[evaluate] summary written to {out_eval / 'SUMMARY.json'}", flush=True)


def run_smoke(args) -> None:
    """Offline software test on a synthetic bundle-shaped fixture (no data)."""
    from types import SimpleNamespace

    registry = load_registry(args.registry)
    names = tuple(registry["cells"][0]["feature_names"])
    signs = np.asarray(registry["cells"][0]["confidence_signs"], dtype=np.float64)
    rng = np.random.default_rng(0)
    n = 120
    bundle = SimpleNamespace(
        cell_id="smoke_cell",
        X_raw=rng.normal(size=(n, len(names))),
        feature_names=names,
        confidence_signs=signs,
        row_ids=tuple(f"row{i}" for i in range(n)),
        group_ids=tuple(f"g{i // 10}" for i in range(n)),
        raw_trace_length=rng.integers(20, 400, size=n).astype(np.float64),
        dataset_family="smoke", task_type="smoke", bundle_sha256="smoke",
    )
    config = ContinuousDeemConfig(epochs=3)
    X_risk, _, _ = donor_risk_matrix(bundle.X_raw, bundle.X_raw, names)
    crossfit = crossfit_continuous_deem(
        bundle.X_raw, names, signs, bundle.group_ids, bundle.raw_trace_length,
        seed=0, config=config,
    )
    gates, _ = cross_view_dufs(crossfit.residuals, names, crossfit.folds,
                               bundle.row_ids, config=DufsConfig())
    graphs = {
        "uniform": build_inventory_graph(crossfit.residuals, names, bundle.row_ids, k=7),
        "dufs": build_inventory_graph(crossfit.residuals, names, bundle.row_ids,
                                      k=7, gates=gates),
    }
    for arm, mechanism, kind in GRAPH_ARMS:
        result = fit_continuous_deem(
            X_risk, names, seed=0, config=config,
            graph_config=GraphDeemConfig(lambda_=0.1, mechanism=mechanism),
            laplacian=symmetric_normalized_laplacian(graphs[kind]),
        )
        assert np.isfinite(result.score).all(), arm
        print(f"[smoke] {arm} ({mechanism}/{kind}) ok: "
              f"score_sd={float(np.std(result.score)):.4g} "
              f"healthy={result.health.get('healthy')}", flush=True)
    print("[smoke] all graph arms pass on the synthetic fixture", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="label-free graph-arm fits on the 24 cells")
    fit.add_argument("--registry", type=Path, required=True)
    fit.add_argument("--bundle-dir", type=Path, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    fit.add_argument("--workers", type=int, default=1)
    fit.add_argument("--cells", nargs="*", default=None)

    evaluate = sub.add_parser("evaluate", help="score fits against the v1 sidecars")
    evaluate.add_argument("--registry", type=Path, required=True)
    evaluate.add_argument("--bundle-dir", type=Path, required=True)
    evaluate.add_argument("--sidecar-dir", type=Path, required=True)
    evaluate.add_argument("--frozen-stage-a", type=Path, required=True)
    evaluate.add_argument("--out-dir", type=Path, required=True)

    smoke = sub.add_parser("smoke", help="offline synthetic-fixture software test")
    smoke.add_argument("--registry", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "fit":
        run_fit(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()
