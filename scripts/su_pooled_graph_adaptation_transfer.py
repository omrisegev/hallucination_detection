#!/usr/bin/env python3
"""Frozen retrospective transfer of the prespecified SU cleaning adaptation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.harp_global_contribution_teacher import (  # noqa: E402
    PROCESS_MODELS,
    PROCESS_SUBSETS,
    SEMGRAD_DATASETS,
    mixed_v2_matrix,
    process_items,
    telemetry_only,
)
from scripts.su_pooled_graph_adaptation_sidecar import (  # noqa: E402
    candidate_score,
    contribution_state,
    feature_cross_mask,
    pcr_weights,
    psd_projection,
    sha256_file,
    sym,
    write_csv,
    write_json,
)
from spectral_utils.dependency_fusion import sparse_upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "su-pooled-graph-adaptation-transfer-v1-2026-08-23"
SELECTION_ROOT = REPO / "results" / "su_pooled_graph_adaptation_conservative_v2"
DEFAULT_OUT = REPO / "results" / "su_pooled_graph_adaptation_transfer_v1"
CURRENT_ARM = "iu_observed_mean"
PRIMARY_ARM = "iu_cross_sparse_mean"
NRM_REFERENCE_PP = {
    "processbench_qwen": 0.557,
    "processbench_llama": 1.580,
    "semgrad": 1.310,
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def frozen_rows(selection_root: Path) -> dict[str, dict]:
    with (selection_root / "FULL_SELECTION.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["arm"]: row for row in csv.DictReader(handle)}
    return {name: rows[name] for name in (CURRENT_ARM, PRIMARY_ARM)}


def external_cells(*, labels: bool) -> list[dict]:
    cells = []
    for model in PROCESS_MODELS:
        for subset in PROCESS_SUBSETS:
            path = REPO / "dataset_cache" / "repgrid" / f"pb_{model}" / f"processbench_{subset}.pkl"
            items = process_items(path)
            telemetry = [telemetry_only(row) for _, row in items]
            F, names, _, _ = mixed_v2_matrix(telemetry)
            cells.append({
                "cell": f"{model}__{subset}",
                "domain": "processbench_qwen",
                "group": subset,
                "ids": np.asarray([key for key, _ in items]),
                "F": F,
                "names": names,
                "labels": np.asarray([int(row["label"] == -1) for _, row in items]) if labels else None,
            })
    root = REPO / "dataset_cache" / "repgrid" / "pb_llama31_8b"
    for subset in PROCESS_SUBSETS:
        items = process_items(root / f"processbench_{subset}.pkl")
        telemetry = [telemetry_only(row) for _, row in items]
        F, names, _, _ = mixed_v2_matrix(telemetry)
        cells.append({
            "cell": f"llama31_8b__{subset}",
            "domain": "processbench_llama",
            "group": subset,
            "ids": np.asarray([key for key, _ in items]),
            "F": F,
            "names": names,
            "labels": np.asarray([int(row["label"] == -1) for _, row in items]) if labels else None,
        })
    semgrad_root = REPO / "local_cache" / "semgrad_bem_regraded"
    for dataset in SEMGRAD_DATASETS:
        cache = load_pickle(semgrad_root / f"raw_semgrad_{dataset}_T0.0_bem.pkl")
        keys, telemetry, target = [], [], []
        for key in sorted(cache):
            candidates = cache[key].get("candidates")
            if not candidates:
                continue
            candidate = candidates[0]
            keys.append(str(key))
            telemetry.append(telemetry_only(candidate))
            if labels:
                target.append(int(candidate["bem_correct"]))
        F, names, _, _ = mixed_v2_matrix(telemetry)
        cells.append({
            "cell": f"semgrad__{dataset}",
            "domain": "semgrad",
            "group": dataset,
            "ids": np.asarray(keys),
            "F": F,
            "names": names,
            "labels": np.asarray(target) if labels else None,
        })
    return cells


def state_score(state, direction, trust):
    row = {
        "payload": {
            "baseline__frozen": state["baseline"],
            "residuals__frozen": state["residuals"],
            "presence__frozen": state["presence"].astype(np.int8),
        }
    }
    return candidate_score(row, "frozen", direction, trust)


def fit_command(args) -> None:
    selection_root = args.selection_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "scores").mkdir()
    selection = frozen_rows(selection_root)
    current_direction = np.asarray(json.loads(selection[CURRENT_ARM]["direction"]), dtype=float)
    primary_direction = np.asarray(json.loads(selection[PRIMARY_ARM]["direction"]), dtype=float)
    current_trust = float(selection[CURRENT_ARM]["selected_trust"])
    primary_trust = float(selection[PRIMARY_ARM]["selected_trust"])
    primary_alpha = float(selection[PRIMARY_ARM]["selected_alpha"])
    manifest = {}
    cells = external_cells(labels=False)
    for index, cell in enumerate(cells, 1):
        print(f"[{index:02d}/{len(cells)}] {cell['cell']}", flush=True)
        F, names = np.asarray(cell["F"], dtype=float), tuple(cell["names"])
        iu = upcr_fit(F, **IU_FIT_DEFAULTS)
        current_state = contribution_state(F, names, iu.w)
        su = sparse_upcr_fit(F)
        C = sym(F @ F.T / F.shape[1])
        cross_sparse = np.where(feature_cross_mask(names), su.decomposition.sparse, 0.0)
        clean, clean_diag = psd_projection(C - primary_alpha * cross_sparse)
        primary_weights = pcr_weights(clean, iu.rho_hat, 2)
        primary_state = contribution_state(F, names, primary_weights)
        payload = {
            "ids": cell["ids"],
            "iu": current_state["baseline"],
            "current": state_score(current_state, current_direction, current_trust),
            "primary": state_score(primary_state, primary_direction, primary_trust),
        }
        path = out / "scores" / f"{cell['cell']}.npz"
        np.savez_compressed(path, **payload)
        manifest[cell["cell"]] = {
            "domain": cell["domain"],
            "group": cell["group"],
            "score_sha256": sha256_file(path),
            "n": int(F.shape[1]),
            "n_features": int(F.shape[0]),
            "primary_clean_diagnostics": clean_diag,
        }
    write_json(out / "SCORE_FREEZE_MANIFEST.json", {
        "version": VERSION,
        "selection_root": str(selection_root),
        "selection_run_sha256": sha256_file(selection_root / "RUN_DEFINITION.json"),
        "selection_complete_sha256": sha256_file(selection_root / "REPORT_COMPLETE.json"),
        "current": selection[CURRENT_ARM],
        "primary": selection[PRIMARY_ARM],
        "cells": manifest,
        "labels_read_during_fit": False,
        "source_sha256": sha256_file(Path(__file__)),
    })
    write_json(out / "FIT_COMPLETE.json", {
        "version": VERSION,
        "n_cells": len(cells),
        "labels_read": False,
    })


def report_command(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    out = args.out.resolve()
    manifest = json.loads((out / "SCORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("transfer source changed after score freeze")
    labelled = {cell["cell"]: cell for cell in external_cells(labels=True)}
    rows = []
    for cell_name, meta in manifest["cells"].items():
        path = out / "scores" / f"{cell_name}.npz"
        if sha256_file(path) != meta["score_sha256"]:
            raise RuntimeError(f"score hash mismatch: {cell_name}")
        payload = np.load(path, allow_pickle=False)
        cell = labelled[cell_name]
        if not np.array_equal(payload["ids"].astype(str), cell["ids"].astype(str)):
            raise RuntimeError(f"row identity mismatch: {cell_name}")
        y = cell["labels"]
        iu_auc = float(roc_auc_score(y, payload["iu"]))
        current_auc = float(roc_auc_score(y, payload["current"]))
        primary_auc = float(roc_auc_score(y, payload["primary"]))
        rows.append({
            "cell": cell_name,
            "domain": meta["domain"],
            "group": meta["group"],
            "n": len(y),
            "n_correct": int(np.sum(y)),
            "iu_auroc": iu_auc,
            "current_auroc": current_auc,
            "primary_auroc": primary_auc,
            "current_delta_vs_iu_pp": 100 * (current_auc - iu_auc),
            "primary_delta_vs_iu_pp": 100 * (primary_auc - iu_auc),
            "primary_minus_current_pp": 100 * (primary_auc - current_auc),
        })
    write_csv(out / "CELL_METRICS.csv", rows)
    domains = sorted({row["domain"] for row in rows})
    summary = []
    for domain in domains:
        selected = [row for row in rows if row["domain"] == domain]
        groups = sorted({row["group"] for row in selected})
        def equal_group(field):
            return float(np.mean([
                np.mean([row[field] for row in selected if row["group"] == group])
                for group in groups
            ]))
        summary.append({
            "domain": domain,
            "n_cells": len(selected),
            "n_groups": len(groups),
            "current_delta_vs_iu_pp": equal_group("current_delta_vs_iu_pp"),
            "primary_delta_vs_iu_pp": equal_group("primary_delta_vs_iu_pp"),
            "primary_minus_current_pp": equal_group("primary_minus_current_pp"),
            "nrm_reference_pp": NRM_REFERENCE_PP[domain],
        })
    write_csv(out / "SUMMARY.csv", summary)

    x = np.arange(len(summary))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x - width, [row["current_delta_vs_iu_pp"] for row in summary], width, label="current pooled graph")
    ax.bar(x, [row["primary_delta_vs_iu_pp"] for row in summary], width, label="IU + cross-family clean")
    ax.bar(x + width, [row["nrm_reference_pp"] for row in summary], width, label="Family-NRM reference")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, [row["domain"] for row in summary])
    ax.set_ylabel("Equal-dataset-family AUROC delta vs IU-PCR (pp)")
    ax.set_title("Frozen retrospective transfer")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "TRANSFER_COMPARISON.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Frozen transfer: SU-aware pooled graph adaptation",
        "",
        "The current and prespecified cross-family-cleaned directions/configurations were frozen on the original development families before these labels were opened by this script. These targets are historically known and therefore retrospective stress tests, not prospective confirmation.",
        "",
        "| domain | current vs IU | cross-clean vs IU | cross-clean minus current | Family-NRM reference |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['domain']}` | {row['current_delta_vs_iu_pp']:+.3f}pp | "
            f"{row['primary_delta_vs_iu_pp']:+.3f}pp | {row['primary_minus_current_pp']:+.3f}pp | "
            f"{row['nrm_reference_pp']:+.3f}pp |"
        )
    lines.extend(["", "No transfer result may be used to retune alpha, direction, lambda, or trust.", ""])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "REPORT_COMPLETE.json", {
        "version": VERSION,
        "n_cells": len(rows),
        "summaries": summary,
        "retrospective": True,
        "labels_opened_after_score_hash_verification": True,
    })
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "report"):
        command = sub.add_parser(name)
        command.add_argument("--selection-root", type=Path, default=SELECTION_ROOT)
        command.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "fit":
        fit_command(args)
    else:
        report_command(args)


if __name__ == "__main__":
    main()
