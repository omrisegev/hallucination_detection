#!/usr/bin/env python3
"""Cross-family supervised teacher in IU provenance-contribution space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family as original_family,
    load_contract,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
    resolve_data_path,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_contribution_score,
    fit_contribution_transform,
    iu_family_contributions,
)
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "harp-global-contribution-teacher-v1-2026-08-12"
DEFAULT_OUT = REPO / "results" / "harp_global_contribution_teacher_v1"
PRIOR_STRENGTH = 0.3
BOOTSTRAP_DRAWS = 20000
MIN_POSITIVES = 20
PROCESS_MODELS = ("qwen3_4b", "qwen3_8b")
PROCESS_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
SEMGRAD_DATASETS = ("sciq", "truthfulqa")
TELEMETRY_KEYS = (
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def process_items(path):
    cache = load_pickle(resolve_data_path(Path(path)))
    return [
        (str(key), cache[key])
        for key in sorted(cache)
        if not cache[key]["align_diag"]["problems"]
    ]


def telemetry_only(row):
    return {name: row.get(name) for name in TELEMETRY_KEYS}


def contribution_cell(name, group, domain, F, feature_names, correctness):
    correctness = np.asarray(correctness, dtype=int)
    if correctness.shape != (F.shape[1],) or not np.all(
        np.isin(correctness, (0, 1))
    ):
        raise ValueError(f"invalid correctness vector: {name}")
    fitted = upcr_fit(F, **IU_FIT_DEFAULTS)
    space = iu_family_contributions(F, feature_names, fitted.w)
    transform = fit_contribution_transform(
        space, np.arange(F.shape[1], dtype=int)
    )
    baseline, residuals = transform.apply(
        space.baseline_score, space.contributions
    )
    aligned = np.zeros((F.shape[1], len(VIEW_ORDER)), dtype=float)
    presence = np.zeros(len(VIEW_ORDER), dtype=int)
    for index, family_name in enumerate(space.families):
        target_index = VIEW_ORDER.index(family_name)
        aligned[:, target_index] = residuals[:, index]
        presence[target_index] = 1
    cardinality = cardinality_balanced_contribution_score(
        space, fitted.w
    )
    return {
        "cell": name,
        "group": group,
        "domain": domain,
        "baseline": baseline,
        "residuals": aligned,
        "correctness": correctness,
        "cardinality_score": cardinality.score,
        "presence": presence,
        "families": space.families,
        "space": space,
        "weights": fitted.w,
        "n": len(correctness),
        "n_correct": int(np.sum(correctness)),
    }


def load_original_cells(bundle_path):
    cells = []
    with np.load(bundle_path, allow_pickle=True) as data:
        for cell in INSCOPE:
            correctness = np.asarray(data[f"{cell}__labels"], dtype=int)
            if int(np.sum(correctness)) < MIN_POSITIVES:
                continue
            F, names = load_contract(data, cell, "mixed_v2")
            cells.append(contribution_cell(
                cell,
                original_family(cell),
                "original_23",
                F,
                names,
                correctness,
            ))
    return cells


def load_qwen_processbench_cells():
    cells = []
    for model in PROCESS_MODELS:
        for subset in PROCESS_SUBSETS:
            path = (
                REPO
                / "dataset_cache"
                / "repgrid"
                / f"pb_{model}"
                / f"processbench_{subset}.pkl"
            )
            items = process_items(path)
            telemetry = [telemetry_only(row) for _, row in items]
            correctness = [int(row["label"] == -1) for _, row in items]
            F, names, _, _ = mixed_v2_matrix(telemetry)
            cells.append(contribution_cell(
                f"{model}__{subset}",
                subset,
                "processbench_qwen",
                F,
                names,
                correctness,
            ))
    return cells


def load_llama_processbench_cells():
    cells = []
    root = REPO / "dataset_cache" / "repgrid" / "pb_llama31_8b"
    for subset in PROCESS_SUBSETS:
        items = process_items(root / f"processbench_{subset}.pkl")
        telemetry = [telemetry_only(row) for _, row in items]
        correctness = [int(row["label"] == -1) for _, row in items]
        F, names, _, _ = mixed_v2_matrix(telemetry)
        cells.append(contribution_cell(
            f"llama31_8b__{subset}",
            subset,
            "processbench_llama",
            F,
            names,
            correctness,
        ))
    return cells


def load_semgrad_cells():
    cells = []
    root = REPO / "local_cache" / "semgrad_bem_regraded"
    for dataset in SEMGRAD_DATASETS:
        cache = load_pickle(root / f"raw_semgrad_{dataset}_T0.0_bem.pkl")
        telemetry, correctness = [], []
        for key in sorted(cache):
            candidates = cache[key].get("candidates")
            if not candidates:
                continue
            candidate = candidates[0]
            telemetry.append(telemetry_only(candidate))
            correctness.append(int(candidate["bem_correct"]))
        F, names, _, _ = mixed_v2_matrix(telemetry)
        cells.append(contribution_cell(
            f"semgrad__{dataset}",
            dataset,
            "semgrad",
            F,
            names,
            correctness,
        ))
    return cells


def fit_global_delta(cells):
    if not cells:
        raise ValueError("at least one source cell is required")
    baseline = np.concatenate([cell["baseline"] for cell in cells])
    residuals = np.vstack([cell["residuals"] for cell in cells])
    correctness = np.concatenate([cell["correctness"] for cell in cells])
    sample_weights = []
    for cell in cells:
        y = cell["correctness"]
        positive = int(np.sum(y == 1))
        negative = int(np.sum(y == 0))
        if not positive or not negative:
            raise ValueError(f"one-class source cell: {cell['cell']}")
        sample_weights.append(
            np.where(y == 1, 0.5 / positive, 0.5 / negative) / len(cells)
        )
    sample_weights = np.concatenate(sample_weights)

    def objective(delta):
        score = baseline + residuals @ delta
        probability = expit(score)
        loss = float(np.sum(
            sample_weights
            * (np.logaddexp(0.0, score) - correctness * score)
        ))
        loss += 0.5 * PRIOR_STRENGTH * float(np.dot(delta, delta))
        gradient = residuals.T @ (
            sample_weights * (probability - correctness)
        )
        gradient += PRIOR_STRENGTH * delta
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(len(VIEW_ORDER), dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"global teacher fit failed: {fitted.message}")
    return np.asarray(fitted.x, dtype=float), {
        "n_source_cells": len(cells),
        "n_source_samples": len(correctness),
        "objective": float(fitted.fun),
        "n_iter": int(fitted.nit),
        "delta_norm": float(np.linalg.norm(fitted.x)),
    }


def evaluate_cell(cell, delta, regime):
    baseline = cell["baseline"]
    teacher = baseline + cell["residuals"] @ delta
    cardinality = cell["cardinality_score"]
    y = cell["correctness"]
    iu_auc = float(roc_auc_score(y, baseline))
    teacher_auc = float(roc_auc_score(y, teacher))
    cardinality_auc = float(roc_auc_score(y, cardinality))
    return {
        "version": VERSION,
        "regime": regime,
        "domain": cell["domain"],
        "group": cell["group"],
        "cell": cell["cell"],
        "n": cell["n"],
        "n_correct": cell["n_correct"],
        "families_present": "|".join(cell["families"]),
        "iu_auroc": iu_auc,
        "global_teacher_auroc": teacher_auc,
        "cardinality_auroc": cardinality_auc,
        "global_teacher_delta_pp": 100 * (teacher_auc - iu_auc),
        "cardinality_delta_pp": 100 * (cardinality_auc - iu_auc),
        "teacher_correction_scale": float(np.std(
            cell["residuals"] @ delta
        )),
    }


def grouped_summary(rows, domain, method_key):
    selected = [row for row in rows if row["domain"] == domain]
    groups = sorted({row["group"] for row in selected})
    deltas = np.asarray([
        np.mean([
            row[f"{method_key}_auroc"] - row["iu_auroc"]
            for row in selected if row["group"] == group
        ])
        for group in groups
    ])
    seed = int(hashlib.sha256(
        f"{VERSION}:{domain}:{method_key}".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = deltas[
        rng.integers(0, len(deltas), size=(BOOTSTRAP_DRAWS, len(deltas)))
    ].mean(axis=1)
    cell_deltas = np.asarray([
        row[f"{method_key}_auroc"] - row["iu_auroc"] for row in selected
    ])
    return {
        "version": VERSION,
        "domain": domain,
        "method": method_key,
        "n_cells": len(selected),
        "n_groups": len(groups),
        "cell_macro_delta_pp": float(100 * np.mean(cell_deltas)),
        "equal_group_delta_pp": float(100 * np.mean(deltas)),
        "equal_group_ci_low_pp": float(100 * np.quantile(draws, 0.025)),
        "equal_group_ci_high_pp": float(100 * np.quantile(draws, 0.975)),
        "wins": int(np.sum(cell_deltas > 0)),
        "losses": int(np.sum(cell_deltas < 0)),
        "ties": int(np.sum(cell_deltas == 0)),
        "worst_cell_delta_pp": float(100 * np.min(cell_deltas)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    original = load_original_cells(args.bundle)
    external = (
        load_qwen_processbench_cells()
        + load_llama_processbench_cells()
        + load_semgrad_cells()
    )
    rows, coefficient_rows = [], []

    original_groups = sorted({cell["group"] for cell in original})
    for heldout_group in original_groups:
        source = [cell for cell in original if cell["group"] != heldout_group]
        heldout = [cell for cell in original if cell["group"] == heldout_group]
        delta, diagnostics = fit_global_delta(source)
        for cell in heldout:
            rows.append(evaluate_cell(cell, delta, "original_lofo"))
        coefficient_rows.extend({
            "version": VERSION,
            "regime": "original_lofo",
            "heldout_group": heldout_group,
            "family": family,
            "delta": float(delta[index]),
            **diagnostics,
        } for index, family in enumerate(VIEW_ORDER))

    source_delta, source_diagnostics = fit_global_delta(original)
    for cell in external:
        rows.append(evaluate_cell(cell, source_delta, "source23_transfer"))
    coefficient_rows.extend({
        "version": VERSION,
        "regime": "source23_transfer",
        "heldout_group": "all_external",
        "family": family,
        "delta": float(source_delta[index]),
        **source_diagnostics,
    } for index, family in enumerate(VIEW_ORDER))

    summaries = []
    for domain in sorted({row["domain"] for row in rows}):
        for method in ("global_teacher", "cardinality"):
            summaries.append(grouped_summary(rows, domain, method))
    write_csv(args.out / "cell_results.csv", rows)
    write_csv(args.out / "coefficients.csv", coefficient_rows)
    write_csv(args.out / "summary.csv", summaries)

    lookup = {
        (row["domain"], row["method"]): row for row in summaries
    }
    semgrad = lookup[("semgrad", "global_teacher")]
    result = {
        "version": VERSION,
        "status": "supervised_research_instrument",
        "prior_strength": PRIOR_STRENGTH,
        "view_order": list(VIEW_ORDER),
        "source23_delta": source_delta.tolist(),
        "source23_diagnostics": source_diagnostics,
        "semgrad_both_positive": bool(all(
            row["global_teacher_delta_pp"] > 0
            for row in rows if row["domain"] == "semgrad"
        )),
        "semgrad_equal_dataset_delta_pp": semgrad["equal_group_delta_pp"],
        "not_a_deployable_unsupervised_method": True,
    }
    write_json(args.out / "RESULT.json", result)
    sources = {
        "script": sha256_file(Path(__file__)),
        "spec": sha256_file(REPO / "SPEC_HARP_GLOBAL_CONTRIBUTION_TEACHER_V1.md"),
        "module": sha256_file(
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "bundle": sha256_file(args.bundle),
    }
    write_json(args.out / "RUN_DEFINITION.json", {
        "version": VERSION,
        "sources": sources,
        "prior_strength": PRIOR_STRENGTH,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "training_scope": "original_23_cells_only_for_external_transfer",
        "external_labels_used_for_fit": False,
    })

    def signed(value):
        return f"{float(value):+.3f}"

    lines = [
        "# HARP-inspired global contribution teacher",
        "",
        "**Status:** supervised proof-of-feasibility; not a deployable "
        "unsupervised method.",
        "",
        "One six-family correction was trained on the original 23 cells. "
        "External target labels never entered that fit.",
        "",
        "| evaluation domain | method | equal-group delta vs IU | 95% interval | W/L | worst |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['domain']} | `{row['method']}` "
            f"| {signed(row['equal_group_delta_pp'])}pp "
            f"| [{signed(row['equal_group_ci_low_pp'])}, "
            f"{signed(row['equal_group_ci_high_pp'])}] "
            f"| {row['wins']}/{row['losses']} "
            f"| {signed(row['worst_cell_delta_pp'])}pp |"
        )
    lines.extend([
        "",
        "## Source-23 teacher coefficients",
        "",
        "| family | delta |",
        "|---|---:|",
    ])
    for family, value in zip(VIEW_ORDER, source_delta):
        lines.append(f"| `{family}` | {value:+.6f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "If the global teacher transfers where cardinality balancing fails, "
        "then the target correction is present and reusable in contribution "
        "space, while the current label-free nuisance proxy is insufficient. "
        "That is evidence for continuing self-supervised target-direction "
        "research, not for deploying these supervised coefficients.",
        "",
    ])
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
