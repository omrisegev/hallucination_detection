#!/usr/bin/env python3
"""Independent-example frozen transfer of CB-CS-IU to SemGrad cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
)
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_iu_fit,
    leverage_balanced_contribution_score,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_path,
)


VERSION = "cardinality-balanced-semgrad-confirmation-v1-2026-08-12"
DEFAULT_CACHE = REPO / "local_cache" / "semgrad_bem_regraded"
DEFAULT_OUT = REPO / "results" / "cardinality_balanced_semgrad_v1"
SPEC = REPO / "SPEC_CARDINALITY_BALANCED_SEMGRAD_CONFIRMATION_V1.md"
REGISTRY = REPO / "results" / "data_readiness_2026_08_11" / "dataset_registry.json"
DATASETS = ("sciq", "truthfulqa")
EXPECTED_ROWS = {"sciq": 1000, "truthfulqa": 817}
TELEMETRY_KEYS = (
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
)
FORBIDDEN_FIT_KEYS = ("bem_correct", "bem_score", "label")
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
DUFS_LAMBDA = 0.1
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_CHUNK = 250
METHODS = (
    "iu",
    "cardinality",
    "leverage",
    "dufs_liu",
    "uniform",
    "reverse_cardinality",
)
COMPARISONS = (
    ("cardinality", "iu"),
    ("leverage", "iu"),
    ("dufs_liu", "iu"),
    ("cardinality", "leverage"),
    ("cardinality", "dufs_liu"),
    ("cardinality", "uniform"),
    ("cardinality", "reverse_cardinality"),
)


def data_path(cache_root, dataset):
    return Path(cache_root) / f"raw_semgrad_{dataset}_T0.0_bem.pkl"


def bem_manifest_path(cache_root, dataset):
    return Path(cache_root) / f"raw_semgrad_{dataset}_T0.0_bem_manifest.json"


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


def load_cache(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def telemetry_without_targets(path):
    """Return only whitelisted telemetry; target values remain inaccessible."""
    cache = load_cache(path)
    ids, rows = [], []
    for key in sorted(cache):
        candidates = cache[key].get("candidates")
        if not candidates:
            continue
        candidate = candidates[0]
        telemetry = {
            name: candidate.get(name) for name in TELEMETRY_KEYS
        }
        if set(telemetry) & set(FORBIDDEN_FIT_KEYS):
            raise RuntimeError("a target key entered the telemetry whitelist")
        ids.append(str(key))
        rows.append(telemetry)
    return ids, rows


def source_paths():
    return {
        "script": Path(__file__),
        "spec": SPEC,
        "candidate_module": REPO / "spectral_utils" / "contribution_subspace.py",
        "feature_contract": REPO / "spectral_utils" / "dufs_liu_feature_contract.py",
        "feature_registry": REPO / "spectral_utils" / "specrage_views.py",
        "feature_extraction": REPO / "spectral_utils" / "feature_utils.py",
        "dufs_module": REPO / "spectral_utils" / "adapted_dufs.py",
        "laplacian_module": REPO / "spectral_utils" / "laplacian_upcr.py",
        "mixed_v2_dependency": (
            REPO / "scripts" / "leverage_balanced_processbench_transfer.py"
        ),
        "data_readiness_registry": REGISTRY,
    }


def direction_score(baseline, residuals, direction):
    direction = np.asarray(direction, dtype=float)
    raw = residuals @ direction
    scale = float(np.std(raw))
    correction = (
        np.zeros(len(baseline), dtype=float)
        if scale <= 1e-12 or np.linalg.norm(direction) <= 1e-12
        else raw / (len(direction) * scale)
    )
    return baseline + correction


def fit_scores(cache_root, out):
    cache_root, out = Path(cache_root), Path(out)
    score_dir = out / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    diagnostics, data_hashes, bem_manifest_hashes, score_hashes = [], {}, {}, {}

    for dataset in DATASETS:
        path = data_path(cache_root, dataset)
        manifest_path = bem_manifest_path(cache_root, dataset)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_data_hash = sha256_file(path)
        if manifest["output_sha256"] != actual_data_hash:
            raise RuntimeError(f"BEM output hash mismatch: {dataset}")

        row_ids, telemetry = telemetry_without_targets(path)
        if len(row_ids) != EXPECTED_ROWS[dataset]:
            raise RuntimeError(f"unexpected row count: {dataset}")
        F, names, availability, contract = mixed_v2_matrix(telemetry)
        fitted = cardinality_balanced_iu_fit(F, names)
        primary = fitted.balanced
        leverage = leverage_balanced_contribution_score(
            fitted.contribution_space, fitted.baseline.w
        )
        baseline = primary.baseline_score
        if not np.array_equal(baseline, leverage.baseline_score):
            raise RuntimeError("candidate variants do not share the IU baseline")

        _, residuals = primary.transform.apply(
            fitted.contribution_space.baseline_score,
            fitted.contribution_space.contributions,
        )
        uniform = direction_score(
            baseline,
            residuals,
            np.ones(len(fitted.contribution_space.families)),
        )
        reverse = baseline - primary.correction

        gates, gate_diag = adapted_dufs_soft_gates(
            F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
        )
        graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
        dufs_path = laplacian_iu_path(
            F, (0.0, DUFS_LAMBDA), graph=graph
        )
        dufs_iu = dufs_path[0.0].baseline.w @ F
        dufs = dufs_path[DUFS_LAMBDA].w @ F
        standardized_dufs_iu = (
            dufs_iu - np.mean(dufs_iu)
        ) / np.std(dufs_iu)
        iu_identity_error = float(np.max(np.abs(
            standardized_dufs_iu - baseline
        )))
        if iu_identity_error > 1e-10:
            raise RuntimeError(
                f"DUFS and CB IU baselines disagree: {dataset} "
                f"{iu_identity_error:.3e}"
            )

        score_path = score_dir / f"semgrad__{dataset}.npz"
        np.savez_compressed(
            score_path,
            row_ids=np.asarray(row_ids),
            feature_names=np.asarray(names),
            family_names=np.asarray(fitted.contribution_space.families),
            family_cardinality=primary.family_cardinality,
            iu_risk=-baseline,
            cardinality_risk=-primary.score,
            leverage_risk=-leverage.score,
            dufs_liu_risk=-dufs,
            uniform_risk=-uniform,
            reverse_cardinality_risk=-reverse,
            cardinality_delta=primary.delta,
            cardinality_effective_weights=primary.effective_weights,
            cardinality_intercept=np.asarray(primary.intercept),
            leverage_delta=leverage.delta,
            leverage_effective_weights=leverage.effective_weights,
            leverage_intercept=np.asarray(leverage.intercept),
        )
        data_hashes[dataset] = actual_data_hash
        bem_manifest_hashes[dataset] = sha256_file(manifest_path)
        score_hashes[dataset] = sha256_file(score_path)
        diagnostics.append({
            "version": VERSION,
            "dataset": dataset,
            "n_rows": len(row_ids),
            "n_features": int(F.shape[0]),
            "n_families": len(fitted.contribution_space.families),
            "family_cardinality": json.dumps(
                primary.family_cardinality.astype(int).tolist()
            ),
            "target_keys_accessed_during_fit": False,
            "fit_key_whitelist": json.dumps(list(TELEMETRY_KEYS)),
            "contribution_reconstruction_error": (
                fitted.contribution_space.diagnostics["reconstruction_error"]
            ),
            "cardinality_weight_reconstruction_error": (
                primary.diagnostics["weight_reconstruction_error"]
            ),
            "leverage_weight_reconstruction_error": (
                leverage.diagnostics["weight_reconstruction_error"]
            ),
            "cardinality_orthogonality": primary.diagnostics[
                "baseline_correction_covariance"
            ],
            "cardinality_correction_scale": primary.diagnostics[
                "correction_scale"
            ],
            "expected_correction_scale": 1.0 / len(
                fitted.contribution_space.families
            ),
            "dufs_iu_identity_error": iu_identity_error,
            "dufs_effective_feature_count": gate_diag.get(
                "effective_feature_count"
            ),
            "availability": json.dumps(availability, sort_keys=True),
            "contract": json.dumps(contract, sort_keys=True, default=str),
        })
        print(f"semgrad__{dataset}: scores frozen", flush=True)

    write_csv(out / "fit_diagnostics.csv", diagnostics)
    fit_manifest = {
        "version": VERSION,
        "status": "scores_frozen_before_cb_bem_pairing",
        "datasets": list(DATASETS),
        "expected_rows": EXPECTED_ROWS,
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "primary": "cardinality_balanced_contribution_subspace_iu",
        "target": "bem_error = not candidate['bem_correct']",
        "forbidden_fit_keys": list(FORBIDDEN_FIT_KEYS),
        "telemetry_key_whitelist": list(TELEMETRY_KEYS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "dufs": {
            "seeds": list(DUFS_SEEDS),
            "epochs": DUFS_EPOCHS,
            "k": DUFS_K,
            "lambda": DUFS_LAMBDA,
        },
        "data_sha256": data_hashes,
        "bem_manifest_sha256": bem_manifest_hashes,
        "score_sha256": score_hashes,
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths().items()
        },
    }
    write_json(out / "FIT_MANIFEST.json", fit_manifest)
    print(out / "FIT_MANIFEST.json")


def verify_freeze(cache_root, out, manifest):
    if manifest["version"] != VERSION:
        raise RuntimeError("fit manifest version mismatch")
    for name, path in source_paths().items():
        if sha256_file(path) != manifest["source_sha256"][name]:
            raise RuntimeError(f"source changed after fit: {name}")
    for dataset in DATASETS:
        if sha256_file(data_path(cache_root, dataset)) != manifest[
            "data_sha256"
        ][dataset]:
            raise RuntimeError(f"data changed after fit: {dataset}")
        if sha256_file(bem_manifest_path(cache_root, dataset)) != manifest[
            "bem_manifest_sha256"
        ][dataset]:
            raise RuntimeError(f"BEM manifest changed after fit: {dataset}")
        score_path = out / "scores" / f"semgrad__{dataset}.npz"
        if sha256_file(score_path) != manifest["score_sha256"][dataset]:
            raise RuntimeError(f"score changed after fit: {dataset}")


def bootstrap_seed(dataset):
    return int(hashlib.sha256(
        f"{VERSION}:{dataset}:paired-stratified".encode()
    ).hexdigest()[:8], 16)


def _method_sort_structure(target, score):
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    combined_score = np.concatenate([score[positive], score[negative]])
    combined_positive = np.concatenate([
        np.ones(len(positive), dtype=bool),
        np.zeros(len(negative), dtype=bool),
    ])
    order = np.argsort(combined_score, kind="mergesort")
    sorted_score = combined_score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    return order, combined_positive[order], starts


def paired_stratified_bootstrap(target, score_map, *, seed):
    """Vectorized exact weighted-AUROC bootstrap shared across all methods."""
    target = np.asarray(target, dtype=int)
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    n_positive, n_negative = len(positive), len(negative)
    if not n_positive or not n_negative:
        raise ValueError("both target classes are required")
    structures = {
        method: _method_sort_structure(target, np.asarray(score, dtype=float))
        for method, score in score_map.items()
    }
    output = {
        method: np.empty(BOOTSTRAP_DRAWS, dtype=float)
        for method in score_map
    }
    rng = np.random.default_rng(seed)
    p_probability = np.full(n_positive, 1.0 / n_positive)
    n_probability = np.full(n_negative, 1.0 / n_negative)
    for first in range(0, BOOTSTRAP_DRAWS, BOOTSTRAP_CHUNK):
        last = min(first + BOOTSTRAP_CHUNK, BOOTSTRAP_DRAWS)
        size = last - first
        positive_count = rng.multinomial(
            n_positive, p_probability, size=size
        )
        negative_count = rng.multinomial(
            n_negative, n_probability, size=size
        )
        counts = np.concatenate([positive_count, negative_count], axis=1)
        for method, (order, is_positive, starts) in structures.items():
            sorted_count = counts[:, order]
            grouped_positive = np.add.reduceat(
                sorted_count * is_positive[None, :], starts, axis=1
            )
            grouped_negative = np.add.reduceat(
                sorted_count * (~is_positive)[None, :], starts, axis=1
            )
            negative_below = (
                np.cumsum(grouped_negative, axis=1) - grouped_negative
            )
            u_statistic = np.sum(
                grouped_positive
                * (negative_below + 0.5 * grouped_negative),
                axis=1,
            )
            output[method][first:last] = (
                u_statistic / (n_positive * n_negative)
            )
    return output


def contrast_row(scope, method, reference, point_delta, draws):
    return {
        "version": VERSION,
        "scope": scope,
        "method": method,
        "reference": reference,
        "delta_pp": float(100 * point_delta),
        "ci_low_pp": float(100 * np.quantile(draws, 0.025)),
        "ci_high_pp": float(100 * np.quantile(draws, 0.975)),
        "probability_nonpositive": float(np.mean(draws <= 0)),
    }


def hierarchical_draws(delta_by_dataset, method, reference):
    matrix = np.vstack([
        delta_by_dataset[dataset][(method, reference)]
        for dataset in DATASETS
    ])
    seed = int(hashlib.sha256(
        f"{VERSION}:{method}:{reference}:hierarchical".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    dataset_draw = rng.integers(
        0, len(DATASETS), size=(BOOTSTRAP_DRAWS, len(DATASETS))
    )
    within_draw = rng.integers(
        0, BOOTSTRAP_DRAWS, size=(BOOTSTRAP_DRAWS, len(DATASETS))
    )
    return matrix[dataset_draw, within_draw].mean(axis=1)


def report_scores(cache_root, out):
    cache_root, out = Path(cache_root), Path(out)
    manifest = json.loads(
        (out / "FIT_MANIFEST.json").read_text(encoding="utf-8")
    )
    verify_freeze(cache_root, out, manifest)

    cell_rows, summaries = [], []
    delta_by_dataset = {}
    for dataset in DATASETS:
        cache = load_cache(data_path(cache_root, dataset))
        row_ids, target = [], []
        for key in sorted(cache):
            candidates = cache[key].get("candidates")
            if not candidates:
                continue
            candidate = candidates[0]
            row_ids.append(str(key))
            # First per-row authoritative-target access in this protocol.
            target.append(int(not bool(candidate["bem_correct"])))
        target = np.asarray(target, dtype=int)
        bem_manifest = json.loads(
            bem_manifest_path(cache_root, dataset).read_text(encoding="utf-8")
        )
        if int(np.sum(target)) != bem_manifest["bem_incorrect_count"]:
            raise RuntimeError(f"BEM class count mismatch: {dataset}")

        with np.load(
            out / "scores" / f"semgrad__{dataset}.npz",
            allow_pickle=False,
        ) as scores:
            if list(scores["row_ids"].astype(str)) != row_ids:
                raise RuntimeError(f"row alignment changed: {dataset}")
            score_map = {
                method: np.asarray(scores[f"{method}_risk"], dtype=float)
                for method in METHODS
            }
        auroc = {
            method: float(roc_auc_score(target, score))
            for method, score in score_map.items()
        }
        bootstrap = paired_stratified_bootstrap(
            target, score_map, seed=bootstrap_seed(dataset)
        )
        delta_by_dataset[dataset] = {}
        for method, reference in COMPARISONS:
            delta_draws = bootstrap[method] - bootstrap[reference]
            delta_by_dataset[dataset][(method, reference)] = delta_draws
            summaries.append(contrast_row(
                dataset,
                method,
                reference,
                auroc[method] - auroc[reference],
                delta_draws,
            ))
        cell_rows.append({
            "version": VERSION,
            "dataset": dataset,
            "n": len(target),
            "n_bem_error": int(np.sum(target)),
            **{f"{method}_auroc": auroc[method] for method in METHODS},
        })

    for method, reference in COMPARISONS:
        point_delta = np.mean([
            row[f"{method}_auroc"] - row[f"{reference}_auroc"]
            for row in cell_rows
        ])
        summaries.append(contrast_row(
            "equal_dataset_hierarchical",
            method,
            reference,
            point_delta,
            hierarchical_draws(delta_by_dataset, method, reference),
        ))
    write_csv(out / "cell_results.csv", cell_rows)
    write_csv(out / "summary.csv", summaries)

    primary_cells = {
        row["scope"]: row for row in summaries
        if row["method"] == "cardinality" and row["reference"] == "iu"
    }
    hierarchical = primary_cells["equal_dataset_hierarchical"]
    fit_diag = list(csv.DictReader(
        (out / "fit_diagnostics.csv").open(encoding="utf-8")
    ))
    invariant_max = {
        "contribution_reconstruction": max(abs(float(
            row["contribution_reconstruction_error"]
        )) for row in fit_diag),
        "cardinality_weight_reconstruction": max(abs(float(
            row["cardinality_weight_reconstruction_error"]
        )) for row in fit_diag),
        "orthogonality": max(abs(float(
            row["cardinality_orthogonality"]
        )) for row in fit_diag),
        "trust_scale": max(abs(
            float(row["cardinality_correction_scale"])
            - float(row["expected_correction_scale"])
        ) for row in fit_diag),
        "iu_identity": max(abs(float(
            row["dufs_iu_identity_error"]
        )) for row in fit_diag),
    }
    reverse_positive = all(
        next(
            row for row in summaries
            if row["scope"] == dataset
            and row["method"] == "cardinality"
            and row["reference"] == "reverse_cardinality"
        )["delta_pp"] > 0
        for dataset in DATASETS
    )
    primary_positive = all(
        primary_cells[dataset]["delta_pp"] > 0 for dataset in DATASETS
    )
    worst_delta = min(
        primary_cells[dataset]["delta_pp"] for dataset in DATASETS
    )
    max_invariant = max(invariant_max.values())
    gates = [
        {
            "name": "positive CB delta in both datasets",
            "passed": primary_positive,
            "value": {
                dataset: primary_cells[dataset]["delta_pp"]
                for dataset in DATASETS
            },
        },
        {
            "name": "positive equal-dataset hierarchical interval",
            "passed": hierarchical["ci_low_pp"] > 0,
            "value": [hierarchical["ci_low_pp"], hierarchical["ci_high_pp"]],
        },
        {
            "name": "tail safety",
            "passed": worst_delta >= -1.0,
            "value": worst_delta,
        },
        {
            "name": "CB beats reversed direction in both datasets",
            "passed": reverse_positive,
            "value": reverse_positive,
        },
        {
            "name": "numerical invariants",
            "passed": max_invariant < 1e-10,
            "value": max_invariant,
        },
    ]
    result = {
        "version": VERSION,
        "status": "independent_examples_historical_labels_visible",
        "all_primary_gates_passed": bool(all(gate["passed"] for gate in gates)),
        "primary": hierarchical,
        "per_dataset_primary": {
            dataset: primary_cells[dataset] for dataset in DATASETS
        },
        "gates": gates,
        "invariant_max": invariant_max,
        "claim_boundary": (
            "CB was frozen before its SemGrad evaluation and the examples are "
            "independent, but BEM labels and prior baseline results historically "
            "existed in the repository."
        ),
    }
    write_json(out / "RESULT.json", result)

    def signed(value):
        return f"{float(value):+.3f}"

    lines = [
        "# Frozen CB-CS-IU transfer to SemGrad",
        "",
        "**Status:** independent-example frozen transfer with historical label "
        "visibility disclosed.",
        "",
        f"Across SciQ and TruthfulQA with equal dataset weight, CB-CS-IU "
        f"changed BEM-error AUROC by **{signed(hierarchical['delta_pp'])}pp** "
        f"versus ordinary IU. The hierarchical paired interval is "
        f"[{signed(hierarchical['ci_low_pp'])}, "
        f"{signed(hierarchical['ci_high_pp'])}]pp.",
        "",
        "## Per-dataset primary result",
        "",
        "| dataset | IU AUROC | CB AUROC | delta | paired 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in cell_rows:
        row = primary_cells[cell["dataset"]]
        lines.append(
            f"| {cell['dataset']} | {cell['iu_auroc']:.4f} "
            f"| {cell['cardinality_auroc']:.4f} "
            f"| {signed(row['delta_pp'])}pp "
            f"| [{signed(row['ci_low_pp'])}, {signed(row['ci_high_pp'])}] |"
        )
    lines.extend([
        "",
        "## Equal-dataset mechanism contrasts",
        "",
        "| contrast | delta | hierarchical 95% interval |",
        "|---|---:|---:|",
    ])
    for row in summaries:
        if row["scope"] != "equal_dataset_hierarchical":
            continue
        lines.append(
            f"| `{row['method']} - {row['reference']}` "
            f"| {signed(row['delta_pp'])}pp "
            f"| [{signed(row['ci_low_pp'])}, {signed(row['ci_high_pp'])}] |"
        )
    lines.extend(["", "## Frozen gates", ""])
    for gate in gates:
        lines.append(
            f"- **{'PASS' if gate['passed'] else 'FAIL'} — "
            f"{gate['name']}:** `{gate['value']}`"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "Fit received telemetry-only dictionaries and could not access "
        "`bem_correct`, `bem_score`, or the temporary ROUGE-L `label`. Data, "
        "BEM manifests, scores, source files, and row identities were verified "
        "before BEM-error evaluation.",
        "",
        "These are independent answer-level examples and a different benchmark "
        "protocol from the development evidence. They are not pristine in the "
        "strongest historical sense because their labels and earlier IU/DUFS "
        "results already existed elsewhere in the repository.",
        "",
    ])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "report"))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.phase == "fit":
        fit_scores(args.cache_root, args.out)
    else:
        report_scores(args.cache_root, args.out)


if __name__ == "__main__":
    main()
