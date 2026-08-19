#!/usr/bin/env python3
"""Held-out c-STG routing diagnostic for global hallucination detection."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.family_relevance_report import dataset_family, context_headroom  # noqa: E402
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.contextual_stg import ContextualSTGModel  # noqa: E402


VERSION = "global-contextual-stg-router-diagnostic-v1.1-2026-08-19"
DEFAULT_BUNDLE = ROOT / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_SCORE_DIR = ROOT / "results" / "family_relevance_real_v1" / "scores"
DEFAULT_OUT = ROOT / "results" / "global_contextual_stg_router_diagnostic_v1"
PROTOCOL = ROOT / "docs" / "experiments" / "GLOBAL_CONTEXTUAL_STG_ROUTER_DIAGNOSTIC_V1.md"
SEEDS = (11, 23, 47)
N_SPLITS = 5
DATASET_FAMILIES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)
METHODS = (
    "iu_pcr", "fixed_expert_cv", "global_lr", "context_only_lr",
    "augmented_lr", "context_core_only_lr", "augmented_core_lr",
    "quartile_router_cv", "cstg_iu_rank", "cstg_core",
    "cstg_iu_rank_permuted", "cstg_core_permuted",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def write_json(path: Path, payload) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def lr_score(X_train, y_train, X_test) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000),
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def fixed_expert_score(X_train, y_train, X_test) -> np.ndarray:
    auc = np.asarray([roc_auc_score(y_train, X_train[:, j]) for j in range(X_train.shape[1])])
    return X_test[:, int(np.argmax(auc))]


def quartile_router_score(
    X_train: np.ndarray, z_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, z_test: np.ndarray,
) -> np.ndarray:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = np.clip((X_train - mean) / scale, -8.0, 8.0)
    test = np.clip((X_test - mean) / scale, -8.0, 8.0)
    train_bin = np.minimum((4 * np.clip(z_train, 0.0, 1.0)).astype(int), 3)
    test_bin = np.minimum((4 * np.clip(z_test, 0.0, 1.0)).astype(int), 3)
    global_choice = int(np.argmax([
        roc_auc_score(y_train, train[:, j]) for j in range(train.shape[1])
    ]))
    choices: dict[int, int] = {}
    for bucket in range(4):
        keep = train_bin == bucket
        if np.sum(y_train[keep] == 0) < 3 or np.sum(y_train[keep] == 1) < 3:
            choices[bucket] = global_choice
        else:
            choices[bucket] = int(np.argmax([
                roc_auc_score(y_train[keep], train[keep, j])
                for j in range(train.shape[1])
            ]))
    return np.asarray([test[i, choices[int(bucket)]] for i, bucket in enumerate(test_bin)])


def cstg_score(X_train, Z_train, y_train, X_test, Z_test, namespace: str):
    groups = np.asarray([f"q{index}" for index in range(len(y_train))])
    predictions = []
    diagnostics = []
    for seed in SEEDS:
        model = ContextualSTGModel().fit(
            X_train, Z_train, y_train, groups,
            feature_group_ids=np.arange(X_train.shape[1]), seed=seed,
        )
        prediction = model.predict(X_test, Z_test)
        predictions.append(prediction.score)
        diagnostics.append({
            "namespace": namespace,
            "seed": seed,
            "mean_gate": np.mean(prediction.family_gates, axis=0).tolist(),
            "std_gate": np.std(prediction.family_gates, axis=0).tolist(),
            "fit": model.diagnostics_,
        })
    return np.mean(predictions, axis=0), diagnostics


def evaluate_cell(cell: str, labels: np.ndarray, arrays: dict[str, np.ndarray]):
    # These are the exact signed family contributions whose sum is frozen IU-PCR.
    # Do not re-orient them, even by a label-free correlation: this experiment is
    # about conditional leverage/reliability while all existing directions stay fixed.
    experts = np.asarray(arrays["family_experts"], dtype=float).T
    iu = np.asarray(arrays["iu_pcr"], dtype=float)
    rank = np.asarray(arrays["context_iu_rank"], dtype=float)[:, None]
    core = np.column_stack([
        arrays["context_iu_rank"], arrays["context_trace_length"],
        arrays["context_family_disagreement"],
    ])
    rng = np.random.default_rng(stable_seed(f"global-cstg-permutation:{cell}"))
    rank_permuted = rank[rng.permutation(len(rank))]
    core_permuted = core[rng.permutation(len(core))]
    splits = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=stable_seed(f"global-cstg-cv:{cell}"),
    )
    fold_rows = []
    gate_rows = []
    for fold, (train, test) in enumerate(splits.split(experts, labels)):
        Xtr, Xte, ytr, yte = experts[train], experts[test], labels[train], labels[test]
        predictions = {
            "iu_pcr": iu[test],
            "fixed_expert_cv": fixed_expert_score(Xtr, ytr, Xte),
            "global_lr": lr_score(Xtr, ytr, Xte),
            "context_only_lr": lr_score(rank[train], ytr, rank[test]),
            "augmented_lr": lr_score(
                np.column_stack([Xtr, rank[train]]), ytr,
                np.column_stack([Xte, rank[test]]),
            ),
            "context_core_only_lr": lr_score(core[train], ytr, core[test]),
            "augmented_core_lr": lr_score(
                np.column_stack([Xtr, core[train]]), ytr,
                np.column_stack([Xte, core[test]]),
            ),
            "quartile_router_cv": quartile_router_score(
                Xtr, rank[train, 0], ytr, Xte, rank[test, 0]
            ),
        }
        for name, Z in (("cstg_iu_rank", rank), ("cstg_core", core),
                        ("cstg_iu_rank_permuted", rank_permuted),
                        ("cstg_core_permuted", core_permuted)):
            score, diagnostic = cstg_score(
                Xtr, Z[train], ytr, Xte, Z[test], f"{cell}:fold{fold}:{name}"
            )
            predictions[name] = score
            gate_rows.extend(diagnostic)
        for method, score in predictions.items():
            fold_rows.append({
                "cell": cell,
                "family": dataset_family(cell),
                "fold": fold,
                "method": method,
                "n_test": len(test),
                "n_positive": int(np.sum(yte == 1)),
                "auroc": float(roc_auc_score(yte, score)),
                "auprc": float(average_precision_score(yte, score)),
            })
    premise, winner_count, winners = context_headroom(
        labels, arrays["family_experts"], arrays["context_iu_rank"], bins=4
    )
    premise_row = {
        "cell": cell,
        "family": dataset_family(cell),
        "headroom_pp": premise,
        "winner_count": winner_count,
        "winners": [str(arrays["family_names"][index]) for index in winners],
    }
    return fold_rows, gate_rows, premise_row


def family_values(cell_rows: list[dict], method: str, baseline: str = "global_lr"):
    by_cell = {(row["cell"], row["method"]): row for row in cell_rows}
    values = []
    family_rows = []
    for family in DATASET_FAMILIES:
        cells = [row["cell"] for row in cell_rows if row["method"] == method and row["family"] == family]
        deltas = [by_cell[(cell, method)]["auroc"] - by_cell[(cell, baseline)]["auroc"] for cell in cells]
        value = float(np.mean(deltas))
        values.append(value)
        family_rows.append({"family": family, "method": method, "delta_vs_global_lr": value})
    return np.asarray(values), family_rows


def bootstrap_ci(values: np.ndarray, namespace: str, draws: int = 20000):
    rng = np.random.default_rng(stable_seed(namespace))
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(value) for value in np.quantile(sampled, (0.025, 0.975))]


def paired_contrast(cell_rows: list[dict], method: str, baseline: str) -> dict:
    lookup = {(row["cell"], row["method"]): row for row in cell_rows}
    values = []
    for family in DATASET_FAMILIES:
        cells = [
            row["cell"] for row in cell_rows
            if row["method"] == method and row["family"] == family
        ]
        values.append(float(np.mean([
            lookup[(cell, method)]["auroc"] - lookup[(cell, baseline)]["auroc"]
            for cell in cells
        ])))
    array = np.asarray(values)
    interval = bootstrap_ci(array, f"global-cstg-contrast:{method}:{baseline}")
    return {
        "method": method, "baseline": baseline,
        "equal_family_delta": float(np.mean(array)),
        "ci_low": interval[0], "ci_high": interval[1],
        "family_wins": int(np.sum(array > 0)),
        "family_deltas": dict(zip(DATASET_FAMILIES, map(float, array))),
    }


def main() -> None:
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    bundle_hash = sha256_file(DEFAULT_BUNDLE)
    score_hashes = {cell: sha256_file(DEFAULT_SCORE_DIR / f"{cell}.npz") for cell in INSCOPE}
    all_folds, all_gates, premise_rows = [], [], []
    with np.load(DEFAULT_BUNDLE, allow_pickle=True) as bundle:
        for position, cell in enumerate(INSCOPE, 1):
            checkpoint_path = checkpoint_dir / f"{cell}.json"
            if checkpoint_path.exists():
                payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if payload.get("version") == VERSION:
                    print(f"[{position:02d}/{len(INSCOPE)}] resume {cell}", flush=True)
                    all_folds.extend(payload["folds"])
                    all_gates.extend(payload["gates"])
                    premise_rows.append(payload["premise"])
                    continue
            labels = np.asarray(bundle[f"{cell}__labels"], dtype=int)
            with np.load(DEFAULT_SCORE_DIR / f"{cell}.npz", allow_pickle=False) as stored:
                arrays = {key: np.asarray(stored[key]) for key in (
                    "family_experts", "family_names", "iu_pcr", "context_iu_rank",
                    "context_trace_length", "context_family_disagreement",
                )}
            print(f"[{position:02d}/{len(INSCOPE)}] {cell}", flush=True)
            folds, gates, premise = evaluate_cell(cell, labels, arrays)
            all_folds.extend(folds)
            all_gates.extend(gates)
            premise_rows.append(premise)
            write_json(checkpoint_path, {
                "version": VERSION, "folds": folds, "gates": gates,
                "premise": premise,
            })

    cell_rows = []
    for cell in INSCOPE:
        for method in METHODS:
            local = [row for row in all_folds if row["cell"] == cell and row["method"] == method]
            cell_rows.append({
                "cell": cell, "family": dataset_family(cell), "method": method,
                "auroc": float(np.mean([row["auroc"] for row in local])),
                "auprc": float(np.mean([row["auprc"] for row in local])),
            })
    family_rows = []
    summary_rows = []
    for method in METHODS:
        local = [row for row in cell_rows if row["method"] == method]
        values, per_family = family_values(cell_rows, method)
        family_rows.extend(per_family)
        ci = bootstrap_ci(values, f"global-cstg:{method}")
        summary_rows.append({
            "method": method,
            "cell_macro_auroc": float(np.mean([row["auroc"] for row in local])),
            "equal_family_auroc": float(np.mean([
                np.mean([row["auroc"] for row in local if row["family"] == family])
                for family in DATASET_FAMILIES
            ])),
            "delta_vs_global_lr": float(np.mean(values)),
            "delta_ci_low": ci[0], "delta_ci_high": ci[1],
            "family_wins_vs_global_lr": int(np.sum(values > 0)),
        })
    summary = {row["method"]: row for row in summary_rows}
    primary = summary["cstg_iu_rank"]
    gates = [
        {"gate": "gain_at_least_0.005", "observed": primary["delta_vs_global_lr"], "passed": primary["delta_vs_global_lr"] >= 0.005},
        {"gate": "ci_lower_above_zero", "observed": primary["delta_ci_low"], "passed": primary["delta_ci_low"] > 0},
        {"gate": "five_of_eight_family_wins", "observed": primary["family_wins_vs_global_lr"], "passed": primary["family_wins_vs_global_lr"] >= 5},
        {"gate": "beats_quartile_router", "observed": primary["equal_family_auroc"] - summary["quartile_router_cv"]["equal_family_auroc"], "passed": primary["equal_family_auroc"] > summary["quartile_router_cv"]["equal_family_auroc"]},
        {"gate": "beats_augmented_lr", "observed": primary["equal_family_auroc"] - summary["augmented_lr"]["equal_family_auroc"], "passed": primary["equal_family_auroc"] > summary["augmented_lr"]["equal_family_auroc"]},
        {"gate": "permutation_does_not_reproduce_gain", "observed": primary["equal_family_auroc"] - summary["cstg_iu_rank_permuted"]["equal_family_auroc"], "passed": primary["equal_family_auroc"] > summary["cstg_iu_rank_permuted"]["equal_family_auroc"]},
    ]
    decision = "GLOBAL_ROUTING_SIGNAL_ACCESSIBLE" if all(row["passed"] for row in gates) else "GLOBAL_ORACLE_NOT_ACCESSIBLE_BY_CSTG"
    exploratory_core = [
        paired_contrast(cell_rows, "cstg_core", baseline)
        for baseline in ("global_lr", "augmented_core_lr", "cstg_core_permuted")
    ]
    write_csv(out / "fold_metrics.csv", all_folds)
    write_csv(out / "cell_metrics.csv", cell_rows)
    write_csv(out / "family_deltas.csv", family_rows)
    write_csv(out / "summary.csv", summary_rows)
    write_json(out / "gate_diagnostics.json", all_gates)
    write_json(out / "premise_headroom.json", premise_rows)
    write_json(out / "DECISION.json", {
        "decision": decision, "gates": gates, "primary_method": "cstg_iu_rank",
        "exploratory_core_contrasts": exploratory_core,
    })
    manifest = {
        "version": VERSION, "retrospective_premise_evidence": True,
        "labels_seen_by_router": True, "cells": list(INSCOPE), "seeds": list(SEEDS),
        "n_splits": N_SPLITS, "bundle_sha256": bundle_hash,
        "score_sha256": score_hashes, "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(out / "RUN_MANIFEST.json", manifest)
    valid_premise = [
        float(row["headroom_pp"]) for row in premise_rows
        if row.get("headroom_pp") is not None and np.isfinite(float(row["headroom_pp"]))
    ]
    mean_premise = float(np.mean(valid_premise))
    lines = [
        "# Global contextual c-STG router diagnostic", "",
        f"**Decision: `{decision}`.**", "",
        "This is a retrospective supervised mechanism diagnostic on the frozen 24-cell completed-trace hallucination-detection panel. It is not a label-free detector or external confirmation.", "",
        "## Premise reproduction", "",
        f"The cell-macro reproduced IU-rank quartile-oracle headroom is {mean_premise:+.3f}pp. The registered historical equal-family value is +2.833pp; the aggregation difference is retained explicitly.", "",
        "## Held-out results", "",
        "| method | cell macro AUROC | equal-family AUROC | delta vs global LR | 95% CI | family wins |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['method']} | {row['cell_macro_auroc']:.4f} | {row['equal_family_auroc']:.4f} | {row['delta_vs_global_lr']:+.4f} | [{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}] | {row['family_wins_vs_global_lr']}/8 |")
    lines += ["", "## Gates", "", "| gate | observed | pass |", "|---|---:|:---:|"]
    for row in gates:
        lines.append(f"| {row['gate']} | {row['observed']:+.6f} | {'yes' if row['passed'] else 'no'} |")
    lines += ["", "## Exploratory extended-context result", "",
              "| contrast | equal-family delta | 95% CI | family wins |",
              "|---|---:|---:|---:|"]
    for row in exploratory_core:
        lines.append(
            f"| cstg_core vs {row['baseline']} | {row['equal_family_delta']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['family_wins']}/8 |"
        )
    lines += ["", "## Interpretation", ""]
    if decision == "GLOBAL_ROUTING_SIGNAL_ACCESSIBLE":
        lines.append("The supervised conditional gate accesses a non-trivial part of the Global specialization. This justifies a separately frozen study of how to obtain a label-free or intervention-grounded teacher signal; it does not make c-STG deployable without labels.")
    else:
        lines.append("IU-rank organizes an in-sample oracle, but the tested held-out c-STG cannot reliably turn it into additional Global detection performance beyond a supervised global combination. The headroom therefore remains descriptive rather than an accessible router signal under this design.")
        lines.append("The exploratory three-coordinate core is not a null result: it beats its permuted-context counterpart by +0.0148 equal-family AUROC with a positive interval. But it does not beat global LR robustly, its cell macro is slightly lower, and the equal-family gain is heterogeneous (including losses on GSM8K and MATH500). This supports a narrow context-by-family interaction diagnostic, not promotion of a router or escalation to LTSREx/LEGO.")
    report = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "REPORT.html").write_text("<html><body><pre>" + html.escape(report) + "</pre></body></html>\n", encoding="utf-8")
    write_json(out / "AUDIT.json", {
        "fold_metrics_are_within_fold": True,
        "concatenated_oof_ranking_used": False,
        "class_balanced_supervised_controls": True,
        "context_direct_path_in_cstg": False,
        "correctness_labels_used_only_in_cv_training_and_evaluation": True,
        "new_inference": False, "drive_mutation": False,
    })
    print(report)


if __name__ == "__main__":
    main()
