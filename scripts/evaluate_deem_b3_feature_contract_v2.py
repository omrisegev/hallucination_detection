#!/usr/bin/env python3
"""Strict post-freeze evaluation of B3 Feature Contract V2 versus frozen B3."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import load_registry, load_target_free_bundle  # noqa: E402


SCHEMA = "deem_b3_feature_contract_v2_evaluation_2026_08_25"
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("B3", "B3_V2")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing empty evaluation table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_content_hash(value: Mapping[str, Any]) -> None:
    copy = dict(value)
    expected = copy.pop("content_sha256", None)
    if canonical_sha256(copy) != expected:
        raise ValueError("canonical content hash mismatch")


def metric(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if y.shape != score.shape or len(np.unique(y)) != 2 or not np.isfinite(score).all():
        raise ValueError("invalid score/label pair")
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def aggregate(per_cell: list[dict[str, Any]], method: str, name: str) -> dict[str, Any]:
    rows = [row for row in per_cell if row["method"] == method]
    families: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        families[row["dataset_family"]].append(float(row[name]))
    family_means = {family: float(np.mean(values)) for family, values in families.items()}
    return {
        "method": method,
        "metric": name,
        "equal_family": float(np.mean(list(family_means.values()))),
        "cell_macro": float(np.mean([row[name] for row in rows])),
        "qa_macro": float(np.mean([row[name] for row in rows if row["task_type"] == "QA"])),
        "math_macro": float(np.mean([row[name] for row in rows if row["task_type"] == "math"])),
        "family_means": family_means,
    }


def paired_bootstrap(
    per_cell: list[dict[str, Any]], name: str, *, draws: int, seed: int
) -> dict[str, Any]:
    lookup = {(row["cell_id"], row["method"]): row for row in per_cell}
    by_family: dict[str, list[float]] = defaultdict(list)
    for cell in sorted({row["cell_id"] for row in per_cell}):
        base = lookup[(cell, "B3")]
        candidate = lookup[(cell, "B3_V2")]
        by_family[base["dataset_family"]].append(float(candidate[name] - base[name]))
    family_delta = {family: float(np.mean(values)) for family, values in by_family.items()}
    families = sorted(by_family)
    rng = np.random.default_rng(seed)
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.choice(families, len(families), replace=True)
        values = []
        for family in selected:
            cell_values = np.asarray(by_family[family], dtype=np.float64)
            values.append(float(np.mean(cell_values[rng.integers(0, len(cell_values), len(cell_values))])))
        distribution[draw] = float(np.mean(values))
    observed = float(np.mean(list(family_delta.values())))
    null = []
    family_array = np.asarray([family_delta[family] for family in families])
    for signs in itertools.product((-1.0, 1.0), repeat=len(families)):
        null.append(float(np.mean(family_array * np.asarray(signs))))
    null_array = np.asarray(null)
    return {
        "metric": name,
        "observed": observed,
        "bootstrap95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "exact_family_signflip_one_sided_p": float(np.mean(null_array >= observed - 1e-15)),
        "family_delta": family_delta,
        "lofo": {
            omit: float(np.mean([value for family, value in family_delta.items() if family != omit]))
            for omit in families
        },
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    definition = json.loads((args.run_dir / "RUN_DEFINITION.json").read_text(encoding="utf-8"))
    freeze = json.loads((args.run_dir / "SCORE_FREEZE.json").read_text(encoding="utf-8"))
    completion = json.loads((args.run_dir / "FIT_COMPLETE.json").read_text(encoding="utf-8"))
    check_content_hash(definition)
    check_content_hash(freeze)
    check_content_hash(completion)
    if definition["config_sha256"] != sha256_file(args.config):
        raise ValueError("config drift")
    expected_sources = {
        "runner": ROOT / "scripts/run_deem_b3_feature_contract_v2.py",
        "core": ROOT / "spectral_utils/deem_b3_feature_contract_v2.py",
        "frozen_b3_core": ROOT / "spectral_utils/residual_graph_deem.py",
        "config": args.config,
    }
    for name, path in expected_sources.items():
        if definition["source_hashes"][name] != sha256_file(path):
            raise ValueError(f"fit source drift: {name}")
    if freeze["expected_fit_artifacts"] != 120 or len(freeze["records"]) != 120:
        raise ValueError("V2 fit roster incomplete")
    if not completion["all_healthy"] or completion["fit_count"] != 120:
        raise ValueError("V2 fit completion unhealthy")
    if any(definition[key] for key in ("labels_accessed_during_fit", "target_module_imported_during_fit")):
        raise ValueError("fit firewall flag failed")
    if any(freeze[key] for key in ("labels_accessed_during_fit", "target_module_imported_during_fit")):
        raise ValueError("score-freeze firewall flag failed")

    baseline_manifest = json.loads((args.baseline_dir / "SCORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    check_content_hash(baseline_manifest)
    baseline_hashes = {row["path"]: row["sha256"] for row in baseline_manifest["artifacts"]}
    record_lookup = {(row["cell_id"], int(row["seed"])): row for row in freeze["records"]}
    if len(record_lookup) != 120:
        raise ValueError("duplicate V2 score-freeze records")

    bundles, scores, stability = {}, {}, []
    for cell_row in registry["cells"]:
        cell = str(cell_row["cell_id"])
        bundle = load_target_free_bundle(args.bundle_dir / f"{cell}.npz")
        bundles[cell] = bundle
        scores[cell] = {}
        method_seed_scores: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
        for seed in SEEDS:
            record = record_lookup[(cell, seed)]
            npz_path = args.run_dir / record["npz"]
            json_path = args.run_dir / record["json"]
            if sha256_file(npz_path) != record["npz_sha256"] or sha256_file(json_path) != record["json_sha256"]:
                raise ValueError(f"V2 frozen artifact hash mismatch: {cell} seed {seed}")
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            check_content_hash(metadata)
            if not metadata["health"]["healthy"] or metadata["labels_accessed_during_fit"]:
                raise ValueError("V2 metadata health/firewall failure")
            with np.load(npz_path, allow_pickle=False) as data:
                score = np.asarray(data["score"], dtype=np.float64)
                logit = np.asarray(data["logit"], dtype=np.float64)
                contribution = np.asarray(data["contributions"], dtype=np.float64)
                row_ids = tuple(str(value) for value in data["row_id"].tolist())
                if row_ids != bundle.row_ids or score.shape != (len(bundle.row_ids),):
                    raise ValueError("V2 row alignment mismatch")
                if not np.allclose(score, 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700))), atol=1e-12, rtol=0):
                    raise ValueError("V2 sigmoid reconstruction mismatch")
                if np.max(np.abs(float(data["aligned_bias"]) + contribution.sum(axis=1) - logit)) > 1e-8:
                    raise ValueError("V2 contribution reconstruction mismatch")
            method_seed_scores["B3_V2"].append(score)

            relative = f"fits/{cell}/B3__seed{seed}.npz"
            baseline_path = args.baseline_dir / relative
            if baseline_hashes.get(relative) != sha256_file(baseline_path):
                raise ValueError(f"frozen B3 hash mismatch: {cell} seed {seed}")
            with np.load(baseline_path, allow_pickle=False) as data:
                base_score = np.asarray(data["score"], dtype=np.float64)
            if base_score.shape != score.shape or not np.isfinite(base_score).all():
                raise ValueError("baseline score shape/finite mismatch")
            method_seed_scores["B3"].append(base_score)
        for method in METHODS:
            matrix = np.stack(method_seed_scores[method], axis=0)
            scores[cell][method] = matrix.mean(axis=0)
            pairwise = [
                float(spearmanr(matrix[i], matrix[j]).statistic)
                for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))
            ]
            stability.append({
                "cell_id": cell,
                "method": method,
                "median_seed_spearman": float(np.median(pairwise)),
                "min_seed_spearman": float(np.min(pairwise)),
            })
    pre_label = {
        "schema": SCHEMA + "_pre_label",
        "fit_count": 120,
        "cell_count": 24,
        "score_freeze_sha256": sha256_file(args.run_dir / "SCORE_FREEZE.json"),
        "baseline_freeze_sha256": sha256_file(args.baseline_dir / "SCORE_FREEZE_MANIFEST.json"),
        "all_score_and_row_checks_passed": True,
        "labels_imported": False,
    }
    pre_label["content_sha256"] = canonical_sha256(pre_label)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_dir / "PRE_LABEL_FREEZE.json", pre_label)
    return {"registry": registry, "config": config, "bundles": bundles, "scores": scores, "stability": stability, "pre_label": pre_label}


def draw_plot(path: Path, comparison: Mapping[str, Any]) -> None:
    width, height = 1100, 570
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    try:
        title = ImageFont.truetype(font_path, 23)
        body = ImageFont.truetype(font_path, 14)
        label = ImageFont.truetype(font_path, 13)
    except OSError:
        title = body = label = ImageFont.load_default()
    draw.text((35, 20), "B3 Feature Contract V2 vs frozen B3", fill="#20242a", font=title)
    draw.text((35, 54), "Equal-dataset-family AUROC deltas; positive means V2 is better", fill="#20242a", font=body)
    deltas = comparison["family_delta"]
    values = list(deltas.values()) + [float(comparison["equal_family_auroc_delta"])]
    names = list(deltas) + ["OVERALL"]
    limit = max(0.0025, max(abs(value) for value in values) * 1.15)
    center, scale = 540, 390 / limit
    draw.line((center, 85, center, 520), fill="#111111", width=2)
    for index, (name, value) in enumerate(zip(names, values)):
        y = 100 + index * 43
        x = center + value * scale
        color = "#3973ac" if value >= 0 else "#a64b4b"
        draw.text((35, y + 3), name, fill="#20242a", font=label)
        draw.rectangle((min(center, x), y, max(center, x), y + 22), fill=color)
        draw.text((max(center, x) + 8 if value >= 0 else min(center, x) - 82, y + 3), f"{value:+.5f}", fill="#20242a", font=label)
    draw.text((35, 548), "Source: frozen mean-of-5 scores; 24 cells; labels opened only after score freeze.", fill="#20242a", font=label)
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/b3_feature_contract_v2")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/b3_frozen")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/bundles")
    parser.add_argument("--sidecar-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/label_sidecars")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deem_b3_feature_contract_v2.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/b3_feature_contract_v2_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = preflight(args)
    # This is the only point at which the target-bearing module is imported.
    labels_module = importlib.import_module("spectral_utils.residual_graph_deem_labels")
    per_cell = []
    for cell_row in state["registry"]["cells"]:
        cell = str(cell_row["cell_id"])
        sidecar = labels_module.load_label_sidecar(args.sidecar_dir / f"{cell}.npz")
        y = labels_module.join_labels_by_id(state["bundles"][cell], sidecar)
        for method in METHODS:
            per_cell.append({
                "cell_id": cell,
                "dataset_family": state["bundles"][cell].dataset_family,
                "task_type": state["bundles"][cell].task_type,
                "method": method,
                **metric(y, state["scores"][cell][method]),
            })
    summaries = [aggregate(per_cell, method, name) for method in METHODS for name in ("auroc", "auprc")]
    bootstrap = paired_bootstrap(
        per_cell, "auroc",
        draws=int(state["config"]["evaluation"]["bootstrap_draws"]),
        seed=int(state["config"]["evaluation"]["bootstrap_seed"]),
    )
    auprc_bootstrap = paired_bootstrap(
        per_cell, "auprc",
        draws=int(state["config"]["evaluation"]["bootstrap_draws"]),
        seed=int(state["config"]["evaluation"]["bootstrap_seed"]) + 1,
    )
    lookup = {(row["method"], row["metric"]): row for row in summaries}
    base = lookup[("B3", "auroc")]
    candidate = lookup[("B3_V2", "auroc")]
    cell_lookup = {(row["cell_id"], row["method"]): row for row in per_cell}
    cell_deltas = [
        float(cell_lookup[(cell, "B3_V2")]["auroc"] - cell_lookup[(cell, "B3")]["auroc"])
        for cell in sorted(state["bundles"])
    ]
    tolerance = float(state["config"]["evaluation"]["tie_tolerance"])
    comparison = {
        "reference": "B3",
        "candidate": "B3_V2",
        "b3_equal_family_auroc": base["equal_family"],
        "b3_v2_equal_family_auroc": candidate["equal_family"],
        "equal_family_auroc_delta": bootstrap["observed"],
        "auroc_bootstrap95": bootstrap["bootstrap95"],
        "auroc_exact_family_signflip_one_sided_p": bootstrap["exact_family_signflip_one_sided_p"],
        "equal_family_auprc_delta": auprc_bootstrap["observed"],
        "auprc_bootstrap95": auprc_bootstrap["bootstrap95"],
        "qa_auroc_delta": candidate["qa_macro"] - base["qa_macro"],
        "math_auroc_delta": candidate["math_macro"] - base["math_macro"],
        "cell_macro_auroc_delta": candidate["cell_macro"] - base["cell_macro"],
        "wins": int(sum(value > tolerance for value in cell_deltas)),
        "ties": int(sum(abs(value) <= tolerance for value in cell_deltas)),
        "losses": int(sum(value < -tolerance for value in cell_deltas)),
        "worst_cell_delta": float(min(cell_deltas)),
        "family_delta": bootstrap["family_delta"],
    }
    promotion = (
        comparison["equal_family_auroc_delta"] >= float(state["config"]["evaluation"]["promotion_delta"])
        and comparison["auroc_bootstrap95"][0] > 0
        and comparison["auroc_exact_family_signflip_one_sided_p"] <= 0.05
    )
    decision = {
        "schema": SCHEMA + "_decision",
        "decision": "PROMOTE_B3_V2_BASELINE" if promotion else "KEEP_FROZEN_B3_BASELINE",
        "promotion": bool(promotion),
        "comparison": comparison,
        "labels_opened_after_pre_label_freeze": True,
    }
    write_csv(args.out_dir / "PER_CELL_METRICS.csv", per_cell)
    write_csv(args.out_dir / "SEED_STABILITY.csv", state["stability"])
    atomic_write_json(args.out_dir / "SUMMARY.json", summaries)
    atomic_write_json(args.out_dir / "COMPARISON.json", comparison)
    atomic_write_json(args.out_dir / "DECISION.json", decision)
    draw_plot(args.out_dir / "b3_v2_family_deltas.png", comparison)
    report = [
        "# B3 Feature Contract V2 evaluation",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        f"- Frozen B3 equal-family AUROC: {comparison['b3_equal_family_auroc']:.9f}",
        f"- B3 V2 equal-family AUROC: {comparison['b3_v2_equal_family_auroc']:.9f}",
        f"- Delta AUROC: {comparison['equal_family_auroc_delta']:+.9f}",
        f"- Descriptive 95% bootstrap: [{comparison['auroc_bootstrap95'][0]:+.9f}, {comparison['auroc_bootstrap95'][1]:+.9f}]",
        f"- Exact family sign-flip one-sided p: {comparison['auroc_exact_family_signflip_one_sided_p']:.6f}",
        f"- Wins/ties/losses: {comparison['wins']}/{comparison['ties']}/{comparison['losses']}",
        f"- Delta AUPRC: {comparison['equal_family_auprc_delta']:+.9f}",
        "",
        "This isolates the cleaned feature contract and revised four-block grouping; no additional input network was added.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    output_inventory = {
        path.name: sha256_file(path)
        for path in sorted(args.out_dir.iterdir())
        if path.is_file() and path.name != "EVALUATION_FREEZE.json"
    }
    evaluation_freeze = {
        "schema": SCHEMA + "_freeze",
        "source_sha256": sha256_file(Path(__file__)),
        "pre_label_freeze_sha256": sha256_file(args.out_dir / "PRE_LABEL_FREEZE.json"),
        "outputs": output_inventory,
    }
    evaluation_freeze["content_sha256"] = canonical_sha256(evaluation_freeze)
    atomic_write_json(args.out_dir / "EVALUATION_FREEZE.json", evaluation_freeze)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
