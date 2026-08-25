#!/usr/bin/env python3
"""Build and audit a conservative, target-free Feature Contract V2.

This stage deliberately does not fit a detector.  It only:
1. replaces the duplicate H15/Hsaved entropy coordinates by an exact common/
   support-difference transform;
2. moves trace length from the expert inventory to a context sidecar;
3. removes the exactly derived high/low FFT ratio when its parents exist; and
4. measures remaining predictive-distribution redundancy with group-CV.

No correctness sidecar or target module is imported by this file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from spectral_utils.group_free_research import derive_feature_dag
from spectral_utils.residual_graph_deem import (
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (
    assert_no_target_fields,
    load_registry,
    load_target_free_bundle,
)


SCHEMA = "feature_contract_v2_2026_08_25"
PREDICTIVE_FEATURES = (
    "epr",
    "mean_logprob_entropy",
    "mean_top1_logprob",
    "logprob_margin",
    "varentropy",
    "renyi_entropy_2",
    "topk_tail_mass",
)
N_PANELS = 20
N_GROUPS = 180
N_FOLDS = 5
RIDGE_ALPHA = 1.0


def stable_hex(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_top_k(repo: Path, cell: Mapping[str, object]) -> tuple[int, Path]:
    cell_id = str(cell["cell_id"])
    path = repo / "dataset_cache" / "repgrid" / cell_id / "manifest.json"
    if not path.is_file() or sha256_file(path) != cell["source"]["manifest_sha256"]:
        raise ValueError(f"manifest binding failed: {cell_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    top_k = int(value["logprob_top_k"])
    if top_k < 2:
        raise ValueError(f"invalid manifest top-k: {cell_id}")
    return top_k, path


def transform_contract(
    X: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...], dict[str, float]]:
    lookup = {name: index for index, name in enumerate(names)}
    if "epr" not in lookup or "mean_logprob_entropy" not in lookup:
        raise ValueError("entropy pair missing from registered inventory")
    h15 = X[:, lookup["epr"]]
    hsaved = X[:, lookup["mean_logprob_entropy"]]
    common = 0.5 * (h15 + hsaved)
    difference = hsaved - h15

    columns: list[np.ndarray] = []
    output_names: list[str] = []
    for name in names:
        if name == "epr":
            output_names.extend(("entropy_common", "entropy_support_delta"))
            columns.extend((common, difference))
        elif name in {"mean_logprob_entropy", "trace_length"}:
            continue
        elif name == "hl_ratio" and {"low_band_power", "high_band_power"} <= set(names):
            continue
        else:
            output_names.append(str(name))
            columns.append(X[:, lookup[name]])
    transformed = np.column_stack(columns).astype(np.float64, copy=False)
    restored_h15 = common - 0.5 * difference
    restored_hsaved = common + 0.5 * difference
    common_sd = max(float(np.std(common)), 1e-15)
    metrics = {
        "entropy_roundtrip_max_abs": float(
            max(np.max(np.abs(restored_h15 - h15)), np.max(np.abs(restored_hsaved - hsaved)))
        ),
        "entropy_delta_to_common_sd": float(np.std(difference) / common_sd),
        "entropy_pearson": float(np.corrcoef(h15, hsaved)[0, 1]),
    }
    if {"hl_ratio", "low_band_power", "high_band_power"} <= set(names):
        expected = X[:, lookup["high_band_power"]] / (X[:, lookup["low_band_power"]] + 1e-12)
        metrics["hl_ratio_roundtrip_max_abs"] = float(
            np.max(np.abs(expected - X[:, lookup["hl_ratio"]]))
        )
    else:
        metrics["hl_ratio_roundtrip_max_abs"] = float("nan")
    return transformed, tuple(output_names), metrics


def select_panel(
    cell_id: str,
    row_ids: Sequence[str],
    group_ids: Sequence[str],
    panel: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        grouped.setdefault(str(group), []).append(index)
    if len(grouped) < N_GROUPS:
        raise ValueError(f"{cell_id}: only {len(grouped)} groups")
    groups = sorted(
        grouped,
        key=lambda group: stable_hex(SCHEMA, "group", panel, cell_id, group),
    )[:N_GROUPS]
    indices = [
        min(
            grouped[group],
            key=lambda index: stable_hex(
                SCHEMA, "row", panel, cell_id, group, row_ids[index]
            ),
        )
        for group in groups
    ]
    return np.asarray(indices, dtype=np.int64), tuple(groups)


def fold_ids(cell_id: str, panel: int, groups: Sequence[str]) -> np.ndarray:
    order = sorted(
        range(len(groups)),
        key=lambda index: stable_hex(SCHEMA, "fold", panel, cell_id, groups[index]),
    )
    folds = np.empty(len(groups), dtype=np.int64)
    for rank, index in enumerate(order):
        folds[index] = rank % N_FOLDS
    return folds


def donor_transform(
    values: np.ndarray, lengths: np.ndarray, train: np.ndarray, held: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    log_length = np.log1p(lengths.astype(np.float64))
    mean_l = float(log_length[train].mean())
    scale_l = max(float(log_length[train].std()), 1e-12)
    z_train = (log_length[train] - mean_l) / scale_l
    z_held = (log_length[held] - mean_l) / scale_l
    design_train = np.column_stack(
        (np.ones(len(train)), z_train, z_train**2, z_train**3)
    )
    design_held = np.column_stack(
        (np.ones(len(held)), z_held, z_held**2, z_held**3)
    )
    coefficient = np.linalg.lstsq(design_train, values[train], rcond=None)[0]
    residual_train = values[train] - design_train @ coefficient
    residual_held = values[held] - design_held @ coefficient
    center = residual_train.mean(axis=0)
    scale = residual_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (residual_train - center) / scale, (residual_held - center) / scale


def panel_r2(
    values: np.ndarray,
    lengths: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    errors = np.zeros(values.shape[1], dtype=np.float64)
    baselines = np.zeros(values.shape[1], dtype=np.float64)
    for fold in range(N_FOLDS):
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        donor, test = donor_transform(values, lengths, train, held)
        for target in range(values.shape[1]):
            peers = [index for index in range(values.shape[1]) if index != target]
            gram = donor[:, peers].T @ donor[:, peers]
            coefficient = np.linalg.solve(
                gram + RIDGE_ALPHA * np.eye(len(peers)),
                donor[:, peers].T @ donor[:, target],
            )
            prediction = test[:, peers] @ coefficient
            errors[target] += float(np.sum((test[:, target] - prediction) ** 2))
            baselines[target] += float(np.sum(test[:, target] ** 2))
    return 1.0 - errors / np.maximum(baselines, 1e-12)


def equal_family_mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    families = sorted({str(row["dataset_family"]) for row in rows})
    family_values = [
        np.mean([float(row[key]) for row in rows if row["dataset_family"] == family])
        for family in families
    ]
    return float(np.mean(family_values))


def plot_summary(path: Path, summary: Mapping[str, object]) -> None:
    cv = summary["predictive_cv_primary_k50"]
    width, height = 1120, 560
    left, bar_start, bar_width = 35, 255, 450
    top, row_height = 90, 53
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#20242a}.title{font-size:22px;font-weight:700}.sub{font-size:14px}.label{font-size:13px}.value{font-size:12px;font-weight:700}</style>',
        '<text x="35" y="36" class="title">Feature Contract V2 — target-free input cleanup</text>',
        '<text x="35" y="62" class="sub">Held-out reconstruction from the other six predictive features, after donor-only length control (23 K=50 cells)</text>',
    ]
    for index, record in enumerate(cv):
        y = top + index * row_height
        value = float(record["equal_dataset_family_r2"])
        color = "#b56b2d" if bool(record["compact_candidate_redundant"]) else "#3973ac"
        elements.append(f'<text x="{left}" y="{y + 18}" class="label">{record["short_name"]}</text>')
        elements.append(f'<rect x="{bar_start}" y="{y}" width="{bar_width}" height="22" fill="#edf0f3"/>')
        elements.append(f'<rect x="{bar_start}" y="{y}" width="{bar_width * max(value, 0):.2f}" height="22" fill="{color}"/>')
        elements.append(f'<text x="{bar_start + bar_width + 10}" y="{y + 17}" class="value">R² {value:.3f}</text>')
    gate_x = bar_start + 0.95 * bar_width
    elements.append(f'<line x1="{gate_x}" y1="{top - 10}" x2="{gate_x}" y2="{top + 7 * row_height - 20}" stroke="#111" stroke-dasharray="5 4"/>')
    elements.append(f'<text x="{gate_x - 24}" y="{top - 16}" class="label">0.95 advisory gate</text>')

    decisions = summary["contract_actions"]
    x0 = 865
    elements.append('<text x="825" y="92" class="sub">Conservative decisions</text>')
    labels = (("Keep", "keep", "#3973ac"), ("Merge", "merge", "#b56b2d"), ("Context", "context", "#777"), ("Drop exact", "drop", "#a64b4b"))
    for index, (label, key, color) in enumerate(labels):
        y = 125 + index * 67
        count = int(decisions[key])
        elements.append(f'<text x="825" y="{y + 17}" class="label">{label}</text>')
        elements.append(f'<rect x="905" y="{y}" width="{count * 5}" height="22" fill="{color}"/>')
        elements.append(f'<text x="{915 + count * 5}" y="{y + 17}" class="value">{count}</text>')
    elements.append('<text x="825" y="430" class="label">Orange = advisory compact candidate</text>')
    elements.append('<text x="825" y="453" class="label">Blue = retain distinct measurement</text>')
    elements.append('<text x="35" y="540" class="label">Source: 48,607 target-free rows across 24 cells; no correctness labels used.</text>')
    elements.append('</svg>')
    path.with_suffix(".svg").write_text("\n".join(elements) + "\n", encoding="utf-8")

    # Pillow is used instead of Matplotlib because this audit must also run in
    # minimal/headless cluster environments.
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    try:
        title_font = ImageFont.truetype(font_path, 22)
        body_font = ImageFont.truetype(font_path, 14)
        label_font = ImageFont.truetype(font_path, 13)
        value_font = ImageFont.truetype(font_path, 12)
    except OSError:
        title_font = body_font = label_font = value_font = ImageFont.load_default()
    draw.text((35, 14), "Feature Contract V2 — target-free input cleanup", fill="#20242a", font=title_font)
    draw.text(
        (35, 48),
        "Held-out reconstruction from the other six predictive features, after donor-only length control (23 K=50 cells)",
        fill="#20242a",
        font=body_font,
    )
    for index, record in enumerate(cv):
        y = top + index * row_height
        value = float(record["equal_dataset_family_r2"])
        color = "#b56b2d" if bool(record["compact_candidate_redundant"]) else "#3973ac"
        draw.text((left, y + 3), str(record["short_name"]), fill="#20242a", font=label_font)
        draw.rectangle((bar_start, y, bar_start + bar_width, y + 22), fill="#edf0f3")
        draw.rectangle((bar_start, y, bar_start + int(bar_width * max(value, 0)), y + 22), fill=color)
        draw.text((bar_start + bar_width + 10, y + 3), f"R² {value:.3f}", fill="#20242a", font=value_font)
    draw.line((gate_x, top - 10, gate_x, top + 7 * row_height - 20), fill="#111111", width=1)
    draw.text((gate_x - 58, top - 30), "0.95 advisory gate", fill="#20242a", font=value_font)
    draw.text((825, 84), "Conservative decisions", fill="#20242a", font=body_font)
    for index, (label, key, color) in enumerate(labels):
        y = 125 + index * 67
        count = int(decisions[key])
        draw.text((825, y + 3), label, fill="#20242a", font=label_font)
        draw.rectangle((905, y, 905 + count * 5, y + 22), fill=color)
        draw.text((915 + count * 5, y + 3), str(count), fill="#20242a", font=value_font)
    draw.text((825, 430), "Orange = advisory compact candidate", fill="#20242a", font=label_font)
    draw.text((825, 453), "Blue = retain distinct measurement", fill="#20242a", font=label_font)
    draw.text((35, 535), "Source: 48,607 target-free rows across 24 cells; no correctness labels used.", fill="#20242a", font=value_font)
    image.save(path, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("configs/residual_graph_deem_24cell_v1_registry.json"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("local_cache/deem_b3_moe_v1/bundles"))
    parser.add_argument("--out-dir", type=Path, default=Path("local_cache/deem_b3_moe_v1/feature_contract_v2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path.cwd().resolve()
    registry = load_registry(args.registry)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contract_dir = args.out_dir / "bundles"
    contract_dir.mkdir(parents=True, exist_ok=True)
    dag = {row["feature_name"]: row for row in derive_feature_dag()}

    lineage_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []

    for name, record in dag.items():
        lineage_rows.append({
            "feature_name": name,
            "source_stream": record["source_stream"],
            "operator": record["operator"],
            "implementation_file": record["implementation"]["source_file"],
            "implementation_line": record["implementation"]["source_line"],
        })
        if name in {"epr", "mean_logprob_entropy"}:
            action, reason = "merge", "same post-warper distribution; exact common/difference transform"
        elif name == "trace_length":
            action, reason = "context", "measurement exposure and mechanical transform confound"
        elif name == "hl_ratio":
            action, reason = "drop", "exactly reconstructed from high_band_power and low_band_power"
        else:
            action, reason = "keep", "no exact target-free redundancy proof"
        decision_rows.append({
            "feature_name": name,
            "action": action,
            "reason": reason,
            "safe_contract_v2": action != "drop",
        })

    for cell in registry["cells"]:
        cell_id = str(cell["cell_id"])
        bundle_path = args.bundle_dir / f"{cell_id}.npz"
        bundle = load_target_free_bundle(bundle_path)
        if bundle.cell_id != cell_id or bundle.manifest_sha256 != cell["source"]["manifest_sha256"]:
            raise ValueError(f"bundle/registry mismatch: {cell_id}")
        top_k, manifest_path = manifest_top_k(repo, cell)
        transformed, names_v2, metrics = transform_contract(bundle.X_raw, bundle.feature_names)
        arrays = {
            "schema": np.asarray(SCHEMA),
            "cell_id": np.asarray(cell_id),
            "X_contract_raw": transformed,
            "feature_names": np.asarray(names_v2, dtype=str),
            "row_id": np.asarray(bundle.row_ids, dtype=str),
            "group_id": np.asarray(bundle.group_ids, dtype=str),
            "raw_trace_length": np.asarray(bundle.raw_trace_length, dtype=np.int64),
            "dataset_family": np.asarray(bundle.dataset_family),
            "task_type": np.asarray(bundle.task_type),
            "manifest_declared_top_k": np.asarray(top_k, dtype=np.int64),
            "source_bundle_sha256": np.asarray(bundle.bundle_sha256),
            "source_manifest_sha256": np.asarray(bundle.manifest_sha256),
            "ordered_row_id_sha256": np.asarray(canonical_sha256(list(bundle.row_ids))),
            "entropy_delta_scaling_policy": np.asarray("scale_by_entropy_common_not_own_sd"),
        }
        assert_no_target_fields(arrays)
        output_path = contract_dir / f"{cell_id}.npz"
        digest = atomic_save_npz(output_path, **arrays)
        artifact_rows.append({
            "cell_id": cell_id,
            "path": output_path.relative_to(args.out_dir).as_posix(),
            "sha256": digest,
            "n_rows": len(bundle.row_ids),
            "n_features": len(names_v2),
        })

        lookup = {name: index for index, name in enumerate(bundle.feature_names)}
        predictive = bundle.X_raw[:, [lookup[name] for name in PREDICTIVE_FEATURES]]
        panel_values = []
        for panel in range(N_PANELS):
            indices, groups = select_panel(cell_id, bundle.row_ids, bundle.group_ids, panel)
            folds = fold_ids(cell_id, panel, groups)
            panel_values.append(
                panel_r2(predictive[indices], bundle.raw_trace_length[indices], folds)
            )
        r2 = np.mean(np.stack(panel_values), axis=0)
        for index, name in enumerate(PREDICTIVE_FEATURES):
            cv_rows.append({
                "cell_id": cell_id,
                "dataset_family": bundle.dataset_family,
                "manifest_declared_top_k": top_k,
                "feature_name": name,
                "held_r2_from_other_six": float(r2[index]),
            })

        cell_rows.append({
            "cell_id": cell_id,
            "dataset_family": bundle.dataset_family,
            "n_rows": len(bundle.row_ids),
            "manifest_declared_top_k": top_k,
            "original_features": len(bundle.feature_names),
            "contract_v2_features": len(names_v2),
            **metrics,
            "source_bundle_sha256": bundle.bundle_sha256,
            "contract_bundle_sha256": digest,
            "manifest_file_sha256": sha256_file(manifest_path),
        })

    primary_cv = [row for row in cv_rows if row["manifest_declared_top_k"] == 50]
    cv_summary = []
    short = {
        "epr": "H15 entropy",
        "mean_logprob_entropy": "Hsaved entropy",
        "mean_top1_logprob": "Top-1 confidence",
        "logprob_margin": "Top1−Top2 margin",
        "varentropy": "Varentropy",
        "renyi_entropy_2": "Rényi-2 entropy",
        "topk_tail_mass": "Top-k tail mass",
    }
    for name in PREDICTIVE_FEATURES:
        rows = [row for row in primary_cv if row["feature_name"] == name]
        family_names = sorted({str(row["dataset_family"]) for row in rows})
        family_values = {
            family: float(np.mean([
                row["held_r2_from_other_six"] for row in rows if row["dataset_family"] == family
            ]))
            for family in family_names
        }
        mean = float(np.mean(list(family_values.values())))
        compact = bool(mean >= 0.95 and min(family_values.values()) >= 0.90)
        cv_summary.append({
            "feature_name": name,
            "short_name": short[name],
            "equal_dataset_family_r2": mean,
            "min_family_r2": float(min(family_values.values())),
            "max_family_r2": float(max(family_values.values())),
            "compact_candidate_redundant": compact,
            "family_values": family_values,
        })

    action_counts = {
        action: sum(row["action"] == action for row in decision_rows)
        for action in ("keep", "merge", "context", "drop")
    }
    k50_cells = [row for row in cell_rows if row["manifest_declared_top_k"] == 50]
    los_cells = [row for row in cell_rows if row["manifest_declared_top_k"] != 50]
    summary = {
        "schema": SCHEMA,
        "scope": "target-free feature cleanup only; no detector or labels",
        "n_cells": len(cell_rows),
        "n_rows": int(sum(row["n_rows"] for row in cell_rows)),
        "contract_actions": action_counts,
        "safe_changes": [
            "replace epr/Hsaved entropy by exact common/support-delta coordinates",
            "move trace length to context sidecar",
            "drop exact-derived hl_ratio when low/high parents exist",
        ],
        "entropy_roundtrip_max_abs": float(max(row["entropy_roundtrip_max_abs"] for row in cell_rows)),
        "hl_ratio_roundtrip_max_abs": float(np.nanmax([row["hl_ratio_roundtrip_max_abs"] for row in cell_rows])),
        "entropy_pair_pearson_mean_k50": float(np.mean([row["entropy_pearson"] for row in k50_cells])),
        "entropy_delta_to_common_sd_mean_k50": float(np.mean([row["entropy_delta_to_common_sd"] for row in k50_cells])),
        "feature_count_original_range": [int(min(row["original_features"] for row in cell_rows)), int(max(row["original_features"] for row in cell_rows))],
        "feature_count_v2_range": [int(min(row["contract_v2_features"] for row in cell_rows)), int(max(row["contract_v2_features"] for row in cell_rows))],
        "manifest_top_k_primary": {"top_k": 50, "n_cells": len(k50_cells)},
        "manifest_top_k_sensitivity": [{"cell_id": row["cell_id"], "top_k": row["manifest_declared_top_k"]} for row in los_cells],
        "predictive_cv_primary_k50": cv_summary,
        "compact_candidate_rule": "R2>=0.95 equal-family and >=0.90 in every dataset family; advisory only",
        "labels_accessed": False,
        "target_module_imported": False,
    }

    write_csv(args.out_dir / "FEATURE_LINEAGE.csv", lineage_rows)
    write_csv(args.out_dir / "FEATURE_DECISIONS.csv", decision_rows)
    write_csv(args.out_dir / "CELL_METRICS.csv", cell_rows)
    write_csv(args.out_dir / "PREDICTIVE_BLOCK_CV.csv", cv_rows)
    write_csv(args.out_dir / "ARTIFACTS.csv", artifact_rows)
    atomic_write_json(args.out_dir / "SUMMARY.json", summary)
    plot_summary(args.out_dir / "feature_contract_v2.png", summary)

    report_lines = [
        "# Feature Contract V2 — target-free cleanup",
        "",
        "This stage does not fit or evaluate a hallucination detector.",
        "",
        "## Safe contract changes",
        "",
        "- `epr` and `mean_logprob_entropy` become the exact invertible pair `entropy_common` and `entropy_support_delta`.",
        "- `trace_length` leaves the expert inventory and remains available as `raw_trace_length` context.",
        "- `hl_ratio` is removed only when its low/high parents are present; it is exactly reconstructable.",
        "- Every other feature is retained in the safe contract. CV redundancy is advisory for a later compact contract.",
        "",
        "## Headline diagnostics",
        "",
        f"- Entropy round-trip max error: {summary['entropy_roundtrip_max_abs']:.3e}.",
        f"- FFT ratio round-trip max error: {summary['hl_ratio_roundtrip_max_abs']:.3e}.",
        f"- Mean K50 entropy-pair Pearson: {summary['entropy_pair_pearson_mean_k50']:.6f}.",
        f"- Mean natural support-delta/common SD ratio: {summary['entropy_delta_to_common_sd_mean_k50']:.6f}.",
        f"- Feature-count range: {summary['feature_count_original_range']} -> {summary['feature_count_v2_range']}.",
        "",
        "## Advisory compact candidates",
        "",
    ]
    for record in cv_summary:
        report_lines.append(
            f"- {record['feature_name']}: equal-family held R2={record['equal_dataset_family_r2']:.4f}, "
            f"min-family={record['min_family_r2']:.4f}; compact-candidate={record['compact_candidate_redundant']}."
        )
    report_lines.extend([
        "",
        "The LOS-Net K=1000 cell is excluded from the K=50 primary predictive-shape summary and retained as sensitivity only.",
    ])
    (args.out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    inventory = {
        path.relative_to(args.out_dir).as_posix(): sha256_file(path)
        for path in sorted(args.out_dir.rglob("*"))
        if path.is_file() and path.name != "FREEZE.json"
    }
    freeze = {
        "schema": SCHEMA + "_freeze",
        "source_sha256": sha256_file(Path(__file__)),
        "registry_sha256": sha256_file(args.registry),
        "inventory": inventory,
        "inventory_content_sha256": canonical_sha256(inventory),
        "labels_accessed": False,
        "target_module_imported": False,
    }
    atomic_write_json(args.out_dir / "FREEZE.json", freeze)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
