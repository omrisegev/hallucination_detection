#!/usr/bin/env python3
"""Create compact A/B and variant plots for the Phase-2 score release.

The default plots are target-free: score vectors and frozen innovation-map
diagnostics only.  If a post-audit evaluation directory is supplied, the
script additionally renders ProcessBench F1/CI and PRMBench guard plots.  The
plot manifest records every input hash so figures cannot be mistaken for a
different freeze.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

_MPL_CACHE = Path("/private/tmp/matplotlib-token-temporal-b3")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
CELLS = (
    "processbench_gsm8k_qwen3_4b", "processbench_math_qwen3_4b",
    "processbench_olympiadbench_qwen3_4b", "processbench_omnimath_qwen3_4b",
    "processbench_gsm8k_qwen3_8b", "processbench_math_qwen3_8b",
    "processbench_olympiadbench_qwen3_8b", "processbench_omnimath_qwen3_8b",
)
METHODS = (
    "LOCAL_TOKEN_B3", "LOCAL_TOKEN_B3_SELF_INNOV",
    "LOCAL_TOKEN_B3_ROOK_ALL_INNOV", "LOCAL_TOKEN_B3_ROOK_PSTG_INNOV",
    "LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL",
)


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_freeze(root: Path) -> dict[str, Any]:
    import hashlib
    from spectral_utils.reconstruction_benchmark.io import canonical_json_bytes
    freeze_path = root / "SCORE_FREEZE_MANIFEST.json"
    value = json.loads(freeze_path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop("payload_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if digest != expected:
        raise RuntimeError(f"score freeze payload hash failed: {freeze_path}")
    if tuple(value.get("expected_cells", ())) != CELLS or tuple(value.get("method_ids", ())) != METHODS:
        raise RuntimeError("Phase-2 plot roster mismatch")
    for binding in value["records"]:
        record_path = root / str(binding["record_path"])
        if _sha256(record_path) != binding["record_sha256"]:
            raise RuntimeError(f"frozen record changed: {binding['cell_id']}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record_body = dict(record)
        record_digest = record_body.pop("payload_sha256", None)
        if record_digest != hashlib.sha256(canonical_json_bytes(record_body)).hexdigest():
            raise RuntimeError(f"frozen record payload hash failed: {binding['cell_id']}")
        score_path = record_path.parent / str(record["score_path"])
        if _sha256(score_path) != binding["score_sha256"]:
            raise RuntimeError(f"frozen score changed: {binding['cell_id']}")
    return value


def _records(root: Path, freeze: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]]:
    from spectral_utils.reconstruction_benchmark.io import load_npz_no_pickle
    output = {}
    for binding in freeze["records"]:
        record_path = root / str(binding["record_path"])
        record = json.loads(record_path.read_text(encoding="utf-8"))
        output[str(binding["cell_id"])] = (record, load_npz_no_pickle(record_path.parent / str(record["score_path"])))
    return output


def _plot_variant_correlations(records: Mapping[str, tuple[dict[str, Any], dict[str, np.ndarray]]], out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, cell_id in zip(axes.flat, CELLS):
        scores = np.asarray(records[cell_id][1]["token_step_scores"], dtype=float)
        corr = np.corrcoef(scores)
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(cell_id.replace("processbench_", ""), fontsize=9)
        ax.set_xticks(range(len(METHODS)), ["B3", "self", "rook", "PSTG", "nonrook"], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(METHODS)), ["B3", "self", "rook", "PSTG", "nonrook"], fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation of step scores")
    fig.suptitle("Phase 2 variant agreement (target-free)")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_variant_deltas(records: Mapping[str, tuple[dict[str, Any], dict[str, np.ndarray]]], out: Path) -> None:
    names = ["self", "rook", "PSTG", "nonrook"]
    values = []
    for cell_id in CELLS:
        score = np.asarray(records[cell_id][1]["token_step_scores"], dtype=float)
        baseline = score[0]
        values.append([float(np.mean(score[i] - baseline)) for i in range(1, 5)])
    matrix = np.asarray(values)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(CELLS))
    width = 0.19
    for index, name in enumerate(names):
        ax.bar(x + (index - 1.5) * width, matrix[:, index], width, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [cell.replace("processbench_", "") for cell in CELLS], rotation=45, ha="right")
    ax.set_ylabel("mean step-score delta vs B3")
    ax.set_title("A/B variant score deltas by cell (target-free)")
    ax.legend(ncol=4)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_ab_comparison(
    records_a: Mapping[str, tuple[dict[str, Any], dict[str, np.ndarray]]],
    records_b: Mapping[str, tuple[dict[str, Any], dict[str, np.ndarray]]],
    out: Path,
) -> None:
    labels = ["B3", "self", "rook", "PSTG", "nonrook"]
    values = []
    for cell_id in CELLS:
        left = np.asarray(records_a[cell_id][1]["token_step_scores"], dtype=float)
        right = np.asarray(records_b[cell_id][1]["token_step_scores"], dtype=float)
        if left.shape != right.shape:
            raise RuntimeError(f"A/B score shape differs for {cell_id}")
        values.append([float(np.mean(np.abs(right[i] - left[i]))) for i in range(len(METHODS))])
    matrix = np.asarray(values)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(CELLS))
    width = 0.15
    for i, label in enumerate(labels):
        ax.bar(x + (i - 2) * width, matrix[:, i], width, label=label)
    ax.set_xticks(x, [cell.replace("processbench_", "") for cell in CELLS], rotation=45, ha="right")
    ax.set_ylabel("mean absolute step-score difference")
    ax.set_title("Independent A/B score agreement by variant")
    ax.legend(ncol=5)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_support(records: Mapping[str, tuple[dict[str, Any], dict[str, np.ndarray]]], out: Path) -> None:
    aggregate = np.zeros((9, 9), dtype=float)
    count = 0
    for cell_id in CELLS:
        record = records[cell_id][0]
        pstg = record["fit_diagnostics"]["methods"]["LOCAL_TOKEN_B3_ROOK_PSTG_INNOV"]
        for item in pstg.get("innovation_maps", ()):
            aggregate += np.asarray(item["support"], dtype=float)
            count += 1
    if count:
        aggregate /= count
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(aggregate, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(9), [f"s{i}" for i in range(9)])
    ax.set_yticks(range(9), [f"t{i}" for i in range(9)])
    ax.set_xlabel("lagged source channel")
    ax.set_ylabel("target channel")
    ax.set_title("PSTG rook-support frequency across cells/folds")
    fig.colorbar(image, ax=ax, label="selected in fold")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_evaluation(evaluation_root: Path, out: Path) -> None:
    rows = _read_csv(evaluation_root / "PROCESSBENCH_MACRO.csv")
    labels = [row["method_id"].replace("LOCAL_TOKEN_", "").replace("_INNOV", "") for row in rows]
    values = np.asarray([float(row["official_macro_f1"]) for row in rows])
    low = np.zeros(len(rows))
    high = np.zeros(len(rows))
    reference = next(
        float(row["official_macro_f1"])
        for row in rows if row["method_id"] == "LOCAL_IU29"
    )
    promotion_path = evaluation_root / "PROMOTION_DECISION.csv"
    if promotion_path.exists():
        promotion = {row["method_id"]: row for row in _read_csv(promotion_path)}
        for i, row in enumerate(rows):
            item = promotion.get(row["method_id"])
            if item:
                absolute_low = reference + float(item["f1_delta_ci_low"])
                absolute_high = reference + float(item["f1_delta_ci_high"])
                low[i] = max(0.0, values[i] - absolute_low)
                high[i] = max(0.0, absolute_high - values[i])
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(np.arange(len(rows)), values, yerr=np.vstack((low, high)) if np.any(low + high) else None, capsize=3)
    ax.set_xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
    ax.set_ylabel("ProcessBench macro F1")
    ax.set_title("Phase 2 post-audit evaluation: macro F1 and paired CI")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build(freeze_a: Path, out_root: Path, *, freeze_b: Path | None = None, evaluation_root: Path | None = None) -> dict[str, Any]:
    if out_root.exists():
        raise FileExistsError(f"plot output already exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)
    freeze = _verified_freeze(freeze_a)
    records = _records(freeze_a, freeze)
    outputs = []
    for name, fn in (
        ("variant_correlations.png", _plot_variant_correlations),
        ("variant_deltas.png", _plot_variant_deltas),
        ("pstg_support_frequency.png", _plot_support),
    ):
        path = out_root / name
        fn(records, path)
        outputs.append(path)
    if freeze_b is not None:
        freeze_b_value = _verified_freeze(freeze_b)
        records_b = _records(freeze_b, freeze_b_value)
        # Reuse the variant delta plot for the second run with an explicit name;
        # keeping the A/B trees separate avoids silently averaging incompatible
        # score releases.
        path = out_root / "variant_deltas_B.png"
        _plot_variant_deltas(records_b, path)
        outputs.append(path)
        path = out_root / "ab_score_difference.png"
        _plot_ab_comparison(records, records_b, path)
        outputs.append(path)
    if evaluation_root is not None:
        path = out_root / "processbench_macro_f1.png"
        _plot_evaluation(evaluation_root, path)
        outputs.append(path)
    manifest = {
        "schema_version": "token-local-temporal-innovation-b3-plot-manifest-v1",
        "score_freeze_a_sha256": _sha256(freeze_a / "SCORE_FREEZE_MANIFEST.json"),
        "score_freeze_b_sha256": _sha256(freeze_b / "SCORE_FREEZE_MANIFEST.json") if freeze_b else None,
        "evaluation_manifest_sha256": _sha256(evaluation_root / "EVALUATION_MANIFEST.json") if evaluation_root and (evaluation_root / "EVALUATION_MANIFEST.json").exists() else None,
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    from spectral_utils.reconstruction_benchmark.io import atomic_write_json, canonical_json_bytes, sha256_bytes
    manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    atomic_write_json(out_root / "PLOT_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-a", required=True)
    parser.add_argument("--freeze-b")
    parser.add_argument("--evaluation-dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = build(Path(args.freeze_a).resolve(), Path(args.out_dir).resolve(), freeze_b=Path(args.freeze_b).resolve() if args.freeze_b else None, evaluation_root=Path(args.evaluation_dir).resolve() if args.evaluation_dir else None)
    print(json.dumps({"status": "PASS", **result}, indent=2))


if __name__ == "__main__":
    main()
