"""Post-evaluation descriptive mechanism summary for uniform multiscale v1."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_direct_probability_temporal as evaluator
from scripts import run_renyi_position_temporal_fusion as base
from scripts import run_uniform_multiscale_fusion_v1 as run
from spectral_utils import uniform_multiscale_fusion as model


def main():
    manifest = json.loads((run.OUT / "MANIFEST.json").read_text(encoding="utf8"))
    base.configure_sources(Path(manifest["source_root"]), Path(manifest["contract_root"]))
    records = json.loads((evaluator.old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8"))["records"]
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        joined = {key: saved[key] for key in saved.files}
    with np.load(run.OUT / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        scores = {name: np.asarray(saved["steps__" + name]) for name in run.METHODS}
    calibration = json.loads((run.OUT / "CALIBRATION.json").read_text(encoding="utf8"))["thresholds"]
    metrics, per = evaluator.evaluate_arrays(records, joined, scores, calibration_thresholds=calibration, fold_auc=True)
    pairs = [
        ("q15_raw_equal", "reference_equal4_token_raw"),
        ("q15_raw_equal", "q15_scale_equal"),
        ("q50_raw_equal", "q50_scale_equal"),
        ("ms8_raw_equal", "ms8_scale_equal"),
        ("q15_raw_equal", "ms8_scale_equal"),
        ("q15_raw_equal", "q15_scale_simplex"),
    ]
    contrasts = evaluator.paired_bootstrap(
        records, joined, per, draws=10_000, pairs=pairs, primary_pairs=set(), primary_ci=0.95
    )
    for first, second in pairs:
        contrasts[first + "_minus_" + second]["pb_delta"] = metrics[first]["pb_all8"] - metrics[second]["pb_all8"]

    con = sqlite3.connect(run.OUT / "CHECKPOINT.sqlite")
    scales = []
    weights = {support: [] for support in model.SUPPORTS}
    try:
        for _, blob, text in con.execute("select key,payload,info from models"):
            info = json.loads(text)
            if len(info["excluded_folds"]) != 1:
                continue
            with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                scales.append(np.asarray(saved["scale"]))
            for support in model.SUPPORTS:
                weights[support].append(np.asarray(info["fits"][support]["weights"]))
    finally:
        con.close()
    mean_scale = np.mean(scales, axis=0)
    scale_shares = {}
    for support, columns in model.SUPPORTS.items():
        share = mean_scale[columns] / mean_scale[columns].sum()
        scale_shares[support] = {model.FEATURE_NAMES[int(column)]: float(value) for column, value in zip(columns, share)}
    weight_summary = {}
    for support, rows in weights.items():
        value = np.stack(rows)
        columns = model.SUPPORTS[support]
        weight_summary[support] = {
            model.FEATURE_NAMES[int(column)]: {
                "mean": float(value[:, j].mean()),
                "min": float(value[:, j].min()),
                "max": float(value[:, j].max()),
            }
            for j, column in enumerate(columns)
        }
    full_centered = {}
    metric_payload = json.loads((run.OUT / "METRICS.json").read_text(encoding="utf8"))["metrics"]
    for support in model.SUPPORTS:
        full = metric_payload[support + "_scale_simplex"]
        centered = metric_payload[support + "_scale_simplex_centered"]
        full_centered[support] = {
            "pb_delta": full["pb_all8"] - centered["pb_all8"],
            "prm_within_delta": full["prm_within"] - centered["prm_within"],
            "prm_fold_auc_delta": full["prm_fold_auc"] - centered["prm_fold_auc"],
            "prm_pooled_oof_delta": full["prm_pooled_oof_descriptive"] - centered["prm_pooled_oof_descriptive"],
            "prmscore_delta": full["prmscore_q08"] - centered["prmscore_q08"],
        }
    base.atomic_json(run.OUT / "POSTHOC_CONTRASTS.json", contrasts)
    base.atomic_json(
        run.OUT / "MECHANISM_SUMMARY.json",
        {
            "schema": "uniform-multiscale-mechanism-summary-v1",
            "scope": "post-evaluation descriptive; no candidate reselection",
            "raw_equal_implicit_scale_shares": scale_shares,
            "outer_fold_simplex_weights": weight_summary,
            "full_minus_centered": full_centered,
        },
    )
    print(base.dumps({"status": "PASS", "posthoc_pairs": len(pairs), "supports": list(model.SUPPORTS)}))


if __name__ == "__main__":
    main()
