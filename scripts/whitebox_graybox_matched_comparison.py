#!/usr/bin/env python3
"""Freeze and report the exact-row white-box versus gray-box comparison.

This is a retrospective comparison, not a new registered benchmark.  The fit
phase reconstructs the gray-box complete-case row IDs without reading labels,
fits the frozen mixed-v2 solvers, intersects those IDs with the frozen white-box
score rows, and hashes label-free score bundles.  Only then does evaluation
reopen the raw caches and define hallucination as ``1 - correctness``.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import pickle
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_repgrid_featcache import H16, candidate_feats  # noqa: E402
from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    fit_one_subset,
    load_contract,
)
from spectral_utils.paper_benchmark_suite import DEPLOYED_FIT  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "whitebox-vs-graybox-matched-v1-2026-08-19"
OUT = REPO / "results" / "whitebox_vs_graybox_matched_v1"
WHITE = REPO / "results" / "whitebox_depth_distributed_pure_v1"
GRAY = REPO / "results" / "hard_filter_dufs_liu_24cell"
GRAY_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
RAW_ROOT = REPO / "dataset_cache" / "whitebox_layer_fusion_v1" / "repgrid"
SEED = 20260812
TIE_TOLERANCE = 0.001

CELLS = OrderedDict((
    ("gsm8k_t1.0", ("lapeigvals_gsm8k_llama8b", "raw_gsm8k_T1.0.pkl")),
    ("triviaqa_t1.0", ("spilled_triviaqa_llama8b", "raw_trivia_qa_T1.0.pkl")),
    ("sciq_t1.0", ("sciq_llama8b", "raw_sciq_T1.0.pkl")),
    ("truthfulqa_t0.5", ("truthfulqa_llama8b", "raw_truthfulqa_T0.5.pkl")),
    ("squadv2_t0.5", ("se_squad_v2_llama8b", "raw_squad_v2_T0.5.pkl")),
    ("nq_open_t0.5", ("se_nq_open_llama8b", "raw_nq_open_T0.5.pkl")),
    ("gsm8k_r1distill_t0.0", ("ars_gsm8k_r1distill8b", "raw_gsm8k_T0.0.pkl")),
    ("gsm8k_mistral24b_t1.0", ("lapeigvals_gsm8k_mistral24b", "raw_gsm8k_T1.0.pkl")),
    ("gsm8k_nemo_t1.0", ("lapeigvals_gsm8k_nemo", "raw_gsm8k_T1.0.pkl")),
    ("gsm8k_phi35_t1.0", ("lapeigvals_gsm8k_phi35", "raw_gsm8k_T1.0.pkl")),
    ("gsm8k_mistral7b_t1.0", ("noise_gsm8k_mistral7b", "raw_gsm8k_T1.0.pkl")),
    ("gsm8k_phi3mini_t1.0", ("noise_gsm8k_phi3mini", "raw_gsm8k_T1.0.pkl")),
    ("triviaqa_qwen3_t0.6", ("semenergy_triviaqa_qwen3_8b", "raw_trivia_qa_T0.6.pkl")),
))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(values.std())
    if scale < 1e-12:
        raise ValueError("cannot standardize a constant score")
    return (values - values.mean()) / scale


def metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def cell_bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return tuple(float(x) for x in np.quantile(sampled.mean(axis=1), (0.025, 0.975)))


def raw_path(gray_cell: str, filename: str) -> Path:
    return RAW_ROOT / gray_cell / filename


def gray_complete_case_ids(path: Path) -> list[str]:
    """Reproduce build_repgrid_featcache's row mask without opening labels."""
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    kept = []
    for problem_idx in sorted(raw):
        for candidate_idx, candidate in enumerate(raw[problem_idx]["candidates"]):
            values = candidate_feats(candidate, allow_short=False)
            if all(np.isfinite(values.get(name, np.nan)) for name in H16):
                kept.append(f"{problem_idx}:{candidate_idx}")
    return kept


def historical_gray_auc() -> dict[str, float]:
    output = {}
    with (GRAY / "per_cell_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["contract"] == "mixed_v2" and row["filter"] == "full" and row["solver"] == "dufs_liu":
                output[row["cell"]] = float(row["auroc"])
    return output


def fit_and_freeze() -> dict:
    """Fit scores and freeze hashes. This function never reads correctness labels."""
    try:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    score_dir = OUT / "scores"
    score_dir.mkdir(exist_ok=True)
    bundle = np.load(GRAY_BUNDLE, allow_pickle=True)
    score_rows = []
    for index, (white_cell, (gray_cell, raw_name)) in enumerate(CELLS.items(), 1):
        print(f"[fit {index:02d}/{len(CELLS)}] {white_cell}", flush=True)
        gray_ids = gray_complete_case_ids(raw_path(gray_cell, raw_name))
        matrix, _ = load_contract(bundle, gray_cell, "mixed_v2")
        if matrix.shape[1] != len(gray_ids):
            raise RuntimeError(f"gray row count mismatch for {gray_cell}")

        gray_final_map, _ = fit_one_subset(matrix)
        gray_correct = np.asarray(gray_final_map["dufs_liu"], dtype=float)
        gray_upcr_fit = upcr_fit(matrix, **DEPLOYED_FIT)
        gray_upcr = np.asarray(gray_upcr_fit.w @ matrix, dtype=float)
        gray_anchor = np.asarray(bundle[f"{gray_cell}__anchor"], dtype=float)
        if np.corrcoef(gray_upcr, gray_anchor)[0, 1] < 0:
            gray_upcr = -gray_upcr

        white_file = WHITE / "scores" / f"{white_cell}.npz"
        white = np.load(white_file, allow_pickle=False)
        white_ids = [str(value) for value in white["row_ids"]]
        white_by_id = dict(zip(white_ids, np.asarray(white["upcr"], dtype=float)))
        gray_final_by_id = dict(zip(gray_ids, gray_correct))
        gray_upcr_by_id = dict(zip(gray_ids, gray_upcr))
        common = [row_id for row_id in white_ids if row_id in gray_final_by_id]
        if len(common) < 20:
            raise RuntimeError(f"too few common rows for {white_cell}")

        white_risk = np.asarray([white_by_id[row_id] for row_id in common], dtype=float)
        gray_final_risk = -np.asarray([gray_final_by_id[row_id] for row_id in common], dtype=float)
        gray_upcr_risk = -np.asarray([gray_upcr_by_id[row_id] for row_id in common], dtype=float)
        hybrid = 0.5 * (zscore(white_risk) + zscore(gray_final_risk))
        score_file = score_dir / f"{white_cell}.npz"
        np.savez_compressed(
            score_file,
            row_ids=np.asarray(common),
            white_pure_upcr=white_risk,
            gray_mixed_v2_dufs_liu=gray_final_risk,
            gray_mixed_v2_upcr=gray_upcr_risk,
            exploratory_equal_z_hybrid=hybrid,
        )
        score_rows.append({
            "white_cell": white_cell,
            "gray_cell": gray_cell,
            "raw_file": str(raw_path(gray_cell, raw_name).relative_to(REPO)),
            "n_white": len(white_ids),
            "n_gray": len(gray_ids),
            "n_common": len(common),
            "white_score_sha256": sha256_file(white_file),
            "matched_score_file": str(score_file.relative_to(OUT)),
            "matched_score_sha256": sha256_file(score_file),
        })

    source_manifest = {
        "version": VERSION,
        "gray_bundle": str(GRAY_BUNDLE.relative_to(REPO)),
        "gray_bundle_sha256": sha256_file(GRAY_BUNDLE),
        "white_source_freeze": str((WHITE / "SOURCE_FREEZE_MANIFEST.json").relative_to(REPO)),
        "white_source_freeze_sha256": sha256_file(WHITE / "SOURCE_FREEZE_MANIFEST.json"),
        "runner": str(Path(__file__).relative_to(REPO)),
        "runner_sha256": sha256_file(Path(__file__)),
        "cells": score_rows,
    }
    write_json(OUT / "SOURCE_FREEZE_MANIFEST.json", source_manifest)
    freeze = {
        "version": VERSION,
        "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True,
        "score_files": [
            {
                "cell": row["white_cell"],
                "file": row["matched_score_file"],
                "sha256": row["matched_score_sha256"],
            }
            for row in score_rows
        ],
    }
    write_json(OUT / "SCORE_FREEZE_MANIFEST.json", freeze)
    return source_manifest


def evaluation_labels(path: Path, common_ids: list[str]) -> tuple[np.ndarray, dict[str, str]]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    labels, groups = [], {}
    for row_id in common_ids:
        problem, candidate = (int(value) for value in row_id.split(":"))
        labels.append(1 - int(bool(raw[problem]["candidates"][candidate].get("label", False))))
        groups[row_id] = str(problem)
    return np.asarray(labels, dtype=int), groups


def evaluate(source_manifest: dict) -> tuple[list[dict], list[dict], dict]:
    bundle = np.load(GRAY_BUNDLE, allow_pickle=True)
    historic = historical_gray_auc()
    per_cell = []
    for source in source_manifest["cells"]:
        white_cell, gray_cell = source["white_cell"], source["gray_cell"]
        score_path = OUT / source["matched_score_file"]
        if sha256_file(score_path) != source["matched_score_sha256"]:
            raise RuntimeError(f"score freeze mismatch for {white_cell}")
        frozen = np.load(score_path, allow_pickle=False)
        if any("label" in key.lower() for key in frozen.files):
            raise RuntimeError(f"label-like field found in score bundle for {white_cell}")
        common = [str(value) for value in frozen["row_ids"]]
        labels, groups = evaluation_labels(REPO / source["raw_file"], common)

        # Independently reproduce the gray row order/labels after the score freeze.
        gray_ids = gray_complete_case_ids(REPO / source["raw_file"])
        with (REPO / source["raw_file"]).open("rb") as handle:
            raw = pickle.load(handle)
        gray_correct = np.asarray([
            int(bool(raw[int(row.split(":")[0])]["candidates"][int(row.split(":")[1])].get("label", False)))
            for row in gray_ids
        ], dtype=int)
        if not np.array_equal(gray_correct, np.asarray(bundle[f"{gray_cell}__labels"], dtype=int)):
            raise RuntimeError(f"gray raw/bundle label-order mismatch for {gray_cell}")

        row = {
            "white_cell": white_cell,
            "gray_cell": gray_cell,
            "n_white": source["n_white"],
            "n_gray": source["n_gray"],
            "n_common": len(common),
            "n_problem_groups": len(set(groups.values())),
            "hallucination_prevalence": float(labels.mean()),
        }
        for name in (
            "white_pure_upcr",
            "gray_mixed_v2_dufs_liu",
            "gray_mixed_v2_upcr",
            "exploratory_equal_z_hybrid",
        ):
            auc, ap = metrics(labels, np.asarray(frozen[name], dtype=float))
            row[f"{name}__auroc"] = auc
            row[f"{name}__auprc"] = ap
        row["white_minus_gray_final__auroc"] = row["white_pure_upcr__auroc"] - row["gray_mixed_v2_dufs_liu__auroc"]
        row["white_minus_gray_final__auprc"] = row["white_pure_upcr__auprc"] - row["gray_mixed_v2_dufs_liu__auprc"]
        row["white_minus_gray_same_upcr__auroc"] = row["white_pure_upcr__auroc"] - row["gray_mixed_v2_upcr__auroc"]
        row["white_minus_gray_same_upcr__auprc"] = row["white_pure_upcr__auprc"] - row["gray_mixed_v2_upcr__auprc"]
        row["hybrid_minus_gray_final__auroc"] = row["exploratory_equal_z_hybrid__auroc"] - row["gray_mixed_v2_dufs_liu__auroc"]
        row["hybrid_minus_gray_final__auprc"] = row["exploratory_equal_z_hybrid__auprc"] - row["gray_mixed_v2_dufs_liu__auprc"]
        row["white_gray_final_spearman"] = float(spearmanr(
            frozen["white_pure_upcr"], frozen["gray_mixed_v2_dufs_liu"]
        ).statistic)

        # AUROC is invariant to flipping correctness/score together; this checks
        # the refit against the historical full-row gray artifact.
        full_gray_auc = float(roc_auc_score(
            gray_correct,
            -np.asarray(frozen["gray_mixed_v2_dufs_liu"], dtype=float)
        )) if len(common) == len(gray_ids) and common == gray_ids else None
        if full_gray_auc is not None and abs(full_gray_auc - historic[gray_cell]) > 1e-3:
            raise RuntimeError(f"gray refit drift exceeds tolerance for {gray_cell}")
        per_cell.append(row)

    methods = OrderedDict((
        ("white_pure_upcr", "White pure distributed-depth · deployed U-PCR"),
        ("gray_mixed_v2_dufs_liu", "Gray mixed-v2 · DUFS-LIU"),
        ("gray_mixed_v2_upcr", "Gray mixed-v2 · deployed U-PCR control"),
        ("exploratory_equal_z_hybrid", "Exploratory equal-z white + gray"),
    ))
    headline = []
    for key, label in methods.items():
        headline.append({
            "method": key,
            "display_method": label,
            "macro_auroc": float(np.mean([row[f"{key}__auroc"] for row in per_cell])),
            "macro_auprc_hallucination": float(np.mean([row[f"{key}__auprc"] for row in per_cell])),
            "n_cells": len(per_cell),
            "n_common_candidates": int(sum(row["n_common"] for row in per_cell)),
            "status": "posthoc_validation_blocked" if key != "exploratory_equal_z_hybrid" else "exploratory_posthoc",
        })
    headline.sort(key=lambda row: row["macro_auroc"], reverse=True)

    comparisons = []
    for name, lhs, rhs, seed_offset in (
        ("white_minus_gray_final", "white_pure_upcr", "gray_mixed_v2_dufs_liu", 0),
        ("white_minus_gray_same_upcr", "white_pure_upcr", "gray_mixed_v2_upcr", 10),
        ("hybrid_minus_gray_final", "exploratory_equal_z_hybrid", "gray_mixed_v2_dufs_liu", 20),
    ):
        for metric_name in ("auroc", "auprc"):
            delta = np.asarray([
                row[f"{lhs}__{metric_name}"] - row[f"{rhs}__{metric_name}"]
                for row in per_cell
            ])
            lo, hi = cell_bootstrap(delta, SEED + seed_offset + (1 if metric_name == "auprc" else 0))
            comparisons.append({
                "contrast": name,
                "lhs": lhs,
                "rhs": rhs,
                "metric": metric_name,
                "macro_delta": float(delta.mean()),
                "cell_bootstrap_ci_low": lo,
                "cell_bootstrap_ci_high": hi,
                "wins": int(np.sum(delta > TIE_TOLERANCE)),
                "ties": int(np.sum(np.abs(delta) <= TIE_TOLERANCE)),
                "losses": int(np.sum(delta < -TIE_TOLERANCE)),
                "tie_tolerance": TIE_TOLERANCE,
                "inference": "descriptive post-hoc equal-cell bootstrap",
            })
    audit = {
        "version": VERSION,
        "status": "POSTHOC / PRELIMINARY / WHITE VALIDATION BLOCKED",
        "n_cells": len(per_cell),
        "n_common_candidates": int(sum(row["n_common"] for row in per_cell)),
        "white_candidates": int(sum(row["n_white"] for row in per_cell)),
        "gray_candidates": int(sum(row["n_gray"] for row in per_cell)),
        "score_files_verified_before_labels": True,
        "labels_seen_during_fit": False,
        "matched_row_ids_exact": True,
        "gray_raw_bundle_label_order_exact": True,
        "old_gray_auprc_not_comparable": "historical report used correctness=1; this report recomputes AUPRC with hallucination=1",
        "mean_white_gray_final_spearman": float(np.mean([row["white_gray_final_spearman"] for row in per_cell])),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    return per_cell, headline, comparisons, audit


def delta_svg(per_cell: list[dict]) -> str:
    ordered = sorted(per_cell, key=lambda row: row["white_minus_gray_final__auroc"], reverse=True)
    width, left, right, row_h = 980, 260, 60, 30
    height = 70 + row_h * len(ordered)
    scale = (width - left - right) / 0.12
    zero = left + 0.07 * scale
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="White minus gray AUROC by cell">',
             '<style>text{font:13px system-ui;fill:#334155}.axis{stroke:#94a3b8}.pos{fill:#0f766e}.neg{fill:#be123c}</style>',
             f'<line class="axis" x1="{zero:.1f}" y1="25" x2="{zero:.1f}" y2="{height-25}"/>']
    for idx, row in enumerate(ordered):
        y = 45 + idx * row_h
        delta = row["white_minus_gray_final__auroc"]
        x = zero + delta * scale
        start, bar_w = min(zero, x), max(1.0, abs(x - zero))
        cls = "pos" if delta >= 0 else "neg"
        parts.append(f'<text x="8" y="{y+5}">{html.escape(row["white_cell"])}</text>')
        parts.append(f'<rect class="{cls}" x="{start:.1f}" y="{y-10}" width="{bar_w:.1f}" height="15" rx="2"><title>{delta:+.4f}</title></rect>')
        parts.append(f'<text x="{x + (6 if delta >= 0 else -6):.1f}" y="{y+3}" text-anchor="{"start" if delta >= 0 else "end"}">{delta:+.3f}</text>')
    parts.append('</svg>')
    return "".join(parts)


def render_reports(per_cell: list[dict], headline: list[dict], comparisons: list[dict], audit: dict) -> None:
    lookup = {row["contrast"] + "__" + row["metric"]: row for row in comparisons}
    end_auc = lookup["white_minus_gray_final__auroc"]
    end_ap = lookup["white_minus_gray_final__auprc"]
    hybrid_auc = lookup["hybrid_minus_gray_final__auroc"]
    by_method = {row["method"]: row for row in headline}
    md = f"""# Exact-row white-box versus gray-box comparison

**Status: POSTHOC / PRELIMINARY / WHITE VALIDATION BLOCKED.**

This comparison uses the exact intersection of **{audit['n_common_candidates']:,} candidates** in the same 13 dataset/model cells. Hallucination is the positive class (`1 = incorrect`) for both methods. The historical gray-box AUPRC is not reused because it treated correctness as positive.

| Method | Macro AUROC | Hallucination AUPRC |
|---|---:|---:|
| White pure distributed-depth U-PCR | {by_method['white_pure_upcr']['macro_auroc']:.6f} | {by_method['white_pure_upcr']['macro_auprc_hallucination']:.6f} |
| Gray mixed-v2 DUFS-LIU | {by_method['gray_mixed_v2_dufs_liu']['macro_auroc']:.6f} | {by_method['gray_mixed_v2_dufs_liu']['macro_auprc_hallucination']:.6f} |
| Gray mixed-v2 deployed-U-PCR control | {by_method['gray_mixed_v2_upcr']['macro_auroc']:.6f} | {by_method['gray_mixed_v2_upcr']['macro_auprc_hallucination']:.6f} |
| Exploratory equal-z white + gray | {by_method['exploratory_equal_z_hybrid']['macro_auroc']:.6f} | {by_method['exploratory_equal_z_hybrid']['macro_auprc_hallucination']:.6f} |

## Decision

White-box alone has no reliable aggregate performance advantage over the final gray-box system: AUROC delta {end_auc['macro_delta']:+.6f}, 95% equal-cell bootstrap interval [{end_auc['cell_bootstrap_ci_low']:+.6f}, {end_auc['cell_bootstrap_ci_high']:+.6f}]; AUPRC delta {end_ap['macro_delta']:+.6f} [{end_ap['cell_bootstrap_ci_low']:+.6f}, {end_ap['cell_bootstrap_ci_high']:+.6f}]. The same-U-PCR control is also a practical tie.

White-box does have a coverage advantage: {audit['white_candidates']:,} scorable candidates versus {audit['gray_candidates']:,} under the gray 30-feature complete-case contract. The exploratory two-score average gains {hybrid_auc['macro_delta']:+.6f} AUROC, with interval [{hybrid_auc['cell_bootstrap_ci_low']:+.6f}, {hybrid_auc['cell_bootstrap_ci_high']:+.6f}]. Its lower bound is close to zero and the comparison is post-hoc, so it is not a promoted result.

Mean per-cell Spearman correlation between final risk scores is {audit['mean_white_gray_final_spearman']:.4f}; most information is shared, with bounded complementary signal.

## Leakage and claim boundary

- Fit reconstructed row availability and produced scores without reading correctness labels.
- Score bundles were hashed before evaluation opened labels.
- Candidate IDs and gray raw/bundle label order match exactly.
- The comparison and hybrid were proposed after observing both component studies; all inference is descriptive.
- The white capture still lacks corrected live Gate B and the architecture-fidelity pilot.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")

    svg = delta_svg(per_cell)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    (fig_dir / "white_minus_gray_auroc.svg").write_text(svg, encoding="utf-8")
    rows_html = "".join(
        f"<tr><td>{html.escape(row['white_cell'])}</td><td>{row['n_common']:,}</td>"
        f"<td>{row['white_pure_upcr__auroc']:.4f}</td><td>{row['gray_mixed_v2_dufs_liu__auroc']:.4f}</td>"
        f"<td>{row['white_minus_gray_final__auroc']:+.4f}</td></tr>"
        for row in sorted(per_cell, key=lambda item: item["white_minus_gray_final__auroc"], reverse=True)
    )
    method_rows = "".join(
        f"<tr><td>{html.escape(row['display_method'])}</td><td>{row['macro_auroc']:.4f}</td>"
        f"<td>{row['macro_auprc_hallucination']:.4f}</td><td>{html.escape(row['status'])}</td></tr>"
        for row in sorted(headline, key=lambda item: item["macro_auroc"], reverse=True)
    )
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>White vs gray matched comparison</title><style>
:root{{color-scheme:light dark;--bg:#f8fafc;--panel:#fff;--text:#172033;--muted:#64748b;--line:#cbd5e1;--warn:#9a3412}}@media(prefers-color-scheme:dark){{:root{{--bg:#0f172a;--panel:#172033;--text:#e2e8f0;--muted:#94a3b8;--line:#334155;--warn:#fdba74}}}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:28px 18px 60px}}section{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin:16px 0;overflow:auto}}h1{{font-size:clamp(1.8rem,5vw,3rem);margin:.2em 0}}.status{{color:var(--warn);font-weight:800;letter-spacing:.04em}}.metric{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.metric div{{border:1px solid var(--line);border-radius:10px;padding:14px}}.metric strong{{display:block;font-size:1.55rem}}table{{border-collapse:collapse;width:100%;min-width:640px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}small,.muted{{color:var(--muted)}}svg{{width:100%;height:auto;min-width:700px}}</style></head><body><main>
<p class="status">POSTHOC / PRELIMINARY / WHITE VALIDATION BLOCKED</p><h1>Exact-row white-box versus gray-box</h1><p>Same 13 dataset/model cells and the exact intersection of {audit['n_common_candidates']:,} candidates. Positive class is hallucination for every AUPRC.</p>
<section class="metric"><div><span>White AUROC</span><strong>{by_method['white_pure_upcr']['macro_auroc']:.4f}</strong></div><div><span>Gray AUROC</span><strong>{by_method['gray_mixed_v2_dufs_liu']['macro_auroc']:.4f}</strong></div><div><span>White − gray</span><strong>{end_auc['macro_delta']:+.4f}</strong><small>CI {end_auc['cell_bootstrap_ci_low']:+.4f} to {end_auc['cell_bootstrap_ci_high']:+.4f}</small></div><div><span>Exploratory hybrid</span><strong>{by_method['exploratory_equal_z_hybrid']['macro_auroc']:.4f}</strong></div></section>
<section><h2>Headline methods</h2><table><thead><tr><th>Method</th><th>AUROC</th><th>Hallucination AUPRC</th><th>Status</th></tr></thead><tbody>{method_rows}</tbody></table></section>
<section><h2>Per-cell AUROC difference</h2><p class="muted">Positive favors white. Negative favors gray.</p>{svg}</section>
<section><h2>Per-cell audit table</h2><table><thead><tr><th>Cell</th><th>Common n</th><th>White</th><th>Gray</th><th>Delta</th></tr></thead><tbody>{rows_html}</tbody></table></section>
<section><h2>Decision</h2><p>White-box alone is a practical aggregate tie, not a robust improvement. It scores {audit['white_candidates']:,} candidates versus {audit['gray_candidates']:,} for the gray complete-case contract. The equal-z hybrid is promising but post-hoc; its AUROC delta is {hybrid_auc['macro_delta']:+.4f}, with the lower interval bound close to zero.</p><p>Mean risk-score Spearman correlation: {audit['mean_white_gray_final_spearman']:.4f}. The white capture remains blocked on corrected live Gate B and architecture fidelity.</p></section>
</main></body></html>"""
    (OUT / "REPORT.html").write_text(report, encoding="utf-8")


def write_run_definition() -> None:
    write_json(OUT / "RUN_DEFINITION.json", {
        "version": VERSION,
        "status": "posthoc_preliminary_validation_blocked",
        "cells": list(CELLS),
        "positive_class": "hallucination = 1 - correctness",
        "white_method": "pure distributed-depth deployed U-PCR",
        "gray_final_method": "mixed-v2 DUFS-LIU, seeds 11/23/37, epochs 80, k=7, lambda=0.1",
        "same_solver_control": "mixed-v2 deployed U-PCR",
        "exploratory_hybrid": "equal mean of within-cell z-scored white and gray-final risk scores",
        "row_contract": "exact row_id intersection after reproducing gray H16 complete-case mask",
        "bootstrap": {"unit": "equal cells", "draws": 20000, "seed_base": SEED},
        "labels_seen_during_fit": False,
        "retrospective": True,
    })


def write_report_manifest() -> None:
    files = [path for path in OUT.rglob("*") if path.is_file() and path.name != "REPORT_MANIFEST.json"]
    write_json(OUT / "REPORT_MANIFEST.json", {
        "version": VERSION,
        "artifacts": [
            {"file": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(files)
        ],
    })


def main() -> None:
    write_run_definition()
    source_manifest = fit_and_freeze()
    per_cell, headline, comparisons, audit = evaluate(source_manifest)
    write_csv(OUT / "per_cell_metrics.csv", per_cell)
    write_csv(OUT / "headline_summary.csv", headline)
    write_csv(OUT / "paired_comparisons.csv", comparisons)
    write_json(OUT / "AUDIT.json", audit)
    render_reports(per_cell, headline, comparisons, audit)
    write_report_manifest()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
