"""Run the bounded answer-only localization pilot.

The script has two explicit phases. ``--phase scores`` reads telemetry only and
writes frozen score/fit artifacts. ``--phase evaluate`` reads the frozen pilot
scores and then joins the registered PRM step labels. It never changes Claude's
worktree or its result namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MAIN = Path(__file__).resolve().parents[1]
CAPSULE = MAIN / "local_cache" / "short_cycle01_code"
LIVE = Path(r"C:/Users/omris/TAU/hd_jlsml_v2_wt")
CELL = LIVE / "results/joint_lsml_optimization_v2/cells/prmbench_qwen3_8b.npz"
LABELS = LIVE / "results/joint_lsml_optimization_v2/labels/prmbench_qwen3_8b_labels.npz"
OUT = MAIN / "results" / "localization_short_cycle01"
sys.path.insert(0, str(CAPSULE))
import spectral_utils  # noqa: E402

spectral_utils.__path__.append(str(MAIN / "spectral_utils"))
from spectral_utils.short_cycle_localization import SETTINGS, fit_answer  # noqa: E402

STREAM_NAMES = (
    "trace_length_series", "entropy_series", "entropy_rolling_spectral_entropy",
    "entropy_rolling_low_band_power", "entropy_rolling_high_band_power",
    "entropy_rolling_hl_ratio", "entropy_rolling_dominant_freq",
    "entropy_rolling_spectral_centroid", "entropy_stft_high_series",
    "entropy_stft_frame_entropy", "entropy_rolling_tail_ratio",
    "entropy_sw_var_series", "entropy_pe_series", "entropy_rolling_rs_hurst",
    "entropy_cusum_abs_series", "spilled_series", "spilled_sw_var_series",
    "spilled_cusum_abs_series", "spilled_rolling_min", "energy_series",
    "energy_rolling_min", "energy_sw_var_series", "energy_cusum_abs_series",
    "top1_logprob_series", "logprob_margin_series", "topk_entropy_series",
    "topk_varentropy_series", "topk_renyi2_series", "topk_tail_mass_series",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_ids(row_ids: np.ndarray, offsets: np.ndarray) -> list[int]:
    lengths = np.diff(offsets)
    candidates = [int(i) for i in np.flatnonzero((lengths >= 1024) & (lengths <= 2048))]
    # Hash selection is deterministic and does not inspect labels.
    candidates.sort(key=lambda i: hashlib.sha256(str(row_ids[i]).encode()).hexdigest())
    selected = candidates[:30]
    if len(selected) < 30:
        raise RuntimeError(f"only {len(selected)} eligible long answers")
    if len({str(row_ids[i]) for i in selected}) != len(selected):
        raise RuntimeError("duplicate selected row IDs")
    return selected


def load_row(bundle, row: int):
    offsets = np.asarray(bundle["token_offsets"], dtype=np.int64)
    lo, hi = int(offsets[row]), int(offsets[row + 1])
    a, b = int(bundle["step_row_offsets"][row]), int(bundle["step_row_offsets"][row + 1])
    starts = np.asarray(bundle["step_starts"][a:b], dtype=np.int64) - lo
    ends = np.asarray(bundle["step_ends"][a:b], dtype=np.int64) - lo
    return np.asarray(bundle["raw"][lo:hi], dtype=np.float64), starts, ends


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos = labels == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return float("nan") if not npos or not nneg else float(
        (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    )


def phase_scores() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(CELL, allow_pickle=False) as bundle:
        row_ids = np.asarray(bundle["row_ids"])
        offsets = np.asarray(bundle["token_offsets"], dtype=np.int64)
        selected = select_ids(row_ids, offsets)
        config = {
            "pilot": "localization-short-cycle01",
            "population": "prmbench_qwen3_8b",
            "selected_rows": selected,
            "selected_row_ids": [str(row_ids[i]) for i in selected],
            "lengths": [int(offsets[i + 1] - offsets[i]) for i in selected],
            "settings": SETTINGS,
            "stream_names": STREAM_NAMES,
            "cell_sha256": sha(CELL), "labels_accessed": False,
            "runner_sha256": sha(Path(__file__)),
            "fitting_module_sha256": sha(MAIN / "spectral_utils" / "short_cycle_localization.py"),
            "source_capsule": json.loads((CAPSULE / "UPSTREAM_SOURCES.json").read_text()),
        }
        (OUT / "CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        rows = []
        for pos, row in enumerate(selected, 1):
            started = time.monotonic()
            raw, starts, ends = load_row(bundle, row)
            try:
                arrays, report = fit_answer(raw, STREAM_NAMES, starts, ends)
                status = "OK"
                np.savez_compressed(OUT / f"row_{row}.npz", **arrays)
            except Exception as exc:  # preserve typed row failure and continue
                report = {"labels_accessed": False, "error": f"{type(exc).__name__}: {exc}"}
                status = "FAILED"
            rows.append({"row": row, "row_id": str(row_ids[row]), "status": status,
                         "tokens": int(len(raw)), "elapsed_seconds": time.monotonic() - started,
                         "report": report})
            (OUT / "scores_progress.json").write_text(json.dumps({"complete": pos, "total": len(selected), "rows": rows}, indent=2), encoding="utf-8")
            print(f"{pos}/{len(selected)} row={row} status={status} elapsed={rows[-1]['elapsed_seconds']:.2f}s", flush=True)
    final = {"config": config, "rows": rows, "labels_accessed": False,
             "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows)}
    (OUT / "SCORES_FROZEN.json").write_text(json.dumps(final, indent=2), encoding="utf-8")


def phase_evaluate() -> None:
    frozen = json.loads((OUT / "SCORES_FROZEN.json").read_text(encoding="utf-8"))
    if frozen.get("labels_accessed") is not False:
        raise RuntimeError("score freeze is not label-free")
    with np.load(LABELS, allow_pickle=False) as labels:
        label_ids = [str(x) for x in labels["row_ids"]]
        label_pos = {x: i for i, x in enumerate(label_ids)}
        by_method = {m: [] for m in ("joint_modelinv_lam0", "iu", "equal")}
        counts = {m: {"answers": 0, "steps": 0} for m in by_method}
        per_row = []
        answer_values = {}
        for item in frozen["rows"]:
            if item["status"] != "OK":
                continue
            row_id = item["row_id"]
            if row_id not in label_pos:
                raise RuntimeError(f"missing label row {row_id}")
            with np.load(OUT / f"row_{item['row']}.npz", allow_pickle=False) as scores:
                flag_lo, flag_hi = int(labels["step_flag_offsets"][label_pos[row_id]]), int(labels["step_flag_offsets"][label_pos[row_id] + 1])
                target = np.asarray(labels["step_error_flags"][flag_lo:flag_hi], dtype=int)
                row_out = {"row": item["row"], "row_id": row_id, "n_steps": len(target)}
                for method in by_method:
                    key = method + "__step"
                    if key not in scores.files:
                        row_out[method] = {"status": "UNAVAILABLE_PRELABEL_FIT"}
                        continue
                    values = np.asarray(scores[key], dtype=float)
                    if len(values) != len(target):
                        raise RuntimeError(f"step alignment mismatch {row_id} {method}")
                    by_method[method].extend(zip(target.tolist(), values.tolist()))
                    counts[method]["answers"] += 1; counts[method]["steps"] += len(target)
                    fit_status = next((r["report"]["methods"].get(method, {}).get("status")
                                       for r in frozen["rows"] if r["row_id"] == row_id), "UNKNOWN")
                    row_out[method] = {"status": fit_status, "auroc": auc(target, values), "positive": int(target.sum()), "steps": len(target)}
                    answer_values.setdefault(row_id, {"target": target})[method] = values
                per_row.append(row_out)
        available_ids = {m: {x["row_id"] for x in per_row if x.get(m, {}).get("status") not in (None, "UNAVAILABLE_PRELABEL_FIT", "MISSING")} for m in by_method}
        complete_ids = available_ids
        strict_ids = {m: {x["row_id"] for x in per_row if x.get(m, {}).get("status") == "OK"} for m in by_method}
        common_ids = set.intersection(*complete_ids.values()) if complete_ids else set()
        common_strict_ids = set.intersection(*strict_ids.values()) if strict_ids else set()
        common_values = {m: [] for m in by_method}
        for item in per_row:
            if item["row_id"] not in common_ids:
                continue
            with np.load(OUT / f"row_{item['row']}.npz", allow_pickle=False) as scores:
                label_index = label_pos[item["row_id"]]
                lo, hi = int(labels["step_flag_offsets"][label_index]), int(labels["step_flag_offsets"][label_index + 1])
                target = np.asarray(labels["step_error_flags"][lo:hi], dtype=int)
                for method in by_method:
                    common_values[method].extend(zip(target.tolist(), np.asarray(scores[method + "__step"], float).tolist()))
        # Grouped bootstrap resamples whole answers, preserving within-answer
        # dependence. It is descriptive for this 30-answer pilot, not a fresh
        # benchmark confidence interval.
        rng = np.random.default_rng(2026090602)
        bootstrap = {}
        ids = sorted(common_strict_ids)
        for other in ("iu", "equal"):
            deltas = []
            for _ in range(2000):
                draw = rng.choice(ids, size=len(ids), replace=True)
                left_t, left_s, right_s = [], [], []
                for row_id in draw:
                    payload = answer_values[row_id]
                    left_t.extend(payload["target"]); left_s.extend(payload["joint_modelinv_lam0"]); right_s.extend(payload[other])
                da = auc(np.asarray(left_t), np.asarray(left_s)) - auc(np.asarray(left_t), np.asarray(right_s))
                if np.isfinite(da):
                    deltas.append(float(da))
            bootstrap[f"joint_minus_{other}"] = {
                "n_draws": len(deltas), "mean": float(np.mean(deltas)),
                "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
                "unit": "answer/source group", "strict_common_answers": len(ids),
            }
        status_counts = {}
        for method in by_method:
            status_counts[method] = {}
            for row in per_row:
                status = row.get(method, {}).get("status", "MISSING")
                status_counts[method][status] = status_counts[method].get(status, 0) + 1
        summary = {"labels_accessed": True, "selection_frozen_before_labels": True,
                   "source_label_sha256": sha(LABELS), "counts": counts, "per_row": per_row,
                   "coverage": {m: len(ids) for m, ids in available_ids.items()},
                   "strict_coverage": {m: len(ids) for m, ids in strict_ids.items()},
                   "fit_status_counts": status_counts,
                   "common_complete_answers": len(common_ids), "common_strict_answers": len(common_strict_ids),
                   "auroc": {m: auc(np.asarray([x[0] for x in vals]), np.asarray([x[1] for x in vals])) if vals else float("nan") for m, vals in by_method.items()},
                   "common_complete_auroc": {m: auc(np.asarray([x[0] for x in vals]), np.asarray([x[1] for x in vals])) if vals else float("nan") for m, vals in common_values.items()},
                   "grouped_bootstrap_strict_common": bootstrap}
    (OUT / "EVALUATION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["auroc"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("scores", "evaluate"), required=True)
    args = parser.parse_args()
    phase_scores() if args.phase == "scores" else phase_evaluate()


if __name__ == "__main__":
    main()
