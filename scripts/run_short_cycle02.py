"""Run the registered fixed-group Joint diagnostic in two firewalled phases."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
CAPSULE = MAIN / "local_cache" / "short_cycle01_code"
LIVE = Path(r"C:/Users/omris/TAU/hd_jlsml_v2_wt")
LABELS = LIVE / "results/joint_lsml_optimization_v2/labels/prmbench_qwen3_8b_labels.npz"
SOURCE = MAIN / "results" / "localization_short_cycle01"
OUT = MAIN / "results" / "localization_short_cycle02"
PROTOCOL = MAIN / "docs" / "experiments" / "LOCALIZATION_SHORT_CYCLE_02_20260907.md"

sys.path.insert(0, str(CAPSULE))
import spectral_utils  # noqa: E402

spectral_utils.__path__.append(str(MAIN / "spectral_utils"))
from spectral_utils.short_cycle02_localization import (  # noqa: E402
    METHODS as NEW_METHODS,
    SETTINGS,
    score_fixed4_answer,
)
from spectral_utils.window_localization import WINDOW_FEATURE_NAMES, make_window_plan  # noqa: E402


METHOD_KEYS = {
    "joint_internal_modelinv_lam0": "joint_modelinv_lam0",
    "joint_fixed4_modelinv_lam0": "joint_fixed4_modelinv_lam0",
    "fixed4_continuous_lsml": "fixed4_continuous_lsml",
    "iu": "iu",
    "equal": "equal",
}
PAIRS = (
    ("joint_fixed4_modelinv_lam0", "iu"),
    ("joint_fixed4_modelinv_lam0", "joint_internal_modelinv_lam0"),
    ("fixed4_continuous_lsml", "joint_fixed4_modelinv_lam0"),
    ("fixed4_continuous_lsml", "iu"),
    ("iu", "equal"),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    ordered = scores[order]
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and ordered[end + 1] == ordered[start]:
            end += 1
        if end > start:
            ranks[order[start:end + 1]] = (start + end + 2) / 2.0
        start = end + 1
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if not n_positive or not n_negative:
        return float("nan")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def phase_scores() -> None:
    source_config = json.loads((SOURCE / "CONFIG.json").read_text(encoding="utf-8"))
    source_frozen = json.loads((SOURCE / "SCORES_FROZEN.json").read_text(encoding="utf-8"))
    if source_config.get("labels_accessed") is not False or source_frozen.get("labels_accessed") is not False:
        raise RuntimeError("short-cycle-1 score source is not label-firewalled")
    if source_config["selected_row_ids"] != [row["row_id"] for row in source_frozen["rows"]]:
        raise RuntimeError("short-cycle-1 selected roster drift")

    OUT.mkdir(parents=True, exist_ok=True)
    module_path = MAIN / "spectral_utils" / "short_cycle02_localization.py"
    config = {
        "pilot": "localization-short-cycle02-fixed4",
        "population": source_config["population"],
        "selected_rows": source_config["selected_rows"],
        "selected_row_ids": source_config["selected_row_ids"],
        "lengths": source_config["lengths"],
        "window_settings": source_config["settings"],
        "new_method_settings": SETTINGS,
        "new_methods": list(NEW_METHODS),
        "reference_methods": ["joint_internal_modelinv_lam0", "iu", "equal"],
        "source_score_freeze_sha256": sha(SOURCE / "SCORES_FROZEN.json"),
        "source_config_sha256": sha(SOURCE / "CONFIG.json"),
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": sha(Path(__file__)),
        "fitting_module_sha256": sha(module_path),
        "labels_accessed": False,
    }
    (OUT / "CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    for position, source_row in enumerate(source_frozen["rows"], 1):
        started = time.monotonic()
        row_number = int(source_row["row"])
        try:
            if source_row["status"] != "OK":
                raise RuntimeError(f"source row unavailable: {source_row['status']}")
            with np.load(SOURCE / f"row_{row_number}.npz", allow_pickle=False) as source:
                values = np.asarray(source["feature_values"], dtype=np.float64)
                step_starts = np.asarray(source["step_starts"], dtype=np.int64)
                step_ends = np.asarray(source["step_ends"], dtype=np.int64)
                frozen_window_starts = np.asarray(source["window_starts"], dtype=np.int64)
                frozen_window_ends = np.asarray(source["window_ends"], dtype=np.int64)
            token_count = int(source_row["tokens"])
            plan = make_window_plan(token_count, 32, 32)
            if not np.array_equal(plan.starts, frozen_window_starts) or not np.array_equal(plan.ends, frozen_window_ends):
                raise RuntimeError("frozen window geometry drift")
            arrays, report = score_fixed4_answer(
                values, WINDOW_FEATURE_NAMES, plan, step_starts, step_ends,
            )
            if not arrays or not any(method + "__step" in arrays for method in NEW_METHODS):
                raise RuntimeError("no new method produced scores")
            np.savez_compressed(OUT / f"row_{row_number}.npz", **arrays)
            status = "OK"
        except Exception as error:
            report = {"labels_accessed": False, "error": f"{type(error).__name__}: {error}"}
            status = "FAILED"
        row = {
            "row": row_number,
            "row_id": source_row["row_id"],
            "status": status,
            "tokens": int(source_row["tokens"]),
            "elapsed_seconds": time.monotonic() - started,
            "report": report,
        }
        rows.append(row)
        (OUT / "scores_progress.json").write_text(json.dumps({
            "complete": position, "total": len(source_frozen["rows"]), "rows": rows,
        }, indent=2), encoding="utf-8")
        print(
            f"{position}/{len(source_frozen['rows'])} row={row_number} "
            f"status={status} elapsed={row['elapsed_seconds']:.2f}s",
            flush=True,
        )
    frozen = {
        "config": config,
        "rows": rows,
        "labels_accessed": False,
        "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
    }
    (OUT / "SCORES_FROZEN.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")


def _method_status(method, old_row, new_row):
    key = METHOD_KEYS[method]
    source = new_row if method in NEW_METHODS else old_row
    if source.get("status") != "OK":
        return "UNAVAILABLE_PRELABEL_FIT"
    return source.get("report", {}).get("methods", {}).get(key, {}).get("status", "MISSING")


def _paired_summary(answer_values, statuses, left, right, rng):
    ids = sorted(
        row_id for row_id, payload in answer_values.items()
        if left in payload and right in payload
        and statuses[row_id].get(left) == "OK"
        and statuses[row_id].get(right) == "OK"
    )
    labels = []
    left_scores = []
    right_scores = []
    for row_id in ids:
        payload = answer_values[row_id]
        labels.extend(payload["target"])
        left_scores.extend(payload[left])
        right_scores.extend(payload[right])
    observed_left = auc(labels, left_scores) if labels else float("nan")
    observed_right = auc(labels, right_scores) if labels else float("nan")
    deltas = []
    for _ in range(2000):
        draw = rng.choice(ids, size=len(ids), replace=True)
        target_draw = []
        left_draw = []
        right_draw = []
        for row_id in draw:
            payload = answer_values[str(row_id)]
            target_draw.extend(payload["target"])
            left_draw.extend(payload[left])
            right_draw.extend(payload[right])
        difference = auc(target_draw, left_draw) - auc(target_draw, right_draw)
        if np.isfinite(difference):
            deltas.append(float(difference))
    return {
        "left": left,
        "right": right,
        "strict_common_answers": len(ids),
        "strict_common_steps": len(labels),
        "left_auroc": observed_left,
        "right_auroc": observed_right,
        "observed_delta": observed_left - observed_right,
        "bootstrap_draws": len(deltas),
        "bootstrap_mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "bootstrap_ci95": [
            float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)),
        ] if deltas else [float("nan"), float("nan")],
        "bootstrap_unit": "answer/source group",
    }


def phase_evaluate() -> None:
    source_frozen = json.loads((SOURCE / "SCORES_FROZEN.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "SCORES_FROZEN.json").read_text(encoding="utf-8"))
    if frozen.get("labels_accessed") is not False:
        raise RuntimeError("new score freeze is not label-free")
    if [row["row_id"] for row in source_frozen["rows"]] != [row["row_id"] for row in frozen["rows"]]:
        raise RuntimeError("source/new row roster mismatch")

    old_rows = {row["row_id"]: row for row in source_frozen["rows"]}
    new_rows = {row["row_id"]: row for row in frozen["rows"]}
    methods = tuple(METHOD_KEYS)
    available = {method: [] for method in methods}
    statuses = {}
    answer_values = {}
    per_row = []
    with np.load(LABELS, allow_pickle=False) as label_file:
        label_ids = [str(value) for value in label_file["row_ids"]]
        label_positions = {row_id: index for index, row_id in enumerate(label_ids)}
        for row_id in frozen["config"]["selected_row_ids"]:
            if row_id not in label_positions:
                raise RuntimeError(f"missing labels for {row_id}")
            old_row = old_rows[row_id]
            new_row = new_rows[row_id]
            row_number = int(old_row["row"])
            label_index = label_positions[row_id]
            lo = int(label_file["step_flag_offsets"][label_index])
            hi = int(label_file["step_flag_offsets"][label_index + 1])
            target = np.asarray(label_file["step_error_flags"][lo:hi], dtype=np.int64)
            answer_values[row_id] = {"target": target.tolist()}
            statuses[row_id] = {}
            row_result = {
                "row": row_number, "row_id": row_id, "steps": len(target),
                "positive_steps": int(target.sum()),
            }
            old_path = SOURCE / f"row_{row_number}.npz"
            new_path = OUT / f"row_{row_number}.npz"
            with np.load(old_path, allow_pickle=False) as old_scores:
                new_scores_context = np.load(new_path, allow_pickle=False) if new_path.exists() else None
                try:
                    for method in methods:
                        status = _method_status(method, old_row, new_row)
                        statuses[row_id][method] = status
                        key = METHOD_KEYS[method] + "__step"
                        score_file = new_scores_context if method in NEW_METHODS else old_scores
                        if score_file is None or key not in score_file.files:
                            row_result[method] = {"status": "UNAVAILABLE_PRELABEL_FIT"}
                            continue
                        values = np.asarray(score_file[key], dtype=np.float64)
                        if len(values) != len(target) or not np.isfinite(values).all():
                            raise RuntimeError(f"invalid aligned scores: {row_id} {method}")
                        answer_values[row_id][method] = values.tolist()
                        available[method].extend(zip(target.tolist(), values.tolist()))
                        row_result[method] = {
                            "status": status,
                            "auroc": auc(target, values),
                        }
                finally:
                    if new_scores_context is not None:
                        new_scores_context.close()
            per_row.append(row_result)

    coverage = {
        method: sum(method in payload for payload in answer_values.values())
        for method in methods
    }
    strict_coverage = {
        method: sum(statuses[row_id].get(method) == "OK" and method in answer_values[row_id]
                    for row_id in answer_values)
        for method in methods
    }
    status_counts = {method: {} for method in methods}
    for method in methods:
        for row_id in answer_values:
            status = statuses[row_id].get(method, "MISSING")
            status_counts[method][status] = status_counts[method].get(status, 0) + 1

    rng = np.random.default_rng(2026090702)
    pairwise = {
        f"{left}_minus_{right}": _paired_summary(
            answer_values, statuses, left, right, rng,
        )
        for left, right in PAIRS
    }
    output = {
        "labels_accessed": True,
        "selection_frozen_before_labels": True,
        "source_label_sha256": sha(LABELS),
        "coverage": coverage,
        "strict_coverage": strict_coverage,
        "fit_status_counts": status_counts,
        "pooled_available_auroc": {
            method: auc(
                [item[0] for item in values], [item[1] for item in values],
            ) if values else float("nan")
            for method, values in available.items()
        },
        "pairwise_strict": pairwise,
        "per_row": per_row,
    }
    (OUT / "EVALUATION.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "coverage": coverage,
        "strict_coverage": strict_coverage,
        "pooled_available_auroc": output["pooled_available_auroc"],
        "pairwise_strict": pairwise,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("scores", "evaluate"), required=True)
    arguments = parser.parse_args()
    if arguments.phase == "scores":
        phase_scores()
    else:
        phase_evaluate()


if __name__ == "__main__":
    main()

