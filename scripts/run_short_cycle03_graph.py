"""Run the answer-only Joint-LIU graph test in score/evaluation phases."""
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
OUT = MAIN / "results" / "localization_short_cycle03_graph"
PROTOCOL = MAIN / "docs" / "experiments" / "LOCALIZATION_SHORT_CYCLE_03_GRAPH_20260907.md"

sys.path.insert(0, str(CAPSULE))
import spectral_utils  # noqa: E402

spectral_utils.__path__.append(str(MAIN / "spectral_utils"))
from spectral_utils.short_cycle03_graph_localization import (  # noqa: E402
    METHODS as NEW_METHODS,
    SETTINGS,
    score_graph_answer,
)
from spectral_utils.window_localization import WINDOW_FEATURE_NAMES, make_window_plan  # noqa: E402


METHOD_KEYS = {
    "internal_joint_liu010_answer_only": "internal_joint_liu010_answer_only",
    "permctl_graph_internal_joint_liu010_answer_only": "permctl_graph_internal_joint_liu010_answer_only",
    "joint_modelinv_lam0": "joint_modelinv_lam0",
    "iu": "iu",
    "equal": "equal",
}
PAIRS = (
    ("internal_joint_liu010_answer_only", "joint_modelinv_lam0"),
    ("permctl_graph_internal_joint_liu010_answer_only", "joint_modelinv_lam0"),
    ("internal_joint_liu010_answer_only", "permctl_graph_internal_joint_liu010_answer_only"),
    ("internal_joint_liu010_answer_only", "iu"),
    ("joint_modelinv_lam0", "iu"),
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


def _row_permutation_seed(row_id: str) -> int:
    suffix = int(hashlib.sha256(str(row_id).encode()).hexdigest()[:8], 16)
    return int(SETTINGS["permutation_seed_base"] + suffix)


def phase_scores() -> None:
    source_config = json.loads((SOURCE / "CONFIG.json").read_text(encoding="utf-8"))
    source_frozen = json.loads((SOURCE / "SCORES_FROZEN.json").read_text(encoding="utf-8"))
    if source_config.get("labels_accessed") is not False or source_frozen.get("labels_accessed") is not False:
        raise RuntimeError("source scores are not label-firewalled")
    OUT.mkdir(parents=True, exist_ok=True)
    module_path = MAIN / "spectral_utils" / "short_cycle03_graph_localization.py"
    gate_path = MAIN / "spectral_utils" / "adapted_dufs.py"
    config = {
        "pilot": "localization-short-cycle03-answer-only-graph",
        "population": source_config["population"],
        "selected_rows": source_config["selected_rows"],
        "selected_row_ids": source_config["selected_row_ids"],
        "lengths": source_config["lengths"],
        "settings": SETTINGS,
        "new_methods": list(NEW_METHODS[1:]),
        "reference_methods": ["joint_modelinv_lam0", "iu", "equal"],
        "source_score_freeze_sha256": sha(SOURCE / "SCORES_FROZEN.json"),
        "source_config_sha256": sha(SOURCE / "CONFIG.json"),
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": sha(Path(__file__)),
        "fitting_module_sha256": sha(module_path),
        "adapted_dufs_sha256": sha(gate_path),
        "labels_accessed": False,
    }
    (OUT / "CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    for position, source_row in enumerate(source_frozen["rows"], 1):
        started = time.monotonic()
        row_number = int(source_row["row"])
        joint_meta = source_row.get("report", {}).get("methods", {}).get("joint_modelinv_lam0", {})
        try:
            if "groups" not in joint_meta:
                raise RuntimeError("UNAVAILABLE_SOURCE_INTERNAL_PARTITION")
            with np.load(SOURCE / f"row_{row_number}.npz", allow_pickle=False) as source:
                if "joint_modelinv_lam0__window" not in source.files:
                    raise RuntimeError("UNAVAILABLE_SOURCE_LAMBDA0_SCORE")
                feature_values = np.asarray(source["feature_values"], dtype=np.float64)
                step_starts = np.asarray(source["step_starts"], dtype=np.int64)
                step_ends = np.asarray(source["step_ends"], dtype=np.int64)
                expected_lambda0 = np.asarray(source["joint_modelinv_lam0__window"], dtype=np.float64)
                frozen_window_starts = np.asarray(source["window_starts"], dtype=np.int64)
                frozen_window_ends = np.asarray(source["window_ends"], dtype=np.int64)
            plan = make_window_plan(int(source_row["tokens"]), 32, 32)
            if not np.array_equal(plan.starts, frozen_window_starts) or not np.array_equal(plan.ends, frozen_window_ends):
                raise RuntimeError("frozen window geometry drift")
            arrays, report = score_graph_answer(
                feature_values,
                WINDOW_FEATURE_NAMES,
                plan,
                step_starts,
                step_ends,
                joint_meta["groups"],
                source_row["report"]["shared"]["active_features"],
                permutation_seed=_row_permutation_seed(source_row["row_id"]),
            )
            replay = np.asarray(arrays["joint_modelinv_lam0_replay__window"], dtype=np.float64)
            replay_error = float(np.max(np.abs(replay - expected_lambda0)))
            if replay_error > 1e-10:
                raise RuntimeError(f"LAMBDA0_REPLAY_DRIFT: {replay_error}")
            report["lambda0_replay_max_abs_error"] = replay_error
            np.savez_compressed(OUT / f"row_{row_number}.npz", **arrays)
            status = "OK"
        except Exception as error:
            report = {
                "labels_accessed": False,
                "error": f"{type(error).__name__}: {error}",
            }
            status = "UNAVAILABLE" if "UNAVAILABLE_SOURCE" in str(error) else "FAILED"
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
    if method in NEW_METHODS[1:]:
        if new_row.get("status") != "OK":
            return "UNAVAILABLE_PRELABEL_FIT"
        return new_row["report"]["methods"].get(method, {}).get("status", "MISSING")
    return old_row.get("report", {}).get("methods", {}).get(method, {}).get("status", "MISSING")


def _pairwise(answer_values, statuses, left, right, rng):
    ids = sorted(
        row_id for row_id, payload in answer_values.items()
        if left in payload and right in payload
        and statuses[row_id].get(left) == "OK"
        and statuses[row_id].get(right) == "OK"
    )
    target = []
    left_scores = []
    right_scores = []
    for row_id in ids:
        payload = answer_values[row_id]
        target.extend(payload["target"])
        left_scores.extend(payload[left])
        right_scores.extend(payload[right])
    left_auc = auc(target, left_scores) if target else float("nan")
    right_auc = auc(target, right_scores) if target else float("nan")
    differences = []
    for _ in range(2000):
        draw = rng.choice(ids, len(ids), replace=True)
        draw_target = []
        draw_left = []
        draw_right = []
        for value in draw:
            payload = answer_values[str(value)]
            draw_target.extend(payload["target"])
            draw_left.extend(payload[left])
            draw_right.extend(payload[right])
        difference = auc(draw_target, draw_left) - auc(draw_target, draw_right)
        if np.isfinite(difference):
            differences.append(float(difference))
    return {
        "left": left,
        "right": right,
        "strict_common_answers": len(ids),
        "strict_common_steps": len(target),
        "left_auroc": left_auc,
        "right_auroc": right_auc,
        "observed_delta": left_auc - right_auc,
        "bootstrap_draws": len(differences),
        "bootstrap_mean_delta": float(np.mean(differences)),
        "bootstrap_ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "bootstrap_unit": "answer/source group",
    }


def phase_evaluate() -> None:
    source_frozen = json.loads((SOURCE / "SCORES_FROZEN.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "SCORES_FROZEN.json").read_text(encoding="utf-8"))
    if frozen.get("labels_accessed") is not False:
        raise RuntimeError("graph scores were not frozen label-free")
    old_rows = {row["row_id"]: row for row in source_frozen["rows"]}
    new_rows = {row["row_id"]: row for row in frozen["rows"]}
    methods = tuple(METHOD_KEYS)
    answer_values = {}
    statuses = {}
    per_row = []
    with np.load(LABELS, allow_pickle=False) as label_file:
        label_ids = [str(value) for value in label_file["row_ids"]]
        positions = {row_id: index for index, row_id in enumerate(label_ids)}
        for row_id in frozen["config"]["selected_row_ids"]:
            old_row = old_rows[row_id]
            new_row = new_rows[row_id]
            row_number = int(old_row["row"])
            label_index = positions[row_id]
            lo = int(label_file["step_flag_offsets"][label_index])
            hi = int(label_file["step_flag_offsets"][label_index + 1])
            target = np.asarray(label_file["step_error_flags"][lo:hi], dtype=np.int64)
            answer_values[row_id] = {"target": target.tolist()}
            statuses[row_id] = {}
            row_result = {"row": row_number, "row_id": row_id, "steps": len(target)}
            with np.load(SOURCE / f"row_{row_number}.npz", allow_pickle=False) as old_scores:
                new_scores = np.load(OUT / f"row_{row_number}.npz", allow_pickle=False) if (OUT / f"row_{row_number}.npz").exists() else None
                try:
                    for method, key_name in METHOD_KEYS.items():
                        status = _method_status(method, old_row, new_row)
                        statuses[row_id][method] = status
                        score_file = new_scores if method in NEW_METHODS[1:] else old_scores
                        key = key_name + "__step"
                        if score_file is None or key not in score_file.files:
                            row_result[method] = {"status": "UNAVAILABLE_PRELABEL_FIT"}
                            continue
                        values = np.asarray(score_file[key], dtype=np.float64)
                        if len(values) != len(target) or not np.isfinite(values).all():
                            raise RuntimeError(f"invalid step alignment: {row_id} {method}")
                        answer_values[row_id][method] = values.tolist()
                        row_result[method] = {"status": status, "auroc": auc(target, values)}
                finally:
                    if new_scores is not None:
                        new_scores.close()
            per_row.append(row_result)

    coverage = {
        method: sum(method in payload for payload in answer_values.values())
        for method in methods
    }
    strict_coverage = {
        method: sum(
            method in answer_values[row_id] and statuses[row_id].get(method) == "OK"
            for row_id in answer_values
        ) for method in methods
    }
    rng = np.random.default_rng(2026090704)
    pairwise = {
        f"{left}_minus_{right}": _pairwise(
            answer_values, statuses, left, right, rng,
        ) for left, right in PAIRS
    }
    output = {
        "labels_accessed": True,
        "selection_frozen_before_labels": True,
        "source_label_sha256": sha(LABELS),
        "coverage": coverage,
        "strict_coverage": strict_coverage,
        "pairwise_strict": pairwise,
        "per_row": per_row,
    }
    (OUT / "EVALUATION.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "coverage": coverage,
        "strict_coverage": strict_coverage,
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

