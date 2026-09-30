"""Audit frozen pilots without refitting or changing their artifacts.

This is retrospective evaluation of already inspected development data. It
records actual imports, source hashes, source-group bootstrap and within-answer
ranking. No candidate selection or new confirmation claim is made here.
"""
from __future__ import annotations

import collections
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LIVE = Path("C:/Users/omris/TAU/hd_jlsml_v2_wt")
CELL = LIVE / "results/joint_lsml_optimization_v2/cells/prmbench_qwen3_8b.npz"
LABELS = LIVE / "results/joint_lsml_optimization_v2/labels/prmbench_qwen3_8b_labels.npz"
OUT = ROOT / "results/localization_short_cycles_audit_20260907"
DIRS = ("localization_short_cycle01", "localization_short_cycle02",
        "localization_short_cycle03_graph")
METHODS = {
    "iu": (0, "iu"), "equal": (0, "equal"),
    "joint_lambda0": (0, "joint_modelinv_lam0"),
    "joint_fixed4": (1, "joint_fixed4_modelinv_lam0"),
    "cont_fixed4": (1, "fixed4_continuous_lsml"),
    "joint_graph010": (2, "internal_joint_liu010_answer_only"),
    "joint_graph_permuted": (2, "permctl_graph_internal_joint_liu010_answer_only"),
}
PAIRS = (("iu", "equal"), ("joint_lambda0", "iu"),
         ("joint_fixed4", "iu"), ("cont_fixed4", "iu"),
         ("joint_graph010", "joint_lambda0"),
         ("joint_graph010", "joint_graph_permuted"),
         ("joint_graph010", "iu"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def auc(target, score):
    return (float(roc_auc_score(target, score))
            if len(np.unique(target)) == 2 else None)


def paired_summary(rows, left, right, seed=2026090704, draws=2000):
    chosen = [r for r in rows if r["strict"].get(left) and r["strict"].get(right)]
    groups = sorted({r["group_id"] for r in chosen})
    index = {group: [i for i, r in enumerate(chosen) if r["group_id"] == group]
             for group in groups}
    if not chosen:
        return {"answers": 0, "groups": 0}
    def metrics(indices):
        y = np.concatenate([chosen[i]["y"] for i in indices])
        values = [auc(y, np.concatenate([chosen[i]["scores"][m] for i in indices]))
                  for m in (left, right)]
        return None if any(v is None for v in values) else values
    point = metrics(range(len(chosen)))
    rng = np.random.default_rng(seed)
    differences, within_differences = [], []
    within_by_row = []
    for row in chosen:
        pair = [auc(row["y"], row["scores"][m]) for m in (left, right)]
        within_by_row.append(None if any(v is None for v in pair) else pair[0] - pair[1])
    for _ in range(draws):
        draw = rng.integers(len(groups), size=len(groups))
        indices = [i for g in draw for i in index[groups[g]]]
        pair = metrics(indices)
        if pair is not None:
            differences.append(pair[0] - pair[1])
        within_draw = [within_by_row[i] for i in indices if within_by_row[i] is not None]
        if within_draw:
            within_differences.append(float(np.mean(within_draw)))
    mixed = [r for r in chosen if len(np.unique(r["y"])) == 2]
    within = {m: float(np.mean([auc(r["y"], r["scores"][m]) for r in mixed]))
              for m in (left, right)} if mixed else {}
    return {
        "answers": len(chosen), "groups": len(groups),
        "steps": sum(len(r["y"]) for r in chosen),
        "pooled_step_auc": dict(zip((left, right), point)) if point else None,
        "observed_delta": point[0] - point[1] if point else None,
        "source_group_bootstrap_ci95": np.quantile(differences, [.025, .975]).tolist()
        if differences else None,
        "bootstrap_valid_draws": len(differences), "seed": seed,
        "within_answer_macro_auc": within, "mixed_label_answers": len(mixed),
        "within_answer_delta": within[left] - within[right] if mixed else None,
        "within_answer_source_group_ci95": np.quantile(within_differences, [.025, .975]).tolist()
        if within_differences else None,
        "within_answer_bootstrap_valid_draws": len(within_differences),
        "qualification": "retrospective pairwise available/OK fits; not confirmation",
    }


def main():
    capsule = ROOT / "local_cache/short_cycle01_code"
    sys.path.insert(0, str(capsule))
    import spectral_utils
    spectral_utils.__path__.append(str(ROOT / "spectral_utils"))
    for name in ("short_cycle_localization", "short_cycle02_localization",
                 "short_cycle03_graph_localization"):
        importlib.import_module(f"spectral_utils.{name}")
    lineage = {}
    for name, module in sorted(sys.modules.items()):
        if name.startswith("spectral_utils") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            lineage[name] = {"path": str(path), "sha256": sha(path)}
    upstream = []
    for entry in read_json(capsule / "UPSTREAM_SOURCES.json"):
        path = capsule / "spectral_utils" / entry["file"]
        actual = sha(path)
        upstream.append({**entry, "capsule_sha256": actual,
                         "matches_recorded_hash": actual == entry["sha256"]})
    snapshots = [read_json(ROOT / "results" / d / "SCORES_FROZEN.json") for d in DIRS]
    row_meta = [{r["row_id"]: r for r in s["rows"]} for s in snapshots]
    assert all(s["labels_accessed"] is False for s in snapshots)
    assert all(set(row_meta[0]) == set(m) for m in row_meta[1:])
    config = read_json(ROOT / "results" / DIRS[0] / "CONFIG.json")
    counts = collections.Counter()
    rows, replay_errors, inspected_paths = [], [], set()
    cell_metadata_hashes = {}
    with np.load(CELL, allow_pickle=False) as cell, np.load(LABELS, allow_pickle=False) as labels:
        cell_ids, label_ids = list(map(str, cell["row_ids"])), list(map(str, labels["row_ids"]))
        assert len(cell_ids) == len(set(cell_ids)) and len(label_ids) == len(set(label_ids))
        cell_pos, label_pos = {s: i for i, s in enumerate(cell_ids)}, {s: i for i, s in enumerate(label_ids)}
        for key in ("row_ids", "group_ids", "token_offsets", "step_row_offsets", "step_starts", "step_ends"):
            values = cell[key]
            cell_metadata_hashes[key] = {"dtype": str(values.dtype), "shape": list(values.shape),
                                         "sha256": hashlib.sha256(values.tobytes()).hexdigest()}
        for row_id in sorted(row_meta[0]):
            ci, li = cell_pos[row_id], label_pos[row_id]
            assert row_meta[0][row_id]["row"] == ci
            lo, hi = map(int, labels["step_flag_offsets"][[li, li + 1]])
            target = np.asarray(labels["step_error_flags"][lo:hi], dtype=int)
            assert set(target.tolist()).issubset({0, 1})
            row = {"row_id": row_id, "group_id": str(cell["group_ids"][ci]),
                   "y": target, "scores": {}, "strict": {}}
            base = ROOT / "results" / DIRS[0] / f"row_{ci}.npz"
            inspected_paths.add(base)
            with np.load(base, allow_pickle=False) as original:
                a, b = map(int, cell["step_row_offsets"][[ci, ci + 1]])
                offset = int(cell["token_offsets"][ci])
                for name in ("step_starts", "step_ends"):
                    assert np.array_equal(original[name], cell[name][a:b] - offset)
                for method, (source, key) in METHODS.items():
                    item = row_meta[source][row_id]
                    meta = item.get("report", {}).get("methods", {}).get(key, {})
                    path = ROOT / "results" / DIRS[source] / f"row_{ci}.npz"
                    if not path.exists():
                        continue
                    inspected_paths.add(path)
                    with np.load(path, allow_pickle=False) as arrays:
                        score_key = f"{key}__step"
                        if score_key not in arrays.files:
                            continue
                        score = np.asarray(arrays[score_key], dtype=float)
                        assert score.shape == target.shape and np.isfinite(score).all()
                        for name in ("step_starts", "step_ends", "window_starts", "window_ends"):
                            assert np.array_equal(arrays[name], original[name])
                        row["scores"][method] = score
                        row["strict"][method] = meta.get("status") == "OK"
                        counts[(method, str(meta.get("status")), str(meta.get("multistart_status")))] += 1
                        if source == 2 and key.startswith("internal_joint"):
                            for level in ("window", "token", "step"):
                                replay_errors.append(float(np.max(np.abs(
                                    arrays[f"joint_modelinv_lam0_replay__{level}"]
                                    - original[f"joint_modelinv_lam0__{level}"]))))
            rows.append(row)
    original_eval = read_json(ROOT / "results" / DIRS[0] / "EVALUATION.json")
    reproduction = {}
    for method, old_name in (("iu", "iu"), ("equal", "equal"), ("joint_lambda0", "joint_modelinv_lam0")):
        chosen = [r for r in rows if method in r["scores"]]
        value = auc(np.concatenate([r["y"] for r in chosen]),
                    np.concatenate([r["scores"][method] for r in chosen]))
        old = original_eval["auroc"][old_name]
        assert abs(value - old) < 1e-12
        reproduction[method] = {"recomputed": value, "reported": old, "absolute_error": abs(value-old)}
    results = {f"{a}_minus_{b}": paired_summary(rows, a, b) for a, b in PAIRS}
    for name, result in results.items():
        print(name, json.dumps(result), flush=True)
    group_counts = collections.Counter(r["group_id"] for r in rows)
    result = {
        "status": "RETROSPECTIVE_AUDIT_NOT_NEW_EXPERIMENT", "labels_accessed": True,
        "original_artifacts_modified": False, "answers": len(rows),
        "source_groups": len(group_counts),
        "duplicate_groups": {g: n for g, n in group_counts.items() if n > 1},
        "orientation_contract": config["settings"]["orientation"],
        "environment": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__},
        "cell_metadata_hashes": cell_metadata_hashes,
        "actual_imports": lineage, "capsule_upstream_checks": upstream,
        "source_hashes": {str(p): sha(p) for p in [Path(__file__), LABELS]
                          + [ROOT / "results" / d / "SCORES_FROZEN.json" for d in DIRS]},
        "inspected_array_sha256": {str(p): sha(p) for p in sorted(inspected_paths)},
        "max_lambda0_replay_error": max(replay_errors), "cycle1_reproduction": reproduction,
        "fit_status_counts": [{"method": k[0], "status": k[1], "multistart": k[2], "count": v}
                              for k, v in sorted(counts.items())],
        "paired_comparisons": results,
        "limitations": [
            "Source groups are the existing source_idx contract; broader problem duplicates need a separate audit.",
            "Imported source hashes describe current replay environment, not proof of every historical executed dependency.",
            "OK matches the historical convergence filter; this audit does not recompute Jacobian identifiability.",
            "Within-answer AUC excludes single-class answers; it is not a complete clean-answer localization metric.",
            "No ProcessBench outcomes or clean/no-error threshold are tested in these pilots.",
            "Historical feature signs are borrowed calibration, not learned independently within each answer.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Audit saved: {OUT / 'AUDIT.json'}", flush=True)


if __name__ == "__main__":
    main()
