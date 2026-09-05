"""Joint L-SML optimization v2 — label-free structure runner.

Protocol: docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md

Stages (all label-free; labels are extracted once at `load` into a separated
namespace that only the evaluator reads):

    python scripts/joint_lsml_optimization_v2/run_v2.py load
    python scripts/joint_lsml_optimization_v2/run_v2.py folds
    python scripts/joint_lsml_optimization_v2/run_v2.py structure [--cells ...] [--workers N]
    python scripts/joint_lsml_optimization_v2/run_v2.py check

`structure` fits every registered arm per (cell x outer fold) and per
(cell x outer x inner fold), scores all rows, fits Module-B reducers on the
deployed-IU substrate, and freezes everything with hashes BEFORE any label
decode.  The evaluator (evaluate_v2.py) is a separate process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from spectral_utils.feature_contract import confidence_sign_vector  # noqa: E402
from spectral_utils.fixed_application_pipelines import (  # noqa: E402
    SHARED_GLOBAL_FEATURES,
    SHARED_TOKEN_VIEWS,
    raw_token_feature_matrix,
)
from spectral_utils.joint_lsml_localization import prepare_active23  # noqa: E402
from spectral_utils.joint_lsml_v2_localization import (  # noqa: E402
    IU_ROSTER,
    LSML_ROSTER,
    DEPLOYED_IU_ROW,
    fit_v2_arms,
)
from spectral_utils.trajectory_reducer import (  # noqa: E402
    ORDERSTAT_K,
    fit_orderstat_weights,
    fit_position_bin_weights,
    reduce_with_weights,
    step_order_statistics,
    step_position_bins,
)

# The main checkout holds the telemetry caches; the sparse worktree holds code.
DATA_ROOT = Path(r"C:\Users\omris\TAU\hallucination_detection")
OUT = REPO / "results" / "joint_lsml_optimization_v2"

RETAINED_23 = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28)
PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
PB_MODELS = {"q4": "pb_qwen3_4b", "q8": "pb_qwen3_8b"}
PRM_CELL = "prmbench_qwen3_8b"
N_OUTER = 5
N_INNER = 5
SEED = 20260905
FOLD_NAMESPACE = "joint_lsml_optimization_v2"

TELEMETRY_KEYS = (
    "token_entropies", "token_spilled_energies", "token_logsumexp", "top_k_logprobs",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_roster() -> list[str]:
    cells = [f"pb_{subset}_{tag}" for subset in PB_SUBSETS for tag in PB_MODELS]
    return cells + [PRM_CELL]


# ── stage: load ──────────────────────────────────────────────────────────────

def _extract_row(row: dict) -> tuple[np.ndarray, list[tuple[int, int]]]:
    telemetry = {key: row[key] for key in TELEMETRY_KEYS}
    matrix = raw_token_feature_matrix(telemetry)
    spans = [(int(lo), int(hi)) for lo, hi in row["step_token_spans"]]
    if spans and spans[-1][1] > len(matrix):
        raise ValueError("step span exceeds token trace")
    return matrix, spans


def _write_cell(cell_id: str, entries: dict[str, dict], group_of: dict[str, str]) -> None:
    row_ids = sorted(entries)
    matrices, starts, ends, span_offsets = [], [], [], [0]
    offsets = [0]
    for row_id in row_ids:
        matrix, spans = _extract_row(entries[row_id])
        base = offsets[-1]
        matrices.append(matrix)
        offsets.append(base + len(matrix))
        for lo, hi in spans:
            starts.append(base + lo)
            ends.append(base + hi)
        span_offsets.append(len(starts))
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "cells" / f"{cell_id}.npz",
        raw=np.vstack(matrices).astype(np.float64),
        token_offsets=np.asarray(offsets, dtype=np.int64),
        row_ids=np.asarray(row_ids),
        group_ids=np.asarray([group_of[row_id] for row_id in row_ids]),
        step_starts=np.asarray(starts, dtype=np.int64),
        step_ends=np.asarray(ends, dtype=np.int64),
        step_row_offsets=np.asarray(span_offsets, dtype=np.int64),
    )


def _write_labels(cell_id: str, payload: dict[str, np.ndarray]) -> None:
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "labels" / f"{cell_id}_labels.npz", **payload)


def stage_load() -> None:
    for subset in PB_SUBSETS:
        for tag, directory in PB_MODELS.items():
            cell_id = f"pb_{subset}_{tag}"
            path = DATA_ROOT / "dataset_cache" / "repgrid" / directory / f"processbench_{subset}.pkl"
            with open(path, "rb") as handle:
                data = pickle.load(handle)
            entries, first_error, group_of = {}, {}, {}
            for key, row in data.items():
                row_id = f"{subset}::{row['id']}"
                entries[row_id] = row
                first_error[row_id] = int(row["label"])
                group_of[row_id] = f"{subset}::{row['id']}"
            _write_cell(cell_id, entries, group_of)
            row_ids = sorted(entries)
            _write_labels(cell_id, {
                "row_ids": np.asarray(row_ids),
                "first_error": np.asarray([first_error[r] for r in row_ids], dtype=np.int64),
            })
            print(f"loaded {cell_id}: {len(entries)} rows")
    path = DATA_ROOT / "dataset_cache" / "four_localization" / "prmbench_qwen3_8b_telemetry_full" / "prmbench_telemetry.pkl"
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    entries, group_of = {}, {}
    error_steps, classifications, n_steps = {}, {}, {}
    for key, row in data.items():
        row_id = str(row["idx"])
        entries[row_id] = row
        group_of[row_id] = str(row["source_idx"])
        error_steps[row_id] = [int(step) for step in row.get("error_steps", [])]
        classifications[row_id] = str(row.get("classification", ""))
        n_steps[row_id] = len(row["step_token_spans"])
    _write_cell(PRM_CELL, entries, group_of)
    row_ids = sorted(entries)
    flat_flags, flag_offsets = [], [0]
    for row_id in row_ids:
        flags = np.zeros(n_steps[row_id], dtype=np.int64)
        for step in error_steps[row_id]:
            if 0 <= step < len(flags):
                flags[step] = 1
        flat_flags.extend(flags.tolist())
        flag_offsets.append(len(flat_flags))
    _write_labels(PRM_CELL, {
        "row_ids": np.asarray(row_ids),
        "step_error_flags": np.asarray(flat_flags, dtype=np.int64),
        "step_flag_offsets": np.asarray(flag_offsets, dtype=np.int64),
        "classification": np.asarray([classifications[r] for r in row_ids]),
    })
    print(f"loaded {PRM_CELL}: {len(entries)} rows")


# ── stage: folds (label-free, deterministic) ─────────────────────────────────

def _assign(groups: list[str], strata: dict[str, str], namespace: str, n_folds: int) -> dict[str, int]:
    """Deterministic stratified round-robin over SHA-sorted groups (label-free)."""
    by_stratum: dict[str, list[str]] = {}
    for group in sorted(set(groups)):
        by_stratum.setdefault(strata[group], []).append(group)
    assignment: dict[str, int] = {}
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda g: hashlib.sha256(f"{namespace}\0{g}".encode()).hexdigest(),
        )
        for position, group in enumerate(ordered):
            assignment[group] = position % n_folds
    return assignment


def stage_folds() -> None:
    folds: dict[str, dict] = {}
    pb_groups: list[str] = []
    pb_strata: dict[str, str] = {}
    for subset in PB_SUBSETS:
        bundle = np.load(OUT / "cells" / f"pb_{subset}_q4.npz", allow_pickle=False)
        for group in bundle["group_ids"]:
            group = str(group)
            pb_groups.append(group)
            pb_strata[group] = subset
    outer = _assign(pb_groups, pb_strata, FOLD_NAMESPACE, N_OUTER)
    inner: dict[str, dict[str, int]] = {}
    for k in range(N_OUTER):
        train_groups = [g for g, fold in outer.items() if fold != k]
        inner[str(k)] = _assign(
            train_groups, pb_strata, f"{FOLD_NAMESPACE}/outer{k}/inner", N_INNER
        )
    folds["processbench"] = {"outer": outer, "inner": inner}

    bundle = np.load(OUT / "cells" / f"{PRM_CELL}.npz", allow_pickle=False)
    prm_groups = sorted({str(g) for g in bundle["group_ids"]})
    prm_strata = {g: "prmbench" for g in prm_groups}
    outer_prm = _assign(prm_groups, prm_strata, FOLD_NAMESPACE, N_OUTER)
    inner_prm: dict[str, dict[str, int]] = {}
    for k in range(N_OUTER):
        train_groups = [g for g, fold in outer_prm.items() if fold != k]
        inner_prm[str(k)] = _assign(
            train_groups, prm_strata, f"{FOLD_NAMESPACE}/outer{k}/inner", N_INNER
        )
    folds["prmbench"] = {"outer": outer_prm, "inner": inner_prm}

    payload = json.dumps(folds, sort_keys=True)
    (OUT / "folds").mkdir(parents=True, exist_ok=True)
    (OUT / "folds" / "folds.json").write_text(payload, encoding="utf-8")
    print("folds written:", hashlib.sha256(payload.encode()).hexdigest()[:16],
          "| PB groups:", len(outer), "| PRM groups:", len(outer_prm))


# ── stage: structure (the heavy label-free fits) ─────────────────────────────

def _load_cell(cell_id: str):
    bundle = np.load(OUT / "cells" / f"{cell_id}.npz", allow_pickle=False)
    return {key: bundle[key] for key in bundle.files}


def _preparation(cell, fit_row_mask):
    return prepare_active23(
        cell["raw"], cell["token_offsets"], [str(r) for r in cell["row_ids"]],
        retained_indices=list(RETAINED_23),
        confidence_signs_29=confidence_sign_vector(SHARED_GLOBAL_FEATURES),
        stream_names_29=SHARED_TOKEN_VIEWS,
        raw_feature_names_29=SHARED_GLOBAL_FEATURES,
        fit_row_mask=fit_row_mask,
    )


def _row_masks(cell, fold_map: dict[str, int], held: int) -> tuple[np.ndarray, np.ndarray]:
    groups = [str(g) for g in cell["group_ids"]]
    train = np.asarray([fold_map.get(g, -1) not in (held, -1) for g in groups], dtype=bool)
    test = np.asarray([fold_map.get(g, -1) == held for g in groups], dtype=bool)
    return train, test


def _step_scores(token_risk: np.ndarray, cell) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step B0 scores (top-10 mean + span max) and per-row detector (max token risk)."""
    starts, ends = cell["step_starts"], cell["step_ends"]
    top10 = np.empty(len(starts), dtype=np.float64)
    span_max = np.empty(len(starts), dtype=np.float64)
    for index, (lo, hi) in enumerate(zip(starts, ends)):
        values = token_risk[int(lo):int(hi)]
        values = values[np.isfinite(values)]
        if values.size == 0:
            top10[index] = span_max[index] = np.nan
            continue
        k = min(10, len(values))
        top10[index] = float(np.partition(values, -k)[-k:].mean())
        span_max[index] = float(values.max())
    offsets = cell["token_offsets"]
    detector = np.asarray([
        float(np.nanmax(token_risk[int(offsets[i]):int(offsets[i + 1])]))
        for i in range(len(offsets) - 1)
    ])
    return top10, span_max, detector


def _fit_and_score(cell_id: str, cell, mask_train: np.ndarray, *, seed: int, tag: str, out_dir: Path):
    prep = _preparation(cell, mask_train)
    fitted = fit_v2_arms(
        prep, seed=seed, cell_key=cell_id,
        domain="prmbench" if cell_id == PRM_CELL else "processbench",
    )
    arrays: dict[str, np.ndarray] = {}
    for arm, weight in fitted["weights"].items():
        risk = prep.token_risk(weight)
        top10, span_max, detector = _step_scores(risk, cell)
        arrays[f"{arm}__w"] = np.asarray(weight, dtype=np.float64)
        arrays[f"{arm}__top10"] = top10.astype(np.float32)
        arrays[f"{arm}__spanmax"] = span_max.astype(np.float32)
        arrays[f"{arm}__detector"] = detector.astype(np.float32)
    np.savez_compressed(out_dir / f"scores_{tag}.npz", **arrays)
    meta = {
        "cell": cell_id, "tag": tag, "seed": seed,
        "n_arms": len(fitted["weights"]),
        "failures": fitted["failures"],
        "fallback_events": fitted["fallback_events"],
        "internal_grouping_status": fitted["internal_grouping_status"],
        "internal_K": fitted["internal_K"],
        "gated_affinity_grouping_status": fitted["gated_affinity_grouping_status"],
        "gate_seed_std": fitted["gate_diagnostics"].get("mean_seed_std"),
        "deployed_iu_matches_grid": fitted["deployed_iu_matches_grid"],
        "row_meta": {
            arm: {k: v for k, v in meta_row.items()
                  if isinstance(v, (int, float, str, bool, list)) or v is None}
            for arm, meta_row in fitted["row_meta"].items()
        },
        "labels_accessed": False,
    }
    (out_dir / f"meta_{tag}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return fitted, prep


def _module_b(cell_id: str, cell, prep, deployed_weight, mask_train, out_dir: Path) -> None:
    risk = prep.token_risk(deployed_weight)
    matrix, lengths = step_order_statistics(risk, cell["step_starts"], cell["step_ends"])
    bins = step_position_bins(risk, cell["step_starts"], cell["step_ends"])
    step_rows = np.repeat(
        np.arange(len(cell["row_ids"])), np.diff(cell["step_row_offsets"])
    )
    train_steps = mask_train[step_rows]
    b1_weights, b1_meta = fit_orderstat_weights(matrix[train_steps], lengths[train_steps])
    b1_scores = reduce_with_weights(matrix, lengths, b1_weights)
    b2b_weights, b2b_meta = fit_position_bin_weights(bins[train_steps])
    b2b_scores = bins @ b2b_weights
    centered = matrix[train_steps] - matrix[train_steps].mean(axis=0, keepdims=True)
    try:
        centered_weights, _ = fit_orderstat_weights(centered + matrix[train_steps].mean(), lengths[train_steps])
    except Exception:
        centered_weights = np.full(ORDERSTAT_K, np.nan)
    np.savez_compressed(
        out_dir / "moduleb.npz",
        orderstats=matrix.astype(np.float32), lengths=lengths,
        position_bins=bins.astype(np.float32), step_rows=step_rows,
        b1_weights=b1_weights, b1_scores=b1_scores.astype(np.float32),
        b2b_weights=b2b_weights, b2b_scores=b2b_scores.astype(np.float32),
        centered_profile=centered_weights,
    )
    (out_dir / "moduleb_meta.json").write_text(json.dumps({
        "b1": b1_meta, "b2b": b2b_meta, "substrate": DEPLOYED_IU_ROW,
        "labels_accessed": False,
    }, indent=1), encoding="utf-8")


def _run_cell(cell_id: str) -> str:
    import torch

    torch.set_num_threads(1)
    started = time.time()
    cell = _load_cell(cell_id)
    folds = json.loads((OUT / "folds" / "folds.json").read_text(encoding="utf-8"))
    panel = "prmbench" if cell_id == PRM_CELL else "processbench"
    outer_map = folds[panel]["outer"]
    for k in range(N_OUTER):
        out_dir = OUT / "structure" / cell_id / f"outer{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "COMPLETE.json").exists():
            continue
        mask_train, _ = _row_masks(cell, outer_map, k)
        fitted, prep = _fit_and_score(
            cell_id, cell, mask_train, seed=SEED + 100 * k, tag="outer", out_dir=out_dir
        )
        if DEPLOYED_IU_ROW in fitted["weights"]:
            _module_b(cell_id, cell, prep, fitted["weights"][DEPLOYED_IU_ROW], mask_train, out_dir)
        inner_map = folds[panel]["inner"][str(k)]
        for j in range(N_INNER):
            inner_dir = out_dir / f"inner{j}"
            inner_dir.mkdir(exist_ok=True)
            if (inner_dir / "scores_inner.npz").exists():
                continue
            groups = [str(g) for g in cell["group_ids"]]
            inner_train = np.asarray([
                outer_map.get(g, -1) not in (k, -1) and inner_map.get(g, -1) != j
                for g in groups
            ], dtype=bool)
            _fit_and_score(
                cell_id, cell, inner_train,
                seed=SEED + 100 * k + 10 * j + 1, tag="inner", out_dir=inner_dir,
            )
        manifest = {
            path.name: _sha(path)
            for path in sorted(out_dir.rglob("*"))
            if path.is_file() and path.name != "MANIFEST.json"
        }
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        (out_dir / "COMPLETE.json").write_text(json.dumps({
            "cell": cell_id, "outer": k, "elapsed_s": round(time.time() - started, 1),
        }), encoding="utf-8")
        print(f"[{cell_id}] outer{k} complete ({time.time() - started:.0f}s)", flush=True)
    return cell_id


def stage_structure(cells: list[str], workers: int) -> None:
    (OUT / "structure").mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        for cell_id in cells:
            _run_cell(cell_id)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_cell, cell_id): cell_id for cell_id in cells}
            for future in as_completed(futures):
                print("cell complete:", future.result(), flush=True)


# ── stage: check ─────────────────────────────────────────────────────────────

def stage_check() -> None:
    problems = []
    for cell_id in cell_roster():
        for k in range(N_OUTER):
            out_dir = OUT / "structure" / cell_id / f"outer{k}"
            if not (out_dir / "COMPLETE.json").exists():
                problems.append(f"{cell_id}/outer{k}: incomplete")
                continue
            manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
            for name, expected in manifest.items():
                matches = list(out_dir.rglob(name))
                if not matches:
                    problems.append(f"{cell_id}/outer{k}/{name}: missing")
                elif _sha(matches[0]) != expected:
                    problems.append(f"{cell_id}/outer{k}/{name}: hash drift")
    if problems:
        print("CHECK FAILED")
        for problem in problems[:40]:
            print(" -", problem)
        raise SystemExit(1)
    print(f"CHECK PASSED: {len(cell_roster())} cells x {N_OUTER} outer folds frozen and hash-stable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("load", "folds", "structure", "check"))
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "load":
        stage_load()
    elif args.stage == "folds":
        stage_folds()
    elif args.stage == "structure":
        stage_structure(args.cells or cell_roster(), args.workers)
    else:
        stage_check()


if __name__ == "__main__":
    main()
