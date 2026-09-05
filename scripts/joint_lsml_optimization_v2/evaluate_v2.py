"""Joint L-SML optimization v2 — label-stage evaluator.

Runs ONLY after `run_v2.py structure` + `check` have frozen every score
artifact.  This is the single process allowed to read
results/joint_lsml_optimization_v2/labels/.  It fits nothing on telemetry:
model selection picks among frozen arrays; the only label-consuming fits are
the pre-registered PB thresholds, Module-B alpha selection, and the Module-B
LR competitor (all disclosed development).

Outputs: results/joint_lsml_optimization_v2/evaluation/
    inner_selection.json      per panel x outer fold x family: selected config
    headline.json             point estimates + paired bootstrap CIs + gates
    per_cell.csv              PB per-cell F1 per reported arm
    moduleb.json              Module-B readouts + tail-vs-bulk profiles
    REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from spectral_utils.joint_lsml_v2_localization import (  # noqa: E402
    DEPLOYED_IU_ROW,
    DEPLOYED_UPCR_PORT_ROW,
    EQUAL_ALL23_METHOD,
    EQUAL_FAMILY_METHOD,
    IU_ROSTER,
    LSML_ROSTER,
    SUCCESSOR_S1,
    SUCCESSOR_S2,
)
from spectral_utils.trajectory_reducer import (  # noqa: E402
    blend_step_scores,
    fit_lr_orderstats,
    score_lr_orderstats,
)

OUT = REPO / "results" / "joint_lsml_optimization_v2"
EVAL = OUT / "evaluation"
N_OUTER = 5
N_INNER = 5
PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
PB_MODELS = ("q4", "q8")
PRM_CELL = "prmbench_qwen3_8b"
IU_IDS = tuple(row_id for row_id, _ in IU_ROSTER)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PB_BOOT = 2000
PRM_BOOT = 10000
BOOT_SEED = 20260906

PB_FLOOR = 0.010
PRM_FLOOR = 0.005
PB_NONINF = -0.005
PRM_NONINF = -0.0025


# ── loading helpers ──────────────────────────────────────────────────────────

def _cell(cell_id: str) -> dict[str, np.ndarray]:
    bundle = np.load(OUT / "cells" / f"{cell_id}.npz", allow_pickle=False)
    return {key: bundle[key] for key in bundle.files}


def _labels(cell_id: str) -> dict[str, np.ndarray]:
    bundle = np.load(OUT / "labels" / f"{cell_id}_labels.npz", allow_pickle=False)
    return {key: bundle[key] for key in bundle.files}


def _scores(cell_id: str, outer: int, inner: int | None = None) -> dict[str, np.ndarray]:
    base = OUT / "structure" / cell_id / f"outer{outer}"
    path = base / "scores_outer.npz" if inner is None else base / f"inner{inner}" / "scores_inner.npz"
    bundle = np.load(path, allow_pickle=False)
    scores = {key: bundle[key] for key in bundle.files}
    if inner is None:
        continuity = base / "scores_continuity.npz"
        if continuity.exists():
            extra = np.load(continuity, allow_pickle=False)
            scores.update({key: extra[key] for key in extra.files})
    return scores


def _folds() -> dict:
    return json.loads((OUT / "folds" / "folds.json").read_text(encoding="utf-8"))


def _arm_ids(scores: dict[str, np.ndarray]) -> list[str]:
    return sorted({key.rsplit("__", 1)[0] for key in scores})


def _masks(cell, fold_map, held):
    groups = [str(g) for g in cell["group_ids"]]
    train = np.asarray([fold_map.get(g, -1) not in (held, -1) for g in groups], dtype=bool)
    test = np.asarray([fold_map.get(g, -1) == held for g in groups], dtype=bool)
    return train, test


def _step_rows(cell) -> np.ndarray:
    return np.repeat(np.arange(len(cell["row_ids"])), np.diff(cell["step_row_offsets"]))


# ── ProcessBench mechanics ───────────────────────────────────────────────────

def _pb_row_features(cell, scores: dict[str, np.ndarray], arm: str):
    """Per-row (detector, locator step index) from frozen arrays."""
    detector = np.asarray(scores[f"{arm}__detector"], dtype=np.float64)
    top10 = np.asarray(scores[f"{arm}__top10"], dtype=np.float64)
    offsets = cell["step_row_offsets"]
    locator = np.asarray([
        int(np.nanargmax(top10[offsets[i]:offsets[i + 1]])) if offsets[i + 1] > offsets[i] else 0
        for i in range(len(offsets) - 1)
    ])
    return detector, locator


def _pb_f1(hit_error: np.ndarray, is_clean: np.ndarray, predicted_clean: np.ndarray,
           cell_of_row: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Equal-subset macro-F1 over the 8 (subset x model) cells."""
    w = np.ones(len(is_clean)) if weights is None else weights
    values = []
    for cell_index in range(int(cell_of_row.max()) + 1):
        rows = cell_of_row == cell_index
        err = rows & ~is_clean
        cln = rows & is_clean
        wa = float(np.sum(w[err] * hit_error[err])) / max(float(np.sum(w[err])), 1e-12)
        wc = float(np.sum(w[cln] * predicted_clean[cln])) / max(float(np.sum(w[cln])), 1e-12)
        values.append(0.0 if wa + wc <= 0 else 2.0 * wa * wc / (wa + wc))
    return float(np.mean(values))


def _pb_threshold(detector_train, hit_if_flag_train, clean_train, cell_train, *,
                  weights=None) -> float:
    """Grid-fit one pooled threshold maximizing train macro-F1 (lowest argmax)."""
    grid = np.unique(np.quantile(detector_train, np.linspace(0.01, 0.99, 99)))
    best_tau, best_value = float(grid[0]), -1.0
    for tau in grid:
        flagged = detector_train >= tau
        f1 = _pb_f1(hit_if_flag_train & flagged, clean_train, ~flagged, cell_train, weights)
        if f1 > best_value + 1e-12:
            best_value, best_tau = f1, float(tau)
    return best_tau


class PBPanel:
    """Assembled OOF per-row features for one arm across all 8 cells."""

    def __init__(self, arm: str, folds: dict, cells: dict, labels: dict,
                 selected_by_fold: dict[int, str] | None = None):
        rows = []
        for subset_index, subset in enumerate(PB_SUBSETS):
            for model_index, tag in enumerate(PB_MODELS):
                cell_id = f"pb_{subset}_{tag}"
                cell = cells[cell_id]
                label = labels[cell_id]
                first_error = label["first_error"]
                fold_map = folds["processbench"]["outer"]
                cell_code = subset_index * len(PB_MODELS) + model_index
                for k in range(N_OUTER):
                    _, test = _masks(cell, fold_map, k)
                    arm_k = arm if selected_by_fold is None else selected_by_fold[k]
                    scores = _scores(cell_id, k)
                    detector, locator = _pb_row_features(cell, scores, arm_k)
                    for row_index in np.flatnonzero(test):
                        rows.append((
                            str(cell["group_ids"][row_index]), cell_code, k,
                            detector[row_index],
                            bool(first_error[row_index] == -1),
                            bool(locator[row_index] == first_error[row_index]),
                        ))
        self.group = np.asarray([row[0] for row in rows])
        self.cell = np.asarray([row[1] for row in rows], dtype=np.int64)
        self.fold = np.asarray([row[2] for row in rows], dtype=np.int64)
        self.detector = np.asarray([row[3] for row in rows], dtype=np.float64)
        self.clean = np.asarray([row[4] for row in rows], dtype=bool)
        self.hit = np.asarray([row[5] for row in rows], dtype=bool)

    def crossfit_macro_f1(self, weights: np.ndarray | None = None) -> tuple[float, dict]:
        hit_all = np.zeros(len(self.detector), dtype=bool)
        clean_pred = np.zeros(len(self.detector), dtype=bool)
        activation = {}
        for k in range(N_OUTER):
            train = self.fold != k
            test = ~train
            tau = _pb_threshold(
                self.detector[train], self.hit[train], self.clean[train], self.cell[train],
                weights=None if weights is None else weights[train],
            )
            flagged = self.detector[test] >= tau
            hit_all[test] = self.hit[test] & flagged
            clean_pred[test] = ~flagged
            for cell_index in np.unique(self.cell):
                err = test & (self.cell == cell_index) & ~self.clean
                if err.any():
                    activation[(int(cell_index), k)] = float(np.mean(self.detector[err] >= tau))
        f1 = _pb_f1(hit_all, self.clean, clean_pred, self.cell, weights)
        return f1, {"activation": activation}


def _pb_paired_bootstrap(panel_a: PBPanel, panel_b: PBPanel, *, n_boot: int, seed: int):
    """Percentile CI of macro-F1(A) - macro-F1(B) with in-replicate threshold refit."""
    assert np.array_equal(panel_a.group, panel_b.group)
    groups = panel_a.group
    unique = np.unique(groups)
    index_of = {g: i for i, g in enumerate(unique)}
    group_index = np.asarray([index_of[g] for g in groups])
    subset_of_group = np.zeros(len(unique), dtype=np.int64)
    for g, cell_code in zip(group_index, panel_a.cell):
        subset_of_group[g] = cell_code // len(PB_MODELS)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for draw in range(n_boot):
        counts = np.zeros(len(unique))
        for subset_index in range(len(PB_SUBSETS)):
            members = np.flatnonzero(subset_of_group == subset_index)
            sampled = rng.choice(members, size=len(members), replace=True)
            np.add.at(counts, sampled, 1.0)
        weights = counts[group_index]
        keep = weights > 0
        f1_a, _ = _subset_crossfit(panel_a, weights, keep)
        f1_b, _ = _subset_crossfit(panel_b, weights, keep)
        deltas[draw] = f1_a - f1_b
    return float(np.mean(deltas)), (
        float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
    )


def _subset_crossfit(panel: PBPanel, weights: np.ndarray, keep: np.ndarray):
    hit_all = np.zeros(len(panel.detector), dtype=bool)
    clean_pred = np.zeros(len(panel.detector), dtype=bool)
    for k in range(N_OUTER):
        train = (panel.fold != k) & keep
        test = (panel.fold == k) & keep
        if not train.any() or not test.any():
            continue
        tau = _pb_threshold(
            panel.detector[train], panel.hit[train], panel.clean[train], panel.cell[train],
            weights=weights[train],
        )
        flagged = panel.detector[test] >= tau
        hit_all[test] = panel.hit[test] & flagged
        clean_pred[test] = ~flagged
    f1 = _pb_f1(hit_all[keep], panel.clean[keep], clean_pred[keep], panel.cell[keep], weights[keep])
    return f1, None


# ── PRMBench mechanics ───────────────────────────────────────────────────────

def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    sorted_scores = scores[order]
    lo = 0
    while lo < len(sorted_scores):
        hi = lo
        while hi + 1 < len(sorted_scores) and sorted_scores[hi + 1] == sorted_scores[lo]:
            hi += 1
        if hi > lo:
            ranks[order[lo:hi + 1]] = (lo + hi + 2) / 2.0
        lo = hi + 1
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class PRMPanel:
    """Assembled OOF per-step scores for one arm (or per-fold selection)."""

    def __init__(self, folds, cell, labels, score_key: str = "spanmax",
                 arm: str | None = None, selected_by_fold: dict[int, str] | None = None,
                 direct_scores_by_fold: dict[int, np.ndarray] | None = None):
        step_rows = _step_rows(cell)
        flags = labels["step_error_flags"]
        fold_map = folds["prmbench"]["outer"]
        self.scores = np.full(len(flags), np.nan)
        self.flags = np.asarray(flags, dtype=np.int64)
        self.group = np.asarray([str(g) for g in cell["group_ids"]])[step_rows]
        for k in range(N_OUTER):
            _, test = _masks(cell, fold_map, k)
            test_steps = test[step_rows]
            if direct_scores_by_fold is not None:
                self.scores[test_steps] = direct_scores_by_fold[k][test_steps]
            else:
                arm_k = arm if selected_by_fold is None else selected_by_fold[k]
                scores = _scores(PRM_CELL, k)
                self.scores[test_steps] = np.asarray(
                    scores[f"{arm_k}__{score_key}"], dtype=np.float64
                )[test_steps]
        self.valid = np.isfinite(self.scores)

    def auroc(self) -> float:
        return _auroc(self.flags[self.valid], self.scores[self.valid])


def _prm_paired_bootstrap(panel_a: PRMPanel, panel_b: PRMPanel, *, n_boot: int, seed: int):
    valid = panel_a.valid & panel_b.valid
    flags = panel_a.flags[valid]
    a = panel_a.scores[valid]
    b = panel_b.scores[valid]
    groups = panel_a.group[valid]
    unique, group_index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    step_lists = [np.flatnonzero(group_index == g) for g in range(len(unique))]
    for draw in range(n_boot):
        sampled_groups = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([step_lists[g] for g in sampled_groups])
        deltas[draw] = _auroc(flags[rows], a[rows]) - _auroc(flags[rows], b[rows])
    return float(np.mean(deltas)), (
        float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
    )


# ── inner selection ──────────────────────────────────────────────────────────

def _inner_select(folds, cells, labels) -> dict:
    """Per panel x outer fold x family: argmax of 5-inner-fold mean metric."""
    selection = {"processbench": {}, "prmbench": {}}
    families = {"lsml": list(LSML_ROSTER), "iu": list(IU_IDS)}

    # PRMBench: step AUROC on inner-val steps
    cell = cells[PRM_CELL]
    label = labels[PRM_CELL]
    step_rows = _step_rows(cell)
    flags = label["step_error_flags"]
    outer_map = _folds()["prmbench"]["outer"]
    for k in range(N_OUTER):
        inner_map = folds["prmbench"]["inner"][str(k)]
        means: dict[str, list[float]] = {}
        for j in range(N_INNER):
            scores = _scores(PRM_CELL, k, inner=j)
            groups = [str(g) for g in cell["group_ids"]]
            val_rows = np.asarray([
                outer_map.get(g, -1) != k and inner_map.get(g, -1) == j for g in groups
            ], dtype=bool)
            val_steps = val_rows[step_rows]
            for family, roster in families.items():
                for config in roster:
                    key = f"{config}__spanmax"
                    if key not in scores:
                        continue
                    value = _auroc(flags[val_steps], np.asarray(scores[key], float)[val_steps])
                    means.setdefault(config, []).append(value)
        selection["prmbench"][str(k)] = {
            family: max(
                (config for config in roster if config in means),
                key=lambda c: (np.nanmean(means[c]), -roster.index(c)),
            )
            for family, roster in families.items()
        }

    # ProcessBench: inner-cross-fitted macro-F1 over the 8 cells
    pb_outer = folds["processbench"]["outer"]
    for k in range(N_OUTER):
        inner_map = folds["processbench"]["inner"][str(k)]
        means = {}
        for j in range(N_INNER):
            rows = {"detector": [], "clean": [], "hit": [], "cell": [], "split": []}
            per_config: dict[str, dict[str, list]] = {}
            for subset_index, subset in enumerate(PB_SUBSETS):
                for model_index, tag in enumerate(PB_MODELS):
                    cell_id = f"pb_{subset}_{tag}"
                    cell_pb = cells[cell_id]
                    first_error = labels[cell_id]["first_error"]
                    groups = [str(g) for g in cell_pb["group_ids"]]
                    in_outer_train = np.asarray([pb_outer.get(g, -1) not in (k, -1) for g in groups], bool)
                    in_val = in_outer_train & np.asarray([inner_map.get(g, -1) == j for g in groups], bool)
                    in_tr = in_outer_train & ~in_val
                    scores = _scores(cell_id, k, inner=j)
                    cell_code = subset_index * len(PB_MODELS) + model_index
                    for family, roster in families.items():
                        for config in roster:
                            if f"{config}__detector" not in scores:
                                continue
                            detector, locator = _pb_row_features(cell_pb, scores, config)
                            bucket = per_config.setdefault(config, {
                                "detector": [], "clean": [], "hit": [], "cell": [], "is_val": [],
                            })
                            for mask, is_val in ((in_tr, False), (in_val, True)):
                                for row_index in np.flatnonzero(mask):
                                    bucket["detector"].append(detector[row_index])
                                    bucket["clean"].append(first_error[row_index] == -1)
                                    bucket["hit"].append(locator[row_index] == first_error[row_index])
                                    bucket["cell"].append(cell_code)
                                    bucket["is_val"].append(is_val)
            for config, bucket in per_config.items():
                detector = np.asarray(bucket["detector"], float)
                clean = np.asarray(bucket["clean"], bool)
                hit = np.asarray(bucket["hit"], bool)
                cell_code = np.asarray(bucket["cell"], int)
                is_val = np.asarray(bucket["is_val"], bool)
                tau = _pb_threshold(detector[~is_val], hit[~is_val], clean[~is_val], cell_code[~is_val])
                flagged = detector[is_val] >= tau
                f1 = _pb_f1(hit[is_val] & flagged, clean[is_val], ~flagged, cell_code[is_val])
                means.setdefault(config, []).append(f1)
        selection["processbench"][str(k)] = {
            family: max(
                (config for config in roster if config in means),
                key=lambda c: (np.nanmean(means[c]), -roster.index(c)),
            )
            for family, roster in families.items()
        }
    return selection


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    folds = _folds()
    cells = {cell_id: _cell(cell_id) for cell_id in
             [f"pb_{s}_{t}" for s in PB_SUBSETS for t in PB_MODELS] + [PRM_CELL]}
    labels = {cell_id: _labels(cell_id) for cell_id in cells}

    selection = _inner_select(folds, cells, labels)
    (EVAL / "inner_selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    print("inner selection:", json.dumps(selection["prmbench"]["0"]))

    headline: dict[str, dict] = {"prmbench": {}, "processbench": {}, "gates": {}}

    # PRMBench arms
    prm_named = {
        "tuned_lsml": {int(k): selection["prmbench"][k]["lsml"] for k in selection["prmbench"]},
        "tuned_iu": {int(k): selection["prmbench"][k]["iu"] for k in selection["prmbench"]},
    }
    prm_panels = {
        name: PRMPanel(folds, cells[PRM_CELL], labels[PRM_CELL], selected_by_fold=by_fold)
        for name, by_fold in prm_named.items()
    }
    for arm in (SUCCESSOR_S1, SUCCESSOR_S2, DEPLOYED_IU_ROW, DEPLOYED_UPCR_PORT_ROW,
                EQUAL_ALL23_METHOD, EQUAL_FAMILY_METHOD, "prov5_cont",
                "fixed_family_cont_unguarded",
                "permctl_gate_prov5_cont", "permctl_graph_internal_joint_liu010"):
        prm_panels[arm] = PRMPanel(folds, cells[PRM_CELL], labels[PRM_CELL], arm=arm)
    for name, panel in prm_panels.items():
        headline["prmbench"][name] = {"auroc": panel.auroc()}
    for name, reference in (
        ("tuned_lsml_vs_tuned_iu", ("tuned_lsml", "tuned_iu")),
        ("s1_vs_deployed_iu", (SUCCESSOR_S1, DEPLOYED_IU_ROW)),
        ("s2_vs_deployed_iu", (SUCCESSOR_S2, DEPLOYED_IU_ROW)),
    ):
        mean, ci = _prm_paired_bootstrap(
            prm_panels[reference[0]], prm_panels[reference[1]],
            n_boot=PRM_BOOT, seed=BOOT_SEED,
        )
        headline["prmbench"][name] = {"delta": mean, "ci95": ci}

    # ProcessBench arms
    pb_named = {
        "tuned_lsml": {int(k): selection["processbench"][k]["lsml"] for k in selection["processbench"]},
        "tuned_iu": {int(k): selection["processbench"][k]["iu"] for k in selection["processbench"]},
    }
    pb_panels = {
        name: PBPanel("", folds, cells, labels, selected_by_fold=by_fold)
        for name, by_fold in pb_named.items()
    }
    for arm in (SUCCESSOR_S1, SUCCESSOR_S2, DEPLOYED_IU_ROW, DEPLOYED_UPCR_PORT_ROW,
                EQUAL_ALL23_METHOD, EQUAL_FAMILY_METHOD, "prov5_cont",
                "fixed_family_cont_unguarded"):
        pb_panels[arm] = PBPanel(arm, folds, cells, labels)
    for name, panel in pb_panels.items():
        f1, extras = panel.crossfit_macro_f1()
        headline["processbench"][name] = {"macro_f1": f1, **extras}
    for name, reference in (
        ("tuned_lsml_vs_tuned_iu", ("tuned_lsml", "tuned_iu")),
        ("s1_vs_deployed_iu", (SUCCESSOR_S1, DEPLOYED_IU_ROW)),
        ("s2_vs_deployed_iu", (SUCCESSOR_S2, DEPLOYED_IU_ROW)),
    ):
        mean, ci = _pb_paired_bootstrap(
            pb_panels[reference[0]], pb_panels[reference[1]],
            n_boot=PB_BOOT, seed=BOOT_SEED,
        )
        headline["processbench"][name] = {"delta": mean, "ci95": ci}

    # Development + promotion gates (Section 7)
    def _gate(delta_entry, floor):
        lo, hi = delta_entry["ci95"]
        if lo > 0 and delta_entry["delta"] >= floor:
            return "SUPPORT"
        if hi < 0:
            return "HARM"
        return "NULL"

    headline["gates"]["development_pb"] = _gate(headline["processbench"]["tuned_lsml_vs_tuned_iu"], PB_FLOOR)
    headline["gates"]["development_prm"] = _gate(headline["prmbench"]["tuned_lsml_vs_tuned_iu"], PRM_FLOOR)
    for successor, key in ((SUCCESSOR_S1, "s1"), (SUCCESSOR_S2, "s2")):
        pb = headline["processbench"][f"{key}_vs_deployed_iu"]["ci95"]
        prm = headline["prmbench"][f"{key}_vs_deployed_iu"]["ci95"]
        superiority = (pb[0] > 0) + (prm[0] > 0)
        noninferior = (pb[0] > PB_NONINF) and (prm[0] > PRM_NONINF)
        headline["gates"][f"{key}_promotion"] = (
            "PROMOTE" if superiority >= 1 and noninferior else "NOT_PROMOTED"
        )

    (EVAL / "headline.json").write_text(json.dumps(headline, indent=1), encoding="utf-8")
    print(json.dumps(headline["gates"], indent=1))

    # ── Module B (PRMB primary; PB secondary handled in report step) ─────────
    moduleb: dict[str, dict] = {}
    cell = cells[PRM_CELL]
    label = labels[PRM_CELL]
    step_rows = _step_rows(cell)
    fold_map = folds["prmbench"]["outer"]
    flags = label["step_error_flags"]

    def _assemble(fold_scores: dict[int, np.ndarray]) -> PRMPanel:
        return PRMPanel(folds, cell, label, direct_scores_by_fold=fold_scores)

    b_scores: dict[str, dict[int, np.ndarray]] = {"b0": {}, "b1": {}, "b2a": {}, "b2b": {}, "b3": {}}
    profiles: dict[str, list] = {"b1": [], "b3": [], "centered": []}
    for k in range(N_OUTER):
        bundle = np.load(OUT / "structure" / PRM_CELL / f"outer{k}" / "moduleb.npz", allow_pickle=False)
        matrix = np.asarray(bundle["orderstats"], dtype=np.float64)
        lengths = np.asarray(bundle["lengths"], dtype=np.int64)
        train_rows, _ = _masks(cell, fold_map, k)
        train_steps = train_rows[step_rows]
        b_scores["b0"][k] = matrix[:, 0]  # span max == top-1 order statistic
        b_scores["b1"][k] = np.asarray(bundle["b1_scores"], dtype=np.float64)
        b_scores["b2b"][k] = np.asarray(bundle["b2b_scores"], dtype=np.float64)
        profiles["b1"].append(np.asarray(bundle["b1_weights"]).tolist())
        profiles["centered"].append(np.asarray(bundle["centered_profile"]).tolist())
        # B2a: alpha by inner-mean AUROC on outer-train steps
        inner_map = folds["prmbench"]["inner"][str(k)]
        alpha_means = []
        for alpha in ALPHA_GRID:
            blended = blend_step_scores(matrix, lengths, alpha)
            values = []
            for j in range(N_INNER):
                groups = [str(g) for g in cell["group_ids"]]
                val_rows = np.asarray([
                    fold_map.get(g, -1) != k and inner_map.get(g, -1) == j for g in groups
                ], dtype=bool)
                val_steps = val_rows[step_rows]
                values.append(_auroc(flags[val_steps], blended[val_steps]))
            alpha_means.append(np.nanmean(values))
        best_alpha = float(ALPHA_GRID[int(np.nanargmax(alpha_means))])
        b_scores["b2a"][k] = blend_step_scores(matrix, lengths, best_alpha)
        moduleb.setdefault("b2a_alpha", {})[str(k)] = best_alpha
        # B3: LR on outer-train steps (official flags)
        model, lr_meta = fit_lr_orderstats(
            matrix[train_steps], lengths[train_steps], flags[train_steps], seed=BOOT_SEED + k
        )
        b_scores["b3"][k] = score_lr_orderstats(model, matrix)
        profiles["b3"].append(lr_meta["coefficient_profile"])

    panels_b = {name: _assemble(scores) for name, scores in b_scores.items()}
    for name, panel in panels_b.items():
        moduleb.setdefault("auroc", {})[name] = panel.auroc()
    for name in ("b1", "b2a", "b2b", "b3"):
        mean, ci = _prm_paired_bootstrap(panels_b[name], panels_b["b0"], n_boot=PRM_BOOT, seed=BOOT_SEED + 7)
        moduleb.setdefault("vs_b0_descriptive", {})[name] = {"delta": mean, "ci95": ci}

    # ── Amendment R1: the 3x3 substrate x trajectory-fuser grid ──────────────
    # Primary Module-B contrast: inner-CV-selected best of the 9 combos vs the
    # frozen top-10-mean control on the SAME substrate (PRMB primary).
    grid_combos = [f"{s}__{f}" for s in ("iu_c2_s25_l2_exoff", "internal_cont", "internal_joint")
                   for f in ("sml", "iu", "joint")]
    grid_fold_scores: dict[str, dict[int, np.ndarray]] = {}
    b0_fold_scores: dict[str, dict[int, np.ndarray]] = {}
    grid_status: dict[str, dict[int, str]] = {}
    for k in range(N_OUTER):
        path = OUT / "structure" / PRM_CELL / f"outer{k}" / "moduleb_grid.npz"
        if not path.exists():
            continue
        bundle = np.load(path, allow_pickle=False)
        for combo in grid_combos:
            key = f"{combo}__scores"
            if key in bundle.files:
                grid_fold_scores.setdefault(combo, {})[k] = np.asarray(bundle[key], float)
                grid_status.setdefault(combo, {})[k] = "OK"
            else:
                grid_status.setdefault(combo, {})[k] = "MISSING_OR_BLOCKED"
            substrate = combo.rsplit("__", 1)[0]
            b0_key = f"{substrate}__b0"
            if b0_key in bundle.files:
                b0_fold_scores.setdefault(substrate, {})[k] = np.asarray(bundle[b0_key], float)
    inner_maps = folds["prmbench"]["inner"]
    combo_selection: dict[str, str] = {}
    for k in range(N_OUTER):
        candidate_means = {}
        for combo in grid_combos:
            if k not in grid_fold_scores.get(combo, {}):
                continue
            scores = grid_fold_scores[combo][k]
            values = []
            for j in range(N_INNER):
                groups = [str(g) for g in cell["group_ids"]]
                val_rows = np.asarray([
                    fold_map.get(g, -1) != k and inner_maps[str(k)].get(g, -1) == j
                    for g in groups
                ], dtype=bool)
                val_steps = val_rows[step_rows]
                values.append(_auroc(flags[val_steps], scores[val_steps]))
            candidate_means[combo] = float(np.nanmean(values))
        if candidate_means:
            combo_selection[str(k)] = max(
                candidate_means, key=lambda c: (candidate_means[c], -grid_combos.index(c))
            )
    if len(combo_selection) == N_OUTER:
        winner_scores = {k: grid_fold_scores[combo_selection[str(k)]][k] for k in range(N_OUTER)}
        winner_b0 = {
            k: b0_fold_scores[combo_selection[str(k)].rsplit("__", 1)[0]][k]
            for k in range(N_OUTER)
        }
        panel_winner = _assemble(winner_scores)
        panel_b0_same = _assemble(winner_b0)
        mean, ci = _prm_paired_bootstrap(panel_winner, panel_b0_same, n_boot=PRM_BOOT, seed=BOOT_SEED + 11)
        moduleb["grid_primary"] = {
            "selected_by_fold": combo_selection,
            "winner_auroc": panel_winner.auroc(),
            "b0_same_substrate_auroc": panel_b0_same.auroc(),
            "delta": mean, "ci95": ci,
            "gate": "SUPPORT" if ci[0] > 0 else ("HARM" if ci[1] < 0 else "NULL"),
        }
    grid_descriptive = {}
    for combo in grid_combos:
        by_fold = grid_fold_scores.get(combo, {})
        if len(by_fold) == N_OUTER:
            grid_descriptive[combo] = _assemble(by_fold).auroc()
    moduleb["grid_descriptive_auroc"] = grid_descriptive
    moduleb["grid_status"] = {combo: status for combo, status in grid_status.items()}

    moduleb["profiles"] = profiles
    (EVAL / "moduleb.json").write_text(json.dumps(moduleb, indent=1), encoding="utf-8")
    print("module B primary:", json.dumps(moduleb.get("grid_primary", {}), indent=1)[:400])


if __name__ == "__main__":
    main()
