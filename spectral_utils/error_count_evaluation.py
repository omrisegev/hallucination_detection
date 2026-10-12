"""Sealed, source-paired evaluation of label-free per-answer error counts.

Estimator fitting belongs elsewhere. This module first authenticates frozen
predictions, then loads labels for evaluation only. Count failures are -1 and
remain outside every metric denominator. All arms share each bank's ranking.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import time

import numpy as np
from .advisor_controls import f1, metrics, top_flags
from .advisor_statistics import answer_auc

ARMS = ("LOCAL_MOM_MEAN", "A_MOM_MEAN", "A_MOM_MEDIAN", "B_TRIPLE_MEAN",
        "B_TRIPLE_MEDIAN", "D_ZERO_MEAN", "D_ZERO_MEDIAN")
CONTROLS = ("Q80", "R2_COUNT", "ENTROPY_GATE_Q80", "LEGACY_MOM")
METHODS = ("IU_PCR", "LSML_CONT")
MULTIPLICITY = 7
ALPHA = .05 / MULTIPLICITY
LENGTH_BINS = ((1, 5), (6, 10), (11, 20), (21, None))
TYPE_CODES = {"redundency": "NR", "circular": "NCL", "counterfactual": "ES",
              "step_contradiction": "SC", "domain_inconsistency": "DC",
              "confidence": "CI", "missing_condition": "PS", "deception": "DR",
              "multi_solutions": "MS"}
CATEGORIES = {"simplicity": ("redundency", "circular"),
              "soundness": ("counterfactual", "step_contradiction", "domain_inconsistency", "confidence"),
              "sensitivity": ("missing_condition", "deception", "multi_solutions")}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, names)
        writer.writeheader()
        writer.writerows(clean(rows))


def validate_prediction_seal(package):
    """Reject missing, changed or unlisted prediction files before label access."""
    package = Path(package).resolve()
    seal_path = package / "PREDICTION_SEAL.json"
    seal = json.loads(seal_path.read_text())
    if seal.get("status") not in ("FROZEN_BEFORE_EVALUATION", "SEALED", "FROZEN", "COMPLETE"):
        raise ValueError("Prediction seal does not declare a completed freeze")
    files = seal.get("files", seal.get("predictions", seal.get("artifacts")))
    if isinstance(files, list):
        files = {item["path"]: item for item in files}
    if not isinstance(files, dict) or not files:
        raise ValueError("Prediction seal has no file hashes")
    verified = {}
    for relative, specification in files.items():
        target = (package / relative).resolve()
        if not target.is_relative_to(package):
            raise ValueError("Prediction seal file escapes its package")
        expected = specification if isinstance(specification, str) else specification.get("sha256")
        if not expected or sha256(target) != expected:
            raise ValueError(f"Prediction seal mismatch: {relative}")
        verified[target.relative_to(package).as_posix()] = expected
    for bank in ("N16", "N56"):
        if f"predictions/{bank}.npz" not in verified:
            raise ValueError(f"Unsealed {bank} prediction bank")
    repository = package.parents[1]
    for relative, expected in seal.get("code_sha256", {}).items():
        if sha256(repository/relative) != expected:
            raise ValueError(f"Sealed code identity mismatch: {relative}")
    if "protocol_sha256" in seal and sha256(package/"PROTOCOL.json") != seal["protocol_sha256"]:
        raise ValueError("Sealed protocol identity mismatch")
    return {"seal_sha256": sha256(seal_path), "verified_files": verified,
            "declared_status": seal["status"]}


@dataclass
class Predictions:
    bank: str
    idx: np.ndarray
    offsets: np.ndarray
    scores: dict
    ranks: dict
    rank_coverage: dict
    counts: dict
    method_counts: dict
    group: np.ndarray
    fold: np.ndarray


def load_predictions(path, bank):
    """Inspect predictions without importing any evaluation labels."""
    with np.load(path, allow_pickle=False) as archive:
        if any(key in archive.files for key in ("y", "label", "labels", "classification", "true_count")):
            raise ValueError("Prediction bank contains evaluation labels")
        idx = archive["idx"].astype(str)
        off = archive["offsets"].astype(np.int64)
        if len(set(idx)) != len(idx) or len(off) != len(idx) + 1 or off[0] != 0 or np.any(np.diff(off) <= 0):
            raise ValueError("Invalid prediction answer identity/offsets")
        group = archive["group"].astype(str)
        fold = archive["fold"].copy()
        if group.shape != idx.shape or fold.shape != idx.shape or fold.dtype.kind not in "iu" or np.any((fold < 0) | (fold > 4)):
            raise ValueError("Invalid prediction group/fold identities")
        group_folds = {}
        for source, part in zip(group, fold):
            if source in group_folds and group_folds[source] != part:
                raise ValueError("Prediction source group crosses folds")
            group_folds[source] = part
        counts = {key[7:]: archive[key].copy() for key in archive.files if key.startswith("count__") and "__" not in key[7:]}
        method_counts = {method: {key[len(f'count__{method}__'):]: archive[key].copy()
                                 for key in archive.files if key.startswith(f"count__{method}__")} for method in METHODS}
        for arm, count in counts.items():
            if count.shape != idx.shape or count.dtype.kind not in "iu":
                raise ValueError(f"Invalid count array: {arm}")
            if np.any((count < -1) | (count > np.diff(off))):
                raise ValueError(f"Out-of-range count: {arm}")
        for method, specific in method_counts.items():
            for arm, count in specific.items():
                if arm not in counts or count.shape != idx.shape or count.dtype.kind not in "iu" or np.any((count < -1) | (count > np.diff(off))):
                    raise ValueError(f"Invalid method-specific count: {method}/{arm}")
                if arm in ARMS and not np.array_equal(count, counts[arm]):
                    raise ValueError(f"Candidate allocation changed with ranking: {method}/{arm}")
        for arm in ARMS + CONTROLS:
            if arm not in counts:
                raise ValueError(f"Missing preregistered count: {arm}")
        for key in archive.files:
            if key.startswith("posterior__"):
                values = archive[key]
                if values.shape != (off[-1],):
                    raise ValueError(f"Invalid posterior shape: {key}")
                finite = np.isfinite(values)
                if np.any((values[finite] < 0) | (values[finite] > 1)):
                    raise ValueError(f"Out-of-range posterior: {key}")
        scores, ranks, coverage = {}, {}, {}
        for method in METHODS:
            score = archive[f"{method}_score"].copy()
            rank = archive[f"rank_{method}"].copy()
            if score.shape != (off[-1],) or rank.shape != score.shape or rank.dtype.kind not in "iu":
                raise ValueError(f"Invalid shared ranking: {method}")
            covered = np.ones(len(idx), bool)
            for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
                if np.all(rank[a:b] == -1):
                    covered[i] = False
                    continue
                if not np.array_equal(np.sort(rank[a:b]), np.arange(a, b)) or not np.isfinite(score[a:b]).all():
                    raise ValueError(f"Invalid answer permutation: {method}/{idx[i]}")
                ordered = score[rank[a:b]]
                if np.any(ordered[1:] > ordered[:-1]):
                    raise ValueError(f"Ranking is not descending: {method}/{idx[i]}")
                expected = a + np.argsort(-score[a:b], kind="stable")
                if not np.array_equal(rank[a:b], expected):
                    raise ValueError(f"Ranking does not use canonical stable ties: {method}/{idx[i]}")
            scores[method], ranks[method], coverage[method] = score, rank, covered
    return Predictions(bank, idx, off, scores, ranks, coverage, counts, method_counts, group, fold)


def authenticate_label_source(package, frozen, seal):
    """Authenticate the exact frozen label bytes before loading any label array."""
    package, frozen = Path(package).resolve(), Path(frozen).resolve()
    if "INPUT_FREEZE.json" not in seal["verified_files"]:
        raise ValueError("Input freeze is not authenticated by the prediction seal")
    freeze = json.loads((package/"INPUT_FREEZE.json").read_text())
    repository = package.parents[1]
    label_path = frozen/"inputs/population.npz"
    matching = [entry for relative, entry in freeze.get("inputs", {}).items()
                if (repository/relative).resolve() == label_path]
    if len(matching) != 1 or not isinstance(matching[0], dict) or not matching[0].get("sha256"):
        raise ValueError("Frozen label source has no unique authenticated input hash")
    expected = matching[0]["sha256"]
    actual = sha256(label_path)
    if actual != expected:
        raise ValueError("Frozen label source hash mismatch before label loading")
    return {"path": str(label_path), "sha256": actual, "input_freeze_sha256": seal["verified_files"]["INPUT_FREEZE.json"]}


def flags_from_shared_rank(rank, offsets, count, rank_coverage=None):
    """Return placement flags plus explicit coverage; failed counts stay missing."""
    count = np.asarray(count)
    covered = count >= 0
    if rank_coverage is not None:
        covered &= np.asarray(rank_coverage, bool)
    flags = np.zeros(int(offsets[-1]), bool)
    for i, (a, b, k) in enumerate(zip(offsets[:-1], offsets[1:], count)):
        if covered[i]:
            if k > b - a or k < 0:
                raise ValueError("Invalid count")
            flags[rank[a:a + int(k)]] = True
    return flags, covered


def partial_spearman(estimate, truth, length, weights=None):
    """Rank partial correlation controlling rank(length); constants are undefined."""
    estimate, truth, length = [np.asarray(x, float) for x in (estimate, truth, length)]
    if weights is None:
        weights = np.ones(len(estimate))
    weights = np.asarray(weights, float)
    selected = weights > 0
    if int(selected.sum()) < 3:
        return np.nan
    x, y, z = [_weighted_midranks(v, weights[None, :])[0] for v in (estimate, truth, length)]
    return float(_partial_correlations(x[None, :], y[None, :], z[None, :], weights[None, :])[0])


def _weighted_midranks(values, weights):
    # Bootstrap multiplicities require ranks recomputed in the resampled population.
    _, inverse = np.unique(values, return_inverse=True)
    levels = int(inverse.max()) + 1
    flat = inverse[None, :] + levels * np.arange(len(weights))[:, None]
    masses = np.bincount(flat.ravel(), weights=weights.ravel(), minlength=len(weights)*levels).reshape(len(weights), levels)
    middle = np.cumsum(masses, axis=1) - .5*masses
    total = weights.sum(1)
    return np.divide(middle[:, inverse], total[:, None],
                     out=np.zeros_like(weights, dtype=float), where=total[:, None] > 0)


def _partial_correlations(x, y, z, weights):
    total = weights.sum(1)
    safe = np.maximum(total, 1)
    x, y, z = [v - (weights*v).sum(1)[:, None]/safe[:, None] for v in (x, y, z)]
    vx, vy, vz = [(weights*v*v).sum(1)/safe for v in (x, y, z)]
    xy, xz, yz = [(weights*a*b).sum(1)/safe for a, b in ((x, y), (x, z), (y, z))]
    controlled = vz > 1e-20
    xy -= np.divide(xz*yz, vz, out=np.zeros_like(vz), where=controlled)
    vx -= np.divide(xz*xz, vz, out=np.zeros_like(vz), where=controlled)
    vy -= np.divide(yz*yz, vz, out=np.zeros_like(vz), where=controlled)
    valid = (vx > 1e-20) & (vy > 1e-20) & (total >= 3)
    result = np.divide(xy, np.sqrt(np.maximum(vx*vy, 0)), out=np.full(len(total), np.nan), where=valid)
    return np.clip(result, -1., 1.)


class SourceBootstrap:
    def __init__(self, groups, draws=20000, seed=20261012, weights=None):
        self.groups, self.inverse = np.unique(groups, return_inverse=True)
        self.n_groups = len(self.groups)
        self.weights = (np.asarray(weights) if weights is not None else
                        np.random.default_rng(seed).multinomial(self.n_groups, np.full(self.n_groups, 1/self.n_groups), size=draws).astype(np.int16))
        if self.weights.ndim != 2 or self.weights.shape[1] != self.n_groups:
            raise ValueError("Bootstrap group multiplicities mismatch")
        self.draws = len(self.weights)
        self.seed = seed
        self.weights_float = self.weights.astype(float)

    def means(self, values, selected):
        values = np.asarray(values, float)
        if values.ndim == 1:
            values = values[:, None]
        selected = np.asarray(selected, bool)
        sums = np.column_stack([np.bincount(self.inverse[selected], weights=values[selected, j], minlength=self.n_groups)
                                for j in range(values.shape[1])])
        counts = np.bincount(self.inverse[selected], minlength=self.n_groups)
        numerator, denominator = self.weights_float @ sums.astype(float), self.weights_float @ counts.astype(float)
        return np.divide(numerator, denominator[:, None], out=np.full(numerator.shape, np.nan, dtype=float), where=denominator[:, None] > 0)

    def correlation(self, estimate, truth, length, selected):
        selected = np.asarray(selected, bool)
        values = [np.asarray(v)[selected] for v in (estimate, truth, length)]
        if not selected.any() or len(np.unique(values[0])) < 2 or len(np.unique(values[1])) < 2:
            return np.full(self.draws, np.nan)
        result = np.full(self.draws, np.nan)
        inv = self.inverse[selected]
        for start in range(0, self.draws, 128):
            weights = self.weights[start:start+128, inv].astype(float)
            ranks = [_weighted_midranks(v, weights) for v in values]
            result[start:start+len(weights)] = _partial_correlations(*ranks, weights)
        return result


def interval(values, alpha=.05, one_sided=False):
    values = np.asarray(values, float)
    good = values[np.isfinite(values)]
    if not len(good):
        return (np.nan, np.nan)
    return tuple(np.quantile(good, [alpha, 1-alpha] if one_sided else [alpha/2, 1-alpha/2]))


def population_masks(classification, truth, length):
    cls = np.asarray(classification)
    masks = {"ALL_NONCONTROL": cls != "correct", "ALL_ANSWERS": np.ones(len(cls), bool),
             "SYNTHETIC_CLEAN": cls == "correct", "ALL_TRUE_ZERO": truth == 0,
             "NONCONTROL_TRUE_ZERO": (truth == 0) & (cls != "correct"),
             "ERROR_BEARING": truth > 0}
    masks.update({typ: cls == typ for typ in TYPE_CODES})
    for lo, hi in LENGTH_BINS:
        masks[f"LENGTH_{lo}_{hi if hi is not None else 'PLUS'}"] = (cls != "correct") & (length >= lo) & (length <= hi if hi is not None else True)
    return masks


def summarize(y, flags, offsets, count, covered, selected, bootstrap, *, valid_only=False, correlation=False):
    """Compute counts on answers and pooled step confusion on matching coverage."""
    selected = np.asarray(selected, bool)
    take = selected & covered
    truth = np.add.reduceat(np.asarray(y, int), offsets[:-1])
    length = np.diff(offsets)
    step_take = np.repeat(take, length)
    step_group = np.repeat(bootstrap.inverse, length)
    point = metrics(y[step_take], flags[step_take])
    point.update(n_answers=int(take.sum()), n_steps=int(step_take.sum()),
                 n_source_groups=int(len(set(bootstrap.inverse[take]))),
                 n_expected_answers=int(selected.sum()), n_failed_answers=int((selected & ~covered).sum()),
                 coverage=float(take.sum()/max(selected.sum(), 1)),
                 n_error_answers=int((take & (truth > 0)).sum()),
                 n_true_zero_answers=int((take & (truth == 0)).sum()))
    if not take.any():
        for key in ("prm_score", "f1_error", "f1_valid"):
            point[key] = np.nan
    confusion = np.column_stack([np.bincount(step_group[step_take & mask], minlength=bootstrap.n_groups)
                                 for mask in (y & flags, ~y & flags, y & ~flags, ~y & ~flags)])
    draws = bootstrap.weights_float @ confusion.astype(float)
    tp, fp, fn, tn = draws.T
    replicas = {"prm_score": f1(tn, fp, fn) if valid_only else (f1(tp, fp, fn)+f1(tn, fp, fn))/2,
                "f1_error": f1(tp, fp, fn), "f1_valid": f1(tn, fp, fn)}
    for values in replicas.values():
        values[tp+fp+fn+tn == 0] = np.nan
    if valid_only:
        point["prm_score"] = point["f1_valid"]
    for key, num, den in (("error_recall", tp, tp+fn), ("error_precision", tp, tp+fp),
                          ("step_flag_rate", tp+fp, tp+fp+fn+tn)):
        replicas[key] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    p_tp, p_fp, p_fn, p_tn = (point[k] for k in ("tp_error", "fp_error", "fn_error", "tn_valid"))
    point.update(error_recall=p_tp/(p_tp+p_fn) if p_tp+p_fn else np.nan,
                 error_precision=p_tp/(p_tp+p_fp) if p_tp+p_fp else np.nan,
                 step_flag_rate=(p_tp+p_fp)/(p_tp+p_fp+p_fn+p_tn) if step_take.any() else np.nan)
    errors = count-truth
    values = np.column_stack([np.abs(errors), errors, count, truth, count > 0])
    means = bootstrap.means(values, take)
    for j, key in enumerate(("count_mae", "count_bias", "count_mean", "true_count_mean", "answer_flag_rate")):
        point[key] = float(values[take, j].mean()) if take.any() else np.nan
        replicas[key] = means[:, j]
    for key, submask, value in (("zero_falseflag_rate", truth == 0, count > 0),
                                ("error_answer_miss_rate", truth > 0, count == 0)):
        sub = take & submask
        point[key] = float(value[sub].mean()) if sub.any() else np.nan
        replicas[key] = bootstrap.means(value.astype(float), sub)[:, 0]
    if correlation:
        point["partial_spearman_length"] = partial_spearman(count[take], truth[take], length[take]) if take.any() else np.nan
        replicas["partial_spearman_length"] = bootstrap.correlation(count, truth, length, take)
    for key, values in replicas.items():
        low, high = interval(values)
        point[key+"_lo95"], point[key+"_hi95"] = low, high
        point[key+"_finite_draws"] = int(np.isfinite(values).sum())
    return point, replicas


def combined_success(contrasts, full_coverage, bank="N16", method="IU_PCR", promotable=None):
    """Fixed seven-arm intersection-union gate; no shrinking M after failures."""
    decisions = []
    for arm in ARMS:
        matching = {(r["baseline"], r["metric"]): r for r in contrasts
                    if r["bank"] == bank and r["method"] == method and r["arm"] == arm
                    and r["population"] == "ALL_NONCONTROL"}
        required = (("Q80", "count_mae", "upper", 0.), ("LEGACY_MOM", "count_mae", "upper", 0.),
                    ("Q80", "prm_score", "lower", 0.), ("Q80", "error_recall", "lower", -.01))
        gates = {}
        for baseline, metric, side, threshold in required:
            row = matching.get((baseline, metric))
            bound = np.nan if row is None else row[f"simultaneous_{side}"]
            comparison_complete = row is not None and row.get("full_coverage", False)
            baseline_eligible = True if promotable is None else bool(promotable.get(baseline, False))
            gates[f"{metric}_vs_{baseline}"] = bool(comparison_complete and baseline_eligible and np.isfinite(bound)
                                                  and (bound < threshold if side == "upper" else bound > threshold))
        coverage = bool(full_coverage.get(arm, False))
        eligible = True if promotable is None else bool(promotable.get(arm, False))
        passed = coverage and eligible and all(gates.values())
        decisions.append({"arm": arm, "full_coverage": coverage, "identifiable_and_promotable": eligible, "gates": gates,
                          "baseline_eligibility": {baseline: True if promotable is None else bool(promotable.get(baseline, False))
                                                   for baseline in ("Q80", "LEGACY_MOM")},
                          "success": passed,
                          "status": "IMPROVEMENT_INTERNAL_DEVELOPMENT" if passed else "FAILED_INCOMPLETE_COVERAGE" if not coverage else "INELIGIBLE_IDENTIFICATION" if not eligible else "NOT_ESTABLISHED"})
    return {"bank": bank, "method": method, "M": MULTIPLICITY, "one_sided_alpha": ALPHA,
            "minimum_recall_delta": -.01, "any_success": any(r["success"] for r in decisions), "arms": decisions,
            "interpretation": "Internal validation on previously inspected development data; conditional prediction uncertainty; no untouched external test."}


def evaluate_package(package, frozen, draws=20000, seed=20261012):
    package, frozen = Path(package), Path(frozen)
    started = time.time()
    seal = validate_prediction_seal(package)
    predictions = {bank: load_predictions(package/f"predictions/{bank}.npz", bank) for bank in ("N16", "N56")}
    label_identity = authenticate_label_source(package, frozen, seal)
    # This is the first evaluation-label access, strictly after seal authentication.
    with np.load(frozen/"inputs/population.npz", allow_pickle=False) as archive:
        population = {key: archive[key].copy() for key in ("idx", "offsets", "y", "classification", "group", "fold")}
    y = population["y"].astype(bool)
    off, idx, cls, groups = (population[k] for k in ("offsets", "idx", "classification", "group"))
    for pred in predictions.values():
        if not np.array_equal(pred.idx, idx) or not np.array_equal(pred.offsets, off):
            raise ValueError("Prediction/evaluation population identity mismatch")
        if not np.array_equal(pred.group, groups) or not np.array_equal(pred.fold, population["fold"]):
            raise ValueError("Prediction/evaluation canonical group or fold mismatch")
    bootstrap = SourceBootstrap(groups, draws, seed)
    truth, length = np.add.reduceat(y.astype(int), off[:-1]), np.diff(off)
    masks = population_masks(cls, truth, length)
    destination = package/"evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/"BOOTSTRAP_DRAWS.npz", source_groups=bootstrap.groups, multiplicities=bootstrap.weights, seed=seed)
    rows, contrasts, per_answer, fold_rows, checkpoint_rows, ranking_rows, distributions = [], [], [], [], [], [], []
    full_coverage = {}
    promotable = {arm: True for arm in ARMS + CONTROLS}
    for bank in predictions:
        for fold in range(5):
            models_path = package/f"checkpoints/{bank}_fold{fold}/models.json"
            relative = models_path.relative_to(package).as_posix()
            if relative not in seal["verified_files"]:
                raise ValueError(f"Unsealed fitting checkpoint: {relative}")
            model = json.loads(models_path.read_text())
            for arm in ARMS + CONTROLS:
                record = model.get("status", {}).get(arm, {"status": "MISSING"})
                eligible = record.get("status") == "PASS" and record.get("promotable", True)
                if bank == "N16" and arm in promotable:
                    promotable[arm] &= bool(eligible)
                checkpoint_rows.append({"bank": bank, "fold": fold, "arm": arm, "status": record.get("status"),
                                        "promotable": bool(eligible), "failure_reason": record.get("reason"),
                                        "fold_total_s": model.get("timing", {}).get("fold_total_s"),
                                        "IU_fit_s": model.get("timing", {}).get("IU_PCR_fit_s"),
                                        "LSML_fit_s": model.get("timing", {}).get("LSML_CONT_fit_s"),
                                        "B_fit_s": model.get("timing", {}).get("B_triple_fit_s"),
                                        "D_fit_s": model.get("timing", {}).get("D_zero_fit_s")})
    flag_output = {"idx": idx, "offsets": off}
    association_cache = {}
    for bank, pred in predictions.items():
        print("EVALUATING", bank, flush=True)
        for method in METHODS:
            metric_cache = {}
            counts = dict(pred.counts)
            counts.update(pred.method_counts[method])
            auc, pairs = answer_auc(y, pred.scores[method], off)
            top1 = np.full(len(idx), np.nan)
            for i, start in enumerate(off[:-1]):
                if pred.rank_coverage[method][i]:
                    top1[i] = y[pred.ranks[method][start]]
            for population_name, mask in masks.items():
                eligible = mask & pred.rank_coverage[method] & np.isfinite(auc)
                errors = mask & pred.rank_coverage[method] & (truth > 0)
                low_auc, high_auc = interval(bootstrap.means(auc, eligible)[:, 0])
                low_top, high_top = interval(bootstrap.means(top1, errors)[:, 0])
                ranking_rows.append({"bank": bank, "method": method, "population": population_name,
                                     "within_answer_auc": float(auc[eligible].mean()) if eligible.any() else np.nan,
                                     "within_answer_auc_lo95": low_auc, "within_answer_auc_hi95": high_auc,
                                     "top1_on_error": float(top1[errors].mean()) if errors.any() else np.nan,
                                     "top1_on_error_lo95": low_top, "top1_on_error_hi95": high_top,
                                     "n_answers": int((mask & pred.rank_coverage[method]).sum()),
                                     "n_auc_answers": int(eligible.sum()), "n_error_answers": int(errors.sum()),
                                     "n_auc_source_groups": int(len(set(groups[eligible]))),
                                     "n_pairs": int(pairs[eligible].sum()),
                                     "rank_sha256": hashlib.sha256(pred.ranks[method].tobytes()).hexdigest(),
                                     "contract": "same immutable within-answer ranking for every count arm; no global OOF AUC"})
            placements = {arm: flags_from_shared_rank(pred.ranks[method], off, count, pred.rank_coverage[method])
                          for arm, count in counts.items()}
            for arm, (flags, covered) in placements.items():
                flag_output[f"{bank}__{method}__{arm}"] = flags
                flag_output[f"covered__{bank}__{method}__{arm}"] = covered
                if bank == "N16" and method == "IU_PCR":
                    full_coverage[arm] = bool(covered.all())
                for population_name, selected in masks.items():
                    result, replicates = summarize(y, flags, off, counts[arm], covered, selected, bootstrap,
                                                   valid_only=population_name == "multi_solutions")
                    if population_name in ("ALL_ANSWERS", "SYNTHETIC_CLEAN", "ALL_TRUE_ZERO", "NONCONTROL_TRUE_ZERO"):
                        # Overall PRMScore belongs only to noncontrols; clean groups
                        # report abstention/F1-valid rather than a misleading half-score.
                        result["prm_score"] = result["prm_score_lo95"] = result["prm_score_hi95"] = np.nan
                        result["prm_score_finite_draws"] = 0
                        replicates["prm_score"] = np.full(bootstrap.draws, np.nan)
                    if population_name == "ALL_NONCONTROL":
                        key = (bank, arm, hashlib.sha256(covered.tobytes()+counts[arm].tobytes()).hexdigest())
                        if key not in association_cache:
                            take = selected & covered
                            association_cache[key] = (partial_spearman(counts[arm][take], truth[take], length[take]) if take.any() else np.nan,
                                                      bootstrap.correlation(counts[arm], truth, length, take))
                        point, samples = association_cache[key]
                        result["partial_spearman_length"] = point
                        result["partial_spearman_length_lo95"], result["partial_spearman_length_hi95"] = interval(samples)
                        result["partial_spearman_length_finite_draws"] = int(np.isfinite(samples).sum())
                        replicates["partial_spearman_length"] = samples
                    result.update(bank=bank, method=method, arm=arm, population=population_name,
                                  role="candidate" if arm in ARMS else "control" if arm in CONTROLS else "diagnostic")
                    rows.append(result)
                    histogram_take = selected & covered
                    actual = np.bincount(truth[histogram_take], minlength=1)
                    estimated = np.bincount(counts[arm][histogram_take], minlength=1)
                    for k in range(max(len(actual), len(estimated))):
                        a = int(actual[k]) if k < len(actual) else 0
                        b = int(estimated[k]) if k < len(estimated) else 0
                        if a or b:
                            distributions.append({"bank": bank, "method": method, "arm": arm, "population": population_name,
                                                  "count": k, "n_true": a, "n_estimated": b,
                                                  "n_covered": int(histogram_take.sum()), "n_expected": int(selected.sum())})
                    if population_name in ("ALL_NONCONTROL", "SYNTHETIC_CLEAN", "ALL_TRUE_ZERO", "NONCONTROL_TRUE_ZERO", "ERROR_BEARING"):
                        metric_cache[bank, method, arm, population_name] = (result, replicates)
                    elif population_name in TYPE_CODES:
                        metric_cache[bank, method, arm, population_name] = (result, {"prm_score": replicates["prm_score"]})
                for category, types in CATEGORIES.items():
                    entries = [metric_cache[bank, method, arm, typ] for typ in types]
                    samples = np.mean([entry[1]["prm_score"] for entry in entries], axis=0)
                    low, high = interval(samples)
                    rows.append({"bank": bank, "method": method, "arm": arm, "population": category,
                                 "prm_score": float(np.mean([entry[0]["prm_score"] for entry in entries])),
                                 "prm_score_lo95": low, "prm_score_hi95": high,
                                 "convention": "equal subtype mean; MS valid F1 alone"})
                for fold in range(5):
                    result, _ = summarize(y, flags, off, counts[arm], covered, masks["ALL_NONCONTROL"] & (population["fold"] == fold), bootstrap)
                    fold_rows.append(dict(bank=bank, method=method, arm=arm, fold=fold, **result))
                for i, uid in enumerate(idx):
                    per_answer.append({"bank": bank, "method": method, "arm": arm, "idx": str(uid),
                                       "group": str(groups[i]), "fold": int(population["fold"][i]), "classification": str(cls[i]),
                                       "n_steps": int(length[i]), "true_count": int(truth[i]),
                                       "predicted_count": int(counts[arm][i]) if covered[i] else None,
                                       "covered": bool(covered[i])})
            for arm in counts:
                for baseline in CONTROLS:
                    if arm == baseline:
                        continue
                    for population_name in ("ALL_NONCONTROL", "SYNTHETIC_CLEAN", "ALL_TRUE_ZERO", "NONCONTROL_TRUE_ZERO", "ERROR_BEARING"):
                        aflags, acov = placements[arm]
                        bflags, bcov = placements[baseline]
                        common = acov & bcov
                        selected = masks[population_name]
                        if np.array_equal(acov, bcov):
                            left, right = (metric_cache[bank, method, term, population_name] for term in (arm, baseline))
                        else:
                            left = summarize(y, aflags, off, counts[arm], common, selected, bootstrap)
                            right = summarize(y, bflags, off, counts[baseline], common, selected, bootstrap)
                        for metric in left[1]:
                            if metric not in right[1]:
                                continue
                            difference = left[1][metric] - right[1][metric]
                            low, high = interval(difference)
                            simultaneous_low, simultaneous_high = interval(difference, ALPHA, one_sided=True)
                            contrasts.append({"bank": bank, "method": method, "arm": arm, "baseline": baseline,
                                              "population": population_name, "metric": metric,
                                              "difference": left[0][metric]-right[0][metric], "lo95": low, "hi95": high,
                                              "simultaneous_lower": simultaneous_low, "simultaneous_upper": simultaneous_high,
                                              "one_sided_alpha": ALPHA, "finite_draws": int(np.isfinite(difference).sum()),
                                              "n_answers": left[0]["n_answers"], "n_source_groups": left[0]["n_source_groups"],
                                              "paired_common_coverage": True, "full_coverage": bool(common.all())})
            print("COMPLETE", bank, method, flush=True)
    success = combined_success(contrasts, full_coverage, promotable=promotable)
    write_csv(destination/"METRICS.csv", rows)
    write_csv(destination/"CONTRASTS.csv", contrasts)
    write_csv(destination/"FOLD_METRICS.csv", fold_rows)
    write_csv(destination/"CHECKPOINT_STATUS.csv", checkpoint_rows)
    write_csv(destination/"RANKING_METRICS.csv", ranking_rows)
    write_csv(destination/"COUNT_DISTRIBUTIONS.csv", distributions)
    write_csv(destination/"PER_ANSWER.csv", per_answer)
    np.savez_compressed(destination/"FLAGS.npz", **flag_output)
    write_json(destination/"SUCCESS.json", success)
    write_json(destination/"RESULTS.json", {"metrics": rows, "contrasts": contrasts, "folds": fold_rows,
                                          "checkpoints": checkpoint_rows, "ranking": ranking_rows, "distributions": distributions, "success": success})
    status = {"status": "COMPLETE", "n_answers": len(idx), "n_steps": len(y), "n_source_groups": len(bootstrap.groups),
              "n_noncontrol": int((cls != "correct").sum()), "n_synthetic_controls": int((cls == "correct").sum()),
              "n_true_zero": int((truth == 0).sum()), "n_noncontrol_true_zero": int(((truth == 0) & (cls != "correct")).sum()),
              "bootstrap_draws": draws, "seed": seed, "multiplicity": MULTIPLICITY, "seal": seal,
              "label_source_sha256": label_identity["sha256"], "label_identity": label_identity, "runtime_s": time.time()-started,
              "module_sha256": sha256(__file__), "associations": "partial Spearman residual ranks against [1, rank(n_steps)]; recomputed weighted ranks per source draw",
              "limits": ["Previously inspected development data; no untouched external confirmation", "Intervals conditional on fitted OOF predictions",
                         "Counts are valid-step annotation membership, including unchanged out-of-range annotation semantics",
                         "Incomplete arms retain -1 failure counts and common-coverage secondary comparisons only",
                         "One-sided alpha .05/7 intersection-union primary gate; pointwise 95% descriptive intervals elsewhere"]}
    write_json(destination/"EVALUATION_STATUS.json", status)
    return status


def build_report(package):
    """Deterministic standalone HTML and scientific figures from saved evidence."""
    package = Path(package)
    validate_prediction_seal(package)
    destination = package/"report"
    destination.mkdir(exist_ok=True)
    data = json.loads((package/"evaluation/RESULTS.json").read_text())
    status = json.loads((package/"evaluation/EVALUATION_STATUS.json").read_text())
    results, contrasts, success = data["metrics"], data["contrasts"], data["success"]
    # Import the plotting library only when a report is requested.
    os.environ.setdefault("MPLCONFIGDIR", str(destination/"mpl_cache"))
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "label_free_error_count_v1"
    import matplotlib.pyplot as plt
    def save_figure(fig, stem):
        for extension in ("png", "svg"):
            metadata = {"Date": None} if extension == "svg" else {"Software": "label_free_error_count_v1"}
            fig.savefig(destination/f"{stem}.{extension}", dpi=160, metadata=metadata)
        plt.close(fig)
    headline = [r for r in results if r.get("bank") == "N16" and r.get("method") == "IU_PCR" and r.get("population") == "ALL_NONCONTROL"]
    arm_order = [arm for arm in CONTROLS + ARMS if any(r["arm"] == arm for r in headline)]
    def lookup(arm):
        return next(r for r in headline if r["arm"] == arm)
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for bank, axs in zip(("N16", "N56"), axes):
        points = {r["arm"]: r for r in results if r.get("bank") == bank and r.get("method") == "IU_PCR" and r.get("population") == "ALL_NONCONTROL"}
        for ax, key, label in zip(axs, ("count_mae", "prm_score", "error_recall"), ("Count MAE", "Noncontrol PRMScore", "Error-step recall")):
            numeric = np.array([np.nan if points[arm].get(key) is None else points[arm][key] for arm in arm_order])
            lows = np.array([np.nan if points[arm].get(key+"_lo95") is None else points[arm][key+"_lo95"] for arm in arm_order])
            highs = np.array([np.nan if points[arm].get(key+"_hi95") is None else points[arm][key+"_hi95"] for arm in arm_order])
            finite = np.isfinite(numeric) & np.isfinite(lows) & np.isfinite(highs)
            ax.errorbar(numeric[finite], np.arange(len(arm_order))[finite], xerr=np.maximum(np.vstack([numeric-lows, highs-numeric])[:, finite], 0), fmt="o", capsize=3)
            ax.set_yticks(np.arange(len(arm_order)), arm_order, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(label)
            ax.set_title(bank + (" | historical anchor" if bank == "N16" else " | prespecified sensitivity"))
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("Fixed IU-PCR ranking | internal development validation | paired source-group 95% intervals")
    fig.tight_layout()
    save_figure(fig, "count_quality")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for bank, ax in zip(("N16", "N56"), axes):
        selected = [r for r in data["distributions"] if r["bank"] == bank and r["method"] == "IU_PCR" and r["population"] == "ALL_NONCONTROL"]
        truth_histogram = [r for r in selected if r["arm"] == "Q80"]
        if truth_histogram:
            ax.step([r["count"] for r in truth_histogram], [r["n_true"]/r["n_covered"] for r in truth_histogram], where="mid", color="black", lw=2, label="Valid-step truth")
        for arm in arm_order:
            hist = [r for r in selected if r["arm"] == arm]
            if hist:
                ax.step([r["count"] for r in hist], [r["n_estimated"]/r["n_covered"] for r in hist], where="mid", alpha=.7,
                        label=arm + (" (partial)" if hist[0]["n_covered"] != hist[0]["n_expected"] else ""))
        ax.set(xlabel="Error count k", ylabel="Share of covered noncontrol answers", title=bank)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Truth and estimated count distributions | incomplete arms show their covered population only")
    fig.tight_layout()
    save_figure(fig, "count_distributions")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for bank, axs in zip(("N16", "N56"), axes):
        folds = {(r["arm"], r["fold"]): r for r in data["folds"] if r["bank"] == bank and r["method"] == "IU_PCR"}
        for ax, metric in zip(axs, ("count_mae", "prm_score")):
            for arm in ARMS:
                differences = []
                for fold in range(5):
                    left, right = folds[arm, fold], folds["Q80", fold]
                    complete = left["coverage"] == 1 and right["coverage"] == 1
                    differences.append(left[metric]-right[metric] if complete and left[metric] is not None and right[metric] is not None else np.nan)
                ax.plot(range(5), differences, marker="o", label=arm)
            ax.axhline(0, color="black", lw=.8)
            ax.set(xticks=range(5), xlabel="Canonical source fold", ylabel=metric + " difference vs Q80", title=bank)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle("Fold stability | missing markers represent failed/incomplete fold coverage")
    fig.tight_layout()
    save_figure(fig, "fold_stability")
    def fmt(value, digits=4):
        if isinstance(value, bool):
            return "yes" if value else "no"
        return "undefined" if value is None else f"{value:.{digits}f}" if isinstance(value, (float, int)) else html.escape(str(value))
    def table(columns, rows):
        return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{fmt(row.get(key))}</td>" for _, key in columns) + "</tr>" for row in rows) + "</tbody></table></div>"
    hypotheses = {"LOCAL_MOM_MEAN": ("סף מוחלט ללא שינוי ייצוג", "סף אימון גלובלי על ציון הקבוצה המקומי; MoM ותוחלת"),
                  "A_MOM_MEAN": ("שימור מידע בין תשובות", "ראיות לפני תקנון בתשובה, תקנון וסף מהאימון; MoM ותוחלת"),
                  "A_MOM_MEDIAN": ("שינוי החלטת הכמות בלבד", "אותו posterior של A; חציון התפלגות הכמות"),
                  "B_TRIPLE_MEAN": ("עמידות של האמידה לתלות", "שלשה יציבה אחת שנבחרה באימון ללא תוויות; posterior מהשלשה ותוחלת"),
                  "B_TRIPLE_MEDIAN": ("שינוי החלטת הכמות בלבד", "אותו posterior של B; חציון התפלגות הכמות"),
                  "D_ZERO_MEAN": ("מצב נקי מפורש ברמת תשובה", "מודל חבוי ברמת התשובה; תוחלת"),
                  "D_ZERO_MEDIAN": ("שינוי החלטת הכמות בלבד", "אותו posterior של D; חציון התפלגות הכמות")}
    experiment_rows = []
    for decision in success["arms"]:
        row = lookup(decision["arm"])
        hypothesis, change = hypotheses[decision["arm"]]
        experiment_rows.append({"arm": decision["arm"], "hypothesis": hypothesis, "change": change,
                                "result": f"MAE {fmt(row.get('count_mae'))}; PRMScore {fmt(row.get('prm_score'))}; recall {fmt(row.get('error_recall'))}",
                                "conclusion": "שיפור באימות פנימי בלבד" if decision["success"] else "אין תוצאה מלאה בגלל כשל זרוע" if not decision["full_coverage"] else "המודל אינו מזוהה מספיק לקידום" if not decision["identifiable_and_promotable"] else "קריטריון השיפור המשולב לא עבר"})
    conclusion = "נמצא שיפור שעובר את הקריטריון המשולב באימות פנימי על נתוני development שכבר נבדקו." if success["any_success"] else "לא הוכח שיפור שעובר את הקריטריון המשולב במשפחה שנבדקה."
    text = "<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Label-free error-count experiment</title><style>body{font:16px system-ui;max-width:1220px;margin:40px auto;padding:0 22px;color:#173042;background:#f8fafb}h1,h2{line-height:1.25}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0;background:white}th,td{border:1px solid #d7e2e7;padding:9px;text-align:start}th{background:#edf3f6}img{width:100%;background:white}details{margin:22px 0}summary{cursor:pointer;font-weight:600}a{color:#176f9d}.small{font-size:13px;color:#526875}.metric{direction:ltr;unicode-bidi:embed}</style><h1>אמידת מספר צעדים שגויים ללא תוויות</h1>"
    text += f"<p><strong>{conclusion}</strong> הדירוג הראשי של IU-PCR קבוע בין זרועות הכמות. N16 הוא בנק ייחוס היסטורי; N56 הוא בקרת רגישות שנקבעה מראש.</p>"
    text += "<p>שיפור דורש ירידה ב־MAE מול Q80 ומול MoM הקודם, עלייה ב־PRMScore מול Q80, ורווח שמוציא אובדן recall של יותר מנקודת אחוז אחת. תיקון הבחירה בין שבע הזרועות משתמש ב־α=0.05/7 חד־צדדי. רווחים אחרים בדוח הם מזווגים, נקודתיים, של 95%.</p>"
    text += f"<p class='small'>{status['n_noncontrol']:,} תשובות noncontrol; {status['n_synthetic_controls']:,} בקרות נקיות סינתטיות; {status['n_true_zero']:,} תשובות בלי שגיאה תקפה; {status['n_source_groups']:,} קבוצות מקור. {status['bootstrap_draws']:,} דגימות bootstrap של קבוצות מקור שלמות.</p>"
    text += "<img src='count_quality.svg' alt='Count MAE, PRMScore and recall with paired source-group confidence intervals'>"
    text += table([( "זרוע", "arm"), ("השערה", "hypothesis"), ("שינוי", "change"), ("תוצאה", "result"), ("מסקנה", "conclusion")], experiment_rows)
    text += "<h2>תוצאה מול מחיר בזיהוי שגיאות</h2>"
    text += table([("Arm", "arm"), ("Count MAE", "count_mae"), ("Bias", "count_bias"), ("PRMScore", "prm_score"), ("Recall", "error_recall"), ("Precision", "error_precision"), ("Error-answer misses", "error_answer_miss_rate"), ("Partial Spearman | length", "partial_spearman_length"), ("Coverage", "coverage")], headline)
    text += "<details><summary>התפלגות הכמות ויציבות בין folds</summary><img src='count_distributions.svg' alt='True and estimated count distributions for N16 and N56'><img src='fold_stability.svg' alt='Fold count and PRMScore differences versus Q80'></details>"
    for title, population_name in (("סימון שווא בבקרות נקיות", "SYNTHETIC_CLEAN"), ("כל התשובות עם אמת אפס", "ALL_TRUE_ZERO"), ("תשובות noncontrol עם אמת אפס", "NONCONTROL_TRUE_ZERO")):
        subset = [r for r in results if r.get("bank") == "N16" and r.get("method") == "IU_PCR" and r.get("population") == population_name]
        text += f"<details><summary>{title}</summary>" + table([("Arm", "arm"), ("N", "n_answers"), ("Count MAE", "count_mae"), ("False flags", "zero_falseflag_rate"), ("Low 95%", "zero_falseflag_rate_lo95"), ("High 95%", "zero_falseflag_rate_hi95")], subset) + "</details>"
    text += "<details><summary>הבדלים מזווגים וקריטריוני ההצלחה</summary>"
    subset = [r for r in contrasts if r["bank"] == "N16" and r["method"] == "IU_PCR" and r["population"] == "ALL_NONCONTROL" and r["arm"] in ARMS and r["baseline"] in ("Q80", "LEGACY_MOM") and r["metric"] in ("count_mae", "count_bias", "prm_score", "error_recall", "error_answer_miss_rate", "partial_spearman_length")]
    text += table([("Arm", "arm"), ("Baseline", "baseline"), ("Metric", "metric"), ("Difference", "difference"), ("Lo 95%", "lo95"), ("Hi 95%", "hi95"), ("Adjusted lower", "simultaneous_lower"), ("Adjusted upper", "simultaneous_upper"), ("N", "n_answers")], subset) + "</details>"
    for title, subset in (("N56 — בקרת רגישות", [r for r in results if r.get("bank") == "N56" and r.get("method") == "IU_PCR" and r.get("population") == "ALL_NONCONTROL"]),
                          ("L-SML — דירוג משני", [r for r in results if r.get("method") == "LSML_CONT" and r.get("population") == "ALL_NONCONTROL"]),
                          ("תשעת הסוגים והקטגוריות", [r for r in results if r.get("bank") == "N16" and r.get("method") == "IU_PCR" and r.get("population") in tuple(TYPE_CODES) + tuple(CATEGORIES)]),
                          ("חלוקה לפי אורך", [r for r in results if r.get("bank") == "N16" and r.get("method") == "IU_PCR" and r.get("population", "").startswith("LENGTH_")]),
                          ("יציבות בין folds", [r for r in data["folds"] if r["bank"] == "N16" and r["method"] == "IU_PCR"] )):
        text += f"<details><summary>{title}</summary>" + table([("Bank", "bank"), ("Arm", "arm"), ("Population", "population"), ("Fold", "fold"), ("N", "n_answers"), ("MAE", "count_mae"), ("Bias", "count_bias"), ("PRMScore", "prm_score"), ("Recall", "error_recall"), ("Coverage", "coverage")], subset) + "</details>"
    text += "<details><summary>הנחות, כיסוי ומה עדיין פתוח</summary><p>Overall מחושב על noncontrols מתוך ספירות confusion משותפות. MS משתמש ב־valid-step F1 בלבד; Sensitivity כולל PS, DR ו־MS. חישוב אמת הכמות שומר על חברות צעדים תקפים לפי האנוטציה המקורית, בלי לתקן אינדקסים מחוץ לטווח.</p><p>כיוון המחלקה החבויה נקבע בהנחת רוב או שלשה בעלי מהימנות מעל chance ובכיוון חיובי; התוויות אינן מתקנות את הכיוון. MoM וחישוב Bayes מניחים אי־תלות מותנית של המצביעים בהינתן מחלקת הצעד. B בוחרת שלשה יציבה אחת באימון ומשתמשת רק בה לחישוב posterior. בדיקת ה־bootstrap שלה מותנית בייצוג הקבוצות שנלמד; היא אינה משקללת מחדש clustering. קורלציה נמוכה או יציבה אינה מוכיחה את הנחת אי־התלות. S-hat, participation ratio וקורלציות הם אבחונים או constraints, ולא מטרות אופטימיזציה.</p><p>חציון התפלגות הכמות מחושב בהנחת עצמאות מותנית של תוויות הצעדים ופרמטרים קבועים. סכום תוחלות הצעדים אינו דורש עצמאות בין צעדים, אך עדיין תלוי בתקפות מודל ה־posterior. D מוסיף מצב נקי ברמת תשובה; כשלי זיהוי או פתרונות מנוונים מונעים קידום גם כשנשמרים חיזויים אבחוניים.</p><p>רווחי הסמך מותנים בחיזויים שנלמדו ונשמרו. הם אינם מכסים את כל שונות הלמידה או החיפוש ההיסטורי. זרוע שנכשלה נשארת חסרה; השוואות בכיסוי חלקי משניות בלבד. שיפור פנימי אינו אישור חיצוני, ואי־מובהקות אינה שקילות. התוצאות מתייחסות למימושים שנבדקו, ולא לכל אפשרות של אמידת כמות.</p></details>"
    text += "<details><summary>בדיקת דירוג משותף</summary>" + table([("Bank", "bank"), ("Method", "method"), ("Within-answer AUC", "within_answer_auc"), ("Top1 on error", "top1_on_error"), ("N AUC", "n_auc_answers"), ("N error", "n_error_answers")], [r for r in data["ranking"] if r["population"] == "ALL_NONCONTROL"]) + "<p>אותו דירוג נשמר בכל זרוע כמות. לא חושב AUC גלובלי על ציוני OOF.</p></details>"
    text += "<details><summary>כשלי זרועות ועלות התאמה</summary>" + table([("Bank", "bank"), ("Fold", "fold"), ("Arm", "arm"), ("Status", "status"), ("Promotable", "promotable"), ("Reason", "failure_reason"), ("Fold fit seconds", "fold_total_s")], data["checkpoints"]) + "</details>"
    text += "<p><a href='../PROTOCOL.json'>Protocol</a> · <a href='../PREDICTION_SEAL.json'>Prediction seal</a> · <a href='../evaluation/METRICS.csv'>Metrics CSV</a> · <a href='../evaluation/CONTRASTS.csv'>Paired contrasts</a> · <a href='../evaluation/PER_ANSWER.csv'>Per-answer counts</a> · <a href='../evaluation/SUCCESS.json'>Success gates</a> · <a href='../evaluation/EVALUATION_STATUS.json'>Evaluation provenance</a></p></html>"
    output = destination/"index.html"
    output.write_text(text)
    write_csv(destination/"EXPERIMENT_TABLE.csv", experiment_rows)
    write_json(destination/"REPORT_RECEIPT.json", {"html_sha256": sha256(output), "results_sha256": sha256(package/"evaluation/RESULTS.json"),
                                                 "module_sha256": sha256(__file__), "deterministic": True})
    return output
