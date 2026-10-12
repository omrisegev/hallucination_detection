"""Exploratory paired component checks on sealed predictions; no refitting.

This post-evaluation diagnostic leaves the protocol, seven-arm primary family,
prediction seal and frozen evaluator unchanged. Reuses the exact main-run source
bootstrap multiplicities. Partial coverage never becomes a full-population claim.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.error_count_evaluation import (SourceBootstrap, authenticate_label_source,
    flags_from_shared_rank, interval, load_predictions, sha256, summarize,
    validate_prediction_seal, write_csv, write_json)

COMPARISONS = (
    ("C_A_MEDIAN_MINUS_MEAN", "A_MOM_MEDIAN", "A_MOM_MEAN"),
    ("C_B_MEDIAN_MINUS_MEAN", "B_TRIPLE_MEDIAN", "B_TRIPLE_MEAN"),
    ("C_D_MEDIAN_MINUS_MEAN", "D_ZERO_MEDIAN", "D_ZERO_MEAN"),
    ("A_ABSOLUTE_MINUS_LOCAL_THRESHOLD", "A_MOM_MEAN", "LOCAL_MOM_MEAN"),
    ("D_ZERO_MINUS_ORDINARY_PRIOR", "D_ZERO_MEAN", "D_PRIOR_MEAN"),
    ("B_TRIPLE_MINUS_A_ALL_GROUPS", "B_TRIPLE_MEAN", "A_MOM_MEAN"),
    ("LOCAL_THRESHOLD_MINUS_RELATIVE", "LOCAL_MOM_MEAN", "LEGACY_MOM"),
)
METRICS = ("count_mae", "count_bias", "prm_score", "error_recall", "zero_falseflag_rate", "error_answer_miss_rate")
SCOPE = "Post-evaluation exploratory component diagnostic; pointwise paired 95% intervals; no candidate selection or primary-family change."


def semantic_hash(values):
    values = np.asarray(values, dtype="<f8")
    missing = np.isnan(values)
    normalized = values.copy()
    normalized[missing] = 0.
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.packbits(missing).tobytes())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def participation_ratio(weights):
    weights = np.asarray(weights, float)
    energy = weights*weights
    return float(energy.sum()**2/(energy*energy).sum()) if (energy*energy).sum() > 0 else np.nan


def grouping_diagnostics(package, seal):
    rows, pairs, within = [], [], []
    for bank in ("N16", "N56"):
        partitions = []
        for fold in range(5):
            relative = f"checkpoints/{bank}_fold{fold}/models.json"
            if relative not in seal["verified_files"]:
                raise ValueError("Unsealed group-fit state")
            model = json.loads((package/relative).read_text())["LSML_CONT"]
            diagnostic = model["diagnostic"]
            partition = np.asarray(diagnostic["group"], int)
            partitions.append(partition)
            rows.append({"bank": bank, "fold": fold, "K": diagnostic["K"],
                         "cross_weight_PR": participation_ratio(diagnostic["cross_weights"]),
                         "effective_feature_weight_PR": participation_ratio(model["weight"]),
                         "n_features": len(partition), "scope": "Label-free fit diagnostic only; never a score objective."})
            for group, record in enumerate(diagnostic["group_weights"]):
                within.append({"bank": bank, "fold": fold, "group": group,
                               "n_members": len(record["columns"]), "members": ",".join(map(str, record["columns"])),
                               "within_weight_PR": participation_ratio(record["weights"]),
                               "cross_weight": diagnostic["cross_weights"][group],
                               "scope": "Diagnostic only; low correlation or PR does not prove conditional independence."})
        for i in range(5):
            for j in range(i+1, 5):
                if len(partitions[i]) != len(partitions[j]):
                    raise ValueError("Inconsistent canonical bank feature ordering")
                pairs.append({"bank": bank, "left_fold": i, "right_fold": j,
                              "adjusted_rand_index": float(adjusted_rand_score(partitions[i], partitions[j])),
                              "n_features": len(partitions[i]), "scope": "Partition stability over the same bank columns; diagnostic only."})
    return rows, pairs, within


def timing_diagnostics(package, seal):
    rows = []
    for bank in ("N16", "N56"):
        for fold in range(5):
            relative = f"checkpoints/{bank}_fold{fold}/models.json"
            if relative not in seal["verified_files"]:
                raise ValueError("Unsealed timing source")
            model = json.loads((package/relative).read_text())
            for key, seconds in model.get("timing", {}).items():
                if seconds is None or not np.isfinite(seconds) or seconds < 0:
                    raise ValueError("Invalid saved timing")
                rows.append({"bank": bank, "fold": fold, "timing_key": key, "seconds": float(seconds),
                             "kind": "fit" if "fit" in key else "prediction" if "prediction" in key else "total",
                             "source_sha256": seal["verified_files"][relative],
                             "scope": "Saved CPU wall-clock measurement; exact stage definition from the frozen driver; no refit/re-timing."})
    summary = []
    for bank in ("N16", "N56"):
        for key in sorted(set(row["timing_key"] for row in rows if row["bank"] == bank)):
            values = [row["seconds"] for row in rows if row["bank"] == bank and row["timing_key"] == key]
            summary.append({"bank": bank, "timing_key": key, "n_recorded_folds": len(values),
                            "min_s": min(values), "median_s": float(np.median(values)), "max_s": max(values), "sum_s": sum(values),
                            "scope": "Missing failed-stage timing is not imputed as zero; five source folds expected."})
    return rows, summary


def run(package, frozen):
    package, frozen = Path(package).resolve(), Path(frozen).resolve()
    started = time.time()
    seal = validate_prediction_seal(package)
    identity = authenticate_label_source(package, frozen, seal)
    with np.load(frozen/"inputs/population.npz", allow_pickle=False) as archive:
        pop = {key: archive[key].copy() for key in ("idx", "offsets", "y", "classification", "group", "fold")}
    y, off, cls, groups = pop["y"].astype(bool), pop["offsets"], pop["classification"], pop["group"]
    truth = np.add.reduceat(y.astype(int), off[:-1])
    masks = {"ALL_NONCONTROL": cls != "correct", "SYNTHETIC_CLEAN": cls == "correct",
             "ERROR_BEARING": truth > 0, "ALL_TRUE_ZERO": truth == 0}
    draws_path = package/"evaluation/BOOTSTRAP_DRAWS.npz"
    with np.load(draws_path, allow_pickle=False) as archive:
        weights, source_groups, seed = archive["multiplicities"], archive["source_groups"], int(archive["seed"])
    if len(weights) != 20000 or seed != 20261012 or not np.array_equal(np.unique(groups), source_groups):
        raise ValueError("Main-run bootstrap contract differs")
    expected = np.random.default_rng(seed).multinomial(len(source_groups), np.full(len(source_groups), 1/len(source_groups)), size=20000)
    if not np.array_equal(weights, expected):
        raise ValueError("Saved multiplicities disagree with the recorded main-run RNG seed")
    del expected
    bootstrap = SourceBootstrap(groups, weights=weights, seed=seed)
    contrasts, changed, posterior_checks, difference_histograms = [], [], [], []
    for bank in ("N16", "N56"):
        print("COMPONENTS", bank, flush=True)
        pred = load_predictions(package/f"predictions/{bank}.npz", bank)
        if not all(np.array_equal(a, b) for a, b in ((pred.idx, pop["idx"]), (pred.offsets, off), (pred.group, groups), (pred.fold, pop["fold"]))):
            raise ValueError("Prediction/component population identities differ")
        with np.load(package/f"predictions/{bank}.npz", allow_pickle=False) as archive:
            for component, left, right in COMPARISONS[:3]:
                pleft, pright = archive["posterior__"+left], archive["posterior__"+right]
                same = np.array_equal(pleft, pright, equal_nan=True)
                posterior_checks.append({"bank": bank, "component": component, "left": left, "right": right,
                                         "same_posterior_including_missingness": bool(same),
                                         "left_semantic_sha256": semantic_hash(pleft), "right_semantic_sha256": semantic_hash(pright),
                                         "n_finite_steps": int(np.isfinite(pleft).sum()), "scope": SCOPE})
                if not same:
                    raise ValueError("Median/mean comparison does not hold posterior fixed")
        for method in ("IU_PCR", "LSML_CONT"):
            counts = dict(pred.counts)
            counts.update(pred.method_counts[method])
            placements = {arm: flags_from_shared_rank(pred.ranks[method], off, count, pred.rank_coverage[method])
                          for arm, count in counts.items()}
            for component, left, right in COMPARISONS:
                if left not in counts or right not in counts:
                    changed.append({"bank": bank, "method": method, "component": component, "status": "MISSING_DIAGNOSTIC", "scope": SCOPE})
                    continue
                lf, lc = placements[left]
                rf, rc = placements[right]
                common = lc & rc
                difference = counts[left]-counts[right]
                for population, selected in masks.items():
                    take = selected & common
                    metadata = {"bank": bank, "method": method, "component": component, "left": left, "right": right,
                                "population": population, "n_answers": int(take.sum()), "n_expected_answers": int(selected.sum()),
                                "n_source_groups": int(len(set(groups[take]))), "coverage": float(take.sum()/max(selected.sum(), 1)),
                                "full_bank_coverage": bool(common.all()), "full_population_coverage": bool(np.all(common[selected])), "scope": SCOPE}
                    changed.append({**metadata, "status": "COMPLETE" if np.all(common[selected]) else "PARTIAL_COMMON_COVERAGE",
                                    "n_count_changed": int((take & (difference != 0)).sum()),
                                    "n_count_increased": int((take & (difference > 0)).sum()), "n_count_decreased": int((take & (difference < 0)).sum())})
                    if take.any():
                        values, frequencies = np.unique(difference[take], return_counts=True)
                        difference_histograms.extend({**metadata, "count_difference": int(value), "n_answers_at_difference": int(frequency)}
                                                     for value, frequency in zip(values, frequencies))
                    left_point, left_boot = summarize(y, lf, off, counts[left], common, selected, bootstrap)
                    right_point, right_boot = summarize(y, rf, off, counts[right], common, selected, bootstrap)
                    for metric in METRICS:
                        if metric == "prm_score" and population in ("SYNTHETIC_CLEAN", "ALL_TRUE_ZERO"):
                            estimate = left_value = right_value = low = high = np.nan
                            n_draws = 0
                        else:
                            samples = left_boot[metric]-right_boot[metric]
                            low, high = interval(samples)
                            left_value, right_value = left_point[metric], right_point[metric]
                            estimate = left_value-right_value
                            n_draws = int(np.isfinite(samples).sum())
                        contrasts.append({**metadata, "metric": metric, "left_value": left_value, "right_value": right_value,
                                          "difference": estimate, "lo95": low, "hi95": high, "finite_draws": n_draws,
                                          "interpretation": "Pointwise, conditional on saved fits; partial comparisons are not full-population effects."})
    weight_rows, ari_rows, within_rows = grouping_diagnostics(package, seal)
    timing_rows, timing_summary = timing_diagnostics(package, seal)
    destination = package/"diagnostics"
    write_csv(destination/"COMPONENT_CONTRASTS.csv", contrasts)
    write_csv(destination/"COMPONENT_COUNT_CHANGES.csv", changed)
    write_csv(destination/"COMPONENT_COUNT_DIFFERENCE_HISTOGRAMS.csv", difference_histograms)
    write_csv(destination/"GROUP_PARTITION_ARI.csv", ari_rows)
    write_csv(destination/"GROUP_WEIGHT_PR.csv", weight_rows)
    write_csv(destination/"WITHIN_GROUP_WEIGHT_PR.csv", within_rows)
    write_csv(destination/"TIMING.csv", timing_rows)
    write_csv(destination/"TIMING_SUMMARY.csv", timing_summary)
    write_json(destination/"TIMING.json", {"source": "authenticated saved fold model timing", "rows": timing_rows, "summary": timing_summary,
                                           "no_refitting": True, "units": "seconds", "script_sha256": sha256(__file__)})
    write_json(destination/"COMPONENT_POSTERIOR_INVARIANCE.json", posterior_checks)
    status = {"status": "COMPLETE", "scope": SCOPE, "comparisons": COMPARISONS, "bootstrap_draws": 20000, "seed": seed,
              "bootstrap_draws_sha256": sha256(draws_path), "seal_sha256": seal["seal_sha256"], "label_identity": identity,
              "script_sha256": sha256(__file__), "runtime_s": time.time()-started, "n_contrast_rows": len(contrasts),
              "posterior_invariance_all_pass": all(row["same_posterior_including_missingness"] for row in posterior_checks),
              "primary_M_unchanged": 7, "no_refitting": True,
              "diagnostic_limits": ["Post-evaluation diagnostic, not a new preregistered primary test", "No adjustment for component multiple comparisons",
                                    "Bootstrap conditions on saved fitted predictions", "Grouping ARI and PR are diagnostics, not optimization objectives"]}
    write_json(destination/"COMPONENT_RESULTS.json", {"status": status, "contrasts": contrasts, "count_changes": changed,
                                                     "posterior_invariance": posterior_checks, "partition_ari": ari_rows, "weight_pr": weight_rows})
    write_json(destination/"COMPONENT_STATUS.json", status)
    print("COMPONENTS_COMPLETE", len(contrasts), "rows", round(status["runtime_s"], 2), "seconds", flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--frozen", type=Path, default=ROOT/"results/advisor_completion_2026-10-11")
    arguments = parser.parse_args()
    run(arguments.package, arguments.frozen)
