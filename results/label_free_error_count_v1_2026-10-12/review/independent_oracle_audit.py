"""Independent point and paired-bootstrap reconstruction of component oracles."""
import csv
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from independent_sealed_audit import authenticate, point, reconstruct_flags, f1, sha, TYPES

PACKAGE = Path(__file__).resolve().parent.parent
DEST = PACKAGE/"diagnostics"
CATEGORIES = dict(simplicity=("redundency", "circular"), soundness=("counterfactual", "step_contradiction", "domain_inconsistency", "confidence"),
                  sensitivity=("missing_condition", "deception", "multi_solutions"))


def main():
    seal, pop = authenticate()
    metadata = json.loads((DEST/"COUNT_ORACLES.json").read_text())
    assert metadata["main_seal_sha256"] == sha(PACKAGE/"PREDICTION_SEAL.json")
    for name, expected in metadata["outputs"].items(): assert sha(DEST/name) == expected
    assert not metadata["fits_or_selection_performed"] and not metadata["candidate_multiplicity_changed"]
    with np.load(DEST/"COUNT_ORACLE_FLAGS.npz") as z: exported = {k: z[k] for k in z.files}
    metrics = list(csv.DictReader((DEST/"COUNT_ORACLE_METRICS.csv").open()))
    contrasts = list(csv.DictReader((DEST/"COUNT_ORACLE_CONTRASTS.csv").open()))
    off, y, cls = pop["offsets"], pop["y"].astype(bool), pop["classification"]
    n = np.diff(off); truth = np.add.reduceat(y.astype(int), off[:-1]); groups, inverse = np.unique(pop["group"], return_inverse=True)
    masks = dict(ALL_NONCONTROL=cls != "correct", SYNTHETIC_CLEAN=cls == "correct", ALL_TRUE_ZERO=truth == 0,
                 NONCONTROL_TRUE_ZERO=(truth == 0) & (cls != "correct"), ERROR_BEARING=truth > 0)
    masks.update({typ: cls == typ for typ in TYPES})
    expected_rows, placements, compared, error = {}, {}, 0, 0.
    for bank in ("N16", "N56"):
        with np.load(PACKAGE/f"predictions/{bank}.npz") as z: primary = {k: z[k] for k in z.files}
        for method in ("IU_PCR", "LSML_CONT"):
            rank = primary["rank_"+method]; q80 = primary[f"count__{method}__Q80"]
            error_first = np.concatenate([np.r_[rank[a:b][y[rank[a:b]]], rank[a:b][~y[rank[a:b]]]] for a, b in zip(off[:-1], off[1:])])
            configs = (("Q80", rank, q80), ("ORACLE_TRUE_COUNT_FIXED_RANK", rank, truth),
                       ("ORACLE_ERROR_FIRST_Q80_COUNT", error_first, q80), ("ALL_ZERO_NEGATIVE_CONTROL", rank, np.zeros_like(truth)))
            for arm, ordering, count in configs:
                flags, covered = reconstruct_flags(ordering, off, count)
                assert np.array_equal(flags, exported[f"{bank}__{method}__{arm}"])
                if arm == "ORACLE_ERROR_FIRST_Q80_COUNT":
                    assert np.array_equal(np.add.reduceat((flags & y).astype(int), off[:-1]), np.minimum(q80, truth))
                placements[bank, method, arm] = count, flags
                for scope, mask in masks.items():
                    r = point(pop, count, flags, covered, mask, ms=scope == "multi_solutions")
                    if scope in ("SYNTHETIC_CLEAN", "ALL_TRUE_ZERO", "NONCONTROL_TRUE_ZERO"): r["prm_score"] = np.nan
                    expected_rows[bank, method, arm, scope] = r
                for category, types in CATEGORIES.items():
                    expected_rows[bank, method, arm, category] = dict(prm_score=np.mean([expected_rows[bank, method, arm, typ]["prm_score"] for typ in types]))
    for row in metrics:
        expected = expected_rows[row["bank"], row["method"], row["arm"], row["population"]]
        for metric, value in expected.items():
            if metric not in row: continue
            actual = float(row[metric]) if row[metric] else np.nan
            if np.isfinite(value): error = max(error, abs(actual-value)); assert abs(actual-value) < 1e-12
            else: assert not np.isfinite(actual)
            compared += 1
    with np.load(PACKAGE/"evaluation/BOOTSTRAP_DRAWS.npz") as z: weights = z["multiplicities"].astype(float); assert np.array_equal(z["source_groups"], groups)
    checked, interval_error = 0, 0.
    for bank in ("N16", "N56"):
        for method in ("IU_PCR", "LSML_CONT"):
            selected = masks["ALL_NONCONTROL"]; step = np.repeat(selected, n); step_group = np.repeat(inverse, n)
            replicas = {}
            for arm in ("Q80", "ORACLE_TRUE_COUNT_FIXED_RANK", "ORACLE_ERROR_FIRST_Q80_COUNT", "ALL_ZERO_NEGATIVE_CONTROL"):
                count, flags = placements[bank, method, arm]
                sums = np.bincount(inverse[selected], weights=abs(count[selected]-truth[selected]), minlength=len(groups))
                denominator = weights@np.bincount(inverse[selected], minlength=len(groups))
                stats = np.column_stack([np.bincount(step_group[step & mask], minlength=len(groups)) for mask in (y & flags, ~y & flags, y & ~flags, ~y & ~flags)])
                tp, fp, fn, tn = (weights@stats).T
                replicas[arm] = dict(count_mae=(weights@sums)/denominator, prm_score=(f1(tp, fp, fn)+f1(tn, fp, fn))/2,
                                     error_recall=tp/(tp+fn))
            for arm in ("ORACLE_TRUE_COUNT_FIXED_RANK", "ORACLE_ERROR_FIRST_Q80_COUNT", "ALL_ZERO_NEGATIVE_CONTROL"):
                for metric, sample in replicas[arm].items():
                    delta = sample-replicas["Q80"][metric]; low, high = np.quantile(delta, [.025, .975])
                    row = next(r for r in contrasts if (r["bank"], r["method"], r["arm"], r["population"], r["metric"]) == (bank, method, arm, "ALL_NONCONTROL", metric))
                    assert row["promotion_eligible"] == "False" and row["pointwise_only"] == "True"
                    interval_error = max(interval_error, abs(low-float(row["lo95"])), abs(high-float(row["hi95"])))
                    assert interval_error < 1e-12
                    checked += 1
    receipt = dict(status="PASS", matched_flags_exact=True, all_type_and_category_points_verified=True, MS_valid_F1_verified=True,
                   point_values_compared=compared, max_point_error=error, primary_population_paired_intervals_checked=checked,
                   max_paired_interval_error=interval_error, count_oracle_not_global_PRM_maximizer=True,
                   quota_oracle_order_holds_method_specific_Q80_counts=True, zero_control_has_no_error_recall=True,
                   no_fitting_or_selection=True, primary_candidate_family_unchanged=True,
                   oracle_provenance_sha256=sha(DEST/"COUNT_ORACLES.json"), script_sha256=sha(__file__),
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (PACKAGE/"review/INDEPENDENT_ORACLE_AUDIT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__": main()
