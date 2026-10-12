"""Reconstruct all descriptive component effects and saved fit diagnostics."""
import csv
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from independent_sealed_audit import authenticate, point, reconstruct_flags, f1, sha

PACKAGE = Path(__file__).resolve().parent.parent
DEST = PACKAGE/"diagnostics"


def number(value): return float(value) if value not in (None, "") else np.nan


def main():
    seal, pop = authenticate()
    status = json.loads((DEST/"COMPONENT_STATUS.json").read_text())
    assert status["seal_sha256"] == sha(PACKAGE/"PREDICTION_SEAL.json") and status["primary_M_unchanged"] == 7 and status["no_refitting"]
    assert status["script_sha256"] == sha(DEST/"evaluate_component_contrasts.py")
    with np.load(PACKAGE/"evaluation/BOOTSTRAP_DRAWS.npz") as z: weights = z["multiplicities"].astype(float)
    assert sha(PACKAGE/"evaluation/BOOTSTRAP_DRAWS.npz") == status["bootstrap_draws_sha256"]
    off, y, cls = pop["offsets"], pop["y"].astype(bool), pop["classification"]
    length = np.diff(off); truth = np.add.reduceat(y.astype(int), off[:-1]); groups, inverse = np.unique(pop["group"], return_inverse=True)
    step_group = np.repeat(inverse, length)
    masks = dict(ALL_NONCONTROL=cls != "correct", SYNTHETIC_CLEAN=cls == "correct", ERROR_BEARING=truth > 0, ALL_TRUE_ZERO=truth == 0)
    contrast_rows = list(csv.DictReader((DEST/"COMPONENT_CONTRASTS.csv").open()))
    change_rows = list(csv.DictReader((DEST/"COMPONENT_COUNT_CHANGES.csv").open()))
    checked, changed_checked, max_error = 0, 0, 0.
    for bank in ("N16", "N56"):
        with np.load(PACKAGE/f"predictions/{bank}.npz") as z: data = {k: z[k] for k in z.files}
        for left, right in (("A_MOM_MEDIAN", "A_MOM_MEAN"), ("B_TRIPLE_MEDIAN", "B_TRIPLE_MEAN"), ("D_ZERO_MEDIAN", "D_ZERO_MEAN")):
            assert np.array_equal(data["posterior__"+left], data["posterior__"+right], equal_nan=True)
        for method in ("IU_PCR", "LSML_CONT"):
            placement = {}
            for _, left, right in status["comparisons"]:
                for arm in (left, right):
                    if arm not in placement:
                        count = data[f"count__{method}__{arm}"]
                        flags, covered = reconstruct_flags(data["rank_"+method], off, count)
                        placement[arm] = count, flags, covered
            for component, left, right in status["comparisons"]:
                lc, lf, lv = placement[left]; rc, rf, rv = placement[right]; common = lv & rv
                for scope, selected in masks.items():
                    take = selected & common; use = np.repeat(take, length); expected, replicas = [], []
                    for count, flags in ((lc, lf), (rc, rf)):
                        expected.append(point(pop, count, flags, common, selected))
                        error = count-truth
                        answer_sums = np.column_stack([np.bincount(inverse[take], weights=v[take], minlength=len(groups)) for v in (abs(error), error)])
                        den = weights@np.bincount(inverse[take], minlength=len(groups))
                        means = np.divide(weights@answer_sums, den[:, None], out=np.full((len(weights), 2), np.nan), where=den[:, None] > 0)
                        confusion = np.column_stack([np.bincount(step_group[use & m], minlength=len(groups)) for m in (y & flags, ~y & flags, y & ~flags, ~y & ~flags)])
                        tp, fp, fn, tn = (weights@confusion).T
                        sample = dict(count_mae=means[:, 0], count_bias=means[:, 1], prm_score=(f1(tp, fp, fn)+f1(tn, fp, fn))/2,
                                      error_recall=np.divide(tp, tp+fn, out=np.full(len(tp), np.nan), where=tp+fn > 0))
                        for field, sub, value in (("zero_falseflag_rate", truth == 0, count > 0), ("error_answer_miss_rate", truth > 0, count == 0)):
                            active = take & sub
                            num = weights@np.bincount(inverse[active], weights=value[active].astype(int), minlength=len(groups))
                            den = weights@np.bincount(inverse[active], minlength=len(groups))
                            sample[field] = np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)
                        replicas.append(sample)
                    meta = bank, method, component, scope
                    change = next(r for r in change_rows if (r["bank"], r["method"], r["component"], r["population"]) == meta)
                    difference = lc-rc
                    assert int(change["n_answers"]) == take.sum()
                    assert int(change["n_count_changed"]) == (difference[take] != 0).sum()
                    assert int(change["n_count_increased"]) == (difference[take] > 0).sum()
                    assert int(change["n_count_decreased"]) == (difference[take] < 0).sum()
                    assert (change["full_bank_coverage"] == "True") == bool(common.all())
                    changed_checked += 1
                    for field in replicas[0]:
                        row = next(r for r in contrast_rows if (r["bank"], r["method"], r["component"], r["population"], r["metric"]) == meta+(field,))
                        if field == "prm_score" and scope in ("SYNTHETIC_CLEAN", "ALL_TRUE_ZERO"):
                            assert row["difference"] == row["lo95"] == row["hi95"] == "" and int(row["finite_draws"]) == 0
                        else:
                            delta = replicas[0][field]-replicas[1][field]; finite = delta[np.isfinite(delta)]
                            lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
                            value = expected[0][field]-expected[1][field]
                            for actual, target in ((number(row["difference"]), value), (number(row["lo95"]), lo), (number(row["hi95"]), hi)):
                                if np.isfinite(target): max_error = max(max_error, abs(actual-target)); assert abs(actual-target) < 1e-11
                                else: assert not np.isfinite(actual)
                            assert int(row["finite_draws"]) == len(finite)
                        checked += 1
    timing_rows = list(csv.DictReader((DEST/"TIMING.csv").open()))
    timing_summary = list(csv.DictReader((DEST/"TIMING_SUMMARY.csv").open()))
    for row in timing_rows:
        model = json.loads((PACKAGE/f"checkpoints/{row['bank']}_fold{row['fold']}/models.json").read_text())
        assert float(row["seconds"]) == model["timing"][row["timing_key"]]
    for row in timing_summary:
        values = [float(r["seconds"]) for r in timing_rows if (r["bank"], r["timing_key"]) == (row["bank"], row["timing_key"])]
        assert int(row["n_recorded_folds"]) == len(values)
        for key, value in (("min_s", min(values)), ("median_s", np.median(values)), ("max_s", max(values)), ("sum_s", sum(values))): assert abs(float(row[key])-value) < 1e-12
    receipt = dict(status="PASS", comparisons=len(status["comparisons"]), component_contrast_rows_checked=checked,
                   count_change_rows_checked=changed_checked, max_point_and_paired_interval_error=max_error,
                   posterior_identical_for_mean_median=True, partial_common_coverage_checked=True,
                   clean_strata_PRMScore_suppressed=True, timing_rows_checked=len(timing_rows), timing_summary_rows_checked=len(timing_summary),
                   timing_only_authenticated_saved_measurements=True, no_refitting_or_selection=True,
                   component_status_sha256=sha(DEST/"COMPONENT_STATUS.json"), script_sha256=sha(__file__),
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (PACKAGE/"review/INDEPENDENT_COMPONENT_AUDIT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__": main()
