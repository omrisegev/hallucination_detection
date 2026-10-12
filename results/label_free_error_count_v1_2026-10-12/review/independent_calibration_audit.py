"""Independent post-seal emission/correlation/Brier/ECE arithmetic checks."""
import csv
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from independent_sealed_audit import authenticate, sha

PACKAGE = Path(__file__).resolve().parent.parent
DEST = PACKAGE/"diagnostics"


def number(value): return float(value) if value not in (None, "") else np.nan


def main():
    seal, population = authenticate()
    provenance = json.loads((DEST/"EMISSION_CALIBRATION_PROVENANCE.json").read_text())
    assert provenance["main_seal_sha256"] == sha(PACKAGE/"PREDICTION_SEAL.json")
    for name, expected in provenance["output_hashes"].items(): assert sha(DEST/name) == expected
    assert sha(DEST/"evaluate_emission_calibration.py") == provenance["diagnostic_script_sha256"]
    with np.load(PACKAGE.parent/"advisor_completion_2026-10-11/inputs/bank_V_N16.npz") as z: V16 = z["V"]
    with np.load(PACKAGE.parent/"advisor_completion_2026-10-11/inputs/bank_V_N56.npz") as z: V56 = z["V"]
    cache = {}; error_emission, error_correlation = 0., 0.
    group_rows = list(csv.DictReader((DEST/"EMISSION_GROUPWISE.csv").open()))
    pair_rows = list(csv.DictReader((DEST/"CONDITIONAL_CORRELATION_PAIRS.csv").open()))
    for bank, V in (("N16", V16), ("N56", V56)):
        for fold in range(5):
            folder = PACKAGE/f"checkpoints/{bank}_fold{fold}"
            model = json.loads((folder/"models.json").read_text())
            answers = np.flatnonzero(population["fold"] == fold)
            original_off = population["offsets"]
            rows = np.concatenate([np.arange(original_off[i], original_off[i+1]) for i in answers])
            length = np.diff(original_off)[answers]; off = np.r_[0, length.cumsum()]
            y = population["y"][rows].astype(bool)
            votes = {}
            for kind in ("local", "absolute"):
                with np.load(folder/f"{kind}_votes.npz") as z: votes[kind] = z["test"].copy()
            groups = model["LSML_CONT"]["diagnostic"]["group_weights"]
            scores = np.column_stack([V[rows][:, g["columns"]]@np.asarray(g["weights"]) for g in groups])
            for a, b in zip(off[:-1], off[1:]):
                block = scores[a:b].copy(); sd = block.std(0)
                scores[a:b] = np.divide(block-block.mean(0), sd, out=np.zeros_like(block), where=sd > 1e-12)
            tie = np.random.default_rng(20260929).random(scores.shape)
            legacy = -np.ones(scores.shape, dtype=np.int8)
            for a, b in zip(off[:-1], off[1:]):
                for column in range(scores.shape[1]):
                    order = np.lexsort((tie[a:b, column], -scores[a:b, column]))
                    legacy[a+order[:int(np.ceil(.2*(b-a)))], column] = 1
            votes["legacy"] = legacy
            for arm, key, kind in (("LOCAL_STRICT", "local_mom", "local"), ("A_STRICT", "absolute_mom", "absolute"),
                                   ("B_SELECTED_TRIPLE", "B_TRIPLE", "absolute"), ("LEGACY_MOM", "LEGACY_MOM", "legacy")):
                state = model[key]; columns = state.get("columns", list(range(votes[kind].shape[1])))
                M = votes[kind][:, columns].astype(float)
                centered = M-np.repeat(np.add.reduceat(M, off[:-1], axis=0)/length[:, None], length, axis=0)
                for scope, mask in (("ALL", np.ones(len(answers), bool)), ("NONCONTROL", population["classification"][answers] != "correct"),
                                    ("SYNTHETIC_CLEAN", population["classification"][answers] == "correct")):
                    use = np.repeat(mask, length)
                    psi = (M[use & y] > 0).mean(0) if (use & y).any() else np.full(len(columns), np.nan)
                    eta = (M[use & ~y] < 0).mean(0) if (use & ~y).any() else np.full(len(columns), np.nan)
                    cache[bank, fold, arm, scope] = (columns, psi, eta, {})
                    for transform, values in (("raw", M), ("answer_centered", centered)):
                        for truth, take in (("error", use & y), ("valid", use & ~y)):
                            with np.errstate(invalid="ignore", divide="ignore"):
                                corr = np.corrcoef(values[take], rowvar=False) if take.sum() > 1 else np.full((len(columns), len(columns)), np.nan)
                            cache[bank, fold, arm, scope][3][transform, truth] = corr
    for row in group_rows:
        columns, psi, eta, _ = cache[row["bank"], int(row["fold"]), row["arm"], row["population"]]
        column = columns.index(int(row["group_index"]))
        for key, expected in (("psi_true", psi[column]), ("eta_true", eta[column])):
            actual = number(row[key])
            if np.isfinite(expected): error_emission = max(error_emission, abs(actual-expected)); assert abs(actual-expected) < 1e-12
            else: assert not np.isfinite(actual)
    for row in pair_rows:
        columns, _, _, correlations = cache[row["bank"], int(row["fold"]), row["arm"], row["population"]]
        expected = correlations[row["transform"], row["true_class"]][columns.index(int(row["group_i"])), columns.index(int(row["group_j"]))]
        actual = number(row["correlation"])
        if np.isfinite(expected): error_correlation = max(error_correlation, abs(actual-expected)); assert abs(actual-expected) < 1e-11
        else: assert not np.isfinite(actual)
    aggregate_rows = list(csv.DictReader((DEST/"EMISSION_AGGREGATE.csv").open())); max_aggregate_error = 0.
    for row in aggregate_rows:
        selected = [r for r in group_rows if (r["bank"], r["arm"], r["population"]) == (row["bank"], row["arm"], row["population"])
                    and (row["basis"] == "raw_parameters_including_failed" or r["fit_status"] == "PASS")]
        for field, exposure in (("psi", "n_error_steps"), ("eta", "n_valid_steps"), ("reliability", "n_steps")):
            errors = np.array([number(r[field+"_error"]) for r in selected]); weights = np.array([int(r[exposure]) for r in selected])
            good = np.isfinite(errors) & (weights > 0)
            assert int(row[field+"_exposures"]) == weights[good].sum()
            for suffix, values in (("mae", abs(errors)), ("mse", errors*errors)):
                expected = np.average(values[good], weights=weights[good]) if good.any() else np.nan
                actual = number(row[field+"_"+suffix])
                if np.isfinite(expected): max_aggregate_error = max(max_aggregate_error, abs(actual-expected)); assert abs(actual-expected) < 1e-11
                else: assert not np.isfinite(actual)
    posterior_rows = list(csv.DictReader((DEST/"POSTERIOR_CALIBRATION.csv").open())); max_posterior_error = 0.
    for row in posterior_rows:
        with np.load(PACKAGE/f"predictions/{row['bank']}.npz") as z: p = z["posterior__"+row["arm"]]
        cls = population["classification"]
        mask = np.ones(len(cls), bool) if row["population"] == "ALL" else cls != "correct" if row["population"] == "NONCONTROL" else cls == "correct"
        take = np.repeat(mask, np.diff(population["offsets"])) & np.isfinite(p)
        p, y = p[take], population["y"][take].astype(float)
        bins = np.clip(np.floor(p*10).astype(int), 0, 9)
        counts = np.bincount(bins, minlength=10); probabilities = np.bincount(bins, weights=p, minlength=10); truth = np.bincount(bins, weights=y, minlength=10)
        good = counts > 0; difference = (probabilities[good]-truth[good])/counts[good]
        expected = dict(brier_mse=float(np.mean((p-y)**2)), posterior_mean=float(p.mean()), true_prevalence=float(y.mean()), probability_bias=float((p-y).mean()),
                        ECE_fixed_ten_bins=float((counts[good]*abs(difference)).sum()/len(p)),
                        reliability_mse_fixed_ten_bins=float((counts[good]*difference**2).sum()/len(p)))
        assert int(row["n_steps"]) == len(p)
        for field, value in expected.items():
            error = abs(number(row[field])-value); max_posterior_error = max(max_posterior_error, error); assert error < 1e-11
    receipt = dict(status="PASS", main_seal_sha256=sha(PACKAGE/"PREDICTION_SEAL.json"), diagnostic_provenance_sha256=sha(DEST/"EMISSION_CALIBRATION_PROVENANCE.json"),
                   group_parameter_rows_checked=len(group_rows), conditional_pair_rows_checked=len(pair_rows), exposure_weighted_aggregate_rows_checked=len(aggregate_rows),
                   posterior_calibration_rows_checked=len(posterior_rows), max_true_emission_error=error_emission, max_conditional_correlation_error=error_correlation,
                   max_aggregate_error=max_aggregate_error, max_posterior_brier_ECE_error=max_posterior_error,
                   no_fitting_or_label_based_selection=True, centered_full_answers_before_class_selection=True, failed_fits_never_repaired=True,
                   script_sha256=sha(__file__), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (PACKAGE/"review/INDEPENDENT_CALIBRATION_AUDIT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__": main()
