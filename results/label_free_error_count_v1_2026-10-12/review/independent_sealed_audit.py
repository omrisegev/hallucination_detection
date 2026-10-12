"""Reconstruct sealed point arithmetic/PMFs and primary bootstrap gates independently.

No estimator fitting and no imports from study implementation helpers.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
sys.dont_write_bytecode = True
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

PACKAGE = Path(__file__).resolve().parent.parent
ROOT = PACKAGE.parents[1]
FROZEN = ROOT/"results/advisor_completion_2026-10-11"
ARMS = ("LOCAL_MOM_MEAN", "A_MOM_MEAN", "A_MOM_MEDIAN", "B_TRIPLE_MEAN", "B_TRIPLE_MEDIAN", "D_ZERO_MEAN", "D_ZERO_MEDIAN")
CONTROLS = ("Q80", "R2_COUNT", "ENTROPY_GATE_Q80", "LEGACY_MOM")
METHODS = ("IU_PCR", "LSML_CONT")
TYPES = ("redundency", "circular", "counterfactual", "step_contradiction", "domain_inconsistency", "confidence", "missing_condition", "deception", "multi_solutions")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def authenticate():
    seal = json.loads((PACKAGE/"PREDICTION_SEAL.json").read_text())
    for relative, record in seal["files"].items():
        assert sha(PACKAGE/relative) == record["sha256"], relative
    assert sha(PACKAGE/"PROTOCOL.json") == seal["protocol_sha256"]
    for relative, expected in seal["code_sha256"].items():
        assert sha(ROOT/relative) == expected, relative
    assert "INPUT_FREEZE.json" in seal["files"]
    freeze = json.loads((PACKAGE/"INPUT_FREEZE.json").read_text())
    label_path = FROZEN/"inputs/population.npz"
    assert sha(label_path) == freeze["inputs"][str(label_path.relative_to(ROOT))]["sha256"]
    with np.load(label_path, allow_pickle=False) as z:
        population = {k: z[k].copy() for k in ("idx", "offsets", "group", "fold", "y", "classification")}
    off = population["offsets"]
    assert len(population["idx"]) == 6969 and off[-1] == 94203 and len(np.unique(population["group"])) == 707
    assert len(set(population["idx"])) == len(population["idx"]) and (np.diff(off) > 0).all()
    for group in np.unique(population["group"]):
        assert len(np.unique(population["fold"][population["group"] == group])) == 1
    return seal, population


def f1(tp, fp, fn):
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)
    return np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp), where=2*tp+fp+fn > 0)


def partial(x, y, length):
    if len(x) < 3: return np.nan
    x, y, length = (rankdata(v, method="average") for v in (x, y, length))
    design = np.column_stack((np.ones(len(x)), length))
    x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(x@y/np.sqrt((x@x)*(y@y))) if x@x > 1e-12 and y@y > 1e-12 else np.nan


def reconstruct_flags(rank, off, count):
    flags = np.zeros(off[-1], bool)
    covered = count >= 0
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        if covered[i]:
            assert 0 <= count[i] <= b-a and np.array_equal(np.sort(rank[a:b]), np.arange(a, b))
            flags[rank[a:a+count[i]]] = True
            assert flags[a:b].sum() == count[i]
    return flags, covered


def point(pop, count, flags, covered, selected, ms=False):
    off, y, group = pop["offsets"], pop["y"].astype(bool), pop["group"]
    length = np.diff(off); truth = np.add.reduceat(y.astype(int), off[:-1]); take = selected & covered
    use = np.repeat(take, length)
    tp, fp, fn, tn = (int((use & condition).sum()) for condition in (y & flags, ~y & flags, y & ~flags, ~y & ~flags))
    score = float(f1(tn, fp, fn) if ms else (f1(tp, fp, fn)+f1(tn, fp, fn))/2) if take.any() else np.nan
    error = count-truth
    return dict(n_answers=int(take.sum()), n_steps=int(use.sum()), n_source_groups=len(np.unique(group[take])),
                n_expected_answers=int(selected.sum()), n_failed_answers=int((selected & ~covered).sum()),
                coverage=float(take.sum()/max(selected.sum(), 1)), prm_score=score,
                count_mae=float(np.abs(error[take]).mean()) if take.any() else np.nan,
                count_bias=float(error[take].mean()) if take.any() else np.nan,
                count_mean=float(count[take].mean()) if take.any() else np.nan,
                true_count_mean=float(truth[take].mean()) if take.any() else np.nan,
                error_recall=tp/(tp+fn) if tp+fn else np.nan,
                error_precision=tp/(tp+fp) if tp+fp else np.nan,
                step_flag_rate=(tp+fp)/(tp+fp+fn+tn) if use.any() else np.nan,
                zero_falseflag_rate=float((count[take & (truth == 0)] > 0).mean()) if (take & (truth == 0)).any() else np.nan,
                error_answer_miss_rate=float((count[take & (truth > 0)] == 0).mean()) if (take & (truth > 0)).any() else np.nan,
                tp_error=tp, fp_error=fp, fn_error=fn, tn_valid=tn)


def pb(probability):
    result = np.ones(1)
    for p in probability: result = np.convolve(result, [1-p, p])
    return result


def class_logs(votes, model):
    votes = votes[:, model.get("columns", list(range(len(model["psi"]))))]
    psi, eta = np.asarray(model["psi"]), np.asarray(model["eta"])
    return (np.where(votes > 0, np.log1p(-eta), np.log(eta)).sum(1),
            np.where(votes > 0, np.log(psi), np.log1p(-psi)).sum(1))


def closed_triple(mu, pairs, third):
    covariance = pairs-mu[:, [0, 0, 1]]*mu[:, [1, 2, 2]]
    centered_third = third-mu[:, 0]*pairs[:, 2]-mu[:, 1]*pairs[:, 1]-mu[:, 2]*pairs[:, 0]+2*np.prod(mu, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t0 = np.sqrt(covariance[:, 0]*covariance[:, 1]/covariance[:, 2])
        loading = np.column_stack((t0, covariance[:, 0]/t0, covariance[:, 1]/t0))
        skew = centered_third/np.prod(loading, axis=1)
        pi = .5*(1-skew/np.sqrt(skew*skew+4))
        delta = loading/(2*np.sqrt(pi*(1-pi)))[:, None]
        psi = .5*(1+mu)+delta*(1-pi)[:, None]
        eta = .5*(1-mu)+delta*pi[:, None]
    valid = ((covariance > 0).all(1) & (pi > 1e-6) & (pi < 1-1e-6) & np.isfinite(pi) &
             ((psi > 1e-6) & (psi < 1-1e-6) & (eta > 1e-6) & (eta < 1-1e-6)).all(1))
    return valid, pi, covariance


def audit_models(pop):
    largest_pmf_error, largest_posterior_error, largest_ll_error, distributions = 0., 0., 0., 0
    bootstrap_triples, bootstrap_error = 0, 0.
    off, fold, idx, group = (pop[k] for k in ("offsets", "fold", "idx", "group"))
    fold_status = {}
    for bank in ("N16", "N56"):
        for part in range(5):
            folder = PACKAGE/f"checkpoints/{bank}_fold{part}"
            model = json.loads((folder/"models.json").read_text()); identity = json.loads((folder/"TRAINING_IDENTITY.json").read_text())
            test, train = np.flatnonzero(fold == part), np.flatnonzero(fold != part)
            assert identity["train_ids"] == idx[train].tolist() and identity["test_ids"] == idx[test].tolist()
            assert not set(group[train]) & set(group[test])
            with np.load(folder/"predictions.npz") as z: local = {k: z[k] for k in z.files}
            assert np.array_equal(local["idx"], idx[test])
            assert np.array_equal(np.diff(local["offsets"]), np.diff(off)[test])
            for arm, status in model["status"].items():
                failed = status["status"] != "PASS"
                assert bool(np.all(local["count__"+arm] == -1)) == failed, (bank, part, arm)
                if failed: assert np.isnan(local["posterior__"+arm]).all()
            fold_status[f"{bank}_{part}"] = {a: s["status"] for a, s in model["status"].items()}
            for prefix, mean_arm, median_arm, model_key, vote_kind in (
                    ("local", "LOCAL_MOM_MEAN", None, "local_mom", "local"),
                    ("absolute", "A_MOM_MEAN", "A_MOM_MEDIAN", "absolute_mom", "absolute"),
                    ("B", "B_TRIPLE_MEAN", "B_TRIPLE_MEDIAN", "B_TRIPLE", "absolute"),
                    ("D", "D_ZERO_MEAN", "D_ZERO_MEDIAN", "D_ZERO", "absolute")):
                if prefix+"_pmf" not in local: continue
                state = model[model_key]; assert state["status"] == "PASS"
                with np.load(folder/f"{vote_kind}_votes.npz") as z: testing, training = z["test"], z["train"]
                log0, log1 = class_logs(testing, state)
                if prefix == "D":
                    for key in ("columns", "psi", "eta"): assert state[key] == model["B_TRIPLE"][key]
                    rho, theta = state["rho"], state["theta"]
                    active = expit(np.log(theta)-np.log1p(-theta)+log1-log0)
                    L0 = np.add.reduceat(log0, local["offsets"][:-1])
                    L1 = np.add.reduceat(np.logaddexp(np.log1p(-theta)+log0, np.log(theta)+log1), local["offsets"][:-1])
                    structural = expit(np.log(rho)-np.log1p(-rho)+L0-L1) if rho > 0 else np.zeros(len(test))
                    probability = active*np.repeat(1-structural, np.diff(local["offsets"]))
                    assert np.allclose(active, local["D_active_p"], rtol=1e-12, atol=1e-14)
                    assert np.allclose(structural, local["D_structural_clean_p"], rtol=1e-12, atol=1e-14)
                    tr0, tr1 = class_logs(training, state)
                    toff = np.r_[0, np.cumsum(np.diff(off)[train])]
                    clean_ll = np.add.reduceat(tr0, toff[:-1])
                    active_ll = np.add.reduceat(np.logaddexp(np.log1p(-theta)+tr0, np.log(theta)+tr1), toff[:-1])
                    ll = float(np.logaddexp(np.log(rho)+clean_ll, np.log1p(-rho)+active_ll).mean())
                    largest_ll_error = max(largest_ll_error, abs(ll-state["average_log_likelihood"]))
                    assert abs(ll-state["average_log_likelihood"]) < 1e-10
                    assert state.get("promotable", False) and state["hessian"]["healthy"]
                else:
                    probability = expit(np.log(state["prevalence"])-np.log1p(-state["prevalence"])+log1-log0)
                largest_posterior_error = max(largest_posterior_error, float(abs(probability-local["posterior__"+mean_arm]).max()))
                assert np.allclose(probability, local["posterior__"+mean_arm], rtol=1e-12, atol=1e-14)
                poff = local[prefix+"_pmf_offsets"]; assert np.array_equal(np.diff(poff), np.diff(local["offsets"])+1)
                for i, (a, b) in enumerate(zip(local["offsets"][:-1], local["offsets"][1:])):
                    expected = pb(active[a:b])*(1-structural[i]) if prefix == "D" else pb(probability[a:b])
                    if prefix == "D": expected[0] += structural[i]
                    saved = local[prefix+"_pmf"][poff[i]:poff[i+1]]
                    largest_pmf_error = max(largest_pmf_error, float(abs(saved-expected).max()))
                    assert np.allclose(saved, expected, rtol=1e-11, atol=1e-13)
                    assert abs(saved.sum()-1) < 1e-12 and (saved >= 0).all()
                    assert local["count__"+mean_arm][i] == int(np.floor(probability[a:b].sum()+.5))
                    if median_arm: assert local["count__"+median_arm][i] == int(np.searchsorted(np.cumsum(expected), .5, side="left"))
                    distributions += 1
            triple = model["B_TRIPLE"]
            with np.load(folder/"absolute_votes.npz") as z: votes = z["train"].astype(float)
            source_ids, source_inverse = np.unique(np.repeat(group[train], np.diff(off)[train]), return_inverse=True)
            source_counts = np.bincount(source_inverse)
            multiplicity = np.random.default_rng(20261012+part).multinomial(len(source_ids), np.full(len(source_ids), 1/len(source_ids)), size=200)
            assert np.array_equal(triple["bootstrap"]["source_ids"], source_ids)
            assert np.array_equal(triple["bootstrap"]["source_step_counts"], source_counts)
            assert np.array_equal(triple["bootstrap"]["resampled_step_counts"], multiplicity@source_counts)
            assert hashlib.sha256(np.asarray(multiplicity, dtype="<i8").tobytes()).hexdigest() == triple["bootstrap"]["multiplicity_sha256"]
            eligible = [row for row in triple["triple_table"] if row["eligible"]]
            selected = min(eligible, key=lambda r: (r["normalized_prevalence_IQR"], tuple(tuple(t) for t in r["member_indices"])))
            assert selected["columns"] == triple["columns"]
            for row in triple["triple_table"]:
                valid = np.asarray(row["bootstrap_valid_mask"], bool)
                assert valid.sum() == row["bootstrap_valid"]
                x = votes[:, row["columns"]]
                features = np.column_stack((x, x[:, [0, 0, 1]]*x[:, [1, 2, 2]], np.prod(x, axis=1)))
                sums = np.column_stack([np.bincount(source_inverse, weights=features[:, j], minlength=len(source_ids)) for j in range(7)])
                boot = (multiplicity@sums)/(multiplicity@source_counts)[:, None]
                independent_valid, pi, covariance = closed_triple(boot[:, :3], boot[:, 3:6], boot[:, 6])
                assert np.array_equal(independent_valid, valid)
                lower = np.quantile(covariance, .025, axis=0)
                assert np.allclose(lower, row["covariance_lower025"], rtol=1e-11, atol=1e-13)
                saved = np.asarray([np.nan if v is None else v for v in row["bootstrap_prevalence"]])
                if valid.any():
                    error = float(abs(pi[valid]-saved[valid]).max()); bootstrap_error = max(bootstrap_error, error)
                    assert error < 1e-10
                assert row["eligible"] == bool(row["full_fit"]["status"] == "PASS" and valid.sum() >= 180 and (lower > 0).all())
                bootstrap_triples += 1
                if row["eligible"]:
                    assert row["full_fit"]["status"] == "PASS" and row["bootstrap_valid"] >= 180 and min(row["covariance_lower025"]) > 0
                    sample = np.asarray([np.nan if v is None else v for v in row["bootstrap_prevalence"]])[valid]
                    expected = np.subtract(*np.quantile(sample, [.75, .25]))/np.sqrt(row["full_fit"]["prevalence"]*(1-row["full_fit"]["prevalence"]))
                    assert abs(expected-row["normalized_prevalence_IQR"]) < 1e-12
    return dict(count_distributions_checked=distributions, max_pmf_error=largest_pmf_error,
                max_posterior_error=largest_posterior_error, max_D_training_loglik_error=largest_ll_error,
                all_B_selected_minimum_eligible_triple=True, source_bootstrap_triples_independently_recomputed=bootstrap_triples,
                max_B_bootstrap_prevalence_error=bootstrap_error, all_B_source_bootstrap_draws_exact=True, fold_status=fold_status)


def main():
    started = time.time(); seal, pop = authenticate(); model_audit = audit_models(pop)
    off, cls, y = pop["offsets"], pop["classification"], pop["y"].astype(bool)
    length = np.diff(off); truth = np.add.reduceat(y.astype(int), off[:-1]); groups, inverse = np.unique(pop["group"], return_inverse=True)
    masks = dict(ALL_NONCONTROL=cls != "correct", ALL_ANSWERS=np.ones(len(cls), bool), SYNTHETIC_CLEAN=cls == "correct",
                 ALL_TRUE_ZERO=truth == 0, NONCONTROL_TRUE_ZERO=(truth == 0) & (cls != "correct"), ERROR_BEARING=truth > 0)
    masks.update({typ: cls == typ for typ in TYPES})
    for lo, hi in ((1, 5), (6, 10), (11, 20), (21, None)):
        masks[f"LENGTH_{lo}_{hi if hi is not None else 'PLUS'}"] = (cls != "correct") & (length >= lo) & ((length <= hi) if hi else True)
    rows, placements, full = [], {}, {}
    for bank in ("N16", "N56"):
        with np.load(PACKAGE/f"predictions/{bank}.npz") as archive: z = {k: archive[k] for k in archive.files}
        for key in ("idx", "offsets", "group", "fold"): assert np.array_equal(z[key], pop[key])
        for method in METHODS:
            rank, score = z["rank_"+method], z[method+"_score"]
            for a, b in zip(off[:-1], off[1:]): assert np.array_equal(rank[a:b], a+np.argsort(-score[a:b], kind="stable"))
            names = [key.split("__", 2)[2] for key in z if key.startswith(f"count__{method}__")]
            for arm in names:
                count = z[f"count__{method}__{arm}"]
                if arm in ARMS: assert np.array_equal(count, z["count__"+arm])
                flags, covered = reconstruct_flags(rank, off, count); placements[bank, method, arm] = (count, flags, covered)
                full[bank, method, arm] = bool(covered.all())
                for population, selected in masks.items():
                    result = point(pop, count, flags, covered, selected, ms=population == "multi_solutions")
                    if population in ("ALL_ANSWERS", "SYNTHETIC_CLEAN", "ALL_TRUE_ZERO", "NONCONTROL_TRUE_ZERO"): result["prm_score"] = np.nan
                    if population == "ALL_NONCONTROL": result["partial_spearman_length"] = partial(count[selected & covered], truth[selected & covered], length[selected & covered])
                    rows.append(dict(bank=bank, method=method, arm=arm, population=population, **result))
    expected_rows = {(r["bank"], r["method"], r["arm"], r["population"]): r for r in rows}
    comparisons = 0; max_point_error = 0.
    result_path = PACKAGE/"evaluation/RESULTS.json"
    data = json.loads(result_path.read_text()) if result_path.exists() else None
    if data:
        for r in data["metrics"]:
            key = r["bank"], r["method"], r["arm"], r["population"]
            if key not in expected_rows: continue
            for metric, expected in expected_rows[key].items():
                if metric in ("bank", "method", "arm", "population"): continue
                actual = r.get(metric)
                if not np.isfinite(expected): assert actual is None, (key, metric, actual)
                else:
                    error = abs(actual-expected); max_point_error = max(max_point_error, error)
                    assert error < 1e-10, (key, metric, actual, expected)
                comparisons += 1
    boot_receipt = None
    if data:
        with np.load(PACKAGE/"evaluation/BOOTSTRAP_DRAWS.npz") as z: multiplicities = z["multiplicities"]; assert np.array_equal(z["source_groups"], groups)
        assert multiplicities.shape == (20000, 707)
        expected = np.random.default_rng(20261012).multinomial(707, np.full(707, 1/707), size=20000).astype(multiplicities.dtype)
        assert np.array_equal(multiplicities, expected)
        weights = multiplicities.astype(float); alpha = .05/7; checked, max_interval_error, success = 0, 0., {}
        for arm in ARMS:
            gates = []
            for baseline, metric, side, threshold in (("Q80", "count_mae", "upper", 0), ("LEGACY_MOM", "count_mae", "upper", 0),
                                                       ("Q80", "prm_score", "lower", 0), ("Q80", "error_recall", "lower", -.01)):
                ac, af, av = placements["N16", "IU_PCR", arm]; bc, bf, bv = placements["N16", "IU_PCR", baseline]
                common = av & bv & masks["ALL_NONCONTROL"]
                replicas = []
                for count, flags in ((ac, af), (bc, bf)):
                    if metric == "count_mae":
                        sums = np.bincount(inverse[common], weights=abs(count[common]-truth[common]), minlength=707)
                        den = weights @ np.bincount(inverse[common], minlength=707)
                        replicas.append((weights@sums)/den)
                    else:
                        use = np.repeat(common, length); step_group = np.repeat(inverse, length)
                        stats = np.column_stack([np.bincount(step_group[use & mask], minlength=707) for mask in (y & flags, ~y & flags, y & ~flags, ~y & ~flags)])
                        tp, fp, fn, tn = (weights @ stats).T
                        replicas.append(tp/(tp+fn) if metric == "error_recall" else (f1(tp, fp, fn)+f1(tn, fp, fn))/2)
                delta = replicas[0]-replicas[1]
                low, high = np.quantile(delta, [alpha, 1-alpha]); lo95, hi95 = np.quantile(delta, [.025, .975])
                row = next(r for r in data["contrasts"] if (r["bank"], r["method"], r["arm"], r["baseline"], r["population"], r["metric"]) == ("N16", "IU_PCR", arm, baseline, "ALL_NONCONTROL", metric))
                error = max(abs(lo95-row["lo95"]), abs(hi95-row["hi95"]), abs(low-row["simultaneous_lower"]), abs(high-row["simultaneous_upper"]))
                max_interval_error = max(max_interval_error, error); assert error < 1e-10
                gates.append(bool(row["full_coverage"] and (high < threshold if side == "upper" else low > threshold)))
                checked += 1
            success[arm] = bool(full["N16", "IU_PCR", arm] and all(gates))
        assert data["success"]["M"] == 7 and data["success"]["minimum_recall_delta"] == -.01
        for decision in data["success"]["arms"]: assert decision["success"] == success[decision["arm"]]
        boot_receipt = dict(primary_contrasts_checked=checked, source_draws_exact_seed_match=True, max_interval_error=max_interval_error,
                            independent_success=success, any_success=any(success.values()))
    with (PACKAGE/"review/INDEPENDENT_POINT_METRICS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row))); writer.writeheader(); writer.writerows(clean(rows))
    receipt = dict(status="PASS" if data else "SAVED_ARITHMETIC_PASS_EVALUATOR_PENDING", seal_sha256=sha(PACKAGE/"PREDICTION_SEAL.json"),
                   population=dict(n_answers=len(cls), n_steps=len(y), n_groups=len(groups), n_noncontrol=int(masks["ALL_NONCONTROL"].sum()),
                                   n_synthetic=int(masks["SYNTHETIC_CLEAN"].sum()), n_true_zero=int(masks["ALL_TRUE_ZERO"].sum())),
                   model_arithmetic=model_audit, independently_reconstructed_metric_rows=len(rows), evaluator_values_compared=comparisons,
                   max_point_error=max_point_error, primary_bootstrap=boot_receipt, fitting_performed=False,
                   expected_label_hash_authenticated_before_read=True, independent_script_sha256=sha(__file__), runtime_s=time.time()-started,
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (PACKAGE/"review/INDEPENDENT_SEALED_AUDIT.json").write_text(json.dumps(clean(receipt), indent=2, allow_nan=False)+"\n")
    print(json.dumps(clean(receipt), indent=2, allow_nan=False))


if __name__ == "__main__": main()
