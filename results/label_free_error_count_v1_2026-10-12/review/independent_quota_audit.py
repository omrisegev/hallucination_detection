"""Authenticate and reconstruct quota diagnostics; no study-model fitting."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.special import expit

PACKAGE = Path(__file__).resolve().parent.parent
DEST = PACKAGE/"diagnostics"
ROOT = PACKAGE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def main():
    seal = json.loads((DEST/"QUOTA_SEAL.json").read_text())
    for name, record in seal["files"].items(): assert sha(DEST/name) == record["sha256"], name
    assert not any(name.endswith(".log") for name in seal["files"])
    freeze = json.loads((DEST/"QUOTA_INPUT_FREEZE.json").read_text())
    for name, record in freeze["inputs"].items(): assert sha(ROOT/name) == record["sha256"], name
    for name, expected in freeze["code"].items(): assert sha(ROOT/name) == expected, name
    protocol = json.loads((DEST/"QUOTA_PROTOCOL.json").read_text())
    assert protocol["candidate_count_unchanged"] == 7 and protocol["relative_quotas"] == [.1, .2, .3]
    seeds = protocol["permutations"]["seeds"]; assert seeds == list(range(2026101200, 2026101210))
    rows = list(csv.DictReader((DEST/"QUOTA_RESULTS.csv").open()))
    states = json.loads((DEST/"QUOTA_MODELS.json").read_text())
    source = ROOT/"results/advisor_completion_2026-10-11/inputs/population.npz"
    with np.load(source) as z: pop = {key: z[key] for key in ("idx", "offsets", "fold", "group")}
    checked, null_error, statistic_error, score_error = 0, 0., 0., 0.
    full, shuffled = [], []
    for folder in sorted((DEST/"checkpoints").iterdir()):
        bank, fold = ("N16", "FULL") if folder.name == "FULL_REFERENCE_N16" else (folder.name.split("_fold")[0], folder.name.split("_fold")[1])
        with np.load(folder/"relative_votes.npz") as z: data = {key: z[key] for key in z.files}
        off = data["offsets"]; length = np.diff(off)
        for quota in (.1, .2, .3):
            votes = data[f"votes_q{int(quota*100)}"]
            k = np.ceil(quota*length).astype(int)
            count = np.add.reduceat((votes > 0).astype(int), off[:-1], axis=0)
            assert np.array_equal(count, np.broadcast_to(k[:, None], count.shape))
        q20 = data["votes_q20"]
        for seed in seeds:
            vv = data[f"permuted_{seed}"]
            assert np.array_equal(np.add.reduceat((vv > 0).astype(int), off[:-1], axis=0),
                                  np.add.reduceat((q20 > 0).astype(int), off[:-1], axis=0))
        selected = [r for r in rows if r["bank"] == bank and r["fold"] == fold and r["scenario"] in ("RELATIVE_QUOTA", "PERMUTED_QUOTA")]
        for row in selected:
            quota = float(row["quota"])
            votes = data[f"votes_q{int(quota*100)}"] if row["scenario"] == "RELATIVE_QUOTA" else data[f"permuted_{row['permutation_seed']}"]
            masks = {"ALL": np.ones(len(length), bool), "1-5": length <= 5, "6-10": (length >= 6) & (length <= 10),
                     "11-20": (length >= 11) & (length <= 20), "21+": length >= 21}
            choose = masks[row["length_bin"]]; use = np.repeat(choose, length); vv = votes[use]
            n = length[choose]; p = np.ceil(quota*n)/n; pbar = np.average(p, weights=n)
            pair = 4*np.average((p-pbar)**2, weights=n); third = 8*np.average((p-pbar)**3, weights=n)
            null_error = max(null_error, abs(pair-float(row["permutation_null_pair_covariance"])), abs(third-float(row["permutation_null_distinct_third"])))
            rate = float((vv > 0).mean()); centered = vv-vv.mean(0); cov = centered.T@centered/len(vv)
            pairs = cov[np.triu_indices(vv.shape[1], 1)]
            statistic_error = max(statistic_error, abs(rate-float(row["actual_mean_vote_rate"])),
                                  abs(pairs.mean()-float(row["offdiag_covariance_mean"])))
            if row["fit_status"] != "PASS": assert row["prevalence_hat"] == ""
            if fold == "FULL" and row["length_bin"] == "ALL" and row["estimator"] == "STRICT":
                if row["scenario"] == "RELATIVE_QUOTA": full.append(dict(quota=quota, actual_rate=rate, prevalence=float(row["prevalence_hat"])))
                else: shuffled.append(float(row["prevalence_hat"]))
            checked += 1
        if fold == "FULL": continue
        part = int(fold); test, train = np.flatnonzero(pop["fold"] == part), np.flatnonzero(pop["fold"] != part)
        assert not set(pop["group"][train]) & set(pop["group"][test])
        with np.load(folder/"absolute_heldout.npz") as z: held = {key: z[key] for key in z.files}
        assert np.array_equal(held["idx"], pop["idx"][test])
        assert np.array_equal(np.diff(held["offsets"]), np.diff(pop["offsets"])[test])
        assert np.array_equal(held["ABS_ZERO_votes"], np.where(held["absolute_scores"] > 0, 1, -1))
        representation = json.loads((folder/"ABSOLUTE_REPRESENTATION.json").read_text())["model"]
        assert np.array_equal(held["ABS_Q80_votes"], np.where(held["absolute_scores"] > representation["absolute"]["threshold"], 1, -1))
        with np.load(PACKAGE/f"checkpoints/{bank}_fold{part}/predictions.npz") as z:
            for method in ("IU_PCR", "LSML_CONT"):
                score_error = max(score_error, float(abs(held[method+"_score"]-z[method+"_score"]).max()))
            assert np.array_equal(held["ABS_Q80_count"], z["count__A_MOM_MEAN"])
        for scenario in ("ABS_Q80", "ABS_ZERO"):
            relevant = [item for item in states.values() if item["metadata"] == dict(bank=bank, fold=part, scope="OOF_DONOR", scenario=scenario,
                                                                                   quota=None, permutation_seed=None, length_bin="ALL") and item["model"].get("method") == "STRICT_ALL_GROUP_MOM"]
            assert len(relevant) == 1
            state = relevant[0]["model"]
            if state["status"] != "PASS":
                assert (held[scenario+"_count"] == -1).all() and np.isnan(held[scenario+"_posterior"]).all()
                continue
            votes = held[scenario+"_votes"]
            psi, eta, pi = np.asarray(state["psi"]), np.asarray(state["eta"]), state["prevalence"]
            log0 = np.where(votes > 0, np.log1p(-eta), np.log(eta)).sum(1)
            log1 = np.where(votes > 0, np.log(psi), np.log1p(-psi)).sum(1)
            probability = expit(np.log(pi)-np.log1p(-pi)+log1-log0)
            assert np.allclose(probability, held[scenario+"_posterior"], rtol=1e-12, atol=1e-14)
            assert np.array_equal(np.floor(np.add.reduceat(probability, held["offsets"][:-1])+.5).astype(int), held[scenario+"_count"])
    assert checked == 550 and len(rows) == 590 and max(null_error, statistic_error, score_error) < 1e-10
    teacher = json.loads((DEST/"EXACT_TEACHER.json").read_text())
    assert teacher["status"] == "PASS" and all(r["absolute_prevalence_error"] < 1e-8 for r in teacher["rows"])
    receipt = dict(status="PASS", diagnostic_seal_sha256=sha(DEST/"QUOTA_SEAL.json"), immutable_files_verified=len(seal["files"]),
                   quota_rows_reconstructed=checked, all_quota_and_shuffle_cardinalities_exact=True, max_analytic_null_error=null_error,
                   max_vote_statistics_error=statistic_error, archived_ranking_vs_primary_max_error=score_error,
                   ABS_Q80_counts_exact_primary_A_parity=True, frozen_zero_threshold_verified=True, failures_remain_missing=True,
                   no_fitting_performed=True, no_real_label_fields_loaded=True, full_reference_transductive=True,
                   primary_candidate_count=7, full_reference_quota_sweep=full,
                   full_reference_permuted_pi_range=[min(shuffled), max(shuffled)],
                   conclusion="Quota sensitivity exists in this bounded development diagnostic; permutation instability excludes a deterministic prevalence=quota interpretation.",
                   script_sha256=sha(__file__), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (PACKAGE/"review/INDEPENDENT_QUOTA_AUDIT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__": main()
