"""PRMB-only retrospective scoring of saved eleven-channel/depth fusion outputs.

No new fits or LLM inference. Cross-fold q80 of existing OOF scores is diagnostic:
it does not reconstruct nested donor model fitting. Raw and answer-z are separate
panels. Source-group bootstrap resamples frozen decisions (conditional intervals).
"""
from pathlib import Path
import ast
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/prmbench_lsml_month_audit_v2"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def f1(conf):
    tp, fp, tn, fn = np.moveaxis(np.asarray(conf, float), -1, 0)
    return (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2


def main():
    started = time.perf_counter()
    if OUT.exists(): raise FileExistsError("Preserve existing audit version")
    ev = ROOT / "results/localization_full_benchmark_v3/evaluation"
    jp, npz = ev / "JOINED.json", ev / "JOINED.npz"
    folds_path = ROOT / "results/localization_source_group_audit_v1/FOLDS_V2.json"
    meta_path = ROOT / "dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl"
    module_path = ROOT / "spectral_utils/prmbench.py"
    spec = importlib.util.spec_from_file_location("official_prm", module_path)
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    records = json.loads(jp.read_text(encoding="utf8"))["records"]
    with np.load(npz) as z: off, labels = z["offsets"], z["labels"]
    assert len(records) == 13769 and len(labels) == 145597
    meta = {r["idx"]: r for r in pickle.loads(meta_path.read_bytes()).values()}
    idx = np.array([i for i,r in enumerate(records) if r["cell"].startswith("prm")])
    assert len(idx) == 6969
    fm = json.loads(folds_path.read_text())["outer"]
    folds = np.array([fm[r["group_id"]] for r in records])
    groups, gi = np.unique([records[i]["group_id"] for i in idx], return_inverse=True)
    noncontrol = np.array([meta[records[i]["row_id"]]["classification"] != "correct" for i in idx])
    for i in idx:
        m = meta[records[i]["row_id"]]; a,b = off[i:i+2]
        assert m["n_steps"] == b-a
        expected = np.array([j+1 in m["error_steps"] for j in range(b-a)])
        np.testing.assert_array_equal(labels[a:b], expected)
    paths = [ROOT / ".worktrees/depth-feature-fusion-v1/results" / name / "STEP_SCORES.npz"
             for name in ("step_level_bank_baseline_v1", "bank_plus_depth_fusion_v1")]
    scores = {}
    for p in paths:
        with np.load(p) as z:
            for k in z.files:
                if k in ("step_length", "position"): continue
                if k in scores: np.testing.assert_allclose(scores[k], z[k])
                scores[k] = z[k].astype(float)
    np.testing.assert_allclose(scores["equal"], scores["bank11_equal"])
    np.testing.assert_allclose(scores["continuous"], scores["bank11_continuous"])
    del scores["equal"], scores["continuous"]
    # Reconstruct frozen L-SML weights without fitting, then ablate only their
    # magnitudes while retaining the learned partition. This isolates weighting
    # beyond simple balancing of the discovered groups.
    source_dir = ROOT / ".worktrees/depth-feature-fusion-v1"
    fit_path = source_dir / "results/step_level_bank_baseline_v1/RESULTS.json"
    helper_path = source_dir / "spectral_utils/lsml_gate_locator_research.py"
    bank_path = ROOT / ".worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz"
    tree = ast.parse(helper_path.read_text(encoding="utf8"))
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="answer_standardize"
             or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="EPS" for t in n.targets)]
    namespace = {"np":np}; exec(compile(ast.Module(body=nodes,type_ignores=[]),str(helper_path),"exec"),namespace)
    with np.load(bank_path) as z: values = namespace["answer_standardize"](z["level"],off)
    fitted = json.loads(fit_path.read_text(encoding="utf8"))["fits"]["continuous"]
    replay = np.full(len(labels),np.nan); group_equal = replay.copy()
    for fit in fitted:
        mask = np.repeat(folds==fit["fold"],np.diff(off)); w=np.asarray(fit["weights"])
        c=np.asarray(fit["meta"]["groups"]); u,cnt=np.unique(c,return_counts=True)
        gw=np.array([1/(len(u)*cnt[np.flatnonzero(u==g)[0]]) for g in c])
        replay[mask]=values[mask]@w; group_equal[mask]=values[mask]@gw
    np.testing.assert_allclose(replay,scores["bank11_continuous"],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(values.mean(1),scores["bank11_equal"],atol=1e-12,rtol=1e-12)
    scores["bank11_learned_partition_equal"] = group_equal
    rows, auc, conf, decisions, thresholds = [], {}, {}, {}, {}
    for name, score in scores.items():
        assert score.shape == labels.shape and np.isfinite(score).all()
        au = []
        for i in idx:
            a,b = off[i:i+2]; y = labels[a:b].astype(bool); pos = int(y.sum()); neg = len(y)-pos
            au.append((rankdata(score[a:b])[y].sum()-pos*(pos+1)/2)/(pos*neg) if pos and neg else np.nan)
        auc[name] = np.asarray(au)
        assert np.isfinite(auc[name]).sum() == 6030
        if name == "ct7_locator": assert abs(np.nanmean(au)-.7723966352864217) < 1e-10
        for normalization in ("raw", "answer_z"):
            s = score.copy()
            if normalization == "answer_z":
                for i in idx:
                    a,b = off[i:i+2]; sd = s[a:b].std()
                    s[a:b] = (s[a:b]-s[a:b].mean())/sd if sd > 1e-8 else 0
            valid = np.zeros(len(labels), bool); ts = []
            for k in range(5):
                train = np.concatenate([s[off[i]:off[i+1]] for i in idx if folds[i] != k])
                tau = np.quantile(train, .8); ts.append(float(tau))
                for i in idx[folds[idx] == k]:
                    a,b = off[i:i+2]; valid[a:b] = s[a:b] < tau
            key = name + "__" + normalization
            counts = np.zeros((len(groups),4), np.int64)
            for j,i in enumerate(idx):
                if not noncontrol[j]: continue
                a,b = off[i:i+2]; good = labels[a:b] == 0; v = valid[a:b]
                counts[gi[j]] += [(v&good).sum(),(v&~good).sum(),(~v&~good).sum(),(~v&good).sum()]
            out = official.prmbench_evaluate([{"idx":records[i]["row_id"],"labels":valid[off[i]:off[i+1]].astype(int).tolist()} for i in idx], [meta[records[i]["row_id"]] for i in idx])
            official_score = (out["total"]["f1"]+out["total"]["negative_f1"])/2
            assert abs(official_score-f1(counts.sum(0))) < 1e-12
            conf[key] = counts; decisions[key] = valid; thresholds[key] = ts
            rows.append({"method":name,"normalization":normalization,"within_auc":float(np.nanmean(au)),
                         "prmscore_q80_oof_diagnostic":float(official_score),"answers":6969,"auc_eligible":6030,
                         "official_scoreable_answers":int(noncontrol.sum()),"official_steps":int(counts.sum())})
    pairs = [("bank11_continuous","bank11_equal"),("bank11_pc4_continuous","bank11_pc4_equal"),
             ("bank11_nov4_continuous","bank11_nov4_equal"),
             ("bank11_continuous","bank11_learned_partition_equal")]
    comparisons = []
    B = 100000; rng = np.random.default_rng(20260923)
    # Retrospective family:4 contrasts x3 endpoints (within/raw PRM/z PRM).
    family = 12
    for a,b in pairs:
        mask = np.isfinite(auc[a]) & np.isfinite(auc[b]); assert mask.sum() == 6030
        cnt = np.bincount(gi[mask], minlength=len(groups))
        ds = np.bincount(gi[mask], weights=(auc[a]-auc[b])[mask], minlength=len(groups))
        draws = {"within_auc":[],"raw_prmscore":[],"answer_z_prmscore":[]}
        for start in range(0,B,1000):
            w = rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(1000,B-start))
            draws["within_auc"].append((w@ds)/(w@cnt))
            for norm in ("raw","answer_z"):
                draws[norm+"_prmscore"].append(f1(w@conf[a+"__"+norm])-f1(w@conf[b+"__"+norm]))
        for endpoint,d in draws.items():
            d = np.concatenate(d)
            point = np.nanmean(auc[a]-auc[b]) if endpoint=="within_auc" else f1(conf[a+"__"+endpoint.removesuffix("_prmscore")].sum(0))-f1(conf[b+"__"+endpoint.removesuffix("_prmscore")].sum(0))
            comparisons.append({"a":a,"b":b,"endpoint":endpoint,"delta":float(point),
                                "ci95":np.quantile(d,[.025,.975]).tolist(),
                                "ci_bonferroni12":np.quantile(d,[.025/family,1-.025/family]).tolist(),
                                "B":B,"paired_N":6030 if endpoint=="within_auc" else int(noncontrol.sum()),
                                "resampled_source_groups":len(groups),"retrospective":True})
    OUT.mkdir(parents=True)
    with (OUT/"DEPTH_AND_BANK_PRMSCORE.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/"FROZEN_Q80_DECISIONS.npz",**decisions,offsets=off,prm_indices=idx)
    sources = paths+[jp,npz,folds_path,meta_path,module_path,fit_path,helper_path,bank_path,Path(__file__)]
    report={"schema":"saved-score-prmbench-audit-v1","no_new_fit":True,"no_new_inference":True,
            "rows":rows,"contrasts":comparisons,"thresholds":thresholds,
            "limitations":["Retrospective calibration of OOF scores; nested fusion training not reconstructed.",
                            "Intervals condition on fitted weights and thresholds; development comparison only.",
                            "Source score arrays lack embedded IDs; canonical shapes, CT7 anchor and bank11 full weight replay verified.",
                            "Learned-partition equal is a new diagnostic from frozen groups, not a newly fitted model."],
            "source_sha256":{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},
            "seconds":time.perf_counter()-started}
    (OUT/"AUDIT.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps({"rows":rows,"contrasts":comparisons,"seconds":report["seconds"]},indent=2),flush=True)


if __name__ == "__main__": main()
