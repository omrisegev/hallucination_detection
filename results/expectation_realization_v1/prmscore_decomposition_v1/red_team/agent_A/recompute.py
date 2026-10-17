"""Red-team A: independent recomputation from raw per-response artifacts."""
import csv, json, pickle, hashlib, sys, importlib.util
import numpy as np

SE = "C:/Users/omris/TAU/hallucination_detection/.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/"
W = "C:/Users/omris/TAU/hallucination_detection/.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/"
SCORER = "C:/Users/omris/TAU/hallucination_detection/.worktrees/depth-feature-fusion-v1/spectral_utils/prmbench.py"
files = {"answers": SE + "OOF_ANSWERS.csv", "oof": SE + "OOF_STEP_SCORES.npz", "freeze": SE + "INPUT_FREEZE.json",
         "steps": W + "run_20260927_stage_b/STEP_SCORES.npz", "thr": W + "run_20260927_stage_b_thr/THRESHOLDS.json",
         "chan": W + "run_20260927_stage_b_thr/CHANNELS.npz", "scorer": SCORER}
freeze = json.load(open(files["freeze"], encoding="utf-8-sig"))
files["prm_pkl"] = freeze["prm_metadata"]["path"].replace("\\", "/")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for k, p in files.items():
    print(f"SHA256 {k}: {sha(p)}  {p}")
assert sha(files["prm_pkl"]) == freeze["prm_metadata"]["sha256"]

spec = importlib.util.spec_from_file_location("prmbench_official", SCORER)
pbm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pbm)

rows = list(csv.DictReader(open(files["answers"], encoding="utf-8-sig")))
oof = np.load(files["oof"]); st = np.load(files["steps"]); ch = np.load(files["chan"])
off = oof["offsets"]; lab = oof["labels"].astype(bool)
assert len(off) == len(rows) + 1 and np.array_equal(off, st["offsets"]) and np.array_equal(off, ch["offsets"])
thr = json.load(open(files["thr"], encoding="utf-8-sig"))
prm = {v["idx"]: v for v in pickle.load(open(files["prm_pkl"], "rb")).values()}

P = [i for i, r in enumerate(rows) if not r["cell"].startswith("pb_")]
print("PRMBench answers", len(P), "PRM pkl rows", len(prm))
# --- alignment check: labels vs 1-based error_steps (out-of-range inert), n_steps, rewards
nmis = 0
for i in P:
    m = prm[rows[i]["id"]]; a, b = off[i], off[i + 1]
    n = b - a
    assert n == m["n_steps"] == len(m["rewards"]), (i, n, m["n_steps"])
    exp = np.zeros(n, bool)
    for e in m["error_steps"]:
        if 1 <= e <= n: exp[e - 1] = True
    nmis += int(not np.array_equal(exp, lab[a:b]))
print("label/error_steps mismatched answers:", nmis)

fold = np.array([int(r["fold"]) for r in rows]); cls = {i: prm[rows[i]["id"]]["classification"] for i in P}
Pset = np.zeros(len(rows), bool); Pset[P] = True
step_ans = np.repeat(np.arange(len(rows)), np.diff(off))
step_is_P = Pset[step_ans]; step_fold = fold[step_ans]

def ansz(s):
    z = np.empty_like(s, dtype=float)
    for i in range(len(rows)):
        a, b = off[i], off[i + 1]; x = s[a:b].astype(float)
        z[a:b] = (x - x.mean()) / max(x.std(), 1e-8)
    return z

def q80_taus(v):  # tau_k = 0.8 quantile over PRMBench steps of fold (k+1)%5
    return {k: float(np.quantile(v[step_is_P & (step_fold == (k + 1) % 5)], 0.8)) for k in range(5)}

reward = np.full(len(lab), np.nan)
for i in P:
    reward[off[i]:off[i + 1]] = prm[rows[i]["id"]]["rewards"]
prm_risk = 1.0 - reward
risk = {"S_equal": st["S_equal"], "B13_equal": st["B13_equal"],
        "realized_drv": ch["values"][:, list(ch["names"]).index("realized_drv")], "PRM": prm_risk}
for k, v in risk.items(): print("nonfinite on PRMBench steps", k, int((~np.isfinite(v[step_is_P])).sum()))
Z = {k: ansz(np.nan_to_num(v)) for k, v in risk.items()}
invalid = {}
for mth in ("S_equal", "B13_equal"):
    t = thr[mth]; tau = np.array([t[str(f)] for f in step_fold])
    invalid[mth] = ~(Z[mth] < tau)
    mine = q80_taus(Z[mth])
    print(f"tau check {mth}: saved {[round(t[str(k)],6) for k in range(5)]} vs q80 rule {[round(mine[k],6) for k in range(5)]}")
t = q80_taus(Z["realized_drv"]); print("realized_drv taus", t)
invalid["realized_drv"] = ~(Z["realized_drv"] < np.array([t[f] for f in step_fold]))
t = q80_taus(prm_risk); print("PRM_raw taus", t)
invalid["PRM_raw_q80"] = ~(prm_risk < np.array([t[f] for f in step_fold]))
t = q80_taus(Z["PRM"]); print("PRM_z taus", t)
invalid["PRM_z_q80"] = ~(Z["PRM"] < np.array([t[f] for f in step_fold]))
invalid["PRM_native"] = ~(reward >= 0.5)

meta = [{"idx": v["idx"], "error_steps": list(v["error_steps"]), "classification": v["classification"]} for v in prm.values()]
C7 = ["redundency", "circular", "domain_inconsistency", "confidence", "counterfactual", "deception", "missing_condition"]
res = {}
for mth, inv in invalid.items():
    preds = [{"idx": rows[i]["id"], "labels": [0 if x else 1 for x in inv[off[i]:off[i + 1]]]} for i in P]
    out = pbm.prmbench_evaluate(preds, meta)
    tot = out["total"]; bc = out["by_classification"]
    res[mth] = {"prmscore": 0.5 * (tot["f1"] + tot["negative_f1"]),
                "cls": {c: 0.5 * (bc["f1"][c] + bc["negative_f1"][c]) for c in C7},
                "n_scored": out["n_predictions_scored"]}
    ctrl = [i for i in P if cls[i] == "correct"]
    anyinv = np.mean([inv[off[i]:off[i + 1]].any() for i in ctrl])
    stepinv = inv[np.concatenate([np.arange(off[i], off[i + 1]) for i in ctrl])].mean()
    res[mth].update(ctrl_n=len(ctrl), ctrl_ans=anyinv, ctrl_steps=stepinv)
    print(f"{mth:14s} PRMScore={res[mth]['prmscore']:.4f} scored={out['n_predictions_scored']} "
          f"controls n={len(ctrl)} anyinv={anyinv:.4f} stepinv={stepinv:.4f}")
print("per-class S_equal - PRM_z_q80:")
for c in C7:
    print(f"  {c:22s} S={res['S_equal']['cls'][c]:.4f} PRMz={res['PRM_z_q80']['cls'][c]:.4f} diff={res['S_equal']['cls'][c]-res['PRM_z_q80']['cls'][c]:+.4f}")

# --- within-answer AUROC (Mann-Whitney with ties = 0.5)
def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
for k in ("S_equal", "realized_drv", "PRM"):
    a = [auc(risk[k][off[i]:off[i + 1]], lab[off[i]:off[i + 1]]) for i in P
         if 0 < lab[off[i]:off[i + 1]].sum() < off[i + 1] - off[i]]
    print(f"within-AUC {k}: {np.mean(a):.4f} N={len(a)}")

# --- argmax-hit complementarity
E = [i for i in P if lab[off[i]:off[i + 1]].any()]
hit = {k: np.array([lab[off[i] + int(np.argmax(risk[k][off[i]:off[i + 1]]))] for i in E]) for k in ("S_equal", "realized_drv", "PRM")}
print("answers with >=1 error step:", len(E))
for a, b in (("S_equal", "realized_drv"), ("S_equal", "PRM")):
    x, y = hit[a], hit[b]
    print(f"{a} vs {b}: both={int((x&y).sum())} only_{a}={int((x&~y).sum())} only_{b}={int((~x&y).sum())} neither={int((~x&~y).sum())} hitrates {x.mean():.4f}/{y.mean():.4f}")

# --- paired source-group bootstrap of S_equal - realized_drv PRMScore (pooled non-control counts)
NC = [i for i in P if cls[i] != "correct"]
def counts(inv):
    M = np.zeros((len(rows), 4))
    for i in NC:
        v = ~inv[off[i]:off[i + 1]]; e = lab[off[i]:off[i + 1]]
        M[i] = [(v & ~e).sum(), (v & e).sum(), (~v & e).sum(), (~v & ~e).sum()]  # TP FP TN FN
    return M
def ps(c):
    tp, fp, tn, fn = c
    return 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp))
CA, CB = counts(invalid["S_equal"]), counts(invalid["realized_drv"])
print(f"pooled non-control PRMScore S_equal={ps(CA[NC].sum(0)):.4f} realized_drv={ps(CB[NC].sum(0)):.4f} diff={ps(CA[NC].sum(0))-ps(CB[NC].sum(0)):+.4f}")
for label, pool in (("groups of non-control answers", NC), ("groups of all PRMBench answers", P)):
    groups = sorted({rows[i]["source_group"] for i in pool}); gi = {g: j for j, g in enumerate(groups)}
    GA = np.zeros((len(groups), 4)); GB = np.zeros((len(groups), 4))
    for i in NC:
        GA[gi[rows[i]["source_group"]]] += CA[i]; GB[gi[rows[i]["source_group"]]] += CB[i]
    rng = np.random.default_rng(20260927 + 7)
    d = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        d.append(ps(w @ GA) - ps(w @ GB))
    d = np.array(d)
    print(f"bootstrap ({label}, G={len(groups)}): mean={d.mean():+.4f} 95% [{np.quantile(d,.025):+.4f}, {np.quantile(d,.975):+.4f}]")
