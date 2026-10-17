"""Items 1 and 2: PRMScore (two independent paths) and ProcessBench F1 macro-8 from DECISIONS flags."""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from common import load_all, load_official_scorer, PRMCELL

df, off, lab, d, bymeta = load_all()
cells = df.cell.values
ids = df.id.values
target = df.target.values
N = len(df)

RULES = ["R0_frozen", "R2_allocate", "R2pb_allocate", "OFFSET_D1_upcr_full", "OFFSET_D2_lsml_cont_good5",
         "OFFSET_D3_lsml_full", "OFFSET_D4_equal_full", "OFFSET_D5_epr", "OFFSET_D6_length",
         "OFFSET_D6b_length_anchored"]

prm_idx = np.where(cells == PRMCELL)[0]
pb_cells = sorted(c for c in set(cells) if c != PRMCELL)
assert len(pb_cells) == 8

# ---- PRMBench error-step sets from the metadata (1-based -> 0-based, in-range only); check vs OOF labels
err_sets = {}
mism = 0
for i in prm_idx:
    r = bymeta[ids[i]]
    n = off[i + 1] - off[i]
    assert r["n_steps"] == n
    es = {e - 1 for e in r["error_steps"] if 1 <= e <= n}
    err_sets[i] = es
    lab_i = lab[off[i]:off[i + 1]]
    if set(np.where(lab_i == 1)[0].tolist()) != es:
        mism += 1
print("PRMBench label/meta mismatches:", mism)


def prm_confusion(flags, i):
    f = flags[off[i]:off[i + 1]]
    es = err_sets[i]
    e = np.zeros(len(f), bool)
    if es:
        e[list(es)] = True
    tp = int(np.sum(~e & ~f)); fp = int(np.sum(e & ~f)); tn = int(np.sum(e & f)); fn = int(np.sum(~e & f))
    return tp, fp, tn, fn


def prmscore_from_counts(tp, fp, tn, fn):
    P = tp / (tp + fp); R = tp / (tp + fn); f1 = 2 * P * R / (P + R)
    nP = tn / (tn + fn); nR = tn / (tn + fp); nf1 = 2 * nP * nR / (nP + nR)
    return 0.5 * (f1 + nf1), f1, nf1


def my_prmscore(flags):
    tot = np.zeros(4, np.int64)
    for i in prm_idx:
        if bymeta[ids[i]]["classification"] == "correct":
            continue
        tot += prm_confusion(flags, i)
    return prmscore_from_counts(*tot), tot


mod = load_official_scorer()
meta_list = [{"idx": bymeta[ids[i]]["idx"], "error_steps": list(bymeta[ids[i]]["error_steps"]),
              "classification": bymeta[ids[i]]["classification"]} for i in prm_idx]


def official_prmscore(flags):
    preds = [{"idx": ids[i], "labels": [int(not x) for x in flags[off[i]:off[i + 1]]]} for i in prm_idx]
    res = mod.prmbench_evaluate(preds, meta_list)
    t = res["total"]
    return 0.5 * (t["f1"] + t["negative_f1"]), res["n_predictions_scored"], res["n_correct_control_rows"]


def pb_f1(flags):
    out = {}
    for c in pb_cells:
        ii = np.where(cells == c)[0]
        pred = np.empty(len(ii), np.int64)
        for j, i in enumerate(ii):
            f = np.where(flags[off[i]:off[i + 1]])[0]
            pred[j] = f[0] if len(f) else -1
        t = target[ii]
        err = t >= 0
        a_e = float(np.mean(pred[err] == t[err]))
        a_c = float(np.mean(pred[~err] == -1))
        f1 = 0.0 if (a_e + a_c) == 0 else 2 * a_e * a_c / (a_e + a_c)
        out[c] = dict(acc_err=a_e, acc_cor=a_c, f1=f1, n=len(ii), n_err=int(err.sum()))
    return float(np.mean([out[c]["f1"] for c in pb_cells])), out


results = {}
for r in RULES:
    fl = d[r].astype(bool)
    (ps, f1, nf1), tot = my_prmscore(fl)
    offi, nsc, nctl = official_prmscore(fl)
    pbm, per = pb_f1(fl)
    results[r] = dict(prmscore_mine=ps, prmscore_official=offi, f1=f1, neg_f1=nf1, counts=tot.tolist(),
                      n_scored=nsc, n_controls=nctl, pb_f1_macro=pbm, pb_per_cell=per,
                      prm_flag_share=float(np.mean(np.concatenate([fl[off[i]:off[i + 1]] for i in prm_idx]))))
    print(f"{r:32s} PRMScore mine={ps!r} official={offi!r}  PB F1 macro={pbm:.4f}")

json.dump(results, open(__file__.replace("point.py", "point_out.json"), "w"), indent=1)
