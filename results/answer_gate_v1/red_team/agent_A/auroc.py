"""Item 3: answer-level AUROC from the saved A_<detector> arrays (Mann-Whitney with midranks)."""
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from common import load_all, PRMCELL, FEAT

df, off, lab, d, bymeta = load_all()
cells = df.cell.values; ids = df.id.values; target = df.target.values
f = np.load(FEAT, allow_pickle=True)
assert list(f["ids"]) == list(ids), "ANSWER_FEATURES id order != OOF_ANSWERS order"
assert list(f["cells"]) == list(cells)
assert np.array_equal(f["folds"], df.fold.values)


def mw_auc(pos, neg):
    x = np.concatenate([pos, neg]).astype(np.float64)
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


DETS = ["D1_upcr_full", "D2_lsml_cont_good5", "D3_lsml_full", "D4_equal_full", "D5_epr", "D6_length",
        "D6b_length_anchored"]
pb_cells = sorted(c for c in set(cells) if c != PRMCELL)
prm_idx = np.where(cells == PRMCELL)[0]
cls = np.array([bymeta[ids[i]]["classification"] if cells[i] == PRMCELL else "" for i in range(len(df))])
has_err = np.zeros(len(df), bool)
for i in prm_idx:
    has_err[i] = bool(np.any(lab[off[i]:off[i + 1]] == 1))
prm_err = np.array([i for i in prm_idx if cls[i] != "correct" and has_err[i]])
prm_ms = np.array([i for i in prm_idx if cls[i] == "multi_solutions"])
prm_ctl = np.array([i for i in prm_idx if cls[i] == "correct"])
nonctl_noerr_nonms = [i for i in prm_idx if cls[i] not in ("correct", "multi_solutions") and not has_err[i]]
print("PRM erroneous n =", len(prm_err), " multi_solutions n =", len(prm_ms), " controls n =", len(prm_ctl),
      " multi_solutions with an error step:", int(has_err[prm_ms].sum()),
      " error-class answers with no in-range error:", len(nonctl_noerr_nonms))

out = {}
for det in DETS:
    A = d["A_" + det].astype(np.float64)
    assert np.all(np.isfinite(A))
    per = {}
    for c in pb_cells:
        ii = np.where(cells == c)[0]
        e = target[ii] >= 0
        a = mw_auc(A[ii][e], A[ii][~e])
        assert abs(a - roc_auc_score(e, A[ii])) < 1e-12
        per[c] = a
    pbm = float(np.mean(list(per.values())))
    fair = mw_auc(A[prm_err], A[prm_ms])
    ctl = mw_auc(A[prm_err], A[prm_ctl])
    out[det] = dict(pb_macro=pbm, pb_per_cell=per, prm_err_vs_multisol=fair, prm_err_vs_controls=ctl)
    print(f"{det:22s} PB macro={pbm:.4f}  PRM err vs multi_sol={fair:.4f}  err vs controls={ctl:.4f}  "
          + " ".join(f"{c.replace('pb_', '')}={v:.3f}" for c, v in per.items()))
json.dump(out, open(__file__.replace("auroc.py", "auroc_out.json"), "w"), indent=1)
