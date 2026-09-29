"""Item 4: my own paired source-group bootstrap (2,000 draws, seed 777 - deliberately NOT the protocol seed).
PRMBench: resample PRMBench source groups; PRMScore from resampled pooled confusion counts (controls excluded, as official).
ProcessBench: resample PB source groups pooled over the 8 cells (a group spanning cells moves together);
per-cell F1 from weighted accuracies, then macro-8. Also a cell-stratified variant as a sensitivity check."""
import json
import numpy as np
from common import load_all, PRMCELL

B = 2000
rng = np.random.default_rng(777)
df, off, lab, d, bymeta = load_all()
cells = df.cell.values; ids = df.id.values; target = df.target.values; grp = df.source_group.values
pb_cells = sorted(c for c in set(cells) if c != PRMCELL)
prm_idx = np.where(cells == PRMCELL)[0]
pb_idx = np.where(cells != PRMCELL)[0]

RULES = ["R0_frozen", "R2pb_allocate", "OFFSET_D1_upcr_full", "OFFSET_D2_lsml_cont_good5"]

# ---------- per-answer statistics
prm_nc = np.array([i for i in prm_idx if bymeta[ids[i]]["classification"] != "correct"])
conf = {}
for r in RULES:
    fl = d[r].astype(bool)
    M = np.zeros((len(prm_nc), 4))
    for j, i in enumerate(prm_nc):
        f = fl[off[i]:off[i + 1]]; e = lab[off[i]:off[i + 1]] == 1
        M[j] = [np.sum(~e & ~f), np.sum(e & ~f), np.sum(e & f), np.sum(~e & f)]
    conf[r] = M
hit = {}
for r in RULES:
    fl = d[r].astype(bool)
    h = np.zeros(len(pb_idx))
    for j, i in enumerate(pb_idx):
        f = np.where(fl[off[i]:off[i + 1]])[0]
        p = f[0] if len(f) else -1
        h[j] = float(p == target[i])  # target -1 => correct answer => hit iff no flag
    hit[r] = h


def prmscore(c):  # c: (..., 4) tp fp tn fn
    tp, fp, tn, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    P = tp / (tp + fp); R = tp / (tp + fn); f1 = 2 * P * R / (P + R)
    nP = tn / (tn + fn); nR = tn / (tn + fp); nf1 = 2 * nP * nR / (nP + nR)
    return 0.5 * (f1 + nf1)


pb_cell_of = cells[pb_idx]; pb_err = target[pb_idx] >= 0


def pb_macro(h, w):  # w: (B, n) weights
    f1s = []
    for c in pb_cells:
        m = pb_cell_of == c
        me = m & pb_err; mc = m & ~pb_err
        ae = (w[:, me] * h[me]).sum(1) / w[:, me].sum(1)
        ac = (w[:, mc] * h[mc]).sum(1) / w[:, mc].sum(1)
        f = np.where(ae + ac > 0, 2 * ae * ac / np.where(ae + ac > 0, ae + ac, 1), 0.0)
        f1s.append(f)
    return np.mean(f1s, axis=0)


def group_weights(g, B, rng, strata=None):
    ug, inv = np.unique(g, return_inverse=True)
    if strata is None:
        cnt = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=B)  # (B, G)
        return cnt[:, inv].astype(float), len(ug)
    W = np.zeros((B, len(g)))
    for s in np.unique(strata):
        m = strata == s
        ugs, invs = np.unique(g[m], return_inverse=True)
        cnt = rng.multinomial(len(ugs), np.full(len(ugs), 1 / len(ugs)), size=B)
        W[:, m] = cnt[:, invs]
    return W, None


# groups: do PB groups span cells? do PRMB groups overlap PB?
gp = grp[pb_idx]
span = {}
for g_, c_ in zip(gp, pb_cell_of):
    span.setdefault(g_, set()).add(c_)
print("PB source groups:", len(span), " spanning >1 cell:", sum(len(v) > 1 for v in span.values()),
      " max cells per group:", max(len(v) for v in span.values()))
print("PRMB groups (non-control answers):", len(np.unique(grp[prm_nc])),
      " overlap with PB groups:", len(set(grp[prm_nc]) & set(gp)))

Wprm, ngp = group_weights(grp[prm_nc], B, rng)
Wpb, ngb = group_weights(gp, B, rng)
Wpb_strat, _ = group_weights(gp, B, rng, strata=pb_cell_of)


def summarize(point, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi), p_le0=float(np.mean(draws <= 0)),
                p_ge0=float(np.mean(draws >= 0)), sd=float(draws.std()))


out = {}
one = np.ones((1, len(prm_nc)))
for a, b in [("OFFSET_D1_upcr_full", "R0_frozen"), ("OFFSET_D2_lsml_cont_good5", "R0_frozen")]:
    pa = prmscore(conf[a].sum(0)); pbv = prmscore(conf[b].sum(0))
    da = prmscore(Wprm @ conf[a]); db = prmscore(Wprm @ conf[b])
    out[f"PRMScore {a} - {b}"] = summarize(pa - pbv, da - db)
for a, b in [("OFFSET_D1_upcr_full", "R0_frozen"), ("OFFSET_D2_lsml_cont_good5", "R0_frozen"),
             ("OFFSET_D1_upcr_full", "R2pb_allocate"), ("OFFSET_D2_lsml_cont_good5", "R2pb_allocate")]:
    w1 = np.ones((1, len(pb_idx)))
    pt = pb_macro(hit[a], w1)[0] - pb_macro(hit[b], w1)[0]
    dr = pb_macro(hit[a], Wpb) - pb_macro(hit[b], Wpb)
    ds = pb_macro(hit[a], Wpb_strat) - pb_macro(hit[b], Wpb_strat)
    out[f"PB F1 macro {a} - {b} (pooled groups)"] = summarize(pt, dr)
    out[f"PB F1 macro {a} - {b} (cell-stratified groups)"] = summarize(pt, ds)
for k, v in out.items():
    print(f"{k:75s} point={v['point']:+.4f}  95% [{v['lo']:+.4f}, {v['hi']:+.4f}]  "
          f"P(d<=0)={v['p_le0']:.4f} P(d>=0)={v['p_ge0']:.4f}")
json.dump(out, open(__file__.replace("boot.py", "boot_out.json"), "w"), indent=1)
