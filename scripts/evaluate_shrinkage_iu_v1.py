"""Evaluate results/fusion_shrinkage_iu_v1 scores on both benchmarks and write REPORT.md."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402

BENCH = ROOT / "results" / "localization_full_benchmark_v3"
GATE = ROOT / "results" / "fusion_fixed_gate_v1"
FOLDS = ROOT / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
OUT = ROOT / "results" / "fusion_shrinkage_iu_v1"
sys.path.insert(0, str(ROOT / "scripts"))
from run_shrinkage_iu_v1 import VARIANTS, vname  # noqa: E402

SEED = 2026090707
DRAWS = 1000


def auc(y, s):
    y = np.asarray(y, bool); p, n = y.sum(), (~y).sum()
    if not p or not n:
        return np.nan
    return float((rankdata(s)[y].sum() - p * (p + 1) / 2) / (p * n))


def main():
    t0 = time.time()
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    z = np.load(BENCH / "evaluation" / "JOINED.npz")
    recs, arms = j["records"], j["arms"]
    n = len(recs)
    offsets, labels, target = z["offsets"], z["labels"], z["target"]
    cells = np.array([r["cell"] for r in recs]); groups = np.array([r["group_id"] for r in recs])
    pb = np.array([c.startswith("pb_") for c in cells]); prm = ~pb
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    outer = np.array([folds["outer"].get(g, -1) for g in groups])
    gate = json.load(open(GATE / "METRICS.json"))
    thr = {int(k): v for k, v in gate["arms"]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"].items()}
    det = np.load(GATE / "DETECTORS.npz")["entropy_mean"]
    fold_thr = np.array([thr.get(int(f), np.nan) for f in outer])

    names = ["iu_replay"] + [vname(v) for v in VARIANTS]
    S = {m: np.full(len(labels), np.nan) for m in names}
    valid = {m: np.zeros(n, bool) for m in names}
    gmm = {m: np.full(n, -2, int) for m in names}
    alpha = {m: [] for m in names}
    replay_err, seam_err, missing, banks = [], [], 0, {"moment": 0, "context": 0}
    for i, r in enumerate(recs):
        p = OUT / "scores" / f"{r['uid']}.npz"
        if not p.exists():
            missing += 1; continue
        meta = json.loads((OUT / "scores" / f"{r['uid']}.json").read_text(encoding="utf-8"))
        if isinstance(meta.get("replay"), (int, float)):
            replay_err.append(meta["replay"]); seam_err.append(meta.get("seam_max_abs_dw", np.nan))
            banks[meta["bank"]] += 1
        with np.load(p) as a:
            for m in names:
                if m + "__risk" in a.files:
                    S[m][offsets[i]:offsets[i + 1]] = a[m + "__risk"]; valid[m][i] = True
                    gmm[m][i] = int(a[m + "__gmm"][0])
                    if m in meta["info"] and "alpha" in meta["info"][m]:
                        alpha[m].append(meta["info"][m]["alpha"])
    # frozen references from the benchmark evaluation
    refs = {"frozen_dual__iu": "dual__iu", "frozen_dual__equal": "dual__equal", "frozen_entropy_parent": "entropy_parent"}
    for k, arm in refs.items():
        c = arms.index(arm); S[k] = z["scores"][:, c]; valid[k] = z["valid"][:, c]; names.append(k)
        gmm[k] = np.where(z["decision"][:, c], z["predictions"][:, c], -2)
    print(f"loaded {n - missing}/{n}; replay max {max(replay_err):.2e}; seam max {np.nanmax(seam_err):.2e}; banks {banks}")

    # step-level arrays for PB peaks
    def peaks(m):
        pk = np.full(n, -1, int)
        for i in range(n):
            if valid[m][i]:
                s = S[m][offsets[i]:offsets[i + 1]]
                pk[i] = int(np.argmax(s)) if np.isfinite(s).all() else -1
        return pk

    results, per_answer = {}, {}
    for m in names:
        v = valid[m]
        lab_mask = np.zeros(len(labels), bool)
        for i in np.flatnonzero(v & prm):
            lab_mask[offsets[i]:offsets[i + 1]] = True
        lab_mask &= labels >= 0
        pooled = auc(labels[lab_mask] == 1, S[m][lab_mask])
        within = []
        wa = np.full(n, np.nan)
        for i in np.flatnonzero(v & prm):
            y = labels[offsets[i]:offsets[i + 1]]; s = S[m][offsets[i]:offsets[i + 1]]
            ok = y >= 0
            if ok.sum() and (y[ok] == 1).any() and (y[ok] == 0).any():
                wa[i] = auc(y[ok] == 1, s[ok]); within.append(wa[i])
        pk = peaks(m)
        pv = v & pb & (pk >= 0) & np.isfinite(det) & np.isfinite(fold_thr)
        pred_fixed = np.where(det >= fold_thr, pk, -1)
        pred_fixed[~pv] = -1
        met_fixed = pb_metrics(target[pb], pred_fixed[pb], pv[pb], cells[pb])
        pred_g = gmm[m].copy(); vg = v & pb & (pred_g != -2); pred_g[~vg] = -1
        met_gmm = pb_metrics(target[pb], pred_g[pb], vg[pb], cells[pb])
        results[m] = dict(prm_valid=int((v & prm).sum()), prm_pooled=pooled, prm_within=float(np.mean(within)) if within else np.nan,
                          prm_within_n=len(within), pb_valid=int(pv.sum()),
                          pb_fixed=met_fixed["macros"], pb_gmm=met_gmm["macros"],
                          pb_cells_fixed={c: x["f1"] for c, x in met_fixed["cells"].items()},
                          alpha_mean=float(np.mean(alpha[m])) if alpha.get(m) else None,
                          alpha_median=float(np.median(alpha[m])) if alpha.get(m) else None)
        per_answer[m] = dict(within=wa, pred_fixed=pred_fixed, pv=pv)
        print(f"{m:34s} PRMB pooled {pooled:.5f} within {(np.mean(within) if within else np.nan):.5f} (n={len(within)}) | PB fixed-gate all8 "
              f"{met_fixed['macros']['all']*100:6.2f} Q8 {met_fixed['macros']['q8']*100:6.2f} | GMM all8 {met_gmm['macros']['all']*100:6.2f} | valid {int((v&prm).sum())}/{int(pv.sum())}")

    # paired source-group bootstrap: variant minus reference, PRMB within (common valid) and PB fixed-gate all8
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(groups, return_inverse=True)
    draws = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float) for _ in range(DRAWS)]
    contrasts = {}
    pairs = [(vname(v), "iu_replay") for v in VARIANTS] + [("iu_replay", "frozen_entropy_parent"), ("iu_replay", "frozen_dual__equal"),
                                                            (vname(("full", "joint", "lw")), "frozen_entropy_parent"),
                                                            (vname(("solve", "joint", "lw")), "frozen_entropy_parent")]
    for a, b in pairs:
        A, B = per_answer[a], per_answer[b]
        common = np.isfinite(A["within"]) & np.isfinite(B["within"])
        if not common.any() or not A["pv"].any():
            contrasts[f"{a} minus {b}"] = dict(status="EMPTY_ARM"); continue
        dw, dpb = [], []
        for w in draws:
            ww = w[common]
            if ww.sum():
                dw.append(np.average(A["within"][common] - B["within"][common], weights=ww))
            pa = pb_metrics(target[pb], A["pred_fixed"][pb], A["pv"][pb], cells[pb], weights=w[pb])["macros"]["all"]
            pb_ = pb_metrics(target[pb], B["pred_fixed"][pb], B["pv"][pb], cells[pb], weights=w[pb])["macros"]["all"]
            if pa is not None and pb_ is not None:
                dpb.append(pa - pb_)
        contrasts[f"{a} minus {b}"] = dict(
            prm_within_common_n=int(common.sum()),
            prm_within_point=float(np.nanmean(A["within"][common] - B["within"][common])),
            prm_within_ci=[float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5))],
            pb_all8_point=float(results[a]["pb_fixed"]["all"] - results[b]["pb_fixed"]["all"]),
            pb_all8_ci=[float(np.percentile(dpb, 2.5)), float(np.percentile(dpb, 97.5))])
    json.dump(dict(results=results, contrasts=contrasts, replay_max_abs=float(max(replay_err)),
                   seam_max_abs=float(np.nanmax(seam_err)), missing=missing, banks=banks, seconds=time.time() - t0),
              open(OUT / "METRICS.json", "w"), indent=1)
    print("evaluated in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
