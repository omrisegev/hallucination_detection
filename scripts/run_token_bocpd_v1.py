"""BOCPD on per-token entropy as a first-error localizer, all 13,769 rows (2026-09-10).

Omri asked whether BOCPD was tried on the winning token-level representation. It was
not: Step 151 used it as an answer-level view, Step 299 as a readout of the fused
8-token-window trajectory on a 58-answer pilot. Here the verified reset-before-observation
Gaussian BOCPD (`spectral_utils.fused_trajectory_readouts.bocpd_filter`, exact, untruncated)
runs on each answer's z-scored token entropy with the answer's own noise-variance estimate
(`noise_variance`), hazard 1/32 (the module default), prior mean 0 / variance 1 on the
z-scale. Three token curves are read out with the top-10 rule and the shared entropy gate:
  bocpd_reset : posterior change-point probability at the token,
  bocpd_rise  : reset probability x positive standardized surprise (upward change evidence),
  bocpd_surprise : -log predictive density of the token.
Reference: token entropy, top-10. Hazard sensitivity 1/16 and 1/64 for the rise curve.
No labels enter any fit; development evidence.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "localization_full_benchmark_v3"
OUT = ROOT / "results" / "token_bocpd_v1"
sys.path.insert(0, str(ROOT))

ARMS = ("token_entropy", "bocpd_reset", "bocpd_rise", "bocpd_surprise", "bocpd_rise_h16", "bocpd_rise_h64")


def topk(tok, ss, se, k=10):
    return np.asarray([np.sort(tok[a:b])[::-1][:min(k, b - a)].mean() for a, b in zip(ss, se)])


def score_record(args):
    uid, cell, row_index = args
    from spectral_utils.fused_trajectory_readouts import bocpd_filter, noise_variance
    with np.load(BENCH / "scores" / f"{uid}.npz") as z:
        ss, se = z["step_starts"], z["step_ends"]
    d = BENCH / "inputs" / cell
    raw = np.load(d / "raw.npy", mmap_mode="r"); offsets = np.load(d / "token_offsets.npy")
    e = np.asarray(raw[offsets[row_index]:offsets[row_index + 1], 1], float)
    zs = (e - e.mean()) / (e.std() + 1e-12)
    r = float(noise_variance(zs)); r = r if np.isfinite(r) and r > 1e-6 else 1.0
    arrays = {"token_entropy__risk": topk(e, ss, se)}
    b = bocpd_filter(zs, hazard=1 / 32, observation_variance=r, prior_mean=0.0, prior_variance=1.0)
    arrays["bocpd_reset__risk"] = topk(b["reset_probability"], ss, se)
    arrays["bocpd_rise__risk"] = topk(b["rise"], ss, se)
    arrays["bocpd_surprise__risk"] = topk(-b["log_predictive"], ss, se)
    for h, name in ((1 / 16, "bocpd_rise_h16"), (1 / 64, "bocpd_rise_h64")):
        bb = bocpd_filter(zs, hazard=h, observation_variance=r, prior_mean=0.0, prior_variance=1.0)
        arrays[name + "__risk"] = topk(bb["rise"], ss, se)
    for k, v in arrays.items():
        if not np.isfinite(v).all():
            raise ValueError(f"NONFINITE {k}")
    return {"uid": uid, "noise_variance": r}, arrays


def phase_score(workers):
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8")); recs = j["records"]
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    row_in_cell = {}
    for cell in sorted(set(r["cell"] for r in recs)):
        ids = np.load(BENCH / "inputs" / cell / "row_ids.npy", allow_pickle=True); row_in_cell[cell] = {str(x): i for i, x in enumerate(ids)}
    todo = [(r["uid"], r["cell"], row_in_cell[r["cell"]][r["row_id"]]) for r in recs if not (OUT / "scores" / f"{r['uid']}.npz").exists()]
    print(f"{len(recs)} records, {len(todo)} to score, {workers} workers", flush=True)
    t0 = time.time(); done = failures = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_record, a): a for a in todo}
        for f in as_completed(futs):
            uid = futs[f][0]
            try:
                out, arrays = f.result()
            except Exception:  # noqa: BLE001
                failures += 1; (OUT / "scores" / f"{uid}.error").write_text(traceback.format_exc(), encoding="utf-8"); continue
            tmp = OUT / "scores" / f"{uid}.tmp.npz"; np.savez_compressed(tmp, **arrays); tmp.replace(OUT / "scores" / f"{uid}.npz")
            done += 1
            if done % 500 == 0: print(f"{done}/{len(todo)} {time.time() - t0:.0f}s failures={failures}", flush=True)
    print("scoring complete", round(time.time() - t0), "s, failures", failures, flush=True)


def phase_evaluate():
    from scipy.stats import rankdata
    from spectral_utils.historical_fusion_evaluation import pb_metrics
    def auc(y, s):
        y = np.asarray(y, bool); p, n = y.sum(), (~y).sum()
        return float((rankdata(s)[y].sum() - p * (p + 1) / 2) / (p * n)) if p and n else np.nan
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8")); z = np.load(BENCH / "evaluation" / "JOINED.npz")
    recs = j["records"]; n = len(recs); off, lab, tgt = z["offsets"], z["labels"], z["target"]
    cells = np.array([r["cell"] for r in recs]); groups = np.array([r["group_id"] for r in recs]); pb = np.array([c.startswith("pb_") for c in cells])
    gate = json.load(open(ROOT / "results" / "fusion_fixed_gate_v1" / "METRICS.json"))
    thr = {int(k): v for k, v in gate["arms"]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"].items()}
    det = np.load(ROOT / "results" / "fusion_fixed_gate_v1" / "DETECTORS.npz")["entropy_mean"]
    folds = json.load(open(ROOT / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"))
    fthr = np.array([thr.get(int(folds["outer"].get(g, -1)), np.nan) for g in groups])
    S = {a: np.full(len(lab), np.nan) for a in ARMS}; valid = {a: np.zeros(n, bool) for a in ARMS}
    for i, r in enumerate(recs):
        p = OUT / "scores" / f"{r['uid']}.npz"
        if not p.exists(): continue
        with np.load(p) as a:
            for m in ARMS:
                if m + "__risk" in a.files: S[m][off[i]:off[i + 1]] = a[m + "__risk"]; valid[m][i] = True
    res, per = {}, {}; T, C = tgt[pb], cells[pb]
    for m in ARMS:
        v = valid[m]; wa = np.full(n, np.nan); pk = np.full(n, -1)
        for i in np.flatnonzero(v):
            s = S[m][off[i]:off[i + 1]]; pk[i] = int(np.argmax(s))
            if not pb[i]:
                y = lab[off[i]:off[i + 1]]; ok = y >= 0
                if (y[ok] == 1).any() and (y[ok] == 0).any(): wa[i] = auc(y[ok] == 1, s[ok])
        pv = v & pb & np.isfinite(det) & np.isfinite(fthr); pred = np.where(det >= fthr, pk, -1); pred[~pv] = -1
        met = pb_metrics(T, pred[pb], pv[pb], C); err = pb & (tgt >= 0) & v
        res[m] = dict(pb_all8=met["macros"]["all"], pb_q8=met["macros"]["q8"], prm_within=float(np.nanmean(wa)),
                      raw_exact=float(np.mean(pk[err] == tgt[err])), within_one=float(np.mean(np.abs(pk[err] - tgt[err]) <= 1)), valid=int(v.sum()))
        per[m] = dict(within=wa, pred=pred, pv=pv); r_ = res[m]
        print(f"{m:16s} PB all8 {r_['pb_all8']*100:6.2f} Q8 {r_['pb_q8']*100:6.2f} exact {r_['raw_exact']*100:5.1f} within-1 {r_['within_one']*100:5.1f} | PRMB within {r_['prm_within']:.4f} | valid {r_['valid']}")
    rng = np.random.default_rng(2026090707); uniq, inv = np.unique(groups, return_inverse=True)
    draws = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float) for _ in range(1000)]
    contrasts = {}
    for a in ARMS[1:]:
        A, B = per[a], per["token_entropy"]; common = np.isfinite(A["within"]) & np.isfinite(B["within"]); dw, dpb = [], []
        for w in draws:
            dw.append(np.average(A["within"][common] - B["within"][common], weights=w[common]))
            pa = pb_metrics(T, A["pred"][pb], A["pv"][pb], C, weights=w[pb])["macros"]["all"]; pb_ = pb_metrics(T, B["pred"][pb], B["pv"][pb], C, weights=w[pb])["macros"]["all"]
            if pa is not None and pb_ is not None: dpb.append(pa - pb_)
        contrasts[a] = dict(within=float(np.mean(A["within"][common] - B["within"][common])), within_ci=[float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5))],
                            pb=float(res[a]["pb_all8"] - res["token_entropy"]["pb_all8"]), pb_ci=[float(np.percentile(dpb, 2.5)), float(np.percentile(dpb, 97.5))])
        c = contrasts[a]; print(f"{a:16s} minus token_entropy: within {c['within']:+.4f} [{c['within_ci'][0]:+.4f},{c['within_ci'][1]:+.4f}]  PB {c['pb']*100:+.2f}pp [{c['pb_ci'][0]*100:+.2f},{c['pb_ci'][1]*100:+.2f}]")
    json.dump(dict(results=res, contrasts=contrasts), open(OUT / "METRICS.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="score", choices=["score", "evaluate", "smoke"]); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.phase == "smoke":
        j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8")); recs = j["records"]
        for r in (recs[0], recs[9000]):
            ids = np.load(BENCH / "inputs" / r["cell"] / "row_ids.npy", allow_pickle=True); k = {str(x): i for i, x in enumerate(ids)}[r["row_id"]]
            t = time.time(); out, arrays = score_record((r["uid"], r["cell"], k)); print(r["cell"], r["tokens"], out, {k2: np.round(v, 2).tolist()[:6] for k2, v in arrays.items()}, f"{time.time()-t:.2f}s")
    elif a.phase == "score": phase_score(a.workers)
    else: phase_evaluate()
