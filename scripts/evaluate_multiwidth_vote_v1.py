"""Peak-window voting across window widths (Omri's formulation, 2026-09-09).

For each width the answer-only IU localizer selects its most problematic window (the argmax
window; with non-overlapping windows this is the plateau of the maximum in the token curve).
Every selected window votes for the tokens it covers. The tokens with the most votes are the
hallucination location; the predicted step is the step containing the max-vote tokens (ties
broken by the mean z-scored curve inside the step). `vote_top1`: one window per width;
`vote_top3`: each width's three highest windows vote (weights 3, 2, 1) so near-ties count.
Reads the per-width z-scored token curves saved by run_multiwidth_iu_v1.py; same entropy
gate; comparison against width-8 alone with the top-10 readout. Development evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

BENCH = ROOT / "results" / "localization_full_benchmark_v3"
GATE = ROOT / "results" / "fusion_fixed_gate_v1"
FOLDS = ROOT / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
SEED, DRAWS, TOPK = 2026090707, 1000, 10


def auc(y, s):
    y = np.asarray(y, bool); p, n = y.sum(), (~y).sum()
    return float((rankdata(s)[y].sum() - p * (p + 1) / 2) / (p * n)) if p and n else np.nan


def plateaus(tok, k):
    """Token masks of the k highest distinct window plateaus of a piecewise-constant curve."""
    r = np.round(tok, 5)
    levels = np.unique(r)[::-1][:k]
    return [(r == lv) for lv in levels]


def step_topk(tok, ss, se, k=TOPK):
    return np.asarray([np.sort(tok[a:b])[::-1][:min(k, b - a)].mean() for a, b in zip(ss, se)])


def vote_steps(curves, ss, se, k):
    votes = np.zeros(len(next(iter(curves.values()))))
    for tok in curves.values():
        for rank, mask in enumerate(plateaus(tok, k)):
            votes[mask] += (k - rank)
    mean_curve = np.mean(np.vstack(list(curves.values())), 0)
    step_votes = np.asarray([votes[a:b].max() for a, b in zip(ss, se)], float)
    tie = step_topk(mean_curve, ss, se)
    tie = (tie - tie.min()) / (np.ptp(tie) + 1e-9) * 1e-3     # strictly below one vote
    return step_votes + tie


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--widths", required=True)
    a = ap.parse_args(); OUT = ROOT / "results" / a.out; WIDTHS = tuple(int(x) for x in a.widths.split(","))
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    z = np.load(BENCH / "evaluation" / "JOINED.npz")
    recs = j["records"]; n = len(recs); offsets, labels, target = z["offsets"], z["labels"], z["target"]
    cells = np.array([r["cell"] for r in recs]); groups = np.array([r["group_id"] for r in recs])
    pb = np.array([c.startswith("pb_") for c in cells]); prm = ~pb
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    gate = json.load(open(GATE / "METRICS.json"))
    thr = {int(k): v for k, v in gate["arms"]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"].items()}
    det = np.load(GATE / "DETECTORS.npz")["entropy_mean"]
    fthr = np.array([thr.get(int(folds["outer"].get(r["group_id"], -1)), np.nan) for r in recs])
    arms = ["w8_top10", "vote_top1", "vote_top3", "vote_top1_full_coverage_only"]
    S = {x: np.full(len(labels), np.nan) for x in arms}; valid = {x: np.zeros(n, bool) for x in arms}
    n_full = 0
    for i, r in enumerate(recs):
        p = OUT / "scores" / f"{r['uid']}.npz"
        if not p.exists(): continue
        with np.load(p) as f:
            if "w8_raw_tok" not in f.files: continue
            ss, se = f["step_starts"], f["step_ends"]
            curves = {w: f[f"w{w}_z_tok"].astype(float) for w in WIDTHS if f"w{w}_z_tok" in f.files}
            raw8 = f["w8_raw_tok"].astype(float)
        sl = slice(offsets[i], offsets[i + 1])
        S["w8_top10"][sl] = step_topk(raw8, ss, se); valid["w8_top10"][i] = True
        S["vote_top1"][sl] = vote_steps(curves, ss, se, 1); valid["vote_top1"][i] = True
        S["vote_top3"][sl] = vote_steps(curves, ss, se, 3); valid["vote_top3"][i] = True
        if len(curves) == len(WIDTHS):
            n_full += 1; S["vote_top1_full_coverage_only"][sl] = S["vote_top1"][sl]; valid["vote_top1_full_coverage_only"][i] = True
    print(f"widths {WIDTHS}; rows with all widths: {n_full}")
    results, per = {}, {}
    T, C = target[pb], cells[pb]
    for x in arms:
        v = valid[x]
        lab = np.zeros(len(labels), bool)
        for i in np.flatnonzero(v & prm): lab[offsets[i]:offsets[i + 1]] = True
        lab &= labels >= 0
        pooled = auc(labels[lab] == 1, S[x][lab]) if lab.any() else np.nan
        wa = np.full(n, np.nan)
        for i in np.flatnonzero(v & prm):
            y = labels[offsets[i]:offsets[i + 1]]; s = S[x][offsets[i]:offsets[i + 1]]; ok = y >= 0
            if ok.sum() and (y[ok] == 1).any() and (y[ok] == 0).any(): wa[i] = auc(y[ok] == 1, s[ok])
        pk = np.full(n, -1)
        for i in np.flatnonzero(v & pb):
            s = S[x][offsets[i]:offsets[i + 1]]; pk[i] = int(np.argmax(s))
        pv = v & pb & (pk >= 0) & np.isfinite(det) & np.isfinite(fthr)
        pred = np.where(det >= fthr, pk, -1); pred[~pv] = -1
        met = pb_metrics(T, pred[pb], pv[pb], C); err = pb & (target >= 0) & pv
        results[x] = dict(prm_valid=int((v & prm).sum()), prm_pooled=pooled, prm_within=float(np.nanmean(wa)), pb_valid=int(pv.sum()),
                          pb_all8=met["macros"]["all"], pb_q8=met["macros"]["q8"], raw_exact=float(np.mean(pk[err] == target[err])),
                          within_one=float(np.mean(np.abs(pk[err] - target[err]) <= 1)), cells={k: y["f1"] for k, y in met["cells"].items()})
        per[x] = dict(within=wa, pred=pred, pv=pv); r_ = results[x]
        print(f"{x:30s} PRMB pooled {pooled:.5f} within {r_['prm_within']:.5f} | PB all8 {r_['pb_all8']*100:6.2f} Q8 {r_['pb_q8']*100:6.2f} exact {r_['raw_exact']*100:5.1f} within-1 {r_['within_one']*100:5.1f} | valid {r_['prm_valid']}/{r_['pb_valid']}")
    rng = np.random.default_rng(SEED); uniq, inv = np.unique(groups, return_inverse=True)
    draws = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float) for _ in range(DRAWS)]
    contrasts = {}
    for x in arms[1:]:
        A, B = per[x], per["w8_top10"]; common = np.isfinite(A["within"]) & np.isfinite(B["within"])
        dw, dpb = [], []
        for w in draws:
            dw.append(np.average(A["within"][common] - B["within"][common], weights=w[common]))
            pa = pb_metrics(T, A["pred"][pb], A["pv"][pb], C, weights=w[pb])["macros"]["all"]
            pb_ = pb_metrics(T, B["pred"][pb], B["pv"][pb], C, weights=w[pb])["macros"]["all"]
            if pa is not None and pb_ is not None: dpb.append(pa - pb_)
        contrasts[f"{x} minus w8_top10"] = dict(prm_within_point=float(np.mean(A["within"][common] - B["within"][common])),
            prm_within_ci=[float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5))],
            pb_all8_point=float(results[x]["pb_all8"] - results["w8_top10"]["pb_all8"]),
            pb_all8_ci=[float(np.percentile(dpb, 2.5)), float(np.percentile(dpb, 97.5))])
        cc = contrasts[f"{x} minus w8_top10"]
        print(f"{x:30s} minus w8_top10: within {cc['prm_within_point']:+.4f} [{cc['prm_within_ci'][0]:+.4f},{cc['prm_within_ci'][1]:+.4f}]  PB {cc['pb_all8_point']*100:+.2f}pp [{cc['pb_all8_ci'][0]*100:+.2f},{cc['pb_all8_ci'][1]*100:+.2f}]")
    json.dump(dict(results=results, contrasts=contrasts, rows_all_widths=n_full), open(OUT / "VOTE_METRICS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
