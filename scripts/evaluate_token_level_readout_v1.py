"""Token-level representation versus the 8-token window bank, answer-only, all 13,769 rows.

Finding that triggered this (2026-09-10): a single raw token stream read out with the
top-10 token-mean rule beats every window-bank fusion arm on both benchmarks. This script
records the headline arms with paired intervals, per-cell ProcessBench F1 and PRMScore:
token entropy, token top-k varentropy, token logprob margin, IU-PCR fitted on the answer's
own tokens over the nine primitive streams, versus the 27-feature window IU with the same
top-10 readout (ref27 of fusion_onset_innovation_iu_v1). Same entropy gate everywhere.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_cache" / "short_cycle01_code"))
import spectral_utils  # noqa: E402
spectral_utils.__path__.append(str(ROOT / "spectral_utils"))
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.answer_localization_v2 import STREAM_NAMES, PRIMITIVES  # noqa: E402

BENCH = ROOT / "results" / "localization_full_benchmark_v3"
ONSET = ROOT / "results" / "fusion_onset_innovation_iu_v1" / "scores"
OUT = ROOT / "results" / "token_level_readout_v1"
PRM = ROOT / "dataset_cache" / "four_localization" / "prmbench_qwen25math7b_full" / "prmbench_prm.pkl"


def auc(y, s):
    y = np.asarray(y, bool); p, n = y.sum(), (~y).sum()
    return float((rankdata(s)[y].sum() - p * (p + 1) / 2) / (p * n)) if p and n else np.nan


def topk(tok, ss, se, k=10):
    return np.asarray([np.sort(tok[a:b])[::-1][:min(k, b - a)].mean() for a, b in zip(ss, se)])


def zcol(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def main():
    OUT.mkdir(exist_ok=True)
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8")); z = np.load(BENCH / "evaluation" / "JOINED.npz")
    recs = j["records"]; off, lab, tgt = z["offsets"], z["labels"], z["target"]
    cells = np.array([r["cell"] for r in recs]); groups = np.array([r["group_id"] for r in recs]); pb = np.array([c.startswith("pb_") for c in cells])
    det = np.load(ROOT / "results" / "fusion_fixed_gate_v1" / "DETECTORS.npz")["entropy_mean"]
    gate = json.load(open(ROOT / "results" / "fusion_fixed_gate_v1" / "METRICS.json"))
    thr = {int(k): v for k, v in gate["arms"]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"].items()}
    folds = json.load(open(ROOT / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"))
    outer = np.array([folds["outer"].get(g, -1) for g in groups]); fthr = np.array([thr.get(int(f), np.nan) for f in outer])
    prim_idx = [STREAM_NAMES.index(s) for s in PRIMITIVES]; blocks = {}; spans = {}
    for cell in sorted(set(cells)):
        d = BENCH / "inputs" / cell; raw = np.load(d / "raw.npy", mmap_mode="r"); to = np.load(d / "token_offsets.npy")
        rid = np.load(d / "row_ids.npy", allow_pickle=True); idx = {str(x): k for k, x in enumerate(rid)}
        for i, r in enumerate(recs):
            if r["cell"] == cell:
                k = idx[r["row_id"]]; blocks[i] = np.asarray(raw[to[k]:to[k + 1]][:, prim_idx], float)
    for i, r in enumerate(recs):
        with np.load(BENCH / "scores" / f"{r['uid']}.npz") as f:
            spans[i] = (f["step_starts"], f["step_ends"])

    def token_iu(i):
        X = zcol(blocks[i]); keep = X.std(0) > 1e-8
        if keep.sum() < 3:
            return None
        X = X[:, keep]
        try:
            r = upcr_fit(X.T, **dict(IU_FIT_DEFAULTS))
        except Exception:  # noqa: BLE001
            return None
        s = X @ r.w
        if np.corrcoef(s, X[:, 0])[0, 1] < 0:
            s = -s
        return topk(s, *spans[i])

    def window_ref(i):
        p = ONSET / f"{recs[i]['uid']}.npz"
        if not p.exists():
            return None
        with np.load(p) as f:
            return f["ref27__risk"] if "ref27__risk" in f.files else None

    arms = {"token_entropy": lambda i: topk(blocks[i][:, 0], *spans[i]),
            "token_varentropy": lambda i: topk(blocks[i][:, 6], *spans[i]),
            "token_margin": lambda i: topk(-blocks[i][:, 4], *spans[i]),
            "token_iu9": token_iu,
            "window_iu27_top10": window_ref}
    S, per, res = {}, {}, {}
    T, C = tgt[pb], cells[pb]
    for name, fn in arms.items():
        Sa = np.full(len(lab), np.nan); pk = np.full(len(recs), -1); wa = np.full(len(recs), np.nan); valid = np.zeros(len(recs), bool)
        for i in range(len(recs)):
            s = fn(i)
            if s is None or not np.isfinite(s).all():
                continue
            valid[i] = True; Sa[off[i]:off[i + 1]] = s; pk[i] = int(np.argmax(s))
            if not pb[i]:
                y = lab[off[i]:off[i + 1]]; ok = y >= 0
                if (y[ok] == 1).any() and (y[ok] == 0).any():
                    wa[i] = auc(y[ok] == 1, s[ok])
        lm = np.zeros(len(lab), bool)
        for i in np.flatnonzero(valid & ~pb):
            lm[off[i]:off[i + 1]] = True
        lm &= lab >= 0
        pv = pb & valid & np.isfinite(det) & np.isfinite(fthr); pred = np.where(det >= fthr, pk, -1); pred[~pv] = -1
        m = pb_metrics(T, pred[pb], pv[pb], C); err = pb & (tgt >= 0) & valid
        S[name] = Sa; per[name] = dict(within=wa, pred=pred, pv=pv, valid=valid)
        res[name] = dict(pb_all8=m["macros"]["all"], pb_q4=m["macros"]["q4"], pb_q8=m["macros"]["q8"],
                         cells={c: x["f1"] for c, x in m["cells"].items()}, prm_within=float(np.nanmean(wa)),
                         prm_pooled=auc(lab[lm] == 1, Sa[lm]), raw_exact=float(np.mean(pk[err] == tgt[err])),
                         within_one=float(np.mean(np.abs(pk[err] - tgt[err]) <= 1)), valid=int(valid.sum()))
    cl = sorted(res["token_entropy"]["cells"])
    print("per-cell PB F1:"); print(" " * 20 + " ".join(f"{c[3:].replace('olympiadbench', 'olymp')[:11]:>11s}" for c in cl) + "   all8")
    for n_, r_ in res.items():
        print(f"{n_:20s}" + " ".join(f"{r_['cells'][c]*100:11.2f}" for c in cl) + f" {r_['pb_all8']*100:7.2f}   within {r_['prm_within']:.4f} pooled {r_['prm_pooled']:.4f} exact {r_['raw_exact']*100:.1f}")
    rng = np.random.default_rng(2026090707); uniq, inv = np.unique(groups, return_inverse=True)
    draws = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float) for _ in range(1000)]
    contrasts = {}
    for a, b in (("token_entropy", "window_iu27_top10"), ("token_iu9", "token_entropy"), ("token_varentropy", "token_entropy"), ("token_margin", "token_entropy")):
        A, B = per[a], per[b]; common = np.isfinite(A["within"]) & np.isfinite(B["within"]); dw, dpb = [], []
        for w in draws:
            dw.append(np.average(A["within"][common] - B["within"][common], weights=w[common]))
            pa = pb_metrics(T, A["pred"][pb], A["pv"][pb], C, weights=w[pb])["macros"]["all"]
            pb_ = pb_metrics(T, B["pred"][pb], B["pv"][pb], C, weights=w[pb])["macros"]["all"]
            if pa is not None and pb_ is not None:
                dpb.append(pa - pb_)
        contrasts[f"{a} minus {b}"] = dict(within=float(np.mean(A["within"][common] - B["within"][common])),
                                          within_ci=[float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5))],
                                          pb=float(res[a]["pb_all8"] - res[b]["pb_all8"]),
                                          pb_ci=[float(np.percentile(dpb, 2.5)), float(np.percentile(dpb, 97.5))])
        c = contrasts[f"{a} minus {b}"]
        print(f"{a} minus {b}: within {c['within']:+.4f} [{c['within_ci'][0]:+.4f},{c['within_ci'][1]:+.4f}]  PB {c['pb']*100:+.2f}pp [{c['pb_ci'][0]*100:+.2f},{c['pb_ci'][1]*100:+.2f}]")
    meta_all = list(pickle.load(open(PRM, "rb")).values()); mb = {m["idx"]: m for m in meta_all}
    prm = [i for i in range(len(recs)) if not pb[i]]
    for name in ("token_entropy", "token_varentropy", "token_iu9", "window_iu27_top10"):
        Sa = S[name]; pred = {}
        for k in sorted(set(outer[prm].tolist())):
            tr = [i for i in prm if outer[i] != k and per[name]["valid"][i]]; te = [i for i in prm if outer[i] == k and per[name]["valid"][i]]
            tau = float(np.quantile(np.concatenate([Sa[off[i]:off[i + 1]] for i in tr]), 0.8))
            for i in te:
                pred[i] = (~(Sa[off[i]:off[i + 1]] >= tau)).astype(int)
        r = prmbench_evaluate([{"idx": recs[i]["row_id"], "labels": [int(x) for x in v]} for i, v in pred.items()], [mb[recs[i]["row_id"]] for i in pred])
        res[name]["prmscore_q08"] = 0.5 * r["total"]["f1"] + 0.5 * r["total"]["negative_f1"]
        print(f"{name:20s} PRMScore(q0.8) {res[name]['prmscore_q08']:.4f}")
    json.dump(dict(results=res, contrasts=contrasts), open(OUT / "METRICS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
