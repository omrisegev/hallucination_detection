#!/usr/bin/env python
"""Calibration-corrected replay of ct7_token_tail_lsml_v1 (handoff 2026-09-24, sections 2-5).

Same token pipeline as v1; what changes is the evaluation contract: each outer fold fits on three
folds, one fitted bundle scores its calibration fold and its evaluation fold into write-once arrays
(model_id shared by both roles), and thresholds come from that bundle's calibration scores only
(ssl-pseudolabel-residual-v1/scripts/experiments/calfix_{common,evaluate}.py).  Adds the declared
tie-aware candidate, its two controls and a matched order-of-operations pair.
Protocol: results/ct7_token_tail_lsml_calfix_v1/PROTOCOL.json (written before any number).

    python -B scripts/experiments/ct7_token_tail_lsml_calfix_v1.py --config configs/ct7_token_tail_lsml_calfix_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct7_levers_common as L  # noqa: E402
import ct7_token_lsml_v1 as T  # noqa: E402

from spectral_utils.claude_feature_bank_v1 import fit_l_sml_weights, fit_token_standardizer, fuse_token_matrix  # noqa: E402
from spectral_utils.ct7_token_streams import STREAMS, masked_answer_local, masked_step_top10  # noqa: E402

SSL = Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees\ssl-pseudolabel-residual-v1\scripts\experiments'); sys.path.insert(0, str(SSL))
import calfix_common as CC  # noqa: E402
import calfix_evaluate as EV  # noqa: E402

MARKS = {"top20pct": ("frac", 0.20), "top30tok": ("count", 30), "top10tok": ("count", 10)}
READOUTS = {"r10": 10, "r30": 30}


def topk_readout(values, valid, spans, k):
    """Top-k mean over the valid tokens of each step; NaN for a step without valid tokens."""
    out = np.full(len(spans), np.nan)
    for s, (a, b) in enumerate(spans):
        v = values[a:b][valid[a:b]]
        if len(v):
            kk = min(k, len(v)); out[s] = np.partition(v, len(v) - kk)[-kk:].mean()
    return out


def marks(x, rows, mark, tie_aware=False, centred=True):
    """Per stream over the answer's all-valid tokens: 1 on the top-k (historical: stable argsort,
    ties by position; tie-aware: boundary ties share the remaining mass), optionally centred."""
    kind, val = mark; out = np.zeros_like(x); idx = np.flatnonzero(rows); st = np.zeros(3, int)   # constant, boundary tie, n<k
    if len(idx) == 0:
        return out, st
    k = max(1, int(np.ceil(val * len(idx)))) if kind == "frac" else min(int(val), len(idx))
    st[2] = int(kind == "count" and len(idx) < val) * x.shape[1]
    for j in range(x.shape[1]):
        v = x[idx, j]; thr = np.sort(v)[::-1][k - 1]; gt = v > thr; eq = v == thr
        if v.std() <= 1e-12:
            st[0] += 1
        elif eq.sum() > 1 and gt.sum() + eq.sum() > k:
            st[1] += 1
        if tie_aware:
            out[idx[gt], j] = 1.0; out[idx[eq], j] = (k - gt.sum()) / eq.sum()
        else:
            out[idx[np.argsort(-v, kind="stable")[:k]], j] = 1.0
        if centred:
            out[idx, j] -= k / len(idx)
    return out, st


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--draws", type=int, default=20_000)
    args = p.parse_args(); started = time.perf_counter()
    d = L.light_dataset(args.config); stage = d.out; out = stage / f"run_{datetime.now():%Y%m%d_%H%M}"; out.mkdir(parents=True, exist_ok=False)
    pop = CC.Population()
    assert np.array_equal(pop.off, d.off) and np.array_equal(pop.fold, d.fold) and list(pop.ids) == [str(x) for x in d.ids]
    td = T.TokenData(d.c["paths"]["ct7_tokens"], d)
    CC.dump(out / "INPUT_MANIFEST.json", {"script": CC.sha(Path(__file__)), "token_lsml_v1": CC.sha(Path(T.__file__)), "levers_common": CC.sha(HERE / "ct7_levers_common.py"),
                                         "calfix_common": CC.sha(SSL / "calfix_common.py"), "calfix_evaluate": CC.sha(SSL / "calfix_evaluate.py"),
                                         "ct7_token_streams": CC.sha(L.ROOT / "spectral_utils/ct7_token_streams.py"), "claude_feature_bank_v1": CC.sha(L.ROOT / "spectral_utils/claude_feature_bank_v1.py"),
                                         "config": CC.sha(Path(args.config)), "protocol": CC.sha(stage / "PROTOCOL.json"),
                                         **{k: CC.sha(v) for k, v in d.c["paths"].items() if k in ("ct7_tokens", "ct7", "joined", "folds") and Path(v).exists()}})
    prep = T.prepared_matrices(td, despike=True, cols=T.ALL)
    mats = [masked_answer_local(x, v) for x, v, _ in prep]                      # answer-local axis, as v1
    allv = [pr[2] for pr in prep]
    rng = np.random.default_rng(0)
    for i in rng.choice(d.n, 300, replace=False):                                  # readout equivalence
        for kk in (10, 30):
            a = topk_readout(mats[i][:, 0], allv[i], td.sp[i], kk); b = masked_step_top10(mats[i][:, 0], allv[i], td.sp[i], k=kk)
            assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-12)
    MK, degen = {}, {}
    # float32 storage (memory); votes = uncentred marks differ from the centred ones by a per-answer
    # constant per stream, so votes@w differs by a per-answer constant, removed by the final answer-z
    for name, spec, tie, cen in [(m, s, False, True) for m, s in MARKS.items()] + [("tailtie20", MARKS["top20pct"], True, True)]:
        res = [marks(mats[i], allv[i], spec, tie, cen) for i in range(d.n)]; MK[name] = [r[0].astype(np.float32) for r in res]
        st = np.sum([r[1] for r in res], 0); cells = d.n * 7
        degen[name] = {"answer_streams": cells, "constant_rate": st[0] / cells, "boundary_tie_rate": st[1] / cells, "fewer_than_k_rate": st[2] / cells}
    nan_steps = int(sum(np.isnan(topk_readout(np.zeros(len(allv[i])), allv[i], td.sp[i], 10)).sum() for i in range(d.n)))
    degen["steps_without_all_valid_token"] = nan_steps; degen["all_valid_token_rate"] = float(np.mean(np.concatenate(allv)))
    CC.dump(out / "DEGENERACY.json", degen); print("marks built", json.dumps(degen), flush=True)

    B = CC.ScoreBundle(pop); total = pop.total; timing = {}

    def put(method, k, answers_scores, record):
        """answers_scores: dict answer -> step scores (before answer-z) for the cal and eval answers of fold k."""
        full = np.full(total, np.nan)
        for i, s in answers_scores.items():
            full[d.off[i]:d.off[i + 1]] = s
        full = L.answer_z(full, d.off)                                             # NaN steps -> neutral 0 (as v1)
        _fit, cal, ev = CC.roles_of(k); mid = CC.model_id(method, k, record)
        for role, f in (("eval", ev), ("cal", cal)):
            B.put_answers(method, k, role, np.flatnonzero(pop.fold == f), full, mid)
        B.models.append({"method": method, "outer_fold": k, "fit_folds": _fit, "cal_fold": cal, "eval_fold": ev, "model_id": mid, **record})

    fixed = {}                                                                     # model-free references
    s_views, _ = T.ct7_style_arm(td, prep); fixed["readout_then_equal_ct7views"] = L.answer_z(s_views, d.off)
    fixed["ct7"] = L.answer_z(d.references["ct7"], d.off); fixed["ct7_raw"] = d.references["ct7"].astype(float)
    for k in range(5):
        t0 = time.perf_counter(); fit_f, cal, ev = CC.roles_of(k)
        train = np.flatnonzero(np.isin(d.fold, fit_f)); score = np.flatnonzero(np.isin(d.fold, [cal, ev]))
        std = fit_token_standardizer((mats[i][allv[i]] for i in train), cap=T.TOKEN_CAP)
        stdrec = {"standardizer_mean": std["mean"], "standardizer_std": std["std"], "sample_count": std["sample_count"], "fit_answers": int(len(train))}
        W, recs = {"equal": None}, {"equal": {"rule": "equal"}}
        w, meta = fit_l_sml_weights((mats[i][allv[i]] for i in train), std); W["cov_lsml"] = w
        recs["cov_lsml"] = {"rule": "token L-SML on continuous streams", "weights": dict(zip(STREAMS, w)), "K": int(meta["K"]), "groups": np.asarray(meta["c"]).tolist()}
        for m in list(MARKS) + ["tailtie20"]:
            istd = fit_token_standardizer((MK[m][i][allv[i]] for i in train), cap=T.TOKEN_CAP)
            w, mt = fit_l_sml_weights((MK[m][i][allv[i]] for i in train), istd)
            key = f"tail_{m}_lsml" if m in MARKS else "tailtie20_lsml"; W[key] = w
            recs[key] = {"rule": f"token L-SML on {m} marks, applied to continuous streams", "weights": dict(zip(STREAMS, w)), "K": int(mt["K"]), "groups": np.asarray(mt["c"]).tolist(),
                         "mark_standardizer_mean": istd["mean"], "mark_standardizer_std": istd["std"]}
        g = np.asarray(recs["tailtie20_lsml"]["groups"]); pe = np.array([1 / (len(np.unique(g)) * np.count_nonzero(g == q)) for q in g])
        acc = {}
        for i in score:
            sp = td.sp[i]; zs = (mats[i] - std["mean"]) / std["std"]
            for r, w in W.items():
                fl, fe = fuse_token_matrix(mats[i], std, weights=w); fused = fe if w is None else fl
                for ro, kk in READOUTS.items():
                    if r.startswith("tailtie20") and ro == "r30":
                        continue
                    acc.setdefault(f"{r}_{ro}", {})[i] = topk_readout(fused, allv[i], sp, kk)
            acc.setdefault("tailtie20_partition_equal_r10", {})[i] = topk_readout(zs @ pe, allv[i], sp, 10)
            acc.setdefault("tailtie20_votes_r10", {})[i] = topk_readout(MK["tailtie20"][i].astype(float) @ W["tailtie20_lsml"], allv[i], sp, 10)
            # one shared all-valid mask, so a step without valid tokens is NaN in every stream (as in the fused arm)
            acc.setdefault("readout_then_equal_matched_r10", {})[i] = np.column_stack([topk_readout(zs[:, j], allv[i], sp, 10) for j in range(7)]).mean(1)
        for name, sc in acc.items():
            base = name.rsplit("_", 1)[0]
            rec = dict(recs.get(base, {}))
            if name == "tailtie20_partition_equal_r10":
                rec = {"rule": "partition_equal of the tailtie20 fit groups", "weights": dict(zip(STREAMS, pe)), "groups": g.tolist()}
            elif name == "tailtie20_votes_r10":
                rec = {"rule": "tailtie20 weights on uncentred tie-aware marks", "weights": recs["tailtie20_lsml"]["weights"]}
            elif name == "readout_then_equal_matched_r10":
                rec = {"rule": "Top10 per standardized stream over the all-valid mask, then equal mean"}
            put(name, k, sc, {**rec, **stdrec, "readout": name.rsplit("_", 1)[1]})
        for name, full in fixed.items():
            B.put_full(name, k, full, {"rule": "fixed scores, no fitted component", "source": name})
        timing[k] = time.perf_counter() - t0
        print(f"fold {k} done ({timing[k]:.0f}s): " + ", ".join(f"{r} K={recs[r].get('K')}" for r in recs if r != "equal"), flush=True)
    checks = B.finalize(out); print("bundle", checks["status"], flush=True)
    P = json.loads((stage / "PROTOCOL.json").read_text(encoding="utf8"))
    primary = [tuple(x) for x in P["primary_contrasts_P1_prmscore_holm6"]]
    desc = [(f"tail_{m}_lsml_{ro}", f"cov_lsml_{ro}", "tail vs covariance") for m in MARKS for ro in READOUTS] + \
           [(f"tail_{m}_lsml_{ro}", f"equal_{ro}", "tail vs equal") for m in MARKS for ro in READOUTS] + \
           [("cov_lsml_r30", "equal_r30", "covariance vs equal, Top30"), ("equal_r30", "equal_r10", "readout Top30 vs Top10"),
            ("tailtie20_lsml_r10", "tail_top20pct_lsml_r10", "tie-aware vs historical marks"), ("readout_then_equal_ct7views", "readout_then_equal_matched_r10", "per-view masks/normalization vs matched"),
            ("readout_then_equal_ct7views", "equal_r10", "old ordering comparison (unmatched)"), ("ct7", "ct7_raw", "CT7 normalization")] + \
           [(a, "ct7", "vs CT7 reference") for a in ("equal_r10", "cov_lsml_r10", "tailtie20_lsml_r10", "tail_top20pct_lsml_r10", "readout_then_equal_ct7views", "readout_then_equal_matched_r10")]
    res = EV.evaluate(out, out, primary, desc, pop=pop, draws=args.draws)
    CC.dump(out / "RUN_STATUS.json", {"status": "COMPLETE", "timing_fold_seconds": timing, "total_seconds": time.perf_counter() - started, "degeneracy": degen})
    import pandas as pd
    pd.set_option("display.width", 250)
    MT = pd.read_csv(out / "METRICS.csv"); print(MT[["method", "prmscore_P1", "prmscore_P1b", "prmscore_P2", "within_auc", "pb_sla_macro8"]].round(4).to_string(index=False))
    CT = res["contrasts"]; print(CT[(CT.family == "primary") & (CT.endpoint == "prmscore_P1")].round(4).to_string(index=False))
    print("written:", out, f"({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
