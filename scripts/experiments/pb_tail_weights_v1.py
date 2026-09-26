#!/usr/bin/env python
"""pb_tail_weights_v1 (Omri, 2026-09-27): on ProcessBench, do L-SML weights learned from the binary
top-20% tail marks and applied to the continuous step signals beat equal weights on the SAME
representation?  Three representations (the seven CT7 streams read out Top10 per stream and then
fused, which is the ProcessBench leader; family15; bank11), five weight rules (equal, continuous
L-SML, tail, votes, partition-equal), two fit scopes (pooled PB+PRMB fit rows; PB-only fit rows).
Evaluation: calfix contract (fit 3 folds / calibrate (k+1)%5 / evaluate k), frozen calfix evaluator
for PRMScore, within-AUC and PB SLA, plus ProcessBench F1 under three decision rules.
Protocol: results/pb_tail_weights_v1/PROTOCOL.json (written and committed before any number).

    python -B scripts/experiments/pb_tail_weights_v1.py --config configs/pb_tail_weights_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct7_levers_common as L  # noqa: E402
import ct7_token_lsml_v1 as T  # noqa: E402
import ct7_token_tail_lsml_calfix_v1 as TT  # noqa: E402  (topk_readout; puts the ssl calfix directory on sys.path)

from spectral_utils.claude_feature_bank_v1 import fit_token_standardizer  # noqa: E402
from spectral_utils.ct7_token_streams import STREAMS, masked_answer_local  # noqa: E402
import calfix_common as CC  # noqa: E402
import calfix_evaluate as EV  # noqa: E402
import tail_calib_common as TC  # noqa: E402

SSL_ROOT = TT.SSL.parents[1]
V2LOCK = SSL_ROOT / "results/tail_threshold_calibration_v1/TRANSFER_LOCK_V2.json"
V2SHA = "0c5c55996503394edf2faeb1bc066e3fdec3c6eb315c79c0e7d59ca7d61c986c"
PREV_STEP = SSL_ROOT / "results/tail1_transfer_v3/run_20260924_2259"
PREV_TOKEN = L.ROOT / "results/ct7_token_tail_lsml_calfix_v1/run_20260924_1550"
SCR = Path("C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad")
POOL_SHA = "d9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16"
FRAC = 0.20
REPS = ("ct7r", "f15", "b11")
RULES = ("q80", "common_gate", "label")
T0 = time.perf_counter()


def log(*a):
    print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)


def fit_record(f, cols):
    rec = {"weights": dict(zip(cols, np.round(np.asarray(f["weights"], float), 10))), "groups": dict(zip(cols, np.asarray(f["groups"]).tolist())),
           "K": int(f["K"]), "anchor_spearman": float(f["anchor_spearman"]), "small_m_guarded": f["small_m_guarded"]}
    if "fit_sd" in f:
        rec.update({"fit_sd": dict(zip(cols, np.round(np.asarray(f["fit_sd"], float), 8))), "grouping_degenerate": bool(f["grouping_degenerate"])})
    return rec


def first_cross_many(s, taus):
    """First step with score >= tau for each tau (-1 when no step reaches it)."""
    cm = np.maximum.accumulate(s); j = np.searchsorted(cm, np.asarray(taus, float), side="left")
    return np.where(j < len(s), j, -1)


def contrast_lists(P):
    primary = [tuple(x) for x in P["primary_contrasts_holm10"]]
    desc = []
    for r in REPS:
        desc += [(f"{r}_tail", f"{r}_votes", "continuous application vs votes"),
                 (f"{r}_tail", f"{r}_part_equal", "learned weights vs the learned partition with equal weights"),
                 (f"{r}_tail_pb", f"{r}_equal", "PB-only tail vs matched equal"),
                 (f"{r}_tail_pb", f"{r}_cov_lsml_pb", "PB-only fit: tail vs continuous L-SML"),
                 (f"{r}_cov_lsml_pb", f"{r}_cov_lsml", "continuous L-SML: PB-only vs pooled fit"),
                 (f"{r}_tail_pb", f"{r}_votes_pb", "PB-only fit: continuous application vs votes"),
                 (f"{r}_tail_pb", f"{r}_part_equal_pb", "PB-only fit: learned weights vs partition"),
                 (f"{r}_tail", "ct7", "vs CT7 reference")]
    desc += [("ct7r_equal", "ct7r_leader", "column standardization bridge"), ("ct7r_leader", "ct7", "leader row vs CT7"),
             ("ct7r_tail", "f15_tail", "representation under the tail rule"), ("ct7r_tail_pb", "ct7r_leader", "PB-only tail vs the leader row")]
    return primary, desc


# ------------------------------------------------------------------ ProcessBench F1
def pb_evaluate(out, pop, d, primary, desc, draws, seed):
    """ProcessBench F1 under three decision rules with a paired source-group bootstrap over ALL PB answers."""
    Z = np.load(out / "SCORES.npz"); methods = sorted(k[len("eval__"):] for k in Z.files if k.startswith("eval__"))
    E = {m: Z["eval__" + m] for m in methods}; C = {m: Z["cal__" + m] for m in methods}
    off = pop.off; pbi = np.flatnonzero(pop.pb); cells = sorted(set(pop.cells[pbi])); cix = np.array([cells.index(c) for c in pop.cells[pbi]])
    tgt = pop.target[pbi]; err = tgt >= 0; fold = pop.fold[pbi]; nc = len(cells)
    ev_thr = json.loads((out / "THRESHOLDS.json").read_text(encoding="utf8"))

    def rates(pred, idx):
        ok = np.where(err[idx], pred == tgt[idx], pred == -1); r = []
        for ci in range(nc):
            mc = cix[idx] == ci; ce = mc & ~err[idx]; ee = mc & err[idx]
            r.append((ok[ce].mean() if ce.any() else np.nan, ok[ee].mean() if ee.any() else np.nan, int(ce.sum()), int(ee.sum())))
        return r

    def f1(ca, ea):
        return 2 * ca * ea / (ca + ea) if ca + ea > 0 else 0.0

    def macro(pred, idx):
        return float(np.nanmean([f1(a, b) for a, b, _, _ in rates(pred, idx)]))

    PRED, THR = {}, {}
    for m in methods:
        e, c = E[m], C[m]
        am = np.array([int(np.argmax(e[off[i]:off[i + 1]])) for i in pbi])           # cvf_v2 locator
        PRED[m, "common_gate"] = np.where(d.gate[pbi], am, -1)
        pq = np.full(len(pbi), -9); pl = np.full(len(pbi), -9); THR[m] = []
        for k in range(5):
            _fit, cal, ev = CC.roles_of(k)
            cs = c[pop.rows(pop.fold == cal)]; tau80 = float(np.quantile(cs, .8)); grid = np.quantile(cs, EV.QGRID)
            assert abs(tau80 - ev_thr[m]["P1"][k]["tau"]) < 1e-15, (m, k)                # same threshold as the PRMScore P1 panel
            ca_ = np.flatnonzero(fold == cal)
            G = np.stack([first_cross_many(c[off[pbi[j]]:off[pbi[j] + 1]], grid) for j in ca_])
            gs = [macro(G[:, g], ca_) for g in range(len(grid))]; best = int(np.argmax(gs)); tau_l = float(grid[best])
            for j in np.flatnonzero(fold == ev):
                pq[j], pl[j] = first_cross_many(e[off[pbi[j]]:off[pbi[j] + 1]], [tau80, tau_l])
            THR[m].append({"outer_fold": k, "cal_fold": cal, "tau_q80": tau80, "tau_label": tau_l, "label_quantile": float(EV.QGRID[best]), "label_cal_macro_f1": float(gs[best])})
        assert (pq > -9).all() and (pl > -9).all(), m
        PRED[m, "q80"] = pq; PRED[m, "label"] = pl
    allidx = np.arange(len(pbi)); point, cellrows = {}, []
    for m in methods:
        point[m] = {"method": m}
        for rule in RULES:
            r = rates(PRED[m, rule], allidx); fs = [f1(a, b) for a, b, _, _ in r]
            point[m].update({f"pb_f1_{rule}": float(np.mean(fs)), f"pb_clean_acc_{rule}": float(np.mean([a for a, _, _, _ in r])),
                             f"pb_err_acc_{rule}": float(np.mean([b for _, b, _, _ in r])), f"pb_flag_rate_{rule}": float(np.mean(PRED[m, rule] >= 0))})
            cellrows += [{"method": m, "rule": rule, "cell": cells[ci], "n_clean": r[ci][2], "n_err": r[ci][3], "clean_acc": r[ci][0], "err_acc": r[ci][1], "f1": fs[ci]} for ci in range(nc)]
    # anchor: the frozen CT7 gate + argmax reproduces CT7's standing macro F1
    anchor = point["ct7"]["pb_f1_common_gate"] - L.CT7_MACRO_F1
    assert abs(anchor) < 1e-12, anchor
    # bootstrap over the source groups of all PB answers
    ug, ginv = np.unique(pop.groups[pbi], return_inverse=True); H = len(ug)
    n_clean = np.zeros((H, nc)); np.add.at(n_clean, (ginv[~err], cix[~err]), 1.0)
    n_err = np.zeros((H, nc)); np.add.at(n_err, (ginv[err], cix[err]), 1.0)
    OK = {}
    for key, pred in PRED.items():
        ok = np.where(err, pred == tgt, pred == -1).astype(float)
        a = np.zeros((H, nc)); np.add.at(a, (ginv[~err], cix[~err]), ok[~err]); b = np.zeros((H, nc)); np.add.at(b, (ginv[err], cix[err]), ok[err]); OK[key] = (a, b)
    rng = np.random.default_rng(seed); est = {key: np.empty(draws) for key in PRED}; pos = 0
    while pos < draws:
        nb = min(2000, draws - pos); V = rng.multinomial(H, np.full(H, 1 / H), size=nb).astype(float)
        dc = V @ n_clean; de = V @ n_err
        assert (dc > 0).all() and (de > 0).all()
        for key, (a, b) in OK.items():
            ca = (V @ a) / dc; ea = (V @ b) / de
            est[key][pos:pos + nb] = np.divide(2 * ca * ea, ca + ea, out=np.zeros_like(ca), where=(ca + ea) > 0).mean(1)
        pos += nb
    rows = []
    for fam, pairs in (("primary", primary), ("descriptive", desc)):
        for a, b, why in pairs:
            for rule in RULES:
                dd = est[a, rule] - est[b, rule]
                rows.append({"family": fam, "a": a, "b": b, "why": why, "endpoint": f"pb_f1_{rule}", "delta": point[a][f"pb_f1_{rule}"] - point[b][f"pb_f1_{rule}"],
                             "ci95_lo": float(np.quantile(dd, .025)), "ci95_hi": float(np.quantile(dd, .975)), "p_boot": EV.pvalue(dd)})
    CT = pd.DataFrame(rows); CT["p_holm_primary"] = np.nan
    sel = (CT.family == "primary") & (CT.endpoint == "pb_f1_q80"); CT.loc[sel, "p_holm_primary"] = EV.holm(CT.loc[sel, "p_boot"].to_numpy())
    CT.to_csv(out / "PB_CONTRASTS.csv", index=False); pd.DataFrame(list(point.values())).to_csv(out / "PB_METRICS.csv", index=False)
    pd.DataFrame(cellrows).to_csv(out / "PB_CELLS.csv", index=False); CC.dump(out / "PB_THRESHOLDS.json", THR)
    return point, CT, {"pb_answers": int(len(pbi)), "pb_error_answers": int(err.sum()), "pb_clean_answers": int((~err).sum()), "pb_source_groups_all": H,
                       "cells": cells, "draws": draws, "seed": seed, "common_gate_ct7_anchor_diff": anchor, "gate_open_rate": float(d.gate[pbi].mean())}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--draws", type=int, default=20_000)
    args = ap.parse_args()
    d = L.light_dataset(args.config)                                   # replays CT7 gated macro-F1 and within-AUC
    stage = d.out; P = json.loads((stage / "PROTOCOL.json").read_text(encoding="utf8"))
    out = stage / f"run_{datetime.now():%Y%m%d_%H%M}"; out.mkdir(parents=True, exist_ok=False)
    pop = CC.Population(); off = pop.off; n = pop.n
    assert np.array_equal(pop.off, d.off) and np.array_equal(pop.fold, d.fold) and list(pop.ids) == [str(x) for x in d.ids]
    assert np.array_equal(pop.pb, d.pb) and np.array_equal(pop.target[pop.pb], d.target[d.pb])
    man = json.loads((PREV_STEP / "INPUT_MANIFEST.json").read_text(encoding="utf8"))
    code = {"calfix_common": CC.sha(TT.SSL / "calfix_common.py"), "calfix_evaluate": CC.sha(TT.SSL / "calfix_evaluate.py"), "tail_calib_common": CC.sha(TT.SSL / "tail_calib_common.py")}
    assert (code["calfix_common"], code["calfix_evaluate"], code["tail_calib_common"]) == (man["calfix_common_sha256"], man["evaluator_sha256"], man["tail_calib_common_sha256"]), code
    assert CC.extgen_hashes() == man["extgen"]
    assert CC.sha(V2LOCK) == V2SHA and CC.sha(SCR / "pool_z.npy") == POOL_SHA
    CC.dump(out / "INPUT_MANIFEST.json", {"script": CC.sha(Path(__file__)), "protocol": CC.sha(stage / "PROTOCOL.json"), "config": CC.sha(Path(args.config)), **code,
                                          "extgen": CC.extgen_hashes(), "lock_v2": V2SHA, "pool_z": POOL_SHA, "ct7_token_tail_calfix_script": CC.sha(Path(TT.__file__)),
                                          "ct7_token_lsml_v1": CC.sha(Path(T.__file__)), "levers_common": CC.sha(HERE / "ct7_levers_common.py"),
                                          "ct7_token_streams": CC.sha(L.ROOT / "spectral_utils/ct7_token_streams.py"), "claude_feature_bank_v1": CC.sha(L.ROOT / "spectral_utils/claude_feature_bank_v1.py"),
                                          **{k: CC.sha(v) for k, v in d.c["paths"].items() if k in ("ct7_tokens", "ct7", "joined", "folds") and Path(v).exists()}})
    rng = np.random.default_rng(0)                                     # first-crossing helper vs brute force
    for _ in range(200):
        s = rng.standard_normal(rng.integers(1, 30)); taus = rng.standard_normal(5)
        want = [next((j for j, v in enumerate(s) if v >= t), -1) for t in taus]
        assert list(first_cross_many(s, taus)) == want

    # ---------------------------------------------------------------- family15 and bank11 (tail1_transfer_v3 construction)
    R = json.loads(V2LOCK.read_text(encoding="utf8"))["recipe"]
    POOL = np.load(SCR / "pool_z.npy"); PN = json.loads((SCR / "pool_names.json").read_text(encoding="utf8"))
    names = R["channels_48"]; ix = {c: j for j, c in enumerate(names)}
    Zc = pop.answer_standardize(POOL[:, [PN.index(c) for c in names]]); del POOL
    S3 = R["families_15"]; FAMS = list(S3); sign = R["source_signs"]; B11N = list(R["bank11"])
    F15 = pop.answer_standardize(np.column_stack([np.column_stack([Zc[:, ix[c]] * sign[c] for c in mem]).mean(1) for mem in S3.values()]))
    B11 = Zc[:, [ix[c] for c in B11N]].copy(); del Zc
    STEPREP = {"f15": (F15, FAMS, FAMS.index("level_entropy")), "b11": (B11, B11N, 0)}
    MARKS, DEGEN = {}, {}
    for r, (X, _cols, _a) in STEPREP.items():
        MARKS[r], DEGEN[r] = CC.tail_marks(X, off, FRAC, tie_aware=True)
    log("family15/bank11 built", {r: {k: round(v, 4) for k, v in g.items()} for r, g in DEGEN.items()})

    # ---------------------------------------------------------------- CT7 token streams (ct7_token_tail_lsml_calfix_v1 preparation)
    td = T.TokenData(d.c["paths"]["ct7_tokens"], d)
    prep = T.prepared_matrices(td, despike=True, cols=T.ALL)
    mats = [masked_answer_local(x, v) for x, v, _ in prep]; allv = [p[2] for p in prep]; sp = td.sp
    del prep, td
    log("token streams prepared")

    B = CC.ScoreBundle(pop); diag = {"K": {}, "groups": {}, "fit_rows": {}, "ct7r_column_standardization": {}, "mark_degeneracy": {**DEGEN, "ct7r": []}}

    def learned(r, s, X, M, anc, cols, fr, emit):
        t = TC.lsml_fit_scaled(M[fr], anc, X[fr], standardize=True, loading_scale="unit"); w = np.asarray(t["weights"], float)
        rec = {"scope": s or "pooled", "fit_rows": int(len(fr)), **fit_record(t, cols)}
        emit(f"{r}_tail{s}", X @ w, {"rule": "tie-aware top-20% marks, standardized, L-SML; weights applied to the continuous columns", **rec})
        emit(f"{r}_votes{s}", M @ (w / np.asarray(t["fit_sd"], float)), {"rule": "tail weights applied to the standardized marks", **rec})
        pe = CC.partition_equal(t["groups"])
        emit(f"{r}_part_equal{s}", X @ pe, {"rule": "equal weight per tail-fit group, applied to the continuous columns", "applied_weights": dict(zip(cols, pe)), **rec})
        diag["K"].setdefault(f"{r}_tail{s}", []).append(int(t["K"])); diag["groups"].setdefault(f"{r}_tail{s}", []).append(np.asarray(t["groups"]).tolist())

    for k in range(5):
        tk = time.perf_counter(); fit_f, cal, ev = CC.roles_of(k); fit_ans = np.isin(pop.fold, fit_f)
        FR = {"": pop.rows(fit_ans), "_pb": pop.rows(fit_ans & pop.pb)}
        diag["fit_rows"][k] = {s or "pooled": int(len(v)) for s, v in FR.items()}

        def emit_step(m, raw, rec):
            B.put_full(m, k, pop.answer_z(raw), rec)

        def emit_ct7r(m, raw, rec):
            B.put_full(m, k, L.answer_z(raw, off), rec)

        emit_step("ct7", pop.ct7, {"rule": "frozen CT7 step scores, answer-z"})
        for r, (X, cols, anc) in STEPREP.items():
            emit_step(f"{r}_equal", X.mean(1), {"rule": "equal"})
            for s, fr in FR.items():
                f = CC.lsml_fit(X[fr], anc)
                emit_step(f"{r}_cov_lsml{s}", X @ f["weights"], {"rule": "continuous L-SML (calfix_common.lsml_fit)", "scope": s or "pooled", "fit_rows": int(len(fr)), **fit_record(f, cols)})
                diag["K"].setdefault(f"{r}_cov_lsml{s}", []).append(int(f["K"]))
                learned(r, s, X, MARKS[r], anc, cols, fr, emit_step)
        # CT7 readouts: donor token standardizer on the fit folds, Top10 per stream over the all-valid mask
        train = np.flatnonzero(fit_ans)
        std = fit_token_standardizer((mats[i][allv[i]] for i in train), cap=T.TOKEN_CAP)
        Rk = np.empty((pop.total, 7))
        for i in range(n):
            zs = (mats[i] - std["mean"]) / std["std"]
            Rk[off[i]:off[i + 1]] = np.column_stack([TT.topk_readout(zs[:, j], allv[i], sp[i], 10) for j in range(7)])
        assert np.isfinite(Rk).all()
        stdrec = {"standardizer_mean": std["mean"], "standardizer_std": std["std"], "sample_count": std["sample_count"]}
        emit_ct7r("ct7r_leader", Rk.mean(1), {"rule": "Top10 per standardized stream over the all-valid mask, then equal mean (readout_then_equal_matched_r10)", **stdrec})
        mu = Rk[FR[""]].mean(0); sd = Rk[FR[""]].std(0); assert (sd > CC.EPS).all()
        X = (Rk - mu) / sd; M, dg = CC.tail_marks(X, off, FRAC, tie_aware=True)
        diag["ct7r_column_standardization"][k] = {"mean": dict(zip(STREAMS, mu)), "sd": dict(zip(STREAMS, sd))}; diag["mark_degeneracy"]["ct7r"].append(dg)
        emit_ct7r("ct7r_equal", X.mean(1), {"rule": "equal mean of the column-standardized readouts", "column_mean": mu, "column_sd": sd, **stdrec})
        for s, fr in FR.items():
            f = TC.lsml_fit_scaled(X[fr], 0, standardize=True, loading_scale="unit")
            emit_ct7r(f"ct7r_cov_lsml{s}", X @ f["weights"], {"rule": "continuous L-SML (lsml_fit_scaled)", "scope": s or "pooled", "fit_rows": int(len(fr)),
                                                           "column_mean": mu, "column_sd": sd, **fit_record(f, list(STREAMS))})
            diag["K"].setdefault(f"ct7r_cov_lsml{s}", []).append(int(f["K"]))
            learned("ct7r", s, X, M, 0, list(STREAMS), fr, emit_ct7r)
        log(f"fold {k} done ({time.perf_counter() - tk:.0f}s); K", {m: v[-1] for m, v in diag["K"].items()})

    # ---------------------------------------------------------------- replay gates
    old_step = np.load(PREV_STEP / "SCORES.npz"); old_tok = np.load(PREV_TOKEN / "SCORES.npz")
    pairs = [("ct7", old_step, "ct7"), ("f15_equal", old_step, "F15_equal"), ("f15_cov_lsml", old_step, "F15_cov_lsml"), ("f15_tail", old_step, "F15_tailstd_lsml"),
             ("b11_equal", old_step, "B11_equal"), ("b11_cov_lsml", old_step, "B11_lsml"), ("ct7r_leader", old_tok, "readout_then_equal_matched_r10")]
    replay = {new: max(float(np.abs(B.scores[new, r] - src[f"{r}__{o}"]).max()) for r in CC.ROLES) for new, src, o in pairs}
    ok = all(v < 1e-12 for v in replay.values()); replay["status"] = "PASS" if ok else "FAIL"
    CC.dump(out / "REPLAY.json", replay); log("replay", replay); assert ok, replay
    checks = B.finalize(out); CC.dump(out / "DIAGNOSIS.json", diag); log("bundle", checks["status"], checks["methods"], "rows")

    # ---------------------------------------------------------------- evaluation
    primary, desc = contrast_lists(P)
    res = EV.evaluate(out, out, primary, desc, pop=pop, draws=args.draws)
    log("frozen evaluator done")
    pbp, pbct, pbchk = pb_evaluate(out, pop, d, primary, desc, args.draws, 20260927)
    log("ProcessBench F1 done", pbchk)
    CT = res["contrasts"].copy()
    sel = (CT.family == "primary") & (CT.endpoint == "pb_sla_macro8"); CT["p_holm_primary_sla"] = np.nan
    CT.loc[sel, "p_holm_primary_sla"] = EV.holm(CT.loc[sel, "p_boot"].to_numpy()); CT.to_csv(out / "CONTRASTS_WITH_SLA_HOLM.csv", index=False)
    MT = pd.read_csv(out / "METRICS.csv").merge(pd.DataFrame(list(pbp.values())), on="method")
    MT["K_per_fold"] = MT.method.map(lambda m: " ".join(map(str, diag["K"].get(m, diag["K"].get(m.replace("_votes", "_tail").replace("_part_equal", "_tail"), [])))))
    MT.to_csv(out / "SUMMARY_METRICS.csv", index=False)
    prim = []
    for a, b, why in primary:
        g = lambda df, e: df[(df.a == a) & (df.b == b) & (df.endpoint == e)].iloc[0]
        s_, f_, p_ = g(CT, "pb_sla_macro8"), g(pbct, "pb_f1_q80"), g(CT, "prmscore_P1")
        prim.append({"a": a, "b": b, "why": why,
                     "sla": [s_.delta, s_.ci95_lo, s_.ci95_hi, s_.p_holm_primary_sla], "f1_q80": [f_.delta, f_.ci95_lo, f_.ci95_hi, f_.p_holm_primary],
                     "prmscore_P1": [p_.delta, p_.ci95_lo, p_.ci95_hi, p_.p_holm_primary]})
    CC.dump(out / "SUMMARY.json", {"primary_contrasts": prim, "pb_checks": pbchk, "replay": replay, "K": diag["K"]})
    CC.dump(out / "RUN_STATUS.json", {"status": "COMPLETE", "seconds": time.perf_counter() - T0})
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["method", "pb_sla_macro8", "pb_f1_q80", "pb_f1_common_gate", "pb_f1_label", "prmscore_P1", "within_auc", "K_per_fold"]
    print(MT[cols].sort_values("pb_sla_macro8", ascending=False).round(4).to_string(index=False))
    for r in prim:
        print(r["a"], "vs", r["b"], {k: [round(float(x), 4) for x in r[k]] for k in ("sla", "f1_q80", "prmscore_P1")})
    log("written", out)


if __name__ == "__main__":
    main()
