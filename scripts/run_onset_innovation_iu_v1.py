"""Onset (C7) and self-innovation (C8) streams inside the answer-only IU bank, all 13,769 rows.

Follows Omri's 2026-09-09 decision to try "when does it change" features for exact-step
location. Both features are ported from the Claude worktree (Phase 2 atomic candidates,
scripts/reasoning_localization/run_phase2_atomic_remaining.py) into the answer-only
pipeline, fitted from the current answer only:

* C7 onset: entropy z-scored over the answer's tokens; burst = max(diff(z) - 1.36, 0);
  rebound onset = positive increments of max(z - running_min(z) - 1.33, 0); onset =
  max(burst, rebound). Thresholds 1.36 / 1.33 are frozen constants borrowed from the
  Claude line (chosen there on other data; declared, not tuned here).
* C8 self-innovation: for each of the nine primitive streams, weighted-ridge AR(1) with
  predictors [1, log1p(position), x[t-1]] fitted on this answer's tokens (ridge 1.0,
  intercept unpenalized), residual divided by its RMS; innovation stream = |residual|.

Each added stream enters the moment bank exactly like a primitive (level / sd / slope per
8-token window), the bank is standardized and oriented with the frozen answer-only rules,
IU-PCR is fit with the frozen defaults, and step scores use the top-10 token-mean readout.
All arms use the moment bank for every record (no dual route), so the 27-feature arm is
the matched reference. Arms: ref27, +c7 (30), +c8 (54), +c7+c8 (57), c7_only curve,
equal57. ProcessBench uses the saved entropy-q0.3 fold gates. No labels enter any fit.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
GATE = ROOT / "results" / "fusion_fixed_gate_v1"
FOLDS = ROOT / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
OUT = ROOT / "results" / "fusion_onset_innovation_iu_v1"
sys.path.insert(0, str(ROOT / "local_cache" / "short_cycle01_code"))
import spectral_utils  # noqa: E402
spectral_utils.__path__.append(str(ROOT / "spectral_utils"))

TOPK, WIDTH = 10, 8
EDIS_TAU_B, EDIS_TAU_R, RIDGE = 1.36, 1.33, 1.0
ARMS = ("ref27", "c7_30", "c8_54", "c7c8_57", "c7_only", "equal57")
SEED, DRAWS = 2026090707, 1000


def edis_onset(z):
    z = np.asarray(z, float)
    burst = np.zeros_like(z)
    if len(z) > 1:
        burst[1:] = np.maximum(np.diff(z) - EDIS_TAU_B, 0.0)
    rebound_excess = np.maximum(z - np.minimum.accumulate(z) - EDIS_TAU_R, 0.0)
    rebound_onset = np.maximum(np.diff(np.concatenate(([0.0], rebound_excess))), 0.0)
    return np.maximum(burst, rebound_onset)


def self_innovation(x):
    """|residual| of a ridge AR(1) with log-position trend, fitted on this answer's tokens."""
    x = np.asarray(x, float); n = len(x)
    if n < 3:
        return np.zeros(n)
    pos = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n - 1), np.log1p(pos[1:]), x[:-1]])
    y = x[1:]
    pen = np.diag([0.0, RIDGE, RIDGE])
    beta = np.linalg.solve(X.T @ X + pen, X.T @ y)
    res = y - X @ beta
    scale = float(np.sqrt(np.mean(res ** 2)))
    out = np.zeros(n)
    if np.isfinite(scale) and scale > 1e-8:
        out[1:] = np.abs(res) / scale
    return out


def moments(series, plan):
    """level / sd / slope per window for a list of token series (mirrors moment_matrix)."""
    S = np.column_stack(series)
    coords = np.linspace(-.5, .5, plan.width); den = coords @ coords
    vals = []
    for lo, hi in zip(plan.starts, plan.ends):
        chunk = S[lo:hi]; mean, sd = chunk.mean(0), chunk.std(0)
        slope = coords @ (chunk - mean) / den
        vals.append(np.column_stack((mean, sd, slope)).ravel())
    return np.asarray(vals)


def step_topk(tok, ss, se, k=TOPK):
    return np.asarray([np.sort(tok[a:b])[::-1][:min(k, b - a)].mean() for a, b in zip(ss, se)])


def score_record(args):
    uid, cell, row_index, tokens = args
    from spectral_utils.answer_localization_v2 import moment_plan, moment_matrix, prepare_local, MIN_WINDOWS, PRIMITIVES, STREAM_NAMES
    from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
    from spectral_utils.short_cycle_localization import scaled_oriented_weight
    from spectral_utils.upcr import upcr_fit
    from spectral_utils.window_localization import windows_to_tokens

    with np.load(BENCH / "scores" / f"{uid}.npz") as z:
        starts, ends = z["step_starts"], z["step_ends"]
    d = BENCH / "inputs" / cell
    raw = np.load(d / "raw.npy", mmap_mode="r"); offsets = np.load(d / "token_offsets.npy")
    block = np.asarray(raw[offsets[row_index]:offsets[row_index + 1]], float)
    out = {"uid": uid, "cell": cell, "arms": {}, "checks": {}}
    plan = moment_plan(tokens, WIDTH)
    if len(plan.fit_indices) < MIN_WINDOWS:
        out["checks"]["error"] = "TOO_FEW_FIT_WINDOWS"; return out, {}
    prim = [block[:, STREAM_NAMES.index(s)] for s in PRIMITIVES]
    base_vals = moments(prim, plan); base_names = [f"{s}__{op}" for s in PRIMITIVES for op in ("level", "sd", "slope")]
    ref_vals, ref_names = moment_matrix(block, plan)
    out["checks"]["bank_replay"] = float(np.max(np.abs(base_vals - ref_vals))); assert ref_names == base_names
    ent = block[:, STREAM_NAMES.index("entropy_series")]
    zent = (ent - ent.mean()) / (ent.std() + 1e-12)
    c7 = edis_onset(zent)
    c8 = [self_innovation(x) for x in prim]
    banks = {
        "ref27": (base_vals, base_names),
        "c7_30": (np.column_stack([base_vals, moments([c7], plan)]), base_names + [f"c7_onset__{op}" for op in ("level", "sd", "slope")]),
        "c8_54": (np.column_stack([base_vals, moments(c8, plan)]), base_names + [f"{s}_innov__{op}" for s in PRIMITIVES for op in ("level", "sd", "slope")]),
    }
    banks["c7c8_57"] = (np.column_stack([banks["c8_54"][0], moments([c7], plan)]), banks["c8_54"][1] + [f"c7_onset__{op}" for op in ("level", "sd", "slope")])
    arrays = {"step_starts": starts, "step_ends": ends}
    for arm in ("ref27", "c7_30", "c8_54", "c7c8_57", "equal57"):
        vals, names = banks["c7c8_57"] if arm == "equal57" else banks[arm]
        try:
            zz, anchor, summary = prepare_local(vals, names, plan.fit_indices)
            fit = zz[plan.fit_indices]
            if arm == "equal57":
                w0 = np.ones(fit.shape[1]) / fit.shape[1]
            else:
                iu = upcr_fit(fit.T, **dict(IU_FIT_DEFAULTS))
                if iu.abstained: raise ValueError("ABSTAINED")
                w0 = iu.w
            w, _ = scaled_oriented_weight(w0, fit, anchor)
            tok = np.asarray(windows_to_tokens(plan, -(zz @ w)), float)
            if not np.isfinite(tok).all(): raise ValueError("NONFINITE")
            arrays[arm + "__risk"] = step_topk(tok, starts, ends)
            out["arms"][arm] = {"ok": True, "active_p": int(summary["active_p"])}
        except Exception as exc:  # noqa: BLE001
            out["arms"][arm] = {"ok": False, "error": str(exc)[:80]}
    # c7 curve alone (token-level onset, top-10 per step)
    arrays["c7_only__risk"] = step_topk(c7, starts, ends); out["arms"]["c7_only"] = {"ok": True}
    return out, arrays


def phase_score(workers):
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8")); recs = j["records"]
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    row_in_cell = {}
    for cell in sorted(set(r["cell"] for r in recs)):
        ids = np.load(BENCH / "inputs" / cell / "row_ids.npy", allow_pickle=True); row_in_cell[cell] = {str(x): i for i, x in enumerate(ids)}
    todo = [(r["uid"], r["cell"], row_in_cell[r["cell"]][r["row_id"]], r["tokens"]) for r in recs if not (OUT / "scores" / f"{r['uid']}.npz").exists()]
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
            (OUT / "scores" / f"{uid}.json").write_text(json.dumps(out), encoding="utf-8")
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
    recs = j["records"]; n = len(recs); offsets, labels, target = z["offsets"], z["labels"], z["target"]
    cells = np.array([r["cell"] for r in recs]); groups = np.array([r["group_id"] for r in recs]); pb = np.array([c.startswith("pb_") for c in cells]); prm = ~pb
    folds = json.loads(FOLDS.read_text(encoding="utf-8")); gate = json.load(open(GATE / "METRICS.json"))
    thr = {int(k): v for k, v in gate["arms"]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"].items()}
    det = np.load(GATE / "DETECTORS.npz")["entropy_mean"]; fthr = np.array([thr.get(int(folds["outer"].get(r["group_id"], -1)), np.nan) for r in recs])
    S = {a: np.full(len(labels), np.nan) for a in ARMS}; valid = {a: np.zeros(n, bool) for a in ARMS}; replay = []; active = {a: [] for a in ARMS}
    for i, r in enumerate(recs):
        p = OUT / "scores" / f"{r['uid']}.npz"
        if not p.exists(): continue
        meta = json.loads((OUT / "scores" / f"{r['uid']}.json").read_text(encoding="utf-8"))
        if "bank_replay" in meta["checks"]: replay.append(meta["checks"]["bank_replay"])
        with np.load(p) as a:
            for m in ARMS:
                if m + "__risk" in a.files:
                    S[m][offsets[i]:offsets[i + 1]] = a[m + "__risk"]; valid[m][i] = True
                    if "active_p" in meta["arms"].get(m, {}): active[m].append(meta["arms"][m]["active_p"])
    print(f"bank replay max abs diff {max(replay):.2e}; active features median: " + ", ".join(f"{m}={np.median(active[m]):.0f}" for m in ARMS if active[m]))
    results, per = {}, {}; T, C = target[pb], cells[pb]
    for m in ARMS:
        v = valid[m]; lab = np.zeros(len(labels), bool)
        for i in np.flatnonzero(v & prm): lab[offsets[i]:offsets[i + 1]] = True
        lab &= labels >= 0; pooled = auc(labels[lab] == 1, S[m][lab]) if lab.any() else np.nan
        wa = np.full(n, np.nan)
        for i in np.flatnonzero(v & prm):
            y = labels[offsets[i]:offsets[i + 1]]; s = S[m][offsets[i]:offsets[i + 1]]; ok = y >= 0
            if ok.sum() and (y[ok] == 1).any() and (y[ok] == 0).any(): wa[i] = auc(y[ok] == 1, s[ok])
        pk = np.full(n, -1)
        for i in np.flatnonzero(v & pb):
            s = S[m][offsets[i]:offsets[i + 1]]; pk[i] = int(np.argmax(s)) if np.isfinite(s).all() else -1
        pv = v & pb & (pk >= 0) & np.isfinite(det) & np.isfinite(fthr); pred = np.where(det >= fthr, pk, -1); pred[~pv] = -1
        met = pb_metrics(T, pred[pb], pv[pb], C); err = pb & (target >= 0) & pv
        results[m] = dict(prm_valid=int((v & prm).sum()), prm_pooled=pooled, prm_within=float(np.nanmean(wa)), pb_valid=int(pv.sum()), pb_all8=met["macros"]["all"], pb_q8=met["macros"]["q8"],
                          raw_exact=float(np.mean(pk[err] == target[err])), within_one=float(np.mean(np.abs(pk[err] - target[err]) <= 1)), cells={k: x["f1"] for k, x in met["cells"].items()})
        per[m] = dict(within=wa, pred=pred, pv=pv); r_ = results[m]
        print(f"{m:10s} PRMB pooled {pooled:.5f} within {r_['prm_within']:.5f} (n={int(np.isfinite(wa).sum())}) | PB all8 {r_['pb_all8']*100:6.2f} Q8 {r_['pb_q8']*100:6.2f} exact {r_['raw_exact']*100:5.1f} within-1 {r_['within_one']*100:5.1f} | valid {r_['prm_valid']}/{r_['pb_valid']}")
    rng = np.random.default_rng(SEED); uniq, inv = np.unique(groups, return_inverse=True)
    draws = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float) for _ in range(DRAWS)]
    contrasts = {}
    for a, b in [(m, "ref27") for m in ARMS if m != "ref27"] + [("c7c8_57", "equal57")]:
        A, B = per[a], per[b]; common = np.isfinite(A["within"]) & np.isfinite(B["within"]); dw, dpb = [], []
        for w in draws:
            dw.append(np.average(A["within"][common] - B["within"][common], weights=w[common]))
            pa = pb_metrics(T, A["pred"][pb], A["pv"][pb], C, weights=w[pb])["macros"]["all"]; pb_ = pb_metrics(T, B["pred"][pb], B["pv"][pb], C, weights=w[pb])["macros"]["all"]
            if pa is not None and pb_ is not None: dpb.append(pa - pb_)
        contrasts[f"{a} minus {b}"] = dict(prm_within_point=float(np.mean(A["within"][common] - B["within"][common])), prm_within_ci=[float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5))],
                                          pb_all8_point=float(results[a]["pb_all8"] - results[b]["pb_all8"]), pb_all8_ci=[float(np.percentile(dpb, 2.5)), float(np.percentile(dpb, 97.5))])
        cc = contrasts[f"{a} minus {b}"]
        print(f"{a:10s} minus {b}: within {cc['prm_within_point']:+.4f} [{cc['prm_within_ci'][0]:+.4f},{cc['prm_within_ci'][1]:+.4f}]  PB {cc['pb_all8_point']*100:+.2f}pp [{cc['pb_all8_ci'][0]*100:+.2f},{cc['pb_all8_ci'][1]*100:+.2f}]")
    json.dump(dict(results=results, contrasts=contrasts, bank_replay_max=float(max(replay))), open(OUT / "METRICS.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="score", choices=["score", "evaluate", "smoke"]); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.phase == "smoke":
        j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8")); recs = j["records"]
        for r in (recs[0], recs[9000], recs[12000]):
            ids = np.load(BENCH / "inputs" / r["cell"] / "row_ids.npy", allow_pickle=True); k = {str(x): i for i, x in enumerate(ids)}[r["row_id"]]
            t = time.time(); out, arrays = score_record((r["uid"], r["cell"], k, r["tokens"]))
            print(r["cell"], r["tokens"], out["checks"], {m: v.get("active_p", v) for m, v in out["arms"].items()}, f"{time.time()-t:.2f}s")
    elif a.phase == "score": phase_score(a.workers)
    else: phase_evaluate()
