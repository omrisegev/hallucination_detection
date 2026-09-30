"""Multi-width window fusion for answer-only IU localization (Omri's idea, 2026-09-09).

Run the unchanged answer-only IU localizer at window widths 8, 16 and 32 on every record
(same bank route as the frozen dual__iu, same standardization/orientation/IU defaults),
spread each width's window risks to tokens with the frozen rule, z-score each token curve
within the answer, and combine the curves across widths by MEAN and by MIN (agreement).
Read out every curve with the adopted top-10 token-mean rule per step, gate ProcessBench
with the saved entropy-q0.3 fold thresholds, and compare with each single width alone.

Width 8 with the max-token readout must replay the frozen dual__iu step scores exactly
(implementation gate). Widths 16/32 need >= 8 fitting windows (128 / 256 tokens); the
combinations use whichever widths are available for the record and report coverage.
No labels enter any fit; development evidence.

--phase score   : per-record npz with z-scored token curves per width (3 workers).
--phase evaluate: metrics, paired source-group bootstrap, METRICS.json.
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
OUT = ROOT / "results" / "fusion_multiwidth_iu_v1"
sys.path.insert(0, str(ROOT / "local_cache" / "short_cycle01_code"))
import spectral_utils  # noqa: E402
spectral_utils.__path__.append(str(ROOT / "spectral_utils"))

WIDTHS = (8, 16, 32)          # overridden by --widths
TOPK = 10
SEED, DRAWS = 2026090707, 1000


def step_topk(tok, starts, ends, k=TOPK):
    out = []
    for a, b in zip(starts, ends):
        t = np.sort(tok[a:b])[::-1]
        out.append(t[:min(k, len(t))].mean())
    return np.asarray(out, float)


def score_record(args):
    uid, cell, row_index, tokens, widths = args          # widths passed explicitly: workers are spawned processes
    from spectral_utils.answer_localization_v2 import moment_plan, moment_matrix, prepare_local, MIN_WINDOWS
    from spectral_utils.fusion_context_bank import context_matrix
    from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
    from spectral_utils.short_cycle_localization import scaled_oriented_weight
    from spectral_utils.upcr import upcr_fit
    from spectral_utils.window_localization import windows_to_tokens

    meta = json.loads((BENCH / "scores" / f"{uid}.json").read_text(encoding="utf-8"))
    with np.load(BENCH / "scores" / f"{uid}.npz") as z:
        frozen = {k: z[k] for k in z.files}
    d = BENCH / "inputs" / cell
    raw = np.load(d / "raw.npy", mmap_mode="r")
    offsets = np.load(d / "token_offsets.npy")
    block = np.asarray(raw[offsets[row_index]:offsets[row_index + 1]], float)
    starts, ends = frozen["step_starts"], frozen["step_ends"]
    out = {"uid": uid, "cell": cell, "widths": {}, "replay": None}
    if not meta["methods"]["dual__iu"].get("valid", False):
        out["replay"] = "FROZEN_IU_INVALID"
        return out, {}
    bank = "context" if meta["methods"]["dual__iu"].get("route") == "context_joint" else "moment"
    out["bank"] = bank
    arrays = {"step_starts": starts, "step_ends": ends}
    for width in widths:
        try:
            plan = moment_plan(tokens, width)
            if len(plan.fit_indices) < MIN_WINDOWS:
                raise ValueError("TOO_FEW_FIT_WINDOWS")
            values, names = (moment_matrix if bank == "moment" else context_matrix)(block, plan)
            z, anchor, _ = prepare_local(values, names, plan.fit_indices)
            fit = z[plan.fit_indices]
            iu = upcr_fit(fit.T, **dict(IU_FIT_DEFAULTS))
            if iu.abstained:
                raise ValueError("ABSTAINED")
            w, _ = scaled_oriented_weight(iu.w, fit, anchor)
            risk = -(z @ w)
            tok = np.asarray(windows_to_tokens(plan, risk), float)
            if not np.isfinite(tok).all():
                raise ValueError("NONFINITE_TOKENS")
            if width == 8:
                steps_max = np.asarray([tok[a:b].max() for a, b in zip(starts, ends)])
                out["replay"] = float(np.max(np.abs(steps_max - frozen["dual__iu__risk"])))
                arrays["w8_raw_tok"] = tok.astype(np.float32)
            sd = tok.std()
            if sd < 1e-8:
                raise ValueError("DEGENERATE_CURVE")
            arrays[f"w{width}_z_tok"] = ((tok - tok.mean()) / sd).astype(np.float32)
            out["widths"][str(width)] = {"ok": True, "n_fit_windows": int(len(plan.fit_indices))}
        except Exception as exc:  # noqa: BLE001
            out["widths"][str(width)] = {"ok": False, "error": str(exc)[:80]}
    return out, arrays


def phase_score(workers, widths=WIDTHS, out=OUT):
    global OUT; OUT = out
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    recs = j["records"]
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    row_in_cell = {}
    for cell in sorted(set(r["cell"] for r in recs)):
        ids = np.load(BENCH / "inputs" / cell / "row_ids.npy", allow_pickle=True)
        row_in_cell[cell] = {str(x): i for i, x in enumerate(ids)}
    todo = [(r["uid"], r["cell"], row_in_cell[r["cell"]][r["row_id"]], r["tokens"], tuple(widths)) for r in recs
            if not (OUT / "scores" / f"{r['uid']}.npz").exists()]
    print(f"{len(recs)} records, {len(todo)} to score, {workers} workers", flush=True)
    t0 = time.time(); done = 0; failures = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_record, a): a for a in todo}
        for f in as_completed(futs):
            uid = futs[f][0]
            try:
                out, arrays = f.result()
            except Exception:  # noqa: BLE001
                failures += 1
                (OUT / "scores" / f"{uid}.error").write_text(traceback.format_exc(), encoding="utf-8")
                continue
            tmp = OUT / "scores" / f"{uid}.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            tmp.replace(OUT / "scores" / f"{uid}.npz")
            (OUT / "scores" / f"{uid}.json").write_text(json.dumps(out), encoding="utf-8")
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(todo)} {time.time() - t0:.0f}s failures={failures}", flush=True)
    print("scoring complete", round(time.time() - t0), "s, failures", failures, flush=True)


def phase_evaluate():
    from scipy.stats import rankdata
    from spectral_utils.historical_fusion_evaluation import pb_metrics

    def auc(y, s):
        y = np.asarray(y, bool); p, n = y.sum(), (~y).sum()
        return float((rankdata(s)[y].sum() - p * (p + 1) / 2) / (p * n)) if p and n else np.nan

    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    z = np.load(BENCH / "evaluation" / "JOINED.npz")
    recs = j["records"]; n = len(recs)
    offsets, labels, target = z["offsets"], z["labels"], z["target"]
    cells = np.array([r["cell"] for r in recs]); groups = np.array([r["group_id"] for r in recs])
    pb = np.array([c.startswith("pb_") for c in cells]); prm = ~pb
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    gate = json.load(open(GATE / "METRICS.json"))
    thr = {int(k): v for k, v in gate["arms"]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"].items()}
    det = np.load(GATE / "DETECTORS.npz")["entropy_mean"]
    fthr = np.array([thr.get(int(folds["outer"].get(r["group_id"], -1)), np.nan) for r in recs])

    arms = ["w8_max_bridge", "w8_top10_raw"] + [f"w{w}_z" for w in WIDTHS] + ["mean_all", "min_all", "median_all"]
    S = {a: np.full(len(labels), np.nan) for a in arms}; valid = {a: np.zeros(n, bool) for a in arms}
    replay = []; avail = {w: 0 for w in WIDTHS}; missing = 0
    for i, r in enumerate(recs):
        p = OUT / "scores" / f"{r['uid']}.npz"
        if not p.exists():
            missing += 1; continue
        meta = json.loads((OUT / "scores" / f"{r['uid']}.json").read_text(encoding="utf-8"))
        if isinstance(meta.get("replay"), (int, float)):
            replay.append(meta["replay"])
        with np.load(p) as a:
            if "w8_raw_tok" not in a.files:
                continue
            ss, se = a["step_starts"], a["step_ends"]
            curves = {w: a[f"w{w}_z_tok"].astype(float) for w in WIDTHS if f"w{w}_z_tok" in a.files}
            for w in curves: avail[w] += 1
            raw8 = a["w8_raw_tok"].astype(float)
            sl = slice(offsets[i], offsets[i + 1])
            def put(name, steps):
                S[name][sl] = steps; valid[name][i] = True
            put("w8_max_bridge", np.asarray([raw8[x:y].max() for x, y in zip(ss, se)]))
            put("w8_top10_raw", step_topk(raw8, ss, se))
            for w in WIDTHS:
                if w in curves: put(f"w{w}_z", step_topk(curves[w], ss, se))
            if 8 in curves:
                stack = np.vstack([curves[w] for w in WIDTHS if w in curves])
                put("mean_all", step_topk(stack.mean(0), ss, se)); put("min_all", step_topk(stack.min(0), ss, se))
                put("median_all", step_topk(np.median(stack, 0), ss, se))
    print(f"loaded {n - missing}/{n}; replay max {max(replay):.2e}; widths available {avail}")
    # frozen check: w8_max_bridge must equal frozen dual__iu scores
    c = j["arms"].index("dual__iu"); m = valid["w8_max_bridge"]
    ref = z["scores"][:, c]
    diff = np.nanmax(np.abs(S["w8_max_bridge"][np.repeat(m, np.diff(offsets))] - ref[np.repeat(m, np.diff(offsets))]))
    print("bridge vs frozen dual__iu step scores, max abs diff:", diff)

    results, per = {}, {}
    T, C = target[pb], cells[pb]
    for a in arms:
        v = valid[a]
        lab = np.zeros(len(labels), bool)
        for i in np.flatnonzero(v & prm): lab[offsets[i]:offsets[i + 1]] = True
        lab &= labels >= 0
        pooled = auc(labels[lab] == 1, S[a][lab]) if lab.any() else np.nan
        wa = np.full(n, np.nan)
        for i in np.flatnonzero(v & prm):
            y = labels[offsets[i]:offsets[i + 1]]; s = S[a][offsets[i]:offsets[i + 1]]; ok = y >= 0
            if ok.sum() and (y[ok] == 1).any() and (y[ok] == 0).any(): wa[i] = auc(y[ok] == 1, s[ok])
        pk = np.full(n, -1)
        for i in np.flatnonzero(v & pb):
            s = S[a][offsets[i]:offsets[i + 1]]; pk[i] = int(np.argmax(s)) if np.isfinite(s).all() else -1
        pv = v & pb & (pk >= 0) & np.isfinite(det) & np.isfinite(fthr)
        pred = np.where(det >= fthr, pk, -1); pred[~pv] = -1
        met = pb_metrics(T, pred[pb], pv[pb], C)
        err = pb & (target >= 0) & pv
        results[a] = dict(prm_valid=int((v & prm).sum()), prm_pooled=pooled, prm_within=float(np.nanmean(wa)),
                          pb_valid=int(pv.sum()), pb_all8=met["macros"]["all"], pb_q8=met["macros"]["q8"],
                          raw_exact=float(np.mean(pk[err] == target[err])), within_one=float(np.mean(np.abs(pk[err] - target[err]) <= 1)),
                          cells={k: x["f1"] for k, x in met["cells"].items()})
        per[a] = dict(within=wa, pred=pred, pv=pv)
        r_ = results[a]
        print(f"{a:14s} PRMB pooled {pooled:.5f} within {r_['prm_within']:.5f} (n={int(np.isfinite(wa).sum())}) | PB all8 {r_['pb_all8']*100:6.2f} Q8 {r_['pb_q8']*100:6.2f} exact {r_['raw_exact']*100:5.1f} within-1 {r_['within_one']*100:5.1f} | valid {r_['prm_valid']}/{r_['pb_valid']}")

    rng = np.random.default_rng(SEED); uniq, inv = np.unique(groups, return_inverse=True)
    draws = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float) for _ in range(DRAWS)]
    contrasts = {}
    for a in [f"w{w}_z" for w in WIDTHS] + ["mean_all", "min_all", "median_all"]:
        b = "w8_top10_raw"
        A, B = per[a], per[b]
        common = np.isfinite(A["within"]) & np.isfinite(B["within"]); pvc = A["pv"] & B["pv"]
        dw, dpb = [], []
        for w in draws:
            dw.append(np.average(A["within"][common] - B["within"][common], weights=w[common]))
            pa = pb_metrics(T, A["pred"][pb], A["pv"][pb], C, weights=w[pb])["macros"]["all"]
            pb_ = pb_metrics(T, B["pred"][pb], B["pv"][pb], C, weights=w[pb])["macros"]["all"]
            if pa is not None and pb_ is not None: dpb.append(pa - pb_)
        contrasts[f"{a} minus {b}"] = dict(
            prm_within_point=float(np.mean(A["within"][common] - B["within"][common])), prm_common_n=int(common.sum()),
            prm_within_ci=[float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5))],
            pb_all8_point=float(results[a]["pb_all8"] - results[b]["pb_all8"]), pb_common_valid=int(pvc.sum()),
            pb_all8_ci=[float(np.percentile(dpb, 2.5)), float(np.percentile(dpb, 97.5))])
        cc = contrasts[f"{a} minus {b}"]
        print(f"{a:10s} minus w8_top10: within {cc['prm_within_point']:+.4f} [{cc['prm_within_ci'][0]:+.4f},{cc['prm_within_ci'][1]:+.4f}]  PB {cc['pb_all8_point']*100:+.2f}pp [{cc['pb_all8_ci'][0]*100:+.2f},{cc['pb_all8_ci'][1]*100:+.2f}]")
    json.dump(dict(results=results, contrasts=contrasts, widths_available=avail, replay_max=float(max(replay)),
                   bridge_max_abs_diff=float(diff), missing=missing), open(OUT / "METRICS.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="score", choices=["score", "evaluate", "smoke"]); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--widths", default="8,16,32"); ap.add_argument("--out", default="fusion_multiwidth_iu_v1")
    a = ap.parse_args()
    WIDTHS = tuple(int(x) for x in a.widths.split(",")); OUT = ROOT / "results" / a.out
    assert 8 in WIDTHS, "width 8 is the frozen bridge and must be included"
    if a.phase == "smoke":
        j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8")); recs = j["records"]
        for r in (recs[0], recs[9000], recs[12000]):
            ids = np.load(BENCH / "inputs" / r["cell"] / "row_ids.npy", allow_pickle=True); k = {str(x): i for i, x in enumerate(ids)}[r["row_id"]]
            t = time.time(); out, arrays = score_record((r["uid"], r["cell"], k, r["tokens"], WIDTHS))
            print(r["cell"], r["tokens"], out.get("bank"), "replay", out["replay"], out["widths"], list(arrays), f"{time.time()-t:.2f}s")
    elif a.phase == "score":
        phase_score(a.workers, WIDTHS, OUT)
    else:
        phase_evaluate()
