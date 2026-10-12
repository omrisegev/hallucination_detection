"""Answer-only IU-PCR with structured covariance shrinkage, all 13,769 rows.

Reuses the frozen localization_full_benchmark_v3 pipeline exactly: same raw
telemetry, 8-token window plan, moment/context bank chosen by the frozen
dual route of each record, same standardization/orientation, same IU-PCR
defaults, same window->token->step max readout.  For every record the
original IU is re-fit and must replay the frozen ``dual__iu__risk`` step
scores (exactness gate); the shrinkage variants then differ from it only in
the covariance handed to the IU solve (see spectral_utils/shrinkage_iu.py).

Phase 1 (this script, --phase score): per-record step risks for all variants,
checkpointed as one npz per record under results/fusion_shrinkage_iu_v1/scores.
Phase 2 (--phase evaluate): PRMBench pooled / within-answer AUC, ProcessBench
F1 with the saved entropy-q0.3 fold gates (fusion_fixed_gate_v1) and with the
native per-answer GMM gate, paired source-group bootstraps, REPORT.md.
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
OUT = ROOT / "results" / "fusion_shrinkage_iu_v1"
# Frozen short-cycle code first (joint_lsml, upcr, laplacian_upcr, ...), repo package appended.
sys.path.insert(0, str(ROOT / "local_cache" / "short_cycle01_code"))
import spectral_utils  # noqa: E402
spectral_utils.__path__.append(str(ROOT / "spectral_utils"))

VARIANTS = [("solve", t, a) for t in ("joint", "block", "diag") for a in ("lw", 0.5, 1.0)] + \
           [("subspace", t, a) for t in ("joint", "block", "diag") for a in ("lw", 1.0)] + \
           [("full", t, a) for t in ("joint", "block", "diag") for a in ("lw", 0.5, 1.0)] + \
           [("full", "joint_transform", a) for a in ("lw", 1.0)]
PRIMARY = [("full", "joint", "lw"), ("solve", "joint", "lw")]


def vname(v):
    return f"{v[0]}__{v[1]}__a{v[2]}"


def score_record(args):
    uid, cell, row_index, tokens = args
    from spectral_utils.answer_localization_v2 import moment_plan, moment_matrix, prepare_local, mixture_readout
    from spectral_utils.fusion_context_bank import context_matrix
    from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
    from spectral_utils.short_cycle_localization import scaled_oriented_weight
    from spectral_utils.upcr import upcr_fit, upcr_fit_covariance
    from spectral_utils.window_localization import windows_to_tokens
    from spectral_utils.shrinkage_iu import shrunk_iu_weights

    meta = json.loads((BENCH / "scores" / f"{uid}.json").read_text(encoding="utf-8"))
    with np.load(BENCH / "scores" / f"{uid}.npz") as z:
        frozen = {k: z[k] for k in z.files}
    d = BENCH / "inputs" / cell
    raw = np.load(d / "raw.npy", mmap_mode="r")
    offsets = np.load(d / "token_offsets.npy")
    block = np.asarray(raw[offsets[row_index]:offsets[row_index + 1]], float)
    assert len(block) == tokens
    starts, ends = frozen["step_starts"], frozen["step_ends"]
    out = {"uid": uid, "cell": cell, "valid": {}, "info": {}, "replay": None}
    route = meta["methods"]["dual__iu"].get("route")
    bank = "context" if route == "context_joint" else "moment"
    out["bank"] = bank
    if not meta["methods"]["dual__iu"].get("valid", False):
        out["replay"] = "FROZEN_IU_INVALID"
        return out, {}
    plan = moment_plan(tokens, 8)
    values, names = (moment_matrix if bank == "moment" else context_matrix)(block, plan)
    z, anchor, summary = prepare_local(values, names, plan.fit_indices)
    fit = z[plan.fit_indices]
    active = summary["active_features"]
    iu = upcr_fit(fit.T, **dict(IU_FIT_DEFAULTS))
    arrays = {}

    def admit(name, w):
        w2, _ = scaled_oriented_weight(w, fit, anchor)
        risk = -(z @ w2)
        if not np.isfinite(risk).all():
            raise ValueError("NONFINITE_RISK")
        token = windows_to_tokens(plan, risk)
        steps = np.asarray([token[a:b].max() for a, b in zip(starts, ends)], float)
        if not np.isfinite(steps).all():
            raise ValueError("NONFINITE_STEP_SCORE")
        arrays[name + "__risk"] = steps
        try:
            g = mixture_readout(risk[plan.fit_indices], steps)
            gmm = int(np.argmax(steps)) if g["prediction"] != -1 else -1
        except Exception:  # noqa: BLE001
            gmm = -2  # invalid native decision
        arrays[name + "__gmm"] = np.asarray([gmm], int)
        out["valid"][name] = True

    admit("iu_replay", iu.w)
    ref = frozen["dual__iu__risk"]
    out["replay"] = float(np.max(np.abs(arrays["iu_replay__risk"] - ref)))
    # exactness gate for the covariance seam: full IU on C with alpha=0 must equal IU
    C = fit.T @ fit / len(fit)
    res0 = upcr_fit_covariance(C, **dict(IU_FIT_DEFAULTS))
    out["seam_max_abs_dw"] = float(np.max(np.abs(res0.w - iu.w)))
    for v in VARIANTS:
        name = vname(v)
        level, kind, alpha = v
        by = "transform" if kind.endswith("_transform") else "stream"
        kind = kind.replace("_transform", "")
        try:
            w, info = shrunk_iu_weights(fit, active, iu, level, kind, alpha, by=by,
                                        upcr_fit_covariance=upcr_fit_covariance, iu_kwargs=dict(IU_FIT_DEFAULTS))
            if info.get("abstained"):
                raise ValueError("ABSTAINED")
            admit(name, w)
            out["info"][name] = {"alpha": info["alpha"], "groups": info["groups"]}
        except Exception as exc:  # noqa: BLE001
            out["valid"][name] = False
            out["info"][name] = {"error": str(exc)[:120]}
    return out, arrays


def phase_score(workers):
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    recs = j["records"]
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    row_in_cell = {}
    for cell in sorted(set(r["cell"] for r in recs)):
        ids = np.load(BENCH / "inputs" / cell / "row_ids.npy", allow_pickle=True)
        row_in_cell[cell] = {str(x): i for i, x in enumerate(ids)}
    todo = [(r["uid"], r["cell"], row_in_cell[r["cell"]][r["row_id"]], r["tokens"]) for r in recs
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
            if done % 200 == 0:
                (OUT / "RUN_STATE.json").write_text(json.dumps(dict(
                    phase="SCORING", pid=os.getpid(), completed=done + len(recs) - len(todo), total=len(recs),
                    failures=failures, seconds=time.time() - t0, updated_unix=time.time())), encoding="utf-8")
                print(f"{done}/{len(todo)} {time.time() - t0:.0f}s failures={failures}", flush=True)
    (OUT / "RUN_STATE.json").write_text(json.dumps(dict(phase="SCORING_COMPLETE", completed=len(recs), total=len(recs),
                                                      failures=failures, seconds=time.time() - t0)), encoding="utf-8")
    print("scoring complete", time.time() - t0, "s, failures", failures)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="score", choices=["score", "smoke"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from spectral_utils.shrinkage_iu import self_test
    assert self_test()
    if a.phase == "smoke":
        j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
        recs = j["records"]
        for r in (recs[0], recs[500], recs[9000], recs[12000]):
            ids = np.load(BENCH / "inputs" / r["cell"] / "row_ids.npy", allow_pickle=True)
            k = {str(x): i for i, x in enumerate(ids)}[r["row_id"]]
            t = time.time(); out, arrays = score_record((r["uid"], r["cell"], k, r["tokens"]))
            print(r["cell"], r["tokens"], "bank", out["bank"], "replay", out["replay"], "seam", out.get("seam_max_abs_dw"),
                  "valid", sum(out["valid"].values()), "/", len(out["valid"]),
                  {k2: v["alpha"] for k2, v in out["info"].items() if "alpha" in v and k2.startswith("full__")},
                  f"{time.time() - t:.2f}s")
    else:
        phase_score(a.workers)
