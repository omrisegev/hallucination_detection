#!/usr/bin/env python
"""Item 4, stage 2 (2026-09-23): answer-local fusion on the 8-token window bank, run ONLY when the
measurement gate of stage 1 passed (`WINDOW_PR.json: gate_passed`), or with --force on a dry run.

Per answer, on its own non-overlapping fit windows (never on the dense stride-1 rows): equal,
IU-PCR (`upcr.upcr_fit`, IU_FIT_DEFAULTS, abstentions flagged), shrinkage-IU (joint target,
Ledoit-Wolf alpha corrected by the effective sample size), continuous L-SML (residual K, guard on).
Weights are oriented to correlate positively with the equal-weight score (label-free), applied to
the dense windows, mapped to tokens, read to official steps by Top10 (primary) and overlap mean,
answer-standardized, argmax, frozen CT7 gate. Answers with fewer than 3p fit windows fall back to
equal and are counted; `*_native` rows are valid only where the fit was native.

    python -B scripts/experiments/window_answer_local_fusion_v1.py --config configs/window_representation_b3_v1.json
    python -B scripts/experiments/window_answer_local_fusion_v1.py --dry-run --force
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct7_levers_common as L  # noqa: E402

L.ensure_spectral_package()
from spectral_utils.ct7_token_streams import masked_step_top10  # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402
from spectral_utils.joint_lsml import continuous_lsml_weight_vector  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.shrinkage_iu import ledoit_wolf_alpha, partition, shrink, target_matrix  # noqa: E402
from spectral_utils.upcr import upcr_fit, upcr_fit_covariance  # noqa: E402
from spectral_utils.window_localization import tokens_to_official_steps, windows_to_tokens  # noqa: E402
from spectral_utils.window_moment_bank import alpha_neff, distance_from_equal  # noqa: E402

sys.path.insert(0, str(L.ROOT / "scripts" / "diagnostics"))
import window_pr_measurement_v1 as M  # noqa: E402

SCHEMA = "window-answer-local-fusion-v1"
MIN_FIT_MULTIPLE = 3
ARMS = ("equal", "iu", "shrink_iu", "lsml")


def weight_ipr(w):
    w = np.abs(np.asarray(w, float)); return float(1.0 / np.sum((w / w.sum()) ** 2)) if w.sum() > 0 else float("nan")


def fit_weights(Z, names, arm):
    """Weights for one arm on standardized fit windows Z [n x p]; returns (w, info)."""
    n, p = Z.shape; kw = dict(IU_FIT_DEFAULTS)
    if arm == "equal":
        return np.full(p, 1.0 / p), {}
    if arm == "iu":
        res = upcr_fit(Z.T, on_abstain="flag", **kw)
        return np.asarray(res.w, float), {"abstained": bool(res.abstained)}
    if arm == "shrink_iu":
        C = Z.T @ Z / n; groups, _ = partition(names, "stream"); T = target_matrix(C, groups, "joint")
        a_lw = ledoit_wolf_alpha(Z, C, T)
        n_eff = min(M.effective_n(Z[:, j]) for j in range(p))
        a = alpha_neff(a_lw, n, n_eff)
        res = upcr_fit_covariance(shrink(C, T, a), on_abstain="flag", **kw)
        return np.asarray(res.w, float), {"alpha_lw": a_lw, "alpha": a, "n_eff": n_eff, "abstained": bool(res.abstained)}
    if arm == "lsml":
        _, meta = lsml_continuous(*[Z[:, j] for j in range(p)], groups=None, small_m_guard=True, compute_score_matrix=False)
        return continuous_lsml_weight_vector(meta, p), {"K": int(meta["K"]), "groups": np.asarray(meta["c"]).tolist()}
    raise ValueError(arm)


def run(d, mats, sp, views):
    p = len(views); min_fit = MIN_FIT_MULTIPLE * p; total = int(d.off[-1])
    scores = {f"{arm}_{ro}": np.full(total, np.nan) for arm in ARMS for ro in ("top10", "mean")}
    native = {arm: np.zeros(d.n, bool) for arm in ARMS}
    diag = {arm: {"distance_cosine": [], "distance_l2": [], "ipr": [], "abstained": 0, "failed": 0, "alpha": [], "K": []} for arm in ARMS}
    fit_counts = np.zeros(d.n, int); uncovered = 0
    for i in range(d.n):
        x = mats[i]; a, b = d.off[i:i + 2]
        if len(x) < M.WIDTH:
            uncovered += 1
            for k in scores:
                scores[k][a:b] = 0.0
            continue
        plan, V, names = M.answer_views(x, sp[i], views)
        fit = V[plan.fit_indices]; fit_counts[i] = len(fit)
        mu = fit.mean(0); sd = fit.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
        Z = (fit - mu) / sd; Vz = (V - mu) / sd
        equal_dense = Vz.mean(1)
        for arm in ARMS:
            w = np.full(p, 1.0 / p); info = {}
            if arm != "equal" and len(fit) >= min_fit:
                try:
                    w, info = fit_weights(Z, names, arm)
                    if not np.isfinite(w).all() or np.abs(w).sum() <= 1e-12:
                        raise ValueError("degenerate weights")
                    native[arm][i] = True
                except Exception:
                    diag[arm]["failed"] += 1; w = np.full(p, 1.0 / p); info = {}
            elif arm == "equal":
                native[arm][i] = True
            dense = Vz @ w
            if native[arm][i] and arm != "equal" and np.corrcoef(dense, equal_dense)[0, 1] < 0:
                w = -w; dense = -dense
            if native[arm][i] and arm != "equal":
                dq = distance_from_equal(w); diag[arm]["distance_cosine"].append(dq["cosine"]); diag[arm]["distance_l2"].append(dq["l2"])
                diag[arm]["ipr"].append(weight_ipr(w)); diag[arm]["abstained"] += int(bool(info.get("abstained", False)))
                if "alpha" in info:
                    diag[arm]["alpha"].append(info["alpha"])
                if "K" in info:
                    diag[arm]["K"].append(info["K"])
            tok = windows_to_tokens(plan, dense)
            scores[f"{arm}_top10"][a:b] = masked_step_top10(tok, np.ones(len(tok), bool), sp[i])
            scores[f"{arm}_mean"][a:b] = tokens_to_official_steps(tok, sp[i][:, 0], sp[i][:, 1])
    for k in scores:
        scores[k] = L.answer_z(np.nan_to_num(scores[k]), d.off)
    summary = {arm: {"native_answers": int(native[arm].sum()), "fallback_to_equal": int(d.n - native[arm].sum()),
                     "failed_fits": diag[arm]["failed"], "abstained": diag[arm]["abstained"],
                     "distance_cosine_median": float(np.median(diag[arm]["distance_cosine"])) if diag[arm]["distance_cosine"] else None,
                     "distance_l2_median": float(np.median(diag[arm]["distance_l2"])) if diag[arm]["distance_l2"] else None,
                     "ipr_median": float(np.median(diag[arm]["ipr"])) if diag[arm]["ipr"] else None,
                     "alpha_median": float(np.median(diag[arm]["alpha"])) if diag[arm]["alpha"] else None,
                     "K_counts": {str(k): int(v) for k, v in zip(*np.unique(diag[arm]["K"], return_counts=True))} if diag[arm]["K"] else None}
               for arm in ARMS}
    summary["uncovered_answers_below_width"] = uncovered; summary["min_fit_windows"] = min_fit
    summary["fit_windows_median"] = float(np.median(fit_counts[fit_counts > 0])) if (fit_counts > 0).any() else None
    return scores, native, summary


def build_methods(d, scores, native):
    methods = {}
    if "ct7" in d.references:
        methods["ct7"] = L.method_from_scores(d, d.references["ct7"])
    for arm in ARMS:
        for ro in ("top10", "mean"):
            name = f"window_{arm}_{ro}"; s = scores[f"{arm}_{ro}"]
            methods[name] = L.method_from_scores(d, s, fallback=~native[arm])
            if arm != "equal":
                m = L.method_from_scores(d, s); m["valid"] = native[arm].copy(); methods[name + "_native"] = m
    return methods


def planned(names):
    pairs = []
    def add(a, b, why):
        if a in names and b in names and (a, b, why) not in pairs:
            pairs.append((a, b, why))
    for arm in ("iu", "shrink_iu", "lsml"):
        add(f"window_{arm}_top10", "window_equal_top10", "PRIMARY_learned_minus_equal" if arm == "lsml" else "learned_minus_equal")
        add(f"window_{arm}_top10_native", "window_equal_top10", "learned_minus_equal_native_rows")
        add(f"window_{arm}_mean", "window_equal_mean", "learned_minus_equal_overlap_mean")
    add("window_shrink_iu_top10", "window_iu_top10", "shrinkage_minus_iu")
    for arm in ARMS:
        add(f"window_{arm}_top10", "ct7", "window_minus_ct7")
    add("window_equal_top10", "window_equal_mean", "top10_minus_overlap_mean")
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--draws", type=int)
    args = p.parse_args(); started = time.perf_counter()
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="window_fusion_dry_")); d = M.synthetic_inputs(tmp)
        views = M.DEFAULT_VIEWS; eleven, ct7t = tmp / "TOKEN_MATRICES.npz", tmp / "CT7_TOKEN_MATRICES.npz"; out = d.out
        gate = {"gate_passed": args.force}
    else:
        d = L.light_dataset(args.config); paths = d.c["paths"]; views = d.c.get("views", M.DEFAULT_VIEWS); out = d.out
        eleven, ct7t = Path(paths["tokens"]), Path(paths["ct7_tokens"])
        gate = json.loads((out / "WINDOW_PR.json").read_text(encoding="utf8"))
        L.replay_ct7(d)
    if not gate.get("gate_passed") and not args.force:
        print("stage 1 gate not passed; the fusion stage does not run (docs/experiments/WINDOW_REPRESENTATION_B3_V1.md)"); return
    if not args.dry_run:
        L.run_freeze(out / "fusion", [Path(__file__), L.ROOT / "scripts/experiments/ct7_levers_common.py", L.ROOT / "spectral_utils/window_moment_bank.py",
                                      L.ROOT / "scripts/diagnostics/window_pr_measurement_v1.py", L.ROOT / "spectral_utils/shrinkage_iu.py", L.ROOT / "spectral_utils/upcr.py"],
                     [Path(paths[k]) for k in ("roster", "joined", "folds", "ct7", "tokens", "ct7_tokens") if k in paths] + [out / "WINDOW_PR.json"],
                     {"schema": SCHEMA, "views": views, "development_only": True})
        d.out = out / "fusion"; d.out.mkdir(parents=True, exist_ok=True)
    mats, sp = M.base_streams(d, eleven, ct7t)
    scores, native, summary = run(d, mats, sp, views)
    methods = build_methods(d, scores, native)
    contrasts = planned(list(methods))
    result = L.evaluate_methods(d, methods, contrasts_extra=contrasts, strata_contrasts=contrasts, prmscore=not args.dry_run, draws=args.draws)
    rows = L.summary_rows(d, methods, result); L.write_summary_csv(d.out / "SUMMARY.csv", rows)
    L.dump(d.out / "RESULTS.json", {"schema": SCHEMA, "development_only": True, "note": L.DEVELOPMENT_NOTE, "views": views,
                                    "access": "answer-only fits on each answer's own non-overlapping fit windows; no other answers, no labels",
                                    "fits": summary, "pb": result["pb"], "prm": result["prm"], "strata": result["strata"],
                                    "timing": result["timing"], "total_seconds": time.perf_counter() - started, "dry_run": bool(args.dry_run)})
    print(f"{'method':32s} {'SLA':>7s} {'F1':>7s} {'within':>7s}")
    for r in rows:
        print(f"{r['method']:32s} {100*r['sla_macro8']:7.2f} {100*r['f1_common_gate']:7.2f} {r['within_auc']:7.4f}")
    print(json.dumps({k: v for k, v in summary.items() if k in ARMS}, indent=1))
    print("written:", d.out, f"({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
