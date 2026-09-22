#!/usr/bin/env python3
"""Digit-free 20-stream ladder: does a partition earn anything without digit streams?

Question (Omri, 2026-09-17, after Step 399 addendum showed the leading Joint arm is
"auto K=3 partition + block averaging"): on Codex's proposed 20-stream digit-free bank,
which of the partition, the within-group rule and the cross-group rule carries any
localization quality, at EVERY admissible K (3..8), under BOTH label-free grouping rules
(Joint's residual-affinity spectral rule and Continuous L-SML's Eq.15 score rule)?

Contract (identical to results/digitfree_broad50_v1): the frozen 13,769-answer development
population, v3 labels / v2 source groups, five outer source folds, answer-local step
standardization (missing entries neutral zero), weights fit on the pooled steps of the four
training source folds, non-digit tail15 gate (within-cell midrank >= .33), H1 sign anchor.
Digit-free bank of 50 streams extracted by Codex (results/digitfree_broad50_v1/extracted) plus
the pure BOCPD residual channel (results/joint_feature_selection_bocpd_v1/INPUTS.npz).
Development evidence only; no untouched confirmation.

Readouts on every partition, all label-free, sign by H1 anchor:
  ceq         mean within group of standardized streams; virtuals standardized; equal across groups
  csml        same within; cross-group SML eigen-solve at every K (guard OFF: real solve at K=3)
  csml_guard  same within; cross-group SML with the Step-205 guard (Codex's proposal verbatim)
  lsml        Continuous L-SML on the given groups (within SML + cross SML, guard on)
  jrel        Joint factor model on the partition; group weight v_g/(u_g^2 + s_g^2/n_g) with
              v_g, u_g, s_g^2 the group means of the fitted global/group loadings and noise
              variances (the ONE arm where the fitted model enters the weights)
Global rows: equal20, H1 singleton, IU-PCR (frozen defaults), Continuous L-SML with its own K,
and the K each grouping rule would select on its own (stability tie-break / Eq.14 residual).
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_digitfree_broad50_v1 import load_data  # noqa: E402
from scripts.run_lsml_gate_locator_research_v1 import score_locator  # noqa: E402
from spectral_utils.digitfree_broad50 import NAMES as BANK50  # noqa: E402
from spectral_utils.fusion_utils import (  # noqa: E402
    _spectral_cluster_precomputed, detect_dependent_groups, lsml_continuous, sml_fuse_signed,
)
from spectral_utils.joint_lsml import canonicalize_labels, covariance_matrix, residual_affinity  # noqa: E402
from spectral_utils.joint_pair_jacobian import fit_joint_pairs_checked  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.lsml_gate_locator_research import (  # noqa: E402
    FusionRecipe, _orient, continuous_lsml_weight_vector, fit_fusion_weights,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402

SCHEMA = "digitfree20-ladder-v1"
OUT = ROOT / "results/digitfree20_ladder_v1"
SEED = 399200
K_RANGE = (3, 4, 5, 6, 7, 8)
RULES = ("affinity", "eq15")
READOUTS = ("ceq", "csml", "csml_guard", "lsml", "jrel")
ROSTER20 = (  # Codex's proposal, 2026-09-17 (other machine); BOCPD residual appended
    "rank_1_risk", "rank_3_risk", "surprisal", "top1_loggap", "censored_rank50", "mass_above", "top2_ratio",
    "H0lim", "a0.25", "H1", "ve0", "ve0.75", "ve1",
    "H0lim_prefix_innovation", "a0.25_prefix_innovation", "ve0.75_prefix_innovation", "ve1_prefix_innovation",
    "top15_turnover", "top50_truncated_js",
    "bocpd_residual",
)
ANCHOR = ROSTER20.index("H1")


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def build_matrix(data):
    idx = [BANK50.index(n) for n in ROSTER20 if n != "bocpd_residual"]
    with np.load(ROOT / "results/joint_feature_selection_bocpd_v1/INPUTS.npz") as z:
        bocpd = z["bocpd"].astype(np.float64)
    x = np.column_stack([data["x"][:, idx], bocpd])
    if x.shape != (len(data["x"]), len(ROSTER20)) or not np.isfinite(x).all():
        raise ValueError("roster matrix malformed")
    return x


# ---------------------------------------------------------------- partitions
def partitions_affinity(train, row_folds, seed):
    """Residual-affinity spectral partitions at every K, with block-deletion stability."""
    from sklearn.metrics import adjusted_rand_score
    full = residual_affinity(covariance_matrix(train))[0]
    deletions = [residual_affinity(covariance_matrix(train[row_folds != f]))[0] for f in np.unique(row_folds)]
    out = {}
    for k in K_RANGE:
        labels = canonicalize_labels(_spectral_cluster_precomputed(full, k, seed=seed + 1000 * k))
        parts = [canonicalize_labels(_spectral_cluster_precomputed(a, k, seed=seed + 1000 * k + j))
                 for j, a in enumerate(deletions)]
        ari = [float(adjusted_rand_score(labels, p)) for p in parts]
        sizes = [int(np.sum(labels == g)) for g in np.unique(labels)]
        out[k] = {"labels": labels, "sizes": sizes, "median_ari": float(np.median(ari)), "min_ari": float(min(ari)),
                  "admissible": bool(len(sizes) == k and min(sizes) >= 2)}
    adm = [k for k in K_RANGE if out[k]["admissible"]]
    selected = sorted(adm, key=lambda k: (-out[k]["median_ari"], -out[k]["min_ari"], k))[0] if adm else None
    return out, selected


def partitions_eq15(train):
    """Continuous L-SML's Eq.15 score-matrix partitions at every K, plus its own residual choice."""
    views = [train[:, j] for j in range(train.shape[1])]
    out = {}
    for k in K_RANGE:
        _, c, residual, _ = detect_dependent_groups(views, K_range=[k], method="residual")
        labels = canonicalize_labels(c)
        sizes = [int(np.sum(labels == g)) for g in np.unique(labels)]
        out[k] = {"labels": labels, "sizes": sizes, "residual": float(residual), "admissible": bool(len(sizes) == k)}
    auto_k, _, _, _ = detect_dependent_groups(views, K_range=list(K_RANGE), method="residual")
    return out, int(auto_k)


# ------------------------------------------------------------------ readouts
def group_means(train, labels):
    groups = np.unique(labels)
    members = [np.flatnonzero(labels == g) for g in groups]
    virtual = np.column_stack([train[:, m].mean(axis=1) for m in members])
    return members, virtual


def readout_weights(train, labels, cov, seed):
    """All partition readouts as raw per-stream weight vectors (orientation by caller)."""
    members, virtual = group_means(train, labels)
    k = len(members)
    sd = virtual.std(axis=0, ddof=1)
    w = {}
    meta = {}
    # ceq: mean within, standardized virtual, equal across groups
    w["ceq"] = np.zeros(train.shape[1])
    for m, s in zip(members, sd):
        w["ceq"][m] = (1.0 / len(m)) / s / k
    # csml / csml_guard: mean within, cross SML on the virtuals
    for name, guard in (("csml", False), ("csml_guard", True)):
        if k == 1:
            cross = np.array([1.0])
        else:
            _, cross = sml_fuse_signed(*[virtual[:, i] for i in range(k)], small_m_guard=guard)
        vec = np.zeros(train.shape[1])
        for m, c in zip(members, cross):
            vec[m] = c / len(m)
        w[name] = vec
        meta[name] = {"cross": np.asarray(cross, float).tolist()}
    # lsml: Continuous L-SML with the groups supplied (within SML + cross SML, guard on)
    _, lm = lsml_continuous(*[train[:, j] for j in range(train.shape[1])], groups=np.asarray(labels, int),
                            compute_score_matrix=False, small_m_guard=True)
    w["lsml"] = continuous_lsml_weight_vector(lm, train.shape[1])
    meta["lsml"] = {"cross": np.asarray(lm["cross_weights"], float).tolist(),
                    "guarded": [list(v) for v in lm["small_m_guarded"]]}
    # jrel: Joint factor model on the partition; group weight from fitted loadings
    sizes = [len(m) for m in members]
    if k >= 3 and min(sizes) >= 2:
        try:
            fit = fit_joint_pairs_checked(cov, labels, anchor_index=ANCHOR, seed=seed, starts=5)
            j = fit.joint
            valid = bool(j.converged and j.multistart_audit["status"] == "PASS"
                         and j.jacobian_audit.get("full_global_rank", False)
                         and np.isfinite(j.jacobian_audit.get("condition_number", np.inf))
                         and j.jacobian_audit.get("condition_number", np.inf) <= 1e8)
            v = np.asarray(j.global_loading, float); u = np.asarray(j.group_loading, float)
            noise = np.maximum(np.diag(cov) - v ** 2 - u ** 2, 1e-6)
            vec = np.zeros(train.shape[1]); rel = []
            for m in members:
                vg = v[m].mean(); ug = u[m].mean(); sg = noise[m].mean() / len(m)
                r = vg / (ug ** 2 + sg)
                rel.append(float(r)); vec[m] = r / len(m)
            w["jrel"] = vec
            meta["jrel"] = {"valid": valid, "group_reliability": rel, "converged_starts": int(j.converged_starts),
                            "multistart": j.multistart_audit["status"],
                            "relative_offdiag_misfit": float(j.relative_offdiag_misfit),
                            "pair_status": fit.pair_audit.get("status"),
                            "condition_number": float(j.jacobian_audit.get("condition_number", np.nan)),
                            "v_group_mean": [float(v[m].mean()) for m in members],
                            "u_group_mean": [float(u[m].mean()) for m in members]}
            if not valid:
                meta["jrel"]["fallback"] = "ceq"; w["jrel"] = w["ceq"].copy()
        except Exception as exc:  # explicit, reported fallback
            meta["jrel"] = {"valid": False, "failure": f"{type(exc).__name__}: {exc}", "fallback": "ceq"}
            w["jrel"] = w["ceq"].copy()
    else:
        meta["jrel"] = {"valid": False, "failure": "INADMISSIBLE_PARTITION", "sizes": sizes, "fallback": "ceq"}
        w["jrel"] = w["ceq"].copy()
    return w, meta


# ---------------------------------------------------------------------- folds
def run_fold(x, data, outer):
    offsets = np.asarray(data["offsets"]); folds = np.asarray(data["folds"])
    row_folds = np.repeat(folds, np.diff(offsets))
    test = row_folds == outer; train = x[~test]
    t0 = time.perf_counter(); seed = SEED + outer
    weights, meta = {}, {"outer": int(outer), "partitions": {}}
    cov = covariance_matrix(train)

    weights["equal20"] = np.ones(x.shape[1]) / x.shape[1]
    weights["H1"] = np.eye(x.shape[1])[ANCHOR]
    try:
        iu = upcr_fit(train.T, **dict(IU_FIT_DEFAULTS))
        weights["iu"], o = _orient(train, iu.w, ANCHOR)
        meta["iu"] = {**o, "abstained": bool(iu.abstained), "n_components_used": int(iu.n_components_used)}
    except Exception as exc:
        meta["iu"] = {"status": "FAILED", "reason": str(exc)}
    cw, cm = fit_fusion_weights(train, FusionRecipe("d20", ROSTER20, "continuous", anchor=ANCHOR), seed=seed)
    weights["continuous_auto"] = cw
    meta["continuous_auto"] = {"K": cm["K"], "groups": cm["groups"], "residual": cm["residual"]}

    aff, aff_selected = partitions_affinity(train, row_folds[~test], seed + 100)
    eq, eq_selected = partitions_eq15(train)
    meta["selected_K"] = {"affinity_stability": aff_selected, "eq15_residual": eq_selected}
    for rule, parts in (("affinity", aff), ("eq15", eq)):
        for k in K_RANGE:
            p = parts[k]
            key = f"{rule}_K{k}"
            meta["partitions"][key] = {kk: vv for kk, vv in p.items() if kk != "labels"}
            meta["partitions"][key]["labels"] = np.asarray(p["labels"], int).tolist()
            raw, rmeta = readout_weights(train, np.asarray(p["labels"], int), cov, seed + 10 * k + (0 if rule == "affinity" else 5))
            for readout, vec in raw.items():
                w, o = _orient(train, vec, ANCHOR)
                weights[f"{key}_{readout}"] = w
                meta[f"{key}_{readout}"] = {**o, **rmeta.get(readout, {})}
    scores = {arm: x[test] @ w for arm, w in weights.items()}
    for arm, w in weights.items():
        meta.setdefault(arm, {})["weights"] = np.asarray(w, float).tolist()
    meta["seconds"] = time.perf_counter() - t0
    return test, scores, meta


# ------------------------------------------------------------------ bootstrap
def bootstrap(data, results, draws, seed):
    """Vectorized paired source-group bootstrap of PB macro-F1 and PRMB within-AUC for every arm."""
    target = np.asarray(data["target"]); cells = data["cells"].astype(str)
    unique, g = np.unique(data["groups"], return_inverse=True); ng = len(unique)
    arms = list(results); cellnames = sorted(c for c in set(cells) if c.startswith("pb_"))
    base = np.zeros((ng, len(cellnames), 2)); hit = np.zeros((ng, len(cellnames), len(arms), 2))
    for j, c in enumerate(cellnames):
        cl = (cells == c) & (target == -1); er = (cells == c) & (target >= 0)
        base[:, j, 0] = np.bincount(g, weights=cl, minlength=ng); base[:, j, 1] = np.bincount(g, weights=er, minlength=ng)
        for a, arm in enumerate(arms):
            ok = results[arm]["prediction"] == target
            hit[:, j, a, 0] = np.bincount(g, weights=cl & ok, minlength=ng)
            hit[:, j, a, 1] = np.bincount(g, weights=er & ok, minlength=ng)
    v0 = results[arms[0]]["within_values"]; valid = np.isfinite(v0)
    wn = np.bincount(g, weights=valid, minlength=ng)
    ws = np.column_stack([np.bincount(g, weights=np.nan_to_num(results[a]["within_values"]), minlength=ng) for a in arms])
    rng = np.random.default_rng(seed); pb = np.empty((draws, len(arms))); within = np.empty((draws, len(arms)))
    for b0 in range(0, draws, 200):
        n = min(200, draws - b0)
        counts = rng.multinomial(ng, np.full(ng, 1 / ng), size=n).astype(float)
        tb = np.einsum("bg,gck->bck", counts, base); th = np.einsum("bg,gcak->bcak", counts, hit)
        ca = th[:, :, :, 0] / np.maximum(tb[:, :, None, 0], 1e-12); ea = th[:, :, :, 1] / np.maximum(tb[:, :, None, 1], 1e-12)
        f1 = np.divide(2 * ca * ea, ca + ea, out=np.zeros_like(ca), where=(ca + ea) > 0)
        pb[b0:b0 + n] = f1.mean(axis=1)
        within[b0:b0 + n] = (counts @ ws) / (counts @ wn)[:, None]
    return arms, pb, within


def interval(d, point):
    return {"point": float(point), "low": float(np.quantile(d, .025)), "high": float(np.quantile(d, .975)),
            "probability_positive": float(np.mean(d > 0))}


def contrasts(metrics, arms, pb, within):
    col = {a: i for i, a in enumerate(arms)}
    out = {}
    def add(name, a, b):
        if a in col and b in col:
            out[name] = {"pb": interval(pb[:, col[a]] - pb[:, col[b]], metrics[a]["pb"] - metrics[b]["pb"]),
                         "within": interval(within[:, col[a]] - within[:, col[b]], metrics[a]["within"] - metrics[b]["within"])}
    for rule in RULES:
        for k in K_RANGE:
            key = f"{rule}_K{k}"
            add(f"{key}_ceq - equal20", f"{key}_ceq", "equal20")
            for r in ("csml", "csml_guard", "lsml", "jrel"):
                add(f"{key}_{r} - {key}_ceq", f"{key}_{r}", f"{key}_ceq")
            add(f"{key}_jrel - {key}_csml", f"{key}_jrel", f"{key}_csml")
            add(f"affinity_K{k}_ceq - eq15_K{k}_ceq", f"affinity_K{k}_ceq", f"eq15_K{k}_ceq")
    for a in ("continuous_auto", "iu", "H1"):
        add(f"{a} - equal20", a, "equal20")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_data(); x = build_matrix(data)
    print(f"data: {len(data['target'])} answers, {len(x)} steps, {x.shape[1]} streams", flush=True)
    oof = {}; fold_meta = []
    for outer in args.folds:
        dest = out / f"fold_{outer}.npz"; info = out / f"fold_{outer}.json"
        if dest.is_file() and info.is_file():
            with np.load(dest) as z:
                test = z["test"]; sc = {a: z[a] for a in z.files if a != "test"}
            fold_meta.append(json.loads(info.read_text()))
        else:
            test, sc, meta = run_fold(x, data, outer)
            np.savez_compressed(dest, test=test, **sc)
            info.write_text(json.dumps(clean(meta), indent=1, sort_keys=True) + "\n")
            fold_meta.append(clean(meta))
            print(f"fold {outer}: {meta['seconds']:.0f}s, selected K {meta['selected_K']}", flush=True)
        for arm, s in sc.items():
            oof.setdefault(arm, np.full(len(x), np.nan))[test] = s
    complete = all(np.isfinite(v).all() for v in oof.values())
    if not complete:
        print("partial folds: metrics skipped"); return
    gate = np.asarray(data["gate"], bool)
    results = {arm: score_locator(s, gate, data) for arm, s in oof.items()}
    metrics = {arm: {"pb": r["pb"], "within": r["within"], "within_n": r["within_n"],
                     "pb_cells": {c: v["f1"] for c, v in r["pb_cells"].items()}} for arm, r in results.items()}
    arms, pb, within = bootstrap(data, results, args.draws, SEED + 999)
    ctr = contrasts(metrics, arms, pb, within)
    fallbacks = {}
    for fm in fold_meta:
        for key, m in fm.items():
            if key.endswith("_jrel") and isinstance(m, dict) and m.get("fallback"):
                fallbacks[key] = fallbacks.get(key, 0) + 1
    report = {"schema": SCHEMA, "roster": ROSTER20, "anchor": "H1", "gate": "tail15 within-cell midrank >= .33 (non-digit)",
              "metrics": metrics, "contrasts": ctr, "selected_K": [fm["selected_K"] for fm in fold_meta],
              "partitions": [{k: {kk: vv for kk, vv in v.items()} for k, v in fm["partitions"].items()} for fm in fold_meta],
              "jrel_fallback_folds": fallbacks, "bootstrap": {"draws": args.draws, "seed": SEED + 999, "unit": "source group"},
              "seconds": time.time() - t0}
    (out / "RUN.json").write_text(json.dumps(clean(report), indent=1, sort_keys=True) + "\n")
    np.savez_compressed(out / "OOF.npz", **oof)
    lines = ["| arm | PB macro % | PRMB within |", "|---|---:|---:|"]
    for arm in sorted(metrics, key=lambda a: -metrics[a]["pb"]):
        lines.append(f"| {arm} | {100 * metrics[arm]['pb']:.2f} | {metrics[arm]['within']:.4f} |")
    print("\n".join(lines))
    print("selected K per fold:", [fm["selected_K"] for fm in fold_meta])
    print("jrel fallbacks:", fallbacks)


if __name__ == "__main__":
    main()
