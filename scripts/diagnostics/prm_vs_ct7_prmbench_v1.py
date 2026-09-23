#!/usr/bin/env python
"""PRMBench measurement (2026-09-23): the supervised Qwen2.5-Math-PRM-7B beside CT7, matched.

Two questions, measurement only (no new method arm, nothing fitted on labels):

1. Matched comparison. The PRM's step rewards are already in `prmbench_prm.pkl` (the file every
   PRMScore run loads). Score them as a locator (risk = 1 - reward) under the SAME endpoints as
   CT7: within-answer AUROC, pooled fold AUROC, and PRMScore under our two label-free/inner threshold
   rules (q80 and inner-selected, `cvf_v2.scoring.prmscores`) as well as the PRM's native 0.5
   threshold. Paired source-group intervals for the within-AUC differences, per-classification
   within-AUC.
2. Independence. Is the PRM a conditionally independent view of CT7's evidence? Within-label
   (conditional) correlation of the answer-standardized PRM risk with each CT7 view and the CT7 mean,
   and the conditional participation ratio (the 1.80 statistic) with and without the PRM. Plus a
   descriptive error-complementarity table: on erroneous PRMBench answers, does the argmax step
   of each method land on an error step, and how do the two hit sets overlap.

Labels enter the endpoints and the independence measurement only, as in Steps 414 / 418.

    python -B scripts/diagnostics/prm_vs_ct7_prmbench_v1.py --config configs/prm_vs_ct7_prmbench_v1.json
    python -B scripts/diagnostics/prm_vs_ct7_prmbench_v1.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import ct7_levers_common as L  # noqa: E402

L.ensure_spectral_package()
from spectral_utils.family_equal_readout import FAMILIES_421, partition_scores  # noqa: E402
from spectral_utils.window_moment_bank import conditional_participation_ratio  # noqa: E402

SCHEMA = "prm-vs-ct7-prmbench-v1"
SEED = 20260923


def prm_risk(d) -> tuple[np.ndarray, dict]:
    """Per-step PRM risk (1 - reward) on PRMBench steps; NaN elsewhere."""
    risk = np.full(int(d.off[-1]), np.nan); bad = 0
    for i in np.flatnonzero(d.prm):
        r = d.meta_by_id[d.ids[i]]["rewards"]
        r = np.asarray(json.loads(r) if isinstance(r, str) else r, float)
        a, b = d.off[i:i + 2]
        if r.shape != (b - a,) or not np.isfinite(r).all():
            bad += 1; continue
        risk[a:b] = 1.0 - r
    return risk, {"prm_answers": int(d.prm.sum()), "reward_length_or_nan_failures": bad,
                  "reward_min": float(np.nanmin(1 - risk)), "reward_max": float(np.nanmax(1 - risk))}


def conditional_corr(x: np.ndarray, Y: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of x with each column of Y after within-label centring."""
    m = np.column_stack([x, Y]).astype(float)
    for cls in (True, False):
        sel = y == cls
        m[sel] -= m[sel].mean(0)
    c = np.corrcoef(m.T)
    return c[0, 1:]


def paired_within(d, auc: dict, a: str, b: str, draws: int) -> dict:
    A, B = auc[a], auc[b]
    mask = d.prm & np.isfinite(A) & np.isfinite(B)
    rng = np.random.default_rng(SEED)
    diffs = np.array([A[idx].mean() - B[idx].mean() for idx in L.group_draws(d.groups, mask, rng, draws)])
    out = L.interval(diffs); out["point_pp"] = float(100 * (A[mask].mean() - B[mask].mean()))
    out["answers"] = int(mask.sum()); out["draws"] = int(draws)
    return out


def hits(d, scores: np.ndarray, answers: np.ndarray) -> np.ndarray:
    """1 if the argmax step of the answer is a labelled error step (PRMBench labels)."""
    out = np.zeros(len(answers), bool)
    for k, i in enumerate(answers):
        a, b = d.off[i:i + 2]
        out[k] = bool(d.labels[a + int(np.argmax(scores[a:b]))])
    return out


def measure(d, profiles: np.ndarray, *, draws: int, prmscore: bool = True) -> dict:
    S = L.cvf().scoring
    started = time.perf_counter()
    risk, prm_info = prm_risk(d)
    ok = np.isfinite(risk) | ~np.repeat(d.prm, np.diff(d.off))
    assert prm_info["reward_length_or_nan_failures"] == 0, prm_info
    ct7 = profiles.mean(1)
    fam, _ = partition_scores(profiles, [g for g in _groups421()], d.off, rule="answer")
    filled = np.where(np.isfinite(risk), risk, 0.0)
    arms = {"ct7": ct7, "ct7_z": L.answer_z(ct7, d.off), "fam421_answer": fam,
            "prm_risk": filled, "prm_risk_z": L.answer_z(filled, d.off)}
    methods = {k: L.method_from_scores(d, v) for k, v in arms.items()}
    for m in methods.values():
        m["valid"] = d.prm.copy()          # PRMBench only; PB is not scored by the PRM file
    prm, auc = {}, {}
    for name, m in methods.items():
        prm[name], auc[name] = S.prm_metrics(d, m)
    by_class = {}
    cls = np.array([d.meta_by_id[d.ids[i]]["classification"] if d.prm[i] else "" for i in range(d.n)])
    for c in sorted(set(cls[d.prm])):
        sel = d.prm & (cls == c)
        by_class[c] = {"answers": int(sel.sum()),
                       **{name: (float(np.nanmean(auc[name][sel])) if np.isfinite(auc[name][sel]).any() else None)
                          for name in ("ct7", "fam421_answer", "prm_risk")}}
    contrasts = {f"{a}_minus_{b}": paired_within(d, auc, a, b, draws)
                 for a, b in (("prm_risk", "ct7"), ("prm_risk", "fam421_answer"), ("fam421_answer", "ct7"))}
    t_metrics = time.perf_counter() - started

    # independence on labelled PRMBench steps (labels used for measurement only)
    steps = np.repeat(d.prm, np.diff(d.off)) & ok
    y = d.labels[steps] == 1
    V = profiles[steps]; pz = arms["prm_risk_z"][steps]; cz = arms["ct7_z"][steps]
    from spectral_utils.family_equal_readout import family_means
    F = family_means(profiles, _groups421())[steps]
    independence = {
        "steps": int(steps.sum()), "error_steps": int(y.sum()),
        "conditional_corr_prm_with_views": dict(zip(L.CT7_VIEWS, conditional_corr(pz, V, y).round(4).tolist())),
        "conditional_corr_prm_with_ct7_mean": float(conditional_corr(pz, cz[:, None], y)[0]),
        "marginal_corr_prm_with_ct7_mean": float(np.corrcoef(pz, cz)[0, 1]),
        "pr_ct7_seven_views": conditional_participation_ratio(V, y),
        "pr_ct7_seven_views_plus_prm": conditional_participation_ratio(np.column_stack([V, pz]), y),
        "pr_three_families": conditional_participation_ratio(F, y),
        "pr_three_families_plus_prm": conditional_participation_ratio(np.column_stack([F, pz]), y),
        "pr_ct7_mean_and_prm": conditional_participation_ratio(np.column_stack([cz, pz]), y),
    }

    # complementarity: erroneous PRMBench answers, argmax on an error step
    err = np.flatnonzero(d.prm & np.array([d.labels[d.off[i]:d.off[i + 1]].any() for i in range(d.n)]))
    h_c, h_p, h_f = hits(d, arms["ct7"], err), hits(d, arms["prm_risk"], err), hits(d, arms["fam421_answer"], err)
    phi = float(np.corrcoef(h_c, h_p)[0, 1]) if h_c.std() and h_p.std() else None
    rng = np.random.default_rng(SEED + 1)
    null = []                               # hit-set overlap expected if the two were independent
    for _ in range(200):
        null.append(float((h_c & rng.permutation(h_p)).mean()))
    complementarity = {
        "erroneous_answers": int(len(err)),
        "hit_rate": {"ct7": float(h_c.mean()), "fam421_answer": float(h_f.mean()), "prm_risk": float(h_p.mean())},
        "both": float((h_c & h_p).mean()), "ct7_only": float((h_c & ~h_p).mean()),
        "prm_only": float((~h_c & h_p).mean()), "neither": float((~h_c & ~h_p).mean()),
        "union": float((h_c | h_p).mean()), "phi": phi,
        "both_if_independent_mean": float(np.mean(null)),
        "note": "argmax hit on any labelled error step; descriptive, labels used for evaluation only",
    }
    per_answer_rank_corr = float(np.corrcoef(
        *[np.argsort(np.argsort(auc[k][np.isfinite(auc["ct7"]) & np.isfinite(auc["prm_risk"])])) for k in ("ct7", "prm_risk")])[0, 1])

    tables = None; t_score = 0.0
    if prmscore:
        started = time.perf_counter()
        tables = S.prmscores(d, {k: methods[k] for k in ("ct7_z", "fam421_answer", "prm_risk", "prm_risk_z")})
        t_score = time.perf_counter() - started
    summary = {}
    for name in methods:
        row = {"within_auc": prm[name]["within_auc"], "eligible": prm[name]["eligible"],
               "pooled_step_auroc_mean_folds": prm[name]["step_auroc_mean_folds"]}
        if tables and name in tables:
            for rule, v in tables[name].items():
                row[f"prmscore_{rule}"] = v["prmscore"]
        summary[name] = row
    if tables:
        summary["supervised_qwen25math_prm7b_native"] = {"prmscore_native_threshold_0.5":
                                                         tables["supervised_qwen25math_prm7b"]["native_threshold_0.5"]["prmscore"]}
    return {"summary": summary, "prm_input": prm_info, "within_auc_contrasts": contrasts,
            "within_auc_by_classification": by_class, "independence": independence,
            "complementarity": complementarity, "per_answer_within_auc_rank_corr_ct7_prm": per_answer_rank_corr,
            "timing": {"metrics_seconds": t_metrics, "prmscore_seconds": t_score}}


def _groups421() -> list[int]:
    g = [0] * 7
    for k, (_, members) in enumerate(FAMILIES_421.items()):
        for j in members:
            g[j] = k
    return g


def synthetic(tmp: Path):
    """Synthetic population with a PRM metadata file (rewards informative, partly independent)."""
    d, profiles = L.synthetic_dataset(tmp, n_answers=120, seed=3)
    rng = np.random.default_rng(4); meta = {}
    classes = ["missing_condition", "confidence", "redundency"]
    for k, i in enumerate(np.flatnonzero(d.prm)):
        a, b = d.off[i:i + 2]; lab = d.labels[a:b]
        logit = 1.5 - 2.0 * lab + 0.5 * profiles[a:b, 0] * 0 + rng.standard_normal(b - a)
        meta[k] = {"idx": d.ids[i], "classification": classes[k % 3], "n_steps": int(b - a),
                   "error_steps": (np.flatnonzero(lab) + 1).tolist(), "rewards": (1 / (1 + np.exp(-logit))).tolist()}
    pickle.dump(meta, open(tmp / "prm.pkl", "wb"))
    d.meta_by_id = {m["idx"]: m for m in meta.values()}
    return d, profiles


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(); started = time.perf_counter()
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="prm_vs_ct7_dry_"))
        d, profiles = synthetic(tmp); draws = 50
    else:
        d = L.light_dataset(args.config); paths = d.c["paths"]; draws = int(d.c["bootstrap_draws"])
        assert d.meta_by_id is not None, "prm_metadata (prmbench_prm.pkl) is required"
        L.run_freeze(d.out, [Path(__file__), L.ROOT / "scripts/experiments/ct7_levers_common.py",
                             L.ROOT / "spectral_utils/family_equal_readout.py", Path(args.config).resolve()],
                     [Path(paths[k]) for k in ("roster", "joined", "folds", "ct7", "profiles", "prm_metadata")],
                     {"schema": SCHEMA, "development_only": True, "draws": draws})
        profiles = L.load_ct7_profiles(d, Path(paths["profiles"]), Path(paths["profile_validation"]))
    record = measure(d, profiles, draws=draws)
    if not args.dry_run:
        within_ct7 = record["summary"]["ct7"]["within_auc"]
        assert abs(within_ct7 - L.CT7_WITHIN_AUC) < 1e-12, within_ct7
    record.update({"schema": SCHEMA, "development_only": True, "note": L.DEVELOPMENT_NOTE,
                   "access": "PRM rows are a supervised external verifier (high-access reference), not a label-free arm",
                   "seconds": time.perf_counter() - started, "dry_run": bool(args.dry_run)})
    L.dump(d.out / "MEASUREMENT.json", record)
    print(json.dumps({k: record[k] for k in ("summary", "within_auc_contrasts", "independence", "complementarity")}, indent=1))
    print("written:", d.out / "MEASUREMENT.json", f"({record['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
