#!/usr/bin/env python
"""
rescore_phase15_selfconsistency.py — Item-5 follow-up #1 (Step 0d).

Phase-15 left 5 same-T=1.0 MATH-500 / Qwen2.5-Math-7B passes cached on Drive.
Item 5 (Step 152) failed its decision gate against an LW-SE baseline that later
turned out to be fragile (NLI truncation suspected, MATH-500 row sign-flipped).
The proper, zero-GPU re-test: an answer-agreement self-consistency baseline
computed from those 5 cached passes, fused with the single-pass L-SML GOOD_5
score, re-checked against the original Item-5 gate
(rho < 0.75 AND fused > max(single arms) + 1pp).

Two modes, auto-detected:

FULL RESCORE — runs when the 5 raw per-pass caches (Drive:
  cache/phase15_temperature/math500_qwen7b_T1.0_run{0..4}.pkl) are copied into
  local_cache/ (or local_cache/phase15_temperature/). Extracts the final boxed
  answer per pass, computes answer-agreement self-consistency
  (spectral_utils.baselines.self_consistency_score), builds the single-pass
  L-SML GOOD_5 score from run0 (anchor_orient(epr), production rule per the
  Step-0b anchor test), fuses (z-score + average), and re-checks the gate.

PARTIAL (results pkl only) — runs now against
  local_cache/phase15_results.pkl. That pkl stores the Step-158 analysis
  (per-arm fused scores + labels + cross-pass correlation matrices) but NOT
  the generated texts, so the answer-agreement arm cannot be built from it.
  What CAN be re-derived label-free from it — and is reported — is the Item-5
  gate applied to the same-T K=5 entropy-averaging arm (Item 6's condition A):
  cross-pass rho and the fused-vs-single delta with its paired-bootstrap CI.

Usage:
    PYTHONPATH=. python scripts/rescore_phase15_selfconsistency.py
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.baselines import self_consistency_score
from spectral_utils.data_loaders import _extract_math_answer
from spectral_utils.feature_utils import extract_all_features
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient

GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]
N_PASSES = 5
GATE_PP = 0.01
RESULTS_PKL = os.path.join("local_cache", "phase15_results.pkl")
OUT_JSON = os.path.join("results", "repgrid", "phase15_rescore.json")

RUN_DIRS = ["local_cache", os.path.join("local_cache", "phase15_temperature"),
            os.path.join("cache", "phase15_temperature")]


def _find_run_paths():
    paths = []
    for r in range(N_PASSES):
        name = f"math500_qwen7b_T1.0_run{r}.pkl"
        p = next((os.path.join(d, name) for d in RUN_DIRS
                  if os.path.exists(os.path.join(d, name))), None)
        paths.append(p)
    return paths


def full_rescore(run_paths):
    """Answer-agreement SC from the 5 raw passes + fusion with 1-pass L-SML."""
    passes = []
    for p in run_paths:
        with open(p, "rb") as f:
            passes.append(pickle.load(f))
    common = sorted(set.intersection(*[
        {k for k, v in c.items() if v.get("done")} for c in passes]))
    print(f"  common done samples across {N_PASSES} passes: {len(common)}")

    answers = {k: [_extract_math_answer(c[k]["full_text"]) or None for c in passes]
               for k in common}
    sc = np.array([self_consistency_score(answers[k]) for k in common])

    base = passes[0]
    f_raw = {k: extract_all_features(base[k]["token_entropies"]) for k in common}
    keys = [k for k in common if f_raw[k] is not None and np.isfinite(sc[common.index(k)])]
    labels = np.array([int(base[k]["label"]) for k in keys])
    feats = {f: np.array([f_raw[k][f] for k in keys]) for f in GOOD_5}
    sc_v = np.array([sc[common.index(k)] for k in keys])

    fused, _ = lsml_continuous_pipeline(feats, GOOD_5, FEATURE_SIGNS)
    anchor = zscore(feats["epr"] * FEATURE_SIGNS["epr"])
    lsml, _flip = anchor_orient(np.asarray(fused, dtype=float), anchor)

    lsml_auc, lsml_lo, lsml_hi = boot_auc(labels, lsml)
    sc_auc, sc_lo, sc_hi = boot_auc(labels, sc_v)
    rho = float(spearmanr(lsml, sc_v).statistic)
    combo = 0.5 * (zscore(lsml) + zscore(sc_v))
    fu_auc, fu_lo, fu_hi = boot_auc(labels, combo)

    best_single = max(lsml_auc, sc_auc)
    gate = (abs(rho) < 0.75) and (fu_auc > best_single + GATE_PP)
    print(f"  L-SML 1-pass GOOD_5   {lsml_auc:.4f} [{lsml_lo:.4f}, {lsml_hi:.4f}]")
    print(f"  SC answer-agreement   {sc_auc:.4f} [{sc_lo:.4f}, {sc_hi:.4f}] (K={N_PASSES})")
    print(f"  Spearman rho          {rho:+.3f}  ({'OK' if abs(rho) < 0.75 else 'too correlated'})")
    print(f"  fused (z+avg)         {fu_auc:.4f} [{fu_lo:.4f}, {fu_hi:.4f}]")
    print(f"  Item-5 gate (rho<0.75 AND fused > best_single + 1pp): "
          f"{'PASS' if gate else 'FAIL'}")
    return {
        "mode": "full-rescore", "n": len(keys),
        "lsml": (lsml_auc, lsml_lo, lsml_hi), "sc": (sc_auc, sc_lo, sc_hi),
        "rho": rho, "fused": (fu_auc, fu_lo, fu_hi), "gate_pass": bool(gate),
    }


def partial_from_results(res):
    """What is honestly derivable from phase15_results.pkl alone."""
    q2 = res["q2"]
    aucs = {k: tuple(float(x) for x in v) for k, v in q2["aucs"].items()}
    base = aucs["single pass T=1.0 (base)"]
    a_lsml = aucs["A: K=5 same-T, L-SML"]
    a_avg = aucs["A: K=5 same-T, simple avg"]
    rho_m = np.asarray(q2["A_rho"], dtype=float)
    off = rho_m[~np.eye(rho_m.shape[0], dtype=bool)]
    rho_mean = float(off.mean())
    d_ab = tuple(float(x) for x in q2["delta_A_minus_base"])

    print("  stored Step-158 arms (AUROC [95% CI]):")
    for name, (a, lo, hi) in aucs.items():
        print(f"    {name:32s} {a:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"  cross-pass Spearman-like rho (A arm, off-diag mean): {rho_mean:+.3f}")
    print(f"  delta A_LSML - base: {d_ab[0]:+.4f}  CI [{d_ab[1]:+.4f}, {d_ab[2]:+.4f}]")

    gate = (abs(rho_mean) < 0.75) and (a_lsml[0] > base[0] + GATE_PP)
    print(f"\n  Item-5 gate applied to the same-T K=5 entropy-averaging arm "
          f"(Item 6 condition A):")
    print(f"    rho {rho_mean:+.3f} < 0.75  AND  fused {a_lsml[0]:.4f} > "
          f"base {base[0]:.4f} + 1pp  ->  {'PASS' if gate else 'FAIL'}")
    print("    (This is the reconciling read of Items 5 vs 6: extra passes spent on "
          "averaging the same cheap entropy signal clear the gate; extra passes "
          "spent on NLI semantic clustering did not.)")
    return {
        "mode": "partial-results-pkl", "aucs": aucs,
        "a_rho_offdiag_mean": rho_mean, "delta_A_minus_base": d_ab,
        "gate_same_T_averaging_pass": bool(gate),
    }


def main():
    if not os.path.exists(RESULTS_PKL):
        print(f"WAITING: {RESULTS_PKL} not yet available")
        print("Drop the file into local_cache/ and re-run.")
        return 1
    with open(RESULTS_PKL, "rb") as f:
        res = pickle.load(f)

    run_paths = _find_run_paths()
    have = [p for p in run_paths if p]
    print(f"raw T=1.0 pass caches found: {len(have)}/{N_PASSES}")

    if len(have) == N_PASSES:
        report = full_rescore(run_paths)
    else:
        print("PARTIAL mode — phase15_results.pkl has the fused arrays and labels "
              "but not the generated texts, so the answer-agreement arm needs the "
              "raw pass caches. Copy from Drive cache/phase15_temperature/:")
        for r in range(N_PASSES):
            print(f"  math500_qwen7b_T1.0_run{r}.pkl  ->  local_cache/")
        print()
        report = partial_from_results(res)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
