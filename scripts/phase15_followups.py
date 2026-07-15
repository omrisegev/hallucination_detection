#!/usr/bin/env python
"""
phase15_followups.py — Step-158 CPU follow-ups #2, #4, #5, #6 on the Phase-15 data.

Step 158 (Item 6, temperature variation) left an 8-item follow-up list, all flagged
"pure CPU once the caches are downloaded". Only #1 (self-consistency fusion, closed
via Step 174) and the temperature sweep itself got done. This script closes four more,
using data that is now locally available:

  F1 — K-sweep for Condition A (#2): AUROC(K) for K=1..5 same-T=1.0 passes.
  F2 — new feature families (#4): spilled-energy suite + extended logprob features
       (varentropy / Renyi-2 / tail-mass) on the Phase-15 traces, orthogonality vs GOOD_5.
  F3 — fairer diversity set B' (#5): {T=0.6, T=1.0, T=1.5} vs a matched K=3 same-T arm.
  F4 — cross-temperature probing (#6): does a hot pass's score predict the COLD
       (T=1.0 run0) label, vs its own-temperature label?

Follow-ups #3 (anchor/sign robustness across T) and #7 (length-controlled AUROC per T)
are NOT attempted here — they need raw per-sample GOOD_5 feature values / trace lengths
at T != 1.0, and phase15_results.pkl only stores scalar per-(feature,temp) AUROCs, not
per-sample arrays. They stay flagged as needing an extra Drive pull (see BLOCKED section
in the output).

Usage:
    PYTHONPATH=. python scripts/phase15_followups.py
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

from spectral_utils.feature_utils import extract_all_features
from spectral_utils.fusion_utils import (
    zscore, boot_auc, paired_boot_delta_auc,
    simple_average_fusion, lsml_continuous, lsml_continuous_pipeline,
    multipass_lsml_continuous,
)
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient
from spectral_utils.repgrid_scoring import (
    logprob_features, logprob_features_extended,
    LOGPROB_FEATS, LOGPROB_SIGNS, LOGPROB_FEATS_EXT, LOGPROB_SIGNS_EXT,
)

GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]
N_PASSES = 5
GATE_PP = 0.01
SPILLED_FEATS = ["epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled", "min_spilled"]
# PROGRESS.md-documented signs (validated GSM8K, Step 131); not in streaming_utils.FEATURE_SIGNS.
SPILLED_SIGNS = {"epr_spilled": -1, "sw_var_peak_spilled": -1,
                  "cusum_max_spilled": -1, "min_spilled": -1}

RESULTS_PKL = os.path.join("local_cache", "phase15_results.pkl")
OUT_JSON = os.path.join("results", "repgrid", "phase15_followups.json")
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


def _load_passes(run_paths):
    passes = []
    for p in run_paths:
        with open(p, "rb") as f:
            passes.append(pickle.load(f))
    common = sorted(set.intersection(*[
        {k for k, v in c.items() if v.get("done")} for c in passes]))
    return passes, common


def _build_good5_runs_feats(passes, common):
    """GOOD_5 feats_dict per pass, restricted to keys valid (trace >= 8 tok) in ALL passes."""
    per_pass_raw = [{k: extract_all_features(c[k]["token_entropies"]) for k in common}
                     for c in passes]
    valid_keys = [k for k in common if all(pr[k] is not None for pr in per_pass_raw)]
    labels = np.array([int(passes[0][k]["label"]) for k in valid_keys])
    runs_feats = [{f: np.array([pr[k][f] for k in valid_keys]) for f in GOOD_5}
                  for pr in per_pass_raw]
    return valid_keys, labels, runs_feats


def f1_k_sweep(runs_feats, labels):
    """AUROC(K) for K = 1..5 same-T=1.0 passes (multipass_lsml_continuous handles the
    per-K fallback: K=1 -> per-pass score, K=2 -> simple average, K>=3 -> L-SML)."""
    out = {}
    for K in range(1, N_PASSES + 1):
        res = multipass_lsml_continuous(runs_feats[:K], GOOD_5, FEATURE_SIGNS,
                                        anchor_feature="epr")
        auc, lo, hi = boot_auc(labels, res["fused"])
        out[K] = {"auc": float(auc), "lo": float(lo), "hi": float(hi)}
        print(f"  K={K}: AUROC {auc:.4f} [{lo:.4f}, {hi:.4f}]")
    return out


def f2_new_feature_families(passes, valid_keys, labels, base_good5_score):
    """Spilled-energy suite + extended logprob features vs GOOD_5, on run0."""
    base = passes[0]
    report = {"spilled": {}, "logprob": {}}

    # --- spilled energy ---
    spilled_vals = {f: [] for f in SPILLED_FEATS}
    for k in valid_keys:
        feats = extract_all_features(base[k]["token_entropies"],
                                     spilled_energies=base[k]["token_spilled_energies"])
        for f in SPILLED_FEATS:
            spilled_vals[f].append(feats[f])
    spilled_vals = {f: np.asarray(v, dtype=float) for f, v in spilled_vals.items()}

    print("\n  spilled-energy features (individual, signed):")
    best_spilled, best_spilled_auc = None, 0.5
    for f in SPILLED_FEATS:
        signed = spilled_vals[f] * SPILLED_SIGNS[f]
        auc, lo, hi = boot_auc(labels, signed)
        rho = float(spearmanr(signed, base_good5_score).statistic)
        report["spilled"][f] = {"auc": float(auc), "lo": float(lo), "hi": float(hi), "rho_vs_good5": rho}
        print(f"    {f:24s} AUROC {auc:.4f} [{lo:.4f},{hi:.4f}]  rho-vs-GOOD5 {rho:+.3f}")
        if abs(auc - 0.5) > abs(best_spilled_auc - 0.5):
            best_spilled, best_spilled_auc = f, auc

    report["spilled"]["best_feature"] = best_spilled
    print(f"  best spilled feature: {best_spilled} (AUROC {best_spilled_auc:.4f})")

    # --- extended logprob features ---
    lp_basic = {f: [] for f in LOGPROB_FEATS}
    lp_ext = {f: [] for f in LOGPROB_FEATS_EXT}
    for k in valid_keys:
        b = logprob_features(base[k]["top_k_logprobs"])
        e = logprob_features_extended(base[k]["top_k_logprobs"])
        for f in LOGPROB_FEATS:
            lp_basic[f].append(b[f])
        for f in LOGPROB_FEATS_EXT:
            lp_ext[f].append(e[f])
    lp_basic = {f: np.asarray(v, dtype=float) for f, v in lp_basic.items()}
    lp_ext = {f: np.asarray(v, dtype=float) for f, v in lp_ext.items()}
    all_lp_signs = {**LOGPROB_SIGNS, **LOGPROB_SIGNS_EXT}
    all_lp_vals = {**lp_basic, **lp_ext}

    print("\n  logprob features (basic + extended, individual, signed):")
    best_lp, best_lp_auc = None, 0.5
    for f, vals in all_lp_vals.items():
        signed = vals * all_lp_signs[f]
        auc, lo, hi = boot_auc(labels, signed)
        rho = float(spearmanr(signed, base_good5_score).statistic)
        report["logprob"][f] = {"auc": float(auc), "lo": float(lo), "hi": float(hi), "rho_vs_good5": rho}
        print(f"    {f:24s} AUROC {auc:.4f} [{lo:.4f},{hi:.4f}]  rho-vs-GOOD5 {rho:+.3f}")
        if abs(auc - 0.5) > abs(best_lp_auc - 0.5):
            best_lp, best_lp_auc = f, auc
    report["logprob"]["best_feature"] = best_lp
    print(f"  best logprob feature: {best_lp} (AUROC {best_lp_auc:.4f})")

    # --- fusion tests: GOOD_5 + best spilled / best logprob as a 6th view ---
    base_feats = {f: np.array([extract_all_features(base[k]["token_entropies"])[f]
                               for k in valid_keys]) for f in GOOD_5}
    for label, extra_name, extra_vals, extra_sign in [
        ("GOOD_5 + " + str(best_spilled), best_spilled, spilled_vals[best_spilled], SPILLED_SIGNS[best_spilled]),
        ("GOOD_5 + " + str(best_lp), best_lp, all_lp_vals[best_lp], all_lp_signs[best_lp]),
    ]:
        feats6 = dict(base_feats)
        feats6[extra_name] = extra_vals
        signs6 = {**FEATURE_SIGNS, extra_name: extra_sign}
        fused, _ = lsml_continuous_pipeline(feats6, GOOD_5 + [extra_name], signs6)
        anchor = zscore(base_feats["epr"] * FEATURE_SIGNS["epr"])
        fused, _ = anchor_orient(np.asarray(fused, dtype=float), anchor)
        auc, lo, hi = boot_auc(labels, fused)
        rho = float(spearmanr(extra_vals * extra_sign, base_good5_score).statistic)
        delta, dlo, dhi = paired_boot_delta_auc(labels, fused, base_good5_score)
        gate = (abs(rho) < 0.75) and (delta > GATE_PP)
        print(f"\n  fusion test [{label}]: AUROC {auc:.4f} [{lo:.4f},{hi:.4f}]  "
              f"rho {rho:+.3f}  delta-vs-GOOD5 {delta:+.4f} [{dlo:+.4f},{dhi:+.4f}]  "
              f"gate {'PASS' if gate else 'FAIL'}")
        report[f"fusion_{extra_name}"] = {
            "auc": float(auc), "lo": float(lo), "hi": float(hi), "rho": rho,
            "delta_vs_good5": delta, "delta_lo": dlo, "delta_hi": dhi, "gate_pass": bool(gate),
        }
    return report


def f3_fairer_diversity(q1, q2, k3_scores_by_idx, labels_idx_map):
    """B' = {0.6, 1.0, 1.5} vs a matched K=3 same-T Condition A, on the shared index set."""
    common_idx = np.asarray(q2["common_idx"])
    labels = np.asarray(q2["labels"])

    temps = [0.6, 1.0, 1.5]
    views = [zscore(np.asarray(q1[t]["scores"])[common_idx]) for t in temps]
    avg_fused, _ = simple_average_fusion(*views)
    lsml_fused, _ = lsml_continuous(*views)
    lsml_fused, _ = anchor_orient(np.asarray(lsml_fused, dtype=float), avg_fused)

    auc_avg, lo_avg, hi_avg = boot_auc(labels, avg_fused)
    auc_lsml, lo_lsml, hi_lsml = boot_auc(labels, lsml_fused)
    rho = np.array([[spearmanr(views[i], views[j]).statistic for j in range(3)]
                    for i in range(3)])
    off = rho[~np.eye(3, dtype=bool)]
    print(f"  B' = {{0.6, 1.0, 1.5}}: simple-avg AUROC {auc_avg:.4f} [{lo_avg:.4f},{hi_avg:.4f}]  "
          f"L-SML AUROC {auc_lsml:.4f} [{lo_lsml:.4f},{hi_lsml:.4f}]  "
          f"off-diag rho mean {off.mean():+.3f}")

    report = {
        "b_prime_avg": {"auc": float(auc_avg), "lo": float(lo_avg), "hi": float(hi_avg)},
        "b_prime_lsml": {"auc": float(auc_lsml), "lo": float(lo_lsml), "hi": float(hi_lsml)},
        "rho_offdiag_mean": float(off.mean()),
    }

    # Matched comparison: K=3 same-T=1.0 (from F1), on the intersection of index spaces.
    if k3_scores_by_idx is not None:
        # k3_scores_by_idx: {sample_index(int): fused_score} from F1's K=3 pass,
        # labels_idx_map: {sample_index(int): label} from F1 (should match q2 on overlap).
        shared = sorted(set(common_idx.tolist()) & set(k3_scores_by_idx.keys()))
        if len(shared) >= 20:
            y_shared = np.array([labels_idx_map[i] for i in shared])
            a_k3 = np.array([k3_scores_by_idx[i] for i in shared])
            b_common_lookup = {int(ci): lsml_fused[j] for j, ci in enumerate(common_idx)}
            b_bprime = np.array([b_common_lookup[i] for i in shared])
            delta, dlo, dhi = paired_boot_delta_auc(y_shared, b_bprime, a_k3)
            auc_k3, _, _ = boot_auc(y_shared, a_k3)
            print(f"  matched K=3 same-T (n={len(shared)}): AUROC {auc_k3:.4f}  "
                  f"delta(B' L-SML - K=3 same-T) {delta:+.4f} [{dlo:+.4f},{dhi:+.4f}]")
            report["k3_same_t_matched"] = {
                "n": len(shared), "auc_k3_same_t": float(auc_k3),
                "delta_bprime_minus_k3": delta, "delta_lo": dlo, "delta_hi": dhi,
            }
        else:
            print(f"  matched K=3 comparison skipped — only {len(shared)} shared indices")
            report["k3_same_t_matched"] = None
    return report


def f4_cross_temperature_probing(q1, q2):
    """Does a hot pass's fused score predict the COLD (T=1.0 run0) label?"""
    common_idx = np.asarray(q2["common_idx"])
    cold_labels = np.asarray(q2["labels"])
    report = {}
    for t in [0.3, 0.6, 1.5, 2.0]:
        hot_scores_on_cold = np.asarray(q1[t]["scores"])[common_idx]
        auc_cold, lo_c, hi_c = boot_auc(cold_labels, hot_scores_on_cold)
        auc_own = q1[t]["auc"]
        print(f"  T={t}: vs COLD(T=1.0) label AUROC {auc_cold:.4f} [{lo_c:.4f},{hi_c:.4f}]  "
              f"vs OWN label AUROC {auc_own:.4f}")
        report[t] = {"auc_vs_cold_label": float(auc_cold), "lo": float(lo_c), "hi": float(hi_c),
                     "auc_vs_own_label": float(auc_own)}
    return report


def main():
    if not os.path.exists(RESULTS_PKL):
        print(f"WAITING: {RESULTS_PKL} not found — needed for F3/F4 (Step-158 q1/q2 dict).")
        return 1
    with open(RESULTS_PKL, "rb") as f:
        res = pickle.load(f)
    q1, q2 = res["q1"], res["q2"]

    run_paths = _find_run_paths()
    have = [p for p in run_paths if p]
    print(f"raw T=1.0 pass caches found: {len(have)}/{N_PASSES}")
    if len(have) != N_PASSES:
        print("Need all 5 raw per-pass caches for F1/F2 — copy from Drive "
              "cache/phase15_temperature/math500_qwen7b_T1.0_run{0..4}.pkl into local_cache/.")
        return 1

    passes, common = _load_passes(run_paths)
    print(f"common done samples across {N_PASSES} passes: {len(common)}")
    valid_keys, labels, runs_feats = _build_good5_runs_feats(passes, common)
    print(f"valid (>=8 tok in all passes) samples: {len(valid_keys)}")

    print("\n=== F1 — K-sweep for Condition A (same-T=1.0) ===")
    f1 = f1_k_sweep(runs_feats, labels)
    k1_score = np.asarray(
        multipass_lsml_continuous(runs_feats[:1], GOOD_5, FEATURE_SIGNS, anchor_feature="epr")["fused"],
        dtype=float)
    k3_res = multipass_lsml_continuous(runs_feats[:3], GOOD_5, FEATURE_SIGNS, anchor_feature="epr")
    k3_score = np.asarray(k3_res["fused"], dtype=float)
    k3_scores_by_idx = {int(k): float(k3_score[i]) for i, k in enumerate(valid_keys)}
    labels_idx_map = {int(k): int(labels[i]) for i, k in enumerate(valid_keys)}

    print("\n=== F2 — new feature families (spilled energy + extended logprob) ===")
    f2 = f2_new_feature_families(passes, valid_keys, labels, k1_score)

    print("\n=== F3 — fairer diversity set B' = {0.6, 1.0, 1.5} ===")
    f3 = f3_fairer_diversity(q1, q2, k3_scores_by_idx, labels_idx_map)

    print("\n=== F4 — cross-temperature probing ===")
    f4 = f4_cross_temperature_probing(q1, q2)

    print("\n=== BLOCKED — needs an extra Drive pull ===")
    print("  #3 anchor/sign robustness across T: needs raw per-sample GOOD_5 feature values")
    print("     at T != 1.0 (phase15_results.pkl only has scalar per-(feature,temp) AUROCs).")
    print("  #7 length-controlled AUROC per T: needs per-sample trace_length at T != 1.0.")
    print("  Both need: cache/phase15_temperature/math500_qwen7b_T{0.3,0.6,1.5,2.0}_run0.pkl")
    print("  copied into local_cache/ alongside the existing T=1.0 run{0..4} files.")

    out = {
        "n_valid": len(valid_keys), "f1_k_sweep": f1, "f2_new_features": f2,
        "f3_fairer_diversity": f3, "f4_cross_temperature": f4,
        "blocked": ["#3 anchor/sign robustness across T", "#7 length-controlled AUROC per T"],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
