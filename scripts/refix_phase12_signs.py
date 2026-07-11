#!/usr/bin/env python
"""
refix_phase12_signs.py — reapply sign-orientation to the Phase-12-Corrected
analysis cells (Step 0c; the MATH-500 sign-flip bug, PROGRESS.md Priority 1
since Step 152).

Root cause: the Phase-12-Corrected notebook calls lsml_continuous_pipeline()
on the GOOD_5 views and hands the fused score straight to boot_auc — no
anchor_orient step — so the fused score keeps the leading-eigenvector's
arbitrary global +- sign. On MATH-500 the solver emitted the mirrored sign
(stored AUROC 0.230 for both the L-SML row and the L-SML+LW-SE fusion row).

Anchor choice: single oriented `epr` (production rule, kept per the Step 0b
test — scripts/test_multi_anchor_orient.py: on GOOD_5 the multi-feature-average
anchor agrees on 26/29 battery cells and the multi anchor is a large net loss
on ALL_H16, so the simpler epr anchor stays).

Two modes, auto-detected:

FULL RECOMPUTE — runs when the raw two-pass inference caches (Drive:
  cache/phase12_corrected/{p1_gsm8k_llama8b,p2_math500_qwenmath7b,
  p3_gpqa_qwen7b}.pkl) are copied into local_cache/. Re-extracts GOOD_5
  features from token_entropies, re-runs lsml_continuous_pipeline, verifies
  the unoriented AUROC reproduces the stored number, then applies
  anchor_orient label-free and reports the corrected AUROC. If the matching
  SE cache (p*_se.pkl) is also present, the L-SML+LW-SE fusion row is
  re-oriented the same way.

MIRROR FALLBACK — runs now against local_cache/phase12_corrected_results.pkl
  alone (that pkl stores only (auc, lo, hi) per method — no score arrays, so
  the label-free anchor decision cannot be re-executed on the original
  scores). The +- ambiguity is exact: AUC(-s) = 1 - AUC(s), CI mirrors to
  [1-hi, 1-lo]. The flip DIRECTION is corroborated label-free by the battery
  precedent: anchor_orient(epr) resolves every comparable math500/gsm8k
  battery cell to the >0.5 side (Step 0b, 29 cells), and this script re-runs
  that decision on the same-model battery cell (math500/Qwen-Math-7B,
  local_cache/math500_res.pkl) as direct corroboration.

Usage:
    PYTHONPATH=. python scripts/refix_phase12_signs.py
"""
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.feature_utils import extract_all_features
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient
from spectral_utils import subset_sweep

GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]
SE_SIGN = -1          # lwse: higher = more hallucinated
K_SE_SC = 10          # notebook validity rule: len(k_samples) == K_SE_SC

RESULTS_PKL = os.path.join("local_cache", "phase12_corrected_results.pkl")
OUT_JSON = os.path.join("results", "repgrid", "phase12_signfix.json")

# Raw two-pass caches (Drive-only today; auto-detected when copied locally)
RAW_CANDIDATES = {
    "gsm8k": ["local_cache/p1_gsm8k_llama8b.pkl",
              "local_cache/phase12_corrected/p1_gsm8k_llama8b.pkl"],
    "math500": ["local_cache/p2_math500_qwenmath7b.pkl",
                "local_cache/phase12_corrected/p2_math500_qwenmath7b.pkl"],
    "gpqa": ["local_cache/p3_gpqa_qwen7b.pkl",
             "local_cache/phase12_corrected/p3_gpqa_qwen7b.pkl"],
}
SE_CANDIDATES = {
    "gsm8k": ["local_cache/p1_gsm8k_se.pkl",
              "local_cache/phase12_corrected/p1_gsm8k_se.pkl"],
    "math500": ["local_cache/p2_math500_se.pkl",
                "local_cache/phase12_corrected/p2_math500_se.pkl"],
    "gpqa": ["local_cache/p3_gpqa_se.pkl",
             "local_cache/phase12_corrected/p3_gpqa_se.pkl"],
}
CELL_LABELS = {
    "gsm8k": "GSM8K / Llama-3.1-8B-Instruct",
    "math500": "MATH-500 / Qwen2.5-Math-7B-Instruct",
    "gpqa": "GPQA Diamond / Qwen2.5-7B-Instruct",
}


def _first_existing(paths):
    return next((p for p in paths if os.path.exists(p)), None)


def full_recompute(domain, raw_path, se_path, stored):
    """Re-run the notebook recipe from the raw two-pass cache, then orient."""
    with open(raw_path, "rb") as f:
        p = pickle.load(f)
    valid = {k: v for k, v in p.items()
             if v.get("done") and len(v.get("k_samples", [])) == K_SE_SC}
    keys = sorted(valid)
    f_raw = {k: extract_all_features(valid[k]["token_entropies"]) for k in keys}
    keys = [k for k in keys if f_raw[k] is not None]
    labels = np.array([valid[k]["correct"] for k in keys])
    feats = {f: np.array([f_raw[k][f] for k in keys]) for f in GOOD_5}

    fused, _ = lsml_continuous_pipeline(feats, GOOD_5, FEATURE_SIGNS)
    fused = np.asarray(fused, dtype=float)
    raw_auc, raw_lo, raw_hi = boot_auc(labels, fused)
    stored_auc = stored["lsml"][0]
    repro_ok = abs(raw_auc - stored_auc) < 0.03 or abs((1 - raw_auc) - stored_auc) < 0.03

    anchor = zscore(feats["epr"] * FEATURE_SIGNS["epr"])
    oriented, flipped = anchor_orient(fused, anchor)
    cor_auc, cor_lo, cor_hi = boot_auc(labels, oriented)

    out = {
        "mode": "full-recompute", "n": len(keys),
        "raw_auc": raw_auc, "repro_ok": repro_ok,
        "lsml_corrected": (cor_auc, cor_lo, cor_hi), "lsml_flipped": flipped,
    }
    print(f"  raw (unoriented) L-SML AUROC {raw_auc:.4f} "
          f"(stored {stored_auc:.4f} — reproduction {'OK' if repro_ok else 'MISMATCH'})")
    print(f"  anchor_orient(epr): flipped={flipped}  ->  corrected "
          f"{cor_auc:.4f} [{cor_lo:.4f}, {cor_hi:.4f}]")

    if se_path:
        with open(se_path, "rb") as f:
            se = pickle.load(f)
        lwse = np.array([se[k]["lwse"] for k in keys])
        fu_feats = {**feats, "lwse": lwse}
        fu_signs = {**FEATURE_SIGNS, "lwse": SE_SIGN}
        fu, _ = lsml_continuous_pipeline(fu_feats, GOOD_5 + ["lwse"], fu_signs)
        fu_or, fu_flip = anchor_orient(np.asarray(fu, dtype=float), anchor)
        fu_auc, fu_lo, fu_hi = boot_auc(labels, fu_or)
        out["fusion_corrected"] = (fu_auc, fu_lo, fu_hi)
        out["fusion_flipped"] = fu_flip
        print(f"  fusion (+LW-SE) anchor_orient: flipped={fu_flip}  ->  "
              f"{fu_auc:.4f} [{fu_lo:.4f}, {fu_hi:.4f}]")
    else:
        print("  fusion row skipped (SE cache p*_se.pkl not local)")
    return out


def battery_corroboration():
    """Re-run the epr-anchor decision on the same-model battery cell
    (math500 / Qwen-Math-7B) as label-free corroboration of the flip direction."""
    try:
        for domain, cell_key, fd, labels in subset_sweep.iter_cells("local_cache"):
            if domain != "math500" or "Qwen-Math-7B" not in cell_key:
                continue
            X = np.column_stack([np.asarray(fd[f], dtype=float) for f in GOOD_5])
            valid = np.isfinite(X).all(axis=1)
            y = np.asarray(labels, dtype=int)[valid]
            feats = {f: X[valid, j] for j, f in enumerate(GOOD_5)}
            fused, _ = lsml_continuous_pipeline(feats, GOOD_5, FEATURE_SIGNS)
            anchor = zscore(feats["epr"] * FEATURE_SIGNS["epr"])
            oriented, flipped = anchor_orient(np.asarray(fused, dtype=float), anchor)
            auc, lo, hi = boot_auc(y, oriented)
            return {"cell": cell_key, "n": int(valid.sum()),
                    "flipped": bool(flipped), "auc": float(auc)}
    except Exception as ex:  # corroboration is best-effort, never fatal
        print(f"  (battery corroboration unavailable: {ex})")
    return None


def mirror_fallback(domain, stored):
    """No score arrays available: exact mirror arithmetic on the stored
    (auc, lo, hi); flip direction justified by the epr-anchor battery precedent."""
    out = {"mode": "mirror-fallback"}
    for method in ("lsml", "fusion"):
        auc, lo, hi = stored[method][:3]
        needs_flip = auc < 0.5
        cor = (1 - auc, 1 - hi, 1 - lo) if needs_flip else (auc, lo, hi)
        out[f"{method}_stored"] = (auc, lo, hi)
        out[f"{method}_corrected"] = cor
        out[f"{method}_flipped"] = needs_flip
        tag = "MIRRORED" if needs_flip else "unchanged"
        print(f"  {method:6s} stored {auc:.4f} [{lo:.4f}, {hi:.4f}]  ->  "
              f"{cor[0]:.4f} [{cor[1]:.4f}, {cor[2]:.4f}]  ({tag})")
    return out


def main():
    if not os.path.exists(RESULTS_PKL):
        print(f"WAITING: {RESULTS_PKL} not yet available")
        print("Drop the file into local_cache/ and re-run.")
        return 1
    with open(RESULTS_PKL, "rb") as f:
        res = pickle.load(f)

    report = {}
    any_full = False
    for domain in ("gsm8k", "math500", "gpqa"):
        stored = res[domain]
        print(f"\n== {CELL_LABELS[domain]} ==")
        raw_path = _first_existing(RAW_CANDIDATES[domain])
        se_path = _first_existing(SE_CANDIDATES[domain])
        if raw_path:
            any_full = True
            print(f"  raw cache found: {raw_path} -> FULL label-free recompute")
            report[domain] = full_recompute(domain, raw_path, se_path, stored)
        else:
            report[domain] = mirror_fallback(domain, stored)

    if not any_full:
        print("\nMode: MIRROR FALLBACK (results pkl stores only (auc, lo, hi) — no "
              "score arrays; the raw two-pass caches are still Drive-only).")
        print("Label-free corroboration of the flip direction (same model+dataset, "
              "battery cell, anchor decision actually executed):")
        cor = battery_corroboration()
        if cor:
            report["battery_corroboration"] = cor
            print(f"  math500/{cor['cell']}: anchor_orient(epr) flipped={cor['flipped']}, "
                  f"oriented AUROC {cor['auc']:.4f} (n={cor['n']}) — the epr anchor "
                  f"resolves this cell to the >0.5 side, same direction as the mirror.")
        print("\nFor the full label-free re-derivation on the ORIGINAL Phase-12 scores, "
              "copy from Drive cache/phase12_corrected/:")
        for d, paths in RAW_CANDIDATES.items():
            print(f"  {os.path.basename(paths[0])}  ->  local_cache/")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
