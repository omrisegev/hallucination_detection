"""
Offline scoring helpers for the replication grid (local CPU).

Loads a cluster raw pkl -> per-candidate feature rows + labels, so OUR methods
(L-SML continuous, U-PCR) can be scored over feature subsets and placed next to each
paper's PUBLISHED AUROC. This module does NOT reproduce any competitor detector — the
competitor number Y comes from the paper (carried in the cell manifest).

Feature sources per candidate:
  - 20 FEAT_NAMES from extract_all_features(H, spilled_energies=DeltaE) — spectral needs
    a trace >= 8 tokens (returns None below that; those rows lack the spectral features).
  - energy views from token_logsumexp (Z_n, the Step-161 raw full-vocab log-partition),
    when the cell captured logsumexp.
  - logprob views from top_k_logprobs (LOS-Net TDS spirit), when present.

A row keeps whatever features are computable; per-subset scoring then selects the rows
where every feature in that subset is finite (so short-QA cells can still be scored on
their energy/logprob views even when the spectral features are unavailable).
"""
import os
import pickle

import numpy as np

from .feature_utils import extract_all_features, FEAT_NAMES, compute_spilled_energy_features
from .fusion_utils import zscore, boot_auc, lsml_continuous_pipeline, upcr_pipeline
from .streaming_utils import FEATURE_SIGNS, anchor_orient

# ── new-view feature names + fixed offline signs (higher oriented value -> more likely
#    CORRECT). Energy views mirror the spilled convention (higher energy/spread -> more
#    suspect -> sign -1). Logprob views: higher confidence -> more likely correct.
ENERGY_FEATS = ["epr_energy", "min_energy", "sw_var_peak_energy", "cusum_max_energy"]
LOGPROB_FEATS = ["mean_top1_logprob", "logprob_margin", "mean_logprob_entropy"]

ENERGY_SIGNS = {"epr_energy": -1, "min_energy": -1,
                "sw_var_peak_energy": -1, "cusum_max_energy": -1}
LOGPROB_SIGNS = {"mean_top1_logprob": +1, "logprob_margin": +1, "mean_logprob_entropy": -1}

# All fixed signs the scorer knows about (base FEATURE_SIGNS + the new views).
ALL_SIGNS = {**FEATURE_SIGNS, **ENERGY_SIGNS, **LOGPROB_SIGNS}


def energy_features_from_logsumexp(token_logsumexp) -> dict:
    """Features of the raw full-vocab log-partition series Z_n (token_logsumexp), mirroring
    the spilled-energy extractor. Reuses compute_spilled_energy_features on Z_n and renames
    the keys to the *_energy namespace so both can coexist as distinct fusion views."""
    z = np.asarray(token_logsumexp, dtype=float)
    if z.size == 0:
        return {k: 0.0 for k in ENERGY_FEATS}
    s = compute_spilled_energy_features(z)  # epr_spilled/sw_var_peak_spilled/cusum_max_spilled/min_spilled
    return {
        "epr_energy":         s["epr_spilled"],
        "min_energy":         s["min_spilled"],
        "sw_var_peak_energy": s["sw_var_peak_spilled"],
        "cusum_max_energy":   s["cusum_max_spilled"],
    }


def logprob_features(top_k_logprobs) -> dict:
    """LOS-Net-style token-distribution features from the saved top-K logprobs.
    top_k_logprobs = {'ids': int32[T,K], 'logprobs': float32[T,K]} (K>=2)."""
    if not isinstance(top_k_logprobs, dict) or "logprobs" not in top_k_logprobs:
        return {k: np.nan for k in LOGPROB_FEATS}
    lp = np.asarray(top_k_logprobs["logprobs"], dtype=float)  # [T, K], sorted desc
    if lp.ndim != 2 or lp.shape[0] == 0:
        return {k: np.nan for k in LOGPROB_FEATS}
    top1 = lp[:, 0]
    top2 = lp[:, 1] if lp.shape[1] > 1 else lp[:, 0]
    p = np.exp(lp)                                   # top-K probs per token
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)       # entropy over the saved top-K
    return {
        "mean_top1_logprob":   float(np.mean(top1)),
        "logprob_margin":      float(np.mean(top1 - top2)),
        "mean_logprob_entropy": float(np.mean(ent)),
    }


LOGPROB_FEATS_EXT = ["varentropy", "renyi_entropy_2", "topk_tail_mass"]
LOGPROB_SIGNS_EXT = {"varentropy": -1, "renyi_entropy_2": -1, "topk_tail_mass": -1}


def logprob_features_extended(top_k_logprobs, tail_k: int = 5) -> dict:
    """Second-order token-distribution features from the saved top-K logprobs, computed
    over the same restricted top-K support as logprob_features() (so all values are a lower
    bound / approximation of the true full-vocab quantity when K < vocab size).

    top_k_logprobs = {'ids': int32[T,K], 'logprobs': float32[T,K]} (K>=2).

    varentropy:        mean per-token variance of surprisal -log p over the top-K support
                        (Kadavath et al. 2022 "varentropy" — dispersion of information content,
                        distinct from the mean surprisal itself).
    renyi_entropy_2:    mean per-token order-2 (collision) Renyi entropy, -log(sum(p^2)),
                        computed on the renormalized top-K distribution.
    topk_tail_mass:     mean per-token probability mass outside the top-`tail_k` (0 if
                        tail_k >= K) — a concentration proxy: near 0 = peaked, larger = flat.
    """
    if not isinstance(top_k_logprobs, dict) or "logprobs" not in top_k_logprobs:
        return {k: np.nan for k in LOGPROB_FEATS_EXT}
    lp = np.asarray(top_k_logprobs["logprobs"], dtype=float)  # [T, K], sorted desc
    if lp.ndim != 2 or lp.shape[0] == 0:
        return {k: np.nan for k in LOGPROB_FEATS_EXT}
    p = np.exp(lp)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)            # renormalized top-K probs

    surprisal = -np.log(p + 1e-12)                             # [T, K]
    mean_surprisal = (p * surprisal).sum(axis=1, keepdims=True)
    varentropy = (p * (surprisal - mean_surprisal) ** 2).sum(axis=1)

    renyi2 = -np.log((p ** 2).sum(axis=1) + 1e-12)

    k = min(tail_k, lp.shape[1])
    tail_mass = 1.0 - p[:, :k].sum(axis=1)
    tail_mass = np.clip(tail_mass, 0.0, 1.0)

    return {
        "varentropy":      float(np.mean(varentropy)),
        "renyi_entropy_2": float(np.mean(renyi2)),
        "topk_tail_mass":  float(np.mean(tail_mass)),
    }


def _candidate_features(cand: dict) -> dict:
    """All computable features for one candidate (missing ones absent/NaN)."""
    feats = extract_all_features(cand.get("token_entropies", []),
                                 spilled_energies=cand.get("token_spilled_energies"))
    out = dict(feats) if feats is not None else {}   # 20 spectral+spilled, or {} if trace<8
    if cand.get("token_logsumexp") is not None:
        out.update(energy_features_from_logsumexp(cand["token_logsumexp"]))
    if cand.get("top_k_logprobs") is not None:
        out.update(logprob_features(cand["top_k_logprobs"]))
    return out


def load_repgrid_cell(pkl_path, label_key="label"):
    """
    Load a cluster raw pkl -> per-candidate feature rows.

    Returns dict:
        rows:        list of {feature_name: value} (one per candidate; missing features absent)
        labels:      np.ndarray[bool]  (judge labels by default; label_key selects the field)
        labels_lex:  np.ndarray[bool]  (lexical grader label, when present, else == labels)
        problem_id:  np.ndarray[int]   (source problem index, for per-problem aggregation)
        n_problems:  int
        available:   sorted list of feature names present on >=1 row
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    rows, labels, labels_lex, pid = [], [], [], []
    for idx in sorted(data.keys()):
        for c in data[idx]["candidates"]:
            rows.append(_candidate_features(c))
            labels.append(bool(c.get(label_key, c.get("label", False))))
            labels_lex.append(bool(c.get("label_lexical", c.get("label", False))))
            pid.append(int(idx))
    avail = sorted({k for r in rows for k in r})
    return {
        "rows": rows,
        "labels": np.asarray(labels, dtype=bool),
        "labels_lex": np.asarray(labels_lex, dtype=bool),
        "problem_id": np.asarray(pid, dtype=int),
        "n_problems": len(data),
        "available": avail,
    }


def subset_matrix(rows, feat_names):
    """Rows x len(feat_names) matrix + valid mask (rows where every feature is finite)."""
    n, m = len(rows), len(feat_names)
    X = np.full((n, m), np.nan)
    for i, r in enumerate(rows):
        for j, f in enumerate(feat_names):
            v = r.get(f, np.nan)
            X[i, j] = v if v is not None else np.nan
    valid = np.isfinite(X).all(axis=1)
    return X, valid


def score_subset(cell, feat_names, method="lsml", signs=ALL_SIGNS,
                 anchor="epr", n_boot=1000):
    """
    Score OUR method on one feature subset of a loaded cell.

    method: 'lsml' (lsml_continuous_pipeline) or 'upcr' (upcr_pipeline).
    Global sign resolved label-free by anchor_orient against the oriented `anchor` view
    (falls back to the subset's first feature if the anchor isn't in the subset). AUROC is
    raw (never max(a,1-a)) with a 95% bootstrap CI.

    Returns dict: auroc, lo, hi, n, valid_rate, method, flipped, k_or_none.
    """
    rows, labels = cell["rows"], cell["labels"]
    X, valid = subset_matrix(rows, feat_names)
    n_valid = int(valid.sum())
    out = {"method": method, "n": n_valid, "valid_rate": n_valid / max(len(rows), 1),
           "auroc": np.nan, "lo": np.nan, "hi": np.nan, "flipped": None}
    if n_valid < 20 or len(feat_names) < 3:
        return out
    y = labels[valid]
    if y.sum() == 0 or y.sum() == len(y):
        return out  # single-class -> AUROC undefined

    feats_dict = {f: X[valid, j] for j, f in enumerate(feat_names)}
    if method == "lsml":
        score, _ = lsml_continuous_pipeline(feats_dict, feat_names, signs)
    else:
        score, *_ = upcr_pipeline(feats_dict, feat_names, signs)

    anchor_feat = anchor if anchor in feat_names else feat_names[0]
    anchor_view = zscore(np.asarray(feats_dict[anchor_feat], dtype=float)
                         * signs.get(anchor_feat, +1))
    score, flipped = anchor_orient(np.asarray(score, dtype=float), anchor_view)
    auc, lo, hi = boot_auc(y.astype(int), score, n=n_boot)
    out.update({"auroc": float(auc), "lo": float(lo), "hi": float(hi), "flipped": bool(flipped)})
    return out
