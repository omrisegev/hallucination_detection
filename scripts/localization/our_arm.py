"""
our_arm.py — enter our L-SML / U-PCR fusion into the Evidence Drop protocol as one more arm.

This is the point of the whole replication: not "can we reproduce their table" but "under
*their* protocol, on *their* metrics, where does our detector land".

WHY THIS FILE EXISTS AT ALL
---------------------------
`spectral_utils.repgrid_scoring.score_subset` is the canonical scorer, and per
`feedback_read_canonical_scorer_first` nothing here re-derives the fusion. But `score_subset`
returns an AUROC and throws the per-sample score away, and selective accuracy / AURC need the
score vector. So `fused_risk` **mirrors `score_subset` line for line** and returns the oriented
score instead of the AUROC — and `assert_mirrors_canonical` proves the mirror is faithful by
recomputing the AUROC from the returned vector and requiring it to match `score_subset` exactly.
If that gate ever fails, this file is wrong and the canonical scorer is right.

ORIENTATION
-----------
The rest of the package orients scores so **higher = more likely correct**. Everything in the
Evidence Drop protocol is a RISK, **higher = more likely hallucinated**. The conversion is a
single negation, done once here and never implicitly.
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from spectral_utils.fusion_utils import (
    zscore, boot_auc, lsml_continuous_pipeline, upcr_pipeline,
)
from spectral_utils.repgrid_scoring import (
    ALL_SIGNS, LOGPROB_SIGNS_EXT, load_repgrid_cell, logprob_features_extended,
    score_subset, subset_matrix,
)
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.subset_sweep import GOOD_5, GOOD_6, H16

# `load_repgrid_cell` -> `_candidate_features` calls `logprob_features` (3 views) but NOT
# `logprob_features_extended` (varentropy / renyi_entropy_2 / topk_tail_mass). So `varentropy`
# never appears in a cell loaded that way — and **GOOD_6, the headline subset per CLAUDE.md, is
# silently unscoreable**: `score_subset` just returns NaN and the row vanishes from the table.
# (`selectors/reference_macros.py:55` guards for exactly this, which is why the selector bench
# never hit it — it reads a featcache built by `scripts/build_repgrid_featcache.py`, not
# `load_repgrid_cell`.)
#
# `load_cell` below re-adds the extended views on top of the canonical loader rather than
# editing the shared module, per the worktree convention. The signs come from the package's own
# LOGPROB_SIGNS_EXT, never hand-typed.
SIGNS = {**ALL_SIGNS, **LOGPROB_SIGNS_EXT}

# The subsets we enter. GOOD_6 is the headline per CLAUDE.md; GOOD_5 is the compatibility
# reference; H16 is the full spectral pool. All three are imported, never hand-typed
# (`feedback_read_canonical_scorer_first`).
SUBSETS = {"GOOD_6": GOOD_6, "GOOD_5": GOOD_5, "H16": H16}

METHODS = ("lsml", "upcr")


def load_cell(pkl_path, label_key: str = "label") -> dict:
    """`load_repgrid_cell` plus the extended logprob views, so GOOD_6 is actually scoreable.

    Everything except the three extra columns comes from the canonical loader untouched.
    """
    import pickle

    cell = load_repgrid_cell(pkl_path, label_key=label_key)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    i = 0
    for idx in sorted(data.keys()):
        for cand in data[idx]["candidates"]:
            tk = cand.get("top_k_logprobs")
            if tk is not None:
                cell["rows"][i].update(logprob_features_extended(tk))
            i += 1
    cell["available"] = sorted({k for r in cell["rows"] for k in r})
    return cell


def fused_risk(cell, feat_names, method: str = "lsml", signs=None, anchor: str = "epr"):
    """(risk, valid_mask) for one feature subset — a faithful mirror of `score_subset`.

    `risk` is defined only on the valid rows (those where every feature in the subset is
    finite), and is oriented **higher = more likely hallucinated**.

    Returns (None, valid) when the subset is not scoreable, using exactly `score_subset`'s
    guards: fewer than 20 valid rows, fewer than 3 features (L-SML is information-free at 3 and
    numerically undetermined at 4 — Step 205), or a single-class label vector.
    """
    signs = SIGNS if signs is None else signs
    rows, labels = cell["rows"], cell["labels"]
    X, valid = subset_matrix(rows, feat_names)
    n_valid = int(valid.sum())
    if n_valid < 20 or len(feat_names) < 3:
        return None, valid
    y = labels[valid]
    if y.sum() == 0 or y.sum() == len(y):
        return None, valid

    feats_dict = {f: X[valid, j] for j, f in enumerate(feat_names)}
    if method == "lsml":
        score, _ = lsml_continuous_pipeline(feats_dict, feat_names, signs)
    else:
        score, *_ = upcr_pipeline(feats_dict, feat_names, signs)

    anchor_feat = anchor if anchor in feat_names else feat_names[0]
    anchor_view = zscore(np.asarray(feats_dict[anchor_feat], dtype=float)
                         * signs.get(anchor_feat, +1))
    score, _flipped = anchor_orient(np.asarray(score, dtype=float), anchor_view)

    # score is higher = more likely CORRECT; the protocol wants higher = more RISK.
    return -np.asarray(score, dtype=float), valid


def assert_mirrors_canonical(cell, feat_names, method="lsml", anchor="epr", tol=1e-12) -> dict:
    """Gate: the AUROC recomputed from `fused_risk` must equal `score_subset`'s, bit-for-bit.

    `boot_auc(labels, scores)` expects scores oriented higher = more likely correct, so the
    risk vector is negated back before comparison. Any drift here means the mirror has diverged
    from the canonical scorer — which is the failure this gate exists to make loud.
    """
    canon = score_subset(cell, feat_names, method=method, signs=SIGNS,
                         anchor=anchor, n_boot=2)
    risk, valid = fused_risk(cell, feat_names, method=method, anchor=anchor)
    if risk is None:
        if np.isfinite(canon["auroc"]):
            raise AssertionError(
                f"fused_risk declined to score {feat_names} but score_subset returned "
                f"{canon['auroc']:.6f} — the guards have diverged")
        return {"skipped": True, "n": canon["n"]}

    auc, _, _ = boot_auc(cell["labels"][valid].astype(int), -risk, n=2)
    if abs(auc - canon["auroc"]) > tol:
        raise AssertionError(
            f"mirror drift on {method}/{feat_names}: fused_risk gives {auc:.12f}, "
            f"score_subset gives {canon['auroc']:.12f}")
    return {"skipped": False, "auroc": float(auc), "n": int(valid.sum())}


# ── step-level features ──────────────────────────────────────────────────────

# ── length-degenerate features ───────────────────────────────────────────────
# `extract_all_features` returns None below 8 tokens, but between 8 and 32 it returns a FULL
# feature dict in which several views are not measurements at all — they are constants that
# `np.isfinite` happily accepts, so `subset_matrix`'s valid mask lets them into the fusion as
# information-free columns. Measured on 30 random steps per length:
#
#   n=8   constant: low_band_power, stft_max_high_power, stft_spectral_entropy
#   n=12  constant: stft_max_high_power, stft_spectral_entropy
#   n=31  constant: stft_max_high_power, stft_spectral_entropy
#   n=40  constant: (none)
#
# Causes, both structural rather than incidental:
#   * `compute_stft_features` has min_len=32 and returns **0.0**, not NaN, below it.
#   * `compute_spectral_features`' low band is `0 < freq <= 0.10`, and the smallest positive
#     rFFT frequency is 1/N, so the band contains NO bins for N < 10 and low_band_power is
#     identically 0. `hl_ratio = high/(low + 1e-12)` then degenerates into high_band_power
#     rescaled by 1e12 — a duplicate column, not a ratio.
#
# These are NaN'd per step so an unmeasurable view is treated as missing, which is what the
# valid mask and the availability report both already know how to handle.
STFT_MIN_LEN = 32
LOW_BAND_MIN_LEN = 10
_STFT_FEATS = ("stft_max_high_power", "stft_spectral_entropy")
_LOW_BAND_FEATS = ("low_band_power", "hl_ratio")


def degenerate_features(n_tokens: int) -> tuple:
    """Feature names that are constants rather than measurements at this trace length."""
    bad = []
    if n_tokens < STFT_MIN_LEN:
        bad.extend(_STFT_FEATS)
    if n_tokens < LOW_BAND_MIN_LEN:
        bad.extend(_LOW_BAND_FEATS)
    return tuple(bad)


def step_feature_rows(row, feat_names=None):
    """Per-step feature dicts for one teacher-forced ProcessBench row.

    Slices every token-aligned series to each step's token range and runs the ordinary
    `extract_all_features` on the slice — so a step is scored exactly like a short trace.

    THE BINDING CONSTRAINT, measured not assumed: `compute_spectral_features` returns None below
    8 tokens and `compute_stft_features` below 32 (`feature_utils.py:33,78`). ProcessBench steps
    are frequently shorter than 32 tokens, so the STFT views are unavailable on most steps and
    the honest step-level pool is smaller than the answer-level one. Callers must report
    availability before reporting any score.
    """
    from spectral_utils.feature_utils import extract_all_features
    from spectral_utils.processbench import slice_series

    ents = row.get("token_entropies")
    spilled = row.get("token_spilled_energies")
    out = []
    for span in row["step_token_spans"]:
        e = slice_series(ents, span)
        if e is None or len(e) < 2:
            out.append({})
            continue
        s = slice_series(spilled, span)
        # Returns None outright when the step is shorter than compute_spectral_features'
        # min_len=8 — an unmeasurable step yields {} rather than a row of zeros.
        feats = extract_all_features(np.asarray(e, dtype=float),
                                     spilled_energies=None if s is None else np.asarray(s, float))
        feats = dict(feats) if feats else {}
        for f in degenerate_features(len(e)):
            if f in feats:
                feats[f] = np.nan
        out.append(feats)
    if feat_names is not None:
        out = [{k: f.get(k, np.nan) for k in feat_names} for f in out]
    return out


def step_feature_availability(rows_of_steps, feat_names) -> dict:
    """Fraction of steps on which each feature is finite. Print this BEFORE any step-level score."""
    avail = {}
    total = sum(len(r) for r in rows_of_steps)
    for f in feat_names:
        ok = sum(1 for r in rows_of_steps for s in r
                 if np.isfinite(s.get(f, np.nan)))
        avail[f] = ok / total if total else 0.0
    return avail


# ── known-answer tests ───────────────────────────────────────────────────────

def smoke() -> None:
    # 1. The subsets come from the canonical definitions, never hand-typed.
    assert GOOD_6 == GOOD_5 + ["varentropy"], GOOD_6
    assert len(H16) == 16, len(H16)
    assert set(SUBSETS) == {"GOOD_6", "GOOD_5", "H16"}

    # 2a. GOOD_6's extra view has a sign in the table we actually pass to the fusion. Without
    #     this, `varentropy` would fall back to +1 and the fusion would orient it backwards.
    assert "varentropy" in SIGNS, "GOOD_6's varentropy has no sign in SIGNS"
    assert SIGNS["varentropy"] == -1, SIGNS["varentropy"]

    # 2. `fused_risk` reproduces `score_subset` exactly on a synthetic cell. Built with a real
    #    signal so the AUROC is not degenerate, and with every GOOD_6 feature present.
    rng = np.random.default_rng(0)
    n = 300
    labels = rng.integers(0, 2, n).astype(bool)
    rows = []
    for i in range(n):
        shift = 1.0 if labels[i] else -1.0
        rows.append({f: float(rng.normal(shift * ALL_SIGNS.get(f, 1), 1.0)) for f in GOOD_6})
    cell = {"rows": rows, "labels": labels}

    for method in METHODS:
        res = assert_mirrors_canonical(cell, GOOD_6, method=method)
        assert not res["skipped"] and res["n"] == n, res

    # 3. Risk orientation: the fused risk must be NEGATIVELY correlated with correctness.
    risk, valid = fused_risk(cell, GOOD_6, method="lsml")
    r = float(np.corrcoef(risk, labels[valid].astype(float))[0, 1])
    assert r < 0, f"risk should be higher for incorrect answers, got corr {r:+.3f}"

    # 4. The guards fire exactly as the canonical scorer's do.
    assert fused_risk(cell, GOOD_6[:2])[0] is None, "fewer than 3 features must not score"
    tiny = {"rows": rows[:10], "labels": labels[:10]}
    assert fused_risk(tiny, GOOD_6)[0] is None, "fewer than 20 valid rows must not score"
    one_class = {"rows": rows, "labels": np.ones(n, dtype=bool)}
    assert fused_risk(one_class, GOOD_6)[0] is None, "single-class must not score"

    # 5. Step features: a long step yields spectral views, an 8-token one does not yield STFT,
    #    and a 1-token step yields nothing at all — the availability story, made concrete.
    row = {
        "token_entropies": list(rng.uniform(0.1, 2.0, 100)),
        "token_spilled_energies": list(rng.uniform(0.1, 2.0, 100)),
        "step_token_spans": [(0, 64), (64, 72), (72, 73)],
    }
    feats = step_feature_rows(row)
    assert np.isfinite(feats[0].get("spectral_entropy", np.nan)), "64-token step should have FFT"
    assert np.isfinite(feats[0].get("stft_spectral_entropy", np.nan)), "64-token step: STFT"
    assert not np.isfinite(feats[1].get("stft_spectral_entropy", np.nan)), \
        "8-token step must NOT report STFT (compute_stft_features returns 0.0, not NaN, below 32)"
    assert not np.isfinite(feats[1].get("low_band_power", np.nan)), \
        "8-token step must NOT report low_band_power (the low band has no rFFT bins below N=10)"
    assert not np.isfinite(feats[1].get("hl_ratio", np.nan)), \
        "8-token step must NOT report hl_ratio (it degenerates to high_band_power * 1e12)"
    assert np.isfinite(feats[1].get("epr", np.nan)), "8-token step SHOULD still have epr"
    assert feats[2] == {}, "1-token step yields no features"

    av = step_feature_availability([feats], ["spectral_entropy", "stft_spectral_entropy", "epr"])
    assert np.isclose(av["spectral_entropy"], 2 / 3), av      # steps 0 and 1
    assert np.isclose(av["stft_spectral_entropy"], 1 / 3), av  # step 0 only
    assert np.isclose(av["epr"], 2 / 3), av

    # 6. `degenerate_features` is a pure function of length, and the boundaries are exact.
    assert degenerate_features(64) == ()
    assert set(degenerate_features(31)) == set(_STFT_FEATS)
    assert set(degenerate_features(9)) == set(_STFT_FEATS) | set(_LOW_BAND_FEATS)
    assert degenerate_features(10) == _STFT_FEATS

    print("our_arm.smoke: PASS (6 checks)")


if __name__ == "__main__":
    smoke()
