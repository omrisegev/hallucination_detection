"""
our_arm.py — enter our fusion into the Evidence Drop protocol as one more arm.

This is the point of the whole replication: not "can we reproduce their table" but "under
*their* protocol, on *their* metrics, where does our detector land".

WHICH ARM IS THE HEADLINE
-------------------------
**U-PCR over the full 46-view `CANONICAL_POOL`, with polarity derived from `sign(rho-hat)` and
the global sign from the cell's own anchor.** Nothing about it is hand-picked: no curated
feature list, no chosen subset size. That is the thesis direction (Step 199 / prior-free).

The maintained entry point for that arm is `scripts/labelfree_standing_report.py:upcr_rho_oriented`
over cells built by `spectral_utils.subset_sweep.prepare_cell(..., feature_pool=CANONICAL_POOL)`.
It is **NOT** `fusion_utils.upcr_pipeline`, and **NOT** `eval_subset_flex(fusion='upcr')` — those
are the legacy path, and scoring them silently reports the wrong detector (the trap recorded in
`project_pool_composition_closed`; this file walked into it once already and `upcr_arm_fit` is the
repair).

`upcr_arm_fit` reproduces `upcr_rho_oriented` and additionally **retains every fitted parameter**
(`w`, `keep`, `derived`, per-view standardisation, the anchor flip). It must: the token-level
trace has to be pushed through the *same* detector as the step-level table, or the picture and
the numbers are not evidence about the same thing. `assert_upcr_mirrors_canonical` imports the
real `upcr_rho_oriented` and requires bit-for-bit equality.

L-SML on `GOOD_6` stays in the file as **one clearly-labelled reference row** — a hand-picked
subset that exists for comparability with the older tables, never as the result.

ORIENTATION
-----------
The rest of the package orients scores so **higher = more likely correct**. Everything in the
Evidence Drop protocol is a RISK, **higher = more likely hallucinated**. The conversion is a
single negation, done once here and never implicitly.
"""
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_REPO, os.path.join(_REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.repgrid_scoring import (
    ALL_SIGNS, LOGPROB_SIGNS_EXT, load_repgrid_cell, logprob_features_extended,
    score_subset, subset_matrix,
)
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.subset_sweep import ALL_SIGNS as POOL_SIGNS
from spectral_utils.subset_sweep import (
    CANONICAL_POOL, GOOD_5, GOOD_6, H16, is_saturated, prepare_cell,
)
from spectral_utils.upcr import upcr_fit

# TWO SIGN TABLES EXIST AND THEY ARE NOT INTERCHANGEABLE. Both are named `ALL_SIGNS`:
#
#   spectral_utils.subset_sweep.ALL_SIGNS      42 views — FEATURE + EXTRA + REPGRID views
#   spectral_utils.repgrid_scoring.ALL_SIGNS   23 views — no anomaly views, and critically no
#                                              varentropy / renyi_entropy_2 / topk_tail_mass
#
# They agree on every key they share, so the bug they cause is not a contradiction — it is a
# silent `.get(f, +1)` fallback. Orienting `varentropy` as +1 when its sign is -1 does not fail,
# warn, or produce a NaN; it just feeds the fusion three backwards columns. `prepare_cell` and
# `labelfree_standing_report.upcr_rho_oriented` both use the subset_sweep table, so the full-pool
# arm MUST use `POOL_SIGNS` throughout. (Caught by the reconstruction assert in `upcr_arm_fit`,
# which is the reason that assert compares against `prepare_cell`'s own matrix rather than
# trusting a re-derivation.)
#
# `SIGNS` below stays on the repgrid table because the L-SML reference rows go through
# `score_subset`, which lives in that module's world.

# exp06's fitted configuration, imported in spirit and copied verbatim from
# `scripts/labelfree_standing_report.py:66`. This file must not silently re-tune it; the
# mirror gate below fails loudly if it drifts from the canonical scorer.
FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

# `load_repgrid_cell` -> `_candidate_features` calls `logprob_features` (3 views) but NOT
# `logprob_features_extended` (varentropy / renyi_entropy_2 / topk_tail_mass). So `varentropy`
# never appears in a cell loaded that way — which silently costs the full pool three of its 30
# views and makes the GOOD_6 reference row unscoreable (`score_subset` returns NaN and the row
# vanishes from the table).
# (`selectors/reference_macros.py:55` guards for exactly this, which is why the selector bench
# never hit it — it reads a featcache built by `scripts/build_repgrid_featcache.py`, not
# `load_repgrid_cell`.)
#
# `load_cell` below re-adds the extended views on top of the canonical loader rather than
# editing the shared module, per the worktree convention. The signs come from the package's own
# LOGPROB_SIGNS_EXT, never hand-typed.
SIGNS = {**ALL_SIGNS, **LOGPROB_SIGNS_EXT}

# Hand-picked subsets. These are REFERENCE ROWS ONLY — they carry prior knowledge the headline
# arm deliberately does not, so they are reported beside U-PCR and clearly labelled, never as
# the contribution (CLAUDE.md "Which method to evaluate"). Imported, never hand-typed
# (`feedback_read_canonical_scorer_first`).
REFERENCE_SUBSETS = {"GOOD_6": GOOD_6, "GOOD_5": GOOD_5, "H16": H16}

# The reference rows run L-SML only. "U-PCR on a fixed subset" is not a thing this project
# reports: the whole point of the arm is that it takes the full pool.
REFERENCE_METHOD = "lsml"


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


def fused_risk(cell, feat_names, method: str = REFERENCE_METHOD, signs=None, anchor: str = "epr"):
    """(risk, valid_mask) for one hand-picked REFERENCE subset — a mirror of `score_subset`.

    This is the reference-row path, not the headline. For the deployed arm use `upcr_arm_fit`.

    `risk` is defined only on the valid rows (those where every feature in the subset is
    finite), and is oriented **higher = more likely hallucinated**.

    Returns (None, valid) when the subset is not scoreable, using exactly `score_subset`'s
    guards: fewer than 20 valid rows, fewer than 3 features (L-SML is information-free at 3 and
    numerically undetermined at 4 — Step 205), or a single-class label vector.
    """
    if method != REFERENCE_METHOD:
        raise ValueError(
            f"fused_risk only serves the L-SML reference rows; got method={method!r}. "
            "The U-PCR arm is full-pool and goes through `upcr_arm_fit` — routing it through a "
            "fixed subset here would score the legacy path (project_pool_composition_closed).")
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
    score, _ = lsml_continuous_pipeline(feats_dict, feat_names, signs)

    anchor_feat = anchor if anchor in feat_names else feat_names[0]
    anchor_view = zscore(np.asarray(feats_dict[anchor_feat], dtype=float)
                         * signs.get(anchor_feat, +1))
    score, _flipped = anchor_orient(np.asarray(score, dtype=float), anchor_view)

    # score is higher = more likely CORRECT; the protocol wants higher = more RISK.
    return -np.asarray(score, dtype=float), valid


def assert_mirrors_canonical(cell, feat_names, method=REFERENCE_METHOD, anchor="epr", tol=1e-12) -> dict:
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


# ── the headline arm: U-PCR over the full pool ───────────────────────────────

@dataclass
class UpcrArm:
    """One fitted U-PCR detector, with every parameter retained so it can be replayed.

    `upcr_rho_oriented` returns only the score. That is enough for a table, but the
    localization report needs the *same* detector applied to sliding token windows, and
    re-fitting per window would be a different detector wearing the same name. So this holds
    the whole transform chain: impute -> hand sign -> z-score -> recover unoriented ->
    derived polarity -> weights -> anchor flip.

    Shapes are all (p,) over `pool`, in `pool` order.
    """
    pool: list
    hand: np.ndarray          # ALL_SIGNS per view (the hand prior U-PCR undoes)
    impute: np.ndarray        # raw median substituted for non-finite entries at fit time
    mu: np.ndarray            # mean of (raw * hand) at fit time
    sd: np.ndarray            # std  of (raw * hand) at fit time
    derived: np.ndarray       # sign(rho_hat_full) — the polarity U-PCR derives for itself
    w: np.ndarray             # fusion weights over the full pool
    keep: np.ndarray          # bool mask of views surviving Algorithm 1's exclusion step
    flipped: bool             # whether anchor_orient negated the fused score
    anchor_name: str
    score: np.ndarray         # fitted score on the fit population, higher = more likely CORRECT
    n: int
    dropped: dict = field(default_factory=dict)
    n_imputed: int = 0

    @property
    def risk(self) -> np.ndarray:
        """Higher = more likely hallucinated. The single negation, done once."""
        return -self.score

    @property
    def n_kept(self) -> int:
        return int(self.keep.sum())

    def apply(self, fd_new) -> np.ndarray:
        """Replay this exact detector on new rows. `fd_new` maps view name -> raw values.

        Raw means **unoriented and unstandardised** — the same form `prepare_cell` consumes.
        Every parameter comes from the fit; nothing is re-estimated, so a single window and a
        million windows get the same transform. Views missing from `fd_new` are imputed to
        their fit-time median, i.e. they contribute exactly their own mean and cannot move the
        score — the honest behaviour for a view that is unmeasurable at this length.
        """
        n = None
        for f in self.pool:
            if f in fd_new and fd_new[f] is not None:
                n = len(np.atleast_1d(np.asarray(fd_new[f], dtype=float)))
                break
        if n is None:
            raise ValueError("fd_new contains none of the fitted pool's views")

        F = np.empty((len(self.pool), n), dtype=float)
        for j, f in enumerate(self.pool):
            x = np.asarray(fd_new.get(f, np.nan), dtype=float)
            x = np.full(n, float(x)) if x.ndim == 0 else x.astype(float, copy=True)
            if len(x) != n:
                raise ValueError(f"{f}: length {len(x)} != {n}")
            bad = ~np.isfinite(x)
            if bad.any():
                x[bad] = self.impute[j]
            v = (x * self.hand[j] - self.mu[j]) / self.sd[j]   # prepare_cell
            F[j] = v * self.hand[j] * self.derived[j]          # upcr_rho_oriented
        s = self.w @ F
        return -s if self.flipped else s


def _rebuild_columns(fd, pool, n):
    """Replay `prepare_cell`'s per-column construction to recover (impute, mu, sd).

    `prepare_cell` returns the standardised matrix but not the constants it standardised with,
    and the token arm needs them. Rather than trust this reconstruction, `upcr_arm_fit` asserts
    the rebuilt matrix equals `ctx.V` exactly — if the shared function ever changes, this fails
    loudly instead of quietly scoring a differently-normalised detector.
    """
    impute = np.zeros(len(pool))
    mu = np.zeros(len(pool))
    sd = np.zeros(len(pool))
    cols = []
    for j, f in enumerate(pool):
        arr = np.asarray(fd[f], dtype=float).copy()
        bad = ~np.isfinite(arr)
        med = float(np.median(arr[~bad])) if bad.any() else float(np.median(arr))
        impute[j] = med
        if bad.any():
            arr[bad] = med
        v = arr * POOL_SIGNS.get(f, +1)
        mu[j], sd[j] = float(v.mean()), float(v.std())
        cols.append((v - mu[j]) / sd[j])
    return np.column_stack(cols), impute, mu, sd


def upcr_arm_fit(fd, labels=None, domain="localization", cell_key="cell",
                 feature_pool=None, tol=1e-10):
    """Fit the deployed U-PCR arm on one population. Returns a `UpcrArm`, or None.

    `fd` maps view name -> raw per-item values (answers, or pooled steps, or windows). The fit
    is **label-free**: `labels` reaches only `prepare_cell`'s single-class / length guard and
    never the estimator. Pass None when there are no labels at all (e.g. pooled ProcessBench
    steps, where only the *first* erroneous step is annotated) and a placeholder is used for
    that guard alone.

    Returns None when `prepare_cell` refuses the population — fewer than 3 usable views after
    dropping missing / constant / saturated ones, or a degenerate label vector.
    """
    feature_pool = CANONICAL_POOL if feature_pool is None else feature_pool
    n = len(np.asarray(next(iter(fd.values())), dtype=float))
    if labels is None:
        # prepare_cell needs two classes to pass its guard; U-PCR itself never sees this.
        labels = np.arange(n) % 2
    ctx = prepare_cell(domain, cell_key, fd, np.asarray(labels), feature_pool=feature_pool)
    if ctx is None:
        return None

    V_re, impute, mu, sd = _rebuild_columns(fd, ctx.pool, n)
    if not np.allclose(V_re, ctx.V, atol=tol, rtol=0):
        raise AssertionError(
            "column reconstruction drifted from prepare_cell (max |diff| "
            f"{np.abs(V_re - ctx.V).max():.3e}) — the standardisation constants captured here "
            "no longer describe the matrix U-PCR is actually fitted on")

    # ── `upcr_rho_oriented`, verbatim, with the parameters kept ──────────────
    hand = np.array([POOL_SIGNS.get(f, +1) for f in ctx.pool], dtype=float)
    V_un = ctx.V * hand
    derived = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
    derived[derived == 0] = 1.0
    F = (V_un * derived).T
    res = upcr_fit(F, **FIT)
    s, flipped = anchor_orient(res.w @ F, ctx.anchor)

    return UpcrArm(
        pool=list(ctx.pool), hand=hand, impute=impute, mu=mu, sd=sd,
        derived=np.asarray(derived, dtype=float), w=np.asarray(res.w, dtype=float),
        keep=np.asarray(res.keep, dtype=bool), flipped=bool(flipped),
        anchor_name=ctx.anchor_name, score=np.asarray(s, dtype=float), n=n,
        dropped=dict(ctx.dropped), n_imputed=int(ctx.n_imputed),
    )


def upcr_risk_from_cell(cell, feature_pool=None):
    """(UpcrArm, valid_mask) for a `load_cell` dict — the answer-level entry point.

    Unlike the fixed-subset path there is no all-features-finite valid mask: `prepare_cell`
    median-imputes sparse gaps and drops a view outright when it is unusable, so every row is
    scored. The returned mask is all-True and exists only so callers share one signature.
    """
    rows, labels = cell["rows"], cell["labels"]
    pool = CANONICAL_POOL if feature_pool is None else feature_pool
    fd = {f: np.array([r.get(f, np.nan) for r in rows], dtype=float) for f in pool
          if any(f in r for r in rows)}
    arm = upcr_arm_fit(fd, labels=np.asarray(labels, dtype=int))
    return arm, np.ones(len(rows), dtype=bool)


CANONICAL_SCORER = "labelfree_standing_report"

# The gate runs the canonical scorer in a SUBPROCESS rooted in the main working tree, and this
# is not fussiness. Two things make an in-process import impossible:
#   * `scripts/labelfree_standing_report.py` is **untracked** — never committed — so a worktree
#     branched off master does not contain it at all;
#   * it imports `scripts/inscope_cells.py`, which imports `spectral_utils.answer_span`, another
#     uncommitted module. `spectral_utils` is already bound to the worktree's copy by the time
#     the gate runs, and a package's `__path__` points at one directory, so that submodule can
#     never resolve here.
# Copying the function into this file would make both problems disappear and would also be the
# exact failure this gate exists to catch — a local copy that drifts from the deployed arm while
# still passing its own test. So: same inputs, real scorer, its own interpreter, compare arrays.
_GATE_SUBPROCESS = r"""
import os, sys
import numpy as np
root = sys.argv[1]
sys.path[:0] = [root, os.path.join(root, "scripts")]
from labelfree_standing_report import upcr_rho_oriented
d = np.load(sys.argv[2], allow_pickle=True)
s, kept = upcr_rho_oriented({"V": d["V"], "pool": list(d["pool"]), "anchor": d["anchor"]})
np.savez(sys.argv[3], score=np.asarray(s, dtype=float), kept=np.int64(kept))
"""


def _main_tree_root():
    """The main working tree that owns this worktree (where the canonical scorer lives)."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                           cwd=_REPO, capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return os.path.dirname(r.stdout.strip())
    except Exception:
        pass
    return os.path.dirname(os.path.dirname(_REPO))  # .worktrees/<name> -> main tree


def canonical_upcr(V, pool, anchor):
    """Run the maintained `upcr_rho_oriented` on (V, pool, anchor). Returns (score, n_kept).

    Raises if the canonical scorer cannot be found or run — a mirror gate that quietly skips is
    worse than no gate, because it reports a pass.
    """
    import subprocess
    import tempfile

    root = _main_tree_root()
    script = os.path.join(root, "scripts", CANONICAL_SCORER + ".py")
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"canonical scorer not found at {script}. It is the maintained entry point for the "
            "deployed U-PCR arm and this gate exists to prove we match it; without it the arm "
            "cannot be certified, so this is a hard failure rather than a skip.")

    # ignore_cleanup_errors: on Windows an NpzFile keeps the archive open, and any stray
    # handle turns teardown into a PermissionError that masks a passing gate.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        inp, outp = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        run = os.path.join(td, "gate.py")
        np.savez(inp, V=np.asarray(V, dtype=float), pool=np.array(list(pool), dtype=object),
                 anchor=np.asarray(anchor, dtype=float))
        with open(run, "w") as f:
            f.write(_GATE_SUBPROCESS)
        r = subprocess.run([sys.executable, run, root, inp, outp],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(outp):
            raise RuntimeError(
                f"canonical scorer subprocess failed (rc={r.returncode}):\n{r.stderr[-2000:]}")
        with np.load(outp) as d:
            return np.array(d["score"], dtype=float), int(d["kept"])


def assert_upcr_mirrors_canonical(fd, labels, tol=1e-12) -> dict:
    """Gate: `upcr_arm_fit` must equal `labelfree_standing_report.upcr_rho_oriented` exactly.

    The canonical function is RUN, not copied — if the maintained arm changes, this gate breaks
    rather than silently reporting a stale detector. `UpcrArm.apply` is checked on the fit
    population too: replaying the frozen parameters must reproduce the fitted score, which is
    what licenses using them on token windows.
    """
    ctx = prepare_cell("localization", "gate", fd, np.asarray(labels),
                       feature_pool=CANONICAL_POOL)
    if ctx is None:
        raise AssertionError("prepare_cell refused the gate population")
    want, want_kept = canonical_upcr(ctx.V, ctx.pool, ctx.anchor)

    arm = upcr_arm_fit(fd, labels=labels)
    if arm is None:
        raise AssertionError("upcr_arm_fit declined a population the canonical scorer accepted")
    d = float(np.abs(arm.score - want).max())
    if d > tol:
        raise AssertionError(f"U-PCR mirror drift: max |diff| {d:.3e} vs canonical")
    if arm.n_kept != want_kept:
        raise AssertionError(f"kept-view drift: {arm.n_kept} vs canonical {want_kept}")

    d_apply = float(np.abs(arm.apply(fd) - arm.score).max())
    if d_apply > 1e-9:
        raise AssertionError(
            f"UpcrArm.apply does not reproduce its own fit (max |diff| {d_apply:.3e}) — the "
            "frozen parameters do not describe the fitted transform, so the token trace would "
            "be a different detector from the table")
    return {"max_diff": d, "apply_max_diff": d_apply, "n_kept": arm.n_kept,
            "p": len(arm.pool), "anchor": arm.anchor_name}


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
    # 1. The reference subsets come from the canonical definitions, never hand-typed.
    assert GOOD_6 == GOOD_5 + ["varentropy"], GOOD_6
    assert len(H16) == 16, len(H16)
    assert set(REFERENCE_SUBSETS) == {"GOOD_6", "GOOD_5", "H16"}

    # 1b. The legacy trap is closed: the U-PCR arm cannot be reached through the fixed-subset
    #     path, because that path is `fusion_utils.upcr_pipeline` and scoring it would report
    #     the wrong detector (project_pool_composition_closed).
    try:
        fused_risk({"rows": [], "labels": np.array([])}, GOOD_6, method="upcr")
    except ValueError as e:
        assert "full-pool" in str(e), e
    else:
        raise AssertionError("fused_risk must refuse method='upcr'")
    assert "upcr_pipeline" not in globals(), "the legacy U-PCR entry point is still imported"

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

    res = assert_mirrors_canonical(cell, GOOD_6, method=REFERENCE_METHOD)
    assert not res["skipped"] and res["n"] == n, res

    # 3. Risk orientation: the fused risk must be NEGATIVELY correlated with correctness.
    risk, valid = fused_risk(cell, GOOD_6)
    r = float(np.corrcoef(risk, labels[valid].astype(float))[0, 1])
    assert r < 0, f"risk should be higher for incorrect answers, got corr {r:+.3f}"

    # 4. The guards fire exactly as the canonical scorer's do.
    assert fused_risk(cell, GOOD_6[:2])[0] is None, "fewer than 3 features must not score"
    tiny = {"rows": rows[:10], "labels": labels[:10]}
    assert fused_risk(tiny, GOOD_6)[0] is None, "fewer than 20 valid rows must not score"
    one_class = {"rows": rows, "labels": np.ones(n, dtype=bool)}
    assert fused_risk(one_class, GOOD_6)[0] is None, "single-class must not score"

    # 4a'. The two sign tables really do disagree by omission, and the U-PCR path is on the one
    #      `prepare_cell` uses. If this ever stops being true the reconstruction assert will
    #      fire, but naming it here says WHY rather than leaving a numeric mismatch to decode.
    for f in ("varentropy", "renyi_entropy_2", "topk_tail_mass"):
        assert POOL_SIGNS[f] == -1, (f, POOL_SIGNS.get(f))
        assert f not in globals()["ALL_SIGNS"], \
            f"{f} appeared in repgrid's ALL_SIGNS — re-check which table the U-PCR path uses"

    # 4b. THE HEADLINE ARM. A synthetic pool with a planted signal: the fit must equal the
    #     canonical `upcr_rho_oriented` bit-for-bit, `apply` must reproduce its own fit, and the
    #     risk must point the right way.
    #
    #     The pool DELIBERATELY spans the spectral views AND the extended logprob views. An
    #     earlier version took the first 12 views that had a repgrid sign, which silently
    #     excluded exactly the three views whose sign lookup was broken — so the gate passed at
    #     drift 0.0 while the real arm was scoring three columns upside down.
    tail = ["varentropy", "renyi_entropy_2", "topk_tail_mass"]
    views = [f for f in CANONICAL_POOL if f in POOL_SIGNS and f not in tail][:9] + tail
    assert len(views) == 12 and all(v in CANONICAL_POOL for v in views), views
    fd = {}
    for f in views:
        shift = np.where(labels, 1.0, -1.0) * POOL_SIGNS.get(f, 1)
        fd[f] = rng.normal(shift, 1.0, n)
    gate = assert_upcr_mirrors_canonical(fd, labels.astype(int))
    assert gate["p"] == 12 and gate["n_kept"] >= 3, gate

    arm = upcr_arm_fit(fd, labels=labels.astype(int))
    r_up = float(np.corrcoef(arm.risk, labels.astype(float))[0, 1])
    assert r_up < 0, f"U-PCR risk should be higher for incorrect answers, got {r_up:+.3f}"

    # 4c. `apply` on a strict subset of rows must equal the corresponding fitted entries —
    #     the property the token trace depends on (no re-standardisation per window).
    sub = {f: v[:37] for f, v in fd.items()}
    assert np.allclose(arm.apply(sub), arm.score[:37], atol=1e-9), "apply is population-dependent"

    # 4d. A view that is entirely unmeasurable is imputed to its fit-time median, so it
    #     contributes a constant and cannot move the ranking. This is what makes short steps
    #     (missing STFT) degrade the score rather than corrupt it.
    gone = dict(fd)
    gone[views[0]] = np.full(n, np.nan)
    s_gone = arm.apply(gone)
    assert np.isfinite(s_gone).all(), "an all-NaN view must not poison the score"
    assert not np.allclose(s_gone, arm.score), "dropping a view should change the score"

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

    print(f"our_arm.smoke: PASS (10 checks)  "
          f"[U-PCR gate: p={gate['p']} kept={gate['n_kept']} "
          f"anchor={gate['anchor']} drift={gate['max_diff']:.1e}]")


if __name__ == "__main__":
    smoke()
