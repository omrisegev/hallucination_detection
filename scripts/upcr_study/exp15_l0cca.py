"""Can a CROSS-CHANNEL criterion reach the room that feature selection has?

WHY THIS EXISTS
    exp13 priced the strongest label-BASED marginal statistic in U-PCR's feature-
    selection channel: the true correlation with correctness IDENTIFIES the good
    features and BUYS NOTHING (+0.08pp over the matched floor, p = 0.62). exp14 then
    priced the label-free menu and found none of six arms clears that floor, with the
    arm that most identifies the good features the WORST performer. Between them, the
    per-feature ranking family is closed: the room is not reachable by scoring features
    one at a time against correctness.

    Every one of those closures is about `rho_hat` — the correlation with correctness.
    l0-CCA's criterion is a DIFFERENT quantity: shared structure between two measurement
    channels. Its gates enter the loss JOINTLY with the canonical directions, so a
    feature's gate depends on the others, which no marginal statistic can express. That
    is why exp13's floor does not bound it and why it is the pre-registered follow-on.

        Lindenbaum, Salhov, AVERBUCH, Kluger, "l0-based Sparse Canonical Correlation
        Analysis", arXiv:2010.05620v2. Digest:
        papers/digests/l0-based-sparse-canonical-correlation-analysis.md
        Implementation: spectral_utils/selectors/l0cca.py

WHAT IS PRE-REGISTERED, WRITTEN BEFORE THE RUN

    THE CHANNEL SPLIT is fixed by the pool's own construction history, so it carries no
    data-dependent choice:
        X = the 16 entropy-trace spectral views (`epr` .. `cusum_shift_idx`)
        Y = the spilled-energy + raw-energy + token-logprob views (up to 14)
    Verified on the loaded pool: X is 5-16 and Y is 13-14 across the 24 cells, and
    N >> D on every half, so the closed-form CCA is well posed everywhere.

    TWO LABEL-FREE ARMS, both scored in ONE pass. The plan sequenced them (probe first,
    trained gates only if the probe justifies it); they are scored together because a
    second arm pre-registered AFTER seeing the first one's number is not pre-registered
    — the same reasoning exp14 applied to the DUFS arm. The probe is still reported
    first and on its own.

      cca_leverage   |leading left/right singular vectors| of the half-A cross-covariance
                     C_xy, concatenated over the two channels. HIGHER = keep. This is the
                     paper's S3.2 gate initialization with the percentile thresholding
                     removed, so it has NO free parameter. One SVD, no training.
                     It is a PROBE OF THE CHANNEL, NOT A CEILING: it is only an
                     initializer in the paper, so a weak result here is evidence, not a
                     bound. That limit must travel with the number wherever it is quoted.
      cca_gates      the trained l0-CCA gate means `mu`, concatenated over the two
                     channels, seed-averaged over 5 stability seeds. HIGHER = keep,
                     SIGNED, for the same reason as DUFS: gates start at MU_INIT = 0.5
                     and Adam drives rejected ones negative, so |mu| would promote the
                     most strongly REJECTED features.
                     lambda_x = lambda_y = lambda is chosen LABEL-FREE by the paper's own
                     criterion — maximize total correlation on held-out rows of half A
                     (inner 75/25 split) — over the fixed grid LAMBDA_GRID. The
                     regularization path is reported, not absorbed: Fig. 2 of the paper
                     claims lambda-robustness, and if OUR support is unstable across
                     lambda that is a finding about our data.

    Concatenating the two channels' mu is legitimate because lambda_x = lambda_y and both
    channels' gates start at 0.5 under one shared loss, so the scales are common by
    construction. Because that is still an assumption, a channel ROUND-ROBIN variant of
    each arm — invariant to any offset between the two channels — is reported as
    sensitivity, outside the primary table and outside the multiplicity family.

    CONTROLS, unchanged and asserted bit-identical to exp13:
      rho        U-PCR's own rho_hat_full — KNOWN to be BELOW chance.
      truecorr   the true correlation with correctness — uses labels; the ceiling of the
                 whole marginal family, and exp13 puts that ceiling ON the floor.

    DECISION RULE, fixed in advance:
      PRIMARY   held-out AUROC of the pruned set, paired over the 24 test sets, against
                the MATCHED PRUNING FLOOR (random order, same construction, same k).
                Positioned against the room, +2.25pp.
      SECONDARY overlap with the good set against a null matched on the arm's OWN
                inside/outside-keep-set composition.
      MULTIPLICITY  Holm-Bonferroni over the TWO label-free arms on the primary endpoint.
                The controls are excluded; their outcomes are known.
      READING   exp14 established that clearing only the overlap test reproduces exp13's
                outcome rather than beating it, AND that across its six label-free arms
                |overlap excess| vs performance had Spearman -0.71 — the nearer an arm
                was to random, the better it scored against a floor that IS random. So an
                arm here must beat the floor WHILE separating from the null. Beating the
                floor alone, with overlap at the null, is the exp14 pattern, not an
                escape from it.
      CARRIED   the room's denominator is not construction-matched: the good set lives
                81.3% inside the deployed keep set while every pruning arm is confined to
                it. About a fifth of the target is unreachable by a pruning arm at all,
                so "% of the room" is reported with that caveat attached.

    KNOWN LIMITS, declared before the run rather than discovered after:
      * Algorithm 1's readout is still "return the s features with largest mu" — a
        per-feature ranking. The escape from the shape exp14 closed is in how the
        statistic is COMPUTED (jointly, set-level), not in how it is READ OUT. If this
        arm fails, that distinction is what the failure is about.
      * Our regime is N >> D; the paper's motivating case is N << D. The degeneracy
        argument for sparsity does not transfer — only nuisance removal does.
      * The paper's Appendix E lists small-batch training as a known weakness. We train
        full batch, so this does not bite, but several cells are small (n = 198 at the
        smallest, i.e. ~99 rows per half against 30 features).

REPRODUCTION
    The splits are exp12's, replayed through exp13's machinery, exactly as exp14 does:
    same per-cell crc32 seed, same replayed consumption, same `nrng` draws in the same
    order. All new randomness is on generators exp13 and exp14 never touch (`trng` for
    tie-breaks, `orng` for the new nulls, `crng` for the CCA row subsample), so the
    published arms are invariant to what is added here by construction. The run asserts,
    per split and to 1e-9, that the deployed AUROC, the pruning floor, both control arms
    and exp13's own rebuilt-set conditional nulls match its splits.csv.
"""
import argparse
import csv
import os
import sys
import zlib

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from exp10_channel_ceilings import (                                # noqa: E402
    FIT, N_SPLITS, derive_cell, derived_arm_gate, fit_cols, _zsc, _all_fields,
)
from exp13_incumbent_anchored_ranking import (                      # noqa: E402
    N_RANDOM, N_OVERLAP_DRAWS, TOL_REPRO,
    true_corr, prune_incumbent, by_stat, cond_overlap_null, load_exp12_splits,
)
from exp14_ranker_menu import (                                     # noqa: E402
    by_score, collapse, paired, holm, load_exp13_splits, _check, ROOM_PP, FLOOR_PP,
)
from spectral_utils.subset_sweep import ALL_SIGNS                   # noqa: E402
from spectral_utils.upcr import upcr_fit                            # noqa: E402
from spectral_utils.feature_utils import FEAT_NAMES                 # noqa: E402
from spectral_utils.selectors.l0cca import (                        # noqa: E402
    cca_leverage_scores, cca_init_mu, l0cca_gates, train_l0cca,
    heldout_total_correlation, N_SEEDS_STABILITY, R_MAX, _prep,
)

OUT_DIR = "15_l0cca"

# The pre-registered channel split. X is the entropy-trace spectral block, frozen as the
# first 16 entries of FEAT_NAMES since Step 154; Y is everything else in the pool, i.e.
# the spilled / raw-energy / token-logprob views appended later.
X_VIEWS = set(list(FEAT_NAMES)[:16])

LAMBDA_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
INNER_VAL_FRAC = 0.25          # inner split of half A, for the label-free lambda choice
MIN_CHANNEL = 3                # a channel below this cannot support a CCA; arm -> NaN

LABEL_FREE = ("cca_leverage", "cca_gates")
CONTROLS = ("rho", "truecorr")
ARMS = LABEL_FREE + CONTROLS
# Outside the primary table and outside the multiplicity family. The round-robin pair
# tests the concatenation assumption; `cca_init_r50` is the paper's literal thresholded
# initialization; the lambda arms ARE the regularization path, scored.
SENSITIVITY = (("cca_leverage_rr", "cca_gates_rr", "cca_init_r50", "chan_rr_random")
               + tuple(f"cca_gates_lam{i}" for i in range(len(LAMBDA_GRID))))

# `chan_rr_random` is the round-robin arms' OWN floor and it is not optional. The
# structural dry run (`--no-cca`, every CCA score NaN) scored the two round-robin arms at
# +0.32pp, p = 0.019 against the pruning floor with no CCA signal in them at all: taking
# one feature per channel in rotation is a channel-BALANCE prior that pays by itself,
# because the good sets are 51% spectral while the marginal rankings pick 32-34%. So a
# round-robin arm must be read against THIS arm — same rotation, random order within each
# channel — and not against the pruning floor, or the balance prior is silently credited
# to l0-CCA. Found in the dry run, before any real number existed.


def channel_round_robin(is_x, score, tiebreak):
    """One feature per CHANNEL in rotation, within a channel by score descending.

    Invariant to any additive or multiplicative offset between the two channels' scores,
    which is exactly the assumption the concatenated arms make. Non-finite scores are
    mapped to -inf rather than left as NaN: tuple comparison short-circuits on
    `nan != nan`, which would silently restore pool-index order (exp14's finding).
    """
    s = np.asarray(score, dtype=float)
    s = np.where(np.isfinite(s), s, -np.inf)
    lanes = []
    for flag in (True, False):
        idx = [int(j) for j in np.where(np.asarray(is_x) == flag)[0]]
        lanes.append(sorted(idx, key=lambda j: (-s[j], float(tiebreak[j]))))
    lanes.sort(key=lambda L: (-s[L[0]], float(tiebreak[L[0]])) if L else (np.inf, 0.0))
    out, pos = [], 0
    while len(out) < len(s):
        for L in lanes:
            if pos < len(L):
                out.append(L[pos])
        pos += 1
    return out


def pick_lambda(Xa, Ya, crng):
    """Choose lambda label-free, by the paper's own criterion.

    Inner 75/25 split of half A; for each lambda train ONE seed, then fit the canonical
    directions on the inner-train rows and measure total correlation on the inner-val
    rows. Returns (lambda, path), where `path` carries the held-out correlation, the open
    -gate count and the seed-0 gates at every lambda so the regularization path can be
    reported rather than absorbed by the choice.

    One seed, not five: this is model selection, and spending the stability budget here
    instead of on the reported gates would buy a better lambda with a noisier estimate.
    """
    n = Xa.shape[0]
    perm = crng.permutation(n)
    n_val = max(int(round(n * INNER_VAL_FRAC)), 2)
    va, tr = perm[:n_val], perm[n_val:]
    path = []
    for lam in LAMBDA_GRID:
        mx, my = train_l0cca(Xa[tr], Ya[tr], lam=lam, seed=0)
        zx, zy = np.clip(mx, 0, 1), np.clip(my, 0, 1)
        hc = heldout_total_correlation(Xa[tr], Ya[tr], Xa[va], Ya[va], zx, zy)
        path.append({"lam": lam, "heldout_corr": hc,
                     "n_open": int((mx > 0).sum() + (my > 0).sum()),
                     "mu": (mx, my)})
    best = max(path, key=lambda r: (r["heldout_corr"], -r["lam"]))
    return best["lam"], path


def run_cell(ck, cell, dcell, rng, nrng, trng, orng, crng, prev12, prev13, args):
    V0 = cell["V"]
    y = dcell["labels"]
    n, m = V0.shape
    hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
    pool = cell["pool"]
    is_x = np.array([f in X_VIEWS for f in pool], dtype=bool)
    xi = [int(j) for j in np.where(is_x)[0]]
    yi = [int(j) for j in np.where(~is_x)[0]]
    ok_split = len(xi) >= MIN_CHANNEL and len(yi) >= MIN_CHANNEL

    split_rows = []
    for rep in range(N_SPLITS):
        # --- exp12's split, reproduced exactly (identical to exp13 and exp14) --------
        idx = rng.permutation(n)
        a_idx, b_idx = idx[: n // 2], idx[n // 2:]
        if len(np.unique(y[a_idx])) < 2 or len(np.unique(y[b_idx])) < 2:
            continue
        raw_a = np.column_stack([_zsc(V0[a_idx, j]) for j in range(m)]) * hand
        raw_b = np.column_stack([_zsc(V0[b_idx, j]) for j in range(m)]) * hand
        try:
            pol = np.sign(upcr_fit(raw_a.T, **FIT).rho_hat_full)
        except Exception:
            continue
        pol[pol == 0] = 1.0
        ca = {"V": raw_a * pol, "anchor": _zsc(dcell["anchor"][a_idx]), "labels": y[a_idx]}
        cb = {"V": raw_b * pol, "anchor": _zsc(dcell["anchor"][b_idx]), "labels": y[b_idx]}

        _, res_a = fit_cols(ca, range(m))
        if res_a is None:
            continue
        start_a = [int(j) for j in np.where(res_a.keep)[0]]

        ref12, ref13 = prev12.get((ck, rep)), prev13.get((ck, rep))
        if ref12 is None or ref13 is None:
            raise RuntimeError(f"{ck} rep {rep}: no exp12/exp13 row — the splits diverged")
        k = int(ref12["k"])
        cols_greedy = [pool.index(f) for f in ref12["greedy_cols"].split("|")]

        auc_dep = fit_cols(cb, range(m))[0]
        _check(ck, rep, "auroc_deployed", auc_dep, float(ref13["auroc_deployed"]))

        Va = ca["V"]
        tb = trng.random(m)                       # one tie-break draw per split

        row = {"cell": ck, "rep": rep, "m": m, "k": k,
               "n_x": len(xi), "n_y": len(yi),
               "n_keep_deployed_halfA": len(start_a),
               "auroc_deployed": auc_dep,
               "auroc_greedy": float(ref12["auroc_greedy"])}

        # --- the two controls, through exp13's own `by_stat` -------------------------
        orders = {"truecorr": by_stat(true_corr(Va, ca["labels"])),
                  "rho": by_stat(np.asarray(res_a.rho_hat_full, dtype=float))}

        # --- the l0-CCA arms ---------------------------------------------------------
        nanv = np.full(m, np.nan)
        if ok_split and not args.no_cca:
            Xa, Ya = _prep(Va[:, xi], Va[:, yi], rng=crng, r_max=R_MAX)

            lev = nanv.copy()
            sx, sy = cca_leverage_scores(Xa, Ya)
            lev[xi], lev[yi] = sx, sy

            ini = nanv.copy()
            ix_, iy_ = cca_init_mu(Xa, Ya, r=50.0)
            ini[xi], ini[yi] = ix_, iy_

            lam, path = pick_lambda(Xa, Ya, crng)
            gx, gy = l0cca_gates(Xa, Ya, lam=lam,
                                 seeds=list(range(N_SEEDS_STABILITY)))
            gates = nanv.copy()
            gates[xi], gates[yi] = gx, gy

            row["cca_lambda"] = float(lam)
            row["cca_n_open"] = int((gx > 0).sum() + (gy > 0).sum())
            row["cca_n_open_x"] = int((gx > 0).sum())
            row["cca_leverage_n_distinct"] = int(len(np.unique(np.round(lev[np.isfinite(lev)], 12))))
            row["cca_init_n_distinct"] = int(len(np.unique(np.round(ini[np.isfinite(ini)], 12))))
            for i, pr in enumerate(path):
                row[f"path_heldout_lam{i}"] = pr["heldout_corr"]
                row[f"path_nopen_lam{i}"] = pr["n_open"]
        else:
            lev = ini = gates = nanv.copy()
            path = [{"mu": (np.full(len(xi), np.nan), np.full(len(yi), np.nan))}
                    for _ in LAMBDA_GRID]
            row["cca_lambda"] = np.nan
            row["cca_n_open"] = -1
            row["cca_n_open_x"] = -1
            row["cca_leverage_n_distinct"] = -1
            row["cca_init_n_distinct"] = -1
            for i in range(len(LAMBDA_GRID)):
                row[f"path_heldout_lam{i}"] = np.nan
                row[f"path_nopen_lam{i}"] = np.nan

        orders["cca_leverage"] = by_score(lev, tb)
        orders["cca_gates"] = by_score(gates, tb)
        orders["cca_leverage_rr"] = channel_round_robin(is_x, lev, tb)
        orders["cca_gates_rr"] = channel_round_robin(is_x, gates, tb)
        orders["cca_init_r50"] = by_score(ini, tb)
        orders["chan_rr_random"] = channel_round_robin(is_x, np.full(m, np.nan), tb)
        for i, pr in enumerate(path):
            v = nanv.copy()
            if ok_split and not args.no_cca:
                v[xi], v[yi] = pr["mu"]
            orders[f"cca_gates_lam{i}"] = by_score(v, tb)

        s = set(start_a)
        # The conditional null depends on the arm ONLY through its inside/outside-keep-
        # set composition, which under pruning every arm shares. Estimated once per
        # composition, as exp14 does, so Monte Carlo noise does not enter the paired test.
        null_cache = {}
        for name in ARMS + SENSITIVITY:
            cols = prune_incumbent(start_a, orders[name], k, m)
            row[f"auroc_{name}_prune"] = fit_cols(cb, cols, exclusion=False)[0]
            if name in SENSITIVITY:
                continue
            row[f"{name}_prune_cols"] = "|".join(pool[c] for c in cols)
            row[f"overlap_{name}_prune"] = len(set(cols) & set(cols_greedy)) / k
            row[f"frac_inside_{name}"] = float(np.mean([c in s for c in cols]))
            row[f"frac_x_{name}"] = float(np.mean([bool(is_x[c]) for c in cols]))
            comp = sum(1 for j in cols if int(j) in s)
            if comp not in null_cache:
                null_cache[comp] = cond_overlap_null(
                    cols_greedy, cols, start_a, m, k, orng)
            row[f"cond_null_{name}_prune"] = null_cache[comp]
        row["n_distinct_null_compositions"] = len(null_cache)
        row["frac_x_greedy"] = float(np.mean([bool(is_x[c]) for c in cols_greedy]))

        _check(ck, rep, "auroc_truecorr_prune", row["auroc_truecorr_prune"],
               float(ref13["auroc_truecorr_prune"]))
        _check(ck, rep, "auroc_rho_prune", row["auroc_rho_prune"],
               float(ref13["auroc_rho_prune"]))

        # --- exp13's floor and control nulls, on `nrng`, in exp13's own order --------
        rnd = []
        for _ in range(N_RANDOM):
            order = [int(j) for j in nrng.permutation(m)]
            rnd.append(fit_cols(cb, prune_incumbent(start_a, order, k, m),
                                exclusion=False)[0])
        row["auroc_random_prune"] = float(np.nanmean(rnd))
        row["n_random_prune_failed"] = int(np.sum(~np.isfinite(rnd)))
        _check(ck, rep, "auroc_random_prune", row["auroc_random_prune"],
               float(ref13["auroc_random_prune"]))

        for name in ("truecorr", "rho"):
            cols_rebuilt = [pool.index(f) for f in ref12[f"{name}_cols_ownk"].split("|")]
            got = cond_overlap_null(cols_greedy, cols_rebuilt, start_a, m, k, nrng)
            _check(ck, rep, f"overlap_{name}_conditional_null", got,
                   float(ref13[f"overlap_{name}_conditional_null"]))

        # --- exp12's random consumption, replayed so the next split lines up ---------
        for _ in range(2 * N_OVERLAP_DRAWS + N_RANDOM):
            rng.choice(m, k, replace=False)

        split_rows.append(row)

    return split_rows


# ---------------------------------------------------------------------------
def check_cca_sign_invariance(cells, n_cells=4):
    """The gates are read off a matrix whose polarity is DERIVED per split, not the hand
    polarity anything else used. That only matters if the objective sees the sign.

    It should not. Flipping column signs sends Cx -> D_x Cx D_x, Cy -> D_y Cy D_y,
    Cxy -> D_x Cxy D_y for sign-diagonal D; the ridge term is unchanged because
    D I D = I, and tr(Cy^-1 Cyx Cx^-1 Cxy) is invariant under that conjugation. Asserted
    here rather than assumed — if it fails, which polarity the arm is fed is an
    undeclared degree of freedom and the pre-registration is incomplete.
    """
    print("\n=== gate: l0-CCA gates invariant to per-column sign flips ===")
    rng = np.random.default_rng(20260804)
    worst = 0.0
    for ck in list(cells)[:n_cells]:
        pool = cells[ck]["pool"]
        V = np.asarray(cells[ck]["V"], dtype=float)
        xi = [j for j, f in enumerate(pool) if f in X_VIEWS]
        yi = [j for j, f in enumerate(pool) if f not in X_VIEWS]
        if len(xi) < MIN_CHANNEL or len(yi) < MIN_CHANNEL:
            continue
        Xa, Ya = _prep(V[:, xi], V[:, yi], rng=np.random.default_rng(7))
        fx = rng.choice([-1.0, 1.0], len(xi))
        fy = rng.choice([-1.0, 1.0], len(yi))
        g0 = np.concatenate(train_l0cca(Xa, Ya, lam=0.1, seed=0))
        g1 = np.concatenate(train_l0cca(Xa * fx, Ya * fy, lam=0.1, seed=0))
        l0 = np.concatenate(cca_leverage_scores(Xa, Ya))
        l1 = np.concatenate(cca_leverage_scores(Xa * fx, Ya * fy))
        d = max(float(np.max(np.abs(g0 - g1))), float(np.max(np.abs(l0 - l1))))
        worst = max(worst, d)
        print(f"  {S.plain_cell(ck)[:34]:34s} max |mu - mu_flipped| = {d:.3e}"
              f"   {'OK' if d < 1e-9 else 'SIGN-DEPENDENT'}")
    if worst >= 1e-9:
        raise SystemExit(
            f"l0-CCA SIGN-INVARIANCE GATE FAILED: max deviation {worst:.3e}. Which "
            "polarity the arm is fed is then a live choice — stop and pre-register it "
            "before scoring.")
    print(f"  PASS (worst {worst:.3e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None, help="smoke subset")
    ap.add_argument("--no-cca", action="store_true",
                    help="structural dry run; every l0-CCA arm becomes NaN")
    ap.add_argument("--check-sign", action="store_true",
                    help="run the sign-invariance gate and exit")
    args = ap.parse_args()

    # A partial run must not be able to masquerade as the real one (exp14's finding).
    partial = bool(args.no_cca or args.cells)
    out = S.outdir(OUT_DIR + ("_partial" if partial else ""))
    cells = S.load()
    S.validity_check(cells)
    if args.check_sign:
        check_cca_sign_invariance(cells)
        return
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    derived_arm_gate(dcells)
    prev12, prev13 = load_exp12_splits(), load_exp13_splits()
    if not args.no_cca:
        check_cca_sign_invariance(cells)

    keys = [k for k in cells if args.cells is None or k in args.cells]
    split_rows = []
    for i, ck in enumerate(keys, 1):
        rng = np.random.default_rng(zlib.crc32(ck.encode()) % (2 ** 32))
        nrng = np.random.default_rng(0xC0FFEE + (zlib.crc32(ck.encode()) % 10 ** 6))
        trng = np.random.default_rng(0x14A11E5 + (zlib.crc32(ck.encode()) % 10 ** 6))
        orng = np.random.default_rng(0x0FF5E7 + (zlib.crc32(ck.encode()) % 10 ** 6))
        # A FIFTH stream, touched by nothing published: the CCA row subsample and the
        # inner lambda split. Adding it cannot move exp13's or exp14's arms.
        crng = np.random.default_rng(0x0CCA55 + (zlib.crc32(ck.encode()) % 10 ** 6))
        rs = run_cell(ck, cells[ck], dcells[ck], rng, nrng, trng, orng, crng,
                      prev12, prev13, args)
        split_rows += rs
        if rs:
            f = lambda key: float(np.nanmean([r[key] for r in rs]))  # noqa: E731
            fl = f("auroc_random_prune")
            print(f"[{i:2d}/{len(keys)}] {S.plain_cell(ck)[:30]:30s} "
                  f"vs floor: good set {(f('auroc_greedy')-fl)*100:+5.2f} | leverage "
                  f"{(f('auroc_cca_leverage_prune')-fl)*100:+5.2f} | gates "
                  f"{(f('auroc_cca_gates_prune')-fl)*100:+5.2f} | lam "
                  f"{f('cca_lambda'):.3f} open {f('cca_n_open'):.1f}", flush=True)

    S.save_csv(os.path.join(out, "splits.csv"), split_rows, _all_fields(split_rows))
    rows = collapse(split_rows)
    S.save_csv(os.path.join(out, "per_cell.csv"), rows, _all_fields(rows))

    summary = {"n_cells": len(rows), "n_splits": len(split_rows),
               "PROVENANCE": ("Pre-registered in this file's docstring before the run. "
                              "Splits are exp12's; exp13's arms asserted identical per "
                              f"split to {TOL_REPRO}."),
               "args": vars(args), "is_partial_run": partial,
               "channel_split": {"X_spectral": sorted(X_VIEWS)},
               "lambda_grid": list(LAMBDA_GRID),
               "label_free_arms": list(LABEL_FREE), "controls": list(CONTROLS),
               "sensitivity_arms_not_in_family": list(SENSITIVITY)}
    if partial:
        summary["WARNING"] = (
            "PARTIAL RUN — --cells and/or --no-cca was passed. Not comparable to the "
            "full run; the l0-CCA arms are NaN under --no-cca.")

    print(f"\n=== the room, re-derived on these rows (exp13 quoted {ROOM_PP:+.2f}pp) ===")
    summary["room"] = paired(rows, "auroc_random_prune", "auroc_greedy",
                             "the good set vs the matched pruning floor")
    summary["floor_vs_deployed"] = paired(rows, "auroc_deployed", "auroc_random_prune",
                                          f"the floor vs deployed (exp13 {FLOOR_PP:+.2f}pp)")

    print("\n=== PRIMARY: each arm vs the matched pruning floor, held out on half B ===")
    prim = summary["primary_vs_floor"] = {}
    for name in ARMS:
        tag = "  (control)" if name in CONTROLS else ""
        prim[name] = paired(rows, "auroc_random_prune", f"auroc_{name}_prune", name + tag)
    adj = holm([(n, prim[n]["p"]) for n in LABEL_FREE])
    summary["primary_holm_adjusted_p"] = adj
    print("  Holm-adjusted p over the two label-free arms: "
          + ", ".join(f"{n}={adj[n]:.3f}" for n in LABEL_FREE))

    print("\n=== how much of the room does each arm recover? "
          "(denominator NOT construction-matched — see docstring) ===")
    frac = summary["fraction_of_room"] = {}
    room = summary["room"]["mean_delta"]
    for name in ARMS:
        frac[name] = prim[name]["mean_delta"] / room if room else float("nan")
        print(f"  {name:22s} {prim[name]['mean_delta']:+6.2f}pp of {room:+.2f}pp "
              f"= {100*frac[name]:5.1f}%")

    print("\n=== SECONDARY: overlap with the good set vs the CONDITIONAL null ===")
    sec = summary["secondary_overlap"] = {}
    for name in ARMS:
        tag = "  (control)" if name in CONTROLS else ""
        sec[name] = paired(rows, f"cond_null_{name}_prune", f"overlap_{name}_prune",
                           name + tag, scale=1.0)

    print("\n=== sensitivity: channel round-robin, the paper's literal init, "
          "and the regularization path ===")
    sc = summary["sensitivity"] = {}
    for name in SENSITIVITY:
        lab = name
        if name.startswith("cca_gates_lam"):
            lab = f"{name}  (lambda = {LAMBDA_GRID[int(name[-1])]})"
        sc[name] = paired(rows, "auroc_random_prune", f"auroc_{name}_prune", lab)

    print("\n=== the round-robin arms against THEIR OWN floor (same rotation, "
          "random within channel) ===")
    rr = summary["roundrobin_vs_channel_balance"] = {}
    for name in ("cca_leverage_rr", "cca_gates_rr"):
        rr[name] = paired(rows, "auroc_chan_rr_random_prune", f"auroc_{name}_prune",
                          f"{name} vs channel balance alone")

    print("\n=== the regularization path, as the label-free criterion sees it ===")
    path = summary["lambda_path"] = []
    for i, lam in enumerate(LAMBDA_GRID):
        hc = [r[f"path_heldout_lam{i}"] for r in rows if np.isfinite(r.get(f"path_heldout_lam{i}", np.nan))]
        no = [r[f"path_nopen_lam{i}"] for r in rows if np.isfinite(r.get(f"path_nopen_lam{i}", np.nan))]
        rec = {"lam": lam,
               "mean_heldout_total_corr": float(np.mean(hc)) if hc else float("nan"),
               "mean_n_open": float(np.mean(no)) if no else float("nan"),
               "vs_floor_pp": sc[f"cca_gates_lam{i}"]["mean_delta"]}
        path.append(rec)
        print(f"  lambda={lam:<5} held-out total corr {rec['mean_heldout_total_corr']:6.3f}"
              f"   open gates {rec['mean_n_open']:5.1f}"
              f"   vs floor {rec['vs_floor_pp']:+5.2f}pp")
    chosen = [r["cca_lambda"] for r in split_rows if np.isfinite(r["cca_lambda"])]
    if chosen:
        vals, cnt = np.unique(chosen, return_counts=True)
        summary["lambda_chosen_distribution"] = {
            float(v): int(c) for v, c in zip(vals, cnt)}
        print("  lambda chosen by the label-free criterion: "
              + ", ".join(f"{v}x{c}" for v, c in zip(vals, cnt)))

    print("\n=== sensitivity: splits at k <= 4 dropped (L-SML undetermined, Step 205) ===")
    rows_k5 = collapse(split_rows, keep=lambda r: r["k"] >= 5)
    summary["n_splits_k_ge_5"] = int(sum(1 for r in split_rows if r["k"] >= 5))
    summary["n_cells_k_ge_5"] = len(rows_k5)
    sens = summary["primary_vs_floor_k_ge_5"] = {}
    if len(rows_k5) >= 6:
        sens["room"] = paired(rows_k5, "auroc_random_prune", "auroc_greedy",
                              "the good set vs the floor")
        for name in ARMS:
            sens[name] = paired(rows_k5, "auroc_random_prune", f"auroc_{name}_prune", name)
    else:
        print("  too few cells survive the filter to report")

    summary["diagnostics"] = {
        "n_random_prune_fits_failed": int(np.sum(
            [r["n_random_prune_failed"] for r in split_rows])),
        "mean_k": float(np.mean([r["k"] for r in split_rows])),
        "mean_keepset": float(np.mean([r["n_keep_deployed_halfA"] for r in split_rows])),
        "n_splits_k_gt_keepset": int(sum(
            1 for r in split_rows if r["k"] > r["n_keep_deployed_halfA"])),
        "mean_n_x": float(np.mean([r["n_x"] for r in split_rows])),
        "mean_n_y": float(np.mean([r["n_y"] for r in split_rows])),
        "mean_cca_n_open": float(np.mean([r["cca_n_open"] for r in split_rows])),
        "n_splits_cca_open_lt_k": int(sum(
            1 for r in split_rows if 0 <= r["cca_n_open"] < r["k"])),
        "mean_cca_init_n_distinct": float(np.mean(
            [r["cca_init_n_distinct"] for r in split_rows])),
        "mean_distinct_null_compositions": float(np.mean(
            [r["n_distinct_null_compositions"] for r in split_rows])),
        "mean_frac_spectral": {
            **{n: float(np.mean([r[f"frac_x_{n}"] for r in rows])) for n in ARMS},
            "greedy": float(np.mean([r["frac_x_greedy"] for r in rows]))},
        "mean_frac_inside_keepset": {
            n: float(np.mean([r[f"frac_inside_{n}"] for r in rows])) for n in ARMS},
    }
    print("\ndiagnostics: " + ", ".join(
        f"{k}={v}" for k, v in summary["diagnostics"].items()
        if not isinstance(v, dict)))
    print("  spectral fraction of the selection: " + ", ".join(
        f"{k}={v:.2f}" for k, v in summary["diagnostics"]["mean_frac_spectral"].items()))

    S.save_json(os.path.join(out, "summary.json"), summary)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
