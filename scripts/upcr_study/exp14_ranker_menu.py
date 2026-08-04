"""Can any label-free per-feature ranking reach the room that feature selection has?

WHY THIS EXISTS
    Every step inside U-PCR has now been priced by letting labels do it perfectly, and
    only ONE has room: which features get kept, about +2.25pp held out (CI [+1.53,
    +3.04], 23W/1L) against a floor that trims the deployed keep set at random.

    exp13 then measured the strongest label-BASED marginal statistic in that channel.
    The true correlation with correctness IDENTIFIES the good features (+0.11 overlap
    over a composition-matched null, p < 1e-4) and BUYS NOTHING (+0.08pp over the
    matched floor, p = 0.62). So what failed there is a SHAPE — score each feature on
    its own, keep the top k — and every candidate with that shape inherits the risk.

    This script prices the label-free menu in the same channel, on exp13's own splits,
    before anything is built on top of a ranker.

WHAT IS PRE-REGISTERED, WRITTEN BEFORE THE RUN
    Eight arms. Each produces an ORDER over the pool from half A alone, which prunes
    the deployed keep set down to the good set's own size k; the pruned set is refit
    and scored on half B. Directions are fixed HERE because a free direction doubles
    the degrees of freedom.

      dufs_gate      the seed-averaged DUFS gate `mu`, param_free=True (Eq. 7), 5
                     stability seeds.  HIGHER mu = keep, SIGNED. Not |mu|: MU_INIT is
                     0.5 and Adam drives rejected gates negative, so |mu| would promote
                     the most strongly REJECTED features. Bracha's first proposal.
      pc_leverage    eigenvalue-weighted squared loading on the top 2 eigenvectors of
                     the half-A feature covariance.  HIGHER = keep.
      redundancy     mean |correlation| to the rest of the pool.  LOWER = keep.
      cluster_size   size of the L-SML dependent group the feature lands in.
                     LOWER = keep.
      pairfit_resid  mean |C_ij - (rho_i + rho_j)| over j != i, from U-PCR's own Eq. 15
                     solve at q = 0 (the residual vector is invariant to g2).
                     LOWER = keep.
      cluster_rr     SET-LEVEL, not a per-feature score: take one feature per L-SML
                     cluster in rotation, ordered within a cluster by the DUFS gate,
                     clusters visited by their best gate. The only arm in the menu that
                     escapes the shape that already failed.
      rho            U-PCR's own rho_hat_full — the control KNOWN to be BELOW chance.
      truecorr       the true correlation with correctness — uses labels; a ceiling on
                     the whole marginal family, and exp13 puts that ceiling ON the floor.

    DECISION RULE, fixed in advance:
      PRIMARY   held-out AUROC of the pruned set, paired over the 24 test sets, against
                the MATCHED PRUNING FLOOR (random order, same construction, same k).
                Positioned against the room, +2.25pp.
      SECONDARY overlap with the good set against a null matched on the arm's OWN
                observed inside/outside-keep-set composition. A uniform-over-pool null
                is too easy for anything living inside the keep set.
      MULTIPLICITY  Holm-Bonferroni over the SIX label-free arms on the primary
                endpoint. The two controls are excluded: their outcomes are known.
      READING   A ranker that clears only the overlap test has REPRODUCED exp13's
                outcome, not beaten it. Written down now so it cannot be renegotiated
                after the numbers are in.
      If the whole menu lands on the floor, the item closes with a ceiling AND an
      impossibility statement — the room is not reachable by per-feature ranking. That
      is a stronger deliverable than a variant that merely loses.

REPRODUCTION
    The splits are exp12's, replayed through exp13's own machinery: the same per-cell
    crc32 seed, the same replayed random consumption, the same `nrng` draws in the same
    order. ALL new randomness is on a THIRD generator (`trng`), so exp13's arms are
    invariant to what is added here by construction rather than by ordering discipline.
    The run asserts, per split and to 1e-9, that the deployed AUROC, the pruning floor,
    both control arms, and exp13's own REBUILT-set conditional nulls match its
    splits.csv. Read that precisely: the nulls REPORTED here are not exp13's. exp13
    measured overlap for the rebuilt top-k against a rebuilt-composition null; this
    script measures it for the PRUNED set against a pruned-composition null, so the two
    controls are re-measured on a different estimand rather than copied. Both keep their
    direction and their significance, which is the point — but +0.09 here and +0.11 in
    exp13 are two measurements, not a discrepancy. Flagged in review.

    A consequence worth stating: under pruning, every arm's keep-set composition is the
    SAME (mean frac-inside 0.9985, one distinct composition on all 120 splits), so the
    conditional null collapses to one common quantity — "draw k uniformly from the keep
    set". The secondary endpoint is therefore a common-null comparison across arms, not
    eight separately matched ones.
"""
import argparse
import csv
import os
import sys
import zlib

import numpy as np
from scipy.linalg import eigh
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
from spectral_utils.subset_sweep import ALL_SIGNS                   # noqa: E402
from spectral_utils.upcr import upcr_fit, additive_design, solve_additive   # noqa: E402
from spectral_utils.fusion_utils import detect_dependent_groups     # noqa: E402
from spectral_utils.selectors.a2_groupfs import (                   # noqa: E402
    dufs_pf_gates, dufs_pf_cell_rng)

# `scripts/nonmono_v2/dufs_pf.py` cannot be imported from here — its module-level
# `from common import ...` resolves to THIS package's `common.py`, which is already in
# sys.modules. The gate code it exposes lives in spectral_utils (imported above) for
# exactly that reason; only the domain lookup is small enough to repeat.
DUFS_BENCH = os.path.join(S.REPO, "results", "selector_bench", "a2_groupfs__c46.csv")

EXP13_DIR = "13_incumbent_anchored_ranking"
OUT_DIR = "14_ranker_menu"

# The six label-free arms carry the multiplicity correction; the two controls do not.
LABEL_FREE = ("dufs_gate", "pc_leverage", "redundancy", "cluster_size",
              "pairfit_resid", "cluster_rr")
CONTROLS = ("rho", "truecorr")
ARMS = LABEL_FREE + CONTROLS
# Scale-dependent re-runs of the two cluster arms. NOT in the primary table and NOT in
# the multiplicity family — they exist because `common.py:42` commits this study to
# showing the loading-scale sensitivity of anything that has one, rather than picking.
SENSITIVITY = ("cluster_size_eigen", "cluster_size_complete",
               "cluster_rr_eigen", "cluster_rr_complete")

# exp13's own reference points, quoted so a drift shows up in the printout rather than
# in a later reading of the CSV.
ROOM_PP = 2.25            # good set minus the matched pruning floor
FLOOR_PP = -0.84          # the matched pruning floor, against the deployed pool


# ---------------------------------------------------------------------------
# orders
# ---------------------------------------------------------------------------
def by_score(score, tiebreak):
    """Best-first order for an ALREADY-ORIENTED score (higher = keep).

    Ties are broken at random rather than by feature index: the pool is laid out
    spectral-then-energy-then-logprob, so a stable index tie-break would silently mean
    "prefer spectral" — a prior, in a study whose whole point is not carrying one.
    Non-finite scores rank last.

    The two CONTROLS deliberately do NOT come through here; they call exp13's `by_stat`
    so their arms stay bit-identical to the published ones (`np.argsort` is not stable,
    `np.lexsort` is, and the reproduction gate is at 1e-9).
    """
    s = np.asarray(score, dtype=float)
    s = np.where(np.isfinite(s), s, -np.inf)
    return [int(j) for j in np.lexsort((np.asarray(tiebreak, dtype=float), -s))]


def pc_leverage(C, k=2):
    """Eigenvalue-weighted squared loading on the top-k eigenvectors of C."""
    m = C.shape[0]
    k = int(min(k, m))
    vals, vecs = eigh(C, subset_by_index=[m - k, m - 1])
    return np.asarray((vecs ** 2) @ vals, dtype=float)


def redundancy(C):
    """Mean |correlation| to the rest of the pool. Returned ORIENTED (negated)."""
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    R = np.abs(C / np.outer(d, d))
    np.fill_diagonal(R, 0.0)
    m = C.shape[0]
    return -R.sum(axis=1) / max(m - 1, 1)


def pairfit_resid(C, loss):
    """Per-feature mean |C_ij - (rho_i + rho_j)| under U-PCR's own Eq. 15 fit.

    Solved at q = 0. Eq. 16 shifts rho by q/2 and b by q, so the residual VECTOR is
    invariant to g2 (upcr.py:56-63) and no g2 choice enters this statistic.

    Returned ORIENTED (negated), i.e. LOW residual = keep, matching the direction
    pre-registered in the module docstring: U-PCR's whole model is the additive story,
    so a feature whose pair covariances that story cannot reproduce is a feature the
    fusion's own assumptions do not hold for. (The opposite reading — a large residual
    means unexploited structure — is arguable, which is exactly why the direction is
    fixed in advance rather than chosen after seeing both.)
    """
    m = C.shape[0]
    A, prs = additive_design(m)
    b = np.array([C[i, j] for i, j in prs], dtype=float)
    rho0 = solve_additive(A, b, loss=loss)
    r = np.abs(b - A @ rho0)
    tot, cnt = np.zeros(m), np.zeros(m)
    for (i, j), rv in zip(prs, r):
        tot[i] += rv
        tot[j] += rv
        cnt[i] += 1
        cnt[j] += 1
    return -tot / np.maximum(cnt, 1.0)


def cluster_roundrobin(assign, gate, tiebreak):
    """One feature per L-SML cluster in rotation; within a cluster, best gate first;
    clusters visited by their best gate. Set-level: a feature's position depends on
    which other features share its cluster, which no marginal statistic can express.

    Non-finite gates are mapped to -inf, not left as NaN: tuple comparison
    short-circuits on `nan != nan`, so a NaN gate would leave Timsort's input order
    untouched and the ranking would silently become the POOL INDEX order — which is
    spectral-then-energy-then-logprob, exactly the prior `by_score`'s random tie-break
    exists to avoid. Flagged in review.
    """
    assign = np.asarray(assign, dtype=int)
    g_ = np.asarray(gate, dtype=float)
    g_ = np.where(np.isfinite(g_), g_, -np.inf)
    order_in = {}
    for g in np.unique(assign):
        idx = [int(j) for j in np.where(assign == g)[0]]
        order_in[int(g)] = sorted(idx, key=lambda j: (-g_[j], float(tiebreak[j])))
    groups = sorted(order_in, key=lambda g: (-g_[order_in[g][0]],
                                             float(tiebreak[order_in[g][0]])))
    out, pos = [], 0
    while len(out) < len(assign):
        for g in groups:
            if pos < len(order_in[g]):
                out.append(order_in[g][pos])
        pos += 1
    return out


def lsml_clusters(V, scale="unit"):
    """L-SML dependent groups on the half-A matrix. Returns (assignment, K, degenerate).

    The PRE-REGISTERED scale is 'unit' — `lsml_continuous`'s own default and the one
    the deployed path uses. But the residual criterion picks K per scale, so the two
    cluster arms are scale-dependent, and `common.py:42` commits this study to
    reporting every scale-dependent number under all three rather than choosing one.
    Both other scales are therefore run as SENSITIVITY arms: outside the primary
    table, outside the multiplicity family, reported so the choice is visible.
    """
    m = V.shape[1]
    K, c, _, _, diag = detect_dependent_groups(
        [V[:, j] for j in range(m)], method="residual",
        loading_scale=scale, return_diag=True)
    return np.asarray(c, dtype=int), int(K), bool(diag.get("degenerate", False))


# ---------------------------------------------------------------------------
def bench_domains(path=DUFS_BENCH):
    """cell -> domain, as the selector bench recorded it. The domain string is part of
    the DUFS seed, so it has to match the bench's or the trainings are a different
    selector wearing the same name."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["cell"]: r["domain"] for r in csv.DictReader(fh)
                if r["variant"] == "a2.dufs_pf"}


def load_exp13_splits():
    path = os.path.join(S.outdir(EXP13_DIR), "splits.csv")
    with open(path, newline="") as f:
        return {(r["cell"], int(r["rep"])): r for r in csv.DictReader(f)}


def _check(ck, rep, name, got, want):
    if not np.isfinite(got) and not np.isfinite(want):
        return
    if abs(got - want) > TOL_REPRO:
        raise RuntimeError(
            f"{ck} rep {rep}: {name} = {got:.12f} != exp13's {want:.12f} — a stream "
            "this script must not touch has moved")


def run_cell(ck, cell, dcell, domain, rng, nrng, trng, orng, prev12, prev13, args):
    V0 = cell["V"]
    y = dcell["labels"]
    n, m = V0.shape
    hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
    pool = cell["pool"]

    split_rows = []
    for rep in range(N_SPLITS):
        # --- exp12's split, reproduced exactly (identical to exp13) -----------------
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

        ref12 = prev12.get((ck, rep))
        ref13 = prev13.get((ck, rep))
        if ref12 is None or ref13 is None:
            raise RuntimeError(f"{ck} rep {rep}: no exp12/exp13 row — the splits diverged")
        k = int(ref12["k"])
        cols_greedy = [pool.index(f) for f in ref12["greedy_cols"].split("|")]

        auc_dep = fit_cols(cb, range(m))[0]
        _check(ck, rep, "auroc_deployed", auc_dep, float(ref13["auroc_deployed"]))

        Va = ca["V"]
        C_a = (Va.T @ Va) / Va.shape[0]
        tb = trng.random(m)                       # one tie-break draw per split

        row = {"cell": ck, "rep": rep, "m": m, "k": k,
               "n_keep_deployed_halfA": len(start_a),
               "auroc_deployed": auc_dep,
               "auroc_greedy": float(ref12["auroc_greedy"])}

        # --- the two controls, through exp13's own `by_stat` ------------------------
        orders = {"truecorr": by_stat(true_corr(Va, ca["labels"])),
                  "rho": by_stat(np.asarray(res_a.rho_hat_full, dtype=float))}

        # --- the six label-free arms ------------------------------------------------
        gate = (dufs_pf_gates(Va, dufs_pf_cell_rng(ck, domain, seed=rep))
                if not args.no_dufs else np.full(m, np.nan))
        assign, K_lsml, degen = lsml_clusters(Va)
        csize = np.array([int(np.sum(assign == assign[j])) for j in range(m)], float)

        orders["dufs_gate"] = by_score(gate, tb)
        orders["pc_leverage"] = by_score(pc_leverage(C_a), tb)
        orders["redundancy"] = by_score(redundancy(C_a), tb)
        orders["cluster_size"] = by_score(-csize, tb)
        orders["pairfit_resid"] = by_score(pairfit_resid(C_a, FIT["loss"]), tb)
        orders["cluster_rr"] = cluster_roundrobin(assign, gate, tb)

        row["lsml_K"] = K_lsml
        row["lsml_degenerate"] = int(degen)
        row["n_distinct_cluster_size"] = int(len(np.unique(csize)))
        row["dufs_n_open"] = int(np.sum(gate > 0)) if np.isfinite(gate).all() else -1

        # the two cluster arms at the other two loading scales — sensitivity only
        for sc in ("eigen", "complete"):
            asg, K_sc, dg_sc = lsml_clusters(Va, scale=sc)
            cs = np.array([int(np.sum(asg == asg[j])) for j in range(m)], float)
            orders[f"cluster_size_{sc}"] = by_score(-cs, tb)
            orders[f"cluster_rr_{sc}"] = cluster_roundrobin(asg, gate, tb)
            row[f"lsml_K_{sc}"] = K_sc
            row[f"lsml_degenerate_{sc}"] = int(dg_sc)

        s = set(start_a)
        # The conditional null depends on the arm ONLY through its inside/outside-keep-
        # set composition, and when k <= |keep set| every arm sits wholly inside — so
        # eight independent 2000-draw estimates of ONE quantity would inject pure Monte
        # Carlo noise into the paired secondary test. Estimated once per composition.
        # Flagged in review.
        null_cache = {}
        for name in ARMS + SENSITIVITY:
            cols = prune_incumbent(start_a, orders[name], k, m)
            row[f"auroc_{name}_prune"] = fit_cols(cb, cols, exclusion=False)[0]
            if name in SENSITIVITY:
                continue
            row[f"{name}_prune_cols"] = "|".join(pool[c] for c in cols)
            row[f"overlap_{name}_prune"] = len(set(cols) & set(cols_greedy)) / k
            row[f"frac_inside_{name}"] = float(np.mean([c in s for c in cols]))
            comp = sum(1 for j in cols if int(j) in s)
            if comp not in null_cache:
                null_cache[comp] = cond_overlap_null(
                    cols_greedy, cols, start_a, m, k, orng)
            row[f"cond_null_{name}_prune"] = null_cache[comp]
        row["n_distinct_null_compositions"] = len(null_cache)

        _check(ck, rep, "auroc_truecorr_prune", row["auroc_truecorr_prune"],
               float(ref13["auroc_truecorr_prune"]))
        _check(ck, rep, "auroc_rho_prune", row["auroc_rho_prune"],
               float(ref13["auroc_rho_prune"]))

        # --- exp13's floor and control nulls, on `nrng`, in exp13's own order -------
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

        # --- exp12's random consumption, replayed so the next split lines up --------
        for _ in range(2 * N_OVERLAP_DRAWS + N_RANDOM):
            rng.choice(m, k, replace=False)

        split_rows.append(row)

    return split_rows


def collapse(split_rows, keep=None):
    out = []
    for ck in dict.fromkeys(r["cell"] for r in split_rows):
        rs = [r for r in split_rows if r["cell"] == ck and (keep is None or keep(r))]
        if not rs:
            continue
        row = {"cell": ck, "n_splits": len(rs)}
        for key in rs[0]:
            if key in ("cell", "rep") or isinstance(rs[0][key], str):
                continue
            vals = [r[key] for r in rs if np.isfinite(float(r[key]))]
            row[key] = float(np.mean(vals)) if vals else np.nan
        out.append(row)
    return out


def paired(rows, base, arm, label, scale=100.0, quiet=False):
    d = np.array([r[arm] - r[base] for r in rows], float) * scale
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(0)
    bs = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(20000)]
    out = {"mean_delta": float(d.mean()),
           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": int(len(d)),
           "p": float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0)
                else float("nan")}
    if not quiet:
        unit = "pp" if scale == 100.0 else ""
        print(f"  {label:46s} {out['mean_delta']:+6.2f}{unit}  "
              f"[{out['ci95'][0]:+.2f}, {out['ci95'][1]:+.2f}]  "
              f"{out['wins']:2d}W/{out['losses']:2d}L  p={out['p']:.4f}")
    return out


def holm(pairs):
    """Holm-Bonferroni step-down. `pairs` is [(name, p)]; returns {name: adjusted p}.

    An arm with a non-finite p (a degenerate all-zero delta, or a dry run) LEAVES the
    family rather than inflating it: counting it in `n` would hand every surviving arm
    a larger multiplier for a test that was never performed. Flagged in review.
    """
    live = [(n_, p) for n_, p in pairs if np.isfinite(p)]
    ordered = sorted(live, key=lambda t: t[1])
    n, run, out = len(ordered), 0.0, {n_: float("nan") for n_, _ in pairs}
    for i, (name, p) in enumerate(ordered):
        run = max(run, min(1.0, (n - i) * p))
        out[name] = run
    return out


# ---------------------------------------------------------------------------
def check_dufs_sign_invariance(cells, doms, n_cells=4):
    """The gates are read off a matrix whose polarity is DERIVED per split, not the
    hand polarity the bench used. That only matters if the objective sees the sign.

    It should not: the self-tuning affinity is built from squared row distances, and
    flipping column j negates the same coordinate in every row, so distances are
    unchanged; the trace term picks up (-1)^2 per column. Both are sign-invariant.
    Asserted here rather than assumed — if it fails, the polarity is a live choice and
    the menu's DUFS arm has an undeclared degree of freedom.
    """
    print("\n=== gate: DUFS gates invariant to per-column sign flips ===")
    rng = np.random.default_rng(20260804)
    worst = 0.0
    for ck in list(cells)[:n_cells]:
        V = np.asarray(cells[ck]["V"], dtype=float)
        flip = rng.choice([-1.0, 1.0], V.shape[1])
        g0 = dufs_pf_gates(V, dufs_pf_cell_rng(ck, doms.get(ck, "repgrid")))
        g1 = dufs_pf_gates(V * flip, dufs_pf_cell_rng(ck, doms.get(ck, "repgrid")))
        d = float(np.max(np.abs(g0 - g1)))
        worst = max(worst, d)
        print(f"  {S.plain_cell(ck)[:34]:34s} max |mu - mu_flipped| = {d:.3e}"
              f"   {'OK' if d < 1e-9 else 'SIGN-DEPENDENT'}")
    if worst >= 1e-9:
        raise SystemExit(
            f"DUFS SIGN-INVARIANCE GATE FAILED: max deviation {worst:.3e}. The gate "
            "depends on feature polarity, so which polarity the menu feeds it is an "
            "undeclared choice — stop and pre-register it before scoring.")
    print(f"  PASS (worst {worst:.3e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None, help="smoke subset")
    ap.add_argument("--no-dufs", action="store_true",
                    help="structural dry run; the two DUFS arms become NaN")
    ap.add_argument("--check-dufs-sign", action="store_true",
                    help="run the sign-invariance gate and exit")
    args = ap.parse_args()

    # A partial run must not be able to masquerade as the real one: `--no-dufs` NaNs
    # out two of the six arms and `--cells` shrinks the paired sample, yet both used to
    # write the same three files. Flagged in review.
    partial = bool(args.no_dufs or args.cells)
    out = S.outdir(OUT_DIR + ("_partial" if partial else ""))
    cells = S.load()
    S.validity_check(cells)
    doms = bench_domains()
    if args.check_dufs_sign:
        check_dufs_sign_invariance(cells, doms)
        return
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    derived_arm_gate(dcells)
    prev12, prev13 = load_exp12_splits(), load_exp13_splits()
    # The menu feeds DUFS the DERIVED-polarity matrix, not the hand-oriented one the
    # published selector saw, so a scoring run asserts the objective is blind to
    # polarity instead of only claiming it.
    if not args.no_dufs:
        check_dufs_sign_invariance(cells, doms)

    keys = [k for k in cells if args.cells is None or k in args.cells]
    split_rows = []
    for i, ck in enumerate(keys, 1):
        rng = np.random.default_rng(zlib.crc32(ck.encode()) % (2 ** 32))
        nrng = np.random.default_rng(0xC0FFEE + (zlib.crc32(ck.encode()) % 10 ** 6))
        # Tie-breaks and the conditional nulls get SEPARATE streams so that changing how
        # many nulls are drawn cannot move a single tie-break, and the primary table is
        # invariant to anything done to the secondary one.
        trng = np.random.default_rng(0x14A11E5 + (zlib.crc32(ck.encode()) % 10 ** 6))
        orng = np.random.default_rng(0x0FF5E7 + (zlib.crc32(ck.encode()) % 10 ** 6))
        rs = run_cell(ck, cells[ck], dcells[ck], doms.get(ck, "repgrid"),
                      rng, nrng, trng, orng, prev12, prev13, args)
        split_rows += rs
        if rs:
            f = lambda key: float(np.nanmean([r[key] for r in rs]))  # noqa: E731
            fl = f("auroc_random_prune")
            print(f"[{i:2d}/{len(keys)}] {S.plain_cell(ck)[:30]:30s} "
                  f"vs floor: good set {(f('auroc_greedy')-fl)*100:+5.2f} | dufs "
                  f"{(f('auroc_dufs_gate_prune')-fl)*100:+5.2f} | rr "
                  f"{(f('auroc_cluster_rr_prune')-fl)*100:+5.2f} | lev "
                  f"{(f('auroc_pc_leverage_prune')-fl)*100:+5.2f} | red "
                  f"{(f('auroc_redundancy_prune')-fl)*100:+5.2f}", flush=True)

    S.save_csv(os.path.join(out, "splits.csv"), split_rows, _all_fields(split_rows))
    rows = collapse(split_rows)
    S.save_csv(os.path.join(out, "per_cell.csv"), rows, _all_fields(rows))

    summary = {"n_cells": len(rows), "n_splits": len(split_rows),
               "PROVENANCE": ("Pre-registered in this file's docstring before the run. "
                              "Splits are exp12's; exp13's arms asserted identical "
                              f"per split to {TOL_REPRO}."),
               "args": vars(args), "is_partial_run": partial,
               "label_free_arms": list(LABEL_FREE), "controls": list(CONTROLS),
               "sensitivity_arms_not_in_family": list(SENSITIVITY)}
    if partial:
        summary["WARNING"] = (
            "PARTIAL RUN — --cells and/or --no-dufs was passed. Not comparable to the "
            "full menu; the DUFS arms are NaN under --no-dufs.")

    print(f"\n=== the room, re-derived on these rows (exp13 quoted {ROOM_PP:+.2f}pp) ===")
    summary["room"] = paired(rows, "auroc_random_prune", "auroc_greedy",
                             "the good set vs the matched pruning floor")
    summary["floor_vs_deployed"] = paired(rows, "auroc_deployed", "auroc_random_prune",
                                          f"the floor vs deployed (exp13 {FLOOR_PP:+.2f}pp)")

    print("\n=== PRIMARY: each arm vs the matched pruning floor, held out on half B ===")
    prim = summary["primary_vs_floor"] = {}
    for name in ARMS:
        tag = "  (control)" if name in CONTROLS else ""
        prim[name] = paired(rows, "auroc_random_prune", f"auroc_{name}_prune",
                            name + tag)
    adj = holm([(n, prim[n]["p"]) for n in LABEL_FREE])
    summary["primary_holm_adjusted_p"] = adj
    print("  Holm-adjusted p over the six label-free arms: "
          + ", ".join(f"{n}={adj[n]:.3f}" for n in LABEL_FREE))

    print("\n=== how much of the room does each arm recover? ===")
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

    print("\n=== sensitivity: the two cluster arms at the other two loading scales ===")
    sc = summary["loading_scale_sensitivity"] = {}
    for name in ("cluster_size", "cluster_rr"):
        for suffix in ("", "_eigen", "_complete"):
            lab = name + (suffix or "  (unit, pre-registered)")
            sc[name + suffix] = paired(rows, "auroc_random_prune",
                                       f"auroc_{name + suffix}_prune", lab)

    print("\n=== sensitivity: splits at k <= 4 dropped (L-SML undetermined, Step 205) ===")
    rows_k5 = collapse(split_rows, keep=lambda r: r["k"] >= 5)
    summary["n_splits_k_ge_5"] = int(sum(1 for r in split_rows if r["k"] >= 5))
    summary["n_cells_k_ge_5"] = len(rows_k5)
    sens = summary["primary_vs_floor_k_ge_5"] = {}
    if len(rows_k5) >= 6:
        sens["room"] = paired(rows_k5, "auroc_random_prune", "auroc_greedy",
                              "the good set vs the floor")
        for name in ARMS:
            sens[name] = paired(rows_k5, "auroc_random_prune", f"auroc_{name}_prune",
                                name)
    else:
        print("  too few cells survive the filter to report")

    summary["diagnostics"] = {
        "n_random_prune_fits_failed": int(np.sum(
            [r["n_random_prune_failed"] for r in split_rows])),
        "mean_k": float(np.mean([r["k"] for r in split_rows])),
        "mean_keepset": float(np.mean([r["n_keep_deployed_halfA"] for r in split_rows])),
        "n_splits_k_gt_keepset": int(sum(
            1 for r in split_rows if r["k"] > r["n_keep_deployed_halfA"])),
        "mean_lsml_K": float(np.mean([r["lsml_K"] for r in split_rows])),
        "n_splits_lsml_degenerate": int(sum(r["lsml_degenerate"] for r in split_rows)),
        "mean_distinct_cluster_sizes": float(np.mean(
            [r["n_distinct_cluster_size"] for r in split_rows])),
        "mean_dufs_n_open": float(np.mean([r["dufs_n_open"] for r in split_rows])),
        "mean_distinct_null_compositions": float(np.mean(
            [r["n_distinct_null_compositions"] for r in split_rows])),
        "mean_frac_inside_keepset": {
            n: float(np.mean([r[f"frac_inside_{n}"] for r in rows])) for n in ARMS},
    }
    print("\ndiagnostics: " + ", ".join(
        f"{k}={v}" for k, v in summary["diagnostics"].items()
        if not isinstance(v, dict)))

    S.save_json(os.path.join(out, "summary.json"), summary)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
