"""Can a label-free SET-LEVEL objective pick a better keep set than U-PCR's threshold?

WHY THIS EXISTS
    Every step inside U-PCR has been priced by letting labels do it perfectly, and only
    ONE has room: which features get kept, about +2.25pp held out (CI [+1.53, +3.04],
    23W/1L) against a floor that trims the deployed keep set at random.

    exp13 priced the strongest label-BASED marginal statistic in that channel (the true
    correlation: +0.08pp, p = 0.62) and exp14 priced the whole label-free per-feature
    menu (none clears the floor; two significantly below it). So the shape "score each
    feature on its own, keep the top k" is closed, and whatever makes a set good is a
    property of the SET.

    A Step-223 scratchpad diagnostic then refuted the obvious set-level candidate. The
    good sets fit the 1-factor model WORSE than matched random subsets of the same size
    drawn from the same keep set — additive misfit +0.131 (23W/1L, p=6e-7), rank-1
    misfit +0.100 (21W/3L, p=2.5e-5) — and are LESS internally correlated, not more
    (cohesion -0.129, 22W/24, p=4e-6). Those diagnostics are re-derived here (`r_diag`),
    because they were computed with no pre-registration and are not quotable until they
    are.

THE HYPOTHESIS THIS SCRIPT TESTS
    exp14's `redundancy` arm (mean |corr| to the whole pool, keep the lowest) was the
    WORST of eight at -3.13pp, yet the good sets are LESS internally correlated. Those
    reconcile only one way: mean |corr| conflates two things that pull in opposite
    directions --- loading on the shared factor (signal, want more) and sharing errors
    with the others (contamination, want less) --- and no marginal statistic can
    separate them. The quantity that does is the correlation NOT explained by the
    factor, i.e. the `Delta_ij = E[h_i h_j]` that upcr.py:86 Eq. 14 assumes to be zero
    and that the diagnostic measures at 0.46 of the off-diagonal norm on the full pool.

    C_S = lambda lambda' + Psi + Delta, fitted with Delta SPARSE. Sparse rather than
    low-rank because that is what the violation measured as: top decile of pairs carries
    ~44% of the residual mass, leading eigenvalue share of |Delta| only ~0.33.

THE ONE CONDITION BEING SWITCHED
    Polarity from sign(rho_hat), the U-PCR fit, the weights, the anchor and the scoring
    are all UNCHANGED. The single change is where the keep set comes from:
        deployed  every view whose rho_hat clears the threshold rule (upcr.py:287-293)
        new arm   the subset S* maximising a label-free composite-reliability objective
    Applied by passing a column subset to fit_cols(..., exclusion=False). No deployed
    file is edited.

WHAT IS PRE-REGISTERED, WRITTEN BEFORE THE RUN
    Five label-free arms. Each selects on half A ONLY, by greedy backward elimination
    from the deployed keep set, and is scored on half B. Directions are fixed inside
    `composite_reliability.OBJECTIVES` (every objective is MAXIMISED; the lower-is-
    better ones are negated in the module) so the search never sees a free direction.

      omega_sparse   M1  (sum lambda)^2 / (1' C 1) under the sparse-Delta fit.
                         THE FLAGSHIP, and the only SELF-SIZING arm besides m_eff:
                         a redundant view grows the denominator faster than the
                         numerator, so there is no K to set.
      resid_dep      M2  -sum |Delta_ij|.  Omri's relaxation used directly, in ABSOLUTE
                         form. The NORMALISED form is what the diagnostic measured and
                         it pointed the wrong way (it falls as its own denominator
                         grows); absolute is what the mechanism predicts.
      m_eff          M3  m / (1 + (m-1) * mean Delta_offdiag).  Effective number of
                         independent views. SELF-SIZING. The direct answer to "why do
                         smaller sets win".
      cohesion_set   M4  -mean |corr| within S.  THE CONTROL, and the cleanest test in
                         the experiment: the SAME quantity as exp14's `redundancy`
                         (-3.13pp, worst of eight) evaluated as a SET objective under a
                         SET SEARCH instead of a marginal top-k. If it clears the floor
                         here where the marginal version lost by 3pp, "it is a set
                         property, not a per-feature one" is demonstrated by a
                         controlled comparison rather than argued.
      loading_sum    M5  sum lambda_i.  The signal half of M1 with the contamination
                         denominator removed. M4 and M5 bracket M1, so a win for M1 can
                         be attributed to one half or the other.

    CONTROLS, outside the multiplicity family:
      oracle_auroc       THE POWER GATE. The same greedy, same size, but J(S) = the
                         in-sample AUROC of the U-PCR fit on half A. It tests the SEARCH
                         PROCEDURE, not an objective: if greedy backward at fixed k
                         cannot beat the floor even when handed labels, then no
                         label-free objective running through it could, and NO NEGATIVE
                         CONCLUSION IS DRAWN. (Step 213's premise-test rule.)
      random_prune       the matched floor, replayed bit-identically from exp13.
      good set           exp12's label-guided ceiling, quoted from its CSV.

    DECISION RULE, fixed in advance:
      PRIMARY    held-out AUROC of S*, paired over the 24 test sets, against the MATCHED
                 PRUNING FLOOR at the same size. Positioned against the room, +2.25pp.
      MULTIPLICITY  FLAT Holm-Bonferroni over the FIVE label-free arms -- flat, not a
                 designated-primary family, for consistency with exp14's flat correction
                 over its six. Best arm needs p < 0.010. Per-arm CIs are reported beside
                 the adjusted p-values and never replaced by them: a marginal effect can
                 fail Holm here and still be real.
      SECONDARY  overlap with exp12's good set against a COMPOSITION-MATCHED null. Per
                 exp14, an arm that clears only the overlap test has reproduced the old
                 outcome, not beaten it -- and there the arm CLOSEST to the null scored
                 best, so overlap excess and performance are always read together.
      SIZE       the two self-sizing arms are run BOTH at the good set's own k (matched
                 to the floor, so the primary is comparable) and in FREE-SIZE mode (the
                 deployable form). Free-size cannot be compared to a fixed-k floor and
                 is reported against the deployed keep set instead.

    M1's NAMED FAILURE MODE, registered before the run. The sparsity prior breaks
    omega's duplicate degeneracy only while a duplicate block fits the budget: a block
    of b views costs C(b,2) pairs against sparsity*C(m,2), so omega is fooled once
    b >~ m*sqrt(sparsity) (~0.32m at the 10% default). Measured exactly at m=12: blocks
    of 2-3 penalised, 4+ fooled. Our pool contains by-construction near-duplicate
    TRIPLES (epr/epr_spilled/epr_energy, the sw_var_peak triple, the cusum_max triple).
    PREDICTION: if M1 fails, it fails by selecting HIGH-cohesion sets. `r_diag` reports
    each arm's selected-set cohesion against the floor's, which is the direct check.

    A NULL RESULT IS A RESULT. If no arm clears the floor while the power gate passes,
    the closure extends from per-feature to set-level and the item closes with a ceiling
    and an impossibility statement -- the same shape of deliverable as exp14's, and
    stronger. That is written here so it cannot be renegotiated afterwards.

REPRODUCTION
    The splits are exp12's, replayed through exp13's machinery exactly as exp14 does:
    same per-cell crc32 seed, same replayed random consumption, `nrng` draws in the same
    order. ALL new randomness is on a SEPARATE generator (`srng` for tie-breaks, `orng`
    for the overlap nulls), so exp13's and exp14's arms are invariant to what is added
    here by construction rather than by ordering discipline. The run asserts, per split
    and to 1e-9, that the deployed AUROC and the pruning floor match exp13's splits.csv
    before any new number is trusted.

Run:  python scripts/upcr_study/exp15_composite_reliability.py
      python scripts/upcr_study/exp15_composite_reliability.py --cells 3   (pilot)
Out:  results/upcr_study/15_composite_reliability/
"""
import argparse
import csv
import os
import sys
import time
import zlib

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                    # noqa: E402

from exp10_channel_ceilings import (                                  # noqa: E402
    FIT, N_SPLITS, derive_cell, derived_arm_gate, fit_cols, _zsc, _all_fields,
)
from exp13_incumbent_anchored_ranking import (                        # noqa: E402
    N_RANDOM, N_OVERLAP_DRAWS, TOL_REPRO,
    prune_incumbent, cond_overlap_null, load_exp12_splits,
)
from spectral_utils.subset_sweep import ALL_SIGNS                     # noqa: E402
from spectral_utils.upcr import upcr_fit                              # noqa: E402
from spectral_utils.composite_reliability import (                    # noqa: E402
    OBJECTIVES, greedy_backward, fit_sparse_factor, delta_structure,
    DEFAULT_SPARSITY, MIN_SET,
)

EXP13_DIR = "13_incumbent_anchored_ranking"
OUT_DIR = "15_composite_reliability"

LABEL_FREE = ("omega_sparse", "resid_dep", "m_eff", "cohesion_set", "loading_sum")
SELF_SIZING = ("omega_sparse", "m_eff")
ROOM_PP = 2.25
FLOOR_PP = -0.84


def _check(ck, rep, name, got, want):
    if not np.isfinite(got) and not np.isfinite(want):
        return
    if abs(got - want) > TOL_REPRO:
        raise RuntimeError(
            f"{ck} rep {rep}: {name} = {got:.12f} != exp13's {want:.12f} — a stream "
            "this script must not touch has moved")


def load_exp13_splits():
    path = os.path.join(S.outdir(EXP13_DIR), "splits.csv")
    with open(path, newline="") as f:
        return {(r["cell"], int(r["rep"])): r for r in csv.DictReader(f)}


def cohesion_of(C, cols):
    cols = sorted(int(c) for c in cols)
    if len(cols) < 2:
        return np.nan
    Cs = C[np.ix_(cols, cols)]
    iu = np.triu_indices(len(cols), 1)
    return float(np.mean(np.abs(Cs[iu])))


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def greedy_oracle(ca, start, target_k, tiebreak, min_set=MIN_SET):
    """The same backward greedy as `greedy_backward`, but scoring each candidate by the
    in-sample AUROC of an actual U-PCR fit on half A. Kept separate because it needs the
    data matrix, not just the covariance."""
    cur = sorted(int(c) for c in start)
    floor_k = max(min_set, int(target_k))
    while len(cur) > floor_k:
        best_key, drop = None, None
        for j in cur:
            cand = [c for c in cur if c != j]
            a = fit_cols(ca, cand, exclusion=False)[0]
            if not np.isfinite(a):
                continue
            key = (a, float(tiebreak[j]))
            if best_key is None or key > best_key:
                best_key, drop = key, j
        if drop is None:
            break
        cur = [c for c in cur if c != drop]
    return sorted(cur)


# ---------------------------------------------------------------------------
def run_cell(ck, cell, dcell, rng, nrng, srng, orng, prev12, prev13, args):
    V0 = cell["V"]
    y = dcell["labels"]
    n, m = V0.shape
    hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
    pool = cell["pool"]

    split_rows = []
    for rep in range(N_SPLITS):
        # --- exp12's split, reproduced exactly (identical to exp13/exp14) ----------
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
            raise RuntimeError(f"{ck} rep {rep}: no exp12/exp13 row — splits diverged")
        k = int(ref12["k"])
        cols_greedy = [pool.index(f) for f in ref12["greedy_cols"].split("|")]

        auc_dep = fit_cols(cb, range(m))[0]
        _check(ck, rep, "auroc_deployed", auc_dep, float(ref13["auroc_deployed"]))

        Va = ca["V"]
        C_a = (Va.T @ Va) / Va.shape[0]
        tb = srng.random(m)                       # one tie-break draw per split

        row = {"cell": ck, "rep": rep, "m": m, "k": k,
               "n_keep_deployed_halfA": len(start_a),
               "auroc_deployed": auc_dep,
               "auroc_greedy": float(ref12["auroc_greedy"]),
               "auroc_keepset": fit_cols(cb, start_a, exclusion=False)[0]}

        # --- the sparsity prior, re-checked on THIS cell ----------------------------
        fit_keep = fit_sparse_factor(C_a[np.ix_(start_a, start_a)],
                                     sparsity=args.sparsity)
        top10, r1 = delta_structure(fit_keep)
        row["delta_top10_share"] = top10
        row["delta_rank1_share"] = r1
        row["delta_nnz"] = fit_keep.n_nonzero
        row["cohesion_keepset"] = cohesion_of(C_a, start_a)
        row["cohesion_good"] = cohesion_of(C_a, cols_greedy)

        # --- the five label-free arms, FIXED k (comparable to the floor) ------------
        sel = {}
        for name in LABEL_FREE:
            fn, _, needs = OBJECTIVES[name]
            cols, jval, steps = greedy_backward(
                C_a, start_a, fn, sparsity=args.sparsity, target_k=k,
                tiebreak=tb, needs_fit=needs)
            sel[name] = cols
            row[f"auroc_{name}"] = fit_cols(cb, cols, exclusion=False)[0]
            row[f"{name}_cols"] = "|".join(pool[c] for c in cols)
            row[f"overlap_{name}"] = len(set(cols) & set(cols_greedy)) / k
            row[f"cohesion_{name}"] = cohesion_of(C_a, cols)
            row[f"J_{name}"] = jval
            row[f"steps_{name}"] = steps

        # --- the two self-sizing arms in FREE-SIZE mode (the deployable form) -------
        for name in SELF_SIZING:
            fn, _, needs = OBJECTIVES[name]
            cols, jval, steps = greedy_backward(
                C_a, start_a, fn, sparsity=args.sparsity, target_k=None,
                tiebreak=tb, needs_fit=needs)
            row[f"auroc_{name}_free"] = fit_cols(cb, cols, exclusion=False)[0]
            row[f"{name}_free_size"] = len(cols)
            row[f"{name}_free_cols"] = "|".join(pool[c] for c in cols)
            row[f"cohesion_{name}_free"] = cohesion_of(C_a, cols)

        # --- THE POWER GATE: the same search, handed labels -------------------------
        cols_or = greedy_oracle(ca, start_a, k, tb)
        row["auroc_oracle_auroc"] = fit_cols(cb, cols_or, exclusion=False)[0]
        row["overlap_oracle_auroc"] = len(set(cols_or) & set(cols_greedy)) / k
        row["cohesion_oracle_auroc"] = cohesion_of(C_a, cols_or)
        sel["oracle_auroc"] = cols_or

        # --- exp13's floor, replayed on `nrng` in exp13's own order -----------------
        rnd, coh_rnd = [], []
        for _ in range(N_RANDOM):
            order = [int(j) for j in nrng.permutation(m)]
            cols_r = prune_incumbent(start_a, order, k, m)
            rnd.append(fit_cols(cb, cols_r, exclusion=False)[0])
            coh_rnd.append(cohesion_of(C_a, cols_r))
        row["auroc_random_prune"] = float(np.nanmean(rnd))
        row["cohesion_random_prune"] = float(np.nanmean(coh_rnd))
        row["n_random_prune_failed"] = int(np.sum(~np.isfinite(rnd)))
        _check(ck, rep, "auroc_random_prune", row["auroc_random_prune"],
               float(ref13["auroc_random_prune"]))

        # exp13 draws its own two conditional nulls from `nrng` AFTER the floor, so the
        # stream must be advanced identically or the next split's floor diverges. Caught
        # by the assert above on rep 1 of the first cell. Replaying them also re-checks
        # both values, which is a free extra gate — exp14 does the same.
        for name in ("truecorr", "rho"):
            cols_rebuilt = [pool.index(f) for f in ref12[f"{name}_cols_ownk"].split("|")]
            got = cond_overlap_null(cols_greedy, cols_rebuilt, start_a, m, k, nrng)
            _check(ck, rep, f"overlap_{name}_conditional_null", got,
                   float(ref13[f"overlap_{name}_conditional_null"]))

        # --- the composition-matched overlap null ----------------------------------
        s = set(start_a)
        null_cache = {}
        for name in list(LABEL_FREE) + ["oracle_auroc"]:
            cols = sel[name]
            row[f"frac_inside_{name}"] = float(np.mean([c in s for c in cols]))
            comp = sum(1 for j in cols if int(j) in s)
            if comp not in null_cache:
                null_cache[comp] = cond_overlap_null(
                    cols_greedy, cols, start_a, m, k, orng)
            row[f"cond_null_{name}"] = null_cache[comp]
        row["frac_inside_greedy"] = float(np.mean([c in s for c in cols_greedy]))

        # --- are the arms actually different objects? (exp14's check) ---------------
        js = [jaccard(sel[a], sel[b_])
              for i, a in enumerate(LABEL_FREE) for b_ in LABEL_FREE[i + 1:]]
        row["mean_pairwise_jaccard"] = float(np.mean(js))
        row["n_to_drop"] = len(start_a) - k

        # --- exp12's random consumption, replayed so the next split lines up --------
        for _ in range(2 * N_OVERLAP_DRAWS + N_RANDOM):
            rng.choice(m, k, replace=False)

        split_rows.append(row)

    return split_rows


def collapse(split_rows):
    out = []
    for ck in dict.fromkeys(r["cell"] for r in split_rows):
        rs = [r for r in split_rows if r["cell"] == ck]
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
    if len(d) == 0:
        return {"mean_delta": float("nan"), "n": 0}
    g = np.random.default_rng(0)
    bs = [float(np.mean(g.choice(d, len(d), replace=True))) for _ in range(20000)]
    out = {"mean_delta": float(d.mean()),
           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": int(len(d)),
           "p": float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0)
                else float("nan")}
    if not quiet:
        unit = "pp" if scale == 100.0 else ""
        print(f"  {label:46s} {out['mean_delta']:+7.2f}{unit}  "
              f"[{out['ci95'][0]:+.2f}, {out['ci95'][1]:+.2f}]  "
              f"{out['wins']:2d}W/{out['losses']:2d}L  p={out['p']:.4f}")
    return out


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, p * (n - i)))
        out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=None, help="pilot on the first N cells")
    ap.add_argument("--sparsity", type=float, default=DEFAULT_SPARSITY)
    args = ap.parse_args()

    out = S.outdir(OUT_DIR)
    cells = S.load()
    S.validity_check(cells)
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    derived_arm_gate(dcells)
    prev12, prev13 = load_exp12_splits(), load_exp13_splits()

    keys = list(cells)[: args.cells] if args.cells else list(cells)
    split_rows, t0 = [], time.time()
    for i, ck in enumerate(keys, 1):
        rng = np.random.default_rng(zlib.crc32(ck.encode()) % (2 ** 32))
        nrng = np.random.default_rng(0xC0FFEE + (zlib.crc32(ck.encode()) % 10 ** 6))
        # NEW streams, separate from exp13's and exp14's, so their arms are invariant
        # to anything added here by construction rather than by ordering discipline.
        srng = np.random.default_rng(0x15C09E + (zlib.crc32(ck.encode()) % 10 ** 6))
        orng = np.random.default_rng(0x15011F + (zlib.crc32(ck.encode()) % 10 ** 6))
        rs = run_cell(ck, cells[ck], dcells[ck], rng, nrng, srng, orng,
                      prev12, prev13, args)
        split_rows += rs
        if rs:
            f = lambda key: float(np.nanmean([r[key] for r in rs]))   # noqa: E731
            fl = f("auroc_random_prune")
            print(f"[{i:2d}/{len(keys)}] {S.plain_cell(ck)[:26]:26s} vs floor: "
                  f"good {(f('auroc_greedy')-fl)*100:+5.2f} | oracle "
                  f"{(f('auroc_oracle_auroc')-fl)*100:+5.2f} | omega "
                  f"{(f('auroc_omega_sparse')-fl)*100:+5.2f} | m_eff "
                  f"{(f('auroc_m_eff')-fl)*100:+5.2f} | coh "
                  f"{(f('auroc_cohesion_set')-fl)*100:+5.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    S.save_csv(os.path.join(out, "splits.csv"), split_rows, _all_fields(split_rows))
    rows = collapse(split_rows)
    S.save_csv(os.path.join(out, "per_cell.csv"), rows, _all_fields(rows))

    summary = {"n_cells": len(rows), "n_splits": len(split_rows),
               "sparsity": args.sparsity, "is_partial_run": bool(args.cells),
               "label_free_arms": list(LABEL_FREE),
               "PROVENANCE": ("Pre-registered in this file's docstring before the run. "
                              "Splits are exp12's; exp13's arms asserted per split to "
                              f"{TOL_REPRO}.")}

    print("\n" + "=" * 84)
    print("THE BAR — re-derived on these rows, not quoted")
    summary["room"] = paired(rows, "auroc_random_prune", "auroc_greedy",
                             "the good set (the room)")
    summary["floor_vs_deployed"] = paired(rows, "auroc_deployed", "auroc_random_prune",
                                          "the matched pruning floor vs deployed")
    print(f"  expected: room {ROOM_PP:+.2f}pp, floor {FLOOR_PP:+.2f}pp")

    print("\nPOWER GATE — the SEARCH handed labels, same size, vs the floor")
    pg = summary["power_gate"] = paired(rows, "auroc_random_prune",
                                        "auroc_oracle_auroc",
                                        "greedy backward on in-sample AUROC")
    gate_ok = pg["mean_delta"] > 0 and pg["ci95"][0] > 0
    summary["power_gate_passed"] = bool(gate_ok)
    print(f"  -> {'PASS' if gate_ok else 'FAIL — NO NEGATIVE CONCLUSION MAY BE DRAWN'}")

    print("\nPRIMARY — each arm vs the matched pruning floor, at the same size")
    prim, pv = {}, {}
    for name in LABEL_FREE:
        prim[name] = paired(rows, "auroc_random_prune", f"auroc_{name}", name)
        pv[name] = prim[name]["p"]
    summary["primary_vs_floor"] = prim
    summary["primary_holm_adjusted_p"] = holm(pv)
    print("\n  Holm-adjusted (flat over the five):")
    for name, p in sorted(summary["primary_holm_adjusted_p"].items(), key=lambda x: x[1]):
        print(f"    {name:16s} {p:.4f}")
    summary["fraction_of_room"] = {
        n: prim[n]["mean_delta"] / summary["room"]["mean_delta"] for n in LABEL_FREE}

    print("\nFREE-SIZE (deployable form) — vs the DEPLOYED keep set, not the floor")
    fs = summary["free_size"] = {}
    for name in SELF_SIZING:
        fs[name] = paired(rows, "auroc_keepset", f"auroc_{name}_free", f"{name} (free)")
        sz = float(np.nanmean([r[f"{name}_free_size"] for r in rows]))
        fs[name]["mean_size"] = sz
        print(f"       -> mean size {sz:.1f} vs deployed keep set "
              f"{np.mean([r['n_keep_deployed_halfA'] for r in rows]):.1f}, "
              f"good set {np.mean([r['k'] for r in rows]):.1f}")

    print("\nSECONDARY — overlap with the good set, vs the composition-matched null")
    sec = summary["secondary_overlap"] = {}
    for name in list(LABEL_FREE) + ["oracle_auroc"]:
        sec[name] = paired(rows, f"cond_null_{name}", f"overlap_{name}",
                           f"{name}: excess over the null", scale=1.0)

    print("\nDIAGNOSTICS — is the sparsity prior holding, and is any arm just cohesion?")
    dg = summary["diagnostics"] = {}
    dg["delta_top10_share"] = float(np.nanmean([r["delta_top10_share"] for r in rows]))
    dg["delta_rank1_share"] = float(np.nanmean([r["delta_rank1_share"] for r in rows]))
    print(f"  Delta top-decile mass share {dg['delta_top10_share']:.3f} "
          f"(uniform 0.10) | leading eigenvalue share {dg['delta_rank1_share']:.3f}")
    print("  selected-set cohesion, each arm minus the floor's "
          "(M1's failure mode is POSITIVE here):")
    for name in list(LABEL_FREE) + ["oracle_auroc"]:
        dg[f"cohesion_excess_{name}"] = paired(
            rows, "cohesion_random_prune", f"cohesion_{name}",
            f"    {name}", scale=1.0)["mean_delta"]
    dg["good_set_cohesion_excess"] = paired(
        rows, "cohesion_random_prune", "cohesion_good",
        "    the good set (the target)", scale=1.0)["mean_delta"]
    dg["mean_pairwise_jaccard"] = float(np.nanmean(
        [r["mean_pairwise_jaccard"] for r in rows]))
    dg["mean_n_to_drop"] = float(np.nanmean([r["n_to_drop"] for r in rows]))
    print(f"  mean pairwise Jaccard between the five arms "
          f"{dg['mean_pairwise_jaccard']:.3f} (1.0 = the same set) | "
          f"mean features to drop {dg['mean_n_to_drop']:.1f}")
    nfail = int(np.sum([r["n_random_prune_failed"] for r in split_rows]))
    summary["n_random_prune_fits_failed"] = nfail
    print(f"  floor fits that failed and were dropped: {nfail}")

    S.save_json(os.path.join(out, "summary.json"), summary)
    print(f"\nWrote -> {out}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
