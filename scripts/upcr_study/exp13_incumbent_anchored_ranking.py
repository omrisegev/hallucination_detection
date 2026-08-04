"""Does the true correlation fail because it is the wrong quantity, or because the arm
rebuilt the feature set from scratch instead of pruning the one we deploy?

WHY THIS EXISTS
    The deciding test (exp12) compared two things that differ in two ways, not one:
      - the good set  = a greedy search STARTED FROM the deployed keep set (~20.9
        features) that flips down to ~11.8, accepting only improving flips
      - the ranking   = a fresh top-k, built with no reference to the incumbent
    So its -0.66pp confounds "the correlation is the wrong quantity" with "the arm threw
    away the incumbent". This script removes the second explanation by giving the ranking
    the incumbent to start from.

    It also replaces the overlap null. exp12 drew the null uniformly over the whole pool,
    but both rankings live almost entirely inside U-PCR's keep set (94.5% / 98.3%) while
    the keep set is only 73.5% of the pool, so the uniform null is too easy for both.

WHAT IS PRE-REGISTERED, WRITTEN BEFORE THE RUN
    Arms, all scored on held-out half B, paired over the 24 test sets:
      prune by true correlation      from the deployed keep set, drop the lowest
                                     |corr(f, y)| down to the good set's own size k
      prune by estimated correlation same, using U-PCR's own rho_hat
      prune at random                the matched floor: same construction, random order
      (carried over unchanged from exp12: the deployed pool, the good set, the two
       rebuilt-from-scratch top-k arms, the rebuilt-from-scratch random floor)

    READING, fixed in advance:
      - If pruning by the true correlation recovers a majority of the good set's gain
        while the rebuilt version did not, the -0.66pp was the rebuild, not the quantity,
        and the deciding test has to be re-read as favouring "right quantity".
      - If pruning by the true correlation still does not separate from the matched
        pruning floor, the rebuild was not the explanation and the quantity genuinely
        does not convert into performance.
      - The gap between the two pruning arms (true vs estimated correlation) is what an
        improved estimator could at most buy, holding everything else fixed.

    Overlap, same two readings, now against the conditional null: a set drawn with the
    SAME observed inside/outside-keep-set composition as the ranking being tested.

REPRODUCTION
    The splits are exp12's, not new ones: the same per-cell crc32 seed, and exp12's
    random consumption (2 x 2000 overlap draws + 25 floor draws per split) is replayed
    call-for-call so every permutation lands at the same stream position. The run asserts
    the recovered deployed AUROC matches exp12's recorded value on every split before any
    new number is trusted. All new randomness uses a SEPARATE generator.
"""
import os
import sys
import csv
import zlib

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from exp10_channel_ceilings import (                                # noqa: E402
    FIT, N_SPLITS, derive_cell, derived_arm_gate, fit_cols, _zsc, _all_fields,
)
from spectral_utils.subset_sweep import ALL_SIGNS                   # noqa: E402
from spectral_utils.upcr import upcr_fit                            # noqa: E402

N_RANDOM = 25            # matched to exp12's floor
N_OVERLAP_DRAWS = 2000   # matched to exp12's overlap null
EXP12_DIR = "12_what_separates_good_features"
TOL_REPRO = 1e-9         # the splits must be bit-identical, not merely close


def true_corr(V, y):
    """Each feature's TRUE correlation with correctness, on the half it is given."""
    y = np.asarray(y, dtype=float)
    yc = y - y.mean()
    if yc.std() < 1e-12:
        return np.zeros(V.shape[1])
    out = np.zeros(V.shape[1])
    for j in range(V.shape[1]):
        v = V[:, j] - V[:, j].mean()
        d = v.std() * yc.std()
        out[j] = float(v @ yc) / (len(y) * d) if d > 1e-12 else 0.0
    return out


def prune_incumbent(start, order, k, m):
    """Keep k features, preferring the incumbent. `order` ranks features best-first.

    Drops from the incumbent in worst-first order; only if k exceeds the incumbent's own
    size does it reach outside, and then in the same order. This is the incumbent-
    respecting analogue of taking a top-k.
    """
    s = set(int(j) for j in start)
    inside = [j for j in order if j in s]
    if k <= len(inside):
        return sorted(inside[:k])
    outside = [j for j in order if j not in s]
    return sorted(inside + outside[: k - len(inside)])


def by_stat(stat):
    return [int(j) for j in np.argsort(-np.abs(np.asarray(stat, dtype=float)))]


def cond_overlap_null(target, cols, start, m, k, rng, draws=N_OVERLAP_DRAWS):
    """Expected overlap with `target` for a set of the SAME size AND the same observed
    inside/outside-keep-set composition as `cols`. The uniform-over-pool null ignores
    that both rankings sit almost entirely inside the keep set."""
    s = set(int(j) for j in start)
    ins = np.array(sorted(s), dtype=int)
    outs = np.array([j for j in range(m) if j not in s], dtype=int)
    n_in = min(sum(1 for j in cols if int(j) in s), len(ins))
    n_out = min(k - n_in, len(outs))
    n_in = k - n_out                       # if outside is too small, take more inside
    tgt = set(int(j) for j in target)
    vals = np.empty(draws)
    for i in range(draws):
        pick = []
        if n_in > 0:
            pick += [int(x) for x in rng.choice(ins, n_in, replace=False)]
        if n_out > 0:
            pick += [int(x) for x in rng.choice(outs, n_out, replace=False)]
        vals[i] = len(tgt & set(pick)) / k
    return float(vals.mean())


def load_exp12_splits():
    path = os.path.join(S.outdir(EXP12_DIR), "splits.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out[(r["cell"], int(r["rep"]))] = r
    return out


# ---------------------------------------------------------------------------
def run_cell(ck, cell, dcell, rng, nrng, prev):
    V0 = cell["V"]
    y = dcell["labels"]
    n, m = V0.shape
    hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
    pool = cell["pool"]

    split_rows = []
    for rep in range(N_SPLITS):
        # --- exp12's split, reproduced exactly -------------------------------------
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

        ref = prev.get((ck, rep))
        if ref is None:
            raise RuntimeError(f"{ck} rep {rep}: no exp12 row — the splits diverged")
        k = int(ref["k"])
        cols_greedy = [pool.index(f) for f in ref["greedy_cols"].split("|")]

        auc_dep = fit_cols(cb, range(m))[0]
        if abs(auc_dep - float(ref["auroc_deployed"])) > TOL_REPRO:
            raise RuntimeError(
                f"{ck} rep {rep}: deployed AUROC {auc_dep:.12f} != exp12's "
                f"{float(ref['auroc_deployed']):.12f} — the splits diverged")

        corr_a = true_corr(ca["V"], ca["labels"])
        rho_a = np.asarray(res_a.rho_hat_full, dtype=float)

        row = {"cell": ck, "rep": rep, "m": m, "k": k, "n_keep_deployed_halfA": len(start_a),
               "auroc_deployed": auc_dep,
               "auroc_greedy": float(ref["auroc_greedy"]),
               "auroc_truecorr_rebuilt": float(ref["auroc_truecorr_ownk"]),
               "auroc_rho_rebuilt": float(ref["auroc_rho_ownk"]),
               "auroc_random_rebuilt": float(ref["auroc_random_ownk"])}

        # --- the new arms: prune the incumbent instead of rebuilding ---------------
        for name, stat in (("truecorr", corr_a), ("rho", rho_a)):
            cols = prune_incumbent(start_a, by_stat(stat), k, m)
            row[f"auroc_{name}_prune"] = fit_cols(cb, cols, exclusion=False)[0]
            row[f"{name}_prune_cols"] = "|".join(pool[c] for c in cols)
            row[f"overlap_{name}_prune"] = len(set(cols) & set(cols_greedy)) / k

        rnd = []
        for _ in range(N_RANDOM):
            order = [int(j) for j in nrng.permutation(m)]
            rnd.append(fit_cols(cb, prune_incumbent(start_a, order, k, m),
                                exclusion=False)[0])
        row["auroc_random_prune"] = float(np.nanmean(rnd))
        row["n_random_prune_failed"] = int(np.sum(~np.isfinite(rnd)))

        # --- the overlap null, conditioned on U-PCR's keep set ---------------------
        s = set(start_a)
        row["frac_inside_greedy"] = np.mean([c in s for c in cols_greedy])
        for name in ("truecorr", "rho"):
            cols = [pool.index(f) for f in ref[f"{name}_cols_ownk"].split("|")]
            row[f"overlap_{name}_rebuilt"] = float(ref[f"overlap_{name}"])
            row[f"overlap_{name}_uniform_null"] = float(
                ref[f"overlap_{name}_random_baseline"])
            row[f"overlap_{name}_conditional_null"] = cond_overlap_null(
                cols_greedy, cols, start_a, m, k, nrng)
            row[f"frac_inside_{name}"] = float(np.mean([c in s for c in cols]))

        # --- exp12's random consumption, replayed so the next split lines up -------
        for _ in range(2 * N_OVERLAP_DRAWS + N_RANDOM):
            rng.choice(m, k, replace=False)

        split_rows.append(row)

    return split_rows


def collapse(split_rows):
    out = []
    for ck in dict.fromkeys(r["cell"] for r in split_rows):
        rs = [r for r in split_rows if r["cell"] == ck]
        row = {"cell": ck, "n_splits": len(rs)}
        for k in rs[0]:
            if k in ("cell", "rep") or isinstance(rs[0][k], str):
                continue
            vals = [r[k] for r in rs if np.isfinite(float(r[k]))]
            row[k] = float(np.mean(vals)) if vals else np.nan
        out.append(row)
    return out


def paired(rows, base, arm, label, scale=100.0):
    d = np.array([r[arm] - r[base] for r in rows], float) * scale
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(0)
    bs = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(20000)]
    out = {"mean_delta": float(d.mean()),
           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": int(len(d)),
           "p": float(wilcoxon(d).pvalue) if len(d) > 5 else float("nan")}
    unit = "pp" if scale == 100.0 else ""
    print(f"  {label:52s} {out['mean_delta']:+6.2f}{unit}  "
          f"[{out['ci95'][0]:+.2f}, {out['ci95'][1]:+.2f}]  "
          f"{out['wins']}W/{out['losses']}L  p={out['p']:.4f}")
    return out


def main():
    out = S.outdir("13_incumbent_anchored_ranking")
    cells = S.load()
    S.validity_check(cells)
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    derived_arm_gate(dcells)
    prev = load_exp12_splits()

    split_rows = []
    for i, ck in enumerate(cells, 1):
        rng = np.random.default_rng(zlib.crc32(ck.encode()) % (2 ** 32))
        nrng = np.random.default_rng(0xC0FFEE + (zlib.crc32(ck.encode()) % 10 ** 6))
        rs = run_cell(ck, cells[ck], dcells[ck], rng, nrng, prev)
        split_rows += rs
        if rs:
            f = lambda key: float(np.nanmean([r[key] for r in rs]))  # noqa: E731
            d = f("auroc_deployed")
            print(f"[{i:2d}/{len(cells)}] {S.plain_cell(ck)[:32]:32s} "
                  f"good set {(f('auroc_greedy')-d)*100:+5.2f}pp | true corr rebuilt "
                  f"{(f('auroc_truecorr_rebuilt')-d)*100:+5.2f}pp | true corr pruned "
                  f"{(f('auroc_truecorr_prune')-d)*100:+5.2f}pp | pruned floor "
                  f"{(f('auroc_random_prune')-d)*100:+5.2f}pp", flush=True)

    S.save_csv(os.path.join(out, "splits.csv"), split_rows, _all_fields(split_rows))
    rows = collapse(split_rows)
    S.save_csv(os.path.join(out, "per_cell.csv"), rows, _all_fields(rows))

    summary = {"n_cells": len(rows), "n_splits": len(split_rows),
               "PROVENANCE": ("Pre-registered in this file's docstring before the run. "
                              "Splits are exp12's, asserted identical per split.")}

    print("\n=== held out on half B, paired over test sets, vs the deployed full pool ===")
    a = summary["arms_vs_deployed"] = {}
    a["good_set"] = paired(rows, "auroc_deployed", "auroc_greedy",
                           "the good set (the ceiling)")
    a["truecorr_prune"] = paired(rows, "auroc_deployed", "auroc_truecorr_prune",
                                 "true correlation, PRUNING the deployed keep set")
    a["rho_prune"] = paired(rows, "auroc_deployed", "auroc_rho_prune",
                            "estimated correlation, pruning")
    a["random_prune"] = paired(rows, "auroc_deployed", "auroc_random_prune",
                               "random pruning, same size (matched floor)")
    a["truecorr_rebuilt"] = paired(rows, "auroc_deployed", "auroc_truecorr_rebuilt",
                                   "true correlation, REBUILT from scratch")
    a["rho_rebuilt"] = paired(rows, "auroc_deployed", "auroc_rho_rebuilt",
                              "estimated correlation, rebuilt")
    a["random_rebuilt"] = paired(rows, "auroc_deployed", "auroc_random_rebuilt",
                                 "random subsets, rebuilt (exp12's floor)")

    print("\n=== was the rebuild the explanation? each arm vs its own matched floor ===")
    b = summary["arms_vs_matched_floor"] = {}
    b["truecorr_prune_vs_prune_floor"] = paired(
        rows, "auroc_random_prune", "auroc_truecorr_prune",
        "true correlation pruning vs the pruning floor")
    b["truecorr_rebuilt_vs_rebuilt_floor"] = paired(
        rows, "auroc_random_rebuilt", "auroc_truecorr_rebuilt",
        "true correlation rebuilt vs the rebuilt floor")
    b["prune_vs_rebuild_truecorr"] = paired(
        rows, "auroc_truecorr_rebuilt", "auroc_truecorr_prune",
        "pruning minus rebuilding, true correlation")
    b["truecorr_prune_vs_ceiling"] = paired(
        rows, "auroc_greedy", "auroc_truecorr_prune",
        "true correlation pruning vs the good set")
    b["estimator_headroom"] = paired(
        rows, "auroc_rho_prune", "auroc_truecorr_prune",
        "what a perfect estimator buys (true minus estimated, pruning)")

    print("\n=== overlap with the good set, against the CONDITIONAL null ===")
    c = summary["overlap"] = {}
    for name, lab in (("truecorr", "true correlation"), ("rho", "estimated correlation")):
        c[f"{name}_uniform_excess"] = paired(
            rows, f"overlap_{name}_uniform_null", f"overlap_{name}_rebuilt",
            f"{lab}: excess over the UNIFORM null", scale=1.0)
        c[f"{name}_conditional_excess"] = paired(
            rows, f"overlap_{name}_conditional_null", f"overlap_{name}_rebuilt",
            f"{lab}: excess over the CONDITIONAL null", scale=1.0)
        c[f"{name}_frac_inside_keepset"] = float(
            np.mean([r[f"frac_inside_{name}"] for r in rows]))
    c["greedy_frac_inside_keepset"] = float(
        np.mean([r["frac_inside_greedy"] for r in rows]))
    c["keepset_frac_of_pool"] = float(
        np.mean([r["n_keep_deployed_halfA"] / r["m"] for r in rows]))
    print(f"  fraction of each set inside U-PCR's keep set: "
          f"good set {c['greedy_frac_inside_keepset']:.3f} | "
          f"true corr {c['truecorr_frac_inside_keepset']:.3f} | "
          f"est corr {c['rho_frac_inside_keepset']:.3f} | "
          f"keep set is {c['keepset_frac_of_pool']:.3f} of the pool")

    nfail = int(np.sum([r["n_random_prune_failed"] for r in split_rows]))
    summary["n_random_prune_fits_failed"] = nfail
    print(f"\nfailed fits silently dropped from the pruning floor: {nfail}")

    S.save_json(os.path.join(out, "summary.json"), summary)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
