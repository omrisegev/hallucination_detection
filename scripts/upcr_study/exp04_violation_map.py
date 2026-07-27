"""
exp04_violation_map.py — Phase B2, REDESIGNED after review.

WHAT THE FIRST VERSION GOT WRONG
--------------------------------
v1 asked: "do pairs inside an L-SML cluster fit the additive model worse than
pairs across clusters?" Answer: yes, 2.03x, 25/25 cells. That looked like strong
support for the clustered variant.

It was a confound. The additive fit error is essentially a restatement of the pair
correlation — Spearman(fit error, |C_ij|) = 0.870. Spectral clustering groups
correlated features BY DEFINITION, so "same-cluster pairs fit worse" is close to a
tautology. Three controls, all run here:

  1. MATCH ON |C_ij|. Compare same- vs cross-cluster fit error within deciles of
     |C_ij|. If clustering carries information beyond correlation magnitude, the
     gap survives. (Review measured 0.997 — it does not.)
  2. RANDOM PARTITION of the same cluster sizes. Any gap here is pure geometry.
  3. MAGNITUDE-ONLY CLUSTERING — average linkage on 1-|corr|, same K, ignoring
     L-SML entirely. If this separates the violation BETTER than L-SML, then
     L-SML's partition is not the thing doing the work.

The honest question is (1), not the raw ratio. A variant that drops same-cluster
equations is only justified if cluster membership predicts assumption violation
*beyond* what pair correlation already tells you.

Reported at all three loading scales (see common.SCALES) — the partition itself
depends on the scale, so the answer might too.

Run:  python scripts/upcr_study/exp04_violation_map.py
Out:  results/upcr_study/04_violation_map/{per_cell.csv,summary.json,index.html}
"""
import os
import sys
import collections

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.upcr import additive_design, solve_additive     # noqa: E402
from spectral_utils.fusion_utils import detect_dependent_groups     # noqa: E402

N_DECILES = 10
SEED = 0


def magnitude_clusters(C, K, seed=SEED):
    """Cluster features on correlation magnitude alone — no L-SML, no Eq. 15
    score matrix. The control that asks whether L-SML's partition is doing
    anything a trivial one would not."""
    from sklearn.cluster import AgglomerativeClustering
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    D = 1.0 - np.abs(corr)
    np.fill_diagonal(D, 0.0)
    model = AgglomerativeClustering(n_clusters=K, metric="precomputed",
                                    linkage="average")
    return model.fit_predict(D)


def ratio(fit_err, same):
    if not same.any() or not (~same).any():
        return float("nan")
    return float(fit_err[same].mean() / (fit_err[~same].mean() + 1e-12))


def decile_matched_ratio(fit_err, same, absC):
    """Same/cross fit-error ratio computed WITHIN deciles of |C_ij|, then
    averaged over deciles that contain both kinds of pair. This is the control
    that decides whether clustering says anything beyond correlation magnitude."""
    edges = np.quantile(absC, np.linspace(0, 1, N_DECILES + 1))
    edges[-1] += 1e-9
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (absC >= lo) & (absC < hi)
        s, x = m & same, m & (~same)
        if s.sum() >= 3 and x.sum() >= 3:
            ratios.append(fit_err[s].mean() / (fit_err[x].mean() + 1e-12))
    return float(np.mean(ratios)) if ratios else float("nan")


def analyse_cell(cell, ck, scale):
    F = cell["V"].T
    m, n = F.shape
    C = (F @ F.T) / n

    views = [cell["V"][:, i] for i in range(m)]
    K, c, resid, _ = detect_dependent_groups(views, method="residual",
                                             loading_scale=scale)
    c = np.asarray(c)
    sizes = sorted(collections.Counter(c).values())

    A, pairs = additive_design(m)
    b = np.array([C[i, j] for i, j in pairs], float)
    rho = solve_additive(A, b, loss="l2")
    fit_err = np.abs(A @ rho - b)

    d = np.sqrt(np.diag(C))
    absC = np.array([abs(C[i, j]) / (d[i] * d[j] + 1e-12) for i, j in pairs])
    same = np.array([c[i] == c[j] for i, j in pairs])

    # control 2: random partition with the SAME size profile
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(m)
    c_rand = np.empty(m, dtype=int)
    pos = 0
    for gi, sz in enumerate(sizes):
        c_rand[perm[pos:pos + sz]] = gi
        pos += sz
    same_rand = np.array([c_rand[i] == c_rand[j] for i, j in pairs])

    # control 3: magnitude-only clustering at the same K
    try:
        c_mag = magnitude_clusters(C, K)
        same_mag = np.array([c_mag[i] == c_mag[j] for i, j in pairs])
        r_mag = ratio(fit_err, same_mag)
    except Exception:
        r_mag = float("nan")

    p = float("nan")
    if same.sum() >= 5 and (~same).sum() >= 5:
        p = float(mannwhitneyu(fit_err[same], fit_err[~same],
                               alternative="greater").pvalue)

    cross_pairs = [pr for pr, s in zip(pairs, same) if not s]
    if cross_pairs:
        A_off, _ = additive_design(m, cross_pairs)
        sv = np.linalg.svd(A_off, compute_uv=False)
        rank, sigma_min = int((sv > 1e-9).sum()), float(sv[-1])
    else:
        rank, sigma_min = 0, 0.0

    return {
        "cell": ck, "scale": scale, "m": m, "K": int(K),
        "cluster_sizes": str(sizes),
        "n_cross_cluster_pairs": int((~same).sum()),
        "spearman_fiterr_vs_abscorr": float(
            spearmanr(fit_err, absC).correlation),
        "ratio_lsml": ratio(fit_err, same),
        "ratio_lsml_decile_matched": decile_matched_ratio(fit_err, same, absC),
        "ratio_random_partition": ratio(fit_err, same_rand),
        "ratio_magnitude_clustering": r_mag,
        "mannwhitney_p": p,
        "design_rank_cross_only": rank,
        "design_sigma_min_cross_only": sigma_min,
        "identifiable": bool(K >= 3 and rank == m),
    }


def main():
    out = S.outdir("04_violation_map")
    cells = S.load()
    S.validity_check(cells)

    rows = []
    for scale in S.SCALES:
        for ck, cell in cells.items():
            rows.append(analyse_cell(cell, ck, scale))
        print(f"  scale={scale}: {len(cells)} cells done", flush=True)

    S.save_csv(os.path.join(out, "per_cell.csv"), rows)

    summary = {"n_cells": len(cells), "by_scale": {}}
    print("\n" + "=" * 78)
    print("B2 REDESIGNED - does cluster membership predict violation BEYOND "
          "correlation?")
    print("=" * 78)
    print(f"{'scale':>9} {'raw ratio':>10} {'decile-matched':>15} "
          f"{'random':>8} {'magnitude':>10} {'ident':>7}")
    for scale in S.SCALES:
        sub = [r for r in rows if r["scale"] == scale]

        def arr(k):
            return np.array([r[k] for r in sub], float)

        raw, matched = arr("ratio_lsml"), arr("ratio_lsml_decile_matched")
        rnd, mag = arr("ratio_random_partition"), arr("ratio_magnitude_clustering")
        rec = {
            "mean_spearman_fiterr_vs_abscorr": float(
                np.nanmean(arr("spearman_fiterr_vs_abscorr"))),
            "mean_ratio_lsml": float(np.nanmean(raw)),
            "n_cells_raw_above_1": int((raw > 1).sum()),
            "mean_ratio_decile_matched": float(np.nanmean(matched)),
            "n_cells_matched_above_1": int(np.nansum(matched > 1)),
            "mean_ratio_random_partition": float(np.nanmean(rnd)),
            "mean_ratio_magnitude_clustering": float(np.nanmean(mag)),
            "n_identifiable": int(sum(1 for r in sub if r["identifiable"])),
        }
        summary["by_scale"][scale] = rec
        print(f"{scale:>9} {rec['mean_ratio_lsml']:10.2f} "
              f"{rec['mean_ratio_decile_matched']:15.3f} "
              f"{rec['mean_ratio_random_partition']:8.3f} "
              f"{rec['mean_ratio_magnitude_clustering']:10.2f} "
              f"{rec['n_identifiable']:5d}/{len(sub)}")

    corr = np.nanmean([r["spearman_fiterr_vs_abscorr"] for r in rows])
    matched_all = np.array([r["ratio_lsml_decile_matched"] for r in rows], float)
    passed = bool(np.nanmean(matched_all) > 1.25)
    verdict = (
        f"PREMISE SURVIVES THE CONTROL - decile-matched ratio "
        f"{np.nanmean(matched_all):.3f} > 1, so cluster membership predicts "
        f"assumption violation beyond correlation magnitude."
        if passed else
        f"PREMISE DOES NOT SURVIVE - the raw effect is a confound. Fit error IS "
        f"pair correlation (Spearman {corr:.3f}); matched on |C_ij| decile the "
        f"same-vs-cross ratio collapses to {np.nanmean(matched_all):.3f}, and a "
        f"random partition gives "
        f"{np.nanmean([r['ratio_random_partition'] for r in rows]):.3f}. "
        f"L-SML membership adds nothing beyond 'these features are correlated', "
        f"so dropping same-cluster equations has no principled justification. "
        f"Phase D's failure is the empirical confirmation.")
    summary["mean_spearman_fiterr_vs_abscorr"] = float(corr)
    summary["kill_switch_passed"] = passed
    summary["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    S.save_json(os.path.join(out, "summary.json"), summary)
    render(out, rows, summary)
    print(f"\nWrote -> {out}")


def render(out, rows, summary):
    body = []
    body.append("<h2>What changed, and why</h2>")
    body.append(
        "<p>The first version of this page reported that feature pairs inside a "
        "cluster fit the model <b>2.03&times; worse</b> than pairs across "
        "clusters, in 25 of 25 test sets, and called that strong support for the "
        "clustered variant. That was a confound. The fit error is very nearly a "
        "restatement of how correlated the pair is "
        f"(Spearman <b>{summary['mean_spearman_fiterr_vs_abscorr']:.3f}</b>), and "
        "clustering groups correlated things by definition. So the raw comparison "
        "cannot distinguish 'clustering finds the violation' from 'clustering "
        "finds correlation'.</p>")
    body.append(
        "<p>The question that actually matters is whether cluster membership "
        "predicts violation <em>beyond</em> correlation magnitude. Three controls "
        "answer it: matching pairs on their correlation decile, a random "
        "partition with identical cluster sizes, and a clustering built on "
        "correlation magnitude alone.</p>")

    body.append("<h2>The controls</h2>")
    for scale in S.SCALES:
        rec = summary["by_scale"][scale]
        body.append(f"<h3>loading scale: {scale}</h3>")
        body.append(S.bar_chart(
            ["L-SML clusters, raw", "L-SML clusters, matched on |corr| decile",
             "random partition, same sizes", "magnitude-only clustering"],
            [rec["mean_ratio_lsml"], rec["mean_ratio_decile_matched"],
             rec["mean_ratio_random_partition"],
             rec["mean_ratio_magnitude_clustering"]],
            "same-cluster fit error / cross-cluster fit error",
            hlines=[("no difference", 1.0)], value_fmt="{:.2f}"))

    body.append("<h2>Per cell, per scale</h2>")
    body.append(S.html_table(
        ["test set", "scale", "K", "cluster sizes", "raw ratio",
         "decile-matched", "random", "magnitude-only", "usable?"],
        [[S.plain_cell(r["cell"]), r["scale"], r["K"], r["cluster_sizes"],
          f"{r['ratio_lsml']:.2f}", f"{r['ratio_lsml_decile_matched']:.3f}",
          f"{r['ratio_random_partition']:.3f}",
          f"{r['ratio_magnitude_clustering']:.2f}",
          "yes" if r["identifiable"] else "NO"]
         for r in sorted(rows, key=lambda r: (r["scale"], r["cell"]))],
        numeric_cols=(2, 4, 5, 6, 7)))

    S.write_page(
        os.path.join(out, "index.html"),
        "Does clustering find the assumption violation, or just find correlation?",
        "U-PCR study, Phase B2 — redesigned after review exposed the original "
        "comparison as a confound",
        [f"Fit error is essentially pair correlation: Spearman "
         f"<b>{summary['mean_spearman_fiterr_vs_abscorr']:.3f}</b>.",
         f"Raw same-vs-cross ratio "
         f"<b>{summary['by_scale']['complete']['mean_ratio_lsml']:.2f}&times;</b>, "
         f"but matched on correlation decile it collapses to "
         f"<b>{summary['by_scale']['complete']['mean_ratio_decile_matched']:.3f}"
         f"</b>.",
         f"A clustering that ignores L-SML entirely and uses correlation "
         f"magnitude alone separates the violation "
         f"<b>{summary['by_scale']['complete']['mean_ratio_magnitude_clustering']:.2f}"
         f"&times;</b>.",
         f"<b>{summary['verdict']}</b>"],
        "".join(body))


if __name__ == "__main__":
    main()
