"""
Experiment 3 - Does size have a sweet spot, and does the fit score find it?

Two questions the whole trimming idea rests on:

  Q1. As you keep more measurements, does accuracy rise, fall, or peak somewhere
      in the middle? If it only rises, no "stop where the curve turns" rule can
      ever work, and the size has to come from somewhere else.
  Q2. Does the fit score - the quantity the trimming algorithm steers by -
      actually predict which combinations are good? Compared WITHIN a size, so
      every comparison is between combinations holding the same number of
      measurements.

PART 1 first audits an exhaustive sweep already on disk (about 1.03 million
scored combinations) to see whether it can be reused for free. IT CANNOT - only
5 of 19 test sets still reproduce, so the cache is reported as a finding and
then set aside.

PART 2 therefore answers both questions with a fresh sample computed on the
current 30-measurement pool.

Writes results/pruning_study/03_size_and_criterion/
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402

SWEEP = os.path.join(S.REPO, "results", "subset_sweep")
N_STALENESS = 60
SIZES = [3, 4, 6, 8, 11, 14, 17, 21, 25, 30]
N_DRAWS = 30
SEED = 0


def staleness_audit(cells):
    rng = np.random.default_rng(SEED)
    rows = []
    for f in sorted(glob.glob(os.path.join(SWEEP, "repgrid__*.npz"))):
        ck = os.path.basename(f).replace("repgrid__", "").replace(".npz", "")
        man = f.replace(".npz", ".manifest.json")
        if ck not in cells or not os.path.exists(man):
            continue
        pool = json.load(open(man, encoding="utf-8"))["pool"]
        cell = cells[ck]
        d = np.load(f, allow_pickle=True)
        mask, auroc = d["mask"], d["auroc"]
        ok = np.isfinite(auroc)
        idx = rng.choice(np.where(ok)[0],
                         size=min(N_STALENESS, int(ok.sum())), replace=False)
        diffs = []
        for i in idx:
            names = [pool[b] for b in range(len(pool)) if (int(mask[i]) >> b) & 1]
            cols = [cell["pool"].index(n) for n in names if n in cell["pool"]]
            if len(cols) != len(names) or len(cols) < 3:
                continue
            live = S.fuse_score(cell, cols)
            if np.isfinite(live):
                diffs.append(abs(live - float(auroc[i])))
        if diffs:
            mx = float(np.max(diffs))
            rows.append({"test_set": S.plain_cell(ck), "test_set_code": ck,
                         "n_rechecked": len(diffs), "max_abs_difference": mx,
                         "median_abs_difference": float(np.median(diffs)),
                         "verdict": "matches" if mx < 1e-3 else "stale"})
            print(f"  {ck[:34]:34s} max diff={mx:.2e}  "
                  f"{'MATCHES' if mx < 1e-3 else 'STALE'}")
    return rows


def live_sweep(cells):
    rng = np.random.default_rng(SEED)
    rows = []
    for ck, cell in cells.items():
        p = len(cell["pool"])
        for size in SIZES:
            if size > p:
                continue
            draws = ([list(range(p))] if size >= p else
                     [sorted(rng.choice(p, size=size, replace=False))
                      for _ in range(N_DRAWS)])
            for cols in draws:
                auc, meta = S.fuse_meta(cell, cols)
                rows.append({
                    "test_set_code": ck, "size": size, "auroc": auc,
                    "fit_misfit": float(meta["residual"]),
                    "n_groups": int(meta["K"]),
                    "measurements_code": ",".join(cell["pool"][i] for i in cols),
                })
        print(f"  {ck[:34]:34s} done")
    return rows


def main():
    out = S.outdir("03_size_and_criterion")
    cells = S.load()
    S.validity_check(cells)

    print("\nPART 1 - can the cached exhaustive sweep be reused?")
    stale = staleness_audit(cells)
    S.save_csv(os.path.join(out, "cache_staleness_audit.csv"), stale)
    n_ok = sum(1 for r in stale if r["verdict"] == "matches")
    worst = max(r["max_abs_difference"] for r in stale)
    print(f"  {n_ok}/{len(stale)} test sets still reproduce; worst diff {worst:.2e}")

    print("\nPART 2 - fresh sample on the current 30-measurement pool")
    raw = live_sweep(cells)
    S.save_csv(os.path.join(out, "sampled_combinations.csv"), raw)
    S.save_npz(os.path.join(out, "sampled_combinations.npz"),
               test_set=np.array([r["test_set_code"] for r in raw]),
               size=np.array([r["size"] for r in raw]),
               auroc=np.array([r["auroc"] for r in raw], float),
               fit_misfit=np.array([r["fit_misfit"] for r in raw], float),
               n_groups=np.array([r["n_groups"] for r in raw]),
               measurements=np.array([r["measurements_code"] for r in raw]))

    # ---------------- aggregate ----------------
    by_cell = {}
    for r in raw:
        by_cell.setdefault(r["test_set_code"], []).append(r)

    curve_rows, crit_rows = [], []
    for ck, rs in by_cell.items():
        sz = np.array([r["size"] for r in rs])
        au = np.array([r["auroc"] for r in rs], float)
        fit = np.array([r["fit_misfit"] for r in rs], float)
        sizes = np.unique(sz)
        mean_c = np.array([np.nanmean(au[sz == s]) for s in sizes])
        max_c = np.array([np.nanmax(au[sz == s]) for s in sizes])
        within = []
        for s in sizes:
            k = (sz == s) & np.isfinite(au) & np.isfinite(fit)
            if k.sum() > 10 and np.ptp(fit[k]) > 0:
                r_ = spearmanr(fit[k], au[k]).statistic
                if np.isfinite(r_):
                    within.append(r_)
                    crit_rows.append({"test_set": S.plain_cell(ck),
                                      "test_set_code": ck, "size": int(s),
                                      "n_combinations": int(k.sum()),
                                      "correlation_fit_vs_accuracy": float(r_)})
        curve_rows.append({
            "test_set": S.plain_cell(ck), "test_set_code": ck,
            "best_size_for_typical_combination": int(sizes[int(np.nanargmax(mean_c))]),
            "best_size_for_best_combination": int(sizes[int(np.nanargmax(max_c))]),
            "typical_at_smallest": float(mean_c[0]),
            "typical_at_largest": float(mean_c[-1]),
            "best_found": float(np.nanmax(max_c)),
            "mean_correlation_fit_vs_accuracy": float(np.mean(within)) if within else np.nan,
        })
    S.save_csv(os.path.join(out, "size_curve_per_test_set.csv"), curve_rows)
    S.save_csv(os.path.join(out, "criterion_power_per_size.csv"), crit_rows)

    all_sizes = SIZES
    # NOT every cell reaches every size: pool sizes run 27-30, and the sampling
    # loop skips `size > p`, so size 30 exists on only 6 of 25 cells. Aggregating
    # over `by_cell` unconditionally hands np.nanmax an EMPTY list, which raises
    # (nanmean merely warns and returns NaN, which is why the crash landed here
    # and not one line earlier). Aggregate over the cells that actually have rows
    # at that size, and record the coverage so a thin size cannot be read as if
    # it were measured on all 25. Found Step 205; pre-existing.
    def _per_size(reduce_fn):
        out = []
        for s in all_sizes:
            vals = [reduce_fn(v) for v in
                    ([r["auroc"] for r in by_cell[c] if r["size"] == s]
                     for c in by_cell) if v]
            out.append(float(np.nanmean(vals)) if vals else float("nan"))
        return out

    size_coverage = [sum(1 for c in by_cell
                         if any(r["size"] == s for r in by_cell[c]))
                     for s in all_sizes]
    mean_of_means = _per_size(np.nanmean)
    mean_of_max = _per_size(np.nanmax)
    thin = [(s, n) for s, n in zip(all_sizes, size_coverage) if n < len(by_cell)]
    if thin:
        print("  NOTE: sizes not measured on all "
              f"{len(by_cell)} cells (pool sizes differ): "
              + ", ".join(f"size {s} on {n}" for s, n in thin))

    corrs = np.array([r["mean_correlation_fit_vs_accuracy"] for r in curve_rows], float)
    rising = sum(1 for r in curve_rows
                 if r["typical_at_largest"] > r["typical_at_smallest"])
    big_best = sum(1 for r in curve_rows if r["best_size_for_typical_combination"] >= 14)
    small_best_of_best = sum(1 for r in curve_rows if r["best_size_for_best_combination"] <= 8)

    curve_chart = S.line_chart(
        [("A typical combination of that size", all_sizes, mean_of_means),
         ("The BEST combination found at that size", all_sizes, mean_of_max)],
        "Number of measurements kept", "Detection accuracy (AUROC)",
        hlines=[("six hand-picked = 0.7594", 0.7594)])
    corr_chart = S.bar_chart(
        [r["test_set"] for r in sorted(curve_rows, key=lambda r: r["mean_correlation_fit_vs_accuracy"])],
        [r["mean_correlation_fit_vs_accuracy"] for r in sorted(curve_rows, key=lambda r: r["mean_correlation_fit_vs_accuracy"])],
        "Correlation between the fit score and real accuracy, within one size",
        value_fmt="{:+.3f}", bar_h=21)

    stale_tbl = S.html_table(
        ["Test set", "Combinations re-scored", "Largest disagreement", "Verdict"],
        [[r["test_set"], r["n_rechecked"], f"{r['max_abs_difference']:.2e}",
          r["verdict"].upper()] for r in sorted(stale, key=lambda r: r["max_abs_difference"])],
        numeric_cols=(1, 2))
    curve_tbl = S.html_table(
        ["Test set", "Best size (typical combination)", "Best size (best combination)",
         "Typical at 3", "Typical at 30", "Best found", "Fit-score correlation"],
        [[r["test_set"], r["best_size_for_typical_combination"],
          r["best_size_for_best_combination"], f"{r['typical_at_smallest']:.4f}",
          f"{r['typical_at_largest']:.4f}", f"{r['best_found']:.4f}",
          f"{r['mean_correlation_fit_vs_accuracy']:+.3f}"]
         for r in sorted(curve_rows, key=lambda r: -r["best_found"])],
        numeric_cols=(1, 2, 3, 4, 5, 6))

    body = f"""
<h2>Part 1 - The cached sweep cannot be reused</h2>
<p>An exhaustive sweep run earlier in the project scored <b>every</b> combination
of an older 16-measurement pool for 19 test sets &mdash; about 1.03 million
combinations. Reusing it would have answered both questions below for free, so
the first step was to check whether those cached scores still reproduce today.</p>
{stale_tbl}
<div class="warn"><b>Only {n_ok} of {len(stale)} test sets still reproduce.</b>
The largest disagreement is <b>{worst:.2f} AUROC</b> &mdash; far too large to be
rounding. The test sets themselves have been re-graded since the sweep was run,
so the cached accuracies describe data we no longer have. <b>The cache is set
aside and everything below is computed fresh.</b> This is worth recording in its
own right: about a million scored combinations are no longer usable, and any
future analysis that reaches for them needs this same check first.</div>

<h2>Part 2 - Fresh measurements on the current 30-measurement pool</h2>
<p>{N_DRAWS} random combinations per size per test set, at sizes
{", ".join(str(s) for s in SIZES)}, scored through exactly the path our headline
numbers use.</p>

<h3>Question 1 - Is there a sweet spot in size?</h3>
{curve_chart}
<p class="note">Two different questions in one picture. The lower line is how
good a <em>typical</em> combination of each size is. The upper line is the best
combination <em>found</em> at that size among those sampled.</p>
{curve_tbl}
<ul>
<li><b>For a typical combination, keeping more is better.</b> In
{rising}/{len(curve_rows)} test sets a typical large combination beats a typical
small one, and in {big_best}/{len(curve_rows)} the best typical size is 14 or
more.</li>
<li><b>But the best combinations found are small</b> &mdash; in
{small_best_of_best}/{len(curve_rows)} test sets the best sampled combination had
8 measurements or fewer, and the upper line sits well above the lower one.</li>
<li><b>Together these say: trimming has a high ceiling and a poor average.</b>
Cutting measurements at random makes things worse; cutting the <em>right</em>
ones makes things much better. All of the value is in choosing well, none of it
in being small.</li>
<li><b>Consequence for the algorithm:</b> a rule of the form "trim while the
curve improves and stop at the turn" has no turn to find on the typical curve.
The size has to be fixed in advance or set by an error-control threshold, not
discovered by following a curve.</li>
</ul>

<h3>Question 2 - Does the fit score know which combinations are good?</h3>
{corr_chart}
<ul>
<li>Mean correlation across test sets: <b>{np.nanmean(corrs):+.3f}</b>
(median {np.nanmedian(corrs):+.3f}), positive in
<b>{int((corrs > 0).sum())}/{len(corrs)}</b> test sets.</li>
<li><b>Used as a score for ranking whole combinations, the fit score carries
little usable information</b>, and its direction is not consistent across test
sets, so it cannot be applied with a fixed sign.</li>
<li><b>Limit of this conclusion.</b> This tests one particular use of the fit
score: ranking entire combinations against each other. It says nothing about the
fit score used as a <em>localizer</em> &mdash; deciding which group to repair &mdash;
which is what Experiment 2 tests. Different quantity, different job.</li>
</ul>

<h2>Saved data</h2>
<ul>
<li><code>cache_staleness_audit.csv</code> &mdash; per test set, whether the old sweep still reproduces</li>
<li><code>sampled_combinations.csv</code> / <code>.npz</code> &mdash; every combination sampled,
its exact measurement list, its accuracy, its fit score and its group count</li>
<li><code>size_curve_per_test_set.csv</code>, <code>criterion_power_per_size.csv</code></li>
</ul>
"""
    S.write_page(
        os.path.join(out, "index.html"),
        "Experiment 3 - Is there a right number of measurements?",
        "Whether accuracy peaks at some middle size, and whether the fit score "
        "can find good combinations. Computed fresh on the current 30-measurement pool.",
        [f"<b>The 1.03M-combination cache on disk is unusable</b> - only {n_ok}/{len(stale)} "
         f"test sets still reproduce (worst disagreement {worst:.2f} AUROC). Everything "
         f"here is recomputed fresh.",
         f"<b>No sweet spot.</b> For a typical combination, more measurements is better "
         f"({rising}/{len(curve_rows)} test sets) - so a 'stop where the curve turns' rule "
         f"has nothing to find.",
         "<b>But the best small combinations score far above typical ones</b>: trimming has "
         "a high ceiling and a poor average. The value is entirely in choosing well.",
         f"<b>The fit score barely ranks combinations</b>: mean correlation "
         f"{np.nanmean(corrs):+.3f}, positive in only {int((corrs > 0).sum())}/{len(corrs)} "
         f"test sets. This does not bear on its use as a localizer (Experiment 2)."],
        body)
    print("\nExperiment 3 complete.")


if __name__ == "__main__":
    main()
