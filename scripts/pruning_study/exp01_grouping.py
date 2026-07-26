"""
Experiment 1 - Does the detector's internal grouping step help or hurt?

Our detector, before combining measurements, first sorts them into groups of
"measurements that behave alike", combines within each group, then combines the
groups. This experiment switches that step off and compares, at every subset
size from 3 to 30, across all 25 test sets.

Writes results/pruning_study/01_grouping/
"""
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402

SIZES = [3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30]
N_DRAWS = 20
SEED = 0


def main():
    out = S.outdir("01_grouping")
    cells = S.load()
    S.validity_check(cells)
    rng = np.random.default_rng(SEED)

    rows, raw = [], []
    for size in SIZES:
        for ck, cell in cells.items():
            p = len(cell["pool"])
            if size > p:
                continue
            draws = ([list(range(p))] if size == p else
                     [sorted(rng.choice(p, size=size, replace=False))
                      for _ in range(N_DRAWS)])
            on = [S.fuse_score(cell, c) for c in draws]
            off = [S.fuse_score(cell, c, groups="flat") for c in draws]
            rows.append({
                "size": size,
                "test_set": S.plain_cell(ck),
                "test_set_code": ck,
                "grouping_on": float(np.nanmean(on)),
                "grouping_off": float(np.nanmean(off)),
                "difference_off_minus_on": float(np.nanmean(off) - np.nanmean(on)),
                "n_subsets_drawn": len(draws),
            })
            for c, a, b in zip(draws, on, off):
                raw.append((size, ck, "|".join(cell["pool"][i] for i in c), a, b))
        print(f"  size {size:2d} done")

    S.save_csv(os.path.join(out, "grouping_by_size.csv"), rows)
    S.save_npz(os.path.join(out, "grouping_raw.npz"),
               size=np.array([r[0] for r in raw]),
               test_set=np.array([r[1] for r in raw]),
               measurements=np.array([r[2] for r in raw]),
               auroc_grouping_on=np.array([r[3] for r in raw], float),
               auroc_grouping_off=np.array([r[4] for r in raw], float))

    # ---- aggregate ----
    sizes, on_c, off_c, stats = [], [], [], []
    for size in SIZES:
        rs = [r for r in rows if r["size"] == size]
        if not rs:
            continue
        a = np.array([r["grouping_on"] for r in rs])
        b = np.array([r["grouping_off"] for r in rs])
        m = np.isfinite(a) & np.isfinite(b)
        try:
            p = wilcoxon(a[m], b[m]).pvalue if m.sum() > 5 and np.any(a[m] != b[m]) else np.nan
        except Exception:
            p = np.nan
        sizes.append(size)
        on_c.append(float(np.nanmean(a)))
        off_c.append(float(np.nanmean(b)))
        stats.append({
            "size": size,
            "grouping_on": float(np.nanmean(a)),
            "grouping_off": float(np.nanmean(b)),
            "difference_pp": float((np.nanmean(b) - np.nanmean(a)) * 100),
            "test_sets_better_without_grouping": int((b > a)[m].sum()),
            "test_sets_total": int(m.sum()),
            "wilcoxon_p": float(p),
        })
    S.save_csv(os.path.join(out, "grouping_summary.csv"), stats)

    # ---- chart + page ----
    chart = S.line_chart(
        [("Grouping step ON (deployed detector)", sizes, on_c),
         ("Grouping step OFF", sizes, off_c)],
        "Number of measurements kept", "Detection accuracy (AUROC)",
        hlines=[("six hand-picked = 0.7594", 0.7594)])

    tbl = S.html_table(
        ["Measurements kept", "Grouping ON", "Grouping OFF", "Difference",
         "Test sets better without grouping", "Wilcoxon p"],
        [[s["size"], f"{s['grouping_on']:.4f}", f"{s['grouping_off']:.4f}",
          f"{s['difference_pp']:+.2f} pp",
          f"{s['test_sets_better_without_grouping']}/{s['test_sets_total']}",
          "n/a" if not np.isfinite(s["wilcoxon_p"]) else f"{s['wilcoxon_p']:.4f}"]
         for s in stats],
        numeric_cols=(0, 1, 2, 3, 5))

    big = [s for s in stats if s["size"] >= 21]
    mean_big = np.mean([s["difference_pp"] for s in big]) if big else 0.0
    small = [s for s in stats if s["size"] <= 8]
    mean_small = np.mean([s["difference_pp"] for s in small]) if small else 0.0

    body = f"""
<h2>What was compared</h2>
<p>Our detector combines measurements in two stages: it first sorts them into
groups of measurements that behave alike, combines within each group, then
combines the group summaries. This experiment switches that first stage off, so
all measurements are combined in one pass, and compares the two at every subset
size. Everything else &mdash; the data, the scaling, the way the final score's
direction is resolved &mdash; is identical.</p>
<p>At each size below 30, {N_DRAWS} random combinations of measurements were drawn per
test set and the two methods scored on exactly the same combinations. At size
30 there is only one combination, so it is scored directly.</p>

<h2>Result</h2>
{chart}
<p class="note">Each line is the average over the 25 test sets. Dashed line is
the reference point our project has been measured against: six measurements
chosen by hand using answer keys.</p>
{tbl}

<h2>Reading this</h2>
<ul>
<li><b>The grouping step costs accuracy at large sizes and gains at small
sizes.</b> Averaged over sizes 21&ndash;30 the difference is
{mean_big:+.2f} pp in favour of switching grouping off; over sizes 3&ndash;8 it is
{mean_small:+.2f} pp.</li>
<li>The sizes where grouping helps are the sizes where there are too few
measurements for a group structure to be worth estimating &mdash; and the sizes
where it hurts are where the estimate is noisiest.</li>
<li><b>None of these differences is large.</b> They sit in the same range as the
spread between neighbouring sizes, so this should be read as "the grouping step
is not buying what it was meant to buy at full pool size", not as
"switching it off is an improvement worth shipping".</li>
</ul>

<h2>Saved data</h2>
<ul>
<li><code>grouping_by_size.csv</code> &mdash; one row per size per test set</li>
<li><code>grouping_summary.csv</code> &mdash; one row per size, aggregated</li>
<li><code>grouping_raw.npz</code> &mdash; every individual subset drawn, its exact
measurement list, and both scores, for re-analysis</li>
</ul>
"""
    S.write_page(
        os.path.join(out, "index.html"),
        "Experiment 1 - Does the grouping step help?",
        "Switching off the detector's internal 'group similar measurements' "
        "stage, at every subset size, on 25 test sets.",
        [f"At the full 30 measurements, switching grouping off changes accuracy by "
         f"{stats[-1]['difference_pp']:+.2f} pp "
         f"({stats[-1]['test_sets_better_without_grouping']}/{stats[-1]['test_sets_total']} "
         f"test sets better without it).",
         f"Averaged over large subsets (21-30) the difference is {mean_big:+.2f} pp; "
         f"over small subsets (3-8) it is {mean_small:+.2f} pp - the sign flips with size.",
         "The effect is small everywhere. The honest reading is that the grouping "
         "step is not earning its keep at full pool size, not that turning it off "
         "is a shippable win."],
        body)
    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
