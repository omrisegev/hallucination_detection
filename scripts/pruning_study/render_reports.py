"""
render_reports.py - rebuild the Experiment 1 and 2 pages from their saved tables.

Both experiments were written with a narrative drafted before their results were
in, and both results contradicted the draft:

  * Experiment 1's draft said the grouping step's effect would flip sign with
    subset size. It does not - switching grouping off is better at EVERY size.
  * Experiment 2's draft assumed the localizer would look good. It does not -
    choosing the worst-fitting group is worse than choosing a random one.

Rather than re-run either experiment (about 40 minutes each), this reads the
saved CSVs and rewrites the pages against what was actually measured. It also
adds the full-pool comparison across all 25 test sets, which the size grid
misses because the number of available measurements varies from 27 to 30.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402


def read(path, floats=(), ints=()):
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            for k in floats:
                if k in r and r[k] not in ("", "nan"):
                    r[k] = float(r[k])
                elif k in r:
                    r[k] = float("nan")
            for k in ints:
                if k in r and r[k] != "":
                    r[k] = int(float(r[k]))
            rows.append(r)
    return rows


# ==========================================================================
def render_exp01():
    out = S.outdir("01_grouping")
    st = read(os.path.join(out, "grouping_summary.csv"),
              floats=("grouping_on", "grouping_off", "difference_pp", "wilcoxon_p"),
              ints=("size", "test_sets_better_without_grouping", "test_sets_total"))

    # Full-pool comparison across ALL 25 test sets (the size grid can't do this:
    # test sets have between 27 and 30 measurements available, so "size 30" only
    # covers the 6 test sets that happen to have all 30).
    cells = S.load()
    S.validity_check(cells)
    on, off, pool_sizes = [], [], []
    for cell in cells.values():
        cols = list(range(len(cell["pool"])))
        pool_sizes.append(len(cols))
        on.append(S.fuse_score(cell, cols))
        off.append(S.fuse_score(cell, cols, groups="flat"))
    on, off = np.array(on), np.array(off)
    from scipy.stats import wilcoxon
    p_full = wilcoxon(on, off).pvalue
    print(f"  full pool, all 25 test sets: on={np.nanmean(on):.4f} "
          f"off={np.nanmean(off):.4f} p={p_full:.4f}")

    sizes = [s["size"] for s in st]
    chart = S.line_chart(
        [("Grouping step ON (deployed detector)", sizes, [s["grouping_on"] for s in st]),
         ("Grouping step OFF", sizes, [s["grouping_off"] for s in st])],
        "Number of measurements kept", "Detection accuracy (AUROC)",
        hlines=[("six hand-picked = 0.7594", 0.7594)])

    tbl = S.html_table(
        ["Measurements kept", "Grouping ON", "Grouping OFF", "Difference",
         "Test sets better without grouping", "Wilcoxon p"],
        [[s["size"], f"{s['grouping_on']:.4f}", f"{s['grouping_off']:.4f}",
          f"{s['difference_pp']:+.2f} pp",
          f"{s['test_sets_better_without_grouping']}/{s['test_sets_total']}",
          "-" if not np.isfinite(s["wilcoxon_p"]) else f"{s['wilcoxon_p']:.4f}"]
         for s in st], numeric_cols=(0, 1, 2, 3, 5))

    grid = [s for s in st if s["test_sets_total"] >= 20]
    n_pos = sum(1 for s in grid if s["difference_pp"] > 0)
    n_sig = sum(1 for s in grid if np.isfinite(s["wilcoxon_p"]) and s["wilcoxon_p"] < 0.05)
    mean_d = float(np.mean([s["difference_pp"] for s in grid]))
    full_d = 100 * (np.nanmean(off) - np.nanmean(on))

    body = f"""
<h2>What was compared</h2>
<p>Our detector combines measurements in two stages: it first sorts them into
groups that behave alike, combines within each group, then combines the group
summaries. This experiment switches that first stage off, so everything is
combined in one pass, and compares the two at every subset size. The data, the
scaling and the way the final score's direction is resolved are identical; only
the grouping stage changes.</p>
<p>At each size, 20 random combinations of measurements were drawn per test set
and both methods scored on <em>exactly the same</em> combinations, so the
comparison is paired.</p>
<p class="note">A note on "30 measurements": the number available varies by test
set &mdash; {min(pool_sizes)} to {max(pool_sizes)}. The size-30 row therefore only
covers the {sum(1 for x in pool_sizes if x == 30)} test sets that have all 30.
The full-pool comparison below uses every test set's own complete set, so it
covers all 25.</p>

<h2>Result</h2>
{chart}
{tbl}

<h3>Full pool, all 25 test sets</h3>
<p>Using each test set's complete set of available measurements:
grouping ON <b>{np.nanmean(on):.4f}</b>, grouping OFF
<b>{np.nanmean(off):.4f}</b> &mdash; a difference of <b>{full_d:+.2f} points</b>
in favour of switching it off, better on
<b>{int((off > on).sum())} of 25</b> test sets (Wilcoxon p = {p_full:.4f}).</p>

<h2>Reading this</h2>
<ul>
<li><b>Switching the grouping step off is better at every single size tested</b>
&mdash; {n_pos} of {len(grid)} sizes positive, and {n_sig} of {len(grid)}
individually significant at p &lt; 0.05. The average difference across the size
grid is {mean_d:+.2f} points.</li>
<li>The consistency matters more than the size of the effect. Any single size
here would be unconvincing on its own; the same sign at every size, on paired
draws, is not.</li>
<li><b>What this does not say.</b> The effect is well under a point at most
sizes. This is evidence that the grouping stage is <em>not earning its keep</em>,
not evidence that removing it is a meaningful accuracy improvement. Nothing here
reaches the six hand-picked measurements (0.7594).</li>
<li>The grouping stage was designed to handle measurements that are
near-duplicates of one another. Experiment 4 shows those duplicates are present
in every test set, so the stage is not failing for want of something to do
&mdash; it is failing at the thing it was built for.</li>
</ul>

<h2>Saved data</h2>
<ul>
<li><code>grouping_by_size.csv</code> &mdash; one row per size per test set</li>
<li><code>grouping_summary.csv</code> &mdash; one row per size, aggregated</li>
<li><code>grouping_raw.npz</code> &mdash; every individual combination drawn, its exact
measurement list, and both scores</li>
</ul>
"""
    S.write_page(
        os.path.join(out, "index.html"),
        "Experiment 1 - Does the grouping step help?",
        "Switching off the detector's internal 'group similar measurements' "
        "stage, at every subset size, on 25 test sets.",
        [f"<b>Switching grouping off is better at every size tested</b> "
         f"({n_pos}/{len(grid)} sizes, {n_sig} individually significant).",
         f"On the full pool across all 25 test sets: {np.nanmean(on):.4f} with "
         f"grouping vs {np.nanmean(off):.4f} without ({full_d:+.2f} points, "
         f"p = {p_full:.4f}).",
         "The effect is small - under a point at most sizes. This says the grouping "
         "stage is not earning its keep, not that removing it is a real improvement.",
         "It was built to handle near-duplicate measurements, and those are present "
         "in every test set - so it is failing at its own job, not idle."],
        body)


# ==========================================================================
def render_exp02():
    out = S.outdir("02_cluster_localized")
    loc = read(os.path.join(out, "localizer_discrimination.csv"),
               floats=("worst_group_misfit_per_pair", "best_group_misfit_per_pair",
                       "spread_ratio"),
               ints=("n_groups", "n_groups_with_2plus_members"))
    comp = read(os.path.join(out, "arm_comparison.csv"),
                floats=("macro_auroc", "difference_vs_worst_group_pp", "wilcoxon_p"),
                ints=("test_sets_better", "test_sets_worse"))
    arms = read(os.path.join(out, "arms_per_test_set.csv"),
                ints=("n_tied_steps",))
    comp.sort(key=lambda r: -r["macro_auroc"])

    ref = "Localize to worst-fitting group"
    get = lambda n: next(c for c in comp if c["method"] == n)      # noqa: E731
    worst_grp, rnd = get(ref), get("Localize to a RANDOM group (control)")
    glob = get("Remove globally (no groups)")
    coin = get("Worst group + coin-flip tie-break (control)")
    smooth = max((c for c in comp if "smoothness" in c["method"]),
                 key=lambda c: c["macro_auroc"])
    rnd_gap = 100 * (rnd["macro_auroc"] - worst_grp["macro_auroc"])
    tied = np.mean([a["n_tied_steps"] for a in arms
                    if a["method"].startswith("Worst group")])

    bar = S.bar_chart([c["method"] for c in comp],
                      [c["macro_auroc"] for c in comp],
                      "Detection accuracy (AUROC), averaged over 25 test sets",
                      hlines=[("six hand-picked", 0.7594)])
    loc_sorted = sorted(loc, key=lambda r: -r["spread_ratio"])
    loc_bar = S.bar_chart([r["test_set"] for r in loc_sorted],
                          [r["spread_ratio"] for r in loc_sorted],
                          "How many times worse the worst repairable group fits "
                          "than the best", value_fmt="{:.1f}x", bar_h=21)
    comp_tbl = S.html_table(
        ["Method", "Accuracy", "vs. worst-group", "Better", "Worse", "Wilcoxon p"],
        [[c["method"], f"{c['macro_auroc']:.4f}",
          f"{c['difference_vs_worst_group_pp']:+.2f} pp",
          c["test_sets_better"], c["test_sets_worse"],
          "-" if not np.isfinite(c["wilcoxon_p"]) else f"{c['wilcoxon_p']:.3f}"]
         for c in comp], numeric_cols=(1, 2, 3, 4, 5))
    loc_tbl = S.html_table(
        ["Test set", "Groups", "Repairable", "Worst misfit", "Best misfit",
         "Spread", "What is in the worst-fitting group"],
        [[r["test_set"], r["n_groups"], r.get("n_groups_with_2plus_members", "-"),
          f"{r['worst_group_misfit_per_pair']:.4f}",
          f"{r['best_group_misfit_per_pair']:.4f}", f"{r['spread_ratio']:.1f}x",
          str(r["worst_group_members"])[:140]] for r in loc_sorted],
        numeric_cols=(1, 2, 3, 4, 5))
    med_spread = float(np.median([r["spread_ratio"] for r in loc]))

    body = f"""
<h2>The algorithm being tested</h2>
<p>Rather than ranking all the measurements and cutting the bottom ones, this
algorithm uses structure the detector <em>already computes and then discards</em>.
The detector sorts measurements into groups that behave alike, and measures how
badly its "everything here is a noisy reading of one hidden thing" model fits.
The idea: find the <b>group where that model fits worst</b> and remove a
measurement from <em>there</em>, on the reasoning that a badly fitting group is
where something does not belong. Repeat until a stopping point.</p>

<div class="warn"><b>This had never actually been run.</b> The prototype in the
repository computed its fit score as the distance between a matrix times its own
leading eigenvector and that eigenvector times its own eigenvalue &mdash; which is
<b>zero by definition</b>. Measured: about 2&times;10<sup>-15</sup>. It was
ranking candidate removals by floating-point rounding error, so the 0.7004
previously recorded against this idea is the score of a coin flip, not of the
idea itself.</div>

<h2>Part 0 - The localizer does discriminate</h2>
<p>Before asking whether it helps, check it points somewhere at all. Groups
holding a single measurement are excluded: they have no internal pairs, so their
misfit is zero by construction and they cannot be repaired.</p>
{loc_bar}
<p class="note">Median spread <b>{med_spread:.1f}&times;</b> between the worst-
and best-fitting repairable group. The localizer is emphatically not
indifferent about where to point.</p>
{loc_tbl}

<h2>Parts 1 and 2 - But pointing there makes things worse</h2>
<p>All arms trim down to {arms[0]['n_kept']} measurements, so size is never a
confound and the only difference is <em>which</em> measurements are removed. Two
controls are included: removing from a <b>randomly chosen</b> group instead of
the worst one, and breaking near-ties with a <b>coin flip</b>.</p>
{bar}
{comp_tbl}

<h2>What actually happened</h2>
<div class="warn"><b>Choosing the worst-fitting group is worse than choosing a
group at random.</b> The random-group control scores
<b>{rnd['macro_auroc']:.4f}</b> against the worst-group localizer's
<b>{worst_grp['macro_auroc']:.4f}</b> &mdash; <b>{rnd_gap:+.2f} points</b> in favour
of <em>not</em> using the localizer, better on {rnd['test_sets_better']} of 25
test sets. This is not a small difference, and it is the wrong way round.</div>

<h3>Why it fails - the useful part</h3>
<p>Look at what sits in the worst-fitting group. Across test sets it is
consistently the cluster of near-duplicate confidence measurements: average
uncertainty per token, average surprise per token, average vocabulary spread,
confidence in the chosen word. Those are among the <b>strongest individual
predictors in the whole pool</b>. They fit the one-cause model badly precisely
<em>because</em> they are near-duplicates &mdash; several readings of the same
underlying quantity, which creates exactly the extra shared structure a
single-factor model cannot absorb.</p>
<p>So "the group where the model fits worst" turns out to be a reliable detector
of <b>where the signal is concentrated</b>, not of where the junk is. Stripping
measurements from it repeatedly removes the informative ones and leaves the weak
ones behind. The premise that bad fit marks something that does not belong is
the part that fails &mdash; and it fails for a reason that generalises: in this
data, redundancy and informativeness travel together.</p>

<h3>The tie-breaker question is real but moot</h3>
<p>Near-ties dominate: about <b>{tied:.0f} of the removal steps per test set</b>
had a runner-up within 10% of the best candidate, so whatever breaks ties really
does make most of the decisions. But across the graph choices the spread is
small &mdash; coin flip {coin['macro_auroc']:.4f}, best smoothness variant
{smooth['macro_auroc']:.4f}. <b>That gap is too small to call either way</b>, and
with the localizer itself pointing the wrong way the tie-breaker is answering a
question that does not arise.</p>

<h3>And trimming at all is the wrong move here</h3>
<p>Every arm lands between {min(c['macro_auroc'] for c in comp):.4f} and
{max(c['macro_auroc'] for c in comp):.4f}. Keeping <em>all</em> the measurements
scores <b>0.7457</b>; the six hand-picked ones score <b>0.7594</b>. Trimming by
any of these rules is worse than not trimming at all &mdash; which matches
Experiment 3's finding that a typical trimmed combination is worse than the full
set.</p>

<h2>Saved data</h2>
<ul>
<li><code>localizer_discrimination.csv</code> &mdash; per test set: group misfits and
exactly which measurements sit in the worst group</li>
<li><code>arms_per_test_set.csv</code> &mdash; every arm on every test set, with the
full list of measurements kept and how many steps were near-ties</li>
<li><code>arm_comparison.csv</code>, <code>arm_scores.json</code> &mdash; aggregates and
raw per-test-set score vectors for re-testing</li>
</ul>
"""
    S.write_page(
        os.path.join(out, "index.html"),
        "Experiment 2 - Trimming by fixing the worst-fitting group",
        "Omri's cluster-localized algorithm, run for the first time, with "
        "controls for the group choice and the tie-breaker.",
        [f"<b>The idea is refuted, and cleanly.</b> Choosing the worst-fitting group "
         f"({worst_grp['macro_auroc']:.4f}) is <b>worse</b> than choosing a random "
         f"group ({rnd['macro_auroc']:.4f}) - a gap of {rnd_gap:+.2f} points.",
         "<b>Why it fails is the valuable part:</b> the worst-fitting group is "
         "reliably the cluster of near-duplicate confidence measurements - among the "
         "strongest predictors in the pool. Bad fit marks where the signal is, not "
         "where the junk is.",
         "The 0.7004 previously recorded against this idea was void anyway: that "
         "prototype's fit score was zero by construction (~2e-15).",
         f"Near-ties drive ~{tied:.0f} of the steps, so the tie-breaker does most of "
         f"the deciding - but its variants differ by too little to call, and the "
         f"question is moot once the localizer points the wrong way."],
        body)


# ==========================================================================
def render_exp03():
    """Rebuild Experiment 3 from its saved sample.

    The original run computed the sweep correctly and saved it, then crashed in
    the aggregation: at size 30 only the 6 test sets that actually hold 30
    measurements contribute, and the others produced empty buckets. The sweep is
    expensive (~70 min), the aggregation is not, so it is redone here from the
    saved data with the empty buckets handled.
    """
    from scipy.stats import spearmanr
    out = S.outdir("03_size_and_criterion")
    d = np.load(os.path.join(out, "sampled_combinations.npz"), allow_pickle=True)
    ck = d["test_set"]
    sz = d["size"].astype(int)
    au = d["auroc"].astype(float)
    fit = d["fit_misfit"].astype(float)
    stale = read(os.path.join(out, "cache_staleness_audit.csv"),
                 floats=("max_abs_difference",), ints=("n_rechecked",))

    # Size 30 is only reachable by the handful of test sets that hold all 30
    # measurements, so including it would compare different test sets at
    # different points on the curve. Curve sizes are restricted to those every
    # test set can reach.
    counts = {int(s): int((sz == s).sum()) for s in np.unique(sz)}
    n_cells = len(set(ck.tolist()))
    curve_sizes = [s for s, n in sorted(counts.items()) if n >= n_cells * 5]
    dropped = [s for s in counts if s not in curve_sizes]

    mean_line, max_line = [], []
    for s in curve_sizes:
        per = [au[(sz == s) & (ck == c)] for c in sorted(set(ck.tolist()))]
        per = [v[np.isfinite(v)] for v in per]
        per = [v for v in per if v.size]
        mean_line.append(float(np.mean([v.mean() for v in per])))
        max_line.append(float(np.mean([v.max() for v in per])))

    rows, corrs = [], []
    for c in sorted(set(ck.tolist())):
        m = ck == c
        s_c, a_c, f_c = sz[m], au[m], fit[m]
        within = []
        for s in curve_sizes:
            k = (s_c == s) & np.isfinite(a_c) & np.isfinite(f_c)
            if k.sum() > 10 and np.ptp(f_c[k]) > 0:
                r_ = spearmanr(f_c[k], a_c[k]).statistic
                if np.isfinite(r_):
                    within.append(r_)
        means = [a_c[(s_c == s) & np.isfinite(a_c)].mean() for s in curve_sizes]
        maxs = [a_c[(s_c == s) & np.isfinite(a_c)].max() for s in curve_sizes]
        mc = float(np.mean(within)) if within else np.nan
        corrs.append(mc)
        rows.append({"test_set": S.plain_cell(c), "test_set_code": c,
                     "best_size_typical": curve_sizes[int(np.argmax(means))],
                     "best_size_best": curve_sizes[int(np.argmax(maxs))],
                     "typical_at_smallest": means[0],
                     "typical_at_largest": means[-1],
                     "best_found": float(max(maxs)),
                     "correlation_fit_vs_accuracy": mc})
    S.save_csv(os.path.join(out, "size_curve_per_test_set.csv"), rows)
    corrs = np.array(corrs, float)

    rising = sum(1 for r in rows if r["typical_at_largest"] > r["typical_at_smallest"])
    big_best = sum(1 for r in rows if r["best_size_typical"] >= 14)
    small_best = sum(1 for r in rows if r["best_size_best"] <= 8)
    n_ok = sum(1 for r in stale if r["verdict"] == "matches")
    worst = max(r["max_abs_difference"] for r in stale)
    gap = 100 * (np.mean(max_line) - np.mean(mean_line))

    curve_chart = S.line_chart(
        [("A typical combination of that size", curve_sizes, mean_line),
         ("The BEST combination found at that size", curve_sizes, max_line)],
        "Number of measurements kept", "Detection accuracy (AUROC)",
        hlines=[("six hand-picked = 0.7594", 0.7594),
                ("keep everything = 0.7457", 0.7457)])
    corr_chart = S.bar_chart(
        [r["test_set"] for r in sorted(rows, key=lambda r: r["correlation_fit_vs_accuracy"])],
        [r["correlation_fit_vs_accuracy"] for r in sorted(rows, key=lambda r: r["correlation_fit_vs_accuracy"])],
        "Correlation between the fit score and real accuracy, within one size",
        value_fmt="{:+.3f}", bar_h=21)
    stale_tbl = S.html_table(
        ["Test set", "Combinations re-scored", "Largest disagreement", "Verdict"],
        [[r["test_set"], r["n_rechecked"], f"{r['max_abs_difference']:.2e}",
          r["verdict"].upper()] for r in sorted(stale, key=lambda r: r["max_abs_difference"])],
        numeric_cols=(1, 2))
    curve_tbl = S.html_table(
        ["Test set", "Best size (typical)", "Best size (best found)",
         "Typical at 3", "Typical at largest", "Best found", "Fit-score correlation"],
        [[r["test_set"], r["best_size_typical"], r["best_size_best"],
          f"{r['typical_at_smallest']:.4f}", f"{r['typical_at_largest']:.4f}",
          f"{r['best_found']:.4f}", f"{r['correlation_fit_vs_accuracy']:+.3f}"]
         for r in sorted(rows, key=lambda r: -r["best_found"])],
        numeric_cols=(1, 2, 3, 4, 5, 6))

    body = f"""
<h2>Part 1 - The cached sweep on disk cannot be reused</h2>
<p>An exhaustive sweep run earlier scored <b>every</b> combination of an older
16-measurement pool for 19 test sets &mdash; about 1.03 million combinations.
Reusing it would have answered both questions below for free, so the first step
was checking whether those cached scores still reproduce.</p>
{stale_tbl}
<div class="warn"><b>Only {n_ok} of {len(stale)} test sets still reproduce</b>,
with disagreements up to <b>{worst:.2f} AUROC</b> &mdash; far beyond rounding. The
test sets have been re-graded since the sweep ran, so those cached accuracies
describe data we no longer have. <b>The cache was set aside and everything below
was recomputed.</b> Worth recording on its own: about a million scored
combinations are stale, and any future analysis reaching for them needs this
same check first.</div>

<h2>Part 2 - Fresh measurements on the current pool</h2>
<p>30 random combinations per size per test set, at sizes
{", ".join(str(s) for s in curve_sizes)}, scored through exactly the path our
headline numbers use.
{"Size 30 was dropped from the curve: only the few test sets holding all 30 measurements can reach it, so it would compare a different set of test sets than every other point." if dropped else ""}</p>

<h3>Question 1 - Is there a best number of measurements?</h3>
{curve_chart}
<p class="note">Two different questions in one picture. The lower line is how
good a <em>typical</em> combination of each size is; the upper line is the best
combination <em>found</em> at that size among those sampled.</p>
{curve_tbl}
<ul>
<li><b>No middle peak.</b> For a typical combination, keeping more is better in
{rising}/{len(rows)} test sets, and the best typical size is 14 or more in
{big_best}/{len(rows)}.</li>
<li><b>But the best combinations found are small</b> &mdash; 8 measurements or
fewer in {small_best}/{len(rows)} test sets &mdash; and the upper line runs about
<b>{gap:.1f} points</b> above the lower one.</li>
<li><b>Together: trimming has a high ceiling and a poor average.</b> Cutting
measurements at random makes things worse; cutting the <em>right</em> ones makes
things much better. All the value is in choosing well, none in being small.</li>
<li><b>Consequence for the algorithm.</b> A rule of the form "trim while the
curve improves, stop at the turn" has no turn to find. Size must be fixed in
advance or set by an error-control threshold &mdash; it cannot be discovered by
following this curve.</li>
</ul>

<h3>Question 2 - Does the fit score know which combinations are good?</h3>
{corr_chart}
<div class="warn"><b>It does - but with the sign reversed from what the algorithm
assumes.</b> Mean correlation <b>{np.nanmean(corrs):+.3f}</b>
(median {np.nanmedian(corrs):+.3f}), and <b>positive in
{int((corrs > 0).sum())} of {len(corrs)} test sets</b>. The fit score is
"misfit": lower means the one-hidden-cause model explains the measurements
better. A <em>positive</em> correlation with accuracy therefore means
combinations that fit the model <b>worse</b> are <b>more accurate</b>.</div>
<ul>
<li>The trimming algorithm removes whichever measurement most <em>improves</em>
the fit. On this evidence that is <b>steering directly against the gradient</b>
&mdash; it optimises for the combinations that score lower.</li>
<li><b>This is the same finding as Experiment 2, seen from another angle.</b>
There, repairing the worst-fitting group was worse than repairing a random one.
Here, worse-fitting combinations score higher. Both say: in this data, poor fit
to a single-cause model marks <em>where the signal is</em>. The strongest
measurements are near-duplicates of one another, and that duplication is exactly
what a one-factor model cannot absorb.</li>
<li><b>The actionable version</b> is that the criterion is not useless &mdash; it is
inverted. A rule that <em>preserves</em> badly-fitting structure, or that trims to
<em>increase</em> misfit, is the version worth testing next. That is a different
algorithm from the one tested here, and it has not been run.</li>
<li><b>Caveat on strength.</b> A correlation of {np.nanmean(corrs):+.2f} is
moderate, not strong. It is enough to say the sign is real and consistent
({int((corrs > 0).sum())}/{len(corrs)}); it is not enough to expect a large gain
from acting on it alone.</li>
</ul>
<div class="warn"><b>Why an earlier reading of this said the opposite.</b> A first
pass used the cached sweep and reported a correlation near zero with an
inconsistent sign. That pass was reading the stale cache audited in Part 1 -
{n_ok}/{len(stale)} of whose test sets no longer reproduce. The number above is
the one computed on current data, and it supersedes it.</div>

<h2>Saved data</h2>
<ul>
<li><code>cache_staleness_audit.csv</code> &mdash; per test set, whether the old sweep still reproduces</li>
<li><code>sampled_combinations.csv</code> / <code>.npz</code> &mdash; all
{len(au):,} combinations sampled, each with its exact measurement list,
accuracy, fit score and group count</li>
<li><code>size_curve_per_test_set.csv</code>, <code>criterion_power_per_size.csv</code></li>
</ul>
"""
    S.write_page(
        os.path.join(out, "index.html"),
        "Experiment 3 - Is there a right number of measurements?",
        "Whether accuracy peaks at a middle size, and whether the fit score can "
        "find good combinations. Recomputed on the current pool.",
        [f"<b>The 1.03M-combination cache on disk is unusable</b> - only {n_ok}/{len(stale)} "
         f"test sets still reproduce (worst disagreement {worst:.2f} AUROC).",
         f"<b>No sweet spot.</b> For a typical combination bigger is better in "
         f"{rising}/{len(rows)} test sets, so a 'stop where the curve turns' rule has "
         f"nothing to find.",
         f"<b>But the best combination at each size sits ~{gap:.1f} points above a typical "
         f"one</b>: trimming has a high ceiling and a poor average. The value is entirely "
         f"in choosing well.",
         f"<b>The fit score does carry signal - pointing the wrong way.</b> Correlation "
         f"with accuracy is {np.nanmean(corrs):+.3f}, positive in "
         f"{int((corrs > 0).sum())}/{len(corrs)} test sets, meaning combinations that fit "
         f"the one-cause model WORSE are MORE accurate. The algorithm minimises misfit, so "
         f"it steers against the gradient.",
         f"An earlier pass reported this correlation as near zero; that pass used the stale "
         f"cache and is superseded by the number above."],
        body)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which:
        render_exp01()
    if "2" in which:
        render_exp02()
    if "3" in which:
        render_exp03()
    print("reports rebuilt")
