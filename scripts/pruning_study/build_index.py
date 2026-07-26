"""
build_index.py - the front page for the trimming study.

Reads whatever each experiment has written and assembles one page linking them
all, with the headline number from each. Safe to re-run at any point; it simply
skips experiments that have not finished yet.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


EXPERIMENTS = [
    ("01_grouping", "Experiment 1 - Does the grouping step help?",
     "Switching off the detector's internal 'group similar measurements' stage, "
     "at every subset size."),
    ("02_cluster_localized",
     "Experiment 2 - Trimming by fixing the worst-fitting group",
     "Omri's cluster-localized algorithm, run for the first time, with controls "
     "for the group choice and the tie-breaker."),
    ("03_size_and_criterion",
     "Experiment 3 - Is there a right number of measurements?",
     "Whether accuracy peaks at a middle size, and whether the fit score can "
     "find good combinations."),
    ("04_weight_diagnostic", "Experiment 4 - Where the label-free weights go wrong",
     "Diagnosing the 1.45-point gap between guessed and learned trust levels."),
    ("05_weighting", "Experiment 5 - How to weight the measurements",
     "Sixteen weighting pipelines as a three-slot factorial."),
]


def headline(slug):
    d = os.path.join(S.OUT_ROOT, slug)
    if slug == "01_grouping":
        r = read(os.path.join(d, "grouping_summary.csv"))
        if r:
            last = r[-1]
            return (f"At all 30 measurements, switching grouping off changes "
                    f"accuracy by {float(last['difference_pp']):+.2f} points "
                    f"({last['test_sets_better_without_grouping']}/"
                    f"{last['test_sets_total']} test sets better without it).")
    if slug == "02_cluster_localized":
        r = read(os.path.join(d, "arm_comparison.csv"))
        if r:
            b = max(r, key=lambda x: float(x["macro_auroc"]))
            return (f"Best arm: {b['method']} at {float(b['macro_auroc']):.4f}. "
                    f"The previously recorded 0.7004 for this idea is void - that "
                    f"prototype's fit score was zero by construction.")
    if slug == "03_size_and_criterion":
        r = read(os.path.join(d, "size_curve_per_test_set.csv"))
        s = read(os.path.join(d, "cache_staleness_audit.csv"))
        if r:
            rising = sum(1 for x in r if float(x["typical_at_largest"]) >
                         float(x["typical_at_smallest"]))
            ckey = ("correlation_fit_vs_accuracy" if "correlation_fit_vs_accuracy"
                    in r[0] else "mean_correlation_fit_vs_accuracy")
            cor = np.nanmean([float(x[ckey]) for x in r])
            extra = ""
            if s:
                ok = sum(1 for x in s if x["verdict"] == "matches")
                extra = (f" The 1.03M-combination cache on disk is unusable - only "
                         f"{ok}/{len(s)} test sets still reproduce.")
            pos = sum(1 for x in r if float(x[ckey]) > 0)
            return (f"No middle peak: bigger is better for a typical combination in "
                    f"{rising}/{len(r)} test sets, so there is no turn for a stopping "
                    f"rule to find. The fit score correlates {cor:+.3f} with accuracy "
                    f"(positive in {pos}/{len(r)}) - meaning worse-fitting combinations "
                    f"score higher, the opposite of what the algorithm assumes.{extra}")
    if slug == "04_weight_diagnostic":
        r = read(os.path.join(d, "weight_diagnostic_per_test_set.csv"))
        if r:
            g = lambda k: np.median([float(x[k]) for x in r])      # noqa: E731
            return (f"A second hidden factor sits at "
                    f"{g('second_factor_vs_first'):.2f} of the first, so the "
                    f"one-cause premise is only approximate. Guessed vs learned "
                    f"trust levels: rank agreement "
                    f"{g('rank_agreement_guessed_vs_learned'):+.3f}, top-5 overlap "
                    f"{g('top5_overlap_out_of_5'):.0f}/5.")
    if slug == "05_weighting":
        r = read(os.path.join(d, "configuration_comparison.csv"))
        if r:
            b = max(r, key=lambda x: float(x["macro_auroc"]))
            lo = min(float(x["macro_auroc"]) for x in r)
            hi = max(float(x["macro_auroc"]) for x in r)
            return (f"All {len(r)} weighting pipelines land between {lo:.4f} and "
                    f"{hi:.4f} - a {100*(hi-lo):.1f}-point spread, all below the "
                    f"six hand-picked measurements (0.7594).")
    return None


def main():
    os.makedirs(S.OUT_ROOT, exist_ok=True)
    cards, done = [], 0
    for slug, title, sub in EXPERIMENTS:
        page = os.path.join(S.OUT_ROOT, slug, "index.html")
        h = headline(slug)
        if os.path.exists(page) and h:
            done += 1
            cards.append(
                f'<h3><a href="{slug}/index.html">{title}</a></h3>'
                f'<p class="sub">{sub}</p><p>{h}</p>')
        else:
            cards.append(f'<h3>{title}</h3><p class="sub">{sub}</p>'
                         f'<p><i>still running</i></p>')

    ref_rows = [[k, f"{v:.4f}"] for k, v in
                sorted(S.REFERENCE_POINTS.items(), key=lambda kv: -kv[1])]
    ref_tbl = S.html_table(["Reference point", "Accuracy (AUROC)"], ref_rows,
                           numeric_cols=(1,))

    body = f"""
<h2>What this study is testing</h2>
<p>One question, in two halves:</p>
<blockquote><b>Can a score computed without any answer keys decide both
<em>which</em> measurements to trust and <em>how many</em> to keep, well enough
to match a set that was hand-picked using answer keys?</b></blockquote>
<p>The detector takes about 30 different measurements from a language model
while it writes an answer, and combines them into one number: how likely the
answer is wrong. Two things in that pipeline are open &mdash; which measurements
to keep, and how much to trust each one. This study attacks both.</p>

<h2>Every result in one table</h2>
<p><b><a href="all_results.html">Open the full results table &rarr;</a></b>
&mdash; every method, arm and configuration measured across all five experiments,
with three comparison columns. Raw file: <code>all_results.csv</code>.</p>

<h2>The reference points everything is measured against</h2>
{ref_tbl}
<p class="note"><b>Which bar matters.</b> An <b>automatic picker</b> chooses its
measurements itself, per test set, with no answer keys &mdash; the best one is
<code>a6.pl_dufs</code> at <b>0.7524</b>, and that is the bar any new automatic
method has to clear. A <b>fixed subset</b> was chosen once <em>using</em> answer
keys and then reused: <code>GOOD_6</code> (0.7594) and <code>LOCO_5</code>
(0.7705, on 24 of 25 test sets) are both of that kind. GOOD_6 is used throughout
as the anti-regression anchor &mdash; every experiment must reproduce it &mdash; but
it is not a fair target for a label-free method.</p>
<p class="note">The gap that matters: the same 30 measurements with the same
directions score 0.7664 when averaged equally and 0.7809 when a model learns the
trust levels from answer keys. That <b>1.45-point</b> difference is the entire
value of knowing how much to trust each measurement, and it is what the
weighting experiments are trying to recover without labels.</p>

<h2>What the five experiments add up to</h2>
<p>Two findings recur across experiments and are the substance of this study.</p>

<h3>1. The fit score is informative, but its sign is backwards</h3>
<p>The trimming algorithm steers by how well a "everything here is a noisy
reading of one hidden thing" model explains the measurements, and removes
whatever most <em>improves</em> that fit. Two independent experiments say that
is the wrong direction:</p>
<ul>
<li><b>Experiment 3</b>: across combinations of equal size, those that fit the
model <em>worse</em> are <em>more</em> accurate (correlation +0.22, positive in
24 of 25 test sets).</li>
<li><b>Experiment 2</b>: repairing the worst-fitting group is <em>worse</em> than
repairing a randomly chosen one, by 2.2 points.</li>
</ul>
<p>The explanation is the same in both cases. The strongest measurements &mdash;
average uncertainty per token, average surprise per token, confidence in the
chosen word &mdash; are near-duplicates of each other. That duplication is exactly
the extra shared structure a single-factor model cannot absorb, so it shows up as
bad fit. <b>In this data, poor fit marks where the signal is concentrated, not
where the junk is.</b> Any rule that trims toward better fit is removing the
informative measurements first.</p>

<h3>2. Trimming has a high ceiling and a poor average</h3>
<p>A typical smaller combination is <em>worse</em> than keeping everything, in
25 of 25 test sets. But the best combination found at each size sits several
points above the typical one, and the best ones found are small. So the entire
value of trimming lives in <em>choosing the right measurements</em> &mdash; there is
no benefit to being small as such, and no middle size that is good on average.
A stopping rule of the form "trim while the curve improves, stop at the turn"
has no turn to find.</p>

<h2>The experiments</h2>
{"".join(cards)}

<h2>How results are reported</h2>
<ul>
<li><b>No result is claimed on a small difference in accuracy.</b> With 25 test
sets, differences below about a point are not distinguishable from sampling
noise, so win/loss counts and significance tests are reported alongside every
average and nothing is adopted on an average alone.</li>
<li><b>Every experiment re-checks the anchor.</b> The six hand-picked
measurements must reproduce 0.7594 before any number is reported; if they do
not, the data being loaded is not the data our existing numbers came from.</li>
<li><b>Every chart has its raw numbers beside it</b> as CSV, so any figure can
be re-derived, re-plotted or re-tested later.</li>
</ul>

<h2>A note on speed</h2>
<p>Two functions in the shared fusion code were doing avoidable work: a matrix
that costs O(m<sup>4</sup>) to build was being computed even when nothing read
it, and two inner loops were written in Python. Fixing all three left the output
bit-identical and made the combining step about 100 times faster at 30
measurements, which is what made these experiments affordable at all.</p>
"""
    S.write_page(os.path.join(S.OUT_ROOT, "index.html"),
                 "Trimming study - can we pick measurements without answer keys?",
                 "Five experiments on 25 test sets. Every number recomputed on "
                 "current data; every chart has its raw data beside it.",
                 ["<b>The fit score the trimming algorithm steers by has its sign "
                  "backwards.</b> Combinations that fit the one-cause model worse are "
                  "more accurate (+0.22, 24/25 test sets), and repairing the "
                  "worst-fitting group is 2.2 points worse than repairing a random one.",
                  "<b>Why:</b> the strongest measurements are near-duplicates of each "
                  "other, and that duplication is what makes the fit look bad. Poor fit "
                  "marks where the signal is, not where the junk is.",
                  "<b>Trimming has a high ceiling and a poor average</b> - a typical "
                  "smaller set is worse than keeping everything (25/25), but the best "
                  "small sets are much better. All the value is in choosing well.",
                  "<b>Nothing tested closes the weight-estimation gap.</b> All 16 "
                  "weighting pipelines land within 1.2 points of each other and below the "
                  "six hand-picked measurements.",
                  "<b>About a million cached scored combinations turned out to be stale</b> "
                  "(only 5/19 test sets still reproduce) - everything here was recomputed."],
                 body)


if __name__ == "__main__":
    main()
