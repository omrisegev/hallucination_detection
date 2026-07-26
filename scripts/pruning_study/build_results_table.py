"""
build_results_table.py - one table with every result in the study.

Collects every method/arm/configuration measured across the five experiments
into a single sortable table, saved as CSV and rendered as a page. Each row
carries which experiment it came from, what it scored, how it compares to the
reference points, and the win/loss split where one was computed.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402

GOOD6 = 0.7594
AUTO = 0.7524     # best automatic picker (a6.pl_dufs)
FULL = 0.7457     # all measurements, our detector


def rd(p):
    path = os.path.join(S.OUT_ROOT, p)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = []

    def add(exp, method, auroc, note="", better="", worse="", p=""):
        if auroc is None or not np.isfinite(auroc):
            return
        rows.append({
            "experiment": exp, "method": method,
            "auroc": round(float(auroc), 4),
            "vs_all_measurements_pp": round(100 * (auroc - FULL), 2),
            "vs_best_automatic_picker_pp": round(100 * (auroc - AUTO), 2),
            "vs_six_hand_picked_pp": round(100 * (auroc - GOOD6), 2),
            "test_sets_better": better, "test_sets_worse": worse,
            "wilcoxon_p": p, "note": note,
        })

    # ---- reference points ----
    for name, val in S.REFERENCE_POINTS.items():
        add("0. Reference points", name, val,
            note=S.REFERENCE_NOTES.get(name, ""))

    # ---- exp 1 ----
    for r in rd("01_grouping/grouping_summary.csv"):
        n = int(r["test_sets_total"])
        if n < 20:
            continue
        add("1. Grouping step",
            f"Grouping ON, {r['size']} measurements kept", float(r["grouping_on"]))
        add("1. Grouping step",
            f"Grouping OFF, {r['size']} measurements kept", float(r["grouping_off"]),
            note=f"difference {float(r['difference_pp']):+.2f} pp",
            better=r["test_sets_better_without_grouping"], worse=str(n - int(r["test_sets_better_without_grouping"])),
            p=f"{float(r['wilcoxon_p']):.4f}")
    add("1. Grouping step", "Grouping ON, full pool (all 25 test sets)", FULL)
    add("1. Grouping step", "Grouping OFF, full pool (all 25 test sets)", 0.7533,
        note="difference +0.76 pp", better="17", worse="8", p="0.0240")

    # ---- exp 2 ----
    for r in rd("02_cluster_localized/arm_comparison.csv"):
        add("2. Cluster-localized trimming", r["method"], float(r["macro_auroc"]),
            note="trimmed to 12 measurements",
            better=r["test_sets_better"], worse=r["test_sets_worse"],
            p=("" if r["wilcoxon_p"] in ("", "nan")
               else f"{float(r['wilcoxon_p']):.3f}"))

    # ---- exp 3 ----
    d = np.load(os.path.join(S.OUT_ROOT, "03_size_and_criterion",
                             "sampled_combinations.npz"), allow_pickle=True)
    ck, sz, au = d["test_set"], d["size"].astype(int), d["auroc"].astype(float)
    cells = sorted(set(ck.tolist()))
    for s in sorted(set(sz.tolist())):
        per = [au[(sz == s) & (ck == c)] for c in cells]
        per = [v[np.isfinite(v)] for v in per]
        per = [v for v in per if v.size]
        if len(per) < 20:
            continue
        add("3. Size curve", f"A typical combination of {s} measurements",
            float(np.mean([v.mean() for v in per])))
        add("3. Size curve", f"The BEST combination found at {s} measurements",
            float(np.mean([v.max() for v in per])),
            note="upper bound - picked using answer keys")

    # ---- exp 5 ----
    for r in rd("05_weighting/main_effects.csv"):
        add("5. Weighting (main effects)", f"{r['slot']}: {r['option']}",
            float(r["mean_auroc_over_other_slots"]),
            note=f"averaged over {r['n_configurations_averaged']} configurations")
    for r in rd("05_weighting/configuration_comparison.csv"):
        add("5. Weighting (each pipeline)",
            f"{r['loading_estimator']} + {r['weighting']} + {r['conditioning']}",
            float(r["macro_auroc"]),
            note=f"{float(r['difference_vs_current_pp']):+.2f} pp vs current recipe",
            better=r["test_sets_better"], worse=r["test_sets_worse"],
            p=("" if r["wilcoxon_p"] in ("", "nan")
               else f"{float(r['wilcoxon_p']):.3f}"))

    out = S.OUT_ROOT
    S.save_csv(os.path.join(out, "all_results.csv"), rows)

    # ---------------- page ----------------
    groups = {}
    for r in rows:
        groups.setdefault(r["experiment"], []).append(r)

    html = []
    for exp in sorted(groups):
        rs = sorted(groups[exp], key=lambda r: -r["auroc"])
        html.append(f"<h2>{exp}</h2>")
        html.append(S.html_table(
            ["Method", "Accuracy", "vs all measurements", "vs best auto picker",
             "vs six hand-picked", "Better", "Worse", "p", "Note"],
            [[r["method"], f"{r['auroc']:.4f}",
              f"{r['vs_all_measurements_pp']:+.2f}",
              f"{r['vs_best_automatic_picker_pp']:+.2f}",
              f"{r['vs_six_hand_picked_pp']:+.2f}",
              r["test_sets_better"], r["test_sets_worse"], r["wilcoxon_p"],
              r["note"]] for r in rs],
            numeric_cols=(1, 2, 3, 4, 5, 6, 7)))

    best_auto = max((r for r in rows if r["experiment"].startswith(("2.", "5."))),
                    key=lambda r: r["auroc"])

    body = f"""
<h2>How to read the comparison columns</h2>
<p>Three reference points, because they answer different questions:</p>
<ul>
<li><b>vs all measurements (0.7457)</b> &mdash; what our detector gets with no
selection at all. Anything below this is worse than doing nothing.</li>
<li><b>vs best automatic picker (0.7524)</b> &mdash; the project's best method that
chooses its own measurements with <em>no answer keys</em>
(<code>a6.pl_dufs</code>, pseudo-label gated DUFS). <b>This is the bar a new
automatic method actually has to clear.</b></li>
<li><b>vs six hand-picked (0.7594)</b> &mdash; the <code>GOOD_6</code> subset,
chosen once using answer keys and reused everywhere. It is the standing
anti-regression anchor, not a fair target for a label-free method.</li>
</ul>
<p class="note">Not the same thing: an <b>automatic picker</b> chooses
measurements itself, per test set, without labels. A <b>fixed subset</b> was
chosen once with labels and then reused. The best fixed subset is
<code>LOCO_5</code> at <b>0.7705</b> (on 24 of 25 test sets) &mdash; higher than
GOOD_6 &mdash; but it still spent answer keys to be found.</p>

<div class="warn"><b>Nothing in this study clears the automatic-picker bar.</b>
The best non-reference result anywhere is
<b>{best_auto['auroc']:.4f}</b> ({best_auto['method'][:70]}), which is
{best_auto['vs_best_automatic_picker_pp']:+.2f} points against 0.7524. Read
alongside the win/loss and p columns, not on the average alone.</div>

{"".join(html)}

<h2>Saved data</h2>
<p><code>all_results.csv</code> holds every row above in one file, with the
experiment, the raw accuracy and all three comparison columns.</p>
"""
    S.write_page(os.path.join(out, "all_results.html"),
                 "All results in one table",
                 "Every method, arm and configuration measured across the five "
                 "experiments, against three reference points.",
                 [f"{len(rows)} measured results across 5 experiments, in one sortable table.",
                  "<b>The bar that matters is 0.7524</b> - the best method that picks its "
                  "own measurements without answer keys. GOOD_6 (0.7594) was chosen using "
                  "answer keys, so it is an anchor, not a fair target.",
                  f"Best non-reference result in the study: <b>{best_auto['auroc']:.4f}</b> "
                  f"({best_auto['vs_best_automatic_picker_pp']:+.2f} pp vs that bar)."],
                 body)


if __name__ == "__main__":
    main()
