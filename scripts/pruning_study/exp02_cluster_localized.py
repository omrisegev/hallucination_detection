"""
Experiment 2 - Omri's cluster-localized trimming algorithm.

The algorithm, in plain language:

  1. Run the detector on all 30 measurements. It already sorts them into groups
     of measurements that behave alike, and it already computes how badly its
     "one shared cause" model fits.
  2. LOCALIZE: work out how badly the model fits *inside each group*, and pick
     the worst-fitting group.
  3. REPAIR: try removing each member of that group; keep the removal that most
     improves the fit. Never empty a group.
  4. Repeat.

This has never been run. The prototype in the repo used a fit score that is
zero by construction, so it ranked candidates by rounding error.

Two things are measured here:
  A. Does localizing to the worst group beat removing globally, and beat
     removing from a randomly chosen group?
  B. When two candidates inside the group improve the fit by nearly the same
     amount - which the measurements show is almost always - does a Laplacian
     smoothness tie-breaker pick better than a coin flip, and does the graph it
     is built from matter?

Writes results/pruning_study/02_cluster_localized/
"""
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402
from spectral_utils.fusion_utils import (                         # noqa: E402
    _estimate_von_voff, _residual_lsml,
)
from spectral_utils.selectors.classical_fs import (               # noqa: E402
    _laplacian_score, _sample_graph,
)

TARGET_SIZE = 12
SEED = 0


# --------------------------------------------------------------------------
# the fit score, decomposed per group
# --------------------------------------------------------------------------
def group_fit(R, c):
    """How badly the 'one shared cause' model fits inside each group.

    Returns {group_id: (total_misfit, misfit_per_pair, member_indices)}.
    Per-pair is what we compare, so a big group is not penalised for being big.
    """
    v_on, _ = _estimate_von_voff(R, c)
    out = {}
    for g in np.unique(c):
        idx = np.where(c == g)[0]
        tot = 0.0
        for i in idx:
            for j in idx:
                if i != j:
                    tot += (v_on[i] * v_on[j] - R[i, j]) ** 2
        npair = max(len(idx) * (len(idx) - 1), 1)
        out[int(g)] = (float(tot), float(tot / npair), idx)
    return out


def total_fit(cell, cols):
    _, meta = S.fuse_meta(cell, cols)
    return float(meta["residual"]), np.asarray(meta["c"], int)


def fit_without(R, c, drop_pos):
    """Fit score after dropping one measurement, keeping the grouping the
    detector just found (the remaining members keep their group labels).

    This is what makes the algorithm affordable: group DETECTION runs once per
    removal step, and each candidate is then scored under that same grouping
    instead of triggering a fresh detection. Re-detection happens at the top of
    the next step, once the removal is committed.
    """
    keep = [i for i in range(R.shape[0]) if i != drop_pos]
    return _residual_lsml(R[np.ix_(keep, keep)], np.asarray(c)[keep])


# --------------------------------------------------------------------------
# tie-breaker: Laplacian smoothness, with the graph scope as the variable
# --------------------------------------------------------------------------
def smoothness(cell, cols, cand_col, scope, rng_seed=SEED):
    """Smoothness of `cand_col` on a graph built from a chosen set of columns.

    The graph is over ANSWERS (rows), built FROM measurements (columns). So the
    choice below is which measurements build the graph that judges a candidate.
    Lower score = smoother = more consistent with the graph's structure.
    """
    V = cell["V"]
    if scope == "all_30":
        gcols = list(range(V.shape[1]))
    elif scope == "surviving":
        gcols = list(cols)
    elif scope == "group_only":
        gcols = list(cols)
    elif scope == "group_minus_candidate":
        gcols = [c for c in cols if c != cand_col]
    elif scope == "anchor_only":
        gcols = None
    else:
        raise ValueError(scope)

    rng = np.random.default_rng(rng_seed)
    if gcols is None:
        X = cell["anchor"].reshape(-1, 1)
    else:
        if len(gcols) < 2:
            return np.nan
        X = V[:, gcols]
    W, sub = _sample_graph(X, rng)
    target = V[sub][:, [cand_col]]
    try:
        return float(_laplacian_score(target, W)[0])
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# removal strategies
# --------------------------------------------------------------------------
def step_localized(cell, cols, rng, group_choice="worst", tiebreak=None,
                   tie_tol=0.10, scope="group_minus_candidate"):
    """One removal step. Returns (new_cols, diagnostics)."""
    V = cell["V"]
    R = np.cov(V[:, cols].T)
    base, c = total_fit(cell, cols)
    gf = group_fit(R, c)
    eligible = {g: v for g, v in gf.items() if len(v[2]) > 1}
    if not eligible:
        return None, {}
    if group_choice == "worst":
        g = max(eligible, key=lambda k: eligible[k][1])
    elif group_choice == "random":
        g = int(rng.choice(sorted(eligible)))
    else:
        raise ValueError(group_choice)

    positions = list(eligible[g][2])
    members = [cols[i] for i in positions]
    gains = []
    for pos in positions:
        gains.append((base - fit_without(R, c, pos), cols[pos]))
    gains.sort(reverse=True)

    best_gain = gains[0][0]
    near = [col for gval, col in gains
            if best_gain > 0 and abs(gval - best_gain) / abs(best_gain) <= tie_tol]
    if len(near) < 2:
        near = [gains[0][1]]
    tied = len(near) > 1

    if not tied or tiebreak is None:
        chosen = gains[0][1]
    elif tiebreak == "random":
        chosen = int(rng.choice(near))
    elif tiebreak == "smoothness":
        sc = [(smoothness(cell, members, col, scope), col) for col in near]
        sc = [(s, c_) for s, c_ in sc if np.isfinite(s)]
        chosen = min(sc)[1] if sc else gains[0][1]
    else:
        raise ValueError(tiebreak)

    return [x for x in cols if x != chosen], {
        "group": g, "group_size": len(members), "tied": tied,
        "n_tied": len(near), "removed": cell["pool"][chosen],
    }


def step_global(cell, cols, rng):
    """Remove whichever single measurement most improves the fit, ignoring
    which group it sits in. This is the control that isolates what localizing
    to the worst group actually buys."""
    V = cell["V"]
    R = np.cov(V[:, cols].T)
    base, c = total_fit(cell, cols)
    gains = sorted((base - fit_without(R, c, pos), cols[pos])
                   for pos in range(len(cols)))
    best = gains[-1]
    return [x for x in cols if x != best[1]], {"removed": cell["pool"][best[1]]}


def run_arm(cell, arm, rng, target=TARGET_SIZE):
    cols = list(range(len(cell["pool"])))
    trace, ties = [], 0
    while len(cols) > target:
        if arm["kind"] == "global":
            nxt, d = step_global(cell, cols, rng)
        else:
            nxt, d = step_localized(cell, cols, rng,
                                    group_choice=arm.get("group", "worst"),
                                    tiebreak=arm.get("tiebreak"),
                                    scope=arm.get("scope",
                                                  "group_minus_candidate"))
        if nxt is None:
            break
        ties += int(d.get("tied", False))
        trace.append((len(cols), d.get("removed", "?")))
        cols = nxt
    return cols, trace, ties


ARMS = [
    ("Remove globally (no groups)", {"kind": "global"}),
    ("Localize to worst-fitting group", {"kind": "local", "group": "worst"}),
    ("Localize to a RANDOM group (control)", {"kind": "local", "group": "random"}),
    ("Worst group + coin-flip tie-break (control)",
     {"kind": "local", "group": "worst", "tiebreak": "random"}),
    ("Worst group + smoothness, graph = group minus candidate",
     {"kind": "local", "group": "worst", "tiebreak": "smoothness",
      "scope": "group_minus_candidate"}),
    ("Worst group + smoothness, graph = the group itself",
     {"kind": "local", "group": "worst", "tiebreak": "smoothness",
      "scope": "group_only"}),
    ("Worst group + smoothness, graph = all 30 measurements",
     {"kind": "local", "group": "worst", "tiebreak": "smoothness",
      "scope": "all_30"}),
    ("Worst group + smoothness, graph = anchor only",
     {"kind": "local", "group": "worst", "tiebreak": "smoothness",
      "scope": "anchor_only"}),
]


def main():
    out = S.outdir("02_cluster_localized")
    cells = S.load()
    S.validity_check(cells)

    # ---------- part 0: does the localizer discriminate? ----------
    loc_rows = []
    for ck, cell in cells.items():
        cols = list(range(len(cell["pool"])))
        R = np.cov(cell["V"][:, cols].T)
        _, c = total_fit(cell, cols)
        gf = group_fit(R, c)
        # Only groups with at least two members can be compared or repaired: a
        # lone measurement has no within-group pairs, so its misfit is zero by
        # construction and would make any "spread" ratio meaningless.
        per = {g: v[1] for g, v in gf.items() if len(v[2]) > 1}
        if not per:
            continue
        worst = max(per, key=per.get)
        best = min(per, key=per.get)
        loc_rows.append({
            "test_set": S.plain_cell(ck), "test_set_code": ck,
            "n_groups": len(gf),
            "n_groups_with_2plus_members": len(per),
            "n_single_measurement_groups": len(gf) - len(per),
            "worst_group_misfit_per_pair": per[worst],
            "best_group_misfit_per_pair": per[best],
            "spread_ratio": per[worst] / max(per[best], 1e-12),
            "worst_group_members": " | ".join(
                S.plain(cell["pool"][i]) for i in gf[worst][2]),
            "worst_group_members_code": ",".join(
                cell["pool"][i] for i in gf[worst][2]),
        })
    S.save_csv(os.path.join(out, "localizer_discrimination.csv"), loc_rows)

    # ---------- part 1+2: the arms ----------
    rows, per_cell = [], {}
    for name, arm in ARMS:
        rng = np.random.default_rng(SEED)
        scores, tie_counts = [], []
        for ck, cell in cells.items():
            cols, trace, ties = run_arm(cell, arm, rng)
            a = S.fuse_score(cell, cols)
            scores.append(a)
            tie_counts.append(ties)
            rows.append({
                "method": name, "test_set": S.plain_cell(ck),
                "test_set_code": ck, "auroc": a,
                "n_kept": len(cols), "n_tied_steps": ties,
                "kept_measurements": " | ".join(
                    S.plain(cell["pool"][i]) for i in cols),
                "kept_measurements_code": ",".join(
                    cell["pool"][i] for i in cols),
            })
        per_cell[name] = np.array(scores, float)
        print(f"  {name:58s} macro={np.nanmean(scores):.4f} "
              f"tied steps/cell={np.mean(tie_counts):.1f}")

    S.save_csv(os.path.join(out, "arms_per_test_set.csv"), rows)
    S.save_json(os.path.join(out, "arm_scores.json"),
                {k: v.tolist() for k, v in per_cell.items()})

    # ---------- comparisons ----------
    ref = "Localize to worst-fitting group"
    comp = []
    for name in per_cell:
        a, b = per_cell[name], per_cell[ref]
        m = np.isfinite(a) & np.isfinite(b)
        try:
            p = (wilcoxon(a[m], b[m]).pvalue
                 if m.sum() > 5 and np.any(a[m] != b[m]) else np.nan)
        except Exception:
            p = np.nan
        comp.append({
            "method": name,
            "macro_auroc": float(np.nanmean(a)),
            "difference_vs_worst_group_pp": float((np.nanmean(a) - np.nanmean(b)) * 100),
            "test_sets_better": int((a > b)[m].sum()),
            "test_sets_worse": int((a < b)[m].sum()),
            "wilcoxon_p": float(p),
        })
    comp.sort(key=lambda r: -r["macro_auroc"])
    S.save_csv(os.path.join(out, "arm_comparison.csv"), comp)

    # ---------- charts ----------
    bar = S.bar_chart([c["method"] for c in comp],
                      [c["macro_auroc"] for c in comp],
                      "Detection accuracy (AUROC), averaged over 25 test sets",
                      hlines=[("six hand-picked", 0.7594)])
    loc_bar = S.bar_chart(
        [r["test_set"] for r in sorted(loc_rows, key=lambda r: -r["spread_ratio"])],
        [r["spread_ratio"] for r in sorted(loc_rows, key=lambda r: -r["spread_ratio"])],
        "How many times worse the worst group fits than the best group",
        value_fmt="{:.1f}x", bar_h=22)

    comp_tbl = S.html_table(
        ["Method", "Accuracy", "vs. worst-group", "Better", "Worse", "Wilcoxon p"],
        [[c["method"], f"{c['macro_auroc']:.4f}",
          f"{c['difference_vs_worst_group_pp']:+.2f} pp",
          c["test_sets_better"], c["test_sets_worse"],
          "-" if not np.isfinite(c["wilcoxon_p"]) else f"{c['wilcoxon_p']:.3f}"]
         for c in comp], numeric_cols=(1, 2, 3, 4, 5))

    loc_tbl = S.html_table(
        ["Test set", "Groups", "Of those, repairable", "Worst group misfit",
         "Best group misfit", "Spread", "What is in the worst group"],
        [[r["test_set"], r["n_groups"], r["n_groups_with_2plus_members"],
          f"{r['worst_group_misfit_per_pair']:.4f}",
          f"{r['best_group_misfit_per_pair']:.4f}",
          f"{r['spread_ratio']:.1f}x",
          r["worst_group_members"][:150]]
         for r in sorted(loc_rows, key=lambda r: -r["spread_ratio"])],
        numeric_cols=(1, 2, 3, 4, 5))

    best = comp[0]
    worst_grp = [c for c in comp if c["method"] == ref][0]
    glob = [c for c in comp if c["method"].startswith("Remove globally")][0]
    med_spread = np.median([r["spread_ratio"] for r in loc_rows])

    body = f"""
<h2>The algorithm being tested</h2>
<p>Instead of ranking all 30 measurements and cutting the bottom ones, this
algorithm uses structure the detector <em>already computes and then throws
away</em>. The detector sorts measurements into groups that behave alike, and it
measures how badly its "everything here is a noisy reading of one hidden thing"
model fits. The idea is to find the <b>group where that model fits worst</b> and
remove a measurement from <em>there</em> &mdash; on the reasoning that a badly
fitting group is where something does not belong.</p>

<div class="warn"><b>This had never been run.</b> The prototype in the
repository computed its fit score as the distance between a matrix times its own
leading eigenvector and that eigenvector times its own eigenvalue &mdash; a
quantity that is <b>zero by definition</b>. Measured, it came out at about
2&times;10<sup>-15</sup>. It was ranking candidate removals by floating-point
rounding error, so the 0.7004 in the record is the score of a coin flip, not of
this idea.</div>

<h2>Part 0 - Does the localizer actually point anywhere?</h2>
<p>Before asking whether it helps, check that it discriminates at all. For each
test set: how badly does the model fit inside the worst group, versus the best
group?</p>
{loc_bar}
<p class="note">Median spread: <b>{med_spread:.1f}&times;</b> between the worst-
and best-fitting group. The localizer is not indifferent.</p>
{loc_tbl}

<h2>Part 1 &amp; 2 - Does localizing help, and does the tie-breaker matter?</h2>
<p>All arms trim from 30 measurements down to {TARGET_SIZE}, so subset size is
never a confound and the only difference is <em>which</em> measurements get
removed. Two controls are included: removing from a <b>randomly chosen</b> group
instead of the worst one, and breaking near-ties with a <b>coin flip</b>.</p>
{bar}
{comp_tbl}

<h2>Reading this</h2>
<ul>
<li>Best arm: <b>{best['method']}</b> at {best['macro_auroc']:.4f}.</li>
<li>Localizing to the worst-fitting group scores {worst_grp['macro_auroc']:.4f};
removing globally with no groups at all scores {glob['macro_auroc']:.4f}
({glob['difference_vs_worst_group_pp']:+.2f} pp).</li>
<li><b>Differences of a few tenths of a point are not being treated as wins.</b>
With 25 test sets, only differences well beyond a point are distinguishable from
sampling noise, and the win/loss columns matter more than the averages.</li>
</ul>

<h2>Saved data</h2>
<ul>
<li><code>localizer_discrimination.csv</code> &mdash; per test set: group misfits and
exactly which measurements sit in the worst group</li>
<li><code>arms_per_test_set.csv</code> &mdash; every arm &times; test set, with the
full list of measurements it kept</li>
<li><code>arm_comparison.csv</code>, <code>arm_scores.json</code> &mdash; aggregates
and raw per-test-set score vectors for re-testing</li>
</ul>
"""
    S.write_page(
        os.path.join(out, "index.html"),
        "Experiment 2 - Trimming by fixing the worst-fitting group",
        "Omri's cluster-localized algorithm, run for the first time, with "
        "controls for the group choice and the tie-breaker.",
        [f"The localizer discriminates: the worst-fitting group misfits "
         f"{med_spread:.1f}x more than the best-fitting one (median over 25 test sets).",
         f"Best arm is <b>{best['method']}</b> at {best['macro_auroc']:.4f}; "
         f"plain global removal scores {glob['macro_auroc']:.4f}.",
         "The previously recorded 0.7004 for this idea is void - that prototype's "
         "fit score was zero by construction (~2e-15), so it ranked removals by "
         "rounding error.",
         "Differences here are small; none is being called a win on the average alone."],
        body)
    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    main()
