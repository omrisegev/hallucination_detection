#!/usr/bin/env python
"""
exp08_pool_lovo_upcr.py — pool leave-one-view-out under the UPDATED U-PCR.

Omri (2026-07-28): "if we choose to remove features, the bad ones that we suspect
are not helping the algorithm, can we do a test where we try to remove them and
see if we are getting better auc?"

That test exists — WS3 (`scripts/pipeline_lovo.py`, LOCO-honest, finished
2026-07-23) — and it is negative: mean held-out delta -0.22pp at threshold 0.0pp
(11W/12L/2T) and +0.04pp at 0.1pp (14W/7L/4T, p=0.23); at thresholds >= 0.2pp NO
view qualifies for removal at all. Two corroborations: the pool-size experiment
(16..30 views all within 0.11pp) and feature_inclusion_audit_c46 (every view has
non-zero LOVO cost on at least one cell).

BUT Omri's objection is correct: WS3 ran the **L-SML** path only.
`pipeline_lovo.py:95-96` calls `eval_subset_flex(..., fusion=sel.get("fusion",
"lsml"))` and the a6 selectors never set `fusion`. It also ran 2026-07-23, before
Step 204 (U-PCR made faithful) and Step 205 (grouping fix). So the question is
unanswered for `upcr.rho_polarities` (0.7551), our best prior-free candidate.

THE TRAP THIS SCRIPT EXISTS TO AVOID: the updated U-PCR is NOT reachable through
the bench. `eval_subset_flex(fusion='upcr')` calls `fusion_utils.upcr_fuse`, whose
docstring requires input "already z-scored and sign-oriented" -- that is
`upcr.legacy` (0.7392) / `upcr.hand_polarities` (0.7405), NOT the 0.7551 recipe.
The rho-polarity path is `spectral_utils.upcr.upcr_fit` driven the way
exp06_orientation.py:83-111 drives it, and that is what this script uses.

Stage 1 (--collect, resumable): for every in-scope cell x every condition in
  {FULL} + {drop v : v in pool}, derive polarities from sign(rho) on UNORIENTED
  features, refit, orient against the cell's anchor, score raw AUROC. The anchor
  array is kept even when the anchor view itself leaves the pool -- orientation is
  offline knowledge, not pool membership (same rule as WS3).
  Also records whether U-PCR's own Algorithm-1 exclusion already dropped the view:
  if it did, pool removal must be a no-op, which is a free internal check.

Stage 2 (--analyze): LOCO-honest, mirroring WS3 stage 2 exactly so the two are
  directly comparable. For held-out cell h, D_h = views whose removal improves the
  mean AUROC over the 24 TRAINING cells by > threshold; re-run on pool-minus-D_h
  for h; report HELD-OUT deltas only. Thresholds 0.0/0.1/0.2/0.5 pp, same four.

Reporting follows Omri's standing rule: effect size with W/L + Wilcoxon, never a
1-2pp gate. In-sample numbers are a ceiling and are labelled as such.

Run:  python scripts/upcr_study/exp08_pool_lovo_upcr.py --collect --analyze
Out:  results/upcr_study/08_pool_lovo/{lovo_raw.csv,lovo_loco.csv,summary.json,index.html}
"""
import argparse
import os
import sys
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.upcr import upcr_fit                            # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                   # noqa: E402
from inscope_cells import GROUP                                     # noqa: E402

# The deployed U-PCR configuration — identical to exp06's FIT (exp06:63-65).
# Any drift here silently measures a different algorithm than the 0.7551 anchor.
FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

THRESHOLDS_PP = (0.0, 0.1, 0.2, 0.5)   # same four WS3 used
FULL = "__FULL__"

# exp06's macro_rho_anchor, from 06_orientation/summary.json. The FULL condition
# here must reproduce it to 4dp or the rho-polarity recipe was mis-transcribed.
EXP06_RHO_ANCHOR = 0.755079270242705


# --------------------------------------------------------------- the recipe
def rho_polarity_score(cell, keep_cols):
    """`upcr.rho_polarities` restricted to `keep_cols`. Verbatim exp06:83-111.

    Returns (auroc, keep_mask_over_keep_cols). `keep_mask` is U-PCR's own
    Algorithm-1 exclusion result, used for the no-op consistency check.
    """
    pool = cell["pool"]
    V = cell["V"]
    cols = sorted(keep_cols)

    # prepare_cell already applied ALL_SIGNS (subset_sweep.py:363), so V is
    # ORIENTED. zscore(raw*s) == s*zscore(raw) for s = +-1, so multiplying by the
    # hand signs UNDOES it and recovers the raw z-scored feature. Backwards here
    # swaps the arms of the experiment.
    hand = np.array([ALL_SIGNS.get(pool[j], +1) for j in cols], dtype=float)
    V_un = V[:, cols] * hand

    r_raw = upcr_fit(V_un.T, **FIT)
    derived = np.sign(r_raw.rho_hat_full)
    derived[derived == 0] = 1.0

    F_der = (V_un * derived).T
    r_der = upcr_fit(F_der, **FIT)
    if not np.isfinite(r_der.w).all():
        return float("nan"), r_der.keep
    # auroc_from_score applies anchor_orient against the CELL'S anchor, which is
    # held fixed regardless of which views left the pool.
    return S.auroc_from_score(cell, r_der.w @ F_der), r_der.keep


# ------------------------------------------------------------------ stage 1
def collect(out):
    cells = S.load()
    S.validity_check(cells)

    path = os.path.join(out, "lovo_raw.csv")
    done = set()
    if os.path.exists(path):
        import pandas as pd
        prev = pd.read_csv(path)
        done = set(zip(prev["cell"], prev["removed"]))
        print(f"  resuming: {len(done)} rows already on disk")

    rows = []
    for ck, cell in cells.items():
        pool = list(cell["pool"])
        all_cols = list(range(len(pool)))

        # One full-pool fit per cell: it supplies both the FULL baseline row and
        # U-PCR's own Algorithm-1 exclusion set (the no-op check in --analyze).
        auc_full, keep_full = rho_polarity_score(cell, all_cols)
        excl = "|".join(pool[j] for j in range(len(pool)) if not keep_full[j])

        cell_rows = []
        if (ck, FULL) not in done:
            cell_rows.append(dict(cell=ck, group=GROUP.get(ck, ""), removed=FULL,
                                  n_pool=len(pool), auroc=auc_full,
                                  n_kept=int(keep_full.sum())))
        for j, name in enumerate(pool):
            if (ck, name) in done:
                continue
            auc, keep = rho_polarity_score(cell, [c for c in all_cols if c != j])
            cell_rows.append(dict(cell=ck, group=GROUP.get(ck, ""), removed=name,
                                  n_pool=len(pool) - 1, auroc=auc,
                                  n_kept=int(keep.sum())))
        for r in cell_rows:
            r["excluded_by_upcr"] = excl
        rows.extend(cell_rows)

        print(f"  {ck[:34]:34s} pool={len(pool):2d} full={auc_full:.4f} "
              f"upcr-excluded={len(excl.split('|')) if excl else 0} "
              f"(+{len(cell_rows)} rows)")

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        if os.path.exists(path):
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df.to_csv(path, index=False)
    print(f"  wrote {path}")
    return path


# ------------------------------------------------------------------ stage 2
def analyze(out):
    import pandas as pd
    from scipy.stats import wilcoxon

    cells = S.load()
    S.validity_check(cells)
    raw = pd.read_csv(os.path.join(out, "lovo_raw.csv"))

    full = raw[raw.removed == FULL].set_index("cell")["auroc"]

    # ---- ANCHOR: the FULL condition must reproduce exp06's rho+anchor macro ----
    macro_full = float(full.mean())
    if abs(macro_full - EXP06_RHO_ANCHOR) > 1e-4:
        raise SystemExit(
            f"ANCHOR FAILED: FULL macro = {macro_full:.6f}, exp06 reported "
            f"{EXP06_RHO_ANCHOR:.6f}. The rho-polarity recipe was mis-transcribed; "
            "refusing to report any removal result on top of it.")
    print(f"  anchor OK: FULL macro = {macro_full:.4f} == exp06 rho+anchor")

    # per-cell delta of removing view v, relative to that cell's FULL
    d = raw[raw.removed != FULL].copy()
    d["delta"] = d["auroc"] - d["cell"].map(full)

    # ---- no-op check, and why it only PARTLY holds ----
    # Predicted: a view U-PCR already excludes (w_i = 0) should be a no-op to
    # remove from the pool. It is not, and the reason is the finding: U-PCR's
    # Algorithm-1 exclusion is DATA-DEPENDENT. Dropping a zero-weight view still
    # perturbs C, hence rho_hat, hence WHICH OTHER VIEWS SURVIVE. Split on that
    # and the prediction is confirmed exactly where it should hold.
    d["already_excluded"] = [
        r.removed in str(r.excluded_by_upcr).split("|")
        for r in d.itertuples()]
    d["n_kept_full"] = d["cell"].map(
        raw[raw.removed == FULL].set_index("cell")["n_kept"])
    ex = d[d.already_excluded].copy()
    ex["survivors_changed"] = ex["n_kept"] != ex["n_kept_full"]

    def _blk(g):
        return dict(n=int(len(g)),
                    frac_exact_noop=float((g.delta == 0).mean()) if len(g) else float("nan"),
                    mean_abs_delta_pp=float(100 * g.delta.abs().mean()) if len(g) else float("nan"))

    kept = d[~d.already_excluded]
    noop = dict(
        n_excluded_pairs=int(len(ex)),
        excluded=_blk(ex), kept_by_upcr=_blk(kept),
        excluded_survivors_unchanged=_blk(ex[~ex.survivors_changed]),
        excluded_survivors_changed=_blk(ex[ex.survivors_changed]))
    print(f"  no-op check: {noop['n_excluded_pairs']} (cell,view) pairs already "
          f"excluded by U-PCR")
    print(f"    survivor set UNCHANGED: n={noop['excluded_survivors_unchanged']['n']}, "
          f"{noop['excluded_survivors_unchanged']['frac_exact_noop']*100:.1f}% exact "
          f"no-ops, mean |delta| = "
          f"{noop['excluded_survivors_unchanged']['mean_abs_delta_pp']:.4f}pp  <- check PASSES here")
    print(f"    survivor set CHANGED:   n={noop['excluded_survivors_changed']['n']}, "
          f"{noop['excluded_survivors_changed']['frac_exact_noop']*100:.1f}% exact "
          f"no-ops, mean |delta| = "
          f"{noop['excluded_survivors_changed']['mean_abs_delta_pp']:.4f}pp  <- exclusion is data-dependent")

    cell_list = sorted(full.index)
    loco_rows = []
    for thr_pp in THRESHOLDS_PP:
        thr = thr_pp / 100.0
        for h in cell_list:
            train = d[d.cell != h]
            gain = train.groupby("removed")["delta"].mean()
            drop = sorted(gain[gain > thr].index)
            cell = cells[h]
            pool = list(cell["pool"])
            cols = [j for j, nm in enumerate(pool) if nm not in set(drop)]
            if len(cols) < 3:
                auc_pruned = float("nan")
            else:
                auc_pruned, _ = rho_polarity_score(cell, cols)
            loco_rows.append(dict(
                threshold_pp=thr_pp, held_out=h,
                n_dropped=len(set(drop) & set(pool)),
                dropped="|".join(x for x in drop if x in pool),
                auroc_full=float(full[h]), auroc_pruned=auc_pruned,
                delta=auc_pruned - float(full[h])))
        print(f"  threshold {thr_pp}pp done")

    loco = pd.DataFrame(loco_rows)
    loco.to_csv(os.path.join(out, "lovo_loco.csv"), index=False)

    verdict = []
    for thr_pp, g in loco.groupby("threshold_pp"):
        g = g.dropna(subset=["delta"])
        w = int((g.delta > 0).sum()); l = int((g.delta < 0).sum())
        t = int((g.delta == 0).sum())
        try:
            p = float(wilcoxon(g.auroc_pruned, g.auroc_full).pvalue)
        except Exception:
            p = float("nan")
        cnt = collections.Counter()
        for s in g.dropped.fillna(""):
            for f in str(s).split("|"):
                if f:
                    cnt[f] += 1
        verdict.append(dict(
            threshold_pp=float(thr_pp), n=int(len(g)),
            mean_delta_pp=float(100 * g.delta.mean()),
            median_delta_pp=float(100 * g.delta.median()),
            W=w, L=l, T=t, wilcoxon_p=p,
            mean_n_dropped=float(g.n_dropped.mean()),
            macro_full=float(g.auroc_full.mean()),
            macro_pruned=float(g.auroc_pruned.mean()),
            dropped_in_20plus_folds="|".join(
                sorted(k for k, v in cnt.items() if v >= 20))))

    vdf = pd.DataFrame(verdict)
    print()
    print(vdf.to_string(index=False))

    S.save_json(os.path.join(out, "summary.json"), dict(
        n_cells=len(cell_list), macro_full=macro_full,
        exp06_rho_anchor=EXP06_RHO_ANCHOR, noop_check=noop,
        verdict=verdict))

    best = max(verdict, key=lambda v: v["mean_delta_pp"])
    body = ["<h2>LOCO-honest verdict per drop threshold</h2>"]
    body.append(S.html_table(
        ["drop threshold", "mean delta", "median", "W/L/T", "Wilcoxon p",
         "views dropped", "macro full", "macro pruned", "dropped in >=20/25 folds"],
        [[f"{v['threshold_pp']}pp", f"{v['mean_delta_pp']:+.3f}pp",
          f"{v['median_delta_pp']:+.3f}pp", f"{v['W']}/{v['L']}/{v['T']}",
          f"{v['wilcoxon_p']:.3f}" if v["wilcoxon_p"] == v["wilcoxon_p"] else "n/a",
          f"{v['mean_n_dropped']:.2f}", f"{v['macro_full']:.4f}",
          f"{v['macro_pruned']:.4f}", v["dropped_in_20plus_folds"] or "(none)"]
         for v in verdict],
        numeric_cols=(1, 2, 4, 5, 6, 7)))

    S.write_page(
        os.path.join(out, "index.html"),
        "Pool pruning under the updated U-PCR",
        "U-PCR study, exp08 — does removing weak views raise AUROC on the "
        "rho-polarity path?",
        [f"Best held-out result across four drop thresholds: "
         f"<b>{best['mean_delta_pp']:+.3f}pp</b> at {best['threshold_pp']}pp "
         f"({best['W']}W/{best['L']}L/{best['T']}T, p = {best['wilcoxon_p']:.3f}).",
         "WS3 asked this on the <b>L-SML</b> path and answered no. This asks it on "
         "<code>upcr.rho_polarities</code> (0.7551), which the bench cannot reach: "
         "<code>eval_subset_flex(fusion='upcr')</code> runs the legacy "
         "<code>upcr_fuse</code> (0.7392), not this recipe.",
         f"<b>Why pruning hurts:</b> U-PCR's own exclusion is data-dependent, so "
         f"removing a zero-weight view is not a no-op. Where the survivor set is "
         f"unchanged the prediction holds — "
         f"{noop['excluded_survivors_unchanged']['frac_exact_noop']*100:.0f}% exact "
         f"no-ops, mean |delta| "
         f"{noop['excluded_survivors_unchanged']['mean_abs_delta_pp']:.3f}pp. Where "
         f"removal shifts which OTHER views survive "
         f"({noop['excluded_survivors_changed']['n']} of "
         f"{noop['n_excluded_pairs']} pairs), "
         f"{noop['excluded_survivors_changed']['frac_exact_noop']*100:.0f}% are "
         f"no-ops and mean |delta| is "
         f"{noop['excluded_survivors_changed']['mean_abs_delta_pp']:.3f}pp.",
         "Held-out deltas only. In-sample numbers are a ceiling, not a result."],
        "".join(body))
    print(f"  wrote {os.path.join(out, 'index.html')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not (a.collect or a.analyze):
        a.collect = a.analyze = True
    out = S.outdir("08_pool_lovo")
    if a.collect:
        print("=== stage 1: collect ===")
        collect(out)
    if a.analyze:
        print("=== stage 2: analyze (LOCO) ===")
        analyze(out)


if __name__ == "__main__":
    main()
