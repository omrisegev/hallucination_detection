#!/usr/bin/env python
"""
run_eval_pipeline.py — the orchestrated evaluation checkpoint (Omri, 2026-07-22).

One command that ends a session with CURRENT numbers instead of stale ones.
The last ~50 steps kept re-deriving comparisons by hand and paying for it
(stale rows after re-grades in three artifact types, OVERRIDE_Y hack,
macro-only comparisons quoting two pools side by side). This script runs the
whole chain in order, resume-safe, and stamps the result:

  1. stale-gate   every bench row's n vs the live featcache n (the Step-193
                  "staleness carrier" failure mode); --fix-stale drops stale
                  rows so stage 2 re-scores them.
  2. bench        every registered selector family on the c46 pool, 25
                  in-scope cells (resume-safe: fresh rows are skipped, new
                  variants — e.g. a freshly registered reference subset —
                  are appended).
  3. compare      selector_compare_inscope.py -> comparison_inscope.csv
  4. matrix       cell_method_matrix.py -> per-cell x method CSV + heatmap
  4b. oracle      cell_oracle_vs_chosen.py -> per-cell ceiling vs OUR
                  ALGORITHM's actual chosen subset
  4c. glossary    build_glossary.py -> GLOSSARY.md (--allow-gaps: a coverage
                  gap warns here, not a pipeline failure; run
                  `build_glossary.py` bare to enforce it as a hard gate)
  5. reports      advisor_inscope_report.py (--skip-reports to omit)
  6. scoreboard   results/checkpoints/scoreboard_<date>_<rev>.csv — the
                  versioned "where we stand" artifact, with a `role` column
                  (reference_macro / OUR ALGORITHM / fs_selector_candidate /
                  router — see GLOSSARY.md) so a fixed subset is never
                  confused with our algorithm's own output.

The legacy all-cell chain (score_repgrid -> repgrid_report -> advisor_report,
which also refreshes scores_lsml_upcr.csv and retires report_figs.OVERRIDE_Y)
is NOT run by default — pass --legacy-reports for it; it rescores the full
50+-cell grid and takes much longer.

Usage:
    python scripts/run_eval_pipeline.py                  # full default chain
    python scripts/run_eval_pipeline.py --fix-stale      # + drop stale rows
    python scripts/run_eval_pipeline.py --skip-reports   # CSVs only, no HTML
"""
import argparse
import datetime as _dt
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    # line_buffering: this script is normally redirected to a log and watched
    # while it runs (bench stages take tens of minutes) — block buffering would
    # show nothing until it exits.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                           line_buffering=True)

from inscope_cells import INSCOPE  # noqa: E402
# Role tagging lives in spectral_utils/glossary.py (single source of truth,
# also feeds GLOSSARY.md) — imported, not reimplemented, so the scoreboard's
# tags and the glossary's tags can never drift apart.
from spectral_utils.glossary import OUR_ALGORITHM, role_of as _role  # noqa: E402

BENCH = os.path.join(REPO, "results", "selector_bench")
CHECKPOINTS = os.path.join(REPO, "results", "checkpoints")
COMPARISON = os.path.join(BENCH, "comparison_inscope.csv")

# simple_stats is the random/MAD floor — cheap, keep it so the floor row on
# the leaderboard is always current too.
SKIP_FAMILIES = ()



def _git_rev():
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            text=True).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPO).returncode != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _run(script, *args):
    cmd = [sys.executable, os.path.join(REPO, "scripts", script), *args]
    print(f"\n=== {script} {' '.join(args)} ===")
    sys.stdout.flush()
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(f"PIPELINE FAIL at {script} (exit {r.returncode})")
        sys.exit(r.returncode)


# ------------------------------------------------------------ stage 1: gate
def stale_gate(fix=False):
    """Compare every in-scope c46 bench row's n against the live cell n."""
    from spectral_utils.selector_bench import iter_prepared_cells
    live_n = {}
    for ctx in iter_prepared_cells(REPO, "c46", ["repgrid"], None):
        live_n[ctx.cell_key] = ctx.V.shape[0]

    total_stale = 0
    for f in sorted(glob.glob(os.path.join(BENCH, "*__c46.csv"))):
        df = pd.read_csv(f)
        mask = df["cell"].isin(INSCOPE) & df["cell"].isin(live_n)
        stale = df[mask & (df["n"] != df["cell"].map(live_n))]
        if len(stale):
            cells = sorted(stale["cell"].unique())
            print(f"  STALE {os.path.basename(f)}: {len(stale)} rows, "
                  f"cells {cells}")
            total_stale += len(stale)
            if fix:
                shutil.copy2(f, f + ".bak")
                df.drop(stale.index).to_csv(f, index=False)
                print(f"    dropped (backup {os.path.basename(f)}.bak); "
                      f"bench stage will re-score")
    if total_stale == 0:
        print("  stale-gate PASS: every bench row matches the live "
              f"featcache n ({len(live_n)} live cells)")
    elif not fix:
        print(f"  stale-gate: {total_stale} stale rows FOUND but NOT fixed "
              "— re-run with --fix-stale")
    return total_stale


# ----------------------------------------------------------- stage 2: bench
def bench_all(seed=0, inscope_only=False):
    """Re-bench every registered family. inscope_only restricts to the 25
    QA+math cells — the ones every headline uses. Worth it: the selector runs
    (and pays full cost) before the resume-skip check, so an unrestricted pass
    re-computes ~26 out-of-scope RAG/GPQA cells it will then discard."""
    from spectral_utils.selectors import registered
    from spectral_utils.selector_bench import bench_selector
    npz_dir = os.path.join(REPO, "results", "subset_sweep")
    cells = list(INSCOPE) if inscope_only else None
    new_total = 0
    for name in sorted(registered()):
        if name in SKIP_FAMILIES:
            continue
        out = os.path.join(BENCH, f"{name}__c46.csv")
        n = bench_selector(name, "c46", REPO, npz_dir, out,
                           seed=seed, domains=["repgrid"], cells=cells)
        print(f"  {name}: {n} new rows")
        new_total += n
    return new_total


def _coverage_matched_deltas(df, best_ref_variant):
    """delta_vs_current_best_ref (macro_all subtraction) is confounded whenever
    n_cells differs between a row and best_ref — e.g. ref.LOCO_5 (24 cells,
    missing inside_coqa_llama7b) vs ref.GOOD_6 (25 cells): the raw macro_all
    delta (+1.11pp) partly reflects GOOD_6 paying for a below-median cell
    LOCO_5 never has to score, not just a genuine win. Recompute on the
    INTERSECTION of cells each row actually shares with best_ref so a smaller-
    coverage variant can never look better purely by dropping hard cells.
    (Omri, 2026-07-23: flagged this exact confound for LOCO_5 vs GOOD_6.)
    """
    from cell_method_matrix import load_bench_rows
    bench = load_bench_rows()
    best_cells = set(bench.loc[bench["variant"] == best_ref_variant, "cell"])
    matched, shared_n = [], []
    for variant in df["variant"]:
        v_cells = set(bench.loc[bench["variant"] == variant, "cell"])
        shared = best_cells & v_cells
        if not shared:
            matched.append(float("nan"))
            shared_n.append(0)
            continue
        v_macro = bench.loc[(bench["variant"] == variant) &
                            (bench["cell"].isin(shared)), "auroc"].mean()
        ref_macro = bench.loc[(bench["variant"] == best_ref_variant) &
                              (bench["cell"].isin(shared)), "auroc"].mean()
        matched.append(v_macro - ref_macro)
        shared_n.append(len(shared))
    df["n_cells_shared_with_best_ref"] = shared_n
    df["delta_vs_current_best_ref_MATCHED"] = matched
    return df


# ------------------------------------------------------ stage 6: scoreboard
def scoreboard(rev, stale_found, new_rows):
    os.makedirs(CHECKPOINTS, exist_ok=True)
    date = _dt.date.today().isoformat()
    out = os.path.join(CHECKPOINTS, f"scoreboard_{date}_{rev}.csv")
    df = pd.read_csv(COMPARISON)
    df.insert(0, "checkpoint_date", date)
    df.insert(1, "git_rev", rev)
    df.insert(2, "stale_rows_at_gate", stale_found)
    df.insert(3, "new_bench_rows", new_rows)
    df["role"] = df["variant"].apply(_role)
    # delta_vs_good5_all is a Step-186 legacy zero-point (predates GOOD_6 and
    # LOCO_5) — keep it (other scripts read it) but also report delta against
    # whichever reference_macro is CURRENTLY on top, so "beats the baseline"
    # never silently means "beats a subset we've already superseded".
    ref_rows = df[df["variant"].str.startswith("ref.")]
    if len(ref_rows):
        best_ref = ref_rows.loc[ref_rows["macro_all"].idxmax()]
        df["current_best_ref"] = best_ref["variant"]
        df["delta_vs_current_best_ref"] = df["macro_all"] - best_ref["macro_all"]
        # raw macro_all subtraction is confounded when n_cells differs (a
        # variant with fewer, easier cells can look ahead without a genuine
        # win) — add the coverage-matched version alongside it, never in
        # place of it, so both are on record.
        try:
            df = _coverage_matched_deltas(df, best_ref["variant"])
        except Exception as e:
            print(f"warning: coverage-matched delta failed ({e}); "
                  f"only the raw (unmatched) delta is available this run")
    df.to_csv(out, index=False)
    # stable pointer to the latest checkpoint
    latest = os.path.join(CHECKPOINTS, "scoreboard_latest.csv")
    shutil.copy2(out, latest)
    print(f"\nscoreboard -> {out}")
    if len(ref_rows):
        mismatched = df[(df["n_cells"] != best_ref["n_cells"]) &
                        (df["variant"] != best_ref["variant"])]
        if len(mismatched):
            print(f"NOTE: {len(mismatched)} rows have different cell "
                  f"coverage than {best_ref['variant']} ({best_ref['n_cells']} "
                  f"cells) — read delta_vs_current_best_ref_MATCHED, not the "
                  f"raw delta_vs_current_best_ref, for those.")
        print(f"current best reference macro: {best_ref['variant']} "
              f"({best_ref['macro_all']:.4f}) — deltas below are vs THIS, "
              f"not vs the legacy GOOD_5 zero-point")
    top = df.sort_values("macro_all", ascending=False).head(10)
    cols = ["variant", "role", "pool_mode", "n_cells", "macro_all"]
    if "delta_vs_current_best_ref" in df.columns:
        cols.append("delta_vs_current_best_ref")
    print(top[cols].to_string(index=False))
    our_row = df[df["variant"] == OUR_ALGORITHM]
    if len(our_row):
        r = our_row.iloc[0]
        print(f"\nOUR ALGORITHM ({OUR_ALGORITHM}): macro {r['macro_all']:.4f}, "
              f"rank {int((df['macro_all'] > r['macro_all']).sum()) + 1} of "
              f"{len(df)} — {int((df['macro_all'] > r['macro_all']).sum())} "
              f"fixed/other subsets currently outperform it")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix-stale", action="store_true",
                    help="drop stale bench rows (backed up) so they re-score")
    ap.add_argument("--skip-reports", action="store_true",
                    help="skip the advisor_inscope HTML regen")
    ap.add_argument("--legacy-reports", action="store_true",
                    help="ALSO regen the all-cell chain (score_repgrid -> "
                         "repgrid_report -> advisor_report); slow")
    ap.add_argument("--inscope-only", action="store_true",
                    help="bench only the 25 QA+math cells (much faster; the "
                         "out-of-scope RAG/GPQA rows feed no headline)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rev = _git_rev()
    print(f"eval pipeline @ {rev} — {_dt.datetime.now():%Y-%m-%d %H:%M}")

    print("\n=== stage 1: stale-gate ===")
    stale_found = stale_gate(fix=args.fix_stale)

    print("\n=== stage 2: bench (resume-safe, all registered families) ===")
    new_rows = bench_all(seed=args.seed, inscope_only=args.inscope_only)

    _run("selector_compare_inscope.py")
    _run("cell_method_matrix.py")
    _run("cell_oracle_vs_chosen.py")
    _run("build_glossary.py", "--allow-gaps")

    if not args.skip_reports:
        _run("advisor_inscope_report.py")
    if args.legacy_reports:
        _run("score_repgrid.py")
        _run("repgrid_report.py")
        _run("advisor_report.py")

    out = scoreboard(rev, stale_found, new_rows)
    print(f"\nDONE. Checkpoint: {os.path.basename(out)}"
          + ("" if args.legacy_reports else
             "\n(note: legacy all-cell chain not regenerated — "
             "scores_lsml_upcr.csv / OVERRIDE_Y unchanged; use "
             "--legacy-reports when needed)"))


if __name__ == "__main__":
    main()
