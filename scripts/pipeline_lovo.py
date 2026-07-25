#!/usr/bin/env python
"""
pipeline_lovo.py — pipeline-level leave-one-view-out redundancy test (WS3).

Omri's claim (2026-07-22): some of the 30 views are redundant IN THE PIPELINE
sense — remove them, run the full FS+fusion algorithm, and results improve.
Step-195's enumeration prune (empty drop list) and the pool-size experiment
(<=+0.11pp) are proxies; neither ever removed a view and re-ran the actual
selector. This script does exactly that.

Stage 1 (--collect, resumable): for every in-scope cell and every condition
  removed_view in {FULL} + pool(30): run the a6_pseudolabel_gates selector on
  the restricted pool, score its a6.pl_dufs (selector of record, Step 194) and
  a6.dufs (unsupervised control) selections with the SAME eval_subset_flex /
  anchor_orient recipe as the bench. The anchor array is kept even when the
  anchor view itself is removed from the pool — orientation is offline
  knowledge, not a pool member.

Stage 2 (--analyze): LOCO-honest drop-set evaluation. For each held-out cell
  h: D_h = views whose removal improves the mean a6.pl_dufs AUROC over the 24
  TRAINING cells by > --drop-threshold. Then (because multi-view removal does
  not compose from single removals) re-run the selector on pool-minus-D_h for
  cell h and compare against h's full-pool run. Report held-out deltas only.

Outputs: results/advisor_inscope/pipeline_lovo.csv        (stage 1 rows)
         results/advisor_inscope/pipeline_lovo_loco.csv   (stage 2 verdicts)

Usage:
    python scripts/pipeline_lovo.py --collect        # long; background it
    python scripts/pipeline_lovo.py --analyze
"""
import argparse
import csv
import dataclasses
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.selector_bench import (   # noqa: E402
    UnlabeledCell, eval_subset_flex, iter_prepared_cells, _cell_rng)
from spectral_utils.selectors import get_selector  # noqa: E402
from inscope_cells import INSCOPE  # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")
COLLECT_CSV = os.path.join(OUT_DIR, "pipeline_lovo.csv")
LOCO_CSV = os.path.join(OUT_DIR, "pipeline_lovo_loco.csv")
FIELDS = ["cell", "removed", "variant", "auroc", "size", "chosen",
          "fallback", "seconds"]
VARIANTS = ("a6.pl_dufs", "a6.dufs")
FULL = "FULL"


def restricted_cell(ctx, keep):
    """UnlabeledCell with only `keep` columns; anchor array preserved."""
    return UnlabeledCell(
        domain=ctx.domain, cell_key=ctx.cell_key,
        pool=[ctx.pool[j] for j in keep],
        pool_bits=np.asarray(ctx.pool_bits)[keep],
        V=ctx.V[:, keep], anchor=ctx.anchor, anchor_name=ctx.anchor_name,
        rho=ctx.rho[np.ix_(keep, keep)],
        dropped=dict(ctx.dropped), n_imputed=ctx.n_imputed)


def run_condition(ctx, keep, seed=0):
    """Run the a6 selector on the restricted pool, score selected subsets on
    the FULL ctx (cols mapped back), return {variant: row}."""
    cell = restricted_cell(ctx, keep)
    rng = _cell_rng(seed, ctx.domain, ctx.cell_key)
    t0 = time.time()
    try:
        sels = get_selector("a6_pseudolabel_gates")(cell, rng, cache=None)
    except Exception as e:
        return {v: dict(auroc=np.nan, size=0, chosen=f"ERROR:{e}",
                        fallback=True, seconds=round(time.time() - t0, 1))
                for v in VARIANTS}
    secs = round(time.time() - t0, 1)
    out = {}
    for sel in sels:
        if sel["variant"] not in VARIANTS:
            continue
        cols_local = np.asarray(sel["cols"], dtype=np.int64)
        cols_full = np.asarray([keep[j] for j in cols_local], dtype=np.int64)
        try:
            r = eval_subset_flex(ctx, cols_full,
                                 fusion=sel.get("fusion", "lsml"),
                                 groups=sel.get("groups"),
                                 K_override=sel.get("K"))
            auroc = float(r["auroc"])
        except Exception:
            auroc = np.nan
        out[sel["variant"]] = dict(
            auroc=auroc, size=len(cols_full),
            chosen="|".join(ctx.pool[j] for j in cols_full),
            fallback=bool(sel.get("fallback", False)), seconds=secs)
    return out


def _existing(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["cell"], row["removed"], row["variant"]))
    return done


def collect(cells_filter=None, seed=0):
    os.makedirs(OUT_DIR, exist_ok=True)
    done = _existing(COLLECT_CSV)
    new_file = not os.path.exists(COLLECT_CSV)
    with open(COLLECT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for ctx in iter_prepared_cells(REPO, "c46", ["repgrid"], None):
            if ctx.cell_key not in INSCOPE:
                continue
            if cells_filter and ctx.cell_key not in cells_filter:
                continue
            p = ctx.V.shape[1]
            conditions = [(FULL, list(range(p)))] + [
                (ctx.pool[v], [j for j in range(p) if j != v])
                for v in range(p)]
            for removed, keep in conditions:
                if all((ctx.cell_key, removed, v) in done for v in VARIANTS):
                    continue
                rows = run_condition(ctx, keep, seed=seed)
                for variant, r in rows.items():
                    if (ctx.cell_key, removed, variant) in done:
                        continue
                    r.update(cell=ctx.cell_key, removed=removed,
                             variant=variant)
                    w.writerow(r)
                    f.flush()               # incremental — survives kill
                    done.add((ctx.cell_key, removed, variant))
                a = rows.get("a6.pl_dufs", {}).get("auroc", np.nan)
                print(f"[lovo] {ctx.cell_key} -{removed}: pl_dufs "
                      f"{a if isinstance(a, str) else round(a, 4)}",
                      flush=True)
    print("collect done")


def _loco_existing():
    done = set()
    if os.path.exists(LOCO_CSV):
        for _, r in pd.read_csv(LOCO_CSV).iterrows():
            done.add((float(r["threshold_pp"]), r["held_out"]))
    return done


def analyze(thresholds=(0.0, 0.1, 0.2, 0.5), seed=0):
    """LOCO-honest pruning verdict, swept over drop thresholds.

    A single threshold is not enough: at 0.0pp every view with a positive mean
    training delta is dropped, which on a noisy 30-view pool can remove half the
    pool at once — that measures "does a small pool help", not "are these views
    redundant". Sweeping the threshold separates the two: a real redundancy
    effect should survive at a strict threshold where only a few views go."""
    df = pd.read_csv(COLLECT_CSV)
    df = df[df["variant"] == "a6.pl_dufs"].copy()
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
    full = df[df["removed"] == FULL].set_index("cell")["auroc"]
    lovo = df[df["removed"] != FULL]
    # Pool sizes vary per cell (27-30 views: some cells lack energy/logprob
    # views), so "complete" cannot be a global view count. A cell's sweep is
    # complete when it removed every view IN ITS OWN pool once — which the
    # collect loop guarantees on clean exit. Use each cell's own p_pool from
    # the live featcache as the target.
    pool_p = {ctx.cell_key: ctx.V.shape[1]
              for ctx in iter_prepared_cells(REPO, "c46", ["repgrid"], None)
              if ctx.cell_key in INSCOPE}
    per_cell = lovo.groupby("cell")["removed"].nunique()
    complete = sorted(c for c in full.index
                      if per_cell.get(c, 0) >= pool_p.get(c, 99))
    print(f"{len(full)} cells with FULL rows; {len(complete)}/25 with a "
          f"complete per-pool LOVO sweep; analyzing those")
    if len(complete) < 3:
        print("not enough complete cells yet — re-run --analyze later")
        return

    delta = (lovo[lovo["cell"].isin(complete)]
             .pivot_table(index="cell", columns="removed", values="auroc")
             .sub(full, axis=0))

    mean_delta = delta.mean(axis=0).sort_values(ascending=False) * 100
    print("\nIN-SAMPLE mean delta per removed view (pp, top/bottom 8) — an "
          "upper bound, NOT the result:")
    print(mean_delta.head(8).round(3).to_string())
    print("  ...")
    print(mean_delta.tail(4).round(3).to_string())
    n_pos = int((mean_delta > 0).sum())
    print(f"  {n_pos}/{len(mean_delta)} views have positive mean in-sample delta")

    ctxs = {c.cell_key: c for c in
            iter_prepared_cells(REPO, "c46", ["repgrid"], None)
            if c.cell_key in complete}
    done = _loco_existing()
    new_file = not os.path.exists(LOCO_CSV)
    fields = ["threshold_pp", "held_out", "n_dropped", "dropped",
              "auroc_full", "auroc_pruned", "delta"]
    with open(LOCO_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for thr_pp in thresholds:
            thr = thr_pp / 100.0
            for h in complete:
                if (float(thr_pp), h) in done:
                    continue
                train = delta.drop(index=h, errors="ignore")
                d_h = sorted(train.columns[train.mean(axis=0) > thr])
                ctx = ctxs.get(h)
                if ctx is None:
                    continue
                if not d_h:
                    held = float(full[h])
                else:
                    keep = [j for j in range(ctx.V.shape[1])
                            if ctx.pool[j] not in d_h]
                    if len(keep) < 3:
                        continue
                    held = run_condition(ctx, keep, seed=seed)[
                        "a6.pl_dufs"]["auroc"]
                row = dict(threshold_pp=thr_pp, held_out=h,
                           n_dropped=len(d_h), dropped="|".join(d_h),
                           auroc_full=float(full[h]), auroc_pruned=float(held),
                           delta=float(held) - float(full[h]))
                w.writerow(row)
                f.flush()
                print(f"[loco thr={thr_pp}pp] {h}: drop {len(d_h)} -> "
                      f"{row['delta']*100:+.2f}pp", flush=True)

    res = pd.read_csv(LOCO_CSV)
    print("\n=== LOCO-HONEST PRUNING VERDICT (held-out only) ===")
    for thr_pp, g in res.groupby("threshold_pp"):
        wins = int((g["delta"] > 0).sum())
        losses = int((g["delta"] < 0).sum())
        print(f"  threshold {thr_pp:>4}pp: mean {g['delta'].mean()*100:+.3f}pp "
              f"({wins}W/{losses}L, {len(g)} folds, "
              f"mean {g['n_dropped'].mean():.1f} views dropped)")
    print(f"wrote {LOCO_CSV}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--cells", default=None, help="comma list filter")
    ap.add_argument("--thresholds", default="0.0,0.1,0.2,0.5",
                    help="comma list of drop thresholds in PP: a view is "
                         "dropped when its mean training delta exceeds this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.collect:
        collect(set(args.cells.split(",")) if args.cells else None,
                seed=args.seed)
    if args.analyze:
        analyze(thresholds=[float(t) for t in args.thresholds.split(",")],
                seed=args.seed)
    if not (args.collect or args.analyze):
        print("nothing to do: pass --collect and/or --analyze")


if __name__ == "__main__":
    main()
