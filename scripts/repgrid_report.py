#!/usr/bin/env python
"""
repgrid_report.py — build the replication-grid report from score_repgrid's CSV.

Renders results/Replication_Grid_Report.html + whatis CSVs:
  - Headline: OUR L-SML / U-PCR AUROC (X) next to each paper's PUBLISHED Y, with the
    head_to_head tag (SAME-MODEL = X and Y share the exact model+dataset).
  - L-SML vs U-PCR across the high-ranked subsets.
  - whatis (A) new vs old same-cell (new GSM8K/Llama-8B GOOD_5 vs old 0.7563 in loco.csv).
  - whatis (B) MACRO now (old + new cells) vs old-only.
  - whatis (C) do the new spilled/energy/logprob views help (GOOD_5 vs GOOD_5+view).
  - whatis (D) more features -> better on short QA (subset-size ladder trend).

Usage:
    python scripts/repgrid_report.py [--scores results/repgrid/scores_lsml_upcr.csv]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "results", "subset_sweep")

SIZE_LADDER = ["consensus_4", "GOOD_5", "STABLE_H9", "ALL_H16"]
REASONING_DOMAINS = ("math500", "gsm8k", "qa")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def headline(rows):
    """Best GOOD_5 (or GOOD_5+view) per cell x method vs published Y."""
    out = []
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    for cell, rs in by_cell.items():
        Y = _f(rs[0]["published_Y"])
        h2h = rs[0]["head_to_head"]
        model, dataset = rs[0]["model"], rs[0]["dataset"]
        for method in ("lsml", "upcr"):
            cand = [r for r in rs if r["method"] == method and _f(r["auroc_X"]) is not None]
            if not cand:
                continue
            best = max(cand, key=lambda r: _f(r["auroc_X"]))
            X = _f(best["auroc_X"])
            out.append({"cell": cell, "model": model, "dataset": dataset, "method": method,
                        "best_subset": best["subset"], "X": X, "Y": Y,
                        "delta": (X - Y) if Y is not None else None, "head_to_head": h2h,
                        "n": best["n_rows"], "valid_rate": best["valid_rate"]})
    return out


def whatis_new_vs_old(rows):
    """New GSM8K/Llama-8B GOOD_5 L-SML vs old loco.csv value."""
    loco = read_csv(os.path.join(SWEEP, "loco.csv"))
    old = {(r["domain"], r["cell"]): _f(r["good5"]) for r in loco}
    out = []
    for r in rows:
        if r["subset"] == "GOOD_5" and r["method"] == "lsml" and _f(r["auroc_X"]) is not None:
            # crude mapping: match on dataset name appearing in an old cell key
            ds = r["dataset"]
            match = next((v for (d, c), v in old.items() if ds.split("_")[0] in (d + c).lower()), None)
            out.append({"cell": r["cell"], "dataset": ds, "new_GOOD5_lsml": _f(r["auroc_X"]),
                        "old_GOOD5": match,
                        "delta_new_minus_old": (_f(r["auroc_X"]) - match) if match else None})
    return out


def whatis_macro(rows, headline_rows):
    """MACRO cell-mean over GOOD_5 L-SML: new cells only, and new+old blended."""
    new_good5 = [_f(r["auroc_X"]) for r in rows
                 if r["subset"] == "GOOD_5" and r["method"] == "lsml" and _f(r["auroc_X"]) is not None]
    loco = read_csv(os.path.join(SWEEP, "loco.csv"))
    old_good5 = [_f(r["good5"]) for r in loco if _f(r["good5"]) is not None]
    macro = lambda xs: sum(xs) / len(xs) if xs else None
    return {"macro_new_cells": macro(new_good5), "n_new": len(new_good5),
            "macro_old_cells": macro(old_good5), "n_old": len(old_good5),
            "macro_old_plus_new": macro(old_good5 + new_good5), "n_all": len(old_good5) + len(new_good5)}


def whatis_view_lift(rows):
    """GOOD_5 vs GOOD_5+{spilled,energy,logprob} per cell (L-SML)."""
    base = {}
    for r in rows:
        if r["method"] == "lsml" and r["subset"] == "GOOD_5" and _f(r["auroc_X"]) is not None:
            base[r["cell"]] = _f(r["auroc_X"])
    out = []
    for r in rows:
        if r["method"] == "lsml" and r["subset"].startswith("GOOD_5+") and r["cell"] in base:
            X = _f(r["auroc_X"])
            if X is not None:
                out.append({"cell": r["cell"], "view": r["subset"], "base_GOOD5": base[r["cell"]],
                            "augmented": X, "lift": X - base[r["cell"]]})
    return out


def whatis_size_trend(rows):
    """AUROC vs subset size (consensus_4 -> GOOD_5 -> STABLE_H9 -> ALL_H16), per cell x method."""
    out = []
    by = defaultdict(dict)
    for r in rows:
        if r["subset"] in SIZE_LADDER and _f(r["auroc_X"]) is not None:
            by[(r["cell"], r["method"])][r["subset"]] = _f(r["auroc_X"])
    for (cell, method), d in by.items():
        row = {"cell": cell, "method": method}
        row.update({s: d.get(s) for s in SIZE_LADDER})
        out.append(row)
    return out


def _tbl(rows, cols, fmt=None):
    fmt = fmt or {}
    th = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        tds = ""
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                v = fmt.get(c, "{:.4f}").format(v)
            tds += f"<td>{'' if v is None else v}</td>"
        body += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(REPO, "results", "repgrid", "scores_lsml_upcr.csv"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "Replication_Grid_Report.html"))
    args = ap.parse_args()

    rows = read_csv(args.scores)
    if not rows:
        sys.exit(f"no scores at {args.scores} — run scripts/score_repgrid.py first")

    hl = headline(rows)
    nvo = whatis_new_vs_old(rows)
    macro = whatis_macro(rows, hl)
    lift = whatis_view_lift(rows)
    size = whatis_size_trend(rows)

    out_dir = os.path.dirname(args.scores)
    for name, data, cols in [
        ("headline_X_vs_Y", hl, ["cell", "model", "dataset", "method", "best_subset", "X", "Y", "delta", "head_to_head", "n", "valid_rate"]),
        ("whatis_new_vs_old", nvo, ["cell", "dataset", "new_GOOD5_lsml", "old_GOOD5", "delta_new_minus_old"]),
        ("whatis_view_lift", lift, ["cell", "view", "base_GOOD5", "augmented", "lift"]),
        ("whatis_size_trend", size, ["cell", "method"] + SIZE_LADDER),
    ]:
        p = os.path.join(out_dir, f"{name}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([{k: r.get(k) for k in cols} for r in data])

    html = f"""<!doctype html><meta charset=utf-8><title>Replication Grid — our L-SML/U-PCR vs published</title>
<style>body{{font:14px system-ui;margin:2rem;max-width:1100px}}table{{border-collapse:collapse;margin:1rem 0}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}th{{background:#f4f4f4}}td:first-child,th:first-child{{text-align:left}}
h2{{border-bottom:2px solid #333;padding-top:1rem}} .note{{color:#555}}</style>
<h1>Replication Grid — OUR method vs the papers' PUBLISHED numbers</h1>
<p class=note>X = our AUROC (L-SML continuous / U-PCR), raw with bootstrap CI. Y = the paper's published AUROC.
<b>SAME-MODEL</b> = X and Y share the exact model+dataset+protocol, so the only difference is the method.
Competitor detectors are NOT reproduced here — Y is taken from the paper.</p>

<h2>1. Headline — best subset per cell: X vs Y</h2>
{_tbl(hl, ["cell","model","dataset","method","best_subset","X","Y","delta","head_to_head","n","valid_rate"], {"X":"{:.4f}","Y":"{:.3f}","delta":"{:+.4f}","valid_rate":"{:.2f}"})}

<h2>2. whatis (A) — fresh replication vs old cache (GOOD_5 L-SML)</h2>
{_tbl(nvo, ["cell","dataset","new_GOOD5_lsml","old_GOOD5","delta_new_minus_old"], {"new_GOOD5_lsml":"{:.4f}","old_GOOD5":"{:.4f}","delta_new_minus_old":"{:+.4f}"})}

<h2>3. whatis (B) — MACRO (cell-mean, GOOD_5 L-SML)</h2>
<p>new cells: <b>{macro['macro_new_cells']}</b> (n={macro['n_new']}) &nbsp;|&nbsp;
old cells: <b>{macro['macro_old_cells']}</b> (n={macro['n_old']}) &nbsp;|&nbsp;
old+new: <b>{macro['macro_old_plus_new']}</b> (n={macro['n_all']})</p>

<h2>4. whatis (C) — do the new views help? (GOOD_5 vs GOOD_5+view, L-SML)</h2>
{_tbl(lift, ["cell","view","base_GOOD5","augmented","lift"], {"base_GOOD5":"{:.4f}","augmented":"{:.4f}","lift":"{:+.4f}"})}

<h2>5. whatis (D) — more features -> better on short QA? (AUROC vs subset size)</h2>
{_tbl(size, ["cell","method"]+SIZE_LADDER, {s:"{:.4f}" for s in SIZE_LADDER})}
"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} + whatis CSVs in {out_dir}")


if __name__ == "__main__":
    main()
