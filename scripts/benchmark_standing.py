#!/usr/bin/env python
"""
benchmark_standing.py — where do we stand, on our own grid and against the papers?

Two questions, one script:

  (1) INTERNAL — among the variants that need no more than a *global* orientation bit
      (no hand-picked subset, no 42 hand signs, no labels at runtime), which is ahead?
      Shortlist: DUFS+L-SML (`a2.dufs`), the leading U-PCR (`upcr.rho_polarities`),
      with the prior-carrying references (GOOD_5/6, LOCO_5, `a6.pl_dufs`) shown as the
      bar to clear rather than as candidates.

  (2) EXTERNAL — against every published number we have verified on our exact cells,
      binned by COST CLASS, because that is the only comparison that means anything:
        BAR A  unsupervised, 1 forward pass, grey-box (logprobs only)  <- our exact class
        BAR B  unsupervised, 1 forward pass, any access               <- + white-box internals
        BAR C  any unsupervised method                                <- + 10-pass / K-sample
        BAR D  best method in the paper                               <- + supervised probes

Inputs (all already on disk, nothing is recomputed here):
  results/advisor_inscope/cell_method_matrix.csv     per-cell AUROC for the bench variants
  results/upcr_study/06_orientation/per_cell.csv     per-cell AUROC for the U-PCR arms
  results/advisor_inscope/competitors_verified.csv   verified published numbers + cost class

Outputs:
  results/benchmark_standing.csv   per-cell merged table (ours + every bar)
  results/BENCHMARK_STANDING.md    the memo

Usage:
    python scripts/benchmark_standing.py
"""
import csv
import os
import statistics as st
import sys

from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(REPO, "results", "advisor_inscope", "cell_method_matrix.csv")
UPCR = os.path.join(REPO, "results", "upcr_study", "06_orientation", "per_cell.csv")
COMPET = os.path.join(REPO, "results", "advisor_inscope", "competitors_verified.csv")
OUT_CSV = os.path.join(REPO, "results", "benchmark_standing.csv")
OUT_MD = os.path.join(REPO, "results", "BENCHMARK_STANDING.md")

# variant -> (label, what prior it carries). Order = display order.
SHORTLIST = [
    ("a2.dufs",     "DUFS + L-SML",        "global anchor bit only"),
    ("upcr.rho",    "U-PCR + sign(rho)",   "global anchor bit only"),
    ("a6.pl_dufs",  "a6.pl_dufs",          "GOOD_6 seed (prior-carrying)"),
    ("ref.GOOD_6",  "GOOD_6",              "hand-picked 6-view subset"),
    ("ref.GOOD_5",  "GOOD_5",              "hand-picked 5-view subset"),
    ("ref.LOCO_5",  "LOCO_5",              "label-chosen 5-view subset"),
]

# QA vs math split — matches the domain convention used by the bench.
QA_MARKERS = ("triviaqa", "coqa", "nq_open", "squad", "sciq", "truthfulqa", "hotpotqa")

BARS = [
    ("A", "our exact class: unsupervised, 1 forward pass, grey-box (logprobs only)",
     lambda r: r["supervision"] == "unsupervised" and r["passes"] == "1" and r["access"] == "grey-box"),
    ("B", "unsupervised + 1 forward pass, any access (adds white-box internals)",
     lambda r: r["supervision"] == "unsupervised" and r["passes"] == "1"),
    ("C", "any unsupervised method, including 10-pass / K-sample",
     lambda r: r["supervision"] == "unsupervised"),
    ("D", "best method in the paper, including supervised probes",
     lambda r: True),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_ours():
    """Merge the bench matrix and the U-PCR orientation table into one per-cell dict.

    The two files are scored by different drivers but on the same 25 in-scope cells;
    the cell keys are asserted identical rather than assumed, because a silent
    partial join here would quietly change every macro below."""
    mat = {r["cell"]: r for r in csv.DictReader(open(MATRIX, newline=""))}
    upcr = {r["cell"]: r for r in csv.DictReader(open(UPCR, newline=""))}
    if set(mat) != set(upcr):
        raise SystemExit(f"cell mismatch: matrix-only={sorted(set(mat) - set(upcr))} "
                         f"upcr-only={sorted(set(upcr) - set(mat))}")
    out = {}
    for cell, row in mat.items():
        vals = {k: _f(row.get(k)) for k, _, _ in SHORTLIST if k != "upcr.rho"}
        vals["upcr.rho"] = _f(upcr[cell]["auroc_rho_anchor"])
        out[cell] = vals
    return out


def read_bars():
    """For each bar, the strongest published number on each of our cells."""
    comp = list(csv.DictReader(open(COMPET, newline="")))
    bars = {}
    for tag, _, pred in BARS:
        best = {}
        for r in comp:
            auroc = _f(r["auroc"])
            if auroc is None or not pred(r):
                continue
            cell = r["cell"]
            if cell not in best or auroc > best[cell][1]:
                best[cell] = (r["method"], auroc, r["paper_slug"])
        bars[tag] = best
    return bars


def paired(ours, cells, a, b):
    """b - a over the cells where both are scored."""
    d = [ours[c][b] - ours[c][a] for c in cells
         if ours[c].get(a) is not None and ours[c].get(b) is not None]
    if not d:
        return None
    return dict(n=len(d), mean=st.mean(d), median=st.median(d),
                wins=sum(x > 0 for x in d), losses=sum(x < 0 for x in d),
                p=wilcoxon(d).pvalue if any(d) else 1.0)


def fmt(s):
    if s is None:
        return "n/a"
    return (f"{s['mean'] * 100:+.2f}pp mean / {s['median'] * 100:+.2f}pp median, "
            f"{s['wins']}W/{s['losses']}L of {s['n']}, p={s['p']:.3f}")


def main():
    ours = read_ours()
    bars = read_bars()
    cells = sorted(ours)
    keys = [k for k, _, _ in SHORTLIST]
    label = {k: lab for k, lab, _ in SHORTLIST}

    # ---- per-cell CSV -------------------------------------------------------
    with open(OUT_CSV, "w", newline="") as f:
        cols = ["cell", "domain"] + keys
        for tag, _, _ in BARS:
            cols += [f"bar{tag}_auroc", f"bar{tag}_method"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cells:
            row = {"cell": c,
                   "domain": "QA" if any(m in c for m in QA_MARKERS) else "math"}
            row.update({k: ours[c].get(k) for k in keys})
            for tag, _, _ in BARS:
                hit = bars[tag].get(c)
                row[f"bar{tag}_auroc"] = hit[1] if hit else None
                row[f"bar{tag}_method"] = hit[0] if hit else None
            w.writerow(row)

    # ---- memo ---------------------------------------------------------------
    L = []
    L.append("# Where we stand — benchmark standing\n")
    L.append(f"Generated by `scripts/benchmark_standing.py` over {len(cells)} in-scope cells. "
             "Every number is a fixed method applied to every cell — no per-cell subset "
             "picking, no oracle. Regenerate after any re-bench.\n")

    L.append("\n## 1. Internal — the shortlist\n")
    L.append("| Method | Prior it needs | macro | QA | math |")
    L.append("|---|---|---:|---:|---:|")
    for k, lab, prior in SHORTLIST:
        vals = [ours[c][k] for c in cells if ours[c].get(k) is not None]
        qa = [ours[c][k] for c in cells
              if ours[c].get(k) is not None and any(m in c for m in QA_MARKERS)]
        ma = [ours[c][k] for c in cells
              if ours[c].get(k) is not None and not any(m in c for m in QA_MARKERS)]
        n = f" ({len(vals)} cells)" if len(vals) != len(cells) else ""
        L.append(f"| {lab} | {prior} | **{st.mean(vals):.4f}**{n} | {st.mean(qa):.4f} | {st.mean(ma):.4f} |")

    L.append("\n### Paired head-to-head\n")
    L.append("| Contrast | Effect |")
    L.append("|---|---|")
    for a, b in [("a2.dufs", "upcr.rho"), ("a2.dufs", "ref.GOOD_6"),
                 ("upcr.rho", "ref.GOOD_6"), ("upcr.rho", "ref.LOCO_5"),
                 ("a2.dufs", "a6.pl_dufs")]:
        L.append(f"| {label[b]} − {label[a]} | {fmt(paired(ours, cells, a, b))} |")

    L.append("\n## 2. External — against the published roster, by cost class\n")
    for tag, desc, _ in BARS:
        B = bars[tag]
        cov = [c for c in cells if c in B]
        L.append(f"\n### Bar {tag} — {desc}\n")
        L.append(f"Covers {len(cov)} of {len(cells)} cells (the rest have no published "
                 f"number in this class).\n")
        L.append("| Method | vs this bar |")
        L.append("|---|---|")
        for k, lab, _ in SHORTLIST:
            cs = [c for c in cov if ours[c].get(k) is not None]
            d = [ours[c][k] - B[c][1] for c in cs]
            if not d:
                continue
            L.append(f"| {lab} | {st.mean(d) * 100:+.2f}pp mean / {st.median(d) * 100:+.2f}pp "
                     f"median, {sum(x > 0 for x in d)}W/{sum(x < 0 for x in d)}L of {len(cs)}, "
                     f"p={wilcoxon(d).pvalue:.3f} |")

    L.append("\n## 3. Per-cell, against the strongest same-cost-class published number (Bar B)\n")
    B = bars["B"]
    L.append("| Cell | DUFS+L-SML | U-PCR+sign(rho) | GOOD_6 | Published (Bar B) | Best of ours − them |")
    L.append("|---|---:|---:|---:|---|---:|")
    for c in cells:
        if c not in B:
            continue
        best = max(v for v in (ours[c]["a2.dufs"], ours[c]["upcr.rho"]) if v is not None)
        L.append(f"| {c} | {ours[c]['a2.dufs']:.4f} | {ours[c]['upcr.rho']:.4f} | "
                 f"{ours[c]['ref.GOOD_6']:.4f} | {B[c][1]:.4f} ({B[c][0]}) | "
                 f"{(best - B[c][1]) * 100:+.2f}pp |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
