#!/usr/bin/env python
"""
cell_method_matrix.py — the per-cell x method AUROC matrix (Omri's point 5).

Every comparison artifact so far (comparison.csv, comparison_inscope.csv) is
macro-only; the per-cell numbers exist but are scattered across the long-format
per-family bench CSVs (results/selector_bench/*__{h16,c46}.csv). This script
pivots them into ONE cell(25) x method matrix and renders it as a CSV plus an
HTML heatmap, so "how do we do on each cell, with the latest algorithm steps"
is a single artifact.

House rules (advisor chain): every number read from CSV at build time, nothing
hand-typed; guardrail_scan on the HTML is a build failure (exit 1).

Outputs:
  results/advisor_inscope/cell_method_matrix.csv
  results/advisor_inscope/cell_method_matrix.html

Usage:
    python scripts/cell_method_matrix.py            # curated headline columns
    python scripts/cell_method_matrix.py --all      # every variant in the bench
    python scripts/cell_method_matrix.py --variants ref.GOOD_6,a6.pl_dufs
"""
import argparse
import datetime as _dt
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from advisor_report import esc, guardrail_scan          # noqa: E402
from inscope_cells import INSCOPE, QA_CELLS, MATH_CELLS  # noqa: E402

BENCH = os.path.join(REPO, "results", "selector_bench")
OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")

# Headline columns: the fixed reference subsets plus every learned selector
# that made the comparison_inscope leaderboard's top block (Steps 186-195).
DEFAULT_VARIANTS = [
    "ref.LOCO_5", "ref.GOOD_6", "ref.GOOD_5",
    "a6.pruned_dufs", "a6.pl_dufs", "a6.dufs", "a6.pl_rank",
    "a6.fp_dufs", "a6.fp_rank",                    # WS4 Arm A (two-stage full-pool)
    "a6.pl_dufs@loco5", "a6.pl_dufs@central4",     # WS4 Arm B (seed sweep)
    "a2.dufs", "a2.select",
    "a1.router@good5", "a1.router@loco5",          # WS6
    "ref.top_macro_5", "ref.STABLE_H9", "ref.consensus_4", "ref.ALL_H16",
]


def load_bench_rows():
    files = sorted(glob.glob(os.path.join(BENCH, "*__c46.csv")) +
                   glob.glob(os.path.join(BENCH, "*__h16.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["cell"].isin(INSCOPE)].copy()
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
    # prefer the c46-pool row when a (variant, cell) exists in both pools
    df["_pool_rank"] = (df["pool_mode"] == "c46").astype(int)
    df = (df.sort_values("_pool_rank")
            .drop_duplicates(["variant", "cell"], keep="last")
            .drop(columns="_pool_rank"))
    return df


def build_matrix(df, variants):
    cells = QA_CELLS + MATH_CELLS          # QA block first, then math
    mat = df[df["variant"].isin(variants)].pivot_table(
        index="cell", columns="variant", values="auroc", aggfunc="first")
    mat = mat.reindex(index=[c for c in cells if c in mat.index])
    # order columns by macro over the cells they cover, best first
    order = mat.mean(axis=0, skipna=True).sort_values(ascending=False)
    mat = mat[order.index]
    return mat


# ---------------------------------------------------------------- heatmap ---
# Diverging around AUROC 0.5 (chance): blue arm above, red arm below, neutral
# gray midpoint — dataviz diverging rule. Values are printed in every cell, so
# color is redundant with text (the contrast-relief channel).
_NEUTRAL = (240, 239, 236)      # #f0efec
_BLUE = (26, 82, 158)           # deep blue pole (AUROC 1.0)
_RED = (176, 42, 42)            # red pole (AUROC 0.0)


def _lerp(c0, c1, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))


def cell_color(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return None, None
    if a >= 0.5:
        t = min((a - 0.5) / 0.5, 1.0)
        rgb = _lerp(_NEUTRAL, _BLUE, t)
    else:
        t = min((0.5 - a) / 0.5, 1.0)
        rgb = _lerp(_NEUTRAL, _RED, t)
    ink = "#ffffff" if t > 0.55 else "#1a1a19"
    return "#%02x%02x%02x" % rgb, ink


PAGE_CSS = """
body{font-family:Segoe UI,system-ui,sans-serif;margin:24px;color:#1a1a19;background:#fcfcfb}
h1{font-size:20px}
table{border-collapse:separate;border-spacing:2px;font-size:12px}
th{font-weight:600;text-align:left;padding:3px 6px;color:#444}
th.colh{writing-mode:vertical-rl;transform:rotate(180deg);vertical-align:bottom;
        max-height:170px;font-size:11px}
td{padding:3px 6px;text-align:right;border-radius:4px;min-width:44px}
td.rowh{text-align:left;font-weight:500;border-radius:0;white-space:nowrap}
td.blank{color:#999;text-align:center;background:repeating-linear-gradient(
        45deg,#f4f3f0,#f4f3f0 4px,#eceae6 4px,#eceae6 8px)}
tr.macro td{font-weight:700;border-top:2px solid #ccc}
tr.block th{padding-top:10px;color:#666;font-size:11px;text-transform:uppercase}
.note{color:#555;font-size:12px;max-width:900px;line-height:1.5}
.legend{display:flex;gap:14px;align-items:center;font-size:12px;margin:10px 0}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px}
.wrap{overflow-x:auto}
"""


def render_html(mat, out_html):
    variants = list(mat.columns)
    macro = mat.mean(axis=0, skipna=True)
    cover = mat.notna().sum(axis=0)

    def row_html(cell):
        tds = [f'<td class="rowh">{esc(cell)}</td>']
        for v in variants:
            a = mat.loc[cell, v] if v in mat.columns else np.nan
            bg, ink = cell_color(a)
            if bg is None:
                tds.append('<td class="blank" title="not scorable on this '
                           'cell (missing views)">&mdash;</td>')
            else:
                tds.append(f'<td style="background:{bg};color:{ink}" '
                           f'title="{esc(cell)} / {esc(v)}">{a:.3f}</td>')
        return "<tr>" + "".join(tds) + "</tr>"

    head = ('<tr><th></th>' +
            "".join(f'<th class="colh" title="{esc(v)}">{esc(v)}</th>'
                    for v in variants) + "</tr>")
    qa_rows = "".join(row_html(c) for c in mat.index if c in QA_CELLS)
    math_rows = "".join(row_html(c) for c in mat.index if c in MATH_CELLS)
    macro_tds = [f'<td class="rowh">macro (mean over covered cells)</td>']
    for v in variants:
        bg, ink = cell_color(macro[v])
        macro_tds.append(f'<td style="background:{bg};color:{ink}" '
                         f'title="{esc(v)}: covers {cover[v]}/{len(mat)} '
                         f'cells">{macro[v]:.4f}</td>')

    n_qa = sum(1 for c in mat.index if c in QA_CELLS)
    n_math = sum(1 for c in mat.index if c in MATH_CELLS)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Per-cell x method AUROC matrix</title><style>{PAGE_CSS}</style></head><body>
<h1>Per-cell &times; method AUROC — {len(mat)} in-scope cells, {len(variants)} methods</h1>
<p class="note">Every value is read from the selector-bench CSVs at build time
(c46 pool preferred, h16 fallback). Columns are ordered by macro AUROC over the
cells each method covers; a hatched cell means the method is not scorable there
(a fixed subset whose member views are missing from that cell's cache).
Generated {GEN_DATE}.</p>
<div class="legend">
 <span><span class="sw" style="background:#b02a2a"></span> 0.0 (anti-oriented)</span>
 <span><span class="sw" style="background:#f0efec;border:1px solid #ddd"></span> 0.5 (chance)</span>
 <span><span class="sw" style="background:#1a529e"></span> 1.0</span>
</div>
<div class="wrap"><table>
{head}
<tr class="block"><th colspan="{len(variants) + 1}">QA ({n_qa} cells)</th></tr>
{qa_rows}
<tr class="block"><th colspan="{len(variants) + 1}">Math ({n_math} cells)</th></tr>
{math_rows}
<tr class="macro">{''.join(macro_tds)}</tr>
</table></div>
<p class="note">Column names are shorthand — see <code>GLOSSARY.md</code>
(repo root) for what each one means.</p>
</body></html>"""

    hits = guardrail_scan(html)
    if hits:
        print(f"GUARDRAIL FAIL: {hits}")
        sys.exit(1)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


GEN_DATE = _dt.date.today().isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default=None,
                    help="comma list; default = curated headline set")
    ap.add_argument("--all", action="store_true",
                    help="every variant present in the bench CSVs")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    df = load_bench_rows()
    if args.all:
        variants = sorted(df["variant"].unique())
    elif args.variants:
        variants = args.variants.split(",")
    else:
        present = set(df["variant"])
        variants = [v for v in DEFAULT_VARIANTS if v in present]
        missing = [v for v in DEFAULT_VARIANTS if v not in present]
        if missing:
            print(f"note: not in bench yet, skipped: {missing}")

    mat = build_matrix(df, variants)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "cell_method_matrix.csv")
    out_html = os.path.join(args.out_dir, "cell_method_matrix.html")
    mat.round(4).to_csv(out_csv)
    render_html(mat, out_html)
    print(f"matrix {mat.shape[0]} cells x {mat.shape[1]} methods")
    print(f"wrote {out_csv}\nwrote {out_html}")


if __name__ == "__main__":
    main()
