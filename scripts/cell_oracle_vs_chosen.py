#!/usr/bin/env python
"""
cell_oracle_vs_chosen.py — per-cell: what's the best possible subset, and what
did OUR ALGORITHM actually pick? (Omri, 2026-07-22/23)

Two things this answers that cell_method_matrix.py does not:
  1. The per-cell CEILING — the best subset the c46 30-view enumeration found
     for that cell (sizes 3-5, results/subset_sweep_c46/sweep_summary.csv
     `best_auroc`/`best_feats`). This is a LABEL-PEEKING in-sample number
     (chosen using labels on that same cell) — a ceiling/upper-bound, NOT an
     honest achievable number. Report it as such; it is not directly
     comparable to any label-free row.
  2. What `a6.pl_dufs` — the label-free selector ADOPTED AS THE SELECTOR OF
     RECORD (HISTORY Step 194, supersedes a2.dufs) — actually chose on that
     cell: its AUROC, its feature list, and its Jaccard overlap with the
     oracle-best subset's feature list.

Reference macros (GOOD_5/GOOD_6/LOCO_5/...) are NOT our algorithm's output —
they are hand-curated fixed subsets. a6.pl_dufs IS our algorithm's output.
Both are shown, tagged, so the two are never conflated.

Output: results/advisor_inscope/cell_oracle_vs_chosen.csv (+ .html)
Usage:  python scripts/cell_oracle_vs_chosen.py
"""
import datetime as _dt
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from advisor_report import esc, guardrail_scan  # noqa: E402
from inscope_cells import INSCOPE, QA_CELLS, MATH_CELLS  # noqa: E402

SWEEP = os.path.join(REPO, "results", "subset_sweep_c46", "sweep_summary.csv")
BENCH = os.path.join(REPO, "results", "selector_bench",
                     "a6_pseudolabel_gates__c46.csv")
REFMACRO = os.path.join(REPO, "results", "selector_bench",
                        "reference_macros__c46.csv")
OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")
OUT_CSV = os.path.join(OUT_DIR, "cell_oracle_vs_chosen.csv")
OUT_HTML = os.path.join(OUT_DIR, "cell_oracle_vs_chosen.html")

CHOSEN_VARIANT = "a6.pl_dufs"
CHOSEN_LABEL = "a6.pl_dufs (selector of record, Step 194)"


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def build():
    sweep = pd.read_csv(SWEEP)
    sweep = sweep[sweep["cell_key"].isin(INSCOPE)].copy()

    bench = pd.read_csv(BENCH)
    bench = bench[(bench["variant"] == CHOSEN_VARIANT) &
                  (bench["cell"].isin(INSCOPE))].copy()
    bench["auroc"] = pd.to_numeric(bench["auroc"], errors="coerce")

    ref = pd.read_csv(REFMACRO)
    ref = ref[ref["cell"].isin(INSCOPE)].copy()
    ref["auroc"] = pd.to_numeric(ref["auroc"], errors="coerce")
    good5 = ref[ref["variant"] == "ref.GOOD_5"].set_index("cell")["auroc"]
    good6 = ref[ref["variant"] == "ref.GOOD_6"].set_index("cell")["auroc"]

    rows = []
    for cell in INSCOPE:
        s = sweep[sweep["cell_key"] == cell]
        b = bench[bench["cell"] == cell]
        oracle_auroc = float(s["best_auroc"].iloc[0]) if len(s) else float("nan")
        oracle_feats = (s["best_feats"].iloc[0].split("|")
                        if len(s) and pd.notna(s["best_feats"].iloc[0]) else [])
        oracle_size = len(oracle_feats)
        our_auroc = float(b["auroc"].iloc[0]) if len(b) else float("nan")
        our_feats = (b["chosen"].iloc[0].split("|")
                    if len(b) and pd.notna(b["chosen"].iloc[0]) else [])
        rows.append(dict(
            cell=cell, domain="QA" if cell in QA_CELLS else "math",
            oracle_auroc=oracle_auroc, oracle_size=oracle_size,
            oracle_feats="|".join(oracle_feats),
            our_auroc=our_auroc, our_size=len(our_feats),
            our_feats="|".join(our_feats),
            gap_pp=(oracle_auroc - our_auroc) * 100
                   if pd.notna(oracle_auroc) and pd.notna(our_auroc) else float("nan"),
            good5_auroc=float(good5.get(cell, float("nan"))),
            good6_auroc=float(good6.get(cell, float("nan"))),
            feat_overlap_jaccard=round(jaccard(oracle_feats, our_feats), 3),
        ))
    return pd.DataFrame(rows)


PAGE_CSS = """
body{font-family:Segoe UI,system-ui,sans-serif;margin:24px;color:#1a1a19;background:#fcfcfb}
h1{font-size:19px} p.note{color:#555;max-width:900px;line-height:1.5;font-size:13px}
table{border-collapse:collapse;font-size:12px;width:100%}
th{background:#f0efec;text-align:left;padding:6px 8px;position:sticky;top:0}
td{padding:6px 8px;border-bottom:1px solid #eee;vertical-align:top}
td.num{text-align:right;white-space:nowrap}
td.feats{max-width:340px;font-size:11px;color:#444;word-break:break-word}
tr.block td{background:#f7f6f3;font-weight:600;color:#666;text-transform:uppercase;font-size:11px}
.tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600}
.tag-ceiling{background:#f0e6c8;color:#7a5b00}
.tag-ours{background:#dbe8f7;color:#0d4a8f}
.wrap{overflow-x:auto}
</style>
"""


def render_html(df):
    def row(r):
        gap_color = "#0ca30c" if r["gap_pp"] <= 1.0 else (
            "#fab219" if r["gap_pp"] <= 5.0 else "#d03b3b")
        return f"""<tr>
<td>{esc(r['cell'])}</td>
<td class="num">{r['oracle_auroc']:.3f}</td>
<td class="num">{r['oracle_size']}</td>
<td class="feats">{esc(r['oracle_feats'])}</td>
<td class="num">{r['our_auroc']:.3f}</td>
<td class="num">{r['our_size']}</td>
<td class="feats">{esc(r['our_feats'])}</td>
<td class="num" style="color:{gap_color};font-weight:600">{r['gap_pp']:+.2f}</td>
<td class="num">{r['feat_overlap_jaccard']:.2f}</td>
<td class="num">{r['good5_auroc']:.3f}</td>
<td class="num">{r['good6_auroc']:.3f}</td>
</tr>"""

    body = []
    for dom in ("QA", "math"):
        sub = df[df["domain"] == dom]
        body.append(f'<tr class="block"><td colspan="10">{dom} '
                    f'({len(sub)} cells)</td></tr>')
        body.extend(row(r) for _, r in sub.iterrows())

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Per-cell: oracle ceiling vs our algorithm's chosen subset</title>
<style>{PAGE_CSS}</style></head><body>
<h1>Per-cell &mdash; oracle ceiling vs
<span class="tag tag-ours">OUR ALGORITHM</span> ({esc(CHOSEN_LABEL)})</h1>
<p class="note">
<span class="tag tag-ceiling">CEILING</span> = the c46 30-view sweep's best
subset for THAT cell (sizes 3-5, results/subset_sweep_c46/). This is
LABEL-PEEKING and IN-SAMPLE (chosen using that cell's own labels) &mdash;
it is an upper bound on what any label-free method could achieve on this
cell, not an honest, achievable number. It is shown to answer "how much is
left on the table", not as a competitor row.<br><br>
<span class="tag tag-ours">OUR ALGORITHM</span> = a6.pl_dufs, the label-free
selector adopted as the SELECTOR OF RECORD (HISTORY Step 194) &mdash; this is
the actual output of our FS pipeline, run and scored with zero label access.
GOOD_5/GOOD_6 (right two columns) are shown for reference only: they are
hand-curated FIXED subsets, not our algorithm's output.<br><br>
Gap (pp) = oracle ceiling &minus; our algorithm, colored green (&le;1pp),
amber (&le;5pp), red (&gt;5pp). Feature overlap = Jaccard between the
oracle's chosen features and ours &mdash; low overlap with a small gap means
different subsets reach similar performance; high gap with low overlap means
our algorithm is missing the cell's actual signal.
Generated {_dt.date.today().isoformat()}.</p>
<div class="wrap"><table>
<tr><th>cell</th><th>ceiling AUROC</th><th>size</th>
<th>ceiling features</th><th>ours AUROC</th><th>size</th>
<th>our features</th><th>gap (pp)</th><th>feat overlap</th>
<th>GOOD_5</th><th>GOOD_6</th></tr>
{"".join(body)}
</table></div>
<p class="note">See <code>GLOSSARY.md</code> (repo root) for what every
nickname in this project's reports means.</p>
</body></html>"""
    hits = guardrail_scan(html)
    if hits:
        print(f"GUARDRAIL FAIL: {hits}")
        sys.exit(1)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    df = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    df.round(4).to_csv(OUT_CSV, index=False)
    render_html(df)
    print(f"{len(df)} cells | mean ceiling {df['oracle_auroc'].mean():.4f} | "
          f"mean ours {df['our_auroc'].mean():.4f} | "
          f"mean gap {df['gap_pp'].mean():+.2f}pp | "
          f"mean feat-overlap {df['feat_overlap_jaccard'].mean():.3f}")
    print(f"wrote {OUT_CSV}\nwrote {OUT_HTML}")


if __name__ == "__main__":
    main()
