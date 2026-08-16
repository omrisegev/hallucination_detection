#!/usr/bin/env python
"""
Recover REFRAIN's *base* reflection-trigger vocabulary from the PDF's underline geometry.

Why this script exists
----------------------
REFRAIN's Table 5 prints four trigger categories, but its caption says underlining marks
the Section 5.2 **in-category expansions** and bold marks the **new category** — and the
headline REFRAIN row (91.20 / 1.61M on MATH-500) uses neither. Table 2 reports those two
expansions as separate variants with different numbers, so implementing the printed list
verbatim reproduces the *ablation*, not the method.

Plain-text PDF extraction discards underlining, so `papers/extracted/...md` cannot answer
the question. But a LaTeX `\\underline` is a drawn line segment in the PDF content stream.
This script tests each character's midpoint against the thin horizontal segments beneath
its line, and prints the categories split into base versus expansion.

Its output is the evidence behind `spectral_utils/paper_exact/refrain.py:V_BASE`. Re-run it
whenever that constant is questioned; do not edit the constant from memory.

Usage:
    python scripts/recover_refrain_vocabulary.py [--pdf PATH] [--page 13]
"""
import argparse
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_GLOB = os.path.join(REPO_ROOT, "papers", "Stop When Enough*.pdf")


def underline_segments(page, top_lo, top_hi, max_width=400.0):
    """Thin horizontal segments in the band — candidate underlines.

    `max_width` excludes the table's own full-width rules, which are the same shape.
    """
    segs = []
    for l in page.lines:
        if abs(l["y1"] - l["y0"]) >= 1.5:
            continue
        x0, x1 = min(l["x0"], l["x1"]), max(l["x0"], l["x1"])
        if (x1 - x0) >= max_width or not (top_lo <= l["top"] <= top_hi):
            continue
        segs.append((x0, x1, l["top"]))
    return segs


def annotate(page, top_lo, top_hi, x_min=190.0):
    """Yield (line_top, annotated_text) with underlined runs wrapped in [ ]."""
    chars = [c for c in page.chars if top_lo <= c["top"] <= top_hi and c["x0"] > x_min]
    segs = underline_segments(page, top_lo, top_hi)
    rows = {}
    for c in chars:
        rows.setdefault(round(c["top"] / 4), []).append(c)
    for key in sorted(rows):
        cs = sorted(rows[key], key=lambda c: c["x0"])
        bottom = max(c["bottom"] for c in cs)
        out, prev = [], None
        for c in cs:
            mid = (c["x0"] + c["x1"]) / 2.0
            under = any(bottom - 2.5 <= s[2] <= bottom + 5.0 and s[0] - 0.5 <= mid <= s[1] + 0.5
                        for s in segs)
            if under != prev:
                out.append("[" if under else "]")
                prev = under
            out.append(c["text"])
        if prev:
            out.append("]")
        yield cs[0]["top"], "".join(out).lstrip("]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--page", type=int, default=13, help="0-indexed page holding Table 5")
    ap.add_argument("--top-lo", type=float, default=78.0)
    ap.add_argument("--top-hi", type=float, default=200.0)
    args = ap.parse_args()

    try:
        import pdfplumber
    except ImportError:
        sys.exit("pdfplumber is required: pip install pdfplumber")

    pdf_path = args.pdf or (sorted(glob.glob(DEFAULT_GLOB)) or [None])[0]
    if not pdf_path or not os.path.exists(pdf_path):
        sys.exit(f"REFRAIN PDF not found (looked for {DEFAULT_GLOB})")

    from spectral_utils.paper_exact.manifest import sha256_file
    print(f"pdf    : {pdf_path}")
    print(f"sha256 : {sha256_file(pdf_path)}")
    print(f"page   : {args.page} (0-indexed)\n")
    print("[...] marks an underlined run = a Section 5.2 in-category expansion.")
    print("Everything outside the brackets is the BASE vocabulary V used by the")
    print("headline REFRAIN numbers.\n")

    with pdfplumber.open(pdf_path) as pdf:
        if args.page >= len(pdf.pages):
            sys.exit(f"page {args.page} out of range ({len(pdf.pages)} pages)")
        page = pdf.pages[args.page]
        for top, line in annotate(page, args.top_lo, args.top_hi):
            print(f"  top={top:6.1f}  {line}")

    from spectral_utils.paper_exact.refrain import (
        V_BASE, V_CHECK, V_SHIFT, V_UNCERT, V_RETRO, V_INCAT_EXPANSION, V_NEW_CATEGORY)
    print("\n--- constants currently frozen in spectral_utils/paper_exact/refrain.py ---")
    for name, vals in (("V_CHECK", V_CHECK), ("V_SHIFT", V_SHIFT),
                       ("V_UNCERT", V_UNCERT), ("V_RETRO", V_RETRO)):
        print(f"  {name:9s} ({len(vals)}): {'; '.join(vals)}")
    print(f"  V_BASE            : {len(V_BASE)} phrases  <- used by the headline run")
    print(f"  V_INCAT_EXPANSION : {len(V_INCAT_EXPANSION)} phrases (Table 2 ablation)")
    print(f"  V_NEW_CATEGORY    : {len(V_NEW_CATEGORY)} phrases (Table 2 ablation)")


if __name__ == "__main__":
    main()
