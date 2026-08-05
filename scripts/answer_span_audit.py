#!/usr/bin/env python
"""
answer_span_audit.py — which cells generate run-on traces, and which of those a crop fixes.

Produces the evidence table behind `spectral_utils.answer_span.RUNON_CELLS`. Three
columns decide it (criteria (a)/(b)/(c) in that module):

  at_cap     % of generations pinned at max_new_tokens — the model never stopped.
  ans_frac   answer tokens / trace tokens, median. Low = the features are mostly
             measuring text that is not the answer.
  unusable   % of cropped spans that are not an answer at all (empty, a chat-template
             echo like [/INST], a fabricated `Question:` turn, a markdown rule).
             This is the column that separates "croppable" from "must re-generate".

`ans_frac` is only meaningful for SHORT-ANSWER cells, whose grader reads
`first_answer_line`. Cells that reason first and answer last (all math, plus
losnet_hotpotqa_mistral7b) score low here by construction and are excluded from the
verdict — cropping those would keep the preamble and discard the answer. For them
`at_cap` alone is the diagnostic, and it reports truncation, a different defect.

Usage:
    python scripts/answer_span_audit.py
    python scripts/answer_span_audit.py --csv results/answer_span/audit.csv
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_cells import INSCOPE_ALL, QA_CELLS_ALL              # noqa: E402
from spectral_utils.answer_span import (                         # noqa: E402
    RUNON_CELLS, UNREPAIRABLE_CELLS, runon_stats,
)

# Cells whose answer is the LAST thing in the trace, not the first line. Criterion
# (a) fails for these, so ans_frac / unusable carry no verdict.
REASONS_FIRST = {"losnet_hotpotqa_mistral7b"}

AT_CAP_MIN = 50.0      # (b) the model mostly never stopped
ANS_FRAC_MAX = 50.0    # (b) the answer is a minority of the trace
UNUSABLE_MAX = 10.0    # (c) the cropped span is an answer on nearly every row


def cap_of(cell):
    man = os.path.join(REPO, "cache", "repgrid", cell, "manifest.json")
    if os.path.exists(man):
        with open(man) as f:
            m = json.load(f)
        for k in ("max_new", "max_new_tokens"):
            if m.get(k):
                return int(m[k])
    return None


def audit(cell):
    pk = sorted(glob.glob(os.path.join(REPO, "cache", "repgrid", cell, "*.pkl")))
    if not pk:
        return None
    with open(pk[0], "rb") as f:
        data = pickle.load(f)
    cap = cap_of(cell)
    n_tok, ans, at_cap, unusable = [], [], [], []
    for qi in sorted(data.keys()):
        for c in data[qi]["candidates"]:
            s = runon_stats(c, cap=cap)
            n_tok.append(s["n_tokens"])
            ans.append(s["n_answer"] if s["n_answer"] is not None else 0)
            at_cap.append(bool(s["at_cap"]))
            unusable.append(s["unusable"])
    n_tok, ans = np.asarray(n_tok, float), np.asarray(ans, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_tok > 0, ans / n_tok, np.nan)
    return dict(
        cell=cell, group="QA" if cell in QA_CELLS_ALL else "math", n=len(n_tok), cap=cap,
        at_cap=100.0 * float(np.mean(at_cap)),
        med_len=float(np.median(n_tok)), med_ans=float(np.median(ans)),
        ans_frac=100.0 * float(np.nanmedian(frac)),
        unusable=100.0 * float(np.mean(unusable)),
    )


def verdict(r):
    if r["cell"] in REASONS_FIRST or r["group"] == "math":
        return "N/A (answers last)" + (
            f" — TRUNCATED {r['at_cap']:.0f}%" if r["at_cap"] >= AT_CAP_MIN else "")
    if r["at_cap"] < AT_CAP_MIN or r["ans_frac"] >= ANS_FRAC_MAX:
        return "healthy"
    if r["unusable"] >= UNUSABLE_MAX:
        return "UNREPAIRABLE — re-generate"
    return "CROP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "answer_span",
                                                  "audit.csv"))
    args = ap.parse_args()

    # INSCOPE_ALL, not INSCOPE. This audit MEASURES the raw pkls and its verdicts
    # are what `RUNON_CELLS` / `UNREPAIRABLE_CELLS` (and hence the Step-216
    # rejection) are DERIVED FROM. Iterating the post-rejection roster would make
    # the registry check below circular: the rejected cell would stop being
    # measured, the audit would report no unrepairable cells, and the drift check
    # would fire against the very registry the audit exists to justify.
    rows = []
    for cell in INSCOPE_ALL:
        r = audit(cell)
        if r is None:
            print(f"[skip] {cell}: no raw pkl")
            continue
        r["verdict"] = verdict(r)
        rows.append(r)

    hdr = (f"{'cell':<34}{'grp':<5}{'n':>6}{'cap':>6}{'at_cap':>8}{'med_len':>9}"
           f"{'med_ans':>9}{'ans%':>7}{'unusable%':>11}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["group"], -r["at_cap"])):
        print(f"{r['cell']:<34}{r['group']:<5}{r['n']:>6}{str(r['cap']):>6}"
              f"{r['at_cap']:>7.1f}%{r['med_len']:>9.0f}{r['med_ans']:>9.0f}"
              f"{r['ans_frac']:>6.1f}%{r['unusable']:>10.1f}%  {r['verdict']}")

    crop = sorted(r["cell"] for r in rows if r["verdict"] == "CROP")
    bad = sorted(r["cell"] for r in rows if r["verdict"].startswith("UNREPAIRABLE"))
    print(f"\nCROP           -> {crop}")
    print(f"re-generate    -> {bad}")

    # the registry must agree with what was just measured
    drift = []
    if crop != sorted(RUNON_CELLS):
        drift.append(f"RUNON_CELLS={sorted(RUNON_CELLS)} but audit says {crop}")
    if bad != sorted(UNREPAIRABLE_CELLS):
        drift.append(f"UNREPAIRABLE_CELLS={sorted(UNREPAIRABLE_CELLS)} but audit says {bad}")
    if drift:
        print("\nREGISTRY DRIFT:\n  " + "\n  ".join(drift))

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv}")
    if drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
