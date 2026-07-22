#!/usr/bin/env python
"""
reconcile_competitors.py — one-time diff of the two independently-verified competitor
number sets, so no published number is ever re-litigated a third time (Step 194).

Source A (authoritative): results/advisor_inscope/competitors_verified.csv
    Rebuilt from scratch at Step 193/193b/193d directly against the papers' own
    tables (papers/extracted/), with paper_slug/paper_table/paper_evidence per row.
    61/62 rows verified, 17/18 anchors. Carries the two known corrections:
      * 4 of 5 stored "LapEigvals" anchors are actually the paper's AttentionScore
        baseline (values right, method name + supervision tag wrong);
      * lapeigvals_gsm8k_llama8b's 0.925 was Mistral-Small-24B's LapEigvals — a
        different model's column; its own model reads 0.720 (AttentionScore) / 0.872
        (LapEigvals probe).

Source B (older, previously verified in earlier steps):
    B1  results/repgrid/published_baselines.csv           (cell-keyed, 0-100 scale)
    B2  results/reasoning_benchmark.csv  is_ours != yes   ((dataset, model, method)-keyed, %)
    B3  results/repgrid/scores_lsml_upcr.csv published_Y  (cell-keyed anchor, 0-1)
    These fed the results/action_items/* pages.

For every (cell, method) pair the classification is:
    MATCH    same value (|delta| <= 0.005 after scale normalization), same method name
    RENAMED  same value, different method NAME — the Step-193b mislabel class
    DELTA    different value — resolution recorded, A wins (it carries paper evidence)
    ONLY_A / ONLY_B  present in one source only (coverage difference, not a conflict)

Output: results/advisor_inscope/reconciliation.csv (+ console summary).
The verification.html page renders this table; regenerate it after running this.
"""

import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI = os.path.join(REPO, "results", "advisor_inscope")
REPGRID = os.path.join(REPO, "results", "repgrid")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def to_unit(v):
    """Normalize 0-100-scale AUROCs to 0-1. Published AUROCs below 1.5 are
    already unit-scale (no real detector sits below 1.5%)."""
    if v is None:
        return None
    return v / 100.0 if v > 1.5 else v


# Method-name aliases across generations of the tables. Maps VARIANT spellings to
# one canonical key so a pure rename is not misread as a coverage difference.
# NOTE: "LapEigvals" is deliberately NOT aliased to "LapEigvals AttentionScore" —
# that pair is the Step-193b mislabel, and the whole point is to surface it as
# RENAMED rows, not to hide it in normalization.
ALIASES = {
    "selfcheckgpt": "SelfCheckGPT",
    "selfcheckgpt (official)": "SelfCheckGPT",
    "semantic entropy": "Semantic Entropy",
    "semantic entropy nli": "Semantic Entropy",
    "se (iclr'23)": "SE-ICLR23",
    "semantic entropy (kuhn iclr'23)": "SE-ICLR23",
    "eigenscore": "EigenScore (INSIDE)",
    "inside": "EigenScore (INSIDE)",
    "inside (eigenscore)": "EigenScore (INSIDE)",
    "attentionscore": "LapEigvals AttentionScore",
    "lapeigvals attentionscore": "LapEigvals AttentionScore",
    "attentionscore (lapeigvals paper)": "LapEigvals AttentionScore",
    "lapeigvals probe": "LapEigvals probe",
    "hcpd (arxiv 2606.12900)": "HCPD",
    "noise injection": "Noise Injection",
    "ars (ccs)": "ARS (CCS)",
    "internal-states+rc": "Internal-States+RC",
    "internal-states + reasoning-consistency": "Internal-States+RC",
}


def canon(method):
    m = (method or "").strip()
    return ALIASES.get(m.lower(), m)


def load_a():
    """Authoritative rows keyed (cell, canonical method)."""
    out = {}
    for r in read(os.path.join(AI, "competitors_verified.csv")):
        v = to_unit(fnum(r.get("auroc")))
        if v is None:
            continue
        out[(r["cell"], canon(r["method"]))] = {
            "auroc": v, "method": r["method"], "supervision": r.get("supervision", ""),
            "verified_by": r.get("verified_by", ""), "caveat": r.get("caveat", ""),
            "role": r.get("role", ""),
        }
    return out


def load_b():
    """Older rows keyed (cell, canonical method) -> list of (source, value, method, extra)."""
    out = {}

    def add(cell, method, val, source, extra=""):
        v = to_unit(fnum(val))
        if v is None or not cell:
            return
        out.setdefault((cell, canon(method)), []).append(
            {"source": source, "auroc": v, "method": method, "extra": extra})

    for r in read(os.path.join(REPGRID, "published_baselines.csv")):
        add(r["cell"], r["method"], r["auroc"], "published_baselines.csv",
            r.get("note", ""))

    # scores_lsml_upcr: one published anchor per cell, repeated across rows — dedupe.
    seen = set()
    for r in read(os.path.join(REPGRID, "scores_lsml_upcr.csv")):
        key = (r["cell"], r.get("Y_method", ""))
        if key in seen or not r.get("published_Y"):
            continue
        seen.add(key)
        add(r["cell"], r["Y_method"], r["published_Y"], "scores_lsml_upcr.csv",
            r.get("head_to_head", ""))

    # reasoning_benchmark has no cell key; map via each cell's (dataset, model) as
    # recorded in scores_lsml_upcr (the RB rows fed the same action_items pages).
    cell_dm = {}
    for r in read(os.path.join(REPGRID, "scores_lsml_upcr.csv")):
        ds = (r.get("dataset") or "").lower().replace("-", "").replace("_", "")
        md = (r.get("model") or "").split("/")[-1].lower()
        cell_dm.setdefault((ds, md), r["cell"])
    for r in read(os.path.join(REPO, "results", "reasoning_benchmark.csv")):
        if (r.get("is_ours") or "") == "yes":
            continue
        ds = (r.get("dataset") or "").lower().replace("-", "").replace("_", "")
        md = (r.get("model") or "").split("/")[-1].lower()
        cell = cell_dm.get((ds, md))
        if cell:
            add(cell, r["method"], r["auroc"], "reasoning_benchmark.csv",
                f"citable={r.get('citable', '')}; {r.get('note', '')}")
    return out


# The Step-193b/193d resolutions, keyed by (cell, old method name). Everything else
# that disagrees gets a generic "A wins (paper evidence)" note.
KNOWN_RESOLUTIONS = {
    ("lapeigvals_gsm8k_llama8b", "LapEigvals"):
        "old 0.925 is Mistral-Small-24B's LapEigvals column — wrong MODEL "
        "(Step 193b); own-model values are 0.720 (AttentionScore) / 0.872 (probe)",
    ("lapeigvals_gsm8k_llama3b", "LapEigvals"):
        "value right, method mislabeled: it is the paper's AttentionScore "
        "baseline (unsupervised), not supervised LapEigvals (Step 193b)",
    ("lapeigvals_gsm8k_phi35", "LapEigvals"):
        "value right, method mislabeled -> AttentionScore (Step 193b)",
    ("lapeigvals_gsm8k_mistral24b", "LapEigvals"):
        "value right, method mislabeled -> AttentionScore (Step 193b)",
    ("lapeigvals_gsm8k_nemo", "LapEigvals"):
        "value right, method mislabeled -> AttentionScore (Step 193b)",
}


def main():
    A, B = load_a(), load_b()
    rows = []
    cells_a = {c for c, _ in A}

    for (cell, mkey), a in sorted(A.items()):
        # Prefer same-canonical-name matches; fall back to EXACT-value matches
        # under a different name (a transcribed table value is identical across
        # generations, so renames match to the 4th decimal — a looser tolerance
        # would pair genuinely different methods that happen to score nearby).
        matched_sources = []
        for (bc, bm), lst in B.items():
            if bc != cell or bm != mkey:
                continue
            for b in lst:
                matched_sources.append((b, "same-name"))
        if not matched_sources:
            for (bc, bm), lst in B.items():
                if bc != cell or bm == mkey:
                    continue
                for b in lst:
                    if abs(b["auroc"] - a["auroc"]) <= 0.0005:
                        matched_sources.append((b, "renamed"))
        if not matched_sources:
            rows.append(dict(cell=cell, method_a=a["method"], method_b="",
                             source_b="", auroc_a=f"{a['auroc']:.4f}", auroc_b="",
                             delta="", status="ONLY_A",
                             resolution="new/renamed row introduced by the Step-193 "
                                        "rebuild (no old counterpart at this value)"))
            continue
        for b, how in matched_sources:
            d = a["auroc"] - b["auroc"]
            if how == "renamed":
                status = "RENAMED"
                res = KNOWN_RESOLUTIONS.get(
                    (cell, b["method"]),
                    "same value, different method name — old name superseded by "
                    "the Step-193 paper-verified name")
            elif abs(d) <= 0.005:
                status, res = "MATCH", ""
            else:
                status = "DELTA"
                res = KNOWN_RESOLUTIONS.get(
                    (cell, b["method"]),
                    "A wins: carries paper_table + paper_evidence "
                    "(competitors_verified.csv, Step 193)")
            rows.append(dict(cell=cell, method_a=a["method"], method_b=b["method"],
                             source_b=b["source"], auroc_a=f"{a['auroc']:.4f}",
                             auroc_b=f"{b['auroc']:.4f}", delta=f"{d:+.4f}",
                             status=status, resolution=res))

    # old rows with no counterpart in A at all (by cell+canonical name AND by value)
    for (cell, mkey), lst in sorted(B.items()):
        if (cell, mkey) in A:
            continue
        for b in lst:
            if any(abs(b["auroc"] - a["auroc"]) <= 0.0005
                   for (ac, _), a in A.items() if ac == cell):
                continue        # consumed above as RENAMED
            if cell not in cells_a:
                res = ("cell has no row in competitors_verified.csv (out-of-scope "
                       "cell or no published anchor) — coverage fact, not a conflict")
            elif b["source"] == "reasoning_benchmark.csv":
                res = ("baseline row from our OWN benchmark runs "
                       f"({b.get('extra', '').split(';')[0]}) — "
                       "competitors_verified.csv holds published paper-table values "
                       "only, so absence is structural, not a conflict")
            else:
                res = KNOWN_RESOLUTIONS.get(
                    (cell, b["method"]),
                    "old row not carried into the verified table — check whether it "
                    "was dropped deliberately (baseline of a baseline, protocol "
                    "mismatch) before ever quoting it again")
            rows.append(dict(cell=cell, method_a="", method_b=b["method"],
                             source_b=b["source"], auroc_a="",
                             auroc_b=f"{b['auroc']:.4f}", delta="",
                             status="ONLY_B", resolution=res))

    out_path = os.path.join(AI, "reconciliation.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell", "method_a", "method_b", "source_b",
                                           "auroc_a", "auroc_b", "delta", "status",
                                           "resolution"])
        w.writeheader()
        w.writerows(rows)

    tally = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print(f"wrote {len(rows)} rows -> {out_path}")
    for k in ("MATCH", "RENAMED", "DELTA", "ONLY_A", "ONLY_B"):
        print(f"  {k:8s} {tally.get(k, 0)}")
    for r in rows:
        if r["status"] in ("DELTA", "RENAMED"):
            print(f"  {r['status']:8s} {r['cell']:34s} {r['method_b'] or '-':28s} "
                  f"{r['auroc_b'] or '-':>7s} -> {r['method_a'] or '-':28s} "
                  f"{r['auroc_a'] or '-':>7s}  {r['resolution'][:60]}")


if __name__ == "__main__":
    main()
