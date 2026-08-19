#!/usr/bin/env python
"""
Advisor-facing comparison grid for the two lanes we can already report: error
LOCALIZATION on ProcessBench, and EARLY (prefix) DETECTION.

Numbers are read from the frozen Stage-4 scorer-transfer audit under
`results/local_online_comprehensive_v1/` and never retyped. Every source file is hashed into
the output, so a number in the grid can be traced to the bytes it came from.

Grid shape follows the house rule for ours-versus-theirs tables: ONE grid per lane, one column
per method, one row per metric, the direction stated in the row label, and the winner marked
explicitly. Access tiers are never mixed inside a grid — a step-supervised PRM and an 8-pass
72B critic go in their own panel, because ranking them beside a single pass over
log-probabilities we already had would imply a comparison that is not being made.

Two things this script refuses to do:

  * collapse a parity into a win. Every delta is reported with its grouped 95% interval and a
    verdict of `ours wins` / `parity` / `ours loses` decided by whether the interval excludes
    zero — not by the sign of the point estimate.
  * present the DeepConf columns as the published method. Those are our approximate proxies
    (`deepconf_w32`, `deepconf_w64`) computed from saved log-probabilities, and they are
    labelled as proxies everywhere. The pinned official confidence is what the M2 acquisition
    is for, and it is exactly preregistered null N7 of the frozen prefix-lane registry.

Usage:
    python scripts/localization_online_comparison_grid.py
    python scripts/localization_online_comparison_grid.py --out results/advisor_localization_online
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The grid marks winners with a check mark, and a Windows console defaults to cp1252, which
# cannot encode it. The files are UTF-8 regardless; this only keeps the echo from dying.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 — older interpreters / redirected streams
    pass

SRC = os.path.join(REPO_ROOT, "results", "local_online_comprehensive_v1")

#: Descriptive names. Codenames like "step272" or "iu28" mean nothing to a reader who was not
#: in the room, so every column says what the method does.
LABELS = {
    "iu28_registered":
        "28-stream causal prefix (ours)",
    "finalist_global_detector_local_locator":
        "Global detector + step-top-5 locator (new architecture)",
    "max_entropy__step_top5mean":
        "Maximum entropy + step-top-5 locator",
    "max_entropy__persistent_q90_3":
        "Maximum entropy + persistent-q90 locator",
    "max_entropy__peak":
        "Maximum entropy + peak locator",
    "max_entropy":              "Maximum entropy",
    "mean_entropy":             "Mean entropy",
    "step272_twohead":          "Two-head trajectory (Step 272)",
    "gl_liu_v1_replay":         "GL-LIU replay",
    "mind_the_gap":             "Mind the Gap / Evidence-Drop",
    "deepconf_w64":             "DeepConf lowest-group confidence, window 64 — our proxy",
    "deepconf_w32":             "DeepConf lowest-group confidence, window 32 — our proxy",
    "qwen_prm":                 "Qwen2.5-Math-PRM-7B (step-level supervision)",
    "qwen72b_critic":           "Qwen2.5-72B critic (8-sample vote)",
    "qwen3_judge_control":      "Qwen3-8B judge control",
}

#: Which method is "ours" in each lane, i.e. the reference the paired intervals are against.
OURS = {"local": "finalist_global_detector_local_locator", "online": "iu28_registered"}

LANE_TITLE = {
    "local": "Error localization — ProcessBench, macro F1 over four families",
    "online": "Early (prefix) detection — AUROC at 64 and 128 observed tokens",
}
LANE_METRIC = {"local": "ProcessBench macro F1", "online": "AUROC@64/128"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(name):
    path = os.path.join(SRC, name)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f)), {"path": os.path.relpath(path, REPO_ROOT),
                                         "sha256": sha256_file(path)}


def verdict(ci_low, ci_high):
    """A verdict is decided by the interval, never by the sign of the point estimate.

    Deltas in the source audit are always (this method - the lane's DIRECT REFERENCE), where
    the direct reference is the strongest same-access competitor. So the verdict is stated
    against that reference, which is the comparison that was actually computed. In the
    early-detection lane the reference happens to be our own method; in the localization lane
    it is maximum entropy, and our architecture is one of the rows being judged against it.

    This is exactly why the audit's localization headline is a parity and not a +0.0048 win.
    """
    if ci_low is None or ci_high is None:
        return "reference"
    if ci_low > 0:
        return "beats reference"
    if ci_high < 0:
        return "loses to reference"
    return "parity"


def build_lane(lane, agg, intervals):
    ours = OURS[lane]
    prim = {r["candidate"]: (float(r["primary"]), r["access_tier"])
            for r in agg if r["task"] == lane}
    iv = {r["candidate"]: r for r in intervals if r["task"] == lane}

    def row_for(cand):
        value, tier = prim[cand]
        d = iv.get(cand)
        lo = float(d["ci_low"]) if d else None
        hi = float(d["ci_high"]) if d else None
        return {
            "candidate": cand,
            "label": LABELS.get(cand, cand),
            "primary": value,
            "access_tier": tier,
            "is_ours": cand == ours,
            "delta_vs_ours": float(d["delta"]) if d else None,
            "ci_low": lo, "ci_high": hi,
            "family_wins": int(d["family_wins"]) if d else None,
            "family_losses": int(d["family_losses"]) if d else None,
            # The audit computes deltas as (candidate - direct reference), where the direct
            # reference is the strongest same-access competitor, not necessarily "ours". Only
            # rows whose reference IS our method get a verdict against us.
            "delta_reference": d["reference"] if d else None,
            "verdict": verdict(lo, hi) if d else "reference",
        }

    same = sorted([row_for(c) for c, (_, t) in prim.items() if t == "A"],
                  key=lambda r: -r["primary"])
    higher = sorted([row_for(c) for c, (_, t) in prim.items() if t != "A"],
                    key=lambda r: -r["primary"])
    refs = {r["delta_reference"] for r in same if r["delta_reference"]}
    if len(refs) > 1:
        raise SystemExit(f"{lane}: rows are paired against more than one reference {refs}; "
                         "one grid cannot state a single delta row for them")
    reference = (refs.pop() if refs else ours)
    return {"same_access": same, "higher_access": higher, "reference": reference,
            "reference_label": LABELS.get(reference, reference), "ours": ours}


def md_grid(lane, rows):
    """One grid: a column per method, a row per metric, direction in the row label."""
    cols = rows["same_access"]
    head = "| Metric (direction) | " + " | ".join(
        (f"**{c['label']}**" if c["is_ours"] else c["label"]) for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"

    best = max(c["primary"] for c in cols)
    line_primary = f"| {LANE_METRIC[lane]} — higher is better | " + " | ".join(
        (f"**{c['primary']:.4f}** ✅" if abs(c["primary"] - best) < 1e-12
         else f"{c['primary']:.4f}") for c in cols) + " |"

    def cell(c, key, fmt="{:+.4f}"):
        v = c[key]
        return "—" if v is None else fmt.format(v)

    ref, ref_label = rows["reference"], rows["reference_label"]
    isref = lambda c: c["candidate"] == ref            # noqa: E731

    line_delta = (f"| Paired delta vs the reference — positive is better than the reference "
                  f"| " + " | ".join(
                      ("reference" if isref(c) else cell(c, "delta_vs_ours"))
                      for c in cols) + " |")
    line_ci = "| Grouped 95% interval on that delta | " + " | ".join(
        ("—" if isref(c) or c["ci_low"] is None
         else f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]") for c in cols) + " |"
    line_fam = "| Families won / lost vs the reference (of 4) | " + " | ".join(
        ("—" if isref(c) or c["family_wins"] is None
         else f"{c['family_wins']}/{c['family_losses']}") for c in cols) + " |"
    line_v = "| Verdict vs the reference | " + " | ".join(
        ("**reference**" if isref(c) else
         (f"**{c['verdict']}**" if c["verdict"] == "beats reference" else c["verdict"]))
        for c in cols) + " |"

    ours_label = LABELS.get(rows["ours"], rows["ours"])
    note = (f"Every paired delta in this grid is against **{ref_label}**, the lane's direct "
            f"reference — that is the comparison the audit actually computed. Our method here "
            f"is **{ours_label}**"
            + (", which *is* the reference." if ref == rows["ours"] else
               f", which is one of the rows judged against it."))
    out = [f"### {LANE_TITLE[lane]}", "",
           "Same access throughout: one generated trace, log-probabilities only, one model "
           "pass. Deltas are paired by source question across the Qwen3-8B and "
           "Llama-3.1-8B scorer copies.", "", note, "",
           head, sep, line_primary, line_delta, line_ci, line_fam, line_v, ""]

    if rows["higher_access"]:
        out += ["#### Higher-access panel — not comparable to the grid above", "",
                "These see more than we do: step-level supervision, or eight sampled passes "
                "of a 72B model. They are compute ceilings on the same questions, never "
                "same-access deltas.", "",
                "| Method | What it sees | " + LANE_METRIC[lane] + " |", "|---|---|---:|"]
        seen = {"qwen_prm": "step-level PRM800K supervision, 1 pass",
                "qwen72b_critic": "no labels, 8 sampled passes of a 72B model",
                "qwen3_judge_control": "no labels, 1 pass, judge prompt"}
        for r in rows["higher_access"]:
            out.append(f"| {r['label']} | {seen.get(r['candidate'], '—')} "
                       f"| {r['primary']:.4f} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results",
                                                  "advisor_localization_online"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    agg, src_agg = read_csv("STAGE_4_AGGREGATE.csv")
    intervals, src_iv = read_csv("STAGE_4_INTERVALS.csv")
    dec_path = os.path.join(SRC, "STAGE_4_DECISION.json")
    with open(dec_path, encoding="utf-8") as f:
        decision = json.load(f)

    lanes = {lane: build_lane(lane, agg, intervals) for lane in ("online", "local")}

    report = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [src_agg, src_iv,
                    {"path": os.path.relpath(dec_path, REPO_ROOT),
                     "sha256": sha256_file(dec_path)}],
        "protocol_sha256": decision.get("protocol_sha256"),
        "preregistered_verdict": decision.get("verdict"),
        "ours_per_lane": OURS,
        "lanes": lanes,
        "caveats": [
            "The DeepConf columns are OUR approximate proxies from saved log-probabilities, "
            "not the published method. The pinned official confidence is being acquired (M2) "
            "and is preregistered null N7 of the prefix-lane claim registry.",
            "Tier-B rows are same-question compute ceilings, not same-access deltas.",
            "Potential tokens remaining are not realized savings.",
            "The localization result is a PARITY: the point estimate favours the new "
            "architecture by +0.0048 macro F1 and its grouped interval crosses zero.",
        ],
    }
    with open(os.path.join(args.out, "COMPARISON.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = ["# Where our method stands: localization and early detection", "",
          f"Generated {report['written_utc']} from the frozen Stage-4 scorer-transfer audit.",
          f"Protocol `{report['protocol_sha256'][:16]}…`. "
          f"Preregistered verdict of that audit: **`{report['preregistered_verdict']}`**.", "",
          md_grid("online", lanes["online"]),
          md_grid("local", lanes["local"]),
          "## Caveats that travel with these numbers", ""]
    md += [f"- {c}" for c in report["caveats"]]
    md += ["", "## Sources (hashed)", ""]
    md += [f"- `{s['path']}` — `{s['sha256'][:16]}…`" for s in report["sources"]]
    text = "\n".join(md) + "\n"
    with open(os.path.join(args.out, "COMPARISON.md"), "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
