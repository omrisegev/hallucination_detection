#!/usr/bin/env python
"""Diagnostics for results/cumulative_vote_fusion_v1: what changes on the long-chain subsets.

Reads the OOF predictions and fits written by `cumulative_vote_fusion_v1.py` plus two committed
length sources (the prefix lane's `final_length`; the early-online score files' `trace_length`)
and writes ANALYSIS.md + ANALYSIS.json.  Labels are used only for evaluation.  The lower-quantile
readout sweep is descriptive (label-selected) and is reported as such, never as a candidate.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "cumulative_vote_fusion_v1"
PREFIX = ROOT / "results" / "fair_paper_exact_comparisons_v1" / "lanes" / "prefix" / "PER_QUESTION.jsonl"
NAMES = ["single:family6", "single:unified28", "single:gl_liu", "single:max_ent", "single:mind_gap"]
SUBS = ["gsm8k", "math", "olympiadbench", "omnimath"]
LONG = ["olympiadbench", "omnimath"]
SEED = 20260920


def load_lengths():
    lengths = {}
    if PREFIX.exists():
        with open(PREFIX, encoding="utf8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("final_length") is not None:
                    lengths[r["row_id"]] = int(r["final_length"])
    return lengths


def boot_delta(hit, base, n=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    d = hit.astype(float) - base.astype(float)
    m = len(d)
    draws = np.array([d[rng.integers(0, m, m)].mean() for _ in range(n)])
    return [float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def wquant(row, w, q):
    lo, hi = int(row.min()), int(row.max())
    grid = np.arange(lo, hi + 1)
    w = w / w.sum()
    cdf = ((row[:, None] <= grid[None, :]) * w[:, None]).sum(0)
    return int(grid[min(int(np.searchsorted(cdf, q, side="left")), len(grid) - 1)])


def main():
    rows = [r for r in csv.DictReader(open(RES / "PREDICTIONS_pooled.csv", encoding="utf8")) if int(r["label"]) != -1]
    rep = json.load(open(RES / "REPORT_pooled.json", encoding="utf8"))
    lab = np.array([int(r["label"]) for r in rows])
    fam = np.array([r["family"] for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    S = np.stack([np.array([int(r[n]) for r in rows]) for n in NAMES], 1)
    P = {n: S[:, j] for j, n in enumerate(NAMES)}
    for extra in ("median", "ds_mode"):
        P[extra] = np.array([int(r[extra]) for r in rows])
    mx = S.max(1)
    fam6 = P["single:family6"]
    long_mask = np.isin(fam, LONG)
    out = {}
    md = ["# Cumulative-vote fusion v1 — long-chain diagnostics\n"]

    # 1. fitted parameters (pooled scope), averaged over folds
    fits = rep["fits"]
    avg = lambda k: np.mean([f[k] for f in fits], 0)
    out["fit_avg"] = {"sml_w": avg("sml_w").tolist(), "lsml_w": avg("lsml_w").tolist(),
                      "ds_psi": avg("ds_psi").tolist(), "ds_eta": avg("ds_eta").tolist(),
                      "lsml_groups_per_fold": [f["lsml"]["c"] for f in fits]}
    md.append("## 1. Label-free fitted parameters (pooled scope, mean over 5 folds)\n")
    md.append("| localizer | SML w | L-SML w | psi = P(votes <=n | truly <=n) (not-late) | eta = P(votes >n | truly >n) (not-early) |")
    md.append("|---|---:|---:|---:|---:|")
    for j, n in enumerate(NAMES):
        md.append(f"| {n[7:]} | {out['fit_avg']['sml_w'][j]:+.3f} | {out['fit_avg']['lsml_w'][j]:+.3f} | {out['fit_avg']['ds_psi'][j]:.3f} | {out['fit_avg']['ds_eta'][j]:.3f} |")
    md.append(f"\nL-SML groups per fold (index = localizer order above): {out['fit_avg']['lsml_groups_per_fold']}\n")

    # 2. offsets per subset
    md.append("## 2. Signed offset (prediction − first error) per subset, erroneous answers\n")
    md.append("| subset | n | label median / p90 | localizer | early | exact | late | mean offset |")
    md.append("|---|---:|---|---|---:|---:|---:|---:|")
    out["offsets"] = {}
    for s in SUBS + ["all"]:
        m = np.ones(len(lab), bool) if s == "all" else fam == s
        out["offsets"][s] = {}
        for n in NAMES + ["median", "ds_mode"]:
            off = P[n][m] - lab[m]
            d = {"early": float(np.mean(off < 0)), "exact": float(np.mean(off == 0)), "late": float(np.mean(off > 0)), "mean": float(off.mean())}
            out["offsets"][s][n] = d
            md.append(f"| {s} | {m.sum()} | {np.median(lab[m]):.0f} / {np.percentile(lab[m], 90):.0f} | {n} | {d['early']:.2f} | {d['exact']:.2f} | {d['late']:.2f} | {d['mean']:+.2f} |")

    # 3. lateness vs the uniform-guess null, by label position, long subsets
    md.append("\n## 3. Is the lateness more than argmax noise? Mean offset by true position vs a uniform guess over [0, max locator]\n")
    md.append("Long subsets (OlympiadBench + Omni-MATH), erroneous answers.\n")
    md.append("| first error at | n | uniform guess | family6 | max_ent | gl_liu | mind_gap | unified28 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    out["null_comparison_long"] = {}
    for k in range(0, 7):
        m = long_mask & (lab == k)
        if m.sum() == 0:
            continue
        rec = {"n": int(m.sum()), "uniform": float(np.mean(mx[m] / 2 - k))}
        for n in NAMES:
            rec[n[7:]] = float(np.mean(P[n][m] - k))
        out["null_comparison_long"][k] = rec
        md.append(f"| {k} | {rec['n']} | {rec['uniform']:+.2f} | {rec['family6']:+.2f} | {rec['max_ent']:+.2f} | {rec['gl_liu']:+.2f} | {rec['mind_gap']:+.2f} | {rec['unified28']:+.2f} |")

    # 4. accuracy and lateness vs chain depth proxy (max locator), all subsets
    md.append("\n## 4. Incumbent (family6) accuracy and lateness vs chain depth (max locator across the five)\n")
    md.append("| depth bucket | n | SLA % | late | early | label median |")
    md.append("|---|---:|---:|---:|---:|---:|")
    out["depth"] = []
    for lo, hi in [(0, 2), (3, 4), (5, 7), (8, 10 ** 6)]:
        k = (mx >= lo) & (mx <= hi)
        rec = {"lo": lo, "hi": hi, "n": int(k.sum()), "sla": float(np.mean(fam6[k] == lab[k])),
               "late": float(np.mean(fam6[k] > lab[k])), "early": float(np.mean(fam6[k] < lab[k])), "label_median": float(np.median(lab[k]))}
        out["depth"].append(rec)
        md.append(f"| {lo}–{hi if hi < 10**6 else '∞'} | {rec['n']} | {100 * rec['sla']:.1f} | {rec['late']:.2f} | {rec['early']:.2f} | {rec['label_median']:.0f} |")

    # 5. token length terciles within subset (prefix-lane final_length, where available)
    lengths = load_lengths()
    have = np.array([r["row_id"] in lengths for r in rows])
    md.append(f"\n## 5. By token length (prefix-lane `final_length`, available for {int(have.sum())} of {len(rows)} erroneous answers)\n")
    md.append("| subset | tercile (tokens) | n | family6 SLA | family6 late | mind_gap SLA | mind_gap late | median SLA |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    out["length_terciles"] = {}
    if have.sum():
        L = np.array([lengths.get(r["row_id"], -1) for r in rows])
        for s in SUBS:
            m = (fam == s) & have
            if m.sum() < 30:
                continue
            cuts = np.percentile(L[m], [33.3, 66.7])
            ter = np.digitize(L[m], cuts)
            out["length_terciles"][s] = {"cuts": cuts.tolist(), "rows": []}
            for t in range(3):
                k = np.where(m)[0][ter == t]
                rec = {"n": int(len(k)), "family6_sla": float(np.mean(fam6[k] == lab[k])), "family6_late": float(np.mean(fam6[k] > lab[k])),
                       "mind_gap_sla": float(np.mean(P["single:mind_gap"][k] == lab[k])), "mind_gap_late": float(np.mean(P["single:mind_gap"][k] > lab[k])),
                       "median_sla": float(np.mean(P["median"][k] == lab[k]))}
                out["length_terciles"][s]["rows"].append(rec)
                label = ["short", "mid", "long"][t]
                md.append(f"| {s} | {label} (cuts {cuts[0]:.0f}/{cuts[1]:.0f}) | {rec['n']} | {100 * rec['family6_sla']:.1f} | {rec['family6_late']:.2f} | {100 * rec['mind_gap_sla']:.1f} | {rec['mind_gap_late']:.2f} | {100 * rec['median_sla']:.1f} |")

    # 6. positional prior
    md.append("\n## 6. Positional prior: where predictions land vs where errors are (all erroneous, %)\n")
    bins = list(range(0, 8))
    md.append("| | " + " | ".join(str(b) for b in bins) + " | 8+ |")
    md.append("|---|" + "---:|" * (len(bins) + 1))
    md.append("| label | " + " | ".join(f"{100 * np.mean(lab == b):.1f}" for b in bins) + f" | {100 * np.mean(lab >= 8):.1f} |")
    out["positional"] = {"label": [float(np.mean(lab == b)) for b in bins] + [float(np.mean(lab >= 8))]}
    for n in NAMES:
        out["positional"][n[7:]] = [float(np.mean(P[n] == b)) for b in bins] + [float(np.mean(P[n] >= 8))]
        md.append(f"| {n[7:]} | " + " | ".join(f"{100 * np.mean(P[n] == b):.1f}" for b in bins) + f" | {100 * np.mean(P[n] >= 8):.1f} |")

    # 7. lower-quantile readouts (descriptive sweep)
    md.append("\n## 7. Lower-quantile readouts of the fused cumulative vote (DESCRIPTIVE sweep; q is label-selected here, not a candidate)\n")
    w_by_fold = {f["fold"]: np.clip(np.array(f["sml_w"]), 0, None) for f in fits}
    Q = {"min": S.min(1), "2nd_min": np.sort(S, 1)[:, 1], "median": P["median"]}
    for q in (0.2, 0.3, 0.4, 0.5):
        Q[f"sml_q{q}"] = np.array([wquant(S[i], w_by_fold[fold[i]], q) for i in range(len(rows))])
    md.append("| readout | " + " | ".join(SUBS + ["all"]) + " | Δ vs family6 on long subsets (95% CI) | late / early on long |")
    md.append("|---|" + "---:|" * 5 + "---|---|")
    out["quantile_readouts"] = {}
    for k, p in Q.items():
        vals = [float(np.mean(p[fam == s] == lab[fam == s])) for s in SUBS] + [float(np.mean(p == lab))]
        d = boot_delta((p == lab)[long_mask], (fam6 == lab)[long_mask])
        out["quantile_readouts"][k] = {"sla": vals, "delta_long_vs_family6": d,
                                       "late_long": float(np.mean(p[long_mask] > lab[long_mask])), "early_long": float(np.mean(p[long_mask] < lab[long_mask]))}
        md.append(f"| {k} | " + " | ".join(f"{100 * v:.2f}" for v in vals) + f" | {100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}] | {out['quantile_readouts'][k]['late_long']:.2f} / {out['quantile_readouts'][k]['early_long']:.2f} |")
    vals = [float(np.mean(fam6[fam == s] == lab[fam == s])) for s in SUBS] + [float(np.mean(fam6 == lab))]
    md.append(f"| family6 (reference) | " + " | ".join(f"{100 * v:.2f}" for v in vals) + f" | — | {np.mean(fam6[long_mask] > lab[long_mask]):.2f} / {np.mean(fam6[long_mask] < lab[long_mask]):.2f} |")

    json.dump(out, open(RES / "ANALYSIS.json", "w"), indent=1)
    (RES / "ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
