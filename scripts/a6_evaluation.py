#!/usr/bin/env python
"""
a6_evaluation.py — evaluate the a6 pseudo-label-gates bench run against the two
PRE-REGISTERED gates (Step 194, registered before the bench finished):

  Mechanism gate:   mean rho(gate value, that view's own oriented AUROC) > 0.30
                    (a2.dufs baseline: +0.149, per-cell range -0.085..+0.342,
                     scripts/selector_choice_analysis.py).
                    For a6 the rho is computed over the views that HAVE learned
                    gates — the seed views are excluded (they are always-on by
                    construction, and including them at gate=1.0 would inflate
                    rho with information the gates never learned).

  Performance gate: macro AUROC over the 25 in-scope cells beats a2.dufs
                    (0.7502) by >= +1.0pp with Wilcoxon p < 0.05.
                    Aspirational target: GOOD_6 0.7594.

Also reports: the a6.dufs control vs a2.dufs (harness sanity — must be ~equal),
and pl_dufs vs pl_rank (does gate training add anything over ranking by
agreement with the pseudo-label).

Inputs:  results/selector_bench/a6_pseudolabel_gates__c46.csv
         results/selector_bench/a2_groupfs__c46.csv
         results/selector_bench/reference_macros__c46.csv
         results/selector_bench/inscope_feature_orientation.csv
Output:  results/advisor_inscope/a6_evaluation.csv + console verdicts.
"""

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from subset_gap_analysis import spearman            # noqa: E402
from inscope_cells import INSCOPE                   # noqa: E402

BENCH = os.path.join(REPO, "results", "selector_bench")
AI = os.path.join(REPO, "results", "advisor_inscope")

RHO_GATE_THRESHOLD = 0.30
PERF_DELTA_PP = 1.0
A2_BASELINE_RHO = 0.149


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def by_cell(rows, variant):
    out = {}
    for r in rows:
        if r["variant"] == variant and r["cell"] in INSCOPE:
            out[r["cell"]] = r
    return out


def macro(d):
    vals = [fnum(r["auroc"]) for r in d.values()]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def paired(d_a, d_b):
    cells = sorted(set(d_a) & set(d_b))
    a = [fnum(d_a[c]["auroc"]) for c in cells]
    b = [fnum(d_b[c]["auroc"]) for c in cells]
    keep = [(x, y, c) for x, y, c in zip(a, b, cells) if x is not None and y is not None]
    return ([x for x, _, _ in keep], [y for _, y, _ in keep], [c for _, _, c in keep])


def main():
    a6 = read(os.path.join(BENCH, "a6_pseudolabel_gates__c46.csv"))
    a2 = read(os.path.join(BENCH, "a2_groupfs__c46.csv"))
    ref = read(os.path.join(BENCH, "reference_macros__c46.csv"))
    orient = read(os.path.join(BENCH, "inscope_feature_orientation.csv"))

    af = {}
    for r in orient:
        v = fnum(r["oriented_auroc"])
        if v is not None:
            af.setdefault(r["cell"], {})[r["feature"]] = v

    pl = by_cell(a6, "a6.pl_dufs")
    pln = by_cell(a6, "a6.pl_dufs_noseed")
    plr = by_cell(a6, "a6.pl_rank")
    ctrl = by_cell(a6, "a6.dufs")
    a2d = by_cell(a2, "a2.dufs")
    g6 = by_cell(ref, "ref.GOOD_6")

    n_cells = len({r["cell"] for r in a6 if r["cell"] in INSCOPE})
    n_fb = sum(1 for r in a6 if r["cell"] in INSCOPE and r["fallback"] == "True")
    print(f"a6 bench coverage: {n_cells}/25 in-scope cells, {n_fb} fallbacks")
    if n_cells < 25:
        print("!! PARTIAL RUN — verdicts below are provisional\n")

    # ---- mechanism gate ----------------------------------------------------
    out_rows, rhos = [], []
    for cell, r in sorted(pl.items()):
        try:
            diag = json.loads(r.get("diag_json") or "{}")
        except json.JSONDecodeError:
            diag = {}
        gates = diag.get("feat_gate_means_learned")
        agree = diag.get("agreement_with_pl", {})
        rho_cell = float("nan")
        learned = [g for g in (gates or []) if g is not None]
        # agreement_with_pl preserves selectable-pool order (dict insertion order ==
        # sel_cols order == pool order), so its keys name the learned gates 1:1.
        if learned and cell in af and len(agree) == len(learned):
            pairs = [(g, af[cell].get(nm)) for g, nm in zip(learned, agree)]
            pairs = [(g, a) for g, a in pairs if a is not None]
            if len(pairs) >= 5:
                rho_cell, _ = spearman([g for g, _ in pairs], [a for _, a in pairs])
        if rho_cell == rho_cell:
            rhos.append(rho_cell)
        out_rows.append(dict(
            cell=cell,
            auroc_pl_dufs=r["auroc"],
            auroc_pl_noseed=(pln.get(cell) or {}).get("auroc", ""),
            auroc_pl_rank=(plr.get(cell) or {}).get("auroc", ""),
            auroc_a6_ctrl=(ctrl.get(cell) or {}).get("auroc", ""),
            auroc_a2_dufs=(a2d.get(cell) or {}).get("auroc", ""),
            auroc_good6=(g6.get(cell) or {}).get("auroc", ""),
            rho_gate_vs_featauroc=f"{rho_cell:.4f}" if rho_cell == rho_cell else "",
            n_selected=diag.get("n_gated_open", ""),
            seed_views="|".join(diag.get("seed_views", [])),
            lambda3_mult=diag.get("lambda3_mult", ""),
        ))

    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    mech = mean_rho > RHO_GATE_THRESHOLD
    print(f"\nMECHANISM GATE: mean rho(learned gate, view oriented AUROC) = {mean_rho:+.3f} "
          f"over {len(rhos)} cells (range {min(rhos):+.3f}..{max(rhos):+.3f})"
          if rhos else "\nMECHANISM GATE: no rho computable")
    print(f"  threshold > {RHO_GATE_THRESHOLD:+.2f} (a2 baseline {A2_BASELINE_RHO:+.3f}) "
          f"-> {'PASS' if mech else 'FAIL'}")
    print("  note: a6 rho excludes the always-on seed views; a2's baseline includes "
          "its full pool — quote both definitions when reporting.")

    # ---- performance gate --------------------------------------------------
    m_pl, n_pl = macro(pl)
    m_a2, _ = macro(a2d)
    m_g6, _ = macro(g6)
    m_ctrl, _ = macro(ctrl)
    m_plr, _ = macro(plr)
    m_pln, _ = macro(pln)
    print(f"\nmacros (over available cells): a6.pl_dufs {m_pl:.4f} ({n_pl} cells) | "
          f"a6.pl_dufs_noseed {m_pln:.4f} | a6.pl_rank {m_plr:.4f} | "
          f"a6.dufs ctrl {m_ctrl:.4f} | a2.dufs {m_a2:.4f} | GOOD_6 {m_g6:.4f}")

    a, b, cells = paired(pl, a2d)
    if len(a) >= 5:
        diff = [x - y for x, y in zip(a, b)]
        try:
            stat_p = wilcoxon(a, b).pvalue
        except ValueError:
            stat_p = float("nan")
        d_pp = 100 * (np.mean(a) - np.mean(b))
        w = sum(1 for x in diff if x > 1e-9)
        l = sum(1 for x in diff if x < -1e-9)
        perf = (d_pp >= PERF_DELTA_PP) and (stat_p < 0.05)
        print(f"\nPERFORMANCE GATE: a6.pl_dufs vs a2.dufs on {len(a)} paired cells: "
              f"{d_pp:+.2f}pp, {w}W/{l}L, Wilcoxon p = {stat_p:.4f}")
        print(f"  threshold >= +{PERF_DELTA_PP}pp AND p < 0.05 -> {'PASS' if perf else 'FAIL'}")

    # ---- sanity + ablation -------------------------------------------------
    a, b, _ = paired(ctrl, a2d)
    if a:
        print(f"\nharness sanity: a6.dufs control vs a2.dufs: "
              f"{100*(np.mean(a)-np.mean(b)):+.2f}pp over {len(a)} cells "
              f"(must be ~0 for the arms to be comparable)")
    a, b, _ = paired(pl, plr)
    if a:
        print(f"gates-vs-ranking ablation: a6.pl_dufs vs a6.pl_rank: "
              f"{100*(np.mean(a)-np.mean(b)):+.2f}pp over {len(a)} cells")
    a, b, _ = paired(pl, pln)
    if a:
        print(f"seed contribution: a6.pl_dufs vs a6.pl_dufs_noseed: "
              f"{100*(np.mean(a)-np.mean(b)):+.2f}pp over {len(a)} cells")

    out_path = os.path.join(AI, "a6_evaluation.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
