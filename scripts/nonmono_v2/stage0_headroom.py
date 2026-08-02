#!/usr/bin/env python
"""
stage0_headroom.py — is there anything for a transform to win, and can it reach it?

TWO QUESTIONS, ANSWERED BEFORE THE TRANSFORM BENCH RUNS SO NEITHER ANSWER CAN BE
REVERSE-ENGINEERED FROM RESULTS.

S0.2 — THE CEILING. AUROC is invariant to any monotone reparametrisation of the
score, so the cross-fitted gap between an unconstrained bin map of the FUSED score
and the best monotone reading of it IS the total non-monotone headroom of the whole
pipeline on that cell. If the fused score is already monotone in P(correct), the
fusion has absorbed whatever the individual views were doing and no per-view
reshaping can add anything -- full stop, whatever the per-view curves look like.

    Pre-registered declaration: G1 asks for >= +0.5pp macro over 24 cells, i.e.
    +12.0pp summed. Let H = sum over cells of max(headroom_pp, 0). If H < 12.0pp on
    BOTH arms, G1 is declared ARITHMETICALLY UNREACHABLE now, and the scoped gate
    G1s becomes the primary endpoint. G1 is computed and reported either way.

S0.3 — THE CHANNEL. Step 217's transform moved the fused score by EXACTLY zero on
19 of 24 cells (arm A) and on 4 of the 5 cells that carry the detections (arm B).
The reason is `upcr.py:289`:

    keep = (rho_full >= min_frac*var_y) & (rho_full >= rho_max/exclude_frac)

rho_hat is estimated from LINEAR covariance, so a U-shaped view has rho_hat ~ 0 and
is dropped before fusion -- after which its column contents cannot affect anything.
That is simultaneously why the experiment was null and the only channel through
which a working transform can act: monotonising a view makes rho_hat non-zero and
lets it back in. So `kept` and `dufs_selected` are pre-registered here as the
SECONDARY ENDPOINTS, and they are far more sensitive than the macro.

Outputs: results/nonmono_v2/{headroom.csv, mechanism.csv, stage0.json}
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (OUT, GROUP, INSCOPE, load_cells_cached, lsml_score,   # noqa: E402
                    upcr_score, dufs_selection, d1_shape_gain, n_min,
                    isoboot_labels, besag_clifford, iso_fit)
from sklearn.metrics import roc_auc_score                                 # noqa: E402

G1_PER_CELL_PP = 0.5           # gate G1's per-cell macro requirement
RECURRENT = ["rpdi", "pe_mean", "cusum_shift_idx", "hurst_exponent"]
SHAPE_TEST = os.path.join(os.path.dirname(OUT), "nonmono_transform", "shape_test.csv")


def headroom(score, y, rng, n_perm=199):
    """Non-monotone headroom of one fused score, with its monotone-null p-value.

    Same statistic as the per-view detector (`d1_shape_gain`), so the ceiling and
    the per-view effects are on one scale and directly comparable."""
    g, sd, kbest = d1_shape_gain(score, y)
    if not np.isfinite(g):
        return dict(gain=float("nan"), sd=float("nan"), k=-1, p=float("nan"))
    p, h, b, _ = besag_clifford(
        lambda yy: d1_shape_gain(score, yy, seeds=(0,))[0],
        lambda r: isoboot_labels(score, y, r),
        g, rng, n_target=10, b_max=n_perm)
    return dict(gain=float(g), sd=float(sd), k=int(kbest), p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=199,
                    help="isotonic-bootstrap draws for the headroom p-value")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cells = load_cells_cached(use_cache=not args.no_cache)
    dufs = dufs_selection()
    rng = np.random.default_rng(0)

    # flagged pairs from Step 217, for the mechanism table
    flagged = set()
    if os.path.exists(SHAPE_TEST):
        with open(SHAPE_TEST, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["exceeds_null"] == "1":
                    flagged.add((r["cell"], r["feature"]))

    hrows, mrows = [], []
    print(f"\n{'='*100}\nS0.2  NON-MONOTONE HEADROOM OF THE FUSED SCORE\n{'='*100}")
    print(f"{'cell':<34}{'dom':<6}{'n':>6}{'armA auc':>10}{'hd pp':>8}{'p':>7}"
          f"{'armB auc':>10}{'hd pp':>8}{'p':>7}{'keep':>6}")
    print("-" * 100)

    for ck in INSCOPE:
        cell = cells[ck]
        V, y, pool, anchor = cell["V"], cell["labels"], cell["pool"], cell["anchor"]
        chosen = [f for f in dufs.get(ck, []) if f in pool]
        colsA = [pool.index(f) for f in chosen]

        sA = lsml_score(V, anchor, colsA) if len(colsA) >= 3 else None
        sB, resB, rho1 = upcr_score(V, pool, anchor)

        hA = headroom(sA, y, rng, args.n_perm) if sA is not None else \
            dict(gain=float("nan"), sd=float("nan"), k=-1, p=float("nan"))
        hB = headroom(sB, y, rng, args.n_perm)
        aucA = float(roc_auc_score(y, sA)) if sA is not None else float("nan")
        aucB = float(roc_auc_score(y, sB))

        for arm, auc, h, nk in (("dufs_lsml", aucA, hA, len(colsA)),
                                ("upcr", aucB, hB, int(resB.keep.sum()))):
            hrows.append(dict(
                cell=ck, domain=GROUP.get(ck, "?"), arm=arm, n=len(y),
                n_pos=int(y.sum()), n_min=n_min(y),
                auroc_fused=round(auc, 4),
                headroom_pp=(round(h["gain"] * 100, 3) if np.isfinite(h["gain"]) else ""),
                headroom_sd_pp=(round(h["sd"] * 100, 3) if np.isfinite(h["sd"]) else ""),
                best_k=h["k"],
                p_isoboot=(round(h["p"], 4) if np.isfinite(h["p"]) else ""),
                n_views_used=nk))

        print(f"{ck:<34}{GROUP.get(ck,'?'):<6}{len(y):>6}"
              f"{aucA:>10.4f}{hA['gain']*100:>8.2f}{hA['p']:>7.3f}"
              f"{aucB:>10.4f}{hB['gain']*100:>8.2f}{hB['p']:>7.3f}"
              f"{int(resB.keep.sum()):>6}", flush=True)

        # ── S0.3 mechanism: why a transform on view j can or cannot act ──────
        rho_kept = np.asarray(resB.rho_hat_full, dtype=float)
        targets = sorted({f for (c, f) in flagged if c == ck} | set(RECURRENT))
        for f in targets:
            if f not in pool:
                continue
            j = pool.index(f)
            x = V[:, j]
            auc_v = float(roc_auc_score(y, x))
            mrows.append(dict(
                cell=ck, domain=GROUP.get(ck, "?"), feature=f,
                flagged=int((ck, f) in flagged),
                recurrent=int(f in RECURRENT),
                n=len(y), n_min=n_min(y),
                single_view_auc=round(auc_v, 4),
                single_view_auc_oriented=round(max(auc_v, 1 - auc_v), 4),
                upcr_rho_hat=round(float(rho_kept[j]), 6),
                upcr_rho_rank=int(np.sum(rho_kept > rho_kept[j]) + 1),
                upcr_kept=int(bool(resB.keep[j])),
                upcr_w=round(float(resB.w[j]), 6) if resB.keep[j] else "",
                dufs_selected=int(f in chosen),
            ))

    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("headroom.csv", hrows), ("mechanism.csv", mrows)):
        p = os.path.join(OUT, name)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved {p}  ({len(rows)} rows)")

    # ══ the pre-registered declaration ═══════════════════════════════════════
    verdict = {}
    print(f"\n{'='*100}\nTHE PRE-REGISTERED DECLARATION\n{'='*100}")
    print(f"G1 asks for >= {G1_PER_CELL_PP}pp macro over {len(INSCOPE)} cells "
          f"= {G1_PER_CELL_PP*len(INSCOPE):+.1f}pp summed.\n")
    need = G1_PER_CELL_PP * len(INSCOPE)
    for arm in ("dufs_lsml", "upcr"):
        v = [r for r in hrows if r["arm"] == arm and r["headroom_pp"] != ""]
        pos = [max(float(r["headroom_pp"]), 0.0) for r in v]
        sig = [r for r in v if r["p_isoboot"] != "" and float(r["p_isoboot"]) < 0.05]
        H = float(np.sum(pos))
        H_sig = float(np.sum([max(float(r["headroom_pp"]), 0.0) for r in sig]))
        verdict[arm] = dict(H=round(H, 2), H_significant=round(H_sig, 2),
                            n_cells=len(v), n_sig=len(sig),
                            reachable=bool(H >= need))
        print(f"  {arm:<12} H = {H:6.2f}pp over {len(v)} cells   "
              f"({len(sig)} cells with p<0.05, contributing {H_sig:.2f}pp)   "
              f"-> {'G1 REACHABLE' if H >= need else 'G1 UNREACHABLE'}")
    both_unreachable = not any(verdict[a]["reachable"] for a in verdict)
    verdict["declaration"] = ("G1 IS ARITHMETICALLY UNREACHABLE; G1s becomes primary"
                              if both_unreachable else
                              "G1 stays the primary endpoint")
    print(f"\n  ==> {verdict['declaration']}")

    # ── the channel: how often can a transform even act? ────────────────────
    print(f"\n{'='*100}\nS0.3  THE CHANNEL — can a transform on these views reach the fusion?\n{'='*100}")
    fl = [r for r in mrows if r["flagged"]]
    print(f"  flagged (cell,view) pairs present in the live pool : {len(fl)}")
    if fl:
        kept = sum(r["upcr_kept"] for r in fl)
        sel = sum(r["dufs_selected"] for r in fl)
        print(f"    kept by U-PCR (arm B can see it)               : {kept}/{len(fl)}"
              f"   -> {len(fl)-kept} are EXCLUDED before fusion")
        print(f"    selected by DUFS (arm A can see it)            : {sel}/{len(fl)}"
              f"   -> {len(fl)-sel} never enter arm A")
        both = sum(1 for r in fl if not r["upcr_kept"] and not r["dufs_selected"])
        print(f"    invisible to BOTH arms                         : {both}/{len(fl)}")
    rec = [r for r in mrows if r["recurrent"]]
    if rec:
        print(f"  the 4 recurrent views across all cells ({len(rec)} rows):")
        print(f"    kept by U-PCR   {sum(r['upcr_kept'] for r in rec):>3}/{len(rec)}")
        print(f"    chosen by DUFS  {sum(r['dufs_selected'] for r in rec):>3}/{len(rec)}")

    p = os.path.join(OUT, "stage0.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(dict(g1_needs_pp=need, n_cells=len(INSCOPE), verdict=verdict), fh, indent=1)
    print(f"\nsaved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
