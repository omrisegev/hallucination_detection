#!/usr/bin/env python
"""
test_sign_repair.py — the pre-registered repair for the sign-recovery mechanism.

PRE-REGISTERED IN HISTORY STEP 210, VERBATIM:
  "A better label-free relative-sign estimator. Candidate: Z2 synchronisation on
   the sign pattern of the correlation matrix (spectral_utils/orientation.
   z2_sign_recovery, already written and unused on this path).
   GATE: lift recovery above 0.90 on the four failing cells while moving the
   healthy ones by less than 0.5pp."

The four failing cells (L-SML recovery < 0.90), from Step 210:
  seiclr_triviaqa_opt30b -1.269 | inside_coqa_llama7b -0.122
  ars_gsm8k_r1distill8b   0.153 | noise_gsm8k_phi3mini 0.755

PREMISE CHECK FIRST. Step 204 measured L-SML to be sign-gauge invariant — 1150 of
1150 sign vectors bit-identical. If that holds, feeding better signs INTO L-SML
is a no-op by construction and cannot be the repair. Gate P below tests it on
today's data before anything else runs, because the whole design depends on it.

So the repair is tested where signs can actually bind:
  A  z2 + simple average        replaces L-SML with an EXPLICIT sign estimator
  B  z2 + U-PCR                 replaces U-PCR's own sign(rho) polarity step
  C  z2 -> L-SML                the no-op control; must reproduce the baseline
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, os.path.join(REPO, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import INSCOPE, QA_CELLS                             # noqa: E402
from inscope_bench_common import load_cells, assert_good6, good6_cols   # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous                 # noqa: E402
from spectral_utils.orientation import z2_sign_recovery                 # noqa: E402
from spectral_utils.streaming_utils import anchor_orient                # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                       # noqa: E402
from spectral_utils.upcr import upcr_fit                                # noqa: E402

OUT = os.path.join(REPO, "results", "action_items_jul2026", "_data")

FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

FAILING = ["seiclr_triviaqa_opt30b", "inside_coqa_llama7b",
           "ars_gsm8k_r1distill8b", "noise_gsm8k_phi3mini"]
WEAK = FAILING + ["losnet_hotpotqa_mistral7b", "truthfulqa_llama8b",
                  "internalstates_gsm8k_qwen25_7b",
                  "trace_math500_qwenmath15b_k10", "lapeigvals_gsm8k_llama3b"]
MIN_DENOM_PP = 2.0


def A(y, s):
    return float(roc_auc_score(y, s))


def orient(s, anchor):
    return anchor_orient(np.asarray(s, float), anchor)[0]


def main():
    cells = load_cells()
    ok, macro = assert_good6(cells, verbose=True)
    if not ok:
        raise SystemExit(f"validity gate failed: {macro:.4f}")
    print(f"GATE 0 (validity): GOOD_6 macro {macro:.4f} PASS\n")

    with open(os.path.join(REPO, "results/selector_bench/a2_groupfs__c46.csv"),
              newline="", encoding="utf-8") as f:
        dufs = {r["cell"]: r for r in csv.DictReader(f)
                if r["variant"] == "a2.dufs_pf"}

    rows, invariance = [], []
    for ck in INSCOPE:
        c = cells[ck]
        y = np.asarray(c["labels"], int)
        V, pool, anc = c["V"], c["pool"], c["anchor"]
        cols = sorted({pool.index(f) for f in dufs[ck]["chosen"].split("|")
                       if f in pool})
        Vs = V[:, cols]

        # oracle per-view polarity, for the ladder's top rung only
        araw = np.array([roc_auc_score(y, V[:, j]) for j in range(len(pool))])
        osign = np.where(araw >= 0.5, 1.0, -1.0)

        # ── the estimator under test
        z2s = z2_sign_recovery(Vs)                    # on the deployed subset
        z2f = z2_sign_recovery(V)                     # on the full pool

        # ── baselines
        b_lsml = A(y, orient(lsml_continuous(*[Vs[:, i] for i in
                                              range(Vs.shape[1])])[0], anc))
        hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], float)
        V_un = V * hand
        der = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
        der[der == 0] = 1.0
        Fb = (V_un * der).T
        rb = upcr_fit(Fb, **FIT)
        b_upcr = A(y, orient(rb.w @ Fb, anc))

        # ── GATE P: is L-SML actually sign-invariant on today's data?
        c_lsml_z2 = A(y, orient(lsml_continuous(*[(Vs * z2s)[:, i] for i in
                                                  range(Vs.shape[1])])[0], anc))
        rng = np.random.default_rng(0)
        rs = rng.choice([-1.0, 1.0], size=Vs.shape[1])
        c_lsml_rand = A(y, orient(lsml_continuous(*[(Vs * rs)[:, i] for i in
                                                    range(Vs.shape[1])])[0], anc))
        invariance.append((ck, b_lsml, c_lsml_z2, c_lsml_rand))

        # ── ARM A: z2 + simple average (explicit signs, no L-SML)
        a_z2avg = A(y, orient((Vs * z2s).mean(axis=1), anc))
        a_z2avg_full = A(y, orient((V * z2f).mean(axis=1), anc))

        # ── ARM B: z2 replaces sign(rho) inside U-PCR
        Fz = (V_un * z2_sign_recovery(V_un)).T
        rz = upcr_fit(Fz, **FIT)
        a_z2upcr = A(y, orient(rz.w @ Fz, anc))

        # ── ladder context for the recovery ratio
        a_or = roc_auc_score(y, (Vs * osign[cols]).mean(axis=1))
        a_no = roc_auc_score(y, Vs.mean(axis=1))
        r2, r3 = max(a_or, 1 - a_or), max(a_no, 1 - a_no)
        cost = (r2 - r3) * 100
        rec = (lambda v: (round(((max(v, 1 - v) - r3) * 100) / cost, 3)
                          if cost > 0.5 else None))

        rows.append(dict(
            cell=ck, group="QA" if ck in QA_CELLS else "math",
            weak=ck in WEAK, failing=ck in FAILING,
            cost_pp=round(cost, 2), unstable=cost < MIN_DENOM_PP,
            base_lsml=round(b_lsml, 4), base_upcr=round(b_upcr, 4),
            z2_avg=round(a_z2avg, 4), z2_avg_full=round(a_z2avg_full, 4),
            z2_upcr=round(a_z2upcr, 4),
            d_z2avg=round((a_z2avg - b_lsml) * 100, 2),
            d_z2avg_full=round((a_z2avg_full - b_lsml) * 100, 2),
            d_z2upcr=round((a_z2upcr - b_upcr) * 100, 2),
            rec_lsml=rec(b_lsml), rec_z2avg=rec(a_z2avg),
            rec_z2upcr=rec(a_z2upcr),
            upcr_kept=int(rb.keep.sum()), z2upcr_kept=int(rz.keep.sum()),
        ))
        print(f"  {ck:32s} lsml {b_lsml:.4f} -> z2avg {a_z2avg:.4f} "
              f"({(a_z2avg - b_lsml) * 100:+.2f})   upcr {b_upcr:.4f} -> "
              f"z2upcr {a_z2upcr:.4f} ({(a_z2upcr - b_upcr) * 100:+.2f})",
              flush=True)

    # ── GATE P ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    mx_z2 = max(abs(b - z) for _, b, z, _ in invariance)
    mx_rd = max(abs(b - r) for _, b, _, r in invariance)
    print(f"GATE P (premise): L-SML sign-gauge invariance on today's data\n"
          f"  max |baseline - z2-signed|     = {mx_z2:.2e}\n"
          f"  max |baseline - random-signed| = {mx_rd:.2e}\n"
          f"  -> {'CONFIRMED invariant' if max(mx_z2, mx_rd) < 1e-9 else 'NOT invariant'}"
          f" — feeding signs into L-SML "
          f"{'is a no-op, as Step 204 measured' if max(mx_z2, mx_rd) < 1e-9 else 'DOES bind'}")

    # ── the pre-registered gate ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PRE-REGISTERED GATE")
    fail = [r for r in rows if r["failing"]]
    heal = [r for r in rows if not r["weak"]]
    for arm, rk, dk in (("A  z2 + simple average", "rec_z2avg", "d_z2avg"),
                        ("B  z2 inside U-PCR", "rec_z2upcr", "d_z2upcr")):
        lifted = [r for r in fail if r[rk] is not None and r[rk] >= 0.90]
        moved = [r for r in heal if abs(r[dk]) >= 0.5]
        print(f"\n{arm}")
        print(f"  (i)  recovery >= 0.90 on the four failing cells: "
              f"{len(lifted)}/4  -> {'PASS' if len(lifted) == 4 else 'FAIL'}")
        for r in fail:
            print(f"         {r['cell']:32s} {r['rec_lsml']} -> {r[rk]}"
                  f"   ({r[dk]:+.2f}pp AUROC)"
                  + ("   [ratio unstable]" if r["unstable"] else ""))
        print(f"  (ii) healthy cells moved < 0.5pp: "
              f"{len(heal) - len(moved)}/{len(heal)}  -> "
              f"{'PASS' if not moved else 'FAIL'}")
        for r in moved:
            print(f"         {r['cell']:32s} {r[dk]:+.2f}pp")

    # ── full regression ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FULL REGRESSION, all 25 cells (macro AUROC)")
    for k, lbl in (("base_lsml", "DUFS + L-SML (baseline)"),
                   ("base_upcr", "U-PCR + sign(rho) (baseline)"),
                   ("z2_avg", "z2 + simple average, deployed subset"),
                   ("z2_avg_full", "z2 + simple average, full pool"),
                   ("z2_upcr", "z2 inside U-PCR")):
        v = np.array([r[k] for r in rows])
        qa = np.array([r[k] for r in rows if r["group"] == "QA"])
        mt = np.array([r[k] for r in rows if r["group"] == "math"])
        print(f"  {lbl:40s} macro {v.mean():.4f}   QA {qa.mean():.4f}   "
              f"math {mt.mean():.4f}")
    print()
    for dk, base, lbl in (("d_z2avg", "base_lsml", "z2+avg  vs  DUFS+L-SML"),
                          ("d_z2avg_full", "base_lsml", "z2+avg(full) vs DUFS+L-SML"),
                          ("d_z2upcr", "base_upcr", "z2+U-PCR  vs  U-PCR")):
        d = np.array([r[dk] for r in rows])
        w = sum(1 for x in d if x > 0)
        ls = sum(1 for x in d if x < 0)
        try:
            p = stats.wilcoxon(d).pvalue
        except ValueError:
            p = float("nan")
        print(f"  {lbl:32s} {d.mean():+.2f}pp mean / {np.median(d):+.2f}pp median, "
              f"{w}W/{ls}L, p={p:.4f}")

    with open(os.path.join(OUT, "_repair.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, invariance_max=max(mx_z2, mx_rd),
                       good6_macro=macro), f, indent=1)
    print(f"\nwrote {os.path.join(OUT, '_repair.json')}")


if __name__ == "__main__":
    main()
