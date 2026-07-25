#!/usr/bin/env python
"""
validate_k_rules.py — H2: do the label-free K rules actually predict oracle-K?

Extension H's H2 asks for a per-cell subset size derived from the covariance
spectrum instead of the fixed K=15. Step 200 reported the new `eff_rank` /
`mp_floor` rules as "successfully dynamically adapts feature set size per cell",
but that was measured on an un-z-scored pool and its own CSV showed a constant
K=3 on 25/25 cells.

This applies the SAME honest test that refuted D1 (the residual elbow) in Step
198: compare each label-free rule against the per-cell ORACLE K -- the prefix
size that actually maximises fused AUROC -- rather than against downstream AUROC.
Correlating with AUROC is not predicting the optimal size.

PRE-REGISTERED BAR (fixed before looking at the output):
  adopt a rule only if  Spearman(pred_k, oracle_k) >= 0.30 with p < 0.05
  AND its macro AUROC at the predicted K is >= the macro at fixed K=15.
Anything else is reported as REFUTED, exactly as D1 was.

Labels are used ONLY inside `oracle_k` / the macro columns, which are audit
entry points -- never inside a rule.

Usage:  python scripts/validate_k_rules.py
Writes: results/advisor_inscope/k_rule_validation.csv       (per cell)
        results/advisor_inscope/k_rule_validation_rules.csv (per rule)
"""
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

from spectral_utils.fusion_utils import lsml_continuous               # noqa: E402
from spectral_utils.selectors.adaptive_k import (                     # noqa: E402
    predict_k, raw_k, oracle_k, validate)
from inscope_bench_common import load_cells, assert_good6             # noqa: E402
from inscope_cells import GROUP                                       # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")
RULES = ('eff_rank', 'mp_floor', 'stability', 'elbow_fwd', 'knee', 'plateau',
         'gap_step', 'fixed')
SPEARMAN_BAR = 0.30
P_BAR = 0.05


def consensus_ranking(V):
    """Label-free feature ranking: |corr| with the full-pool L-SML consensus.

    Same construction a7/D2 use, so the K rules are validated on the ranking they
    would actually be applied to.
    """
    p = V.shape[1]
    fused, _ = lsml_continuous(*[V[:, c] for c in range(p)])
    y = np.asarray(fused, dtype=np.float64)
    yc = y - y.mean()
    Vc = V - V.mean(0, keepdims=True)
    den = np.linalg.norm(Vc, axis=0) * np.linalg.norm(yc)
    corr = np.abs((Vc * yc[:, None]).sum(0) / np.maximum(den, 1e-12))
    corr[~np.isfinite(corr)] = 0.0
    return list(np.argsort(corr)[::-1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = load_cells()
    ok, macro_g6 = assert_good6(cells)
    if not ok:
        print("FAIL — refusing to report (SPEC_gap_ladder §8).")
        sys.exit(1)

    cd, rankings, rows = {}, {}, []
    for ck, cell in sorted(cells.items()):
        V = cell["V"]
        rank = consensus_ranking(V)
        cd[ck] = {"V": V, "labels": cell["labels"], "group": GROUP.get(ck, "?")}
        rankings[ck] = rank
        k_star, auc_star, _ = oracle_k(V, rank, cell["labels"])
        row = {"cell": ck, "group": GROUP.get(ck, "?"), "p_pool": V.shape[1],
               "oracle_k": k_star, "oracle_auc": round(float(auc_star), 4)}
        for r in RULES:
            row[f"k_{r}"] = predict_k(V, rank, rule=r)
        for r in ("eff_rank", "mp_floor", "stability"):
            row[f"raw_{r}"] = round(float(raw_k(V, rank, r)), 3)
        rows.append(row)
        print(f"  {ck:34s} oracle_k={k_star:2d} eff_rank={row['k_eff_rank']:2d} "
              f"(raw {row['raw_eff_rank']:5.2f})  mp={row['k_mp_floor']:2d}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "k_rule_validation.csv"), index=False)

    # per-rule agreement with oracle-K + macro AUROC at the predicted K
    _, summary = validate(cd, rankings, rules=RULES)
    sm = pd.DataFrame(summary)
    fixed_macro = float(sm.loc[sm["rule"] == "fixed", "macro_auc"].iloc[0])

    def verdict(r):
        rs, p, m = r["spearman_r"], r["p_value"], r["macro_auc"]
        if rs is None or p is None or m is None:
            return "REFUTED (degenerate)"
        if rs >= SPEARMAN_BAR and p < P_BAR and m >= fixed_macro:
            return "ADOPT"
        return "REFUTED"

    sm["verdict"] = sm.apply(verdict, axis=1)
    sm["const_k"] = [df[f"k_{r}"].nunique() == 1 for r in sm["rule"]]
    sm.to_csv(os.path.join(OUT_DIR, "k_rule_validation_rules.csv"), index=False)

    print(f"\n{'='*92}")
    print(f"H2 — label-free K rules vs ORACLE-K   (bar: Spearman >= {SPEARMAN_BAR}, "
          f"p < {P_BAR}, macro >= fixed-K {fixed_macro:.4f})")
    print(f"{'='*92}")
    print(sm[["rule", "spearman_r", "p_value", "mean_abs_dk", "median_k",
              "macro_auc", "oracle_macro_auc", "const_k", "verdict"]]
          .to_string(index=False))
    print(f"\noracle_k: min {df.oracle_k.min()}  median "
          f"{int(df.oracle_k.median())}  max {df.oracle_k.max()}")
    adopted = sm[sm.verdict == "ADOPT"]["rule"].tolist()
    print(f"\nADOPTED: {adopted if adopted else 'NONE — every rule REFUTED'}")
    print(f"\nwrote {os.path.join(OUT_DIR, 'k_rule_validation.csv')}")
    print(f"wrote {os.path.join(OUT_DIR, 'k_rule_validation_rules.csv')}")


if __name__ == "__main__":
    main()
