"""
exp11_posthoc_controls.py — the six measurements that were NOT pre-registered.

WHY THIS FILE EXISTS
--------------------
Phase 1 pre-registered four ceilings (which features get kept, how the two directions
blend, the three hard-coded constants, the label-variance guess) and measured them in
`exp10_channel_ceilings.py`. Two review passes then asked for six more measurements that
nobody had planned. Those six were originally computed in throwaway session scratch, which
means the single most consequential number in the phase — the oracle sign ceiling, which
retired an entire line of work — was not reproducible from the repo.

That is fixed here. Everything in this file is POST-HOC: chosen after seeing the data.
That is not disqualifying (an oracle is a bounding measurement, and the sign ceiling
overturned the section that commissioned it rather than flattering it), but it has a
different evidential status from a pre-registered number and must be labelled wherever it
is quoted.

THE SIX
-------
  sign_ceiling     What is getting EVERY feature's direction right worth? The plan never
                   priced this channel — it asserted the mask/mixing/g2 triple was "the
                   whole space" and ring-fenced direction into a separate line on the
                   strength of a GLOBAL orientation result. One script would have retired
                   that line up front. Answer: -0.06pp, 17/24 cells exactly unchanged.
  mix_splithalf    The blend ceiling under the SAME protocol as the feature-drop ceiling.
                   Reported in-sample it is +0.49pp; held out across rows it is +0.25pp.
                   Comparing an in-sample ceiling against a split-half one is not a
                   comparison.
  random_control   Size-matched RANDOM subsets, label-free. The pre-registered null
                   (selection on shuffled labels) lands BELOW this, i.e. it is an easier
                   bar than chance. This is the honest floor.
  shallow_search   Best of N random size-matched subsets scored on the selection half —
                   how much of the gain a shallow label-guided search recovers.
  loco_transfer    Does the good feature set transfer BETWEEN test sets? Build a keep-set
                   from the other 23 cells' oracle choices, apply it to the held-out one.
  rho_ranking      Does U-PCR's own ranking find the good features? Overlap between the
                   oracle's set and the top-k by |rho_hat|, against a random-overlap
                   baseline; plus top-k-by-rho masks at every size.

Run:  python scripts/upcr_study/exp11_posthoc_controls.py [--only sign_ceiling,...]
Out:  results/upcr_study/11_posthoc_controls/
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from exp10_channel_ceilings import (                                # noqa: E402
    FIT, N_SPLITS, derive_cell, derived_arm_gate, fit_cols, _zsc, _all_fields,
)
from spectral_utils.subset_sweep import ALL_SIGNS                   # noqa: E402
from spectral_utils.upcr import upcr_fit                            # noqa: E402

ORACLE_CSV = "r2_oracle_mask.csv"      # written by exp10 --runs r2
N_RANDOM = 25                          # random subsets per split for the floor
N_SHALLOW = 20                         # budget for the shallow label search


# ---------------------------------------------------------------------------
def oracle_signs(V_raw, y):
    """The direction the labels say each feature reads in."""
    return np.array([1.0 if roc_auc_score(y, V_raw[:, j]) >= 0.5 else -1.0
                     for j in range(V_raw.shape[1])])


def run_sign_ceiling(cells, dcells):
    """THE measurement the plan omitted. Give every feature its true direction, refit,
    rescore. If this is ~0 then no sign estimator, however good, can pay."""
    rows = []
    for ck, cell in cells.items():
        hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
        V_raw = np.asarray(cell["V"], dtype=float) * hand
        y = cell["labels"]
        auc_dep, _ = fit_cols(dcells[ck], range(V_raw.shape[1]))
        pol = oracle_signs(V_raw, y)
        oracle_cell = dict(cell)
        oracle_cell["V"] = V_raw * pol
        auc_orc, _ = fit_cols(oracle_cell, range(V_raw.shape[1]))
        rows.append({"cell": ck, "auroc_deployed": auc_dep,
                     "auroc_oracle_sign": auc_orc,
                     "delta_pp": (auc_orc - auc_dep) * 100.0,
                     "unchanged": bool(abs(auc_orc - auc_dep) < 1e-12)})
        print(f"  {S.plain_cell(ck)[:34]:34s} {auc_dep:.4f} -> {auc_orc:.4f} "
              f"({rows[-1]['delta_pp']:+.2f}pp)"
              f"{'   [exactly unchanged]' if rows[-1]['unchanged'] else ''}", flush=True)
    return rows


def run_mix_splithalf(cells, dcells, seed=0):
    """The blend ceiling under the feature-drop ceiling's own protocol: choose the angle
    on half the rows, score on the other half."""
    rng = np.random.default_rng(seed)
    rows = []
    thetas = np.linspace(0.0, np.pi, 721, endpoint=False)
    for ck, cell in dcells.items():
        V0, y = cells[ck]["V"], cell["labels"]
        n, m = cell["V"].shape
        hand = np.array([ALL_SIGNS.get(f, +1) for f in cells[ck]["pool"]], dtype=float)
        b_dep, b_orc = [], []
        for _ in range(N_SPLITS):
            idx = rng.permutation(n)
            a_idx, b_idx = idx[: n // 2], idx[n // 2:]
            if len(np.unique(y[a_idx])) < 2 or len(np.unique(y[b_idx])) < 2:
                continue
            raw_a = np.column_stack([_zsc(V0[a_idx, j]) for j in range(m)]) * hand
            raw_b = np.column_stack([_zsc(V0[b_idx, j]) for j in range(m)]) * hand
            try:
                pol = np.sign(upcr_fit(raw_a.T, **FIT).rho_hat_full)
            except Exception:
                continue
            pol[pol == 0] = 1.0
            ca = {"V": raw_a * pol, "anchor": _zsc(cell["anchor"][a_idx]),
                  "labels": y[a_idx]}
            cb = {"V": raw_b * pol, "anchor": _zsc(cell["anchor"][b_idx]),
                  "labels": y[b_idx]}
            auc_a, res_a = fit_cols(ca, range(m))
            auc_b, _ = fit_cols(cb, range(m))
            if res_a is None or res_a.used_simple_average:
                continue
            keep = res_a.keep
            nk = int(keep.sum())
            if nk < 2:
                continue
            Fa, Fb = ca["V"][:, keep].T, cb["V"][:, keep].T
            Ca = (Fa @ Fa.T) / Fa.shape[1]
            ev, evec = eigh(Ca, subset_by_index=[nk - 2, nk - 1])
            v1, v2 = evec[:, ::-1][:, 0], evec[:, ::-1][:, 1]
            # choose the angle on half A, score it on half B
            aucs_a = [S.auroc_from_score(ca, (np.cos(t) * v1 + np.sin(t) * v2) @ Fa)
                      for t in thetas]
            t_best = thetas[int(np.nanargmax(aucs_a))]
            b_orc.append(S.auroc_from_score(
                cb, (np.cos(t_best) * v1 + np.sin(t_best) * v2) @ Fb))
            b_dep.append(auc_b)
        if not b_orc:
            continue
        rows.append({"cell": ck, "auroc_splithalf_deployed": float(np.nanmean(b_dep)),
                     "auroc_splithalf_best_angle": float(np.nanmean(b_orc)),
                     "delta_pp": (float(np.nanmean(b_orc)) -
                                  float(np.nanmean(b_dep))) * 100.0,
                     "n_splits": len(b_orc)})
        print(f"  {S.plain_cell(ck)[:34]:34s} {rows[-1]['auroc_splithalf_deployed']:.4f}"
              f" -> {rows[-1]['auroc_splithalf_best_angle']:.4f} "
              f"({rows[-1]['delta_pp']:+.2f}pp)", flush=True)
    return rows


def _read_oracle_cols(out_root, cells):
    import csv
    p = os.path.join(out_root, "10_channel_ceilings", ORACLE_CSV)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} — run exp10_channel_ceilings.py --runs r2 first")
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        names = [f for f in r["oracle_cols"].split("|") if f]
        out[r["cell"]] = names
    return out


def run_random_control(cells, dcells, oracle_sets, seed=0):
    """The honest floor: subsets of the SAME SIZE the oracle chose, picked at random,
    scored on held-out rows. Separates 'which features' from 'how many'."""
    rng = np.random.default_rng(seed)
    rows = []
    for ck, cell in dcells.items():
        V0, y = cells[ck]["V"], cell["labels"]
        n, m = cell["V"].shape
        k = max(3, len(oracle_sets.get(ck, [])) or 10)
        hand = np.array([ALL_SIGNS.get(f, +1) for f in cells[ck]["pool"]], dtype=float)
        rnd, dep, shallow = [], [], []
        for _ in range(N_SPLITS):
            idx = rng.permutation(n)
            a_idx, b_idx = idx[: n // 2], idx[n // 2:]
            if len(np.unique(y[a_idx])) < 2 or len(np.unique(y[b_idx])) < 2:
                continue
            raw_a = np.column_stack([_zsc(V0[a_idx, j]) for j in range(m)]) * hand
            raw_b = np.column_stack([_zsc(V0[b_idx, j]) for j in range(m)]) * hand
            try:
                pol = np.sign(upcr_fit(raw_a.T, **FIT).rho_hat_full)
            except Exception:
                continue
            pol[pol == 0] = 1.0
            ca = {"V": raw_a * pol, "anchor": _zsc(cell["anchor"][a_idx]),
                  "labels": y[a_idx]}
            cb = {"V": raw_b * pol, "anchor": _zsc(cell["anchor"][b_idx]),
                  "labels": y[b_idx]}
            dep.append(fit_cols(cb, range(m))[0])
            draws = [sorted(rng.choice(m, k, replace=False)) for _ in range(N_RANDOM)]
            rnd += [fit_cols(cb, c, exclusion=False)[0] for c in draws[:5]]
            # shallow label search: pick the best of N on half A, score on half B
            cand = draws[:N_SHALLOW]
            sa = [fit_cols(ca, c, exclusion=False)[0] for c in cand]
            best = cand[int(np.nanargmax(sa))]
            shallow.append(fit_cols(cb, best, exclusion=False)[0])
        if not dep:
            continue
        rows.append({
            "cell": ck, "k": k,
            "auroc_splithalf_deployed": float(np.nanmean(dep)),
            "auroc_random_same_size": float(np.nanmean(rnd)),
            "auroc_shallow_label_search": float(np.nanmean(shallow)),
            "random_delta_pp": (float(np.nanmean(rnd)) - float(np.nanmean(dep))) * 100.0,
            "shallow_delta_pp": (float(np.nanmean(shallow)) -
                                 float(np.nanmean(dep))) * 100.0,
        })
        print(f"  {S.plain_cell(ck)[:34]:34s} k={k:2d} deployed "
              f"{rows[-1]['auroc_splithalf_deployed']:.4f} | random "
              f"{rows[-1]['random_delta_pp']:+.2f}pp | shallow "
              f"{rows[-1]['shallow_delta_pp']:+.2f}pp", flush=True)
    return rows


def run_loco_transfer(cells, dcells, oracle_sets):
    """Does the good feature set transfer between test sets? Build the keep-set from the
    OTHER 23 cells' oracle choices, apply it to the held-out one."""
    from collections import Counter
    rows = []
    for ck, cell in dcells.items():
        others = Counter()
        for k2, names in oracle_sets.items():
            if k2 != ck:
                others.update(names)
        pool = cell["pool"]
        ranked = [f for f, _ in others.most_common() if f in pool]
        own_k = max(3, len(oracle_sets.get(ck, [])) or 10)
        auc_dep, _ = fit_cols(cell, range(len(pool)))
        out = {"cell": ck, "auroc_deployed": auc_dep}
        for label, k in (("top10", 10), ("own_size", own_k)):
            cols = [pool.index(f) for f in ranked[:k]]
            a, _ = fit_cols(cell, cols, exclusion=False)
            out[f"auroc_loco_{label}"] = a
            out[f"delta_loco_{label}_pp"] = (a - auc_dep) * 100.0
        rows.append(out)
        print(f"  {S.plain_cell(ck)[:34]:34s} deployed {auc_dep:.4f} | LOCO top10 "
              f"{out['delta_loco_top10_pp']:+.2f}pp | LOCO own-size "
              f"{out['delta_loco_own_size_pp']:+.2f}pp", flush=True)
    return rows


def run_rho_ranking(cells, dcells, oracle_sets, seed=0):
    """Does U-PCR's own ranking find the good features? If the overlap sits at the
    random baseline, no better ESTIMATE of that quantity can help — the quantity itself
    is the wrong one."""
    rng = np.random.default_rng(seed)
    rows = []
    for ck, cell in dcells.items():
        pool = cell["pool"]
        m = len(pool)
        auc_dep, res = fit_cols(cell, range(m))
        if res is None:
            continue
        want = set(pool.index(f) for f in oracle_sets.get(ck, []) if f in pool)
        k = max(len(want), 1)
        order = np.argsort(-np.abs(res.rho_hat_full))
        topk = set(int(j) for j in order[:k])
        overlap = len(want & topk) / k if k else float("nan")
        # random-overlap baseline for the same k, same m
        base = float(np.mean([len(want & set(rng.choice(m, k, replace=False))) / k
                              for _ in range(2000)])) if k else float("nan")
        out = {"cell": ck, "k": k, "overlap_with_rho_topk": overlap,
               "random_overlap_baseline": base, "auroc_deployed": auc_dep}
        for kk in (6, 8, 10, 12, 14, 16):
            cols = [int(j) for j in order[:kk]]
            a, _ = fit_cols(cell, cols, exclusion=False)
            out[f"auroc_rho_top{kk}"] = a
            out[f"delta_rho_top{kk}_pp"] = (a - auc_dep) * 100.0
        rows.append(out)
        print(f"  {S.plain_cell(ck)[:34]:34s} k={k:2d} overlap {overlap:.3f} "
              f"(random {base:.3f})  top10 {out['delta_rho_top10_pp']:+.2f}pp", flush=True)
    return rows


# ---------------------------------------------------------------------------
def paired(rows, a, b, name):
    x = np.array([r[a] for r in rows], float)
    y = np.array([r[b] for r in rows], float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = (y - x) * 100.0
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(20000)]
    p = float(wilcoxon(x, y).pvalue) if len(d) >= 5 and np.any(d != 0) else float("nan")
    out = {"mean_delta_pp": float(d.mean()), "median_delta_pp": float(np.median(d)),
           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
           "n": int(len(d)), "p": p}
    print(f"\n  {name}: {out['mean_delta_pp']:+.2f}pp mean / "
          f"{out['median_delta_pp']:+.2f}pp median, 95% CI "
          f"[{out['ci95'][0]:+.2f}, {out['ci95'][1]:+.2f}], "
          f"{out['wins']}W/{out['losses']}L over {out['n']}, p={out['p']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated: sign_ceiling,mix_splithalf,random_control,"
                         "loco_transfer,rho_ranking")
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else {
        "sign_ceiling", "mix_splithalf", "random_control", "loco_transfer",
        "rho_ranking"}

    out = S.outdir("11_posthoc_controls")
    cells = S.load()
    S.validity_check(cells)
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    derived_arm_gate(dcells)
    oracle_sets = _read_oracle_cols(S.OUT_ROOT, cells)

    summary = {"n_cells": len(cells),
               "PROVENANCE": "ALL POST-HOC — chosen after seeing the data. Label as such "
                             "wherever quoted; do not table these beside the "
                             "pre-registered ceilings without a marker."}

    if "sign_ceiling" in want:
        print("\n=== What is getting EVERY feature's direction right worth? ===")
        rows = run_sign_ceiling(cells, dcells)
        S.save_csv(os.path.join(out, "sign_ceiling.csv"), rows, _all_fields(rows))
        summary["sign_ceiling"] = paired(rows, "auroc_deployed", "auroc_oracle_sign",
                                         "oracle sign vs deployed")
        summary["sign_ceiling"]["n_cells_exactly_unchanged"] = int(
            sum(1 for r in rows if r["unchanged"]))

    if "mix_splithalf" in want:
        print("\n=== The blend ceiling, held out across rows ===")
        rows = run_mix_splithalf(cells, dcells)
        S.save_csv(os.path.join(out, "mix_splithalf.csv"), rows, _all_fields(rows))
        summary["mix_splithalf"] = paired(rows, "auroc_splithalf_deployed",
                                          "auroc_splithalf_best_angle",
                                          "best angle (split-half) vs deployed")

    if "random_control" in want:
        print("\n=== The honest floor: random subsets of the oracle's own size ===")
        rows = run_random_control(cells, dcells, oracle_sets)
        S.save_csv(os.path.join(out, "random_control.csv"), rows, _all_fields(rows))
        summary["random_same_size"] = paired(rows, "auroc_splithalf_deployed",
                                             "auroc_random_same_size",
                                             "random same-size vs deployed")
        summary["shallow_label_search"] = paired(rows, "auroc_splithalf_deployed",
                                                 "auroc_shallow_label_search",
                                                 "shallow label search vs deployed")

    if "loco_transfer" in want:
        print("\n=== Does the good feature set transfer between test sets? ===")
        rows = run_loco_transfer(cells, dcells, oracle_sets)
        S.save_csv(os.path.join(out, "loco_transfer.csv"), rows, _all_fields(rows))
        summary["loco_top10"] = paired(rows, "auroc_deployed", "auroc_loco_top10",
                                       "LOCO top-10 vs deployed")
        summary["loco_own_size"] = paired(rows, "auroc_deployed", "auroc_loco_own_size",
                                          "LOCO at own size vs deployed")

    if "rho_ranking" in want:
        print("\n=== Does U-PCR's own ranking find the good features? ===")
        rows = run_rho_ranking(cells, dcells, oracle_sets)
        S.save_csv(os.path.join(out, "rho_ranking.csv"), rows, _all_fields(rows))
        ov = np.array([r["overlap_with_rho_topk"] for r in rows], float)
        bs = np.array([r["random_overlap_baseline"] for r in rows], float)
        summary["rho_ranking"] = {
            "mean_overlap": float(np.nanmean(ov)),
            "mean_random_baseline": float(np.nanmean(bs)),
            "beats_random": bool(np.nanmean(ov) > np.nanmean(bs)),
            "top_k_deltas_pp": {f"top{k}": float(np.nanmean(
                [r[f"delta_rho_top{k}_pp"] for r in rows]))
                for k in (6, 8, 10, 12, 14, 16)},
        }
        print(f"\n  overlap with the |rho| ranking {np.nanmean(ov):.3f} vs random "
              f"{np.nanmean(bs):.3f} -> "
              f"{'above' if np.nanmean(ov) > np.nanmean(bs) else 'AT OR BELOW'} chance")

    S.save_json(os.path.join(out, "summary.json"), summary)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
