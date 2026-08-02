#!/usr/bin/env python
"""
stage3_ensemble.py — do the repaired views help the FUSION, and why / why not.

THE QUESTION THIS ANSWERS
-------------------------
The single-view unit test settled that the fold works on the view itself: on the
27 (cell, feature) pairs with >= 5pp of measured non-monotone headroom the
symmetric family recovers ~73% of it, and `sciq_llama8b/pe_mean` goes 0.434 ->
0.699. That is a fact about one column. It says nothing about the fused score,
and the whole point of Step 217 was that the two came apart.

Step 217 came apart for a MECHANICAL reason that is now identified, so this run
is built around it rather than repeating it:

  ARM B (U-PCR). `upcr.py:287-293` keeps a view only if
      rho_hat >= min_frac*var_y  AND  rho_hat >= rho_max/exclude_frac
  and rho_hat is estimated from LINEAR covariance. A U-shaped view has
  rho_hat ~ 0, is dropped, and after that its contents cannot move the fused
  score by any amount -- which is exactly why Step 217's arm B was BIT-IDENTICAL
  on 4 of the 5 cells carrying the shapes. That exclusion is not only the
  blocker, it is the CHANNEL: a fold that monotonises the view gives it a real
  rho_hat and lets it back in. `keep` flipping False -> True is therefore the
  sensitive endpoint, and it is recorded per view whether or not the macro moves.

  ARM A (L-SML over DUFS). Step 217 froze the selected view set, and 34 of the 38
  candidates are views DUFS does not select. Freezing the selection removes the
  channel by construction, which is where the 19/24 exact zeros came from. Here
  `dufs_pf` is RE-RUN on the transformed matrix (bit-exact reproduction of
  `a2_groupfs__c46.csv` gated in `dufs_pf.py`), so "does repairing a view get it
  selected" is a measurable event. The frozen arm is kept alongside, labelled, so
  the difference between the two is visible rather than argued.

THE SELECTION RULES, AND WHICH ONE IS THE HONEST NUMBER
------------------------------------------------------
Which views to fold has to be decided without labels, and that is the weak link:
the consensus detector correlates with true headroom at only Spearman +0.309.
Four rules are run, and they are NOT interchangeable:

  oracle_headroom   folds the 34 views selected BY MEASURED HEADROOM, i.e. using
                    the answer key to choose. A CEILING, not a deployable result.
  free_adaptive     folds the views the consensus detector flags, with the
                    transform picked by its AUROC against the PSEUDO-label.
                    Fully label-free -> this is the deployable number.
  free_fixed        same views, one transform for all of them (`mode_centre`).
                    One decision for the whole method instead of 34.
  all_fold          folds every view in the pool. The control that shows what
                    indiscriminate folding costs.
  placebo_k         folds a random set of CONTROL views (headroom < 3pp), matched
                    in count and in transform mix. Adding and reshaping columns
                    perturbs the selector and the rho vector all by itself; if
                    the placebo also "gains", a positive elsewhere is machinery,
                    not shape. This is the gate that makes the result falsifiable.

MODES. `replace` swaps the parent column for its fold. `add_orth` keeps the
parent and appends the fold with its rank-linear component projected out --
plain `add` is theoretically inadmissible for arm B, because U-PCR's Eq. 15/21
estimate rho from cov(f_i,f_j) = rho_i rho_j var(y) ASSUMING conditional
independence, and a deterministic function of a column already in the matrix
biases the whole rho vector rather than just its own entry (Step 217 measured
induced |rho| up to 1.000).

Uncertainty is quoted as a PAIRED CELL-LEVEL BOOTSTRAP CI, not the Wilcoxon p:
the macro has 24 independent units and that is what governs it.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm, spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (OUT, GROUP, INSCOPE, load_cells_cached, pct, kde_modes,   # noqa: E402
                    lsml_score, upcr_score, n_min)
from unit_test_transforms import cv_auc                                       # noqa: E402
from transform_selection import (consensus_score, consensus_pseudo_labels,    # noqa: E402
                                 cv_consensus_map, t_squared, t_dist_median,
                                 t_abs_rank, t_centre, CGRID)
from dufs_pf import bench_domains, dufs_pf_cols                               # noqa: E402
from spectral_utils.subset_sweep import GOOD_6                                # noqa: E402
from sklearn.metrics import roc_auc_score                                     # noqa: E402

SEL_JSON = os.path.join(OUT, "transform_selection.json")
CONS_THR = 0.03          # the consensus-detector threshold, from its precision curve
MIN_GAIN = 0.02          # a transform must beat the raw view by 2pp to be adopted
N_PLACEBO = 3
BOOT = 5000
TIER_A = ("squared", "dist_median", "abs_rank", "mode_centre",
          "consensus_centre", "consensus_map")


# ══════════════════════════════════════════════════════════════════════════════
# Rebuilding a named transform (the selection JSON stores the name, not the vector)
# ══════════════════════════════════════════════════════════════════════════════
def build_options(x, y, V, j, names=TIER_A):
    """{name: vector} for the label-free menu, plus the pseudo-label it is scored on.

    Every one of these is computable without an answer key. `consensus_centre` and
    `consensus_map` use the mean of the OTHER views as a stand-in label; that is a
    circularity worth naming -- reshaping a view to agree with the consensus makes
    it more redundant, which is the opposite of what L-SML and U-PCR want (both
    assume conditional independence given y). It is included because it is
    deployable and measured, not because it is safe."""
    u = pct(x)
    cons = consensus_score(V, j)
    pseudo = consensus_pseudo_labels(cons) if cons is not None else None
    out = {"identity": x}
    if "squared" in names:
        out["squared"] = t_squared(x, u)
    if "dist_median" in names:
        out["dist_median"] = t_dist_median(x, u)
    if "abs_rank" in names:
        out["abs_rank"] = t_abs_rank(x, u)
    _, _, mode_pct = kde_modes(x)
    if "mode_centre" in names and np.isfinite(mode_pct):
        out["mode_centre"] = t_centre(x, u, c=float(mode_pct))
    if pseudo is not None:
        if "consensus_centre" in names:
            cur = np.array([cv_auc(t_centre(x, u, c=float(c)), pseudo) for c in CGRID])
            if np.isfinite(cur).any():
                out["consensus_centre"] = t_centre(
                    x, u, c=float(CGRID[int(np.nanargmax(cur))]))
        if "consensus_map" in names:
            out["consensus_map"] = cv_consensus_map(x, pseudo)
    return out, pseudo


def orthogonalise(t, x):
    """t with its rank-linear component removed, so `add_orth` spans the same 2-D
    space as `add` while leaving the parent's rank information un-duplicated."""
    r = pct(x)
    r = (r - r.mean()) / max(r.std(), 1e-12)
    t = np.asarray(t, float)
    return t - (float(t @ r) / max(float(r @ r), 1e-12)) * r


def zs(v):
    v = np.asarray(v, float)
    s = float(np.std(v))
    return (v - np.mean(v)) / s if s > 1e-12 else np.zeros_like(v)


def apply_config(cell, cfg, mode):
    """(V, pool, induced_rho) with {feature: transform_name} applied.

    Columns are z-scored after transformation so the fusion sees the scale it sees
    on the canonical path (`apply_config` in nonmono_transform_bench.py:437)."""
    V, pool = cell["V"], list(cell["pool"])
    cols = [V[:, k] for k in range(V.shape[1])]
    induced, added = {}, {}
    for f, vec in sorted(cfg.items()):
        if f not in pool:
            continue
        j = pool.index(f)
        x = V[:, j]
        t = zs(vec)
        if mode == "replace":
            cols[j] = t
        else:
            t = zs(orthogonalise(t, x))
            name = f"{f}__t"
            cols.append(t)
            pool.append(name)
            added[f] = name
            induced[f] = float(abs(spearmanr(x, t).statistic))
    return np.column_stack(cols), pool, induced, added


# ══════════════════════════════════════════════════════════════════════════════
# The arms
# ══════════════════════════════════════════════════════════════════════════════
def score_arms(cell, V, pool, dufs_frozen_names, domain, want_dufs=True):
    """Every arm on one (V, pool). Returns metrics + the arm-B fit objects."""
    y, anchor = cell["labels"], cell["anchor"]
    out = {}

    s, res, rho1 = upcr_score(V, pool, anchor)
    out["upcr"] = float(roc_auc_score(y, s))
    out["_upcr_rho"] = {f: float(rho1[i]) for i, f in enumerate(pool)}
    out["_upcr_keep"] = {f: bool(res.keep[i]) for i, f in enumerate(pool)}
    out["_upcr_w"] = {f: float(res.w[i]) for i, f in enumerate(pool)}
    out["upcr_n_kept"] = int(res.keep.sum())

    frozen = [i for i, f in enumerate(pool) if f in set(dufs_frozen_names)]
    sc = lsml_score(V, anchor, frozen)
    out["lsml_dufs_frozen"] = float(roc_auc_score(y, sc)) if sc is not None else np.nan

    g6 = [i for i, f in enumerate(pool) if f in set(GOOD_6)]
    sc = lsml_score(V, anchor, g6)
    out["lsml_good6"] = float(roc_auc_score(y, sc)) if sc is not None else np.nan

    if want_dufs:
        sel, _ = dufs_pf_cols(V, cell["key"], domain)
        sc = lsml_score(V, anchor, sel)
        out["lsml_dufs"] = float(roc_auc_score(y, sc)) if sc is not None else np.nan
        out["_dufs_names"] = [pool[i] for i in sel]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# One cell, every configuration
# ══════════════════════════════════════════════════════════════════════════════
def run_cell(payload):
    ck, cell, panels, domain, dufs_frozen, want_dufs, seed = payload
    cell = dict(cell, key=ck)
    V0, pool = cell["V"], cell["pool"]
    y = cell["labels"]
    rng = np.random.default_rng(abs(hash(ck)) % (2 ** 31) + seed)

    # ── per-view menu, once ───────────────────────────────────────────────────
    menu, pseudo_auc, true_auc = {}, {}, {}
    for f in {p["feature"] for p in panels}:
        if f not in pool:
            continue
        j = pool.index(f)
        opts, pseudo = build_options(V0[:, j], y, V0, j)
        menu[f] = opts
        if pseudo is not None:
            pseudo_auc[f] = {k: cv_auc(v, pseudo) for k, v in opts.items()}
        true_auc[f] = {k: cv_auc(v, y) for k, v in opts.items()}

    by_feat = {p["feature"]: p for p in panels}
    cands = [p["feature"] for p in panels
             if p["is_candidate"] and p["chosen"] != "identity"]
    ctrls = [p["feature"] for p in panels if not p["is_candidate"]]

    # ── the four selection rules ──────────────────────────────────────────────
    sels = {}
    sels["oracle_headroom"] = {f: by_feat[f]["chosen"] for f in cands
                               if by_feat[f]["chosen"] in menu.get(f, {})}

    flagged = [p["feature"] for p in panels
               if p["cons_gain"] != "" and float(p["cons_gain"]) > CONS_THR
               and p["feature"] in menu]
    adaptive = {}
    for f in flagged:
        pa = pseudo_auc.get(f)
        if not pa:
            continue
        free = {k: v for k, v in pa.items() if k != "identity" and np.isfinite(v)}
        if not free:
            continue
        best = max(free, key=free.get)
        if free[best] - pa["identity"] >= MIN_GAIN:
            adaptive[f] = best
    sels["free_adaptive"] = adaptive
    sels["free_fixed"] = {f: "mode_centre" for f in flagged
                          if "mode_centre" in menu.get(f, {})}

    # `abs_rank` needs no fitting of any kind, so the "fold everything" control can
    # cover the WHOLE pool rather than only the views under test — which is what
    # makes it a fair statement of what indiscriminate folding costs.
    for f in pool:
        menu.setdefault(f, {})["abs_rank"] = t_abs_rank(V0[:, pool.index(f)],
                                                        pct(V0[:, pool.index(f)]))
    sels["all_fold"] = {f: "abs_rank" for f in pool}

    mix = [by_feat[f]["chosen"] for f in cands]
    for b in range(N_PLACEBO):
        pool_ctrl = [f for f in ctrls if f in menu]
        k = min(len(mix), len(pool_ctrl))
        if k == 0:
            sels[f"placebo_{b}"] = {}
            continue
        pick = rng.choice(pool_ctrl, size=k, replace=False)
        names = rng.permutation(mix)[:k]
        sels[f"placebo_{b}"] = {f: (t if t in menu[f] else "abs_rank")
                                for f, t in zip(pick, names)}

    # ── score ─────────────────────────────────────────────────────────────────
    base = score_arms(cell, V0, pool, dufs_frozen, domain, want_dufs)
    rows, mech = [], []
    for sname, sel in sels.items():
        for mode in ("replace", "add_orth"):
            if sname == "all_fold" and mode == "add_orth":
                continue                       # 30 extra columns is not a method
            cfg = {f: menu[f][t] for f, t in sel.items() if t in menu.get(f, {})}
            if not cfg:
                for arm in ("upcr", "lsml_dufs_frozen", "lsml_good6",
                            *(("lsml_dufs",) if want_dufs else ())):
                    rows.append(dict(cell=ck, domain=GROUP.get(ck, "?"), sel=sname,
                                     mode=mode, arm=arm, n_views=0,
                                     auroc_base=base[arm], auroc_cfg=base[arm],
                                     delta_pp=0.0))
                continue
            V, pl, induced, added = apply_config(cell, cfg, mode)
            cur = score_arms(cell, V, pl, dufs_frozen, domain, want_dufs)
            for arm in ("upcr", "lsml_dufs_frozen", "lsml_good6",
                        *(("lsml_dufs",) if want_dufs else ())):
                rows.append(dict(
                    cell=ck, domain=GROUP.get(ck, "?"), sel=sname, mode=mode,
                    arm=arm, n_views=len(cfg),
                    auroc_base=round(float(base[arm]), 5),
                    auroc_cfg=round(float(cur[arm]), 5),
                    delta_pp=round(float((cur[arm] - base[arm]) * 100), 4)))
            for f, t in sel.items():
                if f not in cfg or f not in true_auc:
                    continue          # `all_fold` covers pool views never unit-tested
                key = added.get(f, f)
                mech.append(dict(
                    cell=ck, domain=GROUP.get(ck, "?"), sel=sname, mode=mode,
                    feature=f, transform=t, n_min=n_min(y),
                    headroom_pp=by_feat.get(f, {}).get("headroom_pp", np.nan),
                    sv_auc_base=round(float(true_auc[f]["identity"]), 4),
                    sv_auc_cfg=round(float(true_auc[f][t]), 4),
                    rho_base=round(base["_upcr_rho"].get(f, np.nan), 5),
                    rho_cfg=round(cur["_upcr_rho"].get(key, np.nan), 5),
                    keep_base=int(base["_upcr_keep"].get(f, False)),
                    keep_cfg=int(cur["_upcr_keep"].get(key, False)),
                    w_cfg=round(cur["_upcr_w"].get(key, np.nan), 6),
                    dufs_base=int(f in base.get("_dufs_names", [])),
                    dufs_cfg=int(key in cur.get("_dufs_names", [])),
                    induced_rho=round(induced.get(f, np.nan), 4)))
    return ck, rows, mech, base


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════
def boot_ci(d, b=BOOT, seed=0):
    """Paired cell-level bootstrap on the macro delta. The 24 cells are the
    independent units; resampling rows would understate the uncertainty by an
    order of magnitude."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(rows, arm, sel, mode):
    r = [x for x in rows if x["arm"] == arm and x["sel"] == sel and x["mode"] == mode]
    if not r:
        return None
    d = np.array([x["delta_pp"] for x in r], float)
    moved = int(np.sum(np.abs(d) > 1e-9))
    lo, hi = boot_ci(d)
    try:
        p = float(wilcoxon(d).pvalue) if moved else 1.0
    except ValueError:
        p = 1.0
    qa = np.array([x["delta_pp"] for x in r if x["domain"] == "QA"], float)
    mt = np.array([x["delta_pp"] for x in r if x["domain"] == "math"], float)
    return dict(arm=arm, sel=sel, mode=mode, n_cells=len(r), n_moved=moved,
                macro_delta_pp=round(float(d.mean()), 3),
                ci_lo=round(lo, 3), ci_hi=round(hi, 3), wilcoxon_p=round(p, 4),
                wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()),
                worst_pp=round(float(d.min()), 3), best_pp=round(float(d.max()), 3),
                qa_delta_pp=round(float(qa.mean()), 3) if len(qa) else float("nan"),
                math_delta_pp=round(float(mt.mean()), 3) if len(mt) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-dufs", action="store_true",
                    help="skip the arm-A DUFS re-run (fast smoke path)")
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()

    cells = load_cells_cached()
    with open(SEL_JSON, encoding="utf-8") as fh:
        sel_json = json.load(fh)
    by_cell = {}
    for p in sel_json["panels"]:
        by_cell.setdefault(p["cell"], []).append(p)
    dom = bench_domains()
    frozen = {}
    import csv as _csv
    with open(os.path.join(os.path.dirname(OUT), "selector_bench",
                           "a2_groupfs__c46.csv"), newline="", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if r["variant"] == "a2.dufs_pf":
                frozen[r["cell"]] = [f for f in r["chosen"].split("|") if f]

    keys = args.cells or [c for c in INSCOPE if c in by_cell]
    payloads = [(ck, cells[ck], by_cell[ck], dom[ck], frozen[ck],
                 not args.no_dufs, 0) for ck in keys]
    print(f"{len(payloads)} cells, dufs re-run = {not args.no_dufs}\n")

    rows, mech, bases = [], [], {}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for ck, r, m, b in ex.map(run_cell, payloads):
                rows += r
                mech += m
                bases[ck] = {k: v for k, v in b.items() if not k.startswith("_")}
                print(f"  done {ck}", flush=True)
    else:
        for pl in payloads:
            ck, r, m, b = run_cell(pl)
            rows += r
            mech += m
            bases[ck] = {k: v for k, v in b.items() if not k.startswith("_")}
            print(f"  done {ck}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for name, data in (("ensemble_bench.csv", rows), ("ensemble_mechanism.csv", mech)):
        if not data:
            continue
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    arms = ["upcr", "lsml_dufs_frozen", "lsml_good6"] + \
           ([] if args.no_dufs else ["lsml_dufs"])
    sels = ["oracle_headroom", "free_adaptive", "free_fixed", "all_fold"] + \
           [f"placebo_{b}" for b in range(N_PLACEBO)]
    summ = [s for arm in arms for sel in sels for mode in ("replace", "add_orth")
            if (s := summarise(rows, arm, sel, mode))]

    # ── M1: the exclusion channel ─────────────────────────────────────────────
    print(f"\n{'='*112}")
    print("M1 — U-PCR EXCLUSION RECOVERY.  Does repairing a view get it back into `keep`?")
    print(f"{'='*112}")
    for sel in ("oracle_headroom", "free_adaptive"):
        for mode in ("replace", "add_orth"):
            m = [x for x in mech if x["sel"] == sel and x["mode"] == mode]
            if not m:
                continue
            was_out = [x for x in m if not x["keep_base"]]
            recov = [x for x in was_out if x["keep_cfg"]]
            drho = np.array([abs(x["rho_cfg"]) - abs(x["rho_base"]) for x in m], float)
            dsel = [x for x in m if x["dufs_cfg"] and not x["dufs_base"]]
            print(f"\n  {sel:<18}{mode:<10}{len(m)} views")
            print(f"    excluded at base           {len(was_out)}/{len(m)}")
            print(f"    -> re-admitted after fold  {len(recov)}/{max(len(was_out),1)}"
                  f"   ({100*len(recov)/max(len(was_out),1):.0f}%)")
            print(f"    mean |rho_hat| change      {np.nanmean(drho):+.4f}")
            print(f"    newly DUFS-selected        {len(dsel)}/{len(m)}")
    m = [x for x in mech if x["sel"] == "oracle_headroom" and x["mode"] == "replace"]
    if m:
        sv = np.array([100 * (x["sv_auc_cfg"] - x["sv_auc_base"]) for x in m])
        dr = np.array([abs(x["rho_cfg"]) - abs(x["rho_base"]) for x in m])
        ok = np.isfinite(sv) & np.isfinite(dr)
        if ok.sum() > 5:
            rh, pv = spearmanr(sv[ok], dr[ok])
            print(f"\n  Spearman(single-view gain, |rho_hat| gain) = {rh:+.3f}  "
                  f"p = {pv:.2e}   n = {int(ok.sum())}")

    # ── the macro table ───────────────────────────────────────────────────────
    n_cells = len({r["cell"] for r in rows})
    print(f"\n{'='*112}")
    print(f"FUSED AUROC — macro delta over {n_cells} cells.  CI is a PAIRED CELL-LEVEL "
          f"BOOTSTRAP ({n_cells} units), quoted before the p.")
    if n_cells < len(INSCOPE):
        drop = [c for c in INSCOPE if c not in {r["cell"] for r in rows}]
        print(f"  ({len(INSCOPE)-n_cells} cell(s) carry no tested pair and are absent: "
              f"{', '.join(drop)} — n_min < MIN_NMIN)")
    print(f"{'='*112}")
    hdr = (f"{'arm':<18}{'selection':<16}{'mode':<9}{'views':>6}{'mv':>4}"
           f"{'macro':>8}{'95% CI':>17}{'p':>8}{'W/L':>8}{'worst':>8}"
           f"{'QA':>7}{'math':>7}")
    print(hdr)
    print("-" * len(hdr))
    for arm in arms:
        for s in summ:
            if s["arm"] != arm:
                continue
            nv = float(np.mean([x["n_views"] for x in rows
                                if x["arm"] == arm and x["sel"] == s["sel"]
                                and x["mode"] == s["mode"]]))
            ci = f"[{s['ci_lo']:+.2f},{s['ci_hi']:+.2f}]"
            wl = f"{s['wins']}/{s['losses']}"
            print(f"{s['arm']:<18}{s['sel']:<16}{s['mode']:<9}{nv:>6.1f}"
                  f"{s['n_moved']:>4}{s['macro_delta_pp']:>+8.2f}{ci:>17}"
                  f"{s['wilcoxon_p']:>8.3f}{wl:>8}{s['worst_pp']:>+8.2f}"
                  f"{s['qa_delta_pp']:>+7.2f}{s['math_delta_pp']:>+7.2f}")
        print()

    # ── the gates, written down before the run ────────────────────────────────
    print(f"{'='*112}")
    print("GATES")
    print(f"{'='*112}")
    for arm in arms:
        for sel in ("oracle_headroom", "free_adaptive"):
            for mode in ("replace", "add_orth"):
                s = next((z for z in summ if (z["arm"], z["sel"], z["mode"])
                          == (arm, sel, mode)), None)
                if s is None:
                    continue
                pl = [z for z in summ if z["arm"] == arm and z["mode"] == mode
                      and z["sel"].startswith("placebo")]
                pl_mx = max((z["macro_delta_pp"] for z in pl), default=float("nan"))
                g1 = s["macro_delta_pp"] >= 0.5 and s["wilcoxon_p"] < 0.05
                g2 = s["worst_pp"] >= -2.0
                g4 = np.isfinite(pl_mx) and pl_mx < 0.2
                print(f"  {arm:<18}{sel:<16}{mode:<9}"
                      f"G1 {'PASS' if g1 else 'FAIL'}  "
                      f"G2 {'PASS' if g2 else 'FAIL'}  "
                      f"G4 {'PASS' if g4 else 'FAIL'} (max placebo "
                      f"{pl_mx:+.2f}pp)")

    with open(os.path.join(OUT, "ensemble_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(cons_thr=CONS_THR, min_gain=MIN_GAIN, boot=BOOT,
                       baselines=bases, summary=summ), fh, indent=1)
    print(f"\nwrote {OUT}\\ensemble_bench.csv, ensemble_mechanism.csv, "
          f"ensemble_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
