"""
Experiment 5 - How should the measurements be weighted?

Five different proposals have been made for closing the 1.45-point gap between
guessed and learned trust levels. They are NOT five competing methods - they
occupy three independent slots in one pipeline, so they are run as a factorial
and the effect of each slot is reported separately. Running them as a flat
bake-off would confound the slots.

  Slot A - condition the relationship matrix first
      none
      random-matrix cleaning (Laloux et al. 1999 eigenvalue clipping at the
          Marchenko-Pastur edge), then re-zero the diagonal and re-decompose

  Slot B - estimate how strongly each measurement tracks the truth
      leading eigenvector (what we do now)
      triplet method-of-moments (median over triples - robust to local breakage)
      low-rank plus sparse split (Robust-PCA style; separates the shared cause
          from pairwise nuisance links)
      robust re-weighted fit (downweights pairs the model fits worst)

  Slot C - turn those into weights
      weight by signal strength (what we do now)
      weight by signal divided by that measurement's own noise ("precision")

Plus a plain average as a floor.

IMPORTANT ON THE STATISTICS: with 16 combinations, 25 test sets and only 1.45
points of headroom, the best-looking combination will look good by luck. So the
headline is the MAIN EFFECT of each slot, averaged over the others - not the
winning cell. No combination is declared a winner on a small average difference.

Writes results/pruning_study/05_weighting/
"""
import itertools
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402
from spectral_utils.streaming_utils import anchor_orient          # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

EPS = 1e-9


# ---------------------------------------------------------------- slot A
def condition_none(R, n):
    return R


def condition_rmt(R, n):
    """Random-matrix eigenvalue clipping (Laloux/Bouchaud style).

    Eigenvalues below the Marchenko-Pastur upper edge are indistinguishable
    from noise; replace them all by their average, keeping the trace. Then the
    diagonal is re-zeroed downstream, so unlike plain shrinkage this DOES move
    the eigenvector we care about.
    """
    p = R.shape[0]
    d = np.sqrt(np.clip(np.diag(R), EPS, None))
    C = R / np.outer(d, d)
    vals, vecs = np.linalg.eigh(C)
    q = p / max(n, p + 1)
    edge = (1.0 + np.sqrt(q)) ** 2
    noise = vals < edge
    if noise.sum() > 1:
        vals = vals.copy()
        vals[noise] = vals[noise].mean()
    C2 = vecs @ np.diag(vals) @ vecs.T
    return C2 * np.outer(d, d)


# ---------------------------------------------------------------- slot B
def loadings_eigenvector(R):
    Roff = R - np.diag(np.diag(R))
    vals, vecs = np.linalg.eigh(Roff)
    v = vecs[:, -1]
    if np.sum(v > 0) < len(v) / 2:
        v = -v
    return v * np.sqrt(max(vals[-1], EPS))


def loadings_triplets(R):
    """a_i^2 = median over pairs (j,k) of (R_ij * R_ik) / R_jk."""
    p = R.shape[0]
    a2 = np.zeros(p)
    for i in range(p):
        vals = []
        for j in range(p):
            if j == i:
                continue
            for k in range(j + 1, p):
                if k == i:
                    continue
                den = R[j, k]
                if abs(den) > 1e-6:
                    vals.append(R[i, j] * R[i, k] / den)
        a2[i] = np.median(vals) if vals else 0.0
    a = np.sqrt(np.clip(a2, 0, None))
    ref = loadings_eigenvector(R)
    return a * np.sign(np.where(np.abs(ref) < EPS, 1.0, ref))


def loadings_lowrank_sparse(R, lam_frac=0.15, iters=30):
    """Split the off-diagonal relationships into a single shared cause plus a
    sparse set of pairwise nuisance links, then read the shared cause.

    The diagonal is treated as UNOBSERVED throughout (it is contaminated by each
    measurement's own noise, which is why the method discards it) - fitting it
    as an observed zero would bias the shared-cause estimate.
    """
    Roff = R - np.diag(np.diag(R))
    mask = ~np.eye(R.shape[0], dtype=bool)
    lam = lam_frac * np.median(np.abs(Roff[mask]))
    a = loadings_eigenvector(R)
    Sp = np.zeros_like(Roff)
    for _ in range(iters):
        resid = Roff - np.outer(a, a)
        resid[~mask] = 0.0
        Sp = np.sign(resid) * np.clip(np.abs(resid) - lam, 0, None)
        Sp[~mask] = 0.0
        target = Roff - Sp
        target[~mask] = 0.0
        vals, vecs = np.linalg.eigh(target)
        v = vecs[:, -1]
        if np.sum(v > 0) < len(v) / 2:
            v = -v
        a = v * np.sqrt(max(vals[-1], EPS))
    return a


def loadings_robust_irls(R, iters=25):
    """Rank-one fit that downweights the pairs it fits worst."""
    Roff = R - np.diag(np.diag(R))
    mask = ~np.eye(R.shape[0], dtype=bool)
    a = loadings_eigenvector(R)
    for _ in range(iters):
        resid = np.abs(Roff - np.outer(a, a))
        scale = np.median(resid[mask]) + EPS
        u = 1.0 / (1.0 + (resid / scale) ** 2)
        u[~mask] = 0.0
        num = (u * Roff) @ a
        den = (u @ (a ** 2)) + EPS
        a_new = num / den
        if not np.all(np.isfinite(a_new)) or np.linalg.norm(a_new) < EPS:
            break
        a = a_new / np.linalg.norm(a_new) * np.linalg.norm(a)
    return a


# ---------------------------------------------------------------- slot C
def weights_signal(a, R):
    return a


def weights_precision(a, R):
    """w ~ a / var(noise_j), with var(noise_j) = R_jj - a_j^2.

    This is the one recipe that READS the diagonal the method normally throws
    away, so it is also the only one where conditioning the diagonal can matter.
    """
    noise = np.diag(R) - a ** 2
    noise = np.clip(noise, 1e-6 * max(np.median(np.abs(np.diag(R))), EPS), None)
    return a / noise


SLOT_A = {"no conditioning": condition_none,
          "random-matrix cleaning": condition_rmt}
SLOT_B = {"leading eigenvector (current)": loadings_eigenvector,
          "triplet method-of-moments": loadings_triplets,
          "low-rank plus sparse split": loadings_lowrank_sparse,
          "robust re-weighted fit": loadings_robust_irls}
SLOT_C = {"weight by signal (current)": weights_signal,
          "weight by signal over noise": weights_precision}


def score_config(cell, cols, fa, fb, fc):
    V = cell["V"][:, cols]
    n = V.shape[0]
    R = np.cov(V.T)
    try:
        Rc = fa(R, n)
        a = fb(Rc)
        w = fc(a, Rc)
        if not np.all(np.isfinite(w)) or np.linalg.norm(w) < EPS:
            return np.nan
        fused = V @ w
        if fused.std() < 1e-12:
            return np.nan
        sc, _ = anchor_orient(np.asarray(fused, float), cell["anchor"])
        return float(roc_auc_score(cell["labels"], sc))
    except Exception:
        return np.nan


def main():
    out = S.outdir("05_weighting")
    cells = S.load()
    S.validity_check(cells)

    rows = {}
    detail = []
    for (na, fa), (nb, fb), (nc, fc) in itertools.product(
            SLOT_A.items(), SLOT_B.items(), SLOT_C.items()):
        key = (na, nb, nc)
        vals = []
        for ck, cell in cells.items():
            cols = list(range(len(cell["pool"])))
            v = score_config(cell, cols, fa, fb, fc)
            vals.append(v)
            detail.append({"conditioning": na, "loading_estimator": nb,
                           "weighting": nc, "test_set": S.plain_cell(ck),
                           "test_set_code": ck, "auroc": v})
        rows[key] = np.array(vals, float)
        print(f"  {na[:22]:22s} | {nb[:28]:28s} | {nc[:26]:26s} "
              f"-> {np.nanmean(vals):.4f}")

    # plain average floor
    plain = []
    for ck, cell in cells.items():
        V = cell["V"]
        sg = np.sign([np.corrcoef(V[:, j], cell["anchor"])[0, 1]
                      for j in range(V.shape[1])])
        sg[sg == 0] = 1
        m = (V * sg).mean(1)
        sc, _ = anchor_orient(m, cell["anchor"])
        plain.append(roc_auc_score(cell["labels"], sc))
    plain = np.array(plain, float)
    print(f"  plain average (direction from the anchor)  -> {np.nanmean(plain):.4f}")

    S.save_csv(os.path.join(out, "all_configurations.csv"), detail)
    S.save_json(os.path.join(out, "config_scores.json"),
                {" | ".join(k): v.tolist() for k, v in rows.items()})
    S.save_npz(os.path.join(out, "plain_average.npz"), auroc=plain)

    base = rows[("no conditioning", "leading eigenvector (current)",
                 "weight by signal (current)")]

    # ---------- main effects ----------
    def main_effect(slot_idx, names):
        eff = []
        for nm in names:
            sel = [v for k, v in rows.items() if k[slot_idx] == nm]
            eff.append((nm, float(np.nanmean([np.nanmean(v) for v in sel])),
                        len(sel)))
        return eff

    me_rows = []
    for idx, (slot, names) in enumerate([("Conditioning", list(SLOT_A)),
                                         ("Loading estimator", list(SLOT_B)),
                                         ("Weighting", list(SLOT_C))]):
        for nm, val, k in main_effect(idx, names):
            me_rows.append({"slot": slot, "option": nm,
                            "mean_auroc_over_other_slots": val,
                            "n_configurations_averaged": k})
    S.save_csv(os.path.join(out, "main_effects.csv"), me_rows)

    # ---------- per-configuration comparison vs current ----------
    comp = []
    for k, v in rows.items():
        m = np.isfinite(v) & np.isfinite(base)
        try:
            p = (wilcoxon(v[m], base[m]).pvalue
                 if m.sum() > 5 and np.any(v[m] != base[m]) else np.nan)
        except Exception:
            p = np.nan
        comp.append({"conditioning": k[0], "loading_estimator": k[1],
                     "weighting": k[2], "macro_auroc": float(np.nanmean(v)),
                     "difference_vs_current_pp": float((np.nanmean(v) - np.nanmean(base)) * 100),
                     "test_sets_better": int((v > base)[m].sum()),
                     "test_sets_worse": int((v < base)[m].sum()),
                     "wilcoxon_p": float(p)})
    comp.sort(key=lambda r: -r["macro_auroc"])
    S.save_csv(os.path.join(out, "configuration_comparison.csv"), comp)

    # ---------- charts ----------
    me_charts = ""
    for slot, names in [("Conditioning the relationship matrix", list(SLOT_A)),
                        ("Estimating how much each measurement tracks the truth",
                         list(SLOT_B)),
                        ("Turning that into weights", list(SLOT_C))]:
        idx = {"Conditioning the relationship matrix": 0,
               "Estimating how much each measurement tracks the truth": 1,
               "Turning that into weights": 2}[slot]
        eff = main_effect(idx, names)
        me_charts += f"<h3>{slot}</h3>" + S.bar_chart(
            [e[0] for e in eff], [e[1] for e in eff],
            "Accuracy averaged over every setting of the other two slots",
            bar_h=30)

    cfg_chart = S.bar_chart(
        [f"{c['loading_estimator']} + {c['weighting'].replace('weight by ','')}"
         f" ({c['conditioning'].replace(' conditioning','')})" for c in comp],
        [c["macro_auroc"] for c in comp],
        "Detection accuracy (AUROC) over 25 test sets",
        hlines=[("six hand-picked", 0.7594)], bar_h=24)

    comp_tbl = S.html_table(
        ["Conditioning", "Loading estimator", "Weighting", "Accuracy",
         "vs current", "Better", "Worse", "Wilcoxon p"],
        [[c["conditioning"], c["loading_estimator"], c["weighting"],
          f"{c['macro_auroc']:.4f}", f"{c['difference_vs_current_pp']:+.2f} pp",
          c["test_sets_better"], c["test_sets_worse"],
          "-" if not np.isfinite(c["wilcoxon_p"]) else f"{c['wilcoxon_p']:.3f}"]
         for c in comp], numeric_cols=(3, 4, 5, 6, 7))

    best = comp[0]
    body = f"""
<h2>Why this is a factorial and not a bake-off</h2>
<p>Five proposals were put forward for this gap. They are not five rival
methods &mdash; they sit in three independent slots of one pipeline: how the
relationship matrix is cleaned first, how each measurement's reliability is
estimated from it, and how those reliabilities become weights. Running them as
one flat list would mix the slots together and make it impossible to say which
part mattered.</p>

<div class="warn"><b>On reading the numbers below.</b> There are
{len(comp)} configurations, 25 test sets, and only about 1.45 points of headroom
in total. The best-looking configuration will look good partly by luck. So the
main result is the <b>effect of each slot averaged over the others</b>, and no
configuration is called a winner on a small difference in average accuracy.</div>

<h2>Main effects - what each slot is worth</h2>
{me_charts}

<h2>Every configuration</h2>
{cfg_chart}
{comp_tbl}
<p class="note">Plain average of all 30 measurements, with each direction taken
from the anchor measurement: <b>{np.nanmean(plain):.4f}</b>. Note that a plain
average needs a direction for every measurement, while the method being tested
does not &mdash; so this floor is not a like-for-like comparison, it is a
sign-privileged baseline.</p>

<h2>Reading this</h2>
<ul>
<li>Current recipe (leading eigenvector, weight by signal, no conditioning):
<b>{np.nanmean(base):.4f}</b>.</li>
<li>Highest-scoring configuration: {best['loading_estimator']} +
{best['weighting']} + {best['conditioning']}, at
<b>{best['macro_auroc']:.4f}</b> ({best['difference_vs_current_pp']:+.2f} pp,
better on {best['test_sets_better']} of 25 test sets).</li>
<li>Whether that difference is real should be judged from the win/loss split and
the Wilcoxon column, not the average &mdash; and from whether the same slot option
also wins in the main-effects chart, where luck has less room to operate.</li>
</ul>

<h2>Saved data</h2>
<ul>
<li><code>all_configurations.csv</code> &mdash; every configuration on every test set</li>
<li><code>main_effects.csv</code> &mdash; each slot option averaged over the others</li>
<li><code>configuration_comparison.csv</code> &mdash; per-configuration comparison against the current recipe</li>
<li><code>config_scores.json</code>, <code>plain_average.npz</code> &mdash; raw per-test-set score vectors</li>
</ul>
"""
    S.write_page(
        os.path.join(out, "index.html"),
        "Experiment 5 - How to weight the measurements",
        "Sixteen weighting pipelines as a three-slot factorial, plus a plain-average floor.",
        [f"Current recipe scores <b>{np.nanmean(base):.4f}</b> on the full 30 measurements.",
         f"Highest-scoring configuration reaches <b>{best['macro_auroc']:.4f}</b> "
         f"({best['difference_vs_current_pp']:+.2f} pp, better on "
         f"{best['test_sets_better']}/25 test sets).",
         "Reported as main effects per slot rather than a winning combination - with "
         "16 configurations and 1.45 points of headroom, best-of-16 is mostly luck.",
         "No configuration is being adopted on the strength of a sub-point average."],
        body)
    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
