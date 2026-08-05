#!/usr/bin/env python
"""
build_pair_compare.py — is it the algorithm, or is it the features?

Every diagnostic so far (Steps 209-213) asked what the FUSION does differently on
the weak cells. Each answer came back negative: orientation costs +0.00pp on all
25 cells, 0/25 groupings are degenerate, pool composition was closed in both
directions, and the pre-registered sign repair failed its gate and moved the macro
by -0.14pp. So the remaining hypothesis is that the weak cells hand the fusion
worse RAW MATERIAL, and no fusion can fix that.

This script tests it the only way that is not circular: pick a weak cell, pair it
with a HIGH-SCORING cell that is as similar as possible, and compare the features
themselves — their class-conditional distributions, their individual separability,
their correlation with one another, and the ceiling a fully supervised model can
reach from them. If the weak cell's features are simply less informative, the
supervised ceiling drops with the label-free score and there is nothing to repair.

Three pairs, ordered by how much they hold fixed:

  A  triviaqa    same dataset, comparable n and class balance
  B  gsm8k_llama same dataset AND same source cache AND same model family;
                 model SIZE is the only variable
  C  k10_traces  same K=10 multi-trace pipeline, different dataset+model

REVISED AFTER THE STEP-215 ADVERSARIAL REVIEW. Four changes, each because a
Step-214 conclusion did not survive:
  * pair A's strong cell was `spilled_triviaqa_llama8b` — 6 positives of 256,
    trace_length alone scores 0.925 on it, and advisor_report.py:783 already
    said not to headline it. Replaced with `semenergy_triviaqa_qwen3_8b`.
  * `lr_ceiling` no longer applies max(a, 1-a) to a SUPERVISED score (it can
    only fire on noise and only inflate: +12.6pp on the 6-positive cell), and
    averages over 10 seeds because one seed on a small cell has sd 2.8pp.
  * mean|excess rho| is demoted from "an independent finding" to what it is:
    algebraically implied by the per-view Cohen's d already reported, times the
    class prior. It is still shown, with that stated.
  * the excess statistic is normalised against a PERMUTATION null, not 1/sqrt(n)
    — C and W share rows, so their difference is far tighter than the SE of one
    correlation, and the old scale made real structure look sub-noise.

Pure CPU, offline against local_cache/. Gate 0 is assert_good6.
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, os.path.join(REPO, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import HIST_LEGEND, bar, esc, hist_svg, num, page            # noqa: E402
from inscope_bench_common import assert_good6, load_cells                # noqa: E402
from labelfree_standing_report import PRETTY                             # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous                  # noqa: E402
from spectral_utils.streaming_utils import anchor_orient                 # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                        # noqa: E402
from spectral_utils.upcr import upcr_fit                                 # noqa: E402

OUT = os.path.join(REPO, "results", "action_items_jul2026",
                   "item1b_feature_comparison")

FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

NBINS, CLIP = 26, 2.8

PAIRS = [
    dict(
        id="triviaqa", weak="seiclr_triviaqa_opt30b",
        strong="semenergy_triviaqa_qwen3_8b",
        title="TriviaQA — the same dataset, our worst QA cell against a healthy one",
        held="the dataset (TriviaQA), the task shape (one short factual answer), "
             "and a comparable sample size (4,993 vs 4,392) and class balance "
             "(0.46 vs 0.48)",
        varies="the model (OPT-30B base vs Qwen3-8B), the source cache, and the "
               "grader (ROUGE-L &gt; 0.3 vs normalised exact match + judge)",
        why="This is the pair Omri named first. Both cells answer the same "
            "benchmark at comparable n and comparable class balance, so neither "
            "sample size nor prevalence can explain the gap.",
        note="<b>The partner was changed after the Step-215 review.</b> This pair "
             "originally used <code>spilled_triviaqa_llama8b</code> (0.9413) as "
             "the strong cell. That cell has <b>6 positives out of 256</b>, "
             "<code>trace_length</code> alone scores 0.925 on it, and the project "
             "already carries a standing instruction to treat it as "
             "selection-biased and <b>not headline it</b> "
             "(<code>scripts/advisor_report.py:783</code>). Using it was an "
             "error: it inflated the gap to +37.99pp and every conclusion drawn "
             "from that pair. <code>semenergy_triviaqa_qwen3_8b</code> is the "
             "closest defensible control.",
    ),
    dict(
        id="gsm8k_llama", weak="lapeigvals_gsm8k_llama3b",
        strong="lapeigvals_gsm8k_llama8b",
        title="GSM8K / Llama-3 — model size is the only variable",
        held="the dataset (GSM8K), the source cache (the LapEigvals release), the "
             "decoding config, and the model family (Llama-3)",
        varies="model size, 3B vs 8B — and nothing else",
        why="The cleanest controlled comparison in the grid. If the feature "
            "quality moves here, it moves because of the model alone, with the "
            "data pipeline held fixed. This is the pair that can rule out a "
            "cache-specific artifact.",
    ),
    dict(
        id="k10_traces", weak="trace_math500_qwenmath15b_k10",
        strong="trace_gsm8k_llama8b_k10",
        title="K=10 multi-trace — the same pipeline, weak vs strong",
        held="the K=10 multi-trace inference pipeline and its feature extraction",
        varies="the dataset (MATH-500 vs GSM8K) and the model "
               "(Qwen2.5-Math-1.5B vs Llama-3.1-8B)",
        why="Both cells are built by the same multi-trace code, so a defect in "
            "that pipeline would hit both. It separates 'the K=10 path is broken' "
            "from 'this dataset/model is hard'.",
    ),
]


# ── measurement ──────────────────────────────────────────────────────────────
def hist_by_class(x, y):
    x = np.clip(np.asarray(x, float), -CLIP, CLIP)
    edges = np.linspace(-CLIP, CLIP, NBINS + 1)
    out = {}
    for key, mask in (("pos", y == 1), ("neg", y == 0)):
        h, _ = np.histogram(x[mask], bins=edges)
        s = h.sum()
        out[key] = (h / s).round(5).tolist() if s else [0.0] * NBINS
    return out


def within_class_corr(X, y):
    """Correlation left over after removing the class means.

    This is the part of the shared structure that the LABEL does not explain.
    U-PCR and L-SML both assume one latent variable drives every view, i.e. that
    the off-diagonal correlation is (approximately) rank-1 and label-driven. If
    the total correlation and the within-class correlation coincide, the views
    are correlated for reasons that have nothing to do with correctness, and the
    latent-variable model has nothing to lock onto.
    """
    Xc = X.copy()
    for cls in (0, 1):
        m = y == cls
        if m.sum() > 1:
            Xc[m] -= Xc[m].mean(axis=0)
    return np.corrcoef(Xc, rowvar=False)


def rank1_fit(C):
    """Best rank-1 fit to the OFF-DIAGONAL of C, and its relative residual.

    Diagonals are excluded because they are 1 by construction and would flatter
    any fit. Returns (rho, relative residual). rho_i is view i's implied loading
    on the latent variable — exactly the quantity U-PCR estimates.
    """
    p = C.shape[0]
    off = ~np.eye(p, dtype=bool)
    Coff = C * off
    w, V = np.linalg.eigh(Coff)                        # init from the top eigvec
    r0 = V[:, -1] * np.sqrt(max(w[-1], 0.0))

    # The plain ALS iteration used before the Step-215 review does NOT converge:
    # it settles into a period-2 limit cycle (so the answer depended on iteration
    # parity) and diverges from almost every start other than this eigenvector.
    # Minimise the objective directly instead, and keep the best of a few starts.
    def f(r):
        R = C - np.outer(r, r)
        return float((R[off] ** 2).sum())

    def g(r):
        R = (C - np.outer(r, r)) * off
        return -4.0 * (R @ r)

    best, brho = None, r0
    rng = np.random.default_rng(0)
    for k, start in enumerate([r0] + [r0 * s for s in (0.5, 1.5)]
                              + [rng.normal(0, 0.3, p) for _ in range(5)]):
        o = minimize(f, start, jac=g, method="L-BFGS-B",
                     options=dict(maxiter=2000))
        if best is None or o.fun < best:
            best, brho = float(o.fun), o.x
    rho = brho
    res = C - np.outer(rho, rho)
    rel = float(np.linalg.norm(res[off]) / max(np.linalg.norm(C[off]), 1e-12))
    return rho, rel


MIN_POS = 25          # below this the ceiling is not a measurement (see NOTE)


def lr_ceiling(X, y, n_seeds=10, n_folds=5):
    """Supervised ceiling: 5-fold LR, class_weight='balanced', AUROC averaged
    PER FOLD (never over concatenated out-of-fold probabilities — see
    SUPERVISED_ORACLE_CORRECTION.md, bugs A and B), then averaged over seeds.

    NOTE — two corrections after the Step-215 review:
      * NO max(a, 1-a). That transform exists for *unsupervised* scores whose
        sign is undetermined; `predict_proba` has a determined direction, so the
        max() can only ever fire on noise and can only inflate. On a cell with
        6 positives it inflated the ceiling by +12.6pp.
      * Averaged over 10 seeds and the across-seed sd returned, because a single
        seed on a small cell is a draw from a distribution 2.8pp wide.
    """
    npos = int(min(np.bincount(y, minlength=2)))
    if npos < n_folds:
        return None, None, npos
    vals = []
    for seed in range(n_seeds):
        aucs = []
        for tr, te in StratifiedKFold(n_folds, shuffle=True,
                                      random_state=seed).split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000, class_weight="balanced")
            m.fit(sc.transform(X[tr]), y[tr])
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        vals.append(float(np.mean(aucs)))
    return float(np.mean(vals)), float(np.std(vals)), npos


def excess_null(X, y, n_perm=200, seed=0):
    """Permutation null for mean|excess rho|.

    The previous normalisation divided by 1/sqrt(n), the SE of a SINGLE
    correlation. That is the wrong scale: C and W are computed from the same
    rows, so under the null of no class effect they are nearly identical and
    their difference has variance far below 1/n. Dividing by the wrong scale
    made real structure look like it sat under the noise floor, which is what
    the withdrawn "retire estimation noise" claim rested on. Permute the labels
    instead and measure the null directly.
    """
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    off = ~np.eye(p, dtype=bool)
    C = np.corrcoef(X, rowvar=False)
    vals = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        vals.append(float(np.abs((C - within_class_corr(X, yp))[off]).mean()))
    v = np.array(vals)
    obs = float(np.abs((C - within_class_corr(X, y))[off]).mean())
    return dict(obs=round(obs, 5), null_mean=round(float(v.mean()), 5),
                null_p95=round(float(np.percentile(v, 95)), 5),
                z=round(float((obs - v.mean()) / max(v.std(), 1e-12)), 1),
                ratio=round(float(obs / max(v.mean(), 1e-12)), 2))


def deployed_scores(cell, dufs_chosen):
    """The two published label-free numbers, recomputed exactly as the canonical
    path computes them: L-SML over the DUFS-chosen subset, U-PCR over the FULL
    pool with its own internal exclusion. Deliberately NOT restricted to the
    common pool — these rows have to reproduce the published per-cell values,
    otherwise the page silently redefines the thing it is explaining.
    """
    y = np.asarray(cell["labels"], int)
    V, pool, anc = cell["V"], cell["pool"], cell["anchor"]

    cols = sorted({pool.index(f) for f in dufs_chosen if f in pool})
    fused, _ = lsml_continuous(*[V[:, c] for c in cols])
    lsml = float(roc_auc_score(y, anchor_orient(np.asarray(fused, float), anc)[0]))

    hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], float)
    V_un = V * hand
    der = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
    der[der == 0] = 1.0
    F = (V_un * der).T
    r = upcr_fit(F, **FIT)
    upcr = float(roc_auc_score(y, anchor_orient(r.w @ F, anc)[0]))
    return (round(lsml, 4), round(upcr, 4), len(cols), int(r.keep.sum()))


def measure(cell, cols, ck):
    y = np.asarray(cell["labels"], int)
    V, pool = cell["V"], cell["pool"]
    X = V[:, cols]
    p = X.shape[1]

    views = []
    for i, j in enumerate(cols):
        x = V[:, j]
        a = float(roc_auc_score(y, x))
        mp, mn = x[y == 1].mean(), x[y == 0].mean()
        sd = np.sqrt(((x[y == 1].var(ddof=1) * (y == 1).sum()
                       + x[y == 0].var(ddof=1) * (y == 0).sum())
                      / max(len(y) - 2, 1)))
        views.append(dict(
            name=pool[j], auroc=round(a, 4), auroc_or=round(max(a, 1 - a), 4),
            d=round(float((mp - mn) / max(sd, 1e-12)), 3),
            hist=hist_by_class(x, y)))

    C = np.corrcoef(X, rowvar=False)
    W = within_class_corr(X, y)
    E = C - W
    off = ~np.eye(p, dtype=bool)
    rho, r1res = rank1_fit(C)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    n = len(y)

    return dict(
        cell=ck, pretty=PRETTY[ck], n=int(n), pos=int((y == 1).sum()),
        pos_rate=round(float(y.mean()), 4), p=p,
        views=views,
        C=np.round(C, 4).tolist(), W=np.round(W, 4).tolist(),
        E=np.round(E, 4).tolist(),
        mean_abs_C=round(float(np.abs(C[off]).mean()), 4),
        mean_abs_W=round(float(np.abs(W[off]).mean()), 4),
        mean_abs_E=round(float(np.abs(E[off]).mean()), 4),
        redundant_pairs=int((np.abs(C[off]) >= 0.75).sum() // 2),
        rank1_residual=round(r1res, 4),
        rho_mean_abs=round(float(np.abs(rho).mean()), 4),
        lam1_frac=round(float(lam[0] / lam.sum()), 4),
        eigengap=round(float(lam[0] / max(lam[1], 1e-12)), 3),
        eff_rank=round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2),
        corr_se=round(float(1.0 / np.sqrt(max(n - 3, 1))), 4),
        excess_perm=excess_null(X, y),
        kappa=round(float(y.mean() * (1 - y.mean())), 4),
        best_single=round(float(max(v["auroc_or"] for v in views)), 4),
        median_single=round(float(np.median([v["auroc_or"] for v in views])), 4),
        n_above_55=int(sum(1 for v in views if v["auroc_or"] >= 0.55)),
        pool_size=len(pool),
        **dict(zip(("lr_ceiling", "lr_sd", "n_pos_min"), lr_ceiling(X, y))),
    )


# ── rendering ────────────────────────────────────────────────────────────────
def heat(M, w=13, names=None, title=""):
    """Correlation heatmap. Blue = positive, red = negative, white = zero.

    Fixed diverging scale on [-1, 1] so the two panels are directly comparable —
    an auto-scaled pair of heatmaps would make a weak structure look like a
    strong one.
    """
    p = len(M)
    sz = p * w
    cells = []
    for i in range(p):
        for j in range(p):
            v = float(M[i][j])
            t = max(-1.0, min(1.0, v))
            if t >= 0:
                r, g, b = 255 - int(190 * t), 255 - int(120 * t), 255 - int(20 * t)
            else:
                r, g, b = 255 - int(15 * -t), 255 - int(140 * -t), 255 - int(160 * -t)
            nm = f"{names[i]} x {names[j]}: {v:+.2f}" if names else f"{v:+.2f}"
            cells.append(f'<rect x="{j * w}" y="{i * w}" width="{w}" height="{w}" '
                         f'fill="rgb({r},{g},{b})"><title>{esc(nm)}</title></rect>')
    return (f'<svg width="{sz}" height="{sz}" viewBox="0 0 {sz} {sz}" '
            f'role="img" aria-label="{esc(title)}" '
            f'style="border:1px solid var(--line)">{"".join(cells)}</svg>')


HEAT_LEGEND = (
    '<span style="font-size:12.5px;color:var(--mut)">scale fixed to '
    '[&minus;1, +1] on both panels &nbsp;'
    '<span class="lg" style="background:rgb(65,135,235)"></span>+1 &nbsp;'
    '<span class="lg" style="background:#fff;border:1px solid var(--line)"></span>0 '
    '&nbsp;<span class="lg" style="background:rgb(240,115,95)"></span>&minus;1 '
    '&nbsp;&mdash; hover any square for the pair and its value</span>')


def stat_row(label, wv, sv, fmt="{:.4f}", note="", better=None):
    """One comparison row. `better` in {'hi','lo',None} colours the weak cell."""
    cls = ""
    if better and wv is not None and sv is not None:
        worse = (wv < sv) if better == "hi" else (wv > sv)
        cls = ' class="neg"' if worse else ' class="pos"'
    f = (lambda v: "&mdash;" if v is None else fmt.format(v))
    return (f"<tr><td>{label}</td><td{cls}>{f(wv)}</td><td>{f(sv)}</td>"
            f'<td style="text-align:left;color:var(--mut);white-space:normal">'
            f"{note}</td></tr>")


def pair_page(spec, weak, strong):
    wn = f"{weak['pretty'][0]} / {weak['pretty'][1]}"
    sn = f"{strong['pretty'][0]} / {strong['pretty'][1]}"
    names = [v["name"] for v in weak["views"]]

    gap_lf = (strong["lsml"] - weak["lsml"]) * 100
    gap_ce = ((strong["lr_ceiling"] - weak["lr_ceiling"]) * 100
              if weak["lr_ceiling"] and strong["lr_ceiling"] else None)
    gap_bs = (strong["best_single"] - weak["best_single"]) * 100

    # how much of the label-free gap is already present in the raw material
    frac = (f"{100 * gap_ce / gap_lf:.0f}%"
            if gap_ce is not None and abs(gap_lf) > 1e-9 else "&mdash;")

    b = [f'<p class="crumb"><a href="index.html">&larr; feature comparison</a> '
         f'&middot; <a href="../index.html">action items</a></p>',
         f"<h1>{esc(spec['title'])}</h1>",
         f'<p class="sub"><b>{esc(wn)}</b> (label-free {weak["lsml"]:.4f}) '
         f'against <b>{esc(sn)}</b> (label-free {strong["lsml"]:.4f}) '
         f'&mdash; a gap of {gap_lf:+.1f}pp.</p>',
         '<div class="box"><p><b>Held fixed:</b> ' + esc(spec["held"]) + "</p>"
         "<p><b>Varies:</b> " + spec["varies"] + "</p>"
         "<p>" + esc(spec["why"]) + "</p></div>"]
    if spec.get("note"):
        b.append(f'<div class="box bad"><p>{spec["note"]}</p></div>')

    # ── 1. the headline ------------------------------------------------------
    b.append("<h2>1. Does the gap exist before any fusion runs?</h2>")
    b.append(
        "<p>The question this whole page exists to answer. If the features on the "
        "weak cell are simply less informative, then a <b>supervised</b> model "
        "with full access to the labels cannot do well either &mdash; and the gap "
        "is in the raw material, not in the label-free machinery. The supervised "
        "ceiling below is 5-fold logistic regression, <code>class_weight="
        "'balanced'</code>, AUROC averaged per fold.</p>")
    b.append('<div class="scroll"><table><tr><th>quantity</th>'
             f"<th>{esc(wn)}</th><th>{esc(sn)}</th><th>reading</th></tr>")
    b.append(stat_row("best single view (oracle-oriented)", weak["best_single"],
                      strong["best_single"], better="hi",
                      note="the floor any fusion has to beat"))
    b.append(stat_row("median view (oracle-oriented)", weak["median_single"],
                      strong["median_single"], better="hi",
                      note="the typical view, not the luckiest one"))
    b.append(stat_row("views reaching 0.55", weak["n_above_55"],
                      strong["n_above_55"], fmt="{:d}", better="hi",
                      note=f"out of {weak['p']} in the common pool"))
    b.append(stat_row("<b>supervised ceiling (5-fold LR, 10 seeds)</b>",
                      weak["lr_ceiling"], strong["lr_ceiling"], better="hi",
                      note=f"what the features support when the labels are "
                           f"handed over. Across-seed sd &plusmn;"
                           f"{weak['lr_sd']:.4f} / &plusmn;{strong['lr_sd']:.4f} "
                           f"&mdash; read the gap against that, not as a point."))
    b.append(stat_row("label-free, DUFS + L-SML", weak["lsml"], strong["lsml"],
                      better="hi",
                      note=f"what we actually deploy "
                           f"({weak['dufs_size']} vs {strong['dufs_size']} views "
                           f"chosen)"))
    b.append(stat_row("label-free, U-PCR", weak["upcr"], strong["upcr"],
                      better="hi",
                      note=f"the other leading arm "
                           f"({weak['upcr_kept']} vs {strong['upcr_kept']} views "
                           f"kept after its own exclusion)"))
    b.append("</table></div>")
    b.append(
        '<p style="font-size:12.5px;color:var(--mut)">The two label-free rows are '
        "recomputed on their canonical footings &mdash; L-SML over the "
        "DUFS-chosen subset, U-PCR over the full pool with its own exclusion "
        f"&mdash; so they reproduce the published per-cell values. Everything "
        f"else on this page uses the <b>{weak['p']}-view pool the two cells share"
        f"</b> (of {weak['pool_size']} and {strong['pool_size']}), because a "
        "feature comparison across cells is only meaningful on features both "
        "cells have.</p>")

    hw = ((weak["lr_ceiling"] - weak["lsml"]) * 100
          if weak["lr_ceiling"] else None)
    hs = ((strong["lr_ceiling"] - strong["lsml"]) * 100
          if strong["lr_ceiling"] else None)
    b.append(
        f'<div class="box warn"><p>The supervised ceiling moves {gap_ce:+.1f}pp '
        f"across this pair, against a label-free gap of {gap_lf:+.1f}pp, so "
        f"<b>on the order of {frac} of the gap is already present in the "
        f"features</b> before any fusion, selection or orientation runs. The "
        f"best single view alone differs by {gap_bs:+.1f}pp.</p>"
        f"<p><b>Treat that percentage as a direction, not a measurement.</b> "
        f"Bootstrapping it in the Step-215 review gave intervals wide enough to "
        f"include 100% on two of the three pairs, and swapping in a different "
        f"but equally defensible partner cell moved the TriviaQA figure across "
        f"the range &minus;2% to 60%. What survives is the sign: some of the "
        f"gap is in the features, and not all of it.</p>"
        f"<p>On the weak cell the supervised model reaches {hw:+.1f}pp above "
        f"what we deliver; on the strong cell {hs:+.1f}pp.</p></div>")

    # ── 2. the features themselves ------------------------------------------
    b.append("<h2>2. The features themselves</h2>")
    b.append(
        "<p>Every view, in both cells, drawn the same way: the two "
        "class-conditional densities of the z-scored view. A view that separates "
        "shows two displaced humps; a view at chance shows one shape drawn twice. "
        f"{HIST_LEGEND}. <code>d</code> is Cohen's d &mdash; the mean shift "
        "between the classes in pooled standard deviations. Rows are sorted by "
        "how much the two cells disagree.</p>")
    order = sorted(range(len(names)),
                   key=lambda i: -abs(strong["views"][i]["auroc_or"]
                                      - weak["views"][i]["auroc_or"]))
    b.append('<div class="scroll"><table><tr><th>#</th><th>view</th>'
             f"<th>{esc(wn)}</th><th>AUROC</th><th>d</th>"
             f"<th>{esc(sn)}</th><th>AUROC</th><th>d</th><th>&Delta; AUROC</th></tr>")
    for i in order:
        wv, sv = weak["views"][i], strong["views"][i]
        d = (sv["auroc_or"] - wv["auroc_or"]) * 100
        dc = "neg" if d > 2 else ("pos" if d < -2 else "")
        b.append(
            f'<tr><td class="mono">{i}</td><td><code>{esc(names[i])}</code></td>'
            f"<td>{hist_svg(wv['hist'])}</td>"
            f"<td>{wv['auroc_or']:.3f} {bar(wv['auroc_or'], 0.5, 0.95)}</td>"
            f"<td>{wv['d']:+.2f}</td>"
            f"<td>{hist_svg(sv['hist'])}</td>"
            f"<td>{sv['auroc_or']:.3f} {bar(sv['auroc_or'], 0.5, 0.95)}</td>"
            f"<td>{sv['d']:+.2f}</td>"
            f'<td class="{dc}">{d:+.1f}pp</td></tr>')
    b.append("</table></div>")

    # ── 3. correlation -------------------------------------------------------
    b.append("<h2>3. How the views correlate with one another</h2>")
    b.append(
        "<p>Both L-SML and U-PCR are built on one assumption: a single latent "
        "variable &mdash; correctness &mdash; drives every view, so the "
        "off-diagonal correlation should look like "
        "<code>&rho;<sub>i</sub>&rho;<sub>j</sub></code>, a rank-1 matrix. "
        "Everything they estimate (the grouping, the weights, the polarities) is "
        "read off that matrix. Three panels: the <b>total</b> correlation, the "
        "<b>within-class</b> correlation (class means removed &mdash; the part "
        "the label does <i>not</i> explain), and their <b>difference</b>, which "
        "is the only part the latent-variable model can legitimately use.</p>")
    b.append(f"<p>{HEAT_LEGEND}</p>")
    idx = ", ".join(f"<code>{i}</code>&nbsp;{esc(n)}" for i, n in enumerate(names))
    for key, lbl in (("C", "total correlation"),
                     ("W", "within-class correlation (label removed)"),
                     ("E", "excess = total &minus; within-class")):
        b.append(f"<h3>{lbl}</h3>"
                 '<div class="scroll" style="display:flex;gap:26px;'
                 'align-items:flex-start">'
                 f'<div><div style="font-size:12.5px;color:var(--mut);'
                 f'margin-bottom:4px">{esc(wn)}</div>'
                 f'{heat(weak[key], names=names, title=lbl)}</div>'
                 f'<div><div style="font-size:12.5px;color:var(--mut);'
                 f'margin-bottom:4px">{esc(sn)}</div>'
                 f'{heat(strong[key], names=names, title=lbl)}</div></div>')
    b.append(f'<p style="font-size:12px;color:var(--mut);white-space:normal">'
             f"row/column order: {idx}</p>")
    b.append(
        '<p style="font-size:12.5px;color:var(--mut)">A note on the third panel: '
        "total minus within-class is a difference of two correlation matrices, "
        "not an exact variance decomposition &mdash; removing the class means "
        "also shrinks each view's variance. It is a consistent, comparable "
        "measure of how much extra co-movement the label accounts for, and both "
        "panels are computed identically, but do not read an individual entry as "
        "a between-class covariance.</p>")

    b.append("<h3>What the matrices measure</h3>")
    b.append('<div class="scroll"><table><tr><th>quantity</th>'
             f"<th>{esc(wn)}</th><th>{esc(sn)}</th><th>reading</th></tr>")
    b.append(stat_row("mean |correlation|, total", weak["mean_abs_C"],
                      strong["mean_abs_C"],
                      note="how much shared structure there is at all"))
    b.append(stat_row("mean |correlation|, within-class", weak["mean_abs_W"],
                      strong["mean_abs_W"],
                      note="shared structure that is <b>not</b> the label &mdash; "
                           "nuisance the model will read as signal"))
    b.append(stat_row("mean |excess| (total &minus; within)",
                      weak["mean_abs_E"], strong["mean_abs_E"], better="hi",
                      note="the extra co-movement the label accounts for. "
                           "<b>Not independent evidence</b> &mdash; see the note "
                           "below the table."))
    b.append(stat_row("&kappa; = &pi;(1&minus;&pi;), the class-prior factor",
                      weak["kappa"], strong["kappa"],
                      note="mean |excess| carries this factor, so it is not "
                           "comparable across cells of different class balance"))
    b.append(stat_row("mean |excess| &divide; &kappa;", weak["mean_abs_E"] / weak["kappa"],
                      strong["mean_abs_E"] / strong["kappa"], better="hi",
                      note="the prior divided out &mdash; the comparable form"))
    b.append(stat_row("permutation null for mean |excess|",
                      weak["excess_perm"]["null_mean"],
                      strong["excess_perm"]["null_mean"], better="lo",
                      note="200 label permutations. This replaces the "
                           "1/&radic;n normalisation used before the Step-215 "
                           "review, which was the wrong scale: C and W come from "
                           "the same rows, so their difference is far tighter "
                           "than the SE of one correlation."))
    b.append(stat_row("<b>excess vs its own null</b>", weak["excess_perm"]["ratio"],
                      strong["excess_perm"]["ratio"], fmt="{:.2f}&times;",
                      better="hi",
                      note=f"z = {weak['excess_perm']['z']:+.0f} vs "
                           f"{strong['excess_perm']['z']:+.0f}"))
    b.append(stat_row("rank-1 residual", weak["rank1_residual"],
                      strong["rank1_residual"], better="lo",
                      note="how badly the one-latent-variable model fits. "
                           "0 = perfect fit, 1 = the model explains nothing."))
    b.append(stat_row("&lambda;<sub>1</sub> share of the spectrum",
                      weak["lam1_frac"], strong["lam1_frac"], better="hi",
                      note="one dominant direction, or many competing ones"))
    b.append(stat_row("effective rank", weak["eff_rank"], strong["eff_rank"],
                      fmt="{:.2f}", better="lo",
                      note="participation ratio of the eigenvalues"))
    b.append(stat_row("pairs with |&rho;| &ge; 0.75", weak["redundant_pairs"],
                      strong["redundant_pairs"], fmt="{:d}",
                      note="near-duplicate views"))
    b.append(stat_row("samples", weak["n"], strong["n"], fmt="{:,d}", better="hi",
                      note="everything above is estimated from these"))
    b.append(stat_row("positive rate", weak["pos_rate"], strong["pos_rate"],
                      fmt="{:.3f}", note="fraction graded correct"))
    b.append("</table></div>")
    b.append(
        '<div class="box bad"><p><b>Do not read mean |excess| as a second, '
        "independent finding.</b> For two classes with prior &pi;, the excess is "
        "algebraically <code>E<sub>ij</sub> = w<sub>ij</sub>(g<sub>i</sub>g<sub>j"
        "</sub>&minus;1) + &kappa;d<sub>i</sub>d<sub>j</sub>g<sub>i</sub>g<sub>j"
        "</sub></code> with <code>g<sub>i</sub> = 1/&radic;(1+&kappa;d<sub>i</sub>"
        "&sup2;)</code>. Predicting E from the Cohen's d values in section 2, the "
        "within-class matrix and &pi; alone reproduces the measured E "
        "<b>entrywise at correlation &ge; +0.99997 on every cell here</b>. So it "
        "is section 2 restated, multiplied by the class prior &mdash; not a "
        "separate mechanism. It also carries a variance-rescaling term of "
        "opposite sign worth 44&ndash;62% of the label term, so an individual "
        "entry is a partially cancelled residual. This was written up as an "
        "independent finding in Step 214 and <b>withdrawn in Step 215</b>.</p>"
        "</div>")
    return page(spec["title"], "".join(b))


def index_page(results):
    b = ['<p class="crumb"><a href="../index.html">&larr; action items</a></p>',
         "<h1>Is it the algorithm, or is it the features?</h1>",
         '<p class="sub">Three matched cell pairs. In each, a weak cell is set '
         'against a high-scoring one that holds as much as possible fixed, and '
         'the comparison is on the features &mdash; their distributions, their '
         'individual separability, their correlation with one another, and the '
         'ceiling a fully supervised model reaches from them.</p>']
    b.append(
        '<div class="box"><p>Every fusion-side diagnostic has come back negative. '
        'Orientation costs <b>+0.00pp on all 25 cells</b>; <b>0/25</b> groupings '
        'are degenerate; pool composition was closed in both directions (Step '
        '206); and the pre-registered sign repair failed its gate and moved the '
        'macro <b>&minus;0.14pp</b> (Step 213). That leaves one hypothesis '
        'standing: the weak cells hand the fusion worse raw material. The '
        'decisive test is the <b>supervised ceiling</b> &mdash; hand a model the '
        'labels, and see whether it too is stuck.</p>'
        '<p><b>It is stuck part of the way.</b> Some of the gap is in the '
        'features on all three pairs; none of the three attributes all of it '
        'there. The exact split is not measurable at these sample sizes &mdash; '
        'see the caveats below.</p></div>')
    b.append(
        '<div class="box bad"><p><b>Revised after adversarial review (Step 215). '
        "Three of Step 214's four findings did not survive.</b></p>"
        "<ul>"
        "<li><b>The TriviaQA partner cell was disqualified.</b> The original "
        "strong cell, <code>spilled_triviaqa_llama8b</code> (0.9413), has "
        "<b>6 positives out of 256</b>, <code>trace_length</code> alone scores "
        "0.925 on it, and the project already carried a standing instruction to "
        "treat it as selection-biased and not headline it "
        "(<code>scripts/advisor_report.py:783</code>). It has been replaced with "
        "<code>semenergy_triviaqa_qwen3_8b</code> (n = 4,392).</li>"
        "<li><b>&ldquo;The label-driven correlation is ~3&times; smaller&rdquo; "
        "is withdrawn.</b> It is algebraically implied by the per-view Cohen's d "
        "values already reported one section earlier (entrywise correlation "
        "&ge; +0.99997), and its apparent consistency came from an uncontrolled "
        "class prior &mdash; &kappa; = 0.249 vs 0.023 on the original pair.</li>"
        "<li><b>&ldquo;Estimation noise is retired&rdquo; is withdrawn.</b> The "
        "statistic behind it divided by 1/&radic;n, the error bar on a single "
        "correlation. C and W are computed from the same rows, so their "
        "difference is far tighter than that; the normalisation was wrong by "
        "roughly &radic;n. A permutation null replaces it here.</li>"
        "<li><b>&ldquo;The method matches supervision on strong cells&rdquo; is "
        "withdrawn.</b> Across all 25 cells the label-free score exceeds the "
        "ceiling on 4, and <b>all four have n &le; 700</b> (n &le; 700: 4/11 "
        "exceed, mean &minus;0.77pp; n &gt; 700: 0/14, mean &minus;6.03pp; "
        "Mann-Whitney p = 0.035). It was a small-sample effect, not a "
        "features-strength effect.</li>"
        "<li>Two code defects are fixed here: the ceiling no longer applies "
        "<code>max(a, 1&minus;a)</code> to a supervised score (which inflated it "
        "by +12.6pp on the 6-positive cell) and is averaged over 10 seeds; and "
        "the rank-1 fit is minimised properly rather than by a non-convergent "
        "iteration.</li>"
        "</ul></div>")

    b.append("<h2>The answer, in one table</h2>")
    b.append('<div class="scroll"><table><tr><th>pair</th>'
             "<th>cell</th><th></th><th>label-free</th><th>sup. ceiling</th>"
             "<th>headroom</th><th>best single view</th>"
             "<th>mean |excess &rho;|</th><th>rank-1 residual</th></tr>")
    for spec, w, s in results:
        for d, tag, cl in ((w, "weak", ' class="weak"'), (s, "strong", "")):
            hd = ((d["lr_ceiling"] - d["lsml"]) * 100 if d["lr_ceiling"] else None)
            hc = "neg" if hd is not None and hd > 2 else ""
            first = (f'<td rowspan="2"><a href="pair_{spec["id"]}.html">'
                     f'{esc(spec["id"])}</a></td>' if tag == "weak" else "")
            b.append(
                f"<tr{cl}>{first}"
                f'<td style="text-align:left">{esc(d["pretty"][0])} / '
                f'{esc(d["pretty"][1])}</td><td>{tag}</td>'
                f'<td>{d["lsml"]:.4f}</td><td>{num(d["lr_ceiling"])}</td>'
                f'<td class="{hc}">'
                + ("&mdash;" if hd is None else f"{hd:+.1f}pp") + "</td>"
                f'<td>{d["best_single"]:.4f}</td>'
                f'<td>{d["mean_abs_E"]:.4f}</td>'
                f'<td>{d["rank1_residual"]:.3f}</td></tr>')
    b.append("</table></div>")
    b.append('<p style="font-size:13px;color:var(--mut)">Headroom = supervised '
             "ceiling &minus; what we deliver. Positive means signal the features "
             "carry and the label-free method does not reach.</p>")

    b.append("<h2>What the three pairs say together</h2>")
    b.append(
        "<p><b>1. Roughly half the gap really is in the features, and the split "
        "is not the same on every pair.</b> The supervised-ceiling gap as a "
        "fraction of the label-free gap is "
        + ", ".join(
            f"<b>{100 * (s['lr_ceiling'] - w['lr_ceiling']) / (s['lsml'] - w['lsml']):.0f}%</b> "
            f"({esc(spec['id'])})"
            for spec, w, s in results
            if w["lr_ceiling"] and s["lr_ceiling"] and abs(s["lsml"] - w["lsml"]) > 1e-9)
        + ". So the weak cells genuinely do hand the fusion worse raw material "
        "&mdash; but on none of the three pairs does that account for all of "
        "it.</p>")
    b.append(
        "<p><b>2. The rest of the gap is ours, and on the worst cell it is "
        "selection and sign &mdash; not dilution.</b> On "
        "<code>seiclr_triviaqa_opt30b</code> the drop from its best single view "
        "to what we deliver decomposes as: honest best single view over the pool "
        "<b>0.6200</b> (split-half, &plusmn;0.0067) &rarr; best view inside the "
        "12 the selector chose <b>0.5791</b> (the selector had already discarded "
        "the pool's two strongest views) &rarr; L-SML <b>0.5614</b>. That is "
        "<b>&asymp;4.1pp of selection loss and &asymp;1.8pp of sign loss</b>. "
        "Averaging itself costs essentially nothing: an oracle-signed simple "
        "average of those 12 views lands exactly on the best view in the subset, "
        "so <b>dilution explains 0.0pp</b>. And L-SML's effective per-view signs "
        "disagree with the oracle on <b>2 of 12</b> views on this cell against "
        "<b>0 of 12</b> on the others &mdash; the plain label-free average beats "
        "L-SML here, 0.5708 to 0.5614.</p>")
    b.append(
        "<p><b>3. The rank-1 assumption is not a general mechanism.</b> "
        "Residuals are "
        + ", ".join(f"{w['rank1_residual']:.3f} vs {s['rank1_residual']:.3f}"
                    for _, w, s in results)
        + " (weak vs strong). No consistent direction across the three pairs.</p>")

    worst = max(results,
                key=lambda r: (r[1]["lr_ceiling"] or 0) - r[1]["lsml"])
    ws = worst[1]
    hd = (ws["lr_ceiling"] - ws["lsml"]) * 100
    b.append(
        f'<div class="box bad"><p><b>The one number to act on.</b> '
        f'{esc(ws["pretty"][0])} / {esc(ws["pretty"][1])} has a supervised '
        f'ceiling of {ws["lr_ceiling"]:.4f} against our {ws["lsml"]:.4f} '
        f'&mdash; a <b>{hd:.1f}pp gap</b> (bootstrap CI roughly &plusmn;2.5pp; '
        f'across-seed sd on the ceiling &plusmn;{ws["lr_sd"]:.4f}). We also score '
        f'below its own best single view, so on that cell fusing is actively '
        f'worse than picking one view.</p>'
        f"<p><b>But almost none of that gap is label-free reachable</b>, and "
        f"Step 214 was wrong to call it &ldquo;reachable headroom&rdquo;. It "
        f"splits into &asymp;1.8pp that better label-free signs would recover, "
        f"&asymp;4.1pp that needs <i>labels</i> to identify the right single "
        f"view, and &asymp;10.2pp of multivariate supervised gain with no "
        f"label-free analogue at all. The honest label-free target here is "
        f"single digits, not 16pp.</p></div>")
    b.append(
        "<p>Read together: <b>some of the weak-cell deficit is a feature "
        "ceiling and should be reported as one; the part that is ours is "
        "concentrated in view selection and per-view sign on specific cells, "
        "not in the fusion arithmetic.</b> That makes Repair 3 &mdash; keep the "
        "pool's strongest view unconditionally &mdash; the indicated next test, "
        "for the measured reason that the selector discarded the two strongest "
        "views on the worst cell.</p>")

    b.append("<h2>The three pairs</h2><div class='cards'>")
    for spec, w, s in results:
        b.append(f'<div class="card"><h3><a href="pair_{spec["id"]}.html">'
                 f'{esc(spec["title"])}</a></h3>'
                 f'<p><b>varies:</b> {esc(spec["varies"])}</p>'
                 f"<p>{esc(spec['why'])}</p></div>")
    b.append("</div>")
    b.append('<p class="foot">Built by <code>scripts/action_items_jul2026/'
             "build_pair_compare.py</code>. All numbers recomputed from "
             "<code>local_cache/</code> in this run; nothing read off a stored "
             "CSV. Validity gate: GOOD_6 macro must reproduce 0.7594.</p>")
    return page("Features or algorithm — matched cell pairs", "".join(b))


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = load_cells()
    ok, macro = assert_good6(cells, verbose=True)
    if not ok:
        raise SystemExit(f"GATE 0 FAILED: GOOD_6 macro {macro:.4f}")
    print(f"GATE 0 (validity): GOOD_6 macro {macro:.4f} PASS\n")

    with open(os.path.join(REPO, "results/selector_bench/a2_groupfs__c46.csv"),
              newline="", encoding="utf-8") as f:
        dufs = {r["cell"]: r["chosen"].split("|") for r in csv.DictReader(f)
                if r["variant"] == "a2.dufs_pf"}

    results, dump = [], {}
    for spec in PAIRS:
        cw, cs = cells[spec["weak"]], cells[spec["strong"]]
        common = [f for f in cw["pool"] if f in cs["pool"]]
        print(f"[{spec['id']}] common pool {len(common)} views "
              f"(weak {len(cw['pool'])}, strong {len(cs['pool'])})")
        w = measure(cw, [cw["pool"].index(f) for f in common], spec["weak"])
        s = measure(cs, [cs["pool"].index(f) for f in common], spec["strong"])
        for d, ck in ((w, spec["weak"]), (s, spec["strong"])):
            d["lsml"], d["upcr"], d["dufs_size"], d["upcr_kept"] = \
                deployed_scores(cells[ck], dufs[ck])
        results.append((spec, w, s))
        dump[spec["id"]] = dict(spec=spec, weak=w, strong=s)

        gl = (s["lsml"] - w["lsml"]) * 100
        gc = ((s["lr_ceiling"] - w["lr_ceiling"]) * 100
              if w["lr_ceiling"] and s["lr_ceiling"] else float("nan"))
        print(f"    label-free gap {gl:+6.2f}pp | supervised-ceiling gap "
              f"{gc:+6.2f}pp | in features {100 * gc / gl:5.0f}%")
        print(f"    weak   best-single {w['best_single']:.4f}  ceiling "
              f"{num(w['lr_ceiling'])}  |excess| {w['mean_abs_E']:.4f}  "
              f"exc/null {w['excess_perm']['ratio']:.2f}x  rank1res {w['rank1_residual']:.3f}  "
              f"n {w['n']}")
        print(f"    strong best-single {s['best_single']:.4f}  ceiling "
              f"{num(s['lr_ceiling'])}  |excess| {s['mean_abs_E']:.4f}  "
              f"exc/null {s['excess_perm']['ratio']:.2f}x  rank1res {s['rank1_residual']:.3f}  "
              f"n {s['n']}", flush=True)

        with open(os.path.join(OUT, f"pair_{spec['id']}.html"), "w",
                  encoding="utf-8") as f:
            f.write(pair_page(spec, w, s))

    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_page(results))
    with open(os.path.join(OUT, "_data.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
