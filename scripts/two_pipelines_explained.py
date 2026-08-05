#!/usr/bin/env python
"""
two_pipelines_explained.py — how the two label-free arms actually work.

WHY THIS EXISTS
---------------
`results/action_items/labelfree_standing.html` is the SCOREBOARD: where the two arms
land against GOOD_6 and against the published roster. It says nothing about how
either one works. This page is the companion explainer, written for a reader who
knows the theory but has never seen our code:

  * U-PCR + sign(rho)            (Dror, Nadler, Bilal & Kluger, arXiv:1703.02965)
  * DUFS parameter-free + L-SML  (Lindenbaum et al. NeurIPS 2021; Jaffe et al. 2016)

Every equation is labelled with its paper reference, every claim carries a measured
number, and every place our implementation departs from the source is listed with
what the departure is worth.

NUMBER PROVENANCE
-----------------
Three classes, and the page marks which is which:
  FILE   read live from results/ (orientation summary, selector bench, standings)
  SESSION ablations run on 2026-07-29/30 against local_cache via the canonical
         prepare_cell -> lsml_continuous / upcr_fit -> anchor_orient -> raw AUROC path.
         Recompute with the scripts named in ABLATION_PROVENANCE below.
  PAPER  quoted from papers/extracted/*.md

Nothing here is recomputed at build time, so the page renders in under a second and
cannot drift from the scoreboard.

Usage:
    python scripts/two_pipelines_explained.py
Out:
    results/action_items/two_pipelines_explained.html
"""
import csv
import html
import json
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(REPO, "results", "action_items", "two_pipelines_explained.html")

ABLATION_PROVENANCE = (
    "Session ablations, 2026-07-29/30. Each re-ran the canonical path "
    "(subset_sweep.prepare_cell over CANONICAL_POOL -> upcr.upcr_fit or "
    "fusion_utils.lsml_continuous -> streaming_utils.anchor_orient -> raw "
    "roc_auc_score) over the 25 in-scope cells from scripts/inscope_cells.py, "
    "with the deployed U-PCR configuration held fixed at "
    "loss='l2', exclusion=True, difficulty_gate=False, simple_avg_fallback=True, "
    "recompute_after_exclusion=True, g2_projection_k=1, scale_ratio=0.25."
)


# ── numbers ──────────────────────────────────────────────────────────────────
def read_orientation():
    p = os.path.join(REPO, "results/upcr_study/06_orientation/summary.json")
    with open(p) as f:
        return json.load(f)


def read_dufs_bench():
    """a2.dufs_pf and a2.dufs, restricted to the in-scope roster."""
    from inscope_cells import INSCOPE
    rows = [r for r in csv.DictReader(open(
        os.path.join(REPO, "results/selector_bench/a2_groupfs__c46.csv"), newline=""))
        if r["cell"] in INSCOPE]
    out, freq = {}, {}
    for v in ("a2.dufs_pf", "a2.dufs", "a2.select"):
        d = {r["cell"]: r for r in rows if r["variant"] == v}
        if len(d) < len(INSCOPE):
            continue
        au = [float(d[c]["auroc"]) for c in INSCOPE]
        rm = [float(d[c]["rand_med"]) for c in INSCOPE]
        sz = [float(d[c]["size"]) for c in INSCOPE]
        K = [float(d[c]["K"]) for c in INSCOPE]
        out[v] = dict(macro=st.mean(au), size=st.mean(sz), K=st.mean(K),
                      size_lo=min(sz), size_hi=max(sz),
                      d_rand=st.mean(a - r for a, r in zip(au, rm)),
                      w=sum(1 for a, r in zip(au, rm) if a > r),
                      l=sum(1 for a, r in zip(au, rm) if a < r))
        if v == "a2.dufs_pf":
            for c in INSCOPE:
                for f in d[c]["chosen"].split("|"):
                    freq[f] = freq.get(f, 0) + 1
    return out, freq


def read_standings():
    p = os.path.join(REPO, "results/benchmark_standing.csv")
    if not os.path.exists(p):
        return {}
    rows = list(csv.DictReader(open(p, newline="")))
    out = {}
    for col in ("upcr.rho", "a2.dufs", "ref.GOOD_6", "ref.GOOD_5"):
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
        if vals:
            out[col] = st.mean(vals)
    return out


# SESSION ablations. See ABLATION_PROVENANCE.
ABL = {
    # U-PCR: what carries the result
    "upcr_deployed":      0.7551,
    "upcr_unif_weights":  0.7469,
    "upcr_no_exclusion":  0.7545,
    "upcr_no_both":       0.7329,
    "upcr_v1_only":       0.7541,
    "upcr_v1v2_unit":     0.7137,
    "upcr_lsml_on_kept":  0.7536,
    "cos_w_v1":           0.9888,
    "cos_w_v1_min":       0.9606,
    "keep_mean":          21.00,
    # g2
    "g2_sr025_macro":     0.7551,
    "g2_sr100_macro":     0.7539,
    "g2_sr025_pinned":    21,
    "g2_sr100_pinned":    0,
    "g2_abs_025":         0.2338,
    "g2_abs_100":         0.4322,
    "g2_abs_200":         0.4460,
    "g2_k1":              0.7551,
    "g2_k2":              0.7541,
    # scaling
    "scale_z_upcr":       0.7551, "scale_z_g6":      0.7594, "scale_z_diag":  1.000,
    "scale_mm_upcr":      0.7417, "scale_mm_g6":     0.7528, "scale_mm_diag": 0.112,
    "scale_mmnc_upcr":    0.7050, "scale_mmnc_g6":   0.7528,
    # L-SML
    "lsml_g6":            0.7594, "avg_g6":     0.7592, "best1_g6":     0.7523,
    "lsml_pf":            0.7507, "avg_pf":     0.6830, "best1_pf":     0.7559,
    "lsml_full":          0.7457, "avg_full":   0.6267, "best1_full":   0.7593,
    "oracleK_g6":         0.7610, "oracleK_full": 0.7573,
    "K_hit_g6":           14, "K_hit_full": 6,
    "K_rho_g6":          -0.411, "K_rho_full": -0.198,
    # sign recovery
    "sign_vs_oracle":     0.916,
    # prepare_cell funnel
    "pool_mean":          28.72, "pool_lo": 27, "pool_hi": 30, "pool_full30": 6,
    "n_saturated":        26, "n_constant": 1, "n_imputed": 0,
    # grouping vs families
    "fam_same":           0.265, "fam_diff": 0.153,
}


# ── html helpers ─────────────────────────────────────────────────────────────
def esc(s):
    return html.escape(str(s), quote=True)


def lin(v, d0, d1, r0, r1):
    if d1 == d0:
        return r0
    return r0 + (v - d0) * (r1 - r0) / (d1 - d0)


def hbar(labels, values, axis, d0=None, d1=None, fmt="{:.4f}", hi=None, width=880):
    """Horizontal bar chart. `hi` = set of indices to highlight."""
    hi = hi or set()
    L, X0, X1, ROW, TOP = 300, 310, width - 70, 30, 12
    n = len(labels)
    H = TOP + n * ROW + 40
    d0 = min(values) * 0.985 if d0 is None else d0
    d1 = max(values) * 1.004 if d1 is None else d1
    s = [f'<svg viewBox="0 0 {width} {H}" width="100%" role="img">']
    ticks = 5
    for t in range(ticks + 1):
        v = d0 + (d1 - d0) * t / ticks
        x = lin(v, d0, d1, X0, X1)
        s.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP+n*ROW}" '
                 f'class="gx"/>')
        s.append(f'<text x="{x:.1f}" y="{TOP+n*ROW+15}" class="tk" '
                 f'text-anchor="middle">{fmt.format(v)}</text>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        cy = TOP + i * ROW + ROW / 2
        s.append(f'<text x="{L-8}" y="{cy+4:.1f}" class="rl" text-anchor="end">'
                 f'{esc(lb)}</text>')
        x = lin(v, d0, d1, X0, X1)
        cls = "bh" if i in hi else "bn"
        s.append(f'<rect x="{X0}" y="{cy-8:.1f}" width="{max(x-X0,1):.1f}" height="16" '
                 f'class="{cls}"/>')
        s.append(f'<text x="{x+7:.1f}" y="{cy+4:.1f}" class="bl">{fmt.format(v)}</text>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-6}" class="ax" text-anchor="middle">'
             f'{esc(axis)}</text>')
    s.append("</svg>")
    return "".join(s)


def funnel(stages, width=880):
    """stages = [(label, value, note)] descending."""
    X0, W, TOP, ROW = 250, width - 330, 14, 62
    n = len(stages)
    H = TOP + n * ROW + 10
    mx = max(v for _, v, _ in stages)
    s = [f'<svg viewBox="0 0 {width} {H}" width="100%" role="img">']
    for i, (lb, v, note) in enumerate(stages):
        y = TOP + i * ROW
        w = max(W * v / mx, 30)
        x = X0 + (W - w) / 2
        s.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="38" rx="5" '
                 f'class="fn{min(i,4)}"/>')
        s.append(f'<text x="{x+w/2:.1f}" y="{y+24}" class="fv" text-anchor="middle">'
                 f'{v:g}</text>')
        s.append(f'<text x="{X0-14}" y="{y+17}" class="rl" text-anchor="end">'
                 f'{esc(lb)}</text>')
        s.append(f'<text x="{X0-14}" y="{y+31}" class="rs" text-anchor="end">'
                 f'{esc(note)}</text>')
        if i < n - 1:
            s.append(f'<path d="M {X0+W/2} {y+40} l -5 -5 h 10 z" class="fa"/>')
    s.append("</svg>")
    return "".join(s)


def fig(cap, sub, svg, note=""):
    out = ['<figure class="fg">', f'<figcaption><b>{cap}</b>']
    if sub:
        out.append(f'<span class="fsub">{sub}</span>')
    out.append("</figcaption>")
    out.append(svg)
    if note:
        out.append(f'<p class="fnote">{note}</p>')
    out.append("</figure>")
    return "".join(out)


def table(headers, rows, numeric=(), cls=""):
    h = "".join(f'<th class="{"num" if i in numeric else ""}">{c}</th>'
                for i, c in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f'<td class="{"num" if i in numeric else ""}">{c}</td>'
                      for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return f'<table class="{cls}"><tr>{h}</tr>{"".join(body)}</table>'


def code(txt):
    return f'<pre class="cd">{esc(txt)}</pre>'


# ── page ─────────────────────────────────────────────────────────────────────
def main():
    O = read_orientation()
    D, FREQ = read_dufs_bench()
    S = read_standings()
    pf = D.get("a2.dufs_pf", {})
    du = D.get("a2.dufs", {})

    P = []
    A = P.append

    # ---------------------------------------------------------------- header
    A('<div class="hero"><div class="hw">')
    A('<h1>How the two label-free arms actually work</h1>')
    A('<div class="sub">A mechanism walkthrough of <b>U-PCR + sign(rho)</b> and '
      '<b>DUFS parameter-free + L-SML</b>, as implemented — pseudo-algorithm, block '
      'by block, every deviation from the source papers, and a measurement of which '
      'parts of each pipeline actually carry the result.</div>')
    A('<div class="pills">'
      '<span class="pill">Companion to labelfree_standing.html, which is the scoreboard</span>'
      '<span class="pill">25 in-scope cells &#183; 10 QA + 15 math</span>'
      '<span class="pill">Generated by scripts/two_pipelines_explained.py</span>'
      '</div></div></div>')
    A('<div class="page">')

    # ---------------------------------------------------------------- tl;dr
    A('<div class="card">')
    A('<span class="badge bfind">Read this first</span>')
    A('<h2>The one-paragraph version</h2>')
    A('<p>Both arms share the same skeleton and differ in one middle block. Both end with '
      'a single hand-set bit. The interesting finding is not that either selector wins &mdash; '
      'neither does &mdash; but <b>which part of each pipeline is load-bearing</b>. We ablated '
      'every component. Selection contributes almost nothing in either arm, and so does the '
      'weight estimate. What carries the number is <b>label-free orientation</b> and <b>a '
      'dependency-aware fusion over a moderately-sized pool</b>. That is why eight algorithm '
      'families all converged to roughly the same 75%.</p>')

    A('<div class="kv">')
    for lab, val, note in (
            ("U-PCR + sign(rho)", O["macro_rho_anchor"], "macro over 25 cells"),
            ("DUFS param-free + L-SML", pf.get("macro", float("nan")), "macro over 25 cells"),
            ("GOOD_6 (hand-picked)", ABL["lsml_g6"], "the bar to clear"),
    ):
        A(f'<div class="kvi"><div class="kvn">{val*100:.1f}%</div>'
          f'<div class="kvl">{esc(lab)}<br><span class="dim">{esc(note)}</span></div></div>')
    A(f'<div class="kvi"><div class="kvn">1</div><div class="kvl">hand-set bits left '
      f'in either arm<br><span class="dim">the global anchor direction</span></div></div>')
    A('</div>')
    A('</div>')

    # ---------------------------------------------------------------- substrate
    A('<div class="card">')
    A('<span class="badge bdone">Shared substrate</span>')
    A('<h2>1. How a cell becomes <span class="mono">(V, anchor, labels)</span></h2>')
    A('<p>A <b>cell</b> is one (dataset, model, decoding config) pair &mdash; 25 of them, '
      'frozen in <span class="mono">scripts/inscope_cells.py</span>. Every arm below consumes '
      'the output of a single function, <span class="mono">subset_sweep.prepare_cell</span>, '
      'which is the only place a feature matrix is born.</p>')
    A(code("""for each feature f in CANONICAL_POOL (46 names, frozen bit order):
    if f not in cell:            drop 'missing'
    if length mismatch:          drop 'length-mismatch'
    if all non-finite:           drop 'all-nonfinite'
    if some non-finite:          median-impute those entries
    if std < 1e-8:               drop 'constant'
    if >40% ties on one value:   drop 'saturated'
    keep:  column = zscore(arr * ALL_SIGNS[f])        <-- hand signs applied HERE

V      = column_stack(columns)                        # (n, p)
anchor = first available of [epr, low_band_power, spectral_entropy, cusum_max],
         oriented and z-scored
rho    = |Spearman(V)|                                # (p, p)   <-- NOT used by either arm
return None if p < 3 or labels are single-class"""))

    A('<div class="grid2">')
    A('<div><h3>What comes out, and who reads it</h3>')
    A(table(["Output", "Shape", "Consumed by"], [
        ['<span class="mono">V</span>', "n &times; p, z-scored, sign-oriented", "<b>both arms</b>"],
        ['<span class="mono">anchor</span>', "n", "<b>both arms</b>, final step only"],
        ['<span class="mono">pool</span>', "p names", "<b>both arms</b>"],
        ['<span class="mono">labels</span>', "n", "<b>AUROC only</b> &mdash; never selection or fusion"],
        ['<span class="mono">rho</span>', "p &times; p Spearman", '<span class="dim">neither arm</span>'],
    ]))
    A('</div>')
    A('<div><h3>Two traps in that listing</h3>'
      '<p><b>V arrives already hand-oriented.</b> Line 383 applies '
      '<span class="mono">ALL_SIGNS</span>, so both U-PCR call sites multiply by '
      '<span class="mono">hand</span> first to <em>recover</em> the raw view before deriving '
      'polarity themselves. This works because <span class="mono">zscore(raw&#183;s) == '
      's&#183;zscore(raw)</span> for s = &plusmn;1. Getting it backwards silently scores the '
      'incumbent arm instead of the new one; both files carry a warning comment.</p>'
      '<p><b>Two different objects are called "rho".</b> See the box below &mdash; this one '
      'is not the one in the arm\'s name.</p></div>')
    A('</div>')

    A('<div class="warn"><b>Name collision, worth killing on sight.</b> '
      '<span class="mono">cell.rho</span> is a <b>p&times;p |Spearman| matrix</b> between '
      'features &mdash; observable, and read by <em>neither</em> of these two arms. '
      'U-PCR\'s <span class="mono">rho_hat</span> is a <b>length-m vector</b>, '
      '<span class="mono">rho_i = Cov(f_i, Y)</span> &mdash; each view against the '
      '<em>unobservable target</em>. <b>"U-PCR + sign(rho)" means the second one.</b> '
      'Everywhere below, "rho-hat" is the vector and "the Spearman matrix" is the matrix.</div>')

    A('<h3>Where the views go: three separate shrink steps, routinely conflated</h3>')
    A(fig("The pool funnel",
          "46 canonical names down to what each selector actually fuses",
          funnel([
              ("CANONICAL_POOL", 46, "frozen bit order, append-only"),
              ("live views", 30, "16 RAG/GPQA-era views never populate in-scope"),
              ("after prepare_cell", ABL["pool_mean"],
               f"mean; range {ABL['pool_lo']}-{ABL['pool_hi']}, only "
               f"{ABL['pool_full30']}/25 cells carry all 30"),
              ("U-PCR keeps", ABL["keep_mean"], "its Algorithm-1 exclusion step"),
              ("DUFS-PF keeps", pf.get("size", 0), "gates open, mu > 0"),
          ]),
          f"The quality filters are near-inert: <b>{ABL['n_saturated']} saturated + "
          f"{ABL['n_constant']} constant drops across all 25 cells combined</b> "
          f"(~1.3 views per cell), and <b>{ABL['n_imputed']} imputations anywhere</b>. "
          "Repeat offenders: min_spilled (9 saturated + 1 constant), stft_max_high_power (5), "
          "stft_spectral_entropy (5), trace_length (5), dominant_freq (2). The stft views "
          "saturate only on QA cells and trace_length only on math &mdash; short answers have "
          "too few tokens for the STFT bands to vary, and MATH-500 traces bunch at the token "
          "cap. Selection happens in the last two rows, not the first three."))

    A('<h3>The 30 live views</h3>')
    A(table(["Block", "n", "Source signal", "New at the cluster re-run?"], [
        ["H(n) spectral / STFT / time-domain", 16, "token entropy series", "no &mdash; the original pool"],
        ["spilled energy &Delta;E(n) = &minus;log p(sampled)", 4, "sampled-token logprob", "<b>yes</b>"],
        ["energy Z(n), full-vocab log-partition", 4, "<span class='mono'>token_logsumexp</span>", "<b>yes</b>"],
        ["top-K statistics (incl. <span class='mono'>varentropy</span>)", 6, "saved top-50 logprobs", "<b>yes</b>"],
    ], numeric=(1,)))
    A('<p class="dim">16 &rarr; 30 is exactly the AIRCC re-run: the Colab-era caches stored only '
      '<span class="mono">token_entropies</span>, so the last 14 were <em>not computable at '
      'all</em>, not merely unused.</p>')
    A('</div>')

    # ---------------------------------------------------------------- the rho question
    A('<div class="card">')
    A('<span class="badge bfind">The central mechanism</span>')
    A('<h2>2. How <span class="mono">rho_i = Cov(f_i, Y)</span> is estimated without ever seeing Y</h2>')
    A('<p>This looks circular and is not. It is Theorem 1 of Dror, Nadler, Bilal &amp; Kluger '
      '(arXiv:1703.02965). <b>U-PCR never uses Y. It uses the footprint Y leaves on the '
      'covariance between views &mdash; and that footprint is observable.</b> '
      '<span class="mono">rho</span> is not measured, it is <em>solved for</em>, from a linear '
      'system whose coefficients are ordinary view-view covariances.</p>')

    A('<h3>The model (&sect;4.1)</h3>')
    A('<p>Let <span class="mono">g(x) = E[Y|X=x]</span> be the best predictor anyone could '
      'build. Write each view as <span class="mono">f_i(x) = g(x) + h_i(x)</span>. Define '
      '<span class="mono">g2 = E[g&sup2;] = E[g&#183;Y]</span> (the <b>shared signal</b>) and '
      '<span class="mono">a_i = E[h_i&#183;Y] = E[h_i&#183;g]</span>. Then immediately:</p>')
    A(code("rho_i = E[g Y] + E[h_i Y] = g2 + a_i                              (Eq. 11)"))
    A('<p>and the single assumption &mdash; the regression analogue of Dawid&ndash;Skene:</p>')
    A(code("E[h_i(X) h_j(X)] = 0   for i != j        \"uncorrelated deviations\"   (Eq. 13)"))
    A('<p>Now compute the <b>observable</b> covariance between two different views:</p>')
    A(code("""C_ij = E[f_i f_j] = E[(g + h_i)(g + h_j)]
     = E[g^2] + E[h_i g] + E[g h_j] + E[h_i h_j]
     =   g2   +    a_i   +    a_j   +     0            <-- Eq. 13 kills the last term

C_ij = g2 + a_i + a_j                                                    (Eq. 14)"""))
    A('<p>The left side is computable from the data matrix alone. So <b>each pair of views '
      'gives one linear equation</b> in the unknowns. With m views that is m(m&minus;1)/2 '
      'equations for m unknowns &mdash; over-determined at <b>m &ge; 3</b>, which is exactly '
      'where <span class="mono">upcr_fit</span> raises '
      '<span class="mono">"U-PCR needs m >= 3 (Assumption A4)"</span>.</p>')

    A('<div class="info"><b>The whole method is matrix completion.</b> Treat the optimal '
      'predictor <span class="mono">g</span> as an (m+1)-th view we never observe. In the full '
      '(m+1)&times;(m+1) covariance of (f_1 &hellip; f_m, g), the <b>missing last column IS '
      'rho</b> (since Cov(f_i, g) = rho_i) and the <b>missing corner IS g2</b> = Var(g). '
      'Assumption (13) forces C_ij = rho_i + rho_j &minus; g2, so the observed block determines '
      'the missing column <b>up to one scalar</b> &mdash; and Eq. 20 supplies that scalar.</div>')

    A('<h3>The identity that makes it concrete (m = 3)</h3>')
    A(code("""C_12 = g2 + a_1 + a_2
C_13 = g2 + a_1 + a_3          add the first two, subtract the third:
C_23 = g2 + a_2 + a_3          C_12 + C_13 - C_23 = 2 a_1 + g2

              C_12 + C_13 - C_23 + g2
    rho_1  =  -----------------------
                        2"""))
    A('<p>Three view-view correlations and one scalar, and you have view 1\'s covariance with '
      'the unobservable target. Verified on data generated from the model: the closed form '
      'gives <b>0.24666</b> against a true <b>0.25000</b>. For m &gt; 3 it is the same system '
      'solved by least squares over all pairs &mdash; '
      '<span class="mono">additive_design</span> + <span class="mono">solve_additive</span>, '
      'each design row being <span class="mono">e_i + e_j</span>.</p>')

    A('<h3>Why this yields orientation &mdash; and why one bit is unrecoverable</h3>')
    A(table(["What you do to the data", "What happens to C", "What happens to rho-hat"], [
        ["<b>Flip one view</b> <span class='mono'>f_i &rarr; &minus;f_i</span>",
         "row/column i negates",
         '<b class="win">rho_hat_i flips sign</b> &mdash; detectable'],
        ["<b>Flip every view</b> <span class='mono'>F &rarr; &minus;F</span>",
         "<b>bit-identical</b> (both indices flip, product unchanged)",
         '<b class="loss">rho_hat bit-identical</b> &mdash; undetectable'],
    ]))
    A(code("""base            rho_hat = [ 0.2459  0.3998 -0.2003  0.1991 -0.2479  0.3002]
flip feature 2  rho_hat = [ 0.2099  0.2889 +0.1415  0.1891 -0.0327  0.2417]   <- entry 2 flipped
flip ALL        rho_hat = [ 0.2459  0.3998 -0.2003  0.1991 -0.2479  0.3002]   <- identical"""))
    A('<p><b>Relative polarity lives in the covariance structure; the global sign provably does '
      'not.</b> Measured on all 25 in-scope cells, <span class="mono">max|&Delta;rho|</span> '
      'under a global flip is exactly <span class="mono">0.000e+00</span>. That single contrast '
      'explains both why per-view polarity is free and why the anchor bit is not.</p>')
    A('<p class="dim">Honest caveat: flipping a view <em>breaks</em> assumption (13) for that '
      'view (E[h\'_i h_j] = &minus;2a_j &ne; 0), which is why the other entries drift slightly '
      'above rather than staying pinned.</p>')

    A('<h3>Does it work on our data?</h3>')
    A(fig("Sign recovery and what it is worth",
          "agreement of sign(rho-hat) with two different references, and the AUROC it buys",
          hbar(
              ["sign(rho-hat) vs ORACLE sign(Cov(f_i, label))",
               "sign(rho-hat) vs our 30 hand-declared signs",
               "chance level for this folded statistic"],
              [ABL["sign_vs_oracle"], O["mean_polarity_agreement"], O["polarity_null_mean"]],
              "fraction of views whose polarity agrees (global bit folded out)",
              d0=0.4, d1=1.0, fmt="{:.3f}", hi={0}),
          f"<b>The estimator disagrees with us and agrees with the truth.</b> Agreement with "
          f"the hand signs is {O['mean_polarity_agreement']*100:.1f}%, which is "
          f"<b>at chance</b> (folded null mean {O['polarity_null_mean']*100:.1f}%, "
          f"p={O['p_hand_vs_null']:.2f}); agreement with the label-derived polarity is "
          f"<b>{ABL['sign_vs_oracle']*100:.1f}%</b>. A whole-pool audit found "
          f"<b>{O['n_features_mis_signed']} of {O['n_features_audited']}</b> hand signs have "
          f"mean oriented AUROC below 0.5. The hand prior was the outlier, not the estimator."))

    A(fig("What the orientation choice is worth",
          "macro AUROC over the 25 in-scope cells",
          hbar(["structure-derived polarity + epr anchor (deployed)",
                "42 hand polarities + epr anchor (the incumbent)",
                "structure-derived polarity + structural global rule (anchor-FREE)"],
               [O["macro_rho_anchor"], O["macro_hand_anchor"], O["macro_rho_majority"]],
               "macro AUROC", d0=0.20, d1=0.80, hi={0}),
          f"sign(rho) over hand signs: <b>+{O['rho_anchor_vs_hand']['mean_delta']*100:.2f}pp, "
          f"{O['rho_anchor_vs_hand']['wins']}W/{O['rho_anchor_vs_hand']['losses']}L, "
          f"p={O['rho_anchor_vs_hand']['p']:.5f}</b>. And the fully anchor-free variant "
          f"collapses to <b>{O['macro_rho_majority']:.4f}</b>, losing "
          f"{O['rho_majority_vs_hand']['losses']}/25 &mdash; the structural global rule agrees "
          f"with the anchor in <b>{O['n_cells_global_rule_matches_anchor']} of 25</b> cells. "
          "The claim is not that we removed every prior. It is that we removed everything "
          "except one bit, and can prove that bit is not removable from this estimator."))
    A('</div>')

    # ---------------------------------------------------------------- upcr pipeline
    A('<div class="card">')
    A('<span class="badge bdone">Arm 1</span>')
    A('<h2>3. U-PCR + sign(rho), end to end</h2>')
    A(code("""INPUT   V (n x p) z-scored + hand-oriented;  anchor;  hand (the signs, only to UNDO them)

 0  F <- (V * hand)^T                      # recover raw z-scored views, shape (p, n)

 -- polarity probe -------------------------------------------------
 1  probe <- UPCR_FIT(F)
 2  pol   <- sign(probe.rho_hat_full);  pol[pol == 0] <- +1
 3  F     <- F * pol[:, None]

 -- the real fit ---------------------------------------------------
 4  res   <- UPCR_FIT(F)
 5  score <- res.w @ F

 -- the one hand-set bit -------------------------------------------
 6  score, flipped <- anchor_orient(score, anchor)


UPCR_FIT(F):
 a  C <- F F^T / n ;  var_y <- 0.25 * mean(diag C)          # scale_ratio = 0.25
 b  n_components <- 2 if lambda2 > 0.1 * trace(C) else 1    # fires on 24/25 cells
 c  A, pairs <- additive_design(m)                          # row for (i,j) is e_i + e_j
    rho0 <- lstsq(A, [C_ij for (i,j) in pairs])             # Eq. 15, solved ONCE at q = 0
 d  for q in linspace(0, var_y, 300):                       # Eq. 16, analytic shift
        rho(q) <- rho0 + q/2
        RES(q) <- ||rho(q) - v1 v1^T rho(q)|| / ||rho(q)||  # Eq. 20, k = 1 projection
    g2 <- argmin RES ;  rho <- rho(g2)
 e  keep <- (rho >= 0.05*var_y) AND (rho >= max(rho)/3)     # Algorithm 1 exclusion
    if |keep| < 3: keep <- the 3 largest rho                # Assumption A4 floor
 f  redo (c)-(d) on F[keep] only                            # "Recalculate ..."
 g  if |keep| < 5: w <- uniform                             # Sec 4.3, fires on 0/25 cells
    else:          w <- sum_c (v_c . rho / lambda_c) v_c    # Eq. 21"""))

    A('<h3>Why the pair equations cannot see g2, and what Eq. 20 does about it</h3>')
    A('<p>Every design row sums to 2, so shifting all of <span class="mono">rho</span> by '
      '<span class="mono">q/2</span> shifts every fitted <span class="mono">C_ij</span> by '
      'exactly <span class="mono">q</span> (Eq. 16). The residual is therefore <b>flat in '
      'q</b> &mdash; verified bit-identical at q = 0, 0.6 and 5.0 '
      '(<span class="mono">9.964335e-03</span> every time). That is why the code solves the '
      'system <em>once</em> and slides it, and why g2 needs its own criterion.</p>')

    A('<div class="warn"><b>Var(Y) is assumed known. g2 is not. They are different quantities.</b> '
      'Algorithm 1\'s input line reads <em>"Predictions f_i(x_j), E[Y] and Var(Y)"</em>, and '
      '&sect;4.2 says explicitly that g2 <em>"would seldom be known to the practitioner"</em>. '
      '<span class="mono">g2 = E[g(X)&sup2;]</span> is the variance of the <b>optimal '
      'predictor</b>; <span class="mono">Var(Y) &minus; g2</span> is the irreducible error. '
      'Var(Y) does exactly one job: it sets the upper end of the g2 search interval. '
      'For us <span class="mono">g2/Var(Y)</span> reads as <b>"how much of correctness is '
      'knowable from the trace at all"</b> &mdash; the paper\'s difficulty index, and the basis '
      'of its STOP rule, which we run off.</div>')

    A('<p><b>What A1 costs us.</b> Moment-matching the z-scored rows to Y\'s moments multiplies '
      'every row by the same constant, and a positive constant leaves eigenvectors fixed while '
      'rho and lambda cancel in Eq. 21 &mdash; so the model\'s <b>accuracy rate p drops out '
      'entirely and never needs to be known</b>. What does <em>not</em> cancel is the ratio '
      '<span class="mono">var_y / mean(diag C)</span>. In the paper A1 pins it; for us it is a '
      'free parameter, set empirically to <b>0.25</b>. So: A1 is free about p, and not free '
      'about scale.</p>')

    A(fig("Is the g2 criterion selecting, or just clipped?",
          "absolute g2-hat found under three grid ceilings, in units where mean(diag C) = 1",
          hbar(["scale_ratio = 0.25 (deployed) — grid ceiling 0.25",
                "scale_ratio = 1.0 (the paper's calibration) — ceiling 1.0",
                "scale_ratio = 2.0 — ceiling 2.0"],
               [ABL["g2_abs_025"], ABL["g2_abs_100"], ABL["g2_abs_200"]],
               "mean g2-hat", d0=0.0, d1=0.5, fmt="{:.3f}", hi={0}),
          f"Doubling the grid from [0,1] to [0,2] moves the answer by <b>0.003</b>. The "
          f"criterion has a stable interior argmin near <b>0.44</b> and is genuinely selecting; "
          f"our cap at 0.25 simply truncates <em>below</em> it, so on "
          f"<b>{ABL['g2_sr025_pinned']} of 25</b> cells the search terminates on the boundary "
          f"(at scale_ratio 1.0 that drops to {ABL['g2_sr100_pinned']}/25). "
          f"<b>The measured cost of the cap is &minus;0.12pp</b> "
          f"({ABL['g2_sr025_macro']:.4f} vs {ABL['g2_sr100_macro']:.4f}) &mdash; the clipped "
          f"version is very slightly better. Say \"pinned, and it costs 0.12pp\", not "
          f"\"the criterion does not work\"."))

    A('<h3>Why v1 alone in Eq. 20, but v1 <em>and</em> v2 in Eq. 21?</h3>')
    A('<p>&sect;4.3 scopes the two-component rule explicitly: <em>"&hellip; '
      '<b>Then, Eq. (21) is replaced by</b> &hellip;"</em>. Eq. 20 is written with v1 alone and '
      'stays that way. Conceptually the two steps do different jobs:</p>')
    A(table(["Step", "Job", "Which subspace, and why"], [
        ["Eq. 20", "<b>model selection</b> &mdash; pick the nuisance parameter q",
         "v1 alone. Lemma 2 gives the O(&epsilon;) prediction that rho lies along v1; the "
         "<em>departure from the v1 line</em> IS the criterion's signal."],
        ["Eq. 21", "<b>estimation</b> &mdash; build the weight vector",
         "v1 and v2. To O(&epsilon;&sup2;) the paper shows rho lies in span{v1, v2}, so "
         "discarding the v2 part loses real signal when lambda2 is non-negligible."],
    ]))
    A('<p><b>A seam worth volunteering before it is found:</b> Eq. 20 chooses q by '
      '<em>minimising</em> the very off-v1 component that Eq. 21 then <em>uses</em>. Not '
      'incoherent &mdash; a nuisance parameter fixed under one approximation, a quantity '
      'estimated under a finer one &mdash; but it is an unresolved asymmetry in the paper, not '
      'something we introduced. Measured, it barely matters: k=1 '
      f'<b>{ABL["g2_k1"]:.4f}</b> vs k=2 <b>{ABL["g2_k2"]:.4f}</b> '
      '(&minus;0.09pp, 2W/3L, p=0.345), and only 5 of 25 cells differ at all &mdash; because '
      'when the argmin sits on the grid boundary, the shape of the curve is irrelevant.</p>')
    A('</div>')

    # ---------------------------------------------------------------- lsml
    A('<div class="card">')
    A('<span class="badge bdone">Shared fusion</span>')
    A('<h2>4. The L-SML core</h2>')
    A('<p>This is arm 2\'s fusion and also GOOD_6\'s, so it is the piece the reference bar and '
      'the DUFS arm share. Paper: Jaffe, Fetaya, Nadler, Jiang &amp; Kluger, <i>Unsupervised '
      'Ensemble Learning with Dependent Classifiers</i> (arXiv:1510.05830).</p>')
    A('<p><b>The problem it exists to solve.</b> Plain SML assumes conditionally independent '
      'errors. Our 30 views violate that about as badly as possible &mdash; they are all '
      'deterministic functions of the <em>same</em> token-level trace. L-SML\'s answer is not '
      'to assume independence but to <b>discover the dependence structure and quotient it '
      'out</b>.</p>')

    A('<h3>Lemma 1 &mdash; the structure that makes it possible</h3>')
    A(code("""              /  v_off_i * v_off_j     if c(i) != c(j)     <- different groups
    r_ij  =  <
              \\  v_on_i  * v_on_j      if c(i) == c(j)     <- same group      (Eq. 10)"""))
    A('<p>The covariance is <b>two rank-one matrices stitched together by the grouping</b>. If '
      'you knew the assignment c, you would know exactly what shape R should have. Everything '
      'downstream is: find the c that makes R look most like this.</p>')

    A(code("""INPUT   views (m arrays, z-scored, sign-oriented);  anchor

 1  R <- cov(X^T)                              # layout pinned contiguous (see note)
 2  S <- score_matrix(R)                       # Eq. 15 dependency detector
 3  for K in 2 .. min(m-1, 8):
        c_K   <- spectral_cluster(S, K)
        Delta_K <- residual(R, c_K)            # Eq. 14 reconstruction error
    K*, c <- argmin over K
    (at m <= 4: skip the above, enumerate ALL set partitions exactly)
 4  for each group g:  xi_g <- sml_fuse_signed(views in g)      # WITHIN-group
 5  fused <- sml_fuse_signed(xi_1 ... xi_K)                     # ACROSS groups
 6  fused <- anchor_orient(fused, anchor)"""))
    A('<p>Step 4 collapses each dependent group into a single <b>virtual classifier</b>. By '
      'construction those K virtual classifiers <em>are</em> conditionally independent, so step '
      '5 is legitimately plain SML. The dependence is not modelled away &mdash; it is absorbed '
      'into the group summaries.</p>')

    A('<h3>What the Eq. 14 residual is actually measuring</h3>')
    A(code("""Delta(v_on, v_off, c) = SUM_{i != j}     1_c(i,j) * (v_on_i  v_on_j  - r_ij)^2
                                    + (1 - 1_c(i,j)) * (v_off_i v_off_j - r_ij)^2     (Eq. 14)"""))
    A('<p>For every off-diagonal pair the grouping makes a <b>prediction</b> about what '
      '<span class="mono">r_ij</span> should be, and Delta is the sum of squared prediction '
      'errors. In code it is three lines: build a <b>two-rank-one mosaic</b> '
      '(<span class="mono">pred = where(same_group, outer(v_on,v_on), outer(v_off,v_off))</span>), '
      'subtract R, square, zero the diagonal, sum. Four things about it that are easy to '
      'get wrong:</p>')
    A('<ul>'
      '<li><b>v_on and v_off are fitted, not given.</b> Delta_K is a <em>profiled</em> residual '
      '&mdash; the best the model can do under that grouping.</li>'
      '<li><b>Masked entries are free parameters, not zeros.</b> Lemma 1 says nothing about '
      'i = j, nor about within-group entries when fitting v_off.</li>'
      '<li><b>The parameter count is 2m regardless of K</b>, so no complexity penalty is needed '
      '&mdash; raising K only reassigns which entries each rank-one pattern explains.</li>'
      '<li><b>K = 1 and K = m both collapse to a single rank-one model of R.</b> Only '
      'intermediate K gets two patterns plus a partition, which is strictly more expressive. '
      'The code caps K at m&minus;1 to exclude the all-singleton end. (Honest wrinkle: at the '
      'extremes one vector goes unconstrained, so mid-range K has slightly more effective '
      'freedom &mdash; the criterion is not perfectly flat across K.)</li>'
      '</ul>')
    A('<p><b>And we are not minimising Eq. 14.</b> Lemma 3 says that is NP-hard. Spectral '
      'clustering on the Eq. 15 score matrix generates one candidate per K, and Delta is used '
      'only to <em>rank those few candidates</em>.</p>')

    A('<h3>Two numerical landmines, both already stepped on</h3>')
    A('<ul>'
      '<li><b>At m = 3, Eq. 15 carries no information at all.</b> The double sum is literally '
      'empty. The vectorised form computed it as a difference of large partial sums and left '
      '~1e-17 of cancellation noise, which flipped a real grouping from [0,0,1] to [1,0,1], '
      'moved the residual 0.00297 &rarr; 0.05284 and the AUROC 0.6023 &rarr; 0.5891. Treat any '
      'size-3 fusion as structurally undetermined.</li>'
      '<li><b>At m = 4 a 5.55e-17 difference in R &mdash; BLAS summation order alone &mdash; '
      'flipped the chosen partition and moved AUROC by 9.7pp.</b> Hence the exact enumeration '
      'of all set partitions at m &le; 4 (Bell(4) = 15), and the pinned memory layout at '
      'step 1.</li>'
      '</ul>')

    A('<h3>What it does on our data</h3>')
    A(table(["Columns fused", "mean size", "L-SML", "plain average", "best single view <i>(oracle)</i>", "K chosen"], [
        ["GOOD_6", "6.0", f"<b>{ABL['lsml_g6']:.4f}</b>",
         f"{ABL['avg_g6']:.4f} <span class='dim'>(+0.02pp, p=0.60)</span>",
         f"{ABL['best1_g6']:.4f} <span class='dim'>(+0.71pp)</span>", "K=2 on 19/25"],
        ["DUFS parameter-free", "16.9", f"<b>{ABL['lsml_pf']:.4f}</b>",
         f"{ABL['avg_pf']:.4f} <span class='win'>(+6.78pp, p&lt;1e-4)</span>",
         f"{ABL['best1_pf']:.4f} <span class='loss'>(&minus;0.51pp)</span>", "K=3&ndash;6"],
        ["full pool", "28.7", f"<b>{ABL['lsml_full']:.4f}</b>",
         f"{ABL['avg_full']:.4f} <span class='win'>(+11.90pp, p&lt;1e-4)</span>",
         f"{ABL['best1_full']:.4f} <span class='loss'>(&minus;1.37pp)</span>", "K=4&ndash;8"],
    ], numeric=(1, 2, 3, 4)))
    A('<div class="take"><b>L-SML\'s value is entirely a function of pool redundancy.</b> On the '
      'six hand-picked views it is <b>statistically indistinguishable from a plain average</b> '
      '(+0.02pp, p=0.60); on 17 views it is worth 6.8pp; on 29, 11.9pp. So the fusion is not '
      'what makes GOOD_6 good &mdash; <b>the selection is</b>. And symmetrically, the '
      'label-free arms genuinely <em>need</em> the fusion, because they keep 17&ndash;21 views '
      'and averaging those collapses.</div>')
    A('<p class="dim">The "best single view" column is <b>label-selected</b>, so it is an oracle '
      'and a ceiling, not a competitor &mdash; but it is honest, and it says that fusing 17 '
      'views does not beat picking the right one. Note also that K is capped at '
      '<span class="mono">min(m&minus;1, 8)</span> and the cap binds on 8 of 25 cells at full '
      'pool.</p>')

    A('<h3>Do the discovered groups mean anything?</h3>')
    A(fig("Groups track the view families, partially",
          "probability that two views land in the same L-SML group",
          hbar(["same family (e.g. two H(n) views)", "different family"],
               [ABL["fam_same"], ABL["fam_diff"]],
               "P(same L-SML group)", d0=0.0, d1=0.35, fmt="{:.3f}", hi={0}),
          "A 1.7&times; enrichment holding on <b>25 of 25 cells</b>. So it is discovering, "
          "without being told, that views computed from the same underlying series are "
          "dependent &mdash; exactly what the model claims. But 0.265 is far from 1.0: "
          "<b>it detects family structure; the groups are not the families.</b>"))

    A('<h3>Is the Eq. 14 residual a good rule for choosing K?</h3>')
    A(table(["", "GOOD_6 (K &isin; 2&ndash;5)", "full pool (K &isin; 2&ndash;8)"], [
        ["argmin(residual) == argmax(AUROC)", f"{ABL['K_hit_g6']}/25 <span class='dim'>(chance ~6)</span>",
         f"{ABL['K_hit_full']}/25 <span class='dim'>(chance ~3.6)</span>"],
        ["Spearman(residual, AUROC) across K", f"{ABL['K_rho_g6']:+.3f}", f"{ABL['K_rho_full']:+.3f}"],
        ["AUROC at the chosen K", f"{ABL['lsml_g6']:.4f}", f"{ABL['lsml_full']:.4f}"],
        ["AUROC at the <i>oracle</i> K", f"{ABL['oracleK_g6']:.4f}", f"<b>{ABL['oracleK_full']:.4f}</b>"],
        ["<b>left on the table by the K rule</b>", "<b>0.16pp</b>", "<b>1.17pp</b>"],
    ], numeric=(1, 2)))
    A('<p>Negative Spearman means lower residual &rarr; higher AUROC, so the criterion '
      '<em>is</em> pointing the right way &mdash; but weakly, and it degrades as the pool grows. '
      'The code says as much: <em>"Steps 203/204 already found the residual criterion itself to '
      'be a poor guide."</em></p>')
    A('<div class="info"><b>A concrete research lead.</b> Full pool with oracle K scores '
      f'<b>{ABL["oracleK_full"]:.4f}</b>, against GOOD_6\'s {ABL["lsml_g6"]:.4f} &mdash; '
      'essentially a tie. So on the full 30-view pool the gap to the hand-picked subset is '
      'largely a <b>K-selection</b> problem, not a <em>pool</em> problem, and a better K rule '
      'is worth up to 1.17pp. Oracle K is label-peeking, so this is a ceiling, not a result.</div>')

    A('<p class="dim"><b>One property worth carrying:</b> L-SML is <b>exactly invariant to input '
      'feature signs</b> (Step 201, 1150/1150 sign vectors, bit-identical output). That is why '
      'the 42 hand polarities were already free on this arm, and why sign(rho) &mdash; worth '
      '+1.46pp on U-PCR &mdash; buys nothing here. The only orientation prior on arm 2 is the '
      'single anchor bit at the end.</p>')
    A('</div>')

    # ---------------------------------------------------------------- dufs
    A('<div class="card">')
    A('<span class="badge bdone">Arm 2</span>')
    A('<h2>5. DUFS parameter-free + L-SML</h2>')
    A('<p>A two-stage pipeline with a clean seam: DUFS picks a subset from the unlabeled V, then '
      'the <em>same</em> L-SML above fuses whatever it picked. Nothing is co-designed, so the '
      'only thing changing versus GOOD_6 is the selection step.</p>')
    A('<p><b>The idea in one paragraph.</b> Give every feature a gate '
      '<span class="mono">z_i &isin; [0,1]</span>, multiply the data by the gates, and build a '
      'similarity graph over <em>samples</em> using only the gated data. Then ask: under which '
      'gate settings does the data look most structured on its own graph? Features whose gates '
      'can close without destroying that structure are uninformative. The self-reference is the '
      'point &mdash; the graph is rebuilt from the gated features at every step.</p>')

    A(code("""INPUT   V (n x p) z-scored, sign-oriented;  anchor;  rng

 -- stage 1: DUFS parameter-free gates ------------------------------
 1  X <- subsample rows of V to at most R_MAX = 1200
 2  for each of 5 seeds:
        mu <- 0.5 * ones(p)                        # gate means, the ONLY parameters
        for 120 epochs:
            B  <- random minibatch of 256 rows
            z  <- clip(mu + N(0, 0.5^2), 0, 1)     # stochastic-gate reparameterisation
            Xt <- B * z                            # gated batch
            W  <- self_tuning_affinity(Xt, k=7)    # graph over SAMPLES, built from Xt
            P  <- (D^-1 W)^2                       # 2-step random walk
            tr <- -trace(Xt^T P Xt) / |B|
            Pz <- 0.5 * (1 + erf(mu / (0.5*sqrt2)))   # P(gate i is open)
            loss <- tr / (sum(Pz) + 1e-8)          # <-- Eq. 7.  NO lambda.
            Adam(lr = 2e-2).step()
 3  mu_bar <- mean of the 5 seeds' mu              # determinism
 4  cols <- { i : mu_bar_i > 0 }                   # the STG readout rule
    if |cols| < 3: fall back to the full pool

 -- stage 2: fusion, unchanged --------------------------------------
 5  fused <- lsml_continuous(V[:, cols])
 6  score <- anchor_orient(fused, anchor)"""))

    A('<div class="warn"><b>Notation trap.</b> DUFS\'s <span class="mono">L_rw</span> is '
      '<span class="mono">D&#8315;&sup1;K</span>, the random-walk <em>transition</em> matrix '
      '&mdash; <b>not</b> the Laplacian <span class="mono">I &minus; D&#8315;&sup1;K</span>. '
      'Reading that "L" as a Laplacian gets the sign backwards. Minimising '
      '<span class="mono">&minus;trace(Xt&#7488; P&sup2; Xt)</span> <b>maximises</b> agreement '
      'between diffusion neighbours.</div>')

    A('<h3>Eq. 6 &rarr; Eq. 7: how the last hyperparameter disappears</h3>')
    A(code("""if param_free:  loss = trace / (Pz.sum() + PF_DELTA)     # Eq. 7   <- a2.dufs_pf
else:           loss = trace / d + lam2 * Pz.mean()     # Eq. 6   <- a2.dufs"""))
    A('<p>Eq. 6 is the usual additive trade-off: fit <b>plus</b> lambda &times; sparsity, and you '
      'have to choose lambda. Eq. 7 makes it a <b>ratio</b>: since <span class="mono">trace</span> '
      'is negative, dividing by the number of open gates makes the objective <b>structure per '
      'open gate</b>. Opening a gate must pay for itself, the trade-off becomes scale-free, and '
      'lambda vanishes. This is the paper\'s own device, used for all its two-moons experiments.</p>')
    A('<div class="take"><b>This is a selling point, not just convenience.</b> The DUFS paper '
      'itself picks lambda by sweeping it and keeping the run with the best <b>clustering '
      'accuracy</b> &mdash; which is computed against labels. <b>The paper\'s own protocol '
      'peeks.</b> Eq. 7 removes the parameter entirely, so a2.dufs_pf is not merely cheaper '
      '(5 trainings instead of 20) &mdash; it is the variant that makes this arm <b>honestly '
      'label-free</b>, and it is <em>more</em> faithful to the paper than our lambda rule, '
      'not less.</div>')

    A('<h3>What the gates select</h3>')
    freq_rows = [[f'<span class="mono">{esc(f)}</span>', n,
                  "&#9608;" * max(1, round(n / 25 * 20))]
                 for f, n in sorted(FREQ.items(), key=lambda kv: -kv[1])[:16]]
    A(table(["View", "Cells selected (of 25)", ""], freq_rows, numeric=(1,), cls="freq"))
    A(f'<p><b>The selection is highly stable.</b> Six views are chosen on every single cell '
      f'&mdash; and four of them are top-K/spilled views that <em>did not exist</em> before the '
      f'cluster re-run, which independently corroborates "7 of the 10 most informative views '
      f'are now new ones". GOOD_6\'s own members are picked at 25, 25, 23, 21, 15 and 11: '
      f'<b>the gates half-agree with the hand-picked subset without being told about it.</b></p>')

    A('<h3>Deviations from Lindenbaum et al. 2021</h3>')
    A(table(["", "What we do", "What the paper does", "Why"], [
        ["<b>D1</b>", "self-tuning kernel exp(&minus;d&sup2;/(&gamma;&#7522;&gamma;&#11388;)), k=7",
         "<b>global</b> bandwidth &sigma;_b = max&#7522;(C&#183;&#8214;x&#7522;&minus;x&#8342;&#8214;), k=2, C=5",
         "makes a2.dufs graph-identical to a2.select so the granularity comparison is meaningful. "
         "Genuine departure."],
        ["<b>D2</b>", "Adam, lr 2e-2, 120 epochs, batch 256",
         "SGD, lr 0.3&ndash;1, <b>5,000&ndash;26,000</b> epochs, mostly full batch",
         "CPU budget. A 40&ndash;200&times; compute reduction and the most likely source of any gap."],
        ["<b>D3</b>", "lambda removed entirely (Eq. 7); or chosen label-free by selection stability",
         "sweeps lambda, keeps the run with the best <b>clustering accuracy</b>",
         "<b>ours is stricter</b> &mdash; the paper's protocol peeks at labels."],
    ]))
    A('<p class="dim">Everything else is faithful: the Eq. 6 objective (our 1/d normalisation '
      'rescales both terms equally, so the minimiser is unchanged), L_rw = D&#8315;&sup1;K as '
      '&sect;2.1 defines it, t = 2 (App. S3), STG with mu init 0.5 and fixed sigma, the Eq. 2 '
      'regulariser, and the "retain features such that Z_i &gt; 0" readout.</p>')
    A('</div>')

    # ---------------------------------------------------------------- side by side
    A('<div class="card">')
    A('<span class="badge bfind">Side by side</span>')
    A('<h2>6. Where each arm makes each decision</h2>')
    A(table(["Decision", "U-PCR + sign(rho)", "DUFS parameter-free + L-SML"], [
        ["<b>Per-view polarity</b>",
         "derived: <span class='mono'>sign(rho_hat)</span> from the covariance structure",
         "<b>irrelevant</b> &mdash; L-SML is exactly sign-invariant"],
        ["<b>Global &plusmn; direction</b>",
         "<span class='mono'>anchor_orient</span> against oriented <span class='mono'>epr</span>",
         "identical &mdash; <span class='mono'>anchor_orient</span>"],
        ["<b>How many views survive</b>",
         "falls out of two Algorithm-1 thresholds &mdash; mean <b>21.0</b>",
         f"falls out of the Eq. 7 ratio &mdash; mean <b>{pf.get('size',0):.1f}</b>"],
        ["<b>Which views survive</b>",
         "rho_hat &ge; 0.05&#183;var_y AND rho_hat &ge; max/3",
         "gates open, <span class='mono'>mu &gt; 0</span>, averaged over 5 seeds"],
        ["<b>Handling view dependence</b>",
         "<b>assumed away</b> (Eq. 13 uncorrelated deviations); the eigen-projection absorbs it",
         "<b>modelled explicitly</b> &mdash; L-SML discovers K groups and fuses in two levels"],
        ["<b>Weighting</b>",
         "Eq. 21: w &isin; span{v1, v2}, coefficients from rho_hat",
         "two-level <span class='mono'>sml_fuse_signed</span>"],
        ["<b>Tunable knobs left</b>",
         "<span class='mono'>scale_ratio = 0.25</span> (worth 0.12pp)",
         "<b>none</b> &mdash; that is the point of Eq. 7"],
        ["<b>Free parameters in the weight vector</b>",
         "<b>2</b> &mdash; rho_hat enters only via v1&#183;rho and v2&#183;rho",
         "K group weight vectors + 1 cross-group vector"],
    ]))
    A('<div class="info"><b>The sharpest structural contrast.</b> On a 30-view pool, Eq. 21 '
      'confines the weight vector to a <b>2-dimensional</b> subspace fixed entirely by C, so '
      '<b>28 of rho_hat\'s 30 degrees of freedom are projected away before they touch the '
      'score</b>. L-SML makes the opposite bet: model the dependence explicitly and let every '
      'view keep its own weight inside its group.</div>')
    A('</div>')

    # ---------------------------------------------------------------- what carries it
    A('<div class="card">')
    A('<span class="badge bflag">The finding</span>')
    A('<h2>7. What actually carries the result</h2>')
    A('<p>Every component of both arms was ablated on the same 25 cells. The pattern is '
      'consistent and it is the most useful thing on this page.</p>')

    A(fig("U-PCR, component by component",
          "macro AUROC over 25 in-scope cells; the deployed arm is highlighted",
          hbar(["exclusion ON + Eq.21 weights  (DEPLOYED)",
                "exclusion OFF + Eq.21 weights",
                "exclusion ON + w = v1 only, rho-hat UNUSED",
                "exclusion ON + L-SML instead of Eq.21",
                "exclusion ON + uniform weights",
                "exclusion OFF + uniform weights (plain average of all views)",
                "exclusion ON + v1+v2 with unit coefficients"],
               [ABL["upcr_deployed"], ABL["upcr_no_exclusion"], ABL["upcr_v1_only"],
                ABL["upcr_lsml_on_kept"], ABL["upcr_unif_weights"], ABL["upcr_no_both"],
                ABL["upcr_v1v2_unit"]],
               "macro AUROC", d0=0.70, d1=0.77, hi={0}),
          f"Deleting the exclusion step entirely costs <b>0.06pp (13W/12L, p=0.979)</b>. "
          f"Replacing the whole Eq. 21 weight with the bare leading eigenvector &mdash; using "
          f"rho-hat <b>not at all</b> &mdash; costs <b>0.09pp (11W/13L, p=0.458)</b>. Both are "
          f"ties. And the weights are, in practice, just v1: "
          f"<b>|cos(w, v1)| = {ABL['cos_w_v1']:.4f} mean, {ABL['cos_w_v1_min']:.4f} min.</b>"))

    A('<div class="warn"><b>This corrects a claim in the July advisor email.</b> The email said '
      '<em>"its weighting turned out to be the part that does not matter &hellip; what it really '
      'decides is which features get dropped."</em> The first half is confirmed and is in fact '
      'stronger than stated. <b>The second half does not hold up</b> &mdash; removing the '
      'exclusion step costs 0.06pp at p=0.979. Selection is not carrying the result either.</div>')

    A('<h3>So what is?</h3>')
    A(table(["Component", "What it is worth", "Verdict"], [
        ["U-PCR's exclusion step (its selection)", "<b>+0.06pp</b> <span class='dim'>p=0.979</span>", '<span class="loss">not load-bearing</span>'],
        ["U-PCR's Eq. 21 weights vs bare v1", "<b>+0.09pp</b> <span class='dim'>p=0.458</span>", '<span class="loss">not load-bearing</span>'],
        ["DUFS's choice of <i>which</i> views, vs random same-size",
         f"<b>+{pf.get('d_rand',0)*100:.2f}pp</b> <span class='dim'>{pf.get('w',0)}W/{pf.get('l',0)}L, p=0.300</span>",
         '<span class="loss">not significant</span>'],
        ["DUFS's size reduction, 30 &rarr; 17", "+0.50pp", '<span class="neutral">modest</span>'],
        ["L-SML vs plain average, on 6 views", "<b>+0.02pp</b> <span class='dim'>p=0.60</span>", '<span class="loss">nothing</span>'],
        ["L-SML vs plain average, on 17 / 29 views", "<b>+6.8pp / +11.9pp</b>", '<span class="win">decisive</span>'],
        ["<b>sign(rho) orientation</b>", "<b>+1.46pp</b> <span class='dim'>20W/5L, p=0.0003</span>", '<span class="win"><b>the contribution</b></span>'],
    ]))

    A('<div class="take"><b>This explains why every direction landed at roughly 75%.</b> '
      'Neither arm\'s selection and neither arm\'s weighting is load-bearing. What carries the '
      'number is <b>(a) label-free orientation</b> and <b>(b) a dependency-aware fusion over a '
      'moderately-sized pool</b>. Once you have both, the specific subset is close to '
      'interchangeable &mdash; so of course eight algorithm families converge to the same place. '
      'That reframes the result from <em>"I tried many things and none won"</em> to '
      '<b>"I measured why nothing wins, and selection is not the bottleneck here."</b></div>')

    A('<p><b>One mechanism worth having ready:</b> exclusion and v1-weighting are '
      '<b>substitutes</b>. With uniform weights, exclusion is worth <b>+1.40pp</b> '
      f'({ABL["upcr_unif_weights"]:.4f} vs {ABL["upcr_no_both"]:.4f}); with v1-shaped weights it '
      'is worth nothing, because the leading eigenvector already down-weights weak views '
      '<em>smoothly</em>. You need one of the two, not both.</p>')
    A('</div>')

    # ---------------------------------------------------------------- deviations
    A('<div class="card">')
    A('<span class="badge bflag">Fidelity</span>')
    A('<h2>8. Every deviation, and what it is worth</h2>')
    A('<p>Running the U-PCR pipeline fully paper-faithful scores <b>69.1%</b> against '
      '<b>73.9%</b> for the path we kept, so the deviations stay &mdash; but each is a choice '
      'that has to be defensible on its own.</p>')

    A('<h3>U-PCR: the seven documented deviations</h3>')
    A(table(["#", "Deviation", "What the paper does", "Status"], [
        ["1", "<span class='mono'>loss = 'l2'</span>",
         "Remark 1 / Eq. 15 also offers the <b>absolute</b> loss for robustness when the "
         "uncorrelated-error assumption is violated",
         "L1 available and exact (one LP instead of 300 solves); L2 deployed"],
        ["2", "<span class='mono'>scale_ratio = 0.25</span>",
         "A1 pins var_y; the paper's calibration is Var(f_i) &asymp; Var(Y), i.e. ratio 1",
         "<b>worth &minus;0.12pp to 'fix'</b>; causes g2 to pin at the grid ceiling on 21/25 cells"],
        ["3", "<span class='mono'>difficulty_gate = False</span>",
         "Algorithm 1 STOPS when g2 &lt; 0.1&#183;Var(Y)",
         "would fire on 1/25 cells at our scale_ratio; we always return a score"],
        ["4", "<span class='mono'>simple_avg_fallback = True</span>",
         "&sect;4.3: with &le; 4 experts left, use a plain average",
         "faithful &mdash; but <b>fires on 0 of 25 cells</b>, so it is dead code in practice"],
        ["5", "<span class='mono'>recompute_after_exclusion = True</span>",
         "Algorithm 1: \"Recalculate v1, rho(q), g2 on remaining experts\"",
         "faithful"],
        ["6", "<span class='mono'>g2_projection_k = 1</span>",
         "Eq. 20's residual is against v1 <b>alone</b>; the 2-component rule replaces Eq. 21 only",
         "faithful; the legacy path used 2 almost everywhere. Worth <b>+0.09pp</b>"],
        ["7", "<span class='mono'>exclusion</span> genuinely switchable",
         "Algorithm 1 always excludes",
         "the legacy path could not truly disable it &mdash; min_frac=0 collapsed to rho &ge; 0, "
         "which still drops weak views"],
    ], numeric=(0,)))

    A('<h3>A deviation that is in neither list, upstream of both arms</h3>')
    A('<div class="warn"><b>z-scoring is itself a per-feature rescale.</b> The pipeline is '
      '<b>exactly</b> invariant to a <em>global</em> rescale (verified: multiply every view by '
      '7.3 and <span class="mono">max|&Delta;AUROC| = 0.00e+00</span>) but <b>not</b> to a '
      'per-feature one &mdash; <span class="mono">s&#7522;s&#11388;(g2+a&#7522;+a&#11388;)</span> '
      'is not of the form <span class="mono">g2\'+a&#7522;\'+a&#11388;\'</span>, so Eq. 14\'s '
      'additive structure breaks. We therefore impose the model on the <b>correlation</b> matrix, '
      'not the covariance matrix. Justification: the paper\'s regressors all predict Y in Y\'s '
      'units; our 30 views are entropies in nats, log-partition energies, spectral band powers '
      'and token counts, with no common scale. Some normalisation is mandatory.</div>')
    A(table(["Scaler", "U-PCR + sign(rho)", "L-SML GOOD_6", "mean(diag C)"], [
        ["<b>z-score (deployed)</b>", f"<b>{ABL['scale_z_upcr']:.4f}</b>",
         f"<b>{ABL['scale_z_g6']:.4f}</b>", f"{ABL['scale_z_diag']:.3f}"],
        ["min-max [&minus;1,1], centered",
         f"{ABL['scale_mm_upcr']:.4f} <span class='loss'>(&minus;1.34pp, 5W/20L)</span>",
         f"{ABL['scale_mm_g6']:.4f} <span class='loss'>(&minus;0.66pp, 6W/19L)</span>",
         f"{ABL['scale_mm_diag']:.3f}"],
        ["min-max [&minus;1,1], <b>not</b> centered",
         f"{ABL['scale_mmnc_upcr']:.4f} <span class='loss'>(&minus;5.01pp)</span>",
         f"{ABL['scale_mmnc_g6']:.4f} <span class='dim'>(unchanged)</span>", "0.365"],
    ], numeric=(1, 2, 3)))
    A('<p><b>Three consequences.</b> z-score wins on both arms, on 20 of 25 cells for U-PCR &mdash; '
      'likely because min-max sets each view\'s scale from its two most extreme observations, and '
      'several of our views are heavy-tailed. <b>Centering is not optional for U-PCR</b>: '
      '<span class="mono">upcr_fit</span> computes <span class="mono">C = F @ F.T / n</span>, '
      'which is a covariance only if the rows are pre-centered &mdash; L-SML is immune because '
      '<span class="mono">np.cov</span> centers internally, which is exactly why its row does not '
      'move. And <span class="mono">scale_ratio = 0.25</span> was tuned against a <b>unit</b> '
      'diagonal; under min-max, mean(diag C) drops to 0.112 and var_y would silently shrink '
      '9&times;.</p>')
    A('</div>')

    # ---------------------------------------------------------------- Q&A
    A('<div class="card">')
    A('<span class="badge bfind">Anticipated questions</span>')
    A('<h2>9. The questions this work invites, with answers</h2>')
    QA = [
        ("You call it label-free, but there is still an anchor. How free is it really?",
         "One bit, and it is provably not removable from this estimator. rho-hat is a function "
         "of C alone, and C(&minus;F) == C(F) because a global flip flips both indices in every "
         "product &mdash; measured max|&Delta;rho| under a global flip is exactly 0.000e+00 on "
         "all 25 cells. When we did derive the global sign structurally (g2 &ge; 0 so rho should "
         f"lean positive), it was wrong in <b>25 of 25</b> cells and the macro fell to "
         f"<b>{O['macro_rho_majority']:.4f}</b>. That variant now raises rather than existing "
         "silently. Everything else &mdash; all 30 per-view polarities, the subset, the pool "
         "size &mdash; is derived."),
        ("sign(rho) agrees with your hand signs only 56%. That is chance. So what is it recovering?",
         f"The right reference. Against the <b>label-derived</b> polarity it agrees "
         f"<b>{ABL['sign_vs_oracle']*100:.1f}%</b> (p&lt;1e-4). A whole-pool audit found "
         f"<b>{O['n_features_mis_signed']} of {O['n_features_audited']}</b> hand signs have mean "
         "oriented AUROC below 0.5. The estimator disagrees with us and agrees with the truth; "
         "the hand prior was the outlier."),
        ("Is this the paper's algorithm, or yours?",
         "Algorithm 1's structure is faithful: additive pair equations, the Eq. 16 analytic "
         "shift, Eq. 20 g2 selection, exclusion, refit, Eq. 21 weights with the &lambda;2 rule. "
         "Seven deviations are documented and each is exposed as a flag so the A/B is measurable "
         "rather than arguable. Running fully paper-faithful scores <b>69.1%</b> against "
         "<b>73.9%</b> for the path we kept, so we kept it and say so. The one deliberate "
         "addition is orientation, which the paper never faces because its regressors are all "
         "trained to predict Y and are positively oriented by construction."),
        ("Your g2 is pinned at the grid ceiling on most cells. Is the model-selection step doing anything?",
         f"Yes, and the pinning is a clipping artefact worth 0.12pp. With scale_ratio = 0.25 the "
         f"grid is [0, 0.25] against a unit diagonal, and the argmin lands on the boundary on "
         f"{ABL['g2_sr025_pinned']}/25 cells. Widen it to the paper's calibration and it un-pins "
         f"<b>completely</b> ({ABL['g2_sr100_pinned']}/25) and settles at g2 &asymp; 0.44 &mdash; "
         f"and doubling the grid again moves that by 0.003, so the criterion has a stable "
         f"interior argmin. Macro moves {ABL['g2_sr025_macro']:.4f} &rarr; "
         f"{ABL['g2_sr100_macro']:.4f}. The criterion works; the cap is an empirical choice."),
        ("Are the DUFS gates better than picking views at random?",
         f"Barely, and not significantly: <b>+{pf.get('d_rand',0)*100:.2f}pp, "
         f"{pf.get('w',0)}W/{pf.get('l',0)}L, p=0.300</b> against a random subset of the same "
         f"size. The &lambda;-tuned variant is +{du.get('d_rand',0)*100:.2f}pp (p=0.634). What "
         "pays is the size reduction, worth +0.50pp over full-pool L-SML. Both things are true "
         "at once: the selection is stable, structured and reproducible &mdash; six views on "
         "every cell &mdash; and the AUROC is largely insensitive to which ~17 of 30 you take. "
         "That is a statement about pool redundancy, and the grouping analysis says the same."),
        ("If nothing you built beats the hand-picked subset, what is the contribution?",
         "Two things. First, GOOD_6 cannot say about itself that it was chosen without an answer "
         "key; both label-free arms reach the same bar with one bit of prior, and neither gap is "
         "significant. Second, and more useful: we measured <b>why</b> nothing wins. Selection "
         "contributes 0.06&ndash;0.31pp in either arm, weighting 0.09pp, and orientation 1.46pp. "
         "The bottleneck is not which views you pick."),
        ("Your fusion loses to the single best view on the larger pools. Why fuse at all?",
         "The best single view is <b>label-selected</b> &mdash; an oracle and a ceiling, not "
         f"something a deployed method can reach. It is {ABL['best1_full']:.4f} at full pool "
         f"against L-SML's {ABL['lsml_full']:.4f}. The honest reading is that it bounds the "
         "headroom, and it is exactly what the selection line is trying to approach. Note also "
         f"that full pool with <em>oracle K</em> scores {ABL['oracleK_full']:.4f} &mdash; "
         "essentially tied with GOOD_6 &mdash; so a good chunk of that headroom is a K-selection "
         "problem rather than a selection problem."),
        ("How do I know these numbers are current?",
         "Every in-scope script must reproduce the GOOD_6 macro at <b>0.7594 &plusmn; 0.002</b> "
         "before it may report anything (<span class='mono'>inscope_bench_common.assert_good6</span>); "
         "if it does not, the loaded data is not the data the committed numbers came from and the "
         "run is void. The scoreboard page additionally re-derives both arms per cell and aborts "
         "the build on any drift above 5e-4 from the recorded values. AUROC is always raw, never "
         "max(a, 1&minus;a), and <span class='mono'>UnlabeledCell</span> has no labels field, so "
         "label leakage into selection is structurally impossible."),
    ]
    for q, a in QA:
        A(f'<details class="qa"><summary>{esc(q)}</summary><div>{a}</div></details>')
    A('</div>')

    # ---------------------------------------------------------------- provenance
    A('<div class="card">')
    A('<span class="badge bflag">Provenance</span>')
    A('<h2>10. Where every number came from</h2>')
    A(table(["Class", "Source"], [
        ["<b>FILE</b>", "<span class='mono'>results/upcr_study/06_orientation/summary.json</span> "
                        "(orientation macros, polarity agreement, sign audit); "
                        "<span class='mono'>results/selector_bench/a2_groupfs__c46.csv</span> "
                        "(DUFS macros, sizes, selection frequencies, random-subset floors); "
                        "<span class='mono'>results/benchmark_standing.csv</span>"],
        ["<b>SESSION</b>", esc(ABLATION_PROVENANCE)],
        ["<b>PAPER</b>", "<span class='mono'>papers/extracted/unsupervised-ensemble-regression.md</span>, "
                         "<span class='mono'>papers/extracted/unsupervised-ensemble-learning-with-dependent-classifiers.md</span>, "
                         "and the DUFS/GroupFS fidelity notes in "
                         "<span class='mono'>spectral_utils/selectors/a2_groupfs.py</span>"],
    ]))
    A('<p class="dim">This page recomputes nothing at build time, so it renders instantly and '
      'cannot drift from the scoreboard. Scope is the 25 in-scope cells (10 QA + 15 math); RAG '
      'and GPQA are out (Step 191) and contribute no cell, macro or win to anything above.</p>')
    A('</div>')

    A('</div>')

    page = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            "<title>How the two label-free arms work</title>\n<style>\n"
            + CSS + "\n</style>\n</head>\n<body>\n" + "\n".join(P) + "\n</body>\n</html>\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {OUT}  ({len(page)/1024:.0f} KB)")
    return 0


CSS = """
:root{--blue:#2563eb;--blue-l:#eff6ff;--blue-d:#1e40af;--green:#10b981;--green-l:#ecfdf5;
--green-d:#065f46;--red:#dc2626;--amber:#f59e0b;--amber-l:#fffbeb;--amber-d:#92400e;
--g50:#f8fafc;--g100:#f1f5f9;--g200:#e2e8f0;--g600:#475569;--g700:#334155;--g800:#1e293b;--g900:#0f172a;
--purple:#a855f7;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;font-size:15px;color:var(--g800);
background:var(--g50);line-height:1.65;}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);color:#fff;padding:56px 24px 44px;
border-bottom:4px solid var(--blue);}
.hw{max-width:1120px;margin:0 auto;}
.hero h1{font-size:32px;font-weight:800;letter-spacing:-.02em;margin-bottom:10px;}
.hero .sub{font-size:16px;color:#cbd5e1;max-width:900px;}
.pills{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap;}
.pill{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);padding:4px 12px;
border-radius:20px;font-size:13px;font-weight:500;}
.page{max-width:1120px;margin:0 auto;padding:40px 24px 64px;}
.card{background:#fff;border:1px solid var(--g200);border-radius:14px;padding:36px;
margin-bottom:36px;box-shadow:0 2px 6px rgba(0,0,0,.03);}
.badge{display:inline-block;font-size:12px;font-weight:700;text-transform:uppercase;
letter-spacing:.06em;padding:4px 12px;border-radius:20px;margin-bottom:12px;}
.bdone{background:var(--green-l);color:var(--green-d);}
.bfind{background:#f3e8ff;color:#6b21a8;}
.bflag{background:var(--amber-l);color:var(--amber-d);}
h2{font-size:24px;font-weight:800;letter-spacing:-.015em;color:var(--g900);margin-bottom:12px;}
h3{font-size:17px;font-weight:700;color:var(--g900);margin:26px 0 10px;}
p{margin-bottom:14px;color:var(--g700);}
ul{margin:0 0 14px 20px;color:var(--g700);} li{margin-bottom:7px;}
.take{background:var(--green-l);border-left:4px solid var(--green);padding:16px 20px;
border-radius:0 10px 10px 0;margin:18px 0;} .take b{color:var(--green-d);}
.info{background:var(--blue-l);border-left:4px solid var(--blue);padding:16px 20px;
border-radius:0 10px 10px 0;margin:18px 0;} .info b{color:var(--blue-d);}
.warn{background:var(--amber-l);border-left:4px solid var(--amber);padding:16px 20px;
border-radius:0 10px 10px 0;margin:18px 0;} .warn b{color:var(--amber-d);}
table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px;}
th{background:var(--g100);color:var(--g800);font-weight:700;text-align:left;padding:10px 12px;
border:1px solid var(--g200);vertical-align:bottom;}
td{padding:9px 12px;border:1px solid var(--g200);color:var(--g700);vertical-align:top;}
td.num,th.num{text-align:right;white-space:nowrap;}
table.freq td:last-child{color:var(--purple);letter-spacing:-2px;font-size:11px;}
.win{color:#16a34a;font-weight:700;} .loss{color:var(--red);font-weight:700;}
.neutral{color:var(--g600);font-weight:600;} .dim{color:var(--g600);font-size:13px;}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--g100);
padding:2px 6px;border-radius:4px;color:var(--g800);}
pre.cd{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;line-height:1.55;
background:#0f172a;color:#e2e8f0;padding:18px 20px;border-radius:10px;overflow-x:auto;margin:16px 0;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin:20px 0;}
@media(max-width:860px){.grid2{grid-template-columns:1fr;}}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:20px 0 4px;}
.kvi{background:var(--g50);border:1px solid var(--g200);border-radius:10px;padding:14px 16px;}
.kvi .kvn{font-size:28px;font-weight:800;color:var(--g900);letter-spacing:-.02em;}
.kvi .kvl{font-size:12.5px;color:var(--g600);margin-top:2px;line-height:1.4;}
figure.fg{margin:22px 0;border:1px solid var(--g200);border-radius:12px;padding:18px 20px;background:#fff;}
figure.fg figcaption{margin-bottom:12px;font-size:15px;color:var(--g900);}
figure.fg figcaption .fsub{display:block;font-weight:400;font-size:13px;color:var(--g600);margin-top:2px;}
.fnote{font-size:13px;color:var(--g600);margin:10px 0 0;line-height:1.55;}
.gx{stroke:var(--g200);stroke-width:1;}
.tk{font-size:10.5px;fill:var(--g600);} .ax{font-size:11.5px;fill:var(--g600);}
.rl{font-size:12.5px;fill:var(--g800);} .rs{font-size:11px;fill:var(--g600);}
.bn{fill:#94a3b8;} .bh{fill:var(--blue);} .bl{font-size:11.5px;fill:var(--g800);font-weight:700;}
.fn0{fill:#cbd5e1;} .fn1{fill:#94a3b8;} .fn2{fill:#64748b;} .fn3{fill:var(--blue);} .fn4{fill:var(--purple);}
.fv{font-size:14px;fill:#fff;font-weight:800;} .fa{fill:var(--g200);}
details.qa{border:1px solid var(--g200);border-radius:10px;margin-bottom:10px;background:var(--g50);}
details.qa summary{cursor:pointer;padding:13px 18px;font-weight:700;color:var(--g900);font-size:14.5px;
list-style:none;}
details.qa summary::-webkit-details-marker{display:none;}
details.qa summary::before{content:"\\25B8  ";color:var(--blue);font-weight:800;}
details.qa[open] summary::before{content:"\\25BE  ";}
details.qa[open] summary{border-bottom:1px solid var(--g200);}
details.qa > div{padding:14px 18px 16px;color:var(--g700);font-size:14px;background:#fff;
border-radius:0 0 10px 10px;}
svg{max-width:100%;height:auto;display:block;}
"""


if __name__ == "__main__":
    sys.exit(main())
