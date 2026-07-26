"""
study_common.py — shared helpers for the trimming study (results/pruning_study/).

Everything in this study reports in PLAIN LANGUAGE. Code identifiers like
`epr` or `lsml_continuous` are translated through PLAIN_NAME / METHOD_NAME
before they reach any chart, table or CSV column that a human reads. The raw
code name is always kept alongside in a `*_code` column so results stay
machine-traceable.

Scoring is the canonical path (scripts/inscope_bench_common.py), i.e. exactly
the path the project's headline numbers come from:
    z-scored views over the 30-view pool
    -> continuous L-SML fusion
    -> label-free global sign resolution against the cell's anchor view
    -> raw AUROC (never max(a, 1-a))

The one deviation is `compute_score_matrix=False`, which skips an O(m^4)
matrix that nothing reads on the groups-given path. Verified bit-identical
output, ~103x faster at 30 views.
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import roc_auc_score                        # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous          # noqa: E402
from spectral_utils.streaming_utils import anchor_orient         # noqa: E402

OUT_ROOT = os.path.join(REPO, "results", "pruning_study")

# --------------------------------------------------------------------------
# plain-language names (sourced from GLOSSARY.md)
# --------------------------------------------------------------------------
PLAIN_NAME = {
    "epr": "Average uncertainty per token",
    "trace_length": "Answer length",
    "spectral_entropy": "How flat the uncertainty rhythm is",
    "low_band_power": "Slow drifts in uncertainty",
    "high_band_power": "Fast jitter in uncertainty",
    "hl_ratio": "Jitter-to-drift ratio",
    "dominant_freq": "Speed of the strongest rhythm",
    "spectral_centroid": "Average rhythm speed",
    "stft_max_high_power": "Strongest local burst of jitter",
    "stft_spectral_entropy": "Local rhythm flatness",
    "rpdi": "Late-answer uncertainty drift",
    "sw_var_peak": "Burstiest stretch of the answer",
    "pe_mean": "Ordinal pattern complexity",
    "hurst_exponent": "Self-similarity of the trace",
    "cusum_max": "Size of the biggest shift",
    "cusum_shift_idx": "Where the biggest shift happens",
    "epr_spilled": "Average surprise per token",
    "sw_var_peak_spilled": "Burstiest stretch (surprise)",
    "cusum_max_spilled": "Size of biggest shift (surprise)",
    "min_spilled": "Most confident single token",
    "bocpd_ecp": "Expected changepoints",
    "bocpd_ecp_spilled": "Expected changepoints (surprise)",
    "bocpd_mean_p0": "Changepoint probability",
    "hmm_occupancy": "Time in the high-uncertainty state",
    "hmm_tail_occupancy": "Late time in high-uncertainty state",
    "hmm_switch_rate": "State-switching rate",
    "ar_mse_innov": "Unpredictability of the trace",
    "ar_innov_ratio": "Unpredictability ratio",
    "kalman_mse_innov": "Tracking error",
    "kalman_nis": "Normalised tracking error",
    "mahalanobis": "Distance from typical answers",
    "gmm_nll": "Atypicality (mixture model)",
    "kde_nll": "Atypicality (density estimate)",
    "iforest": "Outlier score",
    "ae": "Reconstruction error",
    "prae": "Reconstruction error (predictive)",
    "epr_energy": "Average vocabulary spread",
    "min_energy": "Narrowest vocabulary moment",
    "sw_var_peak_energy": "Burstiest stretch (vocabulary spread)",
    "cusum_max_energy": "Size of biggest shift (vocabulary spread)",
    "mean_top1_logprob": "Confidence in the chosen word",
    "logprob_margin": "Lead of first choice over second",
    "mean_logprob_entropy": "Spread across candidate words",
    "varentropy": "Variability of surprise",
    "renyi_entropy_2": "Concentration of candidate words",
    "topk_tail_mass": "Probability outside the top few words",
}

CELL_NAME = {
    "epr_triviaqa_mistral24b": "TriviaQA / Mistral-24B",
    "inside_coqa_llama7b": "CoQA / Llama-7B",
    "losnet_hotpotqa_mistral7b": "HotpotQA / Mistral-7B",
    "sciq_llama8b": "SciQ / Llama-8B",
    "se_nq_open_llama8b": "NQ-Open / Llama-8B",
    "se_squad_v2_llama8b": "SQuAD-v2 / Llama-8B",
    "seiclr_triviaqa_opt30b": "TriviaQA / OPT-30B",
    "semenergy_triviaqa_qwen3_8b": "TriviaQA / Qwen3-8B",
    "spilled_triviaqa_llama8b": "TriviaQA / Llama-8B",
    "truthfulqa_llama8b": "TruthfulQA / Llama-8B",
    "ars_gsm8k_r1distill8b": "GSM8K / R1-Distill-8B",
    "internalstates_gsm8k_qwen25_7b": "GSM8K / Qwen2.5-7B",
    "lapeigvals_gsm8k_llama3b": "GSM8K / Llama-3B",
    "lapeigvals_gsm8k_llama8b": "GSM8K / Llama-8B",
    "lapeigvals_gsm8k_mistral24b": "GSM8K / Mistral-24B",
    "lapeigvals_gsm8k_nemo": "GSM8K / Nemo",
    "lapeigvals_gsm8k_phi35": "GSM8K / Phi-3.5",
    "noise_gsm8k_mistral7b": "GSM8K / Mistral-7B",
    "noise_gsm8k_phi3mini": "GSM8K / Phi-3-mini",
    "math500_dsmath7b": "MATH-500 / DeepSeek-Math-7B",
    "math500_qwenmath7b": "MATH-500 / Qwen-Math-7B",
    "math500_r1distill8b": "MATH-500 / R1-Distill-8B",
    "math500_r1distill8b_mn4096": "MATH-500 / R1-Distill-8B (long)",
    "trace_gsm8k_llama8b_k10": "GSM8K / Llama-8B (10 samples)",
    "trace_math500_qwenmath15b_k10": "MATH-500 / Qwen-Math-1.5B (10 samples)",
}

# Reference points, from results/checkpoints/scoreboard_latest.csv.
# "Automatic picker" means a method that chooses the measurements ITSELF, per
# test set, without answer keys. That is a different thing from a FIXED subset
# that was chosen once (using answer keys) and then reused everywhere.
REFERENCE_POINTS = {
    "Ceiling: all measurements, trust levels trained on answer keys": 0.7809,
    "Best fixed subset: LOCO_5 (5 measurements, chosen with answer keys)": 0.7705,
    "Six hand-picked measurements, GOOD_6 (chosen with answer keys)": 0.7594,
    "Best automatic picker: a6.pl_dufs (chooses its own, no answer keys)": 0.7524,
    "All measurements, our detector": 0.7457,
    "Six measurements picked at random": 0.7360,
}

REFERENCE_NOTES = {
    "Best fixed subset: LOCO_5 (5 measurements, chosen with answer keys)":
        "Found by exhaustive enumeration and validated leave-one-test-set-out, so "
        "it is not chosen in-sample the way GOOD_6 was. Scored on 24 of 25 test "
        "sets; on that same 24 it beats GOOD_6 by 0.73 points. Still uses answer "
        "keys to choose the subset once.",
    "Best automatic picker: a6.pl_dufs (chooses its own, no answer keys)":
        "Pseudo-label gated DUFS. It builds a stand-in for the truth by fusing a "
        "few strong 'seed' measurements, then uses that to supervise which "
        "measurements get selected. Label-free AT RUNTIME - it never sees an "
        "answer key. BUT its seed set defaults to GOOD_6, which was itself chosen "
        "using answer keys, so it carries a label-derived prior. It is also the "
        "selector of record by default rather than by merit: BOTH its "
        "pre-registered bars failed (mechanism 0.207 vs a 0.30 bar; performance "
        "+0.22pp vs a +1.0pp bar), and its lead over GOOD_5 is +0.05pp, not "
        "significant. Treat 0.7524 as the number to beat, not as a strong result.",
    "Six hand-picked measurements, GOOD_6 (chosen with answer keys)":
        "The standing anti-regression anchor: every experiment must reproduce "
        "0.7594 before reporting anything.",
}


def plain(code):
    return PLAIN_NAME.get(code, code)


def plain_cell(code):
    return CELL_NAME.get(code, code)


# --------------------------------------------------------------------------
# data + scoring
# --------------------------------------------------------------------------
def load():
    """The 25 in-scope cells through the canonical loader."""
    from inscope_bench_common import load_cells
    return load_cells()


def fuse_score(cell, cols, groups=None, weights=None):
    """Canonical label-free score for a subset of measurements.

    groups=None      -> L-SML discovers its own groups (the deployed detector)
    groups='flat'    -> one group, i.e. the grouping step switched off
    weights=vector   -> bypass L-SML entirely and use the supplied weights
    Returns AUROC, or nan when the subset is too small / degenerate.
    """
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return float("nan")
    V = cell["V"]
    sub = V[:, cols]
    if weights is not None:
        fused = sub @ np.asarray(weights, dtype=float)
    else:
        g = np.zeros(len(cols), int) if groups == "flat" else groups
        fused, _ = lsml_continuous(*[V[:, c] for c in cols], groups=g,
                                   compute_score_matrix=False)
    fused = np.asarray(fused, dtype=float)
    if not np.isfinite(fused).all() or fused.std() < 1e-12:
        return float("nan")
    score, _ = anchor_orient(fused, cell["anchor"])
    return float(roc_auc_score(cell["labels"], score))


def fuse_meta(cell, cols, groups=None):
    """Same fusion, but returns (auroc, meta) so callers can read the groups
    and the goodness-of-fit the detector computed."""
    cols = sorted(set(int(c) for c in cols))
    V = cell["V"]
    g = np.zeros(len(cols), int) if groups == "flat" else groups
    fused, meta = lsml_continuous(*[V[:, c] for c in cols], groups=g,
                                  compute_score_matrix=False)
    fused = np.asarray(fused, dtype=float)
    if not np.isfinite(fused).all() or fused.std() < 1e-12:
        return float("nan"), meta
    score, _ = anchor_orient(fused, cell["anchor"])
    return float(roc_auc_score(cell["labels"], score)), meta


def good6_cols(cell):
    from spectral_utils.subset_sweep import GOOD_6
    return [cell["pool"].index(f) for f in GOOD_6 if f in cell["pool"]]


def validity_check(cells, verbose=True):
    """The project's standing anti-regression anchor: the six hand-picked
    measurements must reproduce 0.7594 macro. If they do not, the loaded data
    is not the data our numbers came from and nothing downstream is valid."""
    vals = [fuse_score(c, good6_cols(c)) for c in cells.values()]
    macro = float(np.nanmean(vals))
    ok = abs(macro - 0.7594) <= 0.002
    if verbose:
        print(f"VALIDITY CHECK: six hand-picked measurements = {macro:.4f} "
              f"(must be 0.7594 +/- 0.002) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("Validity check FAILED - refusing to report numbers.")
    return macro


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------
def outdir(name):
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def save_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {os.path.relpath(path, REPO)}  ({len(rows)} rows)")


def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)
    print(f"  saved {os.path.relpath(path, REPO)}")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved {os.path.relpath(path, REPO)}")


# --------------------------------------------------------------------------
# charts - self-contained inline SVG, no external libraries
# --------------------------------------------------------------------------
PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280",
           "#00707a", "#975a16", "#4a5568"]

_CSS = """
:root{--bg:#fff;--fg:#1a202c;--mut:#4a5568;--line:#cbd5e0;--card:#f7fafc}
@media(prefers-color-scheme:dark){:root{--bg:#12161c;--fg:#e6edf3;--mut:#9aa5b1;--line:#2d3748;--card:#1a2029}}
:root[data-theme=dark]{--bg:#12161c;--fg:#e6edf3;--mut:#9aa5b1;--line:#2d3748;--card:#1a2029}
:root[data-theme=light]{--bg:#fff;--fg:#1a202c;--mut:#4a5568;--line:#cbd5e0;--card:#f7fafc}
body{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
max-width:1100px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:26px;margin:0 0 6px} h2{font-size:19px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:16px;margin:22px 0 8px}
.sub{color:var(--mut);margin:0 0 22px}
.tldr{background:var(--card);border-left:4px solid #2b6cb0;padding:16px 18px;border-radius:6px;margin:0 0 26px}
.tldr h2{margin:0 0 8px;border:0;padding:0;font-size:17px}
.tldr ul{margin:8px 0 0;padding-left:20px} .tldr li{margin:5px 0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;white-space:nowrap}
th{background:var(--card);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.fig{margin:18px 0;overflow-x:auto}
.note{color:var(--mut);font-size:13.5px;margin:8px 0 0}
code{background:var(--card);padding:1px 5px;border-radius:3px;font-size:13px}
.warn{background:#fffaf0;border-left:4px solid #c05621;padding:12px 16px;border-radius:6px;margin:14px 0}
@media(prefers-color-scheme:dark){.warn{background:#2a1f14}}
"""


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def line_chart(series, xlabel, ylabel, width=880, height=380, hlines=None,
               xticks=None, ylim=None):
    """series: list of (label, xs, ys). hlines: list of (label, y)."""
    L, R, T, B = 78, 190, 22, 52
    xs_all = [x for _, xs, _ in series for x in xs]
    ys_all = [y for _, _, ys in series for y in ys if np.isfinite(y)]
    if hlines:
        ys_all += [y for _, y in hlines]
    if not xs_all or not ys_all:
        return "<p>(no data)</p>"
    x0, x1 = min(xs_all), max(xs_all)
    y0, y1 = (ylim if ylim else (min(ys_all), max(ys_all)))
    pad = (y1 - y0) * 0.08 or 0.01
    y0, y1 = y0 - pad, y1 + pad
    pw, ph = width - L - R, height - T - B

    def px(x): return L + (x - x0) / max(x1 - x0, 1e-9) * pw
    def py(y): return T + (y1 - y) / max(y1 - y0, 1e-9) * ph

    o = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'font-family="sans-serif" font-size="12">']
    for i in range(6):
        y = y0 + (y1 - y0) * i / 5
        o.append(f'<line x1="{L}" y1="{py(y):.1f}" x2="{L+pw}" y2="{py(y):.1f}" '
                 f'stroke="var(--line)" stroke-width="1"/>')
        o.append(f'<text x="{L-9}" y="{py(y)+4:.1f}" text-anchor="end" '
                 f'fill="var(--mut)">{y:.3f}</text>')
    for x in (xticks or sorted(set(xs_all))):
        o.append(f'<text x="{px(x):.1f}" y="{T+ph+18}" text-anchor="middle" '
                 f'fill="var(--mut)">{x}</text>')
    for i, (lbl, y) in enumerate(hlines or []):
        o.append(f'<line x1="{L}" y1="{py(y):.1f}" x2="{L+pw}" y2="{py(y):.1f}" '
                 f'stroke="#718096" stroke-width="1.4" stroke-dasharray="6 4"/>')
        o.append(f'<text x="{L+pw+8}" y="{py(y)+4:.1f}" fill="#718096" '
                 f'font-size="11.5">{_esc(lbl)}</text>')
    for i, (lbl, xs, ys) in enumerate(series):
        col = PALETTE[i % len(PALETTE)]
        pts = [f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys) if np.isfinite(y)]
        if pts:
            o.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="{col}" stroke-width="2.2"/>')
        yy = T + 14 + i * 17
        o.append(f'<line x1="{L+pw+8}" y1="{yy}" x2="{L+pw+26}" y2="{yy}" '
                 f'stroke="{col}" stroke-width="2.6"/>')
        o.append(f'<text x="{L+pw+30}" y="{yy+4}" fill="var(--fg)" '
                 f'font-size="11.5">{_esc(lbl)}</text>')
    o.append(f'<text x="{L+pw/2}" y="{height-10}" text-anchor="middle" '
             f'fill="var(--fg)">{_esc(xlabel)}</text>')
    o.append(f'<text x="16" y="{T+ph/2}" text-anchor="middle" fill="var(--fg)" '
             f'transform="rotate(-90 16 {T+ph/2})">{_esc(ylabel)}</text>')
    o.append("</svg>")
    return f'<div class="fig">{"".join(o)}</div>'


def bar_chart(labels, values, xlabel="", width=880, bar_h=26, hlines=None,
              colors=None, value_fmt="{:.4f}"):
    L, R, T, B = 330, 90, 14, 44
    height = T + B + bar_h * len(labels)
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return "<p>(no data)</p>"
    v0 = min(finite + [h[1] for h in (hlines or [])])
    v1 = max(finite + [h[1] for h in (hlines or [])])
    span = (v1 - v0) or 1.0
    v0, v1 = v0 - span * 0.12, v1 + span * 0.06
    pw = width - L - R

    def px(v): return L + (v - v0) / (v1 - v0) * pw

    o = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'font-family="sans-serif" font-size="12.5">']
    for i, (lbl, val) in enumerate(zip(labels, values)):
        y = T + i * bar_h
        col = (colors[i] if colors else PALETTE[i % len(PALETTE)])
        o.append(f'<text x="{L-10}" y="{y+bar_h*0.68:.0f}" text-anchor="end" '
                 f'fill="var(--fg)">{_esc(lbl)}</text>')
        if np.isfinite(val):
            o.append(f'<rect x="{L}" y="{y+4}" width="{max(px(val)-L,1):.1f}" '
                     f'height="{bar_h-9}" fill="{col}" rx="2"/>')
            o.append(f'<text x="{px(val)+7:.1f}" y="{y+bar_h*0.68:.0f}" '
                     f'fill="var(--mut)" font-size="11.5">'
                     f'{value_fmt.format(val)}</text>')
    for lbl, hv in (hlines or []):
        o.append(f'<line x1="{px(hv):.1f}" y1="{T}" x2="{px(hv):.1f}" '
                 f'y2="{T+bar_h*len(labels)}" stroke="#c05621" '
                 f'stroke-width="1.5" stroke-dasharray="5 4"/>')
        o.append(f'<text x="{px(hv):.1f}" y="{T+bar_h*len(labels)+16}" '
                 f'text-anchor="middle" fill="#c05621" font-size="11">'
                 f'{_esc(lbl)}</text>')
    o.append(f'<text x="{L+pw/2}" y="{height-6}" text-anchor="middle" '
             f'fill="var(--fg)">{_esc(xlabel)}</text>')
    o.append("</svg>")
    return f'<div class="fig">{"".join(o)}</div>'


def scatter_chart(xs, ys, xlabel, ylabel, width=880, height=400,
                  highlight=None, diagonal=False):
    L, R, T, B = 78, 30, 22, 52
    fx = [x for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    fy = [y for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if not fx:
        return "<p>(no data)</p>"
    x0, x1 = min(fx), max(fx)
    y0, y1 = min(fy), max(fy)
    xp, yp = (x1 - x0) * .08 or .01, (y1 - y0) * .08 or .01
    x0, x1, y0, y1 = x0 - xp, x1 + xp, y0 - yp, y1 + yp
    pw, ph = width - L - R, height - T - B

    def px(x): return L + (x - x0) / (x1 - x0) * pw
    def py(y): return T + (y1 - y) / (y1 - y0) * ph

    o = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'font-family="sans-serif" font-size="12">']
    for i in range(6):
        y = y0 + (y1 - y0) * i / 5
        o.append(f'<line x1="{L}" y1="{py(y):.1f}" x2="{L+pw}" y2="{py(y):.1f}" '
                 f'stroke="var(--line)"/>')
        o.append(f'<text x="{L-9}" y="{py(y)+4:.1f}" text-anchor="end" '
                 f'fill="var(--mut)">{y:.3f}</text>')
        x = x0 + (x1 - x0) * i / 5
        o.append(f'<text x="{px(x):.1f}" y="{T+ph+18}" text-anchor="middle" '
                 f'fill="var(--mut)">{x:.3f}</text>')
    if diagonal:
        lo, hi = max(x0, y0), min(x1, y1)
        o.append(f'<line x1="{px(lo):.1f}" y1="{py(lo):.1f}" x2="{px(hi):.1f}" '
                 f'y2="{py(hi):.1f}" stroke="#718096" stroke-dasharray="5 4"/>')
    for x, y in zip(xs, ys):
        if np.isfinite(x) and np.isfinite(y):
            o.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4" '
                     f'fill="#2b6cb0" fill-opacity="0.5"/>')
    if highlight:
        for hx, hy, hl in highlight:
            o.append(f'<circle cx="{px(hx):.1f}" cy="{py(hy):.1f}" r="6" '
                     f'fill="none" stroke="#c05621" stroke-width="2.4"/>')
            o.append(f'<text x="{px(hx)+10:.1f}" y="{py(hy)-8:.1f}" '
                     f'fill="#c05621" font-size="11.5">{_esc(hl)}</text>')
    o.append(f'<text x="{L+pw/2}" y="{height-10}" text-anchor="middle" '
             f'fill="var(--fg)">{_esc(xlabel)}</text>')
    o.append(f'<text x="16" y="{T+ph/2}" text-anchor="middle" fill="var(--fg)" '
             f'transform="rotate(-90 16 {T+ph/2})">{_esc(ylabel)}</text>')
    o.append("</svg>")
    return f'<div class="fig">{"".join(o)}</div>'


def html_table(headers, rows, numeric_cols=()):
    o = ["<table><thead><tr>"]
    o += [f"<th>{_esc(h)}</th>" for h in headers]
    o.append("</tr></thead><tbody>")
    for r in rows:
        o.append("<tr>")
        for i, c in enumerate(r):
            cls = ' class="num"' if i in numeric_cols else ""
            o.append(f"<td{cls}>{_esc(c)}</td>")
        o.append("</tr>")
    o.append("</tbody></table>")
    return "".join(o)


def write_page(path, title, subtitle, tldr_lines, body_html):
    tldr = "".join(f"<li>{t}</li>" for t in tldr_lines)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{_CSS}</style></head><body>
<h1>{_esc(title)}</h1><p class="sub">{subtitle}</p>
<div class="tldr"><h2>In short</h2><ul>{tldr}</ul></div>
{body_html}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  saved {os.path.relpath(path, REPO)}")
