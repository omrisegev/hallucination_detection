"""Shared CSS, SVG helpers and vocabulary for the Jul-2026 action-item pages.

Every page is self-contained: no external stylesheet, script, font or image.
"""
import html

CSS = """
:root{--bg:#fff;--fg:#141414;--mut:#666;--line:#e4e4e7;--hi:#fff8e1;--box:#f6f7f9;
 --ok:#1e7e34;--bad:#c0392b;--warn:#b8860b;--acc:#1a4f8a;--pos:#2b6cb0;--neg:#c53030}
*{box-sizing:border-box}
body{margin:0;padding:0 18px 90px;background:var(--bg);color:var(--fg);
 font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto}
a{color:var(--acc)} a:hover{text-decoration:underline}
h1{font-size:26px;margin:30px 0 4px;line-height:1.25}
h2{font-size:20px;margin:38px 0 10px;padding-top:13px;border-top:2px solid var(--line)}
h3{font-size:16.5px;margin:26px 0 7px}
h4{font-size:14.5px;margin:18px 0 6px;color:var(--mut);text-transform:uppercase;
   letter-spacing:.04em}
p{margin:9px 0}
.sub{color:var(--mut);font-size:14px;margin:0 0 20px}
.crumb{font-size:13px;color:var(--mut);margin:16px 0 0}
.box{background:var(--box);border-left:4px solid var(--acc);padding:11px 15px;
 margin:15px 0;border-radius:0 4px 4px 0}
.box.bad{border-left-color:var(--bad);background:#fdf0ee}
.box.warn{border-left-color:var(--warn);background:var(--hi)}
.box.ok{border-left-color:var(--ok);background:#eef7f0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:12px 0}
table{border-collapse:collapse;font-size:13px;width:100%;min-width:600px}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right;
 white-space:nowrap;vertical-align:middle}
th:first-child,td:first-child{text-align:left}
th{background:#f0f0f3;font-weight:600}
tr.weak{background:var(--hi)}
tr.hl{background:#eaf2fb}
td.neg,.neg{color:var(--bad)} td.pos,.pos{color:var(--ok)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
code{background:#f0f0f3;padding:1px 5px;border-radius:3px}
.chip{display:inline-block;padding:0 6px;border-radius:9px;font-size:10.5px;
 font-weight:700;color:#fff;margin-right:3px;line-height:16px}
.c-g6{background:#1a4f8a}.c-dep{background:#8a4f1a}.c-or{background:#1e7e34}
.c-up{background:#5a3d8a}.c-drop{background:#999}
.dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:13.5px;
 margin:10px 0}
.dl dt{color:var(--mut)} .dl dd{margin:0}
.glo dt{font-weight:600;color:var(--fg);margin-top:9px}
.glo dd{margin:0 0 4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px;
 margin:18px 0}
.card{border:1px solid var(--line);border-radius:7px;padding:13px 15px;background:var(--box)}
.card h3{margin:0 0 5px;font-size:16px}
.card p{margin:5px 0;font-size:13.5px;color:var(--mut)}
.foot{color:var(--mut);font-size:12.5px;margin-top:34px;border-top:1px solid var(--line);
 padding-top:13px}
.lg{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;
 margin-right:4px}
@media (prefers-color-scheme:dark){
 :root{--bg:#15171b;--fg:#e9e9ec;--mut:#9aa0a6;--line:#2b2e35;--hi:#2d2a1b;--box:#1d2025;
  --acc:#63a0d8;--bad:#e57373;--ok:#66bb6a;--warn:#d4a017;--pos:#63a0d8;--neg:#e57373}
 th{background:#22252b} code{background:#22252b}
 tr.hl{background:#1b2733} .box.bad{background:#2a1d1c} .box.ok{background:#1a2620}
}
:root[data-theme="dark"]{--bg:#15171b;--fg:#e9e9ec;--mut:#9aa0a6;--line:#2b2e35;
 --hi:#2d2a1b;--box:#1d2025;--acc:#63a0d8;--bad:#e57373;--ok:#66bb6a;--warn:#d4a017;
 --pos:#63a0d8;--neg:#e57373}
:root[data-theme="dark"] th{background:#22252b}
:root[data-theme="light"]{--bg:#fff;--fg:#141414;--mut:#666;--line:#e4e4e7;--hi:#fff8e1;
 --box:#f6f7f9;--acc:#1a4f8a;--bad:#c0392b;--ok:#1e7e34;--warn:#b8860b;
 --pos:#2b6cb0;--neg:#c53030}
:root[data-theme="light"] th{background:#f0f0f3}
"""


def esc(s):
    return html.escape(str(s), quote=True)


def page(title, body, css=CSS):
    return (f"<title>{esc(title)}</title><style>{css}</style>"
            f'<div class="wrap">{body}</div>')


def hist_svg(h, w=132, ht=32, show_axis=False):
    """Two class-conditional densities drawn as overlapping filled areas.

    Blue = correct answers, red = hallucinated ones. Both are densities, so a
    6:1 class imbalance does not make one invisible. A view that separates shows
    two displaced humps; a view at chance shows one shape drawn twice.
    """
    pos, neg = h["pos"], h["neg"]
    n = len(pos)
    m = max(max(pos or [0]), max(neg or [0])) or 1.0
    dx = w / n

    def path(vals):
        pts = [f"0,{ht}"]
        for i, v in enumerate(vals):
            yy = ht - (v / m) * (ht - 2)
            pts.append(f"{i * dx:.1f},{yy:.1f}")
            pts.append(f"{(i + 1) * dx:.1f},{yy:.1f}")
        pts.append(f"{w},{ht}")
        return " ".join(pts)

    mid = w / 2
    ax = (f'<line x1="{mid}" y1="0" x2="{mid}" y2="{ht}" stroke="currentColor" '
          f'stroke-opacity=".25" stroke-dasharray="2 2"/>') if show_axis else ""
    return (f'<svg width="{w}" height="{ht}" viewBox="0 0 {w} {ht}" '
            f'role="img" style="vertical-align:middle;overflow:visible">{ax}'
            f'<polygon points="{path(neg)}" fill="var(--neg)" fill-opacity=".42"/>'
            f'<polygon points="{path(pos)}" fill="var(--pos)" fill-opacity=".42"/>'
            f'<polyline points="{path(pos)}" fill="none" stroke="var(--pos)" '
            f'stroke-width="1.2"/>'
            f'<polyline points="{path(neg)}" fill="none" stroke="var(--neg)" '
            f'stroke-width="1.2"/></svg>')


HIST_LEGEND = ('<span class="lg" style="background:var(--pos);opacity:.6"></span>'
               'correct answers &nbsp; '
               '<span class="lg" style="background:var(--neg);opacity:.6"></span>'
               'hallucinated answers')


def bar(v, lo, hi, w=64, col="var(--acc)"):
    """Horizontal magnitude bar, for scanning a column at a glance."""
    f = 0.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    return (f'<span style="display:inline-block;width:{w}px;height:7px;'
            f'background:var(--line);border-radius:2px;vertical-align:middle">'
            f'<span style="display:block;width:{f * w:.1f}px;height:7px;'
            f'background:{col};border-radius:2px"></span></span>')


def pp(v, nd=2, unit="pp"):
    """Signed number with a colour class."""
    if v is None:
        return "<td>—</td>"
    cls = "neg" if v < -0.05 else ("pos" if v > 0.05 else "")
    return f'<td class="{cls}">{v:+.{nd}f}{unit}</td>'


def num(v, nd=4):
    return "—" if v is None else f"{float(v):.{nd}f}"


# ── vocabulary, used verbatim on every page that needs it ────────────────────
GLOSSARY = [
    ("cell", "One (dataset, model, decoding-config) pair — e.g. CoQA answered by "
     "LLaMA-7B. Twenty-five are in scope. Every number on these pages is computed "
     "inside a single cell; nothing is pooled across cells unless it says so."),
    ("answer / sample", "One generated answer, labelled <b>correct</b> or "
     "<b>hallucinated</b> by that cell's grader. A cell has between 198 and 8,460 "
     "of them."),
    ("view (or feature)", "One number computed from the token-level entropy trace "
     "of one answer — e.g. <code>epr</code>, <code>spectral_entropy</code>, "
     "<code>varentropy</code>. A view is a weak, standalone hallucination detector."),
    ("pool", "Every view available on that cell — 27 to 30 of them. Views can be "
     "missing when a cell's cache lacks the fields they need."),
    ("subset", "The views actually fused. <code>GOOD_6</code> is hand-picked and "
     "fixed; <code>deployed</code> is what the label-free selector (DUFS) chose on "
     "that cell; <code>oracle-5</code> is the best 5 views chosen <i>using the "
     "labels</i>, so it is an upper bound we cannot reach in practice, not a method."),
    ("AUROC", "Probability that a randomly chosen correct answer scores above a "
     "randomly chosen hallucinated one. 0.5 is chance. Reported <b>raw</b> — never "
     "<code>max(a, 1−a)</code> — except where a column explicitly says oracle."),
    ("relative sign", "Each view's polarity: does a <i>higher</i> value mean more "
     "likely correct, or less? There are 2<sup>p</sup> combinations and no labels "
     "to pick from."),
    ("global sign / the anchor", "Even with every relative sign right, the fused "
     "score is only determined up to one overall ±1 — and that bit is "
     "<b>provably not recoverable</b> from the covariance (flipping everything "
     "leaves the covariance identical). So one view, the <b>anchor</b> "
     "(<code>epr</code>), is declared to point the known way, and the fused score "
     "is flipped if it disagrees with it. This is the single prior the label-free "
     "arms still carry."),
    ("true-label anchor", "The counterfactual where that one bit is supplied by "
     "the labels instead of by the anchor view. The gap between the two is the "
     "<b>entire</b> cost of the anchor prior, and these pages report it per cell."),
    ("L-SML", "The fusion. It groups views that fail together, weights within each "
     "group, then combines the groups — all without labels. Its central job is to "
     "recover the relative signs it was not told."),
    ("K and the grouping", "How many dependent groups L-SML splits the subset "
     "into, chosen by minimising the Eq.-14 residual over K. <b>Degenerate</b> "
     "means the winning K beat the runner-up by less than floating-point noise, "
     "so the answer was decided by rounding rather than by data."),
    ("residual", "The Eq.-14 objective L-SML minimises when picking the grouping. "
     "Comparable only within one cell and one subset — never across cells."),
    ("non-monotone gain", "Cross-fitted AUROC of a bin-mean predictor built on "
     "that view, minus the view's own oracle-oriented AUROC. Positive means the "
     "view carries signal that <b>no monotone, sign-oriented use of it can "
     "reach</b> — and every fusion here is monotone in each view. Cross-fitted "
     "over 5 folds, because the in-sample version beats a monotone score even on "
     "pure noise."),
    ("recovery ratio", "Of the AUROC a simple average loses by not knowing the "
     "relative signs, the fraction L-SML gives back. 1.0 means all of it. This is "
     "the quantity that separates the weak cells from the healthy ones."),
]


def glossary_html(subset=None):
    items = [g for g in GLOSSARY if subset is None or g[0] in subset]
    out = ['<dl class="dl glo">']
    for term, defn in items:
        out.append(f"<dt>{term}</dt><dd>{defn}</dd>")
    out.append("</dl>")
    return "".join(out)
