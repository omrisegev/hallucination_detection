#!/usr/bin/env python
"""
selector_deep_report — four companion HTML reports for the Step-186/187
label-free feature-selection bench, written for reading (not just numbers):

  1. methods_protocol.html     — every compared method: paper, algorithm,
                                 assumptions, and whether we validated them.
  2. experiment_results.html   — results with visualization: leaderboards,
                                 per-cell, per-domain, per-model/temperature.
  3. benchmark_vs_published.html — the label-free candidate (a2.select) placed
                                 on the published-paper scoreboard, action-items
                                 style, next to the fixed-subset references.
  4. feature_value_audit.html  — single-feature AUROC of every canonical-pool
                                 feature per cell / domain / overall + noise-
                                 candidate verdicts (pool-curation analysis).

Data: results/selector_bench/*__{h16,c46}.csv (+ the uncommitted A4 CSVs from
.worktrees/antigravity when present), results/subset_sweep/sweep_summary.csv,
results/repgrid/{published_baselines,headline_X_vs_Y}.csv, and per-feature
AUROCs computed live from the prepared cells (fixed offline signs — the same
orientation the fusion consumes).

    python scripts/selector_deep_report.py    # -> results/selector_bench/*.html
"""

import html as html_mod
import os
import re
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from sklearn.metrics import roc_auc_score  # noqa: E402

from spectral_utils.selector_bench import iter_prepared_cells  # noqa: E402
from spectral_utils.subset_sweep import GOOD_5  # noqa: E402

BENCH = os.path.join(REPO_DIR, 'results', 'selector_bench')
SWEEP = os.path.join(REPO_DIR, 'results', 'subset_sweep')
REPG = os.path.join(REPO_DIR, 'results', 'repgrid')
A4_DIR = os.path.join(REPO_DIR, '.worktrees', 'antigravity',
                      'results', 'selector_bench')

KEY = ['domain', 'cell']
COQA = 'inside_coqa_llama7b'

# family palette (validated 6-slot set from the dashboard + one new slot for A4)
FAM = {
    'reference': ('Reference macros (fixed subsets)', '#2a78d6', '#3987e5'),
    'residual':  ('A1 residual-guided',               '#008300', '#1a9e1a'),
    'classical': ('Classical spectral FS',            '#d55181', '#e87ba4'),
    'stats':     ('Simple-stats floor',               '#c98500', '#eda100'),
    'groupfs':   ('A2 GroupFS',                       '#1baf7a', '#22c286'),
    'cae':       ('A3 Concrete-AE',                   '#d95926', '#eb6834'),
    'a4':        ('A4 Antigravity',                   '#8a63d2', '#9a7ce0'),
    'single':    ('Single feature',                   '#6b7280', '#9aa2af'),
}


def fam_of(variant):
    if variant.startswith('ref.'):
        return 'reference'
    if variant.startswith('a1.'):
        return 'residual'
    if variant.startswith('a2.'):
        return 'groupfs'
    if variant.startswith('a3.'):
        return 'cae'
    if variant.startswith('a4.'):
        return 'a4'
    if variant.startswith(('lapscore', 'spec', 'mcfs')):
        return 'classical'
    if variant.startswith(('random', 'mad', 'kurtosis', 'decorr')):
        return 'stats'
    if variant.startswith('epr'):
        return 'single'
    return 'stats'


def esc(s):
    return html_mod.escape(str(s))


# ===========================================================================
# Part A — data
# ===========================================================================

def load_bench():
    import glob
    files = sorted(glob.glob(os.path.join(BENCH, '*__h16.csv')) +
                   glob.glob(os.path.join(BENCH, '*__c46.csv')))
    frames = [pd.read_csv(f) for f in files]
    a4 = []
    for pool in ('h16', 'c46'):
        p = os.path.join(A4_DIR, f'a4_antigravity__{pool}.csv')
        if os.path.exists(p):
            d = pd.read_csv(p)
            if pool == 'c46':
                # A4's c46 arm ran all 51 cells; only the repgrid-19 rows are
                # protocol-comparable to the master c46 arm (same pools).
                d = d[d.domain == 'repgrid']
            a4.append(d)
    df = pd.concat(frames + a4, ignore_index=True)
    df['auroc'] = pd.to_numeric(df['auroc'], errors='coerce')
    return df


def cell_meta():
    """repgrid cell -> (dataset, short model). Primary source: headline_X_vs_Y
    (covers every grid cell); published_baselines only covers 6 cells."""
    pb = pd.read_csv(os.path.join(REPG, 'published_baselines.csv'))
    hl = pd.read_csv(os.path.join(REPG, 'headline_X_vs_Y.csv'))
    meta = {}
    for _, r in hl.drop_duplicates('cell').iterrows():
        meta[r['cell']] = (r['dataset'], r['model'].split('/')[-1])
    for _, r in pb.drop_duplicates('cell').iterrows():
        meta.setdefault(r['cell'], (r['dataset'], r['model'].split('/')[-1]))
    return meta, pb


# published-method display names, mirrored from the canonical strings in
# scripts/advisor_report.py::qa_headline_html and cluster/presets.py comments
BASELINE_LABEL = {
    'semenergy_triviaqa_qwen3_8b': ('Semantic Energy (2508.14496)', 'unsup.'),
    'epr_triviaqa_mistral24b': ('EPR paper baseline', 'unsup.'),
    'seiclr_triviaqa_opt30b': ('Semantic Entropy (ICLR\'23)', 'multi-sample'),
    'inside_coqa_llama7b': ('INSIDE / EigenScore (2402.03744)', 'multi-sample'),
    'losnet_hotpotqa_mistral7b': ('LOS-Net (2503.14043)', 'supervised'),
    'se_squad_v2_llama8b': ('SE-ICLR protocol (adapted)', 'multi-sample'),
    'spilled_triviaqa_llama8b': ('Spilled/Semantic Energy (boundary)', ''),
    'truthfulqa_llama8b': ('TruthfulQA (generation)', ''),
    'se_nq_open_llama8b': ('SE-ICLR protocol (adapted, judge labels)',
                           'multi-sample'),
    'ars_gsm8k_r1distill8b': ('ARS paper', 'supervised'),
    'internalstates_gsm8k_qwen25_7b': ('Internal-States probe (2510.11529)',
                                       'supervised'),
    'noise_gsm8k_phi3mini': ('Noise-Injection consistency (NI)',
                             'multi-sample'),
    'noise_gsm8k_mistral7b': ('Noise-Injection consistency (NI)',
                              'multi-sample'),
}
# the headline Y for the LapEigvals cells mixes the paper's supervised probe and
# its unsupervised AttentionScore per model — name the paper, not one method
for _c in ('llama8b', 'llama3b', 'mistral24b', 'nemo', 'phi35'):
    BASELINE_LABEL[f'lapeigvals_gsm8k_{_c}'] = (
        'LapEigvals paper (2502.17598)', '')


def parse_h16_cell(domain, cell):
    """(model, T) from an h16 roster cell key; None when not encoded."""
    m = re.match(r'(.+)_T(\d\.\d)$', cell)
    if m:
        return m.group(1), m.group(2)
    if '/' in cell:                      # rag: 'Qwen-7B/hotpotqa'
        return cell.split('/')[0], None
    return None, None


def feature_matrices():
    """Per-feature oriented AUROC per cell for both pools + epr-alone.
    The c46 arm is restricted to the 19 replication-grid cells (the harness
    default iterates all 51 — the same trap A4's bench fell into)."""
    out = {}
    for pool in ('c46', 'h16'):
        rows, ns = {}, {}
        for ctx in iter_prepared_cells(REPO_DIR, pool_mode=pool):
            if pool == 'c46' and ctx.domain != 'repgrid':
                continue
            ck = (ctx.domain, ctx.cell_key)
            ns[ck] = ctx.V.shape[0]
            rows[ck] = {f: roc_auc_score(ctx.labels, ctx.V[:, j])
                        for j, f in enumerate(ctx.pool)}
        M = pd.DataFrame(rows)           # features x cells
        out[pool] = (M, ns)
    return out


def feature_verdicts(M):
    """Per-feature aggregate + verdict from the per-cell AUROC matrix."""
    recs = []
    for f, row in M.iterrows():
        v = row.dropna()
        if not len(v):
            continue
        dev = (v - 0.5).abs()
        rec = dict(feature=f, n_cells=len(v), mean=v.mean(), median=v.median(),
                   above=(v > 0.52).mean(), below=(v < 0.48).mean(),
                   mean_dev=dev.mean(), max_dev=dev.max(),
                   best=v.max(), worst=v.min())
        if len(v) < 5:
            verdict = 'LOW-COVERAGE'
        elif rec['mean_dev'] < 0.02 and rec['max_dev'] < 0.06:
            verdict = 'NOISE-CANDIDATE'
        elif rec['below'] > 0.30 and rec['above'] > 0.30:
            verdict = 'SIGN-UNSTABLE'
        elif rec['mean'] < 0.5:
            verdict = 'ANTI-ORIENTED'
        elif rec['mean_dev'] < 0.035:
            verdict = 'WEAK'
        else:
            verdict = 'KEEP'
        rec['verdict'] = verdict
        recs.append(rec)
    return pd.DataFrame(recs).set_index('feature')


# ===========================================================================
# Part B — html scaffolding
# ===========================================================================

CSS = """
:root{
  --bg:#faf9f7; --panel:#ffffff; --ink:#1f2430; --muted:#5b6472;
  --line:#e3e1dc; --accent:#1baf7a; --ref:#2a78d6; --warn:#c98500;
  --bad:#c0392b; --chip:#f0efec; --code:#f4f3f0;
}
:root[data-theme="dark"]{
  --bg:#14171d; --panel:#1c2027; --ink:#e8eaf0; --muted:#9aa2af;
  --line:#2c313a; --accent:#22c286; --ref:#3987e5; --warn:#eda100;
  --bad:#e26d5a; --chip:#252a33; --code:#232830;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14171d; --panel:#1c2027; --ink:#e8eaf0; --muted:#9aa2af;
    --line:#2c313a; --accent:#22c286; --ref:#3987e5; --warn:#eda100;
    --bad:#e26d5a; --chip:#252a33; --code:#232830;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 "Segoe UI",system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;line-height:1.25;margin:6px 0 2px;text-wrap:balance}
h2{font-size:19px;margin:38px 0 8px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}
p{max-width:76ch}
.sub{color:var(--muted);margin:0 0 18px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);margin:0}
table{border-collapse:collapse;font-size:13.5px;margin:10px 0}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;
  font-variant-numeric:tabular-nums}
th{font-size:12px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em;white-space:nowrap}
td.num,th.num{text-align:right}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;
  padding:4px 8px;background:var(--panel)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:16px 18px;margin:14px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.chip{display:inline-block;background:var(--chip);border-radius:999px;
  padding:1px 10px;font-size:12px;color:var(--muted);margin-right:6px}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;
  margin-right:6px;vertical-align:baseline}
.k{font-weight:600}
.good{color:var(--accent);font-weight:600}
.badv{color:var(--bad);font-weight:600}
.warnv{color:var(--warn);font-weight:600}
.mut{color:var(--muted)}
code{background:var(--code);border-radius:4px;padding:1px 5px;font-size:13px}
.note{border-left:3px solid var(--warn);padding:8px 14px;background:var(--panel);
  border-radius:0 8px 8px 0;margin:12px 0;max-width:80ch}
.big{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums}
svg text{font:11.5px "Segoe UI",system-ui,sans-serif;fill:var(--ink)}
svg .mut{fill:var(--muted)}
.legend{margin:8px 0 2px;font-size:13px;color:var(--muted)}
a{color:var(--ref)}
"""

THEME_JS = """
<script>
(function(){var b=document.createElement('button');b.textContent='\\u25d0 theme';
b.style.cssText='position:fixed;top:10px;right:12px;z-index:9;background:var(--panel);'+
'color:var(--muted);border:1px solid var(--line);border-radius:8px;padding:3px 10px;'+
'cursor:pointer;font-size:12px';
b.onclick=function(){var r=document.documentElement;
var cur=r.getAttribute('data-theme')||
(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
r.setAttribute('data-theme',cur==='dark'?'light':'dark');};
document.body.appendChild(b);})();
</script>
"""


def page(title, subtitle, body):
    return (f'<title>{esc(title)}</title>\n<style>{CSS}</style>\n'
            f'<main>\n<p class="eyebrow">Step 186-187 &middot; label-free '
            f'feature-subset selection</p>\n<h1>{esc(title)}</h1>'
            f'<p class="sub">{subtitle}</p>\n{body}\n</main>\n{THEME_JS}')


def hbar(rows, xmin=0.55, xmax=None, width=940, ref_lines=(), title=''):
    """Horizontal bar chart. rows = [(label, value, hex, note)]."""
    if xmax is None:
        xmax = max(v for _, v, _, _ in rows) + 0.012
    bh, gap, lab_w = 21, 7, 250
    h = len(rows) * (bh + gap) + 26
    plot_w = width - lab_w - 74
    def X(v):
        return lab_w + max(0.0, (v - xmin) / (xmax - xmin)) * plot_w
    s = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(title)}" '
         f'style="max-width:100%;height:auto">']
    for rv, rl, rc in ref_lines:
        x = X(rv)
        s.append(f'<line x1="{x:.0f}" y1="4" x2="{x:.0f}" y2="{h-20}" '
                 f'stroke="{rc}" stroke-dasharray="4 4" stroke-width="1.4"/>'
                 f'<text x="{x+4:.0f}" y="{h-6}" class="mut">{esc(rl)} {rv:.3f}</text>')
    y = 4
    for label, v, color, note in rows:
        s.append(f'<text x="{lab_w-8}" y="{y+bh-6}" text-anchor="end">{esc(label)}</text>')
        s.append(f'<rect x="{lab_w}" y="{y}" width="{max(2, X(v)-lab_w):.1f}" '
                 f'height="{bh}" rx="4" fill="{color}"/>')
        s.append(f'<text x="{X(v)+6:.1f}" y="{y+bh-6}">{v:.4f}'
                 f'{(" " + esc(note)) if note else ""}</text>')
        y += bh + gap
    s.append('</svg>')
    return ''.join(s)


def heat_td(v, lo=0.35, hi=0.95, fmt='{:.0f}', scale100=True):
    """Diverging cell around 0.5: blue = below, orange = above, gray = 0.5."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<td class="num mut" style="opacity:.45">&middot;</td>'
    t = max(-1.0, min(1.0, (v - 0.5) / 0.32))
    a = min(0.72, abs(t) * 1.15)
    rgb = '43,120,214' if t < 0 else '235,104,52'
    txt = fmt.format(v * 100 if scale100 else v)
    return (f'<td class="num" style="background:rgba({rgb},{a:.2f})">{txt}</td>')


def verdict_chip(v):
    color = {'KEEP': 'var(--accent)', 'WEAK': 'var(--warn)',
             'NOISE-CANDIDATE': 'var(--bad)', 'SIGN-UNSTABLE': 'var(--bad)',
             'ANTI-ORIENTED': 'var(--warn)', 'LOW-COVERAGE': 'var(--muted)'}.get(v, 'var(--muted)')
    return f'<span style="color:{color};font-weight:600">{esc(v)}</span>'


# ===========================================================================
# Part C1 — methods & protocol page
# ===========================================================================

def method_card(name, fam, paper, algo, assume, validated):
    c = FAM[fam][1]
    rows = ''.join(
        f'<tr><td style="white-space:nowrap">{esc(a)}</td><td>{v}</td></tr>'
        for a, v in validated)
    return (f'<div class="card"><h3 style="margin-top:0">'
            f'<span class="dot" style="background:{c}"></span>{esc(name)}</h3>'
            f'<p class="mut" style="margin:2px 0 8px;font-size:13px">{paper}</p>'
            f'<p style="margin:6px 0">{algo}</p>'
            f'<p style="margin:8px 0 4px"><span class="k">Assumes:</span> {assume}</p>'
            f'<table><tr><th>assumption</th><th>validated on our data?</th></tr>'
            f'{rows}</table></div>')


def build_methods(adm):
    V_OK = '<span class="good">Yes</span>'
    V_NO = '<span class="badv">Refuted</span>'
    V_PART = '<span class="warnv">Partially</span>'

    adm_rows = ''
    for (obj, label) in [('eq14_rel', 'L-SML residual (size-normalized)'),
                         ('eq14_raw', 'L-SML residual (raw Eq-14)'),
                         ('upcr_k1', 'U-PCR projection residual (k=1)'),
                         ('rho_mean', 'mean pairwise |Spearman|'),
                         ('rho_max', 'max pairwise |Spearman|'),
                         ('K', 'detected cluster count K')]:
        sub = adm[adm.objective == obj]
        n_adm = (sub.verdict == 'ADMISSIBLE').sum()
        doms = ', '.join(sub[sub.verdict == 'ADMISSIBLE'].domain) or '&mdash;'
        med = sub.median_spearman.median()
        adm_rows += (f'<tr><td>{label}</td><td class="num">{med:+.3f}</td>'
                     f'<td class="num">{n_adm}/7</td><td>{doms}</td></tr>')

    body = f"""
<h2>What this experiment is</h2>
<p>The July-2026 meeting picked <span class="k">label-free, per-cell feature-subset
selection</span> as the thesis's new algorithmic contribution. The pipeline stays
single-generation and unsupervised: from one generated answer we extract up to 46
scalar views of the entropy trace (the canonical pool), and a fusion step (continuous
L-SML, or U-PCR) combines a <em>subset</em> of them into one hallucination score.
Until now the subset was fixed (GOOD_5), found by a labeled grid search &mdash; prior
knowledge. The question here: can an algorithm choose the subset <em>per cell</em>
(model &times; dataset &times; temperature), from the unlabeled feature matrix alone,
and match or beat the fixed label-derived subsets?</p>

<div class="card"><p style="margin:0" class="k">Pipeline with the new stage</p>
<p style="margin:6px 0 0"><code>entropy / logprob trace</code> &rarr;
<code>46 scalar features (fixed offline signs, z-scored)</code> &rarr;
<span style="color:var(--accent);font-weight:650">&#9655; NEW: label-free subset
selector &#9665;</span> &rarr; <code>same L-SML fusion (unchanged)</code> &rarr;
<code>label-free anchor orientation (epr)</code> &rarr; <code>AUROC</code></p>
<p class="mut" style="margin:8px 0 0;font-size:13px">Selectors receive an
<code>UnlabeledCell</code> object that structurally has no labels field &mdash;
label leakage is impossible by construction. Labels are used only to score the
final AUROC, exactly as for every existing method.</p></div>

<h2>Two experiment arms</h2>
<table>
<tr><th>arm</th><th>cells</th><th>pool</th><th>evaluation</th></tr>
<tr><td class="k">H16</td><td>51 cells (math500, gsm8k, gpqa, rag, qa, trace,
repgrid)</td><td>16 core H(n) features</td><td>exact lookup in the Step-153
exhaustive sweep (all 65,399 subsets per cell) &mdash; gives the exact
percentile-within-size and the per-cell oracle</td></tr>
<tr><td class="k">c46</td><td>19 replication-grid cells</td><td>canonical 46-feature
pool (23&ndash;29 available per cell after drops: entropy + spilled-energy + logprob
families)</td><td>live L-SML scoring; no oracle enumerable at p&asymp;2<sup>46</sup></td></tr>
</table>
<p>Every selector runs on identical fixed cell sets with fixed seeds; failures
emit flagged fallback rows and are never dropped; every AUROC is raw
(<code>roc_auc_score</code> after label-free anchor orientation, never
max-flipped). A mandatory random-subset floor and simple-stats floors run beside
every learned method. There is <span class="k">no pass/fail gatekeeping</span>:
all results are reported and the researcher chooses.</p>

<h2>Compared methods</h2>

<h3>Fixed-subset references (the prior-knowledge baselines)</h3>
<div class="card">
<table>
<tr><th>reference</th><th>what it is</th><th>prior knowledge used</th></tr>
<tr><td class="k">GOOD_5</td><td>epr, spectral_entropy, hl_ratio, sw_var_peak,
pe_mean</td><td>labeled grid search over earlier phases</td></tr>
<tr><td class="k">GOOD_6</td><td>GOOD_5 + varentropy</td><td>labeled; only defined
where varentropy exists (c46 arm)</td></tr>
<tr><td class="k">top_macro_5</td><td>top-5 features by individual macro AUROC
(epr, spectral_entropy, hl_ratio, sw_var_peak, cusum_max)</td><td>labeled
per-feature ranking of the Step-153 sweep</td></tr>
<tr><td class="k">STABLE_H9 / consensus_4 / ALL_H16</td><td>stability-picked 9 /
4-feature consensus / the whole H16 pool</td><td>labeled analyses</td></tr>
<tr><td class="k">epr alone</td><td>the single anchor feature</td><td>none (it is
the pipeline's fixed anchor)</td></tr>
<tr><td class="k">per-cell oracle</td><td>best subset per cell from the exhaustive
sweep</td><td>full label peeking; upper reference only (winner's-curse-inflated:
argmax over 65k subsets on the same labels)</td></tr>
</table></div>

<h3>The six label-free selector families</h3>
"""

    body += method_card(
        'A1 — Residual-guided selection (repo-internal objectives)', 'residual',
        'No external paper — the fusion models\' own structure-fit statistics, in the '
        'SML lineage (Parisi et&nbsp;al. 2014; Jaffe et&nbsp;al. 2015) and the U-PCR '
        'projection model. K-rules from random-matrix theory: Ahn &amp; Horenstein '
        '(2013) eigenvalue-ratio; Kritchman &amp; Nadler (2009) sequential '
        'Tracy&ndash;Widom rank test.',
        'Pick the subset whose <em>fusion-model fit</em> is best: minimize the L-SML '
        'Eq-14 spectral residual (exhaustively from the sweep cache, or greedily), or '
        'the U-PCR projection residual; route between L-SML and U-PCR per cell by '
        'whichever residual is smaller; or keep GOOD_5 and swap only K (the cluster '
        'count) using the rank tests.',
        'a subset on which the fusion model fits its structural assumptions better '
        'will also detect hallucinations better (residual &harr; AUROC correlation).',
        [('residual tracks AUROC within size',
          V_NO + ' &mdash; pre-registered admissibility criterion (median '
          'within-size Spearman &le; &minus;0.10 AND &ge;60% negative cells) fails '
          'globally; weakly admissible only on repgrid (&minus;0.109) and qa '
          '(&minus;0.170). Same fate as the old &rho;&ge;0.75 filter.'),
         ('router: smaller residual picks the better fusion family',
          V_NO + ' &mdash; NOT-USEFUL in every domain; a constant choice is at '
          'least as good.'),
         ('K estimable from subset covariance spectrum',
          V_PART + ' &mdash; estimators disagree with each other (KN subset-vs-pool '
          'agreement 2%) yet ALL land within &plusmn;0.2pp of GOOD_5: K is simply '
          'not a lever here.')])

    body += method_card(
        'Classical spectral FS — Laplacian Score / SPEC / MCFS', 'classical',
        'He, Cai &amp; Niyogi, NIPS 2005 (Laplacian Score); Zhao &amp; Liu, ICML '
        '2007 (SPEC &phi;2); Cai, Zhang &amp; He, KDD 2010 (MCFS).',
        'Build a Gaussian kNN similarity graph over <em>samples</em>; rank each '
        'feature by how well it respects that graph: LS = degree-centred Laplacian '
        'Rayleigh quotient; SPEC = normalized-Laplacian quotient with the trivial '
        'eigendirection removed; MCFS = L1-regress the top spectral embeddings on '
        'the features, score = max |coefficient|.',
        'the discriminative signal varies smoothly along the sample manifold '
        '(cluster assumption): features aligned with graph structure are the '
        'informative ones.',
        [('graph smoothness &harr; label relevance',
          V_NO + ' on our cells &mdash; correct/incorrect answers do not form '
          'separated sample clusters in feature space; all three land at '
          '&asymp;0.70 (c46) / 0.61&ndash;0.63 (h16), below the kurtosis floor.'),
         ('MCFS: discrete multi-cluster structure',
          V_NO + ' &mdash; unstable on weak-cluster data (documented); its Lasso '
          'needed a scale-free fix (eigenvector targets shrink as 1/&radic;n) '
          'to produce non-zero scores at n&ge;1200 at all.')])

    body += method_card(
        'A2 — GroupFS: selection through group discovery (+ DUFS baseline)',
        'groupfs',
        'Lifshitz, Lindenbaum, Mishne, Meir &amp; Benisty, '
        '&ldquo;Unsupervised Feature Selection Through Group Discovery&rdquo;, '
        'AAAI 2026 (arXiv:2511.09166). No official code &mdash; clean-room torch-CPU '
        'reimplementation grounded in the fetched paper; predecessor DUFS '
        '(Lindenbaum et&nbsp;al., 2021, Gated Laplacian) reproduced as the '
        'same-family baseline <code>a2.dufs</code>.',
        'Jointly discover latent <em>groups</em> of related features and gate whole '
        'groups on/off. Three-term loss: sample-graph diffusion smoothness of the '
        'gated data + feature-graph smoothness of the group embedding (with '
        'orthogonality) + group-sparsity via stochastic gates; Gumbel-Softmax '
        'feature&rarr;group assignment, warm-started from spectral clustering of '
        'the feature graph. Selection = features of the open-gate groups. Documented '
        'deviation: under our CPU budget the paper\'s joint group-gate training '
        'saturates (all gates open), so selection pressure comes from per-feature '
        'DUFS gates aggregated to group granularity; discovery mechanics kept.',
        'informative structure lives in correlated feature groups; data varies '
        'smoothly on a sample manifold; group sparsity separates signal groups '
        'from nuisance.',
        [('feature groups exist and are discoverable',
          V_OK + ' &mdash; planted-groups smoke: ARI &ge; 0.6, noise excluded; on '
          'real cells it finds a stable 8-feature core + cell-dependent periphery.'),
         ('its groups improve L-SML\'s clustering step',
          V_PART + ' &mdash; feeding discovered groups to L-SML '
          '(<code>a2.select+groups</code> 0.7282) &asymp; letting L-SML cluster '
          'alone (0.7323): clustering is not the bottleneck.'),
         ('selection transfers to detection AUROC',
          V_OK + ' on the 46-view pool: 0.7323 vs GOOD_5 0.7328 label-free '
          '(10W/2T/7L per cell); ' + V_NO + ' on the already-curated H16 pool '
          '(0.6337) &mdash; nothing left to select there.')])

    body += method_card(
        'A3 — Concrete Autoencoder', 'cae',
        'Bal&#305;n, Abid &amp; Zou, &ldquo;Concrete Autoencoders: Differentiable '
        'Feature Selection and Reconstruction&rdquo;, ICML 2019 '
        '(arXiv:1901.09346). Torch-CPU reimplementation, linear decoder.',
        'A concrete (Gumbel-Softmax) selector layer with k slots picks k features; '
        'a decoder reconstructs ALL features from them; temperature annealed '
        '10 &rarr; 0.01; best-of-3-seeds by held-out reconstruction MSE + a '
        'closed-form greedy swap polish on the same objective; k from the val-MSE '
        'elbow.',
        'features that best linearly reconstruct the whole matrix are the '
        'informative ones (reconstruction &asymp; relevance).',
        [('reconstruction &asymp; label relevance',
          V_NO + ' &mdash; the Step-151 negative replicated at scale: best '
          '0.7099 (c46) / 0.62 (h16). Independently confirmed by A4\'s greedy '
          'CSSP on the same objective: identical h16 macro (0.6124 = 0.6124 at '
          'k=5), chosen-set Jaccard 0.52, per-cell r=0.85 &mdash; the objective, '
          'not the optimizer, is the dead end.')])

    body += method_card(
        'A4 — Antigravity track (anchor-affinity / intrinsic-dim / greedy CSSP)',
        'a4',
        'No single paper &mdash; three heuristics built on the shared harness in '
        'the parallel Antigravity worktree (uncommitted branch '
        '<code>selector/a4-antigravity-unsupervised</code>). CSSP = classic column '
        'subset selection; rank rules reuse the A1 estimators.',
        '(i) <code>a4.anchor_*</code>: rank features by |Spearman| with the epr '
        'anchor, keep the top-s (self-supervision by proxy). '
        '(ii) <code>a4.intrinsic_k_ah</code>: subset size = eigenvalue-ratio '
        'intrinsic dimension + 3, same anchor ranking. '
        '(iii) <code>a4.recon_*</code>: greedy forward selection minimizing linear '
        'reconstruction error of the full matrix (deterministic CSSP analog of A3). '
        'Plus GOOD_5 with K forced from the full-pool spectrum.',
        'features that agree with the strongest single view make a good ensemble; '
        'the covariance spectrum\'s effective rank prescribes the subset size.',
        [('anchor-correlates form a good ensemble',
          V_NO + ' &mdash; it selects the anchor\'s clones (maximum redundancy): '
          'a4.anchor_s4 0.6593 &asymp; epr alone 0.6606 (25W/26L per cell). The '
          'ensemble adds nothing over its anchor; GOOD_5\'s +1.1pp comes from '
          'complementary features this objective cannot find. Best <em>learned</em> '
          'selector on H16 nonetheless.'),
         ('reconstruction &asymp; relevance (CSSP)',
          V_NO + ' &mdash; converges exactly to A3 (see above); confirms the '
          'objective-level refutation with a 100&times; cheaper optimizer.'),
         ('smoke gate', V_OK + ' &mdash; 13/13 known-answer tests pass in the '
          'worktree; deterministic; zero fallback rows.')])

    body += method_card(
        'Simple-stats floors', 'stats',
        'No paper &mdash; the mandatory guardrails: random subsets, |kurtosis|, '
        'MAD, greedy decorrelation. (Random-floor discipline motivated by '
        'Rajabinasab et&nbsp;al. 2026.)',
        'Rank features by a one-line statistic of the unlabeled column and take '
        'the top-s. Any learned selector that cannot beat these is not learning '
        'anything about the data.',
        'none (that is the point).',
        [('learned methods clear the dumb floor',
          V_PART + ' &mdash; embarrassingly, |kurtosis| (top-6) scores 0.7199 on '
          'the 46-view grid, above every classical-FS method and A3/A4; only the '
          'GroupFS family and the K-rule ties clear it.')])

    body += f"""
<h3>The clustering-swap sub-experiment (from the theorem-validation lead)</h3>
<p>The L-SML theorem validation found the Lemma-4 within-vs-between-cluster
correlation contrast never materialized (negative in 4/4 inspected cells), so the
bench folded in three ways to replace or override L-SML's internal spectral
clustering: forced group assignments from GroupFS (<code>a2.select+groups</code>,
<code>a2.groups@good5</code>), forced K from the two rank tests
(<code>*.good5+K_ah</code>, <code>*.good5+K_kn</code> &mdash; both subset-spectrum
and full-pool-spectrum variants exist), and the residual router between L-SML and
U-PCR. <span class="k">Verdict: every variant lands within &plusmn;0.6pp of its
untouched counterpart</span> &mdash; two implementations, opposite K estimates,
same answer: the clustering step is not where performance is lost, and the missing
theorem contrast has no practical cost on this grid.</p>

<h2>Objective admissibility (the pre-registered analysis behind A1)</h2>
<p>Before trusting any selector that optimizes a label-free objective, we measured
whether each candidate objective even <em>correlates</em> with AUROC among
same-size subsets (Spearman, within cell &times; size, over the 65k-subset sweep;
criterion fixed in advance: median &le; &minus;0.10 and &ge;60% of units negative).</p>
<div class="scroll"><table>
<tr><th>objective</th><th class="num">median &rho; (all domains)</th>
<th class="num">domains admissible</th><th>which</th></tr>
{adm_rows}
</table></div>
<p class="mut">Consequence: no label-free objective is globally admissible &mdash;
selection cannot be reduced to optimizing any single one of these statistics. The
methods that work (GroupFS) do not optimize a fit statistic; they exploit
structure (groups + smoothness + sparsity).</p>

<h2>Verification discipline</h2>
<p>Every new component passed a standalone known-answer unit test on synthetic
data with an obvious expected result <em>before</em> integration
(<code>scripts/smoke_selectors.py</code>, 17 tests: planted-K rank tests, fusion
regression against pre-refactor fixtures, planted-group ARI, planted-factor
recovery, mask round-trips, lookup-vs-live &le;1e&minus;6, no-label-leak
assertion). Gate order mirrored the cluster rule: component smoke &rarr; 2-cell
bench &rarr; full bench.</p>
"""
    return page('Methods, algorithms & experiment protocol',
                'What each compared selector is, where it was published, what it '
                'assumes, and whether those assumptions hold on our data.', body)


# ===========================================================================
# Part C2 — results page
# ===========================================================================

def build_results(df, comp, epr_c46, epr_h16, cmeta):
    theme = 'light'  # colors picked to work on both; family hex from FAM light

    def macro(pool, variant):
        d = df[(df.pool_mode == pool) & (df.variant == variant)]
        return d.auroc.mean() if len(d) else np.nan

    # ---- c46 leaderboard (headline arm)
    lead_c = (df[df.pool_mode == 'c46'].groupby('variant')
              .agg(macro=('auroc', 'mean'), n=('auroc', 'size'),
                   size=('size', 'mean')).reset_index())
    lead_c = lead_c[lead_c.n >= 19].sort_values('macro', ascending=False)
    keep = [v for v in lead_c.variant
            if not v.startswith(('random_s4', 'random_s6', 'mad', 'decorr',
                                 'a1.relres_exh', 'a1.minres_exh',
                                 'lapscore_s', 'spec_s', 'mcfs_s4', 'mcfs_s6'))]
    lead_c = lead_c[lead_c.variant.isin(keep)].head(24)
    rows = []
    g5 = macro('c46', 'ref.GOOD_5')
    for _, r in lead_c.iterrows():
        f = fam_of(r.variant)
        note = ''
        if r.variant.startswith('a4.'):
            note = '(A4)'
        rows.append((r.variant, r.macro, FAM[f][1], note))
    rows.append(('epr alone (single feature)', epr_c46, FAM['single'][1], ''))
    rows.sort(key=lambda t: -t[1])
    bars_c46 = hbar(rows, xmin=0.60,
                    ref_lines=[(g5, 'GOOD_5', FAM['reference'][1])],
                    title='c46 leaderboard')

    # ---- h16 leaderboard (selected)
    picks_h = ['ref.top_macro_5', 'a1.good5+K_kn', 'ref.GOOD_5',
               'a4.good5+K_intrinsic_ah', 'a1.router@good5', 'ref.consensus_4',
               'a4.anchor_s4', 'a4.intrinsic_k_ah', 'kurtosis_s6',
               'ref.STABLE_H9', 'ref.ALL_H16', 'a2.dufs', 'a2.select',
               'a3.cae_k3', 'lapscore_adapt', 'a4.recon_s5', 'random_s5']
    rows = [(v, macro('h16', v), FAM[fam_of(v)][1],
             '(A4)' if v.startswith('a4.') else '')
            for v in picks_h if not np.isnan(macro('h16', v))]
    rows.append(('epr alone (single feature)', epr_h16, FAM['single'][1], ''))
    rows.sort(key=lambda t: -t[1])
    oracle = 0.7472
    bars_h16 = hbar(rows, xmin=0.55, xmax=0.76,
                    ref_lines=[(macro('h16', 'ref.GOOD_5'), 'GOOD_5',
                                FAM['reference'][1]),
                               (oracle, 'per-cell oracle', '#888888')],
                    title='h16 leaderboard')

    # ---- per-cell dot plot on c46: GOOD_5 vs GOOD_6 vs a2.select
    piv = df[(df.pool_mode == 'c46') &
             df.variant.isin(['ref.GOOD_5', 'ref.GOOD_6', 'a2.select'])] \
        .pivot_table(index='cell', columns='variant', values='auroc')
    piv = piv.sort_values('ref.GOOD_5', ascending=False)
    W, lab_w, rh = 940, 250, 24
    lo_v, hi_v = 0.52, max(0.97, piv.max().max() + 0.01)
    plot_w = W - lab_w - 30
    def PX(v):
        return lab_w + (v - lo_v) / (hi_v - lo_v) * plot_w
    H = len(piv) * rh + 42
    s = [f'<svg viewBox="0 0 {W} {H}" style="max-width:100%;height:auto">']
    for gx in (0.6, 0.7, 0.8, 0.9):
        s.append(f'<line x1="{PX(gx):.0f}" y1="6" x2="{PX(gx):.0f}" y2="{H-30}" '
                 f'stroke="var(--line)" stroke-width="1"/>'
                 f'<text x="{PX(gx):.0f}" y="{H-16}" text-anchor="middle" '
                 f'class="mut">{gx:.1f}</text>')
    y = 8
    for cell, r in piv.iterrows():
        ds, mdl = cmeta.get(cell, ('', ''))
        s.append(f'<text x="{lab_w-8}" y="{y+13}" text-anchor="end">{esc(cell)}</text>')
        a, b, c = r.get('ref.GOOD_5'), r.get('ref.GOOD_6'), r.get('a2.select')
        if not np.isnan(a) and not np.isnan(c):
            x1, x2 = sorted((PX(a), PX(c)))
            s.append(f'<line x1="{x1:.0f}" y1="{y+9}" x2="{x2:.0f}" y2="{y+9}" '
                     f'stroke="var(--line)" stroke-width="2.5"/>')
        for v, colkey, rr in ((b, 'reference', 3.5), (a, 'reference', 5.5),
                              (c, 'groupfs', 5.5)):
            if v is not None and not np.isnan(v):
                fill = FAM[colkey][1]
                op = '0.45' if rr < 5 else '1'
                s.append(f'<circle cx="{PX(v):.1f}" cy="{y+9}" r="{rr}" '
                         f'fill="{fill}" fill-opacity="{op}"/>')
        y += rh
    s.append('</svg>')
    dots = ''.join(s)

    # ---- per-domain table (h16) for headline methods
    doms = ['math500', 'gsm8k', 'gpqa', 'repgrid', 'trace', 'qa', 'rag']
    meth = ['ref.top_macro_5', 'ref.GOOD_5', 'a1.good5+K_kn', 'a4.anchor_s4',
            'kurtosis_s6', 'a2.select', 'a3.cae_k3']
    dh = df[df.pool_mode == 'h16']
    dom_tbl = ['<table><tr><th>method</th><th class="num">51-cell macro</th>' +
               ''.join(f'<th class="num">{d}</th>' for d in doms) + '</tr>']
    for v in meth + ['epr']:
        if v == 'epr':
            cells_html = '<tr><td><span class="dot" style="background:' + \
                FAM['single'][1] + '"></span>epr alone</td>' + \
                f'<td class="num k">{epr_h16 * 100:.1f}</td>' + \
                ''.join('<td class="num mut">&middot;</td>' for d in doms) + '</tr>'
            dom_tbl.append(cells_html)
            continue
        sub = dh[dh.variant == v]
        row = [f'<tr><td><span class="dot" style="background:'
               f'{FAM[fam_of(v)][1]}"></span>{esc(v)}</td>'
               f'<td class="num k">{sub.auroc.mean() * 100:.1f}</td>']
        for d in doms:
            m = sub[sub.domain == d].auroc.mean()
            row.append(heat_td(m, fmt='{:.1f}'))
        dom_tbl.append(''.join(row) + '</tr>')
    dom_tbl.append('</table>')
    dom_html = ''.join(dom_tbl)

    # ---- repgrid per-dataset (c46): a2.select delta vs GOOD_5
    sub = df[(df.pool_mode == 'c46') &
             df.variant.isin(['ref.GOOD_5', 'ref.GOOD_6', 'a2.select'])]
    piv2 = sub.pivot_table(index='cell', columns='variant', values='auroc')
    piv2['dataset'] = [cmeta.get(c, ('?', ''))[0] for c in piv2.index]
    piv2['model'] = [cmeta.get(c, ('', '?'))[1] for c in piv2.index]
    ds_rows = []
    for ds, g in piv2.groupby('dataset'):
        ds_rows.append(
            f'<tr><td>{esc(ds)}</td><td class="num">{len(g)}</td>'
            f'<td class="num">{g["ref.GOOD_5"].mean() * 100:.1f}</td>'
            f'<td class="num">{g["ref.GOOD_6"].mean() * 100:.1f}</td>'
            f'<td class="num">{g["a2.select"].mean() * 100:.1f}</td>'
            + heat_td((g['a2.select'] - g['ref.GOOD_5']).mean() + 0.5,
                      fmt='{:+.1f}') + '</tr>')
    # per model family
    def mf(m):
        m = m.lower()
        for k in ('llama', 'qwen', 'mistral', 'phi', 'nemo', 'opt', 'deepseek',
                  'r1distill'):
            if k in m:
                return {'r1distill': 'deepseek'}.get(k, k)
        return 'other'
    piv2['fam'] = piv2['model'].map(mf)
    fam_rows = []
    for f_, g in piv2.groupby('fam'):
        fam_rows.append(
            f'<tr><td>{esc(f_)}</td><td class="num">{len(g)}</td>'
            f'<td class="num">{g["ref.GOOD_5"].mean() * 100:.1f}</td>'
            f'<td class="num">{g["a2.select"].mean() * 100:.1f}</td>'
            + heat_td((g['a2.select'] - g['ref.GOOD_5']).mean() + 0.5,
                      fmt='{:+.1f}') + '</tr>')

    # ---- temperature slice (h16 reasoning cells with encoded T)
    dh2 = dh[dh.variant.isin(['ref.GOOD_5', 'a2.select', 'a4.anchor_s4',
                              'kurtosis_s6'])].copy()
    parsed = dh2.apply(lambda r: parse_h16_cell(r.domain, r.cell), axis=1)
    dh2['T'] = [t for _, t in parsed]
    t_rows = []
    for T, g in dh2[dh2['T'].notna()].groupby('T'):
        p = g.pivot_table(index='cell', columns='variant', values='auroc')
        t_rows.append(
            f'<tr><td>T = {esc(T)}</td><td class="num">{p.shape[0]}</td>'
            f'<td class="num">{p["ref.GOOD_5"].mean() * 100:.1f}</td>'
            f'<td class="num">{p["a2.select"].mean() * 100:.1f}</td>'
            f'<td class="num">{p["a4.anchor_s4"].mean() * 100:.1f}</td>'
            f'<td class="num">{p["kurtosis_s6"].mean() * 100:.1f}</td></tr>')

    coqa_d = piv.loc[COQA, 'a2.select'] - piv.loc[COQA, 'ref.GOOD_5'] \
        if COQA in piv.index else np.nan

    legend = ''.join(
        f'<span class="chip"><span class="dot" style="background:{FAM[k][1]}">'
        f'</span>{esc(FAM[k][0])}</span>'
        for k in ['reference', 'groupfs', 'residual', 'a4', 'stats',
                  'classical', 'cae', 'single'])

    body = f"""
<p class="legend">{legend}</p>

<h2>Headline: the 46-view pool (19 replication-grid cells)</h2>
<p>The deployment-shaped arm: give the algorithm the raw, uncurated canonical
pool and no labels. <span class="k">GroupFS (<code>a2.select</code>) matches the
label-derived fixed subsets</span> &mdash; 0.7323 vs GOOD_5 0.7328 and
top_macro_5 0.7364 &mdash; choosing a different subset in almost every cell
(12 distinct subsets / 19 cells, sizes 9&ndash;28). Only GOOD_6 (0.7440) still
leads it. A4's variants (purple, from the uncommitted Antigravity worktree,
repgrid-19 rows only) tie GOOD_5 when keeping GOOD_5's features and swapping K,
and trail when actually selecting.</p>
<div class="card">{bars_c46}</div>

<h2>Per cell: dynamic selection vs the fixed subsets</h2>
<p>Small blue dot = GOOD_6, large blue = GOOD_5, green = a2.select (label-free);
the gray bar spans GOOD_5&harr;a2.select. It wins 10 cells (up to +5.5pp), ties
2, loses 7 &mdash; and one catastrophic cell
(<code>inside_coqa_llama7b</code>, {coqa_d * 100:+.1f}pp) erases an otherwise
positive macro: excluding it, a2.select 0.7427 vs GOOD_5 0.7355 (+0.7pp) and
top_macro_5 0.7406 (+0.2pp); GOOD_6 0.7482 stays ahead.</p>
<div class="card">{dots}</div>

<h2>Per dataset and per model family (46-view grid)</h2>
<div class="grid2">
<div class="card"><h3 style="margin-top:0">By dataset</h3>
<table><tr><th>dataset</th><th class="num">cells</th><th class="num">GOOD_5</th>
<th class="num">GOOD_6</th><th class="num">a2.select</th>
<th class="num">&Delta; vs GOOD_5 (pp)</th></tr>{''.join(ds_rows)}</table></div>
<div class="card"><h3 style="margin-top:0">By model family</h3>
<table><tr><th>family</th><th class="num">cells</th><th class="num">GOOD_5</th>
<th class="num">a2.select</th><th class="num">&Delta; (pp)</th></tr>
{''.join(fam_rows)}</table>
<p class="mut" style="font-size:13px">Positive &Delta; concentrates on gsm8k
cells and the 24B/8B mid-size models; the coqa loss sits in the llama row.</p>
</div></div>

<h2>The curated 16-feature pool (51 cells) — where selection does NOT help</h2>
<p>On H16 every learned selector loses to the fixed references: the pool is
already the product of curation, so there is little left to select away and
mistakes are expensive. The best <em>learned</em> selector here is A4's
anchor-affinity family &mdash; but it is indistinguishable from its own anchor:
<span class="k">epr alone scores 0.6606</span>, a4.anchor_s4 0.6593. The per-cell
oracle (0.7472, +7.6pp over GOOD_5) remains uncaptured by every family.</p>
<div class="card">{bars_h16}</div>

<h2>Per domain (h16 arm, macro AUROC &times;100)</h2>
<div class="scroll">{dom_html}</div>
<p class="mut">Reading: fixed references win everywhere; learned selection hurts
most on rag/qa (short traces, weak features); repgrid + gsm8k are where a2.select
is closest. Cell shading: orange above 50, blue below.</p>

<h2>Temperature (h16 cells with T encoded in the key)</h2>
<div class="card"><table>
<tr><th>temperature</th><th class="num">cells</th><th class="num">GOOD_5</th>
<th class="num">a2.select</th><th class="num">a4.anchor_s4</th>
<th class="num">kurtosis_s6</th></tr>{''.join(t_rows)}</table>
<p class="mut" style="font-size:13px">T=1.5 rows are the four math500 reasoning
cells (relabeled per Step 184); T=1.0 covers gsm8k + gpqa + qa. Coverage is thin
&mdash; temperature is not a separable axis on this roster; the T=1.5 reasoning
cells are exactly where all methods (including selection) score highest, so the
apparent T effect is a domain effect.</p></div>

<h2>Reading the whole experiment</h2>
<div class="card"><ul style="margin:4px 0;padding-left:20px">
<li><span class="k">Dynamic label-free selection reaches fixed label-derived
quality on the raw pool</span> (a2.select &asymp; GOOD_5/top_macro_5), with zero
fallbacks &mdash; the intended deployment win. One robustness fix (the coqa cell)
would put it strictly ahead of both.</li>
<li><span class="k">GOOD_6 is still the best fixed subset</span> (+1.2pp over
a2.select); the label-free ceiling has not caught the best curated set.</li>
<li><span class="k">Clustering / K is not a lever</span>: every swap variant
lands within &plusmn;0.6pp of its untouched counterpart.</li>
<li><span class="k">Fit-statistic objectives are refuted</span> (admissibility),
<span class="k">reconstruction objectives are refuted twice</span> (A3 = A4-CSSP),
and <span class="k">anchor-affinity collapses to epr alone</span>.</li>
<li><span class="k">The per-cell oracle prize (+7.6pp, concentrated on RAG/GPQA)
is untouched</span> by all six families &mdash; see the benchmark page for what
that means going forward.</li>
<li><span class="k">The feature audit reframes pool curation:</span> no pool
feature is flat noise; 13/30 are informative but carried with the wrong fixed
sign on this grid (L-SML absorbs that via negative weights) &mdash; see the
feature-value-audit page.</li>
</ul></div>
"""
    return page('Experiment results — label-free selection bench',
                'Six selector families vs the fixed-subset references: overall, '
                'per cell, per domain/dataset, per model family and temperature. '
                'AUROC, higher is better; all label-free at selection time.', body)


# ===========================================================================
# Part C3 — benchmark vs published page
# ===========================================================================

def build_benchmark(df, cmeta, pb, headline):
    hl = headline[headline.method == 'lsml'].set_index('cell')
    # name the published method per cell by matching Y against published_baselines
    pb = pb.copy()
    pb['auroc01'] = pb['auroc'] / 100.0

    def pub_name(cell, y):
        sub = pb[pb.cell == cell]
        m = sub[(sub.auroc01 - y).abs() < 5e-4]
        if len(m):
            r = m.iloc[0]
            return r['method'], r['supervision']
        if cell in BASELINE_LABEL:
            return BASELINE_LABEL[cell]
        return 'cited baseline (roster)', ''

    def col(pool, variant):
        return df[(df.pool_mode == pool) & (df.variant == variant)] \
            .set_index('cell').auroc

    g5 = col('c46', 'ref.GOOD_5')
    g6 = col('c46', 'ref.GOOD_6')
    a2c46 = col('c46', 'a2.select')
    a2h16 = col('h16', 'a2.select')

    def verd(x, y):
        if np.isnan(x) or np.isnan(y):
            return '<td class="mut">&middot;</td>'
        d = x - y
        if d > 0.01:
            return f'<td class="good">&#9650; {d * 100:+.1f}</td>'
        if d < -0.01:
            return f'<td class="badv">&#9660; {d * 100:+.1f}</td>'
        return f'<td class="warnv">&asymp; {d * 100:+.1f}</td>'

    cells = [c for c in a2c46.index if c in hl.index]
    counts = {k: [0, 0, 0] for k in ('a2c46', 'a2h16', 'g5', 'g6')}
    rows = []
    for c in sorted(cells, key=lambda c: -(hl.loc[c, 'Y']
                                           if np.isfinite(hl.loc[c, 'Y'])
                                           else -9)):
        y = float(hl.loc[c, 'Y'])
        name, supv = pub_name(c, y)
        ds, mdl = cmeta.get(c, ('', ''))
        vals = {'a2c46': a2c46.get(c, np.nan), 'a2h16': a2h16.get(c, np.nan),
                'g5': g5.get(c, np.nan), 'g6': g6.get(c, np.nan)}
        for k, v in vals.items():
            if not np.isnan(v):
                d = v - y
                counts[k][0 if d > 0.01 else (2 if d < -0.01 else 1)] += 1
        supv_chip = (f' <span class="chip">{esc(supv)}</span>') if supv else ''
        y_txt = f'{y * 100:.1f}' if np.isfinite(y) else '&mdash;'
        rows.append(
            f'<tr><td>{esc(c)}<br><span class="mut" style="font-size:12px">'
            f'{esc(ds)} &middot; {esc(mdl)}</span></td>'
            f'<td>{esc(name)}{supv_chip}<br><span class="mut" '
            f'style="font-size:12px">published</span></td>'
            f'<td class="num k">{y_txt}</td>'
            f'<td class="num">{vals["g5"] * 100:.1f}</td>{verd(vals["g5"], y)}'
            f'<td class="num">{vals["g6"] * 100:.1f}</td>{verd(vals["g6"], y)}'
            f'<td class="num k" style="color:var(--accent)">'
            f'{vals["a2c46"] * 100:.1f}</td>{verd(vals["a2c46"], y)}'
            f'<td class="num">{vals["a2h16"] * 100:.1f}</td>{verd(vals["a2h16"], y)}'
            f'</tr>')

    def cline(k, label):
        w, t, l = counts[k]
        return (f'<tr><td>{label}</td><td class="num good">{w}</td>'
                f'<td class="num warnv">{t}</td><td class="num badv">{l}</td></tr>')

    n = len(cells)
    body = f"""
<div class="note"><span class="k">What changed vs the action-items pages:</span>
the green column is <code>a2.select</code> &mdash; the subset is chosen
<em>per cell, label-free, with no prior knowledge</em> (GroupFS on the 46-view
pool), then fused by the same L-SML. The fixed-subset columns (GOOD_5 / GOOD_6)
are the label-derived references those pages used. Published numbers come from
<code>results/repgrid/published_baselines.csv</code> exactly as cited; verdict
margin &plusmn;1pp. All our scores are single-generation and unsupervised at
inference; several published baselines are supervised or multi-sample (chips).</p></div>

<h2>Scoreboard ({n} replication-grid cells)</h2>
<div class="scroll"><table>
<tr><th>cell</th><th>published method</th><th class="num">pub.</th>
<th class="num">GOOD_5</th><th>vs</th>
<th class="num">GOOD_6</th><th>vs</th>
<th class="num">a2.select (46-pool)</th><th>vs</th>
<th class="num">a2.select (16-pool)</th><th>vs</th></tr>
{''.join(rows)}
</table></div>

<h2>Verdict counts vs the published number</h2>
<div class="grid2"><div class="card">
<table><tr><th>ours</th><th class="num">&#9650; above</th>
<th class="num">&asymp; within 1pp</th><th class="num">&#9660; below</th></tr>
{cline('g5', 'GOOD_5 (fixed, label-derived)')}
{cline('g6', 'GOOD_6 (fixed, label-derived)')}
{cline('a2c46', 'a2.select on 46-view pool (label-free)')}
{cline('a2h16', 'a2.select on 16-feature pool (label-free)')}
</table></div>
<div class="card"><p style="margin:0 0 6px" class="k">How to read it</p>
<p style="margin:0;font-size:14px">The label-free 46-pool candidate keeps
essentially the same scoreboard as the label-derived fixed subsets &mdash;
same wins, same losses, within a point almost everywhere &mdash; while removing
the grid search and the labels from the pipeline. The 16-pool variant is shown
for completeness: with the pool already curated, dynamic selection only loses
ground, so the 46-view pool is the right home for it.</p></div></div>

<p class="mut">Provenance: published values and citations exactly as recorded in
the replication grid (see <code>results/action_items/item4_benchmarking.html</code>
for the original fixed-subset scoreboard and the per-paper notes; supervision
chips mark supervised or multi-sample baselines, which are not like-for-like
with our single-pass unsupervised scores).</p>
"""
    return page('Benchmark vs published baselines — label-free candidate',
                'The per-cell, label-free subset selector on the same '
                'published-paper scoreboard the action-items reports use.', body)


# ===========================================================================
# Part C4 — feature value audit page
# ===========================================================================

def build_feature_audit(mats, cmeta):
    Mc, ns_c = mats['c46']
    Mh, ns_h = mats['h16']
    # order features by mean oriented AUC desc
    vc = feature_verdicts(Mc)
    vh = feature_verdicts(Mh)
    Mc = Mc.loc[vc.sort_values('mean', ascending=False).index]

    # per-dataset aggregation (c46)
    ds_of = {c: cmeta.get(c[1], ('?', ''))[0] for c in Mc.columns}
    datasets = sorted(set(ds_of.values()))

    # ---- main heatmap: 46 features x 19 cells
    cells_sorted = sorted(Mc.columns, key=lambda c: ds_of[c])
    head = ('<tr><th>feature</th><th class="num">mean</th>' +
            ''.join(f'<th style="writing-mode:vertical-rl;transform:rotate('
                    f'180deg);max-height:150px;font-size:10.5px">'
                    f'{esc(c[1])}</th>' for c in cells_sorted) + '</tr>')
    body_rows = []
    for f, row in Mc.iterrows():
        tds = ''.join(heat_td(row.get(c, np.nan)) for c in cells_sorted)
        body_rows.append(
            f'<tr><td style="white-space:nowrap">{esc(f)}</td>'
            f'<td class="num k">{vc.loc[f, "mean"] * 100:.1f}</td>{tds}</tr>')
    heat_main = f'<div class="scroll"><table>{head}{"".join(body_rows)}</table></div>'

    # ---- per-dataset means heatmap
    agg = {}
    for ds in datasets:
        cols = [c for c in Mc.columns if ds_of[c] == ds]
        agg[ds] = Mc[cols].mean(axis=1)
    A = pd.DataFrame(agg).loc[Mc.index]
    head2 = ('<tr><th>feature</th><th class="num">overall</th>' +
             ''.join(f'<th class="num">{esc(d)}</th>' for d in datasets) +
             '<th>verdict</th></tr>')
    rows2 = []
    for f in A.index:
        tds = ''.join(heat_td(A.loc[f, d]) for d in datasets)
        rows2.append(f'<tr><td>{esc(f)}</td>'
                     f'<td class="num k">{vc.loc[f, "mean"] * 100:.1f}</td>{tds}'
                     f'<td>{verdict_chip(vc.loc[f, "verdict"])}</td></tr>')
    heat_ds = f'<div class="scroll"><table>{head2}{"".join(rows2)}</table></div>'

    # ---- verdict summary table
    vt_rows = []
    for f, r in vc.sort_values(['verdict', 'mean'],
                               ascending=[True, False]).iterrows():
        vt_rows.append(
            f'<tr><td>{esc(f)}</td><td>{verdict_chip(r.verdict)}</td>'
            f'<td class="num">{r.n_cells:.0f}</td>'
            f'<td class="num">{r["mean"] * 100:.1f}</td>'
            f'<td class="num">{r.best * 100:.1f}</td>'
            f'<td class="num">{r.worst * 100:.1f}</td>'
            f'<td class="num">{r.above * 100:.0f}%</td>'
            f'<td class="num">{r.below * 100:.0f}%</td></tr>')

    noise = vc[vc.verdict == 'NOISE-CANDIDATE'].index.tolist()
    unstable = vc[vc.verdict == 'SIGN-UNSTABLE'].index.tolist()
    weak = vc[vc.verdict == 'WEAK'].index.tolist()
    keep = vc[vc.verdict == 'KEEP'].index.tolist()
    anti = vc[vc.verdict == 'ANTI-ORIENTED'].sort_values('mean').index.tolist()
    in_good5 = [f for f in anti + noise + unstable + weak if f in GOOD_5]

    # h16 per-domain means (shows where the fixed sign flips between domains)
    dom_of_h = {c: c[0] for c in Mh.columns}
    hdoms = ['math500', 'gsm8k', 'gpqa', 'repgrid', 'trace', 'qa', 'rag']
    headD = ('<tr><th>feature</th>' +
             ''.join(f'<th class="num">{esc(d)}</th>' for d in hdoms) +
             '<th class="num">all 51</th></tr>')
    rowsD = []
    for f in vh.sort_values('mean', ascending=False).index:
        tds = []
        for d in hdoms:
            cols = [c for c in Mh.columns if dom_of_h[c] == d]
            v = Mh.loc[f, cols].mean() if cols else np.nan
            tds.append(heat_td(v))
        rowsD.append(f'<tr><td>{esc(f)}</td>{"".join(tds)}'
                     f'<td class="num k">{vh.loc[f, "mean"] * 100:.1f}</td></tr>')
    heat_hdom = f'<div class="scroll"><table>{headD}{"".join(rowsD)}</table></div>'

    # ---- h16 heatmap transposed: 51 cells x 16 features
    featsH = vh.sort_values('mean', ascending=False).index.tolist()
    headH = ('<tr><th>cell</th>' +
             ''.join(f'<th style="writing-mode:vertical-rl;transform:rotate('
                     f'180deg);font-size:10.5px">{esc(f)}</th>'
                     for f in featsH) + '</tr>')
    rowsH = []
    for c in sorted(Mh.columns):
        tds = ''.join(heat_td(Mh.loc[f, c] if f in Mh.index else np.nan)
                      for f in featsH)
        rowsH.append(f'<tr><td style="white-space:nowrap;font-size:12px">'
                     f'{esc(c[0])} / {esc(c[1])}</td>{tds}</tr>')
    heat_h16 = f'<div class="scroll" style="max-height:560px;overflow-y:auto">' \
               f'<table>{headH}{"".join(rowsH)}</table></div>'

    body = f"""
<div class="note"><span class="k">Question this page answers:</span> which of the
canonical-pool features carry signal anywhere, and which are noise everywhere and
only dilute the pool. Scores are single-feature AUROCs under the pipeline's
<em>fixed offline orientation</em> (the exact columns the fusion consumes) &mdash;
a value below 50 means the fixed sign convention is wrong on that cell, not that
the feature is worthless there. Verdicts are computed on the 46-view grid
(19 cells); dropping a feature is a pool-curation decision that uses labels
offline &mdash; that is allowed (it is design-time, like choosing the pool was),
but it is <em>not</em> applied anywhere automatically: this page is analysis.</div>

<h2>Verdict summary — the headline is orientation, not noise</h2>
<p>The expected finding was &ldquo;some features are random noise everywhere
&mdash; drop them.&rdquo; That is <span class="k">not</span> what the data says:
<span class="k">zero features are flat noise</span> on this grid. The actual
pathology: <span class="k">{len(anti)} of {len(vc)} features are consistently
ANTI-oriented</span> &mdash; strongly informative, but carried with the wrong
fixed sign (mean AUC 0.27&ndash;0.45, i.e. genuinely 0.55&ndash;0.73 once
flipped). The whole spilled-energy family and several spectral-shape features
are in this bucket, and so is GOOD_5's own <code>hl_ratio</code> (0.378 here vs
positive on the reasoning domains &mdash; see the cross-domain table below).
L-SML absorbs a wrong sign through negative fusion weights, so bench results are
unaffected; anything sign-sensitive (U-PCR's keep-mask, anchor-correlation
priors, human reading of single features) is not.</p>
<div class="grid2">
<div class="card"><p class="big" style="color:var(--accent)">{len(keep)}</p>
<p style="margin:0" class="k">clear keepers</p>
<p class="mut" style="font-size:13px;margin:4px 0 0">consistent oriented signal,
mean deviation &ge; 3.5pp. The top-8 are exactly the stable core GroupFS
rediscovers label-free in all 19 cells &mdash; convergent validation from two
independent routes.</p></div>
<div class="card"><p class="big" style="color:var(--warn)">{len(anti)}</p>
<p style="margin:0" class="k">anti-oriented (informative, wrong fixed sign)</p>
<p class="mut" style="font-size:13px;margin:4px 0 0">
{', '.join(f'<code>{esc(f)}</code>' for f in anti)}</p></div>
<div class="card"><p class="big" style="color:var(--bad)">{len(unstable) + len(weak)}</p>
<p style="margin:0" class="k">sign-unstable / weak</p>
<p class="mut" style="font-size:13px;margin:4px 0 0">
{', '.join(f'<code>{esc(f)}</code>' for f in unstable)} flips side cell-by-cell
(no global sign can serve it); {', '.join(f'<code>{esc(f)}</code>' for f in weak)}
is barely above chance here &mdash; consistent with its known adaptive
suppression inside L-SML.</p></div>
<div class="card"><p class="big">{len(noise)}</p>
<p style="margin:0" class="k">flat-noise candidates</p>
<p class="mut" style="font-size:13px;margin:4px 0 0">mean |AUC&minus;50| &lt; 2pp
and never beyond 6pp in any cell: {', '.join(f'<code>{esc(f)}</code>' for f in noise) or
'<span class="k">none</span> &mdash; every pool feature carries signal somewhere.'}</p></div>
</div>
<p class="mut">GOOD_5 members outside the clear-keeper bucket:
{', '.join(f'<code>{esc(f)}</code>' for f in in_good5) or 'none'}.</p>

<h2>Per-feature verdicts (46-view grid, 19 cells)</h2>
<div class="scroll"><table>
<tr><th>feature</th><th>verdict</th><th class="num">cells</th>
<th class="num">mean AUC</th><th class="num">best</th><th class="num">worst</th>
<th class="num">% cells &gt; 52</th><th class="num">% cells &lt; 48</th></tr>
{''.join(vt_rows)}
</table></div>

<h2>Per feature &times; dataset (mean AUC &times;100)</h2>
{heat_ds}
<p class="mut">Orange = above 50 under the deployed sign, blue = below. A feature
that is orange in one dataset and blue in another is exactly the per-cell
selection opportunity (and the fixed-sign risk) this whole experiment is about.</p>

<h2>Per feature &times; cell (46-view grid)</h2>
{heat_main}

<h2>Cross-domain sign check (16-feature pool, mean AUC per domain, 51 cells)</h2>
<p>This table is the evidence that orientation is <em>domain-dependent</em>, not
a one-off bug: <code>hl_ratio</code>, <code>spectral_centroid</code>,
<code>dominant_freq</code> and friends are orange (correctly oriented) on the
reasoning domains where the sign convention was set, and blue (anti-oriented) on
repgrid/qa/rag &mdash; the trace regime changes the feature&rsquo;s polarity.</p>
{heat_hdom}

<h2>The 16-feature pool per cell (all 51 cells)</h2>
{heat_h16}

<h2>What follows from this</h2>
<div class="card"><ul style="margin:4px 0;padding-left:20px">
<li><span class="k">Nothing needs deleting for being noise.</span> The
&ldquo;drop the random features&rdquo; hypothesis is answered: no feature is
flat-noise across the grid, so pruning the pool buys nothing on signal grounds.
The pool's real cost is <em>selector burden</em>, which the bench already
prices in.</li>
<li><span class="k">The sign convention deserves an offline audit.</span> The
{len(anti)} anti-oriented features (whole spilled-energy family included) are a
candidate one-line offline fix in the sign tables &mdash; harmless to L-SML
(weights absorb it), material to U-PCR and to any anchor-correlation objective
(A4's anchor selector used |&rho;|, so it was immune). Any change must be
re-validated against the canonical repgrid scores before adoption.</li>
<li><span class="k">Sign-unstable and domain-flipping features are the per-cell
selection opportunity</span> &mdash; concrete evidence that one fixed global
subset (and one fixed sign) leaves per-cell value on the table, which is the
thesis of the whole selection experiment.</li>
<li><span class="k">Do not delete WEAK features:</span> fusion exists to extract
value from weakly-informative, decorrelated views.</li>
</ul></div>
"""
    return page('Feature value audit — 46-feature pool',
                'Single-feature AUROC of every canonical feature per cell, per '
                'dataset, and overall — asking which features are noise that '
                'could leave the global pool. Answer: none are noise, but 13 '
                'carry the wrong fixed sign.', body)


# ===========================================================================
# main
# ===========================================================================

def main():
    df = load_bench()
    comp = pd.read_csv(os.path.join(BENCH, 'comparison.csv'))
    adm = pd.read_csv(os.path.join(BENCH, 'admissibility_summary.csv'))
    cmeta, pb = cell_meta()
    headline = pd.read_csv(os.path.join(REPG, 'headline_X_vs_Y.csv'))

    print('[deep-report] computing per-feature AUROC matrices ...')
    mats = feature_matrices()
    Mc, _ = mats['c46']
    Mh, _ = mats['h16']
    epr_c46 = Mc.loc['epr'].mean()
    epr_h16 = Mh.loc['epr'].mean()
    print(f'  c46: {Mc.shape[0]} features x {Mc.shape[1]} cells; '
          f'epr-alone {epr_c46:.4f}')
    print(f'  h16: {Mh.shape[0]} features x {Mh.shape[1]} cells; '
          f'epr-alone {epr_h16:.4f}')

    out = {
        'methods_protocol.html': build_methods(adm),
        'experiment_results.html': build_results(df, comp, epr_c46, epr_h16,
                                                 cmeta),
        'benchmark_vs_published.html': build_benchmark(df, cmeta, pb, headline),
        'feature_value_audit.html': build_feature_audit(mats, cmeta),
    }
    for name, htmldoc in out.items():
        path = os.path.join(BENCH, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(htmldoc)
        print(f'[deep-report] wrote {path} ({len(htmldoc) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
