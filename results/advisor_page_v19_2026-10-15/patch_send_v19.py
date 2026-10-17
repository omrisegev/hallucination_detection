"""Build the send copy of v19 from the published v19 HTML.

Changes, all display-only (no new scores are computed):
1. Eyebrows keep only the deck/appendix label (drops minutes and v16 section pointers).
2. ES legend name matches slide 2 ("Empirical soundness").
3. Slide s07: STEP_INDEX removed from the heatmap, view H and the text; view H shows
   energy_level, q15_VE1, digit_spread, realized_drv (all N16 channels).
4. New appendix slide a8: pairwise error correlation (phi) between features, read from
   results/signal_gallery_extensions_2026-10-12/PAIRWISE_CONDITIONAL.npz.

Run from the repository root: .venv/bin/python results/advisor_page_v19_2026-10-15/patch_send_v19.py
"""
from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mpl_cache'))
import csv, html, io, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

SRC = HERE / 'published_v19_2026-10-15.html'
DST = HERE / 'published_v19_2026-10-15_send.html'
GALLERY = json.loads((ROOT / 'docs/reviews/signal_visualization_options_20261012/DATA.json').read_text())
FEATURE_CSV = ROOT / 'results/signal_gallery_extensions_2026-10-12/FEATURE_PRMSCORE.csv'
PAIRS = ROOT / 'results/signal_gallery_extensions_2026-10-12/PAIRWISE_CONDITIONAL.npz'
H_FEATURES = ['energy_level', 'q15_VE1', 'digit_spread', 'realized_drv']

# Same plotting style as build_v17.py.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':'#203044','text.color':'#203044',
    'axes.edgecolor':'#cbd5df','xtick.color':'#526173','ytick.color':'#526173',
    'svg.fonttype':'none','svg.hashsalt':'advisor-v19-send','figure.facecolor':'white',
    'axes.facecolor':'white','savefig.facecolor':'white'})

def e(v): return html.escape(str(v), quote=True)
def svg_of(name, fig, alt):
    stream = io.StringIO(); fig.savefig(stream, format='svg', bbox_inches='tight', metadata={'Date':None})
    svg = stream.getvalue(); svg = svg[svg.index('<svg'):]
    svg = re.sub(r'<svg\b', '<svg role="img" aria-label="'+e(alt)+'"', svg, count=1)
    (HERE / 'figures_send').mkdir(exist_ok=True)
    (HERE / 'figures_send' / f'{name}.svg').write_text(svg)
    fig.savefig(HERE / 'figures_send' / f'{name}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return svg
def replace_once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)

s = SRC.read_text()

# 1-2. Eyebrows and ES name.
s, n = re.subn(r'(<div class="eyebrow">[^<]*)<span>[^<]*</span>(</div>)', r'\1\2', s)
assert n == 23
s, n = re.subn(r'<b>ES</b> Counterfactual', '<b>ES</b> Empirical soundness', s)
assert n == 2

# 3. Slide s07 figures.
axes = GALLERY['axes']; keys = [a['key'] for a in axes]; codes = [a['code'] for a in axes]
features = sorted(GALLERY['features'], key=lambda f: f['family']); names = [f['id'] for f in features]
vals = np.array([[GALLERY['metrics']['raw'][k][n]['value'] for k in keys] for n in names])
fig, ax = plt.subplots(figsize=(10, 12.6)); im = ax.imshow(vals, vmin=.35, vmax=.9, cmap='YlGnBu', aspect='auto')
ax.set_yticks(range(56), [f'{f["family"]}  {f["id"]}' for f in features], fontsize=7.5)
ax.set_xticks(range(8), codes); ax.xaxis.tick_top(); ax.tick_params(length=0)
for j in range(1, 56):
    if features[j]['family'] != features[j-1]['family']: ax.axhline(j-.5, color='white', lw=2.5)
for j in range(56):
    for k in range(8): ax.text(k, j, f'{vals[j,k]:.2f}', ha='center', va='center', fontsize=6.5,
                               color='white' if vals[j,k] > .72 else '#203044')
for n_, cols in [('ct7_ve0', [0, 1]), ('digit_spread', [2]), ('energy_level', [4, 6, 7])]:
    j = names.index(n_)
    for k in cols: ax.add_patch(Rectangle((k-.49, j-.49), .98, .98, fill=False, edgecolor='#ff9f1c', lw=2))
fig.colorbar(im, ax=ax, fraction=.022, pad=.02, label='Within-answer AUC'); fig.tight_layout()
heat_cap = ('All 56 signals. Within-type error/valid eligible answers; 6,030 in total. Raw within-answer AUC with '
            'label-free signs; no threshold. Orange outlines guide the three findings. MS has no error/valid pairs and is omitted.')
heat_svg = svg_of('signal_heatmap', fig, heat_cap)

rows = list(csv.DictReader(FEATURE_CSV.open()))
score = {f: {r['type']: float(r['value'])*100 for r in rows if r['feature'] == f} for f in H_FEATURES}
fig, ax = plt.subplots(figsize=(10, 3.4))
for n_, c in zip(H_FEATURES, ['#1766a1', '#d4543c', '#3b8c88', '#6f58a5']):
    ax.plot(range(8), [score[n_][k] for k in keys], '-o', label=n_, color=c, lw=2)
ax.set(xticks=range(8), xticklabels=codes, ylim=(35, 85), ylabel='Per-feature PRMScore ×100')
ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
h_cap = ('Gallery H: individual-feature PRMScore across eight types (6,051 noncontrol answers; MS excluded). '
         'All four are N16 channels; energy_level and realized_drv are also backbone signals on slide 14. '
         'Each feature uses its own donor/source-fold Q80 (≥), which differs from pooled development Q80 for the main fused scores. '
         'No family fusion or CI is shown.')
h_svg = svg_of('selected_feature_prmscore', fig, h_cap)

i = s.index('id="s07"'); j = s.index('</section>', i)
sec = s[i:j]
figs = list(re.finditer(r'<svg.*?</svg>', sec, flags=re.S)); assert len(figs) == 2
caps = list(re.finditer(r'<figcaption>.*?</figcaption>', sec, flags=re.S)); assert len(caps) == 2
new = (sec[:figs[0].start()] + heat_svg + sec[figs[0].end():caps[0].start()] + f'<figcaption>{e(heat_cap)}</figcaption>'
       + sec[caps[0].end():figs[1].start()] + h_svg + sec[figs[1].end():caps[1].start()] + f'<figcaption>{e(h_cap)}</figcaption>'
       + sec[caps[1].end():])
new = replace_once(new, '<p>Digit alternatives help; step position also carries signal.</p>', '<p>Digit alternatives help most.</p>')
new = replace_once(new, '<b>CI</b><p>STEP_INDEX beats every individual channel. Interpret with care.</p>',
                   '<b>CI / SC / ES</b><p>energy_level leads; it is also a backbone signal on slide 14.</p>')
new = replace_once(new, ' הדגשה נוספת: DC אינו נשען על אותו baseline של מיקום חזק כמו CI.', '')
new = replace_once(new, ' השורה האחרונה היא מיקום הצעד.', '')
new = replace_once(new, 'ב־H כל עקומה היא PRMScore של ערוץ יחיד',
                   'ב־H מוצגים ארבעה ערוצים מ־N16, ושניים מהם (energy_level, realized_drv) הם גם אותות ה־backbone של שקף 14. כל עקומה היא PRMScore של ערוץ יחיד')
assert 'STEP_INDEX' not in new
s = s[:i] + new + s[j:]

# 4. Appendix a8: error correlation between features.
z = np.load(PAIRS, allow_pickle=False)
cols = z['columns'].astype(str).tolist(); assert z['contexts'][0] == 'ALL_NONCONTROL'
assert z['truth_classes'].astype(str).tolist() == ['valid', 'error']
fam = {f['id']: f['family'] for f in GALLERY['features']}
inN16 = {f['id']: f['inN16'] for f in GALLERY['features']}
order56 = [f['id'] for f in features]
order16 = [n_ for n_ in order56 if inN16[n_]]; assert len(order16) == 16
PHI = {'error': z['phi'][0, 1], 'valid': z['phi'][0, 0]}
NS = {'error': int(z['sample_counts'][0, 1]), 'valid': int(z['sample_counts'][0, 0])}
def sub(P, order):
    idx = [cols.index(n_) for n_ in order]; return P[np.ix_(idx, idx)]
def summary(order):
    out = {}
    for cl, P in PHI.items():
        Q = sub(P, order); iu = np.triu_indices(len(order), 1); v = Q[iu]
        same = np.array([fam[order[a]] == fam[order[b]] for a, b in zip(*iu)])
        out[cl] = {'all': float(np.nanmean(v)), 'same': float(np.nanmean(v[same])), 'cross': float(np.nanmean(v[~same]))}
    return out
S16, S56 = summary(order16), summary(order56)
norm = TwoSlopeNorm(vmin=-.3, vcenter=0, vmax=1)
TITLES = {'error': 'Missed errors · erroneous steps', 'valid': 'False flags · valid steps'}
def phi_figure(order, numbers):
    n_ = len(order); size = (12, 6.6) if numbers else (12, 7.2)
    fig, axs = plt.subplots(1, 2, figsize=size, constrained_layout=True)
    labels = [f'{fam[x]}  {x}' for x in order]
    for ax, cl in zip(axs, ['error', 'valid']):
        Q = sub(PHI[cl], order).copy(); np.fill_diagonal(Q, np.nan)
        im = ax.imshow(np.ma.masked_invalid(Q), cmap='RdBu_r', norm=norm, aspect='equal')
        ax.set_title(f'{TITLES[cl]} (n = {NS[cl]:,} steps)', fontsize=10.5, loc='left')
        fs = 7.5 if numbers else 5
        ax.set_xticks(range(n_), [fam[x] if not numbers else x for x in order], rotation=90, fontsize=fs)
        ax.set_yticks(range(n_), labels if ax is axs[0] else ['']*n_, fontsize=fs); ax.tick_params(length=0)
        for b in range(1, n_):
            if fam[order[b]] != fam[order[b-1]]:
                ax.axhline(b-.5, color='#203044', lw=.8); ax.axvline(b-.5, color='#203044', lw=.8)
        if numbers:
            for a in range(n_):
                for b in range(n_):
                    if a != b and np.isfinite(Q[a, b]):
                        ax.text(b, a, f'{Q[a,b]:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=5.6,
                                color='white' if abs(Q[a, b]) > .55 else '#203044')
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.colorbar(im, ax=axs, fraction=.025, pad=.01, label='φ of mistake indicators')
    return fig
cap16 = ('N16 channels, ordered by family; dark lines separate families. φ = correlation between two features\' mistake indicators '
         'at each feature\'s own Q80 cut, computed separately on erroneous steps (missed errors) and valid steps (false flags). '
         'All 6,211 noncontrol answers, steps pooled. Diagonal omitted.')
cap56 = 'All 56 channels, same definition and order as the N16 view; axis ticks show the family.'
svg16 = svg_of('error_phi_n16', phi_figure(order16, True), cap16)
svg56 = svg_of('error_phi_n56', phi_figure(order56, False), cap56)
f2 = lambda v: f'{v:.2f}'
a8 = f'''<section class="slide appendix-slide" id="a8" aria-labelledby="a8-title"><div class="eyebrow">APPENDIX</div><h1 id="a8-title">Do our features make the same mistakes?</h1><p class="lead">Correlation of mistakes between pairs of features, each flagged at its own Q80 cut. Mistakes are correlated mainly inside a family.</p><div class="tabs"><button class="view active" data-panel="phi16">N16 · 16 channels</button><button class="view" data-panel="phi56">All 56 channels</button></div><div data-view="phi16"><figure class="chart">{svg16}<figcaption>{e(cap16)}</figcaption></figure></div><div data-view="phi56" hidden><figure class="chart">{svg56}<figcaption>{e(cap56)}</figcaption></figure></div><div class="callout "><strong>How to read it</strong><p>A mistake is a missed error on an erroneous step, or a false flag on a valid step. φ = 0: two features make unrelated mistakes; φ = 1: they always make the same ones.</p></div><ul class="points"><li>N16 average φ: {f2(S16['error']['all'])} for missed errors, {f2(S16['valid']['all'])} for false flags.</li><li>All 56: inside a family {f2(S56['error']['same'])} / {f2(S56['valid']['same'])}; across families {f2(S56['error']['cross'])} / {f2(S56['valid']['cross'])} (missed errors / false flags).</li><li>The PRM is not in this matrix; its error correlation with our features is not measured here.</li></ul><div class="callout research"><strong>Limit</strong><p>Pairwise only: small φ does not establish joint independence. Steps are pooled across answers with no adjustment for answer, position or length, and no interval. It measures binary mistakes at Q80, not the continuous scores, which are far more correlated.</p></div><aside class="speaker" lang="he" dir="rtl"><h3>הערות דובר</h3><p>השקף עונה על השאלה האם הפיצ׳רים שלנו טועים יחד. לכל פיצ׳ר מסמנים שגיאה לפי סף Q80 משלו, ובודקים זוגות: באיזו מידה שני פיצ׳רים מפספסים את אותם צעדים שגויים, ובאיזו מידה הם מסמנים בטעות את אותם צעדים תקינים.</p><p><b>מה רואים:</b> הקורלציה גבוהה בעיקר בתוך משפחה. בין משפחות היא נמוכה בממוצע. כלומר פיצ׳רים מאותה משפחה נוטים לטעות על אותם צעדים, ופיצ׳רים ממשפחות שונות פחות.</p><p><b>גבול המסקנה:</b> זו קורלציה זוגית על החלטות בינאריות, ולא בדיקת אי־תלות. הציונים הרציפים עצמם מתואמים הרבה יותר. ה־PRM לא נמצא במטריצה הזאת.</p></aside><div class="source"><b>Evidence</b> results/signal_gallery_extensions_2026-10-12/PAIRWISE_CONDITIONAL.npz (phi, context ALL_NONCONTROL); scripts/build_signal_gallery_extensions.py (pairwise_binary_errors); drawn by results/advisor_page_v19_2026-10-15/patch_send_v19.py</div></section>'''
k = s.index('id="a7"'); end = s.index('</section>', k) + len('</section>')
s = s[:end] + a8 + s[end:]
s = replace_once(s, '<span class="nav-num">A07</span>Paper reference rows and evaluator conventions</a>',
                 '<span class="nav-num">A07</span>Paper reference rows and evaluator conventions</a><a href="#a8"><span class="nav-num">A08</span>Do our features make the same mistakes?</a>')
DST.write_text(s)
print(json.dumps({'N16': S16, 'N56': S56, 'n': NS}, indent=1))
print('wrote', DST.name, len(s))
