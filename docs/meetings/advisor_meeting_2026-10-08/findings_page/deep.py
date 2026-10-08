import re, html
from pathlib import Path
HERE = Path(__file__).parent
SL = HERE.parent / 'deck' / 'project' / 'slides'
MAIN = 'https://claude.ai/artifact/NQNTYpFqVyx6xRnRS4NENV'
BLOB = {'2b4fa8aa81fd23672092e1d6ee944bc0': 'fig/M2_mtg_per_cell_light.png', 'b95dffc295a23c947f2261f0b222b535': 'fig/P1_planted_natural_light.png',
        '75d2e40c4952e4f88dc2f25420b3a5a5': 'fig/G1_gain_prmbench_dssigned_light.png'}
for p in (HERE / 'blobs').glob('*.png'): BLOB[p.stem] = f'blobs/{p.name}'
PAGES = {
    'deep_mtg': ('Mind the Gap Deep Dive', 'How Mind the Gap works, how we reproduced it, and how it compares with our method per cell.',
                 ['metrics', 'competitors', 'mtgmethod', 'mtgvsours', 'pb']),
    'deep_tf': ('Teacher Forcing Deep Dive', 'What teacher forcing is, how we run it, the open prompt-format issue, and the check on answers the model wrote itself.',
                ['teacherforcing', 'tfconfig', 'written', 'ownanswers', 'planted']),
    'deep_pipeline': ('Tail Pipeline Deep Dive', 'The binary tail-vote pipeline of the last three weeks, component by component, and why fusion saturates.',
                      ['candidates', 'learned3w', 'pipeline', 'tails', 'terms', 'tailsq', 'filter', 'merge', 'between', 'mom', 'estacc', 'hem', 'tci', 'withinloss', 'gainA', 'saturation']),
}
CSS = (HERE / 'template.html').read_text(encoding='utf8')
CSS = CSS[CSS.index('<style>'):CSS.index('</style>') + 8]
EXTRA = """<style>
.slide{border-top:2px solid var(--fg);margin-top:40px;padding-top:6px}
.slide div{display:block;min-width:0}.slide p,.slide li{margin:6px 0}
.slide img{background:var(--plate);border-radius:6px;padding:8px;max-width:100%;height:auto!important;width:auto!important}
.slide table{margin:10px 0}.slide td,.slide th{white-space:normal;text-align:left;vertical-align:top}
.slide svg{max-width:100%;height:auto;background:var(--plate);border-radius:6px}
.src{color:var(--muted);font-size:13px}
details{margin-top:10px;border:1px solid var(--line);border-radius:6px;padding:8px 12px}
details div{white-space:pre-line;direction:rtl;text-align:right;font-size:15px}
x-shape,x-icon,x-connector{display:none}
</style>"""


def strip_styles(s):
    parts = re.split(r'(<svg.*?</svg>)', s, flags=re.S)
    return ''.join(p if p.startswith('<svg') else re.sub(r'\s(style|data-[a-z-]+)="[^"]*"', '', p) for p in parts)


def convert(sid, n):
    s = (SL / f'{sid}.html').read_text(encoding='utf8')
    notes = re.search(r'<aside>(.*?)</aside>', s, re.S); notes = notes.group(1).strip() if notes else ''
    s = re.sub(r'<aside>.*?</aside>', '', s, flags=re.S)
    foot = re.search(r'<p style="position:absolute[^"]*">(.*?)</p>', s, re.S)
    src = re.sub(r'^\s*\d+\s*·\s*', '', foot.group(1)) if foot else ''
    s = re.sub(r'<p style="position:absolute[^"]*">.*?</p>', '', s, flags=re.S)
    title = re.search(r'<h[12][^>]*>(.*?)</h[12]>', s, re.S); title = title.group(1).strip() if title else sid
    s = re.sub(r'<h[12][^>]*>.*?</h[12]>', '', s, count=1, flags=re.S)
    s = re.sub(r'^\s*<section[^>]*>', '', s.strip()); s = re.sub(r'</section>\s*$', '', s)
    for bid, path in BLOB.items(): s = s.replace(f'/_blob/{bid}', path)
    missing = re.findall(r'/_blob/[0-9a-f]{32}', s)
    s = strip_styles(s)
    nb = f'<details><summary>Speaker notes (Hebrew)</summary><div>{notes}</div></details>' if notes else ''
    return f'<section class="slide" id="{sid}"><h2>{n}. {title}</h2>{s}<p class="src">Source: {src}</p>{nb}</section>', missing


for page, (title, lede, ids) in PAGES.items():
    body, miss = [], []
    for i, sid in enumerate(ids, 1):
        b, m = convert(sid, i); body.append(b); miss += m
    toc = ''.join(f'<a href="#{sid}">{i}</a>' for i, sid in enumerate(ids, 1))
    out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>{title}</title>{CSS}{EXTRA}</head>
<body><div class="wrap"><nav><a href="{MAIN}">← Findings page</a><a href="deep_mtg.html">Mind the Gap</a><a href="deep_tf.html">Teacher forcing</a><a href="deep_pipeline.html">Tail pipeline</a><span>Slides: {toc}</span></nav>
<h1>{title}</h1><p class="lede">{lede} Collected from the full slide deck; figures are light-theme images.</p>
{''.join(body)}</div></body></html>"""
    (HERE / f'{page}.html').write_text(out, encoding='utf8')
    print(page, len(out), 'missing blobs:', miss)
