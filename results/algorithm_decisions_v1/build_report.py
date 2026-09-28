"""Builds REPORT_HE.html (Hebrew, RTL) for algorithm_decisions_v1 from a run directory.  Presentation only (no fitting).
Usage: python build_report.py RUN_DIR [OUT_HTML]"""
import json, html, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
RUN = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'run_20260928'); OUTF = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'REPORT_HE.html'
M = pd.read_csv(RUN / 'METRICS.csv'); C = pd.read_csv(RUN / 'CONTRASTS.csv').set_index('contrast_id')
SEL = json.loads((RUN / 'SELECTION.json').read_text(encoding='utf8')); NUL = json.loads((RUN / 'NULLS.json').read_text(encoding='utf8'))
CONC = json.loads((RUN / 'CONCENTRATION.json').read_text(encoding='utf8'))
ACT = [json.loads(l) for l in open(RUN / 'ACTIVITY.jsonl', encoding='utf8')]
GR = pd.read_csv(RUN / 'GROUPS.csv'); LOGO = pd.read_csv(RUN / 'LOGO.csv'); SH = pd.read_csv(RUN / 'SHIFT.csv'); EST = pd.read_csv(RUN / 'ESTIMATES.csv')
MTG = HERE.parent / 'mtg_reproduction_v1'; MP = json.loads((MTG / 'PROTOCOL.json').read_text(encoding='utf8'))['target_table3']; MG = pd.read_csv(MTG / 'GRID.csv')
STATUS = json.loads((RUN / 'RUN_STATUS.json').read_text(encoding='utf8'))

BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
BANK_HE = {'B13': '13', 'B16': '13 + ספרות', 'B20': '20', 'B23': '20 + ספרות', 'B32': '32', 'B35': '32 + ספרות', 'B51': '51', 'B54': '51 + ספרות'}
POS_HE = {'P0': 'בלי נטרול מיקום', 'P1': 'נטרול מיקום בלמידה בלבד', 'P2': 'נטרול מיקום מלא'}
W_HE = {'EQ': 'ממוצע', 'SML': 'SML', 'HEM': 'אומדני hem'}
B_HE = {'EQ': 'משקל שווה', 'SML': 'SML', 'DSM': 'אומדני DS', 'HEM': 'אומדני hem'}
def vname(v):
    p, f = v.split('__')
    if f == 'BASE': return f'ממוצע פשוט ({POS_HE[p]})'
    if f == 'ORACLE': return 'תקרה עם תוויות'
    w, b = f.split('_'); return f'בתוך קבוצה: {W_HE[w]} · בין קבוצות: {B_HE[b]} ({POS_HE[p]})'
def e(s): return html.escape(str(s))
def f4(x, d=4):
    if x is None or (isinstance(x, float) and x != x): return '—'
    s = f'{abs(x):.{d}f}'
    return ('−' + s) if x < 0 and s.strip('0.') else s
def num(x, d=4): return f'<span class="num" dir="ltr">{f4(x, d)}</span>'
def code(c): return f'<code dir="ltr">{e(c)}</code>'
met = {(r.method, r.metric, r.stratum): r.estimate for r in M.itertuples()}
def wa(m): return met.get((m, 'within_auc', 'all'))
def con(a, b, ep='prm_within_auc'):
    k = f'{a} - {b}'
    if k not in C.index or pd.isna(C.loc[k].get(f'{ep}_delta', np.nan)): return None
    r = C.loc[k]; return float(r[f'{ep}_delta']), float(r[f'{ep}_lo']), float(r[f'{ep}_hi'])
def verdict(t):
    if t is None: return 'na', '—'
    d, lo, hi = t
    return ('gain', 'עדיף') if lo > 0 else ('loss', 'נחות') if hi < 0 else ('ns', 'אין הבדל')
def cell(t, span=0.02):
    if t is None: return '<td class="na">—</td>'
    d, lo, hi = t; v, _ = verdict(t); a = min(abs(d) / span, 1) * (0.85 if v != 'ns' else 0.3)
    return f'<td class="hh {"p" if d > 0 else "n"}{" sig" if v != "ns" else ""}" style="--a:{a:.3f}" title="[{f4(lo)}, {f4(hi)}]"><span dir="ltr">{f4(d)}</span></td>'
def tabs(group, items, default=0):
    btns = ''.join(f'<button type="button" role="tab" data-g="{group}" data-i="{i}" aria-selected="{"true" if i == default else "false"}">{lab}</button>' for i, (lab, _) in enumerate(items))
    return f'<div class="tabs" role="tablist">{btns}</div>' + ''.join(f'<div class="panel" data-g="{group}" data-i="{i}"{"" if i == default else " hidden"}>{b}</div>' for i, (_, b) in enumerate(items))

cand = SEL['candidate']; base_mean = SEL['P0__BASE_mean_within_auc_8banks']; TB = pd.DataFrame(SEL['table'])
OF = SEL['one_factor_at_a_time']; centre = OF['centre']

# ---------------- selection table: every variant x bank, delta vs the plain average
rows = []
for r in TB.itertuples():
    tds = ''.join(cell(con(f'{bk}__{r.variant}', f'{bk}__P0__BASE')) for bk in BANKS)
    chip = '<span class="chip gain">מועמד</span>' if r.variant == cand else ('<span class="chip ns">כשיר</span>' if r.eligible else '<span class="chip loss">הפסד בבנק</span>')
    rows.append(f'<tr{" class=cand" if r.variant == cand else ""}><th scope="row">{e(vname(r.variant))} {chip}</th><td class="num">{f4(r.mean_within_auc_8banks)}</td>'
                f'<td class="num">{len(r.wins)}</td><td class="num">{len(r.losses)}</td>{tds}</tr>')
sel_table = (f'<div class="tablewrap"><table class="selt"><caption>כל 38 הווריאנטים, ממוינים לפי הממוצע על 8 הבנקים. בכל תא: ההפרש מהממוצע הפשוט (סינון DS ואז ממוצע, בלי נטרול מיקום) באותו בנק; מודגש = רווח סמך 95% לא כולל אפס. '
             f'הממוצע הפשוט עצמו: {f4(base_mean)} בממוצע על 8 הבנקים.</caption><thead><tr><th scope="col">וריאנט</th><th scope="col">ממוצע 8 בנקים</th><th scope="col">ניצחונות</th><th scope="col">הפסדים</th>'
             + ''.join(f'<th scope="col">{BANK_HE[b]}</th>' for b in BANKS) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')

# ---------------- one factor at a time around the centre
def of_table():
    labs = list(OF['rows'][BANKS[0]].keys())
    head = ''.join(f'<th scope="col">{BANK_HE[b]}</th>' for b in BANKS)
    body = []
    for lab in labs:
        if lab == 'digits_added': continue
        kind, val = lab.split('=') if '=' in lab else (lab, '')
        name = {'position': f'מיקום: {POS_HE.get(val, val)}', 'within': f'בתוך קבוצה: {W_HE.get(val, val)}', 'between': f'בין קבוצות: {B_HE.get(val, val)}', 'plain_average_P0': 'ממוצע פשוט'}[kind]
        tds = []
        for bk in BANKS:
            r = OF['rows'][bk].get(lab, {})
            if 'delta_vs_centre' in r: tds.append(cell((r['delta_vs_centre'], r['lo'], r['hi'])))
            else: tds.append(f'<td class="centre">{f4(r.get("within_auc"))}</td>')
        body.append(f'<tr><th scope="row">{e(name)}</th>{"".join(tds)}</tr>')
    dg = []
    for bk in BANKS:
        r = OF['rows'][bk].get('digits_added'); dg.append(cell((r['delta'], r['lo'], r['hi'])) if r else '<td class="na">—</td>')
    body.append(f'<tr><th scope="row">הוספת ספרות (מול אותו בנק בלי)</th>{"".join(dg)}</tr>')
    return (f'<div class="tablewrap"><table class="selt"><caption>מרכז: {e(vname(centre))}{"" if OF["centre_is_candidate"] else " (המועמד הוא הממוצע הפשוט, ולכן המרכז הוא הווריאנט הטוב ביותר)"}. '
            f'שורה עם ערך אחד = המרכז עצמו (AUC בתוך תשובה); שאר השורות = ההפרש מהמרכז כשמשנים רכיב אחד.</caption><thead><tr><th scope="col">שינוי</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')

# ---------------- position: plain average and L-SML (SML_SML) under P0/P1/P2
def pos_table():
    body = []
    for f, lab in (('BASE', 'ממוצע פשוט'), ('SML_SML', 'L-SML עם איחוד')):
        for p in ('P0', 'P1', 'P2'):
            tds = []
            for bk in BANKS:
                a = f'{bk}__{p}__{f}'; t = None if (p == 'P0' and f == 'BASE') else con(a, f'{bk}__P0__BASE')
                cid = f'{a} - {bk}__P0__BASE'; sw = NUL.get(cid, {}).get('whole_answer_same_length_swap', {}).get('mean')
                if t is None: tds.append(f'<td class="centre">{f4(wa(a))}</td>'); continue
                c_ = cell(t); tds.append(c_[:-5] + (f'<small class="swn">מיקום לבד {f4(sw)}</small>' if sw is not None else '') + '</td>')
            body.append(f'<tr><th scope="row">{lab}, {POS_HE[p]}</th>{"".join(tds)}</tr>')
    return ('<div class="tablewrap"><table class="selt"><caption>הפרש מהממוצע הפשוט בלי נטרול. "מיקום לבד" = ממוצע ההפרש כשמחליפים תוויות בין תשובות באותו אורך (כמה מההפרש מוסבר במיקום).</caption>'
            '<thead><tr><th scope="col">שיטה ומצב מיקום</th>' + ''.join(f'<th scope="col">{BANK_HE[b]}</th>' for b in BANKS) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')

# ---------------- estimates vs truth
def est_tables():
    rows = []
    for bk in BANKS:
        er = EST[(EST.bank == bk) & (EST.learn == 'raw')]
        g = GR[(GR.bank == bk) & (GR.learn == 'raw') & (GR.within == 'EQ')]
        prev_ds = er.groupby('fold').prevalence_hat.first().mean(); prev_true = er.groupby('fold').prevalence_true.first().mean()
        hem_prev = g.groupby('fold').HEM_prevalence.first().mean() if 'HEM_prevalence' in g else np.nan
        dsm_prev = g.groupby('fold').DSM_prevalence.first().mean() if 'DSM_prevalence' in g else np.nan
        mae_ch = float(np.mean(np.abs(er.pi_hat - er.pi_true)))
        mae_dsm = float(np.mean(np.abs(g.DSM_pi - g.pi_true_group_mark))) if 'DSM_pi' in g and g.DSM_pi.notna().any() else np.nan
        rows.append(f'<tr><th scope="row">{BANK_HE[bk]}</th><td>{num(prev_true, 3)}</td><td>{num(prev_ds, 3)}</td><td>{num(dsm_prev, 3)}</td><td>{num(hem_prev, 3)}</td><td>{num(mae_ch, 3)}</td><td>{num(mae_dsm, 3)}</td></tr>')
    return ('<div class="tablewrap"><table class="assum"><caption>ממוצע על 5 folds, למידה על הבנק המקורי. השכיחות האמיתית של צעדים שגויים היא כ-0.14.</caption><thead><tr><th scope="col">בנק</th><th scope="col">שכיחות אמיתית</th>'
            '<th scope="col">שכיחות נאמדת<br><small>DS על ערוצים</small></th><th scope="col">שכיחות נאמדת<br><small>DS על קבוצות</small></th><th scope="col">שכיחות נאמדת<br><small>hem</small></th>'
            '<th scope="col">שגיאה ממוצעת בדיוק<br><small>ערוצים, DS</small></th><th scope="col">שגיאה ממוצעת בדיוק<br><small>קבוצות, DS</small></th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
def logo_table():
    L = LOGO.copy(); L['dprev'] = L.prevalence_after - L.prevalence_before; L['err_before'] = (L.prevalence_before - L.prevalence_true).abs(); L['err_after'] = (L.prevalence_after - L.prevalence_true).abs()
    rows = []
    for bk in BANKS:
        x = L[L.bank == bk]
        if not len(x): continue
        agg = x.groupby('members').agg(dprev=('dprev', 'mean'), dpi=('mean_abs_dpi_hat_others', 'mean'), eb=('err_before', 'mean'), ea=('err_after', 'mean'), mb=('mae_pi_before', 'mean'), ma=('mae_pi_after', 'mean'), n=('fold', 'count')).sort_values('dpi', ascending=False)
        top = agg.iloc[0]
        rows.append(f'<tr><th scope="row">{BANK_HE[bk]}</th><td>{" · ".join(code(c) for c in top.name.strip("[]").replace(chr(39), "").split(", "))}</td><td>{num(top.dpi, 3)}</td><td>{num(top.dprev, 3)}</td>'
                    f'<td>{num(top.eb, 3)} ← {num(top.ea, 3)}</td><td>{num(top.mb, 3)} ← {num(top.ma, 3)}</td></tr>')
    return ('<div class="tablewrap"><table class="assum"><caption>לכל בנק: הקבוצה שהוצאתה מזיזה את אומדני שאר הערוצים הכי הרבה (בלי תוויות), ומה זה עושה מול האמת. ממוצע על folds.</caption><thead><tr><th scope="col">בנק</th><th scope="col">הקבוצה</th>'
            '<th scope="col">תזוזת האומדנים<br><small>ממוצע |שינוי|, בלי תוויות</small></th><th scope="col">שינוי בשכיחות<br><small>בלי תוויות</small></th><th scope="col">שגיאת השכיחות<br><small>לפני ← אחרי</small></th>'
            '<th scope="col">שגיאת הדיוק<br><small>לפני ← אחרי</small></th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
def shift_table():
    rows = []
    for (bw, bo), x in SH.groupby(['with_digits', 'without'], sort=False):
        rows.append(f'<tr><th scope="row">{BANK_HE[bo]} ← {BANK_HE[bw]}</th><td>{num(x.prevalence_without.mean(), 3)} ← {num(x.prevalence_with.mean(), 3)}</td><td>{num(x.mean_dpi_hat_shared.mean(), 3)}</td>'
                    f'<td>{int(x.shared_decisions_identical.sum())} מתוך {len(x)}</td><td>{num(x.spearman_pi_without.mean(), 3)} ← {num(x.spearman_pi_with_shared.mean(), 3)}</td></tr>')
    return ('<div class="tablewrap"><table class="assum"><caption>מה הוספת שלושת פיצ\'רי הספרות עושה לאומדני DS של הערוצים הקיימים (אותו fold).</caption><thead><tr><th scope="col">בנק</th><th scope="col">שכיחות נאמדת<br><small>בלי ← עם</small></th>'
            '<th scope="col">תזוזה ממוצעת בדיוק הנאמד</th><th scope="col">folds עם אותן החלטות סינון</th><th scope="col">דירוג האומדנים מול האמת<br><small>בלי ← עם</small></th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')

# ---------------- ProcessBench per cell with Mind the Gap
PBC = ['pb_gsm8k_q4', 'pb_math_q4', 'pb_olympiadbench_q4', 'pb_omnimath_q4', 'pb_gsm8k_q8', 'pb_math_q8', 'pb_olympiadbench_q8', 'pb_omnimath_q8']
def pb_table():
    rows = []
    def add(name, vals, cls=''):
        mac = float(np.mean(vals)); rows.append((mac, f'<tr class="{cls}"><th scope="row">{name}</th>' + ''.join(f'<td class="num">{v:.2f}</td>' for v in vals) + f'<td class="num"><b>{mac:.2f}</b></td></tr>'))
    col = MP['columns'].index('Shannon Drop')
    add('Mind the Gap, כפי שפורסם (Shannon Drop)', [MP['Qwen3-4B' if c.endswith('q4') else 'Qwen3-8B'][c.split('_')[1]][col] for c in PBC], 'ext')
    x = MG[(MG.signal == 'shannon') & (MG.config == 'drop|paper|earlier|sum') & (MG.dec == 'argmax')].set_index('cell').sla_all
    add('Mind the Gap, השחזור שלנו (אותו כלל החלטה)', [x[c] for c in PBC], 'ext')
    for m, lab in (('ct7', 'CT7'), ('fam421', 'fam421')): add(lab, [100 * met[(m, 'sla', c)] for c in PBC])
    for bk in ('B13', 'B16'):
        for v in dict.fromkeys(['P0__BASE', cand, centre]):
            add(f'{BANK_HE[bk]}: {e(vname(v))}', [100 * met[(f'{bk}__{v}', 'sla', c)] for c in PBC], 'cand' if v == cand else '')
    head = ''.join(f'<th scope="col" dir="ltr">{c.replace("pb_", "").replace("_q4", " 4B").replace("_q8", " 8B")}</th>' for c in PBC)
    return (f'<div class="tablewrap"><table class="assum"><caption>ProcessBench: אחוז התשובות השגויות שבהן הצעד עם הציון הגבוה ביותר הוא השגיאה הראשונה. ממוין לפי הממוצע.</caption><thead><tr><th scope="col">שיטה</th>{head}<th scope="col">ממוצע 8</th></tr></thead><tbody>'
            + ''.join(r for _, r in sorted(rows, key=lambda z: -z[0])) + '</tbody></table></div>')

# ---------------- activity summary
def act_table():
    rows = []
    for bk in BANKS:
        for lr, lab in (('raw', 'מקורי'), ('position_adjusted', 'בלי מיקום')):
            ds = [d for d in ACT if d['bank'] == bk and d['learn'] == lr]
            if not ds: continue
            merged = sum(bool(d.get('merged')) for d in ds); K = [d.get('K') for d in ds]
            guard = sum(any(g[0] == 'cross' for g in (d.get('lsml_fit', {}).get('small_m_guarded') or [])) for d in ds)
            zero = sum(sum(1 for w in (d.get('between', {}).get('EQ', {}).get('DSM') or []) if w == 0) for d in ds)
            small = sum(len([s for s in (d.get('hem', {}).get('source') or []) if s != 'latent']) for d in ds)
            fb = sum(bool(d.get('partition_anchor_fallback')) for d in ds)
            rows.append(f'<tr><th scope="row">{BANK_HE[bk]}, {lab}</th><td>{merged}/5</td><td dir="ltr">{",".join(map(str, K))}</td><td>{guard}/5</td><td>{zero}</td><td>{small}</td><td>{fb}</td></tr>')
    return ('<div class="tablewrap"><table class="assum"><caption>לכל בנק ולכל fold: האם כל רכיב באמת פעל.</caption><thead><tr><th scope="col">בנק ובנק הלמידה</th><th scope="col">האיחוד פעל</th><th scope="col">מספר קבוצות לכל fold</th>'
            '<th scope="col">ה-guard של L-SML בין קבוצות<br><small>(3 קבוצות = משקל שווה)</small></th><th scope="col">קבוצות עם משקל 0<br><small>DS בין קבוצות, סכום</small></th><th scope="col">קבוצות קטנות ב-hem<br><small>סכום</small></th>'
            '<th scope="col">q15_H1 סונן<br><small>עוגן חלופי לחלוקה</small></th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')

cand_he = e(vname(cand)); nw = len([r for r in SEL['table'] if r['qualifies']]); ne = len([r for r in SEL['table'] if r['eligible']])
best = SEL['best_overall']; bestm = next(r['mean_within_auc_8banks'] for r in SEL['table'] if r['variant'] == best)
CSS = (HERE.parent / 'lsml_merge_step_v1' / 'REPORT_HE.html').read_text(encoding='utf8').split('<style>')[1].split('</style>')[0]
CSS += """
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr));gap:10px;margin:.6rem 0 1rem}
.stage{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:10px 12px;display:grid;gap:4px;align-content:start}
.stage .sn{font-family:var(--mono);font-size:.75rem;color:var(--muted)}.stage h3{margin:0;font-size:1rem}.stage p{margin:0;font-size:.86rem}
.stage .dec{font-size:.8rem;border-radius:999px;padding:1px 8px;justify-self:start;border:1px solid currentColor}
.dec.done{color:var(--gain)}.dec.new{color:var(--avg)}
tr.cand th,tr.cand td{background:color-mix(in srgb,var(--gain) 12%,var(--surface))}tr.ext th,tr.ext td{background:color-mix(in srgb,var(--band) 8%,var(--surface))}
td.centre{font-family:var(--mono);text-align:center;font-weight:600}td.na{color:var(--muted);text-align:center}small.swn{display:block;color:var(--muted);font-family:var(--mono);font-size:.68rem;font-weight:400}
table.dec td{font-family:var(--body);text-align:start;font-size:.88rem}
.selt th{white-space:normal;min-width:260px;font-size:.82rem}
"""
JS = """document.querySelectorAll('.tabs button').forEach(function(b){b.addEventListener('click',function(){var g=b.dataset.g,i=b.dataset.i;
document.querySelectorAll('.tabs button[data-g="'+g+'"]').forEach(function(x){x.setAttribute('aria-selected',x.dataset.i===i?'true':'false')});
document.querySelectorAll('.panel[data-g="'+g+'"]').forEach(function(p){p.hidden=p.dataset.i!==i});});});"""
STAGES = [('0', 'ייצוג', 'בכל תשובה: ציון z לכל ערוץ (מטריצה רציפה), וסימון 20% הצעדים העליונים לכל ערוץ (בינארי).', 'קבוע', 'done'),
          ('1', 'סינון', 'DS על הסימונים הבינאריים; ערוץ נשאר אם הדיוק הנאמד שלו מעל 0.5.', 'הוחלט לפני הניסוי', 'done'),
          ('2', 'מיקום', 'האם מנטרלים את המיקום בתשובה בזמן הלמידה.', 'הוכרע בניסוי', 'new'),
          ('3', 'קבוצות', 'חלוקה של L-SML על הסימונים הבינאריים, ואז שלב האיחוד (יחס בליעה מתחת ל-0.5).', 'הוחלט לפני הניסוי', 'done'),
          ('4', 'משקל בתוך קבוצה', 'ממוצע, SML, או אומדני hem.', 'הוכרע בניסוי', 'new'),
          ('5', 'משקל בין קבוצות', 'שווה, SML, אומדני DS, או אומדני hem.', 'הוכרע בניסוי', 'new')]
flow = '<div class="flow">' + ''.join(f'<div class="stage"><span class="sn">שלב {n}</span><h3>{t}</h3><p>{d}</p><span class="dec {c}">{s}</span></div>' for n, t, d, s, c in STAGES) + '</div>'
page = f"""<title>הכרעות באלגוריתם</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Assistant:wght@400;500;600;700&family=Frank+Ruhl+Libre:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap" dir="rtl" lang="he">
<header><h1>הכרעות באלגוריתם</h1>
<p class="lede">ניסוי {code('algorithm_decisions_v1')}. 8 בנקים (13, 20, 32, 51 ערוצים, כל אחד בלי ועם שלושת פיצ'רי הספרות), 3 מצבי מיקום, 12 כללי שקלול, 5 folds לפי קבוצות מקור.
PRMBench: 6,030 תשובות עם צעד שגוי ותקין. ProcessBench: 4,442 תשובות שגויות ב-8 cells. מצב הריצה: {e(STATUS.get('status'))}.</p></header>
<section class="summary" aria-labelledby="sum"><h2 id="sum">בקצרה</h2><ul>
<li><b>המועמד לפי הכלל שהוקפא מראש:</b> {cand_he}, כלומר סינון DS ואחריו ממוצע פשוט של הערוצים שנשארו. {nw} וריאנטים עמדו בכלל הבחירה ו-{ne} היו כשירים: כל וריאנט אחר הפסיד באופן מובהק לפחות בבנק אחד.</li>
<li><b>הקרוב ביותר:</b> {e(vname(best))}, {num(bestm)} בממוצע על 8 הבנקים מול {num(base_mean)} לממוצע הפשוט. רווח אמיתי ורחב ב-32 ערוצים (עם ובלי ספרות), רווח דק ב-13 וב-51 ועוד ספרות, והפסד רחב ב-20 ועוד ספרות וב-51.</li>
<li><b>L-SML</b> נמוך מהממוצע הפשוט בכל 8 הבנקים, גם כשלומדים בלי מיקום.</li>
<li><b>אומדני DS כמשקלים בין קבוצות</b> הם השקלול הטוב ביותר בין קבוצות, ובבנק של 32 ערוצים הם מגיעים לתקרה של משקלים שחושבו עם תוויות. אבל החלוקה לקבוצות עצמה עולה יותר ממה שהם מחזירים ברוב הבנקים.</li>
<li><b>מיקום:</b> לא מנטרלים. נטרול בלמידה בלבד כמעט לא משנה, ונטרול מלא פוגע בכל הבנקים.</li>
<li><b>ספרות:</b> משפרות את הממוצע הפשוט בכל בנק; מעבר למיקום הרווח מובהק ב-20, 32 ו-51 אבל לא ב-13.</li>
</ul></section>
<h2 id="decide">ההחלטות לפי שלבי האלגוריתם</h2>
<div class="tablewrap"><table class="assum dec"><thead><tr><th scope="col">שלב</th><th scope="col">ההחלטה</th><th scope="col">על סמך</th></tr></thead><tbody>
<tr><th scope="row">סינון</th><td>DS על הסימונים הבינאריים, להשאיר ערוץ אם הדיוק הנאמד מעל 0.5</td><td>Steps 451, 455, 456; הספרות לא משנות את ההחלטות</td></tr>
<tr><th scope="row">מיקום</th><td>לא לנטרל</td><td>נטרול בלמידה: שינוי של עד 0.0015, והפסד ב-32 ערוצים; נטרול מלא: הפסד בכל 8 הבנקים</td></tr>
<tr><th scope="row">קבוצות ומשקלים</th><td>לא נכנסים למועמד</td><td>אף שילוב של משקלים בתוך קבוצה ובין קבוצות לא נמנע מהפסד בכל הבנקים</td></tr>
<tr><th scope="row">שימוש באומדנים</th><td>כמסנן: כן. כמשקלים בין קבוצות: רק כווריאנט משני שתלוי בבנק</td><td>עדיף על משקל שווה לקבוצה ב-6 מתוך 8 בנקים (אחרי ניכוי מיקום), מנצח את הממוצע רק ב-32 ערוצים</td></tr>
<tr><th scope="row">בינארי מול רציף</th><td>בינארי לאמידה (סינון, חלוקה, אומדני קבוצה), רציף לציון שמשלבים</td><td>ללא שינוי</td></tr>
<tr><th scope="row">L-SML</th><td>לא במועמד</td><td>נמוך מהממוצע בכל 8 הבנקים</td></tr>
<tr><th scope="row">ספרות</th><td>להוסיף, בכפוף להחלטתך על ההחרגה</td><td>רווח בכל בנק; מעבר למיקום מובהק ב-3 מתוך 4</td></tr>
</tbody></table></div>
<h2 id="algo">האלגוריתם בשלבים</h2>{flow}
<h2 id="selection">כל הווריאנטים מול הממוצע הפשוט</h2>{sel_table}
<h2 id="stages">מה כל רכיב תורם</h2>{of_table()}
<h2 id="position">מיקום: לנטרל ולהשתמש ב-L-SML?</h2>{pos_table()}
<h2 id="estimates">האומדנים מול האמת</h2>{est_tables()}<h3>איזו קבוצה שולטת באומדנים</h3>{logo_table()}<h3>מה הספרות עושות לאומדנים</h3>{shift_table()}
<h2 id="pb">ProcessBench לפי cell, מול Mind the Gap</h2>{pb_table()}
<h2 id="activity">האם כל רכיב פעל</h2>{act_table()}
<h2 id="redteam">מה הצוות האדום תיקן</h2>
<ul>
<li>כל המספרים שוחזרו בקוד עצמאי, כולל בנייה מחדש מהקלטים הגולמיים, והכלל בחר שוב את הממוצע הפשוט.</li>
<li>הרווחים של הווריאנט הקרוב ביותר ב-13 וב-51 ועוד ספרות מרוכזים באחוז אחד של התשובות. רק הרווח ב-32 ערוצים רחב ואמיתי.</li>
<li>המנגנון שהצעתי, שאומדני DS נותנים משקל 0 לקבוצות ברמת ניחוש, הופרך: אין אף קבוצה עם משקל 0 ואף קבוצה ברמת ניחוש. מה שהם עושים בפועל הוא לתת לקבוצת הרמה את המשקל הגבוה ביותר בכל 40 המקרים, עם יחסים מוגזמים בגלל אומדנים מנופחים.</li>
<li>"האיחוד פעל בכל מקום שבו הרמה פוצלה" נכון רק בלמידה על הבנק המקורי (10 מתוך 14 בלמידה בלי מיקום).</li>
<li>"קבוצת הרמה שולטת באומדנים בכל בנק" נכון רק לפי אחד משני מדדים; הוצאתה מקרבת את השכיחות לאמת ב-5 מתוך 8 בנקים, לא 6.</li>
<li>ב-13 ערוצים הרווח של הספרות מעבר למיקום לא מובהק (0.0022).</li>
</ul>
<p class="files lede">קבצים: {code(str(RUN.relative_to(HERE.parent.parent)))} (METRICS.csv, CONTRASTS.csv, SELECTION.json, NULLS.json, ACTIVITY.jsonl, GROUPS.csv, LOGO.csv, SHIFT.csv); פרוטוקול {code('results/algorithm_decisions_v1/PROTOCOL.json')}.</p>
</div><script>{JS}</script>"""
OUTF.write_text(page, encoding='utf8'); print('written', OUTF, len(page))
