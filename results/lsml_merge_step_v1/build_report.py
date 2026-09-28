"""Builds REPORT_HE.html (Hebrew, RTL) for lsml_merge_step_v1 from report_data.json. Presentation only."""
import json, html
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'report_data.json').read_text(encoding='utf8'))

BANKS = ['B13', 'B16', 'B20', 'B32', 'B51']
BANK_HE = {'B13': '13 ערוצים', 'B16': '13 + ספרות', 'B20': '20 ערוצים', 'B32': '32 ערוצים', 'B51': '51 ערוצים'}
METHODS = [  # (code, Hebrew name, family)
    ('DSF_equal', 'ממוצע פשוט אחרי סינון DS', 'avg'),
    ('ALL_equal', 'ממוצע של כל הערוצים', 'avg'),
    ('SF_equal', 'סינון פשוט (מתאם), ואז ממוצע', 'avg'),
    ('BD_equal', 'כלל הרצועה בלי היפוך, ואז ממוצע', 'band'),
    ('BF_equal', 'כלל הרצועה עם היפוך, ואז ממוצע', 'band'),
    ('ALL_lsml', 'L-SML על כל הערוצים', 'lsml'),
    ('DSF_lsml', 'L-SML כמו שהוא (אחרי סינון DS)', 'lsml'),
    ('DSF_lsml_mC', 'L-SML עם איחוד על החלוקה שלו', 'lsml'),
    ('DSF_lsml_B', 'L-SML על החלוקה הבינארית', 'lsml'),
    ('DSF_lsml_mB', 'L-SML על החלוקה הבינארית אחרי איחוד', 'lsml'),
    ('BF_lsml', 'כלל הרצועה עם היפוך, ואז L-SML', 'band'),
    ('BF_lsml_mB', 'כלל הרצועה עם היפוך, L-SML על בינארית אחרי איחוד', 'band'),
    ('DSF_Gbin', 'משקל שווה לקבוצה: חלוקה בינארית', 'group'),
    ('DSF_GmB', 'משקל שווה לקבוצה: בינארית אחרי איחוד', 'group'),
    ('DSF_Gcont', 'משקל שווה לקבוצה: החלוקה של L-SML', 'group'),
    ('DSF_GmC', 'משקל שווה לקבוצה: החלוקה של L-SML אחרי איחוד', 'group'),
]
REFS = [('ct7', 'CT7 (ייחוס)'), ('fam421', 'fam421 (ייחוס)')]
NAME = {c: h for c, h, _ in METHODS} | dict(REFS)
FAM = {c: f for c, _, f in METHODS} | {'ct7': 'ref', 'fam421': 'ref'}
NUM = {c: i + 1 for i, (c, _, _) in enumerate(METHODS)} | {'ct7': 17, 'fam421': 18}
FAM_HE = {'avg': 'ממוצע', 'lsml': 'L-SML', 'group': 'משקל שווה לקבוצה', 'band': 'כלל הרצועה', 'ref': 'ייחוס'}

met = {}
for r in D['metrics']:
    met[(r['method'], r['metric'], r['stratum'])] = r['estimate']
def mv(bk, code, metric, stratum='all'):
    key = code if code in ('ct7', 'fam421', 'step_index') else f'{bk}__{code}'
    return met.get((key, metric, stratum))
CON = {r['contrast_id']: r for r in D['contrasts'] if r.get('endpoint') == 'prm_within_auc'}
PBCON = {r['contrast_id']: r for r in D['pb_contrasts']}

def e(s): return html.escape(str(s))
def f4(x, d=4):
    if x is None or x != x: return '—'
    s = f'{abs(x):.{d}f}'
    return ('−' + s) if x < 0 and s.strip('0.') else s
def num(x, d=4, cls='num'): return f'<span class="{cls}" dir="ltr">{f4(x, d)}</span>'
def code(c): return f'<code dir="ltr">{e(c)}</code>'

# ------------------------------------------------------------------ 1. all methods x banks (difference bars)
def all_methods_block(metric, stratum, label, span):
    rows = []
    for c, h, fam in METHODS + [(r[0], r[1], 'ref') for r in REFS]:
        cells = []
        for bk in BANKS:
            v = mv(bk, c, metric, stratum); base = mv(bk, 'DSF_equal', metric, stratum)
            if v is None or v != v: cells.append('<td class="dcell na">לא רץ</td>'); continue
            d = v - base; w = min(abs(d) / span, 1) * 50
            side = 'r' if d >= 0 else 'l'
            bar = '' if c == 'DSF_equal' else f'<i class="bar {fam} {side}" style="width:{w:.1f}%"></i>'
            cells.append(f'<td class="dcell"><div class="track" dir="ltr"><i class="zero"></i>{bar}</div>'
                         f'<div class="dv"><b dir="ltr">{f4(v)}</b>{"" if c == "DSF_equal" else f"<span class=dd dir=ltr>{f4(d)}</span>"}</div></td>')
        rows.append(f'<tr class="fam-{fam}{" base" if c == "DSF_equal" else ""}"><th scope="row"><span class="mnum">{NUM[c]}</span>'
                    f'<span class="sw {fam}"></span>{e(h)}</th>{"".join(cells)}</tr>')
    head = ''.join(f'<th scope="col">{BANK_HE[b]}</th>' for b in BANKS)
    return (f'<div class="tablewrap"><table class="allm"><caption>{label}. בכל תא: הערך, וההפרש מהממוצע הפשוט אחרי סינון DS באותו בנק '
            f'(פס ימינה = טוב ממנו, שמאלה = גרוע ממנו; קצה הסקאלה {f4(span, 3)}).</caption>'
            f'<thead><tr><th scope="col">שיטה</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

# ------------------------------------------------------------------ 2. yes/no questions (forest)
QUESTIONS = [
    ('DSF_lsml_mC', 'DSF_lsml', 'האם שלב האיחוד משפר את L-SML על החלוקה שלו?', True),
    ('DSF_lsml_mB', 'DSF_lsml', 'האם חלוקה בינארית ואיחוד משפרים את L-SML כמו שהוא?', True),
    ('DSF_lsml_mB', 'DSF_equal', 'השאלה המרכזית: האם L-SML עם האיחוד עוקף ממוצע פשוט של אותם ערוצים?', True),
    ('BF_equal', 'DSF_equal', 'האם כלל הרצועה (הורדה והיפוך) עדיף על סינון DS?', True),
    ('BD_equal', 'DSF_equal', 'האם הורדת הרצועה 0.45 עד 0.55 לבדה עוזרת?', False),
    ('BF_equal', 'BD_equal', 'האם היפוך הערוצים ההפוכים עוזר לממוצע?', False),
    ('DSF_lsml_B', 'DSF_lsml', 'חלוקה בינארית במקום החלוקה של L-SML, עם משקלי L-SML', False),
    ('DSF_GmB', 'DSF_equal', 'האם משקל שווה לכל קבוצה (אחרי איחוד) עוקף ממוצע פשוט?', False),
    ('DSF_lsml_mB', 'DSF_GmB', 'האם משקלי L-SML מוסיפים על משקל שווה לקבוצות באותה חלוקה?', False),
    ('ALL_lsml', 'ALL_equal', 'L-SML על כל הערוצים מול ממוצע של כל הערוצים', False),
]
def h2h_pair(bk, a, b):
    H = D['h2h'][bk]; m = H['methods']
    if a not in m or b not in m: return None
    i, j = m.index(a), m.index(b); return H['delta'][i][j], H['lo'][i][j], H['hi'][i][j]
def question_block(a, b, q, primary):
    rows = []; vals = []
    for bk in BANKS:
        cid = f'{bk}__{a} - {bk}__{b}'; r = CON.get(cid)
        if r is not None and r.get('delta') == r.get('delta'):
            d = r['delta']; lo, hi = (r['ci_adj_lo'], r['ci_adj_hi']) if primary else (r['ci95_lo'], r['ci95_hi'])
        else:
            t = h2h_pair(bk, a, b)
            if t is None: rows.append((bk, None)); continue
            d, lo, hi = t
        pos = CON.get(f'{bk}__{a}_pos - {bk}__{b}_pos'); swap = D['nulls'].get(cid, {}).get('whole_answer_same_length_swap', {}).get('mean') if primary else None
        rows.append((bk, (d, lo, hi, pos['delta'] if pos else None, swap))); vals += [lo, hi] + ([pos['delta']] if pos else []) + ([swap] if swap is not None else [])
    span = max(0.004, max(abs(v) for v in vals) * 1.08) if vals else 0.01
    def x(v): return 50 + 50 * v / span
    out = []
    for bk, t in rows:
        if t is None: out.append(f'<div class="frow"><span class="fb">{BANK_HE[bk]}</span><span class="fnote">לא רץ על בנק זה</span></div>'); continue
        d, lo, hi, pd_, sw = t; verdict = 'gain' if lo > 0 else 'loss' if hi < 0 else 'ns'
        vtxt = {'gain': 'עדיף', 'loss': 'נחות', 'ns': 'אין הבדל מובהק'}[verdict]
        if abs(d) < 1e-12 and abs(lo) < 1e-12 and abs(hi) < 1e-12: verdict, vtxt = 'ns', 'השלב לא פעל'
        marks = f'<i class="ci {verdict}" style="left:{x(lo):.2f}%;width:{max(x(hi) - x(lo), .4):.2f}%"></i><i class="pt {verdict}" style="left:{x(d):.2f}%"></i>'
        if pd_ is not None: marks += f'<i class="pp" style="left:{x(pd_):.2f}%" title="אחרי הסרת מיקום"></i>'
        if sw is not None: marks += f'<i class="sw0" style="left:{x(sw):.2f}%" title="מה שמיקום לבד נותן"></i>'
        extra = (f'<span class="fx">אחרי הסרת מיקום {num(pd_)}</span>' if pd_ is not None else '') + (f'<span class="fx">מיקום לבד {num(sw)}</span>' if sw is not None else '')
        out.append(f'<div class="frow"><span class="fb">{BANK_HE[bk]}</span><div class="ftrack" dir="ltr"><i class="zero"></i>{marks}</div>'
                   f'<span class="fv">{num(d)} <span class="fci" dir="ltr">[{f4(lo)}, {f4(hi)}]</span></span><span class="chip {verdict}">{vtxt}</span>{extra}</div>')
    tag = '<span class="tag prim">שאלה ראשית, רווח סמך מתוקן (Bonferroni, 40 השוואות)</span>' if primary else '<span class="tag">שאלה משנית, רווח סמך 95%</span>'
    return (f'<article class="q"><h3>{e(q)}</h3><p class="qsub">{e(NAME[a])} <span class="minus">פחות</span> {e(NAME[b])} {tag}</p>'
            f'<div class="fscale" dir="ltr"><span>{f4(-span, 3)}</span><span>0</span><span>{f4(span, 3)}</span></div>{"".join(out)}</article>')

# ------------------------------------------------------------------ 3. head-to-head matrices
def h2h_block(bk):
    H = D['h2h'][bk]; m = H['methods']; order = [c for c, _, _ in METHODS if c in m] + [c for c, _ in REFS]
    idx = [m.index(c) for c in order]
    head = ''.join(f'<th scope="col" title="{e(NAME[c])}"><span class="mnum">{NUM[c]}</span></th>' for c in order)
    rows = []
    for c, i in zip(order, idx):
        tds = []
        for c2, j in zip(order, idx):
            if i == j: tds.append('<td class="diag">—</td>'); continue
            d, lo, hi = H['delta'][i][j], H['lo'][i][j], H['hi'][i][j]; sig = lo > 0 or hi < 0
            a = min(abs(d) / 0.02, 1) * (0.85 if sig else 0.35)
            tds.append(f'<td class="hh {"p" if d > 0 else "n"}{" sig" if sig else ""}{" hi" if a > .55 else ""}" style="--a:{a:.3f}" title="{e(NAME[c])} פחות {e(NAME[c2])}: {f4(d)} [{f4(lo)}, {f4(hi)}]"><span dir="ltr">{f4(d, 4)}</span></td>')
        rows.append(f'<tr><th scope="row"><span class="mnum">{NUM[c]}</span>{e(NAME[c])} <span class="mean" dir="ltr">{f4(H["mean"][i])}</span></th>{"".join(tds)}</tr>')
    return (f'<div class="tablewrap"><table class="h2h"><caption>{BANK_HE[bk]}: כל תא = השיטה בשורה פחות השיטה בעמודה (AUC בתוך תשובה, PRMBench). '
            f'מודגש עם מסגרת = רווח הסמך 95% לא כולל אפס (bootstrap מזווג לפי קבוצות מקור, 4,000 הגרלות, בלי תיקון להשוואות מרובות: מבט חקרני).</caption>'
            f'<thead><tr><th scope="col">שיטה (ממוצע)</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

# ------------------------------------------------------------------ 4. per cell
PBC = ['pb_gsm8k_q4', 'pb_gsm8k_q8', 'pb_math_q4', 'pb_math_q8', 'pb_olympiadbench_q4', 'pb_olympiadbench_q8', 'pb_omnimath_q4', 'pb_omnimath_q8']
PBC_HE = {c: c.replace('pb_', '').replace('_q', ' q') for c in PBC}
CLASSES = sorted({s.split('=')[1] for (_, _, s) in met if s.startswith('class=') and s != 'class=correct'})
def cell_block(bk, cols, title):
    head = ''.join(f'<th scope="col">{h}</th>' for _, _, _, h in cols)
    rows = []
    for c, h, fam in METHODS + [(r[0], r[1], 'ref') for r in REFS]:
        if mv(bk, c, 'within_auc') is None: continue
        tds = []
        for metric, stratum, span, _ in cols:
            v = mv(bk, c, metric, stratum); base = mv(bk, 'DSF_equal', metric, stratum)
            if v is None or v != v: tds.append('<td>—</td>'); continue
            d = v - base; a = min(abs(d) / span, 1) * 0.8
            tds.append(f'<td class="hh {"p" if d > 0 else "n"}{" hi" if c != "DSF_equal" and a > .55 else ""}" style="--a:{0 if c == "DSF_equal" else a:.3f}" title="הפרש מהממוצע הפשוט: {f4(d)}"><span dir="ltr">{f4(v)}</span></td>')
        rows.append(f'<tr class="{"base" if c == "DSF_equal" else ""}"><th scope="row"><span class="mnum">{NUM[c]}</span><span class="sw {fam}"></span>{e(h)}</th>{"".join(tds)}</tr>')
    return (f'<div class="tablewrap"><table class="cells"><caption>{title} צבע: ירוק = טוב מהממוצע הפשוט אחרי סינון DS באותה עמודה, אדום = גרוע ממנו.</caption>'
            f'<thead><tr><th scope="col">שיטה</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
CELL_COLS = [('within_auc', 'all', 0.02, 'PRMBench<br><small>AUC בתוך תשובה</small>'), ('prmscore', 'all', 0.02, 'PRMBench<br><small>PRMScore</small>')] + \
            [('sla', c, 0.05, f'<span dir="ltr">{PBC_HE[c]}</span><br><small>ProcessBench</small>') for c in PBC] + [('sla', 'macro8', 0.04, 'ProcessBench<br><small>ממוצע 8</small>')]
CLASS_COLS = [('within_auc', f'class={c}', 0.03, f'<span dir="ltr">{c.replace("_", " ")}</span>') for c in CLASSES if any(v == v for (m_, mt, st), v in met.items() if st == f'class={c}' and mt == 'within_auc' and v is not None)]

# ------------------------------------------------------------------ 5. the L-SML assumption
SRC_HE = {'part_cont': 'החלוקה של L-SML', 'part_contM': 'החלוקה של L-SML אחרי איחוד', 'part_bin': 'חלוקה בינארית', 'part_binM': 'חלוקה בינארית אחרי איחוד'}
MK_HE = {'values': 'ערכים רציפים', 'marks': 'סימוני 20% העליונים', 'values_position_removed': 'ערכים רציפים אחרי הסרת מיקום'}
def assum_table(bk, mk):
    A = D['assumption'][bk]; rows = []
    for s in ('part_cont', 'part_contM', 'part_bin', 'part_binM'):
        st = A['stats'].get(f'{mk}|{s}')
        if st is None: continue
        rd = A['random'][f'{mk}|{s}']; K = len(A['partitions'][s]['groups'])
        rows.append(f'<tr><th scope="row">{SRC_HE[s]} <small>({K} קבוצות)</small></th><td>{num(st["clean"]["between_mean"], 3)}</td><td>{num(st["error"]["between_mean"], 3)}</td>'
                    f'<td>{num(max(st["clean"]["between_max"], st["error"]["between_max"]), 3)}</td><td>{num(100 * (st["clean"]["between_share_gt_0.1"] + st["error"]["between_share_gt_0.1"]) / 2, 0)}%</td>'
                    f'<td>{num(rd["random_mean"], 3)}</td><td>{num((st["clean"]["within_mean"] + st["error"]["within_mean"]) / 2, 3)}</td></tr>')
    return ('<div class="tablewrap"><table class="assum"><thead><tr><th scope="col">חלוקה</th><th scope="col">בין קבוצות, צעדים תקינים<br><small>ממוצע |מתאם|</small></th>'
            '<th scope="col">בין קבוצות, צעדים שגויים<br><small>ממוצע |מתאם|</small></th><th scope="col">בין קבוצות<br><small>מקסימום</small></th><th scope="col">זוגות בין קבוצות<br><small>עם |מתאם| מעל 0.1</small></th>'
            '<th scope="col">חלוקה אקראית<br><small>באותם גדלים, בין קבוצות</small></th><th scope="col">בתוך קבוצות<br><small>ממוצע |מתאם|</small></th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
def heat(bk, mk, cls):
    A = D['assumption'][bk]; C = A['matrices'][mk][cls]; g = A['partitions']['part_binM']['labels']; ch = A['channels']
    order = sorted(range(len(ch)), key=lambda j: (g[j], j)); cells = []
    for a in order:
        tds = []
        for b in order:
            v = C[a][b]; edge = (' gl' if b != order[0] and g[b] != g[order[order.index(b) - 1]] else '') + (' gt' if a != order[0] and g[a] != g[order[order.index(a) - 1]] else '')
            if a == b: tds.append(f'<td class="cm diag{edge}"></td>'); continue
            tds.append(f'<td class="cm {"p" if v > 0 else "n"}{" same" if g[a] == g[b] else ""}{" hi" if abs(v) / .6 > .55 else ""}{edge}" style="--a:{min(abs(v) / .6, 1):.3f}" title="{e(ch[a])} × {e(ch[b])}: {f4(v, 3)}"><span dir="ltr">{int(round(abs(v) * 100))}</span></td>')
        cells.append(f'<tr><th scope="row" dir="ltr"><span class="gdot g{g[a] % 6}"></span>{e(ch[a])}</th>{"".join(tds)}</tr>')
    return (f'<figure class="hm"><figcaption>{"צעדים תקינים" if cls == "clean" else "צעדים שגויים"}</figcaption><div class="tablewrap"><table class="corr" dir="ltr">'
            f'<tbody>{"".join(cells)}</tbody></table></div></figure>')
def groups_list(bk):
    A = D['assumption'][bk]; gs = A['partitions']['part_binM']['groups']
    return '<ol class="groups">' + ''.join(f'<li><span class="gdot g{i % 6}"></span>' + ' · '.join(code(c) for c in gr) + '</li>' for i, gr in enumerate(gs)) + '</ol>'
def top_pairs(bk, mk):
    tp = D['assumption'][bk]['top_pairs'][mk]
    return ('<div class="tablewrap"><table class="pairs"><thead><tr><th scope="col">זוג ערוצים מקבוצות שונות</th><th scope="col">מתאם, צעדים תקינים</th><th scope="col">מתאם, צעדים שגויים</th></tr></thead><tbody>'
            + ''.join(f'<tr><td>{code(a)} × {code(b)}</td><td>{num(c, 3)}</td><td>{num(er, 3)}</td></tr>' for _, a, b, c, er in tp[:6]) + '</tbody></table></div>')

# ------------------------------------------------------------------ tabs helper
def tabs(group, items, default=0):
    btns = ''.join(f'<button type="button" role="tab" id="t-{group}-{i}" data-g="{group}" data-i="{i}" aria-selected="{"true" if i == default else "false"}">{lab}</button>' for i, (lab, _) in enumerate(items))
    panels = ''.join(f'<div class="panel" data-g="{group}" data-i="{i}"{"" if i == default else " hidden"}>{body}</div>' for i, (_, body) in enumerate(items))
    return f'<div class="tabs" role="tablist">{btns}</div>{panels}'

# ------------------------------------------------------------------ numbers used in the prose
def c(bk, a, b): return CON[f'{bk}__{a} - {bk}__{b}']
cen = {bk: c(bk, 'DSF_lsml_mB', 'DSF_equal')['delta'] for bk in BANKS}
posd = {bk: CON[f'{bk}__DSF_lsml_mB_pos - {bk}__DSF_equal_pos']['delta'] for bk in BANKS}
swp = {bk: D['nulls'][f'{bk}__DSF_lsml_mB - {bk}__DSF_equal']['whole_answer_same_length_swap']['mean'] for bk in BANKS}
AS = D['assumption']
def bm(bk, mk, s): st = AS[bk]['stats'][f'{mk}|{s}']; return (st['clean']['between_mean'] + st['error']['between_mean']) / 2
pdig = D['posthoc_digits']

pos_rows = ''.join(f'<tr><th scope="row">{BANK_HE[bk]}</th><td>{num(cen[bk])}</td><td>{num(swp[bk])}</td><td>{num(posd[bk])}</td></tr>' for bk in BANKS)
assum_rows = ''.join(f'<tr><th scope="row">{BANK_HE[bk]}</th><td>{num(bm(bk, "values", "part_binM"), 3)}</td><td>{num(AS[bk]["random"]["values|part_binM"]["random_mean"], 3)}</td>'
                     f'<td>{num(bm(bk, "values_position_removed", "part_binM"), 3)}</td><td>{num(bm(bk, "marks", "part_binM"), 3)}</td>'
                     f'<td>{num(max(AS[bk]["stats"]["values|part_binM"]["clean"]["between_max"], AS[bk]["stats"]["values|part_binM"]["error"]["between_max"]), 3)}</td></tr>' for bk in BANKS)

metric_tabs = tabs('all', [('AUC בתוך תשובה (PRMBench)', all_methods_block('within_auc', 'all', 'PRMBench, AUC בתוך תשובה, 6,030 תשובות', 0.03)),
                           ('PRMScore (PRMBench)', all_methods_block('prmscore', 'all', 'PRMBench, PRMScore רשמי, 6,969 תשובות', 0.03)),
                           ('ProcessBench', all_methods_block('sla', 'macro8', 'ProcessBench, מיקום השגיאה הראשונה בלי שער, ממוצע 8 cells', 0.05))])
h2h_tabs = tabs('h2h', [(BANK_HE[bk], h2h_block(bk)) for bk in BANKS])
cell_tabs = tabs('cell', [(BANK_HE[bk], cell_block(bk, CELL_COLS, f'{BANK_HE[bk]}: כל השיטות על כל cell.')) for bk in BANKS])
class_tabs = tabs('cls', [(BANK_HE[bk], cell_block(bk, CLASS_COLS, f'{BANK_HE[bk]}: AUC בתוך תשובה לפי סוג השגיאה ב-PRMBench.')) for bk in BANKS])
assum_tabs = tabs('as', [(BANK_HE[bk], tabs(f'as{bk}', [(MK_HE[mk], assum_table(bk, mk)) for mk in ('values', 'marks', 'values_position_removed')])) for bk in BANKS])
heat_tabs = tabs('hm', [(BANK_HE[bk], groups_list(bk) + tabs(f'hm{bk}', [(MK_HE[mk], f'<div class="hmpair">{heat(bk, mk, "clean")}{heat(bk, mk, "error")}</div>' + top_pairs(bk, mk))
                                                                       for mk in ('values', 'values_position_removed', 'marks')])) for bk in ('B13', 'B16', 'B20')])
questions = ''.join(question_block(*q) for q in QUESTIONS)
digits_parts = ''.join(f'<li><span class="gdot g{i % 6}"></span>' + ' · '.join(code(x) for x in g) + '</li>' for i, g in enumerate(D['partitions']['B16']['bin']['per_fold'][0]['after']))

CSS = """
:root{--ground:#F2F5F7;--surface:#FFFFFF;--ink:#16202A;--muted:#566272;--rule:#D8DEE5;--soft:#E9EEF2;
--avg:#1F5E8E;--lsml:#C2622A;--group:#5E7D26;--band:#7650A6;--ref:#8A94A0;--gain:#1F7A55;--loss:#B2372F;--ns:#A3ADB8;--pos:#1F5E8E;--neg:#C2622A;
--display:'Frank Ruhl Libre',Georgia,serif;--body:'Assistant','Segoe UI',Arial,sans-serif;--mono:'IBM Plex Mono',Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--ground:#0E141A;--surface:#151E27;--ink:#E3E9EF;--muted:#9AA7B4;--rule:#283440;--soft:#1C2732;
--avg:#6FA7D8;--lsml:#E48A52;--group:#A5C465;--band:#B38FE0;--ref:#7F8A96;--gain:#5BC291;--loss:#E57A73;--ns:#5C6773;--pos:#6FA7D8;--neg:#E48A52}}
:root[data-theme="dark"]{color-scheme:dark;--ground:#0E141A;--surface:#151E27;--ink:#E3E9EF;--muted:#9AA7B4;--rule:#283440;--soft:#1C2732;
--avg:#6FA7D8;--lsml:#E48A52;--group:#A5C465;--band:#B38FE0;--ref:#7F8A96;--gain:#5BC291;--loss:#E57A73;--ns:#5C6773;--pos:#6FA7D8;--neg:#E48A52}
body{background:var(--ground);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.6}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
h1,h2{font-family:var(--display);font-weight:700;text-wrap:balance;line-height:1.2}
h1{font-size:2.3rem;margin:0 0 .3rem}h2{font-size:1.6rem;margin:2.8rem 0 .6rem;padding-top:1rem;border-top:1px solid var(--rule)}
h3{font-size:1.1rem;margin:0 0 .2rem;text-wrap:balance}
p,li{max-width:72ch}.lede{color:var(--muted);font-size:.95rem;margin:0}
code{font-family:var(--mono);font-size:.82em;background:var(--soft);padding:0 .3em;border-radius:3px;white-space:nowrap}
.num,.fci,.dd,.mean{font-family:var(--mono);font-variant-numeric:tabular-nums}
.summary{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:18px 22px;margin-top:22px;display:grid;gap:10px}
.summary h2{border:0;margin:0;padding:0;font-size:1.25rem}.summary ul{margin:0;padding-inline-start:1.2rem;display:grid;gap:6px}
.answer{border-inline-start:4px solid var(--avg);padding:6px 14px;background:var(--surface);margin:12px 0;border-radius:0 6px 6px 0}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.88rem;color:var(--muted);margin:.4rem 0 .8rem}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-inline-end:6px;vertical-align:middle;background:var(--ref)}
.sw.avg,.bar.avg{background:var(--avg)}.sw.lsml,.bar.lsml{background:var(--lsml)}.sw.group,.bar.group{background:var(--group)}.sw.band,.bar.band{background:var(--band)}.sw.ref,.bar.ref{background:var(--ref)}
.mnum{display:inline-block;min-width:1.6em;font-family:var(--mono);font-size:.75rem;color:var(--muted)}
.tablewrap{overflow-x:auto;margin:.4rem 0 1rem}
table{border-collapse:collapse;font-size:.88rem;background:var(--surface)}
caption{caption-side:top;text-align:start;color:var(--muted);font-size:.85rem;padding:0 0 6px}
th,td{border-bottom:1px solid var(--rule);padding:5px 8px;text-align:start;vertical-align:middle}
thead th{font-weight:600;font-size:.82rem;color:var(--muted);background:var(--soft)}
tbody th{font-weight:500;white-space:nowrap}
tr.base th,tr.base td{background:color-mix(in srgb,var(--avg) 9%,var(--surface))}
.allm td.dcell{min-width:118px}.allm td.na{color:var(--muted);font-size:.8rem}
.track{position:relative;height:8px;background:var(--soft);border-radius:4px;overflow:hidden}
.track .zero,.ftrack .zero{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--muted)}
.bar{position:absolute;top:0;bottom:0}.bar.r{left:50%}.bar.l{right:50%}
.dv{display:flex;justify-content:space-between;gap:6px;font-size:.8rem;margin-top:2px}.dv b{font-family:var(--mono);font-weight:600}.dd{color:var(--muted)}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:.6rem 0 .4rem}
.tabs button{font:inherit;font-size:.88rem;border:1px solid var(--rule);background:var(--surface);color:var(--ink);padding:4px 12px;border-radius:999px;cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.tabs button:focus-visible{outline:2px solid var(--avg);outline-offset:2px}
.qgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,540px),1fr));gap:14px}
.q{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
.qsub{color:var(--muted);font-size:.84rem;margin:.1rem 0 .5rem}.minus{font-weight:700}
.tag{display:inline-block;font-size:.72rem;border:1px solid var(--rule);border-radius:999px;padding:0 8px;margin-inline-start:6px;color:var(--muted)}
.tag.prim{border-color:var(--avg);color:var(--avg)}
.fscale{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-inline-start:92px;margin-inline-end:0;padding-inline-end:0}
.frow{display:grid;grid-template-columns:86px minmax(120px,1fr) auto auto;align-items:center;gap:4px 8px;padding:3px 0;border-top:1px dashed var(--rule);font-size:.84rem}
.frow .fx{grid-column:2 / -1;font-size:.74rem;color:var(--muted);display:inline;margin-inline-end:10px}
.frow .fnote{grid-column:2 / -1;color:var(--muted);font-size:.8rem}
.ftrack{position:relative;height:18px}
.ci{position:absolute;top:8px;height:3px;border-radius:2px}.pt{position:absolute;top:4px;width:10px;height:10px;border-radius:50%;margin-left:-5px}
.ci.gain,.pt.gain{background:var(--gain)}.ci.loss,.pt.loss{background:var(--loss)}.ci.ns,.pt.ns{background:var(--ns)}
.pp{position:absolute;top:3px;width:10px;height:10px;margin-left:-6px;border:2px solid var(--avg);transform:rotate(45deg);background:var(--surface)}
.sw0{position:absolute;top:1px;width:2px;height:16px;margin-left:-1px;background:var(--ink);opacity:.55}
.fv{font-size:.8rem;white-space:nowrap}.fci{color:var(--muted);font-size:.72rem}
.chip{font-size:.72rem;padding:1px 8px;border-radius:999px;white-space:nowrap;border:1px solid currentColor}
.chip.gain{color:var(--gain)}.chip.loss{color:var(--loss)}.chip.ns{color:var(--muted)}
.keyrow{display:flex;flex-wrap:wrap;gap:16px;font-size:.82rem;color:var(--muted);margin:.3rem 0 .8rem}
.keyrow i{display:inline-block;vertical-align:middle;margin-inline-end:6px;position:static}
.k-pt{width:10px;height:10px;border-radius:50%;background:var(--loss)}.k-pp{width:9px;height:9px;border:2px solid var(--avg);transform:rotate(45deg)}.k-sw{width:2px;height:14px;background:var(--ink);opacity:.55}
td.hh{text-align:center;font-family:var(--mono);font-size:.74rem;background:color-mix(in srgb,var(--gain) calc(var(--a) * 100%),var(--surface))}
td.hh.n{background:color-mix(in srgb,var(--loss) calc(var(--a) * 100%),var(--surface))}
td.hh.sig{font-weight:700;outline:1.5px solid var(--ink);outline-offset:-2px}
td.diag{text-align:center;color:var(--muted)}
.h2h th .mean{color:var(--muted);font-size:.75rem;margin-inline-start:6px}
.cells td.hh{min-width:64px}
.assum td{font-family:var(--mono);text-align:center}
.hmpair{display:flex;flex-wrap:wrap;gap:16px}.hm{margin:0}.hm figcaption{font-weight:600;font-size:.9rem;margin-bottom:4px}
table.corr{font-size:.62rem;background:var(--surface)}
table.corr th{font-family:var(--mono);font-weight:400;font-size:.66rem;text-align:right;padding:1px 6px;border:0;white-space:nowrap}
td.cm{width:22px;height:20px;padding:0;text-align:center;border:0;color:var(--ink);background:color-mix(in srgb,var(--pos) calc(var(--a) * 100%),var(--surface))}
td.cm.n{background:color-mix(in srgb,var(--neg) calc(var(--a) * 100%),var(--surface))}
td.cm.same{box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--ink) 25%,transparent)}
td.cm.gl{border-left:3px solid var(--ink)}td.cm.gt{border-top:3px solid var(--ink)}td.hi,td.cm.hi{color:var(--surface)}td.cm.diag{background:var(--soft)}
.gdot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-inline-end:6px;vertical-align:middle}
.g0{background:#1F5E8E}.g1{background:#C2622A}.g2{background:#5E7D26}.g3{background:#7650A6}.g4{background:#B2372F}.g5{background:#8A94A0}
.groups{display:grid;gap:4px;padding-inline-start:1.2rem;font-size:.88rem}
.pairs td{font-size:.82rem}
.files code{white-space:normal}
@media (max-width:640px){h1{font-size:1.8rem}.frow{grid-template-columns:70px minmax(90px,1fr);}.frow .fv,.frow .chip{grid-column:2}.fscale{margin-inline-start:74px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""
JS = """
document.querySelectorAll('.tabs button').forEach(function(b){b.addEventListener('click',function(){
  var g=b.dataset.g,i=b.dataset.i;
  document.querySelectorAll('.tabs button[data-g="'+g+'"]').forEach(function(x){x.setAttribute('aria-selected',x.dataset.i===i?'true':'false')});
  document.querySelectorAll('.panel[data-g="'+g+'"]').forEach(function(p){p.hidden=p.dataset.i!==i});
});});
"""
fam_legend = ''.join(f'<span><i class="sw {k}"></i>{v}</span>' for k, v in FAM_HE.items())
central_cells = ''.join(f'<li>{BANK_HE[bk]}: {num(cen[bk])}</li>' for bk in BANKS)

page = f"""<title>שלב האיחוד ב-L-SML</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Assistant:wght@400;500;600;700&family=Frank+Ruhl+Libre:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap" dir="rtl" lang="he">
<header>
<h1>שלב האיחוד ב-L-SML</h1>
<p class="lede">ניסוי {code('lsml_merge_step_v1')}, 28 בספטמבר 2026. פרוטוקול קפוא {code('6665e28ca')}, ריצה {code('577ab40e4')}, תיעוד {code('50f9f7642')}.
PRMBench: 6,969 תשובות (6,030 עם צעד שגוי וצעד תקין). ProcessBench: 4,442 תשובות שגויות ב-8 cells. חמישה folds לפי קבוצות מקור.
צוות אדום של שלושה סוכנים: כל המספרים שוחזרו, הפרשנות תוקנה בכמה נקודות (מפורט בסוף).</p>
</header>

<section class="summary" aria-labelledby="sum">
<h2 id="sum">בקצרה</h2>
<ul>
<li><b>שלב האיחוד תיקן את החלוקה:</b> בכל fold שבו L-SML פיצל את משפחת הרמה לשתי קבוצות, השלב איחד אותן, ולא איחד שום דבר אחר. על 13 הערוצים זו בדיוק החלוקה הידנית משלב B2.</li>
<li><b>L-SML עם החלוקה המתוקנת לא עוקף ממוצע פשוט של אותם ערוצים</b> באף בנק. AUC בתוך תשובה, L-SML עם איחוד פחות ממוצע פשוט: <span class="inline-list">{', '.join(f'{BANK_HE[bk]} {num(cen[bk])}' for bk in BANKS)}</span>.</li>
<li><b>רוב ההפסד הוא מיקום:</b> כשמסירים את השפעת המיקום בתשובה, L-SML עם האיחוד עוקף את הממוצע ב-20, 32 ו-51 ערוצים, ומפסיד ב-13 וב-16.</li>
<li><b>כלל הרצועה (0.45 עד 0.55) לא עדיף על סינון DS</b> באף בנק. הורדת הרצועה עזרה רק ב-32 ערוצים, והיפוך הערוצים ההפוכים פגע.</li>
<li><b>הנחת L-SML לא מתקיימת:</b> בין ערוצים מקבוצות שונות נשאר מתאם ממוצע של 0.14 עד 0.37 (לפי הבנק), גם כשבודקים רק צעדים שגויים או רק צעדים תקינים, והזוג הגרוע מגיע ל-0.7 עד 0.9. ה-clustering טוב מכל אחת מ-2,000 חלוקות אקראיות באותם גדלים, אבל רחוק מאפס. בצעדים שגויים התלות גבוהה יותר מאשר בתקינים.</li>
<li><b>פיצ'רי הספרות</b> לא צורפו לשום קבוצה: ה-clustering שם אותם תמיד בקבוצה נפרדת של שלושה. הם מעלים את הממוצע הפשוט ל-{num(mv('B16', 'DSF_equal', 'within_auc'))}, אבל רוב הרווח הוא מיקום.</li>
</ul>
</section>

<h2 id="all">כל השיטות על כל הבנקים</h2>
<p>כל שורה היא שיטה, כל עמודה היא בנק ערוצים. הבסיס להשוואה בכל בנק הוא הממוצע הפשוט אחרי סינון DS (השורה המודגשת). הצבע מסמן את משפחת השיטה.</p>
<div class="legend">{fam_legend}</div>
{metric_tabs}

<h2 id="questions">שאלות כן או לא</h2>
<p>כל כרטיס עונה על שאלה אחת: השיטה הראשונה פחות השנייה, AUC בתוך תשובה ב-PRMBench, בכל אחד מחמשת הבנקים. ארבע השאלות הראשונות הוגדרו מראש בפרוטוקול, ולהן רווח סמך מתוקן להשוואות מרובות.</p>
<div class="keyrow"><span><i class="k-pt"></i>ההפרש ורווח הסמך (ירוק עדיף, אדום נחות, אפור אין הבדל מובהק)</span><span><i class="k-pp"></i>אותו הפרש אחרי הסרת מיקום</span><span><i class="k-sw"></i>מה שמיקום לבד נותן (החלפת תוויות בין תשובות באותו אורך)</span></div>
<div class="qgrid">{questions}</div>

<h2 id="h2h">השוואה של כל שיטה מול כל שיטה</h2>
<p>לכל בנק, טבלה של כל הזוגות. המספרים בכותרות העמודות תואמים את מספרי השורות. מעבר עם העכבר על תא מציג את רווח הסמך.</p>
{h2h_tabs}

<h2 id="cells">כל השיטות על כל cell</h2>
<p>ב-PRMBench יש cell אחד ({code('prmbench_qwen3_8b')}), וב-ProcessBench שמונה: ארבעה מערכי נתונים, כל אחד ב-q4 וב-q8. q4 ו-q8 חולקים את אותן תשובות, כך שיש בפועל ארבעה מערכי נתונים בלתי תלויים. ב-ProcessBench המדד הוא האם הצעד עם הציון הגבוה ביותר הוא השגיאה הראשונה.</p>
{cell_tabs}
<h3 style="margin-top:1.2rem">PRMBench לפי סוג השגיאה</h3>
{class_tabs}

<h2 id="assumption">האם הקבוצות מקיימות את הנחת L-SML?</h2>
<p>L-SML מניח שערוצים מקבוצות שונות טועים באופן בלתי תלוי כשיודעים את התווית האמיתית. בדקנו זאת ישירות: חישבנו את המתאם בין כל שני ערוצים בנפרד על הצעדים התקינים ובנפרד על הצעדים השגויים. אם ההנחה מתקיימת, המתאם בין ערוצים מקבוצות שונות צריך להיות קרוב לאפס בשתי הקבוצות. בתוך קבוצה מותר מתאם.
החישוב על החלוקות של fold 0, על {num(D['fold0_steps']['prm_fit_steps'], 0)} צעדי PRMBench מה-folds שעליהם הותאם המודל ({num(D['fold0_steps']['error_steps'], 0)} מהם שגויים). התוויות שימשו כאן לאבחון בלבד.</p>
<div class="tablewrap"><table class="assum"><caption>סיכום לכל בנק, לחלוקה הבינארית אחרי איחוד: ממוצע |מתאם| בין קבוצות (ממוצע של צעדים תקינים ושגויים).</caption>
<thead><tr><th scope="col">בנק</th><th scope="col">החלוקה שנמצאה</th><th scope="col">חלוקה אקראית באותם גדלים</th><th scope="col">אחרי הסרת מיקום</th><th scope="col">על סימוני 20% העליונים</th><th scope="col">הזוג הגרוע ביותר</th></tr></thead>
<tbody>{assum_rows}</tbody></table></div>
<div class="answer"><b>התשובה: לא.</b> ה-clustering מוצא מבנה אמיתי: התלות בין הקבוצות שלו נמוכה מזו של כל אחת מ-2,000 חלוקות אקראיות באותם גדלים, ובתוך הקבוצות המתאם גבוה פי שניים בערך. אבל ההנחה של L-SML רחוקה מלהתקיים:
<ul><li>בין קבוצות נשאר מתאם ממוצע של 0.14 (32 ערוצים) עד 0.37 (13 ערוצים, צעדים שגויים), ו-60% עד 90% מהזוגות בין קבוצות מעל 0.1.</li>
<li>התלות גבוהה יותר בצעדים השגויים מאשר בתקינים, בכל הבנקים. כלומר ערוצים מקבוצות שונות נוטים לטעות יחד דווקא כשיש שגיאה.</li>
<li>המפרים הגדולים: ב-13 וב-16 ערוצים {code('chosen_surprisal')} (בקבוצה עם ערוצי המימוש) מול ערוצי הרמה {code('q15_VE1')} ו-{code('q15_H1')}, כ-0.58; ב-20 ערוצים {code('ct7_H0lim_prefix_innovation')} מול {code('ct7_ve0')} וערוצי הרמה, מעל 0.8.</li>
<li>הסרת המיקום לא מורידה את התלות בין הקבוצות, ואפילו מעלה אותה מעט. לכן מיקום אינו מה שמפר את ההנחה, וההשערה שהעליתי קודם בכיוון הזה לא נתמכת.</li>
<li>על סימוני 20% העליונים (הצורה הבינארית שעליה נמצאה החלוקה) התלות נמוכה יותר, 0.08 עד 0.21, אבל גם שם לא אפס.</li></ul>
זה מסביר למה משקלי L-SML לא עוקפים ממוצע: השלב שמשקלל בין הקבוצות מניח שהן בלתי תלויות בהינתן התווית, וזה לא המצב.</div>
<h3>פירוט לפי חלוקה</h3>
{assum_tabs}
<h3>הקבוצות ומפת המתאמים</h3>
<p>הקבוצות של החלוקה הבינארית אחרי איחוד (fold 0), ומתחתן מטריצת המתאמים בנפרד לצעדים תקינים ולצעדים שגויים. הערוצים מסודרים לפי קבוצה, והקווים העבים מפרידים בין קבוצות. המספר בתא הוא |מתאם| כפול 100, כחול חיובי וכתום שלילי. מחוץ לריבועים שעל האלכסון, ההנחה דורשת תאים ריקים.</p>
{heat_tabs}

<h2 id="digits">פיצ'רי הספרות</h2>
<p><b>לאיזו קבוצה הוספתי אותם?</b> לאף קבוצה. שלושת הפיצ'רים ({code('digit_alternative')}, {code('digit_spread')}, {code('digit_alternative_innovation')}) נוספו לבנק כשלושה ערוצים נפרדים, וה-clustering החליט לבד היכן לשים אותם. בכל חמשת ה-folds, בחלוקה הבינארית וגם בחלוקה של L-SML, לפני האיחוד ואחריו, הם יצרו קבוצה נפרדת משלהם ולא התאחדו עם שום קבוצה אחרת.</p>
<p class="lede">הקבוצות ב-13 + ספרות, חלוקה בינארית אחרי איחוד, fold 0 (ובשאר ה-folds קבוצת הספרות זהה):</p><ol class="groups">{digits_parts}</ol>
<p>התרומה שלהם (לא הוגדרה מראש בפרוטוקול): הממוצע הפשוט אחרי סינון DS עולה מ-{num(mv('B13', 'DSF_equal', 'within_auc'))} ל-{num(mv('B16', 'DSF_equal', 'within_auc'))}, הפרש {num(pdig['B16__DSF_equal - B13__DSF_equal']['delta'])}
(רווח סמך {num(pdig['B16__DSF_equal - B13__DSF_equal']['ci95'][0])} עד {num(pdig['B16__DSF_equal - B13__DSF_equal']['ci95'][1])}). אבל החלפת תוויות בין תשובות באותו אורך לבדה נותנת {num(pdig['B16__DSF_equal - B13__DSF_equal']['nulls_mean_sd']['swap'][0])}, כלומר כ-80% מהרווח הוא מיקום.
אחרי הסרת מיקום נשאר {num(pdig['B16__DSF_equal_pos - B13__DSF_equal_pos']['delta'])} (רווח סמך {num(pdig['B16__DSF_equal_pos - B13__DSF_equal_pos']['ci95'][0])} עד {num(pdig['B16__DSF_equal_pos - B13__DSF_equal_pos']['ci95'][1])}).
ב-ProcessBench הממוצע עולה מ-{num(mv('B13', 'DSF_equal', 'sla', 'macro8'))} ל-{num(mv('B16', 'DSF_equal', 'sla', 'macro8'))}, והשיפור מובהק רק ב-gsm8k q4 וב-MATH. הם גם מזיזים את אומדני DS (השכיחות הנאמדת עולה מכ-0.28 לכ-0.41).</p>
<p class="lede">פתוח להחלטתך: האם הפיצ'רים האלה מחוץ להחרגת הספרות מ-17 בספטמבר. הם לא משתמשים בחוסר הסכמה עם הטוקן שנבחר, אבל ההחרגה כוללת גם פיצ'רים של נוכחות ספרות.</p>

<h2 id="position">מיקום בתשובה</h2>
<p>לכל בנק: ההפרש בשאלה המרכזית (L-SML עם איחוד פחות ממוצע פשוט), מה שמיקום לבד נותן, ואותו הפרש כשהשיטות רצות על בנק שממנו הוסר פרופיל המיקום (ממוצע לפי אורך התשובה ומקום הצעד).</p>
<div class="tablewrap"><table class="assum"><thead><tr><th scope="col">בנק</th><th scope="col">ההפרש</th><th scope="col">מיקום לבד</th><th scope="col">אחרי הסרת מיקום</th></tr></thead><tbody>{pos_rows}</tbody></table></div>
<p>ב-13, 20 ו-51 ערוצים ההפסד כולו מוסבר במיקום, וב-32 כשני שלישים ממנו. ההשערה שהעליתי, שמיקום הוא גורם משותף שמפר את הנחת L-SML, לא נתמכת בבדיקת ההנחה למעלה: הסרת המיקום לא מורידה את התלות בין הקבוצות. למה L-SML מתנהג אחרת על הבנקים בלי מיקום עדיין לא ידוע. בנוסף, הציונים על הבנקים בלי מיקום נמוכים מהממוצע הפשוט על הבנקים המקוריים, ולכן הסרת המיקום מהציון עצמו אינה הפתרון.</p>

<h2 id="redteam">מה הצוות האדום תיקן</h2>
<ul>
<li>ב-32 ערוצים שלב האיחוד לא פעל באף fold. ההפסד שם שייך לחלוקה הבינארית ולמשקלי L-SML, לא לשלב.</li>
<li>ההפסדים בשאלה המרכזית הם ברובם מיקום (ראו למעלה).</li>
<li>ב-13 ערוצים, אחרי האיחוד יש שלוש קבוצות, ובמקרה כזה L-SML עצמו קובע משקלים שווים בין הקבוצות ובתוך הקבוצות של שלושה. שם ההשוואה היא בפועל משקל שווה לקבוצה מול משקל שווה לערוץ.</li>
<li>יחס הבליעה תלוי בגודל הקבוצות: לשני חצאים של אותו גורם הוא (1−r)/(1+(m−1)r), כש-m הוא גודל החצי הקטן. רק ערך הייחוס לקבוצות בלתי תלויות אינו תלוי בגודל.</li>
<li>אחרי הצוות האדום, בדיקת ההנחה בדו״ח הזה הפריכה את המנגנון שהצעתי למיקום (ראו למעלה).</li>
<li>"האיחוד מאחד רק את משפחת הרמה" נכון על הבנקים המקוריים. על הבנקים בלי מיקום ועל כלל הרצועה הוא איחד שש פעמים קבוצה אחרת, קרוב לסף 0.5.</li>
</ul>
<p class="files lede">קבצים: {code('results/lsml_merge_step_v1/SUMMARY.md')}, {code('run_20260928/RED_TEAM.md')}, {code('run_20260928/METRICS.csv')}, {code('CONTRASTS.csv')}, {code('PARTITIONS.json')}; נתוני הדו״ח {code('report_data.json')} (מ-{code('report_data.py')}).</p>
</div>
<script>{JS}</script>
"""
(HERE / 'REPORT_HE.html').write_text(page, encoding='utf8')
print('written', len(page))
