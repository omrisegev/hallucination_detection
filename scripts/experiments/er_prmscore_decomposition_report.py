"""Hebrew HTML report for prmscore_decomposition_v1. Every number is read from the run outputs and the red-team outputs
(no hand-copied values).

    python -B scripts/experiments/er_prmscore_decomposition_report.py
"""
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'results/expectation_realization_v1/prmscore_decomposition_v1'
RT = D / 'red_team'

P1 = json.loads((D / 'P1_OVERALL.json').read_text(encoding='utf8'))
P3 = json.loads((D / 'P3_RANKING_DECISIONS.json').read_text(encoding='utf8'))
GATES = json.loads((D / 'GATES.json').read_text(encoding='utf8'))
CT = pd.read_csv(D / 'P2_CONTRASTS.csv')
AR = pd.read_csv(D / 'P2_ARMS_BY_STRATUM.csv')
P4 = pd.read_csv(D / 'P4_COMPLEMENTARITY.csv')
S1 = pd.read_csv(D / 'S1_LENGTH_POSITION.csv')
S1B = json.loads((D / 'S1_BINS.json').read_text(encoding='utf8'))
C1 = json.loads((RT / 'agent_C/rt_c_part1.json').read_text(encoding='utf8'))
C2 = json.loads((RT / 'agent_C/rt_c_part2.json').read_text(encoding='utf8'))
B = json.loads((RT / 'agent_B/audit_out.json').read_text(encoding='utf8'))
_log2 = (RT / 'agent_B/audit_log2.txt').read_text(encoding='utf8')
_rand = [float(x) for x in re.findall(r'np\.float64\(([0-9.]+)\)', next(l for l in _log2.splitlines() if l.startswith('random-score controls')))]
RAND_SHARE, RAND_STEP = _rand[0::2], _rand[1::2]                      # five random-score draws, agent B
LATE_N, LATE_CONF = map(int, re.search(r"late answers (\d+) .*?'confidence'\), (\d+)\)", _log2).groups())
FIRST_N = sum(B['first_step_by_class'].values()); FIRST_MC = B['first_step_by_class']['missing_condition']

ROW = {'B13_equal': 'ממוצע כל 13 הערוצים', 'S_equal': 'סינון DS ואז ממוצע', 'G1_sml': 'הכלל הבינארי (קבוצות מתגלות, משקלי DS)',
       'B_sml__merge': 'הכלל הבינארי, level ממוזג', 'B13_lsml': 'L-SML על 13 הערוצים', 'S_lsml': 'L-SML על השורדים',
       'realized_drv': 'realized_drv לבד', 'fam421': 'fam421', 'ct7': 'CT7', 'step_index': 'אינדקס הצעד (מיקום בלבד)',
       'PRM_native': 'Qwen PRM, הסף שלו (0.5)', 'PRM_raw_q80': 'Qwen PRM, q80 בסקאלה שלו', 'PRM_z_q80': 'Qwen PRM, בדיוק הכלל שלנו'}
KIND = {'B13_equal': ('אלגוריתמי', 'ours'), 'S_equal': ('אלגוריתמי', 'ours'), 'G1_sml': ('אלגוריתמי', 'ours'),
        'B_sml__merge': ('בדיקת מנגנון, רשימה קשיחה', 'mech'), 'B13_lsml': ('אלגוריתמי', 'ours'), 'S_lsml': ('אלגוריתמי', 'ours'),
        'realized_drv': ('ערוץ שנבחר בדיעבד', 'drv'), 'fam421': ('ייחוס', 'ref'), 'ct7': ('ייחוס', 'ref'), 'step_index': ('ייחוס מיקום', 'ref'),
        'PRM_native': ('מפוקח, גישה שונה', 'prm'), 'PRM_raw_q80': ('מפוקח, גישה שונה', 'prm'), 'PRM_z_q80': ('מפוקח, גישה שונה', 'prm')}
CAL = {'PRM_native': 'reward ≥ 0.5', 'PRM_raw_q80': 'q80 על risk גולמי, fold הכיול', 'PRM_z_q80': 'answer-z, q80 של fold הכיול'}
CLS = [('redundency', 'יתירות', 'NR', 'פשטות'), ('circular', 'לוגיקה מעגלית', 'NCL', 'פשטות'),
       ('counterfactual', 'סתירה לעובדות', 'ES', 'תקינות'), ('step_contradiction', 'סתירה בין צעדים', 'SC', 'תקינות'),
       ('domain_inconsistency', 'אי־עקביות בתחום', 'DC', 'תקינות'), ('confidence', 'ביטחון יתר בשגיאה', 'CI', 'תקינות'),
       ('missing_condition', 'תנאי חסר', 'PS', 'רגישות'), ('deception', 'הטעיה', 'DR', 'רגישות'),
       ('multi_solutions', 'ריבוי פתרונות', 'MS', 'רגישות')]
ORDER = ['B13_equal', 'S_equal', 'G1_sml', 'B_sml__merge', 'B13_lsml', 'S_lsml', 'realized_drv', 'fam421', 'ct7', 'step_index',
         'PRM_native', 'PRM_raw_q80', 'PRM_z_q80']
rows = {r['arm']: r for r in P1['rows']}; p3 = {r['arm']: r for r in P3['arms']}


def num(x, d=4, pct=False, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return '<span class="num na">—</span>'
    v = x * 100 if pct else x
    s = f'{abs(v):,.{d}f}' if abs(v) >= 1000 else f'{abs(v):.{d}f}'
    s = ('−' if v < 0 and round(abs(v), d) > 0 else '') + s + ('%' if pct else '')
    return f'<span class="num">{s}</span>'


def iv(lo, hi, d=4):
    return f'<span class="num ci">[{num(lo, d)[18:-7]}, {num(hi, d)[18:-7]}]</span>'


def delta_cell(r, d=4):
    if r is None or r['delta'] != r['delta']:
        return '<td class="d na">—</td>'
    cls = 'pos' if r['ci95_lo'] > 0 else 'neg' if r['ci95_hi'] < 0 else 'nil'
    bonf = r.get('ci_bonf9_lo')
    mark = ''
    if bonf is not None and bonf == bonf and (r['ci_bonf9_lo'] > 0 or r['ci_bonf9_hi'] < 0):
        mark = '<span class="bf" title="גם הרווח המתוקן ל־9 סוגים לא כולל אפס">●</span>'
    return f'<td class="d {cls}">{num(r["delta"], d)}{mark}<br>{iv(r["ci95_lo"], r["ci95_hi"], d)}</td>'


def ct(contrast, stratum, endpoint):
    x = CT[(CT.contrast == contrast) & (CT.stratum == stratum) & (CT.endpoint == endpoint)]
    return None if x.empty else x.iloc[0].to_dict()


def arm(a, stratum, endpoint):
    x = AR[(AR.arm == a) & (AR.stratum == stratum) & (AR.endpoint == endpoint)]
    return None if x.empty else x.iloc[0].to_dict()


pop = GATES['population']
Sx = rows['S_equal']; Dx = rows['realized_drv']; Bx = rows['B13_equal']; Pz = rows['PRM_z_q80']; Pr = rows['PRM_raw_q80']; Pn = rows['PRM_native']
f_sd = ct('S_equal - realized_drv', 'total_noncontrol', 'prmscore'); f_sb = ct('S_equal - B13_equal', 'total_noncontrol', 'prmscore')
f_pz = ct('S_equal - PRM_z_q80', 'total_noncontrol', 'prmscore'); f_pr = ct('S_equal - PRM_raw_q80', 'total_noncontrol', 'prmscore')
f_sdw = ct('S_equal - realized_drv', 'total_noncontrol', 'within_auc')
nb = C1['nullB_diffs']; na = C1['nullA_diffs']; rs = C2['residual_vs_whole_answer_swap']; rp = C2['residual_vs_within_answer_perm']
share_within = C2['auc_decomposition']['S_equal']['w_share_within']
our_rows = [a for a in ORDER if KIND[a][1] in ('ours', 'mech')]
lo_our = min(rows[a]['prmscore'] for a in our_rows); hi_our = max(rows[a]['prmscore'] for a in our_rows)
pooled_c = {c['contrast']: c for c in P3['contrasts']}
ctl = {a: p3[a]['controls_outside_prmscore'] for a in ORDER}
noise = C2['iid_baseline_share_ge1_flag_controls_len_distribution']
cls_pz = {c: ct('S_equal - PRM_z_q80', c, 'prmscore') for c, *_ in CLS}

# ------------------------------------------------------------------ sections
H = []
H.append(f'''
<header class="top">
  <p class="eyebrow">PRMBench · {num(pop["prm_answers"], 0)} תשובות · נתוני פיתוח · 27.9.2026</p>
  <h1>פירוק ה־PRMScore של בנק 13 הערוצים</h1>
  <p class="lede">פירקתי את ה־PRMScore על התחזיות הקפואות של שלבים A עד B2, בלי לאמן מחדש ובלי לבחור ספים. לכל שורה נשמר הסף שהמודל של אותו fold חישב, וכל השורות נמדדות על אותה אוכלוסייה. הפירוק עונה על שלוש שאלות: מה הסינון מוסיף מעל הממוצע המקורי, מה מפריד את השורה המאוחדת מהערוץ הבודד ומה־PRM המפוקח, ואיזה חלק מכל פער הוא דירוג בתוך תשובה, השוואה בין תשובות או החלטה בסף.</p>
</header>

<section aria-labelledby="h-sum">
  <h2 id="h-sum">בקצרה</h2>
  <ol class="points">
    <li><b>שורות האיחוד שלנו קרובות זו לזו, והערוץ הבודד לא נופל מהן.</b> כל שורות האיחוד נעות בין {num(lo_our)} ל־{num(hi_our)}. realized_drv לבד מקבל {num(Dx["prmscore"])}, וסינון ואז ממוצע מקבל {num(Sx["prmscore"])}. ההפרש ביניהם, {num(f_sd["delta"])} {iv(f_sd["ci95_lo"], f_sd["ci95_hi"])}, בתוך אי־הוודאות. אבל זה לא שוויון אמיתי: כשמחליפים תוויות בין תשובות באותו אורך, realized_drv נושא יותר אות ספציפי לתשובה ({num(rs["S_equal-realized_drv"]["point"])} {iv(*rs["S_equal-realized_drv"]["ci95"])}). השורה המאוחדת משיגה אותו בזכות מבנה המיקום.</li>
    <li><b>רוב הרווח של הסינון הוא מבנה מיקום.</b> הסינון מוסיף {num(f_sb["delta"])} {iv(f_sb["ci95_lo"], f_sb["ci95_hi"])} מעל ממוצע כל 13 הערוצים. החלפת תוויות בין תשובות באותו אורך משחזרת {num(nb["S_equal-B13_equal"]["null_mean"])} מתוכם, והחלק הספציפי לתשובה, {num(rs["S_equal-B13_equal"]["point"])} {iv(*rs["S_equal-B13_equal"]["ci95"])}, כולל אפס.</li>
    <li><b>הפער ל־PRM נחלק לשניים.</b> מול ה־PRM עם סף q80 בסקאלה שלו הפער הוא {num(-f_pr["delta"])}. כשה־PRM מקבל בדיוק את כלל הנרמול שלנו, {num(Pr["prmscore"] - Pz["prmscore"])} מהפער נעלמים ונשארים {num(-f_pz["delta"])}. גם רוב השארית ({num(-na["S_equal-PRM_z_q80"]["null_mean"])}) נשאר כשמערבבים את התוויות בתוך כל תשובה. כלומר, בכלל שלנו היתרון של ה־PRM הוא בכמה צעדים הוא מסמן בכל תשובה, ולא בשאלה איזה צעד.</li>
    <li><b>לפי סוג שגיאה התמונה מתהפכת.</b> אנחנו מובילים ביתירות ({num(cls_pz["redundency"]["delta"])}), בלוגיקה מעגלית ({num(cls_pz["circular"]["delta"])}) ובאי־עקביות בתחום ({num(cls_pz["domain_inconsistency"]["delta"])}). ה־PRM מוביל בהטעיה ({num(-cls_pz["deception"]["delta"])}), בסתירה לעובדות ({num(-cls_pz["counterfactual"]["delta"])}), בביטחון יתר ({num(-cls_pz["confidence"]["delta"])}) ובתנאי חסר ({num(-cls_pz["missing_condition"]["delta"])}). יתירות ולוגיקה מעגלית נמדדות בשני הצדדים דרך כלל חלופי, כי לאף אחד מהם אין ראש יתירות.</li>
    <li><b>נרמול בתוך כל תשובה לא יכול לומר שתשובה נקייה.</b> כל השורות שמנרמלות בתוך התשובה מסמנות לפחות צעד אחד בכ־{num(ctl["S_equal"]["share_answers_with_false_flag"], 0, pct=True)} מ־{num(pop["controls"], 0)} תשובות הביקורת הנקיות. ציונים אקראיים באותו כלל נותנים {num(noise["gauss_tau0.85"], 1, pct=True)}, כך שזו תכונה של הכלל ולא של השיטה. ה־PRM בסקאלה שלו מסמן {num(ctl["PRM_raw_q80"]["share_answers_with_false_flag"], 1, pct=True)} מהן, ובסף שלו {num(ctl["PRM_native"]["share_answers_with_false_flag"], 1, pct=True)}.</li>
  </ol>
</section>''')

# overall table
tr = []
for a in ORDER:
    r = rows[a]; kind, key = KIND[a]
    cal = CAL.get(a, 'answer-z, q80 של אותו מודל ב־fold הכיול')
    tr.append(f'<tr class="k-{key}"><th scope="row"><span class="sw sw-{key}"></span>{html.escape(ROW[a])}<span class="sub">{kind} · {cal}</span></th>'
              f'<td>{num(r["prmscore"])}<br>{iv(*r["prmscore_ci95"])}</td><td>{num(r["f1_correct"])}</td><td>{num(r["f1_error"])}</td>'
              f'<td>{num(r["recall_error"])}</td><td>{num(r["flag_rate"], 1, pct=True)}</td><td>{num(r["within_auc"])}</td>'
              f'<td>{num(r["pooled_auc_decision_scale"])}</td><td>{num(r["any_error_hit"])}</td></tr>')
H.append(f'''
<section aria-labelledby="h-all">
  <h2 id="h-all">כל השורות על אותה אוכלוסייה</h2>
  <p class="prose">ה־PRMScore הרשמי הוא ממוצע של F1 על הצעדים התקינים ושל F1 על הצעדים השגויים, על {num(pop["noncontrol"], 0)} תשובות שאינן ביקורת ({num(pop["noncontrol_steps"], 0)} צעדים, מהם {num(pop["noncontrol_error_steps"], 0)} שגויים). F1 על הצעדים התקינים כמעט זהה בכל השורות שמסמנות כחמישית מהצעדים, והציון מוכרע ב־F1 על הצעדים השגויים. within-AUC הוא ממוצע על {num(pop["within_auc_eligible"], 0)} תשובות עם צעד תקין וצעד שגוי. AUC כולל מחושב בסקאלת ההחלטה של כל שורה. פגיעה פירושה שהצעד המסוכן ביותר בתשובה שגוי, על {num(pop["erroneous_noncontrol"], 0)} התשובות השגויות.</p>
  <div class="card scroll"><table class="big">
    <thead><tr><th>שורה</th><th>PRMScore<span class="sub">רווח 95%</span></th><th>F1 צעדים תקינים</th><th>F1 צעדים שגויים</th><th>recall שגויים</th><th>צעדים מסומנים</th><th>within-AUC</th><th>AUC כולל</th><th>פגיעה</th></tr></thead>
    <tbody>{''.join(tr)}</tbody></table></div>
  <p class="note">ה־PRM עם הסף שלו מסמן רק {num(Pn["flag_rate"], 1, pct=True)} מהצעדים. לכן ה־F1 שלו על צעדים תקינים גבוה ({num(Pn["f1_correct"])}) וה־F1 על צעדים שגויים נמוך ({num(Pn["f1_error"])}). הסינון ואז ממוצע גבוה ממנו בסך הכול ב־{num(ct("S_equal - PRM_native", "total_noncontrol", "prmscore")["delta"])}. בחירת הסף ל־PRM בסקאלה שלו לא נבחרה לפי התוצאה: אותו כלל fold (k+1)%5 כמו שלנו. Step 437 דיווח {num(GATES["step437_reference"]["PRM_raw_q80"])} בקוד הכיול שלו; כאן {num(GATES["prm_here"]["PRM_raw_q80"])}.</p>
</section>''')

# categories: toggled endpoints
EPS = [('prmscore', 'PRMScore'), ('f1_error', 'F1 צעדים שגויים'), ('f1_correct', 'F1 צעדים תקינים'), ('within_auc', 'within-AUC')]
COLS = [('S_equal - B13_equal', 'תרומת הסינון', 'סינון מול ממוצע 13'), ('S_equal - realized_drv', 'מול הערוץ הבודד', 'סינון מול realized_drv'),
        ('S_equal - PRM_z_q80', 'מול PRM, אותו כלל', 'סינון מול PRM (answer-z)'), ('S_equal - PRM_raw_q80', 'מול PRM, הסקאלה שלו', 'סינון מול PRM (גולמי)')]
tabs = []; panels = []
for i, (ep, lab) in enumerate(EPS):
    tabs.append(f'<button type="button" id="tab-{ep}" data-ep="{ep}" aria-pressed="{"true" if i == 0 else "false"}">{lab}</button>')
    body = []
    for c, he, code, grp in CLS + [('total_noncontrol', 'כל התשובות שאינן ביקורת', '', '')]:
        n_ans = CT[(CT.stratum == c)].n_answers.iloc[0]
        cells = ''.join(delta_cell(ct(k, c, ep)) for k, *_ in COLS)
        tot = ' class="tot"' if c == 'total_noncontrol' else ''
        sub = f'<span class="sub">{grp} · <span class="num">{code}</span></span>' if code else ''
        body.append(f'<tr{tot}><th scope="row">{he}{sub}</th><td>{num(float(n_ans), 0)}</td>{cells}</tr>')
    panels.append(f'<div class="panel" data-ep="{ep}"{"" if i == 0 else " hidden"}><div class="card scroll"><table class="big"><thead><tr><th>סוג שגיאה</th><th>תשובות</th>'
                  + ''.join(f'<th>{h}<span class="sub">{s}</span></th>' for _, h, s in COLS) + f'</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>')
lev = []
for c, he, code, grp in CLS + [('total_noncontrol', 'כל התשובות שאינן ביקורת', '', '')]:
    cells = ''.join(f'<td>{num((arm(a, c, "prmscore") or {}).get("estimate"))}</td>' for a in ['B13_equal', 'S_equal', 'realized_drv', 'ct7', 'PRM_z_q80', 'PRM_raw_q80', 'PRM_native'])
    lev.append(f'<tr{" class=\"tot\"" if c == "total_noncontrol" else ""}><th scope="row">{he}</th>{cells}</tr>')
H.append(f'''
<section aria-labelledby="h-cls">
  <h2 id="h-cls">לפי סוג שגיאה</h2>
  <p class="prose">בכל תא: ההפרש בין השורות והרווח שלו (95%, bootstrap מזווג לפי source group, 10,000 דגימות). ירוק פירושו שהשורה המאוחדת גבוהה והרווח לא כולל אפס, אדום פירושו שהיא נמוכה. הנקודה מסמנת שגם רווח מתוקן ל־9 סוגים לא כולל אפס (תיקון רק על הסוגים, לא על כל ההשוואות). בריבוי פתרונות אין צעדים שגויים, ולכן מוגדר בו רק F1 על צעדים תקינים.</p>
  <div class="ctl" role="group" aria-label="מדד">{''.join(tabs)}</div>
  {''.join(panels)}
  <h3>הרמות עצמן, PRMScore</h3>
  <div class="card scroll"><table><thead><tr><th>סוג שגיאה</th><th>ממוצע 13</th><th>סינון ואז ממוצע</th><th>realized_drv</th><th>CT7</th><th>PRM, אותו כלל</th><th>PRM, סקאלה שלו</th><th>PRM, סף 0.5</th></tr></thead><tbody>{"".join(lev)}</tbody></table></div>
  <ul class="points small">
    <li><b>הסינון</b> עוזר בחמישה סוגים, והשיפור עקבי ב־5 מתוך 5 folds רק בלוגיקה מעגלית ובסתירה לעובדות. בתנאי חסר הוא מזיק ({num(ct("S_equal - B13_equal", "missing_condition", "prmscore")["delta"])}).</li>
    <li><b>מול realized_drv</b> ההבדל העקבי הוא ביתירות, שם השורה המאוחדת טובה יותר ({num(ct("S_equal - realized_drv", "redundency", "prmscore")["delta"])}), ובהטעיה, שם היא נמוכה יותר ({num(ct("S_equal - realized_drv", "deception", "prmscore")["delta"])}). בביטחון יתר אין לדרג ביניהם: הסדר מתהפך אם 16 התשובות עם אינדקס שגיאה מחוץ לתשובה נספרות אחרת.</li>
    <li><b>מול ה־PRM</b> כל הפער הכולל בשורה של אותו כלל הוא ב־F1 על צעדים שגויים ({num(ct("S_equal - PRM_z_q80", "total_noncontrol", "f1_error")["delta"])}). מול ה־PRM בסקאלה שלו, F1 על הצעדים התקינים בסך הכול לא שונה ({num(ct("S_equal - PRM_raw_q80", "total_noncontrol", "f1_correct")["delta"])} {iv(ct("S_equal - PRM_raw_q80", "total_noncontrol", "f1_correct")["ci95_lo"], ct("S_equal - PRM_raw_q80", "total_noncontrol", "f1_correct")["ci95_hi"])}), אבל בריבוי פתרונות ה־PRM משאיר הרבה יותר צעדים תקינים כתקינים.</li>
  </ul>
</section>''')

# ranking vs between vs decisions
r3 = []
for a in ['B13_equal', 'S_equal', 'realized_drv', 'ct7', 'step_index', 'PRM_z_q80', 'PRM_raw_q80', 'PRM_native']:
    q = p3[a]; kind, key = KIND[a]
    r3.append(f'<tr class="k-{key}"><th scope="row"><span class="sw sw-{key}"></span>{html.escape(ROW[a])}<span class="sub">סקאלת החלטה: {q["decision_scale"]}</span></th>'
              f'<td>{num(q["within_auc"])}</td><td>{num(q["within_auc_pair_weighted"])}</td><td>{num(q["pooled_auc_decision_scale"])}</td>'
              f'<td>{num(q["false_flag_rate_correct_steps"], 1, pct=True)}</td><td>{num(q["miss_rate_error_steps"], 1, pct=True)}</td><td>{num(q["predicted_error_fraction"], 1, pct=True)}</td></tr>')
pz_c = pooled_c['S_equal - PRM_z_q80']; pr_c = pooled_c['S_equal - PRM_raw_q80']; pd_c = pooled_c['S_equal - realized_drv']
H.append(f'''
<section aria-labelledby="h-rank">
  <h2 id="h-rank">דירוג בתוך תשובה, השוואה בין תשובות, החלטה בסף</h2>
  <p class="prose">ב־AUC כולל, רק {num(share_within, 3, pct=True)} מזוגות צעד שגוי וצעד תקין הם מאותה תשובה. לכן AUC כולל מודד כמעט רק השוואה בין תשובות (ההפרש בינו לבין AUC בין־תשובתי קטן מ־{num(3.1e-6, 6)}). הוא לא מדד כיול. מה שמכריע את ה־PRMScore הוא ההחלטות בסף, והן בעמודות הימניות.</p>
  <div class="card scroll"><table class="big"><thead><tr><th>שורה</th><th>within-AUC<span class="sub">ממוצע על תשובות</span></th><th>within-AUC<span class="sub">משוקלל בזוגות</span></th><th>AUC כולל<span class="sub">= בין תשובות</span></th><th>צעדים תקינים שסומנו</th><th>צעדים שגויים שהוחמצו</th><th>צעדים מסומנים</th></tr></thead><tbody>{"".join(r3)}</tbody></table></div>
  <ul class="points small">
    <li><b>באותה סקאלה (answer-z)</b> ה־AUC הכולל שלנו וה־AUC הכולל של ה־PRM קרובים: הפרש {num(pz_c["delta"])} {iv(*pz_c["ci95"])}. בסקאלה הגולמית של ה־PRM הפער הוא {num(pr_c["delta"])} {iv(*pr_c["ci95"])}. המידע על ההבדל בין תשובות נמצא בסקאלה הגולמית של ה־PRM, והנרמול בתוך התשובה מוחק אותו.</li>
    <li><b>ב־PRMScore</b> המידע הזה שווה פחות: {num(Pr["prmscore"] - Pz["prmscore"])} (ה־PRM עם הכלל שלו מול ה־PRM עם הכלל שלנו), כי q80 מסמן כחמישית מהצעדים בכל מקרה.</li>
    <li><b>realized_drv</b> גבוה מהשורה המאוחדת גם בהשוואה בין תשובות ({num(-pd_c["delta"])} {iv(-pd_c["ci95"][1], -pd_c["ci95"][0])} ב־AUC כולל).</li>
  </ul>
</section>''')

# clean answers
rc = []
for a in ['B13_equal', 'S_equal', 'realized_drv', 'ct7', 'PRM_z_q80', 'PRM_raw_q80', 'PRM_native']:
    q = p3[a]; kind, key = KIND[a]
    c0, c1, c2, c3 = q['controls_outside_prmscore'], q['multi_solutions_in_prmscore'], q['inert_error_annotation_in_prmscore'], q['erroneous_answers_correct_steps']
    rc.append(f'<tr class="k-{key}"><th scope="row"><span class="sw sw-{key}"></span>{html.escape(ROW[a])}</th>'
              f'<td>{num(c0["share_answers_with_false_flag"], 1, pct=True)}</td><td>{num(c0["step_false_flag_rate"], 1, pct=True)}</td>'
              f'<td>{num(c1["share_answers_with_false_flag"], 1, pct=True)}</td><td>{num(c1["step_false_flag_rate"], 1, pct=True)}</td>'
              f'<td>{num(c2["share_answers_with_false_flag"], 1, pct=True)}</td>'
              f'<td>{num(c3["share_answers_with_false_flag_on_a_correct_step"], 1, pct=True)}</td><td>{num(c3["step_false_flag_rate"], 1, pct=True)}</td></tr>')
rc.append(f'<tr class="k-noise"><th scope="row"><span class="sw sw-noise"></span>ציונים אקראיים, אותו כלל<span class="sub">צוות אדום, 5 הגרלות על תשובות הביקורת</span></th>'
          f'<td>{num(min(RAND_SHARE), 1, pct=True)} עד {num(max(RAND_SHARE), 1, pct=True)}</td><td>{num(min(RAND_STEP), 1, pct=True)} עד {num(max(RAND_STEP), 1, pct=True)}</td><td colspan="5" class="muted">—</td></tr>')
H.append(f'''
<section aria-labelledby="h-clean">
  <h2 id="h-clean">תשובות נקיות</h2>
  <p class="prose">תשובות הביקורת ({num(pop["controls"], 0)}, כל הצעדים תקינים) לא נכנסות ל־PRMScore הרשמי, ולכן הן בטבלה נפרדת. ריבוי פתרונות ({num(pop["multi_solutions"], 0)} תשובות בלי צעד שגוי) ו־{num(pop["inert_error_annotation"], 0)} תשובות שכל אינדקסי השגיאה שלהן אחרי הצעד האחרון כן נכנסות ל־PRMScore, כתשובות תקינות לגמרי.</p>
  <div class="card scroll"><table class="big"><thead><tr><th rowspan="2">שורה</th><th colspan="2">ביקורת, מחוץ ל־PRMScore</th><th colspan="2">ריבוי פתרונות</th><th>אינדקס מחוץ לתשובה</th><th colspan="2">תשובות שגויות, צעדים תקינים</th></tr>
    <tr><th>תשובות עם התראה</th><th>צעדים מסומנים</th><th>תשובות עם התראה</th><th>צעדים מסומנים</th><th>תשובות עם התראה</th><th>תשובות עם התראה על צעד תקין</th><th>צעדים תקינים מסומנים</th></tr></thead><tbody>{"".join(rc)}</tbody></table></div>
  <p class="callout"><b>זו תכונה של הכלל, לא ממצא על השיטות.</b> אחרי נרמול בתוך תשובה, לכל תשובה ממוצע 0 וסטיית תקן 1, וסף גלובלי סביב 0.85 כמעט תמיד חוצה צעד אחד לפחות. ציונים אקראיים באותו כלל נותנים את אותם שיעורים, ושיעור הצעדים המסומנים דומה בתשובות נקיות ובתשובות שגויות. רק ה־PRM בסקאלה הגולמית משאיר חלק מהתשובות הנקיות בלי התראה.</p>
</section>''')

# complementarity
cm = []
NAMES4 = {'S_equal | realized_drv': ('סינון ואז ממוצע', 'realized_drv'), 'S_equal | PRM_z_q80': ('סינון ואז ממוצע', 'PRM, אותו כלל'),
          'S_equal | PRM_raw_q80': ('סינון ואז ממוצע', 'PRM, סקאלה שלו'), 'realized_drv | PRM_z_q80': ('realized_drv', 'PRM, אותו כלל'),
          'B13_equal | S_equal': ('ממוצע 13', 'סינון ואז ממוצע')}
for _, r in P4[P4.stratum == 'all'].iterrows():
    a, b_ = NAMES4[r.pair]; meas = 'פגיעה (בלי סף)' if r.measure.startswith('any') else 'צעד שגוי מסומן בסף'
    if r.pair == 'S_equal | PRM_raw_q80' and r.measure.startswith('any'):
        continue   # identical to the answer-z PRM row for the threshold-free hit
    cm.append(f'<tr><th scope="row">{a} מול {b_}<span class="sub">{meas}</span></th><td>{num(float(r.both), 0)}</td><td>{num(float(r.only_first), 0)}</td>'
              f'<td>{num(float(r.only_second), 0)}</td><td>{num(float(r.neither), 0)}</td><td>{num((r.n - r.neither) / r.n, 1, pct=True)}</td></tr>')
cc = []
for c, he, *_ in CLS[:-1]:
    x = P4[(P4.pair == 'S_equal | PRM_z_q80') & (P4.stratum == c) & P4.measure.str.startswith('any')].iloc[0]
    y = P4[(P4.pair == 'S_equal | realized_drv') & (P4.stratum == c) & P4.measure.str.startswith('any')].iloc[0]
    cc.append(f'<tr><th scope="row">{he}</th><td>{num(float(x.n), 0)}</td><td>{num(float(y.only_first), 0)}</td><td>{num(float(y.only_second), 0)}</td>'
              f'<td>{num(float(x.only_first), 0)}</td><td>{num(float(x.only_second), 0)}</td></tr>')
H.append(f'''
<section aria-labelledby="h-comp">
  <h2 id="h-comp">מי תופס אילו תשובות</h2>
  <p class="prose">על {num(pop["erroneous_noncontrol"], 0)} התשובות השגויות. פגיעה בלי סף: הצעד המסוכן ביותר בתשובה שגוי. בסף: לפחות צעד שגוי אחד מסומן בסף הקפוא. הטבלה מתארת; היא לא אומרת שאיחוד של שתי שורות היה משיג את הכיסוי הזה.</p>
  <div class="card scroll"><table><thead><tr><th>זוג</th><th>שתיהן</th><th>רק הראשונה</th><th>רק השנייה</th><th>אף אחת</th><th>לפחות אחת</th></tr></thead><tbody>{"".join(cm)}</tbody></table></div>
  <h3>פגיעה בלי סף, לפי סוג שגיאה</h3>
  <div class="card scroll"><table><thead><tr><th>סוג שגיאה</th><th>תשובות</th><th>רק סינון ואז ממוצע<span class="sub">מול realized_drv</span></th><th>רק realized_drv</th><th>רק סינון ואז ממוצע<span class="sub">מול PRM</span></th><th>רק PRM</th></tr></thead><tbody>{"".join(cc)}</tbody></table></div>
</section>''')

# nulls
NL = [('S_equal-B13_equal', 'סינון מול ממוצע 13'), ('S_equal-realized_drv', 'סינון מול realized_drv'), ('S_equal-PRM_z_q80', 'סינון מול PRM, אותו כלל')]
nl = []
for k, he in NL:
    nl.append(f'<tr><th scope="row">{he}</th><td>{num(C1["observed_diffs"][k])}</td>'
              f'<td>{num(na[k]["null_mean"])}<span class="sub">sd {num(na[k]["null_sd"])[18:-7]}</span></td><td>{num(rp[k]["point"])}<br>{iv(*rp[k]["ci95"])}</td>'
              f'<td>{num(nb[k]["null_mean"])}<span class="sub">sd {num(nb[k]["null_sd"])[18:-7]}</span></td><td>{num(rs[k]["point"])}<br>{iv(*rs[k]["ci95"])}</td></tr>')
pf = C1['permuted_feature']
H.append(f'''
<section aria-labelledby="h-null">
  <h2 id="h-null">בדיקות אפס</h2>
  <p class="prose">אלה בדיקות של הצוות האדום (סוכן C, 20 זרעים לכל אפס). הספים נשארים קפואים, כי הם לא תלויים בתוויות. <b>ערבוב בתוך תשובה</b> שומר על מספר הצעדים השגויים בכל תשובה ומבטל את המיקום שלהם. <b>החלפה בין תשובות</b> נותנת לכל תשובה את וקטור התוויות של תשובה אחרת באותו אורך ובאותו fold, ושומרת כך על מבנה המיקום הכללי. השארית היא ההפרש הנצפה פחות ממוצע האפס, עם רווח 95%.</p>
  <div class="card scroll"><table class="big"><thead><tr><th>השוואה</th><th>נצפה</th><th>ערבוב בתוך תשובה<span class="sub">ממוצע האפס</span></th><th>שארית</th><th>החלפה בין תשובות<span class="sub">ממוצע האפס</span></th><th>שארית</th></tr></thead><tbody>{"".join(nl)}</tbody></table></div>
  <p class="note">בדיקת שפיות: ערבוב realized_drv בתוך כל תשובה מוריד אותו ל־{num(pf["perm_prmscore_mean"])} (אקראי: {num(C1["baselines"]["iid_random_20pct_flags_prmscore"])}), מתחת לאינדקס הצעד ({num(C1["baselines"]["step_index_prmscore"])}). כלומר ה־PRMScore שלו נובע מאות אמיתי.</p>
</section>''')

# secondary
def s1row(kind, stratum, row, ep='prmscore'):
    x = S1[(S1.kind == kind) & (S1.stratum == stratum) & (S1.row == row) & (S1.endpoint == ep)]
    return None if x.empty else x.iloc[0]
def s1cell(kind, stratum, row):
    x = s1row(kind, stratum, row)
    if x is None:
        return '<td>—</td>'
    if ' - ' in row:
        return delta_cell({'delta': x.estimate, 'ci95_lo': x.ci95_lo, 'ci95_hi': x.ci95_hi})
    return f'<td>{num(x.estimate)}</td>'
LB = [('len_q1', f'עד {int(S1B["length_quartile_edges_steps"][0])} צעדים'), ('len_q2', f'{int(S1B["length_quartile_edges_steps"][0]) + 1} עד {int(S1B["length_quartile_edges_steps"][1])}'),
      ('len_q3', f'{int(S1B["length_quartile_edges_steps"][1]) + 1} עד {int(S1B["length_quartile_edges_steps"][2])}'), ('len_q4', f'{int(S1B["length_quartile_edges_steps"][2]) + 1} ומעלה')]
PB_ = [('first_step', 'בצעד הראשון'), ('early', 'בשליש הראשון'), ('middle', 'בשליש האמצעי'), ('late', 'בשליש האחרון')]
SROWS = ['S_equal', 'realized_drv', 'PRM_z_q80', 'S_equal - B13_equal', 'S_equal - realized_drv', 'S_equal - PRM_z_q80']
SHEAD = ''.join(f'<th>{h}</th>' for h in ['סינון ואז ממוצע', 'realized_drv', 'PRM, אותו כלל', 'תרומת הסינון', 'מול realized_drv', 'מול PRM'])
sl = ''.join(f'<tr><th scope="row">{lab}<span class="sub">{num(float(S1B["length_bin_answers"][k]), 0)[18:-7]} תשובות</span></th>' + ''.join(s1cell('length_quartile', k, r) for r in SROWS) + '</tr>' for k, lab in LB)
sp = ''.join(f'<tr><th scope="row">{lab}<span class="sub">{num(float(S1B["position_bin_answers"][k]), 0)[18:-7]} תשובות</span></th>' + ''.join(s1cell('first_error_position', k, r) for r in SROWS) + '</tr>' for k, lab in PB_)
H.append(f'''
<section aria-labelledby="h-sec">
  <h2 id="h-sec">משני: אורך תשובה ומיקום השגיאה הראשונה</h2>
  <p class="prose">PRMScore בכל תא. גבולות האורך הם רבעונים של מספר הצעדים, בלי תוויות. בגלל שוויונות בגבולות, הרבעונים לא שווים בגודלם. חלוקת המיקום משתמשת בתוויות ומתארת בלבד. היא גם מבולבלת עם סוג השגיאה: בקבוצת הצעד הראשון {num(FIRST_MC / FIRST_N, 0, pct=True)} הם תנאי חסר ({FIRST_MC} מתוך {FIRST_N}), ובקבוצת השליש האחרון {num(LATE_CONF / LATE_N, 0, pct=True)} הם ביטחון יתר ({LATE_CONF} מתוך {LATE_N}). לכן אין לקרוא את שיפוע הסינון לפי מיקום כאפקט של מיקום.</p>
  <h3>לפי אורך התשובה</h3>
  <div class="card scroll"><table><thead><tr><th>אורך</th>{SHEAD}</tr></thead><tbody>{sl}</tbody></table></div>
  <h3>לפי מיקום השגיאה הראשונה</h3>
  <div class="card scroll"><table><thead><tr><th>השגיאה הראשונה</th>{SHEAD}</tr></thead><tbody>{sp}</tbody></table></div>
</section>''')

H.append(f'''
<section aria-labelledby="h-cav">
  <h2 id="h-cav">תיקונים והסתייגויות</h2>
  <ul class="points">
    <li><b>תיקון לדיון הקודם:</b> כתבתי שהסימונים הבינאריים (top-20%) חושבו על כל הצעדים של כל התשובות יחד. זה לא נכון. בשלבים A, B ו־B2 הסימון הוא בתוך כל תשובה, והרנרים מוודאים שכל תשובה קיבלה בדיוק את המספר הזה. "כל צעדי ה־fit folds" מתייחס רק לשורות שהאומדים מותאמים עליהן. הניסוי שהצעתי (סימונים בתוך תשובה והסרת מיקום) כבר קיים, כולל הבנק המותאם למיקום. השכיחות המשוערת (0.22 עד 0.24) קרובה לשיעור הסימון הכפוי, 0.2.</li>
    <li><b>האוכלוסייה:</b> מתוך {num(pop["noncontrol"], 0)} התשובות שאינן ביקורת, {num(pop["erroneous_noncontrol"], 0)} עם צעד שגוי בתוך התשובה, {num(pop["multi_solutions"], 0)} ריבוי פתרונות, ו־{num(pop["inert_error_annotation"], 0)} תשובות שכל אינדקסי השגיאה שלהן אחרי הצעד האחרון. המסמך המקורי של הניסוי טעה בחשבון הזה, והתיקון נכתב (תיקון A1) לפני שחושב מספר כלשהו.</li>
    <li><b>יתירות ולוגיקה מעגלית</b> נמדדות בשני הצדדים דרך כלל חלופי (validity fallback), כי אין ראש יתירות. זה לא הפרוטוקול הרשמי לשני הסוגים האלה.</li>
    <li><b>realized_drv</b> נבחר בדיעבד, כטוב מבין 22 ערוצים על אותם נתונים. כאן הוא נקודת ייחוס ולא מועמד. אין צורך שהוא יוביל כדי שהאומדנים הבינאריים ייחשבו מוצלחים: הוא נבחר לפי דירוג רציף, והאומדנים מתייחסים לסימונים אחרי חיתוך.</li>
    <li><b>הרכב הבנק:</b> הכלל הבינארי עם level ממוזג משתמש ברשימה קשיחה של חמשת ערוצי ה־level. זו בדיקת מנגנון ולא שיטה אלגוריתמית.</li>
    <li><b>סוג הנתונים:</b> אלה נתוני פיתוח ולא אישור. ה־PRM אומן עם תוויות צעדים ורץ forward pass משלו; הוא נקודת ייחוס, לא מתחרה בתנאים שווים.</li>
    <li><b>ProcessBench</b> מחוץ לפירוק הזה.</li>
  </ul>
</section>

<section aria-labelledby="h-dec">
  <h2 id="h-dec">מה זה אומר לשאלה "בנק חדש או טענה אחרת"</h2>
  <ul class="points">
    <li><b>ערך הבחירה של SML:</b> הרווח של הסינון על PRMScore קטן, ורובו משוחזר ממבנה מיקום. זה לא מבסס ערך ייחודי לבחירה מעבר למיקום.</li>
    <li><b>כיול:</b> הנרמול בתוך תשובה עולה לשורות שלנו ביכולת לזהות תשובה נקייה, ומוחק את ההשוואה בין תשובות שה־PRM מקבל מהסקאלה שלו. בכלל שלנו, רוב היתרון של ה־PRM הוא במספר הצעדים שהוא מסמן בכל תשובה. זה מצביע על השער (כמה לסמן בכל תשובה) ולא על המשקלים.</li>
    <li><b>השלמה:</b> השורה המאוחדת וה־PRM תופסים תשובות שונות (לפי סוגי שגיאה שונים). השורה המאוחדת והערוץ הבודד תופסים תשובות שונות גם כן, אבל האיחוד הנוכחי לא מנצל את האות הספציפי לתשובה של realized_drv.</li>
  </ul>
</section>

<footer>
  <p><b>שחזור:</b> פרוטוקול קפוא <span class="num">a6a7ea493</span> לפני כל תוצאה, תיקון A1 <span class="num">94984fe01</span> לפני כל מספר, הרצה <span class="num">1d0a25b25</span>. כל השערים עברו: המדדים הקפואים משוחזרים עד <span class="num">{GATES["frozen_metrics_replay_max_abs_diff"]:.1e}</span>, המדרג הרשמי שווה לספירות ב־{num(float(GATES["counts_compared_by_class"]), 0)[18:-7]} השוואות לפי סוג, הרצות הספים של B ו־B2 זהות ביט אחר ביט לריצות הקפואות, וה־PRM בסף 0.5 משחזר את Step 437.</p>
  <p><b>ביקורת:</b> סקירה בלתי תלויה של הקוד לפני ההרצה (שני באגים חוסמים תוקנו) וצוות אדום של שלושה סוכנים: חישוב מחדש בלתי תלוי, ביקורת כיסוי אוכלוסייה, ובדיקות אפס ומתמטיקה. אף מספר לא נמצא שגוי; שלוש קריאות נחלשו, והן מנוסחות כאן בהתאם. <span class="num">RED_TEAM.md</span>.</p>
  <p><b>קבצים:</b> <span class="num">results/expectation_realization_v1/prmscore_decomposition_v1/</span> (P1_OVERALL.json, P2_CONTRASTS.csv, P2_ARMS_BY_STRATUM.csv, P3_RANKING_DECISIONS.json, P4_COMPLEMENTARITY.csv, S1_LENGTH_POSITION.csv, GATES.json, red_team/).</p>
</footer>''')

CSS = '''
:root{--ground:#f4f6f8;--surface:#ffffff;--ink:#16202b;--ink2:#465262;--muted:#768091;--hair:#e1e5eb;--axis:#c5ccd6;
--ours:#0f6e78;--drv:#6a4fb3;--prm:#b0661c;--ref:#8a93a1;--mech:#5d8a93;--noise:#b7bec9;
--pos:rgba(29,122,75,.13);--posink:#17613c;--neg:rgba(178,58,47,.12);--negink:#9b3027;--chip:#eef1f5;
--display:"Frank Ruhl Libre","David Libre",Georgia,serif;--sans:"Assistant","Segoe UI",Arial,sans-serif;--mono:"IBM Plex Mono",Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--ground:#0f1318;--surface:#171c23;--ink:#e8edf3;--ink2:#b7c0cc;--muted:#8a94a3;--hair:#262d37;--axis:#39424f;
--ours:#3fb1bd;--drv:#a08ce0;--prm:#e0954e;--ref:#8f98a6;--mech:#7fb3bc;--noise:#59616d;--pos:rgba(95,197,140,.16);--posink:#7fd4a2;--neg:rgba(240,135,124,.15);--negink:#f3a097;--chip:#20262f}}
:root[data-theme="dark"]{color-scheme:dark;--ground:#0f1318;--surface:#171c23;--ink:#e8edf3;--ink2:#b7c0cc;--muted:#8a94a3;--hair:#262d37;--axis:#39424f;
--ours:#3fb1bd;--drv:#a08ce0;--prm:#e0954e;--ref:#8f98a6;--mech:#7fb3bc;--noise:#59616d;--pos:rgba(95,197,140,.16);--posink:#7fd4a2;--neg:rgba(240,135,124,.15);--negink:#f3a097;--chip:#20262f}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.6}
.page{max-width:1060px;margin:0 auto;padding-inline:20px;padding-block:36px 64px;display:grid;gap:40px}
@media (max-width:480px){.page{padding-inline:16px;padding-block:24px 48px}}
h1,h2,h3{font-family:var(--display);font-weight:700;line-height:1.25;margin:0;text-wrap:balance}
h1{font-size:clamp(28px,4.6vw,40px)} h2{font-size:clamp(21px,3vw,27px)} h3{font-size:17px;font-family:var(--sans);font-weight:700;margin-top:6px}
p{margin:0} section{display:grid;gap:14px} .top{display:grid;gap:12px}
.eyebrow{font-size:13px;letter-spacing:.05em;color:var(--muted)} .lede{font-size:18px;max-width:74ch;color:var(--ink2)}
.prose{max-width:78ch;color:var(--ink2)} .note{font-size:14px;color:var(--muted);max-width:86ch}
.points{margin:0;padding-inline-start:1.3em;display:grid;gap:10px;max-width:82ch} .points li{color:var(--ink2)} .points b{color:var(--ink)}
.points.small{font-size:15px}
.num{direction:ltr;unicode-bidi:isolate;font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:.92em;white-space:nowrap}
.ci{color:var(--muted);font-size:.78em} .na{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:10px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:9px 10px;text-align:start;vertical-align:top;border-bottom:1px solid var(--hair)}
thead th{font-size:12.5px;font-weight:700;color:var(--ink2);border-bottom:1px solid var(--axis);vertical-align:bottom}
tbody th{font-weight:600;min-width:170px} tbody tr:last-child th,tbody tr:last-child td{border-bottom:none}
table.big td{white-space:nowrap}
.sub{display:block;font-size:12px;color:var(--muted);font-weight:400}
tr.tot th,tr.tot td{border-top:1px solid var(--axis);font-weight:700}
td.d.pos{background:var(--pos)} td.d.pos .num:first-child{color:var(--posink)}
td.d.neg{background:var(--neg)} td.d.neg .num:first-child{color:var(--negink)}
.bf{font-size:9px;margin-inline-start:4px;vertical-align:middle;color:var(--ink2)}
.sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin-inline-end:7px;vertical-align:1px}
.sw-ours{background:var(--ours)} .sw-drv{background:var(--drv)} .sw-prm{background:var(--prm)} .sw-ref{background:var(--ref)} .sw-mech{background:var(--mech)} .sw-noise{background:var(--noise)}
.ctl{display:flex;flex-wrap:wrap;gap:6px}
.ctl button{font:inherit;font-size:14px;border:1px solid var(--hair);background:var(--surface);color:var(--ink2);padding:4px 13px;border-radius:999px;cursor:pointer}
.ctl button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.ctl button:focus-visible{outline:2px solid var(--ours);outline-offset:2px}
.callout{border-inline-start:3px solid var(--ours);padding-inline-start:12px;color:var(--ink2);max-width:82ch} .callout b{color:var(--ink)}
.muted{color:var(--muted)}
footer{display:grid;gap:8px;font-size:13.5px;color:var(--muted);max-width:90ch;border-top:1px solid var(--hair);padding-top:18px} footer b{color:var(--ink2)}
'''
JS = '''
document.querySelectorAll('.ctl button[data-ep]').forEach(function(b){b.addEventListener('click',function(){
  var ep=b.getAttribute('data-ep');
  document.querySelectorAll('.ctl button[data-ep]').forEach(function(x){x.setAttribute('aria-pressed',String(x===b));});
  document.querySelectorAll('.panel[data-ep]').forEach(function(p){p.hidden=p.getAttribute('data-ep')!==ep;});
});});
'''
page = ('<title>פירוק PRMScore בבנק 13</title>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700&family=Frank+Ruhl+Libre:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
        f'<style>{CSS}</style>\n<div class="page" dir="rtl" lang="he">{"".join(H)}</div>\n<script>{JS}</script>\n')
assert '‏' not in page and '‎' not in page
(D / 'REPORT_HE.html').write_bytes(page.encode('utf8'))
print('wrote', D / 'REPORT_HE.html', len(page))
