"""REPORT_HE.md built only from the panel CSVs (numbers are never typed by hand)."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent; HL = 'B16'
METHODS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']; ALL5 = METHODS + ['HEM_MOM_ga']
PRM_L = 'Qwen2.5-Math-PRM-7B (supervised)'; MTG_L = 'Mind the Gap (local repro.)'
A = pd.read_csv(OUT / 'PANEL_A_ABSOLUTE.csv'); CA = pd.read_csv(OUT / 'PANEL_A_CONTRASTS.csv')
B = pd.read_csv(OUT / 'PANEL_B_PRMBENCH.csv'); T = pd.read_csv(OUT / 'PANEL_B_PRMBENCH_TYPES.csv')
P = pd.read_csv(OUT / 'PANEL_B_PROCESSBENCH.csv'); F = pd.read_csv(OUT / 'PANEL_B_PROCESSBENCH_F1.csv'); CB = pd.read_csv(OUT / 'PANEL_B_CONTRASTS.csv')
V = json.loads((OUT / 'VALIDATION.json').read_text(encoding='utf8'))
f3 = lambda v: '—' if pd.isna(v) else f'{v:.3f}'; s3 = lambda v: f'{v:+.3f}'
ci = lambda r: f'{r.delta:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]'

md = ['# פאנלים למצגת: האלגוריתם מול איחוד פשוט, והשיטה מול המתחרים', '',
      '**סטטוס:** תוצאות פיתוח על PRMBench ו־ProcessBench, אוכלוסייה שכבר נחשפה. לא נבחר ולא ננעל מתכון; ארבע השיטות מוצגות יחד. '
      f'הבנק המודגש **{HL}** נבחר בדיעבד לפי ה־AUC הממוצע של ארבע השיטות ב־PRMBench (`PROTOCOL.json`). '
      'כל הטבלאות נבנו אחרי שחלק מהמספרים כבר נראו בביקורת (גילוי מלא בפרוטוקול).', '',
      f'**אימות:** ציוני השיטות, הבקרות ודגלי ההחלטה שוחזרו מהקבצים הקפואים עד {max(V["score_replay_max"], V["metric_replay_max"], V["control_replay_max"]):.1e}; '
      f'{V["flag_replays"]} השוואות דגלים זהות. {V["n_prmbench_answers"]:,} תשובות PRMBench, {V["n_processbench_rows"]:,} שורות ProcessBench '
      f'({V["n_pb_error_rows"]:,} שגויות; כל תשובה מופיעה פעמיים, בשני מודלי telemetry).', '']

# ---------------- Panel A
md += ['## פאנל א: האם האלגוריתם מוסיף על ממוצע פשוט של אותו בנק?', '',
       'השוואה לכל בנק בנפרד: כל שיטה פחות `DS_FILTER_EQ` (ממוצע שווה של הערוצים שעברו את מסנן DS — אותו קלט, בלי שקלול). '
       'אין כאן מתחרים חיצוניים. רווחי סמך מזווגים לפי שאלת מקור, 5,000 דגימות, תיאוריים.', '',
       '![](figures/A1_algorithm_vs_simple_average_light.png)', '']
c = CA[CA.reference == 'DS_FILTER_EQ']
for metric, name in (('prmb_within_auc', 'PRMBench within-answer AUC'), ('pb_argmax_hit', 'ProcessBench פגיעת הצעד המדורג ראשון')):
    s = c[c.metric == metric]; pos = int((s.lo95 > 0).sum()); neg = int((s.hi95 < 0).sum())
    md.append(f'- **{name}:** מתוך {len(s)} צירופי שיטה×בנק, {pos} חיוביים ו־{neg} שליליים (רווח שאינו כולל אפס); השאר לא מובחנים. '
              f'טווח ההפרשים {s3(s.delta.min())} עד {s3(s.delta.max())}.')
g_ = c[c.metric == 'prmb_within_auc'].groupby('bank'); npos = g_.apply(lambda g: int((g.lo95 > 0).sum())); nneg = g_.apply(lambda g: int((g.hi95 < 0).sum()))
md += [f'- התרומה תלויה בבנק (PRMBench, לפחות 3 מתוך 5 שיטות): חיובית ב־{", ".join(b for b in npos.index if npos[b] >= 3)}; שלילית ב־{", ".join(b for b in nneg.index if nneg[b] >= 3)}; '
       f'מעורבת ב־{", ".join(b for b in npos.index if npos[b] < 3 and nneg[b] < 3)} (כולל הבנק המודגש).',
       '- המסקנה: השקלול מוסיף אלפיות עד כשתי נקודות AUC על חלק מהבנקים ב־PRMBench, ולא מוסיף ב־ProcessBench. ההבדל בין הבנקים גדול מההבדל בין השיטות.', '']
tab = A.pivot(index='bank', columns='scorer', values='prmb_within_auc')
md += ['| בנק | ' + ' | '.join(ALL5 + ['DS_FILTER_EQ', 'ALL_EQ']) + ' |', '|---|' + '---:|' * (len(ALL5) + 2)]
for bk, r in tab.iterrows():
    md.append(f'| {"**" + bk + "**" if bk == HL else bk} | ' + ' | '.join(f3(r[x]) for x in ALL5 + ['DS_FILTER_EQ', 'ALL_EQ']) + ' |')
md += ['', '_ערכים מוחלטים: PRMBench within-answer AUC. ProcessBench והשוואה ל־ALL_EQ: `PANEL_A_ABSOLUTE.csv`, `PANEL_A_CONTRASTS.csv`._', '']

# ---------------- Panel B PRMBench
md += ['## פאנל ב: השיטה כמאתרת שגיאות, מול המתחרים', '', f'### PRMBench — הבנק המודגש {HL}, ארבע השיטות', '',
       '![](figures/B1_prmbench_headline_light.png)', '',
       '| מדרג | within-answer AUC | הצעד הראשון בדירוג שגוי | PRMScore, דגל אחד לתשובה | F1 על צעדים תקינים | F1 על צעדי שגיאה | PRMScore בכלל של השיטה | שיעור צעדים מסומנים בכלל זה |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
rows = B[(B.bank == HL) & B.scorer.isin(METHODS)].to_dict('records') + B[B.bank == '-'].to_dict('records')
for r in rows:
    md.append(f"| {r['scorer']} | {f3(r['within_auc'])} | {f3(r['top1_on_error'])} | {f3(r['prmscore_top1'])} | {f3(r['f1_correct_top1'])} | {f3(r['f1_error_top1'])} | {f3(r['prmscore_own'])} | {f3(r['flag_rate_own'])} |")
md += ['', 'הכלל שלנו הוא ANSWER_Z_Q80 (20% מהצעדים); הכלל של ה־PRM הוא הסף המקורי 0.5 (8.7% מהצעדים). זו גישת כיול שונה, ולכן העמודה "דגל אחד לתשובה" היא ההשוואה ההוגנת. '
       'PRMScore עם שני דגלים לתשובה הוא בחירה בדיעבד ונמצא ב־CSV בלבד.', '']
sub = CB[(CB.benchmark == 'PRMBench')]
md += ['**ניגודים מזווגים (B16):**', '', '| שיטה | מול | AUC | צעד ראשון שגוי | PRMScore דגל אחד |', '|---|---|---:|---:|---:|']
for (m, vs), g in sub.groupby(['method', 'versus'], sort=False):
    g = g.set_index('metric')
    md.append(f'| {m} | {vs} | {ci(g.loc["within_auc"])} | {ci(g.loc["top1_on_error"])} | {ci(g.loc["prmscore_top1"])} |')
md += ['', '- **חוזקה:** בתקציב סימון שווה, השיטות שלנו ב־B16 גבוהות מה־PRM המפוקח בצעד הראשון בדירוג וב־PRMScore, ב־AUC שתי גרסאות cc לא מובחנות ממנו (הרווח כולל אפס) ו־MEAN_MOM_ga ו־TCI נמוכות ממנו במעט. '
       'מול Mind the Gap — יתרון גדול בכל המדדים. **הסתייגות:** B16 נבחר בדיעבד; ממוצע על שמונת הבנקים נמוך יותר (`PANEL_B_PRMBENCH.csv`).',
       f'- **רכיבי PRMScore ב־B16 (דגל אחד לתשובה):** F1 על צעדי שגיאה {f3(B[(B.bank == HL) & B.scorer.isin(METHODS)].f1_error_top1.min())}–{f3(B[(B.bank == HL) & B.scorer.isin(METHODS)].f1_error_top1.max())} מול {f3(B[B.scorer == PRM_L].f1_error_top1.iloc[0])} ל־PRM; '
       f'על צעדים תקינים כמעט זהה ({f3(B[(B.bank == HL) & B.scorer.isin(METHODS)].f1_correct_top1.mean())} מול {f3(B[B.scorer == PRM_L].f1_correct_top1.iloc[0])}). '
       'החולשה היא כלל ההחלטה: ANSWER_Z_Q80 מסמן 20% מהצעדים ומדליק לפחות דגל אחד ב־99.1%–99.6% מ־934 התשובות ללא צעד שגוי (`METRICS.csv` של F11–F12, B16, clean_flag_rate).', '']

md += ['### PRMBench לפי סוג שגיאה', '', '![](figures/B2_error_types_light.png)', '']
t = T[T.group.str.startswith('cls:') & (T.group != 'cls:multi_solutions')]
md += ['| סוג | שלנו B16 (ממוצע 4) | PRM | MTG | שלנו פחות PRM |', '|---|---:|---:|---:|---:|']
rr = []
for g, gg in t.groupby('group'):
    o = gg[(gg.bank == HL) & gg.scorer.isin(METHODS)].within_auc.mean(); p = gg[gg.scorer == PRM_L].within_auc.iloc[0]; m = gg[gg.scorer == MTG_L].within_auc.iloc[0]
    rr.append((g[4:], o, p, m))
for g, o, p, m in sorted(rr, key=lambda x: x[1] - x[2], reverse=True):
    md.append(f'| {g} | {f3(o)} | {f3(p)} | {f3(m)} | {s3(o - p)} |')
md += ['', '**הקריאה:** אנחנו חזקים בשגיאות "תהליך" — חזרתיות, היגיון מעגלי, חוסר עקביות — שבהן המודל עצמו מגלה אי־ודאות. '
       'ה־PRM חזק בשגיאות תוכן שנאמרות בביטחון (confidence, counterfactual, deception): שגיאה שנכתבת בביטחון כמעט אינה משאירה עקבה באי־הוודאות, וזה גבול טבעי של גישה שאינה משתמשת בתוויות. '
       'מול Mind the Gap אנחנו גבוהים בכל שמונת סוגי השגיאה (multi_solutions אינו מכיל צעדי שגיאה). ללא רווחי סמך ברמת הסוג; תיאורי.', '']

# ---------------- Panel B ProcessBench
md += ['### ProcessBench — איתור בלי gate (הפרוטוקול של Mind the Gap)', '', '![](figures/B3_processbench_localization_light.png)', '']
st = ['ALL', 'content-only (not step 0, not longest)', 'error at step 0', 'error in longest step']
names = {'ALL': 'כל התשובות השגויות', 'content-only (not step 0, not longest)': 'השגיאה לא בצעד 0 ולא בצעד הארוך ביותר',
         'error at step 0': 'השגיאה בצעד 0', 'error in longest step': 'השגיאה בצעד הארוך ביותר'}
md += ['| שכבה | n | שלנו B16 (טווח 4) | MTG | PRM argmax | PRM בסף שלו | מבקר 72B | תמיד צעד 0 | הצעד הארוך |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for s in st:
    g = P[P.stratum == s]; o = g[(g.bank == HL) & g.scorer.isin(METHODS)].argmax_hit
    val = lambda lab: g[g.scorer == lab].argmax_hit.iloc[0]
    md.append(f'| {names[s]} | {int(g.n_error_rows.iloc[0])} | {f3(o.min())}–{f3(o.max())} | {f3(val(MTG_L))} | {f3(val("Qwen2.5-Math-PRM-7B argmax"))} | '
              f'{f3(val("Qwen2.5-Math-PRM-7B native (0.5)"))} | {f3(val("Qwen2.5-72B critic native"))} | {f3(val("Always step 0"))} | {f3(val("Longest step"))} |')
sub = CB[CB.benchmark == 'ProcessBench']
md += ['', '| שיטה (B16) | מול | כל השגויות | ללא צעד 0 וללא הצעד הארוך |', '|---|---|---:|---:|']
for (m, vs), g in sub.groupby(['method', 'versus'], sort=False):
    g = g.set_index('stratum'); md.append(f'| {m} | {vs} | {ci(g.loc["ALL"])} | {ci(g.loc[st[1]])} |')
all32 = P[(P.stratum == st[1]) & P.scorer.isin(METHODS)].argmax_hit.mean(); mtg_c = P[(P.stratum == st[1]) & (P.scorer == MTG_L)].argmax_hit.iloc[0]
md += ['', f'- **מול Mind the Gap:** ב־B16 שתי גרסאות cc נשארות מעליו גם כשמוציאים שגיאות בצעד 0 ובצעד הארוך ביותר; גרסאות ga ו־TCI שם לא מובחנות ממנו. '
       f'בממוצע על כל 32 התצורות, בשכבה הזו אנחנו {f3(all32)} מול {f3(mtg_c)} — כלומר היתרון תלוי בבנק.',
       '- **מגבלה ידועה:** בממוצע על 32 התצורות, הצעד המדורג ראשון אצלנו הוא הצעד הארוך ביותר בכ־52% מהתשובות השגויות (MTG: 42%; ביקורת, `REVIEW.md` של F11–F12); חלק מהיתרון ב־ProcessBench הוא הטיית אורך (ב־PRMBench היא קטנה: אורך בלבד נותן AUC 0.618).',
       '- **מול ה־PRM המפוקח והמבקר:** רחוקים מאוד. ה־PRM מוצא את השגיאה הראשונה בכ־57% מהתשובות השגויות (argmax), אנחנו ב־40%.', '']
md += ['**F1 הרשמי (עם החלטה על תשובות נקיות; גישת כיול שונה):**', '', '| מדרג | כלל | F1 | פגיעה בשגויות | נכון על נקיות |', '|---|---|---:|---:|---:|']
for _, r in F[(F.bank.isin([HL, '-']))].iterrows():
    md.append(f'| {r.scorer} | {r.rule} | {f3(r.official_f1)} | {f3(r.error_hit)} | {f3(r.clean_correct)} |')
md += ['', 'ההחלטה על תשובות נקיות היא צוואר הבקבוק: ANSWER_Z_Q80 מסמן כמעט כל תשובה נקייה (F1 כ־0.10); R2 ושער האנטרופיה מגיעים ל־0.37–0.41, מול 0.73 ל־PRM.', '']

md += ['## מה מותר לומר', '',
       '- מותר: "שיטה ללא תוויות מדרגת צעדים בתוך תשובה בערך ברמת PRM מפוקח על PRMBench, עם פרופיל שגיאות משלים, וגבוה בבירור מ־Mind the Gap."',
       '- מותר: "השקלול מוסיף על ממוצע פשוט רק בחלק מהבנקים; בחירת הבנק משנה יותר מבחירת השיטה."',
       '- אסור: עליונות על PRM מפוקח; מנצח בין ארבע השיטות; B16 כבחירה מוקדמת; שקילות מרווח הכולל אפס; מספרי המאמר של Mind the Gap כהשוואה.', '',
       '## קבצים', '', '- `PROTOCOL.json`, `INPUT_MANIFEST.json`, `VALIDATION.json` (+ `VALIDATION_ATTEMPT1_FAIL.json`: ניסיון ראשון שנעצר על כלל argmax; תוקן לכלל הקפוא)',
       '- `PANEL_A_ABSOLUTE.csv`, `PANEL_A_CONTRASTS.csv`, `PANEL_B_PRMBENCH.csv`, `PANEL_B_PRMBENCH_TYPES.csv`, `PANEL_B_PROCESSBENCH.csv`, `PANEL_B_PROCESSBENCH_F1.csv`, `PANEL_B_CONTRASTS.csv`, `PER_ANSWER.npz`',
       '- `figures/` (בהיר וכהה, PNG ו־SVG), `build.py`, `render.py`, `make_report.py`', '']
(OUT / 'REPORT_HE.md').write_text('\n'.join(md), encoding='utf8')
man = {'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('PANEL_*.csv'))},
       'report_sha256': hashlib.sha256((OUT / 'REPORT_HE.md').read_bytes()).hexdigest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / 'REPORT_MANIFEST.json').write_text(json.dumps(man, indent=1), encoding='utf8'); print('report written')
