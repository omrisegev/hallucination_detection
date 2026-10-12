"""Independent metric replay and readable report for the completed diagnostic."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import hashlib
import html
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contrast_span_oracle_20261005/run_v3'
CODE = ROOT / 'results/monday_fusion_analysis_v1/code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'scripts/experiments'))
import calfix_common as CC

complete = json.loads((OUT / 'COMPLETE.json').read_text(encoding='utf8'))
assert complete['status'] == 'PASS' and complete['n_cells'] == 18
for name, expected in complete['artifacts'].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, name
pop = CC.Population()
summary = pd.read_csv(OUT / 'SUMMARY.csv')
per = pd.read_csv(OUT / 'PER_CELL.csv')
primary = ['plain', 'LR_PC1', 'LR_PC1_single_linear', 'LR_PC1_within_group_linear', 'LR_all_X']
checks = []
for path in sorted(OUT.glob('*.scores.npz')):
    bank, cell = path.name.removesuffix('.scores.npz').split('__', 1)
    saved = np.load(path)
    answers, offsets = saved['answers'], saved['offsets']
    assert len(answers) == int(np.sum(pop.cells == cell))
    assert np.array_equal(answers, np.flatnonzero(pop.cells == cell))
    assert np.array_equal(np.diff(offsets), pop.ns[answers])
    for method in primary:
        if method not in saved:
            assert bank == 'B32' and method == 'LR_PC1_single_linear'
            continue
        values = []
        for i, a, b in zip(answers, offsets[:-1], offsets[1:]):
            score = saved[method][a:b]
            if pop.prm[i] and pop.eligible[i]:
                label = pop.labels[pop.off[i]:pop.off[i + 1]]
                values.append(roc_auc_score(label, score))
            elif not pop.prm[i] and pop.target[i] >= 0:
                tolerance = 8 * np.finfo(float).eps
                chosen = next(j for j, s in enumerate(score) if score.max() - s <= tolerance)
                values.append(float(chosen == pop.target[i]))
        row = per[(per.bank == bank) & (per.cell == cell) & (per.method == method)].iloc[0]
        assert len(values) == row['count']
        error = abs(float(np.mean(values)) - row['mean'])
        assert error < 1e-12, (bank, cell, method, error)
        checks.append(dict(bank=bank, cell=cell, method=method, n_checked=len(values), error=error))
    print('Independent metrics PASS', bank, cell, flush=True)

fits = json.loads((OUT / 'FITS.json').read_text(encoding='utf8'))
span_rows = []
for info in fits:
    for fit in info['fits']:
        if 'basis' in fit:
            basis = np.array(fit['basis'])
            np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
            span_rows.append(dict(bank=info['bank'], cell=info['cell'], fold=info['fold'],
                                  method=fit['method'], rank=fit['rank'], n_channels=len(info['names']),
                                  n_within_pairs=info['n_within_group_pairs']))
pd.DataFrame(span_rows).to_csv(OUT / 'SPAN_RANKS.csv', index=False)
contrasts = pd.read_csv(OUT / 'PAIRED_CONTRASTS.csv')
recovery = []
for bk in ['B13', 'B32']:
    for panel in ['PRMB', 'PB']:
        values = summary[(summary.bank == bk) & (summary.panel == panel)].set_index('method').value
        for method in ['LR_PC1_single_linear', 'LR_PC1_within_group_linear', 'LR_PC1_single_answer_std', 'LR_PC1_within_group_answer_std']:
            if method in values:
                recovery.append(dict(bank=bk, panel=panel, method=method,
                                     gain_vs_plain=values[method] - values['plain'],
                                     gap_to_full=values['LR_all_X'] - values[method],
                                     fraction_plain_to_full=(values[method] - values['plain']) / (values['LR_all_X'] - values['plain']),
                                     fraction_pc1_to_full=(values[method] - values['LR_PC1']) / (values['LR_all_X'] - values['LR_PC1'])))
pd.DataFrame(recovery).to_csv(OUT / 'GAP_RECOVERY.csv', index=False)
review = {'status': 'PASS', 'n_cells': 18, 'n_metric_rows_independently_replayed': len(checks),
          'metric_checks': checks, 'n_fold_fits': len(fits), 'n_unit_tests': 7,
          'max_metric_error': max(c['error'] for c in checks),
          'max_full_span_score_error': max(v for f in fits for v in f['checks'].values()),
          'limitations': ['Same-agent independent metric implementation, not an external review',
                          'Supervised development diagnostic with historical transductive feature selection',
                          'Bootstrap intervals conditional on OOF fitted models; no bootstrap refitting',
                          'No new label-free fusion result, no untouched confirmation',
                          'Named single contrast unavailable in B32; no substitution']}
(OUT / 'REVIEW.json').write_text(json.dumps(review, indent=2), encoding='utf8')

labels = {'plain': 'ממוצע הפיצ׳רים', 'LR_PC1': 'גורם משותף בלבד — משקל נלמד',
          'LR_PC1_single_linear': 'גורם משותף + ניגוד יחיד',
          'LR_PC1_within_group_linear': 'גורם משותף + כל הניגודים בתוך הקבוצות',
          'LR_all_X': 'LR על כל הפיצ׳רים',
          'LR_PC1_single_answer_std': 'גורם משותף + ניגוד יחיד בתקנון בתוך תשובה',
          'LR_PC1_within_group_answer_std': 'גורם משותף + כל ניגודי הקבוצות בתקנון בתוך תשובה'}
tables = []
for panel in ['PRMB', 'PB']:
    rows = []
    for method, label in labels.items():
        row = {'ייצוג': label}
        for bk in ['B13', 'B32']:
            found = summary[(summary.bank == bk) & (summary.panel == panel) & (summary.method == method)]
            row[bk] = 'לא זמין: הערוץ חסר בבנק' if found.empty else f"{float(found.value.iloc[0]):.6f}"
        rows.append(row)
    tables.append('<h2>' + ('PRMBench — AUC בתוך תשובה' if panel == 'PRMB' else 'ProcessBench — איתור השגיאה הראשונה, ממוצע שמונה תאים') + '</h2>' + pd.DataFrame(rows).to_html(index=False, escape=True))

report = '''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">
<title>ניגוד יחיד מול מכלול הניגודים — אבחון מפוקח</title>
<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.7;padding:0 20px;color:#182330}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #ccd4dc;padding:9px;text-align:right}th{background:#edf2f7}code{direction:ltr;unicode-bidi:embed}a{color:#14629a}</style>
<h1>ניגוד יחיד מול מכלול הניגודים</h1>
<p>5 באוקטובר 2026. אבחון מפוקח על נתוני פיתוח. המשקלים נלמדו מתוויות בקיפולי האימון בלבד; אין כאן תוצאה של שיטה חסרת תוויות.</p>
<p>נבדקו שני בנקים בכל תשעת התאים: 13,769 תשובות ייחודיות בכל בנק. המדד ב־PRMBench מחושב על 6,030 תשובות שבהן יש גם צעדים תקינים וגם שגויים. ב־ProcessBench המדד מחושב רק על תשובות שגויות; הוא אינו כולל החלטה אם התשובה תקינה ואינו PRMScore.</p>
<p>הניגוד היחיד הוא realized_drv פחות chosen_surprisal. הוא קיים ב־B13 בלבד. B32 הוא בנק אחר ולא הרחבה של B13; לא הוספנו אליו פיצ׳רים.</p>
<p>מכלול הניגודים כולל את כל זוגות הפיצ׳רים השורדים בתוך קבוצות L-SML הקפואות, ללא בחירה לפי תוויות, צידוד או מסנן MoM נוסף. צמצמנו את ההפרשים לבסיס אורתונורמלי כדי שכפילויות לא ישנו את הענישה על המשקלים. ההשוואה הראשית שומרת על אותה גאומטריית ענישה של LR בכל הייצוגים. בדיקה משלימה מתקננת כל ניגוד בתוך תשובה, כמו בניסוי הקודם, ומשנה בכך את הייצוג.</p>
'''+ ''.join(tables) + '''
<h2>מה הבדיקות מזהות</h2>
<p>השילוב של הממוצע עם כל ההפרשים בין כל זוגות הפיצ׳רים משחזר את מלוא המרחב הלינארי. כך גם הגורם המשותף יחד עם כל השאריות. התאמתם ל־LR המלא היא בדיקת תקינות אלגברית, לא גילוי של מידע חדש. הבדיקה המהותית היא כמה מהשיפור נשמר עם ניגוד יחיד או עם ניגודים בתוך הקבוצות בלבד.</p>
<p>כל התאמות ה־LR משתמשות ב־C=1 ובאיזון מחלקות, ללא חיפוש פרמטרים. התקנון הגלובלי ו־PCA החדשים הותאמו רק בקיפולי האימון. סינון הפיצ׳רים והקבוצות היסטוריים ומותאמים ללא תוויות ברמת התא כולו; זה אינו ניסוי של התאמה מתשובה אחת בלבד.</p>
<p>רווחי הסמך מבוססים על 5,000 דגימות חוזרות של שאלות מקור, בזוגות ועל אותם ציוני OOF; ב־ProcessBench נשמר ממוצע שמונת התאים. הם חקרניים, ללא תיקון לריבוי השוואות וללא התאמה מחדש בכל דגימה.</p>
<h2>בדיקות וקבצים</h2>
<p>שבע בדיקות יחידה עברו. כל 18 התאים הושלמו. ציוני הממוצע וה־LR ההיסטורי שוחזרו; המדדים העיקריים חושבו שוב במימוש נפרד, כולל AUC באמצעות scikit-learn. מקורות וקוד נחתמו ב־SHA256 לפני הריצה ונבדקו ללא שינוי בסיומה. הבדיקה הנפרדת בוצעה בידי אותו סוכן.</p>
<ul><li><a href="SUMMARY.csv">טבלת התוצאות</a></li><li><a href="PAIRED_CONTRASTS.csv">הפרשים ורווחי סמך</a></li><li><a href="GAP_RECOVERY.csv">שיעור הפער שנסגר</a></li><li><a href="SPAN_RANKS.csv">ממדי הייצוגים</a></li><li><a href="REVIEW.json">בדיקות עצמאיות</a></li><li><a href="HISTORICAL_REPLAY.json">שחזור הייחוסים</a></li><li><a href="../PROTOCOL.json">פרוטוקול</a></li><li><a href="../PROTOCOL_AMENDMENT_V2.json">תיקון זמינות הפיצ׳ר</a></li></ul>
</html>'''
(OUT / 'REPORT_HE.html').write_text(report, encoding='utf8')
assert 'ניגוד יחיד' in (OUT / 'REPORT_HE.html').read_text(encoding='utf8')
print('REVIEW PASS:', len(checks), 'metric rows;', len(fits), 'folds', flush=True)
print(pd.DataFrame(recovery).to_string(index=False), flush=True)
