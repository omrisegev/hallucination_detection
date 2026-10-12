"""Validate and render the completed group-score diagnostic."""
from pathlib import Path
import hashlib
import html
import json
import re
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/group_contrast_oracle_20261005'
OUT = BASE / 'run_v1'
complete = json.loads((OUT / 'COMPLETE.json').read_text(encoding='utf8'))
assert complete['status'] == 'PASS' and complete['n_cells'] == 18
for name, expected in complete['artifacts'].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, name
review = json.loads((OUT / 'REVIEW.json').read_text(encoding='utf8'))
assert review['n_metric_rows'] == 234 and review['n_fold_fits'] == 90
summary = pd.read_csv(OUT / 'SUMMARY.csv')
paired = pd.read_csv(OUT / 'PAIRED_CONTRASTS.csv')
assert len(paired) == 48 and set(paired.valid_draws) == {5000}
assert set(summary[summary.panel == 'PRMB'].n_checked) == {6030}
assert set(summary[summary.panel == 'PB'].n_checked) == {4442}
gates = pd.read_json(OUT / 'GATES.json')
assert (gates.groupby('bank').n_answers.sum() == 13769).all()
weights = pd.read_csv(OUT / 'GROUP_WEIGHTS.csv')
signed = weights[weights.method == 'LR_GROUPS'].copy()
signed['negative'] = signed.weight < -1e-8
sign_counts = signed.groupby(['bank', 'cell']).agg(n_coefficients=('weight', 'size'), n_negative=('negative', 'sum')).reset_index()
sign_counts.to_csv(OUT / 'SIGN_COUNTS.csv', index=False)
recovery = []
for bank in ['B13', 'B32']:
    for panel in ['PRMB', 'PB']:
        values = summary[(summary.bank == bank) & (summary.panel == panel)].set_index('method').value
        for reference in ['plain', 'MEAN_EQ']:
            for method in ['LR_GROUPS', 'LR_GROUPS_NONNEG', 'LR_GROUPS_PLUS_WITHIN', 'LR_RAW_GROUP_SPAN']:
                recovery.append(dict(bank=bank, panel=panel, reference=reference, method=method,
                                     gain=values[method] - values[reference], gap_to_full=values.LR_all_X - values[method],
                                     fraction_gap_recovered=(values[method] - values[reference]) / (values.LR_all_X - values[reference])))
pd.DataFrame(recovery).to_csv(OUT / 'GAP_RECOVERY.csv', index=False)
labels = {
    'plain': 'ממוצע כל הפיצ׳רים השורדים',
    'MEAN_EQ': 'ממוצע שווה בין הקבוצות',
    'MEAN_DS': 'מיזוג DS בין הקבוצות',
    'MEAN_SML': 'מיזוג SML בין הקבוצות',
    'MEAN_MOM': 'מיזוג MoM בין הקבוצות',
    'LR_GROUPS': 'LR על ציוני הקבוצות — משקלים חתומים',
    'LR_GROUPS_NONNEG': 'LR על ציוני הקבוצות — משקלים לא שליליים',
    'LR_GROUPS_PLUS_WITHIN': 'LR על ציוני הקבוצות וגם ניגודים פנימיים',
    'LR_RAW_GROUP_SPAN': 'LR במרחב ממוצעי הקבוצות לפני תקנון בתוך תשובה',
    'LR_all_X': 'LR על כל הפיצ׳רים',
    'LR_PC1_within_group_linear': 'ייחוס קודם: גורם משותף וכל הניגודים הפנימיים'}
tables = []
for panel in ['PRMB', 'PB']:
    rows = []
    for method, label in labels.items():
        row = {'שיטה': label}
        for bk in ['B13', 'B32']:
            value = summary[(summary.bank == bk) & (summary.panel == panel) & (summary.method == method)].value.iloc[0]
            row[bk] = f'{value:.6f}'
        rows.append(row)
    tables.append('<h2>' + ('PRMBench — AUC בתוך תשובה, 6,030 תשובות' if panel == 'PRMB' else 'ProcessBench — איתור שגיאה ראשונה, 4,442 תשובות שגויות') + '</h2>' + pd.DataFrame(rows).to_html(index=False, escape=True))
report = '''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">
<title>האם אפשר להסתפק בניגודים בין הקבוצות?</title>
<style>body{font-family:Arial,sans-serif;max-width:1120px;margin:40px auto;padding:0 20px;line-height:1.7;color:#192634}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:9px;border:1px solid #cad4de;text-align:right}th{background:#edf3f8}a{color:#17639b}</style>
<h1>ניגודים בין קבוצות לעומת מידע בתוך הקבוצות</h1>
<p>5 באוקטובר 2026. ניסוי אבחוני מפוקח על נתוני פיתוח. כל משקלי ה־LR נלמדו מתוויות בקיפולי האימון בלבד. אין כאן תוצאה של שיטה חסרת תוויות.</p>
<p>נבדקו B13 ו־B32 על כל תשעת התאים, 13,769 תשובות בכל בנק. הקבוצות, סינון DS, כיווני הפיצ׳רים וממוצעי הפיצ׳רים בתוך הקבוצה נשמרו מהצינור הקיים. ארבעת המיזוגים הקיימים ו־LR קיבלו בדיוק אותה מטריצה של ציוני קבוצות.</p>
<p>השאלה הראשונה: האם שינוי המשקלים בין הקבוצות מספיק? השנייה: האם נדרשים משקלים שליליים, או שגם כוונון חיובי בלבד מספיק? השלישית: כמה מוסיף מידע על ההבדלים בין הפיצ׳רים בתוך כל קבוצה?</p>
''' + ''.join(tables) + '''
<h2>הגדרות ופירוש</h2>
<p>ציון קבוצה בצינור הקיים הוא ממוצע חבריה, שאחריו תקנון בתוך כל תשובה. אותו ציון בדיוק נכנס להשוואה הראשית. LR עם סימנים חופשיים יכול לייצג ניגודים בין הקבוצות גם ללא הוספה מפורשת של כל ההפרשים. הוספת ממוצע וכל הפרשי הקבוצות בבסיס אורתונורמלי שיחזרה את אותו LR ונשמרה כבקרה אלגברית.</p>
<p>הזרוע עם משקלים לא שליליים ממזערת את אותה פונקציית הפסד עם אותה ענישת C=1; רק סימני המקדמים מוגבלים. לכן רווח של הזרוע החתומה עליה, אם קיים, קשור לאפשרות להשתמש במקדמים שליליים ולא רק ללמידה מפוקחת של משקלים חיוביים.</p>
<p>להחזרת המידע הפנימי נוספו כל ההפרשים בתוך הקבוצות, בבסיס ללא כפילויות. ציוני הקבוצות עצמם נשארו זהים. זהו ייצוג אחר מהפיצ׳רים המקוריים בגלל התקנון בתוך תשובה; אין לטעון לשקילות ל־LR המלא. בנפרד, ממוצעי קבוצות לפני התקנון יחד עם הניגודים הפנימיים משחזרים את מלוא המרחב המקורי, והבקרה הזו שיחזרה את ציוני LR מהניסוי הקודם.</p>
<p>LR משתמש באיזון מחלקות וב־C=1 ללא חיפוש פרמטרים. התקנון הגלובלי מותאם רק לאימון. קבוצות וסינון היסטוריים הותאמו ללא תוויות ברמת התא כולו — הגישה אינה התאמה מתשובה יחידה. ב־ProcessBench מדד האיתור מתייחס רק לתשובות שגויות וממוצע שווה של שמונה תאים; אינו מודד זיהוי תשובות תקינות או PRMScore.</p>
<p>רווחי הסמך הם חקרניים: 5,000 דגימות זוגיות לפי שאלת מקור, עם דגימה משותפת לתאי ProcessBench. אין תיקון לריבוי השוואות ואין אימון מחדש בכל דגימה. כל הדוגמאות הן נתוני פיתוח שכבר נחשפו.</p>
<h2>בדיקות ושחזור</h2>
<p>ארבע בדיקות חדשות עברו: אובדן מידע באיגום, אילוץ משקלים ותקינות פונקציית ההפסד, הפרדת קיפולים ושחזור המרחב המלא, ומדדים עם שוויונות. הושלמו 18 תאים ו־90 התאמות קיפולים. כל 234 שורות המדדים נבדקו גם באמצעות ספירת זוגות תקינים־שגויים, בנפרד ממימוש הדירוג המקורי. כל המקורות והקוד נשארו ללא שינוי לפי SHA256. הבדיקה הנפרדת בוצעה בידי אותו סוכן.</p>
<ul><li><a href="SUMMARY.csv">תוצאות</a></li><li><a href="PAIRED_CONTRASTS.csv">הפרשים ורווחי סמך</a></li><li><a href="GROUP_WEIGHTS.csv">משקלי הקבוצות וחבריהן</a></li><li><a href="SIGN_COUNTS.csv">ספירת משקלים שליליים</a></li><li><a href="GAP_RECOVERY.csv">פערים לעומת הייחוסים</a></li><li><a href="REVIEW.json">בדיקות</a></li><li><a href="../PROTOCOL.json">פרוטוקול</a></li></ul></html>'''
(OUT / 'REPORT_HE.html').write_text(report, encoding='utf8')
for target in re.findall(r'href="([^"]+)"', report):
    assert (OUT / target).resolve().exists(), target
assert 'בין קבוצות' in (OUT / 'REPORT_HE.html').read_text(encoding='utf8')
snapshot = OUT / 'implementation'
snapshot.mkdir(exist_ok=True)
for rel in ['spectral_utils/group_contrast_oracle.py', 'spectral_utils/contrast_span_oracle.py',
            'scripts/run_group_contrast_oracle.py', 'scripts/run_contrast_span_oracle.py',
            'scripts/report_group_contrast_oracle.py', 'tests/test_group_contrast_oracle.py']:
    dest = snapshot / Path(rel).name
    if dest.exists():
        assert dest.read_bytes() == (ROOT / rel).read_bytes()
    else:
        shutil.copyfile(ROOT / rel, dest)
(OUT / 'DELIVERY_MANIFEST.json').write_text(json.dumps({str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'DELIVERY_MANIFEST.json'}, indent=2), encoding='utf8')
print('REPORT PASS: 18 cells, 90 folds, 234 independent metric rows, 48 paired contrasts', flush=True)
print(summary[summary.method.isin(['MEAN_EQ', 'MEAN_SML', 'MEAN_MOM', 'LR_GROUPS', 'LR_GROUPS_NONNEG', 'LR_GROUPS_PLUS_WITHIN', 'LR_all_X'])].to_string(index=False), flush=True)
print(sign_counts.to_string(index=False), flush=True)
