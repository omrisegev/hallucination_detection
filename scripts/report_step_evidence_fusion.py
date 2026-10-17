"""Produce a standalone report from the reviewed fixed-bank experiment."""
from pathlib import Path
import sys,json,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_step_evidence_fusion import OUT,write,sha

LABELS={
 'innovation5':'בסיס innovation5',
 'tcn__real':'TCN — התיקון הקיים',
 'evidence__context':'חריגה מתחזית TCN בלבד — שחזור',
 'evidence__end':'תיקון מסוף הצעד בלבד',
 'evidence__sustained':'תיקון מחריגה רציפה בלבד',
 'evidence__equal_context_end':'ממוצע: הקשר + סוף הצעד',
 'evidence__equal_context_sustained':'ממוצע: הקשר + רציפות',
 'evidence__equal':'ממוצע של שלוש הראיות',
 'evidence__iu':'IU-PCR על שלוש הראיות',
 'evidence__equal_shuffled':'ממוצע עם סדר מעורבב',
 'evidence__iu_shuffled':'IU-PCR עם סדר מעורבב',
 'iu__ridge+tcn+noreset':'מוביל PB הקודם: IU של מנבאים',
 'equal__ridge+bocpd+noreset':'מוביל within הקודם: ממוצע מנבאים'}


def run():
    result=json.loads((OUT/'METRICS.json').read_text());audit=json.loads((OUT/'AUDIT.json').read_text())
    if audit['status']!='PASS':raise ValueError('Unreviewed results')
    m=result['metrics'];err=result['error_analysis']
    rows=[];lossrows=[];primary=[]
    for n,label in LABELS.items():
        a=m[n];rows.append(f'<tr><td>{label}</td><td>{100*a["pb_all8"]:.4f}%</td><td>{a["prm_within"]:.6f}</td><td>{a["prmscore_q08"]:.6f}</td></tr>')
        if n in err:
            e=err[n];d=e['tcn__real']
            lossrows.append(f'<tr><td>{label}</td><td>{d["gained"]}</td><td>{d["lost"]}</td><td>{d["net"]:+d}</td><td>{e["recovered_common885_raw"]}</td><td>{e["recovered_common707_final"]}</td></tr>')
    for name,c in result['contrasts'].items():
        if not c['primary']:continue
        a,b=name.split('_minus_');lo,hi=c['pb_ci'];wl,wh=c['prm_within_ci']
        primary.append(f'<tr><td>{LABELS[a]} לעומת {LABELS[b]}</td><td>{100*c["pb_delta"]:+.4f} [{100*lo:+.4f}, {100*hi:+.4f}]</td><td>{c["prm_within_delta_common"]:+.6f} [{wl:+.6f}, {wh:+.6f}]</td></tr>')
    d=result['mechanism']['real']
    doc='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>שילוב ראיות לצעד — Step392</title><style>body{font:17px/1.7 system-ui;background:#f2f5f8;color:#203344;margin:0}main{max-width:1150px;margin:25px auto;padding:32px;background:white;border-radius:14px}table{width:100%;border-collapse:collapse;font-size:15px}td,th{border-bottom:1px solid #dde4eb;padding:9px;text-align:right}th{background:#eaf0f5}a{color:#087d91}code,.num{direction:ltr;unicode-bidi:embed}svg{max-width:100%}</style><main>
<h1>שילוב חריגה מהתחזית, רציפות וסוף הצעד</h1>
<p>ניסוי פיתוח מלא: 13,769 תשובות, 145,597 צעדים. בנק אחד קבוע; ללא חיפוש צירופים, אימון מנבאים או שינוי gate. כל המדדים חושבו מחדש ונבדקו מול חישוב נפרד של PB ו־within. הניסוי אינו אישור על אוכלוסייה חדשה.</p>
<p><strong>הכרעה: שילוב שלוש הראיות אינו מועמד להחלפת TCN.</strong> הממוצע מוריד PB מ־40.9718% ל־39.6301%, ו־IU-PCR ל־39.0007%. הירידות מול TCN מובחנות גם ברווחי הסמך הראשיים של 99.5%. IU מוריד גם within לעומת TCN ולעומת הממוצע. מנגד, הסדר האמיתי עדיף על ערבוב בסיכומי הצורה בדירוג בתוך תשובה, ולכן תוצאה שלילית זו אינה אומרת שאין בהם מידע כרונולוגי.</p>
<p>פשרת פיתוח מעניינת נמצאה בהשוואה המשנית שנרשמה מראש: הקשר + סוף הצעד בממוצע נותן within של 0.765898 ו־PRMScore של 0.644000, עם PB של 39.5281%. מול TCN, הפרש within הוא ‎+0.004306, רווח סמך תיאורי 95% ‎[+0.002627,+0.005931]; הפרש PB הוא ‎−1.4437 נקודות, רווח ‎[−2.3416,−0.5778]. זו אינה הכרעה ראשית מתוקנת לריבוי השוואות, ואין כאן מנצח בכל המדדים.</p>
<p>הניסיון הנוכחי לא פתר את ההחמצות המשותפות: הממוצע של השלושה לא הציל אף אחת מ־707 ההחמצות עם gate פתוח; IU הציל אחת. last4 בסיכום ישיר שבדק Claude ותיקון last4 בעוצמה .25 לציון הבסיס הם אלגוריתמים שונים. הניסוי מראה שהשילוב המרוסן המסוים אינו מעביר את כל פוטנציאל האיתור של last4 לבסיס; הוא אינו סותר את 119 הפגיעות באבחון הקודם.</p>
<h2>מה מיזגנו</h2><p>השארנו פעם אחת את ציון innovation5 הקיים: Top10 נפרד לכל זרם ואז ממוצע. התיקון משלב שלוש ראיות לאותו צעד: שארית TCN חתומה בסיכום Top10; ממוצע ארבעת הטוקנים האחרונים; והחלון הרציף בן עשרה טוקנים בעל הממוצע הגבוה ביותר, בנפרד לכל זרם. אורכי החלונות מתקצרים בצעדים קצרים.</p>
<p>כל ראיה מתוקננת על פני צעדי התשובה. IU-PCR נאמד ממטריצת covariance ממורכזת של שלוש הראיות, עם שני רכיבים ועקומת g2 הקנונית. לאחר השילוב, התיקון מתוקנן ומוכפל ב־0.25 מסטיית התקן של ציוני הבסיס. לכן הבסיס נשמר פעם אחת, וכל החלופות נבדקות באותה עוצמת תיקון. אין נרמול של השונות הכוללת של הציון הסופי.</p>
<p>זהו משקול של ראיות ברמת הצעד, לא משקלי פיצ׳רים המשתנים בכל טוקן. המנבא משתמש בהיסטוריית טוקנים החוצה צעדים; סיכומי הצורה נשארים בתוך הצעד. כל התהליך offline: TCN אומן בתשובות אחרות בהפרדת קבוצות; המשקול והנרמול מקומיים לתשובה; gate של tail15 באחוזון .33 נשאר טרנסדוקטיבי. כיול PRMScore נעשה עם עשר התאמות TCN המחריגות זוגות folds, בנוסף לחמש ההתאמות הראשיות.</p>
<h2>התוצאות</h2><table><tr><th>שיטה</th><th>PB מאקרו F1</th><th>within-AUC</th><th>PRMScore</th></tr>'''+''.join(rows)+'''</table>
<h2>האם המיזוג מוסיף?</h2><p>חמש השוואות ראשיות, שני מדדים: 10,000 דגימות bootstrap מזווגות של קבוצות מקור, ורווחי סמך של 99.5%. הפרש PB בנקודות אחוז. בחירה קודמת של הבנק והתיקון, וההשראה ל־last4 מתוך ההחמצות, השתמשו בתוצאות פיתוח על אותה אוכלוסייה.</p><table><tr><th>השוואה</th><th>הפרש PB ורווח סמך</th><th>הפרש within ורווח סמך</th></tr>'''+''.join(primary)+'''</table>
<h2>שגיאות שנוספו ואבדו</h2><p>השוואה ל־TCN הקיים על 4,442 תשובות PB השגויות. מספר פגיעות אינו PB F1: המדד הראשי מאזן תשובות נקיות ושגויות וממצע תאים. עמודת 885 היא איתור לפני gate בקבוצת ההחמצות הקודמת; עמודת 707 היא איתור סופי בקבוצה שה־gate שלה פתוח. אלו קבוצות אבחון שנבחרו בעבר לפי תוצאות, לא אוכלוסיית איכות עצמאית.</p>
<table><tr><th>שיטה</th><th>נוספו</th><th>אבדו</th><th>נטו</th><th>אותרו מתוך 885 לפני gate</th><th>אותרו מתוך 707 אחרי gate</th></tr>'''+''.join(lossrows)+'''</table>
<h2>האם IU אכן נאמד?</h2>'''+f'''<p>IU תקף מספרית ב־{d["native_answers"]:,} תשובות; מעבר מפורש לממוצע ב־{d["fallback_answers"]:,}. הסיבות: <code>{html.escape(json.dumps(d["fallback_reasons"]))}</code>. חציון המשקלים, בסדר הקשר / סוף / רציפות: <code>{npfmt(d["median_weights"])}</code>. מקדם שלילי מופיע ב־{100*d["negative_weight_fraction"]:.2f}% מההתאמות התקפות; {100*d["spearman_filter_would_reject_fraction"]:.2f}% מכילות זוג ראיות עם מתאם Spearman מוחלט של לפחות .75.</p>
<p>התלות בין הראיות ומספר הצעדים הקטן מגבילים את פירוש המשקלים כאמינות. בשלושה פיצ׳רים יש שלוש משוואות אדיטיביות לשלושה פרמטרים; שארית אפס אינה הוכחת ההנחות. התוצאות המלאות כוללות את מדיניות המעבר לממוצע, ולא מסתירות את הכיסוי.</p>
<p>בקרת הערבוב משנה רק את שני סיכומי הצורה, באמצעות תמורה משותפת לחמשת הזרמים בתוך כל צעד. ראיית TCN נשמרת. לכן זו בקרה לסדר הטוקנים בסיכומי הצורה, לא מבחן חדש לסדר בהקשר של TCN.</p>
<h2>היסטוריה ותוקף</h2><p>Top10, ממוצע רציף ומשקול סטטיסטיקות סדר כבר נבדקו בעבר. הניסוי הנוכחי אינו מציג אותם כחידוש: השאלה היא האם שילוב הסיכומים עם ראיית ההקשר, כתיקון לבסיס, מועיל. אין להסיק מכישלון השילוב הזה שכל fusion או כל שימוש בזמן נכשל. אין לקבע מועמד רק משום שהציל חלק מה־885.</p>
<p><a href="../../docs/experiments/STEP_EVIDENCE_FUSION_20260915.md">הפרוטוקול שנרשם מראש</a> · <a href="METRICS.csv">כל 27 השורות</a> · <a href="METRICS.json">תאים, השוואות ומשקלים</a> · <a href="PB_ERROR_LEDGER.csv">כל מזהי השגיאות וההחלטות</a> · <a href="AUDIT.json">בדיקות ושחזור</a></p></main></html>'''
    (OUT/'REPORT.html').write_text(doc,encoding='utf8')
    write(OUT/'REPORT_AUDIT.json',dict(status='PASS',report_sha256=sha(OUT/'REPORT.html'),
        tables_displayed=len(rows),primary_comparisons=len(primary),error_rows=len(lossrows),
        numerical_sources=['METRICS.json','AUDIT.json'],report_code_sha256=sha(Path(__file__))))
    print('Report written',OUT/'REPORT.html')


def npfmt(x):
    return '['+', '.join(f'{v:.4f}' for v in x)+']'


if __name__=='__main__':run()
