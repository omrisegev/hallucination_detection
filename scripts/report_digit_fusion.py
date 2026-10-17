"""Readable result and mechanism review for the frozen digit-fusion experiment."""
from pathlib import Path
import sys,json,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digit_fusion import OUT,write,sha

LABELS={'innovation5':'בסיס innovation5','tcn__real':'TCN הקיים',
 'digit_standalone':'אי־הסכמה על ספרות בלבד','digit025':'בסיס + ספרות, γ=.25',
 'digit1':'בסיס + ספרות, γ=1 — שחזור משני','presence':'בקרת נוכחות ספרות',
 'rate':'בקרת שיעור אי־הסכמה מתוך הספרות','permuted':'בקרת מיקומי אי־הסכמה מעורבבים',
 'tcn_digit_sum':'TCN + תיקון ספרות נוסף','tcn_digit_matched':'TCN + ספרות, עוצמת תיקון תואמת',
 'bank5_equal':'בנק 5, מיצוע לאחר תקנון','bank5_iu':'בנק 5, IU-PCR',
 'bank6_equal':'בנק 6 עם ספרות, מיצוע לאחר תקנון','bank6_iu':'בנק 6 עם ספרות, IU-PCR',
 'iu__ridge+tcn+noreset':'מוביל PB הקודם — IU על מנבאים',
 'equal__ridge+bocpd+noreset':'מוביל within הקודם — ממוצע מנבאים'}


def run():
    result=json.loads((OUT/'METRICS.json').read_text());a=json.loads((OUT/'AUDIT.json').read_text())
    assert a['status']=='PASS'
    m=result['metrics'];er=result['error_analysis'];mech=result['mechanism']
    rows=[];errors=[];comparisons=[]
    for n,label in LABELS.items():
        v=m[n];rows.append(f'<tr><td>{label}</td><td>{100*v["pb_all8"]:.4f}%</td><td>{v["prm_within"]:.6f}</td><td>{v["prmscore_q08"]:.6f}</td></tr>')
        if n in er:
            e=er[n];b=e['innovation5'];t=e['tcn__real']
            errors.append(f'<tr><td>{label}</td><td>{e["hits"]}</td><td>+{b["gained"]}/−{b["lost"]}</td><td>+{t["gained"]}/−{t["lost"]}</td><td>{e["common885_raw"]}</td><td>{e["common707_final"]}</td><td>{e["common707_positive_final"]}</td></tr>')
    for name,c in result['contrasts'].items():
        x,y=name.split('_minus_');level=100*c['ci_level'];lo,hi=c['pb_ci'];wl,wh=c['prm_within_ci']
        comparisons.append(f'<tr><td>{LABELS[x]} מול {LABELS[y]}</td><td>{level:.4f}%'+(' ראשי' if c['primary'] else ' תיאורי')+f'</td><td>{100*c["pb_delta"]:+.4f} [{100*lo:+.4f},{100*hi:+.4f}]</td><td>{c["prm_within_delta_common"]:+.6f} [{wl:+.6f},{wh:+.6f}]</td></tr>')
    mechanisms=[]
    for name in ('all_answers','variable_digit_answers','pb_error_answers'):
        v=mech[name];mechanisms.append(f'<tr><td>{name}</td><td>{v["answers"]}</td><td>{v["participation5"]:.4f}</td><td>{v["participation6"]:.4f}</td></tr>')
    strata=''.join(f'<tr><td>{k}</td><td>{v["answers"]}</td><td>{v["gained"]}</td><td>{v["lost"]}</td></tr>' for k,v in result['strata'].items())
    doc='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>אי־הסכמה על ספרות ו־Fusion — Step393</title><style>body{font:17px/1.7 system-ui;background:#f1f5f9;color:#203344;margin:0}main{max-width:1250px;margin:25px auto;padding:30px;background:white;border-radius:14px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f7}a{color:#08768d}code{direction:ltr;unicode-bidi:embed}</style><main><h1>אי־הסכמה על ספרות: שחזור עצמאי וניסוי Fusion</h1>
<p>כל 13,769 התשובות ו־145,597 הצעדים נכללו בהשוואה. לא הורץ מעבר מודל חדש. זרם הספרות, שני הציונים של Claude ושלוש שורות המדדים שלו שוחזרו; 18 שורות ייחוס נשארו זהות. כל 30 חבילות המדדים נבדקו בחישוב עצמאי של PB ו־within.</p>
<p><strong>התוצאה: אי־הסכמה על ספרות מועילה לשילוב, אבל IU-PCR הקנוני אינו מנצל אותה נכון בבנק שנבדק.</strong> תיקון הספרות של Claude שוחזר: 41.3300% PB ו־0.776036 within. הוספתו לציון TCN נותנת 42.0781% ו־0.774945, עם PRMScore של 0.652284. מול TCN זו תוספת של 255 פגיעות ואובדן 204: נטו 51.</p>
<p>הזרוע של 42.0781% היא השוואה משנית שנקבעה מראש. מול TCN, רווח סמך תיאורי 95% להפרש PB הוא ‎[+0.127,+2.104] נקודות, ול־within ‎[+0.011450,+0.015226]. אין להציג זאת כאישור ראשי מתוקן או כאישור חיצוני. בבקרת עוצמת התיקון מתקבלים 41.4627% ו־0.772607: רווח within נשאר ברור בהשוואה הראשית, אך יתרון PB אינו מוכרע. גם ההפרש של שחזור Claude מול הבסיס כולל אפס ב־PB תחת התיקון הרחב יותר של המחקר הנוכחי, למרות רווחו החיובי ברווח ההיסטורי של 97.5%.</p>
<p>אי־ההסכמה האמיתית עדיפה על העברתה למיקומי ספרות אחרים באותה תשובה בשני המדדים, גם ברווחים הראשיים המתוקנים. היא עדיפה על עצם נוכחות הספרות ב־within. בקרת שיעור אי־הסכמה מגיעה ל־41.1532% ו־0.774964. לכן יש מידע מעבר לצפיפות הספרות לבדה, אך אין כאן טענה שכל השפעת אורך הוסרה.</p>
<p>בבנק בן שישה פיצ׳רים, מיצוע מתוקנן נותן 41.3961% ו־0.773468, ואילו IU יורד ל־30.5284% ו־0.637115. באבחון נפרד של 30 תשובות עם ספרות משתנות המשקלים תואמים למימוש U-PCR הקנוני. מקדם הספרות שלילי ב־99.98% מהתשובות שבהן הוא משתנה, בחציון ‎−0.2798. זו תוצאה התואמת דיכוי של ראיית אי־ההסכמה, לא התכנסות לממוצע. היא מפריכה את ההבטחה שמקור מידע פחות מתואם יגרום למיזוג הספקטרלי להצליח מעצמו.</p>
<h2>מה השתנה באלגוריתם</h2><p>בטוקן שבו הן הטקסט שסופק והן top1 של המודל הם ספרה יחידה, אי־הסכמה ביניהם מקבלת 1; אחרת 0. מזהי 0–9 נבדקו מול טוקנייזרי Qwen3-4B/8B השמורים, ולא הונחו רק בגלל המספרים בקוד Claude. זו השוואת תוכן הטקסט להעדפת המודל באותו הקשר. הבדיקה אינה משווה מספרים שלמים, ואי־הסכמה אינה מוכיחה טעות מתמטית.</p>
<p>שחזור השילוב: Top10 של זרם הספרות בכל צעד, תקנון של הסיכום בתוך התשובה, ותיקון בעוצמה .25 מסטיית התקן של בסיס innovation5. γ=1 נשמר כשחזור משני בלבד. נבדקו גם נוכחות ספרות, שיעור אי־הסכמה מתוך מספר הספרות, וערבוב אירועי אי־ההסכמה רק בין מיקומי ספרות באותה תשובה.</p>
<p>שילוב עם TCN נבדק בשתי גרסאות: הוספת תיקון הספרות לציון TCN הקיים; ותקנון סכום שתי הראיות כך שעוצמת התיקון תישאר .25. כך ניתן לבדוק אם התוספת מגיעה מכיוון התיקון או מהגדלת עוצמתו.</p>
<p>בדיקת IU נפרדת: אותו בנק בן חמשת זרמי הטוקנים, עם ובלי ספרות. תקנון על כל טוקני התשובה, covariance ממורכז, IU קנוני עם שני רכיבים, ואז Top10 לכל זרם וסכום משוקלל. אותו סדר גם בממוצע. הציון הסופי מוחזר לממוצע ולסטיית התקן של הבסיס. זו החלפת משקול הבנק, לא אותו תיקון .25. עמודה קבועה אינה נספרת כתצוגה חיה; היא מוסרת מהאמידה, ומשקלה אפס.</p>
<h2>התוצאות</h2><table><tr><th>שיטה</th><th>PB מאקרו F1</th><th>within-AUC</th><th>PRMScore</th></tr>'''+''.join(rows)+'''</table>
<h2>הפרשים מזווגים</h2><p>10,000 דגימות bootstrap של קבוצות המקור. שש השוואות ראשיות כפול שני מדדים: רווחי סמך 99.5833%. שאר ההשוואות תיאוריות ברמת 95%. השחזור אינו ניסוי בלתי תלוי: Claude סינן שבע תצוגות באמצעות קבוצות שגיאה מתויגות, והבנק כבר פותח על אוכלוסייה זו. רווחי הסמך אינם מתקנים את כל תהליך הבחירה ההיסטורי.</p><table><tr><th>השוואה</th><th>רמת רווח סמך</th><th>Δ PB בנקודות אחוז</th><th>Δ within</th></tr>'''+''.join(comparisons)+'''</table>
<h2>אילו שגיאות נוספו ואבדו?</h2><p>פגיעות סופיות מתוך 4,442 תשובות PB השגויות. קבוצות 885 ו־707 נשארו כפי שהוגדרו לפני ניסוי הספרות. ספירת 707 עם אות חיובי מפרידה פגיעה שבה יש אי־הסכמה בצעד הנכון מפגיעה שאפשר לקבל משוויון בין ציוני אפס. זהו אבחון של קבוצות שנבחרו לפי תוצאות עבר.</p><table><tr><th>שיטה</th><th>פגיעות</th><th>נוספו/אבדו מול הבסיס</th><th>מול TCN</th><th>885 לפני gate</th><th>707 אחרי gate</th><th>707 ואות ספרות חיובי</th></tr>'''+''.join(errors)+'''</table>
<h2>בקרת אורך והזדמנויות</h2><p>בנוסף לבקרות האיכות בטבלה, אלה תוספות ואובדנים של γ=.25 מול innovation5 לפי אורך הצעד השגוי ומספר הספרות בו. אלו חתכים תיאוריים חופפים בין שתי החלוקות, לא מבחן סיבתי או תיקון לכל השפעת אורך אפשרית.</p><table><tr><th>חתך</th><th>תשובות</th><th>נוספו</th><th>אבדו</th></tr>'''+strata+'''</table>
<h2>מה באמת אומרת אי־התלות?</h2><p>המספר 3.55 בדוח Claude התייחס להוספת כל שבע התצוגות החדשות. הטבלה הבאה מחשבת במפורש את אותו בנק עם תוספת ספרות אחת בלבד, מתוך ממוצע covariance מתוקנן בתוך תשובה. יחס השתתפות הוא ממד אפקטיבי לפי שונות, לא מספר מומחים בלתי תלויים ולא מבחן איכות.</p><table><tr><th>אוכלוסייה</th><th>תשובות</th><th>בנק 5</th><th>בנק 5 + ספרות</th></tr>'''+''.join(mechanisms)+'''</table>'''+f'''<p>ב־{mech["constant_digit_answers"]:,} תשובות זרם הספרות קבוע. חציון משקלי bank6 IU: <code>{html.escape(json.dumps(mech["bank6"]["median_weights"]))}</code>. חציון מקדם הספרות בתשובות שבהן הוא משתנה: <code>{mech["bank6"]["last_weight_median_if_digit_present"]:.6f}</code>. מקדם שלילי מופיע באחד הזרמים ב־{100*mech["bank6"]["negative_weight_fraction"]:.2f}% מהתשובות. התאמת IU תקפה מספרית ב־{mech["bank6"]["native_answers"]:,} תשובות.</p>'''+'''
<p>כמה פונקציות של אותו וקטור q15 אינן בהכרח אותו מספר או אותה אינפורמציה. מתאם גבוה אינו שולל השלמה בזנבות; מתאם נמוך אינו מוכיח אמינות. גם ρ אחיד אינו מחייב ממוצע אם covariance אינו סימטרי בין המומחים. לכן המבחן למיזוג מוצלח הוא התוספת לאיתור מול רכיביו, ולא משקלים שנראים שונים או עלייה ביחס ההשתתפות.</p>
<h2>חוזה גישה ומה לא הוסק</h2><p>המנבא והבנק נבדקים על תשובות נתונות ב־teacher forcing. בדיקת הקוד המפיק מאשרת הסטת logits של טוקן אחד; נבדקו מזהי טוקנים, סדר top-k וגבולות צעדים. אין כאן אימות חדש באמצעות forward pass. בתשובה שנוצרה ב־greedy decoding על ידי אותו מודל, top1 והספרה שנבחרה אמורים להתאים; לכן אין להעביר את הפרשנות אוטומטית ליצירה עצמית.</p>
<p>ה־gate נשאר tail15 באחוזון .33 בכל תא, טרנסדוקטיבי. כיול PRMScore משתמש בקבוצות אחרות ללא תוויות; בזרועות TCN משתמשים בציונים מעשר התאמות המחריגות זוגות folds כדי למנוע חשיפה של מודלי הכיול לקבוצת ההערכה. לא אומן gate חדש, לא נוספו אופרטורים ולא חודש תור FM/DiFlo. לפני טענת הכללה נדרש אישור על שאלות שלא שימשו לפיתוח.</p>
<p><a href="../../docs/experiments/DIGIT_FUSION_20260915.md">פרוטוקול</a> · <a href="METRICS.csv">כל 30 השורות</a> · <a href="METRICS.json">תאים, אבחונים ורווחי סמך</a> · <a href="PB_ERROR_LEDGER.csv">מזהי השגיאות והחלטות כל השיטות</a> · <a href="AUDIT.json">אימות ושחזור</a> · <a href="IU_DIAGNOSTIC_AUDIT.json">אבחון מקדם הספרות</a></p></main></html>'''
    (OUT/'REPORT.html').write_text(doc,encoding='utf8')
    write(OUT/'REPORT_AUDIT.json',dict(status='GENERATED_PENDING_LINK_AND_NUMERIC_REVIEW',
        displayed_rows=len(rows),report_sha256=sha(OUT/'REPORT.html'),source_sha256=sha(Path(__file__))))
    print(OUT/'REPORT.html')


if __name__=='__main__':run()
