"""Render the completed predictor-only experiment without choosing new arms."""
from pathlib import Path
import sys,json,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_aligned_context_predictors import OUT,METHODS,sha,write


def run():
    payload=json.loads((OUT/'METRICS.json').read_text());m=payload['metrics'];p=payload['prediction']
    audit=json.loads((OUT/'AUDIT.json').read_text());state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status']!='COMPLETE_REVIEWED' or audit['status']!='PASS':raise ValueError('review incomplete')
    def table(headers,rows):
        return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
    names=['innovation5',*METHODS,'original4']
    lines=['# מנבאי הקשר לפני Fusion — Step387',
        'השוואת פיתוח מלאה: 13,769 תשובות, 145,597 צעדים, 6,968,779 טוקנים. לא נלמד fusion בין מנבאים.',
        'הממצא: כדאי לשמור את BOCPD כמנבא חלופי לצד Ridge. התיקון שלו משפר within מול innovation5 ברווח הסמך הראשי המתוקן: +.002930, CI99.5% [.000140,.005692]. Ridge שומר על ה-PB הגבוה ביותר, ו-BOCPD על within הגבוה ביותר בנקודות. אין יתרון מובהק של אחד על השני בהשוואה הראשית.',
        'הסייג המנגנוני: גם noreset מגיע ל-within .762839, קרוב ל-BOCPD .763223; ההפרשים מול noreset אינם מובהקים. לכן אין לייחס עדיין את הרווח להחלפת משטר. MSE של Ridge נמוך מזה של BOCPD (.688936 מול .818527), בעוד noreset עם MSE1.015762 עדיין נותן דירוג טוב: דיוק החיזוי אינו מסביר לבדו את איכות התיקון.',
        'BOCPD מוסיף115 פגיעות PB שבהן Ridge מפספס, ומאבד139 שבהן Ridge פוגע. זו השלמה חלקית בפסגות, לא כלל בחירה זמין ללא תוויות ולא הוכחה ש-U-PCR ישפר. ההחלטה: לשמור את Ridge ואת BOCPD, לשמור noreset כבקרה מחייבת, ולא להתאים עדיין fusion. mean16 לא הראה יתרון ראשי.',
        '## מה נבדק',
        'כל מנבא חוזה את אותו וקטור innovation5 המתוקנן לפני קריאת הטוקן הנוכחי. שארית חתומה ממוצעת על חמשת הפיצ׳רים, Top10 של השארית בכל צעד, ואז תיקון במשקל .25 לציון innovation5 המקורי. התיקון ממורכז ומתוקנן בין צעדי אותה תשובה ומוכפל בסטיית התקן של ציון הבסיס. אותו gate של tail15 באחוזון .33; אין שינוי בנק, TopK או סף.',
        'Ridge הוא המנבא החיצוני הקיים, 16 השהיות ומיקום יחסי, בקבוצות מקור נפרדות. mean16 חוזה באמצעות ממוצע העבר הזמין עד 16 טוקנים. BOCPD הוא חמישה מסנני ממוצע גאוסיים נפרדים, עם הסתברות החלפת משטר 1/32 ושונויות תצפית ו-prior קבועות 1. הוא משקלל את תחזית האיפוס והמשך המשטר לפני קריאת הטוקן. noreset הוא אותו מודל ללא החלפות, sum(past)/(t+1). zero הוא תחזית אפס: בקרה לתרומת התצפית הנוכחית עצמה.',
        'כל המסלולים משתמשים בנרמול ובסימנים של תשובה שלמה, ולכן הם offline. mean16/BOCPD/noreset אינם לומדים פרמטרים מתשובות אחרות. zero אינו מסלול נטול היסטוריה מקצה לקצה: בנק innovation5 כבר כולל היסטוריה.',
        '## איכות איתור — יותר גבוה עדיף',
        table(['שיטה','PB F1 %','within-AUC','PRMScore'],[[n,f"{100*m[n]['pb_all8']:.4f}",f"{m[n]['prm_within']:.6f}",f"{m[n]['prmscore_q08']:.6f}"] for n in names]),
        '## השוואות ראשיות',
        '10,000 דגימות bootstrap מזווגות לפי קבוצות מקור. חמישה זוגות ושני מדדים: רווחי סמך 99.5% בתיקון Bonferroni. רווח הכולל אפס אינו מוכיח שקילות. PRMScore משני: כיול .8 בקבוצות אחרות; Ridge שומר את הכיול המקונן המקורי.',
        table(['השוואה','הפרש PB בנקודות אחוז','CI PB','הפרש within','CI within'],
            [[k,f"{100*c['pb_delta']:+.4f}",str([round(100*x,4) for x in c['pb_ci']]),
              f"{c['prm_within_delta_common']:+.6f}",str([round(x,6) for x in c['prm_within_ci']])]
             for k,c in payload['contrasts'].items() if c['primary']]),
        '## דיוק חיזוי — MSE נמוך יותר עדיף',
        'ממוצע על חמשת הפיצ׳רים המתוקננים, עם משקל שווה לכל תשובה. כל הטוקנים כלולים; הפירוט אחרי 16 טוקנים ולפי פיצ׳ר שמור ב-METRICS.json. זהו דיוק חיזוי של טלמטריה, לא דיוק חיזוי נכונות.',
        table(['מנבא','MSE','CI משני 95% להפרש MSE מול Ridge'],[[n,f"{p[n]['scalar_mse_answer_mean']:.6f}",str([round(x,6) for x in p[n]['mse_minus_ridge_secondary95_ci']])] for n in METHODS]),
        '## האם השאריות מביאות מידע שונה?',
        'הטבלה מציגה חציון מתאמים בתוך תשובה מול Ridge. הסרת התצפית המשותפת היא רגרסיה לינארית אבחונית מתוך התשובה; היא אינה ציון חדש, אינה מסירה כל תלות משותפת ואינה מוכיחה את הנחות U-PCR. מתאם חסר בזרם קבוע נשאר null.',
        table(['מנבא','מתאם תחזיות','מתאם שאריות','מתאם לאחר הסרת התצפית המשותפת'],
            [[n,*[round(p[n]['correlations'][k]['quantiles'][1],6) if p[n]['correlations'][k]['quantiles'] else 'null' for k in
              ('prediction_correlation_ridge','signed_residual_correlation_ridge','residual_correlation_after_removing_current')]] for n in METHODS]),
        '## פגיעות שנוספו ואבדו מול Ridge',
        'תשובות PB עם שגיאה, אחרי אותו gate. אלו שינויים בפסגות קיימות, ולא הבטחה ליכולת fusion. פירוק מוקדם/אמצע/מאוחר נמצא ב-METRICS.json.',
        table(['מנבא','פגיעות משותפות','נוספו','אבדו'],[[n,*[payload['peak_transitions'][n]['ridge']['all_errors'][k] for k in ('both','gained','lost')]] for n in METHODS]),
        '## גבולות המסקנה והמשך',
        'חזית הנקודות על שני המדדים: '+', '.join(payload['pareto'])+'. יתרון נקודתי או הפסד אינם מחליפים את ההשוואות המזווגות. הבנק innovation5, מינון .25 ובחירת שארית חתומה מבוססים על פיתוח קודם באותה אוכלוסייה. גם זרוע חדשה על כל התשובות אינה אישור על שאלות חדשות.',
        'הניסוי הזה אינו אימון TCN/FM/DiFlo ואינו השוואת איכות מלאה שלהם. התור העצבי נשאר מושהה ללא שינוי. אין להסיק מתוצאה כאן שמשפחת המנבאים מוצתה. לפני fusion נדרש להחליט אם איכות התיקון וההשלמה בין המנבאים מצדיקות אותו, ולא רק דיוק חיזוי או מתאם שונה.',
        '## אימות ושחזור',
        f"13 בדיקות עברו: 4 בדיקות חיזוי חדשות ו-9 בדיקות הקשר וכיול קיימות. אימות מדדים עצמאי לכל {len(m)} השיטות; כל תשע כותרות הייחוס משוחזרות. חישוב scalar/מיון נפרד לכל תשובה ולכל מנבא, פער מרבי {audit['scoring']['max_readout_delta']:.3g}; פער Ridge מול הארכיון {audit['scoring']['max_ridge_score_delta']:.3g}. התאמת כל חתימות הבנק והמקורות. ללא כשל שקט או השלמת ציונים בממוצע.",
        f"זמן ניקוד {state['scoring_seconds']:.1f} שניות; הערכה ואבחון {state['evaluation_seconds']:.1f} שניות. חישוב CPU עם BLAS thread אחד.",
        '[הפרוטוקול](../../docs/experiments/ALIGNED_CONTEXT_PREDICTORS_20260915.md) · [המדדים והאבחונים](METRICS.json) · [אימות](AUDIT.json) · [מצב](RUN_STATE.json)',
        'מקור רעיון BOCPD: Adams & MacKay, https://arxiv.org/abs/0710.3742. תחזית reset-before-observation והשימוש בה לתיקון ציוני reasoning הם ההתאמה שלנו.']
    md='\n\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(md,encoding='utf8')
    body=[]
    for part in lines:
        if part.startswith('|'):
            rows=part.strip().splitlines();head=rows[0];data=rows[2:]
            def tr(row,tag):return '<tr>'+''.join('<'+tag+'>'+html.escape(c.strip())+'</'+tag+'>' for c in row.strip('|').split('|'))+'</tr>'
            body.append('<div class="scroll"><table>'+tr(head,'th')+''.join(tr(r,'td') for r in data)+'</table></div>')
        elif part.startswith('# '):body.append('<h1>'+html.escape(part[2:])+'</h1>')
        elif part.startswith('## '):body.append('<h2>'+html.escape(part[3:])+'</h2>')
        else:body.append('<p>'+html.escape(part)+'</p>')
    page='<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>Step387 — מנבאים לפני fusion</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 22px;line-height:1.7;background:#f6f8fb;color:#182431}table{border-collapse:collapse;background:white;width:100%;font-size:15px}th,td{border:1px solid #d7e0e8;padding:9px;text-align:right}th{background:#e3edf5}.scroll{overflow:auto}h1,h2{color:#13415a}a{color:#176399}</style>'+''.join(body)+'<p><a href="METRICS.json">METRICS.json</a> · <a href="AUDIT.json">AUDIT.json</a> · <a href="../../docs/experiments/ALIGNED_CONTEXT_PREDICTORS_20260915.md">פרוטוקול</a></p></html>'
    (OUT/'REPORT.html').write_text(page,encoding='utf8')
    (OUT/'REPRODUCE.md').write_text('From the temporal research worktree, with the existing hash-verified data and frozen Ridge artifacts:\n\n'
        '```powershell\npython -B -X utf8 scripts/run_aligned_context_predictors.py\npython -B -X utf8 scripts/evaluate_aligned_context_predictors.py\npython -B -X utf8 scripts/report_aligned_context_predictors.py\n```\n\n'
        'The scoring manifest rejects changed source code or inputs. Each answer is checkpointed in ANSWERS.sqlite. '
        'The runner never loads correctness labels; evaluation is separate. Large arrays/SQLite/diagnostics remain local; '
        'their hashes are published in ARTIFACTS.json. The same source/data bundle is required for replay.\n',encoding='utf8')
    paths=[p for p in OUT.iterdir() if p.is_file() and p.name!='ARTIFACTS.json']
    write(OUT/'ARTIFACTS.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}))


if __name__=='__main__':run()
