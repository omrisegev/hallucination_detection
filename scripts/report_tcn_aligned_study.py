"""Render the frozen seed0 TCN predictor study and its audit trail."""
from pathlib import Path
import sys,json,html,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_tcn_aligned_study import OUT,write,sha


def run():
    state=json.loads((OUT/'RUN_STATE.json').read_text());audit=json.loads((OUT/'AUDIT.json').read_text())
    if state['status']!='COMPLETE_REVIEWED' or audit['status']!='PASS':raise ValueError('Full review required')
    payload=json.loads((OUT/'METRICS.json').read_text());metrics=payload['metrics'];pred=payload['prediction']
    interpretation=json.loads((OUT/'REVIEW.json').read_text()) if (OUT/'REVIEW.json').exists() else {}
    job_cost=[]
    for key,fit in audit['fits'].items():
        path=ROOT/'results/temporal_context_models_v1'/key
        training=json.loads((path/'TRAINING.json').read_text())
        scoring=json.loads((path/'scoring/RUN_STATE.json').read_text())
        job_cost.append([key.rsplit('exclude',1)[1],fit['steps'],fit['best_step'],
            round(training[-1]['seconds'],1),round(scoring['seconds'],1)])
    def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
    parts=['# TCN כמנבא מיושר לפני Fusion — Step388',
        'כל13,769 התשובות /145,597 צעדים /6,968,779 טוקנים. seed0 בלבד. אין fusion בין מנבאים; תוצאות פיתוח.',
        *interpretation.get('paragraphs_he',[]),
        '## השיטה שנבדקה',
        'מנבא TCN חוזה את אותם חמשת הפיצ׳רים של innovation5, עם תוחלת ושונות לכל פיצ׳ר. האימון ללא תוויות נכונות; checkpoint נבחר לפי Gaussian prediction loss בקבוצות validation נפרדות. חמש התאמות outer ועשר התאמות זוגות לכיול PRMScore. אימון fold0 הישן נוצל מחדש, כולל ציוניו המקוריים.',
        'הקלט כולל16 טוקנים קודמים, מסכת קיום ומיקום יחסי. הארכיטקטורה הקיימת: רוחב32, שלוש קונבולוציות kernel3 עם dilations1/2/4 ומסלולים שיוריים. לפלט יש שדה קליטה15 טוקנים; משבצת העבר הרחוקה ביותר אינה משפיעה. לא שינינו את הארכיטקטורה באמצע הניסוי. Ridge משתמש בכל16 ההשהיות. הנרמול והסימנים מתשובה שלמה ולכן המערכת offline.',
        'ציוני השארית נשמרים עם סימן, ממוצעים על5 הפיצ׳רים ואז Top10 בתוך כל צעד. מצרפים לבסיס innovation5 במשקל .25, לאחר התאמת הסקאלה בין ציוני הצעדים בתשובה. אותו tail15 gate באחוזון .33. השונות החזויה אינה משמשת כמשקל fusion בניסוי הזה.',
        'tcn__shuffled מערבב את משבצות ההיסטוריה ו-tcn__zero מאפס את ערכיה, עם אותו מודל מאומן, מסכה ומיקום. אלו התערבויות בזמן ניקוד, לא מודלים שאומנו מחדש על null. אין להסיק ש-zero הוא נטול היסטוריה מקצה לקצה: פיצ׳ר innovation במטרה כבר מכיל היסטוריה. zero ללא הקידומת tcn הוא בקרת תחזית אפס של Step387.',
        'מגבלת בקרת הסדר: הערבוב הקיים פועל על16 משבצות, אך רק15 מגיעות לפלט. לכן הוא עשוי לשנות גם איזה טוקן היסטורי נשמט. ההשוואה אינה בידוד טהור של הסדר באותה קבוצת15 טוקנים נראים; לא יוחס לה הבדל הנובע בהכרח מסדר ההשהיות לבדו.',
        '## איכות איתור',
        'PB הוא מאקרו F1 על שמונת תאי ProcessBench, עם החלטת מיקום השגיאה או אין-שגיאה. within-AUC הוא ממוצע הדירוג בתוך6,030 תשובות PRMBench שבהן יש צעדים משני סוגי התוויות; תשובות חד-מחלקתיות אינן מאפשרות לחשב AUC פנימי. כל13,769 התשובות נוקדו. PRMScore משתמש בכיול המקונן המוחזק בקבוצות אחרות.',
        table(['שיטה','PB F1 %','within-AUC','PRMScore'],[[n,f"{100*m['pb_all8']:.4f}",f"{m['prm_within']:.6f}",f"{m['prmscore_q08']:.6f}"] for n,m in metrics.items()]),
        '## פירוט PB לפי תא',
        'פירוט תיאורי של אותה חבילת תחזיות. אין כאן הכרעת מובהקות נפרדת לכל תא.',
        table(['תא','innovation5','Ridge','BOCPD','TCN'],[[cell,*[f"{100*metrics[n]['pb_cells'][cell]['f1']:.4f}" for n in ('innovation5','ridge','bocpd','tcn__real')]] for cell in metrics['tcn__real']['pb_cells']]),
        '## שש השוואות ראשיות',
        '12 מדדים נבחנים: PB ו-within לכל זוג. 10,000 דגימות bootstrap מזווגות לפי קבוצות מקור; CI99.5833% בתיקון Bonferroni. רווח הכולל אפס אינו מוכיח שקילות. רווחי הסמך אינם כוללים את כל אי-הוודאות של החיפוש ההיסטורי באותה אוכלוסייה.',
        table(['השוואה','הפרש PB בנקודות אחוז','CI PB','הפרש within','CI within'],[[k,f"{100*c['pb_delta']:+.4f}",str([round(100*x,4) for x in c['pb_ci']]),f"{c['prm_within_delta_common']:+.6f}",str([round(x,6) for x in c['prm_within_ci']])] for k,c in payload['contrasts'].items()]),
        '## חיזוי ורגישות להיסטוריה',
        'MSE ממוצע על חמשת הפיצ׳רים ובמשקל שווה לכל תשובה; נמוך יותר עדיף לחיזוי טלמטריה. פירוט לפי פיצ׳ר ואחרי16 טוקנים ב-METRICS.json. המטרה אינה תווית נכונות, ולכן MSE נמוך יותר אינו מבטיח איתור טוב יותר.',
        table(['מנבא','MSE'],[[n,f"{v['scalar_mse_answer_mean']:.6f}"] for n,v in {**payload['reference_prediction'],**pred}.items()]),
        table(['TCN','מתאם שאריות מול Ridge','לאחר הסרת תצפית משותפת','מתאם שאריות מול TCN אמיתי'],[[n,*[round(v['correlations'][k]['quantiles'][1],6) if v['correlations'][k]['quantiles'] else 'null' for k in ('signed_residual_correlation_ridge','residual_correlation_after_removing_current','signed_residual_real_correlation')]] for n,v in pred.items()]),
        'הסרת התצפית המשותפת היא אבחון רגרסיה לינארית בתוך תשובה. היא אינה מסירה כל תלות משותפת, אינה ציון חדש ואינה מוכיחה את הנחות U-PCR.',
        '## פגיעות PB שנוספו ואבדו',
        'TCN עם היסטוריה אמיתית, תשובות עם שגיאה, אחרי אותו gate. אלו השוואות בין פסגות קיימות; אין כאן כלל בחירה ללא תוויות או חסם ביצועים של fusion.',
        table(['מול','פגיעות משותפות','נוספו','אבדו'],[[n,*[v['all_errors'][k] for k in ('both','gained','lost')]] for n,v in payload['peak_transitions'].items()]),
        '## עלות והתכנסות',
        f"זמן הקיר של תור ההשלמה: {state['seconds']/60:.1f} דקות, עם עד3 עבודות במקביל. זמן ההערכה הנוספת: {state['evaluation_seconds']:.1f} שניות. השורה של fold0 כוללת ריצה היסטורית ששימשה מחדש ואינה חלק מעלות התור החדש. זמני האימון והניקוד הם זמני קיר לכל תהליך, לא שעות CPU מצטברות; הניקוד הקיים שומר גם אבחונים שאינם מועמדים בניסוי הזה.",
        table(['folds מוחזקים בחוץ','עדכוני אימון','checkpoint נבחר','אימון בשניות','ניקוד בשניות'],job_cost),
        '## ביקורת ושחזור',
        f"אימות עצמאי של PB ו-pairwise within לכל{len(metrics)} השיטות. כל13 כותרות הייחוס שוחזרו. פער מרבי של readout סקלרי מתחזיות הטוקנים השמורות: {audit['prediction_audit']['max_scalar_readout_delta']:.3g}. כל קבוצות האימון/validation/held, חתימות הקוד והבנק נבדקו. fold0 השמור לא השתנה. ללא תוויות בהתאמה וללא השלמת כשל בממוצע.",
        'שדה הקליטה נבדק גם באמצעות נגזרת והפרעה למשבצת הראשונה. בדיקות הקשר, המודלים, הכיול המקונן והתור נמצאות ב-TESTS.json. אימות ה-readout בוצע על כל האוכלוסייה, לא על תת-מדגם איכות.',
        f"חזית נקודתית על PB/within: {', '.join(payload['pareto'])}. הניסוי משלים seed0 בלבד ואינו בדיקת יציבות בין seeds. הבנק והתיקון החתום .25 נבחרו בפיתוח קודם; נדרש אישור קפוא על שאלות חדשות לפני טענת הכללה.",
        '[פרוטוקול](../../docs/experiments/TCN_ALIGNED_PREDICTOR_20260915.md) · [מדדים](METRICS.json) · [אימות](AUDIT.json) · [מצב](RUN_STATE.json)']
    (OUT/'REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf8',newline='\n')
    body=[]
    for part in parts:
        if part.startswith('|'):
            rows=part.strip().splitlines()
            def row(line,tag):return '<tr>'+''.join('<'+tag+'>'+html.escape(x.strip())+'</'+tag+'>' for x in line.strip('|').split('|'))+'</tr>'
            body.append('<div class="scroll"><table>'+row(rows[0],'th')+''.join(row(x,'td') for x in rows[2:])+'</table></div>')
        elif part.startswith('## '):body.append('<h2>'+html.escape(part[3:])+'</h2>')
        elif part.startswith('# '):body.append('<h1>'+html.escape(part[2:])+'</h1>')
        else:
            escaped=html.escape(part)
            escaped=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',escaped)
            body.append('<p>'+escaped+'</p>')
    page='<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TCN לפני Fusion — Step388</title><style>body{font:17px system-ui;line-height:1.7;max-width:1100px;margin:40px auto;padding:0 22px;background:#f4f7fa;color:#182d39}table{border-collapse:collapse;width:100%;background:white;font-size:15px}th,td{padding:8px;border:1px solid #d4e0e7;text-align:right}th{background:#e1edf3}.scroll{overflow:auto}h1,h2{color:#13425a}a{color:#146780}</style>'+''.join(body)+'<p><a href="METRICS.json">METRICS.json</a> · <a href="AUDIT.json">AUDIT.json</a> · <a href="../../docs/reviews/tcn_aligned_architecture_audit_2026-09-15.md">בדיקת הארכיטקטורה</a></p></html>'
    (OUT/'REPORT.html').write_text(page,encoding='utf8',newline='\n')
    (OUT/'REPRODUCE.md').write_text('This is a completed archived run. Read REPORT.html and AUDIT.json for the preserved evidence. Do not rerun the training runner in place: it rewrites RUN_STATE.json and REUSE.json, including the initial reuse record. For a fresh replication use an isolated copy, retain the source files matching the manifest, and keep the original result directory separately. From that temporal research worktree with the existing frozen context data:\n\n```powershell\npython -B -X utf8 scripts/run_tcn_aligned_study.py\npython -B -X utf8 scripts/analyze_tcn_aligned_predictions.py\npython -B -X utf8 scripts/evaluate_tcn_aligned_study.py\npython -B -X utf8 scripts/report_tcn_aligned_study.py\n```\n\nThe original90-job flow queue must remain paused. Training and scoring use its shared job directory but a separate TCN-only roster. Do not overwrite old fold0 artifacts. Large checkpoints and SQLite/token arrays stay local with hashes.\n',encoding='utf8',newline='\n')
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir() if p.is_file() and p.name not in ('ARTIFACTS.json','QUEUE.lock','COMMIT_PATHS.json')}
    write(OUT/'ARTIFACTS.json',dict(files=files))


if __name__=='__main__':run()
