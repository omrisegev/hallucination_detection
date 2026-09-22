"""Render the completed residual-moment comparison; never refit or rescore."""
from pathlib import Path
import sys,json,csv,html,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_residual_moment_real import OUT, key, sha, write

INTRO=[
('השורה המחקרית', 'החלפת הפיצ׳רים בשאריות אינה מועילה בנתונים האמיתיים תחת החוזה שנבדק. שימוש בשאריות רק לאמידת משקלים, תוך ניקוד הפיצ׳רים המקוריים, נותן מועמד משני קרוב מאוד לבסיס. אין הכרזה על שיטה חדשה מנצחת.'),
('מה הושלם', '20 seeds בחמישה עולמות סינתטיים, ולאחריהם כל 13,769 התשובות, 145,597 הצעדים ו־6,968,779 הטוקנים האמיתיים. 18 זרועות חדשות/בקרות ועוד תשעה ייחוסים. כל ההשוואות הן פיתוח על האוכלוסייה שכבר שימשה למחקר; בחירת innovation5 השתמשה בתוויות פיתוח בעבר.'),
('ההפרדה המרכזית', 'L הוא הפיצ׳ר המקורי ו־R הוא ערכו פחות תחזית ridge מהעבר. שתי אותיות בסוף שם זרוע מציינות: הראשונה — הנתונים לאמידת covariance ומשקלים; השנייה — הנתונים שמנוקדים. למשל RL לומד משאריות ומנקד את הפיצ׳רים המקוריים. בכל תשובה המשקלים קבועים. זהו ניסוי באמידת fusion, ולא routing שמשנה משקל בכל טוקן.'),
('מה למדנו מקלוד', 'בעולם שבו ההפרעה איטית והאות הרצוי מהיר, רגעי שאריות יכולים לשנות את כיוון U-PCR ולהועיל. הוספנו שאריות אימון מחוץ לקבוצת התשובה, יחידות משותפות ובקרה שבה דווקא האות הרצוי איטי. שם ניקוי ההקשר מזיק מאוד. זו מגבלת ההנחה “צפוי = הפרעה”; אין ממנה מסקנה מוקדמת על הסמנטיקה של שגיאות reasoning.'),
('מה הראו הנתונים שלנו', 'מיצוע שאריות יורד מ־39.8314% / 0.760293 ל־36.3349% / 0.651336. U-PCR על שאריות משפר לעומת הממוצע החלש הזה, אבל אינו משחזר את הבסיס. בהשוואות הראשיות, החלפת רגעי רמות ברגעי שאריות משפרת within של native כשהניקוד נשאר על שאריות: כ־0.007393 מקומית וכ־0.000988 חיצונית, ברווחים מתוקנים חיוביים. לכן “fusion לא עובד” תהיה מסקנה שגויה; התרומה קיימת בתוך ייצוג שאיכותו הכוללת נמוכה יותר.'),
('המועמד המשני', 'pooled simplex RL מגיע ל־39.8459% / 0.761112 / PRMScore 0.639269. השינוי מול innovation5 הוא רק 0.0145 נקודת PB ו־0.000819 within. רווח PB משני 95% כולל אפס; within נמצא בקושי מעל אפס ברווח משני שאינו מתקן לבחירת הזרוע המובילה. אין כאן יתרון ראשי מאושר. ה־ridge השיורי הישן, 40.8472% / 0.761620, נשאר עדיף בנקודות על כל הזרועות החדשות; גם הוא זרוע פיתוח משנית.'),
('דירוג לעומת כיול בין תשובות', 'local simplex RL שומר within קרוב לבסיס, 0.759895, אך PRMScore יורד ל־0.598094 לעומת 0.638830. לכן משקול שנראה סביר בדירוג בתוך תשובה יכול עדיין לשנות באופן מזיק את סולם הציונים בין תשובות. ההחלפה ב־R היא גם שינוי נרחב יותר מהוספת innovation של H0 בלבד או מהמסלול השיורי הקודם; הממצא אינו שולל כל שימוש בחדשנות או בחיזוי הקשר.'),
('חיבור ל־Step384', 'האבחון הקודם מצא מבנה הקשר יציב, אך לא מדד איתור. כאן נמדדה איכות אמיתית, ונמצא שפירוק ההקשר לשארית בלבד מסיר מידע מועיל בחוזה שנבדק. כיוון rho המקומי כמעט אינו משתנה בין רמות לשאריות: חציון cosine של 0.99847. זה רחוק מהתיקון הגדול בכיוון האמינות בעולם הסינתטי של הפרעה איטית. יציבות או NLL אינם תחליף לתרומה לאיתור.'),
('מה נשאיר להמשך', 'נשמור את innovation5 ואת מסלול ה־ridge השיורי הקודם; לא נחליף את הציון בשארית ולא נחדש את תור ה־flows. אם בודקים כעת משקול דינמי, השאלה המצומצמת היא האם ההקשר של Step384 משפר את כיוון המשקלים כשהציונים המקוריים נשארים זמינים. נדרשים אותו head סטטי, מיקום בלבד, שכנות אקראית ובקרת שינוי עוצמה. תוצאת RL המשנית מצדיקה לכל היותר להשאיר רגעי שאריות כאבלציה, ולא לבחור אותם מראש כמנצחים. הניסוי הדינמי עדיין לא הורץ.'),
('שער הקושי ועולם השונויות השוות', 'המיפוי ל־eta שהציע קלוד כויל לאחר צפייה בעולמות. כש־g2 בתקרה הוא שקול ל־eta קבוע גבוה יותר, ולכן לא פתחנו חיפוש eta נוסף. כישלון energy בעולם שונויות שוליות שוות מצמצם את טענת ההצלחה של S0. הוא אינו מוכיח שכל מידע ההקשר נעלם: אפילו בגאוסי, מתאם בין אנרגיות חלון תלוי בריבוע covariance הצולב.'),
('גבולות החוזה', 'אין תוויות נכונות באימון ridge או באמידת covariance ומשקלים. ה־ridge נלמד מתשובות אחרות. במקומי covariance נאמד מהתשובה הנוכחית; בחיצוני הוא נאמד מאותה משפחת נתונים/מודל בקבוצות אחרות. גם המקומי משתמש במנבא חיצוני. נרמול התשובה השלמה ומיקום יחסי הופכים את המסלול ל־offline. שער tail15 באחוזון .33 נשאר טרנסדוקטיבי. שום שלב אחרי השגיאה הראשונה לא סומן אוטומטית שגוי.'),
('פרטי המימוש שמבודדים את ההשוואה', 'אותה סטיית תקן של הרמות משמשת גם לשאריות. covariance ממורכז, floor של 1e-6 ותקרת g2 של 0.25 לפי הרמות. native שומר שני רכיבי PCR, ממיר משקלים ליחידות גולמיות ומנרמל את סכום הערכים המוחלטים ל־1; simplex משתמש בכל covariance עם tau=1 ו־eta=.25. Top10 נבחר בנפרד לכל זרם לפני הכפלת המשקל, גם אם המשקל שלילי. זהו חוזה readout מותאם, ולא כל מימוש אפשרי של native IU.'),
('הפרדת מקורות ובדיקות', 'נעשה שימוש חוזר ב־15 מודלי ridge קפואים ונוספו עשרה מודלים המוציאים שלושה folds, רק לצורך שאריות מקורות הכיול המקונן. שארית של תשובת מקור מחושבת במודל שאינו למד את קבוצתה ואינו למד את קבוצת הבדיקה החיצונית. כל הטוקנים נוקדו, כולל 16 הראשונים עם מסכת היסטוריה. 22 בדיקות עברו; 60 התאמות real-data נבדקו מול upcr_fit_covariance; ציוני PB ו־within של כל 27 השיטות נבדקו בחישוב עצמאי. לא הופעל אף fallback מקומי.'),
('אי־ודאות ועלות', '10,000 דגימות bootstrap מזווגות של קבוצות המקור. שמונה השוואות ראשיות בשני מדדים: רווחי 99.6875%; יתר ההשוואות אבחוניות ב־95%. התיקון אינו כולל את כל הבחירה ההסתגלותית ההיסטורית. הסינתטי ארך כ־20 שניות; התאמה וניקוד אמיתיים כ־278 שניות, והערכה וביקורת כ־38 שניות, בתהליכון BLAS אחד. ציונים גדולים נשמרים מקומית עם חתימות.'),
]

def table(headers,rows):
    md='| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'
    md+='\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
    ht='<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'
    ht+=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    return md,ht+'</tbody></table></div>'

def run():
    payload=json.loads((OUT/'METRICS.json').read_text());metrics=payload['metrics']
    syn=json.loads((ROOT/'results/residual_moment_synthetic_v1/RESULTS.json').read_text())
    state=json.loads((OUT/'RUN_STATE.json').read_text());state.pop('quality_labels_used',None)
    state.update(fitting_correctness_labels_used=False,evaluation_correctness_labels_used=True)
    write(OUT/'RUN_STATE.json',state)
    # Read-only diagnostics from complete outer fits, not new quality variants.
    ratios=[];pool=[]
    for f in range(5):
        with np.load(OUT/(key((f,))+'_DIAGNOSTICS.npz')) as d:ratios.extend(d['g2']/d['var_y'][:,None])
        pp=json.loads((OUT/(key((f,))+'_POOL.json')).read_text())
        for cell,item in pp.items():
            fit=item['fit'];pool.append(dict(fold=f,cell=cell,groups=len(item['reference_groups']),
                g2_fraction=[fit[r]['g2']/fit['var_y'] for r in ('L','R')],
                native_fallbacks=[fit[r]['native_fallback'] for r in ('L','R')]))
    ratios=np.asarray(ratios)
    integration=dict(local_g2_ceiling_fractions=np.mean(ratios>=1-1.5/300,axis=0),
        local_g2_fraction_median=np.median(ratios,axis=0),pooled=pool,unit_tests_passed=22,
        output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),disk_free=shutil.disk_usage(OUT).free,
        interpretation='Keep original scores; no residual-only promotion; pooled simplex RL is secondary small signal',
        source_code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/evaluate_residual_moment_real.py',ROOT/'scripts/run_residual_moment_synthetic.py',ROOT/'spectral_utils/residual_moment_fusion.py']})
    write(OUT/'INTEGRATION_SUMMARY.json',integration)
    blocks=[]
    for title,text in INTRO:blocks.append(('text',title,text))
    names=['original4','innovation5','equal__R','local__native__LL','local__native__LR','local__native__RL','local__native__RR',
        'pooled__simplex__LL','pooled__simplex__RL','pooled__simplex__RR','ridge_signed025_secondary']
    mainrows=[[n,f"{100*metrics[n]['pb_all8']:.4f}%",f"{metrics[n]['prm_within']:.6f}",f"{metrics[n]['prmscore_q08']:.6f}"] for n in names]
    blocks.insert(2,('table','השוואות מרכזיות',(['שיטה','PB','within','PRMScore'],mainrows)))
    synthrows=[[world]+[f'{v[n]:.6f}' for n in ('equal__L','equal__R','pooled__native__LL','pooled__native__RL','pooled__native__RR')] for world,v in syn['summary'].items()]
    blocks.append(('table','הסינתטי — יחידות משותפות ושאריות אימון מחוץ לקבוצה',
        (['עולם','equal L','equal R','native LL','native RL','native RR'],synthrows)))
    allrows=[[n,f"{100*m['pb_all8']:.4f}%",f"{m['prm_within']:.6f}",f"{m['prmscore_q08']:.6f}",m['valid_answers']] for n,m in metrics.items()]
    blocks.append(('table','כל הזרועות והייחוסים',(['שיטה','PB','within','PRMScore','תשובות'],allrows)))
    ci=lambda a,scale=1:'['+', '.join(f'{scale*v:+.6f}' for v in a)+']'
    prows=[[name,f"{100*c['pb_delta']:+.4f}",ci(c['pb_ci'],100),f"{c['prm_within_delta_common']:+.6f}",ci(c['prm_within_ci'])] for name,c in payload['contrasts'].items() if c['primary']]
    blocks.append(('table','השוואות ראשיות — 99.6875%',(['השוואה','הפרש PB בנקודות','CI PB','הפרש within','CI within'],prows)))
    crows=[]
    for n in ['innovation5','pooled__simplex__RL','equal__R','local__native__RR']:
        for cell,v in metrics[n]['pb_cells'].items():
            crows.append([n,cell,json.dumps(v,ensure_ascii=False)])
    blocks.append(('table','PB לפי תא — פירוט ישיר ממערך התחזיות',(['שיטה','תא','מדדים וספירות'],crows)))
    srows=[]
    for n in ['equal__R','pooled__simplex__RL','local__native__RR']:
        for stratum,v in payload['early_middle_late'][n].items():srows.append([n,stratum,v['answers'],v['hits'],v['gained'],v['lost']])
    blocks.append(('table','מיקום השגיאה הראשונה — אבחון לאחר הערכה',(['שיטה','אזור','תשובות','פגיעות','נוספו','אבדו'],srows)))
    md=['# רגעי שאריות ו־U-PCR — ניסוי מלא, 15.09.2026\n'];ht=[]
    for kind,title,body in blocks:
        md.append('\n## '+title+'\n');ht.append('<h2>'+html.escape(title)+'</h2>')
        if kind=='text':md.append(body+'\n');ht.append('<p>'+html.escape(body)+'</p>')
        else:
            m,h=table(*body);md.append(m);ht.append(h)
    links=[('הפרוטוקול הקפוא','../../docs/experiments/RESIDUAL_MOMENT_FUSION_20260915.md'),('מדדים, השוואות וחזית Pareto','METRICS.json'),('ביקורת עצמאית','AUDIT.json'),('הפרדת מקורות וחתימות ridge','MODEL_AUDIT.json'),('תוצאות סינתטיות מלאות','../residual_moment_synthetic_v1/RESULTS.json'),('אבחון ההקשר הקודם','../energy_context_stability_v2/REPORT.html'),('יומן ניקוי worktrees','../../docs/reviews/worktree_cleanup_2026-09-15.json')]
    md.append('\n## קבצים\n\n'+'\n'.join(f'- [{t}]({p})' for t,p in links)+'\n')
    ht.append('<h2>קבצים</h2><ul>'+''.join('<li><a href="'+p+'">'+t+'</a></li>' for t,p in links)+'</ul>')
    style='body{font:17px/1.7 system-ui;background:#f1f5f9;color:#203247;margin:0}main{max-width:1200px;margin:24px auto;padding:32px;background:white;border-radius:16px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #d9e2ec;text-align:right}td:first-child{direction:ltr;unicode-bidi:isolate}th{background:#eaf0f7}.scroll{overflow:auto}a{color:#12687d}h2{margin-top:2em}p{max-width:105ch}'
    (OUT/'REPORT.md').write_text('\n'.join(md),encoding='utf8')
    (OUT/'REPORT.html').write_text('<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>רגעי שאריות ו־U-PCR</title><style>'+style+'</style><main><h1>רגעי שאריות ו־U-PCR — ניסוי מלא</h1>'+''.join(ht)+'</main></html>',encoding='utf8')
    with (OUT/'SUMMARY.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['method','PB_percent','within','PRMScore','answers']);w.writerows(allrows)
    # Explicit replay commands and large-artifact hashes for another checkout.
    reproduction='''# Reproduction\n\nUse the frozen data bundle and Step379 ridge NPZ/JSON pairs named in MODEL_AUDIT.json.\nVerify MANIFEST.json and model hashes before use. Large scores remain local.\n\n```powershell\npython scripts/run_residual_moment_synthetic.py\npython scripts/run_residual_moment_real.py\npython scripts/evaluate_residual_moment_real.py\npython scripts/report_residual_moment_fusion.py\n```\n\nThe real scorer does not load correctness labels. The evaluator loads the main\nrepository evaluation contract. Five outer score archives plus ten pair-exclusion\narchives provide complete nested calibration. The ten added triple-exclusion ridge\nfits use the unchanged16384-sample deterministic fit rule. Existing single/pair\nmodels and all original results are immutable. NPY data and frozen source models\nmust be available at their manifest paths; Git summaries alone cannot reproduce\nraw scores without those local/Drive artifacts.\n'''
    (OUT/'REPRODUCE.md').write_text(reproduction,encoding='utf8')
    write(OUT/'ARTIFACTS.json',{str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='ARTIFACTS.json'})
    print(json.dumps(dict(report=str(OUT/'REPORT.html'),local_g2_ceiling_fractions=integration['local_g2_ceiling_fractions'].tolist(),unit_tests=22)))

if __name__=='__main__':run()
