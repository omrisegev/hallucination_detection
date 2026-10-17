"""Report the completed original-level contextual fusion quality experiment."""
from pathlib import Path
import sys,json,html,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_context_weighted_levels import OUT,sha,write
from scripts.report_residual_moment_fusion import table

PARAGRAPHS=[
('מה נבדק', 'המשך ישיר ל־Step384, לפני ההסתעפות של רגעי השאריות: ייצוג הקשר מהעבר בוחר שכנים בקבוצות מקור אחרות; covariance מותנה קובע משקלי fusion; הניקוד משתמש בחמשת הפיצ׳רים המקוריים של innovation5. כל 13,769 התשובות, 145,597 הצעדים ו־6,968,779 הטוקנים נכללו. 19 מדיניות/בקרות ותשעה ייחוסים. אין התאמה לפי תוויות נכונות.'),
('החלטה', 'המשקול הדינמי שנבדק אינו מחליף את בסיס המיצוע. U-PCR הלא־מרוסן נפגע באופן ניכר בשינוי כיוון המשקלים; simplex והבקרה של שתי קבוצות נשארים קרובים יותר לייחוסים, אך אינם עוברים את innovation5 בנקודות PB/within. זהו מבחן איכות של מנגנון מוגדר, ולא סגירה של כל שימוש בהקשר או של FM/DiFlo.'),
('מדוע האבחון הקודם לא הספיק', 'ב־Step384 השכנים לפי הקשר שיפרו NLL של הפיצ׳רים והמשקלים השתנו מעבר לרעש הדגימה המותנה. המדדים האלה אינם מזהים אילו פיצ׳רים אמינים סמנטית. כעת נמדדה המשימה עצמה. בניסוי native, בקרת עוצמה בלבד נשארת קרובה יותר לסטטי, ואילו שינוי הכיוון ממשיך לפגוע גם לאחר הסרת שינוי העוצמה. הפגיעה אינה מוסברת רק בהגדלה ובהקטנה של כל הציונים יחד.'),
('רזולוציית הזמן', 'המשקל מתעדכן בעד 16 נקודות לאורך התשובה, על בסיס 16 הטוקנים הקודמים בכל נקודה. ביניהן מוחזק המשקל האחרון; אין אינטרפולציה מעוגן עתידי. ב־16 הטוקנים הראשונים משתמשים במשקל הסטטי. ציוני כל הטוקנים נשמרים, והיסטוריה חוצה צעדים. זו מדיניות מקוטעת של עד 16 עדכונים, לא התאמת הקשר מדויקת בכל טוקן. היא עלולה להחמיץ שינוי קצר בהקשר; התוצאה אינה שוללת כל רזולוציה אחרת.'),
('אבחון גיל המשקלים לאחר הניקוד', 'על כל האוכלוסייה, 4.87% מתרומות הפיצ׳ר־טוקן שנבחרו ל־Top10 נמצאות בחימום הסטטי. מבין היתר, 45.62% משתמשות במשקל שעודכן לפחות 16 טוקנים קודם; חציון הגיל 14 טוקנים ואחוזון90 הוא 48. זהו אבחון נוסף שנעשה לאחר ראיית הדירוג, ללא תוויות. הוא אינו מוכיח שיישון המשקלים גרם לפגיעה, אבל מחייב לתחום את המסקנה למדיניות 16 העדכונים. קביעת יתרון או כישלון של התאמה מדויקת בכל טוקן תדרוש השוואה נוספת.'),
('הבקרות', 'בכל אחד משלושת מנגנוני המשקול נבדקו static, הקשר energy, מיקום ואורך בלבד, ושכנים אקראיים עם אותו kernel של זרוע energy. נוספו energy עם שינוי עוצמה בלבד, ו־energy עם שינוי כיוון בלבד. כל הזרועות משתמשות באותם טוקני Top10 שנבחרו לפי ערכי הפיצ׳ר המקוריים; המשקל מוכפל לאחר הבחירה. גם משקל שלילי אינו מחליף את הבחירה לטוקני תחתית.'),
('מה פירוש static כאן', 'הבנק innovation5 כבר מכיל מידע מהעבר באמצעות הפיצ׳ר החמישי — H0lim פחות ממוצע העבר. static מתייחס למשקלי fusion קבועים, ולא להיעדר מידע כרונולוגי בפיצ׳רים. לכן הניסוי בודק תוספת של משקול תלוי־הקשר מעל הבנק הקיים; אין להסיק ממנו שהיסטוריה אינה מועילה כלל.'),
('מנגנוני המשקול', 'native משתמש ב־U-PCR הקנוני עם שני רכיבים. מקדמים מומרים ליחידות גולמיות ומחולקים בנורמת L1 של המקדמים הסטטיים, כדי לשמור שינויי עוצמה. simplex משתמש בכל covariance, במשקלים לא־שליליים וב־eta=.25 עם tau=1. group הוא אותה מטרה על ציר יחיד בין {H0lim,VE0,innovation} לבין {VE075,VE1}; הקבוצות מוגדרות מראש, ולא נלמדו באמצעות GroupFS.'),
('הגדרת העוצמה', 'לכל משקל גולמי w מחשבים A=||sd*w||2 / ||sd*w_static||2 באמצעות סקאלות האימון. בקרת עוצמה בלבד היא A*w_static; בקרת כיוון בלבד היא w/A. לכן בידוד העוצמה נעשה בקואורדינטות פיצ׳רים משותפות, ולא באמצעות סכום מקדמים שיכול להתאפס כשיש משקלים שליליים.'),
('מקורות וכיול', '45 ההתאמות החיצוניות של Step384 נוצלו מחדש אחרי בדיקת חתימות, קבוצות ומזהי נקודות. נוספו עשר התאמות PRMB עם הוצאת זוגות folds, כדי שכיול PRMScore לא ילמד ממקורות הבדיקה החיצוניים. בכל התאמה נשארו K=64 קבוצות שכנים שונות ו־.5 borrowing. שער tail15 באחוזון .33 נשמר. הנרמול והסימנים של תשובה שלמה ומיקומה היחסי הופכים את המערכת ל־offline; השער טרנסדוקטיבי.'),
('הסקה סטטיסטית', 'ההשוואות הראשיות הוגדרו מראש עבור simplex: energy מול static, מיקום, אקראי ועוצמה בלבד. ארבע השוואות כפול שני מדדים: רווחי 99.375% מ־10,000 דגימות bootstrap מזווגות של קבוצות מקור. כל שמונת הרווחים הראשיים כוללים אפס: אין יתרון ראשי מאומת, ואין מכאן הוכחת שקילות. native, group, כיוון בלבד והשוואות לבסיס הן משניות ב־95%. הפגיעה של native מול סטטי נשארת שלילית בשני הרווחים המשניים. כל האוכלוסייה שימשה לפיתוח בעבר, כולל בחירת innovation5 בעזרת תוויות; הרווחים אינם מתקנים את כל הבחירה ההסתגלותית.'),
('חיבור ל־FM ו־DiFlo', 'הקווים נשארים פתוחים. FM הוא מודל flow בסיסי; DiFlo מוסיף מטרות עזר; DOT הוא מדד שנגזר מהמסלולים של שניהם. אפשר להשתמש בתיאור הקשר שה-flow לומד לבחירת השכנים באותו צינור U-PCR. אפשר גם לבחון DOT כתוספת למסלול השיורי הקודם או כמידת הסתמכות על משקל מקומי. DOT יחיד אינו אומדן אמינות לכל פיצ׳ר, ואין להניח שסטייה מהתפלגות האימון היא שגיאת reasoning. השילוב עדיין לא אומן או נבדק.'),
('מה ממשיכים ומה לא מקבעים', 'נשמרים innovation5 והמסלול ההיסטורי של ridge עם תוספת שיורית. לפני השקעה במודל הקשר יקר לאותו אומדן משקלים יש להפריד בין בעיית אמידת האמינות לבין מגבלת תדירות העדכון שנבדקה כאן. הממצאים אינם מבחן מלא של flow או DOT כתוספת לציון. לפני חידוש התור העצבי נדרשים רישום רכיבי loss ובקרות הקשר/יישור; ארבע העבודות שהושלמו אינן מסקנת איכות על כל האוכלוסייה. אין הרחבת eta, בנק או רזולוציה בדיעבד בתוך הניסוי הזה.'),
]

def run():
    state=json.loads((OUT/'RUN_STATE.json').read_text());payload=json.loads((OUT/'METRICS.json').read_text())
    if state['status']!='COMPLETE_REVIEWED':raise ValueError('Final audit incomplete')
    metrics=payload['metrics'];audit=json.loads((OUT/'AUDIT.json').read_text())
    interim=json.loads((OUT/'RANKING_INTERIM.json').read_text())['methods']
    for name,m in interim.items():
        np.testing.assert_allclose([m['PB'],m['within']],[metrics[name]['pb_all8'],metrics[name]['prm_within']],atol=2e-14,rtol=0)
    fits=[json.loads(p.read_text()) for p in OUT.glob('*/COMPLETE.json')]
    assert len(fits)==55 and sum(f['reused'] for f in fits)==45
    amplitude={}
    for head in ('native','simplex','group'):
        arrays=[]
        for p in OUT.glob('*__exclude_?/ANCHORS.npz'):
            fit=json.loads(p.with_name('FIT.json').read_text())['preprocessing'];sd=np.asarray(fit['sd'])
            from spectral_utils.energy_context_stability import heads
            h=heads(np.asarray(fit['C']),sd,fit['var_y']);key={'native':'native_a','simplex':'qp_w','group':'group_w'}[head]
            static=h[key][0] if head=='native' else h[key][0]*sd
            with np.load(p) as a:dynamic=a['energy__'+head] if head=='native' else a['energy__'+head]*sd
            arrays.extend(np.linalg.norm(dynamic,axis=1)/np.linalg.norm(static))
        amplitude[head]=dict(q01_q50_q99=np.quantile(arrays,[.01,.5,.99]).tolist(),min=float(np.min(arrays)),max=float(np.max(arrays)))
    summary=dict(decision='NO_16_ANCHOR_POLICY_PROMOTION_EXACT_TOKEN_CONTEXT_UNRESOLVED',fits=55,reused=45,nested_new=10,
        methods=len(metrics),full_population=True,amplitude=amplitude,neff_min=min(f['energy_neff_min'] for f in fits),
        seconds_scoring=state['seconds'],seconds_evaluation=state['evaluation_seconds'],tests=22,
        integration_status='FM/DiFlo open; bridge proposed, not fitted; neural queue paused4/90',
        report_code_sha256=sha(Path(__file__)))
    write(OUT/'SUMMARY.json',summary)
    md=['# משקול לפי הקשר על הפיצ׳רים המקוריים — Step386\n'];ht=[]
    for title,text in PARAGRAPHS:
        md.append('\n## '+title+'\n\n'+text+'\n');ht.append('<h2>'+title+'</h2><p>'+html.escape(text)+'</p>')
        if title=='מה נבדק':
            names=['innovation5','native__static','native__energy','native__amplitude','native__direction','simplex__static','simplex__energy','group__static','group__energy','ridge_signed025_secondary']
            rows=[[n,f"{100*metrics[n]['pb_all8']:.4f}%",f"{metrics[n]['prm_within']:.6f}",f"{metrics[n]['prmscore_q08']:.6f}"] for n in names]
            m,h=table(['שיטה','PB','within','PRMScore'],rows);md.append(m);ht.append(h)
    ci=lambda xs,scale=1:'['+', '.join(f'{scale*x:+.6f}' for x in xs)+']'
    for primary,title in [(True,'ההשוואות הראשיות — 99.375%'),(False,'ההשוואות המשניות — 95%')]:
        rows=[[n,f"{100*c['pb_delta']:+.4f}",ci(c['pb_ci'],100),f"{c['prm_within_delta_common']:+.6f}",ci(c['prm_within_ci'])] for n,c in payload['contrasts'].items() if c['primary']==primary]
        m,h=table(['השוואה','הפרש PB בנקודות','CI PB','הפרש within','CI within'],rows)
        md.extend(['\n## '+title+'\n',m]);ht.extend(['<h2>'+title+'</h2>',h])
    rows=[[n,f"{100*m['pb_all8']:.4f}%",f"{m['prm_within']:.6f}",f"{m['prmscore_q08']:.6f}",m['valid_answers']] for n,m in metrics.items()]
    m,h=table(['שיטה','PB','within','PRMScore','תשובות'],rows);md.extend(['\n## כל התוצאות\n',m]);ht.extend(['<h2>כל התוצאות</h2>',h])
    timing=f"22 בדיקות עברו; חישוב עצמאי של PB ו־within לכל {len(metrics)} השיטות. שחזור סקלרי של readout דינמי על {audit['scalar_readout']['answers']} תשובות בדק את כל 19 המדיניות. הפער המרבי {audit['scalar_readout']['max_scalar_score_difference']:.3g}. כל תשעת הייחוסים שוחזרו. ניקוד והתאמות: {state['seconds']:.1f} שניות; הערכה וביקורת: {state['evaluation_seconds']:.1f} שניות, BLAS בתהליכון אחד."
    md.append('\n## אימות ועלות\n\n'+timing+'\n');ht.append('<h2>אימות ועלות</h2><p>'+timing+'</p>')
    links=[('פרוטוקול הניסוי','../../docs/experiments/CONTEXT_WEIGHTED_LEVEL_FUSION_20260915.md'),('כיצד הקווים יכולים להשתלב','../../docs/reviews/context_fusion_flow_bridge_2026-09-15.md'),('מדדים ותאים','METRICS.json'),('אימות עצמאי','AUDIT.json'),('תוצאות רגעי השאריות','../residual_moment_fusion_v1/REPORT.html')]
    md.append('\n## מקורות מקומיים\n\n'+'\n'.join(f'- [{t}]({p})' for t,p in links));ht.append('<ul>'+''.join('<li><a href="'+p+'">'+t+'</a></li>' for t,p in links)+'</ul>')
    (OUT/'REPORT.md').write_text('\n'.join(md),encoding='utf8')
    style='body{font:17px/1.7 system-ui;background:#f1f5f9;color:#203247;margin:0}main{max-width:1200px;margin:24px auto;padding:32px;background:white;border-radius:16px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #d9e2ec;text-align:right}td:first-child{direction:ltr;unicode-bidi:isolate}th{background:#eaf0f7}.scroll{overflow:auto}a{color:#12687d}h2{margin-top:2em}p{max-width:105ch}'
    (OUT/'REPORT.html').write_text('<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>משקול לפי הקשר — Step386</title><style>'+style+'</style><main><h1>משקול לפי הקשר על הפיצ׳רים המקוריים</h1>'+''.join(ht)+'</main></html>',encoding='utf8')
    with (OUT/'METRICS.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['method','PB','within','PRMScore','answers']);w.writerows(rows)
    write(OUT/'ARTIFACTS.json',{str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='ARTIFACTS.json'})
    print(json.dumps(summary))

if __name__=='__main__':run()
