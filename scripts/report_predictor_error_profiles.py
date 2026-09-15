"""Hebrew diagnostic atlas with inspectable full-token trajectories."""
from pathlib import Path
import sys,json,csv,html,re,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.analyze_predictor_error_profiles import OUT,KEYS,FEATURES,figure_save
from scripts.run_predictor_subset_study import sha,write

def table(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'

def run():
    d=json.loads((OUT/'ANALYSIS.json').read_text());b=json.loads((OUT/'BROAD_ANALYSIS.json').read_text());s=b['summary']
    explanation=[
        'יש הבדלים קבוצתיים ברורים, אבל לא נמצא סימן שמאפיין כל שגיאה שהוחמצה ומבדיל אותה מכל שגיאה שנתפסה. ההתפלגויות חופפות. האבחון מתאר את נקודות העיוורון של המערכת שנבדקה; הוא אינו מוכיח סיבת כשל או גלאי חדש.',
        'הגדרנו תחילה החמצה מול 32 צירופי המנבאים, ואז הרחבנו לפי בקשת עמרי ל-11 ארכיוני ניסויים מלאים: 254 עמודות ציונים, 178 מערכי ציונים שונים ו-173 וקטורי בחירת פסגה שונים. הארכיון כולל גם שיטות חלשות ובקרות. כל קובץ הותאם לייחוס משותף על כל 145,597 הצעדים; לא הסתמכנו על אורך המערך בלבד. אין כאן טענה שכל ניסוי שנעשה אי פעם בפרויקט נכלל, ואין תוצאת FM/DiFlo מלאה חדשה.',
        'מתוך 4,442 תשובות PB עם שגיאה, נשארות 1,528 החמצות סופיות בארכיון הרחב: ב-885 אף שיטה אינה בוחרת את הצעד הנכון גם לפני gate; ב-643 יש לפחות פסגה נכונה אחת שה-gate סוגר. מתוך 885 החמצות המיקום, 707 הן עם gate פתוח ו-178 עם gate סגור. ההשוואה העיקרית להלן היא 707 מול 2,914 תשובות שנתפסו כאשר ה-gate פתוח, כדי לא לערבב סגירת gate עם בחירת צעד.',
        'התשובות שקשה למקם בהן את השגיאה ארוכות יותר, והצעד השגוי תופס חלק קטן יותר מהתשובה: חציון של כ-10.3% לעומת 19.3%. המנבאים מסכימים יותר ביניהם בצעד שהוחמץ, והשינוי בפיצ׳רים לעומת 16 הטוקנים הקודמים קטן יותר. אלה הבדלים ממוצעים עם חפיפה, לא חוק שמזהה שגיאות.',
        'גם לאחר השוואה בתוך אותה משימה ומודל, רבעון אורך ושליש מיקום, נשארים הבדלים באורך היחסי של הצעד, בשינוי הפיצ׳רים ובאי-ההסכמה בין המנבאים. ההתאמה גסה ואינה מסירה כל ערבוב. בשינוי מהעבר אין היסטוריה זמינה לשגיאות בצעד הראשון; המכנים הקטנים יותר מפורטים בטבלה. התאמה אינה אימון מודל או הוכחת סיבתיות.',
        'בקרת גבול הצעד חשובה במיוחד: בארבעת הטוקנים הראשונים, ממוצע הפיצ׳רים המתוקננים בהחמצות הוא 0.822 בצעד השגוי מול 0.762 בצעד התקין הקודם. בקבוצה שנתפסה: 0.910 מול 0.574. יש קפיצה גם בתחילת צעד תקין. לכן אין לפרש את השיא סביב t=0 כחתימה ייחודית של שגיאה. המספרים הם תיאור של זוגות מתוך אותה תשובה, ללא מבחן חדש להבדל הזה.',
        'ההחמצות אינן בעיקר תיקו בין שתי פסגות: ב-707 ההחמצות עם gate פתוח, השגיאה האמיתית מדורגת בחציון במקום החמישי ב-TCN, ב-IU המוביל ובממוצע המוביל. ב-TCN רק 42 מתוך 707 נמצאות בשני המקומות הראשונים. בכל הארכיון יש פסגה במרחק צעד אחד ב-494 תשובות, אך זו בחירה בדיעבד בין המון פסגות ואינה תיקון זמין. אין כיוון יחיד של החמצה: יש בחירות מוקדמות ומאוחרות.',
        'המסקנה שלי להמשך: עוד שינוי משקלים בין אותם מנבאים אינו כרגע ההסבר המשכנע ביותר לאופן חילוץ ההחמצות. כדאי לבודד האם ה-readout מדגיש אורך וגבולות צעדים על חשבון ראיות מתוך הצעד, או שהראיות שאנו מחלצים אינן מבחינות מספיק בין reasoning שגוי לתקין. האבחון אינו מוכיח שאין מידע נוסף בפיצ׳רים או שכל השגיאות נובעות מאותו מנגנון.',
        'הגדרת הקבוצות משתמשת בתוויות ובתוצאות השיטות. בפרט, ההבדל בשארית החתומה קשור לציון שכבר משמש לאיתור ולכן הוא בחלקו תוצאה של הגדרת הקבוצה. גם האורקל על מאות תצורות מוצא פגיעות מקריות. בלי התצורות שסומנו בשמן כבקרות נותרות 946 החמצות מיקום עם gate פתוח, לעומת 707 בארכיון המלא; רשימת שמות הבקרות בקוד היא כלל שקוף, לא סיווג מדעי מושלם.'
    ]
    with (OUT/'ANSWER_DIAGNOSTICS.csv').open(encoding='utf8') as f:allrows=list(csv.DictReader(f))
    with (OUT/'COMMON_MISSES.csv').open(encoding='utf8') as f:rawmiss={r['uid'] for r in csv.DictReader(f) if r['no_archive_peak_correct_even_without_gate']=='True'}
    rawkeys=['tokens','steps','error_start_fraction','error_step_fraction','feature_change_l2','predictor_disagreement','mean_signed_residual']
    rawgroups={}
    for flag,name in [(False,'found_before_gate'),(True,'missed_even_before_gate')]:
        rr=[r for r in allrows if (r['uid'] in rawmiss)==flag];values={}
        for k in rawkeys:
            v=np.array([float(r[k]) for r in rr if r[k] not in ('','None')]);values[k]=dict(n=len(v),median=float(np.median(v)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)))
        rawgroups[name]=dict(n=len(rr),diagnostics=values)
    write(OUT/'RAW_COHORTS.json',dict(scope='All4442 error answers BEFORE gate; secondary descriptive sensitivity, no extra significance test.',groups=rawgroups))
    body=['<h1>מה משותף לשגיאות שכל השיטות מחמיצות? — Step390</h1>']+['<p>'+html.escape(p)+'</p>' for p in explanation]
    body+=['<h2>ההגדרה משנה את קבוצת ההחמצות</h2>',table(['השוואה','נתפסו לפחות פעם אחת','הוחמצו עם gate פתוח','gate סגור'],[
        ['32 צירופי המנבאים',1589,2032,821],['32 וגם חמשת המנבאים הבודדים',4442-d['strict_missed_including_five_singletons'],d['strict_gate_open_missed'],821],
        ['הארכיון הרחב כולל בקרות',s['found'],s['missed_gate_open'],821]]),
        '<p>הספירות הן תשובות תחת מודלים, מקובצות ב-1,979 שאלות מקור. איחוד הפגיעות הוא אורקל בלבד; אי אפשר לבחור את השיטה הנכונה לכל תשובה בלי מידע נוסף. השיטות ההיסטוריות נבדקות כאן תחת אותו gate נוכחי; זה אבחון של פסגות, לא שחזור מדדי הפריסה ההיסטוריים שלהן.</p>']
    rawnames=['אורך תשובה בטוקנים','מספר צעדים','מיקום תחילת השגיאה','אורך הצעד / אורך התשובה','שינוי פיצ׳רים L2 לעומת העבר','אי-הסכמה בין המנבאים','שארית חתומה ממוצעת']
    body+=['<h2>כל 885 החמצות המיקום — גם בלי gate</h2><p>זו הגדרת ההחמצה המחמירה ביחס לארכיון שנבדק: אף שיטה אינה בוחרת את הצעד, גם אם ה-gate אינו חוסם. הטבלה משווה חציונים ל-3,557 תשובות שבהן לפחות פסגה אחת נכונה לפני gate. אין כאן שימוש ב-gate להגדרת הקבוצות. זו בקרת רגישות תיאורית; רווחי הסמך בהמשך מתייחסים להשוואה המבודדת כשה-gate פתוח.</p>',
        table(['מאפיין','פסגה נכונה קיימת: 3,557','אף פסגה נכונה: 885'],[[label]+[f"{rawgroups[g]['diagnostics'][k]['median']:.4f}" for g in ['found_before_gate','missed_even_before_gate']] for label,k in zip(rawnames,rawkeys)]),
        '<img src="RAW_OVERVIEW.png" alt="All885 common localization misses before gate compared with other error answers">']
    names=['אורך תשובה (log2 טוקנים)','מיקום תחילת השגיאה בתשובה','אורך הצעד / אורך התשובה','שינוי פיצ׳רים L2 לעומת העבר','אי-הסכמה בין המנבאים','שארית חתומה ממוצעת']
    body+=['<h2>הבדלים כמותיים — gate פתוח בלבד</h2>',table(['מאפיין','חציון נתפסו','חציון הוחמצו','ממוצע נתפסו אחרי התאמה','ממוצע הוחמצו אחרי התאמה','CI להפרש הוחמצו פחות נתפסו','מכנים בהתאמה'],[
        [names[q],f"{b['cohorts']['found']['medians'][k]:.4f}",f"{b['cohorts']['missed_open']['medians'][k]:.4f}",
         f"{v['matched_found_mean']:.4f}",f"{v['matched_missed_mean']:.4f}",str([round(x,4) for x in v['matched_ci']]),str(v['matched_found_n'])+' / '+str(v['matched_missed_n'])]
        for q,k in enumerate(KEYS) for v in [b['uncertainty']['metrics'][k]]]),
        '<p>10,000 דגימות bootstrap של קבוצות מקור; רווחי 99.5833% עבור שישה מאפיינים בשתי השוואות בתוך כל ניתוח. ההרחבה לארכיון נעשתה אחרי האבחון הראשון ואינה אישור בלתי תלוי שלו; אין תיקון לכל הבחירות במחקר ההיסטורי. משקלי שכבות נקבעו לפי ספירות הרמוניות, מינימום 5 תשובות מכל קבוצה. ב-592 דגימות התרוקנה לפחות שכבה נכללת אחת; המשקלים נורמלו על השכבות הזמינות באותה דגימה. אין הצבת אפסים כתצפיות חסרות.</p>',
        '<h2>התפלגויות על כל התשובות</h2><p>כל קבוצה מנורמלת לסכום 1. ירוק: נתפסה לפחות באחת השיטות; אדום: הוחמצה בכל הארכיון כשה-gate פתוח; אפור: gate סגור. אין חיתוך זנבות.</p><img src="BROAD_HISTOGRAMS.png" alt="Distributions of all PB error answers">',
        '<p><a href="BROAD_HISTOGRAMS.svg">גרף וקטורי להורדה</a> · <a href="HISTOGRAMS.png">אותה בדיקה מול 32 הצירופים בלבד</a></p>',
        '<h2>האם רואים שגיאה — או רק התחלה של צעד?</h2><p>קווים מלאים: הצעד השגוי; מקווקווים: הצעד התקין שקדם לו באותה תשובה. אפס הוא תחילת הצעד, לא הטוקן השגוי המדויק. רק תשובות עם צעד קודם נכללות. כל זוג משתמש באותן תשובות בכל תא זמן. הקווים אחרי תחילת הצעד עשויים לחצות את סופו; רק מדידת ארבעת הטוקנים הראשונים בטקסט הוגבלה במפורש לאורך כל צעד.</p><img src="BOUNDARY_CONTROL.png" alt="Error versus correct step boundary profiles">',
        '<p><a href="BOUNDARY_CONTROL.svg">גרף וקטורי</a> · <a href="ALIGNED_PROFILES.png">פרופילים סביב השגיאה בקבוצות 32 הצירופים</a></p>',
        '<h2>לראות את התשובה עצמה</h2><p>בכל תרשים: חמשת הפיצ׳רים המתוקננים; תחזיות המנבאים מול התצפית; שאריות; ציוני הצעדים. האזור האדום הוא רק הצעד הראשון שסומן שגוי. כל הטוקנים מוצגים ללא החלקה. תחזית סקלרית היא ממוצע תחזיות חמשת הפיצ׳רים, ומשוחזרת בדיוק מהתצפית והשארית השמורה — אינה וקטור הפלט המלא של המודל. ציוני הצעדים תוקננו רק לתצוגה, ללא שינוי ההחלטות.</p>']
    cases=[dict(b['case'],name='common_miss',figure='CASE_COMMON_MISS.png')]+d['cases']
    body+=['<label>תשובה להצגה: <select id="case-select">'+''.join('<option value="case'+str(i)+'">'+html.escape(c['name']+' — '+c['row_id'])+'</option>' for i,c in enumerate(cases))+'</select></label>']
    for i,c in enumerate(cases):
        body+=['<section class="case" id="case'+str(i)+'"'+(' hidden' if i else '')+'><h3>'+html.escape(c['row_id']+' / '+c['cell'])+'</h3><p>'+html.escape(c['question'])+'</p><img src="'+c['figure']+'" alt="Full feature and predictor token traces"><p>השגיאה הראשונה לפי התיוג: צעד '+str(c['first_error_step'])+'.</p><details><summary>טקסט כל הצעדים</summary><ol>'+''.join('<li'+(' class="error"' if q+1==c['first_error_step'] else '')+'>'+html.escape(st)+'</li>' for q,st in enumerate(c['steps']))+'</ol></details></section>']
    body+=['<p>בדוגמת math-404 גם הטיעון בצעד הקודם לשגיאה המתויגת בעייתי: התשובה מחליפה דרישה לשלמות מדויקת בקירוב. לכן יש להבחין בין טעות המיקום לפי הבנצ׳מרק לבין השאלה אם צעד מוקדם יותר כבר מכיל טיעון חשוד. זו הערת קריאה על דוגמה אחת, לא שינוי תוויות או אישור שציוני המערכת מבינים את הטיעון.</p>',
        '<h2>כיסוי הארכיון</h2>',table(['ארכיון','עמודות שנכללו','פגיעות מעבר ל-32 הצירופים'],[[v['archive'],sum(r['included'] for r in v['rows']),s['family_found_beyond_current32'][v['archive']]] for v in b['inventory']]),
        '<p>הפגיעות בין הארכיונים חופפות; אין לסכום את העמודה האחרונה. השוואת הייחוסים המשותפים היא מלאה לכל מערך צעדים; חתימות ופרטי כל זרוע נמצאים ב-BROAD_ANALYSIS.json. אין כאן טענה ליתרון של אחת הזרועות החלשות רק כי היא מוצאת דוגמה אחרת.</p>',
        '<p><a href="COMMON_MISSES.csv">כל מזהי ההחמצות המשותפות</a> · <a href="ANSWER_DIAGNOSTICS.csv">מאפייני כל תשובות השגיאה</a> · <a href="BROAD_ANALYSIS.json">הארכיון והסטטיסטיקה</a> · <a href="ANALYSIS.json">האבחון הראשוני</a> · <a href="AUDIT.json">בדיקות</a></p>']
    page='<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Common missed errors — Step390</title><style>body{font:17px/1.7 system-ui;max-width:1350px;margin:30px auto;padding:24px;background:#f5f8fa;color:#203440}img{width:100%;height:auto;background:white}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{border:1px solid #d0dce3;padding:8px;text-align:right}th{background:#dfedf3}h1,h2{color:#12536c}.scroll{overflow:auto}a{color:#086c91}.error{background:#ffe4e4}li{margin:12px 0;white-space:pre-wrap}select{font:inherit;padding:7px}.case[hidden]{display:none}p{max-width:115ch}</style>'+''.join(body)+'<script>document.getElementById("case-select").onchange=e=>document.querySelectorAll(".case").forEach(s=>s.hidden=s.id!==e.target.value);</script></html>'
    (OUT/'REPORT.html').write_text(page,encoding='utf8')
    (OUT/'REPORT.md').write_text('# Common missed errors — Step390\n\n'+'\n\n'.join(explanation)+'\n\n[הדוח עם התרשימים](REPORT.html)\n',encoding='utf8')
    # A compact figure suitable for showing directly in the conversation.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for q,(k,title) in enumerate([('error_step_fraction','Error-step fraction of answer'),('feature_change_l2','Feature change vs previous16 tokens (L2)'),('predictor_disagreement','Predictor disagreement at error step')]):
        allv=np.array([float(r[k]) for r in allrows if r[k] not in ('','None')]);edges=np.linspace(allv.min(),allv.max(),32)
        for flag,color,label in [(False,'#237c59','Found before gate:3557'),(True,'#c4403e','Missed even before gate:885')]:
            v=np.array([float(r[k]) for r in allrows if r[k] not in ('','None') and (r['uid'] in rawmiss)==flag]);ax[q].hist(v,bins=edges,weights=np.ones(len(v))/len(v),histtype='step',color=color,lw=2,label=label)
        ax[q].set(xlabel=title,ylabel='Fraction of available group / bin');ax[q].grid(alpha=.15)
    ax[0].legend(fontsize=8);fig.suptitle('Common localization misses across the full available archive; gate ignored; distributions overlap');fig.tight_layout();figure_save(fig,'RAW_OVERVIEW');plt.close(fig)
    with (OUT/'ANSWER_DIAGNOSTICS.csv').open(encoding='utf8') as f:rr=list(csv.DictReader(f))
    with (OUT/'COMMON_MISSES.csv').open(encoding='utf8') as f:miss={x['uid'] for x in csv.DictReader(f) if x['gate_open']=='True'}
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for q,key in enumerate(['error_step_fraction','predictor_disagreement']):
        for ismiss,color,label in [(False,'#237c59','Found in archive'),(True,'#c4403e','Missed by archive')]:
            v=np.array([float(r[key]) for r in rr if float(r['gate_percentile'])>=.33 and (r['uid'] in miss)==ismiss]);edges=np.linspace(0,1 if q==0 else .8,31)
            ax[q].hist(v,bins=edges,weights=np.ones(len(v))/len(v),histtype='step',lw=2,color=color,label=label)
        ax[q].set(xlabel='Error-step fraction of answer' if q==0 else 'Predictor disagreement',ylabel='Fraction of group / bin');ax[q].grid(alpha=.15)
    t=(np.array(d['event_bin_edges'][:-1])+np.array(d['event_bin_edges'][1:]))/2
    for c,color in [('0','#237c59'),('1','#c4403e')]:
        av=np.array(b['paired_boundary_means'][c])
        for kind,ls in [(0,'-'),(1,'--')]:ax[2].plot(t,av[kind,:,0],ls=ls,color=color,label=('Found' if c=='0' else 'Missed')+(' error' if kind==0 else ' previous correct'))
    ax[2].axvline(0,color='black',lw=.8);ax[2].set(xlabel='Tokens from step start',ylabel='H0lim (answer z-score)');ax[2].grid(alpha=.15)
    ax[0].legend(fontsize=8);ax[2].legend(fontsize=7);fig.suptitle('Gate-open PB errors: broad archive comparison; descriptive, overlapping distributions');fig.tight_layout();figure_save(fig,'OVERVIEW');plt.close(fig)
    state=json.loads((OUT/'RUN_STATE.json').read_text());state.update(status='COMPLETE_REVIEWED',broad_archives=s['included_archives'],broad_score_columns=s['scored_columns'],common_raw_misses=s['missed_raw_regardless_gate'],no_training=True,descriptive_only=True);write(OUT/'RUN_STATE.json',state)
    write(OUT/'ARTIFACTS.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir() if p.is_file() and p.name not in ['ARTIFACTS.json','COMMIT_PATHS.json']}))

if __name__=='__main__':run()
