"""Render the frozen experiment and descriptive operational-error plots."""
from pathlib import Path
import sys,json,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectral_utils.alternative_views_fusion import METHODS,VIEWS,BANKS,HEADS
from scripts.run_alternative_views_fusion import OUT,sha,write
from scripts.evaluate_alternative_views_fusion import PRIMARY

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'

def run():
    data=json.loads((OUT/'METRICS.json').read_text());metrics=data['metrics'];comp=json.loads((OUT/'ERROR_COMPLEMENTARITY.json').read_text())
    def rows(names):return [[n,f"{100*metrics[n]['pb_all8']:.4f}",f"{metrics[n]['prm_within']:.6f}",f"{metrics[n]['prmscore_q08']:.6f}"] for n in names]
    headers=['Method','PB %','within-AUC','PRMScore']
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    colors=dict(equal='#777777',family_equal='#16834a',iu='#1b5dba',diag='#c47719',block='#9a39a8')
    for bank,marker in [('new7','o'),('augmented12','s')]:
        for h in HEADS:
            n=bank+'__'+h;m=metrics[n];x=m['prm_within'];y=100*m['pb_all8']
            ax.scatter(x,y,color=colors[h],marker=marker,s=70,label=n)
    for n in ('innovation5','digit025','tcn__real','tcn_digit_sum'):
        m=metrics[n];x=m['prm_within'];y=100*m['pb_all8'];ax.scatter(x,y,color='black',marker='x',s=70);ax.annotate(n,(x,y),xytext=(5,6),textcoords='offset points',fontsize=8)
    ax.set(xlabel='PRMB within-answer AUC',ylabel='PB macro F1 (%)',title='Fixed .25 correction: new views and covariance heads\nDevelopment estimates; gate unchanged')
    ax.grid(alpha=.2);ax.margins(x=.12,y=.1);ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3);fig.savefig(OUT/'QUALITY.png',dpi=160);plt.close(fig)
    names=comp['methods'];labels=[n.replace('raw__','') for n in names];phi=np.array(comp['pb_raw']['phi'])
    fig,ax=plt.subplots(figsize=(10,9),layout='constrained');im=ax.imshow(phi,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right');ax.set_yticks(range(len(labels)),labels)
    for i in range(len(labels)):
        for j in range(len(labels)):ax.text(j,i,f'{phi[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(phi[i,j])>.65 else 'black')
    ax.set_title('Correlation of raw localization failures on the same 4,442 PB errors\nOperational binary errors, before gate; descriptive development analysis')
    fig.colorbar(im,ax=ax,shrink=.75,label='Phi correlation');fig.savefig(OUT/'ERROR_CORRELATION.png',dpi=150);plt.close(fig)
    unique=np.array(comp['pb_raw']['row_hits_column_misses']);base_index=names.index('innovation5')
    gains=unique[:9,base_index];losses=unique[base_index,:9]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');x=np.arange(9)
    ax.bar(x-.18,gains,width=.36,label='View hits, innovation5 misses');ax.bar(x+.18,losses,width=.36,label='Innovation5 hits, view misses')
    ax.set_xticks(x,labels[:9],rotation=35,ha='right');ax.set(ylabel='PB error answers',title='Complementarity does not imply a good standalone detector\nRaw peaks before the shared gate');ax.legend(fontsize=8)
    fig.savefig(OUT/'UNIQUE_HITS.png',dpi=160);plt.close(fig)
    contrasts=[]
    for a,b in PRIMARY:
        c=data['contrasts'][a+'_minus_'+b]
        contrasts.append([a+' minus '+b,f"{100*c['pb_delta']:+.4f}",'['+', '.join(f'{100*v:+.4f}' for v in c['pb_ci'])+']',
            f"{c['prm_within_delta_common']:+.6f}",'['+', '.join(f'{v:+.6f}' for v in c['prm_within_ci'])+']'])
    diag=json.loads((OUT/'DIAGNOSTICS.json').read_text());weight_review={};weight_rows=[]
    for bank,idx in [('new7',6),('augmented12',11)]:
        selected=[d['banks'][bank] for d in diag if idx in d['banks'][bank]['live']]
        for h in ('iu','diag','block'):
            w=np.array([d['heads'][h]['weights'][idx] for d in selected]);s=data['mechanism'][bank][h]
            shares=[abs(d['heads'][h]['weights'][idx])/max(np.abs(d['heads'][h]['weights']).sum(),1e-12) for d in selected]
            item=dict(variable_digit_answers=len(w),digit_negative_fraction=float((w<0).mean()),digit_median_weight=float(np.median(w)),median_digit_absolute_weight_share=float(np.median(shares)))
            weight_review[bank+'__'+h]=item
            weight_rows.append([bank+'__'+h,f"{100*item['digit_negative_fraction']:.2f}%",f"{item['digit_median_weight']:.4f}",
                f"{100*item['median_digit_absolute_weight_share']:.2f}%",'-' if s['alpha_quantiles'] is None else f"{s['alpha_quantiles'][2]:.4f}",s['fallback_answers']])
    write(OUT/'WEIGHT_REVIEW.json',weight_review)
    error_rows=[]
    for n in ['add__'+v for v in VIEWS]+[b+'__'+h for b in BANKS for h in HEADS]:
        d=data['error_analysis'][n];r=d['digit025']
        error_rows.append([n,d['hits'],d['raw_hits'],d['common707_final'],r['gained'],r['lost']])
    pb=comp['pb_raw'];prm=comp['prm_answer_balanced'];erows=[]
    for j,n in enumerate(names):erows.append([n,f"{pb['fail_rate'][j]:.4f}",f"{phi[j,base_index]:.4f}",int(unique[j,base_index]),int(unique[base_index,j]),f"{prm['pairwise_ranking_loss'][j]:.4f}"])
    body='''<h1>Step395 — שבע תצוגות, IU-PCR ו־shrinkage</h1>
<p>16.09.2026 · כל 13,769 התשובות, 145,597 הצעדים ו־6,968,779 הטוקנים. 28 זרועות חדשות ו־30 ייחוסים. תוצאות פיתוח על אוכלוסייה שכבר שימשה למחקר.</p>
<p>השאלה היא האם התצוגות משלימות זו את שגיאותיה של זו ביחס לאותו יעד, והאם משקול נלמד מנצל זאת טוב יותר משילוב פשוט. מתאם נמוך בין פיצ׳רים אינו הוכחה לעצמאות שגיאות.</p>
<p><strong>הממצא: אין בניסוי יתרון ל־IU-PCR, עם או בלי shrinkage. בשני הבנקים IU פוגע ב־within לעומת equal, גם ברווחי הסמך המתוקנים. מתן משקל שווה למשפחות משפר within לעומת equal, אך עדיין מפסיד לתיקון הספרות לבדו. נמשיך להחזיק ב־digit025 וב־TCN+digit כמועמדי פיתוח; לא נקדם את הבנקים החדשים כתחליף.</strong></p>
<p>זהו ממצא על המתכונים שנבדקו, לא שלילה של כל fusion או כל shrinkage. בבנק החדש IU אינו משחזר את הכשל הקודם של החסרת ספרות כמעט בכל תשובה; בדרך כלל המקדם חיובי, אבל קטן. גם ייצוב שאכן משנה את המטריצה אינו יוצר יתרון משימתי.</p>
<p><a href="../../docs/reviews/fusion_insertion_map_2026-09-16.md">מפת נקודות השילוב מקצה לקצה והראיות ההיסטוריות</a> · <a href="../../docs/experiments/ALTERNATIVE_VIEWS_FUSION_20260915.md">הפרוטוקול הקפוא</a></p>
<h2>השוואת הייחוס והזרמים הבודדים</h2>'''
    body+=table(headers,rows(['original4','innovation5','tcn__real','digit025','digit1','tcn_digit_sum']+['add__'+v for v in VIEWS]))
    body+='''<p>add = הבסיס innovation5 בתוספת 0.25 כפול סטיית התקן שלו, כפול ציון העזר המתוקנן בתוך התשובה. הבסיס נוסף פעם אחת. TCN+digit sum הוא ייחוס משני מהניסוי הקודם ומשתמש בשני תיקונים; הוא אינו ביקורת תואמת אמפליטודה לזרועות החדשות.</p>
<h2>האם המשקול מועיל?</h2>
<p>new7 מכיל surprisal, rank, mass_above, gap, logtail15, logtail50, digit. augmented12 מוסיף את חמשת הזרמים הקיימים. equal משקלל פיצ׳רים שווה; family_equal משקלל משפחות מקור שווה. IU משתמש במטריצה המקורית; diag ו־block מפעילים IU אחרי shrinkage ליעד אלכסוני או יעד ששומר את הבלוקים של המשפחות.</p>'''
    body+=table(headers,rows([b+'__'+h for b in BANKS for h in HEADS]))+'<img src="QUALITY.png" alt="Quality comparison">'
    body+='''<p>בכל ההשוואות הללו: תקנון הטוקנים בתוך התשובה, covariance ממורכז, Top10 נפרד לכל זרם ואז משקול. וקטור המשקלים קבוע בתוך התשובה, ואינו routing משתנה לכל טוקן. ה־gate נשאר tail15 באחוזון .33. בניגוד לניסוי הבנק הקודם, זו תוספת מוגבלת לבסיס ולא החלפתו.</p>
<p>ה־shrinkage נאמד ללא תוויות משונות מכפלות הפיצ׳רים בחסימות רצופות של עד 16 טוקנים. זו היוריסטיקת ייצוב; חסימות סמוכות אינן בהכרח עצמאיות. יעד block הוא חלוקה לפי מקור הנדסי, לא גילוי קבוצות בעלות שגיאות עצמאיות.</p>
<h2>השוואות ראשיות</h2><p>10,000 דגימות bootstrap מזווגות של קבוצות מקור. CI של 99.6875% עבור שמונה השוואות ושני מדדים. תיקון זה אינו כולל את כל הבחירה ההסתגלותית בהיסטוריית הפרויקט. השוואות נוספות ב־METRICS.json הן משניות עם CI של 95%.</p>'''
    body+=table(['Contrast','Δ PB pp','CI PB','Δ within','CI within'],contrasts)
    body+='<h2>משקלים וייצוב</h2><p>סימן הספרות נבדק רק בתשובות שבהן זרם הספרות משתנה. מקדמי IU אינם מנורמלים לסכום 1; לכן מוצג גם חלק הספרות מתוך סכום הערכים המוחלטים של המקדמים. נרמול ציון העזר מבטל את הסקאלה הכוללת של המקדמים. סיכומי alpha הם על כלל התשובות; כשלי התאמה מסומנים במפורש.</p>'
    body+=table(['Head','Negative digit weight','Median digit coefficient','Median absolute share','Median alpha','Equal fallbacks'],weight_rows)
    body+='''<h2>שגיאות משותפות והצלות</h2><p>PB: השוואת הפסגה לפני ה־gate על אותן 4,442 תשובות שגויות. PRMB: הפסד דירוג על אותם זוגות של צעד שגוי וצעד נכון, עם משקל שווה לכל תשובה ו־0.5 בקשרים. אלו שגיאות תפעוליות; אינן בדיקה ישירה של שאריות רציפות מול המשתנה החבוי של U-PCR. ניתוחים אלו משתמשים בתוויות לאבחון בלבד.</p>'''
    body+=table(['View','PB raw miss rate','Failure phi vs base','Unique raw hits','Lost raw hits','PRMB ranking loss'],erows)
    body+='<img src="ERROR_CORRELATION.png" alt="Failure correlations"><img src="UNIQUE_HITS.png" alt="Unique hits and losses">'
    body+='''<p>קבוצת 707 ההחמצות הפתוחות היא קבוצה שנבחרה לפי כישלונות היסטוריים, ולכן ההצלות בה הן אבחון תיאורי. פסגת digit בתיקו של אפסים אינה ראיה חיובית לאי־הסכמה. אין להסיק מאיחוד פסגות חסם על fusion.</p>'''
    body+=table(['Method','Final hits','Raw hits','Recovered /707','Gained vs digit025','Lost vs digit025'],error_rows)
    body+='''<h2>נאמנות הנתונים והניסוי</h2>
<p>הזנבות חושבו מחדש כ־1 פחות סכום ההסתברויות השמורות. לא מחסירים logsumexp מלוג־הסתברויות שכבר נורמלו. נוסחת הזנבות נבדקה על כל הטוקנים; ציוני הספרות והבסיס שוחזרו במלואם. rank נעצר ב־50 כאשר הטוקן שסופק אינו בקאש. logtail משתמש ברצפה 1e-12; גרסת המסה הגולמית נשמרת כבקרה, משום ש־Top10 של log אינו זהה ל־log של Top10.</p>
<p>כל 58 חבילות PB ו־within נבדקו בחישוב עצמאי; 30 הייחוסים משתחזרים. PRMScore משתמש בכיול באחוזון .8 מקבוצות מקור אחרות; הזרועות החדשות מותאמות בתוך התשובה בלבד, והייחוסים העצביים שומרים את הכיול המקונן הקודם. חמש בדיקות מספריות עברו. תור FM/DiFlo נשאר ללא שינוי.</p>
<p>הנתונים והצינור הם offline; כיול ה־gate טרנסדוקטיבי. התאמה ללא תוויות אינה הופכת את בחירת הפיצ׳רים והניתוח על אוכלוסיית הפיתוח לאישור חיצוני.</p>
<details><summary>כל 58 התוצאות</summary>'''+table(headers,rows(list(metrics)))+'</details>'
    body+='''<p><a href="METRICS.csv">CSV</a> · <a href="METRICS.json">מדדים ורווחי סמך</a> · <a href="PB_ERROR_LEDGER.csv">כל ההחמצות והפסגות</a> · <a href="ERROR_COMPLEMENTARITY.json">שגיאות לפי תא ומיקום</a> · <a href="AUDIT.json">בדיקות</a> · <a href="MANIFEST.json">חתימות ומקורות</a></p>'''
    page='<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Step395 Alternative Views Fusion</title><style>body{font:17px/1.65 Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#1c2837}table{direction:ltr;border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef2f7}img{max-width:100%;height:auto}p{max-width:1000px}a{color:#165ead}h2{margin-top:40px}details{overflow:auto}</style>'+body+'</html>'
    (OUT/'REPORT.html').write_text(page,encoding='utf8')
    state=json.loads((OUT/'RUN_STATE.json').read_text());state.update(status='COMPLETE_REVIEWED',report_sha256=sha(OUT/'REPORT.html'));write(OUT/'RUN_STATE.json',state)
    print(json.dumps(weight_review,indent=2),flush=True)

if __name__=='__main__':run()
