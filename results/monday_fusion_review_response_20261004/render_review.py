"""Render the independent, post-review analysis; never modify experiment outputs."""
from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
D = pd.read_csv(P/'DIRECT_CONTRASTS.csv')
A = pd.read_csv(P/'ABSOLUTE_METRICS.csv')
M = pd.read_csv(P/'MARKS_UNITS.csv')
audit = json.loads((P/'AUDIT.json').read_text(encoding='utf8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
def save(fig, name):
    fig.savefig(P/(name+'.png'), dpi=150, bbox_inches='tight')
    fig.savefig(P/(name+'.svg'), bbox_inches='tight')
    plt.close(fig)
recipes = ['ALL_EQ','DS_FILTER_EQ','MEAN_EQ','MEAN_DS','MEAN_SML','MEAN_MOM','HEM_EQ','HEM_DS','HEM_SML','HEM_MOM']
variants = ['remove_2','remove_3','random_2','random_3']
fig, axes = plt.subplots(1,2,figsize=(12,6.1),layout='constrained')
for ax, benchmark, title in zip(axes,['prm','pb'],['PRMBench: within-answer AUC','ProcessBench: first-error accuracy']):
    t=D[(D.experiment=='level')&(D.comparison=='same_recipe_removal')&(D.benchmark==benchmark)]
    z=t.pivot(index='recipe',columns='variant',values='delta').loc[recipes,variants].to_numpy()*100
    lim=np.max(np.abs(z));im=ax.imshow(z,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(4),['Target -2','Target -3','Random -2','Random -3']);ax.set_yticks(range(10),recipes);ax.set_title(title)
    for (i,j),v in np.ndenumerate(z):ax.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(v)>lim*.55 else '#151515')
    fig.colorbar(im,ax=ax,shrink=.7,label='Change × 100 vs original, same recipe')
fig.suptitle('LEVEL removal: direct effects, equal weight to all 8 banks\nPoint estimates; negative = lower performance',fontsize=13)
save(fig,'LEVEL_DIRECT')

fig,axes=plt.subplots(1,2,figsize=(12,7.3),layout='constrained')
choices=[('channels_answer','Channel votes / answer quota'),('groups_cell','Group votes / cell quota'),('channels_cell','Channel votes / cell quota')]
colors=['#087f8c','#c36a12','#7c43a5']
for ax,bench,title in zip(axes,['prm','pb'],['PRMBench: AUC change × 100','ProcessBench: first-error change (pp)']):
    for i,r in enumerate(['MEAN_DS','MEAN_MOM','HEM_DS','HEM_MOM']):
        for j,(v,label) in enumerate(choices):
            q=D[(D.experiment=='input')&(D.recipe==r)&(D.variant==v)&(D.reference_variant=='groups_answer')&(D.benchmark==bench)].iloc[0]
            x=q.delta*100;y=i+(j-1)*.22
            ax.errorbar(x,y,xerr=np.array([[x-q.lo95*100],[q.hi95*100-x]]),fmt='o',color=colors[j],capsize=3,label=label if i==0 else None)
    ax.axvline(0,color='#333',lw=1);ax.set_yticks(range(4),['MEAN_DS','MEAN_MOM','HEM_DS','HEM_MOM']);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2)
axes[0].legend(loc='lower left',fontsize=8)
fig.suptitle('Estimator inputs vs group votes / answer quota\nPaired source-question bootstrap, 95% unadjusted intervals; 8-bank mean',fontsize=12)
save(fig,'INPUT_DIRECT')

fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for r,col in [('HEM_MOM','#087f8c'),('HEM_EQ','#b45d1a')]:
    vals=[A[(A.experiment=='level')&(A.recipe==r)&(A.variant==v)].prm_auc.item() for v in ['original','random_2']]
    axes[0].plot([0,1],vals,'o-',label=r,color=col)
    for x,y in enumerate(vals):axes[0].annotate(f'{y:.5f}',(x,y),xytext=(4,7),textcoords='offset points')
axes[0].set_xticks([0,1],['Original','Random -2']);axes[0].set_xlim(-.2,1.35);axes[0].set_ylabel('PRMBench within-answer AUC');axes[0].legend();axes[0].set_title('A larger gap can partly reflect a weaker control')
base=A[(A.experiment=='input')&(A.recipe=='HEM_MOM')&(A.variant=='groups_answer')].iloc[0]
alt=A[(A.experiment=='input')&(A.recipe=='HEM_MOM')&(A.variant=='channels_cell')].iloc[0]
vals=[(alt[k]-base[k])*100 for k in ['pb_first_error','pb_f1','pb_clean_flag']]
axes[1].barh(['First-error accuracy','Official F1','Clean-answer flag rate'],vals,color=['#087f8c','#b45d1a','#b45d1a'])
axes[1].axvline(0,color='#333');axes[1].set_xlim(-.85,1.45)
for i,v in enumerate(vals):axes[1].text(v+(.03 if v>0 else -.03),i,f'{v:+.3f}',va='center',ha='left' if v>0 else 'right')
axes[1].set_xlabel('Change in percentage points; q80 decision rule');axes[1].set_title('HEM_MOM: channels / cell vs groups / answer\nHigher clean-answer flag rate is worse',fontsize=10)
save(fig,'SELECTION_AND_TRADEOFF')

def fmt(x):return f'{x:+.3f}'
def row(exp,recipe,variant,bench,comparison=None):
    t=D[(D.experiment==exp)&(D.recipe==recipe)&(D.variant==variant)&(D.benchmark==bench)]
    if comparison:t=t[t.comparison==comparison]
    return t.iloc[0]
numeric=[]
for exp,rec,v,comp,label in [
 ('level','ALL_EQ','remove_2','same_recipe_removal','ממוצע כל הערוצים: הסרה מכוונת של 2'),
 ('level','DS_FILTER_EQ','remove_3','same_recipe_removal','ממוצע אחרי סינון: הסרה מכוונת של 3'),
 ('level','HEM_MOM','random_2','same_recipe_removal','HEM_MOM: בקרת הסרה אקראית של 2'),
 ('input','MEAN_DS','channels_answer','channel_vs_group','MEAN_DS: הצבעות ערוצים במקום קבוצות'),
 ('input','HEM_MOM','channels_answer','channel_vs_group','HEM_MOM: הצבעות ערוצים במקום קבוצות'),
 ('input','HEM_MOM','channels_cell','both_changes','HEM_MOM: ערוצים ומכסה תאית במקום קבוצות ומכסה לתשובה'),
 ('input','MEAN_MOM','channels_cell','quota_channels','MEAN_MOM: מכסה תאית במקום מכסה לתשובה, בהצבעות ערוצים')]:
    vals=[]
    for b in ['prm','pb']:
        q=row(exp,rec,v,b,comp)
        vals.append(f'<span dir="ltr">{fmt(q.delta*100)} [{fmt(q.lo95*100)}, {fmt(q.hi95*100)}]</span>')
    numeric.append('<tr><td>'+label+'</td><td>'+vals[0]+'</td><td>'+vals[1]+'</td></tr>')
marks=M[M.kind=='clean'].groupby('variant')[['mean_marks_per_answer_column','zero_answer_column_fraction']].mean()
marksrows=''.join(f'<tr><td dir="ltr">{v}</td><td>{t.mean_marks_per_answer_column:.3f}</td><td>{t.zero_answer_column_fraction*100:.3f}%</td></tr>' for v,t in marks.iterrows())
claims=[
 ('המספרים שחושבו תקינים','אושר בהיקף הבדיקה הנוכחית','6,480 צבירות חוזרות מתוך תוצאות לתשובה; סטייה מרבית 3.33e−16. זו אינה הרצה חוזרת מהפיצ׳רים הגולמיים.'),
 ('אין להציג את random_2 כשיטה שנבחרה מראש','מסכים','הוגדר כבקרה בתוכנית. קובץ הפרוטוקול לא החריג במפורש את זכאותו לבחירה, והקוד הכניס אותו: עמימות שלא היה צריך לנצל לקידום בקרה. בפועל 12 מתוך 30 מועמדים אקראיים, לא 40 מתוך 40.'),
 ('הסרה מכוונת משנה גם עוגן','העוגן השתנה, אך בדיקת ההמשך לא מצאה השפעה שלו על הסימן','תיקון בעקבות שאלת עמרי: העוגן אינו קובע את החלוקה. ב־720 התאמות SML לא הפך סימן כלל; ב־720 התאמות MoM הכיוון זהה לכיוון לפי סכום חיובי ולפי רוב חיובי. לכן אין כרגע ראיה שהחלפת העוגן מסבירה את הירידה. ראו ANCHOR_CLARIFICATION_20261004.json.'),
 ('השאלות הישירות של הניסויים לא נותחו','מסכים: השמטה בניתוח המקורי','ההשוואות הישירות נוספו כאן, באותו מתכון מול גרסת המקור. הניתוח המאוחר מסומן כחקרני ולא כרישום מראש.'),
 ('הסרת LEVEL פוגעת כמעט תמיד','אושר בהערכות הנקודתיות','כל 20 השוואות ההסרה המכוונת ב־PRMBench שליליות; כל 40 השוואות ההסרה ב־ProcessBench שליליות. לא כל רווח סמך שולל אפס. שתי החריגות החיוביות ב־PRMBench הן HEM בבקרת random_2.'),
 ('שינוי קלט האומדנים אינו מוסיף דבר','הכללה רחבה מדי','נכון שאין שיפור גורף. נמצאו שיפורים ב־HEM_MOM ובאיתור שגיאה ראשונה, לצד ירידה ב־F1. העדר מובהקות אינו הוכחת שוויון.'),
 ('כלל הבחירה מערבב בקרות','מסכים: פגם בקוד הבחירה','DS_FILTER_EQ מוכנס עם יתרון אפס, ואילו משקולים מדורגים מול EQ קבוצתי תואם. יש לתקן את ההגדרה בגרסה חדשה; לא לשכתב SELECTION קפוא.'),
 ('המכסה התאית כמעט אינה משנה דבר','אושר חלקית בלבד','ממוצע הסימונים בתשובות תקינות כמעט אינו משתנה. היסטוגרמות הן של תשובה×עמודה, ולא מאפשרות למדוד כמה תשובות שלמות נותרו ללא סימון. קיימות השפעות מותנות קטנות על הביצועים.'),
 ('השילוב מחבר את שני הזוכים העצמאיים','לא זה המימוש','הקוד בוחר את קלט האומדנים בכפוף למתכון LEVEL: random_2/HEM_MOM עם channels_answer/HEM_MOM. הוא כולל BASE, LEVEL_ONLY, INPUT_ONLY ו־COMBINED. אין בכך הוכחה שכדאי לקדם אותו.'),
 ('אף שיטה אינה עוברת Step 457','יש לדווח, ולהפריד בין כללי הבחירה','לא נמצאה שיטה העוברת את הכלל ההיסטורי. המשתמש ביקש לדווח עליו לצד כלל ממוצע 8 הבנקים. הפיכתו לתנאי וטו תהיה שינוי נוסף בתוכנית, שאינו נובע מעצם הממצא.'),
 ('הריצה נעצרה לפני LR והשוואות מתחרים','אושר','26/648 יחידות מיקום והחלטה; LR, שילוב והשוואות PRM ו־Mind the Gap טרם הושלמו. אין לבחור כלל החלטה על סמך החלק המצומצם.')]
claimrows=''.join('<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in r)+'</tr>' for r in claims)
report='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>בדיקת הביקורת — ניסויי יום שני</title>
<style>body{font:17px/1.65 system-ui,Arial,sans-serif;background:#f4f6f9;color:#18212e;max-width:1200px;margin:auto;padding:28px}h1,h2{line-height:1.3}section{background:white;padding:24px;margin:22px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #cad2dd;padding:10px;text-align:right;vertical-align:top}th{background:#e9eff6}a{color:#1258a1}img{width:100%;background:white;border:1px solid #d1d8e1}code{direction:ltr;unicode-bidi:embed} .note{border-right:5px solid #b87518;padding:12px;background:#fff6df}small{color:#465366} @media(prefers-color-scheme:dark){body{background:#151c28;color:#e9edf6}section{background:#222d3d}th{background:#34445b}a{color:#9ecbff}.note{background:#443821}small{color:#c5d0df}}</style>
<h1>בדיקת הביקורת: מה מאומת, מה דורש תיקון ומה נשאר להחלטת עמרי</h1><p>4 באוקטובר 2026 · ניתוח עצמאי של התוצאות השמורות · כל הנתונים נתוני פיתוח שכבר נחשפו</p>
<section><p><b>עיקר הביקורת על בחירת ה״זוכים״ מוצדק. עם זאת, אין בסיס למסקנה גורפת ששינוי קלט האומדנים אינו מועיל: ההשפעה תלויה במתכון ובמדד.</b> השמטת ההשוואות הישירות וערבוב בקרות הבחירה הן טעויות בניתוח המקורי. שינוי העוגן והדירוג לאחר תקנון לתשובה, לעומת זאת, היו חלק מהתוכנית שאושרה.</p>
<p>נבדקו 2,232 קבצים מול hashes חתומים ונחזרו 6,480 ערכי מדדים. ההפרש המרבי הוא 3.33×10⁻¹⁶. בוצעו 20,000 דגימות bootstrap מזווגות לפי שאלת מקור, עם אותן דגימות בין בנקים ומתכונים. PRMBench: ‏6,030 תשובות כשירות ל־AUC, ‏707 קבוצות מקור; ProcessBench: ‏4,442 תשובות שגויות משני המודלים, ‏2,842 קבוצות מקור באוכלוסיית הדגימה. השקלול שווה לשמונת הבנקים ולשמונת תאי ProcessBench.</p>
<p class="note">גבולות הבדיקה: הצבירה העצמאית התחילה ממדדים ותוצאות שמורים לכל תשובה; לא שחזרתי כאן ציוני צעדים מפיצ׳רים גולמיים, ולא הרצתי מחדש את בדיקת ערבוב התוויות של קלוד. הקבצים המקוריים לא שונו. רווחי הסמך החדשים הם ניתוח חקרני לאחר הביקורת.</p></section>
<section><h2>בדיקת הטענות</h2><table><thead><tr><th>טענה</th><th>מסקנת הבדיקה</th><th>ראיה והשלכה</th></tr></thead><tbody>CLAIMS</tbody></table></section>
<section><h2>ההשוואות הישירות החסרות</h2><p>כל מספר הוא הפרש מול הבקרה הישירה, כפול 100; בסוגריים רווח 95% לא מתוקן. ב־AUC אלו נקודות בסולם 0–100, ובדיוק אלו נקודות אחוז. אין לערבב בין שני המדדים.</p><table><thead><tr><th>השוואה</th><th>PRMBench: ‏AUC</th><th>ProcessBench: דיוק השגיאה הראשונה</th></tr></thead><tbody>NUMERIC</tbody></table>
<p>ב־HEM_MOM, מעבר מקבוצות ומכסה לתשובה לערוצים ומכסה לתא מעלה את דיוק השגיאה הראשונה ב־1.062 נקודות אחוז; גם רווח מתוקן ל־40 ההשוואות באותו ניסוי ובאותו מדד נשאר חיובי: [0.118, 2.062]. ה־AUC ב־PRMBench אינו מראה שיפור מובהק באותה השוואה. תחת כלל q80, ‏F1 יורד ב־0.529 נקודות אחוז ושיעור הדגלים בתשובות תקינות עולה ב־0.441. לשני המדדים האחרונים חושב הפרש נקודתי בלבד בבדיקה זו.</p>
<p>התיקון הוא Bonferroni חקרני בתוך כל משפחת ניסוי ומדד בנפרד, לא תיקון משותף לכל המדדים במחקר. אומדן זנבות רווחים מתוקנים על בסיס 20,000 דגימות הוא קירוב. כל 192 שורות ההשוואה, לרבות הרווחים המתוקנים, נמצאות ב־CSV.</p></section>
<section><h2>הסרת LEVEL: אותה שיטה לפני ואחרי ההסרה</h2><img src="LEVEL_DIRECT.png" alt="שינוי ביצועים לפי מתכון וגרסת הסרה"><p>בכחול שיפור ובאדום ירידה. אלו ממוצעים על כל שמונת הבנקים, ולא טענה שכל בנק או כל רווח סמך שליליים. גם ALL_EQ ו־DS_FILTER_EQ, שאינם תלויים בעוגן של החלוקה, נפגעים. לכן שינוי העוגן אינו יכול להסביר לבדו את כל הירידה.</p><a href="LEVEL_DIRECT.svg">SVG</a></section>
<section><h2>קלט האומדנים: התוצאה תלויה במשקול ובמדד</h2><img src="INPUT_DIRECT.png" alt="השוואות קלט האומדנים עם רווחי סמך"><p>הבקרה בכל הנקודות היא אותו מתכון עם הצבעות קבוצות ומכסה לתשובה. הקווים הם רווחי 95% לא מתוקנים. ‏MEAN ו־HEM מתארים את השקלול בתוך הקבוצות; DS ו־MoM את האמידה למשקול בין הקבוצות.</p><a href="INPUT_DIRECT.svg">SVG</a></section>
<section><h2>מדוע יתרון מול EQ אינו מספיק לבחירת גרסה</h2><img src="SELECTION_AND_TRADEOFF.png" alt="התנהגות הבקרה והחלפה בין דיוק ל־F1"><p>ב־random_2, ‏HEM_MOM עצמו משתפר ב־0.001341 AUC, ובמקביל HEM_EQ נחלש בכ־0.002147. לכן נכון שחלק מהגדלת היתרון נובע מהחלשת הבקרה; לא נכון שכל ההגדלה נובעת מכך. זו עדיין תצפית בבקרה אקראית, ולא הצדקה להכתיר שיטה שנבחרה מראש. ב־ProcessBench אותה הסרה אקראית מורידה את דיוק השגיאה הראשונה בכ־0.570 נקודות אחוז.</p><a href="SELECTION_AND_TRADEOFF.svg">SVG</a></section>
<section><h2>המכסה התאית: יחידת הספירה חשובה</h2><table><tr><th>גרסה</th><th>סימונים בממוצע לתשובה×עמודה תקינה</th><th>יחידות תשובה×עמודה עם אפס סימונים</th></tr>MARKS</table><p>ממוצע שווה של הבנקים ושל MEAN/HEM; ‏934 תשובות תקינות ב־PRMBench. ההיסטוגרמות השמורות מאגדות עמודות. לכן 0.3%–4% אינם שיעור התשובות התקינות ללא שום דגל בכל הערוצים יחד. אי אפשר לחשב את השיעור הזה מההיסטוגרמות לבדן. התוצאות תומכות בטענה שהעברת המכסה כמעט אינה מקטינה את כמות הסימונים הממוצעת בתשובות תקינות; הן אינן מודדות כמה ביטים התחלפו או מוכיחות שהקלט נשאר זהה.</p></section>
<section><h2>החלטות מול התוכנית המאושרת</h2><ol>
<li><b>כלל הבחירה:</b> לשמור את כלל היתרון מול EQ תואם שביקשת, אך לדרג את איכות הבנק וההסרה בהשוואה ישירה למקור; או לעבור לקריטריון בחירה משותף מול הממוצע המקורי. האפשרות השנייה היא שינוי בכלל הבחירה לאחר חשיפת התוצאות. בכל מקרה הבקרות האקראיות יישארו אבחוניות, ולא יתקדמו אוטומטית לשיטה סופית.</li>
<li><b>המתכון המשולב:</b> התוכנית דורשת להריץ אותו, וקלוד מציע לשקול ויתור. הצעתי היא לדחות את קידום השילוב עד לתיקון הבחירה והשלמת LR ושכבות ההחלטה. אין הצדקה לבטל בשמך ניסוי שאישרת, ואין כרגע שני שיפורים מבוססים שאפשר פשוט לצרף.</li>
<li><b>בקרות עוגן ומכסה חדשות:</b> תיקון בעקבות שאלת עמרי: בדיקת הקוד והסימנים אינה תומכת כרגע בצורך בניסוי עוגן כדי להסביר את הירידה. העוגן אינו משפיע על החלוקה ולא שינה סימן לעומת כללי ההכוונה הכלליים במקרים שנבדקו. דירוג שאינו מתוקנן לכל תשובה נותר הצעה לניסוי חדש, ולא תיקון המימוש הקיים.</li>
<li><b>כלל ההחלטה:</b> עדיין לבחירת עמרי מתוך שלוש החלופות לאחר השלמת ההרצות. שיפור באיתור שגיאה ראשונה אינו מחליף את טבלת F1, ‏PRMScore ודגלים בתשובות תקינות.</li></ol>
<p>המשך טכני: נדרש תחילה מקום בדיסק; בתחילת בדיקה זו היו כ־27MB בלבד. הריצה נותרה STOPPED. לא נמחקו קבצים ולא חודשו הרצות. יש לשמור את התוצאות והפרוטוקולים הקפואים ולהוציא תיקון ניתוח בגרסה חדשה. רק אחרי השלמת שכבות ההחלטה, LR ובחירת מתכון ניתן להשלים את ניתוח המתחרים על אותו מתכון.</p></section>
<section><h2>קבצים ושחזור</h2><p><a href="DIRECT_CONTRASTS.csv">השוואות ישירות ורווחי סמך</a> · <a href="ABSOLUTE_METRICS.csv">מדדים מוחלטים</a> · <a href="ANCHOR_AUDIT.csv">בדיקת העוגנים</a> · <a href="MARKS_UNITS.csv">יחידות הסימונים</a> · <a href="AUDIT.json">היקף הבדיקה ו־hashes</a> · <a href="recheck.py">קוד הבדיקה</a> · <a href="render_review.py">יצירת הדוח</a></p>
<p><a href="ANCHOR_CLARIFICATION_20261004.json">תיקון ובדיקת השפעת העוגן בעקבות שאלת עמרי</a> · <a href="../monday_fusion_review_claude_20261004/REVIEW.md">דוח קלוד</a> · <a href="../monday_fusion_analysis_v1/code/docs/experiments/MONDAY_FUSION_ANALYSIS_20261003.md">תוכנית העבודה</a> · <a href="../level_reduction_v1/run_20261003/PROTOCOL.json">פרוטוקול LEVEL</a> · <a href="../estimator_input_v1/run_20261003/PROTOCOL.json">פרוטוקול קלט האומדנים</a></p>
<small>הענף שנבדק: codex/monday-fusion-analysis-20261003, ‏HEAD ‏8a485642e. ריצה מחדש של recheck.py דורשת את קובצי המקור המקומיים; render_review.py משתמש רק בקובצי הבדיקה כאן. התרשימים כאן הם ניתוח ביקורת משלים, ואינם תחליף להשלמת F1–F12 של הניסוי.</small></section></html>'''
report=report.replace('CLAIMS',claimrows).replace('NUMERIC',''.join(numeric)).replace('>MARKS</table>','>'+marksrows+'</table>')
(P/'REVIEW_RESPONSE_HE.html').write_text(report,encoding='utf8')
print('Rendered Hebrew HTML and 3 figures in PNG + SVG.')
