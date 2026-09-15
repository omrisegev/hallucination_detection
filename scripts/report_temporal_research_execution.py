"""Publish a reviewable execution snapshot from completed artifacts only."""
from pathlib import Path
import html
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.temporal_research_features import BASELINE,SUBSETS


def read(path):return json.loads((ROOT/path).read_text(encoding='utf8'))


def render_document(text):
    """Render only the small heading/paragraph/list/table subset used here."""
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
        return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    blocks=[]
    for block in text.strip().split('\n\n'):
        lines=block.splitlines()
        if lines[0].startswith('# '):blocks.append('<h1>'+inline(lines[0][2:])+'</h1>')
        elif lines[0].startswith('## '):blocks.append('<h2>'+inline(lines[0][3:])+'</h2>')
        elif lines[0].startswith('|'):
            rows=[]
            for i,line in enumerate(lines):
                if i==1:continue
                tag='th' if i==0 else 'td'
                rows.append('<tr>'+''.join('<'+tag+'>'+inline(x.strip())+'</'+tag+'>' for x in line.strip('|').split('|'))+'</tr>')
            blocks.append('<table>'+''.join(rows)+'</table>')
        elif all(line.startswith('- ') for line in lines):blocks.append('<ul>'+''.join('<li>'+inline(line[2:])+'</li>' for line in lines)+'</ul>')
        else:blocks.append('<p>'+inline(' '.join(lines))+'</p>')
    return '\n'.join(blocks)


def main():
    b=read('results/temporal_research_baseline_v1/METRICS.json')['metrics']
    m=read('results/temporal_research_mechanism_v1/METRICS.json')['metrics']
    d=read('results/temporal_dufs31_v1/METRICS.json')['metrics']
    repair=read('results/temporal_historical_pb_repair_v2/REPAIR_REVIEW.json')
    assert repair['status']=='PASS' and repair['count']==47 and not repair['changed_headlines']
    names=[BASELINE,'append_innovation__H0lim','append_innovation__VE075','mean__VE0','mean__VE075','RBM12_logit',
           'append_duplicate_H0lim','append_centered_H0lim','append_shuffled_prefix_H0lim','dufs31_k2','dufs31_k3','dufs31_k4']
    metrics={**b,**m,**d};labels={BASELINE:'הרביעייה המקורית', 'append_innovation__H0lim':'הרביעייה + H0lim innovation',
        'append_innovation__VE075':'הרביעייה + VE0.75 innovation','mean__VE0':'VE0 בלבד','mean__VE075':'VE0.75 בלבד',
        'RBM12_logit':'RBM12 — אותו gate','append_duplicate_H0lim':'הוספת עותק H0lim',
        'append_centered_H0lim':'הוספת H0lim ממורכז','append_shuffled_prefix_H0lim':'הוספת innovation עם סדר מעורבב'}
    rows=[]
    for n in names:
        v=metrics[n];rows.append(f"| {labels.get(n,n)} | {100*v['pb_all8']:.4f}% | {v['prm_within']:.6f} | {v['prmscore_q08']:.6f} |")
    linear=ROOT/'results/temporal_linear_context_v1/METRICS.json'
    linear_text='מנבא לינארי: הרצה מלאה בתהליך; אין עדיין סיכום איכות סופי.'
    if linear.exists():
        lm=json.loads(linear.read_text())['metrics']
        lines=[]
        for bank in ('original4','innovation5'):
            name=bank+'__real__squared_residual_0.25';v=lm[name]
            lines.append(f"{bank}: PB {100*v['pb_all8']:.4f}%, within {v['prm_within']:.6f} (הזרוע הראשית).")
        linear_text='המנבא הלינארי סיים הערכה מלאה. '+' '.join(lines)+'''

הזרוע הראשית של שגיאה ריבועית על innovation5 אינה שיפור ברור: PB +0.3072
נקודת אחוז עם רווח סמך 97.5% שחוצה אפס, וירידת within של 0.002768
שרווח הסמך שלה שלילי. מטרת חיזוי טובה אינה בהכרח ציון איתור טוב.

זרוע האבחון הרשומה של שארית עם סימן, בתוספת שיורית במשקל .25, נתנה על
innovation5: PB 40.8472%, within 0.761620, PRMScore 0.641765. מול innovation5
לבדו: PB +1.0159 נקודת אחוז, רווח סמך משני 95% [+0.2877,+1.7480]; within
+0.001328, רווח סמך [-0.000435,+0.003095]. אלו השוואות משניות שנוספו לניתוח
אחרי צפייה בזרועות האבחון, ולא המבחן הראשי שנרשם.

מול אותו מנבא עם סדר ההיסטוריה מעורבב, רווחי הסמך כוללים אפס בשני המדדים.
לפיכך יש ראיה מועילה להתאמת הרקע המקומי, אך עדיין אין יתרון מבוסס לסדר
המדויק של ההשהיות. כל 38 חבילות PB של הניסוי עברו ביקורת אריתמטית עצמאית.
'''
    text='''# ביצוע תוכנית המחקר — 15.09.2026

השלב הראשון בוצע: שחזור עצמאי, תיקון הדיווח, השוואות ייחוס,
כל 15 תתי־הבנקים, innovation וביקורות מנגנון, ו־DUFS על 31 הזרמים הקיימים.
המשך תוכנית המודלים טרם הושלם. כל המספרים כאן הם תוצאות פיתוח על
13,769 תשובות, 145,597 צעדים ו־6,968,779 טוקנים; אין כאן אישור על שאלות חדשות.

## הבסיס והבאגים

עשרת קובצי המקור וחמשת קובצי החוזה תואמים לחתימות המקור. חישוב מחדש
מהנתונים משחזר בדיוק PB 37.474898261944%, within 0.7534358509472404,
ו־PRMScore 0.6344124357811041. זהו שחזור של המדדים מהנתונים;
ארכיוני הציונים החדשים המקוריים לא היו זמינים להשוואה ביטית.

כל 47 חבילות PB תוקנו בתוצרים חדשים, כולל פירוט התאים והפסגות שה־gate
מסתיר. אף ציון כותרת לא השתנה. שדות PRMB ההיסטוריים בתיקונים נשמרו
מהמקור, ואינם מוצגים כשחזור חדש שלהם. H1_native תוקן לאנטרופיה מותנית
ב־top15; אין להסיק מכך ממצא על אנטרופיית כל אוצר המילים.

ה־gate נשאר tail15 Top10 באחוזון .33 בתוך תא. זהו כיול טרנסדוקטיבי.
ביקורת כיול בקבוצות אחרות באותו q נתנה PB 37.4761%; דמיון המספרים כאן
אינו הופך את שני חוזי הגישה לזהים.

## תוצאות מרכזיות באותו gate

| שיטה | PB | within-AUC | PRMScore |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

H0lim innovation הוא ערך הפיצ׳ר פחות הממוצע של הטוקנים הקודמים בתשובה.
הטוקן הנוכחי אינו נכלל ברקע; בתחילת התשובה הערך אפס וסימון ההיסטוריה חסר.
הוא נוסף כזרם חמישי, עם Top10 נפרד וממוצע באותן יחידות. החישוב נבדק
מחדש באמצעות סכום מצטבר סקלרי לכל טוקן, בנפרד מהמימוש שהפיק את התוצאות.

מול הרביעייה המקורית: PB +2.3565 נקודות אחוז, רווח סמך פיתוח 95%
[+1.2973,+3.4287]; within +0.006857, רווח סמך [0.005072,0.008679].
החישוב משתמש ב־10,000 דגימות bootstrap מזווגות של קבוצות מקור.
הוא אינו כולל את כל אי־הוודאות של הבחירה ההסתגלותית לאורך המחקר.

## הסייגים שמשנים את ההמלצה

**אין עדיין יתרון PB מוכח על כלל בחירת הצעד ההיסטורי החזק.** בחירת הפסגה
המוקדמת בין VE0 ל־VE0.75 מגיעה ל־39.3857%. ההפרש לטובת innovation הוא
0.4457 נקודת אחוז, ורווח סמך פיתוח 95% הוא [-0.6910,+1.6031].
first_near_max בסף .25 על ציוני הרביעייה מגיע ל־39.0217%. אלו כללי בחירת
צעד ב־PB; הם אינם שינוי בציוני PRMB ואין לייחס להם שיפור within.
בחירת הצעד הארוך ביותר נתנה 35.1420%, והצעד הראשון 18.6938%.

**הרווח אינו אחיד לאורך התשובה.** בתשובות עם שגיאה ראשונה מוקדמת,
מספר הפגיעות אחרי gate עלה מ־400 ל־533; בשגיאות מאוחרות ירד מ־301 ל־256.
בסך הכול נוספו 294 פגיעות ואבדו 181. אין כאן תוויות טוקן, ואין סימון של
כל הצעדים שאחרי השגיאה הראשונה כשגויים. ניתוח זה מתאר את מיקום השגיאה
בתשובות, בנפרד מפרופילי הפיצ׳רים במיקום הטוקן בפועל.

המתאם הממוצע בתוך תשובה בין H0lim לבין ה־innovation שלו הוא כ־0.976.
התוספת משנה מעט את האות, אך יכולה לשנות משמעותית את הפסגה שנבחרת.
לכן מתאם גבוה אינו מספיק כדי לפסול פיצ׳ר, ושיפור זה עדיין עשוי לכלול
תיקון הטיית מיקום. בקרה ממורכזת ובקרה עם סדר מעורבב אינן משחזרות אותו.

DUFS בחר בעיקר a0.25 ו־ve1.5, ביציבות גבוהה בין folds, ובכל זאת הפסיד
לבסיס. יציבות בחירה והתאמה למבנה הלא־מתויג אינן ראיה לתועלת באיתור שגיאות.
לפי כלל הבנק שהוגדר, להמשך עוברים רק הרביעייה המקורית ובנק innovation5.
הבנק המורחב שולט בחזית PB/within של בנקי הפיצ׳רים שנבדקו במחזור זה;
אין בכך השוואה אחידה לכללי בחירת הצעד או הכרעה סופית על פריסה.

## מודלי ההקשר והכיוון החדש

'''+linear_text+'''

מומשו TCN, conditional flow matching ו־Diverging Flows, כולל אימון ללא
תוויות נכונות, הפרדת קבוצות, checkpoints לפי מטרת האימון ו־readout ששומר
את הבסיס. בדיקות קטנות עברו לאימון ולניקוד TCN ו־DiFlo; הן בדיקות תקינות
ועלות בלבד. אימון DiFlo ראשון עבור innovation5, seed 0, fold 0 הסתיים
אחרי 31,000 עדכונים; נבחר checkpoint מעדכון 28,000. הניקוד הסתיים עבור
כל 2,782 התשובות ב־fold המוחזק בחוץ. זו עדיין אינה הערכת איכות מלאה.

הופעל תור מקומי רציף למחזור seed 0: כל 90 ההתאמות המתוכננות, כולל
שני הבנקים, שלוש השיטות והכיול המקונן. התור מדלג על הרצות שהושלמו,
ממשיך מ־checkpoints ומפעיל הערכה מלאה בסוף. FM רגיל הוא ההרצה הבאה
אחרי DiFlo שהושלם, ולאחריו TCN. זו תמונת מצב; קובץ מצב התור המקושר
בהמשך מציין איזו הרצה פעילה כעת. התור המקורי טיפל רק באימון ובניקוד
DiFlo הראשון; לא היה בו המשך ליתר הסדרה. פער ההפעלה הזה תוקן.

ב־DiFlo ממומשות ענישות repel/curve ו־DOT ביחס לקצה שהמודל יצר בעצמו.
הניקוד אינו מקבל את התצפית העתידית. הפרעות PGD נוצרות ב־logits,
והפיצ׳רים נגזרים מחדש מהסתברויות תקפות. זו התאמה שלנו לטלמטריית reasoning;
המאמר אינו מוכיח שאקסטרפולציה היא שגיאה סמנטית. GMM/KDE היסטוריים אינם DiFlo.
תוקן גם הייחוס השגוי של המאמר: המחברים הם Tsakonas, Ivaldi ו־Mouret.

אני ממשיך לתת לכיוון הזה קדימות, אך לא מציע לבחור בו בשל החדשנות בלבד.
הוא צריך להוסיף מעבר ל־FM רגיל, מעבר לבנק innovation5 ומעבר להטיית מיקום.
הביקורות כוללות הקשר אמיתי, סדר היסטוריה מעורבב ואיפוס העבר. אלה ביקורות
בזמן ניקוד של אותו מודל, ואינן מחליפות מודלי ביקורת שאומנו מחדש.

## מה נשאר פתוח

סדרת TCN/FM/DiFlo המלאה, מספר seeds והכיול המקונן טרם הושלמו.
הוכנה מטריצה של עד 270 התאמות: 3 שיטות × 2 בנקים × 3 seeds × 15
הפרדות folds. הופעל המחזור המקומי של seed 0 בלבד, עם 90 התאמות מתוכננות;
יתר ה־seeds טרם הופעלו. זו אינה סריקת פרמטרים חדשה.
חיבור AIRCC נכשל ב־timeout; GraphTV הקודם נשאר בסטטוס לא ידוע.
לא הוגשה עבודה חדשה לקלאסטר. הגישה נחוצה לאימותו ולהאצת ההמשך.

טרם הושלמו במחזור הנוכחי: IU בהשוואת readout תואמת, shrinkage למספר
יעדים, fusion ורouting עם אילוצי simplex, Network Lasso, מפות בדרגה נמוכה,
CRBM מותנה בהקשר, שליליים באמצעות החלפת היסטוריות מותאמות, ובדיקות
LOCA ודגימת גרף לפי ההנחות שנקבעו. הן נשארות בתוכנית, ולא מסומנות
כתוצאות שליליות או כעבודה שהושלמה. אישור על שאלות חדשות ייעשה לאחר הקפאה.

## תוצרים שניתן לבדוק

- [שחזור ו־15 תתי־בנקים](../../results/temporal_research_baseline_v1/REPORT.md)
- [47 תיקוני PB](../../results/temporal_historical_pb_repair_v2/REPAIR_REVIEW.json)
- [ביקורות מנגנון ו־RBM תואם](../../results/temporal_research_mechanism_v1/REPORT.md)
- [השוואה מזווגת לכללי הצעד](../../results/temporal_feature_diagnostics_v1/READOUT_CONTRASTS.json)
- [מיקום, מתאמים והשלמה](../../results/temporal_feature_diagnostics_v1/FEATURE_DIAGNOSTICS.json)
- [יציבות DUFS](../../results/temporal_feature_diagnostics_v1/DUFS_STABILITY.json)
- [הקפאת הבנקים](../../results/temporal_dufs31_v1/BANK_SELECTION.json)
- [המנבא הלינארי](../../results/temporal_linear_context_v1/RUN_STATE.json)
- [תוצאות המנבא הלינארי והביקורות](../../results/temporal_linear_context_v1/REPORT.md)
- [אימון DiFlo הראשון](../../results/temporal_context_models_v1/diflo__innovation5__seed0__exclude0/RUN_STATE.json)
- [הניקוד שהושלם ל־DiFlo הראשון](../../results/temporal_context_models_v1/diflo__innovation5__seed0__exclude0/scoring/RUN_STATE.json)
- [מצב תור ההמשך המקומי](../../results/temporal_neural_queue_seed0_v1/RUN_STATE.json)
- [digest מאומת של Diverging Flows](../../papers/digests/diverging-flows-2602-13061v2.md)
- [החוזה המלא](../experiments/TEMPORAL_RESEARCH_PROGRAM_20260915.md)
'''
    out=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.md';out.write_text(text,encoding='utf8',newline='\n')
    # Small standalone SVG scatter: points have tooltips, axes show genuine units.
    points=[]
    for n in list(SUBSETS)+['append_innovation__H0lim','append_innovation__VE075','dufs31_k2','dufs31_k3','dufs31_k4']:
        v=metrics[n];x=60+(v['pb_all8']-.36)/.045*650;y=320-(v['prm_within']-.73)/.035*280
        color='#e17b22' if 'innovation' in n else '#8b62b7' if n.startswith('dufs') else '#297a91'
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}"><title>{html.escape(n)}: PB {100*v["pb_all8"]:.4f}%; within {v["prm_within"]:.6f}</title></circle>')
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 380" role="img" aria-label="PB versus within-AUC Pareto plot"><rect width="780" height="380" fill="white"/><path d="M60 30 V320 H740" fill="none" stroke="#526273"/>'
    for val in (36,37,38,39,40):
        x=60+(val/100-.36)/.045*650;svg+=f'<text x="{x}" y="345" text-anchor="middle" font-size="14">{val}%</text>'
    for val in (.73,.74,.75,.76):
        y=320-(val-.73)/.035*280;svg+=f'<text x="50" y="{y+4}" text-anchor="end" font-size="14">{val:.2f}</text>'
    svg+=''.join(points)+'<text x="390" y="372" text-anchor="middle">PB macro F1</text><text x="64" y="18">PRMB within-AUC</text></svg>'
    out.with_suffix('.svg').write_text(svg,encoding='utf8',newline='\n')
    body=render_document(text)
    page='<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ביצוע מחקר ציר הזמן — 15.09.2026</title><style>body{font:17px/1.7 system-ui,sans-serif;color:#243342;background:#f3f6fa;margin:0}main{max-width:1050px;margin:28px auto;background:white;padding:38px;border-radius:16px}h1,h2{line-height:1.35}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:9px;border-bottom:1px solid #dae2eb;text-align:right}th{background:#edf3f8}a{color:#136880}svg{max-width:100%;direction:ltr}code{direction:ltr;unicode-bidi:embed}pre{font:inherit}p{max-width:95ch}</style><main>'+body+'<h2>חזית הבנקים</h2><p>כחול: תתי־בנקים; כתום: innovation; סגול: DUFS. פרטי כל נקודה מופיעים בריחוף. אלו השוואות בנק באותו readout.</p>'+svg+'</main></html>'
    out.with_suffix('.html').write_text(page,encoding='utf8',newline='\n')
    print(out)


if __name__=='__main__':main()
