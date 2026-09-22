"""Standalone comparison of all frozen predictor subsets and selection limits."""
from pathlib import Path
import sys,json,html,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_predictor_subset_study import OUT,sha,write
from spectral_utils.predictor_subset_fusion import METHODS,SUBSETS,subset_name

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def run():
    state=json.loads((OUT/'RUN_STATE.json').read_text());audit=json.loads((OUT/'AUDIT.json').read_text())
    if state['status']!='COMPLETE_REVIEWED' or audit['status']!='PASS':raise ValueError('Full review required')
    data=json.loads((OUT/'METRICS.json').read_text());metrics=data['metrics']
    review=json.loads((OUT/'REVIEW.json').read_text()) if (OUT/'REVIEW.json').exists() else {}
    ids={n:f'C{i+1:02}' for i,n in enumerate(METHODS)}
    ranked=sorted(METHODS,key=lambda n:(-metrics[n]['pb_all8'],-metrics[n]['prm_within'],n))
    rows=[[ids[n],n,len(n.split('__')[1].split('+')),f"{100*metrics[n]['pb_all8']:.4f}",
        f"{metrics[n]['prm_within']:.6f}",f"{metrics[n]['prmscore_q08']:.6f}"] for n in ranked]
    refs=('original4','innovation5','ridge','tcn__real','bocpd','noreset','mean16')
    parts=[
        '# כל צירופי המנבאים — IU-PCR מול מיצוע, Step389',
        'כל13,769 התשובות /145,597 צעדים /6,968,779 טוקנים.32 תצורות חדשות ו-16 ייחוסים. כל האימון הקודם נוצל מחדש; לא אומנו מנבאים חדשים.',
        *review.get('paragraphs_he',[]),
        '## המובילים בסריקה',
        table(['תחום','מוביל PB','מוביל within'],[['כל התצורות',data['leaders']['pb_all8'],data['leaders']['prm_within']]]+
            [[k+' מנבאים',v['pb_all8'],v['prm_within']] for k,v in data['leaders_by_size'].items()]+
            [[k,v['pb_all8'],v['prm_within']] for k,v in data['leaders_by_head'].items()]),
        'הבחירה נעשית באמצעות תוויות הפיתוח. המובילים הם אומדני נקודה בסריקה הזאת; אין כאן אישור על שאלות חדשות. בשוויון משתמשים במדד האחר, אחר כך PRMScore, פחות מנבאים ולבסוף סדר שמות קבוע.',
        '## כל32 התצורות',
        table(['מזהה','שיטה וצירוף','מספר מנבאים','PB F1 %','within-AUC','PRMScore'],rows),
        '## ייחוסים ללא מיזוג חדש',
        table(['שיטה','PB F1 %','within-AUC','PRMScore'],[[n,f"{100*metrics[n]['pb_all8']:.4f}",f"{metrics[n]['prm_within']:.6f}",f"{metrics[n]['prmscore_q08']:.6f}"] for n in refs]),
        '## תרומת IU מול מיצוע של אותו צירוף',
        '16 השוואות ראשיות כפול שני מדדים;10,000 דגימות bootstrap מזווגות לפי קבוצות מקור. רווחי סמך99.84375% בתיקון Bonferroni. רווח הכולל אפס אינו הוכחת שקילות.',
        table(['צירוף','הפרש PB בנקודות אחוז','CI PB','הפרש within','CI within'],[
            [subset_name(s),f"{100*c['pb_delta']:+.4f}",str([round(100*x,4) for x in c['pb_ci']]),
                f"{c['prm_within_delta_common']:+.6f}",str([round(x,6) for x in c['prm_within_ci']])]
            for s in SUBSETS for c in [data['contrasts']['iu__'+subset_name(s)+'_minus_equal__'+subset_name(s)]]]),
        'ב-METRICS.json מופיעות גם השוואות כל32 התצורות מול Ridge,TCN,BOCPD עם רווחי סמך95% חקרניים. הן אינן מתוקנות לבחירת המוביל ולכן אינן אישור ליתרונו לאחר הסריקה. כל התוצאות משתמשות באותה אוכלוסיית פיתוח היסטורית וב-seed0.',
        '## כיצד נעשה המיזוג',
        'בכל טוקן מחשבים לכל מנבא שארית חתומה על אותם חמשת הפיצ׳רים. מתקננים כל עמודת שארית בתוך התשובה. IU מתאים covariance ממורכז ומפיק משקלים בשני הרכיבים הספקטרליים הראשיים; המיצוע מקבל בדיוק אותן עמודות מתוקננות. מכוונים את סימן הציון לכיוון ציון השארית הממוצע לפי covariance ללא תוויות. המשקלים קבועים בתוך תשובה ויכולים להשתנות בין תשובות; אין routing לפי טוקן בניסוי הזה.',
        'ממזגים ברמת הטוקן, מסכמים Top10 לכל צעד, ומוסיפים תיקון חתום.25 לבסיס innovation5 פעם אחת. ה-gate נשאר tail15 באחוזון.33. אין ממוצע חוזר של כמה עותקים של הבסיס ואין שינוי סדר Top10 בין IU לבקרת המיצוע.',
        'Ridge ו-TCN משתמשים במודלים שנלמדו מקבוצות מקור אחרות ללא תוויות נכונות. BOCPD,noreset,mean16 והמשקול משתמשים בתשובה עצמה. הנרמול התשובתי והמיקום היחסי מגדירים מערכת offline. הכיול המקונן משתמש בכל10 זוגות ה-folds, עם שני המנבאים הנלמדים מותאמים ללא שתי קבוצות ה-folds המוחזקות בחוץ.',
        '## אבחון משקלים',
        'המשקלים להלן פועלים על שאריות מתוקננות, לא ביחידות הפיצ׳רים המקוריות. ערכים שליליים מותרים ב-IU הקנוני. בשלושה מנבאים שארית ההתאמה הזוגית מתאפסת לפי בניית המערכת ואינה ראיה לתקפות ההנחות.',
        table(['צירוף','משקלים חציוניים לפי סדר השמות','שיעור תשובות עם משקל שלילי','שארית זוגית חציונית','שיעור g2 בתקרה'],[
            [n,str([round(x,4) for x in v['median_weights']]),f"{v['negative_weight_answer_fraction']:.3f}",
                f"{v['pair_residual_quantiles'][1]:.6f}",f"{v['ceiling_fraction']:.3f}"] for n,v in data['mechanism']['subsets'].items()]),
        '## בדיקות ושחזור',
        f"כל48 השיטות עברו חישוב עצמאי של PB ו-within. כל16 כותרות הייחוס שוחזרו. ה-readout של כל32 התצורות שוחזר באופן עצמאי מסכום התרומות ומיון הטוקנים: פער מרבי {audit['max_independent_readout_delta']:.3g}. פער מרבי במשקלים מול IU הקנוני בבדיקת50 תשובות ללא תוויות: {audit['max_canonical_weight_delta']:.3g}. ציוני חמשת המנבאים הבודדים שוחזרו בכל15 עבודות הניקוד, כולל הכיול.",
        f"זמן קיר לניקוד: {state['seconds']/60:.2f} דקות, עם עד3 תהליכים חד-תהליכוניים במקביל; זמן הערכה: {state['evaluation_seconds']:.2f} שניות. שלוש בדיקות חדשות עברו, כולל45 מטריצות covariance לבדיקת התאמה למימוש הקנוני. אין החלפה שקטה של כשל במיצוע.",
        'מערכי השאריות, המשקלים וקובצי SQLite נשמרים מקומית עם חתימות. דיווח לפי תא, לכל48 השיטות, נמצא ב-METRICS.json. תור ה-flows נשאר מושהה.',
        '[פרוטוקול](../../docs/experiments/PREDICTOR_SUBSET_IU_20260915.md) · [מדדים ורווחי סמך](METRICS.json) · [אימות](AUDIT.json) · [טבלת CSV](METRICS.csv)'
    ]
    if (OUT/'ERROR_ANALYSIS.json').exists():
        e=json.loads((OUT/'ERROR_ANALYSIS.json').read_text());v=e['overlap']
        parts+=['## אילו שגיאות הוחמצו',
            f"ניתוח תיאורי לאחר הבחירה: מתוך {v['error_answers']} תשובות PB עם שגיאה, {v['missed_by_all32']} הוחמצו בכל32 הצירופים. ב-{v['missed_all_gate_closed']} ה-gate סגר את התשובה; ב-{v['missed_all_gate_open']} הוא היה פתוח אך אף צירוף לא בחר את הצעד המתויג. אלה תשובות מודל, לא שאלות מקור עצמאיות. האיחוד בין השיטות הוא אורקל לבחירת פסגה קיימת בלבד.",
            '[ספירות ודוגמאות עם טקסט הצעדים](ERRORS.html) · [כל התחזיות והמזהים](PB_ERROR_LEDGER.csv) · [אבחון מלא](ERROR_ANALYSIS.json).']
    (OUT/'REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    for head,marker,color in [('iu','o','#1769a3'),('equal','s','#dc861c')]:
        names=[n for n in METHODS if n.startswith(head+'__')]
        ax.scatter([metrics[n]['prm_within'] for n in names],[100*metrics[n]['pb_all8'] for n in names],
            s=55,marker=marker,color=color,label=head,alpha=.8)
    for n in ('ridge','tcn__real','bocpd','innovation5'):
        x=metrics[n]['prm_within'];y=100*metrics[n]['pb_all8']
        ax.scatter([x],[y],marker='D',s=50,color='#273541');ax.annotate(n,(x,y),xytext=(5,5),textcoords='offset points',fontsize=8)
    for n in set(data['leaders'].values()):
        ax.annotate(ids[n],(metrics[n]['prm_within'],100*metrics[n]['pb_all8']),xytext=(5,-12),textcoords='offset points',fontsize=10,fontweight='bold')
    ax.set(xlabel='PRMB within-answer AUC',ylabel='ProcessBench macro F1 (%)',title='32 predictor combinations: development point estimates')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'PARETO.svg');fig.savefig(OUT/'PARETO.png',dpi=150);plt.close(fig)
    svg=OUT/'PARETO.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')
    body=[]
    for p in parts:
        if p.startswith('|'):
            lines=p.strip().splitlines()
            def row(line,tag):return '<tr>'+''.join('<'+tag+'>'+html.escape(v.strip())+'</'+tag+'>' for v in line.strip('|').split('|'))+'</tr>'
            body.append('<div class="scroll"><table>'+row(lines[0],'th')+''.join(row(x,'td') for x in lines[2:])+'</table></div>')
        elif p.startswith('## '):
            body.append('<h2>'+html.escape(p[3:])+'</h2>')
            if p=='## כל32 התצורות':body.append('<img src="PARETO.svg" alt="PB versus within for all32 combinations" style="width:100%;max-width:950px">')
        elif p.startswith('# '):body.append('<h1>'+html.escape(p[2:])+'</h1>')
        else:body.append('<p>'+re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',html.escape(p))+'</p>')
    page='<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Predictor subsets — Step389</title><style>body{font:17px system-ui;line-height:1.7;max-width:1200px;margin:35px auto;padding:0 20px;background:#f4f7fa;color:#203340}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #cedae0;padding:7px;text-align:right}th{background:#e1edf2;cursor:pointer}tr:nth-child(even){background:#f6f9fa}.scroll{overflow:auto}h1,h2{color:#174c68}a{color:#126b97}</style>'+''.join(body)
    page+='<script>document.querySelectorAll("table").forEach(t=>{t.querySelectorAll("th").forEach((h,i)=>h.onclick=()=>{let r=Array.from(t.querySelectorAll("tr")).slice(1),d=h.dataset.rev==="1"?1:-1;h.dataset.rev=d===1?"0":"1";r.sort((a,b)=>{let x=a.children[i].textContent,y=b.children[i].textContent;return d*(Number.isFinite(Number(x))&&Number.isFinite(Number(y))?Number(x)-Number(y):x.localeCompare(y));});r.forEach(x=>t.appendChild(x));});});</script></html>'
    (OUT/'REPORT.html').write_text(page,encoding='utf8',newline='\n')
    write(OUT/'ARTIFACTS.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir()
        if p.is_file() and p.name not in ('ARTIFACTS.json','QUEUE.lock','COMMIT_PATHS.json')}))

if __name__=='__main__':run()
