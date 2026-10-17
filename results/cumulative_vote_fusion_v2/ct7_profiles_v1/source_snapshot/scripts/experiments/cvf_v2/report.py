"""Auditable CSV/JSON and a standalone Hebrew HTML report (no remote assets)."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from .core import CHANNELS,ARMS
from .data import dump,digest
from .scoring import collect,pb_metrics,prm_metrics,prmscores
from .uncertainty import bootstrap

def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def diagnostics(d,methods,jobs):
    summary=[];selections=[]
    for job in jobs:
        for name,m in job['models'].items():
            diag=m['diagnostics'];starts=diag.get('starts',[])
            summary.append({'task':job['task'],'fold':job['fold'],'roster':job['roster'],'population':job['population'],'arm':name,
              'status':m['status'],'orientation':m['orientation'],'small_groups':str(diag.get('small_groups',[])),
              'converged':diag.get('converged'),'selected_start':diag.get('selected_start'),
              'selected_iterations':starts[diag['selected_start']]['iterations'] if starts else None,
              'start_likelihood_spread':max(x['log_likelihood'][-1] for x in starts)-min(x['log_likelihood'][-1] for x in starts) if starts else None,
              'job_seconds':job['seconds']})
        if job['stage']=='spectral':
            selections.extend({'task':job['task'],'fold':job['fold'],'roster':job['roster'],'population':job['population'],
              'channel':ch,'readout':r} for ch,r in zip(CHANNELS,job['readouts']))
    csv_write(d.out/'MODEL_DIAGNOSTICS.csv',summary);csv_write(d.out/'READOUT_SELECTIONS.csv',selections)
    stepcounts=np.diff(d.off);lengths=np.load(d.out/'step_lengths.npy');longest=d.peaks(lengths)
    strata=[]
    for name,m in methods.items():
        error=d.pb&(d.target>=0)&m['valid'];rel=np.divide(d.target,np.maximum(stepcounts-1,1))
        groups={'steps_1':stepcounts==1,'steps_2_to_5':(stepcounts>=2)&(stepcounts<=5),'steps_6_to_10':(stepcounts>=6)&(stepcounts<=10),'steps_11_plus':stepcounts>=11}
        for b in range(4):groups[f'error_relative_quarter_{b+1}']=(rel>=b/4)&(rel<((b+1)/4) if b<3 else rel<=1)
        for label,mask in groups.items():
            take=error&mask
            if take.any():strata.append({'method':name,'stratum':label,'n':int(take.sum()),'sla_micro_descriptive':float((m['pred'][take]==d.target[take]).mean()),
              'longest_step_agreement':float((m['pred'][take]==longest[take]).mean()),'mae':float(abs(m['pred'][take]-d.target[take]).mean())})
    csv_write(d.out/'LENGTH_POSITION_DIAGNOSTICS.csv',strata)
    return summary

def html_report(d,pb,pr,ps,uncertainty,diag,coverage):
    # Reviewed interpretation of this frozen run, not an algorithmic winner rule.
    # A different experiment must update its interpretation; never show stale prose.
    reviewed={
      'PB_METRICS.json':'ecbf4c2d778ee4cbfc5c27cafd5836818a5c4004d11fa73fbecb570f5469b776',
      'PRM_RANKING_METRICS.json':'766c80c8fd2b034354446c50d90be849e5ca85de9a5909d9bf56f5668a0adadb',
      'PRMSCORE.json':'19a2ba3c9fe2ab70b2a932bc3e83e230af1dae37c92cb3910017b12a95a9a768',
      'UNCERTAINTY.json':'b97c3a40d61b979958719ec5f811d31ffbfd9badfa33200ff29e9f851b001d19'}
    for name,expected in reviewed.items():
        if digest(d.out/name)!=expected:raise ValueError('Reviewed interpretation does not match '+name+'; update the interpretation for this run.')
    esc=html.escape
    def number(x,percent=False):return '—' if x is None or not np.isfinite(x) else (f'{100*x:.2f}%' if percent else f'{x:.4f}')
    def interval(endpoint,name,scale=100,digits=2):
        v=uncertainty['intervals'].get(endpoint,{}).get(name)
        return '—' if v is None else f'{scale*v["ci95"][0]:.{digits}f}–{scale*v["ci95"][1]:.{digits}f}'
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td dir="ltr">'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    pieces=['<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>מיזוג הצבעות — ניסוי מלא</title>',
      '<style>body{font:16px system-ui;background:#f5f7fa;color:#162438;max-width:1500px;margin:32px auto;padding:0 24px;line-height:1.65}h1,h2{line-height:1.3}section{background:white;padding:24px;margin:20px 0;border-radius:12px}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:14px}td,th{text-align:right;padding:8px 12px;border-bottom:1px solid #dde3eb}th{background:#eaf0f7;position:sticky;top:0}.scroll{overflow:auto;max-height:620px}code,.method{direction:ltr;unicode-bidi:embed}input,select{padding:9px;font:inherit}a{color:#1764a0}.note{border-right:4px solid #3b82a0;padding:12px;background:#edf6fa}details{margin:15px 0}</style>',
      '<h1>ניסוי מיזוג תחזיות בינאריות ורכות</h1>',
      '<p>ניסוי development מלא: 6,800 תשובות ProcessBench, ובנפרד 6,969 תשובות PRMBench. שתי גרסאות הייצוג, שתי משפחות EM, חמישה קיפולי שאלות מקור. לא נקבע סף הצלחה ולא נבחר מועמד אוטומטית.</p>',
      '<p class="note">SLA ללא gate הוא המדד הראשי ב־ProcessBench. עמודת F1 משתמשת באותן החלטות CT7 לכל שיטה. PRMBench בוחן שגיאות בכל צעד, וה־AUROC הראשי הוא ממוצע בתוך תשובה. אין להשוות ישירות בין סקלות המדדים.</p>',
      '<section><h2>הממצאים</h2><p>הרכות מועילה ביחס להצבעות בינאריות שוות: ב־Top5, ה־SLA עולה מ־21.25% ל־32.49% (דלתא 11.24 נקודות, CI95%: 9.41–13.12). DS מעלה את הבסיס הספקטרלי הבינארי מ־21.70% ל־32.98%, אך EM היררכי אינו מוסיף יתרון ברור עליו: 32.71%, ודלתת HEM−DS כוללת אפס ברווח הסמך.</p>',
      '<p>בחירת readout באימון מביאה את המיזוג הרך השווה ל־36.03% SLA ואת L-SML הרציף ל־36.18%. התוספת של למידת המיזוג היא 0.15 נקודות בלבד (CI95%: ‎−0.39 עד ‎+0.68). הפער מול CT7 נשאר ‎−3.70 נקודות SLA (CI95%: ‎−5.35 עד ‎−2.07). ה־F1 עם אותו gate הוא 38.31% מול 41.19%.</p>',
      '<p>ב־PRMBench, אותו רוסטר רך נבחר עם L-SML מגיע ל־0.7615 within-answer AUROC מול 0.7724 של CT7. הוספת EM לבסיס הספקטרלי הבינארי אינה משפרת כאן את ה־AUROC. PRMScore עם סף inner-fold: הרוסטר הרך השווה הנבחר 0.6439, L-SML הנבחר 0.6425, ו־CT7 0.6471; ערכי PRMScore הם אבחון משני ללא בדיקת מובהקות נפרדת.</p>',
      '<p>כל מודלי EM שנבחרו התכנסו, ולא נדרש fallback. זהו מימוש ישים שמאפשר להפריד בין הרכיבים, אבל בניסוי הנוכחי אין עדות ליתרון על CT7. עיקר השיפור ב־ProcessBench מגיע משימור מידע רך ומבחירת readout; התוספת של מיזוג נלמד מעל הרוסטר הנבחר אינה מובחנת מאפס.</p></section>',
      '<section><h2>השוואה ראשית: Top5 קבוע</h2>']
    names=['ct7','token_lsml','token_equal','mindgap']+[f'top5__all__{enc}__{kind}' for enc,kind in ARMS]
    def mainrows(names):
        rows=[]
        for n in names:
            b=pb[n]['macro8'];p=pr.get(n,{});s=ps.get(n,{})
            rows.append([f'<span class="method">{esc(n)}</span>',number(b['sla'],True),interval('pb_sla',n),number(b['f1'],True),interval('pb_common_gate_f1',n),number(p.get('within_auc')),interval('prm_within_auc',n,1,4),
              str(p.get('eligible','—')),number(s.get('quantile_0.8',{}).get('prmscore')),number(s.get('inner_selected',{}).get('prmscore'))])
        return rows
    headers=['שיטה','SLA מאקרו 8','95% CI ל־SLA (%)','F1 עם gate CT7','95% CI ל־F1 (%)','within-answer AUROC','95% CI ל־AUROC','זכאיות AUROC','PRMScore q0.8','PRMScore inner']
    pieces += [table(headers,mainrows(names)),'<p>רווחי הסמך משותפים לפי שאלת מקור; רשימת ההשוואות המזווגות נמצאת בהמשך. hard=בינארי; soft=רך; continuous_lsml על hard הוא בקרת הגישור. ערכי PRMScore הם עם מתאם המעריך הקיים, כולל validity fallback עבור circular/redundency.</p></section>',
      '<section><h2>שני scorers — בנפרד</h2>',
      table(['שיטה','Qwen3-4B SLA','Qwen3-4B F1','Qwen3-8B SLA','Qwen3-8B F1'],[[esc(n),number(pb[n]['q4']['sla'],True),number(pb[n]['q4']['f1'],True),number(pb[n]['q8']['sla'],True),number(pb[n]['q8']['f1'],True)] for n in names]),'</section>',
      '<section><h2>מה השתנה בכל רכיב</h2>']
    core_contrasts=[r for r in uncertainty['contrasts'] if r['a'].startswith('top5__all__') and r['contrast'] in ['soft_minus_binary','learned_minus_equal','em_minus_spectral_initialization','hierarchical_minus_ds','continuous_core_bridge']]
    rows=[[esc(r['endpoint']),f'<span class="method">{esc(r["a"])} − {esc(r["b"])}</span>',f'{100*r["delta"]:+.2f}',f'{100*r["ci95"][0]:+.2f} … {100*r["ci95"][1]:+.2f}',f'{r["p_holm"]:.4g}'] for r in core_contrasts]
    pieces += [table(['מדד','השוואה','הפרש ×100','95% CI ×100','p Holm'],rows),
      f'<p>{uncertainty["planned_comparisons"]} השוואות מתוכננות תוקנו יחד בשיטת Holm. 10,000 דגימות bootstrap; רזולוציית p מינימלית {uncertainty["minimum_raw_p"]:.5f}. אלה אינן בדיקות אישוש על נתונים חדשים.</p></section>',
      '<section><h2>כל הרוסטרים, בקרות וערוצים בודדים</h2><p>selected נבחר מתוויות האימון בלבד; errors משתמש באימון על שגויות בלבד; shuffle משבש סדר טוקנים בכל ערוץ ומתאים את המודלים מחדש. shuffle הוא בקרת מנגנון בזרע אחד.</p><input id="filter" placeholder="סינון לפי שם שיטה" oninput="filterRows()">',
      '<div id="allmethods">'+table(headers,mainrows(list(pb)))+'</div></section>',
      '<section><h2>פירוט לפי scorer ו־subset ב־ProcessBench</h2>']
    cellrows=[]
    for n,b in pb.items():
        for cell,v in b['cells'].items():cellrows.append([esc(n),esc(cell),f'{v["n"]}/{v["total_n"]}',str(v['errors']),number(v['sla'],True),number(v['f1'],True),number(v['tolerance_one'],True),number(v['early'],True),number(v['late'],True),number(v['mae'])])
    pieces += [table(['שיטה','תא','כיסוי','שגויות','SLA','F1','טולרנס 1','מוקדם','מאוחר','מרחק'],cellrows),'</section>',
      '<section><h2>PRMBench — דירוג צעדים בתוך כל fold</h2>']
    native=json.loads((d.out/'HISTORICAL_NATIVE_GATES.json').read_text(encoding='utf8'))
    pieces += [table(['שיטה','תשובות','זכאיות','within AUROC','step AUROC ממוצע folds','step AUPRC ממוצע folds'],[[esc(n),p['answers'],p['eligible'],number(p['within_auc']),number(p['step_auroc_mean_folds']),number(p['step_auprc_mean_folds'])] for n,p in pr.items()]),
      '<p>הסף הנבחר משתמש ב־50 quantiles ובתחזיות inner-fold. הוא נבחר לפי PRMScore באימון ומכויל מחדש על ציוני אימון המודל החיצוני; אין ערבוב סקלות בין מודלים לצורך AUC.</p>',
      '<h3>PRM מפוקח — גישה שונה</h3><p>Qwen2.5-Math-PRM-7B, סף rewards≥0.5: PRMScore '+number(ps['supervised_qwen25math_prm7b']['native_threshold_0.5']['prmscore'])+'. השורה אינה חלק מהשוואת המיזוג ללא תוויות.</p></section>',
      '<section><h2>כיסוי, fallbacks והתכנסות</h2>']
    pieces += [table(['שיטה','PB תקפות','PRMB תקפות','fallback PB','fallback PRMB'],[[esc(n),v['pb'],v['prm'],v['pb_fallback'],v['prm_fallback']] for n,v in coverage.items()]),
      '<p>טבלת האוכלוסייה המלאה משתמשת ב־equal של אותו קידוד רק כאשר המודל נכשל. native_sla ומכני הכיסוי נמצאים ב־PB_METRICS.json. פרמטרי EM הם אומדנים תחת הנחות המודל; קבוצות קטנות, כפילות ערוצים והסכמה גבוהה אינם ראיה לאמינות אמיתית.</p>',
      table(['משימה','fold','רוסטר','אוכלוסיית אימון','מודל','סטטוס','התכנס','איטרציות'],[[esc(str(r[k])) for k in ['task','fold','roster','population','arm','status','converged','selected_iterations']] for r in diag if r['converged'] is not None]),'</section>',
      '<section><h2>השוואות היסטוריות והקשר המאמר</h2><p>טבלת ה־gate המקורי נשמרת בנפרד בקובץ HISTORICAL_NATIVE_GATES.json. העוגנים 35.9237/32.5926 SLA ו־41.1887 F1 של CT7 שוחזרו לפני ההתאמה. לשיטות IU/Joint חסרות 21 תשובות היסטוריות; הכיסוי מוצג במפורש.</p>',
      table(['שיטה היסטורית','gate מקורי','F1 עם gate מקורי','F1 עם gate CT7'],[[esc(n),'CT7 frozen' if n=='ct7' else 'LOCO-5 frozen q=.33',number(native[n]['macro_f1'],True),number(pb[n]['macro8']['f1'],True)] for n in ['ct7','token_lsml','token_equal']]),
      '<p>Mind-the-Gap הראשי חושב מחדש מתוך top-k בקובצי המקור באמצעות adapter קיים: top20, EMA מותאם span5, שיא הירידה השלילית בצעד. mindgap_previous_unadjusted הוא הקובץ ההיסטורי עם EMA אחר. המאמר משתמש גם הוא ב־Qwen3 וב־teacher forcing ב־ProcessBench; ההבדלים שלא הושוו הם פרטי prompt/revision/אוכלוסייה וכלל token→step, שאינו מוגדר דיו במאמר. מספריו אינם שחזור שלנו.</p>',
      '<p>הקשר נפרד מטבלה 3 במאמר, Shannon Drop / Qwen3-8B: GSM8K 46.11%, MATH 32.90%, OlympiadBench 41.52%, Omni-MATH 37.04%. מקור: ה־PDF המקומי והטקסט שחולץ ממנו; אין שימוש במספרים אלה לבחירת הגדרות הניסוי.</p>',
      '<details><summary>נספח Llama מהניסוי הקודם</summary><p>שחזור ציוני התחזיות ההיסטוריות בלבד נשמר ב־LLAMA_APPENDIX.json. מדובר בפיילוט הישן, לא באוכלוסיית Qwen ולא בראיה השוואתית מהניסוי המלא. אין ערבוב שלו במאקרו שמונת התאים.</p></details></section>',
      '<section><h2>מגבלות והחלטת ההמשך</h2><p>כל האוכלוסייה כבר שימשה למחקר development. בחירת readout משתמשת בתוויות ומסומנת בנפרד. קיפולים וקיבוץ מקור מצמצמים זליגה בין שאלות, אך אינם הופכים נתונים שכבר נבדקו לנתוני אישוש. מודלי SML/EM מניחים מבנה תלות מוגבל; מספר הספים גדל עם אורך התשובה ולכן האימון מאוזן ברמת התשובה וה־subset.</p>',
      '<p>ב־Top5 הבינארי השווה 70.08% מתשובות PB השגויות מקבלות תחזית מוקדמת (מאקרו שמונת התאים), לעומת 31.45% בקידוד הרך. בקרת הצעד הארוך מגיעה ל־31.77% SLA; שיבוש הטוקנים משאיר 29.71% במיזוג הרך השווה ואף משפר חלק מהמודלים הספקטרליים החלשים. לכן אין לייחס את כל ההישג למבנה הטוקני. לעומת זאת, ב־PRMBench השיבוש מוריד את אותו AUROC מ־0.7532 ל־0.6108: יש בבנק מידע מקומי שימושי, אך המיזוג החדש אינו מנצל אותו טוב יותר מ־CT7.</p>',
      '<p>בדיקות התקינות כללו התאמה לחישוב brute-force, 1,150 מסלולי likelihood, בדיקת כל גבולות הקיפולים והפעלה מחדש של מודלים שמורים על 3,105 תחזיות בדיקה. תיקון מספרי של שוויון ב־mode שינה 665 תחזיות על פני כל הזרועות, בהתאם לכלל התיקו המוקדם שנקבע מראש; האינדקסים הקודמים נשמרו לביקורת. המספר אינו מספר תשובות ייחודיות.</p>',
      '<p>יש לבחון בנפרד את הדלתא של הרכות, למידת המשקולות, בחירת ה־readout ו־EM, ואת תלותה במשימה ובבקרות האורך/סדר. תוצאה טובה יותר במדד אחד אינה קובעת הצלחה במדד האחר. ההחלטה האם להמשיך בגישה נשארת בידיך; אין בדוח מנגנון לקידום שיטה.</p>',
      '<p>קבצים: <a href="SUMMARY.csv">סיכום CSV</a> · <a href="PAIRED_CONTRASTS.csv">כל ההפרשים המזווגים</a> · <a href="UNCERTAINTY.json">רווחי סמך ו־Holm</a> · <a href="PRMSCORE.json">המעריך הרשמי</a> · <a href="MODEL_DIAGNOSTICS.csv">אבחון מודלים</a> · <a href="LENGTH_POSITION_DIAGNOSTICS.csv">אורך ומיקום</a> · <a href="INPUT_FREEZE.json">קלטים מקובעים</a></p></section>',
      '<script>function filterRows(){const q=document.getElementById("filter").value.toLowerCase();document.querySelectorAll("#allmethods tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}</script></html>']
    (d.out/'REPORT_HE.html').write_text('\n'.join(pieces),encoding='utf8')

def report(d):
    methods,jobs=collect(d);pb={};pr={};within={};coverage={}
    for name,m in methods.items():
        pb[name]=pb_metrics(d,m)
        if (d.prm&m['valid']).any():pr[name],within[name]=prm_metrics(d,m)
        coverage[name]={'pb':int((d.pb&m['valid']).sum()),'prm':int((d.prm&m['valid']).sum()),
          'pb_fallback':int((d.pb&m['fallback']).sum()),'prm_fallback':int((d.prm&m['fallback']).sum())}
    dump(d.out/'PB_METRICS.json',pb);dump(d.out/'PRM_RANKING_METRICS.json',pr);dump(d.out/'COVERAGE.json',coverage)
    native={n:d.pb_metrics(d.peaks(d.references[n]),d.token_gate if n.startswith('token_') else d.gate) for n in ['ct7','token_lsml','token_equal']}
    dump(d.out/'HISTORICAL_NATIVE_GATES.json',native)
    ps=prmscores(d,methods)
    uncertainty=bootstrap(d,methods,within)
    diag=diagnostics(d,methods,jobs)
    np.savez_compressed(d.out/'OOF_STEP_SCORES.npz',offsets=d.off,labels=d.labels,**{k:v['scores'] for k,v in methods.items()})
    answer_rows=[]
    for i,r in enumerate(d.records):
        row={'uid':r['uid'],'id':d.ids[i],'source_group':d.groups[i],'fold':d.fold[i],'cell':d.cells[i],'target':d.target[i],'ct7_gate':int(d.gate[i])}
        for name,m in methods.items():row[name+'__mode']=int(m['pred'][i]) if d.pb[i] and m['valid'][i] else '';row[name+'__fallback']=int(m['fallback'][i])
        answer_rows.append(row)
    csv_write(d.out/'OOF_ANSWERS.csv',answer_rows)
    summary=[]
    for name in methods:
        b=pb[name]['macro8'];p=pr.get(name,{});s=ps.get(name,{})
        summary.append({'method':name,'pb_sla_macro8':b['sla'],'pb_f1_ct7_gate':b['f1'],
          'pb_sla_q4':pb[name]['q4']['sla'],'pb_sla_q8':pb[name]['q8']['sla'],
          **{k:p.get(k) for k in ['within_auc','eligible','step_auroc_mean_folds','step_auprc_mean_folds']},
          'prmscore_q80':s.get('quantile_0.8',{}).get('prmscore'),'prmscore_inner':s.get('inner_selected',{}).get('prmscore'),**coverage[name]})
    csv_write(d.out/'SUMMARY.csv',summary);csv_write(d.out/'PAIRED_CONTRASTS.csv',uncertainty['contrasts'])
    html_report(d,pb,pr,ps,uncertainty,diag,coverage)
    dump(d.out/'REPORT_MANIFEST.json',{'outer_jobs':len(jobs),'inner_jobs':len(list((d.out/'jobs').glob('*__inner*.json'))),
      'outer_fit_seconds':sum(j['seconds'] for j in jobs),'development_only':True,
      'jobs':{str(p.relative_to(d.out)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in (d.out/'jobs').iterdir() if p.is_file()},
      'outputs':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in d.out.iterdir() if p.is_file() and p.name not in ['REPORT_MANIFEST.json']}})
    print('Report complete: '+str(d.out/'REPORT_HE.html'),flush=True)

def render(d):
    """Re-render checked tables/prose without rerunning model fitting or evaluation."""
    def read(name):return json.loads((d.out/name).read_text(encoding='utf8'))
    with open(d.out/'MODEL_DIAGNOSTICS.csv',encoding='utf-8-sig',newline='') as f:diag=list(csv.DictReader(f))
    for row in diag:
        if not row['converged']:row['converged']=None
    html_report(d,read('PB_METRICS.json'),read('PRM_RANKING_METRICS.json'),read('PRMSCORE.json'),read('UNCERTAINTY.json'),diag,read('COVERAGE.json'))
    manifest=read('REPORT_MANIFEST.json')
    manifest['outputs']={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in d.out.iterdir() if p.is_file() and p.name!='REPORT_MANIFEST.json'}
    manifest['evaluation_code']={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    dump(d.out/'REPORT_MANIFEST.json',manifest)
    print('Rendered '+str(d.out/'REPORT_HE.html'),flush=True)
