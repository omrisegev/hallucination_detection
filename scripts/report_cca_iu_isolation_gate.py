"""Independent score audit and Hebrew report for the bounded synthetic gate."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_cca_iu_isolation_gate import OUT,WORLDS,old,save,provenance
from scripts.render_cca_contextual_fusion_proposal import render


def pair_auc(y,s):
    p=s[y==1];n=s[y==0];d=p[:,None]-n
    return float(np.mean((d>0)+.5*(d==0)))


def main():
    rows=[];max_auc=0.;max_score=0.;checks=0
    for world in WORLDS:
        for seed in old.SEEDS:
            r=json.loads((OUT/'cases'/f'{world}_{seed}.json').read_text());rows.append(r)
            test=old._world(seed+100000,world,640)
            with np.load(OUT/'cases'/f'{world}_{seed}_arrays.npz',allow_pickle=False) as f:
                np.testing.assert_array_equal(f['labels'],test[4])
                for method,expected in r['auc'].items():
                    score=f['score__'+method];actual=pair_auc(test[4],score)
                    max_auc=max(max_auc,abs(actual-expected));assert abs(actual-expected)<1e-12
                    key='weight__'+method
                    if key in f:
                        w=f[key];np.testing.assert_allclose(w.sum(axis=1),1,atol=1e-12)
                        assert w.min()>=.75/6-1e-10 and w.max()<=.75/6+.25+1e-10
                        reconstructed=np.sum(w*test[0],axis=1)
                        delta=float(np.max(np.abs(score-reconstructed)));max_score=max(max_score,delta)
                        assert delta<1e-11
                    checks+=1
    assert json.loads((OUT/'PROVENANCE.json').read_text())==provenance()
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    methods=sorted(rows[0]['auc']);lookup={(r['world'],r['method']):r for r in summary['rows']}
    def comparison(a,b):
        result=[];rng=np.random.default_rng(151926);draw=rng.integers(0,20,(10000,20))
        for world in WORLDS:
            d=np.array([r['auc'][a]-r['auc'][b] for r in rows if r['world']==world])
            result.append(dict(world=world,method=a,baseline=b,delta=float(d.mean()),
                               ci95=np.quantile(d[draw].mean(axis=1),[.025,.975]).tolist()))
        return result
    contrasts=[]
    for a,b in [('population_context_full','oracle_rho_context'),
                ('oracle_full','population_context_full'),('dsp_full','random_full'),
                ('second_moment_full','linear_full'),('second_moment_full','second_moment_group'),
                ('second_moment_full','energy_full'),('history_square_only_full','linear_full'),
                ('energy_full','energy_group'),('dsp_full','dsp_group'),
                ('population_context_full','population_context_group')]:
        contrasts.extend(comparison(a,b))
    save(OUT/'AUDIT.json',dict(status='PASS',cases=60,score_bundles=checks,
        max_auc_delta=max_auc,max_weighted_score_delta=max_score,source_hashes_unchanged=True))
    save(OUT/'CONTRASTS.json',contrasts)
    passed=[m for m,g in summary['gates'].items() if all(g.values())]
    cca_passed=any(m in passed for m in ('linear_full','history_square_only_full','second_moment_full'))
    decision=('REVIEW_CCA_SYNTHETIC_PASS_BEFORE_REAL_DATA' if cca_passed else
              'STOP_CCA_FULL_RUN_CONTEXT_WEIGHTING_FEASIBLE' if passed else
              'STOP_FULL_REAL_DATA_GATE_NOT_PASSED')
    save(OUT/'DECISION.json',dict(status=decision,passing_methods=passed,
        scope='synthetic only; no PB/PRMB quality or automatic model promotion',gates=summary['gates']))
    lines=['# בדיקת היתכנות מבודדת: CCA ו־IU מותנה בהקשר',
           '**כל התוצאות במסמך סינתטיות. לא בוצע ניסוי איכות חדש ב־PB או ב־PRMB.**',
           '## מה נבדק',
           'שוחזר S0 ההיסטורי בשלושה עולמות וב־20 seeds לכל עולם. לאחר מכן נבדקו אותם נתוני הווה '
           'עם covariance אוכלוסייה ידוע, אומדני שכנים והקשר נתון או נלמד. היסטוריות העבר הן הרחבה '
           'סינתטית חדשה בת 16 תצפיות באותו משטר; אינן חלק מ־S0 ההיסטורי.',
           'המשקלים החדשים נפתרים על covariance מלא תחת simplex, עם τ=1 ו־η=.25. '
           'native IU בשני רכיבים, גבולות τ ו־ρ אמיתי מופיעים כאבחונים נפרדים. '
           'oracle פירושו מידע סינתטי ידוע על המשטר או המטרה; אינו אלגוריתם לפריסה.',
           '## החלטה',
           '`'+decision+'`',
           'זרועות שעברו את כל תנאי השיפור והבטיחות שהוגדרו: '+(', '.join(passed)+'.' if passed else 'אין.')+
           ' שלוש זרועות CCA לא עברו את שער השיפור. לא נפתח ניסוי איכות מלא של CCA, '
           'ולא החלפנו אותו אוטומטית במועמד חדש על נתונים אמיתיים.',
           '## מה למדנו מהמנגנון',
           '**יש היתכנות למשקול מותנה; אין כרגע הצדקה למורכבות של CCA בהגדרה שנבדקה.** '
           'בעולם האינפורמטיבי, covariance ידוע עם QP נותן .877768 לעומת .849700 סטטי. '
           'עם שכנים והקשר המשטר הנתון מתקבל .879375: אין כאן שחזור של ההפסד ההיסטורי. '
           'ההפרש בין השניים אינו מבודד רק רעש דגימה, משום שהאמידה המקומית כוללת shrinkage שנבחר ללא תוויות.',
           'ב־native IU עם covariance אוכלוסייה מתקבל .973425 בעולם האינפורמטיבי, אך .505853 '
           'בעולם עם nuisance. לכן כשל תחת nuisance יכול להישאר גם ללא רעש covariance. '
           'ה־QP השמרני מצמצם את הפגיעה אך אינו פותר זיהוי סמנטי: .736094 לעומת .739602 סטטי. '
           'PASS בבטיחות משמעו שהנזק בתוך המרווח המותר, ולא שהנזק אפס.',
           'ידיעת Cov(X,target) נותנת ל־QP .911706 ו־.819271, בהתאמה. הפער מאומדן IU '
           'ממקם מגבלה באומדן רגעי המטרה/סולמו; אין לייחס את כולו לרכיב יחיד או לטעון '
           'שהאורקל ניתן ללמידה ללא תוויות. הוא משתמש במידע סינתטי שאינו זמין במשימה.',
           'CCA ליניארי כמעט אינו משנה את הייחוס. ריבועים בעבר בלבד אינם מצילים אותו. '
           'CCA לרגעים שניים לומד תלות מוחזקת מסוימת, אבל מגיע רק ל־.852372. '
           'סיכום אנרגיה פשוט של העבר נותן .874187. העובדה ש־CCA נלמד אינה מספיקה '
           'להצדיק שימוש בו לצורך fusion. אין מכאן שלילה של כל CCA או של רגעים שניים.',
           'בקרת ערבוב שתי הקבוצות נותנת .873368 עם הקשר האנרגיה, קרוב ל־.874187 של QP מלא. '
           'הפער הקטן ויתרונו הסינתטי מפורטים בהשוואות; אין להציגו כשוויון מוכח או '
           'כשיפור משמעותי בנתונים אמיתיים. הכיוון הפשוט הוא מועמד הגיוני יותר לדיון הבא.',
           'השלב הבא המומלץ הוא אבחון ללא תוויות של יציבות C, rho והמשקלים בבנק האמיתי, '
           'בהקשר האנרגיה הפשוט מול מיקום ושכנים אקראיים. הוא טרם בוצע, משום שהשער '
           'הנוכחי אינו מאשר את מועמד CCA המקורי. אין לפתוח סריקת CCA כדי להתאים ל־20 seeds אלה.',
           '## שחזור המקור',
           'כל 60 הרשומות שוחזרו ותואמות עד דיוק נומרי; גם בדיקות המכניקה תואמות. '
           'מקורות S0 ותוצריו לא שונו. [אימות השחזור](LEGACY_REPLAY.json).',
           '## תוצאות ממוצעות — AUC סינתטי',
           '| שיטה | informative | null | coherent nuisance |\n|---|---:|---:|---:|']
    lines[-1]+='\n'+'\n'.join('| '+m+' | '+' | '.join(f"{lookup[(w,m)]['auc']:.6f}" for w in WORLDS)+' |' for m in methods)
    lines+=['## השוואות מבודדות',
            'הפרשים מוחלטים ב־AUC, 95% bootstrap מזווג של 20 seeds עם 10,000 דגימות. '
            'הרווחים אבחוניים, ללא תיקון לכל ריבוי ההשוואות; אין לפרשם כאישור שיפור על משימות reasoning.',
            '| מועמד פחות ביקורת | עולם | הפרש | CI 95% |\n|---|---|---:|---|']
    lines[-1]+='\n'+'\n'.join(f"| {r['method']} − {r['baseline']} | {r['world']} | {r['delta']:+.6f} | [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] |" for r in contrasts)
    lines+=['## שערי קבלה',
            'מול ייחוס סטטי תואם: שיפור informative של לפחות .005 וב־18/20 seeds; '
            'שינוי מוחלט ב־null עד .005; ירידת nuisance ממוצעת לכל היותר .005 וב־seed הגרוע לכל היותר .020.',
            '| זרוע | שיפור | ניצחונות | null | nuisance ממוצע | nuisance קצה |\n|---|---|---|---|---|---|']
    lines[-1]+='\n'+'\n'.join('| '+m+' | '+' | '.join('PASS' if v else 'FAIL' for v in g.values())+' |' for m,g in summary['gates'].items())
    lines+=['## אבחוני התאמה',
            '| עולם | שיטה | alpha ממוצע | שיעור g2 בתקרה | מתאמי CCA מוחזקים ממוצעים |\n|---|---|---:|---:|---|']
    for world in WORLDS:
        for method in ('oracle_full','dsp_full','random_full','linear_full','history_square_only_full','second_moment_full','energy_full'):
            t=[r['telemetry'][method] for r in rows if r['world']==world]
            corr=np.mean([v['held_component_correlations'] for v in t],axis=0).tolist() if 'held_component_correlations' in t[0] else []
            lines[-1]+=f"\n| {world} | {method} | {np.mean([v['alpha'] for v in t]):.3f} | {np.mean([v['g2_ceiling_fraction'] for v in t]):.3f} | {corr} |"
    lines+=['## מגבלות וייחוס',
            'אין טענה ש־S0 מקיים את הנחות הרגעים האדיטיביים. גם ידיעת המשטר אינה מבטיחה '
            'זיהוי אמינות ללא תוויות. ירידה ב־loss, מתאם CCA חיובי ושיפור סינתטי הם שלושה ממצאים שונים. '
            'η=.25 מגביל את עוצמת השינוי ולכן כישלון בהגדרה זו אינו הוכחה לכישלון בכל η. '
            'הגדלת תקציב, שינוי בנק או סריקת פרמטרים לא בוצעו.',
            '## אימות ותוצרים',
            f'אימות עצמאי של {checks} חבילות ציונים עבר: AUC מחושב בזוגות חיובי–שלילי, '
            'וציוני fusion משוחזרים מהמשקלים ומהתצפיות. בדיקות QP מול SLSQP, התאמה למימוש IU הקנוני, '
            'יחידות, זהות, בידוד ההווה מ־CCA ושכנות אקראית מופיעות בבדיקות הקוד.',
            '[פרוטוקול קפוא](../../docs/experiments/CCA_IU_ISOLATION_GATE_20260915.md) · '
            '[טבלה מלאה](SUMMARY.csv) · [השוואות](CONTRASTS.json) · [אימות](AUDIT.json) · '
            '[החלטה](DECISION.json) · [חתימות המקור](PROVENANCE.json)']
    md='\n\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(md,encoding='utf8')
    body,toc=render(md)
    page='<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>CCA IU — בדיקת היתכנות</title><style>body{font:17px/1.7 system-ui;margin:30px auto;max-width:1300px;padding:20px;color:#203344}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:8px;border:1px solid #ccd6dd;text-align:right}.table-wrap{overflow:auto}h2{margin-top:35px}a{color:#006c80}code{direction:ltr;unicode-bidi:isolate}nav{display:grid;gap:8px}</style><nav>'+toc+'</nav>'+body+'</html>'
    (OUT/'REPORT.html').write_text(page,encoding='utf8')
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.rglob('*') if p.is_file() and p.name not in ('ARTIFACT_HASHES.json','RUN_STATE.json')}
    save(OUT/'ARTIFACT_HASHES.json',hashes)
    save(OUT/'RUN_STATE.json',dict(state='COMPLETE_REVIEWED',decision=decision,cases=60,score_bundles=checks))
    print(decision,'PASS audit',checks,flush=True)


if __name__=='__main__':main()
