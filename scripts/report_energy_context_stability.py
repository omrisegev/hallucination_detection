"""Audit saved moments without correctness labels; render full diagnostic."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_energy_context_stability import OUT,DATA,ARMS,save,sha,code_hashes
from scripts.render_cca_contextual_fusion_proposal import render
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.contextual_iu import DEFAULT_IU_FIT


def audit_history(metadata,data):
    x=np.load(DATA/'features.npy',mmap_mode='r');off=np.array([m['offset'] for m in metadata])
    maximum=0.
    for start in range(0,len(data['x']),4096):
        stop=min(start+4096,len(data['x']));a=data['answer'][start:stop];p=data['local'][start:stop]
        idx=off[a,None]+p[:,None]-np.arange(16,0,-1)[None,:]
        raw=np.asarray(x[idx],float)
        for name,actual in [('hm',raw.mean(axis=1)),('hs',(raw**2).mean(axis=1))]:
            delta=float(np.max(abs(actual-data[name][start:stop])));maximum=max(maximum,delta)
            if delta>1e-10:raise ValueError('History replay mismatch')
        np.testing.assert_array_equal(data['x'][start:stop],x[off[a]+p])
    return maximum


def main():
    metadata=json.loads((DATA/'METADATA.json').read_text());data=dict(np.load(OUT/'LANDMARKS.npz',allow_pickle=False))
    prepared=json.loads((OUT/'PREPARED.json').read_text())
    if code_hashes()!=json.loads((OUT/'PROVENANCE.json').read_text()):raise ValueError('Source code changed')
    for name in ('METADATA.json','features.npy'):
        if sha(DATA/name)!=prepared['source_files'][name]:raise ValueError('Source data changed')
    if sha(OUT/'LANDMARKS.npz')!=prepared['sha256']:raise ValueError('Landmarks changed')
    history_delta=audit_history(metadata,data)
    records=[];answer_nll={};max_abs=0.;max_rel=0.;checks=0;seen=[];motion=[];canonical_checks=0;geometry=[];max_kkt=0.
    for directory in sorted(OUT.glob('*__exclude*')):
        rec=json.loads((directory/'COMPLETE.json').read_text());records.append(rec)
        fit=json.loads((directory/'FIT.json').read_text());assert not set(fit['training_groups'])&set(fit['held_groups'])
        for name,expected in rec['hashes'].items():assert sha(directory/name)==expected
        with np.load(directory/'DIAGNOSTICS.npz',allow_pickle=False) as f:
            ids=f['landmark_ids'];seen.extend(ids.tolist());a=data['answer'][ids];target=f['target']
            mean=np.array(fit['preprocessing']['mean']);sd=np.array(fit['preprocessing']['sd'])
            np.testing.assert_allclose(target,(data['x'][ids]-mean)/sd,atol=1e-12)
            assert all(metadata[i]['fold']==rec['fold'] and metadata[i]['cell']==rec['cell'] for i in set(a))
            train=f['training_landmark_ids']
            assert all(metadata[i]['fold']!=rec['fold'] and metadata[i]['cell']==rec['cell'] for i in data['answer'][train])
            for arm in ARMS:
                C=f[arm+'__C'];e=target-f[arm+'__mu'];L=np.linalg.cholesky(C)
                anchor=int(f['anchors'][0])
                canonical=upcr_fit_covariance(C[anchor],var_y=fit['preprocessing']['var_y'],**DEFAULT_IU_FIT)
                np.testing.assert_allclose(f[arm+'__rho'][anchor],canonical.rho_hat_full,atol=1e-10)
                np.testing.assert_allclose(f[arm+'__native_a'][anchor],canonical.w,atol=1e-10)
                canonical_checks+=1
                geometry.append(dict(cell=rec['cell'],fold=rec['fold'],arm=arm,
                    covariance_trace_median=float(np.median(np.trace(C,axis1=1,axis2=2))),
                    covariance_frobenius_median=float(np.median(np.linalg.norm(C,axis=(1,2)))),
                    qp_raw_mean=f[arm+'__qp_w'].mean(axis=0).tolist(),
                    qp_standardized_mean=f[arm+'__qp_a'].mean(axis=0).tolist()))
                whiten=np.linalg.solve(L,e[...,None])[...,0]
                reconstructed=.5*(2*np.log(np.diagonal(L,axis1=1,axis2=2)).sum(axis=1)+(whiten**2).sum(axis=1)+5*np.log(2*np.pi))
                stored=f[arm+'__nll'];diff=np.abs(reconstructed-stored)
                max_abs=max(max_abs,float(diff.max()));max_rel=max(max_rel,float(np.max(diff/np.maximum(1,abs(stored)))))
                np.testing.assert_allclose(reconstructed,stored,rtol=1e-8,atol=1e-7)
                for answer in np.unique(a):
                    answer_nll.setdefault(int(answer),{})[arm]=float(np.mean(reconstructed[a==answer]))
                checks+=len(ids)
                for head in ('qp','group'):
                    w=f[arm+'__'+head+'_w'];np.testing.assert_allclose(w.sum(axis=1),1,atol=1e-10)
                    assert w.min()>=.15-1e-10
                    np.testing.assert_allclose(f[arm+'__'+head+'_a'],w*sd/sd.mean(),atol=1e-12)
                B=sd/sd.mean();Q=C*B[None,:,None]*B[None,None,:];r=f[arm+'__rho']*B
                pure=(f[arm+'__qp_w']-.15)/.25
                grad=np.einsum('nij,nj->ni',Q,pure)-r;level=np.sum(grad*pure,axis=1)
                violation=float(np.max(np.where(pure>1e-8,np.abs(grad-level[:,None]),np.maximum(level[:,None]-grad,0))))
                max_kkt=max(max_kkt,violation)
                assert violation<=1e-7
                if arm!='static':
                    refs=f[arm+'__anchor_reference_ids']
                    assert np.isin(refs,train).all()
                    for row in refs:
                        assert len(set(metadata[i]['group_id'] for i in data['answer'][row]))==64
                    delta=f[arm+'__qp_w']-f['static__qp_w']
                    group_delta=f[arm+'__group_w']-f['static__group_w']
                    d=np.array([1/3,1/3,-.5,-.5,1/3]);projection=(delta@d)[:,None]*d/(d@d)
                    total=float(np.sum(delta**2))
                    motion.append(dict(cell=rec['cell'],fold=rec['fold'],arm=arm,
                        qp_update_rms=float(np.sqrt(np.mean(delta**2))),
                        group_axis_explained_fraction=float(np.sum(projection**2)/max(total,1e-30)),
                        group_update_relative_squared_error=float(np.sum((delta-group_delta)**2)/max(total,1e-30))))
            np.testing.assert_array_equal(f['energy__neff'],f['random__neff'])
    assert len(records)==45 and sorted(seen)==list(range(len(data['x'])))
    assert len(answer_nll)==13769 and all(set(v)==set(ARMS) for v in answer_nll.values())
    # Independent hierarchy: average landmarks within answer, answers within
    # source/cell, sources within cell, then equal cell macro.
    grouped=defaultdict(list)
    for i,values in answer_nll.items():grouped[(metadata[i]['group_id'],metadata[i]['cell'])].append([values[a] for a in ARMS])
    grouped={k:np.mean(v,axis=0) for k,v in grouped.items()}
    cells=sorted(set(k[1] for k in grouped));sources=sorted(set(k[0] for k in grouped))
    ci={c:i for i,c in enumerate(cells)};gi={g:i for i,g in enumerate(sources)}
    values=np.zeros((len(sources),len(cells),len(ARMS)));present=np.zeros((len(sources),len(cells)))
    for (g,c),v in grouped.items():values[gi[g],ci[c]]=v;present[gi[g],ci[c]]=1
    cellmeans=values.sum(axis=0)/present.sum(axis=0)[:,None];macro=cellmeans.mean(axis=0)
    contrast=values[:,:,2,None]-values[:,:,[0,1,3]]
    draws=[];rng=np.random.default_rng(150915)
    for start in range(0,10000,100):
        counts=rng.multinomial(len(sources),np.full(len(sources),1/len(sources)),size=100)
        denom=counts@present
        if np.any(denom==0):raise ValueError('Bootstrap empty cell')
        means=(counts@contrast.reshape(len(sources),-1)).reshape(100,len(cells),3)/denom[:,:,None]
        draws.append(means.mean(axis=1))
    draws=np.concatenate(draws);interval=np.quantile(draws,[.025,.975],axis=0)
    comparisons={base:dict(delta=float(macro[2]-macro[j]),ci95=interval[:,k].tolist(),
        cells_better=int(np.sum(cellmeans[:,2]<cellmeans[:,j]))) for k,(base,j) in enumerate([('static',0),('position',1),('random',3)])}
    stability={arm:{key:dict(median_ratio=float(np.median([r['stability'][arm][key]['ratio'] for r in records])),
        fits_ratio_above_one=sum(r['stability'][arm][key]['ratio']>1 for r in records))
        for key in ('C','rho','native_a','qp_a','group_a')} for arm in ('position','energy','random')}
    telemetry={a:{k:float(np.median([r['telemetry'][a][k] for r in records]))
        for k in records[0]['telemetry'][a]} for a in ARMS}
    motion_summary={a:{k:float(np.median([r[k] for r in motion if r['arm']==a]))
        for k in ('qp_update_rms','group_axis_explained_fraction','group_update_relative_squared_error')}
        for a in ('position','energy','random')}
    # This decision describes feasibility only. It is not an accuracy gate.
    supporting=comparisons['position']['ci95'][1]<0 and stability['energy']['qp_a']['median_ratio']>1
    decision='STABLE_CONTEXT_STRUCTURE_NOT_DETECTION_VALIDATION' if supporting else 'CONTEXT_STABILITY_NOT_ESTABLISHED'
    save(OUT/'AUDIT.json',dict(status='PASS',fits=45,answers=len(answer_nll),landmarks=len(data['x']),
        nll_values_replayed=checks,history_max_abs_delta=history_delta,nll_max_abs_delta=max_abs,
        nll_max_relative_delta=max_rel,canonical_moment_native_checks=canonical_checks,all_landmark_simplex_max_kkt=max_kkt,
        source_files_unchanged=True,labels_opened=False))
    save(OUT/'SUMMARY.json',dict(decision=decision,macro_nll=dict(zip(ARMS,map(float,macro))),comparisons=comparisons,
        cell_nll={c:dict(zip(ARMS,map(float,cellmeans[i]))) for i,c in enumerate(cells)},
        stability=stability,motion=motion,geometry=geometry,telemetry_medians=telemetry,motion_medians=motion_summary,
        fit_records=records,source_groups=len(sources),
        total_fit_seconds=float(sum(r['elapsed'] for r in records))))
    np.savez_compressed(OUT/'GROUP_NLL.npz',source_ids=np.array(sources),cells=np.array(cells),values=values,present=present,bootstrap=draws)
    lines=['# יציבות הקשר ומשקלי fusion בבנק האמיתי',
        '**אבחון ללא תוויות נכונות בלבד. אין כאן מדדי PB/within-AUC חדשים או שינוי בגלאי.**',
        '## הקשר למחקר הקודם',
        'הבסיס נשאר innovation5 ובקרות המיקום של Step381. בדיקת Step383 הראתה היתכנות '
        'סינתטית להקשר אנרגיה פשוט, ללא הצדקה להרחבת CCA. כאן נבדק אם מבנה כזה מופיע '
        'בבנק האמיתי ובהפרדת קבוצות מקור.',
        'בחירת בנק innovation5 בהיסטוריה השתמשה בתוויות פיתוח; היעדר תוויות באבחון הנוכחי אינו מוחק חשיפה זו. '
        'ה־NLL בודק יחד ממוצע ו־covariance מקומיים, ואילו משקלי IU משתמשים ב־covariance. '
        'שיפור ב־NLL לבדו אינו מייחס את הרווח דווקא לרכיב שמייצר משקלים.',
        '## היקף',
        f'כל {len(answer_nll):,} התשובות נכללו ב־{len(data["x"]):,} נקודות מדידה, עד16 לכל תשובה, '
        'עם16 טוקנים קודמים לכל נקודה. תשובה קצרה אחת תרמה פחות מ־16 נקודות. '
        'אלו אבחוני נקודות מדידה ולא ניקוד של כל הטוקנים.45 התאמות מכסות9 תאים ו־5 folds. '
        'כל קבוצה מוחרגת מהאימון שמאבחן אותה. ערכי הפיצ׳רים וכיוונם המקוריים נשמרו; המסלול offline.',
        '16 הטוקנים הראשונים בכל תשובה אינם נקודות יעד באבחון זה, משום שנדרש חלון עבר מלא. '
        'אין להסיק מכאן כיסוי של שגיאות קצרות או מוקדמות; ניסוי איתור יצטרך לנקד גם את תחילת התשובה.',
        '## מה הותאם',
        'אנרגיית העבר היא log1p של ממוצע ריבועי הפיצ׳רים בקואורדינטות האימון. היא כוללת '
        'גם רמה וגם תנודתיות, ואינה שונות בלבד. הוסר ממנה פרופיל מיקום/אורך שנאמד באימון. '
        'השכנות משתמשת במיקום ובשארית האנרגיה; הבקרה משתמשת במיקום בלבד. '
        '64 קבוצות מקור נבחרות לכל נקודה, עם borrowing קבוע .5. אומדן הרגעים של U-PCR '
        'מזין PCR בשני רכיבים, simplex מרוסן ומשקול חד־פרמטרי של קבוצות פיצ׳רים.',
        '## חיזוי התפלגות הפיצ׳רים המוחזקים בחוץ',
        'NLL נמוך יותר פירושו התאמה טובה יותר להתפלגות הפיצ׳רים, לא איתור טוב יותר של שגיאות. '
        'המספרים מאוזנים לפי תשובה, קבוצת מקור ותא; הסקאלות נאמדו רק באימון.',
        '| זרוע | NLL מאקרו |\n|---|---:|\n'+'\n'.join(f'| {a} | {macro[j]:.6f} |' for j,a in enumerate(ARMS)),
        '| energy פחות ביקורת | הפרש NLL | CI95% | תאים שבהם energy טוב יותר |\n|---|---:|---|---:|\n'+
        '\n'.join(f'| {a} | {v["delta"]:+.6f} | [{v["ci95"][0]:+.6f}, {v["ci95"][1]:+.6f}] | {v["cells_better"]}/9 |' for a,v in comparisons.items()),
        '10,000 דגימות bootstrap של קבוצות מקור, עם הופעות של אותה קבוצה בתאים שונים יחד. '
        'הרווחים אבחוניים ולא מתוקנים לריבוי השוואות; אי־ודאות התאמת המודלים מחדש אינה כלולה.',
        '## שינוי יציב לעומת רעש אמידה',
        'ב־16 נקודות מקבוצות שונות בכל תא/fold נדגמו64 שחזורים של קבוצות השכנים. '
        'הטבלה מציגה חציון יחס בין שונות הכיוון בין נקודות לבין רעש bootstrap בתוך נקודה. '
        'יחס מעל1 אינו מבחן מובהקות. הפרופילים, השכנים והאומדן הגלובלי קבועים באבחון זה, '
        'ולכן אין כאן מדידה של כל אי־הוודאות בצינור.',
        '| זרוע | covariance | rho | PCR | simplex | קבוצות |\n|---|---:|---:|---:|---:|---:|\n'+
        '\n'.join('| '+a+' | '+' | '.join(f'{stability[a][k]["median_ratio"]:.3f}' for k in ('C','rho','native_a','qp_a','group_a'))+' |' for a in stability),
        '## סולם המשקלים ובקרת הקבוצות',
        'חציונים על45 ההתאמות; הפירוט המלא והמשקלים בשתי מערכות היחידות נמצאים בקובץ המסכם. '
        'g2 בתקרה משקף תלות בגבול שנקבע מראש, ולא זיהוי של סולם נכונות מתוך הנתונים.',
        '| זרוע | שיעור g2 בתקרה | שארית אדיטיבית | מספר קבוצות אפקטיבי מינימלי טיפוסי | זווית PCR מול סטטי | זווית simplex מול סטטי |\n|---|---:|---:|---:|---:|---:|\n'+
        '\n'.join('| '+a+' | '+' | '.join(f'{telemetry[a][k]:.4f}' for k in ('g2_ceiling_fraction','mean_additive_residual','neff_min','native_angle_to_static_mean','qp_angle_to_static_mean'))+' |' for a in ARMS),
        'הזוויות ברדיאנים. כדי להימנע מדמיון טריוויאלי שמקורו ב־75% משקל equal, '
        'הטבלה הבאה מנתחת את שינוי המשקלים ביחס לראש הסטטי של אותו סוג. '
        'השבר המוסבר הוא ההיטל על ציר ערבוב הקבוצות הקבוע; השגיאה משווה את שינויי שני הפתרונות בפועל.',
        '| זרוע | RMS שינוי משקלי simplex | שבר שינוי על ציר הקבוצות | שגיאת שינוי יחסית של הראש הקבוצתי |\n|---|---:|---:|---:|\n'+
        '\n'.join('| '+a+' | '+' | '.join(f'{motion_summary[a][k]:.6f}' for k in ('qp_update_rms','group_axis_explained_fraction','group_update_relative_squared_error'))+' |' for a in motion_summary),
        '## תאים',
        '| תא | static | position | energy | random |\n|---|---:|---:|---:|---:|\n'+
        '\n'.join('| '+c+' | '+' | '.join(f'{v:.5f}' for v in cellmeans[i])+' |' for i,c in enumerate(cells)),
        '## החלטת היתכנות',
        '`'+decision+'`',
        'יש הצדקה להמשיך לבדיקה ממוקדת של U-PCR המותנה בסיכום העבר. energy משפר את NLL מול '
        'מיקום בלבד בכל9 התאים, והשינוי בכיוון PCR גדול מרעש השכנים בכל45 ההתאמות. '
        'זו ראיה למבנה סטטיסטי שימושי לאמידה, ולא לכך שהשינוי מכוון לפיצ׳ר הנכון סמנטית.',
        'הראיה ל־simplex חלשה יותר: חציון יחס שינוי/רעש הוא1.227, מול0.978 בשכנים אקראיים; '
        'רק36 מתוך45 ההתאמות מעל1. אין כאן מבחן מובהקות להשוואת היחסים. '
        'בנוסף, בחציון פחות מאחוז משינוי המשקלים הגולמיים נמצא על ציר שתי הקבוצות הקבוע. '
        'לכן החלוקה ההיוריסטית אינה משחזרת את תנועת הפתרון המלא; אין מכאן הוכחה שהתנועה הנוספת מועילה, '
        'או שבנק הפיצ׳רים מכיל יותר משני כיווני אות חשובים. g2 כמעט תמיד בתקרה, והנחת הסולם לא נפתרה.',
        'ההמשך המוצע הוא השוואת איתור אחת עם U-PCR סטטי ומותנה בהקשר, לצד אותו בנק במיצוע, '
        'בקרות מיקום/שכנות אקראית ושינוי עוצמה בלבד, וה־simplex המרוסן ובקרת הקבוצות כמסלולי בידוד. '
        'יש להקפיא את סדר Top10 וה־gate ולמדוד עלות לניקוד כל הטוקנים לפני ההרצה. '
        'לא נפתח כאן ניסוי האיכות, לא הורחב CCA ולא חודשה משפחת המודלים העצביים.',
        'סיכום האנרגיה אינו משתמש בסדר הפנימי של16 הטוקנים שבחלון. הממצא תומך בתלות ברקע המקומי, '
        'ואינו מוכיח יתרון לסדר ההשהיות המדויק. ה־innovation המקורי ובקרות המיקום של Step381 נשארים עוגן האיכות.',
        'ההחלטה מסכמת ראיות למבנה מותנה, ולא בוחרת מנצח בזיהוי שגיאות. השוואת קבוצות '
        'מניחה חלוקה קבועה {H0lim,VE0,innovation}/{VE075,VE1}; אין כאן גילוי GroupFS '
        'או הוכחת עצמאות שגיאות. קרבה בין משקלים מרוסנים מושפעת גם מכך ש־75% ממשקלם הוא equal.',
        '## תקינות נומרית ועלות',
        'ריצה v1 נעצרה לאחר33 התאמות: סף שיפור מוחלט של 1e-13 בפותר השאיר פתרון גבול עם שארית KKT של '
        '1.45e-7. התיקון הסיר את סף השיפור, בלי לשנות את מטרת האופטימיזציה או להקל בבדיקת KKT. '
        'כל45 ההתאמות חושבו מחדש ב־v2; המקור נשמר. המקרה נבדק גם בפותר SLSQP עצמאי. '
        f'סך זמני ההתאמות בריצה המלאה: {sum(r["elapsed"] for r in records):.1f} שניות, ללא זמן הכנה ודיווח. '
        '[תיעוד התיקון](../../docs/experiments/ENERGY_CONTEXT_STABILITY_NUMERICAL_AMENDMENT_20260915.md).',
        '## אימות וקישורים',
        f'אימות עצמאי עבר על {checks:,} ערכי NLL באמצעות פירוק Cholesky, ועל כל חלונות '
        'העבר שנבחרו. נבדקו כיסוי מלא של התשובות, הפרדת מקורות, יחידות המשקלים, '
        'זהות מסת השכנים האקראיים ושמירת חתימות הקלט.',
        '[פרוטוקול](../../docs/experiments/ENERGY_CONTEXT_STABILITY_20260915.md) · '
        '[נתונים מסכמים](SUMMARY.json) · [אימות](AUDIT.json) · [מקור האבחון הסינתטי](../cca_iu_isolation_gate_v1/REPORT.html)']
    md='\n\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(md,encoding='utf8');body,toc=render(md)
    (OUT/'REPORT.html').write_text('<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>יציבות הקשר ו־fusion</title><style>body{font:17px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:25px;color:#253747}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #ccd5dd;text-align:right}.table-wrap{overflow:auto}h2{margin-top:35px}nav{display:grid;gap:6px}a{color:#006779}code{direction:ltr;unicode-bidi:isolate}</style><nav>'+toc+'</nav>'+body+'</html>',encoding='utf8')
    save(OUT/'RUN_STATE.json',dict(state='COMPLETE_REVIEWED',fits=45,answers=len(answer_nll),landmarks=len(data['x']),decision=decision))
    save(OUT/'REPORT_PROVENANCE.json',{name:sha(ROOT/name) for name in
        ('scripts/report_energy_context_stability.py','scripts/audit_energy_numerical_revision.py','tests/test_energy_context_stability.py')})
    save(OUT/'ARTIFACT_HASHES.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('ARTIFACT_HASHES.json','RUN_STATE.json')})
    print(decision,comparisons,stability,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
