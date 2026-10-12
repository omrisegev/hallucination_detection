"""Build the final report, including sealed explanatory quota diagnostics.

This is an evaluation/presentation layer. It never fits an estimator or selects
an arm, threshold, sign, bank or hyperparameter from evaluation outcomes.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/error-count-mpl-cache')
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/label_free_error_count_v1_2026-10-12'
FROZEN=ROOT/'results/advisor_completion_2026-10-11'
from spectral_utils.error_count_evaluation import (ARMS,CONTROLS,SourceBootstrap,
    build_report,validate_prediction_seal,flags_from_shared_rank,
    summarize,interval,write_json,write_csv,sha256)


def validate_diagnostics():
    dest=OUT/'diagnostics';seal=json.loads((dest/'QUOTA_SEAL.json').read_text())
    for relative,spec in seal['files'].items():
        path=(dest/relative).resolve()
        if not path.is_relative_to(dest.resolve()) or sha256(path)!=spec['sha256']:
            raise RuntimeError('Quota diagnostic seal mismatch: '+relative)
    return seal


def diagnostic_evaluation():
    validate_prediction_seal(OUT);validate_diagnostics()
    freeze=json.loads((OUT/'INPUT_FREEZE.json').read_text())
    source=FROZEN/'inputs/population.npz';expected=freeze['inputs'][str(source.relative_to(ROOT))]['sha256']
    if sha256(source)!=expected: raise RuntimeError('Evaluation labels changed')
    with np.load(source,allow_pickle=False) as z:
        p={key:z[key] for key in ['idx','offsets','y','classification','group','fold']}
    off=p['offsets'];n=np.diff(off);truth=np.add.reduceat(p['y'].astype(int),off[:-1])
    bootz=np.load(OUT/'evaluation/BOOTSTRAP_DRAWS.npz',allow_pickle=False)
    boot=SourceBootstrap(p['group'],weights=bootz['multiplicities'])
    assert np.array_equal(boot.groups,bootz['source_groups'])
    masks={'ALL_NONCONTROL':p['classification']!='correct','SYNTHETIC_CLEAN':p['classification']=='correct',
           'ALL_TRUE_ZERO':truth==0,'ERROR_BEARING':truth>0}
    rows=[];contrasts=[];predout={'idx':p['idx'],'offsets':off}
    for bank in ['N16','N56']:
        main=np.load(OUT/'predictions'/f'{bank}.npz',allow_pickle=False)
        quantities={name:np.full(len(n),-1,np.int64) for name in ['ABS_ZERO','ABS_Q80']}
        probabilities={name:np.full(off[-1],np.nan) for name in quantities}
        for fold in range(5):
            z=np.load(OUT/f'diagnostics/checkpoints/{bank}_fold{fold}/absolute_heldout.npz',allow_pickle=False)
            answers=np.flatnonzero(p['fold']==fold)
            if not np.array_equal(z['idx'],p['idx'][answers]) or not np.array_equal(z['source_group'],p['group'][answers]):
                raise RuntimeError('Diagnostic heldout identities differ')
            if not np.array_equal(np.diff(z['offsets']),n[answers]): raise RuntimeError('Diagnostic answer lengths differ')
            steps=np.concatenate([np.arange(off[i],off[i+1]) for i in answers])
            for method in ['IU_PCR','LSML_CONT']:
                if not np.array_equal(z[method+'_score'],main[method+'_score'][steps]): raise RuntimeError('Diagnostic changed ranking')
            for name in quantities:
                quantities[name][answers]=z[name+'_count'];probabilities[name][steps]=z[name+'_posterior']
        for name in quantities:
            predout[f'{bank}__count__{name}']=quantities[name];predout[f'{bank}__posterior__{name}']=probabilities[name]
        for method in ['IU_PCR','LSML_CONT']:
            counts={**quantities,'Q80':main[f'count__{method}__Q80'],'LEGACY_MOM':main[f'count__{method}__LEGACY_MOM']}
            for name in quantities:
                flag,cov=flags_from_shared_rank(main['rank_'+method],off,counts[name])
                for pop,selected in masks.items():
                    point,draw=summarize(p['y'],flag,off,counts[name],cov,selected,boot)
                    if pop in ['SYNTHETIC_CLEAN','ALL_TRUE_ZERO']:
                        for key in ['prm_score','prm_score_lo95','prm_score_hi95']:
                            point[key]=np.nan
                        point['prm_score_finite_draws']=0
                    rows.append(dict(bank=bank,method=method,scenario=name,population=pop,role='exploratory_diagnostic_only',**point))
                    for baseline in ['Q80','LEGACY_MOM']:
                        bf,bc=flags_from_shared_rank(main['rank_'+method],off,counts[baseline]);common=cov&bc
                        left,ld=summarize(p['y'],flag,off,counts[name],common,selected,boot)
                        right,rd=summarize(p['y'],bf,off,counts[baseline],common,selected,boot)
                        for metric in ['count_mae','prm_score','error_recall','zero_falseflag_rate','error_answer_miss_rate']:
                            if metric=='prm_score' and pop in ['SYNTHETIC_CLEAN','ALL_TRUE_ZERO']:
                                continue
                            lo,hi=interval(ld[metric]-rd[metric])
                            contrasts.append(dict(bank=bank,method=method,scenario=name,baseline=baseline,population=pop,metric=metric,
                                difference=left[metric]-right[metric],lo95=lo,hi95=hi,n_answers=left['n_answers'],
                                n_source_groups=left['n_source_groups'],full_coverage=bool(common.all()),role='exploratory_pointwise95_only'))
    dest=OUT/'diagnostics'
    np.savez_compressed(dest/'EXPLORATORY_PREDICTIONS.npz',**predout)
    write_csv(dest/'EXPLORATORY_METRICS.csv',rows);write_csv(dest/'EXPLORATORY_CONTRASTS.csv',contrasts)
    result={'status':'COMPLETE','diagnostic_only':True,'primary_candidate_count_unchanged':7,
            'metrics':rows,'contrasts':contrasts,'label_source_sha256':expected,
            'diagnostic_seal_sha256':sha256(dest/'QUOTA_SEAL.json'),'code_sha256':sha256(__file__),
            'scope':'preconfigured mean-zero threshold neutralization; no protected improvement claim, no selection from utility'}
    write_json(dest/'EXPLORATORY_EVALUATION.json',result)
    return result


def supplemental_sections():
    """Explain saved component diagnostics; do not refit or promote them."""
    dest=OUT/'diagnostics'
    component=json.loads((dest/'COMPONENT_RESULTS.json').read_text())
    oracle=json.loads((dest/'COUNT_ORACLES.json').read_text())
    emission=json.loads((dest/'EMISSION_CALIBRATION.json').read_text())
    timing=json.loads((dest/'TIMING.json').read_text())
    def fmt(v):return 'undefined' if v is None else f'{v:.4f}'
    def row(component_name,population,metric):
        return next(r for r in component['contrasts'] if r['bank']=='N16' and r['method']=='IU_PCR'
                    and r['component']==component_name and r['population']==population and r['metric']==metric)
    block="<section id='component-effects'><h2>מה שינה כל רכיב</h2>"
    block+="<p>ההשוואות הישירות הבאות משתמשות באותם 20,000 bootstrap של קבוצות מקור. אלה אבחונים לאחר ההערכה, עם רווחים נקודתיים של 95%; הם אינם מוסיפים זרועות למשפחת ההכרזה. סימן ההפרש הוא שמאל פחות ימין.</p>"
    block+="<table><thead><tr><th>N16 / IU-PCR: comparison</th><th>Population / N</th><th>Metric</th><th>Δ [95% CI]</th></tr></thead><tbody>"
    requested=[('LOCAL_THRESHOLD_MINUS_RELATIVE','ALL_NONCONTROL','count_mae'),
               ('LOCAL_THRESHOLD_MINUS_RELATIVE','ALL_NONCONTROL','prm_score'),
               ('LOCAL_THRESHOLD_MINUS_RELATIVE','ALL_NONCONTROL','error_recall'),
               ('A_ABSOLUTE_MINUS_LOCAL_THRESHOLD','ALL_NONCONTROL','count_mae'),
               ('A_ABSOLUTE_MINUS_LOCAL_THRESHOLD','ALL_NONCONTROL','prm_score'),
               ('C_B_MEDIAN_MINUS_MEAN','ALL_NONCONTROL','count_mae'),
               ('C_D_MEDIAN_MINUS_MEAN','ALL_NONCONTROL','count_mae'),
               ('D_ZERO_MINUS_ORDINARY_PRIOR','ALL_NONCONTROL','count_mae'),
               ('D_ZERO_MINUS_ORDINARY_PRIOR','ALL_NONCONTROL','error_answer_miss_rate'),
               ('D_ZERO_MINUS_ORDINARY_PRIOR','SYNTHETIC_CLEAN','zero_falseflag_rate')]
    for name,pop,metric in requested:
        r=row(name,pop,metric)
        block+=f"<tr><td>{html.escape(r['left'])} − {html.escape(r['right'])}</td><td>{pop} / {r['n_answers']}</td><td>{metric}</td><td>{fmt(r['difference'])} [{fmt(r['lo95'])}, {fmt(r['hi95'])}]</td></tr>"
    block+="</tbody></table>"
    changes=[r for r in component['count_changes'] if r['bank']=='N16' and r['method']=='IU_PCR' and r['population']=='ALL_NONCONTROL']
    b=next(r for r in changes if r['component']=='C_B_MEDIAN_MINUS_MEAN')
    d=next(r for r in changes if r['component']=='C_D_MEDIAN_MINUS_MEAN')
    block+=f"<p>חציון לעומת תוחלת שינה {b['n_count_changed']} מתוך {b['n_answers']} כמויות ב־B ו־{d['n_count_changed']} מתוך {d['n_answers']} ב־D; ה־posteriors זהים, כולל החוסרים. B מול A ניתן להשוות רק על 1,269 מתוך 6,211 noncontrols. D עם ρ=0 מפיק כאן אותן החלטות כמו B; רכיב האפס הוא מקור השינוי הנוסף.</p>"
    block+="<p><a href='../diagnostics/COMPONENT_CONTRASTS.csv'>All component contrasts</a> · <a href='../diagnostics/COMPONENT_POSTERIOR_INVARIANCE.json'>Same-posterior checks</a> · <a href='../diagnostics/COMPONENT_COUNT_DIFFERENCE_HISTOGRAMS.csv'>Count-change histograms</a></p></section>"
    block+="<section id='emission-calibration'><h2>יציבות לעומת כיול לשגיאות המטרה</h2>"
    block+="<p>התוויות משמשות כאן לאבחון של מודלים שכבר הוקפאו, בלי התאמה מחדש. האוכלוסייה בטבלה כוללת את כל 6,969 התשובות; שכיחות האמת היא יחס צעדים תקפים שגויים מתוך 94,203 צעדים.</p>"
    block+="<table><thead><tr><th>Bank / B</th><th>MoM prevalence</th><th>True prevalence</th><th>ψ MAE</th><th>η MAE</th><th>Orientation agreement</th></tr></thead><tbody>"
    for bank in ['N16','N56']:
        r=next(r for r in emission['aggregates'] if r['bank']==bank and r['arm']=='B_SELECTED_TRIPLE' and r['population']=='ALL' and r['basis']=='successful_fits_only')
        block+=f"<tr><td>{bank}</td><td>{r['prevalence_hat_stepweighted']:.2%}</td><td>{r['prevalence_true_stepweighted']:.2%}</td><td>{fmt(r['psi_mae'])}</td><td>{fmt(r['eta_mae'])}</td><td>{r['orientation_agreement_group_fold_fraction']:.1%}</td></tr>"
    block+="</tbody></table><p>כיוון חיובי ו־bootstrap יציב אינם מבטיחים emissions מכוילים. קורלציות זוגיות מותנות בתווית נותרות ניכרות; הן אבחון תיאורי להפרת הנחות אפשרית, ולא מבחן שמוכיח אי־תלות. מירכוז בתוך תשובה עלול בעצמו ליצור תלות. פרמטרים גולמיים של MoM ההיסטורי מופרדים מה־posterior שעבר clipping, וכשלי התאמה לא תוקנו לפי התוויות.</p>"
    block+="<p><a href='../diagnostics/EMISSION_CALIBRATION.json'>Calibration evidence</a> · <a href='../diagnostics/EMISSION_GROUPWISE.csv'>All emission parameters</a> · <a href='../diagnostics/CONDITIONAL_CORRELATION_PAIRS.csv'>Raw and centered conditional correlations</a> · <a href='../diagnostics/POSTERIOR_CALIBRATION.csv'>Posterior calibration</a></p></section>"
    block+="<section id='oracle-headroom'><h2>מרווח לשיפור: אבחוני oracle</h2><p>אלה אבחוני רכיבים שמשתמשים בתוויות הערכה. הכמות האמיתית מצטרפת לדירוג הקבוע; דירוג error-first שומר את כמויות Q80. ההפרשים אינם מצטברים ואינם חסם מתמטי על PRMScore. הם אינם אומדים שניתן להפעיל ללא תוויות.</p>"
    block+="<table><thead><tr><th>N16 / IU-PCR, noncontrols</th><th>MAE</th><th>PRMScore</th><th>Recall</th></tr></thead><tbody>"
    for arm in ['Q80','ORACLE_TRUE_COUNT_FIXED_RANK','ORACLE_ERROR_FIRST_Q80_COUNT','ALL_ZERO_NEGATIVE_CONTROL']:
        r=next(r for r in oracle['overall'] if r['bank']=='N16' and r['method']=='IU_PCR' and r['arm']==arm)
        block+=f"<tr><td>{arm}</td><td>{fmt(r['count_mae'])}</td><td>{fmt(r['prm_score'])}</td><td>{fmt(r['error_recall'])}</td></tr>"
    block+="</tbody></table><p>גם הכמות וגם הסדר בתוך תשובה מגבילים את התוצאה. הימנעות מלאה מסימון נותנת recall אפס. כל תשעת הסוגים נשמרים באבחון: MS נמדד ב־valid-step F1 ו־Sensitivity כולל PS/DR/MS; הבקרות הנקיות מדווחות בנפרד.</p><p><a href='../diagnostics/COUNT_ORACLES.json'>Oracle definitions and overall results</a> · <a href='../diagnostics/COUNT_ORACLE_METRICS.csv'>Nine types and clean strata</a> · <a href='../diagnostics/COUNT_ORACLE_CONTRASTS.csv'>Paired oracle intervals</a></p></section>"
    block+="<details id='timing'><summary>זמני CPU ויציבות הקבוצות</summary><p>זמנים מתוך המודלים החתומים, בשניות. זמנים חסרים של stages שנכשלו אינם מושלמים כאפס. ARI ו־participation ratio הם אבחונים בלבד.</p>"
    block+="<table><thead><tr><th>Bank</th><th>Stage</th><th>Recorded folds</th><th>Min / median / max seconds</th></tr></thead><tbody>"
    keys=['IU_PCR_fit_s','IU_PCR_prediction_s','A_representation_fit_s','B_triple_fit_s','D_zero_fit_s']
    for r in timing['summary']:
        if r['timing_key'] in keys:
            block+=f"<tr><td>{r['bank']}</td><td>{r['timing_key']}</td><td>{r['n_recorded_folds']}</td><td>{fmt(r['min_s'])} / {fmt(r['median_s'])} / {fmt(r['max_s'])}</td></tr>"
    block+="</tbody></table><p><a href='../diagnostics/TIMING_SUMMARY.csv'>All timing summaries</a> · <a href='../diagnostics/TIMING.csv'>All raw timings</a> · <a href='../diagnostics/GROUP_PARTITION_ARI.csv'>Group partition ARI</a> · <a href='../diagnostics/GROUP_WEIGHT_PR.csv'>Group weight participation</a> · <a href='../diagnostics/WITHIN_GROUP_WEIGHT_PR.csv'>Within-group weight participation</a></p></details>"
    return block


def quality_plot(data):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='label_free_error_count_v1_final'
    import matplotlib.pyplot as plt
    order=CONTROLS+ARMS;fig,axes=plt.subplots(2,3,figsize=(20,12))
    for bank,axs in zip(['N16','N56'],axes):
        points={r['arm']:r for r in data['metrics'] if r.get('bank')==bank and r.get('method')=='IU_PCR' and r.get('population')=='ALL_NONCONTROL'}
        labels=[]
        for arm in order:
            r=points[arm];labels.append(arm if r['coverage']==1 else f"{arm} [partial N={r['n_answers']}/{r['n_expected_answers']}]")
        for ax,key,title in zip(axs,['count_mae','prm_score','error_recall'],['Count MAE (errors/answer)','Noncontrol PRMScore (0–1)','Error-step recall (0–1)']):
            for i,arm in enumerate(order):
                r=points[arm];v=r.get(key);lo=r.get(key+'_lo95');hi=r.get(key+'_hi95')
                if v is None or lo is None or hi is None:continue
                partial=r['coverage']!=1
                ax.errorbar([v],[i],xerr=[[max(v-lo,0)],[max(hi-v,0)]],fmt='o',capsize=3,
                            color='0.5' if partial else '#246c9c',markerfacecolor='none' if partial else '#246c9c')
            ax.set_yticks(range(len(order)),labels,fontsize=7);ax.invert_yaxis();ax.set_xlabel(title)
            ax.set_title(bank+(' | historical anchor' if bank=='N16' else ' | prespecified sensitivity'));ax.grid(axis='x',alpha=.2)
    fig.suptitle('Fixed IU-PCR ranking | pointwise source-group 95% intervals | open markers: partial coverage, not full-population comparisons')
    fig.tight_layout()
    for ext in ['png','svg']:
        fig.savefig(OUT/'report'/f'count_quality.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'label_free_error_count_v1_final'})
    plt.close(fig)


def main():
    validate_diagnostics();evaluation=diagnostic_evaluation();build_report(OUT)
    data=json.loads((OUT/'evaluation/RESULTS.json').read_text());quality_plot(data)
    dest=OUT/'report';shutil.copyfile(dest/'REPORT_RECEIPT.json',dest/'BASE_REPORT_RECEIPT.json')
    science=json.loads((OUT/'diagnostics/QUOTA_SCIENCE.json').read_text())
    full=next(x for x in science['quota_sweep'] if x['scope']=='FULL_REFERENCE_TRANSDUCTIVE' and x['estimator']=='STRICT')
    perm=next(x for x in science['permutation_summaries'] if x['scope']=='FULL_REFERENCE_TRANSDUCTIVE' and x['estimator']=='STRICT')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5));xx=[100*x['actual_mean_vote_rate'] for x in full['points']];yy=[100*x['prevalence_hat'] for x in full['points']]
    ax.plot(xx,yy,'o-',label='Estimated MoM prevalence');ax.plot(xx,xx,'--',color='0.5',label='Vote fraction reference')
    for x,y,row in zip(xx,yy,full['points']):ax.annotate(f"q={row['quota']:.1f}",(x,y),xytext=(5,-15),textcoords='offset points')
    ax.set(xlabel='Actual positive vote fraction (%)',ylabel='Estimated latent prevalence (%)',title='Fixed historical N16 groups | predefined quota perturbation')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(dest/f'quota_mechanism.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'label_free_error_count_v1_final'})
    plt.close(fig)
    def number(v):return 'undefined' if v is None else f'{v:.4f}'
    block="<section id='quota-mechanism'><h2>השערת שיעור ההצבעה: בדיקה ונטרול מוגדר</h2>"
    block+="<p>הקרבה בין 22.905% של MoM לבין 23.016% הצבעות אינה מקרית בלבד במבחן הרגישות שנבדק: שינוי המכסה משנה כמעט במקביל את אומדן השכיחות. אין כאן זהות אלגברית או הוכחה שהמכסה מסבירה לבדה את הכשל.</p>"
    block+="<img src='quota_mechanism.svg' alt='Fixed quota perturbation: actual positive vote fraction and estimated latent prevalence'>"
    block+="<table><thead><tr><th>מכסה</th><th>הצבעה בפועל</th><th>MoM</th></tr></thead><tbody>"
    for row in full['points']:block+=f"<tr><td>{row['quota']:.0%}</td><td>{row['actual_mean_vote_rate']:.3%}</td><td>{row['prevalence_hat']:.3%}</td></tr>"
    block+="</tbody></table>"
    block+=f"<p>באותן מכסות בדיוק, ערבוב עצמאי בתוך תשובה הפיק אומדנים בטווח {perm['prevalence_min']:.2%}–{perm['prevalence_max']:.2%}. לכן MoM תלוי גם בהסכמה ובמומנטים מסדר גבוה, והקרבה לשיעור ההצבעה אינה כלל מתמטי. הערבוב שומר גם מבנה תלוי אורך: covariance צפוי של 4·Var(ceil(q·n)/n).</p>"
    block+="<p>אבחון נטרול נוסף קיבע סף אפס על ציוני הקבוצות המוחלטים, שמורכזים באימון: ABS_ZERO. הוא מסיר שיעור הצבעה כפוי, אך ממוצע האימון אינו גבול מובטח לשגיאה. הזרוע הוגדרה לפני צפייה בתוצאות חדשות ונשארת אבחון משני; אין בחירה של q=10% בעקבות קרבתו לשכיחות האמת.</p>"
    block+="<table><thead><tr><th>Bank</th><th>Diagnostic</th><th>N</th><th>MAE</th><th>PRMScore</th><th>Recall</th><th>Clean false flags</th></tr></thead><tbody>"
    for bank in ['N16','N56']:
        row=next(x for x in evaluation['metrics'] if x['bank']==bank and x['method']=='IU_PCR' and x['scenario']=='ABS_ZERO' and x['population']=='ALL_NONCONTROL')
        control=next(x for x in evaluation['metrics'] if x['bank']==bank and x['method']=='IU_PCR' and x['scenario']=='ABS_ZERO' and x['population']=='SYNTHETIC_CLEAN')
        block+=f"<tr><td>{bank}</td><td>ABS_ZERO</td><td>{row['n_answers']}</td><td>{number(row['count_mae'])}</td><td>{number(row['prm_score'])}</td><td>{number(row['error_recall'])}</td><td>{number(control['zero_falseflag_rate'])}</td></tr>"
    block+="</tbody></table><p>רווחים מזווגים נקודתיים של 95% נמצאים בקובץ האבחון; הם אינם קריטריון ההצלחה המתוקן של שבע הזרועות הראשיות. ביטול המכסה לבדו לא זיהה את שיעור השגיאות הסמנטי בניסוי הזה.</p>"
    block+="<p><a href='../diagnostics/QUOTA_PROTOCOL.json'>Diagnostic protocol</a> · <a href='../diagnostics/QUOTA_RESULTS.csv'>All 590 training-only diagnostic rows</a> · <a href='../diagnostics/QUOTA_SEAL.json'>Diagnostic prediction seal</a> · <a href='../diagnostics/EXPLORATORY_EVALUATION.json'>Exploratory evaluation and paired intervals</a> · <a href='../diagnostics/EXACT_TEACHER.json'>Independent-voter synthetic control</a></p></section>"
    block+="<section id='interpretation'><h2>מה הממצאים מגבילים</h2><p>המשפחה שנבדקה אינה מספקת שיפור כמות שעובר את הקריטריון המשולב. שינוי עיגול לחציון כמעט שלא שינה את התוצאה; שלשה יציבה אינה מבטיחה posterior מכויל לשגיאות אמת; ורכיב תשובה נקייה שינה את ההקצאה במחיר MAE ו־recall. A ב־N16 נכשל בארבעה folds בגלל emissions לא־תקפים, והנקודות החלקיות שלו אינן תוצאה מלאה.</p><p>הניסוי הבא שהראיות מצדיקות הוא מודל של ראיות רציפות לפני ספי הצבעה, עם טיפול מפורש בתלות ובהטרוגניות בין תשובות. הוא צריך לבדוק שהמחלקה החבויה המוערכת תואמת לשגיאות המטרה, בלי prior שמוכנס מתוויות. לא הורץ כאן מודל כזה ולא הוכח שאמידת כמות בכלל אינה אפשרית.</p><p><a href='../review/INDEPENDENT_SEALED_AUDIT.json'>Independent arithmetic audit</a> · <a href='../review/BRIDGE_NUMERICAL_REPAIR.json'>Numerical repair receipt</a> · <a href='../AMENDMENTS.json'>Chronological amendments</a></p></section>"
    path=dest/'index.html';text=path.read_text()
    notice="<p class='small'><strong>כיסוי חלקי:</strong> A ב־N16 ו־LOCAL_MOM ב־N56 מוגדרים רק ב־1,269 מתוך 6,211 noncontrols. סמנים חלולים מציינים אוכלוסייה חלקית; אין להשוות את נקודתם לתוצאת אוכלוסייה מלאה.</p>"
    text=text.replace("<img src='count_quality.svg'",notice+"<img src='count_quality.svg'",1)
    # Put the user's new mechanism result directly after the main conclusion.
    first_end=text.find('</p>',text.find('<h1'))+4
    text=text[:first_end]+block+text[first_end:]
    text=text.replace('</html>',supplemental_sections()+"<p><a href='../README.md'>Reproduction and package guide</a> · <a href='../FINDINGS_HE.md'>Hebrew findings</a> · <a href='../review/FINAL_REVIEW.md'>Independent final review</a></p></html>")
    path.write_text(text)
    write_json(dest/'REPORT_RECEIPT.json',{'status':'FINAL_WITH_SEALED_DIAGNOSTICS','html_sha256':sha256(path),
        'builder_sha256':sha256(__file__),'primary_results_sha256':sha256(OUT/'evaluation/RESULTS.json'),
        'quota_science_sha256':sha256(OUT/'diagnostics/QUOTA_SCIENCE.json'),
        'supplemental_inputs':{p.name:sha256(p) for p in [OUT/'diagnostics/COMPONENT_RESULTS.json',OUT/'diagnostics/EMISSION_CALIBRATION.json',OUT/'diagnostics/COUNT_ORACLES.json',OUT/'diagnostics/TIMING.json']},
        'figures':{p.name:sha256(p) for p in sorted(dest.glob('*.svg'))},'browser_pixel_review':'not performed: local file protocol blocked; static PNG figures inspected'})
    print('FINAL_REPORT',path,flush=True)


if __name__=='__main__':main()
