"""Completed position/decision layer checkpoints: implementation consistency only (not quality).
CONTENT/ANSWER_Z_Q80 rows in layers must equal the sealed content-stage chunk rows."""
import json, gzip, io
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT=Path(__file__).resolve().parent
d=ROOT/'results/level_reduction_v1/run_20261004_resume'
sealed=sorted(s.name.removesuffix('.complete.json') for s in (d/'layers').glob('*.complete.json'))
fields=['within_answer_auc','prmscore','pb_first_error_accuracy','pb_official_f1','clean_flag_rate','wrong_answer_auroc','coverage','failures','n_answers']
maxdiff=0.; compared=0; rows_total=0; fail_rows=[]; pos_counts={}; rule_counts={}; nulls=[]; cov=[]
for stem in sealed:
    lay=pd.read_csv(d/'layers'/(stem+'.metrics.csv.gz')); con=pd.read_csv(d/'chunks'/(stem+'.metrics.csv'))
    rows_total+=len(lay)
    for (pos,rule),g in lay.groupby(['position_variant','decision_rule']): pos_counts[(pos,rule)]=pos_counts.get((pos,rule),0)+len(g)
    for _,r in lay[lay.coverage==0].iterrows(): fail_rows.append({'stem':stem,'recipe':r.recipe,'position_variant':r.position_variant,'decision_rule':r.decision_rule,'reason':str(r.get('failure_reason'))[:120]})
    a=lay[(lay.position_variant=='CONTENT')&(lay.decision_rule=='ANSWER_Z_Q80')].set_index('recipe')
    b=con[(con.position_variant=='CONTENT')&(con.decision_rule=='ANSWER_Z_Q80')].set_index('recipe')
    for recipe in b.index:
        if recipe not in a.index: fail_rows.append({'stem':stem,'recipe':recipe,'missing_in_layers':True}); continue
        for f in fields:
            x,y=a.loc[recipe,f],b.loc[recipe,f]
            if pd.isna(x) and pd.isna(y): continue
            if pd.isna(x)!=pd.isna(y): maxdiff=max(maxdiff,1.0); fail_rows.append({'stem':stem,'recipe':recipe,'field':f,'layer':x,'content':y}); continue
            maxdiff=max(maxdiff,abs(float(x)-float(y))); compared+=1
    cov.append({'stem':stem,'n_rows':len(lay),'coverage0_rows':int((lay.coverage==0).sum())})
    dg=json.loads((d/'layers'/(stem+'.diagnostics.json')).read_text(encoding='utf8'))
    nl=dg.get('nulls',{})
    for key,v in nl.items():
        rec={'stem':stem,'score':key,'n_checked':v.get('n_checked'),'singletons':v.get('same_length_singletons'),'scope':v.get('scope')}
        for lab in ['within_answer_shuffle','whole_answer_same_length_swap']:
            if lab in v: rec[lab+'_mean']=v[lab]['mean']; rec[lab+'_p']=v[lab]['p_plus_one']
        rec['observed']=v.get('observed_within_auc',v.get('observed_first_error_accuracy'))
        nulls.append(rec)
nulls=pd.DataFrame(nulls); nulls.to_csv(OUT/'LAYER_NULLS_COMPLETED.csv',index=False)
pd.DataFrame(fail_rows).to_csv(OUT/'LAYER_FAILURE_ROWS.csv',index=False)
summary={'sealed_layer_checkpoints':len(sealed),'expected_level':360,'content_vs_layer_fields_compared':compared,'max_abs_difference':maxdiff,
 'layer_metric_rows':rows_total,'rows_by_position_rule':{f'{k[0]}|{k[1]}':v for k,v in sorted(pos_counts.items())},
 'coverage0_rows':int(sum(c['coverage0_rows'] for c in cov)),'failure_rows_listed':len(fail_rows)}
by=pd.DataFrame(fail_rows)
if len(by) and 'position_variant' in by: summary['failures_by_position_variant']=by.groupby('position_variant').size().to_dict(); summary['failures_by_recipe']=by.groupby('recipe').size().to_dict()
if len(nulls):
    c=nulls[nulls.score.str.endswith('__CONTENT')]
    summary['nulls_content_scores']={'n':int(len(c)),'p_within_shuffle_max':float(c.within_answer_shuffle_p.max()),'p_within_shuffle_le_0.01':int((c.within_answer_shuffle_p<=0.01).sum()),
        'p_same_length_swap_max':float(c.whole_answer_same_length_swap_p.max()),'p_same_length_swap_gt_0.05':int((c.whole_answer_same_length_swap_p>0.05).sum()),
        'null_mean_within_shuffle_range':[float(c.within_answer_shuffle_mean.min()),float(c.within_answer_shuffle_mean.max())]}
    pz=nulls[nulls.score.str.endswith('__POSZ')]
    if len(pz): summary['nulls_POSZ_scores']={'n':int(len(pz)),'observed_range':[float(pz.observed.min()),float(pz.observed.max())],'p_same_length_swap_gt_0.05':int((pz.whole_answer_same_length_swap_p>0.05).sum()),'p_within_shuffle_gt_0.05':int((pz.within_answer_shuffle_p>0.05).sum())}
    summary['singletons_logged_range']=[int(nulls.singletons.min()),int(nulls.singletons.max())]
(OUT/'LAYER_CONSISTENCY.json').write_text(json.dumps(summary,indent=1,default=str),encoding='utf8')
print(json.dumps(summary,indent=1,default=str))
