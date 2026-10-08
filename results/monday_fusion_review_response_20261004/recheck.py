"""Read-only reanalysis of completed arms; exploratory review, no new fitted method."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
from pathlib import Path
import json,hashlib,collections,datetime
import numpy as np
import pandas as pd

ROOT=Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT=Path(__file__).resolve().parent
RUNS={k:ROOT/'results'/v/'run_20261003' for k,v in [('level','level_reduction_v1'),('input','estimator_input_v1')]}
BANKS=['B13','B16','B20','B23','B32','B35','B51','B54']
RECIPES=['ALL_EQ','DS_FILTER_EQ']+[w+'_'+m for w in ['MEAN','HEM'] for m in ['EQ','DS','SML','MOM']]
VAR={'level':['original','remove_2','remove_3','random_2','random_3'],'input':['groups_answer','channels_answer','groups_cell','channels_cell']}
DRAW=20000; SEED=2026100417
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
used={}; verified=0
for exp,path in RUNS.items():
    for name in ['PROTOCOL.json','INPUT_MANIFEST.json','CONTRASTS.csv','SELECTION.json','METRICS.csv']:
        used[str(path/name)]=sha(path/name)
    for stamp in (path/'chunks').glob('*.complete.json'):
        seal=json.loads(stamp.read_text());stem=stamp.name.removesuffix('.complete.json')
        for suffix,digest in seal['artifacts'].items():
            p=stamp.parent/(stem+suffix); assert sha(p)==digest,str(p);used[str(p)]=digest;verified+=1
manifest=json.loads((RUNS['level']/'INPUT_MANIFEST.json').read_text())
ap=Path(manifest['inputs']['oof_answers']['path']);assert sha(ap)==manifest['inputs']['oof_answers']['sha256']
answers=pd.read_csv(ap,usecols=['id','uid','cell','source_group','fold','target'])
used[str(ap)]=sha(ap)
metrics={}
for exp,path in RUNS.items():
    m=pd.read_csv(path/'METRICS.csv');m=m[(m.position_variant=='CONTENT')&(m.decision_rule=='ANSWER_Z_Q80')]
    assert len(m)==len(VAR[exp])*8*9*10 and (m.coverage==1).all() and (m.failures==0).all()
    metrics[exp]=m

# Independent aggregation from per-answer stored AUC / argmax, without production statistic helpers.
raw={}; checks=[]; groupinfo={}
for benchmark in ['prm','pb']:
    mask=answers.cell.str.startswith('prm') if benchmark=='prm' else answers.cell.str.startswith('pb_')
    indices=np.flatnonzero(mask); local={int(a):j for j,a in enumerate(indices)}
    frame=answers.iloc[indices]; cellnames=sorted(frame.cell.unique()); cellid=pd.Categorical(frame.cell,categories=cellnames).codes
    groups,gi=np.unique(frame.source_group,return_inverse=True)
    eligible=None if benchmark=='prm' else frame.target.to_numpy()>=0
    for exp,path in RUNS.items():
        for variant in VAR[exp]:
            averages={recipe:np.zeros(len(indices)) for recipe in RECIPES}
            for bk in BANKS:
                for ce in cellnames:
                    with np.load(path/'chunks'/f'{ce}__{bk}__{variant}.stats.npz') as z:
                        ai=z['answer_indices'];li=np.array([local[int(a)] for a in ai]);assert np.array_equal(answers.iloc[ai].cell.to_numpy(),np.repeat(ce,len(ai)))
                        for recipe in RECIPES:
                            if benchmark=='prm':
                                val=z[recipe+'__ANSWER_Z_Q80__auc'];mask=np.isfinite(val)
                                if eligible is None: eligible=mask.copy()
                                assert np.array_equal(mask,eligible)
                                point=float(val[mask].mean());val=np.nan_to_num(val)
                            else:
                                val=(z[recipe+'__ANSWER_Z_Q80__argmax']==answers.iloc[ai].target.to_numpy()).astype(float)
                                point=float(val[eligible[li]].mean())
                            expected=metrics[exp].loc[(metrics[exp].bank==bk)&(metrics[exp].variant==variant)&(metrics[exp].recipe==recipe)&(metrics[exp].cell==ce),
                                'within_answer_auc' if benchmark=='prm' else 'pb_first_error_accuracy'].item()
                            error=abs(point-expected);assert error<1e-12,(exp,ce,bk,variant,recipe,error)
                            checks.append(error);averages[recipe][li]+=val/8
            for recipe,v in averages.items():raw[benchmark,exp,variant,recipe]=v
    groupinfo[benchmark]=(groups,gi,cellid,eligible,indices)
print('Per-answer metrics reaggregated:',len(checks),'max difference',max(checks),flush=True)

identities=[]
for r in RECIPES:
    for v in VAR['level'][1:]:identities.append({'experiment':'level','recipe':r,'variant':v,'reference_variant':'original','reference_recipe':r,'comparison':'same_recipe_removal','family_size':40})
    for v,ref,kind in [('channels_answer','groups_answer','channel_vs_group'),('groups_cell','groups_answer','quota_groups'),('channels_cell','channels_answer','quota_channels'),('channels_cell','groups_answer','both_changes')]:
        identities.append({'experiment':'input','recipe':r,'variant':v,'reference_variant':ref,'reference_recipe':r,'comparison':kind,'family_size':40})
for r in RECIPES[2:]:
    identities.append({'experiment':'level','recipe':r,'variant':'original','reference_variant':'original','reference_recipe':'DS_FILTER_EQ','comparison':'original_vs_plain','family_size':16})
    identities.append({'experiment':'level','recipe':r,'variant':'original','reference_variant':'original','reference_recipe':r.split('_')[0]+'_EQ','comparison':'original_vs_matched_EQ','family_size':16})
rows=[]
for benchmark in ['prm','pb']:
    groups,gi,ci,eligible,indices=groupinfo[benchmark];ncell=ci.max()+1;ng=len(groups)
    counts=np.zeros((ng,ncell));np.add.at(counts,(gi[eligible],ci[eligible]),1)
    numerator=np.zeros((ng,ncell,len(identities)))
    for j,d in enumerate(identities):
        delta=raw[benchmark,d['experiment'],d['variant'],d['recipe']]-raw[benchmark,d['experiment'],d['reference_variant'],d['reference_recipe']]
        np.add.at(numerator[:,:,j],(gi[eligible],ci[eligible]),delta[eligible])
    point=(numerator.sum(0)/counts.sum(0)[:,None]).mean(0)
    distribution=np.empty((DRAW,len(identities)));rng=np.random.default_rng(SEED)
    for start in range(0,DRAW,200):
        n=min(200,DRAW-start);w=rng.multinomial(ng,np.full(ng,1/ng),size=n).astype(float)
        den=w@counts;num=(w@numerator.reshape(ng,-1)).reshape(n,ncell,-1)
        distribution[start:start+n]=(num/den[:,:,None]).mean(1)
    for j,d in enumerate(identities):
        lo,hi=np.quantile(distribution[:,j],[.025,.975]);k=d['family_size'];alo,ahi=np.quantile(distribution[:,j],[.025/k,1-.025/k])
        rows.append(dict(d,benchmark=benchmark,endpoint='within_answer_auc' if benchmark=='prm' else 'first_error_argmax_accuracy',
            delta=float(point[j]),lo95=float(lo),hi95=float(hi),lo_bonferroni=float(alo),hi_bonferroni=float(ahi),
            n_answers=int(eligible.sum()),n_source_groups=ng,n_cells=ncell,n_banks=8,draws=DRAW,
            interpretation='Exploratory reanalysis after review; never retroactively preregistered'))
    print('Paired intervals complete:',benchmark,flush=True)
pd.DataFrame(rows).to_csv(OUT/'DIRECT_CONTRASTS.csv',index=False)

absolute=[]
for exp,m in metrics.items():
    for (variant,recipe),t in m.groupby(['variant','recipe']):
        pr=t[t.cell.str.startswith('prm')];pb=t[t.cell.str.startswith('pb_')]
        absolute.append({'experiment':exp,'variant':variant,'recipe':recipe,'prm_auc':pr.within_answer_auc.mean(),
            'prmscore':pr.prmscore.mean(),'pb_first_error':pb.pb_first_error_accuracy.mean(),'pb_f1':pb.pb_official_f1.mean(),
            'pb_clean_flag':pb.clean_flag_rate.mean(),'prm_clean_flag':pr.clean_flag_rate.mean(),'n_banks':8,'pb_cells':8})
pd.DataFrame(absolute).to_csv(OUT/'ABSOLUTE_METRICS.csv',index=False)

anchor=[];marks=[]
for exp,path in RUNS.items():
    for p in (path/'chunks').glob('*.diagnostics.json'):
        ce,bk,var=p.name.removesuffix('.diagnostics.json').split('__');d=json.loads(p.read_text())['diagnostics']
        if exp=='level':anchor.append({'cell':ce,'bank':bk,'variant':var,'removed':','.join(d['removed']),'first_removed':d['removed'][0] if d['removed'] else '',
            'anchor':d['anchor'],'fallback':d['anchor_fallback'],'sign_flips':len(d['sign_flips'])})
        elif ce.startswith('prm'):
            for within in ['MEAN','HEM']:
                for kind,h in d[within+'_marks_distribution'].items():
                    total=sum(h['histogram'].values());assert total==h['n_answers']*h['n_columns']
                    marks.append({'bank':bk,'variant':var,'within':within,'kind':kind,'n_answers':h['n_answers'],'n_columns':h['n_columns'],
                        'mean_marks_per_answer_column':sum(int(k)*n for k,n in h['histogram'].items())/total,
                        'zero_answer_column_fraction':h['histogram'].get('0',0)/total})
pd.DataFrame(anchor).to_csv(OUT/'ANCHOR_AUDIT.csv',index=False);pd.DataFrame(marks).to_csv(OUT/'MARKS_UNITS.csv',index=False)

# The combined scheduler does NOT stitch the two independently selected recipes:
# its input variant is selected conditional on the LEVEL recipe.
ls=json.loads((RUNS['level']/'SELECTION.json').read_text());ins=json.loads((RUNS['input']/'SELECTION.json').read_text())
conditional=[x for x in ins['candidates'] if x['recipe']==ls['winner']['recipe'] and x['complete_8_banks_9_cells']]
selected_input=max(conditional,key=lambda x:x['mean_advantage'])
audit={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','sealed_artifacts_verified':verified,
    'per_answer_metric_aggregations_checked':len(checks),'max_metric_difference':max(checks),'bootstrap_draws':DRAW,'seed':SEED,
    'prm_eligible':int(groupinfo['prm'][3].sum()),'prm_source_groups':len(groupinfo['prm'][0]),'pb_erroneous':int(groupinfo['pb'][3].sum()),
    'pb_source_groups':len(groupinfo['pb'][0]),'level_random_candidates':sum(x['variant'].startswith('random') for x in ls['candidates']),
    'level_candidates':len(ls['candidates']),'input_winner':ins['winner'],'conditional_input_for_level_winner':selected_input,
    'scope':'Reaggregated stored per-answer scores/outcomes and audited code; no independent feature refit or rerun of raw-score metric evaluator',
    'input_files':used}
assert all(sha(Path(p))==h for p,h in used.items()),'Input changed during reanalysis'
dump('AUDIT.json',audit)
print(json.dumps({k:v for k,v in audit.items() if k!='input_files'}),flush=True)
