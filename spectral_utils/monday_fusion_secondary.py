"""Descriptive paired intervals for position, decision and within-group effects.

The benchmark's source-question universe and seed are shared by every cell/bank.
Only source sufficient statistics and a bounded bootstrap block are retained.
"""
import json
import numpy as np
import pandas as pd
from spectral_utils.monday_fusion_protocol import RUNS,dump
from spectral_utils.monday_fusion_metrics import prmscore_counts
from spectral_utils.monday_fusion_execution import RECIPES,storage_guard

def ratio(a,b):
    return np.divide(a,b,out=np.full(np.shape(a),np.nan),where=b>0)

def endpoint(values,kind):
    if kind=='within_answer_auc': return ratio(values[...,0],values[...,1])
    if kind=='prmscore':
        result=prmscore_counts(values)
        return np.where(values.sum(-1)>0,result,np.nan)
    if kind=='clean_flag_rate': return ratio(values[...,4],values[...,3])
    if kind=='pb_first_error_accuracy': return ratio(values[...,5],values[...,1])
    if kind=='pb_official_f1':
        ae=ratio(values[...,0],values[...,1]); ac=ratio(values[...,2],values[...,3])
        return np.where(np.isfinite(ae)&np.isfinite(ac),np.where(ae+ac>0,ratio(2*ae*ac,ae+ac),0.),np.nan)
    raise ValueError(kind)

def sufficient(z,key,kind):
    if kind=='within_answer_auc':
        # AUC is decision-invariant.
        prefix=key.rsplit('__',1)[0]
        return np.column_stack([z[prefix+'__auc_sum'],z[prefix+'__auc_count']])
    return z[key+('__confusion' if kind=='prmscore' else '__answer_counts')]

def paired_intervals(universe,local_groups,arrays,comparisons,draws=50000):
    """arrays maps (score key, endpoint) to group sufficient-statistic matrices."""
    index={k:i for i,k in enumerate(universe)}
    local=np.array([index[k] for k in local_groups])
    keys=list(arrays); starts=np.r_[0,np.cumsum([arrays[k].shape[1] for k in keys])]
    matrix=np.column_stack([arrays[k] for k in keys]).astype(float)
    columns={k:slice(starts[i],starts[i+1]) for i,k in enumerate(keys)}
    distributions=np.empty((draws,len(comparisons)),float)
    observed={k:float(endpoint(a.sum(0),k[1])) for k,a in arrays.items()}
    rng=np.random.default_rng(20261003)
    for first in range(0,draws,500):
        n=min(500,draws-first)
        w=rng.multinomial(len(universe),np.full(len(universe),1/len(universe)),size=n)[:,local].astype(float)
        totals=w@matrix
        computed={k:endpoint(totals[:,s],k[1]) for k,s in columns.items()}
        for j,(a,b,kind) in enumerate(comparisons): distributions[first:first+n,j]=computed[(a,kind)]-computed[(b,kind)]
    result=[]
    for j,(a,b,kind) in enumerate(comparisons):
        valid=np.isfinite(distributions[:,j]); values=distributions[valid,j]
        lo,hi=np.quantile(values,[.025,.975]) if len(values) else (np.nan,np.nan)
        result.append({'delta':observed[(a,kind)]-observed[(b,kind)],'lo95':lo,'hi95':hi,
                       'draws':draws,'valid_draws':int(valid.sum()),'n_source_groups':len(local),
                       'bootstrap_universe_source_groups':len(universe),'status':'OK' if valid.all() else 'UNDEFINED_BOOTSTRAP_DRAWS'})
    return result

def layer_contrasts(pop,experiment):
    out=RUNS[experiment]
    if not(out/'POSITION_DECISION_COMPLETE.json').exists(): raise RuntimeError('Complete position/decision layers first')
    selection=json.loads((out/'SELECTION.json').read_text(encoding='utf8'))
    reference='original' if experiment=='level_reduction_v1' else 'groups_answer'
    chosen=selection['winner']['variant'] if selection.get('winner') else reference
    variants={reference,chosen if chosen!='original' else reference}
    universes={True:np.unique(pop.groups[pop.prm]).astype(str),False:np.unique(pop.groups[~pop.prm]).astype(str)}
    directory=out/'secondary'; directory.mkdir(exist_ok=True)
    for path in sorted((out/'layers').glob('*.stats.npz')):
        ce,bk,var=path.name.removesuffix('.stats.npz').split('__')
        if var not in variants: continue
        stem=ce+'__'+bk+'__'+var; dest=directory/(stem+'.csv.gz')
        if (directory/(stem+'.complete.json')).exists(): continue
        storage_guard()
        is_prm=ce.startswith('prm'); z=np.load(path); pairs=[]; meta=[]; arrays={}; missing=[]
        endpoint_names=['within_answer_auc','prmscore','clean_flag_rate'] if is_prm else ['pb_first_error_accuracy','pb_official_f1','clean_flag_rate']
        def add(recipe,pos,rule,control,cpos,crule,comparison,kinds):
            a='__'.join([recipe,pos,rule]); b='__'.join([control,cpos,crule])
            for kind in kinds:
                identity={'bank':bk,'cell':ce,'variant':var,'recipe':recipe,'position_variant':pos,'decision_rule':rule,
                          'control':control,'control_position':cpos,'control_rule':crule,'comparison':comparison,
                          'endpoint':kind,'family':'secondary_descriptive','lo_corrected':None,'hi_corrected':None}
                try:
                    for key in [a,b]:
                        if (key,kind) not in arrays: arrays[(key,kind)]=sufficient(z,key,kind)
                except KeyError:
                    missing.append(dict(identity,status='FAILED_COVERAGE')); continue
                pairs.append((a,b,kind)); meta.append(identity)
        for recipe in RECIPES:
            for pos in ['POS','PRIOR','PRIOR_CF','EQUAL_CONTENT_POSITION','CF_CONTENT_ONLY','POSZ']:
                add(recipe,pos,'ANSWER_Z_Q80',recipe,'CONTENT','ANSWER_Z_Q80','position_vs_content',endpoint_names)
            for rule in ['R2_COUNT','ENTROPY_GATE_Q80','TOP_1','TOP_2','TOP_3']:
                add(recipe,'CONTENT',rule,recipe,'CONTENT','ANSWER_Z_Q80','decision_vs_q80',[k for k in endpoint_names if k not in ['within_answer_auc','pb_first_error_accuracy']])
            if recipe.startswith('HEM_'):
                add(recipe,'CONTENT','ANSWER_Z_Q80',recipe.replace('HEM_','MEAN_'),'CONTENT','ANSWER_Z_Q80','hem_vs_mean_within',endpoint_names)
            if recipe.endswith(('_DS','_SML','_MOM')):
                add(recipe,'CONTENT','ANSWER_Z_Q80',recipe.split('_')[0]+'_EQ','CONTENT','ANSWER_Z_Q80','matched_secondary',[k for k in endpoint_names if k!='within_answer_auc'])
        if pairs:
            results=paired_intervals(universes[is_prm],z['source_groups'],arrays,pairs)
            rows=[dict(a,**b) for a,b in zip(meta,results)]+missing
        else: rows=missing
        pd.DataFrame(rows).to_csv(dest,index=False)
        dump(directory/(stem+'.complete.json'),{'status':'COMPLETE','rows':len(rows)})
        print('Secondary intervals '+stem,flush=True)
    tables=[pd.read_csv(p) for p in sorted(directory.glob('*.csv.gz'))]
    secondary=pd.concat(tables,ignore_index=True); secondary.to_csv(out/'SECONDARY_CONTRASTS.csv',index=False)
    core=pd.read_csv(out/'CONTRASTS.csv')
    if 'family' in core: core=core[core.family!='secondary_descriptive']
    pd.concat([core,secondary],ignore_index=True).to_csv(out/'CONTRASTS.csv',index=False)
    dump(out/'SECONDARY_CONTRASTS_COMPLETE.json',{'status':'COMPLETE','reference_and_selected_variants':sorted(variants),'rows':len(secondary),
        'scope':'All recipes, reference and content-selected variant; position vs content, decision vs q80, HEM vs mean, matched non-AUC endpoints',
        'correction':'Descriptive 95% only, separate from registered primary Bonferroni family'})
