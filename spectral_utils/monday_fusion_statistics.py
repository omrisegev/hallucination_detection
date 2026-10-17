"""Paired source-question bootstrap, same draws for every bank/recipe."""
import json
import numpy as np
import pandas as pd
from spectral_utils.monday_fusion_protocol import RUNS,dump

def core_contrasts(pop,experiment):
    out=RUNS[experiment]; metrics=pd.read_csv(out/'METRICS.csv')
    prm=np.flatnonzero(pop.prm); source,inv=np.unique(pop.groups[prm],return_inverse=True)
    eligible=pop.eligible[prm]; counts=np.bincount(inv[eligible],minlength=len(source)).astype(float)
    sums=[]; identities=[]; missing=[]
    for path in sorted((out/'chunks').glob('prmbench*stats.npz')):
        cell,bank,variant=path.name.removesuffix('.stats.npz').split('__')
        z=np.load(path)
        legacy=np.load(RUNS['level_reduction_v1']/'chunks'/f'{cell}__{bank}__original.stats.npz')
        if not np.array_equal(z['answer_indices'],prm): raise RuntimeError('Unpaired PRMB answer ordering')
        for within in ['MEAN','HEM']:
            control=within+'_EQ__ANSWER_Z_Q80__auc'
            for between in ['DS','SML','MOM']:
                recipe=within+'_'+between; key=recipe+'__ANSWER_Z_Q80__auc'
                identity={'bank':bank,'variant':variant,'recipe':recipe,'control':within+'_EQ','comparison':'matched',
                          'endpoint':'within_answer_auc','position_variant':'CONTENT','decision_rule':'ANSWER_Z_Q80'}
                if control not in z or key not in z or not np.isfinite(z[key][eligible]).all() or not np.isfinite(z[control][eligible]).all():
                    missing.append(dict(identity,status='FAILED_COVERAGE')); continue
                delta=z[key][eligible]-z[control][eligible]
                sums.append(np.bincount(inv[eligible],weights=delta,minlength=len(source)))
                identities.append(identity)
                legacy_key='DS_FILTER_EQ__ANSWER_Z_Q80__auc'
                if legacy_key in legacy and np.isfinite(legacy[legacy_key][eligible]).all():
                    delta=z[key][eligible]-legacy[legacy_key][eligible]
                    sums.append(np.bincount(inv[eligible],weights=delta,minlength=len(source)))
                    identities.append(dict(identity,control='original_DS_FILTER_EQ',comparison='legacy_rule'))
    if not sums: raise RuntimeError('No full-coverage primary contrasts')
    a=np.stack(sums,1); draws=50000; distribution=np.empty((draws,len(sums)),np.float64)
    rng=np.random.default_rng(20261003)
    for start in range(0,draws,1000):
        n=min(1000,draws-start); w=rng.multinomial(len(source),np.full(len(source),1/len(source)),size=n).astype(float)
        distribution[start:start+n]=(w@a)/(w@counts)[:,None]
    multiplicity=240 if experiment=='level_reduction_v1' else 192
    rows=[]
    for j,identity in enumerate(identities):
        lo,hi=np.quantile(distribution[:,j],[.025,.975]); adj=np.quantile(distribution[:,j],[.025/multiplicity,1-.025/multiplicity])
        rows.append(dict(identity,status='OK',delta=float(a[:,j].sum()/counts.sum()),lo95=float(lo),hi95=float(hi),
                         lo_corrected=float(adj[0]) if identity['comparison']=='matched' else None,
                         hi_corrected=float(adj[1]) if identity['comparison']=='matched' else None,
                         family_size=multiplicity if identity['comparison']=='matched' else None,
                         n_source_groups=len(source),n_answers=int(eligible.sum()),draws=draws))
    verdict=[]
    for variant,recipe in sorted(set((d['variant'],d['recipe']) for d in identities)):
        indices=[j for j,d in enumerate(identities) if d['variant']==variant and d['recipe']==recipe and d['comparison']=='matched']
        valid=len(indices)==8
        # All PB cells must also have a score; do not select by omitting failed cells.
        subset=metrics[(metrics['variant']==variant)&(metrics['recipe']==recipe)&(metrics['position_variant']=='CONTENT')&(metrics['decision_rule']=='ANSWER_Z_Q80')]
        valid=valid and len(subset)==72 and bool((subset['coverage']==1).all())
        diagnostic_control=experiment=='level_reduction_v1' and variant.startswith('random_')
        v={'variant':variant,'recipe':recipe,'complete_8_banks_9_cells':valid,'mean_advantage':None,
           'selection_eligible':not diagnostic_control,
           'role':'diagnostic_random_control' if diagnostic_control else 'weighted_candidate'}
        if valid:
            dd=distribution[:,indices].mean(1); lo,hi=np.quantile(dd,[.025,.975])
            mean=float(np.mean([rows[j]['delta'] for j in indices]))
            v.update(mean_advantage=mean,lo95=float(lo),hi95=float(hi),matched_eq_significant_losses=sum(rows[j]['hi95']<0 for j in indices))
            legacy_indices=[j for j,d in enumerate(identities) if d['variant']==variant and d['recipe']==recipe and d['comparison']=='legacy_rule']
            losses=sum(rows[j]['hi95']<0 for j in legacy_indices); wins=sum(rows[j]['lo95']>0 for j in legacy_indices)
            v.update(legacy_no_bank_significant_loss=len(legacy_indices)==8 and losses==0,
                     legacy_significant_wins=wins,legacy_significant_losses=losses,
                     legacy_full_rule_qualifies=len(legacy_indices)==8 and losses==0 and wins>=5)
            absolute=subset[subset.cell=='prmbench_qwen3_8b'].within_answer_auc
            v['absolute_8_bank_mean']=float(absolute.mean()) if len(absolute)==8 else None
            rows.append({'bank':'EQUAL_MEAN_8_BANKS','variant':variant,'recipe':recipe,'endpoint':'within_answer_auc','comparison':'matched',
                         'position_variant':'CONTENT','decision_rule':'ANSWER_Z_Q80','delta':mean,'lo95':float(lo),'hi95':float(hi),
                         'status':'OK','n_source_groups':len(source),'n_answers':int(eligible.sum()),'draws':draws})
        verdict.append(v)
    pd.DataFrame(rows+missing).to_csv(out/'CONTRASTS.csv',index=False)
    # Preserve the registered matched-EQ advantage criterion. The plain mean is
    # a reported reference, not an artificial zero on a different reference scale.
    eligible_v=[v for v in verdict if v['complete_8_banks_9_cells'] and v['selection_eligible']]
    winner=max(eligible_v,key=lambda v:v['mean_advantage']) if eligible_v else None
    legacy_eligible=[v for v in verdict if v['selection_eligible'] and v.get('legacy_full_rule_qualifies')]
    legacy_winner=max(legacy_eligible,key=lambda v:v['absolute_8_bank_mean']) if legacy_eligible else {'variant':'original','recipe':'DS_FILTER_EQ','reason':'No complete recipe has 5 significant wins and no loss'}
    dump(out/'SELECTION.json',{'status':'DEVELOPMENT_CONTENT_SELECTION_PENDING_LAYERS_REVIEW','winner':winner,
                               'candidates':verdict,'legacy_Step457_on_new_per_cell_results':legacy_winner,
                               'analysis_revision':'REVIEW_CORRECTIONS_V2.json; original 20261003 selection preserved',
                               'criterion':'Largest equal-weight 8-bank advantage over the same-variant matched group EQ; never direct evidence that removal improves the original',
                               'plain_mean_role':'Reported absolute and legacy comparison only; not a zero-gain candidate'})
