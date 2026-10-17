"""Matched estimator-input experiments: fixed scores/partition, changed binary votes."""
import copy
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from spectral_utils.monday_fusion_protocol import RUNS,dump
from spectral_utils.monday_fusion_core import quota_marks,between_weights
from spectral_utils.monday_fusion_execution import check_frozen,code_seal,restore_content,score_state,save_chunk,native,completed_chunk

def accuracy(votes,labels,anchor,metadata):
    import er_stage_a as SA
    truth=SA.truth(votes,labels); rows=[]; estimates={}
    for method in ['DS','MOM','SML']:
        row=dict(metadata,estimator=method,n_steps=len(votes),n_votes=votes.shape[1],true_prevalence=truth['prevalence'])
        try:
            if votes.shape[1]<3: raise ValueError('Fewer than 3 votes')
            est=SA.em_estimate(votes,'ds') if method=='DS' else SA.tensor_mom_estimate(votes,anchor) if method=='MOM' else SA.sml_estimate(votes,anchor)
            if method!='SML':
                row.update(prevalence_error=abs(est['prevalence']-truth['prevalence']),estimated_prevalence=est['prevalence'],
                           sensitivity_MAE=float(np.abs(est['psi']-truth['psi']).mean()),specificity_MAE=float(np.abs(est['eta']-truth['eta']).mean()))
            rank=est['t'] if method=='SML' else est['pi']
            row.update(reliability_spearman=float(spearmanr(rank,truth['pi']).statistic),status='OK')
            estimates[method]={'estimated':est,'truth':truth}
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc: row.update(status='FAILED',reason=str(exc))
        rows.append(native(row))
    return rows,estimates

def estimator_state(data,original,bk,ce,vote_type,quota):
    import ds_group_weights as GW
    state={**original,'diagnostics':copy.deepcopy(original['diagnostics']),'failures':dict(original['failures']),
           'scores':{k:v for k,v in original['scores'].items() if k in ('ALL_EQ','DS_FILTER_EQ')},'recipes':{}}
    bank=state['bank']; diag=state['diagnostics']; records=[]
    if 'groups' not in state:
        return state,records
    groups=state['groups']; ai=state['anchor']; ganchor=int(groups[ai]); surv=state['surv']
    channel=quota_marks(bank['V'][:,surv],bank['off'],bank['key'][:,surv],quota=='cell')
    for within in ['MEAN','HEM']:
        base=original['recipes'].get(within+'_EQ')
        if base is None: continue
        z=base['Z']; key=data.env['KEYG'][bank['rows'],:z.shape[1]]
        group_votes=quota_marks(z,bank['off'],key,quota=='cell')
        inputs=channel if vote_type=='channels' else group_votes
        metadata={'bank':bk,'cell':ce,'within':within,'vote_type':vote_type,'quota':quota}
        if data.pop.prm[bank['answers'][0]]:
            ar,est=accuracy(inputs,data.pop.labels[bank['rows']],ai if vote_type=='channels' else ganchor,metadata)
            records+=ar; diag[within+'_estimator_accuracy']=est
        counts=np.add.reduceat((inputs>0).astype(int),bank['off'][:-1],axis=0)
        error=np.array([bool(data.pop.labels[data.pop.off[i]:data.pop.off[i+1]].any()) if data.pop.prm[i] else data.pop.target[i]>=0 for i in bank['answers']])
        diag[within+'_marks_distribution']={}
        for label,mask in [('clean',~error),('erroneous',error)]:
            u,c=np.unique(counts[mask],return_counts=True)
            diag[within+'_marks_distribution'][label]={'histogram':dict(zip(map(str,u),map(int,c))),'n_answers':int(mask.sum()),'n_columns':inputs.shape[1]}
        diag[within+'_vote_correlation']=np.corrcoef(inputs,rowvar=False)
        for method in ['EQ','DS','SML','MOM']:
            name=within+'_'+method
            try:
                w,est=between_weights(method,z,group_votes,state['X'][:,ai],ganchor,
                    channel_votes=channel if vote_type=='channels' else None,groups=groups,channel_anchor=ai)
                score=GW.weighted_group_score(z,w,signed=method=='SML')
                if method=='SML' and name in original['scores']:
                    if np.max(np.abs(score-original['scores'][name]))>1e-9: raise RuntimeError('SML changed despite fixed continuous group scores')
                state['scores'][name]=score
                state['failures'].pop(name,None)
                state['recipes'][name]={'Z':z,'votes':group_votes,'weights':w,'within':base['within'],'estimator':est}
                mass=np.abs(w)/np.abs(w).sum()
                diag[name]={'weights':w,'estimator':est,'effective_groups':float(1/(mass@mass)),
                            'negative_weights':int((w<0).sum()),'vote_type':vote_type,'quota':quota}
                if method=='DS':
                    top=int(np.argmax(w))
                    diag[within+'_top_DS_group']={'group':top,'members':[bank['names'][surv[j]] for j in np.flatnonzero(groups==top)],'weight':float(w[top])}
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc: state['failures'][name]=str(exc)
    return state,records

def run_estimators(data,quota):
    check_frozen(); out=RUNS['estimator_input_v1']; level=RUNS['level_reduction_v1']
    code_seal(out,'estimator_'+quota)
    status=json.loads((level/'RUN_STATUS.json').read_text(encoding='utf8'))
    if status['status'] not in ['CONTENT_COMPLETE_LAYERS_PENDING','COMPLETE']: raise RuntimeError('LEVEL core must complete first')
    if quota=='cell' and not(out/'ANSWER_QUOTA_COMPLETE.json').exists(): raise RuntimeError('Experiment 2a must complete before 2b')
    all_accuracy=[]; count=0
    for ce in sorted(set(data.pop.cells)):
        for bk in data.env['BN']:
            original=restore_content(data,bk,ce)
            for vote_type in ['channels','groups']:
                variant=vote_type+'_'+quota; stem=ce+'__'+bk+'__'+variant; started=time.perf_counter()
                if not completed_chunk(out/'chunks',stem):
                    state,records=estimator_state(data,original,bk,ce,vote_type,quota)
                    metrics,stats=score_state(data,state,bk,ce,variant,out,started)
                    dump(out/'chunks'/(stem+'.accuracy.json'),records)
                    save_chunk(out,stem,state,metrics,stats,extra_artifacts=['.accuracy.json'])
                all_accuracy+=json.loads((out/'chunks'/(stem+'.accuracy.json')).read_text(encoding='utf8'))
                count+=1
                dump(out/'RUN_STATUS.json',{'status':'RUNNING','quota':quota,'completed_cells':count,'expected_cells':144})
                print(f'Estimator inputs {stem}: checkpoint complete',flush=True)
    pd.DataFrame(all_accuracy).to_csv(out/('ESTIMATOR_ACCURACY_'+quota+'.csv'),index=False)
    dump(out/(quota.upper()+'_QUOTA_COMPLETE.json'),{'status':'COMPLETE','cells':count})
    tables=[pd.read_csv(p) for p in sorted((out/'chunks').glob('*.metrics.csv'))]
    pd.concat(tables,ignore_index=True).to_csv(out/'METRICS.csv',index=False)
    acc=[pd.read_csv(p) for p in sorted(out.glob('ESTIMATOR_ACCURACY_*.csv'))]
    pd.concat(acc,ignore_index=True).to_csv(out/'ESTIMATOR_ACCURACY.csv',index=False)
    dump(out/'DIAGNOSTICS.json',{p.stem:json.loads(p.read_text(encoding='utf8')) for p in (out/'chunks').glob('*.diagnostics.json')})
    dump(out/'RUN_STATUS.json',{'status':'CONTENT_COMPLETE_LAYERS_PENDING' if quota=='cell' else 'EXPERIMENT_2A_COMPLETE','completed_cells':count})
