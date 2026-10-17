"""Position, decision and LR layers, with explicit checkpoints and prerequisites."""
import copy
import json
import time
import numpy as np
import pandas as pd
from spectral_utils.monday_fusion_protocol import RUNS,WORK,dump,sha
from spectral_utils.monday_fusion_execution import RECIPES,check_frozen,code_seal,restore_content,native,storage_guard,completed_chunk
from spectral_utils.monday_fusion_core import quota_marks
from spectral_utils.monday_fusion_position import variants
from spectral_utils.monday_fusion_decision import raw_pool,raw_variant,entropy_feature,rules
from spectral_utils.monday_fusion_metrics import evaluate,answer_z,top_flags
from spectral_utils.monday_fusion_nulls import nulls

def prerequisites():
    check_frozen()
    if not (RUNS['estimator_input_v1']/'CELL_QUOTA_COMPLETE.json').exists():
        raise RuntimeError('Both estimator-input experiments must complete before layers')

def input_state(data,experiment,bk,ce,variant):
    import ds_group_weights as GW
    if experiment=='level_reduction_v1': return restore_content(data,bk,ce,variant)
    state=restore_content(data,bk,ce); vote_type,quota=variant.split('_')
    if not completed_chunk(RUNS[experiment]/'chunks',ce+'__'+bk+'__'+variant): raise RuntimeError('Missing estimator-input checkpoint')
    record=json.loads((RUNS[experiment]/'chunks'/(ce+'__'+bk+'__'+variant+'.diagnostics.json')).read_text(encoding='utf8'))
    state['diagnostics']=record['diagnostics']; state['failures']=record['failures']
    bank=state['bank']; bank['quota']=quota; bank['vote_type']=vote_type
    bank['marks']=quota_marks(bank['V'],bank['off'],bank['key'],quota=='cell')
    state['scores']={k:v for k,v in state['scores'].items() if k in ['ALL_EQ','DS_FILTER_EQ']}
    for name in list(state['recipes']):
        if name not in record['recipes']:
            del state['recipes'][name]; continue
        r=state['recipes'][name]; saved=record['recipes'][name]; z=r['Z']
        r['weights']=np.asarray(saved['weights'],float); r['estimator']=saved['estimator']
        r['votes']=quota_marks(z,bank['off'],data.env['KEYG'][bank['rows'],:z.shape[1]],quota=='cell')
        state['scores'][name]=GW.weighted_group_score(z,r['weights'],signed=name.endswith('_SML'))
    return state

def score_layer_chunk(data,state,bk,ce,variant,pool_raw,entropy):
    bank=state['bank']; off=bank['off']; all_scores={}; metrics=[]; diagnostics={}; suff={}; cache={}
    source,gi=np.unique(data.pop.groups[bank['answers']],return_inverse=True)
    suff['source_groups']=source.astype(str)
    raw=raw_variant(data,bank,bk,pool_raw)
    for recipe in RECIPES:
        if recipe not in state['scores']:
            diagnostics[recipe]={'failure':state['failures']}
            for pos in ('CONTENT','POS','PRIOR','PRIOR_CF'):
                for rule in ('ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80','TOP_1','TOP_2','TOP_3'):
                    metrics.append({'bank':bk,'variant':variant,'recipe':recipe,'position_variant':pos,'decision_rule':rule,'cell':ce,
                                    'coverage':0.,'failures':len(bank['answers']),'failure_reason':json.dumps(native(state['failures'])),
                                    'n_answers':0,'n_total':len(bank['answers']),'review_status':'PENDING'})
            continue
        position,diag,failed=variants(data,state,recipe,cache)
        diagnostics[recipe]={'positions':diag,'failures':failed}
        for pos,score in position.items():
            key=recipe+'__'+pos; all_scores[key]=score
            rr=raw if recipe=='ALL_EQ' else raw[:,state['surv']]
            flags,detectors,dd=rules(score,bank,rr,entropy[bank['answers']],data.pop.uids[bank['answers']])
            diagnostics[recipe].setdefault('decisions',{})[pos]=dd
            for k in (1,2,3): flags['TOP_'+str(k)]=top_flags(score,off,np.full(len(off)-1,k))
            for rule,ff in flags.items():
                metric,per=evaluate(data.pop,bank,score,ff,detectors.get(rule))
                row={'bank':bk,'variant':variant,'recipe':recipe,'position_variant':pos,'decision_rule':rule,'cell':ce,
                     'review_status':'PENDING','top2_post_hoc':rule=='TOP_2','n_total':len(bank['answers']),**metric}
                metrics.append(native(row))
                prefix=key+'__'+rule
                if rule=='ANSWER_Z_Q80':
                    mask=np.isfinite(per['auc'])
                    suff[key+'__auc_sum']=np.bincount(gi[mask],weights=per['auc'][mask],minlength=len(source))
                    suff[key+'__auc_count']=np.bincount(gi[mask],minlength=len(source))
                confusion=np.zeros((len(source),4),np.int64); np.add.at(confusion,gi,per['confusion'])
                suff[prefix+'__confusion']=confusion
                err=per['erroneous']; target=data.pop.target[bank['answers']]
                per_counts=np.column_stack([(per['first']==target)&err,err,(per['first']==target)&~err,~err,(per['flag_count']>0)&~err,(per['argmax']==target)&err])
                group_counts=np.zeros((len(source),6),np.int64); np.add.at(group_counts,gi,per_counts.astype(int))
                suff[prefix+'__answer_counts']=group_counts
        for pos,reason in failed.items():
            for rule in ('ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80','TOP_1','TOP_2','TOP_3'):
                metrics.append({'bank':bk,'variant':variant,'recipe':recipe,'position_variant':pos,'decision_rule':rule,'cell':ce,
                                'coverage':0.,'failures':len(bank['answers']),'failure_reason':reason,'n_answers':0,'n_total':len(bank['answers']),'review_status':'PENDING'})
    # All position/decision scores share fixed-score label nulls, computed together.
    diagnostics['nulls']=nulls(data.pop,bank,all_scores)
    return metrics,diagnostics,suff

def run_position_decision(data):
    prerequisites(); pool=raw_pool(data); entropy=entropy_feature(data)
    gate=WORK/'POSITION_REPLAY_GATE.json'
    if not gate.exists() or json.loads(gate.read_text(encoding='utf8')).get('status')!='PASS':
        from spectral_utils.monday_fusion_position_replay import replay_positions
        replay_positions(data)
    for experiment in RUNS:
        out=RUNS[experiment]; code_seal(out,'position_decision')
        directory=out/'layers'; directory.mkdir(exist_ok=True)
        variants_to_run=['original','remove_2','remove_3','random_2','random_3'] if experiment=='level_reduction_v1' else ['channels_answer','groups_answer','channels_cell','groups_cell']
        for ce in sorted(set(data.pop.cells)):
            for bk in data.env['BN']:
                for variant in variants_to_run:
                    stem=ce+'__'+bk+'__'+variant
                    if completed_chunk(directory,stem): continue
                    storage_guard()
                    start=time.perf_counter(); state=input_state(data,experiment,bk,ce,variant)
                    metrics,diagnostics,suff=score_layer_chunk(data,state,bk,ce,variant,pool,entropy)
                    for row in metrics: row['runtime_s']=time.perf_counter()-start
                    pd.DataFrame(metrics).to_csv(directory/(stem+'.metrics.csv.gz'),index=False)
                    dump(directory/(stem+'.diagnostics.json'),native(diagnostics))
                    np.savez_compressed(directory/(stem+'.stats.npz'),**suff)
                    dump(directory/(stem+'.complete.json'),{'status':'COMPLETE','artifacts':{s:sha(directory/(stem+s)) for s in ['.metrics.csv.gz','.diagnostics.json','.stats.npz']}})
                    print('Position/decision checkpoint '+experiment+'/'+stem,flush=True)
        storage_guard()
        pd.concat([pd.read_csv(p) for p in sorted(directory.glob('*.metrics.csv.gz'))],ignore_index=True).to_csv(out/'METRICS.csv',index=False)
        dump(out/'POSITION_DECISION_COMPLETE.json',{'status':'COMPLETE','chunks':len(list(directory.glob('*.complete.json')))})

def subbank(bank,mask):
    indices=np.flatnonzero(mask); oldoff=bank['off']
    steps=np.concatenate([np.arange(oldoff[i],oldoff[i+1]) for i in indices])
    return {**bank,'answers':bank['answers'][indices],'rows':bank['rows'][steps],
            'off':np.r_[0,np.cumsum(np.diff(oldoff)[indices])],'V':bank['V'][steps]},steps

def lr_rows(data,state,bk,ce,variant,roster,with_position):
    from spectral_utils.monday_fusion_lr import nested_lr
    from spectral_utils.monday_fusion_metrics import prmscore_counts
    bank=state['bank']; columns=np.arange(bank['V'].shape[1]) if roster=='full' else state['surv']
    if not len(columns): raise ValueError('LR has no survivor columns')
    pos=data.env['POSZ'][bank['rows']] if with_position else None
    fit=nested_lr(data.pop,bank,columns,pos); rows=[]; folds=[]
    for rule in ['ANSWER_Z_Q80','TOP_1','TOP_2','TOP_3']:
        fm=[]; counts=np.zeros(4)
        for fold in range(5):
            mask=data.pop.fold[bank['answers']]==fold; local,steps=subbank(bank,mask)
            score=fit['oof'][steps]
            flags=None if rule=='ANSWER_Z_Q80' else top_flags(score,local['off'],np.full(len(local['answers']),int(rule[-1])))
            m,detail=evaluate(data.pop,local,score,flags); counts+=detail['confusion'].sum(0); fm.append(m)
            folds.append({'fold':fold,'decision_rule':rule,**m})
        totals={k:sum(m[k] for m in fm) for k in ['n_answers','n_steps','n_auc_answers','n_erroneous','n_clean']}
        def weighted(metric,weight):
            usable=[m for m in fm if m.get(metric) is not None and m.get(weight,0)>0]
            return sum(m[metric]*m[weight] for m in usable)/sum(m[weight] for m in usable) if usable else None
        first=weighted('pb_first_error_accuracy','n_erroneous'); ae=weighted('pb_decision_error_accuracy','n_erroneous')
        clean=weighted('clean_flag_rate','n_clean')
        ac=None if clean is None else 1-clean
        f1=None if ae is None else (2*ae*ac/(ae+ac) if ae+ac>0 else 0.)
        aucs=[m['wrong_answer_auroc'] for m in fm if m['wrong_answer_auroc'] is not None]
        error_distribution={}
        for m in fm:
            for k,v in (m.get('error_count_distribution') or {}).items(): error_distribution[k]=error_distribution.get(k,0)+v
        rows.append({'bank':bk,'variant':variant,'recipe':'LR_'+roster,'position_variant':'POS' if with_position else 'CONTENT',
                     'decision_rule':rule,'cell':ce,'score_scope':'OOF','within_answer_auc':weighted('within_answer_auc','n_auc_answers'),
                     'prmscore':float(prmscore_counts(counts)) if data.pop.prm[bank['answers'][0]] else None,
                     'pb_first_error_accuracy':first,'pb_decision_error_accuracy':ae,'pb_official_f1':f1,'clean_flag_rate':clean,
                     'wrong_answer_auroc':float(np.mean(aucs)) if aucs else None,'answer_auc_aggregation':'mean within-fold AUROC; never pooled',
                     'decision_calibration':'separately within each held-out source fold without labels','coverage':1.,'failures':0,
                     'error_count_distribution':error_distribution or None,
                     'top2_post_hoc':rule=='TOP_2','review_status':'PENDING',**totals})
        score=fit['in_sample']; flags=None if rule=='ANSWER_Z_Q80' else top_flags(score,bank['off'],np.full(len(bank['answers']),int(rule[-1])))
        m,_=evaluate(data.pop,bank,score,flags)
        rows.append({'bank':bk,'variant':variant,'recipe':'LR_'+roster,'position_variant':'POS' if with_position else 'CONTENT',
                     'decision_rule':rule,'cell':ce,'score_scope':'IN_SAMPLE_DIAGNOSTIC','review_status':'PENDING','top2_post_hoc':rule=='TOP_2',**m})
    return rows,folds,{'folds':fit['diagnostics'],'unknown_post_error_steps':fit['n_unknown_post_error_steps'],'in_sample_C':fit['in_sample_C']}

def run_lr(data):
    from sklearn.exceptions import ConvergenceWarning
    prerequisites(); out=RUNS['level_reduction_v1']; code_seal(out,'lr')
    if not all((p/'POSITION_DECISION_COMPLETE.json').exists() for p in RUNS.values()):
        raise RuntimeError('Position/decision layers must complete before LR aggregation')
    directory=out/'lr'; directory.mkdir(exist_ok=True)
    for ce in sorted(set(data.pop.cells)):
        for bk in data.env['BN']:
            for variant in ['original','remove_2','remove_3','random_2','random_3']:
                state=restore_content(data,bk,ce,variant)
                for roster in ('full','survivors'):
                    for position in (False,True):
                        stem=f'{ce}__{bk}__{variant}__{roster}__pos{int(position)}'
                        if completed_chunk(directory,stem): continue
                        storage_guard()
                        started=time.perf_counter()
                        try:
                            rows,folds,diag=lr_rows(data,state,bk,ce,variant,roster,position)
                            status='COMPLETE'
                        except (ValueError,ArithmeticError,ConvergenceWarning,np.linalg.LinAlgError) as exc:
                            status='ARM_FAILURE'; diag={'failure':str(exc)}; folds=[]
                            rows=[{'bank':bk,'variant':variant,'recipe':'LR_'+roster,'position_variant':'POS' if position else 'CONTENT',
                                   'decision_rule':rule,'cell':ce,'score_scope':scope,'coverage':0.,'failures':len(state['bank']['answers']),
                                   'n_answers':0,'n_total':len(state['bank']['answers']),'failure_reason':str(exc),'review_status':'PENDING'}
                                  for rule in ['ANSWER_Z_Q80','TOP_1','TOP_2','TOP_3'] for scope in ['OOF','IN_SAMPLE_DIAGNOSTIC']]
                        for row in rows: row['runtime_s']=time.perf_counter()-started
                        pd.DataFrame(rows).to_csv(directory/(stem+'.metrics.csv'),index=False)
                        dump(directory/(stem+'.fits.json'),native(diag)); dump(directory/(stem+'.fold_metrics.json'),native(folds))
                        dump(directory/(stem+'.complete.json'),{'status':status,'runtime_s':time.perf_counter()-started,
                             'artifacts':{s:sha(directory/(stem+s)) for s in ['.metrics.csv','.fits.json','.fold_metrics.json']}})
                        print('LR checkpoint '+stem+' '+status,flush=True)
    table=pd.concat([pd.read_csv(p) for p in directory.glob('*.metrics.csv')],ignore_index=True)
    table.to_csv(out/'LR_METRICS.csv',index=False)
    base=pd.read_csv(out/'METRICS.csv'); base=base[~base['recipe'].str.startswith('LR_')]
    pd.concat([base,table],ignore_index=True).to_csv(out/'METRICS.csv',index=False)
    dump(out/'LR_COMPLETE.json',{'status':'COMPLETE','chunks':len(list(directory.glob('*.complete.json'))),'failure_rows':int((table.coverage==0).sum())})
    # Experiment 2 uses unchanged original feature banks/survivors, so identical LR fits are reused.
    other=RUNS['estimator_input_v1']; refs=[]
    for variant in ['channels_answer','groups_answer','channels_cell','groups_cell']:
        shared=table[table['variant']=='original'].copy(); shared['variant']=variant
        shared['reference_source']=str(out/'LR_METRICS.csv'); shared['reference_sha256']=sha(out/'LR_METRICS.csv'); refs.append(shared)
    lr2=pd.concat(refs,ignore_index=True); lr2.to_csv(other/'LR_METRICS.csv',index=False)
    base=pd.read_csv(other/'METRICS.csv'); base=base[~base['recipe'].str.startswith('LR_')]
    pd.concat([base,lr2],ignore_index=True).to_csv(other/'METRICS.csv',index=False)
    dump(other/'LR_COMPLETE.json',{'status':'COMPLETE_REUSED_IDENTICAL_BANKS','source':str(out/'LR_METRICS.csv'),'sha256':sha(out/'LR_METRICS.csv')})
