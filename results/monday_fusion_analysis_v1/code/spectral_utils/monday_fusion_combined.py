"""Full-population interaction check; final decision remains Omri's choice."""
import json
import time
import numpy as np
import pandas as pd
from spectral_utils.monday_fusion_protocol import ROOT,WORK,RUNS,dump,sha
from spectral_utils.monday_fusion_execution import restore_content,native,storage_guard,completed_chunk,code_seal
from spectral_utils.monday_fusion_estimators import estimator_state
from spectral_utils.monday_fusion_decision import raw_pool,raw_variant,entropy_feature,rules
from spectral_utils.monday_fusion_metrics import evaluate,top_flags

OUT=ROOT/'results/monday_combined_recipe_v1/run_20261003'

def proposal():
    for out in RUNS.values():
        if not(out/'LR_COMPLETE.json').exists() or not(out/'POSITION_DECISION_COMPLETE.json').exists():
            raise RuntimeError('Complete layers and LR are required before combined recipe')
    level=json.loads((RUNS['level_reduction_v1']/'SELECTION.json').read_text(encoding='utf8'))
    inputs=json.loads((RUNS['estimator_input_v1']/'SELECTION.json').read_text(encoding='utf8'))
    winner=level['winner']
    if winner is None: raise RuntimeError('No full-coverage LEVEL recipe')
    # Keep within/between fixed from experiment 1, choose input variant for that SAME recipe.
    eligible=[r for r in inputs['candidates'] if r['recipe']==winner['recipe'] and r['complete_8_banks_9_cells']]
    input_winner=max(eligible,key=lambda r:r['mean_advantage']) if eligible else {'variant':'groups_answer'}
    vote_type,quota=input_winner['variant'].split('_')
    return {'level_variant':winner['variant'],'recipe':winner['recipe'],'vote_type':vote_type,'quota':quota,
            'position_variant':'CONTENT','selection_scope':'Registered content-primary endpoint; position variants remain diagnostic',
            'controls':['BASE','LEVEL_ONLY','INPUT_ONLY','COMBINED'],
            'level_selection_sha256':sha(RUNS['level_reduction_v1']/'SELECTION.json'),
            'input_selection_sha256':sha(RUNS['estimator_input_v1']/'SELECTION.json')}

def run_combined(data):
    spec=proposal(); OUT.mkdir(parents=True,exist_ok=True)
    code_seal(OUT,'combined')
    if (OUT/'PROPOSAL.json').exists():
        if json.loads((OUT/'PROPOSAL.json').read_text(encoding='utf8'))!=spec: raise RuntimeError('Combined proposal changed')
    else: dump(OUT/'PROPOSAL.json',spec)
    directory=OUT/'chunks'; directory.mkdir(exist_ok=True)
    pool=raw_pool(data); entropy=entropy_feature(data); chosen=spec['recipe']
    input_record={'version':4,'parent_manifest':data.manifest,'raw_manifest':str(WORK/'RAW_INPUT_MANIFEST_V3.json'),
        'raw_manifest_sha256':sha(WORK/'RAW_INPUT_MANIFEST_V3.json'),'selected_components':spec}
    if (OUT/'INPUT_MANIFEST.json').exists():
        if json.loads((OUT/'INPUT_MANIFEST.json').read_text(encoding='utf8'))!=input_record: raise RuntimeError('Combined input manifest changed')
    else: dump(OUT/'INPUT_MANIFEST.json',input_record)
    control=chosen.split('_')[0]+'_EQ' if chosen.startswith(('MEAN_','HEM_')) else 'DS_FILTER_EQ'
    for ce in sorted(set(data.pop.cells)):
        for bk in data.env['BN']:
            for condition in spec['controls']:
                stem=ce+'__'+bk+'__'+condition
                if completed_chunk(directory,stem): continue
                storage_guard()
                variant=spec['level_variant'] if condition in ('LEVEL_ONLY','COMBINED') else 'original'
                state=restore_content(data,bk,ce,variant)
                if condition in ('INPUT_ONLY','COMBINED'):
                    state,_=estimator_state(data,state,bk,ce,spec['vote_type'],spec['quota'])
                bank=state['bank']; raw=raw_variant(data,bank,bk,pool); rows=[]; arrays={'answer_indices':bank['answers'],'global_rows':bank['rows']}
                source,gi=np.unique(data.pop.groups[bank['answers']],return_inverse=True); suff={'source_groups':source.astype(str)}
                for recipe in list(dict.fromkeys([chosen,control,'DS_FILTER_EQ','ALL_EQ'])):
                    if recipe not in state['scores']:
                        for rule in ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80','TOP_1','TOP_2','TOP_3']:
                            rows.append({'bank':bk,'condition':condition,'recipe':recipe,'decision_rule':rule,'cell':ce,'coverage':0.,'failures':len(bank['answers']),
                                         'failure_reason':json.dumps(native(state['failures']))})
                        continue
                    score=state['scores'][recipe]; rawcols=raw if recipe=='ALL_EQ' else raw[:,state['surv']]
                    decisions,detectors,dd=rules(score,bank,rawcols,entropy[bank['answers']],data.pop.uids[bank['answers']])
                    for k in (1,2,3): decisions['TOP_'+str(k)]=top_flags(score,bank['off'],np.full(len(bank['answers']),k))
                    for rule,flags in decisions.items():
                        m,detail=evaluate(data.pop,bank,score,flags,detectors.get(rule))
                        rows.append({'bank':bk,'condition':condition,'recipe':recipe,'decision_rule':rule,'cell':ce,'review_status':'PENDING','top2_post_hoc':rule=='TOP_2',**m})
                        if recipe in (chosen,control):
                            arrays[recipe+'__'+rule+'__flags']=flags
                            if rule=='ANSWER_Z_Q80': arrays[recipe+'__auc']=detail['auc']
                            key=recipe+'__CONTENT__'+rule
                            cm=np.zeros((len(source),4),int); np.add.at(cm,gi,detail['confusion']); suff[key+'__confusion']=cm
                            err=detail['erroneous']; target=data.pop.target[bank['answers']]
                            ac=np.column_stack([(detail['first']==target)&err,err,(detail['first']==target)&~err,~err,(detail['flag_count']>0)&~err,(detail['argmax']==target)&err])
                            am=np.zeros((len(source),6),int); np.add.at(am,gi,ac); suff[key+'__answer_counts']=am
                            if rule=='ANSWER_Z_Q80':
                                valid=np.isfinite(detail['auc']); prefix=recipe+'__CONTENT'
                                suff[prefix+'__auc_sum']=np.bincount(gi[valid],weights=detail['auc'][valid],minlength=len(source))
                                suff[prefix+'__auc_count']=np.bincount(gi[valid],minlength=len(source))
                    if recipe in (chosen,control): arrays[recipe+'__score']=score
                pd.DataFrame(rows).to_csv(directory/(stem+'.metrics.csv'),index=False)
                np.savez_compressed(directory/(stem+'.stats.npz'),**suff)
                dump(directory/(stem+'.fit.json'),native({'diagnostics':state['diagnostics'],'failures':state['failures'],
                    'recipes':{r:{'weights':v['weights']} for r,v in state['recipes'].items()},'level_variant':variant}))
                dump(directory/(stem+'.complete.json'),{'status':'COMPLETE','artifacts':{s:sha(directory/(stem+s)) for s in ['.stats.npz','.metrics.csv','.fit.json']}})
                print('Combined full-population checkpoint '+stem,flush=True)
                dump(OUT/'RUN_STATUS.json',{'status':'RUNNING','last_checkpoint':stem,'completed_chunks':len(list(directory.glob('*.complete.json'))),'expected_chunks':288})
    table=pd.concat([pd.read_csv(p) for p in directory.glob('*.metrics.csv')],ignore_index=True); table.to_csv(OUT/'METRICS.csv',index=False)
    candidates=[]
    for condition in spec['controls']:
        t=table[(table.condition==condition)&(table.decision_rule=='ANSWER_Z_Q80')]
        a=t[t.recipe==chosen]; b=t[t.recipe==control]
        complete=len(a)==72 and len(b)==72 and (a.coverage==1).all() and (b.coverage==1).all()
        gain=None
        if complete:
            ap=a[a.cell=='prmbench_qwen3_8b'].set_index('bank'); bp=b[b.cell=='prmbench_qwen3_8b'].set_index('bank')
            gain=float((ap.within_answer_auc-bp.within_answer_auc).mean())
        candidates.append({'condition':condition,'complete':bool(complete),'mean_advantage':gain})
    valid=[c for c in candidates if c['complete']]
    if not valid: raise RuntimeError('No full-population combined recipe or control is eligible')
    selected=max(valid,key=lambda c:c['mean_advantage'])
    combined_contrasts(data,spec,control)
    final={**spec,'selected_condition':selected['condition'],'candidates':candidates,'banks':list(data.env['BN']),
           'status':'AWAITING_OMRI_DECISION_RULE_AND_CLAUDE_REVIEW'}
    dump(OUT/'RECIPE_SELECTION.json',final)
    comparison=table[(table.condition==selected['condition'])&(table.recipe==chosen)].copy()
    comparison['benchmark']=np.where(comparison.cell.str.startswith('prm'),'PRMBench','ProcessBench')
    comparison.groupby(['benchmark','decision_rule'])[['within_answer_auc','prmscore','pb_official_f1','clean_flag_rate','wrong_answer_auroc']].mean().to_csv(OUT/'DECISION_COMPARISON.csv')
    lr=pd.read_csv(RUNS['level_reduction_v1']/'LR_METRICS.csv'); refs=[]
    for condition in spec['controls']:
        var=spec['level_variant'] if condition in ('LEVEL_ONLY','COMBINED') else 'original'
        subset=lr[lr.variant==var].copy(); subset['condition']=condition; refs.append(subset)
    pd.concat(refs,ignore_index=True).to_csv(OUT/'LR_REFERENCE.csv',index=False)
    dump(OUT/'RUN_STATUS.json',{'status':'AWAITING_OMRI_DECISION_RULE','selection':final,'review':'CLAUDE_PENDING','no_external_evaluation':True})

def combined_contrasts(data,spec,control):
    from spectral_utils.monday_fusion_secondary import sufficient,paired_intervals
    out=OUT/'contrasts'; out.mkdir(exist_ok=True); recipe=spec['recipe']; aggregate={}
    for ce in sorted(set(data.pop.cells)):
        prm=ce.startswith('prm'); universe=np.unique(data.pop.groups[data.pop.prm if prm else ~data.pop.prm]).astype(str)
        endpoints=['within_answer_auc','prmscore','clean_flag_rate'] if prm else ['pb_first_error_accuracy','pb_official_f1','clean_flag_rate']
        for bk in data.env['BN']:
            stem=ce+'__'+bk; files={c:np.load(OUT/'chunks'/(stem+'__'+c+'.stats.npz')) for c in spec['controls']}
            groups=files['BASE']['source_groups']; arrays={}; pairs=[]; metadata=[]; missing=[]
            for condition,z in files.items():
                if not np.array_equal(z['source_groups'],groups): raise RuntimeError('Combined source groups changed')
                for r in [recipe,control]:
                    for rule in ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80','TOP_1','TOP_2','TOP_3']:
                        for kind in endpoints:
                            if kind in ['within_answer_auc','pb_first_error_accuracy'] and rule!='ANSWER_Z_Q80': continue
                            key=r+'__CONTENT__'+rule
                            try: arrays[(condition+'::'+key,kind)]=sufficient(z,key,kind)
                            except KeyError: pass
            for condition in spec['controls']:
                for other,r,comparison in [(condition,control,'matched')]+([ (c,recipe,'component_interaction') for c in ['BASE','LEVEL_ONLY','INPUT_ONLY']] if condition=='COMBINED' else []):
                    for rule in ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80','TOP_1','TOP_2','TOP_3']:
                        for kind in endpoints:
                            if kind in ['within_answer_auc','pb_first_error_accuracy'] and rule!='ANSWER_Z_Q80': continue
                            a=condition+'::'+recipe+'__CONTENT__'+rule; b=other+'::'+r+'__CONTENT__'+rule
                            identity={'bank':bk,'cell':ce,'condition':condition,'control_condition':other,'recipe':recipe,'control':r,
                                'decision_rule':rule,'endpoint':kind,'comparison':comparison,'family':'combined_descriptive'}
                            if (a,kind) not in arrays or (b,kind) not in arrays:
                                missing.append(dict(identity,status='FAILED_COVERAGE')); continue
                            pairs.append((a,b,kind)); metadata.append(identity)
                            if prm and comparison=='matched' and kind=='within_answer_auc':
                                aggregate.setdefault(condition,[]).append((groups.copy(),arrays[(a,kind)]-np.column_stack([arrays[(b,kind)][:,0],np.zeros(len(groups))])))
            if (out/(stem+'.csv.gz')).exists(): continue
            results=paired_intervals(universe,groups,arrays,pairs) if pairs else []
            pd.DataFrame([dict(a,**b) for a,b in zip(metadata,results)]+missing).to_csv(out/(stem+'.csv.gz'),index=False)
            print('Combined paired intervals '+stem,flush=True)
    allrows=pd.concat([pd.read_csv(p) for p in out.glob('*.csv.gz')],ignore_index=True)
    universe=np.unique(data.pop.groups[data.pop.prm]).astype(str); arrays={}; pairs=[]; identities=[]
    for condition,entries in aggregate.items():
        if len(entries)!=8: continue
        if any(not np.array_equal(g,entries[0][0]) for g,a in entries): raise RuntimeError('Bank source alignment')
        value=np.mean([a for g,a in entries],axis=0)
        arrays[(condition,'within_answer_auc')]=value
        arrays[('zero','within_answer_auc')]=np.column_stack([np.zeros(len(value)),value[:,1]])
        pairs.append((condition,'zero','within_answer_auc')); identities.append({'bank':'EQUAL_MEAN_8_BANKS','cell':'prmbench_qwen3_8b',
            'condition':condition,'recipe':recipe,'control':control,'endpoint':'within_answer_auc','comparison':'matched','family':'combined_descriptive'})
    if pairs:
        result=paired_intervals(universe,entries[0][0],arrays,pairs)
        allrows=pd.concat([allrows,pd.DataFrame([dict(a,**b) for a,b in zip(identities,result)])],ignore_index=True)
    allrows.to_csv(OUT/'CONTRASTS.csv',index=False)

def freeze_decision(data,rule,authorization):
    import ds_group_weights as GW
    if rule not in ('ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80') or not authorization:
        raise ValueError('An explicit Omri decision and its conversation text are required')
    if (OUT/'LOCKED_RECIPE.json').exists(): raise RuntimeError('Recipe already frozen; do not overwrite')
    selected=json.loads((OUT/'RECIPE_SELECTION.json').read_text(encoding='utf8')); recipe=selected['recipe']; condition=selected['selected_condition']
    pool=raw_pool(data); entropy=entropy_feature(data)
    arrays={'offsets':data.pop.off}
    for bk in selected['banks']:
        score=np.full(data.pop.total,np.nan); flags=np.zeros(data.pop.total,bool); written=np.zeros(data.pop.total,int)
        for ce in sorted(set(data.pop.cells)):
            stem=ce+'__'+bk+'__'+condition
            if not completed_chunk(OUT/'chunks',stem): raise RuntimeError('Missing combined checkpoint '+stem)
            fit=json.loads((OUT/'chunks'/(stem+'.fit.json')).read_text(encoding='utf8'))
            state=restore_content(data,bk,ce,fit['level_variant']); bank=state['bank']; rows=bank['rows']
            if recipe in state['recipes']:
                value=GW.weighted_group_score(state['recipes'][recipe]['Z'],np.asarray(fit['recipes'][recipe]['weights']),signed=recipe.endswith('_SML'))
            else: value=state['scores'][recipe]
            raw=raw_variant(data,bank,bk,pool); rawcols=raw if recipe=='ALL_EQ' else raw[:,state['surv']]
            decisions,detectors,_=rules(value,bank,rawcols,entropy[bank['answers']],data.pop.uids[bank['answers']])
            actual,_=evaluate(data.pop,bank,value,decisions[rule],detectors[rule])
            saved=pd.read_csv(OUT/'chunks'/(stem+'.metrics.csv')); expected=saved[(saved.recipe==recipe)&(saved.decision_rule==rule)].iloc[0]
            for field in ['within_answer_auc','prmscore','pb_first_error_accuracy','pb_official_f1','clean_flag_rate','wrong_answer_auroc']:
                if actual[field] is not None and abs(actual[field]-expected[field])>1e-12: raise RuntimeError('Final recipe replay mismatch '+stem+'/'+field)
            score[rows]=value; flags[rows]=decisions[rule]; written[rows]+=1
        if not np.isfinite(score).all() or not (written==1).all(): raise RuntimeError('Final coverage/write-once failure '+bk)
        arrays[bk+'__score']=score; arrays[bk+'__flags']=flags
    np.savez_compressed(OUT/'STEP_SCORES.npz',**arrays)
    lock={**selected,'status':'FROZEN','decision_rule':rule,'decision_chosen_by':'Omri','authorization':authorization,
          'step_scores':(OUT/'STEP_SCORES.npz').relative_to(ROOT).as_posix(),'step_scores_sha256':sha(OUT/'STEP_SCORES.npz'),
          'independent_review':'PENDING','external_results_may_be_opened':False}
    dump(OUT/'LOCKED_RECIPE.json',lock)
