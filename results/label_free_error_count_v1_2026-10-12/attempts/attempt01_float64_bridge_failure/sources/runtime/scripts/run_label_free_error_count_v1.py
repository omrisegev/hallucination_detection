"""Preregistered CPU count study. Fitting never loads evaluation labels.

Use the main checkout's Python 3.11 virtual environment. All outputs belong to
the new package; the advisor package and its runtime helpers are read-only.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import ast
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FROZEN = ROOT / 'results/advisor_completion_2026-10-11'
OUT = ROOT / 'results/label_free_error_count_v1_2026-10-12'
SEED = 20261012
ARMS = ['LOCAL_MOM_MEAN', 'A_MOM_MEAN', 'A_MOM_MEDIAN',
        'B_TRIPLE_MEAN', 'B_TRIPLE_MEDIAN', 'D_ZERO_MEAN', 'D_ZERO_MEDIAN']
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
ARCHIVE = FROZEN / 'sources/research/results/monday_fusion_analysis_v1/code/scripts/experiments'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(x):
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    return x


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def frozen_inventory():
    manifest = json.loads((FROZEN / 'PACKAGE_MANIFEST.json').read_text())
    entries = [{'path': 'PACKAGE_MANIFEST.json', 'sha256': sha(FROZEN / 'PACKAGE_MANIFEST.json')}]
    errors = []
    for entry in manifest['files']:
        path = FROZEN / entry['path']
        actual = sha(path) if path.is_file() else None
        entries.append({'path': entry['path'], 'sha256': actual, 'size': path.stat().st_size if path.exists() else None})
        if actual != entry['sha256']: errors.append(entry['path'])
    runtime = []
    for entry in manifest['runtime_files']:
        path = ROOT / entry['repo_relative_path']; actual = sha(path) if path.is_file() else None
        runtime.append({'path': entry['repo_relative_path'], 'sha256': actual})
        if actual != entry['sha256']: errors.append(entry['repo_relative_path'])
    if errors: raise RuntimeError('Frozen manifest mismatch: ' + repr(errors))
    return {'files': entries, 'runtime': runtime, 'manifest_sha256': entries[0]['sha256']}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'version': 1, 'status': 'FROZEN_BEFORE_NEW_NUMERICAL_RUNS',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'development_only': True, 'no_external_untouched_test': True,
        'hypothesis': 'Answer normalization and relative quotas can discard between-answer count evidence; this is not assumed to be the only limitation.',
        'banks': ['N16', 'N56'], 'primary_bank': 'N16', 'N16_status': 'historical reference, not automatic selection',
        'folds': 'FOLDS_V2.json prmbench.outer; entire canonical source groups',
        'training_population': 'all training answers, without consulting classifications or labels',
        'ranking': 'one canonical donor IU-PCR fit per bank/fold; LSML secondary; immutable stable descending ranks',
        'arms': ARMS, 'candidate_count_for_multiplicity': 7,
        'A': {'mu': 'active donor raw mean', 'r': 'pooled within-answer population SD on active observations',
              'coordinates': 'active*(raw-mu)/r', 'weights': 'unchanged canonical LSML within-group coefficients on bridge coordinates',
              'digit_intercept': 'mask-dependent', 'group_scaling': 'absolute donor mean/SD; local canonical answer-z',
              'quantile': .8, 'quantile_method': 'linear', 'vote_ties': 'strict greater than threshold',
              'reconstruction_abs_tolerance': 1e-9, 'scale_zero_tolerance': 1e-12},
        'MoM': {'new_clipping': False, 'legacy_clipping': 'faithful historical control only, reported',
                'rank_one_maxiter': 2000, 'rank_one_tol': 1e-13, 'orientation': 'positive loading sum; majority-better-than-chance assumption'},
        'B': {'input': 'absolute representation fixed in advance', 'triples': 'all learned-group triples',
              'source_bootstrap_draws': 200, 'min_valid': 180, 'covariance_lower_quantile': .025,
              'eligibility': 'full fit strict probability interior, >=180 valid bootstrap fits, every pair covariance .025 quantile > 0',
              'selection': 'min IQR(prevalence)/sqrt(pi*(1-pi)); exact ties by sorted feature member indices',
              'orientation': 'all three positively covarying voters better than chance; not independence proof',
              'posterior': 'selected triple only', 'bootstrap_conditions_on_group_representation': True,
              'sensitivity': ['same triple equal-answer moments', 'same triple equal-source/equal-answer moments']},
        'C': {'mean': 'floor(sum(p)+.5)', 'median': 'smallest k with exact PB CDF >= .5',
              'PB_assumption': 'conditional independent step labels and fixed parameters; mean sum needs no step independence'},
        'D': {'fixed_emissions': 'valid B selected triple', 'parameters': ['rho structural-clean mixture probability', 'theta active prevalence'],
              'starts': [[r,t] for r in [.1,.5,.9] for t in [.25,.75]], 'method': 'L-BFGS-B',
              'bounds': [[0,1],[0,1]], 'maxiter':500, 'maxfun':2000, 'ftol':1e-12, 'gtol':1e-7,
              'near_best_average_ll':1e-8, 'nonident_parameter_difference':.05, 'hessian_condition_limit':1e8,
              'boundary_diagnostic': 'rho=0, refit theta; no new candidate',
              'count_pmf': 'q_clean*delta0+(1-q_clean)*PB(active posterior)',
              'activation': 'preregistered because frozen incumbent already fails k=0',
              'dependency_failure': 'no valid B => FAILED_DEPENDENCY', 'near_clean_boundary_epsilon':1e-6},
        'controls': ['Q80','R2_COUNT','ENTROPY_GATE_Q80','LEGACY_MOM'],
        'control_training': 'all fitted stats, DS survivor selection, thresholds and entropy cutoff training-only; historical transductive replay separate',
        'relative_vote_seed':20260929, 'channel_vote_seed':20260928, 'DS_seed':20260927,
        'seed': SEED, 'bootstrap_draws':20000, 'bootstrap_unit':'canonical whole source group',
        'success': {'coverage':'all five folds, N16', 'MAE':'upper one-sided paired difference <0 vs Q80 and LEGACY_MOM',
                    'PRMScore':'lower one-sided paired difference >0 vs Q80', 'recall':'lower one-sided paired difference >-.01 vs Q80',
                    'alpha':.05/7, 'failed_arms_keep_multiplicity':True},
        'metrics': {'Overall':'pooled macro F1 on noncontrols', 'MAE':'equal-answer noncontrol mean',
                    'MS':'valid-step F1', 'Sensitivity':['missing_condition','deception','multi_solutions'],
                    'partial_Spearman':'residual average-tie ranks on intercept and rank(n_steps)',
                    'length_bins':[5,10,20], 'controls_separate':True,
                    'true_count':'sum valid-step annotation membership; no out-of-range repairs',
                    'interval_limit':'conditional on saved OOF fits, not whole-learning-procedure uncertainty'},
        'retrospective_context':'family motivated by already inspected development evidence; new protocol is prospective only for these new runs',
        'no_truth_prior':True, 'no_supervised_selection':True, 'no_bank_winner_selection':True,
        'no_s_hat_PR_correlation_optimization':True, 'CPU_only':True, 'publication':'local only'
    }
    p = OUT/'PROTOCOL.json'
    if p.exists():
        old=json.loads(p.read_text());protocol['created_utc']=old['created_utc']
        if old!=protocol: raise RuntimeError('Protocol already exists with different choices; append an amendment and use a new run')
    else: dump(p,protocol)
    inventory = frozen_inventory()
    before=OUT/'FROZEN_PACKAGE_BEFORE.json'
    if before.exists():
        if json.loads(before.read_text())!=inventory: raise RuntimeError('Frozen advisor package changed since preparation')
    else: dump(before,inventory)
    inputs={}
    for name in ['population.npz','bank_V_N16.npz','bank_V_N56.npz','bank_N16.npz','bank_N56.npz',
                 'features_thinkfix_mask1.npz','CANONICAL_GROUPS.json','FOLDS_V2.json']:
        path=FROZEN/'inputs'/name;inputs[str(path.relative_to(ROOT))]={'sha256':sha(path),'size':path.stat().st_size}
    for name in ['controls/NUISANCE_INPUTS.npz','controls/DECISION_FLAGS.npz','statistics/MOM_POSTERIOR_QUOTA.npz']:
        path=FROZEN/name;inputs[str(path.relative_to(ROOT))]={'sha256':sha(path),'size':path.stat().st_size}
    state={'protocol_sha256':sha(p),'inputs':inputs,'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'repo_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
           'python':sys.version,'executable':sys.executable,'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','matplotlib']}}
    path=OUT/'INPUT_FREEZE.json'
    if path.exists() and json.loads(path.read_text())!=state: raise RuntimeError('Input/runtime identity changed')
    dump(path,state)
    for d in ['checkpoints','predictions','logs','review','report']: (OUT/d).mkdir(exist_ok=True)
    amendments=OUT/'AMENDMENTS.json'
    if not amendments.exists(): dump(amendments,[])
    print('PREPARED',len(inventory['files']),'frozen files verified',flush=True)


def archived_estimators():
    sys.path.insert(0,str(ARCHIVE))
    import er_stage_a as sa
    import er_stage_b as sb
    sa.CVF=ARCHIVE
    return sa,sb


def load_unlabeled():
    # Read only explicitly permitted fields; y and classification are not loaded.
    with np.load(FROZEN/'inputs/population.npz',allow_pickle=False) as z:
        data={k:z[k] for k in ['idx','offsets','group','fold','feature_rows']}
    groups=json.loads((FROZEN/'inputs/CANONICAL_GROUPS.json').read_text())['rows']
    gm={str(r['row_id']):str(r['canonical_group_id']) for r in groups}
    folds=json.loads((FROZEN/'inputs/FOLDS_V2.json').read_text())['prmbench']['outer']
    assert len(data['idx'])==6969 and data['offsets'][-1]==94203 and len(np.unique(data['group']))==707
    for uid,g,f in zip(data['idx'],data['group'],data['fold']):
        assert gm[str(uid)]==str(g) and folds[str(g)]==int(f)
    nu=np.load(FROZEN/'controls/NUISANCE_INPUTS.npz',allow_pickle=False)
    assert np.array_equal(nu['idx'],data['idx'])
    data['entropy']=nu['answer_entropy']
    f=np.load(FROZEN/'inputs/features_thinkfix_mask1.npz',allow_pickle=False)
    data['digit_active']=f['digit_active'][data['feature_rows']]
    return data


def cut_answers(data, answers):
    answers=np.asarray(answers)
    off=np.asarray(data['offsets'])
    if (answers.ndim!=1 or not len(answers) or answers.dtype.kind not in 'iu' or
            len(np.unique(answers))!=len(answers) or np.any(answers<0) or np.any(answers>=len(off)-1)):
        raise ValueError('Answer selection must be nonempty unique valid integer indices')
    if off.ndim!=1 or off.dtype.kind not in 'iu' or off[0]!=0 or np.any(np.diff(off)<=0):
        raise ValueError('Invalid complete-answer offsets')
    n=np.diff(data['offsets'])[answers]
    rows=np.concatenate([np.arange(data['offsets'][i],data['offsets'][i+1]) for i in answers])
    return rows,np.r_[0,np.cumsum(n)]


def count_flags(flags,offsets):
    return np.add.reduceat(np.asarray(flags,dtype=np.int64),offsets[:-1])


def rank_per_answer(score,offsets):
    blocks=[]
    for a,b in zip(offsets[:-1],offsets[1:]):
        if np.isnan(score[a:b]).all(): blocks.append(np.full(b-a,-1,np.int64))
        elif np.isfinite(score[a:b]).all(): blocks.append(a+np.argsort(-score[a:b],kind='stable'))
        else: raise ValueError('Partially invalid answer ranking')
    return np.concatenate(blocks)


def historical_baselines():
    from spectral_utils.advisor_controls import answer_z,global_z,top_flags,metrics
    from spectral_utils.advisor_lr import fit_labelfree
    from spectral_utils.advisor_statistics import posterior_from_mom
    sa,sb=archived_estimators()
    data=load_unlabeled();off=data['offsets'];z=np.load(FROZEN/'inputs/bank_N16.npz');v=np.load(FROZEN/'inputs/bank_V_N16.npz')
    saved=np.load(FROZEN/'controls/DECISION_FLAGS.npz');mom_saved=np.load(FROZEN/'statistics/MOM_POSTERIOR_QUOTA.npz')
    env={'np':np}
    source=ARCHIVE.parents[1]/'spectral_utils/monday_fusion_decision.py'
    metric_source=source.with_name('monday_fusion_metrics.py')
    for path,names in [(metric_source,{'answer_z','top_flags'}),(source,{'rules'})]:
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    cols=v['columns'].astype(str).tolist();surv=z['G_survivors'].astype(str).tolist();raw=v['raw'][:,[cols.index(x) for x in surv]]
    assert np.max(np.abs(global_z(raw).mean(1)-z['G']))<1e-9
    results={};flags_export={}
    for method in ['IU_PCR','LSML_CONT']:
        flags,_,detail=env['rules'](z['S__'+method],{'off':off},raw,data['entropy'],data['idx'])
        for arm in ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80']:
            assert np.array_equal(flags[arm],saved[method+'__'+arm]),(method,arm)
            flags_export[method+'__'+arm]=flags[arm]
        results[method]={'rules':detail,'flags_exact':True}
    fitted=fit_labelfree(v['V'],'LSML_CONT')
    assert np.max(np.abs(answer_z(fitted.predict(v['V']),off)-z['S__LSML_CONT']))<1e-9
    gs=np.column_stack([answer_z(v['V'][:,q['columns']]@np.asarray(q['weights']),off) for q in fitted.diagnostic['group_weights']])
    votes=sb.random_tie_marks(gs,off,.2,np.random.default_rng(20260929).random(gs.shape))
    estimate=sa.tensor_mom_estimate(votes,None);post=posterior_from_mom(votes,estimate)
    assert np.max(np.abs(post-mom_saved['posterior']))<1e-12
    counts=np.floor(np.add.reduceat(post,off[:-1])+.5).astype(int)
    assert np.array_equal(counts,mom_saved['count'])
    # Historical labels are allowed only to reproduce already-known controls.
    pop=np.load(FROZEN/'inputs/population.npz',allow_pickle=False)
    pool=np.repeat(pop['classification']!='correct',np.diff(off))
    for method in ['IU_PCR','LSML_CONT']:
        flags=top_flags(z['S__'+method],off,counts);assert np.array_equal(flags,mom_saved[method])
        results[method]['Q80']=metrics(pop['y'][pool],flags_export[method+'__ANSWER_Z_Q80'][pool])
        results[method]['MoM']=metrics(pop['y'][pool],flags[pool]);flags_export[method+'__LEGACY_MOM']=flags
    results['MoM_estimate']=estimate;results['clean_controls_flagged']=int((counts[pop['classification']=='correct']>0).sum())
    results['status']='PASS';results['historical_transductive']=True
    np.savez_compressed(OUT/'historical_baselines.npz',idx=data['idx'],offsets=off,count=counts,posterior=post,**flags_export)
    dump(OUT/'HISTORICAL_BASELINE_REPLAY.json',results)
    print('HISTORICAL_BASELINES_PASS',results['clean_controls_flagged'],flush=True)


def freeze_code():
    paths=['scripts/run_label_free_error_count_v1.py','spectral_utils/error_count.py','spectral_utils/error_count_inputs.py',
           'spectral_utils/error_count_evaluation.py','scripts/evaluate_label_free_error_count_v1.py']
    result={p:sha(ROOT/p) for p in paths}
    path=OUT/'CODE_FREEZE.json'
    if path.exists() and json.loads(path.read_text())!=result: raise RuntimeError('Code changed after freeze; use an explicit new run/amendment')
    dump(path,result)
    for p in paths:
        dest=OUT/'sources/runtime'/p
        if dest.exists() and sha(dest)!=result[p]: raise RuntimeError('Archived runtime identity changed: '+p)
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)
    return result


def validate_identity():
    freeze=json.loads((OUT/'INPUT_FREEZE.json').read_text())
    current={k:importlib.metadata.version(k) for k in freeze['versions']}
    if (freeze['python']!=sys.version or freeze['executable']!=sys.executable or current!=freeze['versions']):
        raise RuntimeError('Runtime/executable/dependency identity changed')
    assert sha(OUT/'PROTOCOL.json')==freeze['protocol_sha256']
    for path,entry in freeze['inputs'].items():
        if sha(ROOT/path)!=entry['sha256']: raise RuntimeError('Input changed: '+path)
    return freeze_code()


def predict():
    from spectral_utils.advisor_controls import answer_z
    from spectral_utils.advisor_lr import fit_labelfree,as_dict
    from spectral_utils.advisor_statistics import posterior_from_mom
    from spectral_utils.error_count_inputs import fit_vote_representation,apply_vote_representation
    from spectral_utils.error_count import fit_mom,posterior,count_distribution,fit_stable_triple,fit_fixed_triple,fit_zero_mixture,predict_zero_mixture
    code=validate_identity();sa,sb=archived_estimators();d=load_unlabeled();off=d['offsets'];n=np.diff(off)
    for bank in ['N16','N56']:
        z=np.load(FROZEN/f'inputs/bank_V_{bank}.npz',allow_pickle=False);raw=z['raw'];V=z['V'];cols=z['columns'].astype(str).tolist()
        active=np.ones(raw.shape,bool)
        for j,name in enumerate(cols):
            if name in DIG: active[:,j]=d['digit_active'][:,DIG.index(name)]
        combined={'idx':d['idx'],'offsets':off,'fold':d['fold'],'group':d['group']}
        all_names=ARMS+['Q80','R2_COUNT','ENTROPY_GATE_Q80','LEGACY_MOM','D_PRIOR_MEAN','D_PRIOR_MEDIAN']
        for name in all_names:
            for method in ['IU_PCR','LSML_CONT']: combined[f'count__{method}__{name}']=np.full(len(n),-1,np.int64)
            combined[f'count__{name}']=np.full(len(n),-1,np.int64)
            combined[f'posterior__{name}']=np.full(off[-1],np.nan)
        for method in ['IU_PCR','LSML_CONT']: combined[method+'_score']=np.full(off[-1],np.nan)
        for fold in range(5):
            started=time.time();test=np.flatnonzero(d['fold']==fold);train=np.flatnonzero(d['fold']!=fold)
            assert not set(d['group'][test])&set(d['group'][train])
            tr,toff=cut_answers(d,train);te,eoff=cut_answers(d,test)
            folder=OUT/'checkpoints'/f'{bank}_fold{fold}';folder.mkdir(exist_ok=True)
            ident={'bank':bank,'fold':fold,'input_freeze':sha(OUT/'INPUT_FREEZE.json'),'code':code,'protocol':sha(OUT/'PROTOCOL.json'),
                   'amendments':sha(OUT/'AMENDMENTS.json')}
            ready=folder/'COMPLETE.json'
            if ready.exists():
                receipt=json.loads(ready.read_text())
                if receipt['identity']!=ident or sha(folder/'predictions.npz')!=receipt['predictions_sha256'] or sha(folder/'models.json')!=receipt['models_sha256']:
                    raise RuntimeError('Checkpoint identity mismatch: '+str(folder))
                print('RESUME',bank,fold,flush=True)
                local=np.load(folder/'predictions.npz',allow_pickle=False)
            else:
                print('FIT',bank,fold,'train',len(train),'test',len(test),flush=True)
                dump(folder/'TRAINING_IDENTITY.json',{'train_ids':d['idx'][train],'test_ids':d['idx'][test],
                    'train_groups':np.unique(d['group'][train]),'test_groups':np.unique(d['group'][test]),'identity':ident})
                local_out={};models={'identity':ident,'status':{},'timing':{}};fusion={}
                for method in ['IU_PCR','LSML_CONT']:
                    t=time.time()
                    try:
                        fusion[method]=fit_labelfree(V[tr],method);models[method]=as_dict(fusion[method]);models['timing'][method+'_fit_s']=time.time()-t
                        t=time.time();raw_score=np.asarray(fusion[method].predict(V[te]),float)
                        if raw_score.shape!=(len(te),) or not np.isfinite(raw_score).all(): raise RuntimeError('Invalid canonical holdout prediction')
                        local_out[method+'_score']=answer_z(raw_score,eoff);models['timing'][method+'_prediction_s']=time.time()-t
                    except Exception as exc:
                        fusion.pop(method,None)
                        models[method]={'status':'FAILED','reason':repr(exc)};local_out[method+'_score']=np.full(len(te),np.nan)
                if len(fusion)!=2:
                    for name in all_names:
                        local_out['count__'+name]=np.full(len(test),-1,np.int64);local_out['posterior__'+name]=np.full(len(te),np.nan)
                        for method in ['IU_PCR','LSML_CONT']: local_out[f'count__{method}__{name}']=np.full(len(test),-1,np.int64)
                        models['status'][name]={'status':'FAILED','reason':'FAILED_CANONICAL_FUSER_DEPENDENCY'}
                    local_out['idx']=d['idx'][test];local_out['offsets']=eoff;models['timing']['fold_total_s']=time.time()-started
                    np.savez_compressed(folder/'predictions.npz',**local_out);dump(folder/'models.json',models)
                    dump(ready,{'identity':ident,'predictions_sha256':sha(folder/'predictions.npz'),'models_sha256':sha(folder/'models.json')})
                    for key in combined:
                        if key.startswith('count__'): combined[key][test]=local_out[key]
                        elif key.startswith('posterior__') or key.endswith('_score'): combined[key][te]=local_out[key]
                    print('FOLD_FAILED_FUSER',bank,fold,flush=True)
                    continue
                groups=fusion['LSML_CONT'].diagnostic['group_weights']
                # All arms have a defined failure marker rather than fabricated zero quotas.
                for name in all_names:
                    for method in ['IU_PCR','LSML_CONT']: local_out[f'count__{method}__{name}']=np.full(len(test),-1,np.int64)
                    local_out[f'count__{name}']=np.full(len(test),-1,np.int64)
                    local_out[f'posterior__{name}']=np.full(len(te),np.nan)

                def store_counts(name,c,p=None):
                    c=np.asarray(c,np.int64)
                    assert c.shape==(len(test),) and np.all((c>=0)&(c<=np.diff(eoff)))
                    local_out['count__'+name]=c
                    for method in ['IU_PCR','LSML_CONT']: local_out[f'count__{method}__{name}']=c.copy()
                    if p is not None: local_out['posterior__'+name]=np.asarray(p,float)
                    models['status'][name]={'status':'PASS','n_answers':len(test)}

                def failure(names,exc):
                    for name in names: models['status'][name]={'status':'FAILED','reason':str(exc)}
                    print('ARM_FAILED',bank,fold,','.join(names),str(exc)[:250],flush=True)

                # Q80 and entropy are each method-specific; candidate counts remain identical across rankings.
                for method in ['IU_PCR','LSML_CONT']:
                    ts=answer_z(fusion[method].predict(V[tr]),toff);es=local_out[method+'_score'];tau=float(np.quantile(ts,.8))
                    qc=count_flags(es>=tau,eoff);local_out[f'count__{method}__Q80']=qc
                    h=d['entropy'][train];hm=float(h.mean());hs=float(h.std());hs=hs if hs>1e-12 else 1e-12
                    offset=ts+np.repeat((h-hm)/hs,np.diff(toff));tauoff=float(np.quantile(offset,.8))
                    zero=count_flags(offset>=tauoff,toff)==0;close=int(np.ceil(zero.mean()*len(train)))
                    order=np.lexsort((d['idx'][train].astype(str),h))
                    if close==0: gate=np.ones(len(test),bool);key=None
                    elif close==len(train): gate=np.zeros(len(test),bool);key='ALL_CLOSED'
                    else:
                        k=order[close-1];key=[float(h[k]),str(d['idx'][train][k])]
                        gate=np.array([(float(e),str(u))>tuple(key) for e,u in zip(d['entropy'][test],d['idx'][test])])
                    local_out[f'count__{method}__ENTROPY_GATE_Q80']=qc*gate
                    models[method+'_controls']={'q80':tau,'entropy_mean':hm,'entropy_sd':hs,'offset_q80':tauoff,'close':close,'cutoff_key':key}
                for name in ['Q80','ENTROPY_GATE_Q80']:
                    local_out['count__'+name]=local_out[f'count__IU_PCR__{name}'].copy();models['status'][name]={'status':'PASS'}
                try:
                    channel_votes=sb.random_tie_marks(V[tr],toff,.2,np.random.default_rng(20260928).random(V[tr].shape))
                    ds=sa.em_estimate(channel_votes,'ds',seed=20260927);surv=np.flatnonzero(np.asarray(ds['pi'])>.5)
                    if not len(surv): raise RuntimeError('No donor DS survivors')
                    r=raw[tr][:,surv];mu=r.mean(0);sd=np.maximum(r.std(0),1e-12);g=((r-mu)/sd).mean(1);tau=float(np.quantile(g,.8))
                    eg=((raw[te][:,surv]-mu)/sd).mean(1);store_counts('R2_COUNT',count_flags(eg>=tau,eoff))
                    models['R2_COUNT']={'DS':ds,'columns':surv.tolist(),'survivors':[cols[i] for i in surv],'mean':mu,'sd':sd,'threshold':tau}
                except Exception as exc: failure(['R2_COUNT'],exc)
                train_g=np.column_stack([answer_z(V[tr][:,q['columns']]@np.asarray(q['weights']),toff) for q in groups])
                test_g=np.column_stack([answer_z(V[te][:,q['columns']]@np.asarray(q['weights']),eoff) for q in groups])
                try:
                    mt=sb.random_tie_marks(train_g,toff,.2,np.random.default_rng(20260929).random(train_g.shape))
                    me=sb.random_tie_marks(test_g,eoff,.2,np.random.default_rng(20260929).random(test_g.shape))
                    legacy=sa.tensor_mom_estimate(mt,None);p=posterior_from_mom(me,legacy);store_counts('LEGACY_MOM',np.floor(np.add.reduceat(p,eoff[:-1])+.5),p)
                    models['LEGACY_MOM']=legacy
                except Exception as exc: failure(['LEGACY_MOM'],exc)
                try:
                    t=time.time();rep=fit_vote_representation(raw[tr],V[tr],active[tr],toff,groups,cross_weights=fusion['LSML_CONT'].diagnostic['cross_weights'])
                    models['timing']['A_representation_fit_s']=time.time()-t
                    models['representation']=rep
                    for kind,names in [('local',['LOCAL_MOM_MEAN']),('absolute',['A_MOM_MEAN','A_MOM_MEDIAN'])]:
                        _,mt,td=apply_vote_representation(rep,raw[tr],V[tr],active[tr],toff,kind=kind)
                        t=time.time();_,me,ed=apply_vote_representation(rep,raw[te],V[te],active[te],eoff,kind=kind)
                        models['timing'][kind+'_vote_prediction_s']=time.time()-t
                        models[kind+'_votes']={'train':td,'test':ed}
                        np.savez_compressed(folder/(kind+'_votes.npz'),train=mt,test=me,train_rows=tr,test_rows=te)
                        try:
                            t=time.time();mom=fit_mom(mt,legacy_clip=False);models[kind+'_mom']=mom;models['timing'][kind+'_mom_fit_s']=time.time()-t
                            t=time.time();p=posterior(me,mom);dist=count_distribution(p,eoff);models['timing'][kind+'_count_prediction_s']=time.time()-t
                            store_counts(names[0],dist['mean_half_up'],p)
                            if len(names)>1: store_counts(names[1],dist['median_lower'],p)
                            local_out[kind+'_pmf']=np.asarray(dist['pmf']);local_out[kind+'_pmf_offsets']=np.asarray(dist['pmf_offsets'])
                        except Exception as exc: failure(names,exc)
                        if kind=='absolute': abs_train,abs_test=mt,me
                    t=time.time();step_groups=np.repeat(d['group'][train],np.diff(toff))
                    triple=fit_stable_triple(abs_train,step_groups,[q['columns'] for q in groups],seed=SEED+fold,draws=200)
                    models['B_TRIPLE']=triple;models['timing']['B_triple_fit_s']=time.time()-t
                    try:
                        t=time.time();p=posterior(abs_test,triple);dist=count_distribution(p,eoff);models['timing']['B_count_prediction_s']=time.time()-t
                        store_counts('B_TRIPLE_MEAN',dist['mean_half_up'],p);store_counts('B_TRIPLE_MEDIAN',dist['median_lower'],p)
                        local_out['B_pmf']=np.asarray(dist['pmf']);local_out['B_pmf_offsets']=np.asarray(dist['pmf_offsets'])
                        na=np.diff(toff);wanswer=np.repeat(1/na,na);ug,inv=np.unique(d['group'][train],return_inverse=True);pergroup=np.bincount(inv)
                        wsource=np.repeat(1/(na*pergroup[inv]),na)
                        models['B_weighting_sensitivity']={}
                        for wname,w in [('equal_answer',wanswer),('equal_source_answer',wsource)]:
                            sm=fit_fixed_triple(abs_train,triple['columns'],weights=w);models['B_weighting_sensitivity'][wname]=sm
                            if sm.get('status')=='PASS': local_out[wname+'_posterior']=posterior(abs_test,sm)
                    except Exception as exc: failure(['B_TRIPLE_MEAN','B_TRIPLE_MEDIAN'],exc)
                    if triple.get('status')=='PASS':
                        t=time.time();zero=fit_zero_mixture(abs_train,toff,triple);models['D_ZERO']=zero;models['timing']['D_zero_fit_s']=time.time()-t
                        try:
                            t=time.time();zd=predict_zero_mixture(abs_test,eoff,triple,zero);models['timing']['D_count_prediction_s']=time.time()-t
                            store_counts('D_ZERO_MEAN',zd['mean_half_up'],zd['marginal_p']);store_counts('D_ZERO_MEDIAN',zd['median_lower'],zd['marginal_p'])
                            local_out['D_pmf']=np.asarray(zd['pmf']);local_out['D_pmf_offsets']=np.asarray(zd['pmf_offsets'])
                            local_out['D_structural_clean_p']=np.asarray(zd['structural_clean_p']);local_out['D_active_p']=np.asarray(zd['active_p'])
                            if not zero.get('promotable',True):
                                for arm in ['D_ZERO_MEAN','D_ZERO_MEDIAN']: models['status'][arm]['promotable']=False
                        except Exception as exc: failure(['D_ZERO_MEAN','D_ZERO_MEDIAN'],exc)
                        boundary=zero.get('boundary_model')
                        if boundary and boundary.get('status')=='PASS':
                            bd=predict_zero_mixture(abs_test,eoff,triple,boundary)
                            store_counts('D_PRIOR_MEAN',bd['mean_half_up'],bd['marginal_p']);store_counts('D_PRIOR_MEDIAN',bd['median_lower'],bd['marginal_p'])
                    else: failure(['D_ZERO_MEAN','D_ZERO_MEDIAN'],'FAILED_DEPENDENCY: '+str(triple.get('failure_reason')))
                except Exception as exc:
                    failure([x for x in ARMS if x not in models['status']],exc)
                    models['representation_exception']=traceback.format_exc()
                models['timing']['fold_total_s']=time.time()-started
                local_out['idx']=d['idx'][test];local_out['offsets']=eoff
                np.savez_compressed(folder/'predictions.npz',**local_out);dump(folder/'models.json',models)
                dump(ready,{'identity':ident,'predictions_sha256':sha(folder/'predictions.npz'),'models_sha256':sha(folder/'models.json'),
                            'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
                local=np.load(folder/'predictions.npz',allow_pickle=False)
                print('FOLD_DONE',bank,fold,round(time.time()-started,2),models['status'],flush=True)
            if not np.array_equal(local['idx'],d['idx'][test]) or not np.array_equal(local['offsets'],eoff):
                raise RuntimeError('Checkpoint semantic answer identity/offset mismatch: '+str(folder))
            for key in combined:
                if key.startswith('count__'): combined[key][test]=local[key]
                elif key.startswith('posterior__') or key.endswith('_score'): combined[key][te]=local[key]
        for method in ['IU_PCR','LSML_CONT']:
            combined['rank_'+method]=rank_per_answer(combined[method+'_score'],off)
        np.savez_compressed(OUT/'predictions'/f'{bank}.npz',**combined)
        print('BANK_DONE',bank,flush=True)


def seal():
    validate_identity()
    files={}
    for p in sorted((OUT/'predictions').glob('*.npz')): files[str(p.relative_to(OUT))]={'sha256':sha(p),'size':p.stat().st_size}
    for p in sorted((OUT/'checkpoints').glob('*/*.json')): files[str(p.relative_to(OUT))]={'sha256':sha(p),'size':p.stat().st_size}
    for p in sorted((OUT/'checkpoints').glob('*/*.npz')): files[str(p.relative_to(OUT))]={'sha256':sha(p),'size':p.stat().st_size}
    for name in ['AMENDMENTS.json','INPUT_FREEZE.json']:
        files[name]={'sha256':sha(OUT/name),'size':(OUT/name).stat().st_size}
    if not all('predictions/'+b+'.npz' in files for b in ['N16','N56']): raise RuntimeError('Both banks required before seal')
    state={'status':'FROZEN_BEFORE_EVALUATION','files':files,'code_sha256':json.loads((OUT/'CODE_FREEZE.json').read_text()),
           'protocol_sha256':sha(OUT/'PROTOCOL.json'),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    p=OUT/'PREDICTION_SEAL.json'
    if p.exists():
        old=json.loads(p.read_text());state['created_utc']=old['created_utc']
        if old!=state: raise RuntimeError('Prediction seal changed')
    dump(p,state);print('PREDICTIONS_SEALED',flush=True)


def verify():
    validate_identity()
    after=frozen_inventory();before=json.loads((OUT/'FROZEN_PACKAGE_BEFORE.json').read_text())
    if after!=before: raise RuntimeError('Frozen advisor package was modified')
    dump(OUT/'FROZEN_PACKAGE_AFTER.json',after)
    s=json.loads((OUT/'PREDICTION_SEAL.json').read_text())
    for path,entry in s['files'].items():
        if sha(OUT/path)!=entry['sha256']: raise RuntimeError('Sealed output changed: '+path)
    dump(OUT/'VERIFY.json',{'status':'PASS','frozen_files_unchanged':len(after['files']),'sealed_files_verified':len(s['files'])})
    print('VERIFY_PASS',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','baselines','predict','seal','verify','all'],default='all');args=ap.parse_args()
    if args.stage in ['prepare','all']:prepare()
    if args.stage in ['baselines','all']:historical_baselines()
    if args.stage in ['predict','all']:predict()
    if args.stage in ['seal','all']:seal()
    if args.stage in ['verify','all']:verify()


if __name__=='__main__': main()
