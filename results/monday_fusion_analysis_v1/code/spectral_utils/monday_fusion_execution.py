"""Resumable cell-at-a-time execution with explicit failure rows."""
from pathlib import Path
import json
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
from spectral_utils.monday_fusion_protocol import ROOT,WORK,CODE,RUNS,BRANCH,sha,dump
from spectral_utils.monday_fusion_core import variant_bank,fit_content
from spectral_utils.monday_fusion_metrics import evaluate,answer_z,top_flags

RECIPES=['ALL_EQ','DS_FILTER_EQ']+[w+'_'+b for w in ['MEAN','HEM'] for b in ['EQ','DS','SML','MOM']]

def storage_guard():
    free=shutil.disk_usage(ROOT).free
    if free<100_000_000: raise RuntimeError(f'Only {free/1e6:.1f} MB free; stopped before next artifact write')

def completed_chunk(directory,stem):
    path=directory/(stem+'.complete.json')
    if not path.exists(): return False
    record=json.loads(path.read_text(encoding='utf8'))
    for suffix,digest in record.get('artifacts',{}).items():
        if sha(directory/(stem+suffix))!=digest: raise RuntimeError('Completed checkpoint changed: '+stem+suffix)
    return True

def native(v):
    if isinstance(v,np.ndarray): return native(v.tolist())
    if isinstance(v,np.generic): return native(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    if isinstance(v,dict): return {str(k):native(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [native(x) for x in v]
    return v

def check_frozen():
    for exp,out in RUNS.items():
        rel=(out/'PROTOCOL.json').relative_to(ROOT).as_posix()
        committed=subprocess.check_output(['git','-C',str(ROOT),'show',BRANCH+':'+rel])
        if committed!=(out/'PROTOCOL.json').read_bytes(): raise RuntimeError('Protocol is not committed exactly: '+exp)
    gate=json.loads((WORK/'REPLAY_GATE.json').read_text(encoding='utf8'))
    if gate['status']!='PASS': raise RuntimeError('Complete replay + zero-removal gate is required')

def code_seal(out,stage):
    paths=['scripts/experiments/monday_fusion_run.py']+[p.relative_to(CODE).as_posix() for p in (CODE/'spectral_utils').glob('monday_fusion*.py')]
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',BRANCH],text=True).strip()
    frozen_files=[out/'PROTOCOL.json']+([out/'IMPLEMENTATION_NOTES_V1.json'] if (out/'IMPLEMENTATION_NOTES_V1.json').exists() else [])
    if (out/'REVIEW_CORRECTIONS_V2.json').exists(): frozen_files.append(out/'REVIEW_CORRECTIONS_V2.json')
    for path in frozen_files:
        expected=subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+path.relative_to(ROOT).as_posix()])
        if expected!=path.read_bytes(): raise RuntimeError('Uncommitted protocol/implementation notes: '+str(path))
    for rel in paths:
        expected=subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+rel])
        if expected!=(CODE/rel).read_bytes(): raise RuntimeError('Code is not committed before scoring: '+rel)
    seal={'commit':head,'stage':stage,'paths':{rel:sha(CODE/rel) for rel in paths},'protocol_sha256':sha(out/'PROTOCOL.json')}
    # Keep the exact pre-resume code provenance of reused immutable checkpoints.
    dump(out/('CODE_MANIFEST_'+stage+'_'+head[:12]+'.json'),seal)
    dump(out/('CODE_MANIFEST_'+stage+'.json'),seal)
    dump(out/'CODE_MANIFEST.json',seal)

def zero_removal(data):
    gate=json.loads((WORK/'REPLAY_GATE.json').read_text(encoding='utf8'))
    if gate['status'] not in ('HISTORICAL_PASS_ZERO_REMOVAL_PENDING','PASS'): raise RuntimeError('Historical replay incomplete')
    comparisons={}
    reference=np.load(data.manifest['inputs']['step462']['path'])
    for ce in sorted(set(data.pop.cells)):
        rows=data.pop.rows(data.pop.cells==ce)
        for bk in data.env['BN']:
            actual=variant_bank(data,bk,ce,'original')
            expected=data.env['cell_bank'](bk,rows,ce)
            for key,ref in zip(['V','marks','TT'],expected[:3]):
                if not np.array_equal(actual[key],ref[rows]): raise RuntimeError(f'Zero removal differs: {ce}/{bk}/{key}')
            comparisons[ce+'__'+bk]={'exact':True,'n_steps':len(rows),'columns':actual['V'].shape[1]}
            if bk in ['B16','B23','B35','B54']:
                state=fit_content(data,actual,recipes=['MEAN_DS'])
                for recipe,ref in [('DS_FILTER_EQ','BASE'),('MEAN_DS','GRP')]:
                    if recipe not in state['scores']: raise RuntimeError('Zero-removal replay failed: '+str(state['failures']))
                    error=float(np.max(np.abs(state['scores'][recipe]-reference[bk+'__'+ref][rows])))
                    if error>1e-9: raise RuntimeError(f'New runner zero-removal scores differ: {ce}/{bk}/{recipe}: {error}')
                    comparisons[ce+'__'+bk][recipe+'_max_abs_difference']=error
        print('Zero-removal matrices and legacy scores PASS: '+ce,flush=True)
    gate.update(status='PASS',zero_removal=comparisons)
    dump(WORK/'REPLAY_GATE.json',gate)
    print('Zero-removal gate: all 72 bank/cell matrices, marks and partition inputs exact',flush=True)

def save_chunk(out,stem,state,metrics,statistics,extra_artifacts=()):
    directory=out/'chunks'; directory.mkdir(exist_ok=True)
    if shutil.disk_usage(ROOT).free<100_000_000: raise RuntimeError('Less than 100 MB free; stop before checkpoint')
    # COMPLETE sentinel is last. Interrupted chunks can be recomputed, completed chunks are immutable.
    pd.DataFrame(metrics).to_csv(directory/(stem+'.metrics.csv'),index=False)
    dump(directory/(stem+'.diagnostics.json'),native({'diagnostics':state['diagnostics'],'failures':state['failures'],
        'surv':state['surv'],'groups':state.get('groups'),'anchor':state.get('anchor'),
        'recipes':{k:{'weights':v['weights'],'within':v['within'],'estimator':v['estimator']} for k,v in state['recipes'].items()}}))
    np.savez_compressed(directory/(stem+'.stats.npz'),**statistics)
    dump(directory/(stem+'.complete.json'),{'status':'COMPLETE','artifacts':{suffix:sha(directory/(stem+suffix)) for suffix in ['.metrics.csv','.diagnostics.json','.stats.npz']+list(extra_artifacts)}})

def content_cell(data,bk,ce,variant,out):
    started=time.perf_counter(); bank=variant_bank(data,bk,ce,variant); state=fit_content(data,bank)
    metrics,stats=score_state(data,state,bk,ce,variant,out,started)
    save_chunk(out,ce+'__'+bk+'__'+variant,state,metrics,stats)
    return time.perf_counter()-started

def score_state(data,state,bk,ce,variant,out,started):
    bank=state['bank']
    metrics=[]; stats={'answer_indices':bank['answers']}
    for recipe in RECIPES:
        for rule in ['ANSWER_Z_Q80','TOP_1','TOP_2','TOP_3']:
            row={'bank':bk,'variant':variant,'recipe':recipe,'position_variant':'CONTENT','decision_rule':rule,'cell':ce,
                 'top2_post_hoc':rule=='TOP_2','runtime_s':time.perf_counter()-started,'review_status':'PENDING',
                 'input_manifest_sha256':sha(out/'INPUT_MANIFEST.json'),'n_total':len(bank['answers'])}
            if recipe not in state['scores']:
                reason=state['failures'].get(recipe,state['failures'].get(recipe.split('_')[0],state['failures'].get('GROUPING','No usable survivors')))
                row.update(coverage=0.,failures=len(bank['answers']),failure_reason=reason,n_answers=0,n_steps=0)
            else:
                score=state['scores'][recipe]
                flags=None if rule=='ANSWER_Z_Q80' else top_flags(score,bank['off'],np.full(len(bank['answers']),int(rule[-1])))
                metric,per=evaluate(data.pop,bank,score,flags)
                row.update(metric)
                for metricname in ['auc','confusion','first','argmax','flag_count']:
                    if rule=='ANSWER_Z_Q80': stats[recipe+'__'+rule+'__'+metricname]=per[metricname]
            metrics.append(native(row))
        # Keep the main content comparisons for original banks; all other scores reconstruct from fits.
        retain=(variant=='original' and recipe in ['DS_FILTER_EQ','MEAN_EQ','MEAN_DS','MEAN_SML','MEAN_MOM']) or (variant in ['groups_answer','groups_cell','channels_answer','channels_cell'] and recipe in ['MEAN_DS','MEAN_MOM'])
        if retain and recipe in state['scores']:
            stats[recipe+'__step_score']=state['scores'][recipe]  # float64, no precision loss
    return metrics,stats

def restore_content(data,bk,ce,variant='original'):
    import ds_group_weights as GW
    from spectral_utils.lsml_gate_locator_research import answer_standardize
    stem=ce+'__'+bk+'__'+variant
    if not completed_chunk(RUNS['level_reduction_v1']/'chunks',stem): raise RuntimeError('Missing content checkpoint: '+stem)
    bank=variant_bank(data,bk,ce,variant)
    p=RUNS['level_reduction_v1']/'chunks'/(ce+'__'+bk+'__'+variant+'.diagnostics.json')
    saved=json.loads(p.read_text(encoding='utf8'))
    state={'bank':bank,'diagnostics':saved['diagnostics'],'failures':saved['failures'],
           'surv':np.array(saved['surv'],int),'anchor':saved['anchor'],'recipes':{},'scores':{'ALL_EQ':bank['V'].mean(1)}}
    if len(state['surv']):
        state['X']=bank['V'][:,state['surv']]
        state['scores']['DS_FILTER_EQ']=state['X']@np.full(len(state['surv']),1/len(state['surv']))
    if saved['groups'] is None: return state
    state['groups']=np.array(saved['groups'],int)
    for name,r in saved['recipes'].items():
        wi=None if r['within'] is None else [None if w is None else np.asarray(w,float) for w in r['within']]
        z=GW.group_matrix(state['X'],bank['off'],state['groups'],wi,answer_standardize)
        key=data.env['KEYG'][bank['rows'],:z.shape[1]]
        votes=data.env['SB'].random_tie_marks(z,bank['off'],.2,key)
        weights=np.asarray(r['weights'],float)
        state['recipes'][name]={'Z':z,'votes':votes,'weights':weights,'within':wi,'estimator':r['estimator']}
        state['scores'][name]=GW.weighted_group_score(z,weights,signed=name.endswith('_SML'))
    return state

def run_level(data,feasibility=False):
    check_frozen(); out=RUNS['level_reduction_v1']
    code_seal(out,'feasibility' if feasibility else 'level_content')
    if feasibility:
        # No quality table: complete one cell's largest-bank fit purely to price compute.
        start=time.perf_counter(); bank=variant_bank(data,'B54','pb_gsm8k_q4','remove_3'); state=fit_content(data,bank)
        elapsed=time.perf_counter()-start
        report={'tag':'FEASIBILITY','status':'PASS','cell':'pb_gsm8k_q4','bank':'B54','variant':'remove_3',
                'fit_seconds':elapsed,'n_answers':len(bank['answers']),'n_steps':len(bank['V']),
                'recipes_with_scores':list(state['scores']),'failures':state['failures'],
                'full_fit_count':360,'naive_full_fit_seconds':elapsed*360,'quality_metrics_computed':False}
        dump(out/'FEASIBILITY.json',native(report)); print(json.dumps(native(report)),flush=True); return
    feasible=json.loads((out/'FEASIBILITY.json').read_text(encoding='utf8'))
    if feasible['status']!='PASS': raise RuntimeError('Feasibility has not passed')
    started=time.perf_counter(); complete=0
    for ce in sorted(set(data.pop.cells)):
        for bk in data.env['BN']:
            for variant in ['original','remove_2','remove_3','random_2','random_3']:
                stem=ce+'__'+bk+'__'+variant
                if (out/'chunks'/(stem+'.complete.json')).exists():
                    seal=json.loads((out/'chunks'/(stem+'.complete.json')).read_text(encoding='utf8'))
                    if any(sha(out/'chunks'/(stem+s))!=h for s,h in seal['artifacts'].items()): raise RuntimeError('Completed chunk changed: '+stem)
                else:
                    elapsed=content_cell(data,bk,ce,variant,out)
                    print(f'LEVEL {stem}: checkpoint complete in {elapsed:.1f}s',flush=True)
                complete+=1
                dump(out/'RUN_STATUS.json',{'status':'RUNNING','stage':'content','completed_cells':complete,'expected_cells':360,'runtime_s':time.perf_counter()-started})
    tables=[pd.read_csv(p) for p in sorted((out/'chunks').glob('*.metrics.csv'))]
    pd.concat(tables,ignore_index=True).to_csv(out/'METRICS.csv',index=False)
    diagnostics={p.stem:json.loads(p.read_text(encoding='utf8')) for p in sorted((out/'chunks').glob('*.diagnostics.json'))}
    dump(out/'DIAGNOSTICS.json',diagnostics)
    dump(out/'RUN_STATUS.json',{'status':'CONTENT_COMPLETE_LAYERS_PENDING','completed_cells':complete,'runtime_s':time.perf_counter()-started})
