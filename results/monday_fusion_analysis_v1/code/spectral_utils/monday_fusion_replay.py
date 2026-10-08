"""Run the pinned historical function bodies with relocated, verified inputs."""
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from spectral_utils.monday_fusion_protocol import ROOT, WORK, CODE, SSL, RUNS, sha, dump
from spectral_utils.lsml_gate_locator_research import answer_standardize

class HistoricalData:
    def __init__(self, manifest, pop, banks=None):
        sys.path.insert(0,str(CODE / 'scripts/experiments'))
        import er_stage_a as SA, er_stage_b as SB, tail_calib_common as TC
        import lsml_merge_step as MS, ds_group_weights as GW, position_prior_ds as PP
        from calfix_common import tail_marks
        self.manifest, self.pop = manifest,pop
        paths = {k:Path(v['path']) for k,v in manifest['inputs'].items()}
        lv = np.load(paths['level_bank']); names11 = list(map(str,lv['channels']))
        prof = np.load(paths['ct7_profiles']).astype(float)
        pn = json.loads(paths['ct7_profile_validation'].read_text(encoding='utf8'))['channels']
        raw13 = np.column_stack([lv['level'].astype(float),prof[:,pn.index('chosen_token_z_despiked')],lv['derivative'][:,names11.index('chosen_surprisal')].astype(float)])
        values = answer_standardize(raw13,pop.off)
        df = np.load(paths['digit_features']); act = df['active'].astype(bool); dfv=df['values']; d3=np.zeros((pop.total,3))
        for a,b in zip(pop.off[:-1],pop.off[1:]):
            for j in range(3):
                valid=act[a:b,j]; y=dfv[a:b,j][valid]
                if len(y) and y.std()>1e-12: d3[a:b,j][valid]=(y-y.mean())/y.std()
        pool=np.load(paths['pool_z'],mmap_mode='r')
        pn=json.loads(paths['pool_names'].read_text(encoding='utf8'))
        ps=pd.read_csv(paths['pool_structure']).set_index('channel')
        level=pool[:,[pn.index(c) for c in ['q15_H1','q15_VE1','logprob_margin','true_tail50','energy_level']]].mean(1)
        def hard_stop(reason): raise RuntimeError('Historical gate: '+reason)
        env=dict(np=np,pd=pd,spearmanr=spearmanr,SA=SA,SB=SB,TC=TC,MS=MS,GW=GW,PP=PP,
                 answer_standardize=answer_standardize,tail_marks=tail_marks,hard_stop=hard_stop,
                 off=pop.off,ns=pop.ns,S=pop.total,n=pop.n,fold=pop.fold,
                 names11=names11,values=values,D3=d3,POOL=pool,PN=pn,PS=ps,
                 BN=manifest['banks'],DIG=['digit_alternative','digit_spread','digit_alternative_innovation'],
                 LEVEL=level,PRMC='prmbench_qwen3_8b')
        source=CODE / 'scripts/experiments/per_dataset_fit_run.py'
        tree=ast.parse(source.read_text(encoding='utf8'))
        keep={'zt','answer_z','marks_ok','rows_of','bank','cell_bank','CFFail','stop','n_clip','logit_pi','one_bin_start','prior_fit','slope','fe_score','fit_arms'}
        nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in keep]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
        banks=list(manifest['banks']) if banks is None else list(banks)
        vb={bk:env['bank'](bk) for bk in banks}
        marks={}; tta={}; keys={}
        for bk,v in vb.items():
            keys[bk]=np.random.default_rng(20260928).random(v.shape)
            marks[bk]=SB.random_tie_marks(v,pop.off,.2,keys[bk])
            tta[bk]=tail_marks(v,pop.off,.2,tie_aware=True,centred=True)[0]
        pos=np.arange(pop.total)-np.repeat(pop.off[:-1],pop.ns)
        posz=env['answer_z'](pos.astype(float)); keyp=np.random.default_rng(20260931).random((pop.total,1))
        env.update(VB=vb,MARKS=marks,TTA=tta,KEYB=keys,POSZ=posz,
                   POSM=SB.random_tie_marks(posz[:,None],pop.off,.2,keyp),
                   KEYG=np.random.default_rng(20260929).random((pop.total,13)),
                   BINS={1:np.zeros(pop.total,np.int64),10:PP.position_bins(pop.off,10)},
                   kneed=np.maximum(1,np.ceil(.2*pop.ns)).astype(int),
                   LFCOL={bk:[(i,c[4:]) for i,c in enumerate(manifest['banks'][bk]) if c.startswith('lf__')] for bk in banks})
        self.env=env; self.raw13=raw13; self.digit_raw=dfv

def replay(manifest,pop):
    t=time.perf_counter(); data=HistoricalData(manifest,pop); e=data.env
    code={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*.py')}
    for directory in (ROOT/'spectral_utils/external_generalization',ROOT/'.worktrees/cumulative-vote-fusion-v2/scripts/experiments/cvf_v2'):
        for p in directory.rglob('*.py'): code[str(p)]=sha(p)
    dump(WORK/'REPLAY_CODE_MANIFEST.json',code)
    ref457=np.load(manifest['inputs']['step457']['path'])
    ref462=np.load(manifest['inputs']['step462']['path'])
    fit=pop.rows(np.isin(pop.fold,[2,3,4])); pf=fit[np.repeat(pop.prm,pop.ns)[fit]]; ev=pop.rows(pop.fold==0)
    results={'status':'RUNNING','tolerance':1e-9,'step457':{},'step462':{},'zero_removal':{}}
    def check(bucket,key,actual,expected):
        assert actual.shape==expected.shape, (key,actual.shape,expected.shape)
        err=float(np.max(np.abs(actual-expected)))
        results[bucket][key]={'max_abs_difference':err,'n_steps':len(actual)}
        dump(WORK/'REPLAY_GATE.json',results)
        if not np.isfinite(err) or err>1e-9:
            raise RuntimeError(f'Replay {bucket}/{key}: {err} exceeds frozen 1e-9 tolerance')
    for bk in ['B16','B23','B35','B54']:
        o,_=e['fit_arms'](e['VB'][bk],e['MARKS'][bk],e['TTA'][bk],pf,fit,'Step457 '+bk)
        for arm,ref in [('BASE','BASE'),('GRP','EQ_DSM')]:
            check('step457',bk+'__'+arm,o[arm][ev],ref457[bk+'__P0__'+ref][ev])
        print('Step457 replay '+bk+' PASS',flush=True)
    step_cells=np.repeat(pop.cells,pop.ns)
    for ce in sorted(set(pop.cells)):
        rc=np.flatnonzero(step_cells==ce)
        for bk in ['B16','B23','B35','B54']:
            v,m,tt,_,_=e['cell_bank'](bk,rc,ce)
            o,_=e['fit_arms'](v,m,tt,rc,rc,ce+' '+bk)
            for arm in ['BASE','GRP']:
                check('step462',ce+'__'+bk+'__'+arm,o[arm][rc],ref462[bk+'__'+arm][rc])
            print('Step462 replay '+ce+' '+bk+' PASS',flush=True)
    results.update(status='HISTORICAL_PASS_ZERO_REMOVAL_PENDING',runtime_s=time.perf_counter()-t)
    dump(WORK/'REPLAY_GATE.json',results)
    print(json.dumps({'status':results['status'],'comparisons':len(results['step457'])+len(results['step462']),'runtime_s':results['runtime_s']}),flush=True)
