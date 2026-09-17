"""End-to-end stress of automatic Joint inclusion on the entire benchmark."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digitfree_broad50_v1 import load_data,sha
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from spectral_utils.digitfree_broad50 import NAMES,ANCHOR
from spectral_utils.joint_feature_selection import pruning_path
from spectral_utils.joint_block_balanced import discover_sourcefold_groups
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.lsml_gate_locator_research import fit_fusion_weights,FusionRecipe
PREVIOUS=ROOT/'results/joint_feature_selection_bocpd_v1'
OUT=ROOT/'results/joint_full_redundancy_stress_v1'
SEED=399170
ARMS=('joint_full','joint_auto','continuous','equal_full','equal_selected')


def state(status,**kwargs):
    dump(OUT/'RUN_STATE.json',dict(status=status,time=time.strftime('%Y-%m-%dT%H:%M:%S'),**kwargs))
    print(status,kwargs,flush=True)


def inputs():
    data=load_data()
    with np.load(PREVIOUS/'INPUTS.npz') as z:base=np.column_stack((data['x'],z['bocpd']))
    records=json.loads((ROOT.parents[1]/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    assert len(records)==len(data['target'])==13769
    return data,base,[r['uid'] for r in records]


def main():
    OUT.mkdir(exist_ok=True)
    paths=[Path(__file__),ROOT/'spectral_utils/joint_redundancy_stress.py',ROOT/'spectral_utils/joint_feature_selection.py',
        ROOT/'spectral_utils/joint_block_balanced.py',ROOT/'docs/experiments/JOINT_FULL_REDUNDANCY_STRESS_V1.md',
        PREVIOUS/'INPUTS.npz',PREVIOUS/'SCORES.npz',PREVIOUS/'MANIFEST.json',ROOT/'results/digitfree_broad50_v1/MANIFEST.json']
    manifest=dict(schema='joint-full-redundancy-stress-v1',seed=SEED,noise_seed=401170,hashes={str(p):sha(p) for p in paths})
    mp=OUT/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('manifest drift')
    dump(mp,manifest);state('LOADING');data,base,uids=inputs()
    rowfold=np.repeat(data['folds'],np.diff(data['offsets']));scores={};all_meta={}
    with np.load(PREVIOUS/'SCORES.npz') as z:
        for a in ARMS:scores['base__'+a]=z['B51_bocpd__'+a]
        for a in ('innovation5','historical_bocpd'):scores[a]=z[a]
        np.testing.assert_array_equal(z['gate'],data['gate'])
    for kind in ('duplicates','noise'):
        x=augment(base,data['offsets'],uids,kind);names=tuple(NAMES)+('bocpd_residual',)+tuple(f'{kind}_{i+1}' for i in range(15))
        assert x.shape==(145597,66) and np.isfinite(x).all()
        output={a:np.full(len(x),np.nan) for a in ARMS};folds=[]
        for outer in range(5):
            path=OUT/f'{kind}_fold{outer}';held=rowfold==outer
            if path.with_suffix('.npz').exists() and path.with_suffix('.json').exists():
                with np.load(path.with_suffix('.npz')) as z:
                    for a in ARMS:output[a][held]=z[a]
                folds.append(json.loads(path.with_suffix('.json').read_text()));continue
            started=time.perf_counter();train=x[~held];state('GROUP_DISCOVERY',kind=kind,outer=outer)
            discovery=discover_sourcefold_groups(train,rowfold[~held],seed=SEED+100+outer)
            meta=dict(outer=outer,names=names,discovery=discovery,valid={},weights={})
            weights={'equal_full':np.ones(66)/66}
            try:
                weights['continuous'],m=fit_fusion_weights(train,FusionRecipe(kind,names,'continuous',anchor=ANCHOR),seed=SEED+outer)
                meta['continuous']=m
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:meta['continuous_failure']=str(exc)
            state('JOINT_FIT',kind=kind,outer=outer,group_sizes=discovery.get('group_sizes'))
            try:
                if discovery['status']!='SELECTED':raise ValueError('NO_ADMISSIBLE_PARTITION')
                def notify(step,kept,retention):
                    if step%5==0:state('PRUNING',kind=kind,outer=outer,removed=step,kept=kept,retention=retention)
                fit=pruning_path(train,np.asarray(discovery['labels']),anchor_index=ANCHOR,seed=SEED+200+outer,
                    maximum_removals=58,stop_after_automatic=True,notify=notify)
                weights['joint_full']=fit['path'][0]['weights'];weights['joint_auto']=fit['automatic']['weights']
                selected=np.asarray(fit['automatic']['active']);equal=np.zeros(66);equal[selected]=1/len(selected);weights['equal_selected']=equal
                meta['selection']=fit;meta['selected_names']=[names[i] for i in selected]
                meta['selected_added_features']=int(np.sum(selected>=51))
                if kind=='duplicates':meta['both_copies_retained']=int(sum(i in selected and i+51 in selected for i in range(15)))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:meta['selection_failure']=f'{type(exc).__name__}: {exc}'
            for a in ARMS:
                meta['valid'][a]=a in weights
                if a not in weights:weights[a]=np.eye(66)[ANCHOR]
                output[a][held]=x[held]@weights[a]
            meta['weights']=weights;meta['seconds']=time.perf_counter()-started
            dump(path.with_suffix('.json'),meta);np.savez_compressed(path.with_suffix('.npz'),**{a:output[a][held] for a in ARMS})
            folds.append(meta);state('FOLD_COMPLETE',kind=kind,outer=outer,seconds=meta['seconds'],valid=meta['valid'])
        scores.update({kind+'__'+a:s for a,s in output.items()});all_meta[kind]=folds
    state('EVALUATING');metrics={a:score_locator(s,data['gate'],data) for a,s in scores.items()}
    pairs=[(kind+'__'+a,'base__'+a) for kind in ('duplicates','noise') for a in ('joint_auto','joint_full')]
    intervals=bootstrap(data,metrics,pairs);native={};stability={};preserved={}
    for kind,fs in all_meta.items():
        for a in ARMS:native[kind+'__'+a]=sum(int(np.sum(data['folds']==f['outer'])) for f in fs if f['valid'][a])
        for a in ('joint_auto','joint_full'):
            key=kind+'__'+a;c=intervals[key+' minus base__'+a]
            preserved[key]=bool(native[key]==13769 and c['pb']['low']>-.01 and c['within']['low']>-.002)
            stability[key]=dict(peak_changed=int(np.sum(metrics[key]['peaks']!=metrics['base__'+a]['peaks'])),
                final_prediction_changed=int(np.sum(metrics[key]['prediction']!=metrics['base__'+a]['prediction'])))
    result=dict(status='COMPLETE',metrics={a:scalar_metrics(m) for a,m in metrics.items()},primary_contrasts=intervals,
        native_answers=native,practical_preservation=preserved,stability=stability,
        selected_counts={k:[len(f.get('selected_names',[])) for f in fs] for k,fs in all_meta.items()},
        selected_added={k:[f.get('selected_added_features') for f in fs] for k,fs in all_meta.items()})
    dump(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    lines=['# Full-data Joint redundancy/noise stress','', '| Method | PB % | PRMB within AUC | Native answers |', '|---|---:|---:|---:|']
    for a,m in result['metrics'].items():lines.append(f"| {a} | {100*m['pb']:.4f} | {m['within']:.6f} | {native.get(a,'frozen reference')} |")
    lines+=['','Practical preservation (-1pp PB / -.002 within, corrected intervals): '+str(preserved),
        '', 'Exact-copy prediction invariance is separate; see stability and native failures in RESULTS.json.',
        '', '```json',json.dumps(intervals,indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');state('COMPLETE',practical_preservation=preserved)
    print('\n'.join(lines[:20]),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:state('FAILED',error=str(exc));raise
