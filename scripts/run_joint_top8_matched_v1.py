import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digitfree_broad50_v1 import load_data,sha
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from spectral_utils.digitfree_broad50 import NAMES,ANCHOR,masked_answer_standardize
from spectral_utils.joint_feature_selection import pruning_path
from spectral_utils.joint_block_balanced import discover_sourcefold_groups
from spectral_utils.lsml_gate_locator_research import fit_fusion_weights,FusionRecipe
OUT=ROOT/'results/broad50_top8_v1';PREVIOUS=ROOT/'results/joint_feature_selection_bocpd_v1';SEED=399170


def main():
    assert json.loads((OUT/'RUN_STATE.json').read_text())['status'] in ('EXTRACTION_COMPLETE','FITTING','COMPLETE')
    paths=[Path(__file__),ROOT/'docs/experiments/JOINT_TOP8_MATCHED_V1.md',
        ROOT/'spectral_utils/joint_feature_selection.py',OUT/'EXTRACTION_MANIFEST.json',PREVIOUS/'SCORES.npz',PREVIOUS/'INPUTS.npz']
    manifest=dict(schema='joint-top8-matched-v1',hashes={str(p):sha(p) for p in paths})
    fp=OUT/'FIT_MANIFEST.json'
    if fp.exists() and json.loads(fp.read_text())!=manifest:raise ValueError('fit manifest drift')
    dump(fp,manifest);data=load_data();offsets=data['offsets'];raw=np.full(data['x'].shape,np.nan);available=np.zeros(raw.shape,bool)
    done=np.zeros(len(offsets)-1,bool)
    for p in sorted((OUT/'extracted').glob('*.npz')):
        with np.load(p) as z:
            values=z['values'];mask=z['available'];indexes=z['indexes'];cursor=0
            for i in indexes:
                assert not done[i];a,b=offsets[i:i+2];n=b-a;raw[a:b]=values[cursor:cursor+n];available[a:b]=mask[cursor:cursor+n];cursor+=n;done[i]=True
    assert done.all();x=masked_answer_standardize(raw,available,offsets)
    with np.load(PREVIOUS/'INPUTS.npz') as z:bocpd=z['bocpd']
    rowfold=np.repeat(data['folds'],np.diff(offsets));scores={};details={}
    for bank,values in (('B50',x),('B51_bocpd',np.column_stack((x,bocpd)))):
        names=tuple(NAMES)+(('bocpd_residual',) if bank!='B50' else ())
        arms=('joint_full','joint_auto','equal_full','equal_selected','continuous')
        out={a:np.full(len(x),np.nan) for a in arms};folds=[]
        for outer in range(5):
            p=OUT/f'{bank}_fold{outer}';held=rowfold==outer
            if p.with_suffix('.npz').exists() and p.with_suffix('.json').exists():
                with np.load(p.with_suffix('.npz')) as z:
                    for a in arms:out[a][held]=z[a]
                folds.append(json.loads(p.with_suffix('.json').read_text()));continue
            print('top8 fit',bank,outer,flush=True);dump(OUT/'RUN_STATE.json',dict(status='FITTING',bank=bank,outer=outer))
            train=values[~held];discovery=discover_sourcefold_groups(train,rowfold[~held],seed=SEED+100+outer)
            weights={'equal_full':np.ones(values.shape[1])/values.shape[1]};meta=dict(outer=outer,discovery=discovery,valid={})
            try:
                weights['continuous'],m=fit_fusion_weights(train,FusionRecipe(bank,names,'continuous',anchor=ANCHOR),seed=SEED+outer)
                meta['continuous']=m
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:meta['continuous_failure']=str(exc)
            try:
                if discovery['status']!='SELECTED':raise ValueError('NO_ADMISSIBLE_PARTITION')
                fit=pruning_path(train,np.asarray(discovery['labels']),anchor_index=ANCHOR,seed=SEED+200+outer,stop_after_automatic=True)
                weights['joint_full']=fit['path'][0]['weights'];weights['joint_auto']=fit['automatic']['weights']
                w=np.zeros(values.shape[1]);w[fit['automatic']['active']]=1/len(fit['automatic']['active']);weights['equal_selected']=w
                meta['selection']=fit;meta['selected_names']=[names[i] for i in fit['automatic']['active']]
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:meta['selection_failure']=str(exc)
            for a in arms:
                meta['valid'][a]=a in weights
                if a not in weights:weights[a]=np.eye(values.shape[1])[ANCHOR]
                out[a][held]=values[held]@weights[a]
            meta['weights']=weights;dump(p.with_suffix('.json'),meta);np.savez_compressed(p.with_suffix('.npz'),**{a:out[a][held] for a in arms});folds.append(meta)
        scores.update({bank+'__'+a:s for a,s in out.items()});details[bank]=folds
    with np.load(PREVIOUS/'SCORES.npz') as z:
        for bank in ('B50','B51_bocpd'):scores[bank+'__top10_auto']=z[bank+'__joint_auto']
        for key in ('innovation5','historical_bocpd'):scores[key]=z[key]
    metrics={a:score_locator(s,data['gate'],data) for a,s in scores.items()}
    pairs=[(b+'__joint_auto',b+'__top10_auto') for b in ('B50','B51_bocpd')]
    summary=dict(status='COMPLETE',metrics={a:scalar_metrics(m) for a,m in metrics.items()},
        primary_contrasts=bootstrap(data,metrics,pairs),selected_counts={b:[len(f.get('selected_names',[])) for f in fs] for b,fs in details.items()},
        native_answers={b+'__'+a:sum(int(np.sum(data['folds']==f['outer'])) for f in fs if f['valid'][a]) for b,fs in details.items() for a in fs[0]['valid']})
    dump(OUT/'RESULTS.json',summary);np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    dump(OUT/'RUN_STATE.json',dict(status='COMPLETE'))
    lines=['# Matched Top8 follow-up','', '| Method | PB % | PRMB within AUC |','|---|---:|---:|']
    for a,m in summary['metrics'].items():lines.append(f"| {a} | {100*m['pb']:.4f} | {m['within']:.6f} |")
    lines+=['','Primary readout contrasts,98.75% source-group intervals:','```json',json.dumps(summary['primary_contrasts'],indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
