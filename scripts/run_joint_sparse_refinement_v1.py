"""Full matched composition of sparse membership and fixed information refinement."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_fixed_readout_v1 import inputs,BANKS
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.joint_sparse_membership import alias_coordinates
from spectral_utils.joint_sparse_refinement import refine_joint_membership
from spectral_utils.lsml_gate_locator_research import _orient
from spectral_utils.digitfree_broad50 import ANCHOR
PREVIOUS=ROOT/'results/joint_sparse_membership_v1';OUT=ROOT/'results/joint_sparse_refinement_v1'


def state(status,**kwargs):
    dump(OUT/'RUN_STATE.json',dict(status=status,time=time.strftime('%Y-%m-%dT%H:%M:%S'),**kwargs))
    print(status,kwargs,flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    files=[Path(__file__),ROOT/'spectral_utils/joint_sparse_refinement.py',ROOT/'spectral_utils/joint_feature_selection.py',
        ROOT/'spectral_utils/joint_group_reliability.py',ROOT/'docs/experiments/JOINT_SPARSE_REFINEMENT_V1.md',
        PREVIOUS/'SCORES.npz',PREVIOUS/'AUDIT.json',PREVIOUS/'MANIFEST.json',
        *[PREVIOUS/f'{b}_fold{f}.json' for b in BANKS for f in range(5)]]
    manifest=dict(schema='joint-sparse-refinement-v1',hashes={str(p):sha(p) for p in files})
    mp=OUT/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('MANIFEST_DRIFT')
    dump(mp,manifest);state('LOADING');data,base,uids=inputs();rowfold=np.repeat(data['folds'],np.diff(data['offsets']))
    with np.load(PREVIOUS/'SCORES.npz') as z:
        scores={k:z[k] for k in z.files if k!='gate'};np.testing.assert_array_equal(z['gate'],data['gate'])
    cache={};native={};all_meta=[]
    for bank in BANKS:
        x=base if bank=='base' else augment(base,data['offsets'],uids,bank)
        scores[bank+'__refined']=np.full(len(x),np.nan);native[bank+'__refined']=0
        for outer in range(5):
            started=time.perf_counter();held=rowfold==outer
            parent=json.loads((PREVIOUS/f'{bank}_fold{outer}.json').read_text());aliases=parent['aliases'];z=alias_coordinates(x,aliases)
            path=OUT/f'{bank}_fold{outer}.json'
            if path.exists():
                meta=json.loads(path.read_text())
                if meta['valid']:cache[(outer,meta['training_digest'])]=meta['fit']
            else:
                meta=dict(bank=bank,outer=outer,aliases=aliases,valid=False)
                if not parent['fit']['valid']:meta['failure']='PARENT_FAILED: '+parent['fit']['failure']
                else:
                    initial=np.asarray(parent['fit']['active']);part=np.asarray(parent['fit']['final_labels']);train=z[~held][:,initial]
                    digest=hashlib.sha256(np.ascontiguousarray(train).tobytes()+part.astype(np.int64).tobytes()).hexdigest();key=(outer,digest)
                    try:
                        if key in cache:fit=cache[key];reused=True
                        else:
                            def notify(step,kept,retention):
                                if step%5==0:state('REFINING',bank=bank,outer=outer,removed=step,kept=kept,retention=retention)
                            fit=refine_joint_membership(train,part,seed=399170+200+outer,notify=notify);cache[key]=fit;reused=False
                        w=np.zeros(z.shape[1]);w[initial]=fit['unoriented_weights']
                        expanded_input=np.column_stack((z[~held],x[~held,ANCHOR]))
                        w,orientation=_orient(expanded_input,np.r_[w,0.],z.shape[1]);w=w[:-1]
                        active=initial[np.asarray(fit['active'])];raw=np.zeros(x.shape[1])
                        for j,ids in enumerate(aliases):raw[ids]=w[j]/len(ids)
                        original=[i for j in active for i in aliases[j]]
                        meta.update(valid=True,initial_active=initial,active=active,fit=fit,training_digest=digest,
                            canonical_weights=w,expanded_weights=raw,orientation=orientation,reused_fit=reused,
                            selected_added=int(sum(i>=51 for i in original)),bocpd_retained=50 in original)
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:meta['failure']=f'{type(exc).__name__}: {exc}'
                meta['seconds']=time.perf_counter()-started;dump(path,meta)
            if meta['valid']:
                scores[bank+'__refined'][held]=z[held]@np.asarray(meta['canonical_weights'])
                native[bank+'__refined']+=int(np.sum(data['folds']==outer))
            else:scores[bank+'__refined'][held]=x[held,ANCHOR]
            all_meta.append(meta);state('FOLD_COMPLETE',bank=bank,outer=outer,valid=meta['valid'],kept=len(meta.get('active',[])),seconds=meta['seconds'])
    state('EVALUATING');metrics={k:score_locator(s,data['gate'],data) for k,s in scores.items()}
    pairs=[(b+'__refined',b+'__reliability_auto') for b in BANKS]+[(b+'__refined','base__refined') for b in ('duplicates','noise')]
    ci=bootstrap(data,metrics,pairs)
    preservation={b:bool(native[b+'__refined']==13769 and ci[b+'__refined minus base__refined']['pb']['low']>-.01 and
        ci[b+'__refined minus base__refined']['within']['low']>-.002) for b in ('duplicates','noise')}
    result=dict(status='COMPLETE',metrics={k:scalar_metrics(m) for k,m in metrics.items()},primary_contrasts=ci,
        native_answers=native,practical_preservation=preservation,
        selected_counts={b:[len(m.get('active',[])) for m in all_meta if m['bank']==b] for b in BANKS},
        selected_added={b:[m.get('selected_added') for m in all_meta if m['bank']==b] for b in BANKS},
        bocpd_retained={b:[m.get('bocpd_retained') for m in all_meta if m['bank']==b] for b in BANKS},
        stress_max_score_difference={b:float(np.max(np.abs(scores[b+'__refined']-scores['base__refined']))) for b in ('duplicates','noise')},
        stress_peak_changes={b:int(np.sum(metrics[b+'__refined']['peaks']!=metrics['base__refined']['peaks'])) for b in ('duplicates','noise')})
    dump(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    lines=['# Sparse Joint membership with information refinement','','| Method | PB % | Within AUC | Native |','|---|---:|---:|---:|']
    for k,m in result['metrics'].items():lines.append(f"| {k} | {100*m['pb']:.4f} | {m['within']:.6f} | {native.get(k,'reference/control')} |")
    lines+=['','Practical preservation: '+str(preservation),'Selected counts: '+str(result['selected_counts']),
        'Score differences under stress: '+str(result['stress_max_score_difference']),'','```json',json.dumps(ci,indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');state('COMPLETE',practical_preservation=preservation)
    print('\n'.join(lines[39:47]),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:state('FAILED',error=str(exc));raise
