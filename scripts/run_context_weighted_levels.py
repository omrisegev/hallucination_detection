"""Score fixed original features with source-excluded chronological IU weights."""
from pathlib import Path
import sys,json,time,hashlib
from itertools import combinations
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.energy_context_stability import ContextFit,GroupNeighbors,conditional_moments,heads
from spectral_utils.context_weighted_levels import HEADS,METHODS,score_answer
from scripts.run_residual_moment_real import sha,write,key

SOURCE=ROOT/'results/energy_context_stability_v2'
OUT=ROOT/'results/context_weighted_levels_v1'
EXCLUSIONS=[(f,) for f in range(5)]+list(combinations(range(5),2))
HEAD_KEYS={'native':'native_a','simplex':'qp_w','group':'group_w'}

def restore_fit(values):
    fit=ContextFit()
    for k,v in values.items():setattr(fit,k,np.asarray(v) if isinstance(v,list) else v)
    return fit

def fit_case(cell,excluded,meta,data):
    name=cell+'__'+key(excluded);dest=OUT/name;dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        done=json.loads((dest/'COMPLETE.json').read_text())
        if sha(dest/'ANCHORS.npz')!=done['anchors_sha256'] or sha(dest/'FIT.json')!=done['fit_sha256']:raise ValueError('Changed anchor checkpoint')
        fit=restore_fit(json.loads((dest/'FIT.json').read_text())['preprocessing'])
        with np.load(dest/'ANCHORS.npz') as f:arrays={k:f[k] for k in f.files}
        return fit,arrays,done
    start=time.perf_counter();answer=data['answer'];cells=np.array([m['cell'] for m in meta])[answer]
    folds=np.array([m['fold'] for m in meta])[answer];groups=np.array([m['group_id'] for m in meta])[answer]
    train=np.flatnonzero((cells==cell)&~np.isin(folds,excluded));held=np.flatnonzero((cells==cell)&np.isin(folds,excluded))
    if set(groups[train])&set(groups[held]):raise ValueError('Source overlap')
    source_hashes={}
    if len(excluded)==1:
        old=SOURCE/(cell+'__exclude'+str(excluded[0]));completion=json.loads((old/'COMPLETE.json').read_text())
        for file,value in completion['hashes'].items():
            if sha(old/file)!=value:raise ValueError('Changed Step384 source: '+str(old/file))
            source_hashes[str((old/file).relative_to(ROOT))]=value
        original=json.loads((old/'FIT.json').read_text());fit=restore_fit(original['preprocessing'])
        if set(original['training_groups'])!=set(groups[train]) or set(original['held_groups'])!=set(groups[held]):raise ValueError('Source fit group mismatch')
        with np.load(old/'DIAGNOSTICS.npz') as f:
            np.testing.assert_array_equal(f['landmark_ids'],held);np.testing.assert_array_equal(f['training_landmark_ids'],train)
            arrays={'landmark_ids':held}
            for arm in ('energy','position','random'):
                for head,k in HEAD_KEYS.items():arrays[arm+'__'+head]=f[arm+'__'+k]
            arrays['energy_neff']=f['energy__neff']
    else:
        fit=ContextFit().fit(*(data[k][train] for k in ('x','hm','hs','position','length')),groups[train],answer[train])
        tp,te=fit.transform(*(data[k][train] for k in ('hm','hs','position','length')))
        qp,qe=fit.transform(*(data[k][held] for k in ('hm','hs','position','length')))
        neighbors={'position':GroupNeighbors(tp,groups[train]),'energy':GroupNeighbors(te,groups[train])}
        Z=(data['x'][train]-fit.mean)/fit.sd
        arrays={'landmark_ids':held,'energy_neff':np.empty(len(held))}
        for arm in ('energy','position','random'):
            for head in HEADS:arrays[arm+'__'+head]=np.empty((len(held),5))
        seed=int.from_bytes(hashlib.sha256(('energy-v1/'+name).encode()).digest()[:4],'little');rng=np.random.default_rng(seed)
        for startq in range(0,len(held),256):
            sl=slice(startq,min(startq+256,len(held)));n=len(qe[sl])
            ei,ew,_=neighbors['energy'].nearest(qe[sl]);arrays['energy_neff'][sl]=1/np.sum(ew**2,axis=1)
            for arm in ('energy','position','random'):
                if arm=='energy':idx,kernel=ei,ew
                elif arm=='position':idx,kernel,_=neighbors['position'].nearest(qp[sl])
                else:idx=neighbors['energy'].random(n,rng);kernel=ew
                _,C=conditional_moments(Z[idx],kernel,fit.C);h=heads(C,fit.sd,fit.var_y)
                for head,k in HEAD_KEYS.items():arrays[arm+'__'+head][sl]=h[k]
            if startq%8192==0:print('[nested-context]',name,min(startq+256,len(held)),len(held),flush=True)
    if np.min(arrays['energy_neff'])<32:raise ValueError('Insufficient effective neighbors')
    write(dest/'FIT.json',dict(cell=cell,excluded_folds=excluded,training_groups=sorted(set(groups[train])),
        held_groups=sorted(set(groups[held])),preprocessing=fit.as_dict(),source_hashes=source_hashes))
    np.savez_compressed(dest/'ANCHORS.npz',**arrays)
    done=dict(cell=cell,excluded_folds=excluded,reused=len(excluded)==1,landmarks=len(held),
        held_answers=len(set(answer[held])),seconds=time.perf_counter()-start,
        anchors_sha256=sha(dest/'ANCHORS.npz'),fit_sha256=sha(dest/'FIT.json'),energy_neff_min=float(arrays['energy_neff'].min()))
    write(dest/'COMPLETE.json',done);return fit,arrays,done

def run():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    provenance=json.loads((SOURCE/'PROVENANCE.json').read_text())
    for file,expected in provenance.items():
        if sha(ROOT/file)!=expected:raise ValueError('Step384 code/protocol changed: '+file)
    prepared=json.loads((SOURCE/'PREPARED.json').read_text())
    if sha(SOURCE/'LANDMARKS.npz')!=prepared['sha256']:raise ValueError('Landmark hash mismatch')
    bundle=FeatureBundle(ROOT/'results/temporal_context_data_v1','innovation5');meta=bundle.metadata
    for file,expected in bundle.manifest['files'].items():
        if sha(ROOT/'results/temporal_context_data_v1'/file)!=expected:raise ValueError('Feature bundle changed')
    with np.load(SOURCE/'LANDMARKS.npz') as f:data={k:f[k] for k in f.files}
    manifest=dict(protocol_sha256=sha(ROOT/'docs/experiments/CONTEXT_WEIGHTED_LEVEL_FUSION_20260915.md'),
        code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'spectral_utils/context_weighted_levels.py']},
        source_provenance_sha256=sha(SOURCE/'PROVENANCE.json'),landmarks_sha256=prepared['sha256'],
        data_manifest_sha256=sha(ROOT/'results/temporal_context_data_v1/MANIFEST.json'),methods=METHODS,
        labels_in_fit=False,answers=len(meta),steps=meta[-1]['step_stop'],tokens=int(bundle.length.sum()))
    if (OUT/'MANIFEST.json').exists() and json.loads((OUT/'MANIFEST.json').read_text())!=json.loads(json.dumps(manifest)):
        raise ValueError('Frozen run manifest changed')
    write(OUT/'MANIFEST.json',manifest);fit_stats=[]
    for excluded in EXCLUSIONS:
        path=OUT/(key(excluded)+'.npz');donepath=OUT/(key(excluded)+'_COMPLETE.json')
        if donepath.exists():
            if sha(path)!=json.loads(donepath.read_text())['sha256']:raise ValueError('Score archive changed')
            print('[resume]',key(excluded),flush=True);continue
        scores={m:np.full(manifest['steps'],np.nan) for m in METHODS}
        cells=sorted(set(m['cell'] for m in meta)) if len(excluded)==1 else ['prmbench_qwen3_8b']
        count=0
        for cell in cells:
            fit,arrays,stats=fit_case(cell,excluded,meta,data);fit_stats.append(stats)
            landmark_ids=arrays['landmark_ids'];answer_ids=data['answer'][landmark_ids]
            sh=heads(fit.C,fit.sd,fit.var_y);static={h:sh[k][0] for h,k in HEAD_KEYS.items()}
            unique,first,counts=np.unique(answer_ids,return_index=True,return_counts=True)
            for i,begin,size in zip(unique,first,counts):
                rows=slice(begin,begin+size)
                if not np.all(answer_ids[rows]==i):raise ValueError('Noncontiguous answer anchors')
                m=meta[i];n=m['tokens'];offset=m['offset'];positions=data['local'][landmark_ids[rows]]
                x=np.asarray(bundle.features[offset:offset+n],float);spans=np.asarray(bundle.spans[m['step_start']:m['step_stop']])-offset
                local={h:{a:arrays[a+'__'+h][rows] for a in ('energy','position','random')} for h in HEADS}
                output=score_answer(x,spans,positions,local,static,fit.sd)
                for name,values in output.items():
                    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite contextual score')
                    scores[name][m['step_start']:m['step_stop']]=values
                count+=1
            write(OUT/'RUN_STATE.json',dict(status='SCORING',excluded_folds=excluded,last_cell=cell,answers_in_exclusion=count,
                seconds=time.perf_counter()-start,completed_fits=len(fit_stats),fitting_labels_used=False))
            print('[context-complete]',cell,key(excluded),count,round(time.perf_counter()-start,1),flush=True)
        with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**scores)
        path.with_suffix('.tmp').replace(path)
        write(donepath,dict(excluded_folds=excluded,answers=count,sha256=sha(path)))
    write(OUT/'RUN_STATE.json',dict(status='SCORING_COMPLETE_PENDING_EVALUATION',answers=len(meta),steps=manifest['steps'],
        tokens=manifest['tokens'],seconds=time.perf_counter()-start,methods=METHODS,fitting_labels_used=False))

if __name__=='__main__':
    try:
        with threadpool_limits(limits=1):run()
    except BaseException as e:
        if OUT.exists():write(OUT/'FAILURE.json',dict(error=f'{type(e).__name__}: {e}'))
        raise
