"""Frozen full-answer-population unlabeled landmark diagnostic; no evaluators."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.context_training import METADATA_KEYS
from spectral_utils.energy_context_stability import (history_landmarks,ContextFit,
    GroupNeighbors,conditional_moments,heads,gaussian_nll,bootstrap_metrics,signal_noise,unit)
DATA=ROOT/'results/temporal_context_data_v1'
OUT=ROOT/'results/energy_context_stability_v2'
ARMS=('static','position','energy','random')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)


def code_hashes():
    names=['docs/experiments/ENERGY_CONTEXT_STABILITY_20260915.md',
           'docs/experiments/ENERGY_CONTEXT_STABILITY_NUMERICAL_AMENDMENT_20260915.md',
           'scripts/run_energy_context_stability.py','spectral_utils/energy_context_stability.py',
           'spectral_utils/cca_iu_isolation.py','spectral_utils/upcr.py']
    return {n:sha(ROOT/n) for n in names}


def prepare():
    manifest=json.loads((DATA/'MANIFEST.json').read_text())
    for name in ('METADATA.json','features.npy'):
        if sha(DATA/name)!=manifest['files'][name]:raise ValueError('Source SHA mismatch '+name)
    metadata=json.loads((DATA/'METADATA.json').read_text())
    if len(metadata)!=13769 or any(set(m)!=METADATA_KEYS for m in metadata):raise ValueError('Metadata contract')
    groups={}
    for m in metadata:
        if m['group_id'] in groups and groups[m['group_id']]!=m['fold']:raise ValueError('Source crosses folds')
        groups[m['group_id']]=m['fold']
    prepared=OUT/'LANDMARKS.npz'
    if prepared.exists():
        saved=json.loads((OUT/'PREPARED.json').read_text())
        if saved['source_files']!=manifest['files'] or sha(prepared)!=saved['sha256']:raise ValueError('Prepared source mismatch')
        return metadata,dict(np.load(prepared,allow_pickle=False))
    features=np.load(DATA/'features.npy',mmap_mode='r');result={k:[] for k in ('x','hm','hs','answer','local','position','length')}
    excluded=[]
    for i,m in enumerate(metadata):
        x=np.asarray(features[m['offset']:m['offset']+m['tokens']],float)
        p,hm,hs=history_landmarks(x)
        if not len(p):excluded.append(i);continue
        for key,v in dict(x=x[p],hm=hm,hs=hs,answer=np.full(len(p),i),local=p,
                          position=(p+.5)/len(x),length=np.full(len(p),len(x))).items():result[key].append(v)
    result={k:np.concatenate(v) for k,v in result.items()}
    np.savez_compressed(prepared,**result)
    save(OUT/'PREPARED.json',dict(source_files=manifest['files'],source_manifest_sha256=sha(DATA/'MANIFEST.json'),
        answers=len(metadata),covered_answers=int(len(np.unique(result['answer']))),landmarks=len(result['x']),
        excluded_short_answers=excluded,sha256=sha(prepared)))
    return metadata,result


def fit_case(cell,fold,metadata,data):
    name=f'{cell}__exclude{fold}';dest=OUT/name;dest.mkdir(exist_ok=True)
    done=dest/'COMPLETE.json'
    if done.exists():
        record=json.loads(done.read_text())
        for file,expected in record['hashes'].items():
            if sha(dest/file)!=expected:raise ValueError('Checkpoint artifact mismatch')
        return record
    started=time.time()
    answer=data['answer'];cells=np.array([m['cell'] for m in metadata])[answer]
    folds=np.array([m['fold'] for m in metadata])[answer];groups=np.array([m['group_id'] for m in metadata])[answer]
    train=np.flatnonzero((cells==cell)&(folds!=fold));held=np.flatnonzero((cells==cell)&(folds==fold))
    if set(groups[train])&set(groups[held]):raise ValueError('Source firewall')
    fit=ContextFit().fit(*(data[k][train] for k in ('x','hm','hs','position','length')),groups[train],answer[train])
    train_pos,train_energy=fit.transform(*(data[k][train] for k in ('hm','hs','position','length')))
    query_pos,query_energy=fit.transform(*(data[k][held] for k in ('hm','hs','position','length')))
    Z=(data['x'][train]-fit.mean)/fit.sd;Y=(data['x'][held]-fit.mean)/fit.sd
    neighbors={a:GroupNeighbors(z,groups[train]) for a,z in [('position',train_pos),('energy',train_energy)]}
    seed=int.from_bytes(hashlib.sha256(('energy-v1/'+name).encode()).digest()[:4],'little')
    rng=np.random.default_rng(seed)
    # Deterministic anchors from16 distinct held source groups, one query each.
    source_order=sorted(set(groups[held]),key=lambda g:hashlib.sha256((name+'/'+g).encode()).digest())[:16]
    anchors=[]
    for g in source_order:
        eligible=np.flatnonzero(groups[held]==g)
        anchors.append(int(eligible[int.from_bytes(hashlib.sha256(g.encode()).digest()[:4],'little')%len(eligible)]))
    anchors=np.array(sorted(anchors));static=heads(fit.C,fit.sd,fit.var_y)
    arrays={'landmark_ids':held,'target':Y,'anchors':anchors,'training_landmark_ids':train}
    for arm in ARMS:
        for key in ('mu','C','rho','g2','additive_residual','native_a','qp_w','qp_a','group_w','group_a','beta','condition','nll','neff'):
            shape=(len(held),5,5) if key=='C' else (len(held),5) if key in ('mu','rho','native_a','qp_w','qp_a','group_w','group_a') else (len(held),)
            arrays[arm+'__'+key]=np.empty(shape)
    anchor_data={a:{} for a in ARMS if a!='static'}
    for start in range(0,len(held),256):
        stop=min(start+256,len(held));sl=slice(start,stop);n=stop-start
        ids_e,kw_e,dist=neighbors['energy'].nearest(query_energy[sl])
        for arm in ARMS:
            if arm=='static':
                mu=np.zeros((n,5));C=np.broadcast_to(fit.C,(n,5,5));h={k:np.broadcast_to(v,(n,)+v.shape[1:]) for k,v in static.items()}
                neff=np.full(n,len(set(groups[train])))
            else:
                if arm=='energy':ids,kw=ids_e,kw_e
                elif arm=='position':ids,kw,_=neighbors['position'].nearest(query_pos[sl])
                else:ids=neighbors['energy'].random(n,rng);kw=kw_e
                local=Z[ids];mu,C=conditional_moments(local,kw,fit.C);h=heads(C,fit.sd,fit.var_y)
                neff=1/np.sum(kw**2,axis=1)
                if neff.min()<32:raise ValueError('Insufficient effective support')
                for anchor in anchors[(anchors>=start)&(anchors<stop)]:
                    q=anchor-start;anchor_data[arm][int(anchor)]=(local[q],kw[q],train[ids[q]])
            for key,value in dict(h,mu=mu,nll=gaussian_nll(Y[sl],mu,C),neff=neff).items():arrays[arm+'__'+key][sl]=value
        if start and start%4096==0:print(name,'queries',stop,'/',len(held),flush=True)
    diagnostics={}
    for arm in ('position','energy','random'):
        local=np.array([anchor_data[arm][int(a)][0] for a in anchors]);kw=np.array([anchor_data[arm][int(a)][1] for a in anchors])
        arrays[arm+'__anchor_reference_ids']=np.array([anchor_data[arm][int(a)][2] for a in anchors])
        arrays[arm+'__anchor_kernel']=kw
        boot=bootstrap_metrics(local,kw,fit,np.random.default_rng(seed+1))
        diagnostics[arm]={k:signal_noise(arrays[arm+'__'+k][anchors],boot[k]) for k in boot}
        for key,value in boot.items():arrays[arm+'__bootstrap_'+key]=value
    telemetry={}
    for arm in ARMS:
        qp=arrays[arm+'__qp_a'];group=arrays[arm+'__group_a'];native=arrays[arm+'__native_a']
        telemetry[arm]=dict(g2_ceiling_fraction=float(np.mean(arrays[arm+'__g2']>=fit.var_y*(1-1.5/300))),
            mean_additive_residual=float(arrays[arm+'__additive_residual'].mean()),
            neff_min=float(arrays[arm+'__neff'].min()),condition_median=float(np.median(arrays[arm+'__condition'])),
            qp_group_cosine_mean=float(np.mean(np.sum(unit(qp)*unit(group),axis=1))),
            native_amplitude_median=float(np.median(np.linalg.norm(native,axis=1))),
            qp_amplitude_median=float(np.median(np.linalg.norm(qp,axis=1))),
            native_angle_to_static_mean=float(np.mean(np.arccos(np.clip(unit(native)@unit(static['native_a'])[0],-1,1)))),
            qp_angle_to_static_mean=float(np.mean(np.arccos(np.clip(unit(qp)@unit(static['qp_a'])[0],-1,1)))))
    np.savez_compressed(dest/'DIAGNOSTICS.npz',**arrays)
    save(dest/'FIT.json',dict(cell=cell,held_fold=fold,training_groups=sorted(set(groups[train])),
                             held_groups=sorted(set(groups[held])),preprocessing=fit.as_dict()))
    record=dict(cell=cell,fold=fold,training_landmarks=len(train),held_landmarks=len(held),
        held_answers=len(set(answer[held])),stability=diagnostics,telemetry=telemetry,
        elapsed=time.time()-started,hashes={n:sha(dest/n) for n in ('DIAGNOSTICS.npz','FIT.json')})
    save(done,record);return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--max-fits',type=int);args=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);code=code_hashes();path=OUT/'PROVENANCE.json'
    if path.exists() and json.loads(path.read_text())!=code:raise ValueError('Code/protocol changed')
    save(path,code);started=time.time()
    with threadpool_limits(limits=1):
        metadata,data=prepare();completed=0
        for cell in sorted(set(m['cell'] for m in metadata)):
            for fold in range(5):
                row=fit_case(cell,fold,metadata,data);completed+=1
                save(OUT/'RUN_STATE.json',dict(state='RUNNING',completed=completed,total=45,last=[cell,fold],elapsed=time.time()-started))
                print(cell,fold,'complete',completed,'fit_seconds',round(row['elapsed'],1),flush=True)
                if args.max_fits and completed>=args.max_fits:return
    save(OUT/'RUN_STATE.json',dict(state='COMPLETE_PENDING_AUDIT',completed=45,total=45,elapsed=time.time()-started))


if __name__=='__main__':main()
