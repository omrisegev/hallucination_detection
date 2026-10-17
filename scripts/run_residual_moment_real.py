"""Full-population LABEL-FREE scoring. Evaluation is a separate entry point."""
from pathlib import Path
import sys, json, hashlib, time, shutil
from itertools import combinations
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_context_models import RidgePredictor
from spectral_utils.energy_context_stability import group_weights
from spectral_utils.residual_moment_fusion import residuals, fit_pair, stream_top10, score_crossed, jsonable
from scripts.run_temporal_linear_context import fit as fit_ridge

OUT=ROOT/'results/residual_moment_fusion_v1'
EXCLUSIONS=[(f,) for f in range(5)]+list(combinations(range(5),2))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(path)

def key(excluded):return 'exclude_'+'_'.join(map(str,sorted(excluded)))

class Models:
    def __init__(self,bundle):self.bundle=bundle;self.cache={};self.audit={}
    def get(self,excluded):
        excluded=tuple(sorted(set(excluded)))
        if excluded in self.cache:return self.cache[excluded]
        if len(excluded)<=2:
            path=ROOT/'results/temporal_linear_context_v1'/('innovation5__'+key(excluded)+'_model.npz')
            with np.load(path,allow_pickle=False) as f:
                model=RidgePredictor(f['coefficient'],f['mean'],f['scale'])
        elif len(excluded)==3:
            path=OUT/'models'/('innovation5__'+key(excluded)+'_model.npz')
            model=fit_ridge(self.bundle,excluded,path)
        else:raise ValueError('Unexpected exclusion set')
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        held={m['group_id'] for m in self.bundle.metadata if m['fold'] in excluded}
        allowed={m['group_id'] for m in self.bundle.metadata if m['fold'] not in excluded}
        train=set(meta['training_groups']);val=set(meta['validation_groups'])
        if set(meta['excluded_folds'])!=set(excluded) or train&val or (train|val)&held or train|val!=allowed:
            raise ValueError('Frozen ridge group audit failed')
        if meta.get('correctness_labels_used') is not False:raise ValueError('Undeclared ridge supervision')
        self.cache[excluded]=model
        self.audit[key(excluded)]=dict(path=str(path),sha256=sha(path),metadata_sha256=sha(path.with_suffix('.json')),
            excluded_folds=excluded,training_groups=len(train),validation_groups=len(val),held_groups=len(held))
        print('[ridge-ready]',key(excluded),'new' if len(excluded)==3 else 'reused',flush=True)
        return model


def run():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    start=time.perf_counter();bundle=FeatureBundle(ROOT/'results/temporal_context_data_v1','innovation5');meta=bundle.metadata
    protocol=ROOT/'docs/experiments/RESIDUAL_MOMENT_FUSION_20260915.md'
    manifest=dict(schema='residual-moment-fusion-v1',protocol_sha256=sha(protocol),
        source_code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'spectral_utils/residual_moment_fusion.py',ROOT/'spectral_utils/energy_context_stability.py',ROOT/'spectral_utils/cca_iu_isolation.py',ROOT/'scripts/run_temporal_linear_context.py']},
        data_manifest_sha256=sha(ROOT/'results/temporal_context_data_v1/MANIFEST.json'),
        answers=len(meta),tokens=int(bundle.length.sum()),steps=int(meta[-1]['step_stop']),labels_used=False)
    for name,expected in bundle.manifest['files'].items():
        if sha(ROOT/'results/temporal_context_data_v1'/name)!=expected:raise ValueError('Data hash mismatch: '+name)
    if (OUT/'MANIFEST.json').exists() and json.loads((OUT/'MANIFEST.json').read_text())!=manifest:
        raise ValueError('Immutable experiment manifest changed')
    write(OUT/'MANIFEST.json',manifest)
    if shutil.disk_usage(OUT).free<500_000_000:raise OSError('Insufficient disk for bounded score archive')
    models=Models(bundle)
    for e in EXCLUSIONS+list(combinations(range(5),3)):models.get(e)
    write(OUT/'MODEL_AUDIT.json',models.audit)
    # Landmark observations contain no labels; endpoints include short prefixes.
    ids=[];positions=[]
    for i,n in enumerate(bundle.length):
        pp=np.unique(np.linspace(0,n-1,min(16,n),dtype=int));ids.extend([i]*len(pp));positions.extend(pp)
    ids=np.asarray(ids);positions=np.asarray(positions)
    level=np.asarray(bundle.features[bundle.offset[ids]+positions])[:,bundle.columns].astype(float)
    folds=np.array([m['fold'] for m in meta]);cells=np.array([m['cell'] for m in meta]);groups=np.array([m['group_id'] for m in meta])
    head_keys=['native','simplex'];moment_keys=['L','R']
    method_names=['equal__L','equal__R']+[f'{scope}__{h}__{m}{s}' for scope in ('local','pooled') for h in head_keys for m in moment_keys for s in moment_keys]
    for excluded in EXCLUSIONS:
        ek=key(excluded);path=OUT/(ek+'.npz');done=OUT/(ek+'_COMPLETE.json')
        if done.exists():
            if sha(path)!=json.loads(done.read_text())['scores_sha256']:raise ValueError('Score checkpoint changed')
            print('[resume-complete]',ek,flush=True);continue
        reference=~np.isin(folds[ids],excluded)
        if len(excluded)==2:reference&=~np.char.startswith(cells[ids],'pb_')
        rr=np.flatnonzero(reference);R=np.empty((len(rr),5))
        for donor in range(5):
            take=np.flatnonzero(folds[ids[rr]]==donor)
            if not len(take):continue
            donor_model=models.get(tuple(sorted((*excluded,donor))))
            R[take]=residuals(bundle,ids[rr[take]],positions[rr[take]],donor_model)
        pooled={};pool_audit={}
        for cell in sorted(set(cells[ids[rr]])):
            take=cells[ids[rr]]==cell;refids=ids[rr[take]]
            weight=group_weights(groups[refids],refids)
            pooled[cell]=fit_pair(level[rr[take]],R[take],weight)
            pool_audit[cell]=dict(fit=pooled[cell],reference_answer_ids=np.unique(refids),
                reference_groups=sorted(set(groups[refids])),landmarks=int(take.sum()),
                residual_predictors=[key(tuple(sorted((*excluded,k)))) for k in range(5) if k not in excluded])
        write(OUT/(ek+'_POOL.json'),pool_audit)
        arrays={name:np.full(manifest['steps'],np.nan) for name in method_names}
        query=np.flatnonzero(np.isin(folds,excluded)&(True if len(excluded)==1 else ~np.char.startswith(cells,'pb_')))
        diag_weights=np.empty((len(query),2,2,5));diag_rho=np.empty((len(query),2,5));diag_g2=np.empty((len(query),2))
        diag_C=np.empty((len(query),2,5,5));diag_scale=np.empty((len(query),5));diag_ceiling=np.empty(len(query))
        diag_fallback=np.empty((len(query),2),bool);diag_residual=np.empty((len(query),2));diag_predict=np.empty((len(query),5))
        model=models.get(excluded)
        for j,i in enumerate(query):
            m=meta[i];n=bundle.length[i];offset=bundle.offset[i]
            L=np.asarray(bundle.features[offset:offset+n])[:,bundle.columns].astype(float)
            Rq=residuals(bundle,np.full(n,i),np.arange(n),model)
            spans=np.asarray(bundle.spans[m['step_start']:m['step_stop']])-offset
            summaries={'L':stream_top10(L,spans),'R':stream_top10(Rq,spans)}
            local=fit_pair(L,Rq)
            output={f'equal__{k}':v.mean(axis=1) for k,v in summaries.items()}
            output.update(score_crossed(summaries,local,'local'))
            output.update(score_crossed(summaries,pooled[m['cell']],'pooled'))
            for name,values in output.items():
                if not np.isfinite(values).all():raise FloatingPointError('Nonfinite answer score')
                arrays[name][m['step_start']:m['step_stop']]=values
            for k,rep in enumerate(moment_keys):
                diag_weights[j,k]=[local[rep][h] for h in head_keys];diag_rho[j,k]=local[rep]['rho']
                diag_g2[j,k]=local[rep]['g2'];diag_fallback[j,k]=local[rep]['native_fallback']
                diag_C[j,k]=local['C_'+rep];diag_residual[j,k]=local[rep]['additive_residual']
            diag_scale[j]=local['sd'];diag_ceiling[j]=local['var_y'];diag_predict[j]=Rq.var(0)/np.maximum(L.var(0),1e-12)
            if (j+1)%250==0:
                state=dict(status='SCORING',excluded_folds=excluded,completed=j+1,expected=len(query),seconds=time.perf_counter()-start)
                write(OUT/'RUN_STATE.json',state);print('[real]',ek,j+1,len(query),round(state['seconds'],1),flush=True)
        with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
        path.with_suffix('.tmp').replace(path)
        np.savez_compressed(OUT/(ek+'_DIAGNOSTICS.npz'),ids=query,weights=diag_weights,rho=diag_rho,g2=diag_g2,
            covariance=diag_C,sd=diag_scale,var_y=diag_ceiling,native_fallback=diag_fallback,
            additive_residual=diag_residual,residual_variance_ratio=diag_predict)
        write(done,dict(excluded_folds=excluded,answers=len(query),scores_sha256=sha(path),
            local_native_fallback_counts=diag_fallback.sum(0),seconds_elapsed=time.perf_counter()-start))
        print('[real-complete]',ek,len(query),flush=True)
    write(OUT/'RUN_STATE.json',dict(status='SCORING_COMPLETE_PENDING_EVALUATION',answers=len(meta),steps=manifest['steps'],
        tokens=manifest['tokens'],seconds=time.perf_counter()-start,quality_labels_used=False,methods=method_names))

if __name__=='__main__':
    try:
        with threadpool_limits(limits=1):run()
    except BaseException as e:
        if OUT.exists():write(OUT/'FAILURE.json',dict(error=f'{type(e).__name__}: {e}'))
        raise
