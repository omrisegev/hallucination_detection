"""Historical 24-cell contract: answer summaries, then cell-local unsupervised fusion."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_direct_probability_fusion_v2 as old
from spectral_utils import moment_rbm_fusion as core
from spectral_utils.direct_probability_fusion import top_mean,zscore_columns,_orient
from spectral_utils.varentropy_contribution_fusion import contributions
from spectral_utils.upcr import upcr_fit
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from scripts.inscope_cells import INSCOPE,GROUP

OUT=ROOT/'results/global_moment_fusion_24_v1'
CORE=tuple(core.METHODS)
EXTRA=('var15_raw','var15_equal','var15_iu','var50_raw')
REFERENCES=('augmented_equal','augmented_iu','augmented_joint_lw','entropy','historical_iu_pcr')
METHODS=CORE+EXTRA+tuple('ref_'+m for m in REFERENCES)
NAMES=dict(equal='Moment mean',iu='Moment IU-PCR',rbm_initial='Moment RBM before learning',
    rbm='Moment Gaussian RBM',b3_initial='Moment B3 before learning',b3='Moment B3',
    var15_raw='Varentropy Top-15',var15_equal='Varentropy contributions / mean',
    var15_iu='Varentropy contributions / IU-PCR',var50_raw='Varentropy Top-50',
    ref_augmented_equal='Saved direct probability mean (17 inputs)',
    ref_augmented_iu='Saved direct probability IU-PCR (17 inputs)',
    ref_augmented_joint_lw='Saved direct probability Joint shrinkage (17 inputs)',
    ref_entropy='Saved token entropy',ref_historical_iu_pcr='Historical IU-PCR')
PAIRS=(('iu','ref_historical_iu_pcr'),('rbm','iu'),('b3','rbm'),
       ('iu','equal'),('rbm','rbm_initial'),('b3','b3_initial'))


def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for attempt in range(40):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)


def aggregate_tokens(X):
    return np.array([top_mean(X[:,j],10) for j in range(X.shape[1])])


def inspect_sources(source,v2):
    audit_path=v2/'results/direct_probability_fusion_v2_selected_tail/DATA_AUDIT.json'
    audit=json.loads(audit_path.read_text(encoding='utf8'))
    audit_rows={r['cell']:r for r in audit['historical_24']}
    if set(audit_rows)!=set(INSCOPE):raise ValueError('historical audit roster differs')
    hashes={}
    for cell in INSCOPE:
        row=audit_rows[cell];p=source/row['artifact']
        print('[verify source]',cell,flush=True)
        digest=old.sha256_file(p)
        if digest!=row['sha256']:raise ValueError('source changed: '+cell)
        hashes[str(p)]=digest
    paths=[Path(__file__),ROOT/'spectral_utils/moment_rbm_fusion.py',
        ROOT/'spectral_utils/deem_b3_contract_ablation.py',ROOT/'spectral_utils/residual_graph_deem.py',
        ROOT/'spectral_utils/direct_probability_fusion.py',ROOT/'spectral_utils/direct_probability_fusion_v2.py',
        ROOT/'spectral_utils/varentropy_contribution_fusion.py',ROOT/'spectral_utils/upcr.py',
        ROOT/'spectral_utils/laplacian_upcr.py',ROOT/'scripts/run_direct_probability_fusion_v2.py',
        ROOT/'scripts/verify_global_moment_fusion_24.py',ROOT/'docs/experiments/GLOBAL_MOMENT_FUSION_24_V1.md',
        old.REFERENCE_24,old.HISTORICAL_BUNDLE,audit_path,
        v2/'results/direct_probability_fusion_v2_selected_tail/HISTORICAL_24.json',
        v2/'results/direct_probability_fusion_v2_selected_tail/HISTORICAL_24_SCORES.npz']
    for p in paths:hashes[str(p)]=old.sha256_file(p)
    from importlib.metadata import version
    return dict(schema='global-moment-fusion-24-v1',methods=list(METHODS),cells=list(INSCOPE),hashes=hashes,
        packages={p:version(p) for p in ('numpy','scipy','torch','scikit-learn')},
        fit_scope='All matched answers per cell, label-free transductive; not same-answer fitting',
        aggregation='Top10 token mean for each input coordinate; same historical row population/order/cropping',
        bootstrap='10000 within-cell problem-group draws; 24-cell macros; exploratory 95% intervals')


def one_cell(cell,v2,bundle,prior):
    started=time.perf_counter();path=old._historical_source(cell)
    candidates,groups=old._matched_historical_candidates(old.load_pickle(path),cell,len(bundle[cell+'__labels']))
    n=len(candidates)
    if n!=len(bundle[cell+'__labels']):raise ValueError('population mismatch: '+cell)
    X=[];C=[];v15=[];v50=[]
    for row in candidates:
        lp=old.logprob_matrix(old._topk_payload(row),k=50)
        X.append(aggregate_tokens(core.representation(lp,row['token_spilled_energies'])))
        c=contributions(lp,15);C.append(aggregate_tokens(c))
        v15.append(top_mean(c.sum(axis=1),10));v50.append(top_mean(contributions(lp,50).sum(axis=1),10))
    X=np.asarray(X);C=np.asarray(C)
    fits,failures,seconds=core.fit_matrix(X,'global24:'+cell)
    scores={m:fits[m]['score'] if m in fits else np.full(n,np.nan) for m in CORE}
    scores.update(var15_raw=np.asarray(v15),var50_raw=np.asarray(v50));params={};diagnostics={}
    for m,f in fits.items():
        diagnostics[m]=f['diagnostics']
        for key,val in f['state'].items():params[m+'::'+key]=val
    Z,keep,mean,scale=zscore_columns(C)
    for m in ('var15_equal','var15_iu'):
        tick=time.perf_counter()
        try:
            if Z.shape[1]<3:raise ValueError('too few varying contributions')
            if m=='var15_equal':w=np.full(Z.shape[1],1/Z.shape[1])
            else:
                fitted=upcr_fit(Z.T,**dict(IU_FIT_DEFAULTS))
                if fitted.abstained or fitted.used_simple_average:raise ValueError('IU abstention/fallback')
                w=fitted.w
            s=Z@w;flip=False
            if m=='var15_iu':s,flip,_=_orient(s,scores['var15_raw'])
            if not np.isfinite(s).all():raise ValueError('nonfinite contribution score')
            scores[m]=s;params[m+'::w']=(-w if flip else w)
            diagnostics[m]=dict(columns=np.flatnonzero(keep).tolist(),normalization_mean=mean.tolist(),normalization_scale=scale.tolist())
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
            scores[m]=np.full(n,np.nan);failures[m]=str(e)
        seconds[m]=time.perf_counter()-tick
    # Targets never enter the fitting API. Open them for alignment/evaluation here.
    labels=~np.asarray(bundle[cell+'__labels'],bool)
    np.testing.assert_array_equal(labels,[not bool(r.get('label',False)) for r in candidates])
    safe=cell.replace('-','_')
    np.testing.assert_array_equal(labels,prior[safe+'__label'])
    for m in REFERENCES:scores['ref_'+m]=prior[safe+'__'+m].copy()
    # Replay historical IU from its original feature bundle, beyond copying scores.
    V=np.asarray(bundle[cell+'__V']);names=tuple(str(x) for x in bundle[cell+'__pool'])
    mixed,_,_=old.dufs_liu_mixed_v2_from_bundle(V,names,np.asarray(bundle[cell+'__hand_signs']))
    A=np.asarray(mixed.T,float);historical=upcr_fit(A,**dict(IU_FIT_DEFAULTS))
    np.testing.assert_allclose(-(historical.w@A),scores['ref_historical_iu_pcr'],atol=1e-10,rtol=1e-10)
    metrics={m:float(roc_auc_score(labels,s)) if np.isfinite(s).all() else None for m,s in scores.items()}
    expected=old._historical_references()[cell]
    np.testing.assert_allclose(metrics['ref_historical_iu_pcr'],float(expected['auroc']),atol=1e-12,rtol=0)
    point=dict(cell=cell,domain=GROUP[cell],n=n,problems=len(set(groups)),auroc=metrics,
        failures=failures,diagnostics=diagnostics,fit_seconds=seconds,seconds=time.perf_counter()-started,
        coverage={m:float(np.isfinite(s).mean()) for m,s in scores.items()},review='PENDING')
    out=OUT/'cells'/cell;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'SCORES.npz',labels=labels,groups=groups,X=X,C=C,
        **{'score__'+m:s for m,s in scores.items()},**params)
    save_json(out/'METRICS.json',point)
    return point


def summarize(cells,complete):
    macro={}
    for m in METHODS:
        macro[m]={}
        for label,domain in (('all24',None),('qa9','QA'),('math15','math')):
            rows=[v for v in cells.values() if domain is None or v['domain']==domain]
            valid=[r['auroc'][m] for r in rows if r['auroc'][m] is not None]
            macro[m][label]=float(np.mean(valid)) if complete and len(valid)==len(rows) else None
    result=dict(status='POINTS_COMPLETE' if complete else 'RUNNING',n_cells=len(cells),expected_cells=24,
        fit_scope='Cell-local across matched answers; labels excluded from fit. Historical transductive development contract.',
        macro=macro,cells=cells)
    save_json(OUT/'METRICS.json',result)
    fields=['cell','domain','n']+[NAMES[m] for m in METHODS]
    with (OUT/'SUMMARY.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for cell,r in cells.items():w.writerow(dict(cell=cell,domain=r['domain'],n=r['n'],**{NAMES[m]:r['auroc'][m] for m in METHODS}))
    return result


def bootstrap(result):
    matrices={a+'_minus_'+b:[] for a,b in PAIRS};per_cell={}
    for i,cell in enumerate(INSCOPE):
        print('[bootstrap]',cell,flush=True)
        with np.load(OUT/'cells'/cell/'SCORES.npz') as z:
            per_cell[cell]={}
            for a,b in PAIRS:
                key=a+'_minus_'+b;left=z['score__'+a];right=z['score__'+b]
                if not np.isfinite(left).all() or not np.isfinite(right).all():
                    matrices[key].append(None);per_cell[cell][key]=None;continue
                d=old._paired_group_auc_bootstrap(z['labels'],left,right,z['groups'],draws=10000,seed=20260911+i)
                matrices[key].append(d);per_cell[cell][key]=np.percentile(d,[2.5,97.5]).tolist()
    contrasts={};rng=np.random.default_rng(20260911)
    for a,b in PAIRS:
        key=a+'_minus_'+b;contrasts[key]={}
        for scope,domain in (('all24',None),('qa9','QA'),('math15','math')):
            ix=[i for i,c in enumerate(INSCOPE) if domain is None or GROUP[c]==domain]
            if any(matrices[key][i] is None for i in ix):contrasts[key][scope]=None;continue
            M=np.stack([matrices[key][i] for i in ix]);samples=rng.integers(0,len(ix),(10000,len(ix)))
            draws=M[samples,np.arange(10000)[:,None]].mean(axis=1)
            contrasts[key][scope]=dict(delta=result['macro'][a][scope]-result['macro'][b][scope],
                hierarchical_group_cell_ci95=np.percentile(draws,[2.5,97.5]).tolist(),draws=10000)
    result.update(contrasts=contrasts,cell_intervals=per_cell,interval_scope='Exploratory, conditional on fixed transductive fits; no multiple-comparison winner claim',status='AWAITING_REVIEW')
    save_json(OUT/'METRICS.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--v2-root',type=Path,required=True)
    args=p.parse_args();old.configure_source_root(args.source_root);OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        manifest=inspect_sources(args.source_root,args.v2_root);mp=OUT/'MANIFEST.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('manifest differs; preserve existing result')
        save_json(mp,manifest)
        prior_path=args.v2_root/'results/direct_probability_fusion_v2_selected_tail/HISTORICAL_24_SCORES.npz'
        old_metrics=json.loads((args.v2_root/'results/direct_probability_fusion_v2_selected_tail/HISTORICAL_24.json').read_text())
        cells={}
        with np.load(old.HISTORICAL_BUNDLE,allow_pickle=True) as bundle,np.load(prior_path) as prior:
            for cell in INSCOPE:
                cached=OUT/'cells'/cell/'METRICS.json'
                print('[cell]',cell,flush=True)
                point=json.loads(cached.read_text()) if cached.exists() else one_cell(cell,args.v2_root,bundle,prior)
                for m in REFERENCES:
                    expected=old_metrics['cells'][cell]['historical_iu_pcr'] if m=='historical_iu_pcr' else old_metrics['cells'][cell]['auroc'][m]
                    np.testing.assert_allclose(point['auroc']['ref_'+m],expected,atol=1e-12,rtol=0)
                cells[cell]=point;summarize(cells,False)
                save_json(OUT/'RUN_STATE.json',dict(status='RUNNING',completed_cells=len(cells),expected_cells=24))
        result=summarize(cells,True);bootstrap(result)
        import subprocess
        subprocess.run([sys.executable,str(ROOT/'scripts/verify_global_moment_fusion_24.py'),'--result-dir',str(OUT)],check=True)
        result=json.loads((OUT/'METRICS.json').read_text());result['status']='COMPLETE_REVIEWED';save_json(OUT/'METRICS.json',result)
        save_json(OUT/'RUN_STATE.json',dict(status='COMPLETE',completed_cells=24,review='PASS'))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save_json(OUT/'RUN_STATE.json',dict(status='FAILED',error=f'{type(e).__name__}: {e}'));raise
