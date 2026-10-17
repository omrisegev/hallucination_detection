"""Independent score, membership, calibration and covariance audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,runpy
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_structured_stress_v1 import OUT,PREVIOUS,inputs,KINDS,training_inputs
from scripts.audit_joint_selection_results import metrics
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump
from spectral_utils.joint_structured_stress import structured_augmentation
from spectral_utils.joint_sparse_membership import alias_coordinates,initial_loadings
from spectral_utils.digitfree_broad50 import ANCHOR


def factor_information(c,a):
    regularized=(1-1e-4)*c+1e-4*np.diag(np.maximum(np.diag(c),1e-12))
    return np.sum(a*np.linalg.solve(regularized,a),axis=0)


def audit_model(model,x,offsets,sources):
    for ids in model['aliases']:
        for i in ids:np.testing.assert_array_equal(x[:,i],x[:,ids[0]])
    z=alias_coordinates(x,model['aliases']);membership=model.get('membership');rounds=0
    if membership:
        active=np.arange(z.shape[1]);row_sources=np.repeat(sources,np.diff(offsets))
        _,source_index=np.unique(row_sources,return_inverse=True)
        for r in membership['rounds']:
            np.testing.assert_array_equal(active,r['active_before']);cov=np.cov(np.ascontiguousarray(z[:,active]).T)
            cal=r['calibration'];np.testing.assert_allclose(cal['penalty'],max(np.quantile(cal['null_maxima'],.95),1e-8))
            rng=np.random.default_rng(cal['seed']);signs=rng.choice(np.array([-1.,1.]),size=(source_index.max()+1,len(active)))
            null=np.cov(z[:,active]*signs[source_index],rowvar=False);np.fill_diagonal(null,0.)
            v,u,mask=initial_loadings(cov,np.asarray(r['discovery']['labels']),start=0,seed=cal['seed'])
            maximum=max(np.max(np.abs(null@v)),np.max(np.abs((null*mask)@u)))
            np.testing.assert_allclose(maximum,cal['null_maxima'][0],atol=1e-11,rtol=1e-10)
            converged=[]
            for s in r['sparse']['starts']:
                assert np.all(np.diff(s['objective_trace'])<=1e-9)
                v=np.asarray(s['v']);u=np.asarray(s['u']);assert np.all(v*v+u*u<=np.diag(cov)+1e-12)
                residual=cov-np.outer(v,v)-mask*np.outer(u,u)
                loss=.5*np.square(residual[np.triu_indices(len(v),1)]).sum()+cal['penalty']*(np.abs(v).sum()+np.abs(u).sum())
                np.testing.assert_allclose(loss,s['objective'],atol=1e-10)
                np.testing.assert_array_equal(np.flatnonzero((v!=0)|(u!=0)),s['support'])
                if s['converged']:converged.append(s['support'])
            active=active[np.unique(np.concatenate(converged))];np.testing.assert_array_equal(active,r['active_after']);rounds+=1
        np.testing.assert_array_equal(active,membership['active'])
    if not model['valid']:return 0.,rounds
    fit=model['refinement'];assert fit['converged_starts']>=4 and fit['multistart']['status']=='PASS'
    assert fit['jacobian']['full_global_rank'] and fit['jacobian']['condition_number']<=1e8
    selection=fit['selection'];chosen=selection['automatic'];index=selection['automatic_index']
    assert chosen==selection['path'][index] and chosen['minimum_retention']>=.95
    if index+1<len(selection['path']):assert selection['path'][index+1]['minimum_retention']<.95
    initial=np.asarray(model['initial_active']);active=np.asarray(model['active'])
    np.testing.assert_array_equal(initial,membership['active']);np.testing.assert_array_equal(active,initial[np.asarray(chosen['active'])])
    c0=np.asarray(membership['observed_covariance']);v0=np.asarray(membership['global_loading']);u0=np.asarray(membership['group_loading'])
    labels0=np.asarray(membership['final_labels']);a0=np.zeros((len(v0),1+len(np.unique(labels0))));a0[:,0]=v0
    for j,g in enumerate(np.unique(labels0)):a0[labels0==g,j+1]=u0[labels0==g]
    info=factor_information(c0,a0);np.testing.assert_allclose(info,selection['initial_factor_information'],atol=1e-8,rtol=1e-8)
    for step in selection['path']:
        ids=np.asarray(step['active']);after=factor_information(c0[np.ix_(ids,ids)],a0[ids])
        ratios=np.divide(after,info,out=np.ones_like(after),where=info>1e-8)
        np.testing.assert_allclose(ratios,step['retention'],atol=1e-8,rtol=1e-8)
    cov=np.asarray(fit['observed_covariance']);v=np.asarray(fit['global_loading']);u=np.asarray(fit['group_loading']);labels=np.asarray(fit['labels'])
    np.testing.assert_allclose(cov,np.cov(z[:,active],rowvar=False),atol=1e-12)
    component=np.outer(v,v)+(labels[:,None]==labels[None,:])*np.outer(u,u)
    c=component+np.diag(np.maximum(np.diag(cov)-np.diag(component),0));np.testing.assert_allclose(c,fit['model_covariance'],atol=1e-12)
    local=np.zeros(len(v))
    for g in np.unique(labels):
        vg=np.where(labels==g,v,0.);var=float(vg@c@vg)
        local[labels==g]=v[labels==g]*(vg@vg)/var if var>1e-14 else 0.
    expected=np.zeros(z.shape[1]);expected[active]=local
    rho=spearmanr(z@expected,x[:,ANCHOR]).statistic
    if np.isfinite(rho) and rho<0:expected*=-1
    expected/=np.abs(expected).sum();w=np.asarray(model['canonical_weights'])
    np.testing.assert_allclose(expected,w,atol=1e-12,rtol=1e-12)
    return float(np.max(np.abs(expected-w))),rounds


def checkpoint_checks(data,base,uids,*,require_complete):
    path=OUT/'CHECKPOINT_AUDIT.json';version=sha(Path(__file__));manifest=sha(OUT/'MANIFEST.json')
    cache=json.loads(path.read_text()) if path.exists() else {}
    if cache.get('auditor_sha256')!=version or cache.get('manifest_sha256')!=manifest:cache={}
    checked=cache.get('models',{})
    for bank in ('base_replay',*KINDS):
        x=base if bank=='base_replay' else structured_augmentation(base,data['offsets'],uids,bank)
        for outer in range(5):
            source=OUT/f'{bank}_fold{outer}.json'
            if not source.exists():
                if require_complete:raise AssertionError(str(source))
                continue
            try:d=json.loads(source.read_text())
            except json.JSONDecodeError:
                if require_complete:raise
                continue
            digest=sha(source);key=source.name
            if checked.get(key,{}).get('sha256')==digest:continue
            _,args=training_inputs(data,x,outer)
            error,rounds=audit_model(d['model'],args[0],args[1],args[2])
            checked[key]=dict(sha256=digest,weight_error=error,rounds=rounds)
            dump(path,dict(status='COMPLETED_CHECKPOINTS_PASS',auditor_sha256=version,
                manifest_sha256=manifest,models=checked))
            print('CHECKPOINT_AUDIT_PASS',key,flush=True)
    if require_complete:assert len(checked)==15
    return checked


def main():
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():assert sha(p)==h,p
    tests=runpy.run_path(str(ROOT/'tests/test_joint_structured_stress.py'));passed=[]
    for name,fn in tests.items():
        if name.startswith('test_'):fn();passed.append(name)
    data,base,uids=inputs();checked=checkpoint_checks(data,base,uids,require_complete=True)
    with np.load(OUT/'SCORES.npz') as z:scores={k:z[k] for k in z.files}
    gate=scores.pop('gate');np.testing.assert_array_equal(gate,data['gate'])
    with np.load(PREVIOUS/'SCORES.npz') as z:
        for k in scores:
            if k in z.files:np.testing.assert_array_equal(scores[k],z[k])
    result=json.loads((OUT/'RESULTS.json').read_text());native={};diagnostics={};weight_error=0.;score_error=0.;rounds=0
    for outer in range(5):
        d=json.loads((OUT/f'base_replay_fold{outer}.json').read_text());rows,args=training_inputs(data,base,outer)
        record=checked[f'base_replay_fold{outer}.json'];weight_error=max(weight_error,record['weight_error']);rounds+=record['rounds']
        actual=alias_coordinates(base[~rows],d['model']['aliases'])@np.asarray(d['model']['canonical_weights'])
        np.testing.assert_allclose(actual,scores['base__refined'][~rows],atol=1e-11,rtol=1e-11)
    for kind in KINDS:
        x=structured_augmentation(base,data['offsets'],uids,kind);np.testing.assert_array_equal(x[:,:51],base)
        corr=np.corrcoef(x[:,51:],rowvar=False)
        diagnostics[kind]=dict(median_added_pair_correlation=float(np.median(corr[np.triu_indices(15,1)])),
            minimum_parent_copy_correlation=float(min(np.corrcoef(x[:,i],x[:,51+i])[0,1] for i in range(15))))
        for arm in ('joint','continuous','equal'):native[kind+'__'+arm]=0
        for outer in range(5):
            d=json.loads((OUT/f'{kind}_fold{outer}.json').read_text());model=d['model'];rows,args=training_inputs(data,x,outer)
            record=checked[f'{kind}_fold{outer}.json'];weight_error=max(weight_error,record['weight_error']);rounds+=record['rounds']
            actual=alias_coordinates(x[~rows],model['aliases'])@np.asarray(model['canonical_weights']) if model['valid'] else x[~rows,ANCHOR]
            if model['valid']:np.testing.assert_allclose(actual,x[~rows]@np.asarray(model['expanded_weights']),atol=1e-12,rtol=1e-12)
            for arm in ('joint','continuous','equal'):
                predicted=actual if arm=='joint' else x[~rows]@np.asarray(d['weights'][arm])
                error=float(np.max(np.abs(predicted-scores[kind+'__'+arm][~rows])));score_error=max(score_error,error)
                np.testing.assert_allclose(predicted,scores[kind+'__'+arm][~rows],atol=1e-12,rtol=1e-12)
                if d['valid'][arm]:native[kind+'__'+arm]+=int(np.sum(data['folds']==outer))
    assert native==result['native_answers']
    for name,s in scores.items():
        assert np.isfinite(s).all();pb,within,n=metrics(s,gate,data);m=result['metrics'][name]
        np.testing.assert_allclose([pb,within],[m['pb'],m['within']],atol=1e-12,rtol=0);assert n==m['within_n']==6030
    for k,preserved in result['practical_preservation'].items():
        ci=result['primary_contrasts'][k+'__joint minus base__refined']
        assert all(v['confidence']==.9875 and v['draws']==10000 for v in ci.values())
        assert preserved==(native[k+'__joint']==13769 and ci['pb']['low']>-.01 and ci['within']['low']>-.002)
    audit=dict(status='PASS',answers=len(uids),steps=len(base),tests=passed,metrics_checked=len(scores),
        sparse_rounds_checked=rounds,weight_replay_max_error=weight_error,score_replay_max_error=score_error,
        native_answers=native,score_sha256=sha(OUT/'SCORES.npz'),checks=['frozen dependency hashes',
        'all-five baseline API replay','sparse objectives and independent null covariance',
        'complete95% retention path and final factor covariance','independent group weighting',
        'all held scores and failures','independent PB counts and pairwise AUROC','fixed gate and preservation margins'])
    dump(OUT/'AUDIT.json',audit);dump(OUT/'PERTURBATION_DIAGNOSTIC.json',diagnostics);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    if '--checkpoints-only' in sys.argv:
        data,base,uids=inputs();checked=checkpoint_checks(data,base,uids,require_complete=False)
        print('COMPLETED_CHECKPOINTS_VERIFIED',len(checked),flush=True)
    else:main()
