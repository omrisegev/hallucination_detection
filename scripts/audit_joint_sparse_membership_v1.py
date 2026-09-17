"""Independent sparse membership, null calibration and full score audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,runpy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_sparse_membership_v1 import OUT,PREVIOUS,inputs,BANKS
from scripts.audit_joint_selection_results import metrics
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.joint_sparse_membership import alias_coordinates,initial_loadings
from spectral_utils.digitfree_broad50 import ANCHOR


def main():
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():assert sha(p)==h,p
    test=runpy.run_path(str(ROOT/'tests/test_joint_sparse_membership.py'));passed=[]
    for name,fn in test.items():
        if name.startswith('test_'):fn();passed.append(name)
    data,base,uids=inputs();sizes=np.diff(data['offsets']);rowfold=np.repeat(data['folds'],sizes)
    sources=np.repeat(data['groups'],sizes)
    with np.load(OUT/'SCORES.npz') as z:scores={k:z[k] for k in z.files}
    gate=scores.pop('gate');np.testing.assert_array_equal(gate,data['gate'])
    with np.load(PREVIOUS/'SCORES.npz') as z:
        for k in z.files:
            if k!='gate':np.testing.assert_array_equal(scores[k],z[k])
    result=json.loads((OUT/'RESULTS.json').read_text());native={};cache={};count_rounds=0;score_error=0.;expanded_error=0.
    for bank in BANKS:
        x=base if bank=='base' else augment(base,data['offsets'],uids,bank);native[bank+'__sparse']=0
        for outer in range(5):
            d=json.loads((OUT/f'{bank}_fold{outer}.json').read_text());held=rowfold==outer;groups=d['aliases']
            assert sorted(i for g in groups for i in g)==list(range(x.shape[1]))
            for group in groups:
                for i in group:np.testing.assert_array_equal(x[~held,i],x[~held,group[0]])
            z=alias_coordinates(x,groups);fit=d['fit'];key=(outer,d['training_digest'])
            if key in cache:assert fit==cache[key]
            elif fit['valid']:
                active=np.arange(z.shape[1])
                for r in fit['rounds']:
                    np.testing.assert_array_equal(active,r['active_before'])
                    # Preserve reduction order for the iterative loading
                    # initializer; the signed-observation null is independent.
                    cov=np.cov(np.ascontiguousarray(z[~held][:,active]).T)
                    cal=r['calibration'];np.testing.assert_allclose(cal['penalty'],max(np.quantile(cal['null_maxima'],.95),1e-8))
                    # Independently reconstruct first randomized covariance from
                    # actual signed observations, bypassing source Gram sums.
                    _,source_index=np.unique(sources[~held],return_inverse=True)
                    rng=np.random.default_rng(cal['seed']);signs=rng.choice(np.array([-1.,1.]),size=(source_index.max()+1,len(active)))
                    null=np.cov(z[~held][:,active]*signs[source_index],rowvar=False);np.fill_diagonal(null,0.)
                    v,u,mask=initial_loadings(cov,np.asarray(r['discovery']['labels']),start=0,seed=cal['seed'])
                    maximum=max(np.max(np.abs(null@v)),np.max(np.abs((null*mask)@u)))
                    np.testing.assert_allclose(maximum,cal['null_maxima'][0],atol=1e-11,rtol=1e-10)
                    converged=[]
                    for s in r['sparse']['starts']:
                        assert np.all(np.diff(s['objective_trace'])<=1e-9)
                        v=np.asarray(s['v']);u=np.asarray(s['u'])
                        assert np.all(v*v+u*u<=np.diag(cov)+1e-12)
                        residual=cov-np.outer(v,v)-mask*np.outer(u,u)
                        value=.5*np.square(residual[np.triu_indices(len(v),1)]).sum()+cal['penalty']*(np.abs(v).sum()+np.abs(u).sum())
                        np.testing.assert_allclose(value,s['objective'],atol=1e-10)
                        np.testing.assert_array_equal(np.flatnonzero((v!=0)|(u!=0)),s['support'])
                        if s['converged']:converged.append(s['support'])
                    union=np.unique(np.concatenate(converged));np.testing.assert_array_equal(active[union],r['active_after'])
                    active=active[union];count_rounds+=1
                np.testing.assert_array_equal(active,fit['active']);cache[key]=fit
            if fit['valid']:
                assert fit['converged_starts']>=4 and fit['multistart']['status']=='PASS'
                assert fit['jacobian']['full_global_rank'] and fit['jacobian']['condition_number']<=1e8
                active=np.asarray(fit['active']);v=np.asarray(fit['global_loading']);u=np.asarray(fit['group_loading'])
                labels=np.asarray(fit['final_labels']);cov=np.asarray(fit['observed_covariance'])
                np.testing.assert_allclose(cov,np.cov(z[~held][:,active],rowvar=False),atol=1e-12)
                component=np.outer(v,v)+(labels[:,None]==labels[None,:])*np.outer(u,u)
                np.testing.assert_allclose(component+np.diag(np.maximum(np.diag(cov)-np.diag(component),0)),fit['model_covariance'],atol=1e-12)
                weights=np.asarray(d['canonical_weights']);assert not np.any(weights[np.setdiff1d(np.arange(len(weights)),active)])
                expected=z[held]@weights;native[bank+'__sparse']+=int(np.sum(data['folds']==outer))
                expanded=x[held]@np.asarray(d['expanded_weights'])
                expanded_error=max(expanded_error,float(np.max(np.abs(expanded-expected))))
                np.testing.assert_allclose(expanded,expected,atol=1e-12,rtol=1e-12)
            else:expected=x[held,ANCHOR]
            actual=scores[bank+'__sparse'][held];score_error=max(score_error,float(np.max(np.abs(actual-expected))))
            np.testing.assert_array_equal(expected,actual)
    assert native==result['native_answers']
    np.testing.assert_array_equal(scores['base__sparse'],scores['duplicates__sparse'])
    for name,score in scores.items():
        assert np.isfinite(score).all();pb,within,n=metrics(score,gate,data);m=result['metrics'][name]
        np.testing.assert_allclose([pb,within],[m['pb'],m['within']],atol=1e-12,rtol=0);assert n==m['within_n']==6030
    audit=dict(status='PASS',answers=len(uids),steps=len(base),metrics_checked=len(scores),
        unique_sparse_rounds_checked=count_rounds,tests=passed,score_replay_max_error=score_error,
        alias_expansion_max_error=expanded_error,native_answers=native,score_sha256=sha(OUT/'SCORES.npz'),
        checks=['frozen hashes and controls','training-only alias equality and model sharing',
        'independent signed-observation null covariance','monotone sparse objectives and variance constraints',
        'conservative union support and final native checks','factor covariance and full weight replay',
        'independent PB counts and pairwise within AUC','exact-copy score identity'])
    dump(OUT/'AUDIT.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
