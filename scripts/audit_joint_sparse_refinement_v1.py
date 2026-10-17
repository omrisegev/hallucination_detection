"""Verify full refinement, frozen controls and decoding to original features."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,runpy
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_sparse_refinement_v1 import OUT,PREVIOUS,inputs,BANKS
from scripts.audit_joint_selection_results import metrics
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.joint_sparse_membership import alias_coordinates
from spectral_utils.digitfree_broad50 import ANCHOR


def main():
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():assert sha(p)==h,p
    test=runpy.run_path(str(ROOT/'tests/test_joint_sparse_refinement.py'))
    test['test_refinement_is_identical_after_alias_expansion']()
    data,base,uids=inputs();rowfold=np.repeat(data['folds'],np.diff(data['offsets']))
    with np.load(OUT/'SCORES.npz') as z:scores={k:z[k] for k in z.files}
    gate=scores.pop('gate');np.testing.assert_array_equal(gate,data['gate'])
    with np.load(PREVIOUS/'SCORES.npz') as z:
        for k in z.files:
            if k!='gate':np.testing.assert_array_equal(scores[k],z[k])
    result=json.loads((OUT/'RESULTS.json').read_text());native={};cache={};weight_error=0.;score_error=0.;alias_error=0.
    for bank in BANKS:
        x=base if bank=='base' else augment(base,data['offsets'],uids,bank);native[bank+'__refined']=0
        for outer in range(5):
            d=json.loads((OUT/f'{bank}_fold{outer}.json').read_text());held=rowfold==outer
            parent=json.loads((PREVIOUS/f'{bank}_fold{outer}.json').read_text());assert d['aliases']==parent['aliases']
            z=alias_coordinates(x,d['aliases'])
            if d['valid']:
                assert parent['fit']['valid'];np.testing.assert_array_equal(d['initial_active'],parent['fit']['active'])
                fit=d['fit'];key=(outer,d['training_digest'])
                if key in cache:assert fit==cache[key]
                else:cache[key]=fit
                selection=fit['selection'];a=selection['automatic'];index=selection['automatic_index']
                assert a==selection['path'][index] and a['minimum_retention']>=.95
                if index+1<len(selection['path']):assert selection['path'][index+1]['minimum_retention']<.95
                assert min(a['group_sizes'])>=2
                initial=np.asarray(d['initial_active']);active=np.asarray(d['active'])
                np.testing.assert_array_equal(active,initial[np.asarray(a['active'])])
                assert fit['converged_starts']>=4 and fit['multistart']['status']=='PASS'
                assert fit['jacobian']['full_global_rank'] and fit['jacobian']['condition_number']<=1e8
                cov=np.asarray(fit['observed_covariance']);v=np.asarray(fit['global_loading']);u=np.asarray(fit['group_loading'])
                labels=np.asarray(fit['labels']);c=np.asarray(fit['model_covariance'])
                np.testing.assert_allclose(cov,np.cov(z[~held][:,active],rowvar=False),atol=1e-12)
                components=np.outer(v,v)+(labels[:,None]==labels[None,:])*np.outer(u,u)
                np.testing.assert_allclose(c,components+np.diag(np.maximum(np.diag(cov)-np.diag(components),0)),atol=1e-12)
                local=np.zeros(len(v))
                for g in np.unique(labels):
                    vg=np.where(labels==g,v,0.);var=float(vg@c@vg)
                    local[labels==g]=v[labels==g]*(vg@vg)/var if var>1e-14 else 0.
                expected=np.zeros(z.shape[1]);expected[active]=local
                rho=spearmanr(z[~held]@expected,x[~held,ANCHOR]).statistic
                if np.isfinite(rho) and rho<0:expected*=-1
                expected/=np.abs(expected).sum();w=np.asarray(d['canonical_weights'])
                weight_error=max(weight_error,float(np.max(np.abs(expected-w))))
                np.testing.assert_allclose(expected,w,atol=1e-12,rtol=1e-12)
                reconstructed=z[held]@w;expanded=x[held]@np.asarray(d['expanded_weights'])
                alias_error=max(alias_error,float(np.max(np.abs(expanded-reconstructed))))
                np.testing.assert_allclose(expanded,reconstructed,atol=1e-12,rtol=1e-12)
                native[bank+'__refined']+=int(np.sum(data['folds']==outer))
            else:reconstructed=x[held,ANCHOR]
            actual=scores[bank+'__refined'][held]
            score_error=max(score_error,float(np.max(np.abs(actual-reconstructed))))
            np.testing.assert_array_equal(reconstructed,actual)
    assert native==result['native_answers']
    np.testing.assert_array_equal(scores['base__refined'],scores['duplicates__refined'])
    np.testing.assert_allclose(scores['base__refined'],scores['noise__refined'],atol=2e-14,rtol=0)
    for name,s in scores.items():
        assert np.isfinite(s).all();pb,within,n=metrics(s,gate,data);m=result['metrics'][name]
        np.testing.assert_allclose([pb,within],[m['pb'],m['within']],atol=1e-12,rtol=0);assert n==m['within_n']==6030
    for b,preserved in result['practical_preservation'].items():
        ci=result['primary_contrasts'][b+'__refined minus base__refined']
        assert preserved==(native[b+'__refined']==13769 and ci['pb']['low']>-.01 and ci['within']['low']>-.002)
    audit=dict(status='PASS',answers=len(uids),steps=len(base),metrics_checked=len(scores),
        tests=['test_refinement_is_identical_after_alias_expansion'],weight_replay_max_error=weight_error,
        score_replay_max_error=score_error,alias_expansion_max_error=alias_error,native_answers=native,
        score_sha256=sha(OUT/'SCORES.npz'),checks=['frozen hashes and controls','inherited sparse/alias membership',
        'unchanged95% first crossing and support','checked final fits and factor covariance',
        'independent group weights and orientation','original-coordinate score decoding',
        'full independent PB and within-AUC replay','copy identity and noise roundoff-only change'])
    dump(OUT/'AUDIT.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
