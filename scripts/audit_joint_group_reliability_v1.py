"""Independent replay of model-based outer weighting and complete metrics."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,runpy
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_group_reliability_v1 import OUT,PREVIOUS,inputs,BANKS
from scripts.audit_joint_selection_results import metrics
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.digitfree_broad50 import ANCHOR


def main():
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():assert sha(p)==h,p
    tests=runpy.run_path(str(ROOT/'tests/test_joint_group_reliability.py'));passed=[]
    for name,fn in tests.items():
        if name.startswith('test_'):fn();passed.append(name)
    data,base,uids=inputs();rowfold=np.repeat(data['folds'],np.diff(data['offsets']))
    with np.load(OUT/'SCORES.npz') as z:scores={k:z[k] for k in z.files}
    gate=scores.pop('gate');np.testing.assert_array_equal(gate,data['gate'])
    with np.load(PREVIOUS/'SCORES.npz') as z:
        for name in z.files:
            if name!='gate':np.testing.assert_array_equal(scores[name],z[name])
    result=json.loads((OUT/'RESULTS.json').read_text());native={};weight_error=0.;score_error=0.;diagnostics=[]
    for bank in BANKS:
        x=base if bank=='base' else augment(base,data['offsets'],uids,bank)
        for arm in ('full','auto'):native[bank+'__reliability_'+arm]=0
        for outer in range(5):
            old=json.loads((PREVIOUS/f'{bank}_fold{outer}.json').read_text())
            d=json.loads((OUT/f'{bank}_fold{outer}.json').read_text());held=rowfold==outer
            for arm,a in d['arms'].items():
                frozen=old['arms'][arm];assert a['valid']==frozen['valid'];w=np.asarray(a['weights'])
                key=bank+'__reliability_'+arm
                if not a['valid']:np.testing.assert_array_equal(w,np.eye(x.shape[1])[ANCHOR])
                else:
                    active=np.asarray(frozen['active']);v=np.asarray(frozen['global_loading'])
                    labels=np.asarray(frozen['labels']);c=np.asarray(frozen['model_covariance'])
                    np.testing.assert_array_equal(active,a['active']);local=np.zeros(len(v))
                    for g in np.unique(labels):
                        mask=labels==g;vv=np.where(mask,v,0.);var=float(vv@c@vv)
                        loading=float(vv@vv)
                        local[mask]=v[mask]*loading/var if var>1e-14 else 0.
                    expected=np.zeros(len(w));expected[active]=local
                    rho=spearmanr(x[~held]@expected,x[~held,ANCHOR]).statistic
                    if np.isfinite(rho) and rho<0:expected*=-1
                    expected/=np.abs(expected).sum()
                    weight_error=max(weight_error,float(np.max(np.abs(expected-w))))
                    np.testing.assert_allclose(expected,w,atol=1e-12,rtol=1e-12)
                    native[key]+=int(np.sum(data['folds']==outer))
                    diagnostics.append(dict(bank=bank,outer=outer,support=arm,
                        added_absolute_weight=a['added_absolute_weight'],groups=a['readout']['groups']))
                s=x[held]@w;score_error=max(score_error,float(np.max(np.abs(s-scores[key][held]))))
                np.testing.assert_allclose(s,scores[key][held],atol=1e-12,rtol=1e-12)
    assert native==result['native_answers']
    for name,s in scores.items():
        assert np.isfinite(s).all();pb,within,n=metrics(s,gate,data);m=result['metrics'][name]
        np.testing.assert_allclose([pb,within],[m['pb'],m['within']],atol=1e-12,rtol=0)
        assert n==m['within_n']==6030
    for b,preserved in result['practical_preservation'].items():
        ci=result['primary_contrasts'][b+'__reliability_auto minus base__reliability_auto']
        assert preserved==(native[b+'__reliability_auto']==13769 and ci['pb']['low']>-.01 and ci['within']['low']>-.002)
    audit=dict(status='PASS',answers=len(uids),steps=len(base),metrics_checked=len(scores),tests=passed,
        weight_replay_max_error=weight_error,score_replay_max_error=score_error,native_answers=native,
        score_sha256=sha(OUT/'SCORES.npz'),checks=['frozen hashes and controls',
        'unchanged supports, loadings and failures','independent group weighting formula and sign',
        'fold score replay','independent pairwise AUROC and PB counts','same gate and preservation rule'])
    dump(OUT/'AUDIT.json',audit);dump(OUT/'GROUP_DIAGNOSTIC.json',dict(folds=diagnostics));print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
