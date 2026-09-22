"""Independent factor-system, weight and full metric audit of fixed readout."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, runpy
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_joint_fixed_readout_v1 import OUT, STRESS, inputs, BANKS, frozen_fold
from scripts.audit_joint_selection_results import metrics
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.digitfree_broad50 import ANCHOR


def main():
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():
        assert sha(p) == h, p
    tests = runpy.run_path(str(ROOT/'tests/test_joint_fixed_readout.py')); passed=[]
    for name, fn in tests.items():
        if name.startswith('test_'): fn(); passed.append(name)
    data, base, uids = inputs()
    rowfold = np.repeat(data['folds'], np.diff(data['offsets']))
    with np.load(OUT/'SCORES.npz') as z: scores = {k:z[k] for k in z.files}
    gate = scores.pop('gate'); np.testing.assert_array_equal(gate, data['gate'])
    with np.load(STRESS/'SCORES.npz') as z:
        for name in z.files:
            if name != 'gate': np.testing.assert_array_equal(scores[name], z[name])
    result = json.loads((OUT/'RESULTS.json').read_text())
    native = {}; weight_error = 0.; score_error = 0.; diagnostics = []
    for bank in BANKS:
        x = base if bank == 'base' else augment(base, data['offsets'], uids, bank)
        for arm in ('full', 'auto'): native[bank+'__inverse_'+arm] = 0
        for outer in range(5):
            d = json.loads((OUT/f'{bank}_fold{outer}.json').read_text()); old=frozen_fold(bank, outer)
            held = rowfold == outer
            for arm, a in d['arms'].items():
                key = bank+'__inverse_'+arm; w = np.asarray(a['weights'])
                assert a['valid'] == old['valid']['joint_'+arm]
                if not a['valid']:
                    np.testing.assert_array_equal(w, np.eye(x.shape[1])[ANCHOR])
                else:
                    native[key] += int(np.sum(data['folds'] == outer))
                    active = np.asarray(a['active']); v = np.asarray(a['global_loading']); u = np.asarray(a['group_loading'])
                    labels = np.asarray(a['labels']); observed = np.asarray(a['observed_covariance']); c = np.asarray(a['model_covariance'])
                    expected_active = np.arange(x.shape[1]) if arm == 'full' else old['selection']['automatic']['active']
                    np.testing.assert_array_equal(active, expected_active)
                    np.testing.assert_allclose(np.cov(x[~held][:,active], rowvar=False, ddof=1), observed, atol=1e-12)
                    component = np.outer(v,v)+(labels[:,None] == labels[None,:])*np.outer(u,u)
                    model = component+np.diag(np.maximum(np.diag(observed)-np.diag(component),0))
                    np.testing.assert_allclose(c, model, atol=1e-12)
                    ev, q = np.linalg.eigh((c+c.T)/2)
                    psd = (q*np.maximum(ev,0))@q.T; psd=(psd+psd.T)/2
                    ridge = max(0., (max(ev)-1000*max(min(ev),0))/999, max(ev)*1e-10)
                    np.testing.assert_allclose(ridge,a['inverse']['ridge'],atol=1e-12)
                    expected = np.zeros(len(w)); expected[active] = np.linalg.solve(psd+ridge*np.eye(len(v)),v)
                    rho = spearmanr(x[~held]@expected,x[~held,ANCHOR]).statistic
                    if np.isfinite(rho) and rho < 0: expected *= -1
                    expected /= np.abs(expected).sum()
                    weight_error=max(weight_error,float(np.max(np.abs(expected-w))))
                    np.testing.assert_allclose(w,expected,atol=1e-9,rtol=1e-8)
                    assert a['inverse']['condition_after'] <= 1000*(1+1e-8)
                    assert a['hierarchical_replay_max_error'] < 1e-10
                    diagnostics.append(dict(bank=bank,outer=outer,support=arm,
                        added_absolute_weight=a['added_absolute_weight'],
                        diagonal_clipped_count=a['diagonal_audit']['clipped_count'],
                        diagonal_clipped_mass=a['diagonal_audit']['clipped_mass'],
                        ridge=a['inverse']['ridge']))
                expected_score = x[held]@w
                score_error=max(score_error,float(np.max(np.abs(expected_score-scores[key][held]))))
                np.testing.assert_allclose(expected_score,scores[key][held],atol=1e-12,rtol=1e-12)
    assert native == result['native_answers']
    for name,score in scores.items():
        assert np.isfinite(score).all(); pb,within,n=metrics(score,gate,data)
        m=result['metrics'][name]
        np.testing.assert_allclose([pb,within],[m['pb'],m['within']],atol=1e-12,rtol=0)
        assert n == m['within_n'] == 6030
    for b, valid in result['practical_preservation'].items():
        ci=result['primary_contrasts'][b+'__inverse_auto minus base__inverse_auto']
        assert valid == (native[b+'__inverse_auto']==13769 and ci['pb']['low']>-.01 and ci['within']['low']>-.002)
    audit=dict(status='PASS',answers=len(uids),steps=len(base),metrics_checked=len(scores),
        tests=passed,weight_replay_max_error=weight_error,score_replay_max_error=score_error,
        native_answers=native,score_sha256=sha(OUT/'SCORES.npz'),
        checks=['frozen hashes and controls','same selected supports and historical validity',
            'observed covariance and factor covariance replay','independent eigensolve, ridge and orientation',
            'all fold scores and native failure accounting','independent pairwise AUC and PB counts',
            'unchanged gate and practical preservation decision'])
    dump(OUT/'AUDIT.json',audit); dump(OUT/'READOUT_DIAGNOSTIC.json',dict(folds=diagnostics))
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__': main()
