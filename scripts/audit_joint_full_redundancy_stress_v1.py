"""Independent metric replay, perturbation integrity and selection audit."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, runpy
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_joint_full_redundancy_stress_v1 import inputs, OUT, PREVIOUS, ARMS
from scripts.audit_joint_selection_results import metrics
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.joint_lsml import covariance_matrix
from spectral_utils.joint_pair_jacobian import fit_joint_pairs_checked


def main():
    for path, digest in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():
        assert sha(path) == digest, path
    tests = runpy.run_path(str(ROOT/'tests/test_joint_redundancy_stress.py'))
    passed = []
    for name, fn in tests.items():
        if name.startswith('test_'):
            fn(); passed.append(name)
    data, base, uids = inputs()
    assert len(set(uids)) == len(uids)
    with np.load(OUT/'SCORES.npz') as z:
        scores = {k: z[k] for k in z.files}
    gate = scores.pop('gate')
    np.testing.assert_array_equal(gate, data['gate'])
    with np.load(PREVIOUS/'SCORES.npz') as z:
        for arm in ARMS:
            np.testing.assert_array_equal(scores['base__'+arm], z['B51_bocpd__'+arm])
        for name in ('innovation5', 'historical_bocpd'):
            np.testing.assert_array_equal(scores[name], z[name])
    result = json.loads((OUT/'RESULTS.json').read_text())
    rowfold = np.repeat(data['folds'], np.diff(data['offsets']))
    native = {}; selection = []; structure = []; failures = []; maxerror = 0.
    for kind in ('duplicates', 'noise'):
        x = augment(base, data['offsets'], uids, kind)
        np.testing.assert_array_equal(x[:, :51], base)
        if kind == 'duplicates':
            np.testing.assert_array_equal(x[:, 51:], base[:, :15])
        for arm in ARMS:
            native[kind+'__'+arm] = 0
        for outer in range(5):
            d = json.loads((OUT/f'{kind}_fold{outer}.json').read_text())
            reference = json.loads((PREVIOUS/f'B51_bocpd_fold{outer}.json').read_text())
            held = rowfold == outer
            for arm, weights in d['weights'].items():
                expected = x[held] @ np.asarray(weights)
                actual = scores[kind+'__'+arm][held]
                maxerror = max(maxerror, float(np.max(np.abs(expected-actual))))
                np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=1e-12)
                if d['valid'][arm]:
                    native[kind+'__'+arm] += int(np.sum(data['folds'] == outer))
            fit = d.get('selection')
            if fit:
                a = fit['automatic']; index = fit['automatic_index']
                assert a == fit['path'][index]
                assert a['minimum_retention'] >= .95
                assert min(a['group_sizes']) >= 2
                if index+1 < len(fit['path']):
                    assert fit['path'][index+1]['minimum_retention'] < .95
                w = np.asarray(d['weights']['joint_auto'])
                assert not np.any(w[np.setdiff1d(np.arange(66), a['active'])])
                assert len(a['active']) == result['selected_counts'][kind][outer]
                selection.append(dict(kind=kind, fold=outer, kept=len(a['active']),
                    added=d['selected_added_features'], retention=a['minimum_retention']))
            labels = np.asarray(d['discovery']['labels'])
            diagnostic = dict(kind=kind, fold=outer, group_sizes=d['discovery']['group_sizes'],
                added_per_group=[int(np.sum(labels[51:] == g)) for g in np.unique(labels)],
                selected_added=d.get('selected_added_features'),
                both_copies_retained=d.get('both_copies_retained'),
                initial_factor_information=fit.get('initial_factor_information') if fit else None,
                last_deletion=fit['deletion_audit'][-1] if fit and fit['deletion_audit'] else None,
                arms={})
            for arm in ('joint_full', 'joint_auto'):
                w = np.asarray(d['weights'][arm]); effective = w[:51].copy()
                if kind == 'duplicates':
                    effective[:15] += w[51:]
                diagnostic['arms'][arm] = dict(added_absolute_weight=float(np.abs(w[51:]).sum()),
                    original_effective_weight_l1_change=float(np.abs(effective-reference['weights'][arm]).sum()))
            structure.append(diagnostic)
            if not d['valid']['joint_full']:
                failure = dict(kind=kind, fold=outer, recorded=d.get('selection_failure'))
                try:
                    checked = fit_joint_pairs_checked(covariance_matrix(x[~held]), labels,
                        anchor_index=0, seed=399170+200+outer, starts=5)
                    j = checked.joint
                    failure.update(converged=bool(j.converged), converged_starts=j.converged_starts,
                        multistart=j.multistart_audit, jacobian=j.jacobian_audit,
                        relative_offdiag_misfit=j.relative_offdiag_misfit)
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    failure['reproduced_exception'] = str(exc)
                failures.append(failure)
    assert native == result['native_answers']
    replay = {}
    for name, score in scores.items():
        assert np.isfinite(score).all()
        pb, within, count = metrics(score, gate, data)
        m = result['metrics'][name]
        np.testing.assert_allclose([pb, within], [m['pb'], m['within']], atol=1e-12, rtol=0)
        assert count == m['within_n'] == 6030
        replay[name] = dict(pb=pb, within=within)
    for key, passed_ni in result['practical_preservation'].items():
        arm = key.split('__')[1]
        c = result['primary_contrasts'][key+' minus base__'+arm]
        assert passed_ni == (native[key] == 13769 and c['pb']['low'] > -.01 and c['within']['low'] > -.002)
    audit = dict(status='PASS', answers=len(uids), steps=len(base), tests=passed,
        metrics_checked=len(replay), score_replay_max_error=maxerror, native_answers=native,
        selection=selection, score_sha256=sha(OUT/'SCORES.npz'),
        checks=['frozen dependency hashes', 'unchanged original bank and exact copies',
            'identity-stable standardized noise', 'all fold weights and native coverage',
            '95% first crossing and selected support', 'independent PB counts and pairwise within-AUC',
            'unchanged base controls and gate', 'predeclared practical preservation decision'])
    dump(OUT/'AUDIT.json', audit)
    dump(OUT/'STRUCTURE_DIAGNOSTIC.json', dict(status='COMPLETE',
        interpretation='Post-evaluation structural diagnostics; not a causal readout intervention.',
        folds=structure, failure_refits=failures))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
