"""Reconstruct frozen Joint fits and change only their readout."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_joint_full_redundancy_stress_v1 import inputs, PREVIOUS, OUT as STRESS
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump, score_locator, scalar_metrics
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.joint_feature_selection import checked_fit
from spectral_utils.joint_lsml import covariance_matrix, canonicalize_labels, hierarchical_joint_weights, regularized_joint_map_weights
from spectral_utils.lsml_gate_locator_research import _orient
from spectral_utils.digitfree_broad50 import ANCHOR
OUT = ROOT/'results/joint_fixed_readout_v1'
BANKS = ('base', 'duplicates', 'noise')


def state(status, **kwargs):
    dump(OUT/'RUN_STATE.json', dict(status=status, time=time.strftime('%Y-%m-%dT%H:%M:%S'), **kwargs))
    print(status, kwargs, flush=True)


def frozen_fold(bank, outer):
    path = PREVIOUS/f'B51_bocpd_fold{outer}.json' if bank == 'base' else STRESS/f'{bank}_fold{outer}.json'
    return json.loads(path.read_text())


def main():
    OUT.mkdir(exist_ok=True)
    files = [Path(__file__), ROOT/'docs/experiments/JOINT_FIXED_READOUT_V1.md',
        ROOT/'spectral_utils/joint_feature_selection.py', ROOT/'spectral_utils/joint_lsml.py',
        ROOT/'spectral_utils/joint_pair_jacobian.py', ROOT/'spectral_utils/joint_pair_extension.py',
        ROOT/'spectral_utils/dependency_fusion.py', STRESS/'MANIFEST.json', STRESS/'SCORES.npz',
        PREVIOUS/'INPUTS.npz', *[PREVIOUS/f'B51_bocpd_fold{f}.json' for f in range(5)],
        *[STRESS/f'{b}_fold{f}.json' for b in ('duplicates', 'noise') for f in range(5)]]
    manifest = dict(schema='joint-fixed-readout-v1', target_condition=1000., hashes={str(p):sha(p) for p in files})
    mp = OUT/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest: raise ValueError('MANIFEST_DRIFT')
    dump(mp, manifest); state('LOADING'); data, base, uids = inputs()
    rowfold = np.repeat(data['folds'], np.diff(data['offsets']))
    with np.load(STRESS/'SCORES.npz') as z:
        scores = {k:z[k] for k in z.files if k != 'gate'}
        np.testing.assert_array_equal(z['gate'], data['gate'])
    native = {}; replay_max = 0.; fold_meta = []
    for bank in BANKS:
        x = base if bank == 'base' else augment(base, data['offsets'], uids, bank)
        for arm in ('full', 'auto'):
            scores[bank+'__inverse_'+arm] = np.full(len(x), np.nan)
            native[bank+'__inverse_'+arm] = 0
        for outer in range(5):
            saved = OUT/f'{bank}_fold{outer}.json'; held = rowfold == outer
            if saved.exists():
                meta = json.loads(saved.read_text())
            else:
                started = time.perf_counter(); old = frozen_fold(bank, outer)
                meta = dict(bank=bank, outer=outer, arms={})
                for arm in ('full', 'auto'):
                    oldarm = 'joint_'+arm; state('FIT', bank=bank, outer=outer, support=arm)
                    if not old['valid'][oldarm]:
                        meta['arms'][arm] = dict(valid=False, failure=old.get('selection_failure'),
                            weights=np.eye(x.shape[1])[ANCHOR], historical_failure_preserved=True)
                        continue
                    index = 0 if arm == 'full' else old['selection']['automatic_index']
                    frozen = old['selection']['path'][index]
                    active = np.asarray(frozen['active']); labels = canonicalize_labels(np.asarray(old['discovery']['labels'])[active])
                    train = x[~held][:, active]; cov = covariance_matrix(train)
                    fit = checked_fit(cov, labels, seed=399170+200+outer+index); j = fit.joint
                    _, hw, _ = hierarchical_joint_weights(train, labels, j.global_loading, anchor_index=0, small_m_guard=True)
                    expanded = np.zeros(x.shape[1]); expanded[active] = hw
                    expanded, _ = _orient(x[~held], expanded, ANCHOR)
                    error = float(np.max(np.abs(expanded-old['weights'][oldarm])))
                    np.testing.assert_allclose(expanded, old['weights'][oldarm], atol=1e-11, rtol=1e-10)
                    local, inv = regularized_joint_map_weights(None, j.model_covariance, j.global_loading,
                        mode='liu', lam=0., target_condition=1000.)
                    w = np.zeros(x.shape[1]); w[active] = local
                    w, orientation = _orient(x[~held], w, ANCHOR)
                    meta['arms'][arm] = dict(valid=True, active=active, labels=labels, path_index=index,
                        weights=w, hierarchical_replay_max_error=error, observed_covariance=cov,
                        model_covariance=j.model_covariance, global_loading=j.global_loading,
                        group_loading=j.group_loading, diagonal_audit=j.diagonal_audit,
                        inverse=inv, orientation=orientation, converged_starts=j.converged_starts,
                        misfit=j.relative_offdiag_misfit,
                        added_absolute_weight=float(np.abs(w[51:]).sum()))
                meta['seconds'] = time.perf_counter()-started; dump(saved, meta)
            for arm, a in meta['arms'].items():
                name = bank+'__inverse_'+arm
                scores[name][held] = x[held] @ np.asarray(a['weights'])
                if a['valid']: native[name] += int(np.sum(data['folds'] == outer))
                replay_max = max(replay_max, a.get('hierarchical_replay_max_error', 0.))
            fold_meta.append(meta); state('FOLD_COMPLETE', bank=bank, outer=outer, seconds=meta['seconds'])
    state('EVALUATING')
    metrics = {name:score_locator(s, data['gate'], data) for name, s in scores.items()}
    pairs = [(b+'__inverse_auto', b+'__joint_auto') for b in BANKS]
    pairs += [(b+'__inverse_auto', 'base__inverse_auto') for b in ('duplicates', 'noise')]
    ci = bootstrap(data, metrics, pairs)
    preservation = {b:bool(native[b+'__inverse_auto'] == 13769 and
        ci[b+'__inverse_auto minus base__inverse_auto']['pb']['low'] > -.01 and
        ci[b+'__inverse_auto minus base__inverse_auto']['within']['low'] > -.002) for b in ('duplicates', 'noise')}
    result = dict(status='COMPLETE', metrics={k:scalar_metrics(m) for k,m in metrics.items()},
        primary_contrasts=ci, native_answers=native, practical_preservation=preservation,
        hierarchical_replay_max_error=replay_max,
        fit_seconds=sum(m['seconds'] for m in fold_meta))
    dump(OUT/'RESULTS.json', result); np.savez_compressed(OUT/'SCORES.npz', **scores, gate=data['gate'])
    lines = ['# Fixed-bank Joint model-inverse readout', '', '| Method | PB % | Within AUC | Native |', '|---|---:|---:|---:|']
    for k,m in result['metrics'].items():
        lines.append(f"| {k} | {100*m['pb']:.4f} | {m['within']:.6f} | {native.get(k, 'frozen reference')} |")
    lines += ['', 'Noise inverse heads preserve the original2730-answer H1 fallback.',
        'Only the readout changes. Inverse is the historical lambda0 model map, fixed condition1000.',
        'Practical preservation: '+str(preservation), '', '```json', json.dumps(ci, indent=2), '```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    state('COMPLETE', practical_preservation=preservation); print('\n'.join(lines[:29]), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc: state('FAILED', error=str(exc)); raise
