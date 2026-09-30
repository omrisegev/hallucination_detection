"""Full source-fold runner and evaluator for the predeclared confidence experiment."""
from pathlib import Path
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .lsml_group_confidence import (
    answer_standardize, binary_tail, collapse_duplicates, fit_binary_tree,
    group_contributions, linearized_weights, score,
)
from .external_generalization.artifacts import atomic_json, file_hash
from .external_generalization.fusion import fit_weights, standardize
from .external_generalization.evaluation import confusion, metric, within_auc
from .label_sanity import check_labels
from .prmbench import prmbench_evaluate

ARMS = ('confidence_sum', 'confidence_average', 'confidence_linear', 'binary_likelihood',
        'raw28_equal', 'family15_equal', 'family15_binary_lsml',
        'bank11_lsml', 'bank11_equal', 'ct7_z')
CONTRASTS = ('confidence_average', 'raw28_equal', 'family15_equal', 'family15_binary_lsml')


def load_inputs(root):
    legacy = root/'.worktrees/ssl-pseudolabel-residual-v1/results/named_group_fusion_v1/run_20260924'
    manifest = json.loads((legacy/'INPUT_MANIFEST.json').read_text())
    paths = {k: Path(manifest[k]['path']) for k in ('pool_z', 'pool_names', 'oof_answers', 'oof_step_scores')}
    for k, p in paths.items():
        if file_hash(p) != manifest[k]['sha256']:
            raise ValueError('changed frozen input: '+k)
    paths.update(design=legacy/'DESIGN.json',
                 joined=root/'results/localization_full_benchmark_v3/evaluation/JOINED.json',
                 arrays=root/'results/localization_full_benchmark_v3/evaluation/JOINED.npz',
                 folds=root/'results/localization_source_group_audit_v1/FOLDS_V2.json',
                 source_fits=root/'results/lsml_external_generalization_v1/evaluation/source/VALIDATION_FITS.json',
                 source_scores=root/'results/lsml_external_generalization_v1/evaluation/source/VALIDATION.npz',
                 metadata=root/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl')
    # Do not read target/decision columns while fitting.
    ans = pd.read_csv(paths['oof_answers'], usecols=['uid', 'id', 'source_group', 'fold', 'cell'])
    records = json.loads(paths['joined'].read_text())['records']
    z = np.load(paths['oof_step_scores'])
    off = z['offsets']
    np.testing.assert_array_equal(off, np.load(paths['arrays'])['offsets'])
    assert len(ans) == len(records) == 13769 and off[-1] == 145597
    fm = json.loads(paths['folds'].read_text())['outer']
    for i, row in enumerate(records):
        assert (ans.uid[i], ans.id[i], ans.source_group[i], ans.cell[i], ans.fold[i]) == (
            row['uid'], row['row_id'], row['group_id'], row['cell'], fm[row['group_id']])
    assert ans.groupby('source_group').fold.nunique().max() == 1
    design = json.loads(paths['design'].read_text())
    names = json.loads(paths['pool_names'].read_text())
    pool = np.load(paths['pool_z'])
    assert pool.shape == (off[-1], len(names)) and np.isfinite(pool).all()
    kept = design['kept']
    x = answer_standardize(pool[:, [names.index(c) for c in kept]], off)
    x *= np.array([design['orientation'][c] for c in kept])
    bank11 = answer_standardize(pool[:, :11], off)
    family = design['splits']['S3_M15']
    groups = np.array([next(g for g, cols in enumerate(family.values()) if c in cols) for c in kept])
    virtual = np.column_stack([x[:, groups == g].mean(1) for g in range(len(family))])
    virtual = answer_standardize(virtual, off)
    return {'paths': paths, 'ans': ans, 'off': off, 'x': x, 'bank11': bank11,
            'virtual': virtual, 'groups': groups, 'names': kept, 'family_names': list(family),
            'ct7': answer_standardize(z['ct7'].astype(float), off)}


def run_fit(root, out, command, tests):
    started = time.perf_counter()
    if (out/'FREEZE.json').exists():
        raise FileExistsError('new run required; cannot overwrite a frozen experiment')
    out.mkdir(parents=True, exist_ok=True)
    d = load_inputs(root)
    code = [root/'spectral_utils/lsml_group_confidence.py', Path(__file__),
            root/'scripts/run_lsml_group_confidence.py', root/'tests/test_lsml_group_confidence.py',
            root/'docs/experiments/LSML_GROUP_CONFIDENCE_V1.md']
    freeze = {'command': command, 'time_utc': pd.Timestamp.now(tz='UTC').isoformat(),
              'inputs': {k: {'path': str(p), 'sha256': file_hash(p), 'bytes': p.stat().st_size} for k, p in d['paths'].items()},
              'code': {str(p.relative_to(root)): file_hash(p) for p in code},
              'arms': list(ARMS), 'contrasts': list(CONTRASTS), 'answers': len(d['ans']),
              'steps': int(d['off'][-1]), 'tests': tests,
              'access': 'donor fit 3 / unlabeled pooled PB+PRMB calibration 1 / evaluation 1; development-label-selected bank'}
    atomic_json(out/'FREEZE.json', freeze)
    ans, off, x, groups = (d[k] for k in ('ans', 'off', 'x', 'groups'))
    sf = np.repeat(ans.fold.to_numpy(), np.diff(off))
    binary = binary_tail(x, off)
    bvirtual = binary_tail(d['virtual'], off)
    oof = {arm: np.full(off[-1], np.nan) for arm in ARMS}
    pred = {arm: np.full(off[-1], -1, np.int8) for arm in ARMS}
    writes = np.zeros(off[-1], np.int8)
    sourcefits = json.loads(d['paths']['source_fits'].read_text())
    source = np.load(d['paths']['source_scores'])
    np.testing.assert_array_equal(source['folds'], ans.fold.to_numpy())
    fits, cal_arrays = [], {}
    for fold in range(5):
        calibration = (fold+1) % 5
        train = (sf != fold) & (sf != calibration)
        cal, test = sf == calibration, sf == fold
        b, expansion, gr, membership = collapse_duplicates(binary[train], x[train], groups)
        model = fit_binary_tree(b, gr)
        # Save an unconverged fit before stopping; never hide it in a fallback.
        atomic_json(out/f'MODEL_{fold}.json', model)
        if not model['converged']:
            raise RuntimeError(f'fold {fold}: EM did not converge within 500 iterations')
        z = x @ expansion
        continuous = score(z, model)
        average = score(z, model, average_evidence=True)
        binaryscore = score(binary @ expansion, model, continuous=False)
        linear = z @ linearized_weights(model)
        flsml = fit_weights(standardize(bvirtual[train].astype(float)))
        src = next(r for r in sourcefits if r['test'] == fold)
        assert src['calibration'] == calibration
        raw = {'confidence_sum': continuous, 'confidence_average': average,
               'confidence_linear': linear, 'binary_likelihood': binaryscore,
               'raw28_equal': x.mean(1), 'family15_equal': d['virtual'].mean(1),
               'family15_binary_lsml': d['virtual'] @ np.array(flsml['weights']),
               'bank11_lsml': d['bank11'] @ np.array(src['fit']['weights']),
               'bank11_equal': d['bank11'].mean(1), 'ct7_z': d['ct7']}
        thresholds = {}
        for arm, values in raw.items():
            values = answer_standardize(values, off)
            tau = float(np.quantile(values[cal], .8))
            thresholds[arm] = tau
            oof[arm][test] = values[test]
            pred[arm][test] = values[test] < tau
            cal_arrays[f'{fold}__{arm}'] = values[cal]
        for arm, srcname in [('bank11_lsml', 'frozen_lsml'), ('bank11_equal', 'frozen_equal'), ('ct7_z', 'ct7')]:
            np.testing.assert_allclose(oof[arm][test], source[srcname+'_score'][test], atol=1e-6, rtol=0)
            np.testing.assert_array_equal(pred[arm][test], source[srcname+'_pred'][test])
        writes[test] += 1
        contributions = group_contributions(z[test], model)
        entry = {'test': fold, 'calibration': calibration,
                 'fit_folds': [f for f in range(5) if f not in (fold, calibration)],
                 'train_steps': int(train.sum()), 'calibration_steps': int(cal.sum()), 'test_steps': int(test.sum()),
                 'model_hash': file_hash(out/f'MODEL_{fold}.json'), 'expansion': expansion.tolist(),
                 'duplicate_classes': membership, 'thresholds': thresholds, 'family_lsml': flsml,
                 'group_contribution_sd': contributions.std(0).tolist(),
                 'linear_weights_original': (expansion @ linearized_weights(model)).tolist(),
                 'seconds_so_far': time.perf_counter()-started}
        fits.append(entry)
        atomic_json(out/'FITS.json', fits)
        print('fit fold', fold, 'iterations', model['iterations'], 'seconds', round(time.perf_counter()-started, 1), flush=True)
        if time.perf_counter()-started > 1800:
            raise TimeoutError('registered 30 minute wall cap exceeded')
    np.testing.assert_array_equal(writes, 1)
    assert all(np.isfinite(v).all() for v in oof.values())
    assert all(np.isin(v, [0, 1]).all() for v in pred.values())
    np.savez_compressed(out/'PREDICTIONS.npz', offsets=off, folds=ans.fold.to_numpy(), writes=writes,
                        **{k+'_score': v for k, v in oof.items()}, **{k+'_pred': v for k, v in pred.items()})
    np.savez_compressed(out/'CALIBRATION.npz', **cal_arrays)
    atomic_json(out/'SEAL.json', {'predictions_sha256': file_hash(out/'PREDICTIONS.npz'),
                'calibration_sha256': file_hash(out/'CALIBRATION.npz'),
                'freeze_sha256': file_hash(out/'FREEZE.json'), 'fits_sha256': file_hash(out/'FITS.json'),
                'source_anchor_replay': 'PASS all 13769 rows: scores and decisions',
                'fit_seconds': time.perf_counter()-started, 'answers': len(ans), 'steps': int(off[-1])})


def bootstrap_prmscore(counts, source_groups, draws=20000):
    unique, inverse = np.unique(source_groups, return_inverse=True)
    names = ['confidence_sum', *CONTRASTS]
    grouped = np.zeros((len(unique), len(names)*4))
    for j, arm in enumerate(names):
        np.add.at(grouped[:, j*4:j*4+4], inverse, counts[arm])
    rng = np.random.default_rng(20260924)
    samples = np.empty((draws, len(names)))
    for start in range(0, draws, 128):
        n = min(128, draws-start)
        w = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=n)
        total = (w @ grouped).reshape(n, len(names), 4)
        samples[start:start+n] = metric(total, 'socratic')
    rows = []
    for j, arm in enumerate(CONTRASTS, 1):
        delta = samples[:, 0]-samples[:, j]
        point = float(metric(counts['confidence_sum'].sum(0), 'socratic')-metric(counts[arm].sum(0), 'socratic'))
        lo, hi = np.quantile(delta, [.05/(2*len(CONTRASTS)), 1-.05/(2*len(CONTRASTS))])
        lo95, hi95 = np.quantile(delta, [.025, .975])
        rows.append({'candidate': 'confidence_sum', 'control': arm, 'metric': 'prmscore',
                     'delta': point, 'ci_corrected_lo': float(lo), 'ci_corrected_hi': float(hi),
                     'ci95_lo': float(lo95), 'ci95_hi': float(hi95), 'draws': draws,
                     'family_size': len(CONTRASTS), 'source_groups': len(unique), 'N': len(source_groups)})
    return rows


def evaluate(root, out):
    started = time.perf_counter()
    seal = json.loads((out/'SEAL.json').read_text())
    assert file_hash(out/'PREDICTIONS.npz') == seal['predictions_sha256']
    assert file_hash(out/'CALIBRATION.npz') == seal['calibration_sha256']
    assert file_hash(out/'FREEZE.json') == seal['freeze_sha256']
    assert file_hash(out/'FITS.json') == seal['fits_sha256']
    frozen = json.loads((out/'FREEZE.json').read_text())
    paths = {k: Path(v['path']) for k, v in frozen['inputs'].items()}
    for k, p in paths.items():
        assert file_hash(p) == frozen['inputs'][k]['sha256']
    ans = pd.read_csv(paths['oof_answers'], usecols=['uid', 'id', 'source_group', 'fold', 'cell', 'target'])
    raw = np.load(paths['arrays'])
    off, error = raw['offsets'], raw['labels'].astype(bool)
    np.testing.assert_array_equal(error, np.load(paths['oof_step_scores'])['labels'].astype(bool))
    np.testing.assert_array_equal(ans.target, raw['target'])
    meta = {r['idx']: r for r in pickle.loads(paths['metadata'].read_bytes()).values()}
    prm = ans.cell.str.startswith('prm').to_numpy()
    pb = ~prm
    noncontrol = np.array([prm[i] and meta[ans.id[i]]['classification'] != 'correct' for i in range(len(ans))])
    eligible = np.array([prm[i] and error[a:b].any() and not error[a:b].all() for i, (a, b) in enumerate(zip(off[:-1], off[1:]))])
    for i in np.flatnonzero(prm):
        a, b = off[i:i+2]
        np.testing.assert_array_equal(error[a:b], [j+1 in meta[ans.id[i]]['error_steps'] for j in range(b-a)])
    assert (prm.sum(), noncontrol.sum(), eligible.sum(), (pb & (ans.target >= 0)).sum()) == (6969, 6211, 6030, 4442)
    sanity = check_labels(error[np.repeat(noncontrol, np.diff(off))])
    if not sanity.ok:
        raise ValueError(sanity.summary())
    z = np.load(out/'PREDICTIONS.npz')
    np.testing.assert_array_equal(z['offsets'], off)
    np.testing.assert_array_equal(z['writes'], 1)
    counts, aucs, hits, rows, cells, folds = {}, {}, {}, [], [], []
    behavioral = []
    for arm in ARMS:
        s, p = z[arm+'_score'], z[arm+'_pred']
        assert np.isfinite(s).all() and np.isin(p, [0, 1]).all()
        c = np.zeros((len(ans), 4), dtype=np.int64)
        auc = np.full(len(ans), np.nan)
        hit = np.full(len(ans), np.nan)
        for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
            if noncontrol[i]:
                c[i] = confusion(~error[a:b], p[a:b])
            if eligible[i]:
                auc[i] = within_auc(~error[a:b], s[a:b], np.ones(b-a, bool))
            if pb[i] and ans.target[i] >= 0:
                peak = int(np.flatnonzero(s[a:b] >= s[a:b].max()-8*np.finfo(float).eps)[0])
                hit[i] = float(peak == ans.target[i])
        counts[arm], aucs[arm], hits[arm] = c, auc, hit
        official = prmbench_evaluate(
            [{'idx': ans.id[i], 'labels': p[off[i]:off[i+1]].astype(int).tolist()} for i in np.flatnonzero(prm)],
            [meta[ans.id[i]] for i in np.flatnonzero(prm)])['total']
        ps = float(metric(c.sum(0), 'socratic'))
        np.testing.assert_allclose(ps, (official['f1']+official['negative_f1'])/2, atol=1e-12)
        pbvalues = [float(np.nanmean(hit[ans.cell == cell])) for cell in sorted(set(ans.cell[pb]))]
        for endpoint, value, n in [('prmscore', ps, int(noncontrol.sum())),
                                  ('prm_within_auc', float(np.nanmean(auc)), int(eligible.sum())),
                                  ('pb_sla_macro8', float(np.mean(pbvalues)), int(np.isfinite(hit).sum()))]:
            rows.append({'method': arm, 'metric': endpoint, 'estimate': value, 'N': n,
                         'coverage_answers': len(ans), 'flag': sanity.flag_string() or 'OK',
                         'source': 'PREDICTIONS.npz', 'source_sha256': seal['predictions_sha256']})
        for cell in sorted(set(ans.cell)):
            mask = (ans.cell == cell).to_numpy()
            if cell.startswith('prm'):
                cells.append({'method': arm, 'cell': cell, 'prmscore': float(metric(c[mask].sum(0), 'socratic')),
                              'within_auc': float(np.nanmean(auc[mask])), 'N': int((noncontrol & mask).sum())})
            else:
                cells.append({'method': arm, 'cell': cell, 'pb_sla': float(np.nanmean(hit[mask])), 'N': int(np.isfinite(hit[mask]).sum())})
        for f in range(5):
            mask = ans.fold.to_numpy() == f
            folds.append({'method': arm, 'fold': f, 'prmscore': float(metric(c[mask].sum(0), 'socratic')),
                          'N': int((noncontrol & mask).sum())})
        reference = z['family15_equal_score']
        rank_corr = [float(spearmanr(s[a:b], reference[a:b]).statistic)
                     for i, (a, b) in enumerate(zip(off[:-1], off[1:]))
                     if eligible[i] and np.std(s[a:b]) > 1e-8 and np.std(reference[a:b]) > 1e-8]
        behavioral.append({'method': arm, 'mean_answer_spearman_vs_family_equal': float(np.mean(rank_corr)),
                           'rank_answers': len(rank_corr),
                           'changed_step_decisions_vs_family_equal': int(np.sum(p != z['family15_equal_pred'])),
                           'total_steps': int(off[-1])})
    contrasts = bootstrap_prmscore({k: v[prm] for k, v in counts.items()}, ans.source_group.to_numpy()[prm])
    pd.DataFrame(rows).to_csv(out/'METRICS.csv', index=False)
    pd.DataFrame(cells).to_csv(out/'PER_CELL.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'PER_FOLD.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out/'CONTRASTS.csv', index=False)
    atomic_json(out/'BEHAVIOR.json', behavioral)
    np.savez_compressed(out/'EVALUATION.npz', **{k+'_counts': v for k, v in counts.items()},
                        **{k+'_auc': v for k, v in aucs.items()}, **{k+'_hit': v for k, v in hits.items()})
    atomic_json(out/'STATUS.json', {'status': 'COMPLETE', 'answers': len(ans), 'steps': int(off[-1]),
                'prm_noncontrol_answers': int(noncontrol.sum()), 'prm_noncontrol_steps': int(np.repeat(noncontrol, np.diff(off)).sum()),
                'eligible_auc_answers': int(eligible.sum()), 'pb_error_answers': int((pb & (ans.target >= 0)).sum()),
                'arms': list(ARMS), 'failures': [], 'official_metric_replays': len(ARMS),
                'label_sanity': sanity.summary(), 'evaluation_seconds': time.perf_counter()-started,
                'prediction_hash': seal['predictions_sha256']})
    print(pd.DataFrame(rows).pivot(index='method', columns='metric', values='estimate').round(6).to_string(), flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
