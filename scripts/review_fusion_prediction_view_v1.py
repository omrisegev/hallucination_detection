"""Independent batch-algebra review of the frozen supporting-view audit."""
import os
for option in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[option] = '1'
from collections import Counter
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prediction_audit_io', ROOT/'scripts/audit_fusion_prediction_view_v1.py')
io = importlib.util.module_from_spec(spec); spec.loader.exec_module(io)
OUT = io.OUT


def independent_predictions(x):
    """Explicit batch fit to each past prefix, not the scorer's online update."""
    t, d = x.shape
    ar = x.copy(); last = np.vstack((x[:1], x[:-1])); ema = x.copy()
    state = x[0].copy()
    ar[1] = x[0]
    for i in range(1, t):
        ema[i] = state
        state = (31./33.)*state + (2./33.)*x[i]
        if i < 2: continue
        a, b = x[:i-1], x[1:i]
        ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
        den = np.sum(ac**2, axis=0); slope = np.ones(d)
        np.divide(np.sum(ac*bc, axis=0), den, out=slope, where=den > 0)
        slope = np.clip(slope, -1., 1.)
        ols = b.mean(axis=0) + slope*(x[i-1]-a.mean(axis=0))
        weight = (i-1)/(i-1+16.)
        ar[i] = weight*ols + (1.-weight)*x[i-1]
    return {'ar1': ar, 'last': last, 'ema32': ema}


def centered_rank(a):
    z = (a-a.mean(axis=0))/a.std(axis=0)
    if z.shape[1] == 0: return 0
    s = np.linalg.svd(z, compute_uv=False)
    return int(np.count_nonzero(s > s[0]*max(z.shape)*np.finfo(float).eps))


def summarize(rows):
    out = {'answers': len(rows), 'cells': dict(Counter(r['cell'] for r in rows)),
           'prediction_ratios': {}, 'feature_diagnostics': {}, 'support': {}}
    def summary(v):
        a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
        return {'defined': len(a), 'median': float(np.median(a)) if len(a) else None,
                'q25': float(np.quantile(a, .25)) if len(a) else None,
                'q75': float(np.quantile(a, .75)) if len(a) else None,
                'minimum': float(a.min()) if len(a) else None,
                'maximum': float(a.max()) if len(a) else None}
    for control in ('last', 'ema32'):
        per_stream = []
        for j, name in enumerate(io.PRIMITIVES):
            ratios = [r['diagnostics']['prediction_mse']['ar1'][j] /
                      r['diagnostics']['prediction_mse'][control][j]
                      if r['diagnostics']['prediction_mse'][control][j] > 0 else None for r in rows]
            per_stream.append({'primitive': name, **summary(ratios),
                               'ar1_lower_mse': sum(v is not None and v < 1. for v in ratios)})
        out['prediction_ratios']['ar1_over_'+control] = per_stream
    for bank in ('moment', 'context'):
        for kind in ('ar1', 'last', 'ema32'):
            selected = [r['diagnostics']['banks'][bank+'__'+kind] for r in rows]
            closest = [item['absolute_spearman'] for d in selected for item in d['closest_original']]
            entropy = [abs(v) if v is not None else None for d in selected for v in d['entropy_spearman']]
            out['feature_diagnostics'][bank+'__'+kind] = {
                'active_added': summary([d['active_added'] for d in selected]),
                'closest_original_abs_spearman': summary(closest),
                'entropy_abs_spearman': summary(entropy),
                'near_original_at_090': sum(v is not None and v >= .9 for v in closest),
                'near_original_at_075': sum(v is not None and v >= .75 for v in closest),
                'rank_gain': summary([d['augmented_rank']-d['original_rank'] for d in selected]),
                'original_rank_saturated': sum(d['original_rank'] == d['n_fit_windows']-1 for d in selected)}
    n = [r['diagnostics']['banks']['moment__ar1']['n_fit_windows'] for r in rows]
    out['support'] = {'fit_windows': summary(n), 'n_less_than_36': sum(v < 36 for v in n),
                      'tokens': sum(r['tokens'] for r in rows), 'raw_columns_before': 27, 'raw_columns_after': 36}
    return out


def main():
    started = time.monotonic(); manifest = io.verify(); frozen = io.load(OUT/'AUDIT_FROZEN.json')
    assert frozen['manifest_sha256'] == io.sha(OUT/'MANIFEST.json')
    assert not frozen['labels_decoded'] and not frozen['localization_heads_fitted']
    for path, h in frozen['files'].items(): assert io.sha(path) == h, path
    counts = Counter(); max_prediction_difference = 0.; max_feature_difference = 0.; records = []
    for rec in manifest['selected']:
        uid = rec['uid']; meta = io.load(OUT/'answers'/(uid+'.json')); records.append(meta)
        for key in ('uid', 'row_id', 'group_id', 'tokens', 'cell'): assert meta[key] == rec[key]
        assert meta['manifest_sha256'] == frozen['manifest_sha256']
        assert meta['arrays_sha256'] == io.sha(OUT/'answers'/(uid+'.npz'))
        assert meta['diagnostics']['labels_accessed'] is False
        with np.load(io.PARENT/'inputs'/(uid+'.npz'), allow_pickle=False) as data: raw = data['raw']
        x = raw[:, [io.STREAM_NAMES.index(s) for s in io.PRIMITIVES]]
        expected = independent_predictions(x)
        with np.load(OUT/'answers'/(uid+'.npz'), allow_pickle=False) as a, \
             np.load(io.PARENT/'scores'/(uid+'.npz'), allow_pickle=False) as parent:
            fi = a['fit_indices']; starts = a['window_starts']; ends = a['window_ends']
            full = np.arange(0, len(x)-7, 8)
            np.testing.assert_array_equal(starts, np.unique(np.r_[full, len(x)-8]))
            np.testing.assert_array_equal(starts[fi], full)
            np.testing.assert_array_equal(ends-starts, 8)
            np.testing.assert_array_equal(a['mask'], np.arange(len(x)) > 0)
            np.testing.assert_array_equal(a['fit_pair_counts'], np.maximum(np.arange(len(x))-1, 0))
            assert a['residual_counts'][0] == 7 and np.all(a['residual_counts'][1:] == 8)
            counts['source_and_grid_checks'] += 1
            for kind in ('ar1', 'last', 'ema32'):
                prediction = expected[kind]
                difference = np.max(np.abs(prediction-a[kind+'__predictions']))
                max_prediction_difference = max(max_prediction_difference, float(difference))
                np.testing.assert_allclose(prediction, a[kind+'__predictions'], atol=1e-9, rtol=1e-11)
                err = x-prediction
                values = np.array([np.mean(np.abs(err[max(1, int(lo)):hi]), axis=0) for lo, hi in zip(starts, ends)])
                max_feature_difference = max(max_feature_difference, float(np.max(np.abs(values-a[kind+'__extra']))))
                np.testing.assert_allclose(values, a[kind+'__extra'], atol=1e-9, rtol=1e-11)
                np.testing.assert_allclose(np.mean(err[17:]**2, axis=0), meta['diagnostics']['prediction_mse'][kind], atol=1e-9, rtol=1e-10)
                counts['batch_prediction_stream_traces'] += 9
                counts['residual_window_and_mse_arrays'] += 1
                for bank in ('moment', 'context'):
                    augmented = a[bank+'__'+kind+'__features']; base = parent[bank+'__features']
                    np.testing.assert_array_equal(augmented[:, :27], base)
                    np.testing.assert_array_equal(augmented[:, 27:], a[kind+'__extra'])
                    counts['exact_original_column_replays'] += 1
                    # Redundancy review uses frozen columns, independent np.corrcoef
                    # algebra, and an explicit SVD rank threshold. SciPy rank ties
                    # are intentionally the common primitive in both paths.
                    b, e = base[fi], augmented[fi, 27:]
                    with np.errstate(invalid='ignore', divide='ignore'):
                        ranks = np.column_stack([rankdata(v) for v in np.column_stack((b,e)).T])
                        correlations = np.corrcoef(ranks.T)[27:, :27]
                    d = meta['diagnostics']['banks'][bank+'__'+kind]
                    entropy = d['names'][:27].index('entropy_series__level')
                    actual_entropy = np.array([np.nan if v is None else v for v in d['entropy_spearman']])
                    np.testing.assert_allclose(correlations[:, entropy], actual_entropy, atol=1e-12, rtol=1e-12)
                    for j, old in enumerate(d['closest_original']):
                        if np.isfinite(correlations[j]).any():
                            best = float(np.nanmax(np.abs(correlations[j])))
                            np.testing.assert_allclose(best, old['absolute_spearman'], atol=1e-12, rtol=1e-12)
                            k = d['names'][:27].index(old['original_feature'])
                            assert abs(abs(correlations[j,k])-best) <= 1e-12
                        else: assert old['absolute_spearman'] is None
                    ba = np.ptp(b, axis=0) > 1e-10*np.maximum(1., np.max(np.abs(b), axis=0))
                    ea = np.ptp(e, axis=0) > 1e-10*np.maximum(1., np.max(np.abs(e), axis=0))
                    assert d['active_original'] == int(ba.sum()) and d['active_added'] == int(ea.sum())
                    assert d['original_rank'] == centered_rank(b[:, ba])
                    assert d['augmented_rank'] == centered_rank(np.column_stack((b[:, ba],e[:, ea])))
                    counts['independent_correlation_rank_bundles'] += 1
        if counts['source_and_grid_checks'] % 25 == 0: print('Reviewed', counts['source_and_grid_checks'], '/ 110', flush=True)
    io.save(OUT/'SUMMARY.json', {'status': 'FEASIBILITY_ONLY_NO_QUALITY_RESULT', **summarize(records)})
    io.save(OUT/'REVIEW.json', {'status': 'PASS', 'counts': counts,
        'maximum_prediction_difference': max_prediction_difference,
        'maximum_feature_difference': max_feature_difference,
        'seconds': time.monotonic()-started, 'labels_decoded': False,
        'limitations': ['Same local review session; no separate reviewer agent.',
            'Shared SciPy ranking primitive; independent correlation and batch prediction algebra.',
            'No new IU/Joint fit, localization quality evaluation or winner selection.'],
        'hashes': {str(p): io.sha(p) for p in (Path(__file__), OUT/'MANIFEST.json', OUT/'AUDIT_FROZEN.json', OUT/'SUMMARY.json')}})
    print('Review PASS:', dict(counts), 'max prediction difference', max_prediction_difference, flush=True)


if __name__ == '__main__': main()
