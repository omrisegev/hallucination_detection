"""Independent count-based review of Claude's frozen Step331 gate results.

Reads frozen inputs; writes a separate audit, never modifies the source run.
No new detector, threshold rule, fusion fit, or method selection is performed.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'results/localization_full_benchmark_v3'
SOURCE = ROOT / 'results/fusion_fixed_gate_v1'
OUT = ROOT / 'results/fixed_gate_completion_review_v1'
FOLDS = ROOT / 'results/localization_source_group_audit_v1/FOLDS_V2.json'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def measure(target, prediction, valid, cells, weights=None):
    """Direct weighted counts, independent of the production PB evaluator."""
    weights = np.ones(len(target)) if weights is None else np.asarray(weights)
    result = {}
    for cell in sorted(set(cells)):
        rows = np.flatnonzero(cells == cell)
        clean = rows[target[rows] == -1]
        error = rows[target[rows] >= 0]
        nc, ne = float(weights[clean].sum()), float(weights[error].sum())
        hits = valid & (prediction == target)
        ca = float(weights[clean][hits[clean]].sum() / nc) if nc else None
        ea = float(weights[error][hits[error]].sum() / ne) if ne else None
        f1 = None if ca is None or ea is None else (2 * ca * ea / (ca + ea) if ca + ea else 0.)
        result[str(cell)] = dict(clean=nc, erroneous=ne, clean_acc=ca, err_acc=ea, f1=f1,
                                 valid=int(valid[rows].sum()))
    macros = {}
    for panel in ('q4', 'q8', 'all'):
        values = [m['f1'] for c, m in result.items() if panel == 'all' or c.endswith(panel)]
        macros[panel] = float(np.mean(values)) if values and None not in values else None
    return dict(macros=macros, cells=result)


def gate(d, threshold, peak, valid):
    ok = valid & np.isfinite(d)
    return np.where(ok & (d >= threshold), peak, -1), ok


def fit_scalar(d, peak, valid, target, cells):
    grid = np.quantile(d[valid & np.isfinite(d)], np.linspace(.01, .99, 99))
    scores = [measure(target, *gate(d, t, peak, valid), cells)['macros']['all'] for t in grid]
    return float(grid[int(np.argmax(scores))])


def bootstrap(target, cells, groups, predictions, valids, draws=10000, seed=2026090801):
    """Global canonical-group resampling; Q4/Q8 and duplicate sources stay together."""
    names = sorted(set(cells))
    _, inverse = np.unique(groups, return_inverse=True)
    ng = int(inverse.max()) + 1
    # Denominators + clean/error hit counts for each fixed prediction vector.
    matrix = np.zeros((ng, len(names), 2 + 2 * len(predictions)))
    for c, name in enumerate(names):
        clean, err = (cells == name) & (target == -1), (cells == name) & (target >= 0)
        for j, mask in enumerate((clean, err)):
            matrix[:, c, j] = np.bincount(inverse, weights=mask, minlength=ng)
        for j, (p, v) in enumerate(zip(predictions, valids)):
            hit = v & (p == target)
            matrix[:, c, 2 + j * 2] = np.bincount(inverse, weights=clean & hit, minlength=ng)
            matrix[:, c, 3 + j * 2] = np.bincount(inverse, weights=err & hit, minlength=ng)
    rng = np.random.default_rng(seed)
    values = np.empty((draws, len(predictions)))
    for k in range(draws):
        w = np.bincount(rng.integers(ng, size=ng), minlength=ng)
        counts = (w @ matrix.reshape(ng, -1)).reshape(len(names), -1)
        ca = counts[:, 2::2] / counts[:, 0, None]
        ea = counts[:, 3::2] / counts[:, 1, None]
        values[k] = np.divide(2 * ca * ea, ca + ea, out=np.zeros_like(ca), where=(ca + ea) != 0).mean(axis=0)
        if k < 5:
            expanded = np.repeat(np.arange(len(target)), w[inverse])
            for j, (p, v) in enumerate(zip(predictions, valids)):
                direct = measure(target[expanded], p[expanded], v[expanded], cells[expanded])['macros']['all']
                np.testing.assert_allclose(values[k, j], direct, rtol=0, atol=1e-14)
    results = {}
    for j, name in ((1, 'nested_labels_minus_gmm'), (2, 'quantile_0.3_minus_gmm')):
        diff = values[:, j] - values[:, 0]
        finite = diff[np.isfinite(diff)]
        results[name] = dict(point_difference=measure(target, predictions[j], valids[j], cells)['macros']['all']
                            - measure(target, predictions[0], valids[0], cells)['macros']['all'],
                            bootstrap_mean=float(finite.mean()), ci95=np.quantile(finite, [.025, .975]).tolist(),
                            valid_draws=len(finite))
    return dict(draws=draws, seed=seed, groups=ng, unit='canonical source group across cells/scorers',
                scope='fixed predictions; no refitting or detector/q selection uncertainty', contrasts=results)


def fixtures():
    target = np.array([-1, -1, 0, 1, 2])
    peak = np.array([0, 1, 0, 1, 0])
    d = np.array([.1, .9, .8, .7, np.nan])
    p, v = gate(d, .75, peak, np.ones(5, bool))
    assert p.tolist() == [-1, 1, 0, -1, -1]
    assert v.tolist() == [True, True, True, True, False]
    m = measure(target, p, v, np.array(['pb_fixture_q8'] * 5))
    assert m['cells']['pb_fixture_q8']['clean_acc'] == .5
    assert m['cells']['pb_fixture_q8']['err_acc'] == 1 / 3
    assert abs(m['macros']['all'] - .4) < 1e-14
    # An invalid clean row cannot become a success through prediction=-1.
    p, v = gate(np.array([np.nan, .8]), .5, np.array([0, 0]), np.ones(2, bool))
    assert measure(np.array([-1, 0]), p, v, np.array(['pb_fixture_q8'] * 2))['macros']['all'] == 0


def main():
    start = time.time()
    fixtures()
    paths = [Path(__file__), SOURCE/'METRICS.json', SOURCE/'DETECTORS.npz', SOURCE/'REPORT.md',
             ROOT/'spectral_utils/fixed_gate_readout.py', ROOT/'scripts/evaluate_fixed_gate_v1.py',
             BENCH/'evaluation/JOINED.json', BENCH/'evaluation/JOINED.npz', FOLDS]
    before = {str(p): sha(p) for p in paths}
    save(OUT/'RUN_STATE.json', dict(phase='REVIEWING', pid=os.getpid(), started_unix=start))
    j, reported, folds = load(BENCH/'evaluation/JOINED.json'), load(SOURCE/'METRICS.json'), load(FOLDS)
    with np.load(BENCH/'evaluation/JOINED.npz') as f:
        z = {key: f[key] for key in f.files}
    with np.load(SOURCE/'DETECTORS.npz') as f:
        entropy = f['entropy_mean']
    records = j['records']
    assert len({r['uid'] for r in records}) == len(records) == 13769
    mask = np.array([r['cell'].startswith('pb_') for r in records])
    assert int(mask.sum()) == 6800
    cells = np.array([r['cell'] for r in records])[mask]
    groups = np.array([r['group_id'] for r in records])[mask]
    outer = np.array([folds['outer'][g] for g in groups])
    assert set(outer) == set(range(5))
    target, d = z['target'][mask], entropy[mask]
    for fold in range(5):
        assert set(groups[outer == fold]).isdisjoint(groups[outer != fold])
    # Verify copied stream coordinates against the original source, without importing frozen fit code.
    streams = []
    for path in (ROOT/'spectral_utils/answer_localization_v2.py', ROOT/'spectral_utils/fixed_gate_readout.py'):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STREAM_NAMES' for t in n.targets))
        streams.append(tuple(ast.literal_eval(node.value)))
    assert streams[0] == streams[1]
    stream = streams[0].index('entropy_series')
    index = {(r['cell'], r['row_id']): i for i, r in enumerate(records)}
    raw_review = {}
    for cell in sorted(set(cells)):
        folder = BENCH/'inputs'/cell
        raw = np.load(folder/'raw.npy', mmap_mode='r')
        offsets = np.load(folder/'token_offsets.npy')
        ids = np.load(folder/'row_ids.npy', allow_pickle=True)
        checked = 0
        for k, rid in enumerate(ids):
            x = raw[offsets[k]:offsets[k+1], stream]
            finite = np.asarray(x, float)[np.isfinite(x)]
            actual = float(finite.mean()) if len(finite) else np.nan
            np.testing.assert_allclose(actual, entropy[index[(cell, str(rid))]], rtol=0, atol=1e-14, equal_nan=True)
            checked += 1
        raw_review[str(cell)] = dict(rows=checked, raw_shape=list(raw.shape), raw_sha256=sha(folder/'raw.npy'),
                                     offsets_sha256=sha(folder/'token_offsets.npy'), row_ids_sha256=sha(folder/'row_ids.npy'))
    print('Raw entropy replayed on all6800 PB rows', flush=True)
    arms, transfer, primary = {}, {}, None
    for arm, old in reported['arms'].items():
        m = j['arms'].index(arm)
        peak, valid = z['peaks'][mask, m], z['valid'][mask, m].astype(bool)
        saved_pred, saved_valid = z['predictions'][mask, m], z['decision'][mask, m].astype(bool)
        outputs = {'gmm_bic_saved': (saved_pred, saved_valid)}
        ranges = {}
        for rule in ('nested_labels', 'quantile_0.3'):
            pred = np.full(len(d), -1)
            ok = np.zeros(len(d), bool)
            values = []
            expected = old['rows']['entropy_mean|' + rule]['thresholds']
            for fold in range(5):
                train, test = outer != fold, outer == fold
                if rule == 'nested_labels':
                    threshold = fit_scalar(d[train], peak[train], valid[train], target[train], cells[train])
                else:
                    threshold = float(np.quantile(d[train & valid & np.isfinite(d)], .3))
                np.testing.assert_allclose(threshold, expected[str(fold)], rtol=0, atol=1e-14)
                pred[test], ok[test] = gate(d[test], threshold, peak[test], valid[test])
                values.append(threshold)
            assert np.array_equal(pred[pred >= 0], peak[pred >= 0])
            np.testing.assert_array_equal(ok, saved_valid)
            outputs['entropy_mean|' + rule] = pred, ok
            ranges[rule] = [min(values), max(values)]
        measured = {}
        for key, (p, v) in outputs.items():
            met = measure(target, p, v, cells)
            expected = old['rows'][key]
            for panel in ('all', 'q4', 'q8'):
                np.testing.assert_allclose(met['macros'][panel], expected['macros'][panel], rtol=0, atol=1e-14)
            for cell, values in met['cells'].items():
                for field in ('f1', 'clean_acc', 'err_acc'):
                    np.testing.assert_allclose(values[field], expected['cells'][cell][field], rtol=0, atol=1e-14)
            measured[key] = met
        arms[arm] = dict(results=measured, thresholds_range=ranges, valid_decisions=int(saved_valid.sum()))
        if arm == 'dual__iu':
            primary = outputs
            for src, dst in (('easy','hard'),('hard','easy')):
                key = f'entropy_mean|transfer_{src}_to_{dst}'
                entry = old['rows'][key]
                test = np.isin(cells, entry['evaluated_cells'])
                threshold = fit_scalar(d[~test], peak[~test], valid[~test], target[~test], cells[~test])
                np.testing.assert_allclose(threshold, entry['threshold'], rtol=0, atol=1e-14)
                met = measure(target[test], *gate(d[test], threshold, peak[test], valid[test]), cells[test])
                np.testing.assert_allclose(met['macros']['all'], entry['macros']['all'], rtol=0, atol=1e-14)
                p, v = outputs['entropy_mean|nested_labels']
                matched = measure(target[test], p[test], v[test], cells[test])
                transfer[key] = dict(transfer_macro=met['macros']['all'], matched_nested_macro=matched['macros']['all'],
                                     difference=met['macros']['all']-matched['macros']['all'], cells=entry['evaluated_cells'])
        print('Reviewed', arm, flush=True)
    keys = ['gmm_bic_saved', 'entropy_mean|nested_labels', 'entropy_mean|quantile_0.3']
    boot = bootstrap(target, cells, groups, [primary[k][0] for k in keys], [primary[k][1] for k in keys])
    after = {str(p): sha(p) for p in paths}
    assert before == after, 'Source changed during review; do not publish mixed-version evidence'
    result = dict(status='PASS', population=6800, model_answer_rows=True, arms_reviewed=list(arms),
                  raw_entropy_review=raw_review, arms=arms, transfer=transfer, paired_bootstrap=boot,
                  source_hashes=before, checks=['independent_count_metrics', 'raw_entropy_all_rows', 'stream_contract',
                  'canonical_fold_disjointness', 'all_entropy_thresholds_recomputed', 'frozen_peak_and_coverage',
                  'invalid_clean_and_step_zero_fixtures', 'five_bootstrap_draws_vs_expanded_rows', 'source_hashes_unchanged'],
                  findings=['Six arms evaluated by Step331, sourced from nineteen frozen anchors.',
                  'Quantile threshold uses other answers without labels; detector/q were selected using development outcomes.',
                  'Point difference is distinct from bootstrap mean.',
                  'Transfer is compared with nested evaluation on the same destination cells.',
                  'Failure of tested fused-score summaries is not an impossibility theorem.'],
                  seconds=time.time()-start)
    save(OUT/'REVIEW.json', result)
    p = arms['dual__iu']['results']
    q = boot['contrasts']['quantile_0.3_minus_gmm']
    text = '# Independent review of the fixed entropy gate\n\n'
    text += '**PASS.** All6,800 PB model-answer rows, six evaluated arms, all entropy thresholds and metrics replay.\n\n'
    text += '| IU gate | PB all8 |\n|---|---:|\n'
    for key in keys:
        text += f"| {key.replace('|', ' / ')} | {100*p[key]['macros']['all']:.4f}% |\n"
    text += f"\nQuantile0.3 minus GMM: **{100*q['point_difference']:.4f} pp**, 95% CI [{100*q['ci95'][0]:.4f}, {100*q['ci95'][1]:.4f}] pp, 10,000 canonical-group draws.\n"
    text += '\nIntervals condition on fixed predictions. They exclude calibration refits and detector/q selection uncertainty.\n'
    text += '\n| Transfer | transferred | nested, same destination |\n|---|---:|---:|\n'
    for key, v in transfer.items():
        text += f"| {key.replace('|', ' / ')} | {100*v['transfer_macro']:.4f}% | {100*v['matched_nested_macro']:.4f}% |\n"
    text += '\nQuantile calibration uses other answers. Fusion remains answer-local. This is development evidence, not untouched confirmation.\n'
    text += '\nSource reports are preserved. Detailed source hashes, threshold ranges, cell denominators and audit checks are in REVIEW.json.\n'
    (OUT/'REPORT.md').write_text(text, encoding='utf-8')
    save(OUT/'RUN_STATE.json', dict(phase='COMPLETE_REVIEWED', pid=os.getpid(), seconds=time.time()-start))
    print(text, flush=True)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        fixtures()
        print('PASS fixtures')
    else:
        main()
