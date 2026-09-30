"""Independent frozen-score review; no fitting or changes to Claude's outputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/fusion_shrinkage_iu_v1'
OUT = ROOT / 'results/fusion_shrinkage_iu_codex_review_v1'
BENCH = ROOT / 'results/localization_full_benchmark_v3/evaluation'
METHODS = ['iu_replay', 'full__joint__alw', 'solve__joint__alw',
           'subspace__joint__alw', 'full__joint__a1.0', 'subspace__joint__a1.0']


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pair_auc(y, scores):
    positive = scores[y == 1]
    negative = scores[y == 0]
    if not len(positive) or not len(negative):
        return np.nan
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0).sum() + .5 * (delta == 0).sum()) / delta.size)


def main():
    started = time.time()
    sources = [SOURCE/'METRICS.json', SOURCE/'REPORT.md', BENCH/'JOINED.json', BENCH/'JOINED.npz',
               ROOT/'results/fusion_fixed_gate_v1/METRICS.json',
               ROOT/'results/fusion_fixed_gate_v1/DETECTORS.npz',
               ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json',
               ROOT/'spectral_utils/shrinkage_iu.py', ROOT/'scripts/run_shrinkage_iu_v1.py',
               ROOT/'scripts/evaluate_shrinkage_iu_v1.py', Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    original = load(SOURCE/'METRICS.json')
    joined = load(BENCH/'JOINED.json')
    records = joined['records']; n = len(records)
    assert n == 13769 and len({r['uid'] for r in records}) == n
    with np.load(BENCH/'JOINED.npz', allow_pickle=False) as z:
        offsets, labels, target = z['offsets'], z['labels'], z['target']
        scores = {m: np.full(len(labels), np.nan) for m in METHODS}
        valid = {m: np.zeros(n, bool) for m in METHODS}
        native = {m: np.full(n, -2, int) for m in METHODS}
        for arm in ('dual__iu', 'dual__equal', 'entropy_parent'):
            name = 'frozen_' + arm; column = joined['arms'].index(arm)
            scores[name] = z['scores'][:, column].copy()
            valid[name] = z['valid'][:, column].copy()
            native[name] = np.where(z['decision'][:, column], z['predictions'][:, column], -2)
    cells = np.array([r['cell'] for r in records])
    groups = np.array([r['group_id'] for r in records])
    pb = np.array([c.startswith('pb_') for c in cells]); prm = ~pb
    folded = load(ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json')['outer']
    gate = load(ROOT/'results/fusion_fixed_gate_v1/METRICS.json')
    thresholds = gate['arms']['dual__iu']['rows']['entropy_mean|quantile_0.3']['thresholds']
    with np.load(ROOT/'results/fusion_fixed_gate_v1/DETECTORS.npz', allow_pickle=False) as z:
        detector = z['entropy_mean']
    assert len(detector) == n
    limits = np.array([thresholds.get(str(folded.get(g, -1)), np.nan) for g in groups])
    assert np.isfinite(limits[pb]).all() and np.isfinite(detector[pb]).all()
    open_gate = detector >= limits
    aggregate_hash = hashlib.sha256()
    replay_valid = 0
    for i, record in enumerate(records):
        stem = SOURCE/'scores'/record['uid']
        metadata_bytes = stem.with_suffix('.json').read_bytes()
        arrays_bytes = stem.with_suffix('.npz').read_bytes()
        aggregate_hash.update(record['uid'].encode() + b'\0' + hashlib.sha256(metadata_bytes).digest()
                              + hashlib.sha256(arrays_bytes).digest())
        meta = json.loads(metadata_bytes)
        assert meta['uid'] == record['uid'] and meta['cell'] == record['cell']
        if isinstance(meta['replay'], (float, int)):
            assert meta['replay'] == 0 and meta['seam_max_abs_dw'] == 0
            replay_valid += 1
        sl = slice(offsets[i], offsets[i+1])
        assert offsets[i+1] - offsets[i] == record['steps']
        with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as z:
            for method in METHODS:
                key = method+'__risk'
                assert (key in z.files) == bool(meta['valid'].get(method, False))
                if key not in z.files:
                    continue
                s = z[key]
                assert s.shape == (record['steps'],) and np.isfinite(s).all()
                scores[method][sl] = s; valid[method][i] = True
                native[method][i] = int(z[method+'__gmm'][0])
        if i % 2000 == 0:
            print('Read and validated', i, '/', n, flush=True)
    np.testing.assert_array_equal(scores['iu_replay'], scores['frozen_dual__iu'])
    for v in valid.values():
        np.testing.assert_array_equal(v, valid['iu_replay'])
    pb_cells = sorted(set(cells[pb])); summaries = {}; within = {}; correct = {}; peaks = {}
    for method in scores:
        wa = np.full(n, np.nan); peak = np.full(n, -2, int)
        label_mask = np.zeros(len(labels), bool)
        for i in np.flatnonzero(valid[method]):
            sl = slice(offsets[i], offsets[i+1]); s = scores[method][sl]
            peak[i] = np.argmax(s)
            if prm[i]:
                wa[i] = pair_auc(labels[sl], s); label_mask[sl] = labels[sl] >= 0
        pred = np.where(open_gate, peak, -1)
        ok = valid[method] & (pred == target)
        gmm_ok = valid[method] & (native[method] != -2) & (native[method] == target)
        cell_stats = {}; native_f1 = []
        for cell in pb_cells:
            clean = (cells == cell) & (target == -1); error = (cells == cell) & (target >= 0)
            ca, ea = ok[clean].mean(), ok[error].mean()
            gc, ge = gmm_ok[clean].mean(), gmm_ok[error].mean()
            native_f1.append(2*gc*ge/(gc+ge) if gc+ge else 0.)
            cell_stats[cell] = dict(clean_correct=int(ok[clean].sum()), clean_total=int(clean.sum()),
                error_correct=int(ok[error].sum()), error_total=int(error.sum()),
                raw_exact_peaks=int((peak[error] == target[error]).sum()),
                early=int(((peak[error] < target[error]) & valid[method][error]).sum()),
                late=int(((peak[error] > target[error]) & valid[method][error]).sum()),
                invalid_error=int((~valid[method][error]).sum()),
                suppressed_correct=int(((peak == target) & error & valid[method] & ~open_gate).sum()),
                f1=float(2*ca*ea/(ca+ea) if ca+ea else 0.))
        for cell in pb_cells:
            np.testing.assert_allclose(cell_stats[cell]['f1'], original['results'][method]['pb_cells_fixed'][cell], atol=1e-14, rtol=0)
        y = labels[label_mask] == 1; s = scores[method][label_mask]
        pooled = float((rankdata(s)[y].sum() - y.sum()*(y.sum()+1)/2) / (y.sum()*(~y).sum()))
        summary = dict(prm_pooled=pooled, prm_within=float(np.nanmean(wa)), prm_within_n=int(np.isfinite(wa).sum()),
                       prm_valid=int((prm & valid[method]).sum()), pb_valid=int((pb & valid[method]).sum()),
                       pb_all=float(np.mean([v['f1'] for v in cell_stats.values()])),
                       pb_native_all=float(np.mean(native_f1)), cells=cell_stats)
        ref = original['results'][method]
        for key in ('prm_pooled', 'prm_within', 'prm_within_n', 'prm_valid', 'pb_valid'):
            np.testing.assert_allclose(summary[key], ref[key], atol=1e-14, rtol=0)
        np.testing.assert_allclose(summary['pb_all'], ref['pb_fixed']['all'], atol=1e-14, rtol=0)
        np.testing.assert_allclose(summary['pb_native_all'], ref['pb_gmm']['all'], atol=1e-14, rtol=0)
        summaries[method] = summary; within[method] = wa; correct[method] = ok; peaks[method] = peak
    primary = 'full__joint__alw'; base = 'iu_replay'
    changes = {}
    for cell in pb_cells:
        ix = (cells == cell) & (target >= 0)
        changes[cell] = dict(gained=int((ix & correct[primary] & ~correct[base]).sum()),
            lost=int((ix & ~correct[primary] & correct[base]).sum()),
            f1_delta_pp=100*(summaries[primary]['cells'][cell]['f1']-summaries[base]['cells'][cell]['f1']))
        assert summaries[primary]['cells'][cell]['clean_correct'] == summaries[base]['cells'][cell]['clean_correct']
    pairs = [(primary, base), ('solve__joint__alw', base), (primary, 'subspace__joint__alw'),
             (primary, 'frozen_entropy_parent'), ('subspace__joint__a1.0', 'frozen_entropy_parent')]
    uniq, inv = np.unique(groups, return_inverse=True); ng = len(uniq)
    columns = []; layouts = []
    for a, b in pairs:
        common = np.isfinite(within[a]) & np.isfinite(within[b])
        start = len(columns)
        columns.extend([np.bincount(inv, weights=common, minlength=ng),
                        np.bincount(inv, weights=np.where(common, within[a]-within[b], 0), minlength=ng)])
        for cell in pb_cells:
            clean = (cells == cell) & (target == -1); err = (cells == cell) & (target >= 0)
            for values in (clean, err, clean & correct[a], err & correct[a], clean & correct[b], err & correct[b]):
                columns.append(np.bincount(inv, weights=values, minlength=ng))
        layouts.append((start, len(columns)))
    sufficient = np.column_stack(columns)
    rng = np.random.default_rng(20260909); draws = 10000
    outcomes = [[] for _ in pairs]
    for batch in range(0, draws, 100):
        counts = np.array([np.bincount(row, minlength=ng) for row in rng.integers(0, ng, (100, ng))])
        totals = counts @ sufficient
        for j, (start, end) in enumerate(layouts):
            t = totals[:, start:end]; w = t[:, 1]/t[:, 0]
            c = t[:, 2:].reshape(100, len(pb_cells), 6)
            ca, ea, cb, eb = c[:,:,2]/c[:,:,0], c[:,:,3]/c[:,:,1], c[:,:,4]/c[:,:,0], c[:,:,5]/c[:,:,1]
            fa = np.divide(2*ca*ea, ca+ea, out=np.zeros_like(ca), where=ca+ea>0)
            fb = np.divide(2*cb*eb, cb+eb, out=np.zeros_like(cb), where=cb+eb>0)
            outcomes[j].append(np.column_stack((w, (fa-fb).mean(axis=1))))
    intervals = {}
    for j, (a,b) in enumerate(pairs):
        values = np.concatenate(outcomes[j]); assert np.isfinite(values).all()
        intervals[a+' minus '+b] = dict(scope='primary comparison' if j<2 else 'post-evaluation mechanism/control diagnostic',
             endpoints=['prm_within_auc', 'pb_all_f1'], ci95=np.quantile(values,[.025,.975],axis=0).T.tolist(),
             ci97_5=np.quantile(values,[.0125,.9875],axis=0).T.tolist())
    for path,h in hashes.items():
        assert sha(ROOT/path) == h, 'Source changed during review: '+path
    result = dict(status='PASS_FROZEN_SCORE_METRICS_REPLAY', rows=n, valid_replays=replay_valid,
                  summaries=summaries, primary_error_changes=changes, bootstrap=intervals,
                  bootstrap_draws=draws, bootstrap_groups=ng, bootstrap_seed=20260909,
                  primary_interval_note='97.5% intervals for two primary comparisons within each endpoint; not a correction for all research choices.',
                  ordered_score_files_sha256=aggregate_hash.hexdigest(), source_hashes=hashes,
                  seconds=time.time()-started,
                  scope='Independent saved-score and metric audit; no fusion refits, alpha optimality proof or untouched confirmation. Prior declaration timing is not established by a runtime manifest.')
    OUT.mkdir(exist_ok=True)
    (OUT/'REVIEW.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'primary_error_changes':changes,'bootstrap':intervals,'seconds':result['seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
