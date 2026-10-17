"""Independent replay and separate metric arithmetic for the varentropy expansion fusion v1.

Replay: recompute every bank from the raw saved logprobs, apply the saved
raw-coordinate effective weights and intercepts, take a python-sorted top-10
step mean, and compare to the checkpointed step scores at 2e-10.  Metrics
(full run only) are recomputed with separate arithmetic from SCORES.npz.
"""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_varentropy_expansion_fusion_v1 as run
from spectral_utils.varentropy_expansion_fusion import BANK_COLUMNS, HISTORICAL, IDENTITY_COEFFICIENTS, METHODS, expansion_columns
from spectral_utils.varentropy_contribution_fusion import contributions
from spectral_utils.direct_probability_fusion import zscore_columns

base = run.base


def pb_harmonic(clean_accuracy, error_accuracy):
    total = clean_accuracy + error_accuracy
    return 2 * clean_accuracy * error_accuracy / total if total else 0.


def auc_separate(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s, method='average')[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def replay_one(uid, lp, chosen, spans, arrays, info, methods=METHODS):
    """Return number of checks; raises on any mismatch. ``methods`` is the checkpoint's own method roster."""
    X, _, _ = expansion_columns(lp, chosen); C15 = contributions(lp, 15); anchor = C15.sum(axis=1); checks = 0
    identity = X[:, :135] @ IDENTITY_COEFFICIENTS[:135]
    assert np.max(np.abs(identity - anchor)) <= 1e-9, uid + ': identity'
    assert abs(info['identity_max_discrepancy'] - float(np.max(np.abs(identity - anchor)))) <= 1e-12
    S, W, E, B = arrays['steps'], arrays['weights'], arrays['effective'], arrays['intercepts']
    assert S.shape == (len(spans), len(methods)) and W.shape == (len(methods), 138)
    for j, m in enumerate(methods):
        if m in info['failures']:
            assert np.isnan(S[:, j]).all() and np.isnan(W[j]).all() and np.isnan(B[j]), uid + ':' + m + ' failure must be NaN'
            continue
        Xb = C15 if m in HISTORICAL else X[:, BANK_COLUMNS[m.split('__')[0]]]; width = Xb.shape[1]
        assert np.isfinite(E[j, :width]).all() and np.isnan(E[j, width:]).all() and np.isnan(W[j, width:]).all(), uid + ':' + m + ' padding'
        token = Xb @ E[j, :width] + B[j]
        step = np.array([np.mean(sorted(token[a:b], reverse=True)[:10]) for a, b in spans])   # python sort, not numpy partition
        np.testing.assert_allclose(step, S[:, j], atol=2e-10, rtol=2e-10, err_msg=uid + ':' + m)
        if m != 'B1_hist__raw':
            Z, keep, mean, scale = zscore_columns(Xb)
            np.testing.assert_allclose(E[j, :width][keep], W[j, :width][keep] / scale[keep], rtol=1e-9, atol=1e-12, err_msg=uid + ':' + m + ' effective/standardized')
            assert np.all(W[j, :width][~keep] == 0), uid + ':' + m + ' dropped-column weight'
            if np.std(token) > 1e-12 and np.std(anchor) > 1e-12:
                corr = np.corrcoef(token, anchor)[0, 1]
                assert not (np.isfinite(corr) and corr < -1e-12), uid + ':' + m + ' orientation'
        else:
            # C15 @ ones and C15.sum(axis=1) differ only by floating summation order (~4e-16).
            np.testing.assert_allclose(token, anchor, atol=1e-12, rtol=0, err_msg=uid + ': B1_hist__raw is not the varentropy sum')
        checks += 1
    return checks


def review_metrics(out, records, joined):
    metrics = json.loads((out / 'METRICS.json').read_text(encoding='utf8'))['metrics']
    offsets = joined['offsets']; target = joined['target']; labels = joined['labels']
    cells = np.array([r['cell'] for r in records]); pb = np.char.startswith(cells, 'pb_')
    detector, thr = base.old._gate_contract(records)
    fmap = json.loads(base.old.FOLDS.read_text(encoding='utf8'))['outer']; folds = np.array([fmap[r['group_id']] for r in records])
    rawmeta = {str(r['idx']): r for r in base.old.load_pickle(base.old.PRMB_LABELS).values()}
    checked = 0
    with np.load(out / 'SCORES.npz') as arrays:
        for name, m in metrics.items():
            flat = arrays['steps__' + name]; within = []; peaks = []; valid = []
            for i in range(len(records)):
                s = flat[offsets[i]:offsets[i + 1]]; ok = bool(len(s) and np.isfinite(s).all()); valid.append(ok)
                peaks.append(int(np.argmax(s)) if ok else -1)
                if not pb[i] and ok:
                    y = labels[offsets[i]:offsets[i + 1]]; keep = y >= 0
                    a = auc_separate(y[keep] == 1, s[keep])
                    if np.isfinite(a): within.append(a)
            valid = np.array(valid); peaks = np.array(peaks)
            dv = valid & np.isfinite(detector) & np.isfinite(thr); pred = np.where(detector >= thr, peaks, -1)
            np.testing.assert_array_equal(pred, arrays['prediction__' + name]); np.testing.assert_array_equal(valid, arrays['valid__' + name])
            assert int(valid.sum()) == m['valid_answers']
            cellvalues = []; qvalues = {4: [], 8: []}
            for cell in sorted(set(cells[pb])):
                cc = (cells == cell) & (target < 0); ee = (cells == cell) & (target >= 0)
                f = pb_harmonic(np.sum(cc & dv & (pred == target)) / cc.sum(), np.sum(ee & dv & (pred == target)) / ee.sum())
                cellvalues.append(f); qvalues[int(cell[-1])].append(f)
            np.testing.assert_allclose(np.mean(cellvalues), m['pb_all8'], atol=1e-12)
            for q in (4, 8): np.testing.assert_allclose(np.mean(qvalues[q]), m['pb_q' + str(q)], atol=1e-12)
            if within: np.testing.assert_allclose(np.mean(within), m['prm_within'], atol=1e-12)
            assert len(within) == m['prm_within_n']
            for f, q in m['prmscore_thresholds'].items():
                train = np.flatnonzero(~pb & valid & (folds != int(f))); test = np.flatnonzero(~pb & valid & (folds == int(f)))
                assert not {records[i]['group_id'] for i in train} & {records[i]['group_id'] for i in test}
                np.testing.assert_allclose(np.quantile(np.concatenate([flat[offsets[i]:offsets[i + 1]] for i in train]), .8), q, atol=1e-12)
            confusion = np.zeros(4, dtype=np.int64); pooled_scores = []; pooled_truth = []
            for i in np.flatnonzero(~pb & valid):
                sl = slice(offsets[i], offsets[i + 1]); s = flat[sl]; raw = rawmeta[str(records[i]['row_id'])]
                errors = {int(v) - 1 for v in raw['error_steps']}; truth = np.array([j in errors for j in range(len(s))])
                np.testing.assert_array_equal(labels[sl] == 1, truth); pooled_scores.extend(s); pooled_truth.extend(truth)
                if raw['classification'] == 'correct': continue
                q = m['prmscore_thresholds'][str(folds[i])]; accept = s < q
                confusion += [np.sum(accept & ~truth), np.sum(accept & truth), np.sum(~accept & truth), np.sum(~accept & ~truth)]
            if pooled_scores: np.testing.assert_allclose(auc_separate(pooled_truth, pooled_scores), m['prm_pooled'], atol=1e-12)
            tp, fp, tn, fn = (int(v) for v in confusion)
            def fmeasure(t, f, n):
                precision = t / (t + f) if t + f else -1; recall = t / (t + n) if t + n else -1
                return 2 * precision * recall / (precision + recall) if precision + recall else -1
            prm = .5 * (fmeasure(tp, fp, fn) + fmeasure(tn, fn, fp))
            np.testing.assert_allclose(prm, m['prmscore_conditional'], atol=1e-12)
            if int(np.sum(~pb & valid)) == int((~pb).sum()): np.testing.assert_allclose(prm, m['prmscore_q08'], atol=1e-12)
            else: assert m['prmscore_q08'] is None
            checked += 1
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--skip-source-hashes', action='store_true')
    p.add_argument('--results-dir', type=Path, default=None, help='checkpoint directory to review (default: the full-pass directory; pass .../fast_pass for the fast pass)')
    p.add_argument('--memory-gate-pb-kb', type=int, default=run.MEMORY_GATE_KB['pb']); p.add_argument('--memory-gate-prmb-kb', type=int, default=run.MEMORY_GATE_KB['prm'])
    args = p.parse_args(); gates = {'pb': args.memory_gate_pb_kb, 'prm': args.memory_gate_prmb_kb}
    source = args.source_root.resolve(); base.old.configure_source_root(source); out = args.results_dir.resolve() if args.results_dir else run.OUT
    manifest = json.loads((out / ('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json')).read_text(encoding='utf8'))
    methods = tuple(manifest['methods'])     # the checkpoint's own roster (23 full-pass arms, 19 fast-pass arms)
    hashed = 0
    for path, digest in manifest['hashes'].items():
        if args.skip_source_hashes and Path(path).suffix == '.pkl': continue
        assert base.old.sha256_file(Path(path)) == digest, 'hash drift: ' + path; hashed += 1
    con = sqlite3.connect((out / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite')).as_uri() + '?mode=ro', uri=True)
    assert con.execute('pragma quick_check').fetchone()[0] == 'ok'
    assert json.loads(con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()[0]) == manifest
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    stored = {i: (blob, json.loads(info)) for i, blob, info in con.execute('SELECT idx,payload,info FROM answers')}
    checks = 0; answers = 0; started = time.perf_counter(); failures = []
    with threadpool_limits(limits=1):
        for cell, path, kind, dataset in base.source_specs():
            indices = [i for i in stored if records[i]['cell'] == cell]
            if not indices: continue
            run.wait_for_memory(gates[kind]); print('[replay]', cell, len(indices), flush=True)
            rows = base.old._source_row_map(base.old.load_pickle(path), kind=kind, dataset=dataset)
            for i in indices:
                blob, info = stored[i]; r = records[i]; row = rows[r['row_id']]; assert info['uid'] == r['uid']
                lp = np.asarray(base.old._topk_payload(row)['logprobs'], float); chosen = np.asarray(row['token_spilled_energies'], float)
                spans = np.asarray(row['step_token_spans'], int); assert spans.shape == (r['steps'], 2)
                with np.load(io.BytesIO(blob), allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
                checks += replay_one(r['uid'], lp, chosen, spans, arrays, info, methods); answers += 1
                failures += [dict(uid=r['uid'], method=m, reason=reason) for m, reason in info['failures'].items()]
            del rows
        result = dict(status='PASS', scope='Separate replay arithmetic in the same session; not an external independent review.',
                      results_dir=str(out), methods=list(methods), pending_arms=manifest.get('pending_arms', []),
                      answers_replayed=answers, arm_checks=checks, hashes_verified=hashed, failures=failures,
                      checks=['manifest/checkpoint hashes', 'bank recomputation from raw logprobs', 'identity residual', 'effective vs standardized weights',
                              'python-sorted top-10 replay at 2e-10', 'NaN on declared failures', 'orientation sign vs anchor'],
                      seconds=time.perf_counter() - started)
        if not args.smoke and (out / 'METRICS.json').exists():
            result['metrics_methods_checked'] = review_metrics(out, records, joined); result['checks'].append('separate PB/PRMB metric arithmetic')
    name = 'SMOKE_REVIEW.json' if args.smoke else 'RESULT_REVIEW.json'
    run.atomic_json_retry(out / name, result); print('[review]', name, result['status'], answers, 'answers', checks, 'arm checks', flush=True)


if __name__ == '__main__':
    try: main()
    except SystemExit: raise
    except BaseException as error:
        name = 'SMOKE_REVIEW.json' if '--smoke' in sys.argv else 'RESULT_REVIEW.json'
        target = run.OUT
        if '--results-dir' in sys.argv: target = Path(sys.argv[sys.argv.index('--results-dir') + 1]).resolve()
        run.atomic_json_retry(target / name, dict(status='FAIL', error=f'{type(error).__name__}: {error}')); raise
