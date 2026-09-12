"""Amended depth suite: full measurement of the original second layer with declared failures.

Protocol: docs/experiments/RBM_DEPTH_AMENDMENT_20260912.md. The original driver
(run_rbm_literature_completion.py) is imported, never edited. Differences from its depth branch:

1. An answer whose four oriented first-layer posteriors leave fewer than three varying columns is
   recorded as the named per-answer failure ``COLLAPSED_HIDDEN_VIEWS`` (NaN step scores, counted in
   coverage). The variant definition, seeds, optimizer and readout are otherwise identical, so every
   non-failing answer reproduces the original smoke bit-for-bit (verified by --verify-original-smoke).
2. Optional registered variants ``layer2_logit_exact`` / ``layer2_logit_cd`` feed the oriented
   first-layer *logits* instead of posteriors (enabled by --with-logit-variants, only after the
   diagnosis in depth/SMOKE_DIAGNOSIS.json supports sigmoid saturation as a collapse cause).
3. Evaluation reports full-population metrics (failures count as missed decisions, PRMScore is
   conditional) beside conditional metrics restricted to the answers each method covers, and a
   conditional paired bootstrap on the common covered answers.
"""
import argparse
import io
import json
import os
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_rbm_literature_completion as run  # noqa: E402
from scripts.run_rbm_data_diagnostics import base, csv_write, METRIC_KEYS  # noqa: E402
from spectral_utils import rbm_literature_completion as model  # noqa: E402
from spectral_utils.direct_probability_fusion import step_top_mean, zscore_columns  # noqa: E402
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402

SUITE = 'depth_amended'
ORIGINAL = ('layer2_exact', 'layer2_cd')
LOGIT = ('layer2_logit_exact', 'layer2_logit_cd')
FAILURE = 'COLLAPSED_HIDDEN_VIEWS'
OUT = run.PROGRAM / SUITE
DOC = ROOT / 'docs/experiments/RBM_DEPTH_AMENDMENT_20260912.md'


class CollapsedHiddenViews(ValueError):
    pass


def variants(with_logit):
    return ORIGINAL + (LOGIT if with_logit else ())


def methods(vs):
    return tuple(f'b{bank}_{v}_{s}' for bank in (6, 12) for v in vs for s in ('logit', 'posterior'))


def manifest_for(source, vs):
    m = run.manifest_for(source, 'depth')
    m.update(schema='rbm-literature-completion-v1/depth-amended-20260912', suite=SUITE,
             methods=list(methods(vs)), variants=list(vs), failure_rule=FAILURE,
             amendment=str(DOC.relative_to(ROOT)), original_driver='scripts/run_rbm_literature_completion.py')
    extra = [Path(__file__), DOC, run.PROGRAM / 'depth/SMOKE_DIAGNOSIS.json', run.PROGRAM / 'depth/SMOKE.json']
    m['hashes'].update({str(p): base.old.sha256_file(p) for p in extra})
    return m


def worker(task):
    vs, data, (states, capinfo) = task
    i, uid, spans, anchor, banks = data
    arrays = {}
    meta = dict(uid=uid, n_tokens=len(anchor), models={}, failures={})

    def save(bank, variant, logit, posterior, state, details):
        key = f'b{bank}_{variant}'
        if not np.isfinite(logit).all() or not np.isfinite(posterior).all():
            raise ValueError('nonfinite token score')
        for name, score in (('logit', logit), ('posterior', posterior)):
            arrays['score::' + key + '_' + name] = step_top_mean(score, spans[:, 0], spans[:, 1], 10)
        for name, value in state.items():
            arrays[key + '::' + name] = np.asarray(value)
        meta['models'][key] = details

    for bank, source in banks.items():
        x = source['x']
        p = x.shape[1]
        basekey = f'b{bank}_exact4'
        first = states[basekey + '::theta']
        firstsign = states[basekey + '::signs']
        _, fw, fb = model.unpack(first, p, 4)
        raw = (x @ fw + fb) * firstsign
        for variant in vs:
            key = f'b{bank}_{variant}'
            try:
                use_logit = variant.startswith('layer2_logit')
                hidden = raw if use_logit else expit(raw)
                z, keep, mean, scale = zscore_columns(hidden)
                if z.shape[1] < 3:
                    raise CollapsedHiddenViews(f'{FAILURE}: {int(keep.sum())} varying hidden views of 4')
                if variant.endswith('_exact'):
                    theta, diag = model.exact_fit(z, 1)
                else:
                    purpose = f'bank{bank}:layer2logitcd' if use_logit else f'bank{bank}:layer2cd'
                    theta, diag = model.cd_fit(z, 1, seed=model.seed_for(uid, purpose))
                ell, signs = model.oriented_units(z, theta, 1, anchor)
                logit, post = model.mean_unit_scores(ell)
                save(bank, variant, logit, post,
                     dict(theta=theta, signs=signs, first=first, firstsign=firstsign, keep=keep, mean=mean, scale=scale),
                     dict(type='stacked_logit' if use_logit else 'stacked', h=1, first_h=4,
                          surviving_views=int(keep.sum()), **diag))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError, KeyError) as e:
                meta['failures'][key] = f'{type(e).__name__}: {e}'
                for s in ('logit', 'posterior'):
                    arrays['score::' + key + '_' + s] = np.full(len(spans), np.nan)
    return i, base.packed(**arrays), base.dumps(meta)


def score(source, vs, con, records, joined, reference, workers, smoke):
    done = {i for i, in con.execute('select idx from answers')}
    start = time.perf_counter()
    src = sqlite3.connect(run.modeldb(source).as_uri() + '?mode=ro', uri=True)
    cap = sqlite3.connect((run.PROGRAM / 'capacity/CHECKPOINT.sqlite').as_uri() + '?mode=ro', uri=True)
    statepath = OUT / ('SMOKE_STATE.json' if smoke else 'RUN_STATE.json')
    with ProcessPoolExecutor(max_workers=workers, initializer=base.worker_init) as pool:
        for path in sorted(run.caches(source).glob('cache_*.npz')):
            with np.load(path) as z:
                cache = {k: z[k] for k in z.files if k != 'labels'}
            ids = cache['ids']
            todo = [k for k, i in enumerate(ids) if int(i) not in done]
            if smoke:
                lengths = np.diff(cache['token_offsets'])
                ordered = np.argsort(lengths)
                chosen = {int(ordered[0]), int(ordered[len(ordered) // 2]), int(ordered[int(.95 * (len(ordered) - 1))])}
                todo = [k for k in todo if k in chosen]
            print('[cell]', SUITE, path.stem, 'remaining', len(todo), flush=True)
            for begin in range(0, len(todo), 16):
                tasks = []
                for k in todo[begin:begin + 16]:
                    data = run.prepare_answer(k, cache, src, records, joined, reference)
                    row = cap.execute('select payload,info from answers where idx=?', (data[0],)).fetchone()
                    with np.load(io.BytesIO(row[0])) as z:
                        states = {name: z[name] for name in z.files if not name.startswith('score::')}
                    tasks.append((vs, data, (states, json.loads(row[1]))))
                for row in pool.map(worker, tasks, chunksize=1):
                    con.execute('insert into answers values (?,?,?)', row)
                    done.add(row[0])
                con.commit()
                state = dict(status='SMOKE' if smoke else 'RUNNING', suite=SUITE, completed=len(done), expected=13769,
                             pid=os.getpid(), seconds=time.perf_counter() - start)
                base.atomic_json(statepath, state)
                print('[checkpoint]', SUITE, len(done), '13769', round(state['seconds'], 1), 'seconds', flush=True)
    src.close()
    cap.close()


def comparisons(vs):
    pairs, primary = [], set()
    for bank in (6, 12):
        main = 'posterior' if bank == 6 else 'logit'
        for s in ('posterior', 'logit'):
            for v in vs:
                pairs.append((f'b{bank}_{v}_{s}', f'b{bank}_exact4_{s}'))
                pairs.append((f'b{bank}_{v}_{s}', f'b{bank}_exact1_{s}'))
            if 'layer2_logit_exact' in vs:
                pairs.append((f'b{bank}_layer2_logit_exact_{s}', f'b{bank}_layer2_exact_{s}'))
                pairs.append((f'b{bank}_layer2_logit_cd_{s}', f'b{bank}_layer2_cd_{s}'))
            a, b = f'b{bank}_layer2_exact_{s}', f'b{bank}_exact4_{s}'
            if s == main:
                primary.add((a, b))
    return list(dict.fromkeys(pairs)), primary


def subset(records, joined, per, idx):
    sub_records = [records[i] for i in idx]
    sub_joined = dict(target=joined['target'][idx])
    sub_per = {m: {k: v[idx] for k, v in p.items()} for m, p in per.items()}
    return sub_records, sub_joined, sub_per


def evaluate(source, vs, con, records, joined, reference):
    scores = dict(reference)
    health = []
    for m in methods(vs):
        scores[m] = np.full(int(joined['offsets'][-1]), np.nan)
    with np.load(run.PROGRAM / 'capacity/SCORES.npz') as z:
        for bank in (6, 12):
            for h in (1, 4):
                for s in ('posterior', 'logit'):
                    key = f'b{bank}_exact{h}_{s}'
                    scores[key] = z['steps__' + key]
    for i, blob, info in con.execute('select idx,payload,info from answers order by idx'):
        d = json.loads(info)
        assert d['uid'] == records[i]['uid']
        health.append(dict(idx=i, **d))
        with np.load(io.BytesIO(blob)) as z:
            for m in methods(vs):
                scores[m][joined['offsets'][i]:joined['offsets'][i + 1]] = z['score::' + m]
    assert len(health) == 13769
    failures = {}
    for h in health:
        for key, msg in h['failures'].items():
            failures.setdefault(key, {'count': 0, 'named_collapse': 0, 'other': []})
            failures[key]['count'] += 1
            if FAILURE in msg:
                failures[key]['named_collapse'] += 1
            else:
                failures[key]['other'].append(dict(uid=h['uid'], reason=msg))
    print('[metrics]', SUITE, len(scores), 'configurations', flush=True)
    metrics, per = base.evaluate_arrays(records, joined, scores)
    old = json.loads((run.parent(source) / 'METRICS.json').read_text())['metrics']
    for m in run.REFS:
        for k in METRIC_KEYS:
            np.testing.assert_allclose(metrics[m][k], old[m][k], atol=1e-12, rtol=0)
    pairs, primary = comparisons(vs)
    print('[bootstrap]', SUITE, '10000 source-group draws, full population', flush=True)
    contrasts = base.paired_bootstrap(records, joined, per, draws=10000, pairs=pairs, primary_pairs=primary)
    target = joined['target']
    cells = np.array([r['cell'] for r in records])
    pb = np.char.startswith(cells, 'pb_')
    error = pb & (target >= 0)
    changes = []
    for a, b in pairs:
        c = contrasts[a + '_minus_' + b]
        c['pb_delta'] = metrics[a]['pb_all8'] - metrics[b]['pb_all8']
        new = error & per[a]['decision_valid'] & (per[a]['prediction'] == target)
        oldhit = error & per[b]['decision_valid'] & (per[b]['prediction'] == target)
        gain, loss = new & ~oldhit, oldhit & ~new
        c.update(gained=int(gain.sum()), lost=int(loss.sum()),
                 lost_early=int(np.sum(loss & (per[a]['peak'] < target))),
                 lost_late=int(np.sum(loss & (per[a]['peak'] > target))),
                 lost_gate=int(np.sum(loss & (per[a]['peak'] == target) & (per[a]['prediction'] == -1))),
                 lost_failure=int(np.sum(loss & ~per[a]['valid'])))
        for i in np.flatnonzero(gain | loss):
            changes.append(dict(comparison=a + '_minus_' + b, uid=records[i]['uid'], cell=records[i]['cell'],
                                target=int(target[i]), before=int(per[b]['prediction'][i]),
                                after=int(per[a]['prediction'][i]), peak_after=int(per[a]['peak'][i]),
                                change='gained' if gain[i] else 'lost'))
    # Conditional panel: each method on the answers it covers; paired contrasts on common coverage.
    conditional = {}
    for m in methods(vs):
        v = per[m]['valid']
        res = pb_metrics(target[pb & v], per[m]['prediction'][pb & v], per[m]['decision_valid'][pb & v], cells[pb & v])
        conditional[m] = dict(coverage=float(v.mean()), covered_answers=int(v.sum()), pb_covered=int((pb & v).sum()),
                              pb_all8=res['macros']['all'], pb_q4=res['macros']['q4'], pb_q8=res['macros']['q8'],
                              prm_within=metrics[m]['prm_within'], prm_within_n=metrics[m]['prm_within_n'],
                              prmscore_conditional=metrics[m]['prmscore_conditional'],
                              prmscore_answers=metrics[m]['prmscore_answers'])
    cond_contrasts = {}
    print('[bootstrap]', SUITE, 'conditional (common covered answers), 10000 draws', flush=True)
    for a, b in pairs:
        common = np.flatnonzero(per[a]['valid'] & per[b]['valid'])
        sr, sj, sp = subset(records, joined, per, common)
        c = base.paired_bootstrap(sr, sj, sp, draws=10000, pairs=[(a, b)], primary_pairs=primary)[a + '_minus_' + b]
        ra = pb_metrics(sj['target'][np.char.startswith(np.array([r['cell'] for r in sr]), 'pb_')],
                        sp[a]['prediction'][np.char.startswith(np.array([r['cell'] for r in sr]), 'pb_')],
                        sp[a]['decision_valid'][np.char.startswith(np.array([r['cell'] for r in sr]), 'pb_')],
                        np.array([r['cell'] for r in sr])[np.char.startswith(np.array([r['cell'] for r in sr]), 'pb_')])
        rb = pb_metrics(sj['target'][np.char.startswith(np.array([r['cell'] for r in sr]), 'pb_')],
                        sp[b]['prediction'][np.char.startswith(np.array([r['cell'] for r in sr]), 'pb_')],
                        sp[b]['decision_valid'][np.char.startswith(np.array([r['cell'] for r in sr]), 'pb_')],
                        np.array([r['cell'] for r in sr])[np.char.startswith(np.array([r['cell'] for r in sr]), 'pb_')])
        c.update(common_answers=int(len(common)), pb_all8_a=ra['macros']['all'], pb_all8_b=rb['macros']['all'],
                 pb_delta=(ra['macros']['all'] - rb['macros']['all']) if ra['macros']['all'] is not None and rb['macros']['all'] is not None else None)
        cond_contrasts[a + '_minus_' + b] = c
    base.atomic_json(OUT / 'METRICS.json', dict(
        suite=SUITE, n_answers=13769, metrics=metrics, contrasts=contrasts, conditional=conditional,
        conditional_contrasts=cond_contrasts, failures=failures, variants=list(vs), failure_rule=FAILURE,
        scope=('Full cached development; unlabeled answer-local second layer over the saved exact-H4 first layer; '
               'external fixed entropy gate. Full-population metrics count declared failures as missed decisions; '
               'conditional metrics cover only fitted answers and are not comparable to full-coverage rows without '
               'the coverage column.')))
    base.atomic_json(OUT / 'FIT_HEALTH.json', health)
    np.savez_compressed(OUT / 'SCORES.npz', **{'steps__' + m: s for m, s in scores.items()},
                        **{'prediction__' + m: p['prediction'] for m, p in per.items()},
                        **{'valid__' + m: p['valid'] for m, p in per.items()})
    csv_write(OUT / 'COMPARISON.csv', [dict(method=m, **{k: v[k] for k in METRIC_KEYS}, valid_answers=v['valid_answers'],
                                            prm_within_n=v['prm_within_n'],
                                            coverage=(conditional[m]['coverage'] if m in conditional else 1.0),
                                            pb_all8_conditional=(conditional[m]['pb_all8'] if m in conditional else v['pb_all8']))
                                       for m, v in metrics.items()])
    csv_write(OUT / 'PB_CELLS.csv', [dict(method=m, cell=c, **v) for m, x in metrics.items() for c, v in x['pb_cells'].items()])
    csv_write(OUT / 'CHANGED_SUCCESSES.csv', changes)
    base.atomic_json(OUT / 'RUN_STATE.json', dict(status='SCORED_AWAITING_REVIEW', suite=SUITE, completed=13769, expected=13769))


def verify_original_smoke(con):
    """Non-failing original smoke answers must replay exactly under the amended driver."""
    orig = sqlite3.connect((run.PROGRAM / 'depth/SMOKE.sqlite').as_uri() + '?mode=ro', uri=True)
    checks, failed_orig, failed_new = 0, 0, 0
    for i, blob, info in orig.execute('select idx,payload,info from answers'):
        info = json.loads(info)
        row = con.execute('select payload,info from answers where idx=?', (i,)).fetchone()
        assert row is not None, f'amended smoke lacks answer {i}'
        new_info = json.loads(row[1])
        with np.load(io.BytesIO(blob)) as a, np.load(io.BytesIO(row[0])) as b:
            for key in info['models']:
                for s in ('logit', 'posterior'):
                    np.testing.assert_allclose(a['score::' + key + '_' + s], b['score::' + key + '_' + s], atol=0, rtol=0)
                    checks += 1
        for key in info['failures']:
            failed_orig += 1
            assert key in new_info['failures'] and FAILURE in new_info['failures'][key], (i, key, new_info['failures'].get(key))
            failed_new += 1
    orig.close()
    return dict(status='PASS', exact_replays=checks, original_failures=failed_orig, renamed_failures=failed_new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--with-logit-variants', action='store_true')
    p.add_argument('--verify-original-smoke', action='store_true')
    args = p.parse_args()
    source = args.source_root.resolve()
    vs = variants(args.with_logit_variants)
    OUT.mkdir(parents=True, exist_ok=True)
    records, joined, reference = run.load_contract(source)
    print('[manifest]', SUITE, 'verify cached states, capacity checkpoint and amendment', flush=True)
    manifest = manifest_for(source, vs)
    con = run.connect(OUT / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'), manifest)
    base.atomic_json(OUT / ('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'), manifest)
    with threadpool_limits(limits=1):
        if not args.evaluate_only:
            score(source, vs, con, records, joined, reference, args.workers, args.smoke)
        n = con.execute('select count(*) from answers').fetchone()[0]
        if args.smoke:
            failures = [json.loads(i)['failures'] for i, in con.execute('select info from answers')]
            named = all(FAILURE in msg for f in failures for msg in f.values())
            status = 'PASS' if not any(failures) else ('PASS_WITH_DECLARED_FAILURES' if named else 'FAIL')
            payload = dict(status=status, answers=n, failures=failures, failure_rule=FAILURE,
                           declared_failures=sum(len(f) for f in failures),
                           scope='mechanics and runtime only; no candidate ranking; declared collapses are counted, not fixed')
            if args.verify_original_smoke:
                payload['original_smoke_replay'] = verify_original_smoke(con)
            base.atomic_json(OUT / 'SMOKE.json', payload)
            print('[smoke]', status, payload.get('original_smoke_replay'), flush=True)
        else:
            assert n == 13769
            evaluate(source, vs, con, records, joined, reference)
    con.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'LAST_ERROR.log').write_text(traceback.format_exc(), encoding='utf8')
        raise
