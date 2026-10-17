"""Independent replay and separate metric arithmetic for Renyi-view fusion v2.

Replay: recompute the R6_sel bank from the raw saved logprobs / selected
surprisal, apply the checkpointed raw-coordinate effective weights and
intercepts, take a python-sorted top-10 step mean and compare with the
checkpointed step scores at 2e-10; check failure rows are NaN, padding, the
standardized/effective relation, the orientation against the anchor, the
single-view identities (view__H1 == entropy15 from the frozen representation,
view__Hinf == s_1) and the declared joint group sizes.  Metrics (full run only)
are recomputed with separate arithmetic from SCORES.npz (reuses the cross-rank
reviewer's `review_metrics`, which is bound to the frozen evaluator contract).
"""
import argparse
import io
import os
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
_pre = argparse.ArgumentParser(add_help=False); _pre.add_argument('--roster', default=os.environ.get('RENYI_V2_ROSTER', 'fast'))
os.environ['RENYI_V2_ROSTER'] = _pre.parse_known_args()[0].roster
from scripts import run_renyi_view_fusion_v2 as run
from scripts.review_varentropy_expansion_fusion_v1 import review_metrics
from spectral_utils.renyi_view_fusion_v2 import (BANKS, COLUMN_NAMES, GROUP_OF, VIEW_NAMES, anchor_stream,
                                                 bank_matrix, head_distribution)

METHODS = run.METHODS   # the roster actually scored in the reviewed checkpoint
from spectral_utils.direct_probability_fusion import zscore_columns
from spectral_utils.moment_rbm_fusion import representation

base = run.base


def replay_one(uid, lp, chosen, spans, arrays, info):
    X, names = bank_matrix(lp, chosen, 'R6_sel'); anchor = anchor_stream(lp); checks = 0
    q, s = head_distribution(lp)
    np.testing.assert_allclose(X[:, VIEW_NAMES.index('H1')], representation(lp, chosen)[:, 0], atol=1e-12, rtol=0, err_msg=uid + ': H1 != entropy15')
    np.testing.assert_array_equal(X[:, VIEW_NAMES.index('Hinf')], s.min(axis=1))
    assert np.all(np.diff(X[:, :len(VIEW_NAMES)], axis=1) <= 1e-9), uid + ': H_alpha not non-increasing in alpha'
    S, W, E, B = arrays['steps'], arrays['weights'], arrays['effective'], arrays['intercepts']
    assert S.shape == (len(spans), len(METHODS)) and W.shape == (len(METHODS), len(COLUMN_NAMES))
    for j, m in enumerate(METHODS):
        if m in info['failures']:
            assert np.isnan(S[:, j]).all() and np.isnan(W[j]).all() and np.isnan(B[j]), uid + ':' + m + ' failure must be NaN'
            continue
        prefix, solver = m.split('__')
        cols = list(range(len(COLUMN_NAMES))) if prefix in ('view', 'R6_sel') else [COLUMN_NAMES.index(c) for c in BANKS['R6']]
        assert np.isfinite(E[j]).all() and np.isfinite(W[j]).all(), uid + ':' + m + ' nonfinite weights'
        outside = [k for k in range(len(COLUMN_NAMES)) if k not in cols]
        assert np.all(W[j, outside] == 0) and np.all(E[j, outside] == 0), uid + ':' + m + ' weight outside bank'
        token = X @ E[j] + B[j]
        step = np.array([np.mean(sorted(token[a:b], reverse=True)[:10]) for a, b in spans])   # python sort, not numpy partition
        np.testing.assert_allclose(step, S[:, j], atol=2e-10, rtol=2e-10, err_msg=uid + ':' + m)
        if prefix == 'view':
            k = COLUMN_NAMES.index(solver)
            assert W[j, k] == 1 and E[j, k] == 1 and B[j] == 0 and np.sum(W[j] != 0) == 1, uid + ':' + m + ' single view weights'
            np.testing.assert_array_equal(token, X[:, k])
        else:
            Xb = X[:, cols]; Z, keep, mean, scale = zscore_columns(Xb)
            Wb, Eb = W[j, cols], E[j, cols]
            np.testing.assert_allclose(Eb[keep], Wb[keep] / scale[keep], rtol=1e-9, atol=1e-12, err_msg=uid + ':' + m + ' effective/standardized')
            assert np.all(Wb[~keep] == 0), uid + ':' + m + ' dropped-column weight'
            assert info['diagnostics'][m]['active_columns'] == int(keep.sum()), uid + ':' + m + ' active columns'
            if np.std(token) > 1e-12 and np.std(anchor) > 1e-12:
                corr = np.corrcoef(token, anchor)[0, 1]
                assert not (np.isfinite(corr) and corr < -1e-12), uid + ':' + m + ' orientation'
            if solver == 'joint':
                d = info['diagnostics'][m]
                kept = [COLUMN_NAMES[c] for c, k in zip(cols, keep) if k]
                sizes = {}
                for c in kept: sizes[GROUP_OF[c]] = sizes.get(GROUP_OF[c], 0) + 1
                assert len(sizes) == 3 and min(sizes.values()) >= 3, uid + ': joint fitted on an inadmissible partition'
                assert sorted(d['group_sizes'].values()) == sorted(sizes.values()), uid + ': joint group sizes'
                assert d['n_groups'] == 3
        checks += 1
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--skip-source-hashes', action='store_true')
    p.add_argument('--roster', default=run.ROSTER)
    p.add_argument('--memory-guard-kb', type=int, default=run.MEMORY_GUARD_KB)
    args = p.parse_args(); source = args.source_root.resolve(); base.old.configure_source_root(source)
    out = run.OUT; started = time.perf_counter()
    con = sqlite3.connect(out / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'))
    manifest = json.loads(con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()[0])
    assert manifest['roster'] == run.ROSTER and tuple(manifest['methods']) == tuple(METHODS), 'roster mismatch with checkpoint'
    hash_checks = 0; driver_change = None
    if not args.skip_source_hashes:
        disk = json.loads((out / ('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json')).read_text(encoding='utf8'))
        for path, digest in manifest['hashes'].items():
            if not Path(path).exists(): continue
            now = base.old.sha256_file(Path(path))
            if now == digest: hash_checks += 1; continue
            # Only the driver may differ, and only when the evaluate-only pass recorded the change against
            # the checkpoint's own stored hash (scoring module / inputs / protocol hashes must all match).
            change = disk.get('evaluate_only_driver_change') or disk.get('smoke_resume') or {}
            assert Path(path).name == 'run_renyi_view_fusion_v2.py' and change.get('previous_driver_sha256') == digest                 and change.get('current_driver_sha256') == now, 'hash changed: ' + path
            driver_change = dict(path=path, **change); hash_checks += 1
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    rows_by_idx = {r[0]: (r[1], r[2]) for r in con.execute('SELECT idx, payload, info FROM answers')}
    replayed = checks = 0; failures = {}
    with threadpool_limits(limits=1):
        for cell, path, kind, dataset in base.source_specs():
            idx = [i for i in rows_by_idx if records[i]['cell'] == cell]
            if not idx: continue
            run.memory_guard(args.memory_guard_kb)
            rows = base.old._source_row_map(base.old.load_pickle(path), kind=kind, dataset=dataset)
            for i in idx:
                r = records[i]; row = rows[r['row_id']]
                lp = np.asarray(base.old._topk_payload(row)['logprobs'], float); chosen = np.asarray(row['token_spilled_energies'], float)
                spans = np.asarray(row['step_token_spans'], int); blob, info = rows_by_idx[i]
                arrays = dict(np.load(io.BytesIO(blob))); info = json.loads(info)
                checks += replay_one(r['uid'], lp, chosen, spans, arrays, info); replayed += 1
                for m, why in info['failures'].items(): failures.setdefault(m, []).append(why.split(':')[0])
            del rows
    result = dict(status='PASS', scope='SMOKE' if args.smoke else 'FULL', replayed_answers=replayed, replay_checks=checks,
                  manifest_hash_checks=hash_checks, driver_change_accepted=driver_change,
                  failures_by_method={m: len(v) for m, v in failures.items()},
                  metrics_rederived=None, seconds=None)
    if not args.smoke:
        assert replayed == len(records), f'{replayed} != {len(records)}'
        if run.ROSTER == 'joint':
            with np.load(out / 'SCORES.npz') as z, np.load(run.BASE_OUT / 'fast_pass/SCORES.npz') as fast:
                for m in run.FAST_METHODS:
                    assert np.array_equal(z['steps__' + m], fast['steps__' + m], equal_nan=True), m + ': appended fast-pass scores differ'
            result['fast_pass_arms_identical'] = len(run.FAST_METHODS)
        result['metrics_rederived'] = review_metrics(out, records, joined)
    result['seconds'] = time.perf_counter() - started
    base.atomic_json(out / ('SMOKE_REVIEW.json' if args.smoke else 'RESULT_REVIEW.json'), result)
    print('[review]', result, flush=True)


if __name__ == '__main__':
    main()
