"""Independent replay and separate metric arithmetic for the Renyi alpha sweep v1.

Replay per answer: recompute every view from the raw saved logprobs with an independent formula
(direct numpy: q = p/(sum p + 1e-12) on the top-15, s = -log(q + 1e-12); H_alpha = log(sum q^alpha)/(1-alpha),
H_1 = sum q s, H_inf = min s, limit = mean log q; VE_alpha = sum w s^2 - (sum w s)^2 with w ~ q^alpha),
apply the checkpointed one-hot signed weight, the recorded orientation, a python-sorted top-10 step mean,
and compare with the checkpointed step scores at 2e-10.  Checks: H monotone in alpha; VE_1 == frozen
varentropy15 (contributions sum) at 1e-9; orientation flips only in the VE family and only when the
anchor correlation is negative; failures are NaN.  Metrics (full run) re-derived by the cross-rank reviewer's
`review_metrics` (frozen evaluator contract) for every arm and reference.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ['RENYI_V2_ROSTER'] = 'all'
from scripts import run_renyi_alpha_sweep_v1 as sweep_run     # applies the sweep overrides to the v2 harness
from scripts import run_renyi_view_fusion_v2 as run
from scripts.review_varentropy_expansion_fusion_v1 import review_metrics
from spectral_utils.renyi_alpha_sweep import ALPHA_GRID, VE_GRID, ALPHA_OF, FAMILY_OF, METHODS, VIEW_NAMES
from spectral_utils.varentropy_contribution_fusion import contributions

base = run.base
OUT = run.OUT


def independent_views(lp):
    lp15 = np.asarray(lp, float)[:, :15]; p = np.exp(lp15); q = p / (p.sum(axis=1, keepdims=True) + 1e-12); s = -np.log(q + 1e-12)
    cols = {'H0lim': np.log(q + 1e-12).mean(axis=1)}
    for a in ALPHA_GRID:
        if np.isinf(a): cols['Hinf'] = s.min(axis=1)
        elif a == 1: cols['H1'] = (q * s).sum(axis=1)
        else: cols['a%g' % a] = np.log((q ** a).sum(axis=1)) / (1 - a)
    for a in VE_GRID:
        w = np.full_like(q, 1 / 15) if a == 0 else q ** a / (q ** a).sum(axis=1, keepdims=True)
        m = (w * s).sum(axis=1); cols['ve%g' % a] = (w * s * s).sum(axis=1) - m * m
    return cols


def replay_one(uid, lp, spans, arrays, info):
    cols = independent_views(lp); anchor = contributions(lp, 15).sum(axis=1); checks = 0
    H = np.column_stack([cols[n] for n in VIEW_NAMES if not n.startswith('ve') and n != 'H0lim'])
    assert np.all(np.diff(H, axis=1) <= 1e-9), uid + ': H_alpha not non-increasing'
    np.testing.assert_allclose(cols['ve1'], anchor, atol=1e-9, rtol=0, err_msg=uid + ': VE_1 != varentropy15')
    S, W, B = arrays['steps'], arrays['weights'], arrays['intercepts']
    assert S.shape == (len(spans), len(METHODS)) and W.shape == (len(METHODS), len(VIEW_NAMES))
    for j, m in enumerate(METHODS):
        if m in info['failures']:
            assert np.isnan(S[:, j]).all() and np.isnan(W[j]).all(), uid + ':' + m + ' failure must be NaN'; continue
        name = m[len('view__'):]; k = VIEW_NAMES.index(name); d = info['diagnostics'][m]
        assert np.sum(W[j] != 0) == 1 and abs(W[j, k]) == 1 and B[j] == 0, uid + ':' + m + ' one-hot weight'
        x = cols[name]
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.corrcoef(x, anchor)[0, 1] if len(x) > 1 and anchor.std() > 1e-12 else np.nan
        expect_flip = FAMILY_OF[m] == 'escort_varentropy' and np.isfinite(corr) and corr < 0
        assert bool(d['orientation_flipped']) == bool(expect_flip) and (W[j, k] == (-1 if expect_flip else 1)), uid + ':' + m + ' orientation'
        token = x * W[j, k]
        step = np.array([np.mean(sorted(token[a:b], reverse=True)[:10]) for a, b in spans])
        np.testing.assert_allclose(step, S[:, j], atol=2e-10, rtol=2e-10, err_msg=uid + ':' + m)
        checks += 1
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--memory-guard-kb', type=int, default=1_500_000)
    args = p.parse_args(); source = args.source_root.resolve(); base.old.configure_source_root(source); started = time.perf_counter()
    con = sqlite3.connect(OUT / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'))
    manifest = json.loads(con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()[0])
    assert tuple(manifest['methods']) == tuple(METHODS) and manifest['schema'] == 'renyi-alpha-sweep-v1'
    hash_checks = 0
    for path, digest in manifest['hashes'].items():
        if Path(path).exists():
            assert base.old.sha256_file(Path(path)) == digest, 'hash changed: ' + path; hash_checks += 1
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
                lp = np.asarray(base.old._topk_payload(row)['logprobs'], float); spans = np.asarray(row['step_token_spans'], int)
                blob, info = rows_by_idx[i]; arrays = dict(np.load(io.BytesIO(blob))); info = json.loads(info)
                checks += replay_one(r['uid'], lp, spans, arrays, info); replayed += 1
                for m, why in info['failures'].items(): failures.setdefault(m, []).append(why.split(':')[0])
            del rows
    result = dict(status='PASS', scope='SMOKE' if args.smoke else 'FULL', replayed_answers=replayed, replay_checks=checks,
                  manifest_hash_checks=hash_checks, failures_by_method={m: len(v) for m, v in failures.items()}, metrics_rederived=None)
    if not args.smoke:
        assert replayed == len(records)
        result['metrics_rederived'] = review_metrics(OUT, records, joined)
    result['seconds'] = time.perf_counter() - started
    base.atomic_json(OUT / ('SMOKE_REVIEW.json' if args.smoke else 'RESULT_REVIEW.json'), result)
    print('[review]', result, flush=True)


if __name__ == '__main__':
    main()
