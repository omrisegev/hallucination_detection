"""Capacity-suite interpretation: convergence, hidden-unit health, depth feasibility, task link.

Read-only over the completed, reviewed capacity checkpoint. No full refit. The optional
``--probe`` refits exact H4 with a larger iteration budget on the 27 capacity smoke answers only;
that is a feasibility diagnostic and never a benchmark candidate.

Hidden-unit conditions (per exact-H4 unit, on the answer's standardized bank):
  dead        : oriented logit std <= 1e-8            (no varying signal)
  saturated   : logit std > 1e-8 but posterior std <= 1e-10 (numerical sigmoid saturation)
  duplicate   : |corr(logit_j, logit_k)| >= 0.999 with an earlier non-dead unit
``surviving_views`` applies the depth suite's own rule (``zscore_columns`` on the 4 oriented
posteriors, scale > 1e-10); fewer than three is exactly the depth smoke failure.
"""
import argparse
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_rbm_literature_completion as run  # noqa: E402
from spectral_utils import rbm_literature_completion as model  # noqa: E402
from spectral_utils.direct_probability_fusion import zscore_columns, step_top_mean  # noqa: E402

MODE = {6: 'posterior', 12: 'logit'}   # retained readouts of the two banks


def unit_report(x, theta, anchor):
    p = x.shape[1]
    _, w, b = model.unpack(theta, p, 4)
    ell, signs = model.oriented_units(x, theta, 4, anchor)
    post = expit(ell)
    lstd = ell.std(axis=0)
    pstd = post.std(axis=0)
    dead = lstd <= 1e-8
    sat = (~dead) & (pstd <= 1e-10)
    dup = np.zeros(4, bool)
    live = np.flatnonzero(~dead)
    if len(live) > 1:
        c = np.abs(np.corrcoef(ell[:, live].T))
        np.fill_diagonal(c, 0)
        for a_, j in enumerate(live):
            if any(c[a_, b_] >= .999 for b_ in range(a_)):
                dup[j] = True
    _, keep, _, _ = zscore_columns(post)
    return dict(unit_logit_std=[float(v) for v in lstd], unit_post_std=[float(v) for v in pstd],
                unit_post_mean=[float(v) for v in post.mean(axis=0)], signs=[int(s) for s in signs],
                weight_norms=[float(v) for v in np.linalg.norm(w, axis=0)],
                dead=int(dead.sum()), saturated=int(sat.sum()), duplicate=int(dup.sum()),
                surviving_views=int(keep.sum()), depth_would_fail=bool(keep.sum() < 3))


def peaks(x, theta, h, anchor, spans):
    ell, _ = model.oriented_units(x, theta, h, anchor)
    logit, post = model.mean_unit_scores(ell)
    return {name: int(np.argmax(step_top_mean(s, spans[:, 0], spans[:, 1], 10)))
            for name, s in (('logit', logit), ('posterior', post))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--probe-maxiter', type=int, default=1000)
    args = ap.parse_args()
    out = run.PROGRAM / 'capacity'
    assert json.loads((out / 'RUN_STATE.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((out / 'RESULT_REVIEW.json').read_text())['status'] == 'PASS'
    source = ROOT.parents[1]
    records, joined, reference = run.load_contract(source)
    db = sqlite3.connect((out / 'CHECKPOINT.sqlite').as_uri() + '?mode=ro', uri=True)
    src = sqlite3.connect(run.modeldb(source).as_uri() + '?mode=ro', uri=True)
    pred = np.load(out / 'SCORES.npz')
    smoke_db = sqlite3.connect((out / 'SMOKE.sqlite').as_uri() + '?mode=ro', uri=True)
    smoke_idx = {int(r[0]) for r in smoke_db.execute('select idx from answers')}
    depth_db = sqlite3.connect((run.PROGRAM / 'depth/SMOKE.sqlite').as_uri() + '?mode=ro', uri=True)
    depth_rows = [(int(i), json.loads(inf)) for i, inf in depth_db.execute('select idx,info from answers')]
    depth_fail = {i: inf for i, inf in depth_rows if inf.get('failures')}
    depth_smoke_idx = {i for i, _ in depth_rows}
    rows, probe, diag_depth = [], [], {}
    t0 = time.time()
    with threadpool_limits(limits=1):
        for path in sorted(run.caches(source).glob('cache_*.npz')):
            with np.load(path) as z:
                cache = {k: z[k] for k in z.files if k != 'labels'}
            for k, i in enumerate(cache['ids']):
                i = int(i)
                _, uid, spans, anchor, banks = run.prepare_answer(k, cache, src, records, joined, reference)
                blob, info = db.execute('select payload,info from answers where idx=?', (i,)).fetchone()
                info = json.loads(info)
                t = int(joined['target'][i])
                is_pb = records[i]['cell'].startswith('pb_')
                with np.load(io.BytesIO(blob)) as z:
                    for bank in (6, 12):
                        x = banks[bank]['x']
                        m = info['models']
                        mode = MODE[bank]
                        e1, e4 = m[f'b{bank}_exact1'], m[f'b{bank}_exact4']
                        c1, c4 = m[f'b{bank}_cd1'], m[f'b{bank}_cd4']
                        theta4 = z[f'b{bank}_exact4::theta']
                        ur = unit_report(x, theta4, anchor)
                        pk4, pk1 = f'b{bank}_exact4_{mode}', f'b{bank}_exact1_{mode}'
                        n4, n1 = int(pred['prediction__' + pk4][i]), int(pred['prediction__' + pk1][i])
                        v4, v1 = bool(pred['valid__' + pk4][i]), bool(pred['valid__' + pk1][i])
                        row = dict(idx=i, uid=uid, group_id=records[i]['group_id'], cell=records[i]['cell'], bank=bank,
                                   n_tokens=len(x),
                                   exact1_nll=e1['nll_final'], exact1_iterations=e1['iterations'],
                                   exact1_converged=e1['converged'],
                                   exact4_nll=e4['nll_final'], exact4_nll_initial=e4['nll_initial'],
                                   exact4_iterations=e4['iterations'], exact4_converged=e4['converged'],
                                   exact4_gradient_max=e4['gradient_max'], exact4_message=e4['message'],
                                   exact4_gain_vs_exact1=e1['nll_final'] - e4['nll_final'],
                                   cd1_nll=c1['nll_final'], cd4_nll=c4['nll_final'], cd4_gradient_max=c4['gradient_max'],
                                   dead_units=ur['dead'], saturated_units=ur['saturated'],
                                   duplicate_units=ur['duplicate'], surviving_views=ur['surviving_views'],
                                   depth_would_fail=ur['depth_would_fail'],
                                   min_unit_logit_std=min(ur['unit_logit_std']),
                                   max_unit_logit_std=max(ur['unit_logit_std']),
                                   is_pb=is_pb, target=t, pred_exact1=n1, pred_exact4=n4,
                                   valid_exact1=v1, valid_exact4=v4,
                                   pb_gained=bool(is_pb and v4 and n4 == t and (n1 != t or not v1)),
                                   pb_lost=bool(is_pb and v1 and n1 == t and (n4 != t or not v4)))
                        rows.append(row)
                        if i in depth_smoke_idx:
                            fails = list((depth_fail.get(i) or {}).get('failures', {}).keys())
                            diag_depth.setdefault(uid, {})[f'bank{bank}'] = dict(**ur, smoke_failure=fails)
                        if args.probe and i in smoke_idx:
                            p = x.shape[1]
                            init = model.initial(p, 4, model.seed_for(uid, f'bank{bank}:init4'))
                            th, dg = model.exact_fit(x, 4, theta=init, maxiter=args.probe_maxiter)
                            probe.append(dict(
                                idx=i, uid=uid, cell=records[i]['cell'], bank=bank, n_tokens=len(x),
                                budget100=dict(nll=e4['nll_final'], iterations=e4['iterations'],
                                               gradient_max=e4['gradient_max'], converged=e4['converged'],
                                               peaks=peaks(x, theta4, 4, anchor, spans)),
                                budget_probe=dict(maxiter=args.probe_maxiter, nll=dg['nll_final'],
                                                  iterations=dg['iterations'], gradient_max=dg['gradient_max'],
                                                  converged=dg['converged'], message=dg['message'],
                                                  seconds=dg['seconds'], peaks=peaks(x, th, 4, anchor, spans)),
                                exact1_nll=e1['nll_final'], units_probe=unit_report(x, th, anchor), target=t))
            print('[capacity-convergence]', path.stem, len(rows), f'{time.time() - t0:.0f}s', flush=True)
    run.csv_write(out / 'CAPACITY_CONVERGENCE.csv', rows)
    summary = {}
    for bank in (6, 12):
        sub = [r for r in rows if r['bank'] == bank]
        g = np.array([r['exact4_gradient_max'] for r in sub])
        gain = np.array([r['exact4_gain_vs_exact1'] for r in sub])
        q = np.quantile(g, [.25, .5, .75])
        strata = {}
        for name, mask in (('q1', g <= q[0]), ('q2', (g > q[0]) & (g <= q[1])),
                           ('q3', (g > q[1]) & (g <= q[2])), ('q4', g > q[2])):
            s = [r for r, mm in zip(sub, mask) if mm and r['is_pb']]
            strata[name] = dict(n_pb=len(s), gained=sum(r['pb_gained'] for r in s), lost=sum(r['pb_lost'] for r in s))
        by_cond = {}
        for cond in ('dead_units', 'saturated_units', 'duplicate_units'):
            by_cond[cond] = {}
            for c in sorted({r[cond] for r in sub}):
                s = [r for r in sub if r[cond] == c]
                by_cond[cond][str(c)] = dict(n=len(s), n_pb=sum(r['is_pb'] for r in s),
                                             gained=sum(r['pb_gained'] for r in s if r['is_pb']),
                                             lost=sum(r['pb_lost'] for r in s if r['is_pb']))
        cells = {}
        for c in sorted({r['cell'] for r in sub}):
            s = [r for r in sub if r['cell'] == c]
            cells[c] = dict(n=len(s), depth_would_fail=sum(r['depth_would_fail'] for r in s))
        nfail = sum(r['depth_would_fail'] for r in sub)
        summary[f'bank{bank}'] = dict(
            n=len(sub), retained_readout=MODE[bank],
            exact4_nonconverged=sum(not r['exact4_converged'] for r in sub),
            exact4_at_iteration_cap=sum(r['exact4_iterations'] >= 100 for r in sub),
            exact4_messages={mm: sum(r['exact4_message'] == mm for r in sub) for mm in sorted({r['exact4_message'] for r in sub})},
            exact1_nonconverged=sum(not r['exact1_converged'] for r in sub),
            exact4_gradient_max_quantiles=dict(zip(('p05', 'p25', 'p50', 'p75', 'p95', 'max'),
                                                   [float(v) for v in np.quantile(g, [.05, .25, .5, .75, .95, 1])])),
            exact4_gain_vs_exact1_quantiles=dict(zip(('p05', 'p25', 'p50', 'p75', 'p95'),
                                                     [float(v) for v in np.quantile(gain, [.05, .25, .5, .75, .95])])),
            exact4_worse_than_exact1=int((gain < 0).sum()),
            units=dict(mean_dead=float(np.mean([r['dead_units'] for r in sub])),
                       mean_saturated=float(np.mean([r['saturated_units'] for r in sub])),
                       mean_duplicate=float(np.mean([r['duplicate_units'] for r in sub])),
                       surviving_views_hist={str(v): sum(1 for r in sub if r['surviving_views'] == v) for v in range(5)}),
            depth_expected_failures=nfail, depth_expected_coverage=1 - nfail / len(sub),
            depth_expected_failures_by_cell=cells,
            pb_vs_exact1=dict(gained=sum(r['pb_gained'] for r in sub), lost=sum(r['pb_lost'] for r in sub),
                              by_gradient_quartile=strata, by_unit_condition=by_cond))
    summary['scope'] = ('Descriptive interpretation of the completed capacity suite. Nonconvergence means the registered '
                        'L-BFGS-B iteration cap (maxiter=100), an optimization-budget limitation; unit conditions describe '
                        'representation behaviour; gained/lost counts are associations, not causal effects. CD arms run a '
                        'fixed epoch budget and are never called converged.')
    run.base.atomic_json(out / 'CAPACITY_CONVERGENCE.json', summary)
    exp = {b: summary[b]['depth_expected_failures'] for b in ('bank6', 'bank12')}
    run.base.atomic_json(run.PROGRAM / 'depth/SMOKE_DIAGNOSIS.json', dict(
        scope='Diagnosis of the original depth smoke failures from the saved exact-H4 first layer; read-only',
        failed_smoke_answers={uid: v for uid, v in diag_depth.items() if any(vv['smoke_failure'] for vv in v.values())},
        all_smoke_answers=diag_depth,
        rule='depth requires >=3 of 4 oriented hidden posteriors with std > 1e-10 after zscore_columns',
        expected_full_population_failures=exp,
        expected_full_population_coverage={b: summary[b]['depth_expected_coverage'] for b in exp}))
    if args.probe:
        for r in probe:
            r['peaks_changed'] = {k: r['budget100']['peaks'][k] != r['budget_probe']['peaks'][k] for k in ('logit', 'posterior')}
            r['nll_decrease_vs_budget100'] = r['budget100']['nll'] - r['budget_probe']['nll']
        run.base.atomic_json(out / 'CAPACITY_MAXITER_PROBE.json', dict(
            scope=('FEASIBILITY ONLY: exact H4 refit with a larger iteration budget on the 27 capacity smoke answers; '
                   'no benchmark inference, no candidate'),
            maxiter=args.probe_maxiter, n=len(probe),
            converged_under_probe=sum(r['budget_probe']['converged'] for r in probe),
            peaks_changed=dict(logit=sum(r['peaks_changed']['logit'] for r in probe),
                               posterior=sum(r['peaks_changed']['posterior'] for r in probe)),
            median_nll_decrease=float(np.median([r['nll_decrease_vs_budget100'] for r in probe])),
            rows=probe))
    print('[capacity-convergence] done', len(rows), 'rows', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
