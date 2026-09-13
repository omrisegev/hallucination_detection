"""Renyi-view fusion v2 driver (Stage 3), on the frozen localization evaluator.

Design (Omri, 2026-09-13): fuse several Renyi orders, spending the grid where
the views differ on the top-15 support: alpha in {0.1, 0.25, 0.5, 1, 2, inf}.
Banks R6 / R6_sel; solvers equal, IU-PCR, shrinkage IU and Joint L-SML (declared
tail/head/selected grouping).  Smoke = 27-answer mechanics/feasibility check
only; the full run scores all 13,769 answers and is evaluated on the frozen
contract (PB all-8, PRMB within/pooled, PRMScore q0.8, 10,000-draw paired
bootstrap).  Protocol: docs/experiments/RENYI_VIEW_FUSION_V2.md.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_temporal as base
from spectral_utils.renyi_view_fusion_v2 import (METHODS, NOT_APPLICABLE, MAX_WEIGHTS, COLUMN_NAMES, DIAG_NAMES,
    VIEW_NAMES, SINGLE_VIEWS, FUSED_METHODS, ALPHAS, GROUP_OF, GROUP_NAMES, fit_all, view_diagnostics)
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.fixed_gate_readout import STREAM_NAMES

# Roster passes (Stage-2 pattern): the Joint arm costs ~4.5 s/answer (v2 smoke) versus ~0.02 s for all other arms,
# so the non-Joint roster is scored and evaluated first ('fast'), and the Joint arm separately ('joint'), whose
# evaluation appends the fast-pass step scores (identity-checked) so every contrast is on the same population.
ROSTER = os.environ.get('RENYI_V2_ROSTER', 'fast')
ALL_METHODS = METHODS
FAST_METHODS = tuple(m for m in ALL_METHODS if not m.endswith('__joint'))
JOINT_METHODS = tuple(m for m in ALL_METHODS if m.endswith('__joint'))
METHODS = {'fast': FAST_METHODS, 'joint': JOINT_METHODS, 'all': ALL_METHODS}[ROSTER]
FUSED_IN_ROSTER = tuple(m for m in FUSED_METHODS if m in METHODS)
BASE_OUT = ROOT / 'results/renyi_view_fusion_v2'
OUT = BASE_OUT / (ROSTER + '_pass') if ROSTER != 'all' else BASE_OUT
MEMORY_GUARD_KB = 2_500_000
VIEW_LABEL = {'H0.1': 'Renyi H_0.1', 'H0.25': 'Renyi H_0.25', 'H0.5': 'Renyi H_0.5', 'H1': 'Shannon H_1 (entropy15)',
              'H2': 'Renyi H_2 (collision)', 'Hinf': 'Min-entropy H_inf (= s_1)', 'sel1': 'Selected-token surprisal'}
NAMES = {'view__' + v: 'Single view / ' + VIEW_LABEL[v] for v in SINGLE_VIEWS}
NAMES.update({m: {'R6': 'Six Renyi views', 'R6_sel': 'Six Renyi views + SEL'}[m.split('__')[0]] + ' / ' +
              {'equal': 'oriented equal weights', 'iu': 'IU-PCR learned weights',
               'shrink': 'joint-target LW shrinkage IU', 'joint': 'Joint L-SML (tail/head/selected groups)'}[m.split('__')[1]]
              for m in FUSED_METHODS})
NAMES.update(entropy='Token entropy reference (frozen)', direct_iu='Direct probability IU-PCR (17 inputs, frozen)',
             ref__varentropy15='Raw varentropy15 reference (frozen Step339)',
             ref__varentropy15_iu='Varentropy15 contribution IU (frozen Step339)')
# Primary question: does a combination of Renyi orders localize better than one entropy (H1 = entropy15)?
PRIMARY = {('R6__iu', 'view__H1'), ('R6__equal', 'view__H1'), ('R6_sel__iu', 'view__H1'), ('R6__iu', 'R6__equal')}
REFERENCES = ('ref__varentropy15', 'ref__varentropy15_iu', 'entropy')


def contrast_pairs():
    pairs = []
    for m in FUSED_METHODS:
        pairs += [(m, 'view__' + v) for v in SINGLE_VIEWS]
    pairs += [('R6__iu', 'R6__equal'), ('R6_sel__iu', 'R6_sel__equal'), ('R6_sel__shrink', 'R6_sel__equal'),
              ('R6_sel__shrink', 'R6_sel__iu'), ('R6_sel__joint', 'R6_sel__equal'), ('R6_sel__joint', 'R6_sel__iu'),
              ('R6_sel__joint', 'R6_sel__shrink'), ('R6_sel__iu', 'R6__iu'), ('R6_sel__equal', 'R6__equal')]
    pairs += [('view__' + v, 'view__H1') for v in SINGLE_VIEWS if v != 'H1']
    pairs += [(m, r) for m in FUSED_METHODS for r in ('ref__varentropy15', 'ref__varentropy15_iu')]
    seen, out = set(), []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair); out.append(pair)
    return out


def atomic_json_retry(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(base.dumps(payload) + '\n', encoding='utf8')
    for attempt in range(40):
        try:
            os.replace(tmp, path); return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.05)


base.atomic_json = atomic_json_retry


def clean(obj):
    """JSON-safe copy: NaN/inf -> None, numpy scalars/arrays -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def free_memory_kb():
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out)
    except Exception:
        return -1


def memory_guard(guard_kb, waits=18, sleep_seconds=300):
    if guard_kb <= 0:
        return
    for attempt in range(waits + 1):
        free = free_memory_kb()
        if free < 0 or free > guard_kb:
            return
        if attempt == waits:
            raise RuntimeError(f'BLOCKED: free memory {free} KB stayed below {guard_kb} KB')
        print(f'[memory] free {free} KB < {guard_kb} KB; waiting {sleep_seconds}s ({attempt + 1}/{waits})', flush=True)
        time.sleep(sleep_seconds)


def worker(task):
    i, uid, lp, chosen, spans, expected = task
    started = time.perf_counter()
    fits, failures, seconds = fit_all(lp, chosen, methods=METHODS, uid=uid)
    fit_seconds = time.perf_counter() - started
    # Data-path fidelity: replay the frozen K=50 Renyi-2 and varentropy series from the same rows.
    p50 = np.exp(lp[:, :50]); p50 = p50 / (p50.sum(axis=1, keepdims=True) + 1e-12)
    s50 = -np.log(p50 + 1e-12); h50 = (p50 * s50).sum(axis=1, keepdims=True)
    replay = {'topk_renyi2_series': -np.log((p50 ** 2).sum(axis=1) + 1e-12),
              'topk_varentropy_series': (p50 * (s50 - h50) ** 2).sum(axis=1)}
    errors = {}
    for name, actual in replay.items():
        np.testing.assert_allclose(actual, expected[name], atol=1e-12, rtol=1e-10, err_msg=f'{uid}: frozen {name} mismatch')
        errors[name] = float(np.max(np.abs(actual - expected[name])))
    t0 = time.perf_counter(); views = view_diagnostics(lp, chosen); diag_seconds = time.perf_counter() - t0
    S = np.full((len(spans), len(METHODS)), np.nan); W = np.full((len(METHODS), MAX_WEIGHTS), np.nan)
    E = W.copy(); intercepts = np.full(len(METHODS), np.nan); diag = {}
    for j, m in enumerate(METHODS):
        if m not in fits: continue
        f = fits[m]; S[:, j] = step_top_mean(f['score'], spans[:, 0], spans[:, 1], count=10)
        W[j] = f['weights']; E[j] = f['effective']; intercepts[j] = f['intercept']; diag[m] = f['diagnostics']
    info = dict(uid=uid, n_tokens=int(len(lp)), n_steps=int(len(spans)), failures=failures, seconds=seconds,
                fit_seconds=fit_seconds, diagnostics_seconds=diag_seconds, wall_seconds=time.perf_counter() - started,
                diagnostics=diag, view_diagnostics=views, replay_max_error=errors)
    return i, base.packed(steps=S, weights=W, effective=E, intercepts=intercepts), base.dumps(clean(info))


def manifest_for(source, v2, temporal, varentropy):
    manifest = base.input_manifest(source, v2)
    manifest.update(schema='renyi-view-fusion-v2', methods=list(METHODS), roster=ROSTER, not_applicable=NOT_APPLICABLE,
                    alphas=[('inf' if np.isinf(a) else a) for a in ALPHAS], columns=list(COLUMN_NAMES),
                    groups={c: GROUP_NAMES[g] for c, g in GROUP_OF.items()},
                    temporal_root=str(temporal), varentropy_root=str(varentropy) if varentropy else None,
                    stage='STAGE 3: Renyi-order combination on the frozen contract (design authorized 2026-09-13)')
    files = [Path(__file__), ROOT / 'spectral_utils/renyi_view_fusion_v2.py', ROOT / 'spectral_utils/renyi_view_fusion.py',
             ROOT / 'scripts/test_renyi_view_fusion_v2.py',
             ROOT / 'spectral_utils/varentropy_contribution_fusion.py', ROOT / 'spectral_utils/fixed_gate_readout.py',
             ROOT / 'spectral_utils/joint_lsml.py', ROOT / 'docs/experiments/RENYI_VIEW_FUSION_V2.md',
             temporal / 'results/direct_probability_temporal_v3/SCORES.npz',
             temporal / 'results/direct_probability_temporal_v3/METRICS.json']
    if varentropy:
        files += [varentropy / 'results/varentropy_contribution_fusion_v1/SCORES.npz',
                  varentropy / 'results/varentropy_contribution_fusion_v1/METRICS.json']
    for cell, _, _, _ in base.source_specs():
        files += [base.old.BENCH / 'inputs' / cell / (name + '.npy') for name in ('raw', 'token_offsets', 'row_ids')]
    for path in files:
        manifest['hashes'][str(path)] = base.old.sha256_file(path)
    return manifest


def score(con, records, workers, smoke, max_answers, guard_kb):
    done = {r[0] for r in con.execute('SELECT idx FROM answers')}; started = time.perf_counter()
    detector, _ = base.old._gate_contract(records); processed = 0; load_seconds = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=base.worker_init) as pool:
        for cell, path, kind, dataset in base.source_specs():
            indices = [i for i, r in enumerate(records) if r['cell'] == cell and i not in done]
            if smoke:
                order = sorted((i for i, r in enumerate(records) if r['cell'] == cell), key=lambda i: records[i]['tokens'])
                picks = [order[j] for j in sorted({0, len(order) // 2, min(len(order) - 1, int(.95 * len(order)))})]
                indices = [i for i in picks if i not in done]
                if not indices: print('[skip]', cell, 'smoke picks already checkpointed', flush=True)
            if not indices: continue
            memory_guard(guard_kb)
            print('[load]', cell, len(indices), flush=True); t0 = time.perf_counter()
            rows = base.old._source_row_map(base.old.load_pickle(path), kind=kind, dataset=dataset)
            d = base.old.BENCH / 'inputs' / cell
            raw = np.load(d / 'raw.npy', mmap_mode='r'); to = np.load(d / 'token_offsets.npy')
            lookup = {str(v): j for j, v in enumerate(np.load(d / 'row_ids.npy', allow_pickle=True))}
            load_seconds[cell] = time.perf_counter() - t0
            if smoke:
                for i in indices:
                    if len(rows[records[i]['row_id']]['token_entropies']) != records[i]['tokens']:
                        raise ValueError(f'{records[i]["uid"]}: JOINED tokens differ from the saved trace length')
            for start in range(0, len(indices), 32):
                batch = []
                for i in indices[start:start + 32]:
                    r = records[i]; row = rows[r['row_id']]
                    lp = np.asarray(base.old._topk_payload(row)['logprobs'], float)
                    chosen = np.asarray(row['token_spilled_energies'], float)
                    entropy = np.asarray(row['token_entropies'], float); spans = np.asarray(row['step_token_spans'], int)
                    if lp.shape != (len(entropy), 50): raise ValueError(f'{r["uid"]}: expected T x 50')
                    if chosen.shape != (len(entropy),): raise ValueError(f'{r["uid"]}: selected-token alignment')
                    if spans.shape != (r['steps'], 2): raise ValueError('step count mismatch')
                    with np.load(base.old.BENCH / 'scores' / f'{r["uid"]}.npz') as z:
                        np.testing.assert_array_equal(spans[:, 0], z['step_starts'])
                        np.testing.assert_array_equal(spans[:, 1], z['step_ends'])
                    if kind == 'pb': np.testing.assert_allclose(entropy.mean(), detector[i], atol=1e-12, rtol=0)
                    j = lookup[r['row_id']]
                    expected = {name: np.asarray(raw[to[j]:to[j + 1], STREAM_NAMES.index(name)], float)
                                for name in ('topk_renyi2_series', 'topk_varentropy_series')}
                    if len(expected['topk_renyi2_series']) != len(lp): raise ValueError('frozen token count mismatch')
                    batch.append((i, r['uid'], lp, chosen, spans, expected))
                for item in pool.map(worker, batch, chunksize=1):
                    con.execute('INSERT INTO answers VALUES (?,?,?)', item)
                con.commit(); done.update(item[0] for item in batch); processed += len(batch)
                state = dict(status='SMOKE' if smoke else 'RUNNING', completed=len(done), expected=len(records),
                             last_cell=cell, elapsed_seconds=time.perf_counter() - started, workers=workers,
                             load_seconds=load_seconds)
                base.atomic_json(OUT / ('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'), state)
                print('[checkpoint]', len(done), '/', len(records), round(state['elapsed_seconds'], 1), 'seconds', flush=True)
                if max_answers and processed >= max_answers:
                    del rows, raw; return load_seconds
            del rows, raw
    if not smoke and len(done) == len(records):
        base.atomic_json(OUT / 'RUN_STATE.json', dict(status='SCORING_COMPLETE', completed=len(done), expected=len(records)))
    return load_seconds


def _nanstats(values):
    v = np.asarray([x for x in values if x is not None], float); v = v[np.isfinite(v)]
    if not len(v): return None
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                q10=float(np.quantile(v, .1)), q90=float(np.quantile(v, .9)))


def aggregate_diagnostics(infos, scope):
    """Redundancy summary over scored answers (no labels, no ranking)."""
    views = [r['view_diagnostics'] for r in infos]; n = len(views); cols = list(DIAG_NAMES)
    P = np.array([[[np.nan if v is None else v for v in row] for row in d['pearson']] for d in views], float)
    Sp = np.array([[[np.nan if v is None else v for v in row] for row in d['spearman']] for d in views], float)
    std = np.array([[np.nan if v is None else v for v in d['std']] for d in views], float)
    near = np.array([d['near_constant'] for d in views], bool); dropped = np.array([d['dropped_by_zscore'] for d in views], bool)
    anchor = np.array([[np.nan if v is None else v for v in d['anchor_correlation']] for d in views], float)
    with np.errstate(invalid='ignore'):
        def matrix_summary(M):
            return dict(mean=clean(np.nanmean(M, axis=0)), median=clean(np.nanmedian(M, axis=0)),
                        min=clean(np.nanmin(M, axis=0)), fraction_abs_above_0_95=clean(np.nanmean(np.abs(M) > .95, axis=0)),
                        fraction_abs_above_0_99=clean(np.nanmean(np.abs(M) > .99, axis=0)),
                        n_finite=clean(np.isfinite(M).sum(axis=0)))
        per_column = {c: dict(std=_nanstats(std[:, j]),
                              fraction_near_constant=float(near[:, j].mean()) if j < MAX_WEIGHTS else None,
                              fraction_dropped_by_zscore=float(dropped[:, j].mean()) if j < MAX_WEIGHTS else None,
                              anchor_correlation=_nanstats(anchor[:, j]) if j < MAX_WEIGHTS else None,
                              fraction_anchor_negative=float(np.nanmean(anchor[:, j] < 0)) if j < MAX_WEIGHTS and np.isfinite(anchor[:, j]).any() else None)
                      for j, c in enumerate(cols)}
    fused = {}
    for m in FUSED_IN_ROSTER:
        ds = [r['diagnostics'][m] for r in infos if m in r['diagnostics']]
        entry = dict(n_fitted=len(ds), n_failed=sum(m in r['failures'] for r in infos),
                     failure_reasons=sorted({r['failures'][m].split(':')[0] + ':' + r['failures'][m].split(':')[1][:60]
                                             for r in infos if m in r['failures']}),
                     fraction_global_flip=float(np.mean([d['orientation_flipped'] for d in ds])) if ds else None,
                     mean_column_flips=float(np.mean([d['column_flips'] for d in ds])) if ds else None,
                     active_columns=_nanstats([d['active_columns'] for d in ds]),
                     alpha=_nanstats([d.get('alpha') for d in ds]) if m.endswith('shrink') else None,
                     fraction_alpha_at_one=float(np.mean([d.get('alpha', 0) >= 1 - 1e-12 for d in ds])) if m.endswith('shrink') and ds else None,
                     g2_hat=_nanstats([d.get('g2_hat') for d in ds]) if m.split('__')[1] in ('iu', 'shrink') else None)
        if m.endswith('joint') and ds:
            entry.update(fraction_converged=float(np.mean([d['converged'] for d in ds])),
                         fraction_multistart_pass=float(np.mean([d['multistart_status'] == 'PASS' for d in ds])),
                         fraction_joint_lower_misfit=float(np.mean([d['joint_lower_misfit'] for d in ds])),
                         joint_relative_offdiag_misfit=_nanstats([d['joint_relative_offdiag_misfit'] for d in ds]),
                         hard_relative_offdiag_misfit=_nanstats([d['hard_relative_offdiag_misfit'] for d in ds]),
                         model_covariance_condition=_nanstats([d['model_covariance_condition'] for d in ds]),
                         fraction_condition_above_1e12=float(np.mean([d['model_covariance_condition'] > 1e12 for d in ds])),
                         map_condition_after=_nanstats([d['map_condition_after'] for d in ds]),
                         fraction_map_ridge_positive=float(np.mean([d['map_ridge'] > 0 for d in ds])),
                         fraction_jacobian_full_rank=float(np.mean([d['jacobian_full_global_rank'] for d in ds])),
                         global_loading_cosine_min=_nanstats([d['global_loading_cosine_min'] for d in ds]),
                         anchor_index_counts={c: int(sum(d['anchor_index'] == j for d in ds)) for j, c in enumerate(COLUMN_NAMES)})
        fused[m] = entry
    return dict(scope=scope, n_answers=n, columns=cols, note='Rows/columns of the matrices follow `columns`. '
                'H1 is entropy15 itself; varentropy15 is the orientation anchor; top1_logprob is raw log p_1; '
                'renyi2_k50 is the frozen K=50 topk_renyi2_series (different support from K=15 H2).',
                per_column=per_column, pearson=matrix_summary(P), spearman=matrix_summary(Sp),
                condition_R6=_nanstats([d['condition_R6'] for d in views]),
                condition_R6_sel=_nanstats([d['condition_R6_sel'] for d in views]),
                condition_tail=_nanstats([d['condition_tail'] for d in views]),
                fraction_condition_R6_sel_above_1e6=float(np.mean([(d['condition_R6_sel'] or 0) > 1e6 for d in views])),
                hartley=dict(all_constant=bool(all(d['hartley_constant'] for d in views)),
                             max_std=float(max(d['hartley_std'] for d in views)), value=float(views[0]['hartley_value']),
                             expected=float(np.log(15))),
                h2_k15_vs_renyi2_k50_pearson=_nanstats([d['h2_k15_vs_renyi2_k50_pearson'] for d in views]),
                fused_arms=fused, not_applicable=NOT_APPLICABLE)


def feasibility(infos, load_seconds, workers, wall_seconds):
    n = len(infos); per_arm = {m: float(sum(r['seconds'][m] for r in infos)) for m in METHODS}
    fit_total = float(sum(r['fit_seconds'] for r in infos)); diag_total = float(sum(r['diagnostics_seconds'] for r in infos))
    row_total = float(sum(r['wall_seconds'] for r in infos)); full = 13769
    return dict(scope='SMOKE_FEASIBILITY_ONLY', n_answers=n, workers=workers, full_population=full,
                smoke_wall_seconds=wall_seconds, load_seconds=load_seconds, load_seconds_total=float(sum(load_seconds.values())),
                per_arm_seconds_total=per_arm, per_arm_seconds_per_answer={m: v / n for m, v in per_arm.items()},
                fit_seconds_per_answer=fit_total / n, diagnostics_seconds_per_answer=diag_total / n,
                worker_seconds_per_answer=row_total / n,
                projected_full_fit_seconds=fit_total / n * full / workers,
                projected_full_worker_seconds=row_total / n * full / workers,
                projected_full_wall_seconds=row_total / n * full / workers + float(sum(load_seconds.values())),
                tokens=_nanstats([r['n_tokens'] for r in infos]),
                note='Smoke answers are the shortest/median/95th-percentile traces per cell; the projection is a linear '
                     'extrapolation and excludes evaluation/bootstrap time.')


def evaluate(con, records, joined, temporal, varentropy):
    base.METHODS = METHODS
    scores, telemetry = base.load_scored(con, records, joined['offsets'])
    if ROSTER == 'joint':
        # Append the fast-pass arms (same population, same contract); the fast pass must be complete and evaluated.
        fast_state = json.loads((BASE_OUT / 'fast_pass/RUN_STATE.json').read_text(encoding='utf8'))
        if fast_state.get('status') != 'COMPLETE': raise ValueError('fast pass not COMPLETE; evaluate it first')
        with np.load(BASE_OUT / 'fast_pass/SCORES.npz') as fast:
            for m in FAST_METHODS: scores[m] = fast['steps__' + m]
    with np.load(temporal / 'results/direct_probability_temporal_v3/SCORES.npz') as z:
        for a, b in [('entropy', 'entropy'), ('direct_iu', 'current__iu')]: scores[a] = z['steps__' + b]
    oldmetrics = json.loads((temporal / 'results/direct_probability_temporal_v3/METRICS.json').read_text(encoding='utf8'))
    varmetrics = None
    if varentropy and (varentropy / 'results/varentropy_contribution_fusion_v1/SCORES.npz').exists():
        with np.load(varentropy / 'results/varentropy_contribution_fusion_v1/SCORES.npz') as z:
            scores['ref__varentropy15'] = z['steps__k15__raw']; scores['ref__varentropy15_iu'] = z['steps__k15__iu']
        varmetrics = json.loads((varentropy / 'results/varentropy_contribution_fusion_v1/METRICS.json').read_text(encoding='utf8'))
    metrics, per = base.evaluate_arrays(records, joined, scores)
    for a, b in [('entropy', 'entropy'), ('direct_iu', 'current__iu')]:
        for key in ('pb_all8', 'prm_within', 'prm_pooled', 'prmscore_q08'):
            np.testing.assert_allclose(metrics[a][key], oldmetrics['metrics'][b][key], atol=1e-12, rtol=0)
    if varmetrics:
        for a, b in [('ref__varentropy15', 'k15__raw'), ('ref__varentropy15_iu', 'k15__iu')]:
            for key in ('pb_all8', 'prm_within', 'prm_pooled', 'prmscore_q08'):
                np.testing.assert_allclose(metrics[a][key], varmetrics['metrics'][b][key], atol=1e-12, rtol=0)
    # view__H1 is entropy15 recomputed from the saved top-15 log-probabilities; the frozen 'entropy' reference
    # stream was taken from the cached token_entropies (same definition, generation-time arithmetic). PB and
    # within-answer AUC must agree exactly; pooled AUC / PRMScore may differ at the 1e-8 level through rank ties
    # across answers (observed 8.4e-9 on the fast pass). The discrepancy is recorded, never hidden.
    h1_vs_entropy = {}
    for key in ('pb_all8', 'prm_within', 'prm_pooled', 'prmscore_q08'):
        h1_vs_entropy[key] = float(metrics['view__H1'][key] - metrics['entropy'][key])
        np.testing.assert_allclose(metrics['view__H1'][key], metrics['entropy'][key],
                                   atol=1e-12 if key in ('pb_all8', 'prm_within') else 1e-6, rtol=0)
    pairs = [(a, b) for a, b in contrast_pairs() if a in scores and b in scores]
    print('[evaluate] frozen references reproduced; view__H1 == entropy; bootstrap 10000', flush=True)
    contrasts = base.paired_bootstrap(records, joined, per, pairs=pairs, primary_pairs=PRIMARY, draws=10000)
    pb = np.array([r['cell'].startswith('pb_') for r in records]); target = joined['target']; cases = {}
    for a, b in pairs:
        c = contrasts[a + '_minus_' + b]; c['pb_delta'] = metrics[a]['pb_all8'] - metrics[b]['pb_all8']
        oldhit = pb & per[b]['decision_valid'] & (per[b]['prediction'] == target)
        newhit = pb & per[a]['decision_valid'] & (per[a]['prediction'] == target)
        c.update(gained=int((newhit & ~oldhit).sum()), lost=int((oldhit & ~newhit).sum()))
        cases[a + '_minus_' + b] = [dict(uid=records[i]['uid'], cell=records[i]['cell'], target=int(target[i]),
                                         before=int(per[b]['prediction'][i]), after=int(per[a]['prediction'][i]),
                                         change='gained' if newhit[i] else 'lost') for i in np.flatnonzero(oldhit ^ newhit)]
    infos = [json.loads(r[0]) for r in con.execute('SELECT info FROM answers ORDER BY idx')]
    health = {}
    for m, t in telemetry.items():
        W = np.stack(t.pop('weights')) if t['weights'] else np.zeros((0, MAX_WEIGHTS))
        total = np.abs(W).sum(axis=1, keepdims=True)
        share = np.divide(W, total, out=np.zeros_like(W), where=total > 0)
        health[m] = dict(n_fitted=len(W), n_failed=len(t['failures']), coverage=len(W) / len(records),
                         fit_seconds=t['fit_seconds'], failure_reasons=sorted({f['reason'].split(':')[0] for f in t['failures']}),
                         mean_standardized_coefficients=W.mean(axis=0).tolist() if len(W) else None,
                         mean_absolute_share=np.abs(share).mean(axis=0).tolist() if len(W) else None,
                         mean_negative_share=float(np.maximum(-share, 0).sum(axis=1).mean()) if len(W) else None,
                         columns=list(COLUMN_NAMES))
        t.pop('alphas', None)
    payload = dict(schema='renyi-view-fusion-v2', roster=ROSTER, scored_methods=list(METHODS), view_h1_minus_frozen_entropy=h1_vs_entropy,
                   fast_pass_scores_sha256=base.old.sha256_file(BASE_OUT / 'fast_pass/SCORES.npz') if ROSTER == 'joint' else None,
                   n_answers=len(records), n_steps=int(joined['offsets'][-1]),
                   scope='Full cached localization development; fusion answer-local, gate/calibration external; '
                         'no historical24 result. Stage 3 evaluation on the frozen contract.',
                   metrics=metrics, contrasts=contrasts, primary=[list(p) for p in sorted(PRIMARY)], telemetry=telemetry,
                   not_applicable=NOT_APPLICABLE, historical_references=oldmetrics['historical_references'],
                   mind_gap_reference=oldmetrics['mind_gap_reference'])
    base.atomic_json(OUT / 'METRICS.json', clean(payload)); base.atomic_json(OUT / 'CHANGED_SUCCESSES.json', cases)
    base.atomic_json(OUT / 'FIT_HEALTH.json', clean(health))
    base.atomic_json(OUT / 'DIAGNOSTICS.json', clean(aggregate_diagnostics(infos, 'FULL_DEVELOPMENT_POPULATION')))
    np.savez_compressed(OUT / 'SCORES.npz', **{'steps__' + m: s for m, s in scores.items()},
                        **{'prediction__' + m: p['prediction'] for m, p in per.items()},
                        **{'valid__' + m: p['valid'] for m, p in per.items()})
    rows = []
    for m, x in metrics.items():
        row = dict(method=NAMES.get(m, m), method_id=m, PB_macro_percent=100 * x['pb_all8'], PB_Q4_percent=100 * x['pb_q4'],
                   PB_Q8_percent=100 * x['pb_q8'], PRMB_within_AUC=x['prm_within'], PRMB_pooled_AUC=x['prm_pooled'],
                   PRMScore=x['prmscore_q08'], PRMScore_conditional=x['prmscore_conditional'], valid_answers=x['valid_answers'],
                   within_answers=x['prm_within_n'])
        rows.append(row); print(m, x['pb_all8'], x['prm_within'], x['prm_pooled'], x['prmscore_q08'], flush=True)
    with (OUT / 'SUMMARY.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / 'PB_CELLS.csv').open('w', encoding='utf-8-sig', newline='') as f:
        cells = sorted(next(iter(metrics.values()))['pb_cells']); w = csv.writer(f)
        w.writerow(['method_id'] + [c + '_F1_percent' for c in cells])
        for m, x in metrics.items(): w.writerow([m] + [100 * x['pb_cells'][c]['f1'] for c in cells])
    with (OUT / 'COMPARISON.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['contrast', 'primary', 'ci_level', 'pb_delta_pp', 'pb_ci_low_pp', 'pb_ci_high_pp',
                                       'prm_within_delta_common', 'prm_within_ci_low', 'prm_within_ci_high', 'common_prm_answers',
                                       'gained', 'lost'])
        for key, c in contrasts.items():
            w.writerow([key, c['primary'], c['ci_level'], 100 * c['pb_delta'], 100 * c['pb_ci'][0], 100 * c['pb_ci'][1],
                        c['prm_within_delta_common'], *(c['prm_within_ci'] or [None, None]), c['common_prm_answers'],
                        c['gained'], c['lost']])
    base.atomic_json(OUT / 'COMPARISON.json', clean(contrasts))
    base.atomic_json(OUT / 'RUN_STATE.json', dict(status='COMPLETE', completed=len(records), expected=len(records)))


def connect(path, manifest, smoke, evaluate_only=False):
    """Strict manifest binding; smoke checkpoints (and evaluate-only passes) may resume across a driver-only change.

    Every input, protocol, module and document hash must still match; only this driver's own hash may differ,
    and both hashes are recorded in the manifest (``smoke_resume`` / ``evaluate_only_driver_change``).
    """
    try:
        return base.connect(path, manifest)
    except ValueError:
        if not (smoke or evaluate_only): raise
        import sqlite3
        con = sqlite3.connect(path)
        stored = json.loads(con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()[0])
        key = str(Path(__file__))
        same_meta = {k: v for k, v in stored.items() if k != 'hashes'} == {k: v for k, v in manifest.items() if k != 'hashes'}
        same_hashes = {k: v for k, v in stored['hashes'].items() if k != key} == {k: v for k, v in manifest['hashes'].items() if k != key}
        if not (same_meta and same_hashes and key in stored['hashes']):
            con.close(); raise
        tag = 'smoke_resume' if smoke else 'evaluate_only_driver_change'
        manifest[tag] = dict(previous_driver_sha256=stored['hashes'][key], current_driver_sha256=manifest['hashes'][key],
                             note=('smoke-only resume' if smoke else 'evaluate-only pass') + ' after a driver-only change; all other hashes identical')
        print('[resume] checkpoint accepted across a driver-only change (' + tag + ')', flush=True)
        return con


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--v2-root', type=Path, default=None); p.add_argument('--temporal-root', type=Path, default=None)
    p.add_argument('--varentropy-root', type=Path, default=None)
    p.add_argument('--workers', type=int, default=1); p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-answers', type=int, default=0); p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--memory-guard-kb', type=int, default=MEMORY_GUARD_KB)
    p.add_argument('--allow-full', action='store_true', help='required for a non-smoke run (explicit authorization)')
    args = p.parse_args(); source = args.source_root.resolve()
    print('[roster]', ROSTER, list(METHODS), '->', OUT, flush=True)
    v2 = (args.v2_root or source / '.worktrees/direct-probability-fusion-v2').resolve()
    temporal = (args.temporal_root or source / '.worktrees/direct-probability-temporal-v3').resolve()
    varentropy = (args.varentropy_root or source / '.worktrees/varentropy-contribution-fusion-v1').resolve()
    if not (varentropy / 'results/varentropy_contribution_fusion_v1/SCORES.npz').exists(): varentropy = None
    if not args.smoke and not args.allow_full:
        raise SystemExit('A full run needs --allow-full (explicit authorization; smoke first).')
    base.old.configure_source_root(source); OUT.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for(source, v2, temporal, varentropy)
    con = connect(OUT / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'), manifest, args.smoke, args.evaluate_only)
    base.atomic_json(OUT / ('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'), manifest)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    if len(records) != 13769 or len({r['uid'] for r in records}) != 13769: raise ValueError('benchmark roster mismatch')
    started = time.perf_counter(); load_seconds = {}
    with threadpool_limits(limits=1):
        if not args.evaluate_only:
            load_seconds = score(con, records, args.workers, args.smoke, args.max_answers, args.memory_guard_kb) or {}
        wall = time.perf_counter() - started
        if args.smoke:
            infos = [json.loads(r[0]) for r in con.execute('SELECT info FROM answers ORDER BY idx')]
            base.atomic_json(OUT / 'SMOKE.json', dict(scope='FEASIBILITY_ONLY', n_answers=len(infos),
                n_failures=sum(len(r['failures']) for r in infos),
                failures=[dict(uid=r['uid'], method=m, reason=why) for r in infos for m, why in r['failures'].items()],
                not_applicable=NOT_APPLICABLE, rows=infos))
            base.atomic_json(OUT / 'DIAGNOSTICS.json', clean(aggregate_diagnostics(infos, 'SMOKE_27_ANSWERS_MECHANICS_ONLY')))
            base.atomic_json(OUT / 'FEASIBILITY.json', clean(feasibility(infos, load_seconds, args.workers, wall)))
            base.atomic_json(OUT / 'SMOKE_STATE.json', dict(status='SMOKE_COMPLETE', completed=len(infos), workers=args.workers,
                                                             wall_seconds=wall))
            print('[smoke]', len(infos), 'rows; no benchmark ranking; failures:', sum(len(r['failures']) for r in infos), flush=True)
        else:
            evaluate(con, records, joined, temporal, varentropy)
    con.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        name = 'SMOKE_STATE.json' if '--smoke' in sys.argv else 'RUN_STATE.json'
        path = OUT / name; state = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                     error=f'{type(error).__name__}: {error}')
        atomic_json_retry(path, state)
        raise
