"""Cross-rank varentropy expansion fusion v1 on the frozen localization evaluator; resumable."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_temporal as base
from spectral_utils.varentropy_expansion_fusion import (
    BANKS, BANK_COLUMNS, HISTORICAL, IDENTITY_TOLERANCE, METHODS, SOLVERS, TERM, WIDTH,
    expansion_columns, fit_all, identity_discrepancy, identity_fixed_fusion, is_declared_failure)
from spectral_utils.varentropy_contribution_fusion import contributions
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.fixed_gate_readout import STREAM_NAMES

OUT = ROOT / 'results/varentropy_expansion_fusion_v1'
VAR_STREAM = STREAM_NAMES.index('topk_varentropy_series')
HIST_REFERENCE = {'B1_hist__raw': 'k15__raw', 'B1_hist__equal': 'k15__equal', 'B1_hist__iu': 'k15__iu'}
SOLVER_NAMES = {'equal_identity': 'identity-sign equal weights, anchor-oriented', 'equal_oriented': 'anchor-sign equal weights',
                'iu': 'IU-PCR', 'shrink': 'joint-target shrinkage IU', 'joint': 'Joint L-SML model-inverse map'}
BANK_NAMES = {'B2d_sel': 'D+Pii+selected (33)', 'B2_sel': 'D+Pii+Pij+selected (138)', 'B2d': 'D+Pii (30)', 'B2': 'D+Pii+Pij (135)'}
NAMES = {'B1_hist__raw': 'Top-15 varentropy / Original sum (Step 339 bank)',
         'B1_hist__equal': 'Top-15 contributions / equal weights (Step 339 bank)',
         'B1_hist__iu': 'Top-15 contributions / IU-PCR (Step 339 bank)',
         'entropy': 'Token entropy reference', 'direct_iu': 'Direct probability IU-PCR (17 inputs)',
         'delta_iu': 'Direct probability + change IU-PCR (34 inputs)',
         'ref__k15__raw': 'Frozen Step 339 k15 raw', 'ref__k15__equal': 'Frozen Step 339 k15 equal',
         'ref__k15__iu': 'Frozen Step 339 k15 IU', 'ref__k50__raw': 'Frozen Step 339 k50 raw'}
NAMES.update({f'{b}__{s}': f'{BANK_NAMES[b]} / {SOLVER_NAMES[s]}' for b in BANKS for s in SOLVERS})
PRIMARY = {('B2_sel__iu', 'B2d_sel__iu')}
# Shared-machine resource rule (main agent, 2026-09-12): free physical memory required before loading a cell pickle.
MEMORY_GATE_KB = {'pb': 1_200_000, 'prm': 2_000_000}


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


def clean(value):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def free_physical_kb():
    try:
        import ctypes
        class Status(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile',
                                                  'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
        s = Status(); s.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return int(s.ullAvailPhys // 1024)
    except Exception:
        return None


def wait_for_memory(min_kb, max_minutes=90, poll_minutes=5):
    """Block until free physical memory exceeds ``min_kb`` (resource rule for shared machine)."""
    waited = 0
    while True:
        free = free_physical_kb()
        if free is None or free > min_kb: return free
        if waited >= max_minutes: raise RuntimeError(f'BLOCKED: free memory {free} KB < {min_kb} KB after {waited} minutes')
        print(f'[memory] {free} KB free < gate {min_kb}; waiting {poll_minutes} min', flush=True)
        time.sleep(60 * poll_minutes); waited += poll_minutes


def default_roots(source):
    return dict(v2=source / '.worktrees/direct-probability-fusion-v2',
                temporal=source / '.worktrees/direct-probability-temporal-v3',
                varentropy=source / '.worktrees/varentropy-contribution-fusion-v1')


def comparisons():
    bank = [('B2_sel__iu', 'B2d_sel__iu')] + [(f'B2_sel__{x}', f'B2d_sel__{x}') for x in ('equal_identity', 'equal_oriented', 'shrink', 'joint')]
    bank += [('B2__iu', 'B2d__iu'), ('B2_sel__iu', 'B2__iu'), ('B2d_sel__iu', 'B2d__iu'), ('B2_sel__iu', 'B1_hist__iu'), ('B2d_sel__iu', 'B1_hist__iu')]
    arch = [(f'{b}__{s}', f'{b}__iu') for b in ('B2_sel', 'B2d_sel') for s in ('joint', 'shrink')]
    arch += [(f'{b}__iu', f'{b}__equal_identity') for b in ('B2_sel', 'B2d_sel')]
    ref = [(m, 'entropy') for m in METHODS] + [(m, 'B1_hist__raw') for m in METHODS if m != 'B1_hist__raw']
    return [('bank', a, b) for a, b in bank] + [('architecture', a, b) for a, b in arch] + [('reference', a, b) for a, b in ref]


def worker(task):
    i, uid, lp, chosen, spans, expected50, expected_hist = task
    started = time.perf_counter()
    fits, failures, seconds = fit_all(lp, chosen, uid=uid)
    replay = contributions(lp, 50).sum(axis=1)
    np.testing.assert_allclose(replay, expected50, atol=1e-12, rtol=1e-10, err_msg=f'{uid}: frozen raw Varentropy50 mismatch')
    X, _, _ = expansion_columns(lp, chosen)
    identity = identity_discrepancy(X, lp)
    if not identity <= IDENTITY_TOLERANCE: raise AssertionError(f'{uid}: identity discrepancy {identity}')
    k15_steps = step_top_mean(contributions(lp, 15).sum(axis=1), spans[:, 0], spans[:, 1], count=10)
    fixed_steps = step_top_mean(identity_fixed_fusion(X), spans[:, 0], spans[:, 1], count=10)
    fixed_error = float(np.max(np.abs(fixed_steps - k15_steps)))
    np.testing.assert_allclose(fixed_steps, k15_steps, atol=IDENTITY_TOLERANCE, rtol=0, err_msg=f'{uid}: identity-weighted fusion != k15 raw')
    S = np.full((len(spans), len(METHODS)), np.nan); W = np.full((len(METHODS), WIDTH), np.nan); E = W.copy()
    intercepts = np.full(len(METHODS), np.nan); diag = {}
    for j, m in enumerate(METHODS):
        if m not in fits: continue
        f = fits[m]; S[:, j] = step_top_mean(f['score'], spans[:, 0], spans[:, 1], count=10)
        k = len(f['weights']); W[j, :k] = f['weights']; E[j, :k] = f['effective']; intercepts[j] = f['intercept']
        diag[m] = f['diagnostics']
    hist_error = {}
    for m, ref in HIST_REFERENCE.items():
        j = METHODS.index(m)
        if m in fits:
            hist_error[m] = float(np.max(np.abs(S[:, j] - expected_hist[ref])))
            np.testing.assert_allclose(S[:, j], expected_hist[ref], atol=1e-10, rtol=0, err_msg=f'{uid}: {m} drifted from Step 339')
        else:
            hist_error[m] = None
            if not np.isnan(expected_hist[ref]).all(): raise AssertionError(f'{uid}: {m} failed here but Step 339 scored it')
    info = dict(uid=uid, n_tokens=int(len(lp)), n_steps=int(len(spans)), failures=failures, seconds=seconds,
                wall_seconds=time.perf_counter() - started, diagnostics=diag,
                raw50_max_error=float(np.max(np.abs(replay - expected50))), identity_max_discrepancy=identity,
                identity_fusion_vs_k15raw_max_error=fixed_error, hist_replay_max_error=hist_error)
    return i, base.packed(steps=S, weights=W, effective=E, intercepts=intercepts), base.dumps(clean(info))


def manifest_for(source, roots):
    manifest = base.input_manifest(source, roots['v2'])
    from importlib import metadata
    manifest['packages'] = {n: metadata.version(n) for n in ('numpy', 'scipy', 'scikit-learn')}
    manifest.update(schema='varentropy-expansion-fusion-v1', methods=list(METHODS), width=WIDTH,
                    roots={k: str(v) for k, v in roots.items()})
    files = [Path(__file__), ROOT / 'spectral_utils/varentropy_expansion_fusion.py',
             ROOT / 'spectral_utils/varentropy_contribution_fusion.py', ROOT / 'spectral_utils/joint_lsml.py',
             ROOT / 'spectral_utils/dependency_fusion.py', ROOT / 'spectral_utils/direct_probability_temporal.py',
             ROOT / 'spectral_utils/fixed_gate_readout.py', ROOT / 'scripts/test_varentropy_expansion_fusion.py',
             ROOT / 'docs/experiments/VARENTROPY_EXPANSION_FUSION_V1.md',
             roots['temporal'] / 'results/direct_probability_temporal_v3/SCORES.npz',
             roots['temporal'] / 'results/direct_probability_temporal_v3/METRICS.json',
             roots['varentropy'] / 'results/varentropy_contribution_fusion_v1/SCORES.npz',
             roots['varentropy'] / 'results/varentropy_contribution_fusion_v1/METRICS.json']
    for cell, _, _, _ in base.source_specs():
        files += [base.old.BENCH / 'inputs' / cell / (name + '.npy') for name in ('raw', 'token_offsets', 'row_ids')]
    for path in files: manifest['hashes'][str(path)] = base.old.sha256_file(path)
    return manifest


def smoke_indices(all_indices, rows, records):
    order = sorted(all_indices, key=lambda i: len(rows[records[i]['row_id']]['token_entropies']))
    return [order[j] for j in sorted({0, len(order) // 2, min(len(order) - 1, int(.95 * len(order)))})]


def score(con, records, joined, roots, *, workers, smoke, max_answers, gates):
    done = {r[0] for r in con.execute('SELECT idx FROM answers')}; started = time.perf_counter(); processed = 0
    detector, _ = base.old._gate_contract(records); offsets = joined['offsets']
    with np.load(roots['varentropy'] / 'results/varentropy_contribution_fusion_v1/SCORES.npz') as z:
        hist = {ref: z['steps__' + ref] for ref in HIST_REFERENCE.values()}
    with ProcessPoolExecutor(max_workers=workers, initializer=base.worker_init) as pool:
        for cell, path, kind, dataset in base.source_specs():
            all_indices = [i for i, r in enumerate(records) if r['cell'] == cell]
            indices = [i for i in all_indices if i not in done]
            if not indices: continue
            wait_for_memory(gates[kind])
            print('[load]', cell, len(indices), 'remaining', flush=True)
            rows = base.old._source_row_map(base.old.load_pickle(path), kind=kind, dataset=dataset)
            d = base.old.BENCH / 'inputs' / cell
            raw = np.load(d / 'raw.npy', mmap_mode='r'); to = np.load(d / 'token_offsets.npy')
            lookup = {str(v): j for j, v in enumerate(np.load(d / 'row_ids.npy', allow_pickle=True))}
            if smoke: indices = [i for i in smoke_indices(all_indices, rows, records) if i not in done]
            batch_size = 4 * max(workers, 1)      # keep every worker busy; checkpoint after each batch
            for start in range(0, len(indices), batch_size):
                batch = []
                for i in indices[start:start + batch_size]:
                    r = records[i]; row = rows[r['row_id']]
                    lp = np.asarray(base.old._topk_payload(row)['logprobs'], float)
                    entropy = np.asarray(row['token_entropies'], float); spans = np.asarray(row['step_token_spans'], int)
                    if lp.shape != (len(entropy), 50): raise ValueError(f'{r["uid"]}: expected T x 50')
                    if spans.shape != (r['steps'], 2): raise ValueError(f'{r["uid"]}: step count mismatch')
                    with np.load(base.old.BENCH / 'scores' / f'{r["uid"]}.npz') as z:
                        np.testing.assert_array_equal(spans[:, 0], z['step_starts']); np.testing.assert_array_equal(spans[:, 1], z['step_ends'])
                    if kind == 'pb': np.testing.assert_allclose(entropy.mean(), detector[i], atol=1e-12, rtol=0)
                    j = lookup[r['row_id']]; expected = np.asarray(raw[to[j]:to[j + 1], VAR_STREAM], float)
                    if len(expected) != len(lp): raise ValueError(f'{r["uid"]}: frozen token count mismatch')
                    expected_hist = {ref: hist[ref][offsets[i]:offsets[i + 1]] for ref in hist}
                    batch.append((i, r['uid'], lp, np.asarray(row['token_spilled_energies'], float), spans, expected, expected_hist))
                for item in pool.map(worker, batch, chunksize=1): con.execute('INSERT INTO answers VALUES (?,?,?)', item)
                con.commit(); done.update(item[0] for item in batch); processed += len(batch)
                state = dict(status='SMOKE' if smoke else 'RUNNING', completed=len(done), expected=len(records), last_cell=cell,
                             elapsed_seconds=time.perf_counter() - started, workers=workers)
                base.atomic_json(OUT / ('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'), state)
                print('[checkpoint]', len(done), '/', len(records), round(state['elapsed_seconds'], 1), 'seconds', flush=True)
                if max_answers and processed >= max_answers: return
            del rows, raw
    if not smoke and len(done) == len(records):
        base.atomic_json(OUT / 'RUN_STATE.json', dict(status='SCORING_COMPLETE', completed=len(done), expected=len(records)))


def _stats(values):
    v = np.asarray([x for x in values if x is not None], float); v = v[np.isfinite(v)]
    if not len(v): return None
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def fit_health(infos):
    """Aggregate per-arm diagnostics from checkpoint info rows (smoke or full)."""
    health = {}
    for m in METHODS:
        rows = [r['diagnostics'][m] for r in infos if m in r['diagnostics']]
        reasons = Counter(r['failures'][m].split(':')[0] + ':' + r['failures'][m].split(':', 1)[1][:80] for r in infos if m in r['failures'])
        h = dict(fits=len(rows), failures=int(sum(reasons.values())), failure_reasons=dict(reasons),
                 undeclared_failures=int(sum(1 for r in infos if m in r['failures'] and not is_declared_failure(r['failures'][m]))),
                 seconds=_stats([r['seconds'].get(m) for r in infos]),
                 orientation_flip_rate=float(np.mean([bool(d.get('orientation_flipped')) for d in rows])) if rows else None,
                 anchor_correlation=_stats([d.get('anchor_correlation') for d in rows]),
                 active_columns=_stats([d.get('active_columns') for d in rows]),
                 tiny_scale_columns=_stats([d.get('tiny_scale_columns') for d in rows]),
                 n_lt_p_fits=int(sum(d.get('n_tokens', 0) < d.get('active_columns', 0) for d in rows)) if rows and 'n_tokens' in rows[0] else None)
        if m.endswith('__joint') and rows:
            h.update(converged_rate=float(np.mean([d['converged'] for d in rows])),
                     model_covariance_condition_over_1e12_rate=float(np.mean([d['model_covariance_condition'] > 1e12 for d in rows])),
                     multistart_pass_rate=float(np.mean([d['multistart_status'] == 'PASS' for d in rows])),
                     selected_start_sweeps=_stats([d['selected_start_sweeps'] for d in rows]),
                     joint_relative_offdiag_misfit=_stats([d['joint_relative_offdiag_misfit'] for d in rows]),
                     hard_relative_offdiag_misfit=_stats([d['hard_relative_offdiag_misfit'] for d in rows]),
                     joint_lower_misfit_rate=float(np.mean([d['joint_lower_misfit'] for d in rows])),
                     model_covariance_condition=_stats([d['model_covariance_condition'] for d in rows]),
                     map_ridge=_stats([d['map_ridge'] for d in rows]), map_condition_after=_stats([d['map_condition_after'] for d in rows]),
                     jacobian_full_global_rank_rate=float(np.mean([d['jacobian_full_global_rank'] for d in rows])),
                     jacobian_condition=_stats([d['jacobian_condition'] for d in rows]),
                     global_loading_cosine_min=_stats([d['global_loading_cosine_min'] for d in rows]),
                     failed_monotonicity_starts=_stats([d['failed_monotonicity_starts'] for d in rows]))
        if m.endswith('__shrink') and rows:
            h.update(alpha=_stats([d['alpha'] for d in rows]), alpha_clipped_rate=float(np.mean([d['alpha'] >= 1.0 for d in rows])))
        if (m.endswith('__iu') or m.endswith('__shrink')) and rows and 'analytic_pair_path' in rows[0]:
            h.update(analytic_pair_path_rate=float(np.mean([d['analytic_pair_path'] for d in rows])))
        health[m] = h
    return health


def feasibility(infos, workers, elapsed):
    n = len(infos); wall = sum(r['wall_seconds'] for r in infos)
    per_arm = {m: dict(sum_seconds=float(sum(r['seconds'].get(m, 0.) for r in infos)), mean_seconds=float(np.mean([r['seconds'].get(m, 0.) for r in infos])),
                       max_seconds=float(max(r['seconds'].get(m, 0.) for r in infos))) for m in METHODS + ('bank_build',)}
    projected = wall / n * 13769 / max(workers, 1) if n else None
    return dict(scope='FEASIBILITY_ONLY', n_answers=n, workers=workers, launch_elapsed_seconds=elapsed, fit_wall_seconds_total=wall,
                mean_wall_seconds_per_answer=wall / n if n else None,
                projected_full_fit_seconds=projected, projected_full_fit_hours=projected / 3600 if projected else None,
                projected_full_fit_hours_excluding_joint=(sum(v['sum_seconds'] for m, v in per_arm.items() if not m.endswith('__joint')) / n * 13769 / max(workers, 1) / 3600) if n else None,
                per_arm_seconds=per_arm, identity_max_discrepancy=max(r['identity_max_discrepancy'] for r in infos),
                identity_fusion_vs_k15raw_max_error=max(r['identity_fusion_vs_k15raw_max_error'] for r in infos),
                raw50_max_error=max(r['raw50_max_error'] for r in infos),
                hist_replay_max_error={m: max((r['hist_replay_max_error'][m] or 0.) for r in infos) for m in HISTORICAL},
                tokens=_stats([r['n_tokens'] for r in infos]), steps=_stats([r['n_steps'] for r in infos]),
                fit_health=fit_health(infos))


def evaluate(con, records, joined, roots):
    base.METHODS = METHODS
    scores, telemetry = base.load_scored(con, records, joined['offsets'])
    infos = [json.loads(r[0]) for r in con.execute('SELECT info FROM answers ORDER BY idx')]
    previous = json.loads((roots['varentropy'] / 'results/varentropy_contribution_fusion_v1/METRICS.json').read_text(encoding='utf8'))
    with np.load(roots['varentropy'] / 'results/varentropy_contribution_fusion_v1/SCORES.npz') as z:
        for m in ('k15__raw', 'k15__equal', 'k15__iu', 'k50__raw'): scores['ref__' + m] = z['steps__' + m]
    oldmetrics = json.loads((roots['temporal'] / 'results/direct_probability_temporal_v3/METRICS.json').read_text(encoding='utf8'))
    with np.load(roots['temporal'] / 'results/direct_probability_temporal_v3/SCORES.npz') as z:
        for a, b in [('entropy', 'entropy'), ('direct_iu', 'current__iu'), ('delta_iu', 'delta__iu')]: scores[a] = z['steps__' + b]
    metrics, per = base.evaluate_arrays(records, joined, scores)
    keys = ('pb_all8', 'pb_q4', 'pb_q8', 'prm_within', 'prm_pooled', 'prmscore_q08')
    for m in ('k15__raw', 'k15__equal', 'k15__iu', 'k50__raw'):
        for key in keys: np.testing.assert_allclose(metrics['ref__' + m][key], previous['metrics'][m][key], atol=1e-12, rtol=0, err_msg=m + ' ' + key)
    for m, ref in HIST_REFERENCE.items():
        for key in keys: np.testing.assert_allclose(metrics[m][key], previous['metrics'][ref][key], atol=1e-11, rtol=0, err_msg=m + ' ' + key)
        np.testing.assert_allclose(scores[m], scores['ref__' + ref], atol=1e-10, rtol=0, err_msg=m + ' step scores')
    for a, b in [('entropy', 'entropy'), ('direct_iu', 'current__iu'), ('delta_iu', 'delta__iu')]:
        for key in ('pb_all8', 'prm_within', 'prm_pooled', 'prmscore_q08'):
            np.testing.assert_allclose(metrics[a][key], oldmetrics['metrics'][b][key], atol=1e-12, rtol=0)
    print('[evaluate] Step 339 and temporal-v3 references reproduced; bootstrap 10000', flush=True)
    table = comparisons(); pairs = [(a, b) for _, a, b in table]
    contrasts = base.paired_bootstrap(records, joined, per, pairs=pairs, primary_pairs=PRIMARY, draws=10000, primary_ci=.975)
    pb = np.array([r['cell'].startswith('pb_') for r in records]); target = joined['target']; cases = []
    for tab, a, b in table:
        c = contrasts[a + '_minus_' + b]; c.update(table=tab, pb_delta=metrics[a]['pb_all8'] - metrics[b]['pb_all8'])
        for key in ('prm_within', 'prm_pooled', 'prmscore_q08'):
            c[key + '_delta'] = (metrics[a][key] - metrics[b][key]) if metrics[a][key] is not None and metrics[b][key] is not None else None
        oldhit = pb & per[b]['decision_valid'] & (per[b]['prediction'] == target)
        newhit = pb & per[a]['decision_valid'] & (per[a]['prediction'] == target)
        c.update(gained=int((newhit & ~oldhit).sum()), lost=int((oldhit & ~newhit).sum()))
        cases += [dict(comparison=a + '_minus_' + b, table=tab, uid=records[i]['uid'], cell=records[i]['cell'], target=int(target[i]),
                       before=int(per[b]['prediction'][i]), after=int(per[a]['prediction'][i]), change='gained' if newhit[i] else 'lost')
                  for i in np.flatnonzero(oldhit ^ newhit)]
    weights, failures, coverage = {}, {}, {}
    for m, t in telemetry.items():
        Wm = np.stack(t.pop('weights')) if t['weights'] else np.full((0, WIDTH), np.nan)
        width = 15 if m in HISTORICAL else len(BANK_COLUMNS[m.split('__')[0]]); Wm = Wm[:, :width]
        total = np.nansum(np.abs(Wm), axis=1, keepdims=True); share = np.divide(np.abs(Wm), total, out=np.zeros_like(Wm), where=total > 0)
        terms = ['c'] * 15 if m in HISTORICAL else [TERM[c] for c in BANK_COLUMNS[m.split('__')[0]]]
        weights[m] = dict(n=int(len(Wm)), mean_standardized_coefficients=np.nanmean(Wm, axis=0).tolist() if len(Wm) else None,
                          mean_absolute_share=share.mean(axis=0).tolist() if len(Wm) else None,
                          term_absolute_share={t: float(share[:, [k for k, x in enumerate(terms) if x == t]].sum(axis=1).mean()) for t in dict.fromkeys(terms)} if len(Wm) else None,
                          note='B1_hist raw coefficients act on unnormalized contributions; all other arms on answer-standardized active columns.')
        t.pop('alphas', None)
        failures[m] = t['failures']
        coverage[m] = dict(valid_answers=metrics[m]['valid_answers'], pb_valid=int(metrics[m]['valid_answers'] - metrics[m]['prm_valid_answers']),
                           prm_valid=metrics[m]['prm_valid_answers'], prm_within_n=metrics[m]['prm_within_n'], pb_invalid=metrics[m]['pb_invalid'],
                           failures=len(t['failures']), undeclared_failures=int(sum(1 for f in t['failures'] if not is_declared_failure(f['reason']))))
    payload = dict(schema='varentropy-expansion-fusion-v1', n_answers=len(records), n_steps=int(joined['offsets'][-1]),
                   scope='Full cached localization development; fusion answer-local, gate/calibration external; no historical24 result.',
                   primary=[list(p) for p in PRIMARY], metrics=metrics, contrasts=contrasts, telemetry=telemetry, weights=weights,
                   failures=failures, coverage=coverage, historical_references=previous['historical_references'],
                   mind_gap_reference=previous['mind_gap_reference'])
    base.atomic_json(OUT / 'METRICS.json', clean(payload)); base.atomic_json(OUT / 'FIT_HEALTH.json', clean(fit_health(infos)))
    np.savez_compressed(OUT / 'SCORES.npz', **{'steps__' + m: s for m, s in scores.items()},
                        **{'prediction__' + m: p['prediction'] for m, p in per.items()}, **{'valid__' + m: p['valid'] for m, p in per.items()})
    with (OUT / 'COMPARISON.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['table', 'comparison', 'a', 'b', 'primary', 'ci_level', 'pb_delta', 'pb_ci_lo', 'pb_ci_hi', 'prm_within_delta_common',
                                       'prm_within_ci_lo', 'prm_within_ci_hi', 'common_prm_answers', 'prm_pooled_delta', 'prmscore_delta', 'gained', 'lost'])
        for tab, a, b in table:
            c = contrasts[a + '_minus_' + b]; wci = c['prm_within_ci'] or [None, None]
            w.writerow([tab, a + '_minus_' + b, a, b, c['primary'], c['ci_level'], c['pb_delta'], c['pb_ci'][0], c['pb_ci'][1], c['prm_within_delta_common'],
                        wci[0], wci[1], c['common_prm_answers'], c['prm_pooled_delta'], c['prmscore_q08_delta'], c['gained'], c['lost']])
    cells = sorted(metrics['B1_hist__raw']['pb_cells'])
    with (OUT / 'PB_CELLS.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['method_id', 'method', 'PB_macro_percent', 'PB_Q4_percent', 'PB_Q8_percent'] + cells)
        for m, x in metrics.items(): w.writerow([m, NAMES.get(m, m), 100 * x['pb_all8'], 100 * x['pb_q4'], 100 * x['pb_q8']] + [100 * x['pb_cells'][c]['f1'] for c in cells])
    with (OUT / 'CHANGED_SUCCESSES.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['comparison', 'table', 'uid', 'cell', 'target', 'before', 'after', 'change']); w.writeheader(); w.writerows(cases)
    rows = []
    for m, x in metrics.items():
        rows.append(dict(method=NAMES.get(m, m), method_id=m, PB_macro_percent=100 * x['pb_all8'], PB_Q4_percent=100 * x['pb_q4'], PB_Q8_percent=100 * x['pb_q8'],
                         PRMB_within_AUC=x['prm_within'], PRMB_within_n=x['prm_within_n'], PRMB_pooled_AUC=x['prm_pooled'], PRMScore=x['prmscore_q08'],
                         PRMScore_conditional=x['prmscore_conditional'], valid_answers=x['valid_answers'], pb_invalid=x['pb_invalid'],
                         fit_seconds=telemetry[m]['fit_seconds'] if m in telemetry else None))
        print(m, x['pb_all8'], x['prm_within'], x['prm_pooled'], x['prmscore_q08'], x['valid_answers'], flush=True)
    with (OUT / 'SUMMARY.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    base.atomic_json(OUT / 'RUN_STATE.json', dict(status='COMPLETE', completed=len(records), expected=len(records)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    for a in ('v2-root', 'temporal-root', 'varentropy-root'): p.add_argument('--' + a, type=Path, default=None)
    p.add_argument('--workers', type=int, default=1); p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-answers', type=int, default=0); p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--memory-gate-pb-kb', type=int, default=MEMORY_GATE_KB['pb']); p.add_argument('--memory-gate-prmb-kb', type=int, default=MEMORY_GATE_KB['prm'])
    args = p.parse_args(); source = args.source_root.resolve()
    roots = default_roots(source)
    for k in roots:
        override = getattr(args, k + '_root')
        if override is not None: roots[k] = override.resolve()
    base.old.configure_source_root(source); OUT.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for(source, roots)
    con = base.connect(OUT / ('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'), manifest)
    base.atomic_json(OUT / ('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'), manifest)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    if len(records) != 13769 or len({r['uid'] for r in records}) != 13769: raise ValueError('benchmark roster mismatch')
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if not args.evaluate_only:
            score(con, records, joined, roots, workers=args.workers, smoke=args.smoke, max_answers=args.max_answers, gates={'pb': args.memory_gate_pb_kb, 'prm': args.memory_gate_prmb_kb})
        n = con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
        if args.smoke:
            infos = [json.loads(r[0]) for r in con.execute('SELECT info FROM answers ORDER BY idx')]
            failures = [dict(uid=r['uid'], method=m, reason=reason, declared=is_declared_failure(reason)) for r in infos for m, reason in r['failures'].items()]
            status = 'PASS' if all(f['declared'] for f in failures) else 'FAIL'
            base.atomic_json(OUT / 'SMOKE.json', dict(status=status, scope='FEASIBILITY_ONLY', n_answers=n, failures=failures,
                                                      undeclared_failures=sum(not f['declared'] for f in failures), rows=infos))
            base.atomic_json(OUT / 'FEASIBILITY.json', clean(feasibility(infos, args.workers, time.perf_counter() - started)))
            base.atomic_json(OUT / 'SMOKE_STATE.json', dict(status='SMOKE_' + status, completed=n, expected=len(records)))
            print('[smoke]', status, n, 'rows; failures:', len(failures), 'undeclared:', sum(not f['declared'] for f in failures), flush=True)
        elif n == len(records):
            evaluate(con, records, joined, roots)
        else:
            print('[scoring]', n, '/', len(records), 'answers scored; evaluation waits for completion', flush=True)
    con.close()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as error:
        name = 'SMOKE_STATE.json' if '--smoke' in sys.argv else 'RUN_STATE.json'
        path = OUT / name; state = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=f'{type(error).__name__}: {error}')
        base.atomic_json(path, state)
        raise
