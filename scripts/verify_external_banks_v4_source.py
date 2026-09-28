"""Full raw-source parity gate for the external_banks_v4 channels; no labels, no quality.

Recomputes, for ALL 13,769 development answers, the five NEW raw step channels
(realized_z, realized_drv, ct7_ve1, hist_entropy_series, hist_spilled_series)
and the three digit channels (+ availability mask) from the raw source pickles
listed in the passed 48-channel family gate, and compares them with the frozen
development arrays:

  realized_z           ct7_profiles_v1/profiles.npy column 6 (raw = definitional value)
  realized_drv         DERIVATIVE_CHANNELS.npz 'derivative' column chosen_surprisal
  ct7_ve1              union_top10_profiles.npy column 14 (raw Top10) AND pool_z.npy (answer-z)
  hist_entropy_series  hist29_top10_profiles.npy columns 1/15 (raw Top10) AND pool_z.npy (answer-z)
  hist_spilled_series
  digit_*              digit_family_extension_v1/FEATURES.npz values + active (exact mask)

Answer/step order: localization_full_benchmark_v3 JOINED.json, whose offsets are
asserted equal to the OOF_STEP_SCORES.npz offsets (only 'offsets' is read).
Tolerance: 1e-6 max abs error per channel over all steps; the digit mask must
be identical. Recomputed per-cell arrays are saved next to GATE.json so the
source-side combined bank can be assembled from one consistent code path.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils import external_banks_v4 as E  # noqa: E402
from spectral_utils.family_tail_transfer import answer_standardize  # noqa: E402
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash  # noqa: E402

TOL = 1e-6
POOL_DIR = Path(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/'
                r'2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')
FAMILY_GATE = MAIN/'results/family_tail_external_v1/source_full_v1/GATE.json'
ROSTER = MAIN/'results/localization_full_benchmark_v3/evaluation/JOINED.json'
OOF = MAIN/'.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_STEP_SCORES.npz'
CT7_PROFILES = MAIN/'.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
DERIVATIVE = MAIN/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz'
DIGITS = ROOT/'results/digit_family_extension_v1'
POOL_SHA = 'd9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16'
POOL_CHANNELS = ('ct7_ve1', 'hist_entropy_series', 'hist_spilled_series')
PRODUCERS = {
    'realized_z': '.worktrees/cumulative-vote-fusion-v2/scripts/experiments/cvf_v2/ct7.py prepare() -> '
                  'frozen_locator_ct7.despiked_chosen_token_z over chosen_token_calibration_v1/extracted_sufficient '
                  '(scripts/run_chosen_token_calibration_steps_v2.py)',
    'realized_drv': '.worktrees/token-probability-fusion-v1/scripts/diagnostics/extract_derivative_channels_v1.py '
                    '(derivative_step_readout of claude_feature_bank_v1.build_token_feature_matrix, float64)',
    'ct7_ve1': 'scratchpad union_structure.py (CT7_TOKEN_MATRICES Top10) -> pool_structure.py (answer-z)',
    'hist_entropy_series': 'scratchpad hist29_align.py (localization_full_benchmark_v3 raw.npy Top10) -> pool_structure.py',
    'hist_spilled_series': 'scratchpad hist29_align.py (localization_full_benchmark_v3 raw.npy Top10) -> pool_structure.py',
    'digits': 'scripts/run_digit_family_extension_v1.py extract',
}
CODE = ['spectral_utils/external_banks_v4.py', 'scripts/verify_external_banks_v4_source.py',
        'spectral_utils/digit_feature_family.py', 'spectral_utils/digit_alternative_probability.py',
        'spectral_utils/family_external_features.py', 'spectral_utils/family_hist_features.py',
        'spectral_utils/family_tail_transfer.py', 'spectral_utils/external_generalization/ct7.py',
        'spectral_utils/external_generalization/fusion.py',
        'spectral_utils/external_generalization/_bank11/chosen_token_calibration.py',
        'spectral_utils/external_generalization/_bank11/claude_feature_bank_v1.py',
        'spectral_utils/external_generalization/_bank11/token_feature_views.py',
        'spectral_utils/external_generalization/_bank11/feature_utils.py',
        'spectral_utils/external_generalization/_bank11/temporal_models.py']


def safe_row(row):
    safe = {key: row[key] for key in ('gen_token_ids', 'token_entropies', 'token_spilled_energies',
                                      'token_logsumexp', 'step_token_spans')}
    safe['top_k_logprobs'] = row.get('top_k_logprobs') or row.get('top_k_logprobs_raw')
    return safe


def one(item):
    i, row = item
    started = time.process_time()
    matrix, _ = E.new_step_features(row)
    digits, active = E.digit_step_features(row)
    return i, matrix, digits, active, time.process_time()-started


def fill_like_pool(matrix):
    """Same missing-value rule as family_external_features.step_features / pool_structure.py."""
    x = np.array(matrix, float)
    for j in range(x.shape[1]):
        good = np.isfinite(x[:, j])
        x[~good, j] = x[good, j].mean() if good.any() else 0.
    return x


def nan_error(got, expected):
    """Max abs error per column; NaN on both sides is agreement, NaN on one side is infinite."""
    got = np.asarray(got, float); expected = np.asarray(expected, float)
    both = np.isnan(got) & np.isnan(expected)
    delta = np.abs(got-expected)
    delta[both] = 0.
    delta[np.isnan(delta)] = np.inf
    return delta.max(axis=0)


def source_specs(gate):
    for cell, item in gate['sources'].items():
        dataset = cell.split('_')[1] if cell.startswith('pb_') else None
        yield cell, Path(item['path']), item['sha256'], dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-dir', type=Path, default=POOL_DIR)
    parser.add_argument('--output', type=Path, default=ROOT/'results/external_banks_v4/source_parity')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit-per-cell', type=int, default=0)
    args = parser.parse_args()
    if (args.output/'GATE.json').exists():
        raise FileExistsError('immutable gate already written: '+str(args.output/'GATE.json'))
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    family_gate = json.loads(FAMILY_GATE.read_text())
    assert family_gate['status'] == 'PASS' and family_gate['n_checked'] == 13769 and family_gate['scope'] == 'FULL'
    roster = json.loads(ROSTER.read_text())['records']
    off = np.r_[0, np.cumsum([r['steps'] for r in roster])]
    with np.load(OOF) as z:
        oof_off = z['offsets']
    assert len(roster) == 13769 and off[-1] == 145597 and np.array_equal(off, oof_off)

    # Source arrays (all in JOINED order) with provenance hashes.
    validation = json.loads((CT7_PROFILES/'PROFILE_VALIDATION.json').read_text())
    assert validation['channels'][6] == 'chosen_token_z_despiked'
    assert file_hash(CT7_PROFILES/'profiles.npy') == validation['profile_sha256']
    profiles = np.load(CT7_PROFILES/'profiles.npy')
    with np.load(DERIVATIVE) as z:
        assert list(z['channels'].astype(str)) == list(E.BANK11_NAMES)
        derivative = z['derivative'][:, list(E.BANK11_NAMES).index('chosen_surprisal')].copy()
    union = np.load(args.pool_dir/'union_top10_profiles.npy', mmap_mode='r')
    hist29 = np.load(args.pool_dir/'hist29_top10_profiles.npy', mmap_mode='r')
    hist_names = json.loads((args.pool_dir/'hist29_names.json').read_text())
    assert union.shape == (145597, 18) and hist29.shape == (145597, 29)
    raw_ref = np.column_stack([profiles[:, 6], derivative, union[:, 11+3],
                               hist29[:, hist_names.index('entropy_series')],
                               hist29[:, hist_names.index('spilled_series')]])
    poolpath = args.pool_dir/'pool_z.npy'
    assert file_hash(poolpath) == POOL_SHA
    pool = np.load(poolpath, mmap_mode='r')
    pool_names = json.loads((args.pool_dir/'pool_names.json').read_text())
    z_ref_cols = [pool_names.index(name) for name in POOL_CHANNELS]
    digit_extraction = json.loads((DIGITS/'EXTRACTION.json').read_text())
    assert file_hash(DIGITS/'FEATURES.npz') == digit_extraction['sha256']
    with np.load(DIGITS/'FEATURES.npz') as z:
        digit_ref, digit_active_ref = z['values'], z['active']
        assert np.array_equal(z['offsets'], off)
    vendored = {name: {**item, 'current_sha256': file_hash(MAIN/item['path'])}
                for name, item in E.VENDORED_SOURCES.items()}
    assert all(v['sha256'] == v['current_sha256'] for v in vendored.values()), vendored

    inputs = {str(p): file_hash(p) for p in (FAMILY_GATE, ROSTER, OOF, CT7_PROFILES/'profiles.npy',
              CT7_PROFILES/'PROFILE_VALIDATION.json', DERIVATIVE, args.pool_dir/'union_top10_profiles.npy',
              args.pool_dir/'hist29_top10_profiles.npy', args.pool_dir/'hist29_names.json', poolpath,
              args.pool_dir/'pool_names.json', DIGITS/'FEATURES.npz', DIGITS/'EXTRACTION.json')}
    report = {'status': 'RUNNING', 'version': E.VERSION, 'n_total': len(roster), 'tolerance': TOL,
              'channels': list(E.NEW_NAMES), 'digit_channels': list(E.DIGIT_NAMES),
              'z_checked_channels': list(POOL_CHANNELS), 'raw_value_convention':
              'realized_z is answer-standardized by definition; all other channels are raw step readouts',
              'external_quality_computed': False, 'labels_read': False, 'limit_per_cell': args.limit_per_cell,
              'command': ' '.join([sys.executable, *sys.argv]), 'workers': args.workers, 'producers': PRODUCERS,
              'vendored_sources': vendored, 'inputs_sha256': inputs,
              'code_sha256': {rel: file_hash(ROOT/rel) for rel in CODE}}
    atomic_json(args.output/'GATE.json.running', report)

    raw_max = np.zeros(5); z_max = np.zeros(3); digit_max = np.zeros(3); active_mismatch = 0
    nonfinite_new = 0; checked = 0; checked_steps = 0; cpu = 0.; failures = []; bycell = {}; sources = {}; saved = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for cell, path, expected_sha, dataset in source_specs(family_gate):
            digest = file_hash(path)
            assert digest == expected_sha, path
            sources[cell] = {'path': str(path), 'sha256': digest}
            with path.open('rb') as stream:
                payload = pickle.load(stream)
            rows = list(payload.values()) if isinstance(payload, dict) else payload
            lookup = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = f'{dataset}::{row.get("id")}' if dataset else str(row.get('idx'))
                if key in lookup:
                    raise ValueError('duplicate raw ID '+key)
                lookup[key] = row
            indexes = [i for i, r in enumerate(roster) if r['cell'] == cell]
            if args.limit_per_cell:
                ordered = sorted(indexes, key=lambda i: (roster[i]['tokens'], roster[i]['uid']))
                indexes = sorted({ordered[int(j)] for j in np.linspace(0, len(ordered)-1, args.limit_per_cell)})
            results = {}; cell_raw = np.zeros(5); cell_z = np.zeros(3); cell_digit = np.zeros(3)

            def jobs():
                for i in indexes:
                    row = safe_row(lookup[str(roster[i]['row_id'])])
                    assert len(row['step_token_spans']) == roster[i]['steps']
                    assert len(row['gen_token_ids']) == roster[i]['tokens']
                    yield i, row
            iterator = iter(jobs()); pending = set(); exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < 3*args.workers:
                    try:
                        pending.add(executor.submit(one, next(iterator)))
                    except StopIteration:
                        exhausted = True
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i, matrix, digits, active, seconds = future.result()
                    a, b = off[i:i+2]
                    raw_delta = nan_error(matrix, raw_ref[a:b])
                    z = answer_standardize(fill_like_pool(matrix[:, 2:]), np.array([0, b-a]))
                    z_delta = np.abs(z-pool[a:b][:, z_ref_cols]).max(axis=0)
                    digit_delta = np.abs(digits-digit_ref[a:b]).max(axis=0)
                    mismatch = int((active != digit_active_ref[a:b]).sum())
                    nonfinite_new += int((~np.isfinite(matrix)).sum())
                    raw_max = np.maximum(raw_max, raw_delta); z_max = np.maximum(z_max, z_delta)
                    digit_max = np.maximum(digit_max, digit_delta); active_mismatch += mismatch
                    cell_raw = np.maximum(cell_raw, raw_delta); cell_z = np.maximum(cell_z, z_delta)
                    cell_digit = np.maximum(cell_digit, digit_delta)
                    checked += 1; checked_steps += int(b-a); cpu += seconds
                    bad = {**{n: float(x) for n, x in zip(E.NEW_NAMES, raw_delta) if not x <= TOL},
                           **{n+'__z': float(x) for n, x in zip(POOL_CHANNELS, z_delta) if not x <= TOL},
                           **{n: float(x) for n, x in zip(E.DIGIT_NAMES, digit_delta) if not x <= TOL}}
                    if mismatch:
                        bad['digit_active_mismatch'] = mismatch
                    if bad:
                        failures.append({'cell': cell, 'index': int(i), 'uid': roster[i]['uid'], 'errors': bad})
                        if len(failures) == 1:
                            np.savez_compressed(args.output/'FIRST_FAILURE.npz', got=matrix, expected=raw_ref[a:b],
                                                digits=digits, digit_expected=digit_ref[a:b], active=active,
                                                active_expected=digit_active_ref[a:b])
                        if len(failures) >= 50:
                            atomic_json(args.output/'GATE.json', dict(report, status='FAIL', n_checked=checked,
                                        failures=failures, sources=sources, aborted_after_failures=True))
                            raise AssertionError(failures[:3])
                    results[i] = (matrix, digits, active)
                    if checked % 250 == 0:
                        print('v4 source parity', checked, '/13769', cell,
                              'elapsed', round(time.perf_counter()-started), flush=True)
            order = sorted(results)
            dest = args.output/(cell+'.npz')
            np.savez_compressed(dest, indexes=np.asarray(order), names=np.asarray(E.NEW_NAMES),
                                digit_names=np.asarray(E.DIGIT_NAMES),
                                new=np.concatenate([results[i][0] for i in order]),
                                digits=np.concatenate([results[i][1] for i in order]),
                                digit_active=np.concatenate([results[i][2] for i in order]))
            saved.append({'path': str(dest), 'sha256': file_hash(dest)})
            bycell[cell] = {'answers': len(order), 'raw_max_error': dict(zip(E.NEW_NAMES, map(float, cell_raw))),
                            'z_max_error': dict(zip(POOL_CHANNELS, map(float, cell_z))),
                            'digit_max_error': dict(zip(E.DIGIT_NAMES, map(float, cell_digit)))}
            print('CELL DONE', cell, len(order), float(cell_raw.max()), float(cell_z.max()),
                  float(cell_digit.max()), 'failures so far', len(failures), flush=True)
            del payload, rows, lookup, results
            gc.collect()
    status = 'PASS' if not failures and checked else 'FAIL'
    report.update(status=status, n_checked=checked, steps=checked_steps,
                  scope='FULL' if checked == 13769 else 'FEASIBILITY',
                  per_channel_max_error=dict(zip(E.NEW_NAMES, map(float, raw_max))),
                  per_channel_max_error_answer_z_vs_pool=dict(zip(POOL_CHANNELS, map(float, z_max))),
                  digit_max_error=dict(zip(E.DIGIT_NAMES, map(float, digit_max))),
                  digit_active_mismatches=active_mismatch, nonfinite_new_values=nonfinite_new,
                  failures=failures, cells=bycell, sources=sources, recomputed=saved,
                  elapsed_seconds=time.perf_counter()-started, process_cpu_seconds=cpu)
    atomic_json(args.output/'GATE.json', report)
    os.remove(args.output/'GATE.json.running')
    print(status, checked, '/13769', 'raw', float(raw_max.max()), 'z', float(z_max.max()),
          'digits', float(digit_max.max()), 'active mismatches', active_mismatch, flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
