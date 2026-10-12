"""Matched OOF component oracles and all-zero control, evaluated after seal.

Truth count and error-first order are evaluation-label diagnostics. They are
not practical estimators, selectable candidates, additive gains or a general
mathematical upper bound on pooled PRMScore.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from spectral_utils.error_count_evaluation import (
    CATEGORIES, METHODS, TYPE_CODES, SourceBootstrap, authenticate_label_source,
    flags_from_shared_rank, interval, load_predictions, population_masks,
    summarize, validate_prediction_seal,
)
from evaluate_emission_calibration import sha, write_csv, write_json


def error_first_rank(y, rank, offsets):
    out = np.full_like(rank, -1)
    for a, b in zip(offsets[:-1], offsets[1:]):
        if np.all(rank[a:b] == -1): continue
        existing = rank[a:b]
        out[a:b] = existing[np.argsort(-y[existing].astype(int), kind='stable')]
    return out


def self_test():
    y = np.array([False, True, False, True, False])
    off = np.array([0, 3, 5]);rank = np.array([0, 2, 1, 4, 3])
    oracle = error_first_rank(y, rank, off)
    np.testing.assert_array_equal(oracle, [1, 0, 2, 3, 4])
    truth = np.add.reduceat(y.astype(int), off[:-1])
    flags, covered = flags_from_shared_rank(oracle, off, truth)
    np.testing.assert_array_equal(flags, y)
    assert covered.all()
    flags, _ = flags_from_shared_rank(rank, off, truth)
    assert not (flags & y).any()  # Quantity truth alone leaves the fixed bad ordering bad.
    empty, _ = flags_from_shared_rank(rank, off, np.zeros(2, int))
    assert not empty.any()
    print('COUNT_ORACLE_SELF_TEST_PASS', flush=True)


def main(package, frozen):
    started = time.time();package, frozen = package.resolve(), frozen.resolve()
    destination = package/'diagnostics'
    seal = validate_prediction_seal(package)
    predictions = {bank: load_predictions(package/f'predictions/{bank}.npz', bank)
                   for bank in ['N16', 'N56']}
    label_identity = authenticate_label_source(package, frozen, seal)
    # FIRST LABEL ACCESS follows main-seal and input-freeze label authentication.
    with np.load(frozen/'inputs/population.npz', allow_pickle=False) as archive:
        pop = {k: archive[k] for k in ['idx', 'offsets', 'y', 'classification', 'group', 'fold']}
    off, y = pop['offsets'], pop['y'].astype(bool)
    truth = np.add.reduceat(y.astype(int), off[:-1]);length = np.diff(off)
    draw_path = package/'evaluation/BOOTSTRAP_DRAWS.npz'
    with np.load(draw_path, allow_pickle=False) as archive:
        draw_groups, weights, seed = archive['source_groups'], archive['multiplicities'], int(archive['seed'])
    assert np.array_equal(draw_groups, np.unique(pop['group']))
    assert weights.shape == (20000, len(draw_groups)) and seed == 20261012
    assert np.all(weights >= 0) and np.all(weights.sum(1) == len(draw_groups))
    bootstrap = SourceBootstrap(pop['group'], weights=weights, seed=seed)
    masks = population_masks(pop['classification'], truth, length)
    wanted = ['ALL_NONCONTROL', 'SYNTHETIC_CLEAN', 'ALL_TRUE_ZERO', 'NONCONTROL_TRUE_ZERO',
              'ERROR_BEARING'] + list(TYPE_CODES)
    rows, contrasts, per_answer, flags_export = [], [], [], {'idx': pop['idx'], 'offsets': off}
    for bank, pred in predictions.items():
        assert np.array_equal(pred.idx, pop['idx']) and np.array_equal(pred.offsets, off)
        assert np.array_equal(pred.group, pop['group']) and np.array_equal(pred.fold, pop['fold'])
        for method in METHODS:
            rank = pred.ranks[method];rank_coverage = pred.rank_coverage[method]
            assert rank_coverage.all(), 'Component oracle requires complete matched frozen ranking'
            q80 = pred.method_counts[method].get('Q80', pred.counts['Q80'])
            assert np.all((q80 >= 0) & (q80 <= length))
            placements = {
                'Q80': (rank, q80),
                'ORACLE_TRUE_COUNT_FIXED_RANK': (rank, truth),
                'ORACLE_ERROR_FIRST_Q80_COUNT': (error_first_rank(y, rank, off), q80),
                'ALL_ZERO_NEGATIVE_CONTROL': (rank, np.zeros_like(truth)),
            }
            cache = {}
            for arm, (ordering, count) in placements.items():
                flags, covered = flags_from_shared_rank(ordering, off, count, rank_coverage)
                flags_export[f'{bank}__{method}__{arm}'] = flags
                for population in wanted:
                    point, replicas = summarize(y, flags, off, count, covered, masks[population], bootstrap,
                                                valid_only=population == 'multi_solutions')
                    if population in ['SYNTHETIC_CLEAN', 'ALL_TRUE_ZERO', 'NONCONTROL_TRUE_ZERO']:
                        # Match the main report: clean cohorts report abstention
                        # and valid F1, without a misleading half macro score.
                        point['prm_score'] = point['prm_score_lo95'] = point['prm_score_hi95'] = np.nan
                        point['prm_score_finite_draws'] = 0
                        replicas['prm_score'] = np.full(bootstrap.draws, np.nan)
                    cache[arm, population] = point, replicas
                    rows.append(dict(bank=bank, method=method, arm=arm, population=population,
                                     type_code=TYPE_CODES.get(population),
                                     role='control' if arm == 'Q80' else 'negative_control' if arm.startswith('ALL_ZERO') else 'evaluation_label_oracle',
                                     promotable=False, **point))
                for category, types in CATEGORIES.items():
                    entries = [cache[arm, typ] for typ in types]
                    score = float(np.mean([entry[0]['prm_score'] for entry in entries]))
                    draws = np.mean([entry[1]['prm_score'] for entry in entries], axis=0)
                    lo, hi = interval(draws)
                    mask = np.logical_or.reduce([masks[typ] for typ in types])
                    point = {'prm_score': score, 'n_answers': int(mask.sum()),
                             'n_steps': int(np.repeat(mask, length).sum()),
                             'n_source_groups': len(set(pop['group'][mask]))}
                    cache[arm, category] = point, {'prm_score': draws}
                    rows.append(dict(bank=bank, method=method, arm=arm, population=category,
                                     role='control' if arm == 'Q80' else 'negative_control' if arm.startswith('ALL_ZERO') else 'evaluation_label_oracle',
                                     promotable=False, prm_score_lo95=lo, prm_score_hi95=hi,
                                     convention='equal subtype mean; MS valid-step F1; Sensitivity=PS,DR,MS', **point))
                for i, uid in enumerate(pop['idx']):
                    per_answer.append({'bank': bank, 'method': method, 'arm': arm, 'idx': str(uid),
                        'group': str(pop['group'][i]), 'fold': int(pop['fold'][i]),
                        'n_steps': int(length[i]), 'true_count': int(truth[i]), 'count': int(count[i]),
                        'count_absolute_error': abs(int(count[i])-int(truth[i]))})
            for arm in placements:
                if arm == 'Q80': continue
                for population in wanted + list(CATEGORIES):
                    left, right = cache[arm, population], cache['Q80', population]
                    for metric in left[1]:
                        delta = left[1][metric] - right[1][metric]
                        lo, hi = interval(delta)
                        contrasts.append({'bank': bank, 'method': method, 'arm': arm, 'baseline': 'Q80',
                            'population': population, 'metric': metric,
                            'difference': left[0][metric]-right[0][metric], 'lo95': lo, 'hi95': hi,
                            'finite_draws': int(np.isfinite(delta).sum()), 'pointwise_only': True,
                            'n_answers': left[0]['n_answers'], 'n_source_groups': left[0]['n_source_groups'],
                            'same_20000_source_draws_as_main_evaluation': True, 'promotion_eligible': False})
            print('COUNT_ORACLE_DIAGNOSTIC', bank, method, flush=True)
    # Verify independently rebuilt named Q80 points against the existing main evaluation.
    with (package/'evaluation/METRICS.csv').open() as stream:
        main_rows = list(csv.DictReader(stream))
    for row in rows:
        if row['arm'] != 'Q80' or row['population'] not in wanted: continue
        match = [r for r in main_rows if all(r[k] == row[k] for k in ['bank', 'method', 'arm', 'population'])]
        assert len(match) == 1
        for metric in ['prm_score', 'count_mae', 'count_bias', 'f1_error', 'f1_valid']:
            expected = float(match[0][metric]) if match[0][metric] else np.nan
            actual = row[metric]
            assert (not np.isfinite(expected) and not np.isfinite(actual)) or abs(actual-expected) <= 1e-12, \
                (row['bank'], row['method'], row['population'], metric, actual, expected)
    write_csv(destination/'COUNT_ORACLE_METRICS.csv', rows)
    write_csv(destination/'COUNT_ORACLE_CONTRASTS.csv', contrasts)
    write_csv(destination/'COUNT_ORACLE_PER_ANSWER.csv', per_answer)
    np.savez_compressed(destination/'COUNT_ORACLE_FLAGS.npz', **flags_export)
    write_json(destination/'COUNT_ORACLES.json', {
        'status': 'COMPLETE_EVALUATION_LABEL_DIAGNOSTIC', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'fits_or_selection_performed': False, 'candidate_multiplicity_changed': False,
        'main_seal_sha256': seal['seal_sha256'], 'label_source_identity': label_identity,
        'bootstrap_draws': 20000, 'bootstrap_seed': seed, 'bootstrap_draws_sha256': sha(draw_path),
        'bootstrap_unit': 'whole canonical source group; exactly saved main multiplicities',
        'paired_intervals': 'pointwise 95 percent percentile; evaluation-label oracle diagnostics only',
        'definitions': {'ORACLE_TRUE_COUNT_FIXED_RANK': 'true valid-step membership count with frozen IU/LSML within-answer ordering',
            'ORACLE_ERROR_FIRST_Q80_COUNT': 'frozen method-specific Q80 quota; true error steps ordered first, frozen ordering used within error/valid ties',
            'ALL_ZERO_NEGATIVE_CONTROL': 'zero flags; no error recall; cannot qualify as a successful estimator'},
        'limits': 'Matched component diagnostics are not deployable methods, additive improvements, or a mathematical overall upper bound for pooled PRMScore. MS uses valid-step F1 and synthetic controls are separate.',
        'overall': [row for row in rows if row['population'] == 'ALL_NONCONTROL'],
        'diagnostic_script_sha256': sha(Path(__file__)), 'helper_diagnostic_script_sha256': sha(Path(__file__).with_name('evaluate_emission_calibration.py')),
        'outputs': {name: sha(destination/name) for name in ['COUNT_ORACLE_METRICS.csv', 'COUNT_ORACLE_CONTRASTS.csv',
                    'COUNT_ORACLE_PER_ANSWER.csv', 'COUNT_ORACLE_FLAGS.npz']},
        'runtime_s': time.time()-started})
    assert validate_prediction_seal(package)['seal_sha256'] == seal['seal_sha256']
    print('COUNT_ORACLE_DIAGNOSTIC_COMPLETE', len(rows), len(contrasts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--frozen', type=Path, default=ROOT/'results/advisor_completion_2026-10-11')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: self_test()
    else: main(args.package, args.frozen)
