"""Post-seal held-out emission/dependence diagnostics; never fits a model.

These evaluation-label diagnostics do not choose voters, signs or thresholds.
Failed fits keep their raw parameters and never receive repaired predictions.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from spectral_utils.error_count_evaluation import validate_prediction_seal


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(clean(rows))


def center_answers(values, offsets):
    result = np.array(values, float)
    for a, b in zip(offsets[:-1], offsets[1:]):
        result[a:b] -= result[a:b].mean(0)
    return result


def answer_z(values, offsets):
    result = np.zeros_like(values, dtype=float)
    for a, b in zip(offsets[:-1], offsets[1:]):
        block = values[a:b]
        sd = block.std(0)
        np.divide(block - block.mean(0), sd, out=result[a:b], where=sd > 1e-12)
    return result


def correlation(values):
    """Population Pearson correlations; constant/undersampled columns are undefined."""
    m = values.shape[1]
    if len(values) < 2: return np.full((m, m), np.nan)
    z = np.asarray(values, float) - np.asarray(values, float).mean(0)
    covariance = z.T @ z / len(z)
    denominator = np.sqrt(np.maximum(np.diag(covariance), 0.)[:, None] *
                          np.maximum(np.diag(covariance), 0.)[None, :])
    return np.divide(covariance, denominator, out=np.full_like(covariance, np.nan),
                     where=denominator > 0)


def truth_emissions(votes, y):
    positive = votes > 0
    psi = positive[y].mean(0) if y.any() else np.full(votes.shape[1], np.nan)
    eta = (~positive[~y]).mean(0) if (~y).any() else np.full(votes.shape[1], np.nan)
    return psi, eta


def posterior_replay(votes, psi, eta, prevalence):
    pos = votes > 0
    odds = np.log(prevalence) - np.log1p(-prevalence) + np.sum(
        np.where(pos, np.log(psi) - np.log1p(-eta), np.log1p(-psi) - np.log(eta)), axis=1)
    return expit(odds)


def weighted_error(rows, key, denominator, square=False):
    take = [r for r in rows if r.get(key) is not None and np.isfinite(r[key]) and r[denominator] > 0]
    den = sum(r[denominator] for r in take)
    return (sum(r[denominator] * (r[key] ** 2 if square else abs(r[key])) for r in take) / den
            if den else np.nan), den


def self_test():
    votes = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1], [-1, -1, -1]])
    y = np.array([True, True, False, False])
    psi, eta = truth_emissions(votes, y)
    np.testing.assert_array_equal(psi, [1., .5, .5])
    np.testing.assert_array_equal(eta, [1., .5, .5])
    np.testing.assert_allclose(correlation(votes[y])[1, 2], -1.)
    np.testing.assert_allclose(correlation(votes[~y])[1, 2], 1.)
    assert np.isnan(correlation(votes[y])[0]).all()
    np.testing.assert_array_equal(center_answers(votes, [0, 2, 4]),
                                  [[0, 1, -1], [0, -1, 1], [0, 1, 1], [0, -1, -1]])
    p = posterior_replay(votes[:, :1], np.array([.8]), np.array([.9]), .2)
    np.testing.assert_allclose(p, [.16 / .24, .16 / .24, .04 / .76, .04 / .76])
    print('EMISSION_DIAGNOSTIC_SELF_TEST_PASS', flush=True)


def main(package, frozen):
    started = time.time()
    package, frozen = package.resolve(), frozen.resolve()
    destination = package/'diagnostics'
    seal = validate_prediction_seal(package)  # Authenticate before any label access.
    if 'INPUT_FREEZE.json' not in seal['verified_files']:
        raise RuntimeError('Input freeze is not bound to the prediction seal')
    freeze = json.loads((package/'INPUT_FREEZE.json').read_text())
    inspected = {}
    for relative, specification in freeze['inputs'].items():
        path = ROOT/relative
        if sha(path) != specification['sha256']:
            raise RuntimeError('Frozen input changed: '+relative)
        inspected[relative] = specification['sha256']
    population_path = frozen/'inputs/population.npz'
    if str(population_path.relative_to(ROOT)) not in inspected:
        raise RuntimeError('Evaluation population is not pinned by the input freeze')
    # FIRST EVALUATION-LABEL ACCESS. No subsequent fitting/selection is performed.
    with np.load(population_path, allow_pickle=False) as archive:
        pop = {k: archive[k] for k in ['idx', 'offsets', 'fold', 'group', 'y', 'classification']}
    off, y = pop['offsets'], pop['y'].astype(bool)
    groups_rows, fold_rows, pair_rows, posterior_rows, bin_rows = [], [], [], [], []
    matrices, posterior_totals, covered_groups = {}, {}, {}
    provenance_sources = {}
    historical = frozen/'sources/research/results/monday_fusion_analysis_v1/code/scripts/experiments/er_stage_b.py'
    nodes = [node for node in ast.parse(historical.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == 'random_tie_marks']
    assert len(nodes) == 1
    env = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(historical), 'exec'), env)
    provenance_sources[str(historical.relative_to(ROOT))] = sha(historical)
    for bank in ['N16', 'N56']:
        with np.load(frozen/f'inputs/bank_V_{bank}.npz', allow_pickle=False) as archive:
            V, columns = archive['V'], archive['columns'].astype(str)
            assert np.array_equal(archive['offsets'], off)
        for fold in range(5):
            folder = package/f'checkpoints/{bank}_fold{fold}'
            models = json.loads((folder/'models.json').read_text())
            test = np.flatnonzero(pop['fold'] == fold)
            train = np.flatnonzero(pop['fold'] != fold)
            assert not set(pop['group'][train]) & set(pop['group'][test])
            rows = np.concatenate([np.arange(off[i], off[i+1]) for i in test])
            tr = np.concatenate([np.arange(off[i], off[i+1]) for i in train])
            eoff = np.r_[0, np.cumsum(np.diff(off)[test])]
            local_y = y[rows]
            step_group = np.repeat(pop['group'][test], np.diff(eoff))
            classifications = pop['classification'][test]
            populations = {'ALL': np.ones(len(test), bool), 'NONCONTROL': classifications != 'correct',
                           'SYNTHETIC_CLEAN': classifications == 'correct'}
            votes_by_kind = {}
            for kind in ['local', 'absolute']:
                path = folder/f'{kind}_votes.npz'
                with np.load(path, allow_pickle=False) as archive:
                    assert np.array_equal(archive['test_rows'], rows)
                    assert np.array_equal(archive['train_rows'], tr)
                    votes_by_kind[kind] = archive['test']
                provenance_sources[str(path.relative_to(ROOT))] = sha(path)
            within = models['LSML_CONT']['diagnostic']['group_weights']
            virtual = np.column_stack([V[rows][:, g['columns']] @ np.asarray(g['weights']) for g in within])
            legacy_scores = answer_z(virtual, eoff)
            votes_by_kind['legacy'] = env['random_tie_marks'](legacy_scores, eoff, .2,
                                                   np.random.default_rng(20260929).random(legacy_scores.shape))
            pred_path = folder/'predictions.npz'
            with np.load(pred_path, allow_pickle=False) as archive:
                assert np.array_equal(archive['idx'], pop['idx'][test])
                assert np.array_equal(archive['offsets'], eoff)
                saved = {k: archive[k] for k in archive.files if k.startswith('posterior__')}
            arm_sources = [('LOCAL_STRICT', 'local_mom', 'local', 'LOCAL_MOM_MEAN'),
                           ('A_STRICT', 'absolute_mom', 'absolute', 'A_MOM_MEAN'),
                           ('B_SELECTED_TRIPLE', 'B_TRIPLE', 'absolute', 'B_TRIPLE_MEAN'),
                           ('LEGACY_MOM', 'LEGACY_MOM', 'legacy', 'LEGACY_MOM')]
            for arm, model_key, vote_kind, prediction_arm in arm_sources:
                model = models[model_key]
                status = model.get('status', 'PASS' if arm == 'LEGACY_MOM' else 'MISSING')
                selected = model.get('columns', list(range(votes_by_kind[vote_kind].shape[1])))
                M = votes_by_kind[vote_kind][:, selected]
                assert M.shape == (len(rows), len(selected)) and np.isin(M, [-1, 1]).all()
                psi = np.asarray(model.get('psi_raw', model.get('psi', [None] * len(selected))), float)
                eta = np.asarray(model.get('eta_raw', model.get('eta', [None] * len(selected))), float)
                assert psi.shape == eta.shape == (len(selected),)
                prevalence = model.get('prevalence', np.nan)
                arm_status = models['status'][prediction_arm]['status']
                p_saved = saved['posterior__'+prediction_arm]
                if status == 'PASS' and arm_status == 'PASS':
                    p_replay = posterior_replay(M, np.asarray(model['psi']), np.asarray(model['eta']), prevalence)
                    error = float(np.max(np.abs(p_replay - p_saved)))
                    assert np.isfinite(error) and error <= 1e-12, (bank, fold, arm, error)
                else:
                    assert not np.isfinite(p_saved).any(), (bank, fold, arm, 'Failed fit has predictions')
                    error = None  # No failed model receives a clipped/repaired posterior.
                centered = center_answers(M, eoff)
                for population, answer_mask in populations.items():
                    keep = np.repeat(answer_mask, np.diff(eoff))
                    yy, mm = local_y[keep], M[keep]
                    true_psi, true_eta = truth_emissions(mm, yy)
                    n_error, n_valid = int(yy.sum()), int((~yy).sum())
                    common = {'bank': bank, 'fold': fold, 'arm': arm, 'population': population,
                              'fit_status': status, 'prediction_status': arm_status,
                              'failure_reason': model.get('failure_reason'),
                              'n_answers': int(answer_mask.sum()), 'n_steps': int(keep.sum()),
                              'n_source_groups': len(set(pop['group'][test][answer_mask])),
                              'n_error_steps': n_error, 'n_valid_steps': n_valid,
                              'n_voters': len(selected), 'prevalence_hat': prevalence,
                              'prevalence_true': float(yy.mean()) if len(yy) else np.nan,
                              'posterior_replay_max_abs': error}
                    true_balanced = (true_psi + true_eta) / 2
                    true_delta = true_psi + true_eta - 1
                    hat_delta = psi + eta - 1
                    fold_rows.append(dict(common, prevalence_bias=prevalence-common['prevalence_true'],
                        majority_better_than_chance_true=bool(np.mean(true_balanced > .5) > .5) if np.isfinite(true_balanced).all() else None,
                        all_voters_better_than_chance_true=bool(np.all(true_balanced > .5)) if np.isfinite(true_balanced).all() else None,
                        n_constant_voters=int(np.sum(mm.std(0) == 0)) if len(mm) else 0,
                        raw_out_of_range=int(np.sum((psi <= 0) | (psi >= 1) | (eta <= 0) | (eta >= 1))),
                        legacy_clipped_voters=model.get('out_of_range', 0) if arm == 'LEGACY_MOM' else 0))
                    for j, g in enumerate(selected):
                        indices = within[g]['columns']
                        groups_rows.append(dict(common, group_index=g, member_indices=json.dumps(indices),
                            member_names=json.dumps(columns[indices].tolist()),
                            psi_hat=psi[j], eta_hat=eta[j], psi_true=true_psi[j], eta_true=true_eta[j],
                            balanced_accuracy_hat=(psi[j]+eta[j])/2, balanced_accuracy_true=true_balanced[j],
                            psi_error=psi[j]-true_psi[j], eta_error=eta[j]-true_eta[j],
                            reliability_error=(psi[j]+eta[j])/2-true_balanced[j],
                            orientation_agreement=bool(np.sign(hat_delta[j]) == np.sign(true_delta[j]))
                                if np.isfinite(hat_delta[j]) and np.isfinite(true_delta[j]) else None))
                    for transform, source in [('raw', M), ('answer_centered', centered)]:
                        for cls_name, class_mask in [('error', local_y), ('valid', ~local_y)]:
                            take = keep & class_mask
                            corr = correlation(source[take])
                            key = f'{bank}/fold{fold}/{arm}/{population}/{transform}/{cls_name}'
                            matrices[key] = {'columns': selected, 'matrix': corr,
                                             'n_steps': int(take.sum()), 'n_source_groups': len(set(step_group[take]))}
                            for j in range(len(selected)):
                                for k in range(j+1, len(selected)):
                                    pair_rows.append(dict(common, transform=transform, true_class=cls_name,
                                        group_i=selected[j], group_j=selected[k], correlation=corr[j, k],
                                        n_class_steps=int(take.sum()), n_class_source_groups=len(set(step_group[take]))))
            for arm in ['LOCAL_MOM_MEAN', 'A_MOM_MEAN', 'B_TRIPLE_MEAN', 'D_ZERO_MEAN', 'LEGACY_MOM']:
                p = saved['posterior__'+arm]
                status = models['status'][arm]
                if status['status'] != 'PASS':
                    assert not np.isfinite(p).any()
                    continue
                assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                for population, answer_mask in populations.items():
                    take = np.repeat(answer_mask, np.diff(eoff))
                    pp, yy = p[take], local_y[take]
                    key = (bank, arm, population)
                    total = posterior_totals.setdefault(key, {'n': 0, 'n_answers': 0, 'sum_p': 0., 'sum_y': 0.,
                                                              'sum_brier': 0., 'bin_n': np.zeros(10),
                                                              'bin_p': np.zeros(10), 'bin_y': np.zeros(10)})
                    total['n'] += len(pp); total['n_answers'] += int(answer_mask.sum())
                    total['sum_p'] += pp.sum(); total['sum_y'] += yy.sum(); total['sum_brier'] += np.square(pp-yy).sum()
                    bins = np.minimum((pp * 10).astype(int), 9)
                    total['bin_n'] += np.bincount(bins, minlength=10)
                    total['bin_p'] += np.bincount(bins, weights=pp, minlength=10)
                    total['bin_y'] += np.bincount(bins, weights=yy, minlength=10)
                    covered_groups.setdefault(key, set()).update(pop['group'][test][answer_mask])
            print('EMISSION_DIAGNOSTIC', bank, fold, flush=True)
    aggregates = []
    for bank in ['N16', 'N56']:
        for arm in ['LOCAL_STRICT', 'A_STRICT', 'B_SELECTED_TRIPLE', 'LEGACY_MOM']:
            for population in ['ALL', 'NONCONTROL', 'SYNTHETIC_CLEAN']:
                for basis in ['raw_parameters_including_failed', 'successful_fits_only']:
                    take = [r for r in groups_rows if r['bank'] == bank and r['arm'] == arm and r['population'] == population
                            and (basis == 'raw_parameters_including_failed' or r['fit_status'] == 'PASS')]
                    folds = [r for r in fold_rows if r['bank'] == bank and r['arm'] == arm and r['population'] == population
                             and (basis == 'raw_parameters_including_failed' or r['fit_status'] == 'PASS')]
                    row = {'bank': bank, 'arm': arm, 'population': population, 'basis': basis,
                           'n_folds': len(folds), 'n_group_fold_parameters': len(take),
                           'n_answers': sum(r['n_answers'] for r in folds), 'n_steps': sum(r['n_steps'] for r in folds),
                           'n_source_groups': sum(r['n_source_groups'] for r in folds)}
                    for param, denominator in [('psi', 'n_error_steps'), ('eta', 'n_valid_steps'), ('reliability', 'n_steps')]:
                        row[param+'_mae'], row[param+'_exposures'] = weighted_error(take, param+'_error', denominator)
                        row[param+'_mse'], _ = weighted_error(take, param+'_error', denominator, square=True)
                    row['balanced_emission_mse'] = (row['psi_mse'] + row['eta_mse']) / 2
                    row['prevalence_mse'], _ = weighted_error(folds, 'prevalence_bias', 'n_steps', square=True)
                    denominator = sum(r['n_steps'] for r in folds)
                    row['prevalence_hat_stepweighted'] = sum(r['n_steps']*r['prevalence_hat'] for r in folds) / denominator if denominator else np.nan
                    row['prevalence_true_stepweighted'] = sum(r['n_steps']*r['prevalence_true'] for r in folds) / denominator if denominator else np.nan
                    row['prevalence_bias_stepweighted'] = row['prevalence_hat_stepweighted'] - row['prevalence_true_stepweighted']
                    agreement = [r for r in take if r['orientation_agreement'] is not None]
                    row['orientation_agreement_group_fold_fraction'] = np.mean([r['orientation_agreement'] for r in agreement]) if agreement else np.nan
                    aggregates.append(row)
    for key, total in posterior_totals.items():
        bank, arm, population = key
        n = total['n']
        bin_mean = np.divide(total['bin_p'], total['bin_n'], out=np.full(10, np.nan), where=total['bin_n'] > 0)
        bin_true = np.divide(total['bin_y'], total['bin_n'], out=np.full(10, np.nan), where=total['bin_n'] > 0)
        posterior_rows.append({'bank': bank, 'arm': arm, 'population': population,
            'n_answers': total['n_answers'], 'n_steps': n, 'n_source_groups': len(covered_groups[key]),
            'brier_mse': total['sum_brier']/n if n else np.nan,
            'posterior_mean': total['sum_p']/n if n else np.nan, 'true_prevalence': total['sum_y']/n if n else np.nan,
            'probability_bias': (total['sum_p']-total['sum_y'])/n if n else np.nan,
            'ECE_fixed_ten_bins': np.nansum(total['bin_n']*np.abs(bin_mean-bin_true))/n if n else np.nan,
            'reliability_mse_fixed_ten_bins': np.nansum(total['bin_n']*np.square(bin_mean-bin_true))/n if n else np.nan})
        for j in range(10):
            bin_rows.append({'bank': bank, 'arm': arm, 'population': population, 'bin': j,
                             'lower': j/10, 'upper': (j+1)/10, 'upper_inclusive': j == 9,
                             'n_steps': int(total['bin_n'][j]), 'mean_probability': bin_mean[j], 'error_fraction': bin_true[j]})
    outputs = {'EMISSION_GROUPWISE.csv': groups_rows, 'EMISSION_FOLDWISE.csv': fold_rows,
               'EMISSION_AGGREGATE.csv': aggregates, 'CONDITIONAL_CORRELATION_PAIRS.csv': pair_rows,
               'POSTERIOR_CALIBRATION.csv': posterior_rows, 'POSTERIOR_RELIABILITY_BINS.csv': bin_rows}
    for name, rows in outputs.items(): write_csv(destination/name, rows)
    write_json(destination/'EMISSION_CALIBRATION.json', {
        'status': 'COMPLETE_EVALUATION_ONLY', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'models_refitted': False, 'signs_thresholds_or_voters_selected_from_labels': False,
        'failed_models_repaired': False, 'aggregates': aggregates, 'posterior_calibration': posterior_rows,
        'conditional_correlations': matrices,
        'conventions': {'psi': 'P(vote+ | true error)', 'eta': 'P(vote- | true valid)',
            'balanced_accuracy': '(psi+eta)/2', 'parameter_weights': 'held-out class-step exposures per voter; reliability weighted by held-out step-voter exposures',
            'correlations': 'all answer steps centered before class/population selection; class means removed by Pearson correlation',
            'posterior_calibration': 'uses saved held-out probabilities only; failed model posteriors remain undefined',
            'bins': 'ten fixed equal-width [0,1] bins; final bin includes 1',
            'assumption_limit': 'Class-conditional correlation is descriptive, not an independence proof; source/answer observations are dependent. No IID standard errors or equivalence claims.'},
        'runtime_s': time.time()-started})
    outputs_list = list(outputs) + ['EMISSION_CALIBRATION.json']
    write_json(destination/'EMISSION_CALIBRATION_PROVENANCE.json', {
        'status': 'AUTHENTICATED_POST_SEAL_DIAGNOSTIC', 'main_seal_sha256': seal['seal_sha256'],
        'input_freeze_sha256': sha(package/'INPUT_FREEZE.json'), 'population_sha256': sha(population_path),
        'first_label_access': 'after main seal validation and all input-freeze hashes',
        'diagnostic_script_sha256': sha(Path(__file__)), 'historical_vote_source_and_saved_votes': provenance_sources,
        'verified_input_hashes': inspected,
        'output_hashes': {name: sha(destination/name) for name in outputs_list},
        'main_sealed_files_changed': False, 'amendments_changed': False})
    # A final seal validation confirms none of the diagnostic writes changed main evidence.
    assert validate_prediction_seal(package)['seal_sha256'] == seal['seal_sha256']
    print('EMISSION_CALIBRATION_COMPLETE', len(groups_rows), len(pair_rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--frozen', type=Path, default=ROOT/'results/advisor_completion_2026-10-11')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: self_test()
    else: main(args.package, args.frozen)
