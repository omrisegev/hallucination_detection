"""Bounded, label-free quota/length mechanism diagnostics (amendment A003).

Run with the repository Python 3.11 environment. No real target labels or
classifications are loaded. These are mechanism diagnostics, not additions to
the seven-candidate family or a threshold-selection procedure.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from itertools import product

sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.stats import norm

DEST = Path(__file__).resolve().parent
PACKAGE = DEST.parent
ROOT = PACKAGE.parents[1]
FROZEN = ROOT / 'results/advisor_completion_2026-10-11'
ATTEMPT = PACKAGE / 'attempts/attempt01_float64_bridge_failure'
ARCHIVE = FROZEN / 'sources/research/results/monday_fusion_analysis_v1/code/scripts/experiments'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARCHIVE))
from spectral_utils.error_count import fit_mom, posterior, count_distribution
from spectral_utils.advisor_controls import answer_z
from spectral_utils.advisor_lr import fit_labelfree, as_dict
from spectral_utils.error_count_inputs import fit_vote_representation, apply_vote_representation
import er_stage_a as SA
import er_stage_b as SB

QUOTAS = (.1, .2, .3)
TIE_SEED = 20260929
PERMUTATION_SEEDS = tuple(2026101200 + i for i in range(10))
LENGTH_BINS = ('ALL', '1-5', '6-10', '11-20', '21+')
DIGIT_COLUMNS = ('digit_alternative', 'digit_spread', 'digit_alternative_innovation')


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(clean(rows))


def configuration():
    return dict(version=1, amendment='A003', diagnostic_only=True,
                candidate_count_unchanged=7, no_new_real_labels=True,
                relative_quotas=QUOTAS, tie_seed=TIE_SEED,
                permutations=dict(seeds=PERMUTATION_SEEDS,
                                  operation='independent group-column shuffle inside each answer; preserve every quota'),
                length_bins=LENGTH_BINS,
                input_groups='saved donor LSML_CONT within-group weights from archived first attempt; no refit',
                full_reference='one full N16 label-free LSML fit; transductive historical mechanism reference, separate from OOF donors',
                fits=['archived clipped tensor MoM, raw emissions reconstructed and reported',
                      'new strict all-group MoM, failures remain undefined'],
                absolute=dict(scenarios=['ABS_Q80', 'ABS_ZERO'],
                              zero_boundary='strictly G_abs>0 using frozen donor absolute mean/scale; no heldout threshold fit',
                              q80_boundary='existing strict-greater-than donor group .8 quantile',
                              shared_groups_weights_and_scores=True,
                              heldout='strict-valid model posteriors and half-up counts only; immutable archived rankings'),
                permutation_null=dict(pair_covariance='4*Var_stepweighted(ceil(q*n)/n)',
                                      distinct_third='8*E_stepweighted[(ceil(q*n)/n-pbar)^3]',
                                      caveat='Shuffles retain length/quota mixture structure; surviving pi is not evidence of errors.'),
                exact_teacher=dict(prevalence=.17, thresholds=[-.5, .5, 1.5],
                                   mu_valid=[-.2, 0., .2, .4], mu_error=[1.8, 2.2, 2.6, 3.],
                                   sigma=1., patterns=16, weighted_exact_joint=True,
                                   recovery_tolerance=1e-8, practical_prior=False),
                interpretation='Changing quota changes vote marginals AND dependence/emissions; no algebraic identity or utility claim is inferred.',
                reference='https://proceedings.mlr.press/v38/jaffe15.pdf')


def prepare():
    protocol = json.loads((PACKAGE / 'PROTOCOL.json').read_text())
    amendments = json.loads((PACKAGE / 'AMENDMENTS.json').read_text())
    if protocol['candidate_count_for_multiplicity'] != 7 or not any(item['id'] == 'A003' for item in amendments):
        raise RuntimeError('A003 and unchanged seven-candidate family required before diagnostic preparation')
    inputs = [FROZEN / 'inputs' / name for name in
              ('population.npz', 'bank_V_N16.npz', 'bank_V_N56.npz', 'bank_N16.npz',
               'features_thinkfix_mask1.npz', 'CANONICAL_GROUPS.json', 'FOLDS_V2.json')]
    inputs += [PACKAGE / name for name in ('PROTOCOL.json', 'AMENDMENTS.json', 'INPUT_FREEZE.json')]
    for bank in ('N16', 'N56'):
        for fold in range(5):
            folder = ATTEMPT / 'checkpoints' / f'{bank}_fold{fold}'
            inputs += [folder / name for name in ('models.json', 'TRAINING_IDENTITY.json', 'COMPLETE.json', 'predictions.npz')]
    code = [Path(__file__), ROOT / 'spectral_utils/error_count.py', ROOT / 'spectral_utils/error_count_inputs.py',
            ROOT / 'spectral_utils/advisor_controls.py', ROOT / 'spectral_utils/advisor_lr.py',
            ROOT / 'spectral_utils/fusion_utils.py', ROOT / 'spectral_utils/upcr.py',
            ROOT / 'spectral_utils/laplacian_upcr.py', ARCHIVE / 'er_stage_a.py', ARCHIVE / 'er_stage_b.py']
    state = dict(inputs={str(path.relative_to(ROOT)): dict(sha256=sha(path), size=path.stat().st_size) for path in inputs},
                 code={str(path.relative_to(ROOT)): sha(path) for path in code},
                 runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                              scipy=scipy.__version__, platform=platform.platform()),
                 configuration_sha256=hashlib.sha256(json.dumps(configuration(), sort_keys=True).encode()).hexdigest())
    config_path, freeze_path = DEST / 'QUOTA_PROTOCOL.json', DEST / 'QUOTA_INPUT_FREEZE.json'
    if config_path.exists() and json.loads(config_path.read_text()) != clean(configuration()):
        raise RuntimeError('Diagnostic configuration changed; preserve the existing run and use an amendment/new directory')
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        if 'operational_seal_repair' in previous:
            state['operational_seal_repair'] = previous['operational_seal_repair']
        if previous != clean(state):
            raise RuntimeError('Diagnostic input/code/runtime identity changed')
    dump(config_path, configuration()); dump(freeze_path, state)
    print('QUOTA_PREPARED', len(inputs), 'inputs', flush=True)


def validate_freeze():
    state = json.loads((DEST / 'QUOTA_INPUT_FREEZE.json').read_text())
    if json.loads((DEST / 'QUOTA_PROTOCOL.json').read_text()) != clean(configuration()):
        raise RuntimeError('Diagnostic protocol changed')
    if state['runtime'] != dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                                scipy=scipy.__version__, platform=platform.platform()):
        raise RuntimeError('Diagnostic runtime changed')
    for path, entry in state['inputs'].items():
        if sha(ROOT / path) != entry['sha256']: raise RuntimeError('Changed diagnostic input: ' + path)
    for path, expected in state['code'].items():
        if sha(ROOT / path) != expected: raise RuntimeError('Changed diagnostic code: ' + path)


def archived_mom(votes):
    try:
        estimate = SA.tensor_mom_estimate(votes, None)
        mean = votes.mean(0)
        b, t = float(estimate['b']), np.asarray(estimate['t'], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = t / np.sqrt(1 - b ** 2)
            raw_psi = (1 + mean + delta * (1 - b)) / 2
            raw_eta = (1 - mean + delta * (1 + b)) / 2
        good = bool(0 < estimate['prevalence'] < 1 and np.isfinite(estimate['psi']).all()
                    and np.isfinite(estimate['eta']).all())
        return clean(dict(estimate, status='PASS' if good else 'FAILED', method='ARCHIVED_CLIPPED_MOM',
                          psi_raw=raw_psi, eta_raw=raw_eta, raw_derivation='mean/t/b from exact archived estimate',
                          archived_rank_one_convergence='not reported by archived implementation',
                          failure_reason=None if good else 'NONFINITE_OR_BOUNDARY_ARCHIVED_ESTIMATE'))
    except Exception as exc:
        return dict(status='FAILED', method='ARCHIVED_CLIPPED_MOM', failure_reason=repr(exc))


def select_rows(offsets, answers):
    answers = np.asarray(answers, int)
    rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in answers]) if len(answers) else np.zeros(0, int)
    return rows, np.r_[0, np.cumsum(np.diff(offsets)[answers])]


def length_selection(n, name):
    if name == 'ALL': return np.ones(len(n), bool)
    if name == '1-5': return n <= 5
    if name == '6-10': return (n >= 6) & (n <= 10)
    if name == '11-20': return (n >= 11) & (n <= 20)
    if name == '21+': return n >= 21
    raise ValueError(name)


def vote_statistics(votes, offsets, quota=None):
    n = np.diff(offsets)
    mean = (votes > 0).mean(0)
    centered = votes - votes.mean(0)
    covariance = centered.T @ centered / len(votes)
    offdiag = covariance[np.triu_indices(votes.shape[1], 1)]
    per_answer = np.add.reduceat((votes > 0).sum(1), offsets[:-1])
    result = dict(n_answers=len(n), n_steps=len(votes), n_voters=votes.shape[1],
                  actual_mean_vote_rate=float(mean.mean()), actual_group_vote_rates=mean.tolist(),
                  offdiag_covariance_mean=float(offdiag.mean()), offdiag_covariance_min=float(offdiag.min()),
                  offdiag_covariance_max=float(offdiag.max()),
                  all_negative_vote_answer_fraction=float((per_answer == 0).mean()),
                  all_positive_vote_answer_fraction=float((per_answer == n * votes.shape[1]).mean()))
    if quota is not None:
        k = np.ceil(quota * n).astype(int)
        counts = np.add.reduceat((votes > 0).astype(int), offsets[:-1], axis=0)
        if not np.array_equal(counts, np.broadcast_to(k[:, None], counts.shape)):
            raise RuntimeError('Relative quota was not preserved for every answer/group')
        p = k / n; pbar = float(np.average(p, weights=n))
        result.update(quota_expected_mean_vote_rate=float(k.sum() / n.sum()), quota_preserved=True,
                      permutation_null_pair_covariance=float(4 * np.average((p - pbar) ** 2, weights=n)),
                      permutation_null_distinct_third=float(8 * np.average((p - pbar) ** 3, weights=n)))
    return result


def finite_minmax(values):
    if values is None: return None, None
    array = np.asarray([np.nan if value is None else value for value in values], float)
    return (float(np.nanmin(array)), float(np.nanmax(array))) if np.isfinite(array).any() else (None, None)


def row_from_model(meta, votes, offsets, model, estimator, quota=None):
    summary = vote_statistics(votes, offsets, quota)
    psi_min, psi_max = finite_minmax(model.get('psi_raw'))
    eta_min, eta_max = finite_minmax(model.get('eta_raw'))
    pi = model.get('prevalence')
    valid_pi = pi if model.get('status') == 'PASS' else None
    row = dict(meta, estimator=estimator, fit_status=model.get('status'),
               failure_reason=model.get('failure_reason'), **summary,
               prevalence_hat=valid_pi, raw_prevalence_even_if_failed=pi,
               prevalence_minus_vote_rate=None if valid_pi is None else valid_pi - summary['actual_mean_vote_rate'],
               raw_psi_min=psi_min, raw_psi_max=psi_max, raw_eta_min=eta_min, raw_eta_max=eta_max,
               clipped_voters=model.get('out_of_range', 0) if estimator == 'ARCHIVED_CLIPPED' else 0,
               rank_one_converged=model.get('rank_one_completion', {}).get('converged'),
               rank_one_iterations=model.get('rank_one_completion', {}).get('iterations'))
    if model.get('status') == 'PASS':
        try:
            probability = posterior(votes, model)
            mean = np.add.reduceat(probability, offsets[:-1])
            rounded = np.floor(mean + .5).astype(int)
            row.update(posterior_mean_count_all_answers=float(mean.mean()),
                       posterior_mean_rate_all_steps=float(probability.mean()),
                       predicted_zero_count_fraction_all_answers=float((rounded == 0).mean()))
        except Exception as exc:
            row['posterior_failure'] = repr(exc)
    return row


def independently_permute(votes, offsets, seed):
    rng = np.random.default_rng(seed)
    out = np.array(votes, copy=True)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for column in range(votes.shape[1]): out[a:b, column] = votes[a:b, column][rng.permutation(b - a)]
    if not np.array_equal(np.add.reduceat((votes > 0).astype(int), offsets[:-1], axis=0),
                          np.add.reduceat((out > 0).astype(int), offsets[:-1], axis=0)):
        raise RuntimeError('Permutation changed an answer/group quota')
    return out


def exact_teacher():
    config = configuration()['exact_teacher']
    pi = config['prevalence']
    patterns = np.array(list(product((-1, 1), repeat=4)))
    positives = patterns > 0
    rows = []
    for threshold in config['thresholds']:
        psi = norm.sf(threshold - np.asarray(config['mu_error']))
        eta = norm.cdf(threshold - np.asarray(config['mu_valid']))
        distribution = pi * np.prod(np.where(positives, psi, 1 - psi), axis=1)
        distribution += (1 - pi) * np.prod(np.where(positives, 1 - eta, eta), axis=1)
        for legacy in (False, True):
            model = fit_mom(patterns, legacy_clip=legacy, weights=distribution)
            error = abs(model['prevalence'] - pi) if model.get('status') == 'PASS' else None
            rows.append(dict(threshold=threshold, teacher_prevalence=pi,
                             actual_vote_rate=float(np.sum(distribution[:, None] * positives) / 4),
                             expected_psi=psi, expected_eta=eta, estimated=model,
                             absolute_prevalence_error=error))
    good = all(row['absolute_prevalence_error'] is not None and
               row['absolute_prevalence_error'] <= config['recovery_tolerance'] for row in rows)
    result = dict(status='PASS' if good else 'FAILED', exact_joint=True,
                  conclusion='Under identifiable conditional independence, changing vote marginals need not change latent prevalence.',
                  rows=rows)
    dump(DEST / 'EXACT_TEACHER.json', result)
    if not good: raise RuntimeError('Exact conditional-independent teacher recovery failed')
    print('EXACT_TEACHER_PASS', max(row['absolute_prevalence_error'] for row in rows), flush=True)


def fit():
    validate_freeze()
    if (DEST / 'QUOTA_SEAL.json').exists():
        seal = json.loads((DEST / 'QUOTA_SEAL.json').read_text())
        for path, entry in seal['files'].items():
            if sha(DEST / path) != entry['sha256']: raise RuntimeError('Sealed diagnostic output changed')
        print('QUOTA_RESUME_SEALED', flush=True); return
    started = time.time()
    with np.load(FROZEN / 'inputs/population.npz', allow_pickle=False) as source:
        data = {key: source[key] for key in ('idx', 'offsets', 'group', 'fold', 'feature_rows')}
    with np.load(FROZEN / 'inputs/features_thinkfix_mask1.npz', allow_pickle=False) as source:
        digit_active = source['digit_active'][data['feature_rows']]
    original_freeze = json.loads((PACKAGE / 'INPUT_FREEZE.json').read_text())
    for name in ('population.npz', 'bank_V_N16.npz', 'bank_V_N56.npz', 'bank_N16.npz', 'features_thinkfix_mask1.npz'):
        path = FROZEN / 'inputs' / name
        if sha(path) != original_freeze['inputs'][str(path.relative_to(ROOT))]['sha256']:
            raise RuntimeError('Frozen canonical input changed: ' + name)
    offsets = data['offsets']; rows, model_states, failures = [], {}, []

    def record(meta, votes, off, quota=None, length_fits=False):
        variants = LENGTH_BINS if length_fits else ('ALL',)
        fitted = {}
        for length_name in variants:
            answers = np.flatnonzero(length_selection(np.diff(off), length_name))
            subrows, suboff = select_rows(off, answers)
            if not len(answers):
                rows.append(dict(meta, length_bin=length_name, fit_status='FAILED', failure_reason='EMPTY_LENGTH_BIN'))
                continue
            vv = votes[subrows]
            for estimator in ('ARCHIVED_CLIPPED', 'STRICT'):
                model = archived_mom(vv) if estimator == 'ARCHIVED_CLIPPED' else fit_mom(vv)
                key = '__'.join(str(meta.get(field, 'none')) for field in
                                ('scope', 'bank', 'fold', 'scenario', 'quota', 'permutation_seed')) + '__' + length_name + '__' + estimator
                model_states[key] = dict(model=model, vote_sha256=array_sha(vv), metadata=dict(meta, length_bin=length_name))
                rows.append(row_from_model(dict(meta, length_bin=length_name), vv, suboff, model, estimator, quota))
                if length_name == 'ALL': fitted[estimator] = model
        return fitted

    def relative_family(bank, fold, scope, V, off, groups, model_folder, saved_legacy=None):
        score = np.column_stack([answer_z(V[:, group['columns']] @ np.asarray(group['weights']), off) for group in groups])
        key = np.random.default_rng(TIE_SEED).random(score.shape)
        arrays = dict(offsets=off, scores=score)
        quota20 = None
        for quota in QUOTAS:
            votes = SB.random_tie_marks(score, off, quota, key)
            arrays[f'votes_q{int(quota * 100)}'] = votes
            meta = dict(bank=bank, fold=fold, scope=scope, scenario='RELATIVE_QUOTA', quota=quota, permutation_seed=None)
            fitted = record(meta, votes, off, quota, length_fits=True)
            if quota == .2:
                quota20 = votes
                if saved_legacy and fitted['ARCHIVED_CLIPPED']['status'] == 'PASS':
                    delta = abs(fitted['ARCHIVED_CLIPPED']['prevalence'] - saved_legacy['prevalence'])
                    if delta > 1e-12: raise RuntimeError(f'Archived q20 MoM replay differs by {delta}')
                    dump(model_folder / 'ARCHIVED_Q20_REPLAY.json', dict(prevalence_max_abs=delta, status='PASS'))
        for seed in PERMUTATION_SEEDS:
            permuted = independently_permute(quota20, off, seed)
            arrays[f'permuted_{seed}'] = permuted
            record(dict(bank=bank, fold=fold, scope=scope, scenario='PERMUTED_QUOTA', quota=.2,
                        permutation_seed=seed), permuted, off, .2)
        np.savez_compressed(model_folder / 'relative_votes.npz', **arrays)

    for bank in ('N16', 'N56'):
        with np.load(FROZEN / f'inputs/bank_V_{bank}.npz', allow_pickle=False) as source:
            V, raw, columns = source['V'], source['raw'], source['columns'].astype(str).tolist()
        active = np.ones(raw.shape, bool)
        for j, column in enumerate(columns):
            if column in DIGIT_COLUMNS: active[:, j] = digit_active[:, DIGIT_COLUMNS.index(column)]
        for fold in range(5):
            archive = ATTEMPT / 'checkpoints' / f'{bank}_fold{fold}'
            model = json.loads((archive / 'models.json').read_text())
            identity = json.loads((archive / 'TRAINING_IDENTITY.json').read_text())
            receipt = json.loads((archive / 'COMPLETE.json').read_text())
            if sha(archive / 'models.json') != receipt['models_sha256'] or sha(archive / 'predictions.npz') != receipt['predictions_sha256']:
                raise RuntimeError('Archived model/score checkpoint identity mismatch')
            train, test = np.flatnonzero(data['fold'] != fold), np.flatnonzero(data['fold'] == fold)
            if identity['train_ids'] != data['idx'][train].astype(str).tolist() or identity['test_ids'] != data['idx'][test].astype(str).tolist():
                raise RuntimeError('Archived donor/test semantic identities differ')
            if set(data['group'][train]) & set(data['group'][test]): raise RuntimeError('Source-group overlap')
            train_rows, train_off = select_rows(offsets, train)
            test_rows, test_off = select_rows(offsets, test)
            folder = DEST / 'checkpoints' / f'{bank}_fold{fold}'; folder.mkdir(parents=True, exist_ok=True)
            lsml = model['LSML_CONT']
            if 'diagnostic' not in lsml or lsml['diagnostic']['K'] < 3:
                failures.append(dict(bank=bank, fold=fold, reason='NO_VALID_DONOR_GROUPS')); continue
            groups = lsml['diagnostic']['group_weights']
            dump(folder / 'DONOR_GROUP_MODEL.json', dict(source_model_sha256=sha(archive / 'models.json'),
                 training_identity_sha256=sha(archive / 'TRAINING_IDENTITY.json'), LSML_CONT=lsml,
                 train_ids=data['idx'][train], test_ids=data['idx'][test], train_offsets=train_off, test_offsets=test_off))
            relative_family(bank, fold, 'OOF_DONOR', V[train_rows], train_off, groups, folder, model.get('LEGACY_MOM'))
            try:
                representation = fit_vote_representation(raw[train_rows], V[train_rows], active[train_rows], train_off,
                                                        groups, cross_weights=lsml['diagnostic']['cross_weights'])
                train_score, q80_train, train_diag = apply_vote_representation(representation, raw[train_rows], V[train_rows],
                                                                             active[train_rows], train_off, kind='absolute')
                test_score, q80_test, test_diag = apply_vote_representation(representation, raw[test_rows], V[test_rows],
                                                                         active[test_rows], test_off, kind='absolute')
                if np.max(np.abs(train_score.mean(0))) > 1e-10:
                    raise RuntimeError('ABS_ZERO requires frozen donor-centered absolute scores')
                dump(folder / 'ABSOLUTE_REPRESENTATION.json', dict(model=representation, train=train_diag, test=test_diag))
                heldout = dict(idx=data['idx'][test], offsets=test_off, source_group=data['group'][test],
                               absolute_scores=test_score, train_idx=data['idx'][train], train_offsets=train_off)
                with np.load(archive / 'predictions.npz', allow_pickle=False) as prediction:
                    if not np.array_equal(prediction['idx'], data['idx'][test]) or not np.array_equal(prediction['offsets'], test_off):
                        raise RuntimeError('Archived heldout score semantic identities differ')
                    for method in ('IU_PCR', 'LSML_CONT'):
                        heldout[method + '_score'] = prediction[method + '_score']
                for scenario, training, testing in (
                    ('ABS_Q80', q80_train, q80_test),
                    ('ABS_ZERO', np.where(train_score > 0., 1, -1).astype(np.int8), np.where(test_score > 0., 1, -1).astype(np.int8))):
                    fitted = record(dict(bank=bank, fold=fold, scope='OOF_DONOR', scenario=scenario,
                                         quota=None, permutation_seed=None), training, train_off)
                    heldout[scenario + '_votes'] = testing
                    strict = fitted['STRICT']
                    if strict.get('status') == 'PASS':
                        probability = posterior(testing, strict)
                        dist = count_distribution(probability, test_off)
                        heldout[scenario + '_posterior'] = probability
                        heldout[scenario + '_count'] = np.asarray(dist['mean_half_up'], int)
                        heldout[scenario + '_expected_count'] = np.asarray(dist['mean'], float)
                    else:
                        heldout[scenario + '_posterior'] = np.full(len(testing), np.nan)
                        heldout[scenario + '_count'] = np.full(len(test), -1, int)
                        heldout[scenario + '_expected_count'] = np.full(len(test), np.nan)
                    dump(folder / (scenario + '_HELDOUT_STATUS.json'), dict(status=strict.get('status'),
                         failure_reason=strict.get('failure_reason'), statistics=vote_statistics(testing, test_off),
                         threshold=representation['absolute']['threshold'] if scenario == 'ABS_Q80' else [0.] * len(groups),
                         score_source_sha256=sha(archive / 'predictions.npz')))
                np.savez_compressed(folder / 'absolute_heldout.npz', **heldout)
            except Exception as exc:
                failures.append(dict(bank=bank, fold=fold, reason='ABSOLUTE_REPRESENTATION_FAILURE', exception=repr(exc)))
                dump(folder / 'ABSOLUTE_FAILURE.json', failures[-1])
            print('QUOTA_FOLD_DONE', bank, fold, flush=True)
        if bank == 'N16':
            folder = DEST / 'checkpoints/FULL_REFERENCE_N16'; folder.mkdir(parents=True, exist_ok=True)
            fitted = fit_labelfree(V, 'LSML_CONT')
            with np.load(FROZEN / 'inputs/bank_N16.npz', allow_pickle=False) as reference:
                parity = float(np.max(np.abs(answer_z(fitted.predict(V), offsets) - reference['S__LSML_CONT'])))
            if parity > 1e-9: raise RuntimeError('Full historical N16 canonical score parity failed')
            dump(folder / 'FULL_REFERENCE_MODEL.json', dict(model=as_dict(fitted), transductive=True,
                                                           canonical_score_max_abs=parity, labels_loaded=False))
            relative_family(bank, 'FULL', 'FULL_REFERENCE_TRANSDUCTIVE', V, offsets,
                            fitted.diagnostic['group_weights'], folder)
            print('QUOTA_FULL_REFERENCE_DONE', flush=True)
    write_csv(DEST / 'QUOTA_RESULTS.csv', rows)
    dump(DEST / 'QUOTA_MODELS.json', model_states)
    science = dict(status='COMPLETE', diagnostic_only=True, candidate_count_unchanged=7,
                   labels_loaded=False, n_rows=len(rows), scope_failures=failures, runtime_s=time.time() - started,
                   caveats=configuration()['interpretation'],
                   permutation_null=configuration()['permutation_null'],
                   absolute_zero_caveat='A frozen donor mean is not a guaranteed error boundary; no utility-based threshold choice.',
                   synthetic='See EXACT_TEACHER.json for unchanged latent prevalence under changing vote marginals.',
                   quota_sweep=[], permutation_summaries=[])
    for scope, bank, fold in sorted(set((row.get('scope'), row.get('bank'), str(row.get('fold'))) for row in rows)):
        selected = [row for row in rows if row.get('scope') == scope and row.get('bank') == bank and str(row.get('fold')) == fold
                    and row.get('length_bin') == 'ALL']
        for estimator in ('ARCHIVED_CLIPPED', 'STRICT'):
            sweep = [row for row in selected if row.get('scenario') == 'RELATIVE_QUOTA' and row.get('estimator') == estimator]
            valid = [row for row in sweep if row.get('fit_status') == 'PASS']
            slope = float(np.polyfit([row['actual_mean_vote_rate'] for row in valid], [row['prevalence_hat'] for row in valid], 1)[0]) if len(valid) == 3 else None
            science['quota_sweep'].append(dict(scope=scope, bank=bank, fold=fold, estimator=estimator,
                                              n_valid=len(valid), slope_pi_vs_actual_vote_rate=slope,
                                              points=[{key: row.get(key) for key in ('quota', 'actual_mean_vote_rate', 'prevalence_hat', 'fit_status')} for row in sweep]))
            shuffled = [row for row in selected if row.get('scenario') == 'PERMUTED_QUOTA' and row.get('estimator') == estimator]
            values = [row['prevalence_hat'] for row in shuffled if row.get('fit_status') == 'PASS']
            science['permutation_summaries'].append(dict(scope=scope, bank=bank, fold=fold, estimator=estimator,
                n_valid=len(values), n_attempted=len(shuffled), prevalence_mean=float(np.mean(values)) if values else None,
                prevalence_min=min(values) if values else None, prevalence_max=max(values) if values else None))
    dump(DEST / 'QUOTA_SCIENCE.json', science)
    validate_freeze()
    # Logs remain mutable through the last progress print and are not model
    # evidence. Seal deterministic artifacts only, while retaining logs.
    outputs = [path for path in DEST.rglob('*') if path.is_file() and path.name != 'QUOTA_SEAL.json'
               and path.suffix != '.log' and '__pycache__' not in str(path)]
    dump(DEST / 'QUOTA_SEAL.json', dict(status='FROZEN_BEFORE_EXPLORATORY_LABEL_EVALUATION',
         files={str(path.relative_to(DEST)): dict(sha256=sha(path), size=path.stat().st_size) for path in sorted(outputs)},
         created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), labels_loaded=False, candidate_count_unchanged=7))
    print('QUOTA_COMPLETE', len(rows), 'rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'synthetic', 'fit'), required=True)
    arguments = parser.parse_args()
    if arguments.stage == 'prepare': prepare()
    elif arguments.stage == 'synthetic': exact_teacher()
    else: fit()
