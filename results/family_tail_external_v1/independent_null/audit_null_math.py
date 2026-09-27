"""Independent downstream null/math audit; never imports production scoring math.

The reviewer authored hist26 extraction, so this is NOT an independent audit
of extraction. All quality access requires explicit post-seal authorization.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = OUT.parent
CELLS = {'hard2verify_qwen3_8b': 'hard2verify', 'socratic_qwen3_8b': 'socratic',
         'socratic_qwq32b': 'socratic'}
SEED = 2026092407
ARMS = ('F15_tailtie_lsml', 'F15_equal')
LOCK = ROOT/'results/family_tail_transfer_v1/TRANSFER_LOCK_V1.json'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def z(x, final=False):
    x = np.asarray(x, float)
    sd = x.std(axis=0)
    return np.divide(x-x.mean(axis=0), sd, out=np.zeros_like(x), where=sd > (1e-8 if final else 1e-12))


def families(payload, lock):
    # Deliberately accepts only features/names; no gold or metadata are used.
    x = np.asarray(payload['features'], float)
    names = payload['feature_names']
    recipe = lock['recipe']
    oriented = {name: z(x[:, names.index(name)])*recipe['source_signs'][name]
                for name in recipe['channels_28']}
    return np.column_stack([z(np.column_stack([oriented[n] for n in members]).mean(axis=1))
                            for members in recipe['families_15'].values()])


def predict(family, arm, lock):
    fit = lock['deployment'][arm]
    w = np.array([fit['weights'][n] for n in lock['recipe']['families_15']])
    risk = z(family@w, final=True)
    return risk, (risk < fit['q80_threshold_fold4']).astype(bool)


def score(y, predictions, benchmark):
    y = np.asarray(y, bool)
    p = np.asarray(predictions, bool)
    tp = np.count_nonzero(p & y, axis=1)
    fp = np.count_nonzero(p & ~y, axis=1)
    tn = np.count_nonzero(~p & ~y, axis=1)
    fn = np.count_nonzero(~p & y, axis=1)
    if benchmark == 'socratic':
        # Half of each class F1, summed: PRMScore, not BF1.
        return tp/np.maximum(2*tp+fp+fn, 1) + tn/np.maximum(2*tn+fp+fn, 1)
    good = tp/np.maximum(tp+fn, 1)
    bad = tn/np.maximum(tn+fp, 1)
    return np.divide(2*good*bad, good+bad, out=np.zeros_like(good), where=(good+bad)>0)


def tail_marks(x):
    x = np.asarray(x, float)
    k = int(np.ceil(len(x)*.2))
    result = np.zeros_like(x)
    for j, values in enumerate(x.T):
        cut = np.sort(values)[-k]
        above = values > cut
        tie = values == cut
        result[above, j] = 1
        result[tie, j] = (k-above.sum())/tie.sum()
    return result


def synthetic_math():
    lock = load(LOCK)
    checks = 0
    rng = np.random.default_rng(SEED)
    for n in range(1, 101):
        x = rng.integers(0, 4, size=(n, 7)).astype(float)
        x[:, -1] = 11
        marks = tail_marks(x)
        assert np.all((marks >= 0) & (marks <= 1))
        np.testing.assert_allclose(marks.sum(axis=0), np.ceil(.2*n), atol=1e-12)
        centered = marks-marks.mean(axis=0)
        np.testing.assert_allclose(centered.sum(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(centered[:, -1], 0, atol=1e-12)
        np.testing.assert_allclose(tail_marks(x*17+4), marks)
        checks += 1
    eigen_checks = []
    for covariance in [-10, -1, -.01, .01, 1, 10]:
        matrix = np.array([[0, covariance], [covariance, 0]])
        values, vectors = np.linalg.eigh(matrix)
        v = vectors[:, -1]
        np.testing.assert_allclose(abs(v), np.sqrt(.5), atol=1e-14)
        eigen_checks.append({'off_diagonal': covariance, 'absolute_weights': abs(v).tolist()})
    fit = lock['deployment']['F15_tailtie_lsml']
    group_names = sorted(set(fit['groups'].values()))
    masses = {str(g): {'members': [n for n in fit['weights'] if fit['groups'][n] == g],
                      'l1_mass': sum(abs(v) for n, v in fit['weights'].items() if fit['groups'][n] == g),
                      'l2_norm': float(np.linalg.norm([v for n, v in fit['weights'].items() if fit['groups'][n] == g]))}
              for g in group_names}
    assert fit['K'] == 2 and len(group_names) == 2
    np.testing.assert_allclose([m['l2_norm'] for m in masses.values()], masses[str(group_names[0])]['l2_norm'], atol=1e-12)
    assert np.isclose(sum(abs(v) for v in fit['weights'].values()), 1)
    original = rng.normal(size=(101, 15))
    transformed = original.copy()
    transformed[:, 0] = np.exp(transformed[:, 0])
    np.testing.assert_array_equal(tail_marks(original), tail_marks(transformed))
    continuous_a, _ = predict(z(original), 'F15_tailtie_lsml', lock)
    continuous_b, _ = predict(z(transformed), 'F15_tailtie_lsml', lock)
    continuous_delta = float(np.max(abs(continuous_a-continuous_b)))
    assert continuous_delta > 1e-3
    # Hand-computed, class-balanced 4-step case: TP=TN=FP=FN=1.
    for benchmark in ('hard2verify', 'socratic'):
        np.testing.assert_allclose(score([1, 1, 0, 0], [[1, 0, 1, 0]], benchmark), .5)
        np.testing.assert_allclose(score([1, 1, 0, 0], [[1, 1, 0, 0]], benchmark), 1)
    save('SYNTHETIC_MATH.json', {'status': 'PASS', 'tail_tie_cases': checks,
         'k2_eigen_cases': eigen_checks, 'locked_group_masses': masses,
         'caveat': 'K=2 off-diagonal eigenvectors have equal absolute OUTER coefficients for nonzero covariance. Final group L1 masses differ because within-group eigenvectors have unit L2, not unit L1. No between-group reliability identification. Zero covariance is degenerate.',
         'deployment_caveat': 'Fractional centered quota marks fit the weights, but continuous family scores are deployed. Binary-classifier reliability guarantees do not automatically carry over.',
         'equal_tail_marks_but_continuous_score_max_delta': continuous_delta,
         'lock_sha256': sha(LOCK), 'script_sha256': sha(__file__)})


def run(authorization, draws):
    if authorization != 'root-authorized-after-all-cells-sealed' or draws != 200:
        raise ValueError('requires explicit post-seal authorization and preregistered 200 draws')
    started = time.perf_counter()
    lock = load(LOCK)
    all_seals = load(RUN/'ALL_CELLS_SEALED.json')
    assert all_seals['lock_sha256'] == sha(LOCK)
    rows = {}
    for cell in CELLS:
        rows[cell] = load(RUN/cell/'PREDICTIONS.json')
        seal = load(RUN/cell/'SEAL.json')
        assert seal == all_seals['seals'][cell]
        assert seal['prediction_sha256'] == digest(rows[cell])
        assert seal['lock_sha256'] == sha(LOCK)
        assert len(rows[cell]) == (200 if cell.startswith('hard') else 2995)
    # No evaluator-only labels are accessed until every cell passes its seal.
    labels = {bench: {v['uid']: v for v in load(ROOT/'scratch/external_generalization_private/inputs/evaluator_only'/f'{bench}.json')}
              for bench in set(CELLS.values())}
    methods = list(lock['rows'])
    contrasts = lock['external_primary_contrasts']
    null_output, feature_output, feature_rows = {}, {}, []
    raw_null = {}
    for cell, benchmark in CELLS.items():
        ids = sorted(rows[cell])
        gold = labels[benchmark]
        assert set(ids) == set(gold)
        included = np.concatenate([gold[u]['include'] for u in ids]).astype(bool)
        y = np.concatenate([gold[u]['correct'] for u in ids]).astype(bool)[included]
        p = np.array([np.concatenate([rows[cell][u]['predictions'][a] for u in ids])[included] for a in methods], bool)
        actual = score(y, p, benchmark)
        lengths = [sum(gold[u]['include']) for u in ids]
        off = np.r_[0, np.cumsum(lengths)]
        rng = np.random.default_rng(SEED+(benchmark == 'socratic'))
        global_null = np.array([score(rng.permutation(y), p, benchmark) for _ in range(draws)])
        rng = np.random.default_rng(SEED+100+(benchmark == 'socratic'))
        within_null = np.array([score(np.concatenate([rng.permutation(y[a:b]) for a, b in zip(off[:-1], off[1:])]), p, benchmark)
                                for _ in range(draws)])
        details = {}
        for i, arm in enumerate(methods):
            details[arm] = {'observed': float(actual[i]), 'global_null_mean': float(global_null[:, i].mean()),
                'global_null_95': np.quantile(global_null[:, i], [.025, .975]).tolist(),
                'within_answer_null_mean': float(within_null[:, i].mean()),
                'within_answer_null_95': np.quantile(within_null[:, i], [.025, .975]).tolist(),
                'global_plus_one_upper_p': float((1+(global_null[:, i]>=actual[i]).sum())/(1+draws)),
                'within_plus_one_upper_p': float((1+(within_null[:, i]>=actual[i]).sum())/(1+draws))}
        contrast_results = []
        for arm, control in contrasts:
            a, b = methods.index(arm), methods.index(control)
            contrast_results.append({'arm': arm, 'control': control, 'observed_delta': float(actual[a]-actual[b]),
                'global_null_delta_95': np.quantile(global_null[:, a]-global_null[:, b], [.025, .975]).tolist(),
                'within_answer_null_delta_95': np.quantile(within_null[:, a]-within_null[:, b], [.025, .975]).tolist()})
        null_output[cell] = {'answers': len(ids), 'steps': len(y), 'correct': int(y.sum()),
                             'arms': details, 'contrasts': contrast_results}
        raw_null[cell+'_global'] = global_null
        raw_null[cell+'_within'] = within_null
        changed_pred = {arm: [] for arm in ARMS}
        summaries = {arm: {'answers_score_changed': 0, 'steps_score_changed': 0, 'decisions_changed': 0,
                          'baseline_replay_max_abs': 0.0, 'permuted_score_max_abs_delta': 0.0} for arm in ARMS}
        permuted_family_changed = 0
        for i, uid in enumerate(ids):
            payload = rows[cell][uid]
            original = families(payload, lock)
            f = list(lock['recipe']['families_15']).index('level_entropy')
            permutation_seed = SEED+int(hashlib.sha256(uid.encode()).hexdigest()[:12], 16)
            permutation = np.random.default_rng(permutation_seed).permutation(len(original))
            perturbed = original.copy()
            perturbed[:, f] = original[permutation, f]
            np.testing.assert_array_equal(np.sort(original[:, f]), np.sort(perturbed[:, f]))
            assert np.array_equal(original[:, np.arange(15)!=f], perturbed[:, np.arange(15)!=f])
            permuted_family_changed += int(not np.array_equal(original, perturbed))
            # Gold/metadata mutation cannot affect the features-only entry point.
            fake = dict(payload, correct=[17], labels='mutated', category='adversarial')
            np.testing.assert_array_equal(families(fake, lock), original)
            valid = np.asarray(payload['nonempty'], bool)
            item = {'cell': cell, 'uid': uid, 'seed': permutation_seed, 'nonempty_steps': int(valid.sum()), 'arms': {}}
            for arm in ARMS:
                risk, pred = predict(original, arm, lock)
                risk_alt, pred_alt = predict(perturbed, arm, lock)
                saved = np.array(payload['scores'][arm], float)[valid]
                error = float(np.max(abs(risk-saved)))
                assert error <= 1e-10, (uid, arm, error)
                assert np.array_equal(pred, np.asarray(payload['predictions'][arm], bool)[valid])
                expanded = np.zeros(len(valid), bool)
                expanded[valid] = pred_alt
                changed_pred[arm].append(expanded)
                delta = abs(risk_alt-risk)
                stats = summaries[arm]
                stats['answers_score_changed'] += int(np.any(delta>1e-10))
                stats['steps_score_changed'] += int(np.sum(delta>1e-10))
                stats['decisions_changed'] += int(np.sum(pred_alt!=pred))
                stats['baseline_replay_max_abs'] = max(stats['baseline_replay_max_abs'], error)
                stats['permuted_score_max_abs_delta'] = max(stats['permuted_score_max_abs_delta'], float(delta.max()))
                item['arms'][arm] = {'score_changed_steps': int(np.sum(delta>1e-10)), 'decision_changed_steps': int(np.sum(pred_alt!=pred)), 'max_score_delta': float(delta.max())}
            feature_rows.append(item)
        for arm in ARMS:
            alt = np.concatenate(changed_pred[arm])[included]
            summaries[arm]['original_metric'] = float(actual[methods.index(arm)])
            summaries[arm]['permuted_metric'] = float(score(y, alt[None, :], benchmark)[0])
            summaries[arm]['metric_delta'] = summaries[arm]['permuted_metric']-summaries[arm]['original_metric']
        feature_output[cell] = {'checked': len(ids), 'family_permutation_changes_answers': permuted_family_changed,
                                'label_mutation_invariant_answers': len(ids), 'arms': summaries}
        print('null/math cell complete', cell, len(ids), flush=True)
    np.savez_compressed(OUT/'LABEL_NULL_DRAWS.npz', **raw_null)
    save('LABEL_NULL.json', {'scope': 'FULL 6190 answers; class totals preserved globally or separately within each answer. Same deterministic label draws for the two Socratic backbones. Predictions fixed for all ten arms.',
        'draws_per_scheme_per_cell': draws, 'seed': SEED, 'cells': null_output,
        'interpretation': 'Diagnostic null distributions, not replacements for registered paired question bootstrap; empirical p floor 1/201 and no multiplicity claim.',
        'source_seal_sha256': sha(RUN/'ALL_CELLS_SEALED.json'), 'script_sha256': sha(__file__)})
    save('FEATURE_NULL.json', {'scope': 'FULL', 'n_checked': len(feature_rows), 'n_total': 6190,
        'intervention': 'One UID-seeded permutation of standardized level_entropy family across nonempty steps within each answer, preserving its marginal and the other 14 families. Apply frozen candidate/control weights and thresholds; no fitting. Other answers untouched. Exploratory diagnostic, not a registered competing arm.',
        'seed': SEED, 'family': 'level_entropy', 'cells': feature_output,
        'independence_limit': 'Reviewer implemented hist26 extractor; this review independently reimplements downstream family/scoring/metric math and is not independent extraction verification.',
        'rows': feature_rows, 'lock_sha256': sha(LOCK), 'source_seal_sha256': sha(RUN/'ALL_CELLS_SEALED.json'),
        'script_sha256': sha(__file__), 'elapsed_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic', 'full'], required=True)
    parser.add_argument('--authorization')
    parser.add_argument('--draws', type=int, default=200)
    args = parser.parse_args()
    if args.mode == 'synthetic':
        synthetic_math()
    else:
        run(args.authorization, args.draws)
