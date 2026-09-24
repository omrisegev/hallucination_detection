"""Independent red-team C: bounded feature sensitivity; sealed full-label null.

Feature mode never reads annotations. Label mode requires an explicit post-seal
authorization argument and validates all prediction seals before opening labels.
One process, one BLAS thread; no scoring-package edits.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ[key] = '1'
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from spectral_utils.external_generalization import scoring
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash, require_seal
from spectral_utils.external_generalization.contracts import digest

OUT = Path(__file__).resolve().parent
EVAL = OUT.parent
CELLS = {'hard2verify_qwen3_8b': 'hard2verify', 'socratic_qwen3_8b': 'socratic', 'socratic_qwq32b': 'socratic'}
SEED = 2026092403


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def feature_null():
    bundle_path = EVAL / 'source/BUNDLE.json'
    bundle = load(bundle_path)
    results = []
    source = ROOT / 'scratch/external_generalization_private/evaluation_archives'
    for cell in CELLS:
        timing = load(source / cell / 'TIMING.json')
        ordered = sorted(timing['measurements'], key=lambda r: (r['answer_tokens'], r['uid']))
        indices = np.rint(np.linspace(0, len(ordered) - 1, 12)).astype(int)
        assert len(set(indices)) == 12
        for index in indices:
            uid = ordered[index]['uid']
            path = source / cell / 'records' / (digest(uid) + '.record.json')
            raw = load(path)['payload']['telemetry']
            matrix = scoring.validate_telemetry(raw)
            spans = np.asarray(raw['step_token_spans'], int)
            eligible = np.zeros(len(matrix), bool)
            for left, right in spans:
                eligible[left:right] = True
            positions = np.flatnonzero(eligible)
            seed = SEED + int(digest([cell, uid])[:8], 16)
            permutation = np.random.default_rng(seed).permutation(positions)
            changed = matrix.copy()
            changed[positions, 0] = matrix[permutation, 0]
            assert np.array_equal(matrix[:, 1:], changed[:, 1:])
            assert np.array_equal(np.sort(matrix[positions, 0]), np.sort(changed[positions, 0]))
            with patch.object(scoring, 'validate_telemetry', return_value=matrix):
                baseline = scoring.score_answer(raw, bundle)
            with patch.object(scoring, 'validate_telemetry', return_value=changed):
                perturbed = scoring.score_answer(raw, bundle)
            arms = {}
            for arm in scoring.ALL_ARMS:
                a = np.asarray([np.nan if x is None else x for x in baseline['scores'][arm]])
                b = np.asarray([np.nan if x is None else x for x in perturbed['scores'][arm]])
                valid = np.isfinite(a) & np.isfinite(b)
                assert np.array_equal(np.isfinite(a), np.isfinite(b))
                arms[arm] = {'score_max_abs_delta': float(np.max(np.abs(a[valid] - b[valid]))),
                             'changed_scores_atol1e_10': int(np.sum(np.abs(a[valid] - b[valid]) > 1e-10)),
                             'changed_decisions': int(np.sum(np.asarray(baseline['predictions'][arm]) != np.asarray(perturbed['predictions'][arm])))}
            assert arms['ct7']['score_max_abs_delta'] == 0
            d0, d1 = baseline['local'], perturbed['local']
            item = {'cell': cell, 'uid': uid, 'length_rank': int(index), 'tokens': len(matrix),
                    'valid_step_tokens': len(positions), 'permuted_values_changed': int(np.sum(matrix[:, 0] != changed[:, 0])),
                    'seed': seed, 'telemetry_sha256': file_hash(path), 'arms': arms,
                    'baseline_native': d0['native'], 'perturbed_native': d1['native'],
                    'baseline_fallback': d0.get('reason'), 'perturbed_fallback': d1.get('reason'),
                    'local_weights_changed': d0.get('weights') != d1.get('weights'),
                    'local_partition_changed': d0.get('groups') != d1.get('groups'),
                    'baseline_active_channels': d0['active_channels'], 'perturbed_active_channels': d1['active_channels']}
            results.append(item)
            atomic_json(OUT / 'FEATURE_NULL_PARTIAL.json', results)
            print('FEATURE', len(results), '/36', cell, flush=True)
    summary = {}
    for cell in CELLS:
        subset = [r for r in results if r['cell'] == cell]
        summary[cell] = {'checked': len(subset), 'total': 200 if cell.startswith('hard') else 2995,
                         'native_before': sum(r['baseline_native'] for r in subset),
                         'native_after': sum(r['perturbed_native'] for r in subset),
                         'local_weights_changed': sum(r['local_weights_changed'] for r in subset),
                         'local_partition_changed': sum(r['local_partition_changed'] for r in subset),
                         'arms': {a: {'answers_score_changed': sum(r['arms'][a]['changed_scores_atol1e_10'] > 0 for r in subset),
                                      'steps_score_changed': sum(r['arms'][a]['changed_scores_atol1e_10'] for r in subset),
                                      'decisions_changed': sum(r['arms'][a]['changed_decisions'] for r in subset)} for a in scoring.ALL_ARMS}}
    atomic_json(OUT / 'FEATURE_NULL.json', {'tag': 'FEASIBILITY', 'checked': 36, 'total': 6190,
        'feature': 'bank11 column0 q15_H1 (entropy orientation anchor)',
        'selection': '12 evenly spaced token-length ranks per cell, including shortest/longest; deterministic UID tie order',
        'intervention': 'one permutation across the UNION of all nonempty-step token spans within each answer, CROSSING step boundaries; NOT independent within-step permutations; all other columns and raw CT7 inputs unchanged; local fit reruns',
        'quality_evaluated': False, 'bundle_sha256': file_hash(bundle_path), 'script_sha256': file_hash(__file__),
        'code_hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT/'spectral_utils/external_generalization').rglob('*.py'))},
        'summary': summary, 'records': results})


def raw_metric(y, pred, benchmark):
    tp = np.sum(pred & y, axis=1); fp = np.sum(pred & ~y, axis=1)
    tn = np.sum(~pred & ~y, axis=1); fn = np.sum(~pred & y, axis=1)
    if benchmark == 'socratic':
        return tp / np.maximum(2*tp+fp+fn, 1) + tn / np.maximum(2*tn+fp+fn, 1)
    good = tp / np.maximum(tp+fn, 1); bad = tn / np.maximum(tn+fp, 1)
    return np.divide(2*good*bad, good+bad, out=np.zeros_like(good, dtype=float), where=good+bad > 0)


def label_null(inputs, authorization, draws):
    if authorization != 'root-authorized-after-all-cells-sealed':
        raise ValueError('explicit parent post-seal authorization required')
    all_seals = load(EVAL / 'ALL_CELLS_SEALED.json')
    bundle_hash = file_hash(EVAL / 'source/BUNDLE.json')
    if all_seals['bundle_sha256'] != bundle_hash:
        raise ValueError('bundle changed after seal')
    predictions = {}
    seals = {}
    for cell in CELLS:
        predictions[cell] = load(EVAL / cell / 'PREDICTIONS.json')
        seals[cell] = load(EVAL / cell / 'SEAL.json')
        require_seal(predictions[cell], seals[cell], bundle_hash)
        if seals[cell] != all_seals['seals'][cell]:
            raise ValueError('all-cells seal mismatch')
    # Annotation access occurs only after ALL three seals have passed.
    annotations = {b: {r['uid']: r for r in load(inputs/'evaluator_only'/(b+'.json'))} for b in set(CELLS.values())}
    contrasts = [(v+'_lsml', v+'_'+c) for v in ('frozen','local') for c in ('equal','partition_equal')] + [(v+'_lsml','ct7') for v in ('frozen','local')]
    output = {}
    for cell, bench in CELLS.items():
        ids = sorted(predictions[cell]); gold = annotations[bench]
        assert set(ids) == set(gold)
        mask = np.concatenate([gold[u]['include'] for u in ids]).astype(bool)
        y = np.concatenate([gold[u]['correct'] for u in ids]).astype(bool)[mask]
        pred = np.array([np.concatenate([predictions[cell][u]['predictions'][a] for u in ids])[mask] for a in scoring.ALL_ARMS], bool)
        observed = raw_metric(y, pred, bench)
        prevalence = pred.mean(axis=1)
        correct_prevalence = float(y.mean())
        # This is exact for macro F1's expectation because its denominators
        # depend only on fixed class/prediction totals. BF1 is nonlinear:
        # its plug-in expectation is contextual, the simulated mean is primary.
        independence_reference = (correct_prevalence*prevalence/(correct_prevalence+prevalence)
            + (1-correct_prevalence)*(1-prevalence)/(2-correct_prevalence-prevalence)) if bench == 'socratic' else 2*prevalence*(1-prevalence)
        rng = np.random.default_rng(SEED + (0 if bench == 'hard2verify' else 1))
        null = np.array([raw_metric(rng.permutation(y), pred, bench) for _ in range(draws)])
        lengths = [int(np.sum(gold[u]['include'])) for u in ids]
        edges = np.r_[0, np.cumsum(lengths)]
        within_draws = min(draws, 200)
        within_rng = np.random.default_rng(SEED + 100 + (0 if bench == 'hard2verify' else 1))
        within = []
        for _ in range(within_draws):
            shuffled = np.concatenate([within_rng.permutation(y[a:b]) for a,b in zip(edges[:-1],edges[1:])])
            within.append(raw_metric(shuffled, pred, bench))
        within = np.asarray(within)
        comparison = []
        for left, right in contrasts:
            i, j = scoring.ALL_ARMS.index(left), scoring.ALL_ARMS.index(right)
            d = null[:, i] - null[:, j]
            wd = within[:, i] - within[:, j]
            comparison.append({'left': left, 'right': right, 'observed_delta': float(observed[i]-observed[j]),
                               'null_mean_delta': float(d.mean()), 'null_std_delta': float(d.std()),
                               'null_q025_q975': np.quantile(d, [.025,.975]).tolist(),
                               'within_answer_null_mean_delta': float(wd.mean()),
                               'within_answer_null_std_delta': float(wd.std()),
                               'within_answer_null_q025_q975': np.quantile(wd,[.025,.975]).tolist()})
        output[cell] = {'answers_checked': len(ids), 'answers_total': 200 if bench == 'hard2verify' else 2995,
                        'included_steps': len(y), 'correct_steps': int(y.sum()), 'correct_prevalence': correct_prevalence,
                        'within_answer_draws': within_draws,
                        'arms': {a: {'observed': float(observed[i]), 'null_mean': float(null[:,i].mean()),
                                      'prediction_correct_prevalence': float(prevalence[i]),
                                      'independence_reference_from_expected_counts': float(independence_reference[i]),
                                      'null_std': float(null[:,i].std()), 'null_q025_q975': np.quantile(null[:,i],[.025,.975]).tolist()}
                                 for i,a in enumerate(scoring.ALL_ARMS)}, 'contrasts': comparison}
        for i, arm in enumerate(scoring.ALL_ARMS):
            output[cell]['arms'][arm].update(within_answer_null_mean=float(within[:,i].mean()),
                within_answer_null_std=float(within[:,i].std()),
                within_answer_null_q025_q975=np.quantile(within[:,i],[.025,.975]).tolist())
    atomic_json(OUT/'LABEL_NULL.json', {'checked': 6190, 'total': 6190, 'draws': draws, 'seed': SEED,
        'null': 'global pooled included-step label permutation within dataset; fixed sealed predictions; class counts preserved; identical permutation stream across Socratic backbones',
        'limitation': 'diagnostic random-label null; exchangeability across dependent steps is not assumed for inferential claims; not a source-group bootstrap',
        'within_answer_null': '200 permutations (or draws if lower), full population; permute included-step labels independently inside each answer, preserving every answer class composition; no inferential claims',
        'chance_interpretation': 'macro F1 exact global-shuffle expectation follows fixed truth/prediction prevalence; BF1 reference uses expected counts and is a plug-in, use simulated mean for finite-sample expectation; neither is necessarily 0.5',
        'seals': seals, 'bundle_sha256': bundle_hash, 'script_sha256': file_hash(__file__),
        'annotation_hashes': {b:file_hash(inputs/'evaluator_only'/(b+'.json')) for b in annotations}, 'cells': output})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('feature','labels'), required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--authorization', default='')
    parser.add_argument('--draws', type=int, default=1000)
    args = parser.parse_args()
    if args.mode == 'feature': feature_null()
    else: label_null(args.inputs, args.authorization, args.draws)
