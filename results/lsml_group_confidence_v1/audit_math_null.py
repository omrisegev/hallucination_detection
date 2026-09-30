"""Independent mathematical and scoring-only negative-control audit.

Does not read evaluation artifacts or other reviewers' output, and does not tune.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
from scipy.stats import rankdata

ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
sys.path.insert(0, str(ROOT))
OUT = ROOT/'results/lsml_group_confidence_v1'
from spectral_utils.lsml_group_confidence import answer_standardize, score, fit_binary_tree, collapse_duplicates, binary_tail
from spectral_utils.lsml_group_confidence_experiment import load_inputs

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

seal = json.loads((OUT/'SEAL.json').read_text())
assert sha(OUT/'PREDICTIONS.npz') == seal['predictions_sha256']
assert sha(OUT/'FITS.json') == seal['fits_sha256']
frozen = json.loads((OUT/'FREEZE.json').read_text())
for rel, digest in frozen['code'].items():
    assert sha(ROOT/rel) == digest
d = load_inputs(ROOT)
with np.load(OUT/'PREDICTIONS.npz') as archive:
    z = {key: archive[key] for key in archive.files}
error = np.load(d['paths']['arrays'])['labels'].astype(bool)
off, ans = d['off'], d['ans']
arms = ['confidence_sum', 'confidence_average', 'raw28_equal', 'family15_equal']
eligible = [i for i, (a, b) in enumerate(zip(off[:-1], off[1:]))
            if ans.cell[i].startswith('prm') and error[a:b].any() and not error[a:b].all()]
assert len(eligible) == 6030

# Precompute ranks; null permutations preserve each answer's length and prevalence.
ranked = []
for i in eligible:
    a, b = off[i:i+2]
    y = error[a:b]
    pos, neg = int(y.sum()), int((~y).sum())
    ranked.append((y, np.vstack([rankdata(z[arm+'_score'][a:b]) for arm in arms]), pos, neg))
observed = np.zeros(len(arms))
for y, ranks, pos, neg in ranked:
    observed += (ranks[:, y].sum(1)-pos*(pos+1)/2)/(pos*neg)
observed /= len(eligible)
null = []
seeds = list(range(202609240, 202609260))
for seed in seeds:
    rng = np.random.default_rng(seed)
    total = np.zeros(len(arms))
    for y, ranks, pos, neg in ranked:
        perm = rng.permutation(y)
        total += (ranks[:, perm].sum(1)-pos*(pos+1)/2)/(pos*neg)
    null.append(total/len(eligible))
null = np.asarray(null)

# Feature intervention: use original fitted models and fixed original thresholds.
# Permute one actual feature independently within EVERY answer, then recompute.
xp = d['x'].copy()
column = d['names'].index('q15_H1')
rng = np.random.default_rng(20260924)
for a, b in zip(off[:-1], off[1:]):
    xp[a:b, column] = rng.permutation(xp[a:b, column])
sf = np.repeat(ans.fold.to_numpy(), np.diff(off))
fits = json.loads((OUT/'FITS.json').read_text())
permuted = {arm: np.empty(off[-1]) for arm in arms}
permuted_pred = {arm: np.empty(off[-1], dtype=bool) for arm in arms}
vp = np.column_stack([xp[:, d['groups'] == g].mean(1) for g in range(len(d['family_names']))])
vp = answer_standardize(vp, off)
duplicate_disagreements, models = [], []
for entry in fits:
    f = entry['test']
    model = json.loads((OUT/f'MODEL_{f}.json').read_text())
    assert sha(OUT/f'MODEL_{f}.json') == entry['model_hash']
    models.append({'fold': f, 'converged': model['converged'], 'iterations': model['iterations'],
                   'minimum_objective_increment': float(np.min(np.diff(model['objective']))),
                   'offblock_relative_residual': model['initialization']['offblock_relative_residual'],
                   'group_sizes': np.bincount(model['groups']).tolist()})
    e = np.asarray(entry['expansion'])
    select = sf == f
    baseline = answer_standardize(score(d['x']@e, model), off)
    np.testing.assert_allclose(baseline[select], z['confidence_sum_score'][select], atol=1e-12)
    raw = {'confidence_sum': score(xp@e, model),
           'confidence_average': score(xp@e, model, average_evidence=True),
           'raw28_equal': xp.mean(1), 'family15_equal': vp.mean(1)}
    for arm, values in raw.items():
        ss = answer_standardize(values, off)
        permuted[arm][select] = ss[select]
        permuted_pred[arm][select] = ss[select] < entry['thresholds'][arm]
    # In training-only duplicate collapse, identical bits could diverge outside fit.
    bb = binary_tail(d['x'], off) @ e
    duplicate_disagreements.append({'fold': f, 'fractional_test_binary_cells': int(np.sum((bb[select] != 0)&(bb[select] != 1)))})

feature = {}
for arm in arms:
    auc = []
    for i in eligible:
        a,b = off[i:i+2]
        y = error[a:b]
        p, n = int(y.sum()), int((~y).sum())
        auc.append((rankdata(permuted[arm][a:b])[y].sum()-p*(p+1)/2)/(p*n))
    delta = permuted[arm]-z[arm+'_score']
    feature[arm] = {'mean_absolute_score_change': float(np.mean(np.abs(delta))),
                   'changed_scores_above_1e_minus10': int(np.sum(np.abs(delta)>1e-10)),
                   'changed_step_decisions': int(np.sum(permuted_pred[arm] != z[arm+'_pred'])),
                   'permuted_within_auc': float(np.mean(auc)),
                   'within_auc_change': float(np.mean(auc)-observed[arms.index(arm)])}

spec = importlib.util.spec_from_file_location('audit_tests', ROOT/'tests/test_lsml_group_confidence.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
tests.test_exact_latent_enumeration_and_binary_continuation()
tests.test_linearization_and_saturation()
tests.test_duplicate_invariance_including_binary_equivalent_continuous_columns()
m = {'pi': .35, 'a': [[.15, .85]]*4, 'theta': [[.2, .8]]*3+[[0,1]]*3,
     'groups': [0,0,0,1,2,3], 'mu': [.4]*6, 'sd': [.49]*6}
train, _ = tests.sample(m, 30000, 40)
test, y = tests.sample(m, 30000, 41)
full = fit_binary_tree(train, m['groups'])
subset = [0,3,4,5]
small = fit_binary_tree(train[:, subset], [0,1,2,3])
def loss(pred):
    return float(np.mean(np.logaddexp(0, pred)-y*pred))
full_scores = score(test, full, continuous=False)
small_scores = score(test[:, subset], small, continuous=False)
duplicate_train = np.column_stack([train, train[:,0], train[:,0]])
duplicate_test = np.column_stack([test, test[:,0], test[:,0]])
reduced, expansion, groups, _ = collapse_duplicates(duplicate_train, duplicate_train.astype(float), m['groups']+[0,0])
dup = fit_binary_tree(reduced, groups)
duplicate_scores = score(duplicate_test@expansion, dup, continuous=False)
np.testing.assert_allclose(full_scores, duplicate_scores, atol=1e-12)

result = {
    'audit': 'math and scoring-only negative controls; independent reviewer C',
    'read_summary_or_other_audits': False,
    'hashes': {'PREDICTIONS.npz': sha(OUT/'PREDICTIONS.npz'), 'FITS.json': sha(OUT/'FITS.json'),
               'audit_script': sha(Path(__file__))},
    'math': {
        'verdict': 'Binary likelihood, EM updates, local/global gauge and derivative are algebraically consistent. Continuous scoring is a declared unproved affine likelihood continuation, not a Bernoulli likelihood or original published L-SML estimator.',
        'likelihood_identity': 'p(B_g|Y=y)=L_g0*((1-a_gy)+a_gy*exp(e_g)); L_g0 cancels in likelihood ratio; groups add only after taking log.',
        'member_parameters': 'sensitivity theta_i1; specificity 1-theta_i0 relative to latent alpha, not true Y',
        'member_marginal_parameters': 'Sensitivity to inferred Y: theta_i0+(theta_i1-theta_i0)*a_g1. Specificity: 1-[theta_i0+(theta_i1-theta_i0)*a_g0]. Model-implied, not gold-label measurements.',
        'group_parameters': 'sensitivity a_g1; specificity 1-a_g0 relative to inferred Y (global sign convention fixed by entropy)',
        'EM_MAP': 'Symmetric +0.5 numerator/+1 denominator corresponds Beta(1.5,1.5) MAP, not Jeffreys Beta(0.5,0.5) MAP.',
        'same_parameters_ablation': True,
        'sum_vs_mean': 'Averaging divides all group log evidence including intercept by number of unique binary members. This is a likelihood-temperature intervention, not a control isolating only information from newly added independent features.',
        'limitations': ['No proof named families meet conditional independence; correlated errors can still be double-counted.',
            'Exact clones collapse only within family; cross-family duplicates and near-duplicates are not covered.',
            'No global optimum or finite-sample consistency established by deterministic single-start EM convergence.',
            'Continuous h=mu+sd*z may lie outside [0,1]; singleton contributions are unsaturated, unlike multi-member latent groups.',
            'Groups of two rely on global latent structure for identification; no direct recovery guarantee is tested on real telemetry.',
            'Bank orientation/selection used prior source labels; labels are isolated for this fit but source bank evaluation remains development.',
            'Scoring-only null cannot certify historical feature-selection leakage absence or calibration validity; it checks label/score association disappears.'],
        'fold_models': models, 'duplicate_binary_application_check': duplicate_disagreements},
    'shuffled_label_null': {'scope': 'fixed sealed predictions, independent within-answer error-label permutations, no refit/reselection',
        'eligible_answers_checked': len(eligible), 'eligible_answers_total': 6030, 'seeds': seeds,
        'observed_within_auc': dict(zip(arms, observed.tolist())),
        'per_seed': [dict(seed=seed, **dict(zip(arms,row.tolist()))) for seed,row in zip(seeds,null)],
        'summary': {arm: {'mean': float(null[:,j].mean()), 'sd_across_seeds': float(null[:,j].std(ddof=1)),
                          'min': float(null[:,j].min()), 'max': float(null[:,j].max()),
                          'candidate_minus_arm_mean': float((null[:,0]-null[:,j]).mean()),
                          'candidate_minus_arm_min': float((null[:,0]-null[:,j]).min()),
                          'candidate_minus_arm_max': float((null[:,0]-null[:,j]).max())}
                    for j,arm in enumerate(arms)}},
    'feature_permutation': {'feature': 'q15_H1', 'seed': 20260924, 'within_each_answer': True,
        'answers_checked': len(ans), 'answers_total': 13769, 'steps': int(off[-1]),
        'refit': False, 'recalibrated': False, 'method': 'Original models and thresholds; answer standardization recomputed after scoring.', 'arms': feature},
    'synthetic': {'training_rows': 30000, 'heldout_rows': 30000, 'train_seed':40, 'test_seed':41,
        'three_independent_measurements_loss': loss(full_scores), 'one_measurement_loss': loss(small_scores),
        'same_parameters_mean_evidence_loss': loss(score(test, full, continuous=False, average_evidence=True)),
        'duplicate_measurements_loss': loss(duplicate_scores),
        'duplicate_max_score_difference': float(np.max(np.abs(full_scores-duplicate_scores))),
        'full_converged': full['converged'], 'small_converged': small['converged'], 'duplicate_converged': dup['converged']},
}
(OUT/'AUDIT_MATH_NULL.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'null_summary': result['shuffled_label_null']['summary'], 'feature_permutation': result['feature_permutation'], 'synthetic':result['synthetic']}))
