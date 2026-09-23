"""Read-only audit of frozen token-axis branch; writes only a review evidence JSON."""
from pathlib import Path
import csv
import importlib.util
import json
import sys
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / '.worktrees/review-token-axis-20260921'
OUT = ROOT / 'scratch/token_axis_review_20260921'
OUT.mkdir(parents=True, exist_ok=True)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

cv = module('review_cv', REVIEW / 'scripts/experiments/cumulative_vote_fusion_v1.py')
raw = module('review_raw', REVIEW / 'scripts/experiments/raw_channel_readout_fusion_v1.py')
cv.LANE = ROOT / 'results/fair_paper_exact_comparisons_v1/lanes/localization'
qids, fam, folds, label, S = cv.load_lane()
out = {'branch': '2b321fa3a', 'n_rows': len(S), 'n_error': int((label >= 0).sum())}

# Recompute saved metrics independently from committed predictions.
pred_path = REVIEW / 'results/cumulative_vote_fusion_v1/PREDICTIONS_pooled.csv'
saved = list(csv.DictReader(pred_path.open(encoding='utf8')))
out['saved_metrics'] = {}
for name in ('single:family6', 'single:mind_gap', 'lsml_mode', 'ds_mode'):
    out['saved_metrics'][name] = {}
    for family in cv.SUBSETS + ['all']:
        rr = [r for r in saved if int(r['label']) >= 0 and (family == 'all' or r['family'] == family)]
        out['saved_metrics'][name][family] = float(np.mean([int(r[name]) == int(r['label']) for r in rr]))
    out['saved_metrics'][name]['macro'] = float(np.mean([out['saved_metrics'][name][f] for f in cv.SUBSETS]))

# Refit the five frozen binary models, check fit/readout algebra and actual OOF effect.
out['binary_folds'] = []
for f in sorted(set(folds)):
    tr = (folds != f) & (label >= 0)
    X, owner = cv.disagreement_instances(S[tr])
    y_threshold = np.concatenate([np.arange(row.min(),row.max()) >= lab for row,lab in zip(S[tr],label[tr]) if row.max()>row.min()])
    empirical_psi = np.mean(X[y_threshold]>0,axis=0)
    empirical_eta = np.mean(X[~y_threshold]<0,axis=0)
    score, meta = cv.fusion_utils.lsml_fuse(*X.T)
    flat, _ = cv.fit_lsml(X)
    clipped = np.maximum(flat, 0)
    residual = np.max(np.abs(score - X @ flat))
    exact_modes = []
    flat_modes = []
    for row in S[folds == f]:
        grid = np.arange(row.min(), row.max()+1)
        V = cv.cumulative_votes(row, grid).T
        virtual = []
        for idx, w in meta['group_weights']:
            val = V[:, idx] @ w
            if len(idx) > 1:
                val = np.sign(val)
                val[val == 0] = 1
            virtual.append(val)
        native = np.column_stack(virtual) @ meta['cross_weights']
        # Same CDF convention; isolate replacing flattening by actual binary model.
        denom = np.sum(meta['cross_weights'])
        F = (native / denom + 1)/2
        exact_modes.append(cv.readout_from_cdf(grid, F)[0])
        flat_modes.append(cv.predict_weighted(row[None, :], flat)[0][0])
    te_labels = label[folds == f]
    exact_modes, flat_modes = np.array(exact_modes), np.array(flat_modes)
    out['binary_folds'].append({
        'fold': int(f), 'n_instances': len(X), 'groups': meta['c'].tolist(),
        'cross': meta['cross_weights'].tolist(), 'flat': flat.tolist(),
        'label_using_diagnostic_psi': empirical_psi.tolist(),
        'label_using_diagnostic_eta': empirical_eta.tolist(),
        'max_native_minus_flat': float(residual),
        'different_oof_modes': int(np.sum(exact_modes != flat_modes)),
        'native_hits': int(np.sum((exact_modes == te_labels) & (te_labels >= 0))),
        'flat_hits': int(np.sum((flat_modes == te_labels) & (te_labels >= 0))),
    })

# The two purportedly same sampling matrices disagree even at tau=0.
positions = np.array([[2, 3, 4, 3, 2], [1, 2, 3, 2, 1]])
binary, _ = raw.binary_instances(positions)
cdfs = [(row[:, None] <= np.arange(7)[None, :]).astype(float) for row in positions]
soft = raw.soft_instances(cdfs)
out['tau0_instances'] = {'binary_shape': list(binary.shape), 'soft_shape': list(soft.shape),
                        'binary_sml': cv.fit_sml(binary).tolist(), 'soft_sml': raw.fit_soft(soft)[0].tolist()}

# A fixed additive position penalty reverses a unique maximum, not just ties.
profile = np.array([0., 1., 1.0001])
out['tau0_unique_max'] = {'profile': profile.tolist(), 'hard': raw.argmax_step(profile),
                         'soft': raw.predict_soft(raw.soft_cdf(profile, 1e-6)[None, :], [1])[0]}

# Negative onset profiles have no threshold crossing, but argmax(False...) returns 0.
z = np.array([-3., -2., -1.])
profiles = raw.profiles_for_channel(z, [[0,1],[1,2],[2,3]])
out['negative_onset'] = {'top5': profiles['top5'].tolist(), 'onset80': profiles['onset80'].tolist(),
                         'predicted': raw.argmax_step(profiles['onset80'])}

# Historical depth proxy is smaller than the true error position on some rows.
err = label >= 0
out['depth_proxy'] = {'true_error_beyond_max_prediction': int(np.sum(label[err] > S[err].max(1))),
                      'n_error': int(err.sum())}
out['support_oracle'] = {'any_member_exact': float(np.mean(np.any(S[err] == label[err,None], axis=1))),
    'all_members_late': int(np.sum(np.all(S[err] > label[err,None],axis=1))),
    'all_members_early': int(np.sum(np.all(S[err] < label[err,None],axis=1)))}

# Join the already audited normalized source-question hashes, without reading caches.
meta = json.loads((ROOT / 'results/localization_source_group_audit_v1/PB_QUESTION_METADATA.json').read_text())
hash_by_id = {r['row_id']: r['question_whitespace_sha256'] for r in meta['rows']}
groups = defaultdict(list)
for i,q in enumerate(qids):
    sid = q.split('::',1)[1]
    if sid in hash_by_id:
        groups[hash_by_id[sid]].append(i)
cross_old = [ii for ii in groups.values() if len(set(folds[ii])) > 1]
cross_new = [ii for ii in groups.values() if len(set(raw.stable_fold(qids[i].split('::')[-1], 5) for i in ii)) > 1]
out['source_group_audit'] = {'matched_answers': sum(map(len,groups.values())), 'unique_sources':len(groups),
    'old_folds_cross_source_groups':len(cross_old), 'old_folds_affected_answers':sum(map(len,cross_old)),
    'new_id_hash_cross_source_groups':len(cross_new),'new_id_hash_affected_answers':sum(map(len,cross_new)),
    'example_old': [[qids[i] for i in ii] for ii in cross_old[:2]]}

# Ordinary argmax on raw CDFs need not equal the standardized linear model used at fit.
F = np.array([[.6,.6,1.],[0.,.9,1.]])
w = np.array([1.,1.])
sigma = np.array([.1,1.])
out['soft_scale_counterexample'] = {'F':F.tolist(),'w':w.tolist(),'sigma_train':sigma.tolist(),
    'current_raw_cdf_mode':raw.predict_soft(F,w)[0],
    'standardizer_preserving_mode':raw.predict_soft(F,w/sigma)[0]}

# Disagreement-only conditioning induces dependence even for ideal independent classifiers.
rng = np.random.default_rng(1729)
y = rng.choice([-1,1], 100000)
votes = y[:,None] * np.where(rng.random((len(y),5)) < .8, 1, -1)
dis = votes.min(1) != votes.max(1)
out['disagreement_conditioning_fixture'] = {
    'conditional_corr_full':float(np.corrcoef(votes[y==1].T)[0,1]),
    'conditional_corr_disagreement_only':float(np.corrcoef(votes[(y==1)&dis].T)[0,1]),
    'rows_retained':int(dis.sum())}

# Frozen pilot exposes non-identical binary/soft matrices in real data.
out['pilot_fit_checks'] = {}
for p in (REVIEW / 'results/raw_channel_readout_fusion_v1').glob('pilot_*/REPORT.json'):
    d = json.loads(p.read_text())
    out['pilot_fit_checks'][p.parent.name] = {
        'min_tau0_identity': min(f['tau0_identity_match'] for f in d['fits']),
        'n_instances': [[f['n_bin_instances'],f['n_soft_instances']] for f in d['fits']],
        'negative_soft_weights': [sum(w < 0 for w in f['soft_lsml_w']) for f in d['fits']],
    }

(OUT / 'EVIDENCE.json').write_text(json.dumps(out, indent=2), encoding='utf8')
print(json.dumps(out, indent=2))
