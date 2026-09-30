"""Post-evaluation diagnostics only; never changes predictions or model selection."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.lsml_group_confidence_experiment import load_inputs
from spectral_utils.lsml_group_confidence import binary_tail
from spectral_utils.external_generalization.artifacts import atomic_json


def main():
    out = ROOT/'results/lsml_group_confidence_v1'
    if not (out/'STATUS.json').exists():
        raise RuntimeError('main evaluation must finish first')
    data = load_inputs(ROOT)
    ans, off = data['ans'], data['off']
    b = binary_tail(data['x'], off)
    sf = np.repeat(ans.fold.to_numpy(), np.diff(off))
    labels = np.load(data['paths']['arrays'])['labels'].astype(bool)
    meta = {v['idx']: v for v in pickle.loads(data['paths']['metadata'].read_bytes()).values()}
    noncontrol = np.array([str(ans.cell[i]).startswith('prm') and meta[ans.id[i]]['classification'] != 'correct' for i in range(len(ans))])
    include = np.repeat(noncontrol, np.diff(off))
    fits = json.loads((out/'FITS.json').read_text())
    features, groups = [], []
    for fold in range(5):
        m = json.loads((out/f'MODEL_{fold}.json').read_text())
        theta, a = np.asarray(m['theta']), np.asarray(m['a'])
        gr = np.asarray(m['groups'])
        expansion = np.array(fits[fold]['expansion'])
        # Full run has no duplicate merges; assert before attributing one member.
        np.testing.assert_array_equal(expansion, np.eye(28))
        predicted = theta[:, 0, None] + (theta[:, 1]-theta[:, 0])[:, None]*a[gr]
        mask = include & (sf == fold)
        empirical_sens = b[mask & labels].mean(0)
        empirical_spec = 1-b[mask & ~labels].mean(0)
        for j, name in enumerate(data['names']):
            features.append({'fold': fold, 'feature': name, 'family': data['family_names'][gr[j]],
                             'model_sensitivity_to_latent_Y': predicted[j, 1],
                             'model_specificity_to_latent_Y': 1-predicted[j, 0],
                             'observed_sensitivity_to_true_error': empirical_sens[j],
                             'observed_specificity_to_true_error': empirical_spec[j],
                             'error_steps': int((mask & labels).sum()), 'correct_steps': int((mask & ~labels).sum())})
        sd = np.array(fits[fold]['group_contribution_sd'])
        for g, name in enumerate(data['family_names']):
            groups.append({'fold': fold, 'family': name, 'members': int((gr == g).sum()),
                           'model_group_sensitivity': a[g, 1], 'model_group_specificity': 1-a[g, 0],
                           'continuous_contribution_sd_share': sd[g]/sd.sum(),
                           'model_prevalence': m['pi'],
                           'observed_prmb_error_prevalence': float(labels[mask].mean())})
    f = pd.DataFrame(features)
    f.to_csv(out/'FEATURE_ACCURACY_DIAGNOSTIC.csv', index=False)
    pd.DataFrame(groups).to_csv(out/'GROUP_DIAGNOSTIC.csv', index=False)
    means = f.groupby('feature').mean(numeric_only=True)
    predicted_ba = (means.model_sensitivity_to_latent_Y+means.model_specificity_to_latent_Y)/2
    actual_ba = (means.observed_sensitivity_to_true_error+means.observed_specificity_to_true_error)/2
    result = {'posthoc': True, 'interpretation': 'latent-model parameters are estimates, not measured true-label sensitivities',
              'features': 28, 'fold_feature_estimates': len(f), 'answers': int(noncontrol.sum()), 'steps': int(include.sum()),
              'sensitivity_mae': float(np.abs(f.model_sensitivity_to_latent_Y-f.observed_sensitivity_to_true_error).mean()),
              'specificity_mae': float(np.abs(f.model_specificity_to_latent_Y-f.observed_specificity_to_true_error).mean()),
              'balanced_accuracy_rank_correlation_28_features': float(spearmanr(predicted_ba, actual_ba).statistic),
              'observed_prmb_error_prevalence': float(labels[include].mean()),
              'mean_latent_model_prevalence': float(np.mean([g['model_prevalence'] for g in groups]))}
    atomic_json(out/'MODEL_DIAGNOSTIC.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
