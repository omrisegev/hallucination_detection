"""Label-free descriptive diagnostics; never changes registered predictions."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.family_tail_transfer import load_lock, build_representations, answer_standardize
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash

OUT = ROOT/'results/family_tail_external_v1'


def main():
    seals = json.loads((OUT/'ALL_CELLS_SEALED.json').read_text())
    lock = load_lock()
    families = list(lock['recipe']['families_15'])
    weights = np.array([lock['deployment']['F15_tailtie_lsml']['weights'][f] for f in families])
    cusum = np.array([f in ('energy_cusum', 'entropy_cusum') for f in families])
    report = {'scope': 'Descriptive full-population diagnostics, no labels or refitting',
              'method': 'F15_tailtie_lsml', 'groups': lock['deployment']['F15_tailtie_lsml']['groups'],
              'absolute_weight_mass_cusum': float(abs(weights[cusum]).sum()),
              'absolute_weight_mass_other': float(abs(weights[~cusum]).sum()),
              'warning': 'Weight mass and score contributions are not identified relative group reliability or causal effects.',
              'seal_sha256': file_hash(OUT/'ALL_CELLS_SEALED.json'), 'cells': {}}
    for cell, seal in seals['seals'].items():
        paths = sorted((OUT/cell).glob('shard_*/*.record.json'))
        if len(paths) != seal['answers']:
            raise ValueError('Incomplete sealed population')
        constants = np.zeros(48, int)
        family_constants = np.zeros(15, int)
        counts = np.zeros(4, int)
        contribution_sum = np.zeros((4, 2))
        contribution_abs_sum = np.zeros((4, 2))
        position_score_sum = np.zeros(4)
        max_replay = 0.
        for path in paths:
            row = json.loads(path.read_text())['payload']
            x = np.asarray(row['features'])
            names = row['feature_names']
            constants += x.std(axis=0) <= 1e-12
            v, _ = build_representations(x, names, [0, len(x)])['F15']
            family_constants += v.std(axis=0) <= 1e-12
            contributions = np.column_stack((v[:, cusum]@weights[cusum], v[:, ~cusum]@weights[~cusum]))
            score = answer_standardize(contributions.sum(axis=1), [0, len(x)], final=True)
            valid = np.asarray(row['nonempty'], bool)
            stored = np.array([z for z in row['scores']['F15_tailtie_lsml'] if z is not None])
            max_replay = max(max_replay, float(abs(score-stored).max()))
            q = np.minimum(3, np.flatnonzero(valid)*4//len(valid))
            for j in range(4):
                selected = q == j
                counts[j] += selected.sum()
                contribution_sum[j] += contributions[selected].sum(axis=0)
                contribution_abs_sum[j] += abs(contributions[selected]).sum(axis=0)
                position_score_sum[j] += score[selected].sum()
        if max_replay > 1e-12:
            raise ValueError('Contribution decomposition does not reconstruct saved score')
        report['cells'][cell] = {
            'n_checked': len(paths), 'n_total': seal['answers'],
            'constant_channel_answers': dict(zip(names, constants.tolist())),
            'constant_family_answers': dict(zip(families, family_constants.tolist())),
            'relative_position_step_counts': counts.tolist(),
            'mean_signed_contribution_before_final_z': (contribution_sum/counts[:, None]).tolist(),
            'mean_absolute_contribution_before_final_z': (contribution_abs_sum/counts[:, None]).tolist(),
            'mean_final_score_by_position': (position_score_sum/counts).tolist(),
            'contribution_columns': ['CUSUM families', 'Other families'],
            'score_replay_max_abs_error': max_replay}
    atomic_json(OUT/'REPRESENTATION_DIAGNOSTICS.json', report)
    print('Descriptive representation diagnostics complete:', sum(c['n_checked'] for c in report['cells'].values()))


if __name__ == '__main__':
    main()
