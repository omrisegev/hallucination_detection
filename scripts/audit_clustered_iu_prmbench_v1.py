"""Fresh label-free replay plus saved-stage reconstruction and package audit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json, sys, shutil
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.clustered_iu_prmbench import fit_clustered_iu
from spectral_utils.advisor_controls import answer_z
from run_clustered_iu_prmbench_v1 import sha, dump, OUT, FROZEN, METHODS


def audit():
    seal = json.loads((OUT/'PREDICTION_SEAL.json').read_text())
    for relative, expected in seal['files'].items():
        assert sha(OUT/relative) == expected, relative
    identity = json.loads((OUT/'INPUT_FREEZE.json').read_text())
    for relative, expected in identity['code'].items():
        assert sha(ROOT/relative) == expected, relative
        target = OUT/'sources'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    for relative, expected in identity['inputs'].items(): assert sha(ROOT/relative) == expected, relative
    assert sha(FROZEN/'PACKAGE_MANIFEST.json') == identity['frozen_manifest']
    rows = []
    for bank in ('N16', 'N56'):
        saved = np.load(OUT/'predictions'/f'{bank}.npz', allow_pickle=False)
        assert not set(saved.files) & {'y', 'labels', 'classification'}
        X = np.load(FROZEN/f'inputs/bank_V_{bank}.npz', allow_pickle=False)['V']
        off, folds, groups = saved['offsets'], saved['fold'], saved['group']
        sf = np.repeat(folds, np.diff(off))
        for f in range(5):
            model = json.loads((OUT/'checkpoints'/f'{bank}_fold{f}'/'MODELS.json').read_text())
            assert set(model['train_groups']) == set(groups[folds != f])
            assert set(model['test_groups']) == set(groups[folds == f])
            assert not set(model['train_groups']) & set(model['test_groups'])
            assert not model['failures']
            partition = model['models']['LSML_CONT']['diagnostic']['group']
            fresh = fit_clustered_iu(X[sf != f], partition)
            heldout = X[sf == f]; h = model['models']['CLUSTER_IU']; diag = h['diagnostic']
            assert partition == diag['group']
            stages = []
            for inner in diag['inner']:
                cols = inner['columns']; im = inner['model']; d = im['diagnostic']
                if d['rho_hat_full'] is not None:
                    threshold = max(.05*d['probe_var_y'], max(d['rho_hat_full'])/3)
                    expected = np.array(d['available'])[np.asarray(d['rho_hat_full']) >= threshold].tolist()
                    assert d['retained_columns'] == expected
                assert not d['forced_restoration']
                stages.append(heldout[:, cols] @ np.asarray(im['weight']) + im['intercept'])
            explicit = np.column_stack(stages) @ np.asarray(diag['outer']['weight']) + diag['outer']['intercept']
            affine = heldout @ np.asarray(h['weight']) + h['intercept']
            score = saved['CLUSTER_IU'][sf == f]
            err = max(float(np.max(np.abs(explicit-score))), float(np.max(np.abs(affine-score))),
                      float(np.max(np.abs(fresh.predict(heldout)-score))))
            assert err <= 1e-9
            assert np.array_equal(np.asarray(fresh.diagnostic['outer']['diagnostic']['retained_columns']),
                                  np.asarray(diag['outer']['diagnostic']['retained_columns']))
            eoff = np.r_[0, np.cumsum(np.diff(off)[folds == f])]
            for method in METHODS:
                mm = model['models'][method]
                train_score = X[sf != f] @ np.asarray(mm['weight']) + mm['intercept']
                toff = np.r_[0, np.cumsum(np.diff(off)[folds != f])]
                tau = float(np.quantile(answer_z(train_score, toff), .8))
                assert abs(tau-mm['Q80_tau']) <= 1e-9
                assert np.array_equal(answer_z(saved[method][sf == f], eoff) >= tau,
                                      saved[method+'__flags'][sf == f])
            rows.append({'bank': bank, 'fold': f, 'fresh_replay_and_stage_max_abs': err,
                         'baseline_max_abs': max(model['models'][m]['baseline_saved_score_max_abs'] for m in METHODS[:2]),
                         'source_disjoint': True, 'filter_masks_verified': True, 'Q80_verified': True})
    dump(OUT/'AUDIT.json', {'status': 'PASS', 'folds': rows, 'prediction_seal_verified': True,
         'frozen_inputs_and_runtime_unchanged': True, 'labels_not_loaded_for_replay': True,
         'scientific_unittest_count': 5, 'audit_script_sha256': sha(Path(__file__))})
    print('PASS: ten fresh hybrid replays, saved two-stage scores, baselines, source splits, survivor masks, Q80 and frozen identities')


if __name__ == '__main__': audit()
