"""Label-free scoring of the frozen PRM-boosting recipes on an external cell (AIRCC CPU, eval_mirror_prm_gen_v1).

v2 of score_external.py (frozen; Socratic SCORES), used for ProcessBench (prm_boost_processbench_v1/PREREGISTRATION_HE.md,
section 8). The only change: every L-SML fit is fusion_utils.lsml_continuous_guarded (commit db054926), which drops
constant views before L-SML and gives them weight 0. Without a constant view it is exactly lsml_continuous, so on a
bank with no constant channel this file writes the same SCORES as score_external.py.

Same views, orientation and fusion as results/prm_boosting_controls_v1/lsml_controls.py on PRMBench:
  PRM          risk = logit(1 - reward), answer-z
  HALF_N16     1/2 z(PRM) + 1/2 mean of the 16 N16 channels      (clean recipe, no channel chosen with labels)
  LSML_N16_PRMG  continuous L-SML on PRM + N16, PRM forced into its own group
  LSML_CONT    continuous L-SML on PRM + the 3 PRMBench-chosen channels (frozen recipe of 2026-10-14)
Each fusion is fitted on this cell's own unlabeled steps (per model, per dataset). No labels are read.
Writes SCORES.npz: uids, nonempty, per-arm scores on the non-empty rows, PRM native flags on all steps.

    python3 score_external_v2.py <bank dir with FEATURES.npz and REWARDS.npz> <out SCORES.npz>
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

M = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1/eval_mirror_prm_gen_v1')
spec = importlib.util.spec_from_file_location('ev', M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
# Constant views are dropped before L-SML (weight 0); without them this is exactly lsml_continuous.
from spectral_utils.fusion_utils import lsml_continuous_guarded as lsml_continuous  # noqa: E402

bank, out = Path(sys.argv[1]), Path(sys.argv[2])
F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
assert np.array_equal(F['uids'], R['uids']) and np.array_equal(F['nonempty_lengths'], R['nonempty_lengths'])
ne = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64)
F['cells'] = np.array([EV.CELL] * len(F['uids'])); F['row_ids'] = F['uids']   # Population only reads cells/row_ids keys
fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
ext.BN = dict(ext.BN, TRANSFER=list(EV.S3))
ans = np.arange(len(F['uids']))
pop = ext.Population(F, ['TRANSFER', 'N16'])
bt, bn = pop.cell_bank('TRANSFER', ans), pop.cell_bank('N16', ans)
assert np.array_equal(bt['off'], o)
C3 = np.column_stack([bt['V'][:, bt['names'].index(c)] for c in EV.S3]); N16 = np.asarray(bn['V'], float)
prm = EV.risk(np.clip(R['rewards'][ne], 1e-6, 1 - 1e-6)); zp = EV.az(prm, o)


def lsml(V, groups=None):
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    if np.corrcoef(s, V[:, 0])[0, 1] < 0: s = -s
    return s, d


arms, diag = {'PRM': prm, 'HALF_N16': .5 * zp + .5 * N16.mean(1)}, {}
_, dn = lsml_continuous(*[N16[:, j] for j in range(N16.shape[1])]); cn = np.asarray(dn['c'], int)
arms['LSML_N16_PRMG'], d = lsml(np.column_stack([zp, N16]), groups=np.r_[cn.max() + 1, cn])
diag['LSML_N16_PRMG'] = {'K': int(d['K']), 'groups': np.r_[cn.max() + 1, cn].tolist()}
arms['LSML_CONT'], d = lsml(np.column_stack([zp, C3]))
diag['LSML_CONT'] = {'K': int(d['K']), 'groups': np.asarray(d['c']).tolist()}
np.savez_compressed(out, uids=F['uids'], nonempty=ne, nonempty_lengths=F['nonempty_lengths'],
                    native_valid=R['native_labels'].astype(int), **{f'score__{k}': v for k, v in arms.items()})
out.with_suffix('.json').write_text(json.dumps({'bank': str(bank), 'diag': diag, 'lsml': 'lsml_continuous_guarded',
                                                'lf_sign_flips': bn['diagnostics']['sign_flips'], 'labels_read': False}, indent=1))
print('DONE', out, {k: float(np.corrcoef(v, prm)[0, 1]) for k, v in arms.items()})
