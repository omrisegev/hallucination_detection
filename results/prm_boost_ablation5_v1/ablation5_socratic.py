"""Five-arm ablation on Socratic-PRMBench (local CPU, label-free scoring). Adds MEAN, HALF and EQ arms to
../prm_boost_external_v1/chain_socratic/score_chain_external.py, whose code below is unchanged otherwise:
  TELE_<B>   (arm 1) L-SML on the bank alone          MEAN_<B>  (arm 2) simple mean of the same oriented channels
  PRM        (arm 3) native PRM (and PRM_Q80 in eval)  CHAIN_<B>/PRMG_<B> (arm 4) L-SML boosting
  HALF_<B>   (arm 5) 1/2 z(PRM) + 1/2 mean(bank)       EQ_<B>    mean of z(PRM) and every channel (PRM weight 1/(K+1))
PRMPROB / HALFP / EQP / CHAINP_<B>: same arms with the PRM channel as z(1 - reward) instead of z(logit(1 - reward)).
HALF_N16 and PRMG_N16 (= LSML_N16_PRMG) are parity checks against the frozen ../prm_boost_external_v1/scores_socratic.

    python ablation5_socratic.py <bank dir with FEATURES.npz and REWARDS.npz> <out SCORES.npz>
"""
import importlib.util, json, sys, types
from pathlib import Path
import numpy as np

M = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(M))
# The AIRCC mirror keeps evaluate_amended.py one level up (results/prm_boosting_generalization_v1/), and the
# module resolves its ROOT from __file__. Execute the repo copy unchanged with the mirror's __file__.
_src = M / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
EV = types.ModuleType('ev'); EV.__file__ = str(M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
sys.modules['ev'] = EV
exec(compile(_src.read_text(), str(_src), 'exec'), EV.__dict__)
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402

bank, out = Path(sys.argv[1]), Path(sys.argv[2])
F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
assert np.array_equal(F['uids'], R['uids']) and np.array_equal(F['nonempty_lengths'], R['nonempty_lengths'])
ne = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64)
F['cells'] = np.array([EV.CELL] * len(F['uids'])); F['row_ids'] = F['uids']
fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
BANKS = ('N13', 'N16')
ans = np.arange(len(F['uids']))
pop = ext.Population(F, list(BANKS))
prm = EV.risk(np.clip(R['rewards'][ne], 1e-6, 1 - 1e-6)); zp = EV.az(prm, o)
prob = 1 - np.asarray(R['rewards'][ne], float); zq = EV.az(prob, o)  # probability-scale PRM channel (P-variants)


def lsml(V, groups=None, anchor=None):
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else anchor
    if np.corrcoef(s, ref)[0, 1] < 0: s = -s
    return s, d


arms, diag = {'PRM': prm, 'PRMPROB': prob}, {}
for bk in BANKS:
    b = pop.cell_bank(bk, ans); assert np.array_equal(b['off'], o)
    V = np.asarray(b['V'], float)
    arms[f'TELE_{bk}'], dt = lsml(V, anchor=V.mean(1))
    cn = np.asarray(dt['c'], int)
    arms[f'PRMG_{bk}'], dg = lsml(np.column_stack([zp, V]), groups=np.r_[cn.max() + 1, cn])
    vg = np.asarray(dt['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [EV.az(vg[:, g], o) for g in range(vg.shape[1])])
    arms[f'CHAIN_{bk}'], dc = lsml(VC, groups=np.arange(VC.shape[1]))
    arms[f'MEAN_{bk}'] = V.mean(1)
    arms[f'HALF_{bk}'] = .5 * zp + .5 * V.mean(1)
    arms[f'EQ_{bk}'] = np.column_stack([zp, V]).mean(1)
    arms[f'HALFP_{bk}'] = .5 * zq + .5 * V.mean(1)
    arms[f'EQP_{bk}'] = np.column_stack([zq, V]).mean(1)
    VCq = np.column_stack([zq] + list(VC[:, 1:].T))
    arms[f'CHAINP_{bk}'], _ = lsml(VCq, groups=np.arange(VCq.shape[1]))
    cw = np.abs(np.asarray(dc['cross_weights'], float))
    diag[bk] = {'channels': list(b['names']), 'tele_K': int(dt['K']), 'tele_groups': cn.tolist(),
                'zero_var_channels': [str(n) for n, sd in zip(b['names'], V.std(0)) if sd < 1e-9],
                'CHAIN_n_views': int(VC.shape[1]), 'CHAIN_undetermined': bool(VC.shape[1] <= 3),
                'CHAIN_prm_cross_share': float(cw[0] / cw.sum()),
                'corr_with_prm': {a: float(np.corrcoef(arms[f'{a}_{bk}'], prm)[0, 1]) for a in ('TELE', 'MEAN', 'PRMG', 'CHAIN', 'HALF', 'EQ')}}
np.savez_compressed(out, uids=F['uids'], nonempty=ne, nonempty_lengths=F['nonempty_lengths'],
                    native_valid=R['native_labels'].astype(int), **{f'score__{k}': v for k, v in arms.items()})
out.with_suffix('.json').write_text(json.dumps({'bank': str(bank), 'diag': diag, 'labels_read': False}, indent=1))
print('DONE', out, {k: round(float(np.corrcoef(v, prm)[0, 1]), 3) for k, v in arms.items()})
