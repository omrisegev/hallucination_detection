"""Label-free CHAIN scoring on an external cell (local CPU), added after the frozen Socratic run.

Same inputs, orientation and L-SML entry point as ../score_external.py. Adds the bank-sweep arms of
results/prm_boosting_bank_sweep_v1/lsml_bank_sweep.py for N13 (digit-free) and N16 (3 digit channels):
  TELE_<B>   L-SML on the bank alone (sign anchored to the bank mean)
  PRMG_<B>   L-SML on z(PRM) + bank, PRM forced into its own group  (= LSML_N16_PRMG for B = N16)
  CHAIN_<B>  z(PRM) + answer-z of each TELE_<B> group score, one SML across the K+1 views
LSML_N16_PRMG is recomputed as a parity check against the frozen scores. No labels are read.

CHAIN was defined on PRMBench (2026-10-18 18:59 IDT) before the Socratic outcome was seen (21:23 IDT),
but it is NOT in the frozen Socratic arm list. Report it as a post-registration arm.

    .venv/bin/python score_chain_external.py <bank dir with FEATURES.npz and REWARDS.npz> <out SCORES.npz>
"""
import importlib.util, json, sys, types
from pathlib import Path
import numpy as np

M = Path(__file__).resolve().parents[3]
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


def lsml(V, groups=None, anchor=None):
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else anchor
    if np.corrcoef(s, ref)[0, 1] < 0: s = -s
    return s, d


arms, diag = {'PRM': prm}, {}
for bk in BANKS:
    b = pop.cell_bank(bk, ans); assert np.array_equal(b['off'], o)
    V = np.asarray(b['V'], float)
    arms[f'TELE_{bk}'], dt = lsml(V, anchor=V.mean(1))
    cn = np.asarray(dt['c'], int)
    arms[f'PRMG_{bk}'], dg = lsml(np.column_stack([zp, V]), groups=np.r_[cn.max() + 1, cn])
    vg = np.asarray(dt['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [EV.az(vg[:, g], o) for g in range(vg.shape[1])])
    arms[f'CHAIN_{bk}'], dc = lsml(VC, groups=np.arange(VC.shape[1]))
    cw = np.abs(np.asarray(dc['cross_weights'], float))
    diag[bk] = {'channels': list(b['names']), 'tele_K': int(dt['K']), 'tele_groups': cn.tolist(),
                'CHAIN_n_views': int(VC.shape[1]), 'CHAIN_undetermined': bool(VC.shape[1] <= 3),
                'CHAIN_prm_cross_share': float(cw[0] / cw.sum()),
                'corr_with_prm': {a: float(np.corrcoef(arms[f'{a}_{bk}'], prm)[0, 1]) for a in ('TELE', 'PRMG', 'CHAIN')}}
arms['LSML_N16_PRMG'] = arms['PRMG_N16']
np.savez_compressed(out, uids=F['uids'], nonempty=ne, nonempty_lengths=F['nonempty_lengths'],
                    native_valid=R['native_labels'].astype(int), **{f'score__{k}': v for k, v in arms.items()})
out.with_suffix('.json').write_text(json.dumps({'bank': str(bank), 'diag': diag, 'labels_read': False}, indent=1))
print('DONE', out, {k: round(float(np.corrcoef(v, prm)[0, 1]), 3) for k, v in arms.items()})
