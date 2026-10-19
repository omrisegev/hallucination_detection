import importlib.util, json, sys
from pathlib import Path
import numpy as np

M = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1/eval_mirror_prm_gen_v1')
spec = importlib.util.spec_from_file_location('ev', M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402

bank, out = Path(sys.argv[1]), Path(sys.argv[2])
F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
assert np.array_equal(F['uids'], R['uids']) and np.array_equal(F['nonempty_lengths'], R['nonempty_lengths'])
ne = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64)
F['cells'] = np.array([EV.CELL] * len(F['uids'])); F['row_ids'] = F['uids']

fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
ext.BN = dict(ext.BN, TRANSFER=list(EV.S3))
ans = np.arange(len(F['uids']))

pop = ext.Population(F, ['TRANSFER', 'N16', 'N11', 'N13'])
bt = pop.cell_bank('TRANSFER', ans)
bn16 = pop.cell_bank('N16', ans)
bn11 = pop.cell_bank('N11', ans)
bn13 = pop.cell_bank('N13', ans)
assert np.array_equal(bt['off'], o)

C3 = np.column_stack([bt['V'][:, bt['names'].index(c)] for c in EV.S3])
N16 = np.asarray(bn16['V'], float)
N11 = np.asarray(bn11['V'], float)
N13 = np.asarray(bn13['V'], float)

prm = EV.risk(np.clip(R['rewards'][ne], 1e-6, 1 - 1e-6)); zp = EV.az(prm, o)

def lsml(V, groups=None):
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    if np.corrcoef(s, V[:, 0])[0, 1] < 0: s = -s
    return s, d

arms, diag = {'PRM': prm, 'HALF_N16': .5 * zp + .5 * N16.mean(1)}, {}

arms['LSML_CONT'], d = lsml(np.column_stack([zp, C3]))
diag['LSML_CONT'] = {'K': int(d['K']), 'groups': np.asarray(d['c']).tolist()}

banks_dict = {'N16': N16, 'N11': N11, 'N13': N13}
for bk, V in banks_dict.items():
    stele, dt = lsml_continuous(*[V[:, j] for j in range(V.shape[1])])
    stele = np.asarray(stele, float)
    if np.corrcoef(stele, V[:, 0])[0, 1] < 0: stele = -stele
    arms[f'TELE_{bk}'] = stele
    
    cn = np.asarray(dt['c'], int)
    arms[f'LSML_{bk}_PRMG'], dg = lsml(np.column_stack([zp, V]), groups=np.r_[cn.max() + 1, cn])
    arms[f'AUTO_{bk}'], da = lsml(np.column_stack([zp, V]))
    
    vg = np.asarray(dt['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [EV.az(vg[:, g], o) for g in range(vg.shape[1])])
    arms[f'CHAIN_{bk}'], dc = lsml(VC, groups=np.arange(VC.shape[1]))
    
    diag[bk] = {
        'tele_K': int(dt['K']),
        'PRMG_K': int(dg['K']),
        'AUTO_K': int(da['K']),
        'CHAIN_n_views': int(VC.shape[1]),
        'CHAIN_cross_weights': np.asarray(dc['cross_weights'], float).tolist()
    }

np.savez_compressed(out, uids=F['uids'], nonempty=ne, nonempty_lengths=F['nonempty_lengths'],
                    native_valid=R['native_labels'].astype(int), **{f'score__{k}': v for k, v in arms.items()})
out.with_suffix('.json').write_text(json.dumps({
    'bank': str(bank),
    'diag': diag,
    'labels_read': False
}, indent=1))
print('DONE', out, {k: float(np.corrcoef(v, prm)[0, 1]) for k, v in arms.items()})
