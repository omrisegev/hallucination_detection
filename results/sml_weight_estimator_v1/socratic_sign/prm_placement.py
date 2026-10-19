"""Where the PRM enters L-SML, under three SML weight modes (Socratic, label-free scoring, diagnostic).

Same inputs as ../../prm_boost_ablation5_v1/ablation5_socratic.py (z(PRM) = answer-z of the logit risk, bank V).
Placements:
  AUTO_<B>   L-SML over [z(PRM), channels] with its own group detection: the PRM is not forced into its own group
  FLAT_<B>   one SML stage over [z(PRM), channels], every view its own group (no grouping)
  CHAIN_<B>  shipped construction: L-SML on the channels, then SML over [z(PRM), answer-z group scores]
Modes: signed (shipped sml_fuse_signed + anchor flip toward z(PRM)), abs (|v|, no flip),
       nnr1 (fusion_utils.rank1_offdiag_fit, v >= 0, no flip).
Also saves each arm's effective PRM weight share: the least-squares weights of the fused score on its input views.

    python prm_placement.py <repo> <bank dir> <out SCORES npz>
"""
import json, sys, types, importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

repo, bank, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo))
_src = repo / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
EV = types.ModuleType('ev'); EV.__file__ = str(repo / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
sys.modules['ev'] = EV
exec(compile(_src.read_text(), str(_src), 'exec'), EV.__dict__)
import spectral_utils.fusion_utils as FU  # noqa: E402

SHIPPED = FU.sml_fuse_signed


def sml_abs(*c, gates=None, small_m_guard=False):
    X = np.column_stack(c); R = np.atleast_2d(np.cov(X.T)); w = np.abs(eigh(R - np.diag(np.diag(R)))[1][:, -1])
    return X @ w, w


def sml_nnr1(*c, gates=None, small_m_guard=False):
    X = np.column_stack(c); w = FU.rank1_offdiag_fit(np.cov(X.T), nonneg=True)
    if not np.any(w > 0): w = np.ones(X.shape[1])
    return X @ w, w


MODES = {'signed': SHIPPED, 'abs': sml_abs, 'nnr1': sml_nnr1}


def lsml(V, mode, groups=None, anchor=None):
    FU.sml_fuse_signed = MODES[mode]
    try:
        s, d = FU.lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    finally:
        FU.sml_fuse_signed = SHIPPED
    s = np.asarray(s, float)
    if mode == 'signed':
        ref = V[:, 0] if anchor is None else anchor
        if np.corrcoef(s, ref)[0, 1] < 0: s = -s
    return s, d


F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
assert np.array_equal(F['uids'], R['uids'])
ne = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64)
F['cells'] = np.array([EV.CELL] * len(F['uids'])); F['row_ids'] = F['uids']
fs = importlib.util.spec_from_file_location('prm_fuse', repo / 'results/prm_boosting_v1/fuse.py')
FUSE = importlib.util.module_from_spec(fs); fs.loader.exec_module(FUSE); ext = FUSE.ext
pop = ext.Population(F, ['N13', 'N16']); ans = np.arange(len(F['uids']))
prm = EV.risk(np.clip(R['rewards'][ne], 1e-6, 1 - 1e-6)); zp = EV.az(prm, o)
arms, share = {'PRM': prm}, {}
for bk in ('N13', 'N16'):
    V = np.asarray(pop.cell_bank(bk, ans)['V'], float); VP = np.column_stack([zp, V])
    for mode in MODES:
        s, _ = lsml(VP, mode); arms[f'AUTO_{bk}_{mode}'] = s
        s, _ = lsml(VP, mode, groups=np.arange(VP.shape[1])); arms[f'FLAT_{bk}_{mode}'] = s
        st, dt = lsml(V, mode, anchor=V.mean(1))
        vg = np.asarray(dt['virtual_classifiers'], float)
        VC = np.column_stack([zp] + [EV.az(vg[:, g], o) for g in range(vg.shape[1])])
        s, _ = lsml(VC, mode, groups=np.arange(VC.shape[1])); arms[f'CHAIN_{bk}_{mode}'] = s
    for k in [k for k in arms if k.endswith(tuple(MODES)) and f'_{bk}_' in k]:
        w = np.linalg.lstsq(VP, arms[k], rcond=None)[0]
        share[k] = {'prm_weight_share': float(abs(w[0]) / np.abs(w).sum()), 'prm_weight_sign': float(np.sign(w[0])),
                    'n_negative_channel_weights': int((w[1:] < -1e-9).sum()), 'fit_r2': float(1 - np.var(arms[k] - VP @ w) / np.var(arms[k]))}
np.savez_compressed(out, uids=F['uids'], nonempty=ne, nonempty_lengths=F['nonempty_lengths'],
                    native_valid=R['native_labels'].astype(int), **{f'score__{k}': v for k, v in arms.items()})
Path(out).with_suffix('.json').write_text(json.dumps({'bank': bank.name, 'share': share, 'labels_read': False}, indent=1))
print('DONE', out)
