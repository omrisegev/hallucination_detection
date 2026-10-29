"""Label-free CHAIN / TELE / PRMG scoring on a ProcessBench cell, runnable on the AIRCC mirror or locally.

Logic copied unchanged from ../../prm_boost_external_v1/chain_socratic/score_chain_external.py (the Socratic
post-registration scorer, parity-checked there); only the location of the frozen modules differs:
  M = $PRM_MIRROR, else the current directory (AIRCC: cwd = eval_mirror_prm_gen_v1), else the repo root.
  evaluate_amended.py: M/results/prm_boosting_generalization_v1/evaluate_amended.py (mirror layout) or, in the repo,
  .../run_v1/evaluate_amended.py executed unchanged with the mirror __file__ (its ROOT is resolved from __file__).
Arms: PRM, TELE_<B>, PRMG_<B>, CHAIN_<B> for B in N13 (digit-free) and N16 (3 digit channels); LSML_N16_PRMG =
PRMG_N16 is kept as the parity column against score_external_v2.py. CHAIN is a post-registration arm (defined on
PRMBench 2026-10-18, after the frozen arm list). No labels are read.

ProcessBench pre-registration (../PREREGISTRATION_HE.md, sections 2, 5 and 8):
- every L-SML fit is fusion_utils.lsml_continuous_guarded (commit db054926): constant views are dropped first and get
  weight 0; without a constant view it is exactly lsml_continuous, so TELE/PRMG/CHAIN are unchanged then;
- control arms with the definitions of ../../prm_boost_ablation5_v1/ablation5.py, for B in N13 and N16:
    MEAN_<B>  mean of the bank's answer-z, label-free-oriented channels (no PRM)
    HALF_<B>  1/2 z(PRM) + 1/2 MEAN_<B>       (HALF_N16 doubles as a parity column against score_external_v2.py)
    EQ_<B>    mean of z(PRM) and every bank channel (PRM weight 1/(K+1))
    POS_ONLY  answer-z of the relative step position (k + .5) / n over the non-empty steps
    LEN_ONLY  answer-z of log(1 + step tokens) from the raw telemetry spans (needs the third argument; omitted without it)
  These are controls, not recipes.

    cd $M && python3 -B prm_boost_score_chain_pb.py <bank dir with FEATURES.npz and REWARDS.npz> <out SCORES_CHAIN_<a>.npz> \
        [<external_telemetry.pkl the bank was built from, for LEN_ONLY>]
"""
import importlib.util, json, os, pickle, sys, types
from pathlib import Path
import numpy as np


def _mirror():
    for c in (os.environ.get('PRM_MIRROR'), os.getcwd(), str(Path(__file__).resolve().parents[3])):
        if c and (Path(c) / 'results/prm_boosting_v1/fuse.py').exists():
            return Path(c)
    raise SystemExit('cannot find results/prm_boosting_v1/fuse.py; set PRM_MIRROR')


M = _mirror()
sys.path.insert(0, str(M))
_mirror_ev = M / 'results/prm_boosting_generalization_v1/evaluate_amended.py'
_src = _mirror_ev if _mirror_ev.exists() else M / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
EV = types.ModuleType('ev'); EV.__file__ = str(_mirror_ev)
sys.modules['ev'] = EV
exec(compile(_src.read_text(), str(_src), 'exec'), EV.__dict__)
# Constant views are dropped before L-SML (weight 0); without them this is exactly lsml_continuous.
from spectral_utils.fusion_utils import lsml_continuous_guarded as lsml_continuous  # noqa: E402

bank, out = Path(sys.argv[1]), Path(sys.argv[2]); raw_pkl = Path(sys.argv[3]) if len(sys.argv) > 3 else None
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


def step_log_tokens(raw_pkl, uids, nonempty, lens):
    """log(1 + tokens) of every non-empty step, from the raw (pre-mask) step_token_spans, as ablation5.py's loglen.
    The bank's non-empty steps are exactly the raw spans with hi > lo (build_bank_external drops the others)."""
    raw = pickle.load(open(raw_pkl, 'rb'))
    by_uid = {str(e['idx']): e for e in raw.values() if 'failure' not in e}
    starts = np.r_[0, np.cumsum(lens)]; out = []
    for j, u in enumerate(uids):
        sp = by_uid[str(u)]['step_token_spans']
        if len(sp) != lens[j]: raise SystemExit(f'{u}: {len(sp)} raw spans, bank has {lens[j]} steps')
        keep = np.array([hi > lo for lo, hi in sp])
        if not np.array_equal(keep, nonempty[starts[j]:starts[j + 1]]): raise SystemExit(f'{u}: non-empty steps differ from the bank')
        out.append(np.log1p([hi - lo for lo, hi in sp if hi > lo]))
    return np.concatenate(out)


nst = np.diff(o); pos = np.concatenate([(np.arange(k) + .5) / k for k in nst])
arms, diag = {'PRM': prm, 'POS_ONLY': EV.az(pos, o)}, {}
if raw_pkl is not None:
    loglen = step_log_tokens(raw_pkl, F['uids'], ne, F['nonempty_lengths'].astype(int))
    assert len(loglen) == o[-1]
    arms['LEN_ONLY'] = EV.az(loglen, o)
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
    cw = np.abs(np.asarray(dc['cross_weights'], float))
    diag[bk] = {'channels': list(b['names']), 'tele_K': int(dt['K']), 'tele_groups': cn.tolist(),
                'zero_var_channels': [str(n) for n, sd in zip(b['names'], V.std(0)) if sd < 1e-9],
                'tele_dropped_constant': list(map(int, dt.get('dropped_constant', []))),
                'CHAIN_n_views': int(VC.shape[1]), 'CHAIN_undetermined': bool(VC.shape[1] <= 3),
                'CHAIN_prm_cross_share': float(cw[0] / cw.sum()),
                'corr_with_prm': {a: float(np.corrcoef(arms[f'{a}_{bk}'], prm)[0, 1])
                                  for a in ('TELE', 'PRMG', 'CHAIN', 'MEAN', 'HALF', 'EQ')}}
arms['LSML_N16_PRMG'] = arms['PRMG_N16']
np.savez_compressed(out, uids=F['uids'], nonempty=ne, nonempty_lengths=F['nonempty_lengths'],
                    native_valid=R['native_labels'].astype(int), **{f'score__{k}': v for k, v in arms.items()})
out.with_suffix('.json').write_text(json.dumps({'bank': str(bank), 'raw': None if raw_pkl is None else str(raw_pkl),
                                                'lsml': 'lsml_continuous_guarded', 'diag': diag, 'labels_read': False}, indent=1))
print('DONE', out, {k: round(float(np.corrcoef(v, prm)[0, 1]), 3) for k, v in arms.items()})
