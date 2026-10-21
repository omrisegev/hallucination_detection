"""PRMBench-only copy of results/sml_weight_estimator_v1/processbench_banks/score_banks.py (collection plan section 3,
2026-10-21): same arms and fits, run on the thinkfix_mask1 source bank, PRMBench cell only (ProcessBench is out of focus).
Changes vs the original: cells restricted to prmbench_qwen3_8b; OOF_ANSWERS path is an argument; rank1_offdiag_fit is
taken from ./rank1_fit.py (verbatim) when the mirror's fusion_utils lacks it; no ProcessBench metric.

    python score_banks_prmbench.py <repo> <FEATURES.npz> <PRMBench raw telemetry.pkl> <out dir> <OOF_ANSWERS.csv>

Original docstring:
Telemetry-only L-SML on the nested banks, current SML weights vs the off-diagonal rank-one fit (Step 467).

Mirrors results/monday_cont_baselines_v1/run.py: per cell (8 ProcessBench cells + PRMBench), V = bank['V'] from
the frozen adapter, lsml_continuous with defaults, a fit with K < 3 groups is a failure. Arms:
  LSML_CONT         shipped (native sign: sml_fuse_signed majority rule)          -> parity vs the frozen ABSOLUTE.csv
  LSML_CONT_ORI     same, channels first oriented by sign of corr with the LEVEL mean (the ALL_EQ_ORIENTED rule)
  LSML_ABS_ORI      oriented channels, SML weights |v| (no anchor flip)
  LSML_RANK1_ORI    oriented channels, SML weights = rank1_offdiag_fit(nonneg=True) (no anchor flip)
  ALL_EQ_ORIENTED   plain mean of the oriented channels
Metrics (point estimates): ProcessBench argmax first-error hit, cell-equal mean over the 8 cells (as dev_run.py);
PRMBench within-answer AUC. Labels are read only after all scores exist.

    python score_banks.py <repo> <FEATURES.npz> <PRMBench raw telemetry.pkl> <out dir>
"""
import importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import rankdata

repo, feat, prm_raw, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
OOF = Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo))
import spectral_utils.fusion_utils as FU  # noqa: E402
if not hasattr(FU, 'rank1_offdiag_fit'):
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from rank1_fit import rank1_offdiag_fit as _r1  # noqa: E402
    FU.rank1_offdiag_fit = _r1

fs = importlib.util.spec_from_file_location('prm_fuse', repo / 'results/prm_boosting_v1/fuse.py')
FUSE = importlib.util.module_from_spec(fs); fs.loader.exec_module(FUSE); ext = FUSE.ext
BANKS = ['N11', 'N14', 'N13', 'N16', 'N22', 'N25', 'N53', 'N56', 'N40', 'N43']
SHIPPED = FU.sml_fuse_signed


def sml_abs(*c, gates=None, small_m_guard=False):
    assert gates is None and not small_m_guard
    X = np.column_stack(c); R = np.atleast_2d(np.cov(X.T)); Ro = R - np.diag(np.diag(R))
    w = np.abs(eigh(Ro)[1][:, -1]); return X @ w, w


def sml_rank1(*c, gates=None, small_m_guard=False):
    assert gates is None and not small_m_guard
    X = np.column_stack(c); w = FU.rank1_offdiag_fit(np.cov(X.T), nonneg=True)
    if not np.any(w > 0): w = np.ones(X.shape[1])
    return X @ w, w


def lsml(V, sml):
    FU.sml_fuse_signed = sml
    try:
        s, meta = FU.lsml_continuous(*[V[:, j] for j in range(V.shape[1])])
    finally:
        FU.sml_fuse_signed = SHIPPED
    if int(meta['K']) < 3: raise RuntimeError(f"K={meta['K']} < 3")
    return np.asarray(s, float), {'K': int(meta['K']), 'cross_weights': [float(x) for x in np.asarray(meta['cross_weights'])]}


A = pd.read_csv(OOF, encoding='utf-8-sig')
F = dict(np.load(feat))
assert list(map(str, F['uids'])) == A.uid.astype(str).tolist(), 'answer order differs from OOF_ANSWERS'
cells = A.cell.to_numpy(); N = len(A); pb = np.array([c.startswith('pb_') for c in cells])
pop = ext.Population(F, BANKS); OFF = F['offsets'].astype(np.int64)
ARMS = ['LSML_CONT', 'LSML_CONT_ORI', 'LSML_ABS_ORI', 'LSML_RANK1_ORI', 'ALL_EQ_ORIENTED']
ARGMAX = {(b, a): np.full(N, -9, int) for b in BANKS for a in ARMS}
SC = {}; diag = {}; t0 = time.time()
for bk in BANKS:
    for ce in ['prmbench_qwen3_8b']:
        ans = np.flatnonzero(cells == ce); bank = pop.cell_bank(bk, ans); V = np.asarray(bank['V'], float); off = bank['off']
        lev = pop.level[bank['rows']]
        sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
        Vo = V * sg
        scores = {'ALL_EQ_ORIENTED': Vo.mean(1)}
        for arm, VV, sml in (('LSML_CONT', V, SHIPPED), ('LSML_CONT_ORI', Vo, SHIPPED),
                             ('LSML_ABS_ORI', Vo, sml_abs), ('LSML_RANK1_ORI', Vo, sml_rank1)):
            try:
                s, d = lsml(VV, sml); scores[arm] = s; diag[f'{bk}/{ce}/{arm}'] = d
            except Exception as exc:
                diag[f'{bk}/{ce}/{arm}'] = {'FAILED': f'{type(exc).__name__}: {exc}'[:200]}
        diag[f'{bk}/{ce}/n_flipped_channels'] = int((sg < 0).sum())
        for arm, s in scores.items():
            SC[(bk, arm, ce)] = s
            for j, i in enumerate(ans):
                ss = s[off[j]:off[j + 1]]
                ARGMAX[(bk, arm)][i] = int(np.flatnonzero(ss >= ss.max() - 8 * np.finfo(float).eps)[0])
        print(bk, ce, round(time.time() - t0), flush=True)
(out / 'FIT_DIAGNOSTICS.json').write_text(json.dumps(diag, indent=1))

# ---- labels: read only now, after every score is fixed ------------------------------------------------------
try:
    from spectral_utils.prm_label_contract import prm_error_flags  # noqa: E402
except ImportError:                                  # the AIRCC mirror lacks it: verbatim copy shipped next to this script
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from prm_label_contract import prm_error_flags  # noqa: E402
target = A.target.to_numpy().astype(int)
raw = pickle.load(open(prm_raw, 'rb')); rows = list(raw.values()) if isinstance(raw, dict) else raw
err_steps = {str(r['idx']): r.get('error_steps') for r in rows if isinstance(r, dict) and 'idx' in r}
ids = A.id.astype(str).to_numpy()


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


res = []
for bk in BANKS:
    for arm in ARMS:
        if 'FAILED' in diag.get(f'{bk}/prmbench_qwen3_8b/{arm}', {}) and arm != 'ALL_EQ_ORIENTED':
            res.append({'bank': bk, 'arm': arm, 'prmb_within_auc': np.nan, 'status': 'FAILED'}); continue
        s = SC[(bk, arm, 'prmbench_qwen3_8b')]; ans = np.flatnonzero(cells == 'prmbench_qwen3_8b')
        bank = pop.cell_bank(bk, ans); off = bank['off']; aucs = []
        for j, i in enumerate(ans):
            n = off[j + 1] - off[j]; es = err_steps.get(ids[i])
            if es is None: raise SystemExit('missing error_steps for ' + ids[i])
            y = np.asarray(prm_error_flags(es, n), bool); aucs.append(auc(y, s[off[j]:off[j + 1]]))
        res.append({'bank': bk, 'arm': arm, 'prmb_within_auc': float(np.nanmean(aucs)), 'status': 'COMPLETE'})
pd.DataFrame(res).to_csv(out / 'RESULTS.csv', index=False)
print('DONE', round(time.time() - t0), flush=True)
