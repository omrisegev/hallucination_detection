"""E1b votes: one GLOBAL FUSE threshold per voter instead of within-answer GAP votes (PROTOCOL.json).

Entry point for other experiments (e.g. ds_gate_e2_posrel):

    from votes_e1b import fuse_votes
    out = fuse_votes(bench, prm, arm)          # arm in AUTO, AUTO_NATIVE, TELE16, SUB5
    out['V']        # (steps x voters) int {0,1}, PRM first in AUTO / AUTO_NATIVE; rows follow the cell loaded by
                    # lr_transfer.L.load(bench, prm, 'prm' if arm starts with AUTO else 'sa')
    out['names'], out['tau_quantile'], out['tau_value'], out['shat_init'], out['shat_final'], out['c'] (the cell)

Per cell: raw step features (NOT answer-z) of the arm's telemetry, checked against the stored answer-z channel file;
each voter pooled-standardized, residualized on a cubic in relative position (pooled least squares, answers with >= 2
steps; Step 481), then vote = 1 iff value > tau_j, tau_j a pooled quantile of that voter chosen by coordinate descent
minimizing the FUSE statistic S-hat (Lee et al., arXiv 2604.18547, Prop 2.4 / Eq. 4). Orientation: the stored one
(higher = riskier, as the GAP votes of the corrected E1). No labels anywhere in this file.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ds_gate_v1b_fixed')); sys.path.insert(0, str(HERE.parent / 'bracha_lr_transfer_v1'))
import ds_gate_b as D  # noqa: E402  (read-only reuse of the corrected E1)
import lr_transfer as L  # noqa: E402
E = L.E

TAU = Path(r'C:\Users\omris\TAU\hd_collection_data')
N16N = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
        'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
        'digit_spread', 'digit_alternative_innovation']
SUB5 = ['q15_H1', 'chosen_surprisal', 'top15_turnover', 'dominant_freq16', 'bocpd_p0', 'digit_spread']
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
SHORT = {'qwen_prm_7b': 'qwen_prm', 'qwen_prm_72b': 'qwen_prm_72b', 'pure_prm': 'pure_prm', 'rlhflow_prm': 'rlhflow_prm'}
GRID = np.round(np.arange(.50, .981, .02), 2)
EPS_DEN = 1e-2                       # clip of the Eq. 4 denominators (the paper clips to "a small positive number")
CACHE = HERE / 'cache'


def raw_path(mode, bench, prm):
    """Raw FEATURES.npz of one cell; None when it is not available on this machine."""
    p = SHORT[prm]
    if mode == 'prm' and bench == 'processbench':
        return TAU / 'sub5_all_v1/pb_prm' / f'bank_{p}' / 'FEATURES.npz'
    if mode == 'prm' and bench == 'prmbench':
        return None                  # PRM-backbone PRMBench raw banks exist only on AIRCC (unreachable)
    if mode == 'sa' and bench == 'prmbench':
        return TAU / 'prm_standalone_banks' / f'bank_{prm}' / 'FEATURES.npz'
    return TAU / 'e1b_raw' / f'full_{mode}_{p}_{bench}' / 'FEATURES.npz'


def raw_matrix(mode, bench, prm, c):
    """Raw N16 values aligned with the cell c; act = defined entries (digit channels: digit-active steps only).
    Asserts that answer-standardizing them reproduces the stored channels (E.channels)."""
    p = raw_path(mode, bench, prm)
    if p is None or not p.exists():
        raise FileNotFoundError(f'raw features not available: {mode} {bench} {prm} ({p})')
    F = np.load(p); nm = [str(x) for x in F['names']]; o = F['offsets'].astype(np.int64)
    keys = F['row_ids'].astype(str) if 'row_ids' in F.files and bench == 'prmbench' else F['uids'].astype(str)
    pos = {u: i for i, u in enumerate(keys)}
    rows = np.concatenate([np.arange(o[pos[u]], o[pos[u] + 1]) for u in c.ids]); assert len(rows) == c.off[-1]
    X = F['values'][rows][:, [nm.index(n) for n in N16N]].astype(float)
    act = np.ones_like(X, bool); act[:, [N16N.index(n) for n in DIG]] = F['digit_active'][rows].astype(bool)
    stored = E.channels(c, 'N16')
    Z = np.zeros_like(X)
    for a, b in zip(c.off[:-1], c.off[1:]):
        for j in range(X.shape[1]):
            m = act[a:b, j]; y = X[a:b, j][m]
            if len(y) and y.std() > 1e-12: Z[a:b, j][m] = (y - y.mean()) / y.std()
    err = float(np.max(np.abs(Z - stored)))
    assert err < 1e-4, f'parity {mode} {bench} {prm}: max |answer-z(raw) - stored| = {err:.3g}'
    return X, act


def clean(X, act, off, pool):
    """Pooled z-score per voter, then remove a cubic in relative position (pooled LS on pool & act); undefined -> nan."""
    pos = D.relative_position(off)
    B = np.column_stack([np.ones_like(pos), pos, pos ** 2, pos ** 3])
    out = np.full(X.shape, np.nan)
    for j in range(X.shape[1]):
        m = act[:, j]; v = X[m, j]; sd = v.std()
        if sd <= 1e-12: continue
        z = (X[:, j] - v.mean()) / sd
        fit = m & pool
        beta = np.linalg.lstsq(B[fit], z[fit], rcond=None)[0]
        out[m, j] = (z - B @ beta)[m]
    return out


# ---------------------------------------------------------------- FUSE S-hat (Eq. 4)
def moments(S):
    """S: +-1 votes (n x m). Returns centred columns, Sigma (m x m), T (m x m x m), population central moments."""
    Xc = S - S.mean(0); n = len(S)
    return Xc, Xc.T @ Xc / n, np.einsum('ni,nj,nk->ijk', Xc, Xc, Xc, optimize=True) / n


def shat(Sig, T, eps=EPS_DEN):
    """Eq. 4: sum over j3 of the sample variance, over pairs j1 < j2 < j3, of T[j1,j2,j3] / Sigma[j1,j2]
    (denominators clipped below at eps); j3 with fewer than two pairs contributes nothing."""
    m = Sig.shape[0]; tot = 0.
    for j3 in range(2, m):
        i1, i2 = np.triu_indices(j3, 1)
        if len(i1) < 2: continue
        r = T[i1, i2, j3] / np.maximum(Sig[i1, i2], eps)
        tot += float(np.var(r, ddof=1))
    return tot


def _votes(Z, act, q):
    """{0,1} votes: 1 iff defined and above the pooled q-quantile of the defined values of that voter."""
    V = np.zeros(Z.shape, int); tv = np.full(Z.shape[1], np.nan)
    for j in range(Z.shape[1]):
        m = act[:, j] & np.isfinite(Z[:, j])
        if not m.any(): continue
        tv[j] = np.quantile(Z[m, j], q[j]); V[:, j] = m & (Z[:, j] > tv[j])
    return V, tv


def fuse_taus(Z, act, fixed=(), max_sweeps=5):
    """Coordinate descent over the quantile grid; voters in `fixed` keep their given votes (Z column already {0,1})."""
    m = Z.shape[1]; q = np.full(m, .50)
    cand = {}
    for j in range(m):                                           # all candidate vote columns, once
        if j in fixed:
            cand[j] = {None: np.where(Z[:, j] > .5, 1., -1.)}; continue
        mm = act[:, j] & np.isfinite(Z[:, j]); cand[j] = {}
        for g in GRID:
            if not mm.any(): cand[j][g] = -np.ones(len(Z)); continue
            t = np.quantile(Z[mm, j], g); cand[j][g] = np.where(mm & (Z[:, j] > t), 1., -1.)
    S = np.column_stack([cand[j][None] if j in fixed else cand[j][.50] for j in range(m)])
    Xc, Sig, T = moments(S); s0 = shat(Sig, T); best = s0; trace = [s0]
    for sweep in range(max_sweeps):
        changed = False
        for j in range(m):
            if j in fixed: continue
            cur = q[j]
            for g in GRID:
                if g == cur: continue
                x = cand[j][g]; xc = x - x.mean()
                Sig2 = Sig.copy(); row = Xc.T @ xc / len(x); Sig2[j, :] = row; Sig2[:, j] = row; Sig2[j, j] = xc @ xc / len(x)
                Tj = np.einsum('n,ni,nk->ik', xc, Xc, Xc, optimize=True) / len(x)
                Tj[j, :] = np.einsum('n,nk->k', xc * xc, Xc) / len(x); Tj[:, j] = Tj[j, :]; Tj[j, j] = (xc ** 3).mean()
                T2 = T.copy(); T2[j, :, :] = Tj; T2[:, j, :] = Tj; T2[:, :, j] = Tj
                v = shat(Sig2, T2)
                if v < best - 1e-12:
                    best, q[j], changed = v, g, True
            if q[j] != cur:                                      # commit the best column for voter j
                x = cand[j][q[j]]; S[:, j] = x; Xc, Sig, T = moments(S)
        trace.append(best)
        if not changed: break
    return q, s0, best, trace


def fuse_votes(bench, prm, arm, use_cache=True):
    """Label-free E1b vote matrix for one cell and arm (see module docstring)."""
    CACHE.mkdir(exist_ok=True)
    mode = 'prm' if arm.startswith('AUTO') else 'sa'
    c = L.load(bench, prm, mode)
    cp = CACHE / f'{arm}_{bench}_{prm}.npz'
    if use_cache and cp.exists():
        z = np.load(cp, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        return dict(V=z['V'].astype(int), c=c, **meta)
    X, act = raw_matrix(mode, bench, prm, c)
    names = list(N16N)
    if arm == 'SUB5':
        idx = [N16N.index(n) for n in SUB5]; X, act, names = X[:, idx], act[:, idx], list(SUB5)
    off = c.off; n = np.diff(off); pool = np.repeat(n, n) > 1
    Z = clean(X, act, off, pool)
    fixed = ()
    if arm.startswith('AUTO'):
        risk = np.asarray(c.arms['PRM'], float)                  # logit(1 - reward): > 0 iff reward < 0.5
        if arm == 'AUTO_NATIVE':
            zp = (risk > 0).astype(float); fixed = (0,)          # native flag, not thresholded / cleaned
        else:
            zp = clean(risk[:, None], np.ones((len(risk), 1), bool), off, pool)[:, 0]
        Z = np.column_stack([zp, Z]); act = np.column_stack([np.ones(len(zp), bool), act]); names = ['PRM'] + names
    q, s0, s1, trace = fuse_taus(Z, act, fixed=fixed)
    V, tv = _votes(Z, act, np.where(np.isnan(q), .5, q))
    if fixed: V[:, 0] = (Z[:, 0] > .5).astype(int); tv[0] = np.nan; q[0] = np.nan
    meta = dict(names=names, tau_quantile=[None if np.isnan(x) else float(x) for x in q],
                tau_value=[None if np.isnan(x) else float(x) for x in tv], shat_init=s0, shat_final=s1, shat_trace=trace,
                vote_rate=V.mean(0).tolist())
    np.savez_compressed(cp, V=V.astype(np.uint8), meta=json.dumps(meta))
    return dict(V=V, c=c, **meta)
