"""Label-free structure of the 18-stream union (11-channel bank + CT7 seven) at step level.
Top10 step readout per stream, answer-local standardization, then on the 94,203 labelled
PRMBench steps: conditional (within-label-class) correlation, residual-affinity spectral
partitions K=3..6, conditional PR of the union and of sub-banks. Per-stream within-answer AUROC
is printed as a DIAGNOSTIC column only (labels used for evaluation, never for any fit).
Three-tag: (step, conditional, raw streams)."""
import sys, time, json
import numpy as np
from pathlib import Path
MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); W = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
sys.path.insert(0, str(W))
from spectral_utils.ssl_eval import Frame
from spectral_utils.ssl_s1 import within_auc
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
t0 = time.time()
F = Frame(); off = F.off; n = F.n; ns = F.nsteps
A = np.load(MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz')
B = np.load(MAIN / 'results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz')
spans = A['step_spans']; toff = A['token_offsets']
XA = A['tokens']; XB = B['tokens']; VB = B['valid']
names = list(A['channels']) + ['ct7_' + c for c in B['channels']]
m = len(names)
prof = np.full((int(off[-1]), m), np.nan)
K10 = 10
for i in range(n):
    ta, tb = toff[i:i + 1 + 1]
    xa = XA[ta:tb].astype(float); xb = XB[ta:tb].copy(); xb[~VB[ta:tb]] = np.nan
    x = np.concatenate([xa, xb], 1)
    for s in range(off[i], off[i + 1]):
        a, b = spans[s]; seg = x[a:b]
        k = min(K10, b - a)
        srt = np.sort(np.where(np.isfinite(seg), seg, -np.inf), axis=0)[-k:]
        srt[~np.isfinite(srt)] = np.nan
        prof[s] = np.nanmean(srt, 0)
    if i % 2000 == 0: print(f'{i}/{n} {time.time()-t0:.0f}s', flush=True)
np.save(OUT / 'union_top10_profiles.npy', prof)
# answer-local standardization (finite-masked; constant -> 0)
Z = np.full_like(prof, np.nan)
for i in range(n):
    a, b = off[i:i + 2]; x = prof[a:b]
    mu = np.nanmean(x, 0); sd = np.nanstd(x, 0)
    z = (x - mu) / np.where(sd > 1e-12, sd, np.inf)
    Z[a:b] = np.where(np.isfinite(z), z, 0.0)
np.save(OUT / 'union_top10_z.npy', Z)
# labelled PRMBench steps
prm_steps = np.repeat(F.prm, ns); y = F.labels.astype(bool)
sel = prm_steps & np.isfinite(Z).all(1)
Zs = Z[sel]; ys = y[sel]
print('labelled PRMB steps used', sel.sum(), 'errors', ys.sum())
# conditional (within-label-class) centring
Zc = Zs.copy()
for c in (False, True):
    mm = ys == c; Zc[mm] -= Zc[mm].mean(0)
C = np.corrcoef(Zc.T)
Cm = np.corrcoef(Zs.T)
def pr(Cm_): ev = np.linalg.eigvalsh(Cm_); ev = ev[ev > 0]; return float(ev.sum() ** 2 / (ev ** 2).sum())
# residual-affinity spectral partition (Step 413 route) at K=3..6
from spectral_utils.fusion_utils import _spectral_cluster_precomputed
def residual_affinity(Cm_):
    # inline re-implementation of the Step 413 route: rank-1 off-diagonal loading, affinity = |C - v v^T|
    M = np.array(Cm_, float); v = np.sqrt(np.clip(np.diag(M), 1e-12, None))
    for _ in range(200):                                  # alternating rank-1 fit to the off-diagonal
        for i in range(len(M)):
            oth = np.arange(len(M)) != i
            v[i] = (M[i, oth] @ v[oth]) / max(v[oth] @ v[oth], 1e-12)
    aff = np.abs(M - np.outer(v, v)); np.fill_diagonal(aff, 0.0); return aff, v
res = {'names': names, 'n_steps': int(sel.sum()), 'pr_conditional_union': pr(C), 'pr_marginal_union': pr(Cm)}
idx11 = list(range(11)); idx7 = list(range(11, 18))
res['pr_conditional_11'] = pr(C[np.ix_(idx11, idx11)]); res['pr_conditional_ct7'] = pr(C[np.ix_(idx7, idx7)])
parts = {}
for K in (3, 4, 5, 6):
    try:
        aff, _load = residual_affinity(C)
        lab = _spectral_cluster_precomputed(aff, K)
        parts[K] = [int(v) for v in np.asarray(lab)]
    except Exception as e:
        parts[K] = str(e)
res['partitions'] = parts
# diagnostic: per-stream within-answer AUROC on PRMB (labels used for EVALUATION only)
aucs = {}
for j, nm in enumerate(names):
    vals = []
    for i in np.flatnonzero(F.prm):
        a, b = off[i:i + 2]; yy = y[a:b]
        if yy.any() and (~yy).any(): vals.append(within_auc(yy, Z[a:b, j]))
    aucs[nm] = float(np.nanmean(vals))
res['within_auc_single_diagnostic'] = aucs
json.dump(res, open(OUT / 'union_structure.json', 'w'), indent=1)
np.save(OUT / 'union_cond_corr.npy', C)
# print
print('\nconditional |corr| (step, conditional, raw), 18 streams:')
short = [nm.replace('ct7_', '*')[:12] for nm in names]
print(' ' * 14 + ''.join(f'{s:>7s}' for s in short))
for i in range(m): print(f'{short[i]:14s}' + ''.join(f'{abs(C[i, j]):7.2f}' for j in range(m)))
print('\nPR conditional: union', round(res['pr_conditional_union'], 3), ' 11-bank', round(res['pr_conditional_11'], 3), ' ct7', round(res['pr_conditional_ct7'], 3), ' | marginal union', round(res['pr_marginal_union'], 3))
for K, lab in parts.items():
    if isinstance(lab, list):
        groups = {}
        for nm, g in zip(names, lab): groups.setdefault(g, []).append(nm)
        print(f'K={K}:', ' | '.join(', '.join(v) for v in groups.values()))
    else: print(f'K={K}: {lab}')
print('\nsingle-stream within-AUC (diagnostic):'); [print(f'  {nm:28s} {v:.4f}') for nm, v in sorted(aucs.items(), key=lambda kv: -kv[1])]
print('done', round(time.time() - t0), 's')
