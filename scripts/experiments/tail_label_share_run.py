"""tail_label_share_v1: how much of the between-channel dependence is label-driven, in the ordinary
covariance versus in top-k tail co-exceedance.  Mechanism diagnostic only; protocol in
results/tail_label_share_v1/PROTOCOL.json.  Labels are used to measure, never to select.
"""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/tail_label_share_v1'; OUT.mkdir(parents=True, exist_ok=True)
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
SCR = Path('C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')
INPUTS = {'pool_z': SCR / 'pool_z.npy', 'pool_names': SCR / 'pool_names.json', 'oof_answers': R / 'OOF_ANSWERS.csv',
          'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'pool_structure': ROOT / 'results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv'}
SEED = 20260924; NPERM = 200; EPS = 1e-12
CORE = ['q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level']
DROP = ['ct7_ve1', 'hist_entropy_series', 'hist_spilled_series', 'hist_trace_length_series']
T0 = time.perf_counter()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
off = Zs['offsets']; lab_all = Zs['labels'].astype(int); target = ans.target.to_numpy(); pb = ans.cell.str.startswith('pb_').to_numpy()
POOL = np.load(INPUTS['pool_z']); PN = json.loads(INPUTS['pool_names'].read_text(encoding='utf8')); assert POOL.shape == (int(off[-1]), len(PN))
names = [c for c in PN if c not in DROP]; X_all = POOL[:, [PN.index(c) for c in names]].astype(np.float64); assert np.isfinite(X_all).all()
ps = pd.read_csv(INPUTS['pool_structure']).set_index('channel')
indep = [c for c in names if c not in CORE and abs(ps.loc[c, 'r_level_marginal']) < 0.45]
core_ix = [names.index(c) for c in CORE]; ind_ix = [names.index(c) for c in indep]


def population(bench):
    """Answer slices and 0/1 step labels for one benchmark panel."""
    seg, ys = [], []
    for i in range(len(ans)):
        a, b = int(off[i]), int(off[i + 1])
        if bench == 'prmbench' and not pb[i]:
            y = lab_all[a:b]
            if set(np.unique(y)) - {0, 1}: raise ValueError(f'unexpected PRMBench label values answer {i}')
            if y.any() and (~y.astype(bool)).any(): seg.append((a, b)); ys.append(y)
        elif bench == 'processbench' and pb[i] and target[i] >= 0:
            t = int(target[i]); assert t < b - a
            if t == 0: continue            # a lone error step has no within-answer contrast
            seg.append((a, a + t + 1)); ys.append(np.r_[np.zeros(t, int), 1])
    return seg, ys


def answer_z(block):
    sd = block.std(0); return np.divide(block - block.mean(0), sd, out=np.zeros_like(block), where=sd > EPS)


def tail_ind(block, frac):
    n = len(block); k = 1 if frac is None else max(1, int(np.ceil(frac * n)))
    order = np.argsort(-block, axis=0, kind='stable'); T = np.zeros_like(block)
    np.put_along_axis(T, order[:k], 1.0, axis=0); return T - T.mean(0)


def build(seg, kind):
    parts = []
    for a, b in seg:
        z = answer_z(X_all[a:b])
        parts.append(z if kind == 'covariance' else tail_ind(z, 0.2 if kind == 'tail_top20' else None))
    return np.vstack(parts)


def between(M, starts, lens, y):
    """Between-class covariance summed over answers: sum_a n1 n0 / n * d d^T, divided by total steps."""
    S1 = np.add.reduceat(M * y[:, None], starts, axis=0); n1 = np.add.reduceat(y, starts).astype(float)
    n0 = lens - n1; tot = np.add.reduceat(M, starts, axis=0)          # within-answer centred: tot == 0
    d = S1 / n1[:, None] - (tot - S1) / n0[:, None]
    w = n1 * n0 / lens; return (d * w[:, None]).T @ d / lens.sum()


def offF(A): return float(np.sqrt(np.sum(A ** 2) - np.sum(np.diag(A) ** 2)))


def rank1_offdiag(C, anchor):
    """Rank-1 factor fit to the off-diagonal of a correlation-like matrix (iterated communalities)."""
    A = C.copy(); v = np.sqrt(np.clip(np.abs(np.diag(A)), 0, None));
    for _ in range(200):
        np.fill_diagonal(A, v ** 2); w, U = np.linalg.eigh(A); v_new = U[:, -1] * np.sqrt(max(w[-1], 0))
        if np.max(np.abs(np.abs(v_new) - np.abs(v))) < 1e-10: v = v_new; break
        v = v_new
    return v if v[anchor] >= 0 else -v


def within_auc(y, s):
    y = y.astype(bool); n1 = y.sum(); n0 = len(y) - n1
    return (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rng = np.random.default_rng(SEED); rows, sat_rows, rec_rows = [], [], []
for bench in ['prmbench', 'processbench']:
    seg, ys = population(bench); y = np.concatenate(ys).astype(float)
    lens = np.array([b - a for a, b in seg], float); starts = np.r_[0, np.cumsum(lens)[:-1]].astype(int)
    acc = np.array([np.mean([within_auc(yy, answer_z(X_all[a:b])[:, j]) for (a, b), yy in zip(seg, ys)]) for j in range(len(names))]) - 0.5
    perms = [np.concatenate([rng.permutation(yy) for yy in ys]).astype(float) for _ in range(NPERM)]
    for kind in ['covariance', 'tail_top20', 'tail_top1']:
        t = time.perf_counter(); M = build(seg, kind); N = len(M)
        S = M.T @ M / N; B = between(M, starts, lens, y); share = offF(B) / offF(S)
        null = np.array([offF(between(M, starts, lens, yp)) / offF(S) for yp in perms])
        rows.append({'benchmark': bench, 'object': kind, 'answers': len(seg), 'steps': N, 'error_steps': int(y.sum()),
                     'label_share': share, 'null_mean': float(null.mean()), 'null_q95': float(np.quantile(null, .95)),
                     'share_minus_null': share - float(null.mean()), 'perm_p': float((1 + (null >= share).sum()) / (1 + NPERM)),
                     'core_block_share': offF(B[np.ix_(core_ix, core_ix)]) / offF(S[np.ix_(core_ix, core_ix)]),
                     'indep_x_core_share': float(np.linalg.norm(B[np.ix_(ind_ix, core_ix)]) / np.linalg.norm(S[np.ix_(ind_ix, core_ix)])),
                     'seconds': time.perf_counter() - t})
        for c in indep:
            i = names.index(c)
            sat_rows.append({'benchmark': bench, 'object': kind, 'channel': c, 'signed_acc': acc[i],
                             'satellite_share': float(B[i, core_ix].sum() / np.abs(S[i, core_ix]).sum()),
                             'corr_with_core_mean': float(np.mean(S[i, core_ix] / np.sqrt(S[i, i] * np.diag(S)[core_ix])))})
        sd = np.sqrt(np.diag(S)); Cn = S / np.outer(sd, sd); Bn = B / np.outer(sd, sd)
        for src, Cm in [('observed', Cn), ('oracle_between_only', Bn)]:
            v = rank1_offdiag(Cm, names.index('q15_H1'))
            for subset, ix in [('all', list(range(len(names)))), ('independent', ind_ix)]:
                rho = spearmanr(v[ix], acc[ix]).statistic
                pn = np.array([spearmanr(rng.permutation(v[ix]), acc[ix]).statistic for _ in range(2000)])
                rec_rows.append({'benchmark': bench, 'object': kind, 'source': src, 'subset': subset, 'channels': len(ix),
                                 'spearman_loading_vs_accuracy': float(rho), 'perm_p_one_sided': float((1 + (pn >= rho).sum()) / 2001)})
        print(bench, kind, f'share={share:.4f} null={null.mean():.4f}', flush=True)

pd.DataFrame(rows).to_csv(OUT / 'LABEL_SHARE.csv', index=False)
pd.DataFrame(sat_rows).to_csv(OUT / 'SATELLITE_SHARE.csv', index=False)
pd.DataFrame(rec_rows).to_csv(OUT / 'ACCURACY_RECOVERY.csv', index=False)
(OUT / 'INPUT_MANIFEST.json').write_text(json.dumps({k: {'path': str(v), 'sha256': sha(v)} for k, v in INPUTS.items()} | {
    'channels': names, 'core': CORE, 'independent': indep, 'dropped': DROP, 'script_sha256': sha(Path(__file__)),
    'seconds': time.perf_counter() - T0}, indent=1), encoding='utf8')
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).drop(columns=['seconds']).round(4).to_string())
print(pd.DataFrame(rec_rows).round(4).to_string())
