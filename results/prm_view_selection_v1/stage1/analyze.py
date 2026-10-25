"""Pre-registered analyses of the stage-1 subset search (docs/experiments/PRM_VIEW_SELECTION_STAGE1_PREREG_2026-10-25.md).

    python analyze.py summary          # SEARCH.npz -> ANALYSIS.json, SUBSETS_NAMED.csv, CHANNEL_FREQ.csv, STRUCTURE.csv
    python analyze.py detail           # named subsets: exact re-check, within-AUC, CHAIN_N16, supervised LR -> NAMED_DETAIL.csv
    python analyze.py hem [workers]    # HEM (secondary estimator) on successes + top 300 by min-margin -> HEM.csv
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2] / 'scripts/experiments'))
import stage1 as S1  # noqa: E402
E = S1.E

NAMES = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
         'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
         'digit_spread', 'digit_alternative_innovation']
N_TOP_HEM = 300
PB, SO = list(range(5)), list(range(5, 10))


def names_of(mask):
    return [NAMES[i] for i in S1.subset_of(int(mask))]


def load_search():
    D = dict(np.load(HERE / 'SEARCH.npz'))
    cells = [str(x) for x in D['cells']]
    par = json.loads((HERE / 'PARITY.json').read_text())
    prm = np.array([par[c]['prm_exact'] for c in cells])
    valid_all = D['valid'].all(1)
    mg = D['ps_fused'] - np.maximum(prm[None, :], D['ps_mean'])
    return D, cells, prm, valid_all, mg


def select(mg, valid_all, cols):
    m = np.where(valid_all, mg[:, cols].min(1), -np.inf)
    return int(np.argmax(m)), float(m.max())


def named_selections(D, mg, valid_all):
    named = {'best_all10': select(mg, valid_all, list(range(10))),
             'S_PB (selected on PRMBench)': select(mg, valid_all, PB),
             'S_SO (selected on Socratic)': select(mg, valid_all, SO)}
    for p in range(5):
        named[f'LOPO {S1.PRMS[p]}'] = select(mg, valid_all, [j for j in range(10) if j not in (p, p + 5)])
    return named


def summary():
    D, cells, prm, valid_all, mg = load_search()
    mgb = D['ps_bin'] - np.maximum(prm[None, :], D['ps_mean'])
    win_prm = D['ps_fused'] > prm[None, :]; win_mean = D['ps_fused'] > D['ps_mean']
    succ = valid_all & (mg > 0).all(1)
    size = np.array([bin(int(m)).count('1') for m in D['masks']])
    V = valid_all
    out = dict(n_subsets=int(len(D['masks'])), n_valid_all=int(V.sum()), n_success=int(succ.sum()),
               n_success_binary_scoring=int((V & (mgb > 0).all(1)).sum()),
               n_beat_prm_all10=int((V & win_prm.all(1)).sum()), n_beat_mean_all10=int((V & win_mean.all(1)).sum()),
               n_success_prmbench5=int((V & (mg[:, PB] > 0).all(1)).sum()), n_success_socratic5=int((V & (mg[:, SO] > 0).all(1)).sum()),
               n_beat_prm_socratic5=int((V & win_prm[:, SO].all(1)).sum()),
               by_size={int(k): dict(valid=int((V & (size == k)).sum()), success=int((succ & (size == k)).sum()),
                                     best_min_margin=float(np.max(np.where(V & (size == k), mg.min(1), -np.inf)))) for k in np.unique(size)},
               cells_won_hist={int(k): int((V & ((mg > 0).sum(1) == k)).sum()) for k in range(11)},
               per_cell={c: dict(prm=float(prm[j]), beat_prm=float(win_prm[V, j].mean()), beat_mean=float(win_mean[V, j].mean()),
                                 both=float((mg[V, j] > 0).mean()), best_margin=float(mg[V, j].max()),
                                 best_fused=float(D['ps_fused'][V, j].max())) for j, c in enumerate(cells)},
               min_margin_quantiles={str(q): float(np.quantile(mg[V].min(1), q)) for q in (0, .5, .9, .99, 1)},
               w_prm_share_median=float(np.nanmedian(D['w_prm_share'][V])),
               clip_within_rate=float((D['clip_within'][V] > 0).mean()), clip_between_rate=float((D['clip_between'][V] > 0).mean()))
    named = named_selections(D, mg, valid_all)
    rows = []
    for nm, (i, v) in named.items():
        r = dict(selection=nm, subset=' + '.join(names_of(D['masks'][i])), size=int(size[i]), select_min_margin=v,
                 min_margin_all10=float(mg[i].min()), cells_won=int((mg[i] > 0).sum()))
        for j, c in enumerate(cells):
            r[f'margin {c}'] = float(mg[i, j]); r[f'fused {c}'] = float(D['ps_fused'][i, j]); r[f'mean {c}'] = float(D['ps_mean'][i, j])
        rows.append(r)
    E.write_csv(HERE / 'SUBSETS_NAMED.csv', rows)
    out['named'] = {r['selection']: dict(subset=r['subset'], cells_won=r['cells_won'], min_margin_all10=r['min_margin_all10']) for r in rows}
    out['cross_benchmark'] = {k: [float(mg[named[k][0], j]) for j in (SO if 'PRMBench' in k else PB)]
                              for k in ('S_PB (selected on PRMBench)', 'S_SO (selected on Socratic)')}
    out['lopo_heldout'] = {S1.PRMS[p]: [float(mg[named[f'LOPO {S1.PRMS[p]}'][0], j]) for j in (p, p + 5)] for p in range(5)}
    # channel frequency (successes; if none, the top 1% of valid subsets by min-margin, labelled)
    q99 = np.quantile(mg[V].min(1), .99)
    ref = succ if succ.sum() else (V & (np.where(V, mg.min(1), -np.inf) >= q99))
    out['channel_freq_reference'] = 'successful subsets' if succ.sum() else f'top 1% of valid subsets by min-margin ({int(ref.sum())}; no success)'
    bits = ((D['masks'][:, None] >> np.arange(16)) & 1).astype(bool)
    E.write_csv(HERE / 'CHANNEL_FREQ.csv', [dict(channel=NAMES[k], freq_valid=float(bits[V, k].mean()), freq_reference=float(bits[ref, k].mean()))
                                           for k in range(16)])
    # Omri's structural conditions on binary votes (label-free), from the light caches
    corr = [np.corrcoef(CellData_light(b, p).Vp.T) for b, p in S1.CELLS]
    iv = np.flatnonzero(V); st = np.zeros((len(iv), 2))
    for n, i in enumerate(iv):
        cols = np.asarray(S1.subset_of(int(D['masks'][i]))) + 1; k = len(cols); fr, mp = [], []
        for R in corr:
            sub = R[np.ix_(cols, cols)]
            fr.append(float((R[0, cols] > (sub.sum(1) - 1) / (k - 1)).mean())); mp.append(float((sub.sum() - k) / (k * (k - 1))))
        st[n] = np.mean(fr), np.mean(mp)
    isr = ref[iv]; mm = mg[iv].min(1)
    out['structure'] = dict(frac_channels_prm_gt_mean_other_valid=float(st[:, 0].mean()), frac_reference=float(st[isr, 0].mean()),
                            mean_pairwise_corr_valid=float(st[:, 1].mean()), pairwise_reference=float(st[isr, 1].mean()),
                            spearman_minmargin_vs_frac=float(spearmanr(st[:, 0], mm)[0]),
                            spearman_minmargin_vs_pairwise=float(spearmanr(st[:, 1], mm)[0]))
    E.write_csv(HERE / 'STRUCTURE.csv', [dict(subset=' + '.join(names_of(D['masks'][i])), min_margin=float(mg[i].min()),
                                             cells_won=int((mg[i] > 0).sum()), frac_prm_gt_mean_other=float(st[n, 0]),
                                             mean_pairwise_corr=float(st[n, 1])) for n, i in enumerate(iv)])
    json.dump(out, open(HERE / 'ANALYSIS.json', 'w'), indent=1, default=float)
    print(json.dumps({k: out[k] for k in out if k not in ('per_cell', 'by_size', 'named')}, indent=1, default=float))


def CellData_light(b, p):
    return S1.CellData.from_light(HERE / 'cache' / f'{b}__{p}.npz')


def detail():
    """Named subsets only: exact re-check through the evaluator's loops, within-AUC, CHAIN_N16 and a supervised LR
    on the same views (context)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    D, cells, prm, valid_all, mg = load_search()
    named = named_selections(D, mg, valid_all)
    idx = list(dict.fromkeys(i for i, _ in named.values()))
    rows = {i: dict(subset=' + '.join(names_of(D['masks'][i])), selections='; '.join(k for k, (j, _) in named.items() if j == i)) for i in idx}
    for j, (b, p) in enumerate(S1.CELLS):
        cell = S1.CellData(b, p); c = cell.c
        groups = c.groups if len(c.groups) == c.off[-1] else np.repeat(c.groups, np.diff(c.off))
        ex = lambda s: cell.metric_exact(E.topk_flags(E.answer_z(s, c.off), c.off, E.gap_counts(E.answer_z(s, c.off), c.off)))
        chain = E.measure(c, 'CHAIN_N16', 'GAP', fast=True)[0]
        for i in idx:
            Sx = S1.subset_of(int(D['masks'][i])); cols = np.asarray(Sx) + 1; r = rows[i]
            o = cell.evaluate(Sx, want_detail=True)
            r[f'fused {b}/{p}'] = ex(o['fused']); r[f'mean {b}/{p}'] = ex(o['mean']); r[f'prm {b}/{p}'] = cell.ps_prm
            r[f'exact_ok {b}/{p}'] = bool(abs(r[f'fused {b}/{p}'] - o['ps_fused']) < 1e-9 and abs(r[f'mean {b}/{p}'] - o['ps_mean']) < 1e-9)
            keep = c.keep if c.bench != 'prmbench' else None
            r[f'within_fused {b}/{p}'] = float(np.nanmean(E.per_answer_auc(c.err, o['fused'], c.off, keep)))
            r[f'within_prm {b}/{p}'] = float(np.nanmean(E.per_answer_auc(c.err, cell.Z[:, 0], c.off, keep)))
            r[f'chain_n16 {b}/{p}'] = chain['official'] * (1 if c.bench == 'prmbench' else 100)
            X = cell.Z[:, np.r_[0, cols]]; y = c.err.astype(int); pred = np.zeros(len(y))
            for a, t in GroupKFold(5).split(X, y, groups):
                m_tr = np.zeros(len(y), bool); m_tr[a] = True; m_tr &= cell.pool
                pred[t] = LogisticRegression(class_weight='balanced', max_iter=2000).fit(X[m_tr], y[m_tr]).decision_function(X[t])
            r[f'lr_supervised {b}/{p}'] = ex(pred)
            r[f'w_prm_share {b}/{p}'] = o['w_prm_share']; r[f'K {b}/{p}'] = o['K']
        print(b, p, 'done', flush=True); del cell
    E.write_csv(HERE / 'NAMED_DETAIL.csv', list(rows.values()))


# ----------------------------------------------------------------------------------------------- HEM (secondary)
def hem_weights(V_pool, groups):
    """Exact latent-group EM (cvf_v2.em, kind='hem'); PRM = column 0 in its own group. Linearized non-negative weights."""
    from cvf_v2.em import fit_em

    class _Maj:
        def predict(self, x): return np.asarray(x, float).mean(1)
    mdl = fit_em(V_pool, np.ones(len(V_pool)), 'hem', _Maj(), groups=groups)
    e, t = np.array(mdl.emissions, float), np.array(mdl.transition, float)
    lo = lambda a, b: np.log(a * (1 - b) / (b * (1 - a)))
    within, between = [], np.zeros(len(t))
    for g in range(len(t)):
        idx = np.flatnonzero(groups == g)
        if len(idx) == 1:                                        # alpha is the vote: effective rates given Y
            p1 = t[g, 1] * e[idx[0], 1] + (1 - t[g, 1]) * e[idx[0], 0]
            p0 = t[g, 0] * e[idx[0], 1] + (1 - t[g, 0]) * e[idx[0], 0]
            within.append(np.ones(1)); between[g] = lo(p1, p0); continue
        ww = lo(e[idx, 1], e[idx, 0]); tg = t[g].copy()
        if ww.sum() < 0: ww, tg = -ww, 1 - tg                    # alpha label switch inside the group
        within.append(np.clip(ww, 0, None)); between[g] = lo(tg[1], tg[0])
    if between[groups[0]] < 0: between = -between                # Y label switch: the PRM anchors the direction
    return within, np.clip(between, 0, None), bool(mdl.diagnostics.get('converged'))


_HC = None


def _hem_init():
    global _HC
    _HC = [CellData_light(b, p) for b, p in S1.CELLS]


def _hem_one(mask):
    Sx = S1.subset_of(int(mask)); cols = np.asarray(Sx) + 1; res = []
    for cell in _HC:
        o = cell.evaluate(Sx, want_detail=True)
        if not o['valid']: res.append((np.nan, np.nan, False)); continue
        g = np.r_[0, o['groups'] + 1]
        try:
            within, between, conv = hem_weights(cell.Vp[:, np.r_[0, cols]], g)
        except Exception:                                        # noqa: BLE001
            res.append((np.nan, o['ps_mean'], False)); continue
        comp = [cell.Z[:, 0]] + [cell.ans.z(cell.Z[:, cols[o['groups'] == k]] @ within[k + 1]) if within[k + 1].sum() > 0
                                 else np.zeros(cell.ans.n) for k in range(o['K'])]
        s = sum(between[k] * comp[k] for k in range(len(comp)))
        res.append((cell.metric(cell.flags(s)) if np.any(between > 0) else np.nan, o['ps_mean'], conv))
    return int(mask), res


def hem(workers):
    import multiprocessing as mp
    D, cells, prm, valid_all, mg = load_search()
    succ = valid_all & (mg > 0).all(1)
    iv = np.flatnonzero(valid_all); order = iv[np.argsort(-mg[iv].min(1))]
    pool_idx = list(dict.fromkeys(list(np.flatnonzero(succ)) + list(order[:N_TOP_HEM])))
    t0 = time.time(); rows = []
    with mp.get_context('spawn').Pool(workers, initializer=_hem_init) as pool:
        for n, (mask, res) in enumerate(pool.imap_unordered(_hem_one, [int(D['masks'][i]) for i in pool_idx])):
            i = int(np.flatnonzero(D['masks'] == mask)[0])
            r = dict(subset=' + '.join(names_of(mask)), primary_min_margin=float(mg[i].min()), primary_cells_won=int((mg[i] > 0).sum()))
            hm = []
            for j, (h, mean, conv) in enumerate(res):
                r[f'hem {cells[j]}'] = h; r[f'hem_converged {cells[j]}'] = conv; hm.append(h - max(prm[j], mean))
            r['hem_min_margin'] = float(np.min(hm)); r['hem_cells_won'] = int(np.sum(np.array(hm) > 0))
            rows.append(r)
            if n % 25 == 0: print(f'{n + 1}/{len(pool_idx)} {time.time() - t0:.0f}s', flush=True)
    E.write_csv(HERE / 'HEM.csv', rows)
    hm = np.array([r['hem_min_margin'] for r in rows])
    print(json.dumps(dict(n=len(rows), n_success_hem=int(np.sum(hm > 0)), best_hem_min_margin=float(np.nanmax(hm)),
                          max_hem_cells_won=int(max(r['hem_cells_won'] for r in rows))), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'summary': summary()
    elif cmd == 'detail': detail()
    elif cmd == 'hem': hem(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
