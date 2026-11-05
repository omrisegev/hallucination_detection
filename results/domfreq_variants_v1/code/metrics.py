"""domfreq_variants_v1 stage 2: alignment (Step 0) and metrics (a)-(e) for every variant in every cell.

Labels enter ONLY here, for evaluation/diagnostics. Choosing a variant from these numbers is development-level
bank selection on already-seen data.

    python3 metrics.py <results dir>        (reads cache/CACHE_*.npz, writes RESULTS.json, ALIGNMENT.json,
                                             TOP_TABLE.json, HIST_TOP.npz, SCATTER.npz)
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

S = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1')
R = S / 'results'; BUNDLE = S / 'auto_raw_plus_z_v1/bundle/inputs'
D = Path(sys.argv[1])
PRMS = ('qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm')
SOC_DIR = {'qwen_prm_7b': 'full_qwen_prm_socratic', 'qwen_prm_72b': 'full_qwen_prm_72b_socratic',
           'pure_prm': 'full_pure_prm_socratic', 'rlhflow_prm': 'full_rlhflow_prm_socratic'}
N16 = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
       'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0',
       'realized_z', 'realized_drv', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
# computed from the token entropy series or the top-15 distribution's entropy/varentropy
ENT_FAMILY = ['q15_H1', 'q15_VE1', 'bocpd_p0', 'dominant_freq16']
CELLS = [('prmbench', p) for p in PRMS] + [('socratic', p) for p in PRMS]


def load_cell(bench, tag):
    C = np.load(D / 'cache' / f'CACHE_{bench}_{tag}.npz')
    ids, o, ki = C['ids'].astype(str), C['offsets'], C['keep_index']
    if bench == 'prmbench':
        I = np.load(BUNDLE / f'INPUTS_{tag}.npz')
        assert (I['row_ids'].astype(str) == ids).all(), 'answer order'
        rows = np.concatenate([I['offsets'][k] + ki[o[k]:o[k + 1]] for k in range(len(ids))])
        names = list(I['raw17_names'].astype(str))
        Ch = I['raw17'][rows][:, [names.index(c) for c in N16]].astype(float)
        act = I['digit_active'][rows].astype(bool); y = I['step_error'][rows].astype(bool)
        include = np.ones(len(rows), bool); stored_dom = Ch[:, N16.index('dominant_freq16')]
    else:
        F = np.load(R / 'prm_boost_external_v1' / SOC_DIR[tag] / 'bank_v2/FEATURES.npz')
        assert (F['uids'].astype(str) == ids).all(), 'answer order'
        names = list(F['names'].astype(str))
        ne = F['nonempty'].astype(bool); nl = F['nonempty_lengths']
        assert (np.diff(F['offsets']) == np.diff(o)).all() and (nl == C['n_steps_all']).all(), 'step counts'
        # nonempty mask (all steps) must agree with the cache keep_index
        all_off = np.r_[0, np.cumsum(C['n_steps_all'])]
        mk = np.zeros(all_off[-1], bool)
        for k in range(len(ids)): mk[all_off[k] + ki[o[k]:o[k + 1]]] = True
        assert (mk == ne).all(), 'nonempty mask'
        Ch = F['values'][:, [names.index(c) for c in N16]].astype(float); act = F['digit_active'].astype(bool)
        stored_dom = Ch[:, N16.index('dominant_freq16')]
        G = {g['uid']: g for g in json.load(open(D / 'labels/socratic_evaluator_only.json'))}
        corr = np.concatenate([np.asarray(G[u]['correct'], bool) for u in ids])
        inc = np.concatenate([np.asarray(G[u]['include'], bool) for u in ids])
        assert len(corr) == all_off[-1]
        y, include = ~corr[ne], inc[ne]
    step_idx = ki.astype(float); nall = np.repeat(C['n_steps_all'], np.diff(o)).astype(float)
    return dict(C=C, V=C['V'].astype(float), names=list(C['names'].astype(str)), o=o, y=y, include=include,
                Ch=Ch, act=act, stored_dom=stored_dom, step_idx=step_idx, rel_pos=step_idx / np.maximum(nall - 1, 1),
                step_len=C['step_len'].astype(float), span_lo=C['span_lo'],
                ntok=np.repeat(C['n_tokens'], np.diff(o)), fallback=C['fallback'])


def restrict(c):
    """Keep evaluated steps only (Socratic 'include'); rebuild offsets."""
    m = c['include']
    if m.all(): return c
    a = np.repeat(np.arange(len(c['o']) - 1), np.diff(c['o']))[m]
    out = {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for k, v in c.items()}
    out['o'] = np.r_[0, np.cumsum(np.bincount(a, minlength=len(c['o']) - 1))]
    return out


def within_auc(M, y, o):
    M = np.asarray(M, float); M = M[:, None] if M.ndim == 1 else M
    tot = np.zeros(M.shape[1]); n = 0
    for a, b in zip(o[:-1], o[1:]):
        yy = y[a:b]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if not n1 or not n0: continue
        r = rankdata(M[a:b], axis=0)
        tot += (r[yy].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0); n += 1
    return tot / n, n


def pooled_auc(M, y):
    M = M[:, None] if M.ndim == 1 else M
    r = rankdata(M, axis=0); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def nscore(M):
    r = rankdata(M, axis=0); return ndtri((r - .5) / len(M))


def answer_z(M, o, active=None):
    Z = np.zeros_like(M)
    for a, b in zip(o[:-1], o[1:]):
        x = M[a:b]
        if active is None:
            sd = x.std(0); Z[a:b] = np.where(sd > 1e-8, (x - x.mean(0)) / np.where(sd > 1e-8, sd, 1), 0)
        else:
            for j in range(x.shape[1]):
                m = active[a:b, j]
                if m.sum() > 1 and x[m, j].std() > 1e-8:
                    Z[a:b, j][m] = (x[m, j] - x[m, j].mean()) / x[m, j].std()
    return Z


def r2(Y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    B, *_ = np.linalg.lstsq(X1, Y, rcond=None); res = Y - X1 @ B
    return 1 - (res ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0), res


def spearman_cols(A, B):
    ra = rankdata(A, axis=0); rb = rankdata(B, axis=0)
    ra = (ra - ra.mean(0)) / ra.std(0).clip(1e-12); rb = (rb - rb.mean(0)) / rb.std(0).clip(1e-12)
    return ra.T @ rb / len(A)


def degeneracy(V):
    out = []
    for j in range(V.shape[1]):
        u, cnt = np.unique(V[:, j].astype(np.float32), return_counts=True)
        out.append((float(cnt.max() / len(V)), int(len(u))))
    return out


def main():
    align, cells, signs = {}, {}, {}
    for bench, tag in CELLS:
        c = load_cell(bench, tag)
        # STEP 0 on ALL kept steps (before the include restriction)
        d = np.abs(c['C']['dom_top10'] - c['stored_dom']); dm = np.abs(c['C']['dom_mean'] - c['stored_dom'])
        per_ans = np.array([d[a:b].max() <= 1e-6 for a, b in zip(c['o'][:-1], c['o'][1:])])
        align[f'{bench}/{tag}'] = {'answers': int(len(per_ans)), 'steps': int(len(d)),
                                   'top10_rule_answer_match_rate': float(per_ans.mean()), 'top10_max_abs': float(d.max()),
                                   'plain_mean_step_match_rate': float((dm <= 1e-6).mean()),
                                   'stored_modal_share': float(np.unique(c['stored_dom'], return_counts=True)[1].max() / len(d)),
                                   'stored_distinct': int(len(np.unique(c['stored_dom'])))}
        print(bench, tag, align[f'{bench}/{tag}'], flush=True)
        if per_ans.mean() < 0.99: raise SystemExit('HARD STOP: alignment below 99% in ' + bench + '/' + tag)
        cells[(bench, tag)] = restrict(c)
    json.dump(align, open(D / 'ALIGNMENT.json', 'w'), indent=1)

    names = cells[CELLS[0]]['names']; p = len(names)
    res = {}
    for bench, tag in CELLS:
        c = cells[(bench, tag)]; V, y, o, Ch = c['V'], c['y'], c['o'], c['Ch']
        # reference rows: stored dominant_freq16 and step position
        refs = np.column_stack([c['stored_dom'], c['step_idx'], c['rel_pos'], c['step_len']])
        w_raw, n_ans = within_auc(np.column_stack([V, refs]), y, o)
        pl_raw = pooled_auc(np.column_stack([V, refs]), y)
        key = f'{bench}/{tag}'
        if bench == 'prmbench': signs[tag] = np.where(w_raw[:p] >= .5, 1., -1.)
        sg = signs[tag]                                   # Socratic uses the same PRM's PRMBench sign
        # redundancy
        rho = spearman_cols(V, Ch)                        # [p, 16]
        Zv, Zc = nscore(V), nscore(Ch)
        R2all, resid = r2(Zv, Zc)
        R2ent, _ = r2(Zv, Zc[:, [N16.index(e) for e in ENT_FAMILY]])
        w_res, _ = within_auc(resid, y, o)
        rho_pos = spearman_cols(V, np.column_stack([c['step_idx'], c['rel_pos'], c['step_len']]))
        # equal-weight z-average of the 16 channels, plus the variant (sign sg), no fitted weights
        Z16 = answer_z(Ch, o, None)
        dz = answer_z(Ch[:, [N16.index(x) for x in DIG]], o, c['act'])
        for k, x in enumerate(DIG): Z16[:, N16.index(x)] = dz[:, k]
        s16 = Z16.mean(1); base, _ = within_auc(s16, y, o)
        Zvar = answer_z(V, o, None) * sg
        g, _ = within_auc((16 * s16[:, None] + Zvar) / 17, y, o)
        deg = degeneracy(V)
        fb_share = c['fallback'].mean(0)
        last_tok = c['span_lo'] + c['step_len'] - 1
        rows = {}
        for j, nm in enumerate(names):
            w = nm.split('|')[2]
            if w.startswith('w'):
                W = int(w[1:]); short = float(((last_tok < W - 1) | (c['ntok'] < W)).mean())
            else: short = float(fb_share[j])
            top = np.argsort(-np.abs(rho[j]))[:3]
            rows[nm] = {'sign': int(sg[j]), 'within_raw': float(w_raw[j]), 'within_or': float(sg[j] * (w_raw[j] - .5) + .5),
                        'pooled_or': float(sg[j] * (pl_raw[j] - .5) + .5), 'modal_share': deg[j][0], 'distinct': deg[j][1],
                        'fallback_share': float(fb_share[j]), 'short_or_backfill_share': short,
                        'top3_rho': [(N16[t], float(rho[j, t])) for t in top],
                        'max_abs_rho_ent': float(np.abs(rho[j, [N16.index(e) for e in ENT_FAMILY]]).max()),
                        'max_abs_rho_ent_name': ENT_FAMILY[int(np.argmax(np.abs(rho[j, [N16.index(e) for e in ENT_FAMILY]])))],
                        'R2_all16': float(R2all[j]), 'R2_ent': float(R2ent[j]),
                        'resid_within_or': float(sg[j] * (w_res[j] - .5) + .5),
                        'zavg_gain': float(g[j] - base[0]),
                        'rho_step_index': float(rho_pos[j, 0]), 'rho_rel_pos': float(rho_pos[j, 1]), 'rho_step_len': float(rho_pos[j, 2])}
        rows['_ref'] = {'stored_dominant_freq16_within_raw': float(w_raw[p]), 'step_index_within_raw': float(w_raw[p + 1]),
                        'rel_pos_within_raw': float(w_raw[p + 2]), 'step_len_within_raw': float(w_raw[p + 3]),
                        'stored_dominant_freq16_pooled_raw': float(pl_raw[p]),
                        'zavg16_within': float(base[0]), 'answers_with_both_classes': int(n_ans),
                        'steps_evaluated': int(len(y)), 'error_rate': float(y.mean())}
        res[key] = rows
        print(key, rows['_ref'], flush=True)
    json.dump(res, open(D / 'RESULTS.json', 'w'))

    # selection (declared before reading the table; development-level, label-using):
    # non-reference source series; same PRMBench sign in all 4 PRMs and Socratic on the same side in all 4;
    # modal share < 5% in every cell; rank by the minimum oriented within-AUC over the 8 cells;
    # keep the best configuration per (series, statistic).
    keys = [f'{b}/{t}' for b, t in CELLS]
    cand = []
    for nm in names:
        r = [res[k][nm] for k in keys]
        s = {x['sign'] for x in r}
        cons = len(s) == 1 and all((x['within_raw'] - .5) * x['sign'] > 0 for x in r)
        cand.append({'name': nm, 'consistent': bool(cons), 'min_within_or': min(x['within_or'] for x in r),
                     'mean_within_or': float(np.mean([x['within_or'] for x in r])),
                     'max_modal': max(x['modal_share'] for x in r), 'max_rho_ent': max(x['max_abs_rho_ent'] for x in r),
                     'mean_R2_all16': float(np.mean([x['R2_all16'] for x in r])),
                     'mean_resid_within_or': float(np.mean([x['resid_within_or'] for x in r])),
                     'mean_zavg_gain': float(np.mean([x['zavg_gain'] for x in r])),
                     'min_zavg_gain': float(min(x['zavg_gain'] for x in r)), 'reference': nm.startswith('entropyREF')})
    json.dump(cand, open(D / 'CANDIDATES.json', 'w'), indent=0)
    ok = [x for x in cand if x['consistent'] and x['max_modal'] < .05 and not x['reference']]
    ok.sort(key=lambda x: -x['min_within_or'])
    seen, top = set(), []
    for x in ok:
        k = tuple(x['name'].split('|')[:2])
        if k in seen: continue
        seen.add(k); top.append(x)
    # second view: the most non-redundant (best mean residual within-AUC) among the same filter
    ok2 = sorted(ok, key=lambda x: -x['mean_resid_within_or']); seen2, top2 = set(), []
    for x in ok2:
        k = tuple(x['name'].split('|')[:2])
        if k in seen2: continue
        seen2.add(k); top2.append(x)
    json.dump({'n_variants': p, 'n_pass_filter': len(ok), 'top_by_min_within': top[:12], 'top_by_resid': top2[:12],
               'entropy_reference_best': sorted([x for x in cand if x['reference'] and x['consistent']],
                                                key=lambda x: -x['min_within_or'])[:5]},
              open(D / 'TOP_TABLE.json', 'w'), indent=1)
    # small plotting payloads (plots are drawn off-cluster: no matplotlib in the CPU env)
    sel = [x['name'] for x in top[:6]]
    hp = {}
    for b, t in CELLS:
        c = cells[(b, t)]
        for nm in sel: hp[f'{b}/{t}/{nm}'] = c['V'][:, names.index(nm)].astype(np.float32)
        hp[f'{b}/{t}/__y'] = c['y']
    np.savez_compressed(D / 'HIST_TOP.npz', **hp)
    sc = {k: np.array([[res[k][nm]['within_or'], res[k][nm]['R2_all16'], res[k][nm]['max_abs_rho_ent']] for nm in names]) for k in keys}
    np.savez_compressed(D / 'SCATTER.npz', names=np.array(names), **{k.replace('/', '__'): v for k, v in sc.items()})
    print('DONE metrics; pass filter', len(ok), flush=True)


if __name__ == '__main__':
    main()
