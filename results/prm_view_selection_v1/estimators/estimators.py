"""Between-group estimators (Part A) and the next-step PRM view in CHAIN_N16 (Part B).

Pre-registration: docs/experiments/PRM_VIEW_SELECTION_ESTIMATORS_NEXTSTEP_PREREG_2026-10-28.md (240bc0ea).
Label-free fitting; labels only score.

    python estimators.py time            # one PRMBench cell, all arms, timing only
    python estimators.py run             # 12 cells -> RESULTS.json, RESULTS_A.csv, RESULTS_B.csv, PARITY.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import contextlib, csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'stage1')); sys.path.insert(0, str(HERE.parent / 'stage2'))
sys.path.insert(0, str(ROOT / 'results/prm_boost_math_rigor_v1')); sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import stage1 as S1  # noqa: E402
import stage2 as S2  # noqa: E402
import common as MR  # noqa: E402
from cvf_v2.em import fit_em  # noqa: E402
from collection_parity import h2v_common_uids  # noqa: E402
from spectral_utils import prm_anchored_lsml as A  # noqa: E402

E = S1.E
NAMES = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
         'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
         'digit_spread', 'digit_alternative_innovation']
SETS = {'SUB5': [NAMES.index(n) for n in
                 'q15_H1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0 + digit_spread'.split(' + ')],
        'N16': list(range(16))}
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
BENCHES = ['prmbench', 'socratic', 'hard2verify']
ARMS = ['DS_MAJ', 'DS_SPEC', 'HEM_SPEC', 'HEM_MAJ', 'SML_ONLY']
B, SEED = 2000, 20261028
COMMON = h2v_common_uids()
SEARCH = {s: np.load(HERE.parent / s / 'SEARCH.npz') for s in ('stage1', 'stage2')}
H2V_PRIOR = {(r['prm'], r['version']): r['fused_ps'] for r in
             json.loads((HERE.parent / 'h2v_holdout' / 'H2V_SUB4_SUB5.json').read_text())['rows'] if r['set'] == 'SUB5'}


@contextlib.contextmanager
def common_set():
    """Part A: Hard2Verify cells on the 177-answer common set for fitting and scoring (as h2v_holdout/)."""
    load = E.load
    E.load = lambda bench, prm, **kw: load(bench, prm, **({'uids_keep': COMMON} if bench == 'hard2verify' else {}), **kw)
    try:
        yield
    finally:
        E.load = load


# ----------------------------------------------------------------------------------------------- HEM
class _Score:
    def __init__(self, v): self.v = v
    def predict(self, x): return np.asarray(x, float) @ self.v


class _Maj:
    def predict(self, x): return np.asarray(x, float).mean(1)


def hem_weights(V_pool, groups, start, anchor_group):
    """stage1/analyze.py hem_weights, with the start object passed in and, for anchor_group=None, the majority switch
    (flip if fewer than half of the groups have a positive between weight)."""
    mdl = fit_em(V_pool, np.ones(len(V_pool)), 'hem', start, groups=groups)
    e, t = np.array(mdl.emissions, float), np.array(mdl.transition, float)
    lo = lambda a, b: np.log(a * (1 - b) / (b * (1 - a)))
    within, between = [], np.zeros(len(t))
    for g in range(len(t)):
        idx = np.flatnonzero(groups == g)
        if len(idx) == 1:
            p1 = t[g, 1] * e[idx[0], 1] + (1 - t[g, 1]) * e[idx[0], 0]
            p0 = t[g, 0] * e[idx[0], 1] + (1 - t[g, 0]) * e[idx[0], 0]
            within.append(np.ones(1)); between[g] = lo(p1, p0); continue
        ww = lo(e[idx, 1], e[idx, 0]); tg = t[g].copy()
        if ww.sum() < 0: ww, tg = -ww, 1 - tg
        within.append(np.clip(ww, 0, None)); between[g] = lo(tg[1], tg[0])
    if anchor_group is not None:
        if between[anchor_group] < 0: between = -between
    elif np.sum(between > 0) < len(between) / 2:
        between = -between
    return within, np.clip(between, 0, None), bool(mdl.diagnostics.get('converged'))


# ----------------------------------------------------------------------------------------------- Part A
def fuse(cell, S, arm, with_prm):
    """One fused score. Mirrors CellData.evaluate (with_prm) / TeleCell.evaluate (alone); only the estimator varies."""
    cols = np.asarray(S) + (1 if with_prm else 0)
    K, cg = A.groups_from_cov(cell.R[np.ix_(cols, cols)])
    out = dict(K=int(K), valid=K >= 3)
    if not out['valid']:
        out['reason'] = 'K < 3'; return out
    mean = cell.Z[:, np.r_[0, cols]].mean(1) if with_prm else cell.Z[:, cols].mean(1)
    if with_prm:
        tie_p, tie = cell.Vp[:, 0], cell.V[:, 0]
    else:
        tie = np.where(cell.flags(mean), 1., -1.); tie_p = tie[cell.pool]
    if arm.startswith('HEM'):
        Xcols = np.r_[0, cols] if with_prm else cols
        g = np.r_[0, cg + 1] if with_prm else cg
        Vp = cell.Vp[:, Xcols]
        start = _Score(A.spectral_vote_weights(Vp)) if arm == 'HEM_SPEC' else _Maj()
        try:
            within, between, conv = hem_weights(Vp, g, start, 0 if with_prm else None)
        except ArithmeticError as ex:
            out.update(valid=False, reason=f'HEM: {ex}'); return out
        if between.sum() <= 0:
            out.update(valid=False, reason='all between weights clipped'); return out
        comps = [cell.ans.z(cell.Z[:, Xcols[g == k]] @ within[k]) if within[k].sum() > 0 else np.zeros(cell.ans.n)
                 for k in range(len(between))]
        if with_prm: comps[0] = cell.Z[:, 0]
        fused = sum(between[k] * comps[k] for k in range(len(between)))
        out.update(converged=conv, w=between.tolist())
    else:
        xi_p, comps = ([cell.Vp[:, 0]] if with_prm else []), ([cell.Z[:, 0]] if with_prm else [])
        for k in range(K):
            idx = cols[cg == k]
            w, _ = A.sml_weights_nonneg(cell.R[np.ix_(idx, idx)])
            if len(idx) == 1:
                a_p = cell.Vp[:, idx[0]]
            else:
                sp = cell.Vp[:, idx] @ w
                a_p = np.where(sp > 0, 1., np.where(sp < 0, -1., tie_p))
            xi_p.append(a_p); comps.append(cell.ans.z(cell.Z[:, idx] @ w))
        Bv = np.column_stack(xi_p)
        if arm == 'SML_ONLY':
            wb, _ = A.sml_weights_nonneg(np.cov(Bv.T))
        else:
            wb = A.ds_em_votes(Bv, anchor=0 if with_prm else None, init='spectral' if arm == 'DS_SPEC' else 'majority')['w']
        if wb.sum() <= 0:
            out.update(valid=False, reason='all between weights clipped'); return out
        fused = sum(wb[k] * comps[k] for k in range(len(comps)))
        out['w'] = np.asarray(wb).tolist()
    out.update(fused=fused, mean=mean)
    return out


# ----------------------------------------------------------------------------------------------- metrics
def gapf(c, s):
    z = E.answer_z(s, c.off)
    return E.topk_flags(z, c.off, E.gap_counts(z, c.off))


def official(c, f):
    return float(E.official(c, f)[0]) * (1 if c.bench == 'prmbench' else 100)


def within(c, s):
    return float(np.nanmean(E.per_answer_auc(c.err, s, c.off, c.keep if c.bench != 'prmbench' else None)))


class Boot:
    """Paired bootstrap of the official metric from per-unit confusion counts (PRMBench: source groups; external:
    answers, c.groups = ids)."""

    def __init__(self, c):
        self.c = c
        u, self.inv = np.unique(np.asarray(c.groups).astype(str), return_inverse=True)
        rng = np.random.default_rng(SEED)
        self.W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(float)
        self.nu = len(u); self.k = 1 if c.bench == 'prmbench' else 100

    def units(self, f):
        C = E.counts_by_answer(self.c, f).astype(float); U = np.zeros((self.nu, 4)); np.add.at(U, self.inv, C)
        return U

    def diff(self, fa, fb):
        Ua, Ub = self.units(fa), self.units(fb)
        bench = 'prmbench' if self.c.bench == 'prmbench' else self.c.bench
        d = self.k * (E.official_from_counts(self.W @ Ua, bench) - E.official_from_counts(self.W @ Ub, bench))
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ----------------------------------------------------------------------------------------------- Part A driver
def part_a(bench, prm, arms=ARMS):
    rows, parity = [], []
    with common_set():
        s1 = S1.CellData(bench, prm)
        if bench == 'hard2verify': S2.PRM_ALONE[f'{bench}/{prm}'] = {'prm_exact': s1.ps_prm}
        t = S2.TeleCell().build(bench, prm)
    c = s1.c; bt = Boot(c); j = None
    if bench != 'hard2verify':
        j = list(map(str, SEARCH['stage1']['cells'])).index(f'{bench}/{prm}')
    for sname, S in SETS.items():
        mask = sum(1 << i for i in S)
        for version, cell, with_prm in (('+PRM', s1, True), ('alone', t, False)):
            res = {}
            for arm in arms:
                t0 = time.time(); o = fuse(cell, S, arm, with_prm); o['seconds'] = time.time() - t0
                res[arm] = o
            ref = res['DS_MAJ']
            if ref['valid']:
                f_ref = gapf(c, ref['fused'])
                ps_ref_fast = cell.metric(cell.flags(ref['fused']))
                if j is not None:
                    D = SEARCH['stage1' if with_prm else 'stage2']; i = int(np.flatnonzero(D['masks'] == mask)[0])
                    parity.append(dict(cell=f'{bench}/{prm}', set=sname, version=version, mine=ps_ref_fast,
                                       search=float(D['ps_fused'][i, j]), diff=abs(ps_ref_fast - float(D['ps_fused'][i, j]))))
                elif sname == 'SUB5':
                    prior = H2V_PRIOR[(prm, version)]; mine = official(c, f_ref)
                    parity.append(dict(cell=f'{bench}/{prm}', set=sname, version=version, mine=mine, search=prior, diff=abs(mine - prior)))
            for arm, o in res.items():
                r = dict(bench=bench, prm=prm, set=sname, version=version, arm=arm, K=o['K'], valid=bool(o['valid']),
                         reason=o.get('reason', ''), seconds=o['seconds'], converged=o.get('converged', ''))
                if o['valid']:
                    f = gapf(c, o['fused'])
                    r.update(ps=official(c, f), within=within(c, o['fused']), ps_mean=official(c, gapf(c, o['mean'])),
                             weights=json.dumps([round(x, 4) for x in o['w']]))
                    if arm != 'DS_MAJ' and ref['valid']:
                        r['d_vs_ref'] = r['ps'] - official(c, f_ref); r['ci_vs_ref'] = bt.diff(f, f_ref)
                rows.append(r)
            print(time.strftime('%H:%M:%S'), bench, prm, sname, version,
                  {a: (round(r_['ps'], 2) if r_.get('ps') is not None else r_['reason']) for a, r_ in
                   ((rr['arm'], rr) for rr in rows[-len(arms):])}, flush=True)
    return rows, parity


# ----------------------------------------------------------------------------------------------- Part B
def next_step(risk, off):
    """z(risk) at step k+1 of the same answer; the last step gets the answer mean of the others (0 after answer-z)."""
    out = np.zeros(len(risk))
    for a, b in zip(off[:-1], off[1:]):
        if b - a < 2: continue
        v = np.empty(b - a); v[:-1] = risk[a + 1:b]; v[-1] = v[:-1].mean(); out[a:b] = v
    return E.answer_z(out, off)


def part_b(bench, prm):
    c, VC, w, fused, info = MR.chain_inputs(bench, prm)
    zn = next_step(VC[:, 0], c.off)
    VN = np.column_stack([VC, zn])
    fused_n, dn, flip = MR.lsml(VN, groups=np.arange(VN.shape[1]))
    wn = np.asarray(dn['cross_weights'], float) * (-1. if flip else 1.)
    pos = MR.relative_position(c.off)
    diag = dict(next_corr_prm=float(np.corrcoef(zn, VC[:, 0])[0, 1]), next_corr_pos=float(np.corrcoef(zn, pos)[0, 1]),
                next_weight_share=float(abs(wn[-1]) / np.abs(wn).sum()), prm_share_chain=float(abs(w[0]) / np.abs(w).sum()),
                prm_share_next=float(abs(wn[0]) / np.abs(wn).sum()), next_weight=float(wn[-1]), K=info['K'])
    if bench == 'hard2verify':
        step_mask, _ = MR.eval_rows(c, bench)
        ce = E.load(bench, prm, uids_keep=COMMON)
        s_chain, s_next = fused[step_mask], fused_n[step_mask]
    else:
        ce, s_chain, s_next = c, fused, fused_n
    prm_arm = S1.prm_arm(bench, prm)
    fC, fN, fP = gapf(ce, s_chain), gapf(ce, s_next), gapf(ce, ce.arms[prm_arm])
    status = float(S1.STATUS[(bench, prm, 'CHAIN_N16')]['GAP']); mine = official(ce, fC)
    bt = Boot(ce)
    r = dict(bench=bench, prm=prm, chain_ps=mine, chain_status=status, next_ps=official(ce, fN), prm_ps=official(ce, fP),
             chain_within=within(ce, s_chain), next_within=within(ce, s_next), prm_within=within(ce, ce.arms[prm_arm]),
             ci_next_vs_chain=bt.diff(fN, fC), ci_next_vs_prm=bt.diff(fN, fP), **diag)
    r['d_next_vs_chain'] = r['next_ps'] - r['chain_ps']; r['d_next_vs_prm'] = r['next_ps'] - r['prm_ps']
    print(time.strftime('%H:%M:%S'), 'B', bench, prm, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    return r, dict(cell=f'{bench}/{prm}', mine=mine, status=status, diff=abs(mine - status))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def main(cmd):
    if cmd == 'time':
        t0 = time.time(); rows, par = part_a('prmbench', 'qwen_prm_7b')
        print('parity', par); print('seconds', time.time() - t0)
        t0 = time.time(); r, p = part_b('prmbench', 'qwen_prm_7b'); print('B parity', p, 'seconds', time.time() - t0)
        return
    A_rows, B_rows, PA, PB = [], [], [], []
    for bench in BENCHES:
        for prm in PRMS:
            r, p = part_b(bench, prm); B_rows.append(r); PB.append(p)
            if p['diff'] > 1e-6: raise SystemExit(f'PARITY B FAIL {p}')
            rows, par = part_a(bench, prm); A_rows += rows; PA += par
            bad = [x for x in par if x['diff'] > 1e-9]
            if bad: raise SystemExit(f'PARITY A FAIL {bad}')
            (HERE / 'PARITY.json').write_text(json.dumps(dict(A=PA, B=PB), indent=1))
            (HERE / 'RESULTS.json').write_text(json.dumps(dict(A=A_rows, B=B_rows), indent=1, default=float))
    write_csv(HERE / 'RESULTS_A.csv', A_rows); write_csv(HERE / 'RESULTS_B.csv', B_rows)
    print('done')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'run')
