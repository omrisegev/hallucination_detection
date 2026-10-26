"""Descriptive 8-cell evaluation pre-declared in PREDECLARATION_8CELL_EVAL.md (2026-10-26; CPU; in-sample).

    python eval8.py cache    # POS light caches (the fork's position probe procedure), stage-1 and stage-2 variants
    python eval8.py rule     # label-free rule R on the POS caches -> RULE_POS.json (the set S_R)
    python eval8.py run      # every declared arm + Jaffe-assumption diagnostics -> EVAL8_*.csv, scores/
    python eval8.py comp     # competitor protocols on PRMBench (PRISM, CROP, ARES, SLIDER) -> EVAL8_COMPETITORS.csv

Cells: PRMBench and Socratic x {qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm}; Math-Shepherd dropped by Omri's
decision (2026-10-26). Labels enter only the metrics and the D1-D2 diagnostics.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
VS = HERE.parent
ROOT = VS.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(VS / 'stage1')); sys.path.insert(0, str(VS / 'stage2'))
import stage1 as S1  # noqa: E402
import stage2 as S2  # noqa: E402
from spectral_utils import prm_anchored_lsml as A  # noqa: E402
from spectral_utils import binary_view_estimation as B  # noqa: E402
E = S1.E

PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
CELLS = [(b, p) for b in ('prmbench', 'socratic') for p in PRMS]
NAMES = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
         'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
         'digit_spread', 'digit_alternative_innovation']
FAMILY = {'uncertainty': ['q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level'],
          'realized': ['chosen_surprisal', 'realized_z', 'realized_drv'],
          'dynamics': ['energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0'],
          'digit': ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']}
FAM_IDX = np.array([next(i for i, v in enumerate(FAMILY.values()) if n in v) for n in NAMES])
C3 = ['q15_H1', 'chosen_surprisal', 'digit_spread']
SUB5 = ['q15_H1', 'chosen_surprisal', 'top15_turnover', 'dominant_freq16', 'bocpd_p0', 'digit_spread']
POSC = HERE / 'cache_pos'; SCORES = HERE / 'scores'
key = lambda b, p: f'{b}__{p}'


# ------------------------------------------------------------------------------------------------ representations
def rel_pos(off):
    n = np.diff(off)
    return np.concatenate([(np.arange(k) + .5) / k for k in n])


def residual_cell(b, p):
    """Verbatim logic of position_probe/probe.py:residual_cell (channels residualized on a cubic in relative position,
    pooled least squares on the pool, PRM untouched, channels re-oriented against the PRM, GAP votes recomputed)."""
    o = S1.CellData(b, p); c = o.c
    n = np.diff(c.off); pos = rel_pos(c.off)
    Bm = np.column_stack([np.ones_like(pos), pos, pos ** 2, pos ** 3])
    X = o.Z * o.sign[None, :]
    beta = np.linalg.lstsq(Bm[o.pool], X[o.pool, 1:], rcond=None)[0]
    X[:, 1:] = X[:, 1:] - Bm @ beta
    sign = np.sign([1.] + [np.corrcoef(X[o.pool, 0], X[o.pool, k])[0, 1] for k in range(1, 17)]); sign[sign == 0] = 1
    X = X * sign
    o.Z = np.column_stack([E.answer_z(X[:, k], c.off) for k in range(17)]); o.Z[:, 0] = X[:, 0]
    sd = np.array([[X[a:b2, k].std() for k in range(17)] for a, b2 in zip(c.off[:-1], c.off[1:])])
    const = np.repeat(sd < 1e-8, n, axis=0); V = np.empty_like(o.Z)
    for k in range(17):
        f = E.topk_flags(o.Z[:, k], c.off, E.gap_counts(o.Z[:, k], c.off)); f[const[:, k]] = False; V[:, k] = np.where(f, 1., -1.)
    o.V = V; o.Vp = V[o.pool]; o.R = np.cov(np.ascontiguousarray(o.Vp).T); o.sign = sign   # sign relative to stored values
    return o


def residual_tele(b, p):
    """Telemetry-only POS: stage-2 values (stored orientation) residualized the same way; no PRM anywhere."""
    t = S2.TeleCell().build(b, p); off = t.off; n = np.diff(off); pos = rel_pos(off)
    Bm = np.column_stack([np.ones_like(pos), pos, pos ** 2, pos ** 3])
    beta = np.linalg.lstsq(Bm[t.pool], t.Z[t.pool], rcond=None)[0]
    X = t.Z - Bm @ beta
    t.Z = np.column_stack([E.answer_z(X[:, k], off) for k in range(16)])
    sd = np.array([[X[a:b2, k].std() for k in range(16)] for a, b2 in zip(off[:-1], off[1:])])
    const = np.repeat(sd < 1e-8, n, axis=0); V = np.empty_like(t.Z)
    for k in range(16):
        f = E.topk_flags(t.Z[:, k], off, E.gap_counts(t.Z[:, k], off)); f[const[:, k]] = False; V[:, k] = np.where(f, 1., -1.)
    t.V = V; t.Vp = V[t.pool]; t.R = np.cov(np.ascontiguousarray(t.Vp).T)
    return t


def cmd_cache():
    POSC.mkdir(exist_ok=True)
    for b, p in CELLS:
        o = residual_cell(b, p); o.save_light(POSC / f's1_{key(b, p)}.npz'); del o
        t = residual_tele(b, p); t.save(POSC / f's2_{key(b, p)}.npz'); del t
        print('cached', b, p, flush=True)


def load_cells(b, p):
    return dict(RAW1=S1.CellData.from_light(VS / 'stage1' / 'cache' / f'{key(b, p)}.npz'),
                POS1=S1.CellData.from_light(POSC / f's1_{key(b, p)}.npz'),
                RAW2=S2.TeleCell.load(VS / 'stage2' / 'cache' / f'{key(b, p)}.npz'),
                POS2=S2.TeleCell.load(POSC / f's2_{key(b, p)}.npz'))


# ------------------------------------------------------------------------------------------------ rule R
def cmd_rule():
    rows, cov = [], {}
    for b, p in CELLS:
        o = S1.CellData.from_light(POSC / f's1_{key(b, p)}.npz')
        cov[key(b, p)] = dict(sign=o.sign[1:].tolist(), r0k=o.R[0, 1:].tolist())
        K, g = A.groups_from_cov(o.R[1:, 1:])
        iu = np.triu_indices(16, 1); sa, sb = (FAM_IDX[:, None] == FAM_IDX[None, :])[iu], (g[:, None] == g[None, :])[iu]
        rows.append(dict(cell=key(b, p), K=int(K), rand_vs_families=float((sa == sb).mean()),
                         groups=' | '.join('+'.join(NAMES[i] for i in range(16) if g[i] == j) for j in range(K))))
    audited = [k for k in range(16) if all(cov[c]['sign'][k] > 0 for c in cov)]
    med = {NAMES[k]: float(np.median([cov[c]['r0k'][k] for c in cov])) for k in range(16)}
    picks = {}
    for f, members in FAMILY.items():
        ok = [n for n in members if NAMES.index(n) in audited]
        if ok: picks[f] = max(ok, key=lambda n: med[n])
    out = dict(predeclaration_sha256=hashlib.sha256((HERE / 'PREDECLARATION_8CELL_EVAL.md').read_bytes()).hexdigest(),
               audited=[NAMES[k] for k in audited], excluded_by_audit=[NAMES[k] for k in range(16) if k not in audited],
               median_r0k=med, picks=picks, S_R=list(picks.values()), identifiable_alone=len(picks) >= 3,
               per_cell=cov, lsml_groups_pos=rows)
    (HERE / 'RULE_POS.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('audited', 'excluded_by_audit', 'picks', 'identifiable_alone')}, indent=1))
    for r in rows: print(r['cell'], 'K', r['K'], 'rand', round(r['rand_vs_families'], 2))


# ------------------------------------------------------------------------------------------------ fusions
def eval_ci(cell, S, with_prm, eps=1e-6):
    """F_CI: every channel its own group (Dawid-Skene, K = m), DS-EM between, weights on continuous answer-z."""
    idx = np.r_[0, np.asarray(S) + 1] if with_prm else np.asarray(S)
    ds = A.ds_em_votes(cell.Vp[:, idx], anchor=0 if with_prm else None, eps=eps)
    w = ds['w']
    if w.sum() <= 0:
        return dict(valid=False)
    Zs = cell.Z[:, idx]; mean = Zs.mean(1)
    fused = Zs @ w
    fused_bin = cell.V[:, idx] @ w + 1e-6 * (cell.Z[:, 0] if with_prm else mean)
    return dict(valid=True, fused=fused, mean=mean, fused_bin=fused_bin, share=(w / w.sum()).tolist(),
                prm_share=float(w[0] / w.sum()) if with_prm else np.nan, iters=ds['iters'], n_clip=int((ds['w_raw'] < 0).sum()))


def eval_lsml(cell, S, with_prm):
    o = cell.evaluate(S, want_detail=True)
    if not o['valid']:
        return dict(valid=False, K=o['K'])
    if with_prm:
        wb = o['between']['w']; share = float(wb[0] / wb.sum())
    else:
        share = np.nan
    return dict(valid=True, fused=o['fused'], mean=o['mean'], fused_bin=o['fused_bin'], prm_share=share, K=o['K'],
                groups=' | '.join('+'.join(NAMES[k] for k in np.asarray(S)[o['groups'] == g]) for g in range(o['K'])))


# ------------------------------------------------------------------------------------------------ metrics
def exact_flags(s, off):
    z = E.answer_z(s, off)
    return E.topk_flags(z, off, E.gap_counts(z, off))


def metrics(c, s, bench):
    f = exact_flags(s, c.off)
    C = E.counts_by_answer(c, f)
    ps = float(E.official_from_counts(C.sum(0), 'prmbench')) if bench == 'prmbench' else 100 * float(E.official(c, f)[0])
    keep = None if bench == 'prmbench' else c.keep
    r = E.rates(C)
    out = dict(PRMScore=ps, within_auc=float(np.nanmean(E.per_answer_auc(c.err, s, c.off, keep))), TPR=r['TPR'], FPR=r['FPR'],
               flag_rate=float(f[c.keep].mean()))
    if bench == 'prmbench':
        out.update({f'cat_{k}': v for k, v in E.category_scores(c, f).items()})
    return out


# ------------------------------------------------------------------------------------------------ Jaffe diagnostics
def diagnostics(cell, sets):
    """D1 balanced accuracy of every vote; D2 mean |within-class correlation| by pair type; D3 rank-one misfit."""
    V = cell.V; pool = cell.pool
    err = cell._nyv_m & pool; ok = cell._yv_m & pool
    bal = 0.5 * ((V[err] > 0).mean(0) + (V[ok] < 0).mean(0))
    fam = np.r_[-1, FAM_IDX]                                 # -1 = the PRM
    iu = np.triu_indices(17, 1)
    typ = np.where((fam[iu[0]] == -1) | (fam[iu[1]] == -1), 'prm_channel',
                   np.where(fam[iu[0]] == fam[iu[1]], 'within_family', 'cross_family'))
    out = dict(n_below_half=int((bal[1:] <= .5).sum()), below_half=[NAMES[k] for k in range(16) if bal[k + 1] <= .5],
               bal_prm=float(bal[0]))
    for cls_name, m in (('valid', ok), ('error', err)):
        Cc = np.corrcoef(V[m].T)[iu]
        for t in ('prm_channel', 'cross_family', 'within_family'):
            out[f'abscorr_{t}_{cls_name}'] = float(np.nanmean(np.abs(Cc[typ == t])))
    for t in ('prm_channel', 'cross_family', 'within_family'):
        out[f'abscorr_{t}'] = 0.5 * (out[f'abscorr_{t}_valid'] + out[f'abscorr_{t}_error'])
    for nm, S in sets.items():
        idx = np.r_[0, [NAMES.index(n) + 1 for n in S]]
        Sm = np.cov(cell.Vp[:, idx].T)
        out[f'misfit_{nm}+PRM'] = float(B.rank_one_misfit(Sm, B.rank_one_offdiag(Sm))['overall'])
        Sm2 = Sm[1:, 1:]
        out[f'misfit_{nm}_alone'] = float(B.rank_one_misfit(Sm2, B.rank_one_offdiag(Sm2))['overall']) if len(S) >= 3 else np.nan
    return out, bal


def cmd_run():
    rule = json.loads((HERE / 'RULE_POS.json').read_text()); SR = rule['S_R']
    sets = {'S_R': SR, 'C3': C3, 'SUB5': SUB5}
    SCORES.mkdir(exist_ok=True)
    rows, diag_rows, share_rows = [], [], []
    for b, p in CELLS:
        cells = load_cells(b, p); c = E.load(b, p)
        sc = {}
        # references
        sc['PRM'] = cells['RAW1'].Z[:, 0]
        sc['POSITION'] = rel_pos(c.off)
        for rep in ('RAW', 'POS'):
            c1, c2 = cells[rep + '1'], cells[rep + '2']
            for nm, Sn in sets.items():
                S = [NAMES.index(n) for n in Sn]
                for mode, cell in (('+PRM', c1), ('alone', c2)):
                    with_prm = mode == '+PRM'
                    o = eval_lsml(cell, S, with_prm) if nm == 'SUB5' else eval_ci(cell, S, with_prm)
                    tag = f'{nm}|{rep}|{mode}'
                    if not o['valid']:
                        rows.append(dict(cell=key(b, p), arm=tag, valid=False)); continue
                    sc[tag] = o['fused']; sc[tag + '|mean'] = o['mean']; sc[tag + '|bin'] = o['fused_bin']
                    share_rows.append(dict(cell=key(b, p), arm=tag, prm_share=o['prm_share'], groups=o.get('groups', 'singletons'),
                                           share=json.dumps(o.get('share', None))))
                    if nm != 'SUB5':                                  # pre-declared floor sensitivity for F_CI
                        o4 = eval_ci(cell, S, with_prm, eps=1e-4)
                        if o4['valid']: sc[tag + '|floor1e-4'] = o4['fused']
            d, bal = diagnostics(c1, sets)
            diag_rows.append(dict(cell=key(b, p), rep=rep, **d))
        np.savez(SCORES / f'{key(b, p)}.npz', **{k.replace('|', '__').replace('+', 'P'): v for k, v in sc.items()},
                 names=np.array(list(sc)))
        for arm, s in sc.items():
            m = metrics(c, s, b)
            rows.append(dict(cell=key(b, p), arm=arm, valid=True, **m))
        # parity: the fast metric of the light cell equals the exact evaluator metric for the stage-1 PRM-alone comparator
        rows.append(dict(cell=key(b, p), arm='PRM_parity_ps_prm', valid=True, PRMScore=float(cells['RAW1'].ps_prm)))
        print('done', b, p, flush=True)
        del c, cells
    E.write_csv(HERE / 'EVAL8_ARMS.csv', rows); E.write_csv(HERE / 'EVAL8_DIAGNOSTICS.csv', diag_rows)
    E.write_csv(HERE / 'EVAL8_SHARES.csv', share_rows)


# ------------------------------------------------------------------------------------------------ competitor protocols
def cmd_comp():
    import competitors as CP
    rule = json.loads((HERE / 'RULE_POS.json').read_text())
    arms_want = [f'{nm}|{rep}|{mode}' for nm in ('S_R', 'C3') for rep in ('RAW', 'POS') for mode in ('+PRM', 'alone')]
    arms_want += ['SUB5|POS|+PRM', 'SUB5|POS|alone', 'SUB5|RAW|+PRM', 'SUB5|RAW|alone']
    rows = []
    for prm in PRMS:
        c = E.load('prmbench', prm); s_c = E.load('socratic', prm)
        for cc, b in ((c, 'prmbench'), (s_c, 'socratic')):
            D = np.load(SCORES / f'{key(b, prm)}.npz'); names = list(D['names'])
            for a in arms_want:
                if a in names: cc.arms[a] = D[a.replace('|', '__').replace('+', 'P')]
        arms = [a for a in arms_want if a in c.arms]
        scale = CP.CC.best_scale(c)
        rows += CP.prism_rows(c, [scale, 'CHAIN_N16', 'TELE_N16'] + arms)
        rows += CP.crop_rows(c, ['PRM_native', 'CHAIN_N16', 'TELE_N16'] + arms, s_c)
        rows += CP.ares_rows(c, ['PRM_native', 'CHAIN_N16', 'TELE_N16'] + arms)
        rows += CP.slider_rows(c, [scale, 'CHAIN_N16', 'TELE_N16'] + arms)
        print('comp done', prm, flush=True); del c, s_c
    for r in rows: r['S_R'] = '+'.join(rule['S_R'])
    E.write_csv(HERE / 'EVAL8_COMPETITORS.csv', rows)


if __name__ == '__main__':
    {'cache': cmd_cache, 'rule': cmd_rule, 'run': cmd_run, 'comp': cmd_comp}[sys.argv[1]]()
