"""Stage 7, hygiene for the math peer review (W7, W11, W12), pre-registered in HYGIENE_PREREG.json /
docs/research_notes/math_rigor_2026-10-24/HYGIENE_PREREG.md. Unlabeled fitting throughout; labels only score.

  W11  refit bootstrap: resample source groups (external: answers) with replacement, refit the whole CHAIN_N16
       (telemetry partition + group weights + cross weights + sign) on the resampled rows, apply the frozen refit to
       every answer, weight per-answer metrics by the draw's multiplicities. Compared with the fixed-score bootstrap
       on the same draws.
  W7   partition stability: ARI of each refit's telemetry partition with the full-data partition, and K.
  W12  split-half inductive fit: fit CHAIN on half A of the source groups (stage 4's length-stratified split), apply
       frozen to half B; compare with the exported transductive CHAIN on B and with the PRM on B.

    COLLECTION_DATA=<data root> python stage7_hygiene.py <out_dir> [--probe]
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from stage4_calibrated_clean import split  # noqa: E402

E = K.E
HERE = Path(__file__).resolve().parent
PRE = json.loads((HERE / 'HYGIENE_PREREG.json').read_text()) if (HERE / 'HYGIENE_PREREG.json').exists() else {}


# ------------------------------------------------------------------------------------------- frozen CHAIN fit/apply
def _fit(X, groups=None, anchor=None):
    s, d = K.lsml_continuous_guarded(*[X[:, j] for j in range(X.shape[1])], groups=groups)
    s = np.asarray(s, float); ref = X[:, 0] if anchor is None else anchor
    sign = -1. if np.corrcoef(s, ref)[0, 1] < 0 else 1.
    return dict(gw=[(np.asarray(i, int), np.asarray(w, float)) for i, w in d['group_weights']],
                cw=np.asarray(d['cross_weights'], float), sign=sign, K=int(d['K']), c=np.asarray(d['c'], int))


def _virtual(X, p):
    return np.column_stack([X[:, i[0]].astype(float) if len(i) == 1 else X[:, i] @ w for i, w in p['gw']])


def _apply(X, p):
    vg = _virtual(X, p)
    if vg.shape[1] == 1:
        i, w = p['gw'][0]
        s = X[:, i] @ w if len(i) > 1 else vg[:, 0]
    else:
        s = vg @ p['cw']
    return p['sign'] * s


def fit_chain(V, zp, off, rows, off_rows):
    """CHAIN_N16 fitted on V[rows] (answer offsets off_rows within rows); returns (tele params, cross params)."""
    Vr = V[rows]
    pt = _fit(Vr, anchor=Vr.mean(1))
    vg = _virtual(Vr, pt)
    VC = np.column_stack([zp[rows]] + [E.answer_z(vg[:, g], off_rows) for g in range(vg.shape[1])])
    pc = _fit(VC, groups=np.arange(VC.shape[1]))
    return pt, pc


def apply_chain(V, zp, off, pt, pc):
    vg = _virtual(V, pt)
    VC = np.column_stack([zp] + [E.answer_z(vg[:, g], off) for g in range(vg.shape[1])])
    return _apply(VC, pc)


def rows_of(off, answers):
    """Row indices and offsets of the concatenation of `answers` (repeats allowed)."""
    n = np.diff(off)[answers]
    rows = np.concatenate([np.arange(off[a], off[a + 1]) for a in answers])
    return rows, np.concatenate([[0], np.cumsum(n)])


# ------------------------------------------------------------------------------------------- cell inputs
def cell(bench, prm):
    c = E.load(bench, prm)
    V = E.channels(c, 'N16')
    zp = E.answer_z(c.arms[K.prm_arm(c)], c.off)
    _, ans_eval = K.eval_rows(c, bench)
    return c, V, zp, ans_eval


def gap_flags(c, s):
    z = E.answer_z(s, c.off)
    return E.topk_flags(z, c.off, E.gap_counts(z, c.off))


def per_answer(c, s, ans_eval):
    """Per-answer within-AUC (nan if one class) and per-answer official counts of GAP flags, eval answers only."""
    keep = c.keep if c.bench != 'prmbench' else None
    wa = E.per_answer_auc(c.err, s, c.off, keep); wa[~ans_eval] = np.nan
    C = E.counts_by_answer(c, gap_flags(c, s)) * ans_eval[:, None]
    return wa, C


def weighted(wa, C, w, bench):
    fin = np.isfinite(wa); ww = w * fin
    auc = float(np.sum(np.where(fin, wa, 0) * ww) / ww.sum())
    off = float(E.official_from_counts((C * w[:, None]).sum(0), bench)) * (1. if bench == 'prmbench' else 100.)
    return auc, off


def probe():
    t = time.time(); c, V, zp, ans_eval = cell('prmbench', 'qwen_prm_7b'); t_load = time.time() - t
    t = time.time(); pt, pc = fit_chain(V, zp, c.off, np.arange(c.off[-1]), c.off); t_fit = time.time() - t
    s = apply_chain(V, zp, c.off, pt, pc)
    d = float(np.max(np.abs(s - c.arms['CHAIN_N16'])))
    print(json.dumps({'load_s': t_load, 'fit_s': t_fit, 'max_abs_vs_exported': d, 'K': pt['K']}))


def ci(x):
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def run_cell(bench, prm):
    c, V, zp, ans_eval = cell(bench, prm)
    n_ans = len(c.ids); allrows = np.arange(c.off[-1])
    pt0, pc0 = fit_chain(V, zp, c.off, allrows, c.off)
    s_exp = c.arms['CHAIN_N16']; s_prm = c.arms[K.prm_arm(c)]
    assert np.max(np.abs(apply_chain(V, zp, c.off, pt0, pc0) - s_exp)) < 1e-5, 'parity'
    wa_ch, C_ch = per_answer(c, s_exp, ans_eval); wa_p, C_p = per_answer(c, s_prm, ans_eval)

    # W11 + W7: refit bootstrap over source groups (external: answers)
    u, inv = np.unique(c.groups.astype(str), return_inverse=True)
    rng = np.random.default_rng(PRE['boot_seed'])
    d_auc_fix, d_auc_ref, d_off_fix, d_off_ref, ari, ks = [], [], [], [], [], []
    for _ in range(PRE['refit_draws']):
        w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)))[inv].astype(float)
        rows, off_r = rows_of(c.off, np.repeat(np.arange(n_ans), w.astype(int)))
        pt, pc = fit_chain(V, zp, c.off, rows, off_r)
        wa_r, C_r = per_answer(c, apply_chain(V, zp, c.off, pt, pc), ans_eval)
        a_p, o_p = weighted(wa_p, C_p, w, bench)
        a_f, o_f = weighted(wa_ch, C_ch, w, bench); a_r, o_r = weighted(wa_r, C_r, w, bench)
        d_auc_fix.append(a_f - a_p); d_auc_ref.append(a_r - a_p); d_off_fix.append(o_f - o_p); d_off_ref.append(o_r - o_p)
        ari.append(adjusted_rand_score(pt0['c'], pt['c'])); ks.append(pt['K'])
    one = np.ones(n_ans)
    a_ch, o_ch = weighted(wa_ch, C_ch, one, bench); a_p0, o_p0 = weighted(wa_p, C_p, one, bench)
    kv, kc = np.unique(ks, return_counts=True)
    boot = {'auc_delta': a_ch - a_p0, 'auc_ci_fixed': ci(d_auc_fix), 'auc_ci_refit': ci(d_auc_ref),
            'official_delta': o_ch - o_p0, 'official_ci_fixed': ci(d_off_fix), 'official_ci_refit': ci(d_off_ref),
            'auc_width_ratio': float(np.diff(ci(d_auc_ref))[0] / np.diff(ci(d_auc_fix))[0]),
            'official_width_ratio': float(np.diff(ci(d_off_ref))[0] / max(np.diff(ci(d_off_fix))[0], 1e-12))}
    stab = {'K_full': pt0['K'], 'ari_median': float(np.median(ari)), 'ari_q10': float(np.quantile(ari, .1)),
            'K_counts': {int(k): int(n) for k, n in zip(kv, kc)}, 'modal_K': int(kv[np.argmax(kc)]),
            'modal_K_share': float(kc.max() / len(ks)), 'partition_full': pt0['c'].tolist()}

    # W12: split-half inductive fit
    splits = []
    for seed in range(PRE['split_seeds']):
        A = split(c, PRE['split_seed0'] + seed); B = (~A & ans_eval).astype(float)
        rows, off_r = rows_of(c.off, np.flatnonzero(A))
        pt, pc = fit_chain(V, zp, c.off, rows, off_r)
        wa_i, C_i = per_answer(c, apply_chain(V, zp, c.off, pt, pc), ans_eval)
        a_i, o_i = weighted(wa_i, C_i, B, bench); a_t, o_t = weighted(wa_ch, C_ch, B, bench)
        a_q, o_q = weighted(wa_p, C_p, B, bench)
        splits.append({'seed': seed, 'auc_ind': a_i, 'auc_trans': a_t, 'auc_prm': a_q,
                       'off_ind': o_i, 'off_trans': o_t, 'off_prm': o_q, 'K_A': pt['K'],
                       'ari_A': float(adjusted_rand_score(pt0['c'], pt['c']))})
    med = lambda k: float(np.median([s[k] for s in splits]))
    half = {'med_auc_ind_minus_trans': float(np.median([s['auc_ind'] - s['auc_trans'] for s in splits])),
            'med_off_ind_minus_trans': float(np.median([s['off_ind'] - s['off_trans'] for s in splits])),
            'med_auc_ind_minus_prm': float(np.median([s['auc_ind'] - s['auc_prm'] for s in splits])),
            'med_auc_trans_minus_prm': float(np.median([s['auc_trans'] - s['auc_prm'] for s in splits])),
            'med_auc_ind': med('auc_ind'), 'med_auc_trans': med('auc_trans'), 'med_auc_prm': med('auc_prm'),
            'seeds': splits}
    return {'bench': bench, 'prm': prm, 'n_answers': n_ans, 'n_groups': len(u), 'n_eval': int(ans_eval.sum()),
            'refit_bootstrap': boot, 'partition_stability': stab, 'split_half': half}


def criteria(r):
    b, s, h = r['refit_bootstrap'], r['partition_stability'], r['split_half']
    fx, rf = b['auc_ci_fixed'], b['auc_ci_refit']
    sgn = lambda lo, hi: 1 if lo > 0 else (-1 if hi < 0 else 0)
    e1 = sgn(*fx) == sgn(*rf)
    e2 = s['ari_median'] >= PRE['e2_ari_min'] and s['modal_K_share'] >= PRE['e2_modal_k_min']
    e3 = (h['med_auc_ind_minus_trans'] >= PRE['e3_auc_margin'] and h['med_off_ind_minus_trans'] >= PRE['e3_official_margin']
          and np.sign(h['med_auc_ind_minus_prm']) == np.sign(h['med_auc_trans_minus_prm']))
    return {'e1': bool(e1), 'e2': bool(e2), 'e3': bool(e3)}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); res = []
    for bench in K.BENCHES:
        for prm in E.PRMS:
            t = time.time(); r = run_cell(bench, prm); r['criteria'] = criteria(r); r['seconds'] = time.time() - t
            res.append(r); print(bench, prm, json.dumps(r['criteria']), f"{r['seconds']:.0f}s", flush=True)
            (out / 'HYGIENE_RESULTS.json').write_text(json.dumps(res, indent=1))
    pb = [r for r in res if r['bench'] == 'prmbench']
    n = {k: sum(r['criteria'][k] for r in pb) for k in ('e1', 'e2', 'e3')}
    met = [n['e1'] >= PRE['e1_min_prms'], n['e2'] >= PRE['e2_min_prms'], n['e3'] >= PRE['e3_min_prms']]
    verdict = 'pass' if all(met) else ('partial' if any(met) else 'fail')
    (out / 'HYGIENE_VERDICT.json').write_text(json.dumps({'prmbench_prms_meeting': n, 'endpoints_met': met,
                                                           'verdict': verdict}, indent=1))
    print('verdict', verdict, n)


if __name__ == '__main__':
    if '--probe' in sys.argv:
        probe()
    else:
        main(sys.argv[1])
