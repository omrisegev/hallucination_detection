"""answer_gate_v1: answer-level label-free detectors decide whether and how many steps to flag.
Protocol results/answer_gate_v1/PROTOCOL.json (frozen f745f1834). Step scores (S_equal) and their within-answer ranking are
frozen; only the per-answer flag count changes. Labels enter only the evaluation block.

    python -B scripts/experiments/answer_gate_run.py [run_id] [--draws N] [--shuffles N]
"""
import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; SSL = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
sys.path.insert(0, str(ROOT))
import hashlib  # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402  (byte-identical to the depth-feature-fusion-v1 copy)
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)


def ratio(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b != 0)


def prm_parts(c):
    """c[..., 4] = TP, FP, TN, FN with positive = correct step kept valid (official convention)."""
    tp, fp, tn, fn = np.moveaxis(np.asarray(c, float), -1, 0)
    f1 = ratio(2 * tp, 2 * tp + fp + fn); f1e = ratio(2 * tn, 2 * tn + fn + fp)
    return {'prmscore': (f1 + f1e) / 2, 'f1_correct': f1, 'f1_error': f1e, 'recall_correct': ratio(tp, tp + fn), 'recall_error': ratio(tn, tn + fp),
            'precision_correct': ratio(tp, tp + fp), 'precision_error': ratio(tn, tn + fn), 'flag_rate': ratio(tn + fn, tp + fp + tn + fn)}

STAGE = ROOT / 'results/answer_gate_v1'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
INPUTS = {'features': STAGE / 'ANSWER_FEATURES.npz',
          'step_scores': SSL / 'results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz',
          'thresholds': SSL / 'results/expectation_realization_v1/run_20260927_stage_b_thr/THRESHOLDS.json',
          'dr_decisions': ROOT / 'results/decision_rule_v1/run_20260929/DECISIONS.npz',
          'level_bank': MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz',
          'ct7_profiles': MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy',
          'ct7_profile_validation': MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/PROFILE_VALIDATION.json',
          'oof_answers': R / 'OOF_ANSWERS.csv', 'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'input_freeze': R / 'INPUT_FREEZE.json'}
FIT = dict(loss='l2', exclusion=True, difficulty_gate=False, simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)          # scripts/labelfree_standing_report.py, verbatim
GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
DETECTORS = ['D1_upcr_full', 'D2_lsml_cont_good5', 'D3_lsml_full', 'D4_equal_full', 'D5_epr', 'D6_length']
PRIMARY_DET = ['D1_upcr_full', 'D2_lsml_cont_good5']
DROPPED = ['energy_innovation', 'top50_js']


def lsml_linear(Xfit):
    """continuous L-SML on the columns of Xfit; returns the equivalent linear weight vector and the fused fit score."""
    fused, meta = lsml_continuous(*[Xfit[:, j] for j in range(Xfit.shape[1])])
    W = np.zeros(Xfit.shape[1]); gw = meta['group_weights']; cw = np.asarray(meta['cross_weights'], float)
    if len(gw) == 1:
        idx, w = gw[0]; W[idx] = w
    else:
        for g, (idx, w) in enumerate(gw): W[idx] += cw[g] * np.asarray(w, float)
    return W, np.asarray(fused, float), meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run_id', nargs='?', default='run_20260930')
    ap.add_argument('--draws', type=int, default=10_000); ap.add_argument('--shuffles', type=int, default=20)
    args = ap.parse_args(); OUT = STAGE / args.run_id
    if (OUT / 'RUN_STATUS.json').exists() and json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')).get('status') == 'COMPLETE':
        raise SystemExit(f'{OUT} already holds a finished run; pass a new run id')
    OUT.mkdir(parents=True, exist_ok=True); T0 = time.perf_counter(); timing = {}; gates = {}
    status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'args': vars(args)}; dump(OUT / 'RUN_STATUS.json', status)

    def stop(msg):
        status.update({'status': 'STOPPED', 'reason': msg}); dump(OUT / 'RUN_STATUS.json', status); dump(OUT / 'GATES.json', gates); raise SystemExit('STOP: ' + msg)

    # ------------------------------------------------------------------ population and frozen inputs
    ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
    off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S_ = int(off[-1])
    pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); ids = ans.id.to_numpy(); groups = ans.source_group.to_numpy()
    cells = ans.cell.to_numpy(); target = ans.target.to_numpy()
    meta_path = Path(json.loads(INPUTS['input_freeze'].read_text(encoding='utf-8-sig'))['prm_metadata']['path'])
    meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
    cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
    control = prm & (cls == 'correct'); noncontrol = prm & ~control
    has_err = np.array([labels[off[i]:off[i + 1]].any() for i in range(n)]) & prm
    ms = noncontrol & (cls == 'multi_solutions'); err_nc = has_err & noncontrol
    aid = np.repeat(np.arange(n), ns); step_fold = fold[aid]; prm_step = prm[aid]; pb_step = pb[aid]
    if (int(prm.sum()), int(control.sum()), int(noncontrol.sum()), int(err_nc.sum()), int(ms.sum())) != (6969, 758, 6211, 6035, 160): stop('population')
    F = np.load(INPUTS['features']); X = F['X'].astype(float); names = [str(x) for x in F['names']]
    if not (np.array_equal(F['ids'], ids.astype(str)) and np.array_equal(F['folds'], fold)): stop('feature rows not aligned with OOF_ANSWERS')
    S = np.load(INPUTS['step_scores'])['S_equal'].astype(float)
    tauS = {int(k): float(v) for k, v in json.loads(INPUTS['thresholds'].read_text(encoding='utf8'))['S_equal'].items()}
    zS = np.empty(S_)
    for i in range(n): zS[off[i]:off[i + 1]] = zt(S[off[i]:off[i + 1]])
    DR = np.load(INPUTS['dr_decisions'])

    def top_by_S(counts):
        f = np.zeros(S_, bool)
        for i in range(n):
            k = int(min(counts[i], ns[i]))
            if k > 0: a = off[i]; f[a + np.argsort(-S[a:off[i + 1]], kind='stable')[:k]] = True
        return f

    # ------------------------------------------------------------------ references R0, R2, R2pb
    flags = {}
    R0 = np.zeros(S_, bool)
    for k in range(5): R0[step_fold == k] = zS[step_fold == k] >= tauS[k]
    flags['R0_frozen'] = R0; flags['R2_allocate'] = DR['R2_allocate'].astype(bool)
    gates['R0_equals_decision_rule_v1'] = bool(np.array_equal(R0, DR['R0_frozen']))
    # R2pb: R2 with tau_G calibrated per benchmark (G recomputed exactly as decision_rule_v1)
    lv = np.load(INPUTS['level_bank']); level = lv['level'].astype(float); n11 = list(map(str, lv['channels']))
    prof = np.load(INPUTS['ct7_profiles']).astype(float); pn = json.loads(INPUTS['ct7_profile_validation'].read_text(encoding='utf8'))['channels']
    raw = np.column_stack([level, prof[:, pn.index('chosen_token_z_despiked')], lv['derivative'][:, n11.index('chosen_surprisal')]]); rn = n11 + ['realized_z', 'realized_drv']
    Xs = raw[:, [j for j, c in enumerate(rn) if c not in DROPPED]]
    G = np.full(S_, np.nan); cG = np.zeros(n)
    for k in range(5):
        c = (k + 1) % 5; fitm = ~np.isin(step_fold, [k, c]); evm = step_fold == k
        mu = Xs[fitm].mean(0); sd = np.maximum(Xs[fitm].std(0), 1e-12); g = ((Xs - mu) / sd).mean(1); G[evm] = g[evm]
        for b, bm in (('prm', prm_step), ('pb', pb_step)):
            t = float(np.quantile(g[bm & (step_fold == c)], .8)); sel = evm & bm
            cG += np.bincount(aid[sel], weights=(g[sel] >= t), minlength=n)
    gates['G_recompute_vs_saved_max_abs'] = float(np.max(np.abs(G - DR['G'].astype(float))))
    flags['R2pb_allocate'] = top_by_S(cG)
    gates['R2pb_equals_R2_on_prmbench'] = bool(np.array_equal(flags['R2pb_allocate'][prm_step], flags['R2_allocate'][prm_step]))
    if not (gates['R0_equals_decision_rule_v1'] and gates['G_recompute_vs_saved_max_abs'] < 1e-4 and gates['R2pb_equals_R2_on_prmbench']): stop('references')

    # ------------------------------------------------------------------ detectors: per cell and fold, fit on fit folds
    t = time.perf_counter(); A = {d: np.full(n, np.nan) for d in DETECTORS}          # evaluation-fold detector values
    zA = {d: {k: np.full(n, np.nan) for k in range(5)} for d in DETECTORS}             # fold-k model's zA for eval AND calibration answers
    fitlog = []; e_ix = names.index('epr'); l_ix = names.index('trace_length'); g5 = [names.index(f) for f in GOOD_5]
    sig5 = np.array([FEATURE_SIGNS[f] for f in GOOD_5], float)
    for cell in sorted(set(cells)):
        cm = cells == cell
        for k in range(5):
            c = (k + 1) % 5; fa = cm & ~np.isin(fold, [k, c]); use = cm & np.isin(fold, [k, c])
            assert not (fa & (fold == k)).any()
            Xf = X[fa].copy(); med = np.nanmedian(Xf, 0); Xall = np.where(np.isfinite(X), X, med)     # fit-fold medians only
            mu = Xall[fa].mean(0); sd = Xall[fa].std(0); keep = sd > 1e-12
            Z = np.zeros_like(Xall); Z[:, keep] = (Xall[:, keep] - mu[keep]) / sd[keep]
            anchor_fit = Z[fa, e_ix]
            vals = {}
            # D1: U-PCR, derived polarity (upcr_pipeline_faithful 'derived'), fitted configuration
            Ff = Z[fa][:, keep].T
            probe = upcr_fit(Ff, **FIT); pol = np.sign(probe.rho_hat_full); pol[pol == 0] = 1.0
            res = upcr_fit(Ff * pol[:, None], **FIT); w = np.zeros(len(names)); w[np.flatnonzero(keep)] = res.w * pol
            vals['D1_upcr_full'] = (w, int(res.keep.sum()) if hasattr(res, 'keep') else None)
            polfull = np.zeros(len(names)); polfull[np.flatnonzero(keep)] = pol
            # D2: continuous L-SML on GOOD_5 with FEATURE_SIGNS (original configuration)
            W5, fused5, m5 = lsml_linear(Z[fa][:, g5] * sig5)
            w2 = np.zeros(len(names)); w2[g5] = W5 * sig5
            gates.setdefault('lsml_linear_reproduces_fused_max_abs', 0.0)
            gates['lsml_linear_reproduces_fused_max_abs'] = max(gates['lsml_linear_reproduces_fused_max_abs'], float(np.max(np.abs(Z[fa][:, g5] @ w2[g5] - fused5))))
            vals['D2_lsml_cont_good5'] = (w2, int(m5['K']))
            # D3: continuous L-SML on the full pool with D1's polarities; D4: equal average of the same
            kk = np.flatnonzero(keep); Wf, fusedf, mf = lsml_linear(Z[fa][:, kk] * pol)
            w3 = np.zeros(len(names)); w3[kk] = Wf * pol
            gates['lsml_linear_reproduces_fused_max_abs'] = max(gates['lsml_linear_reproduces_fused_max_abs'], float(np.max(np.abs(Z[fa][:, kk] @ w3[kk] - fusedf))))
            vals['D3_lsml_full'] = (w3, int(mf['K']))
            w4 = np.zeros(len(names)); w4[kk] = pol / len(kk); vals['D4_equal_full'] = (w4, None)
            w5 = np.zeros(len(names)); w5[e_ix] = 1.0; vals['D5_epr'] = (w5, None)
            w6 = np.zeros(len(names)); w6[l_ix] = 1.0; vals['D6_length'] = (w6, None)
            for d, (wd, extra) in vals.items():
                sf = Z[fa] @ wd; _, flipped = anchor_orient(sf, anchor_fit); sgn = -1.0 if flipped else 1.0
                sf = sgn * sf; m_, s_ = sf.mean(), max(sf.std(), 1e-12)
                su = sgn * (Z[use] @ wd); zA[d][k][use] = (su - m_) / s_
                ev = cm & (fold == k); A[d][ev] = sgn * (Z[ev] @ wd)
                fitlog.append({'cell': cell, 'fold': k, 'detector': d, 'flipped': bool(flipped), 'extra': extra, 'views_kept': int(keep.sum()),
                               'imputed_fit_values': int((~np.isfinite(X[fa])).sum())})
    for d in DETECTORS:
        if not np.isfinite(A[d]).all(): stop(f'{d} non-finite on some evaluation answer')
    if not gates['lsml_linear_reproduces_fused_max_abs'] <= 1e-8: stop('L-SML linear weights do not reproduce the fused score')
    timing['detectors_s'] = time.perf_counter() - t
    with open(OUT / 'FIT_LOG.jsonl', 'w', encoding='utf8') as fh:
        for r in fitlog: fh.write(json.dumps(r) + '\n')

    # ------------------------------------------------------------------ OFFSET rule (per-benchmark calibration), weights 1 (primary), 0.5, 2 (descriptive)
    def offset_flags(zAd, wgt):
        f = np.zeros(S_, bool)
        for k in range(5):
            c = (k + 1) % 5; zstep = zAd[k][aid]; D = wgt * zstep + zS
            for bm in (prm_step, pb_step):
                calm = bm & (step_fold == c); evm = bm & (step_fold == k)
                if not np.isfinite(D[calm]).all() or not np.isfinite(D[evm]).all(): raise ArithmeticError('non-finite D')
                tau = float(np.quantile(D[calm], .8)); f[evm] = D[evm] >= tau
        return f
    for d in DETECTORS:
        flags[f'OFFSET_{d}'] = offset_flags(zA[d], 1.0)
    for d in PRIMARY_DET:
        for wgt in (0.5, 2.0): flags[f'OFFSETw{wgt}_{d}'] = offset_flags(zA[d], wgt)
    np.savez_compressed(OUT / 'DECISIONS.npz', offsets=off, **{k: v for k, v in flags.items()}, **{f'A_{d}': A[d].astype(np.float32) for d in DETECTORS})

    # shuffled-A controls (label-free permutation of zA among same cell x fold x step-count answers)
    rng = np.random.default_rng(20261013); shuf = {d: [] for d in PRIMARY_DET}; key = pd.Series(list(zip(cells, fold, ns)))
    grp = key.groupby(key).indices
    for s in range(args.shuffles):
        for d in PRIMARY_DET:
            zp = {k: zA[d][k].copy() for k in range(5)}
            for _, ix in grp.items():
                if len(ix) < 2: continue
                ix = np.asarray(ix); kf = fold[ix[0]]; perm = rng.permutation(ix); zp[kf][ix] = zA[d][kf][perm]
            shuf[d].append(offset_flags(zp, 1.0))

    # ------------------------------------------------------------------ evaluation (labels enter here only)
    t = time.perf_counter(); P = np.flatnonzero(prm); good = ~labels
    def counts_of(fl):
        v = ~fl; c = np.zeros((n, 4))
        c[:, 0] = np.bincount(aid, weights=(v & good), minlength=n); c[:, 1] = np.bincount(aid, weights=(v & ~good), minlength=n)
        c[:, 2] = np.bincount(aid, weights=(~v & ~good), minlength=n); c[:, 3] = np.bincount(aid, weights=(~v & good), minlength=n)
        return c
    def official(fl):
        v = ~fl; res = prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i + 1]].astype(int).tolist()} for i in P], [meta[ids[i]] for i in P])
        return 0.5 * (res['total']['f1'] + res['total']['negative_f1'])
    PBc = sorted(set(cells[pb])); Pb = np.flatnonzero(pb)
    if len(PBc) != 8: stop('ProcessBench cells')
    def pb_pred(fl):
        pr = np.full(n, -2)
        for i in Pb:
            fi = np.flatnonzero(fl[off[i]:off[i + 1]]); pr[i] = int(fi[0]) if len(fi) else -1
        return pr
    Gpb, gpi = np.unique(groups[Pb], return_inverse=True); cix = np.array([PBc.index(c) for c in cells[Pb]])
    def pb_counts(pr):
        a = np.zeros((len(Gpb), len(PBc), 4)); err = target[Pb] >= 0; hit = pr[Pb] == target[Pb]
        np.add.at(a, (gpi, cix, 0), (hit & err).astype(float)); np.add.at(a, (gpi, cix, 1), err.astype(float))
        np.add.at(a, (gpi, cix, 2), (hit & ~err).astype(float)); np.add.at(a, (gpi, cix, 3), (~err).astype(float))
        return a
    def pb_f1(a):
        ae = ratio(a[..., 0], a[..., 1]); ac = ratio(a[..., 2], a[..., 3]); f = np.where((ae == 0) & (ac == 0), 0.0, ratio(2 * ae * ac, ae + ac))
        return f, ae, ac
    ruleset = list(flags)
    CNT = {r: counts_of(flags[r]) for r in ruleset}; OFF = {r: official(flags[r]) for r in ruleset}
    for r in ruleset:
        if abs(float(prm_parts(CNT[r][noncontrol].sum(0))['prmscore']) - OFF[r]) > 1e-12: stop(f'counts vs official {r}')
    gates['R0_prmscore'] = OFF['R0_frozen']; gates['R2_prmscore'] = OFF['R2_allocate']
    if abs(OFF['R0_frozen'] - 0.6565188557935739) > 1e-12 or abs(OFF['R2_allocate'] - 0.6635359749948694) > 1e-12: stop('R0/R2 do not reproduce')
    PBA = {r: pb_counts(pb_pred(flags[r])) for r in ruleset}
    nfl = {r: np.bincount(aid, weights=flags[r], minlength=n) for r in ruleset}
    rows = []
    for r in ruleset:
        pp = prm_parts(CNT[r][noncontrol].sum(0)); f, ae, ac = pb_f1(PBA[r].sum(0)); fl = nfl[r]
        rows.append({'rule': r, 'prmscore': OFF[r], **{k: float(v) for k, v in pp.items() if k != 'prmscore'},
                     'erroneous_with_error_flagged': float((CNT[r][err_nc][:, 2] > 0).mean()),
                     'prm_ms_share_flagged': float((fl[ms] > 0).mean()), 'prm_err_share_flagged': float((fl[err_nc] > 0).mean()),
                     'prm_controls_share_flagged': float((fl[control] > 0).mean()), 'prm_noncontrol_zero_flag': float((fl[noncontrol] == 0).mean()),
                     'pb_f1_macro8': float(np.nanmean(f)), 'pb_acc_erroneous_macro8': float(np.nanmean(ae)), 'pb_acc_correct_macro8': float(np.nanmean(ac)),
                     'pb_correct_share_unflagged': float((fl[pb & (target < 0)] == 0).mean()), 'pb_step_flag_rate': float(fl[pb].sum() / ns[pb].sum()),
                     'pb_per_cell_f1': dict(zip(PBc, map(float, f)))})
    dump(OUT / 'METRICS.json', rows)

    # answer-level detection AUROC (evaluation answers pooled over folds, per cell) with group bootstrap
    def auc(pos, neg):
        s = np.r_[pos, neg]; r_ = rankdata(s); return float((r_[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    det = {}
    rng2 = np.random.default_rng(20261014)
    for d in DETECTORS:
        per = {c: auc(A[d][(cells == c) & (target >= 0)], A[d][(cells == c) & (target < 0)]) for c in PBc}
        e_ms = auc(A[d][err_nc], A[d][ms]); e_ct = auc(A[d][err_nc], A[d][control])
        # length-matched erroneous vs multi_solutions: erroneous answers resampled to the multi_solutions step-count distribution (5 bins)
        bins = np.quantile(ns[ms], [0, .2, .4, .6, .8, 1]); eb = np.clip(np.searchsorted(bins, ns[err_nc], 'right') - 1, 0, 4); mb = np.clip(np.searchsorted(bins, ns[ms], 'right') - 1, 0, 4)
        wts = np.array([(mb == b).mean() / max((eb == b).mean(), 1e-12) for b in eb]); pos_ = A[d][err_nc]; neg_ = A[d][ms]
        cmp_ = (pos_[:, None] > neg_[None, :]) + 0.5 * (pos_[:, None] == neg_[None, :]); e_ms_lm = float((wts[:, None] * cmp_).sum() / (wts.sum() * len(neg_)))
        det[d] = {'pb_auc_per_cell': per, 'pb_auc_macro8': float(np.mean(list(per.values()))), 'prm_err_vs_multi_solutions': e_ms,
                  'prm_err_vs_multi_solutions_length_matched': e_ms_lm, 'prm_err_vs_controls_CONFOUNDED': e_ct}
        # group bootstrap for the PB macro and the fair PRMBench comparison
        bs_pb = []; bs_ms = []
        ugp, gip = np.unique(groups[pb], return_inverse=True); pbi = np.flatnonzero(pb)
        for _ in range(500):
            wg = rng2.multinomial(len(ugp), np.full(len(ugp), 1 / len(ugp))); wa = wg[gip]
            vals_ = []
            for c in PBc:
                cmk = cells[pbi] == c; e_ = cmk & (target[pbi] >= 0); o_ = cmk & (target[pbi] < 0)
                pe = np.repeat(A[d][pbi[e_]], wa[e_]); po = np.repeat(A[d][pbi[o_]], wa[o_])
                if len(pe) and len(po): vals_.append(auc(pe, po))
            bs_pb.append(np.mean(vals_))
        det[d]['pb_auc_macro8_ci95'] = [float(np.quantile(bs_pb, .025)), float(np.quantile(bs_pb, .975))]
        ug, gi_ = np.unique(groups[prm], return_inverse=True); pri = np.flatnonzero(prm)
        for _ in range(500):
            wg = rng2.multinomial(len(ug), np.full(len(ug), 1 / len(ug))); wa = wg[gi_]
            pe = np.repeat(A[d][pri[err_nc[pri]]], wa[err_nc[pri]]); po = np.repeat(A[d][pri[ms[pri]]], wa[ms[pri]])
            if len(pe) and len(po): bs_ms.append(auc(pe, po))
        det[d]['prm_err_vs_multi_solutions_ci95'] = [float(np.quantile(bs_ms, .025)), float(np.quantile(bs_ms, .975))]
    dump(OUT / 'ANSWER_DETECTION.json', det)

    # paired bootstrap of rule contrasts: PRMScore (PRMBench groups) and PB F1 macro-8 (PB groups)
    Gu, gi = np.unique(groups[P], return_inverse=True); gmap = np.full(n, -1); gmap[P] = gi; nci = np.flatnonzero(noncontrol)
    AG = {}
    for r in ruleset:
        a = np.zeros((len(Gu), 4)); np.add.at(a, gmap[nci], CNT[r][nci]); AG[r] = a
    rng3 = np.random.default_rng(20261011); rng4 = np.random.default_rng(20261012)
    bp = {r: np.empty(args.draws) for r in ruleset}; bf = {r: np.empty(args.draws) for r in ruleset}; pos_ = 0
    while pos_ < args.draws:
        nb = min(1000, args.draws - pos_)
        W = rng3.multinomial(len(Gu), np.full(len(Gu), 1 / len(Gu)), size=nb).astype(float)
        Wp = rng4.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float)
        for r in ruleset:
            bp[r][pos_:pos_ + nb] = prm_parts(W @ AG[r])['prmscore']
            bf[r][pos_:pos_ + nb] = np.nanmean(pb_f1(np.einsum('bg,gkc->bkc', Wp, PBA[r]))[0], 1)
        pos_ += nb
    pt = {r: (OFF[r], float(np.nanmean(pb_f1(PBA[r].sum(0))[0]))) for r in ruleset}
    def q(x, a): x = x[np.isfinite(x)]; return [float(np.quantile(x, a / 2)), float(np.quantile(x, 1 - a / 2))]
    cons = []
    pairs = [(f'OFFSET_{d}', ref) for d in DETECTORS for ref in ('R0_frozen', 'R2_allocate', 'R2pb_allocate')] + [('OFFSET_D3_lsml_full', 'OFFSET_D4_equal_full'), ('OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5')]
    for a_, b_ in pairs:
        for ep, B_ in (('prmscore', bp), ('pb_f1_macro8', bf)):
            if (ep == 'prmscore' and b_ == 'R2pb_allocate') or (ep == 'pb_f1_macro8' and b_ == 'R2_allocate'): continue
            d = B_[a_] - B_[b_]; ei = 0 if ep == 'prmscore' else 1
            cons.append({'contrast': f'{a_} - {b_}', 'endpoint': ep, 'delta': pt[a_][ei] - pt[b_][ei], 'ci95': q(d, .05),
                         'p_two_sided': float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    fam = [c for c in cons if c['contrast'] in ('OFFSET_D1_upcr_full - R0_frozen', 'OFFSET_D2_lsml_cont_good5 - R0_frozen')]
    fam.sort(key=lambda c: c['p_two_sided']); still = True; holm = []
    for i, c in enumerate(fam):
        a = .05 / (len(fam) - i); B_ = bp if c['endpoint'] == 'prmscore' else bf; a_, b_ = c['contrast'].split(' - ')
        lo, hi = q(B_[a_] - B_[b_], a); rej = still and (lo > 0 or hi < 0); still = rej
        holm.append({**c, 'holm_alpha': a, 'holm_ci': [lo, hi], 'rejected': bool(rej), 'improves': bool(rej and lo > 0)})
    dump(OUT / 'CONTRASTS.json', cons); dump(OUT / 'HOLM_PRIMARY_FAMILY.json', holm)
    sh = {}
    for d in PRIMARY_DET:
        ps = [official(fl) for fl in shuf[d]]; fs = [float(np.nanmean(pb_f1(pb_counts(pb_pred(fl)).sum(0))[0])) for fl in shuf[d]]
        sh[d] = {'prmscore_observed': OFF[f'OFFSET_{d}'], 'prmscore_shuffled_mean': float(np.mean(ps)), 'prmscore_shuffled_sd': float(np.std(ps)),
                 'pb_f1_observed': pt[f'OFFSET_{d}'][1], 'pb_f1_shuffled_mean': float(np.mean(fs)), 'pb_f1_shuffled_sd': float(np.std(fs))}
    dump(OUT / 'SHUFFLED_A.json', {'draws': args.shuffles, 'within': 'cell x fold x step count', 'detectors': sh})
    timing['eval_s'] = time.perf_counter() - t; timing['total_s'] = time.perf_counter() - T0
    dump(OUT / 'TIMING.json', timing); dump(OUT / 'GATES.json', gates)
    dump(OUT / 'CODE_MANIFEST.json', {'script_sha256': sha(Path(__file__)), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json'),
                                      'inputs': {k: {'path': str(v), 'sha256': sha(v)} for k, v in INPUTS.items()},
                                      'upcr_sha256': sha(ROOT / 'spectral_utils/upcr.py'), 'fusion_utils_sha256': sha(ROOT / 'spectral_utils/fusion_utils.py'),
                                      'prmbench_sha256': sha(ROOT / 'spectral_utils/prmbench.py'), 'streaming_utils_sha256': sha(ROOT / 'spectral_utils/streaming_utils.py')})
    status.update({'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds')}); dump(OUT / 'RUN_STATUS.json', status)
    pd.set_option('display.width', 250)
    print(pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)} for r in rows]).to_string(index=False))
    print(json.dumps(holm, indent=1)); print(json.dumps(timing, indent=1))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        _ap = argparse.ArgumentParser(); _ap.add_argument('run_id', nargs='?', default='run_20260930')
        rid = _ap.parse_known_args()[0].run_id; (STAGE / rid).mkdir(parents=True, exist_ok=True)
        dump(STAGE / rid / 'RUN_STATUS.json', {'status': 'FAILED', 'error': repr(exc), 'at': datetime.now().isoformat(timespec='seconds')}); raise
