"""decision_rule_v1: label-free rules that decide how many steps to flag in each answer, on frozen step scores.
Protocol results/decision_rule_v1/PROTOCOL.json (frozen d3e3cebac). The step scores (S_equal, stage B) and their within-answer
ranking are never changed; only the flag decision changes. Labels enter only the evaluation block.

    python -B scripts/experiments/decision_rule_run.py [run_id] [--draws N] [--perms N]
"""
import argparse
import hashlib
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
sys.path.insert(0, str(DEPTH)); sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
import er_stage_a as A  # noqa: E402

STAGE = ROOT / 'results/decision_rule_v1'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
INPUTS = {'step_scores': SSL / 'results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz',
          'thresholds': SSL / 'results/expectation_realization_v1/run_20260927_stage_b_thr/THRESHOLDS.json',
          'frozen_metrics': SSL / 'results/expectation_realization_v1/run_20260927_stage_b/METRICS.csv',
          'level_bank': MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz',
          'ct7_profiles': MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy',
          'ct7_profile_validation': MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/PROFILE_VALIDATION.json',
          'oof_answers': R / 'OOF_ANSWERS.csv', 'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'input_freeze': R / 'INPUT_FREEZE.json'}
RULES = ['R0_frozen', 'R1_global', 'R2_allocate', 'R3_ds_map', 'R4_ds_expected_f1', 'R5_ds_count']
FAMILY = RULES[1:]
DROPPED = ['energy_innovation', 'top50_js']
CLASSES = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence', 'missing_condition', 'deception', 'multi_solutions']
DS_SEED = 20260927


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


def ds_fit(votes, seed):
    """Dawid-Skene via cvf_v2 EM (as er_stage_a.em_estimate) returning the fitted model too."""
    core, em = A._cvf()
    x = np.asarray(votes, float); w = np.ones(len(x))
    model = em.fit_em(x, w, 'ds', core.fit_spectral(x, w, 'spectral'), groups=None, seed=seed)
    if model.status != 'ok':
        raise ArithmeticError(f'EM status {model.status}')
    c = 1 if model.orientation > 0 else 0; e = np.asarray(model.emissions, float)
    psi, fpr = e[:, c], e[:, 1 - c]; prev = model.prior if c == 1 else 1 - model.prior
    return model, {'psi': psi, 'eta': 1 - fpr, 'prevalence': float(prev), 'converged': bool(model.diagnostics['converged'])}


def ds_posterior(marks01, est, eps=1e-6):
    psi = np.clip(est['psi'], eps, 1 - eps); eta = np.clip(est['eta'], eps, 1 - eps); pi = np.clip(est['prevalence'], eps, 1 - eps)
    lo = np.log(pi / (1 - pi)) + marks01 @ np.log(psi / (1 - eta)) + (1 - marks01) @ np.log((1 - psi) / eta)
    return 1 / (1 + np.exp(-lo))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run_id', nargs='?', default='run_20260929')
    ap.add_argument('--draws', type=int, default=10_000); ap.add_argument('--perms', type=int, default=100); ap.add_argument('--baseline-draws', type=int, default=3)
    ap.add_argument('--ds-population', choices=['all', 'prm'], default='all',
                    help="rows for the DS marks and fit: 'all' fit-fold steps (frozen protocol) or PRMBench fit-fold steps only (post-hoc amendment A1)")
    args = ap.parse_args(); OUT = STAGE / args.run_id
    if (OUT / 'RUN_STATUS.json').exists() and json.loads((OUT / 'RUN_STATUS.json').read_text(encoding='utf8')).get('status') == 'COMPLETE':
        raise SystemExit(f'{OUT} already holds a finished run; pass a new run id')
    OUT.mkdir(parents=True, exist_ok=True); T0 = time.perf_counter(); timing = {}; gates = {}
    status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'args': vars(args)}; dump(OUT / 'RUN_STATUS.json', status)

    def stop(msg):
        status.update({'status': 'STOPPED', 'reason': msg}); dump(OUT / 'RUN_STATUS.json', status); dump(OUT / 'GATES.json', gates); raise SystemExit('STOP: ' + msg)

    # ------------------------------------------------------------------ population
    ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
    off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S_ = int(off[-1])
    pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); ids = ans.id.to_numpy(); groups = ans.source_group.to_numpy()
    cells = ans.cell.to_numpy(); target = ans.target.to_numpy()
    meta_path = Path(json.loads(INPUTS['input_freeze'].read_text(encoding='utf-8-sig'))['prm_metadata']['path'])
    meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
    cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
    control = prm & (cls == 'correct'); noncontrol = prm & ~control
    has_err = np.array([labels[off[i]:off[i + 1]].any() for i in range(n)]) & prm
    eligible = np.array([prm[i] and labels[off[i]:off[i + 1]].any() and (~labels[off[i]:off[i + 1]]).any() for i in range(n)])
    ms = noncontrol & (cls == 'multi_solutions'); inert = noncontrol & ~has_err & ~ms
    aid = np.repeat(np.arange(n), ns); step_fold = fold[aid]; prm_step = prm[aid]; pos = np.arange(S_) - off[aid]
    gates['population'] = {'answers': n, 'prm': int(prm.sum()), 'pb': int(pb.sum()), 'controls': int(control.sum()), 'noncontrol': int(noncontrol.sum()),
                           'erroneous': int((has_err & noncontrol).sum()), 'multi_solutions': int(ms.sum()), 'inert': int(inert.sum()), 'eligible': int(eligible.sum())}
    if (gates['population']['prm'], gates['population']['controls'], gates['population']['noncontrol'], gates['population']['erroneous'],
            gates['population']['multi_solutions'], gates['population']['inert']) != (6969, 758, 6211, 6035, 160, 16):
        stop('population differs from the protocol')

    # ------------------------------------------------------------------ frozen scores, thresholds, raw channels
    t = time.perf_counter()
    S = np.load(INPUTS['step_scores'])['S_equal'].astype(float)
    tauS = {int(k): float(v) for k, v in json.loads(INPUTS['thresholds'].read_text(encoding='utf8'))['S_equal'].items()}
    lv = np.load(INPUTS['level_bank']); level = lv['level'].astype(float); names11 = list(map(str, lv['channels']))
    drv = lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)
    prof = np.load(INPUTS['ct7_profiles']).astype(float); pnames = json.loads(INPUTS['ct7_profile_validation'].read_text(encoding='utf8'))['channels']
    raw = np.column_stack([level, prof[:, pnames.index('chosen_token_z_despiked')], drv]); names = names11 + ['realized_z', 'realized_drv']
    surv = [j for j, c in enumerate(names) if c not in DROPPED]
    if not np.isfinite(raw).all() or len(surv) != 11: stop('raw channels')
    dev = float(np.max(np.abs(answer_standardize(raw, off)[:, surv].mean(1) - S)))
    gates['raw_channels_rebuild_frozen_S_equal_max_abs'] = dev
    if not dev <= 1e-9: stop('raw channels do not rebuild the frozen S_equal')
    Xs = raw[:, surv]; timing['load_s'] = time.perf_counter() - t

    def answer_z(s):
        z = np.empty(S_)
        for i in range(n): z[off[i]:off[i + 1]] = zt(s[off[i]:off[i + 1]])
        return z

    def top_by_score(score, counts):
        """flag, in every answer, the counts[i] steps with the highest score (ties: earlier step first)."""
        f = np.zeros(S_, bool)
        for i in range(n):
            k = int(min(counts[i], ns[i]))
            if k > 0:
                a = off[i]; order = np.argsort(-score[a:off[i + 1]], kind='stable'); f[a + order[:k]] = True
        return f

    def build_rules(Xr, Sr, tau0, seed_ds, record=None):
        """All six rules on raw channels Xr and step score Sr; tau0 = per-fold R0 thresholds (None -> q80 rule)."""
        zS = answer_z(Sr); flags = {r: np.zeros(S_, bool) for r in RULES}; post = np.full(S_, np.nan); G_all = np.full(S_, np.nan)
        for k in range(5):
            c = (k + 1) % 5; fitm = ~np.isin(step_fold, [k, c]); calm = prm_step & (step_fold == c); evm = step_fold == k
            assert not (fitm & evm).any() and not (calm & evm).any()   # hard stop: no fold-k row in a fold-k quantity
            t0 = tau0[k] if tau0 is not None else float(np.quantile(zS[calm], .8))
            flags['R0_frozen'][evm] = zS[evm] >= t0
            mu = Xr[fitm].mean(0); sd = np.maximum(Xr[fitm].std(0), 1e-12); Zg = (Xr - mu) / sd; G = Zg.mean(1); G_all[evm] = G[evm]
            tauG = float(np.quantile(G[calm], .8)); f1 = G >= tauG; flags['R1_global'][evm] = f1[evm]
            dsm = fitm & prm_step if args.ds_population == 'prm' else fitm   # amendment A1: PRMBench fit steps only
            thr = np.quantile(Zg[dsm], .8, axis=0); m01 = (Zg >= thr).astype(float)
            model, est = ds_fit(np.where(m01[dsm] > 0, 1.0, -1.0), seed_ds)
            p = ds_posterior(m01, est)
            if not np.isfinite(p).all(): raise ArithmeticError('posterior not finite')
            post[evm] = p[evm]
            pf = p[fitm & prm_step]; grid = np.unique(np.r_[np.quantile(pf, np.arange(0.005, 0.9951, 0.005)), 0.5]); best = (-1.0, None)
            for cc in grid:
                kept = pf < cc; tp = (1 - pf)[kept].sum(); fp = pf[kept].sum(); tn = pf[~kept].sum(); fn = (1 - pf)[~kept].sum()
                e = float(prm_parts(np.array([tp, fp, tn, fn]))['prmscore'])
                if e > best[0] or (e == best[0] and cc > best[1]): best = (e, float(cc))
            flags['R3_ds_map'][evm] = p[evm] > 0.5; flags['R4_ds_expected_f1'][evm] = p[evm] >= best[1]
            if record is not None:
                model_check = float(np.max(np.abs(model.predict(np.where(m01[:2000] > 0, 1.0, -1.0)) - p[:2000])))
                record[k] = {'tau_S': t0, 'tau_G': tauG, 'mu': mu, 'sd': sd, 'mark_thresholds': thr, 'mark_rate_fit': m01[fitm].mean(0),
                             'ds_psi': est['psi'], 'ds_eta': est['eta'], 'ds_prevalence': est['prevalence'], 'ds_converged': est['converged'],
                             'closed_form_vs_model_posterior_max_abs': model_check, 'r4_cutoff': best[1], 'r4_expected_prmscore_fit': best[0],
                             'posterior_mean_fit_prm': float(pf.mean()), 'ds_flag_rate_map_fit_prm': float((pf > 0.5).mean())}
        # count-based rules, answer by answer (counts from the fold-k decisions of R1 / the fold-k posteriors)
        c1 = np.bincount(aid, weights=flags['R1_global'], minlength=n)
        flags['R2_allocate'] = top_by_score(Sr, c1)
        c5 = np.floor(np.bincount(aid, weights=post, minlength=n) + 0.5)
        flags['R5_ds_count'] = top_by_score(Sr, c5)
        return flags, post, G_all

    t = time.perf_counter(); rec = {}
    try:
        flags, post, G_all = build_rules(Xs, S, tauS, DS_SEED, record=rec)
    except (ArithmeticError, ValueError) as e:
        stop(f'rule construction failed: {e!r}')
    gates['ds_closed_form_vs_model_posterior_max_abs'] = max(r['closed_form_vs_model_posterior_max_abs'] for r in rec.values())
    if not gates['ds_closed_form_vs_model_posterior_max_abs'] <= 1e-8: stop('closed-form DS posterior differs from the EM model')
    gates['ds_converged_all_folds'] = all(r['ds_converged'] for r in rec.values())
    timing['rules_s'] = time.perf_counter() - t
    np.savez_compressed(OUT / 'DECISIONS.npz', offsets=off, posterior=post.astype(np.float32), G=G_all.astype(np.float32), **{r: flags[r] for r in RULES})

    # ------------------------------------------------------------------ evaluation (labels enter here only)
    t = time.perf_counter(); P = np.flatnonzero(prm); good = ~labels
    counts = {}; official = {}
    for r in RULES:
        v = ~flags[r]
        res = prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i + 1]].astype(int).tolist()} for i in P], [meta[ids[i]] for i in P])
        official[r] = {'prmscore': 0.5 * (res['total']['f1'] + res['total']['negative_f1']), 'f1': res['total']['f1'], 'negative_f1': res['total']['negative_f1'],
                       'by_class_f1': res['by_classification']['f1'], 'by_class_negative_f1': res['by_classification']['negative_f1'], 'used_redundancy_head': res['used_redundancy_head']}
        c = np.zeros((n, 4))
        c[:, 0] = np.bincount(aid, weights=(v & good), minlength=n); c[:, 1] = np.bincount(aid, weights=(v & ~good), minlength=n)
        c[:, 2] = np.bincount(aid, weights=(~v & ~good), minlength=n); c[:, 3] = np.bincount(aid, weights=(~v & good), minlength=n)
        counts[r] = c
    frozen = pd.read_csv(INPUTS['frozen_metrics']); fz = float(frozen[(frozen.method == 'S_equal') & (frozen.metric == 'prmscore') & (frozen.stratum == 'all')].estimate.iloc[0])
    gates['R0_reproduces_frozen_prmscore'] = {'frozen': fz, 'here': official['R0_frozen']['prmscore'], 'abs_diff': abs(fz - official['R0_frozen']['prmscore'])}
    if not gates['R0_reproduces_frozen_prmscore']['abs_diff'] <= 1e-12: stop('R0 does not reproduce the frozen PRMScore')
    for r in RULES:
        d = abs(float(prm_parts(counts[r][noncontrol].sum(0))['prmscore']) - official[r]['prmscore'])
        if not d <= 1e-12: stop(f'counts differ from the official scorer for {r}')
    gates['counts_equal_official'] = True

    def within(score):
        return float(np.mean([(rankdata(score[off[i]:off[i + 1]])[labels[off[i]:off[i + 1]]].sum() - labels[off[i]:off[i + 1]].sum() * (labels[off[i]:off[i + 1]].sum() + 1) / 2)
                              / (labels[off[i]:off[i + 1]].sum() * (~labels[off[i]:off[i + 1]]).sum()) for i in np.flatnonzero(eligible)]))
    wa = {'S': within(S), 'G': within(G_all), 'posterior': within(post)}
    place = {'R0_frozen': 'S', 'R1_global': 'G', 'R2_allocate': 'S', 'R3_ds_map': 'posterior', 'R4_ds_expected_f1': 'posterior', 'R5_ds_count': 'S'}
    nflag = {r: np.bincount(aid, weights=flags[r], minlength=n) for r in RULES}
    err_share = np.bincount(aid, weights=labels, minlength=n) / ns
    rows = []
    for r in RULES:
        pp = prm_parts(counts[r][noncontrol].sum(0)); fl = nflag[r]
        def clean(mask): return {'answers': int(mask.sum()), 'share_with_flag': float((fl[mask] > 0).mean()), 'step_flag_rate': float(fl[mask].sum() / ns[mask].sum())}
        rows.append({'rule': r, **{k: float(v) for k, v in pp.items()}, 'official_prmscore': official[r]['prmscore'],
                     'official_by_class_f1': official[r]['by_class_f1'], 'official_by_class_negative_f1': official[r]['by_class_negative_f1'],
                     'used_redundancy_head': official[r]['used_redundancy_head'],
                     'within_auc_of_placement_scale': wa[place[r]], 'placement_scale': place[r],
                     'alloc_corr_flag_share_vs_error_share': float(np.corrcoef(fl[noncontrol] / ns[noncontrol], err_share[noncontrol])[0, 1]),
                     'erroneous_with_error_flagged': float((counts[r][has_err & noncontrol][:, 2] > 0).mean()),
                     'noncontrol_answers_with_no_flag': float((fl[noncontrol] == 0).mean()),
                     'controls': clean(control), 'multi_solutions': clean(ms), 'inert': clean(inert)})
    dump(OUT / 'METRICS.json', rows)

    # ------------------------------------------------------------------ paired source-group bootstrap (PRMBench), classes, Holm
    rng = np.random.default_rng(20261001); Gu, gi = np.unique(groups[P], return_inverse=True); G = len(Gu); gmap = np.full(n, -1); gmap[P] = gi
    strata = ['total'] + CLASSES
    def agg(r):
        a = np.zeros((G, len(strata), 4)); nc_i = np.flatnonzero(noncontrol)
        np.add.at(a, (gmap[nc_i], 0), counts[r][nc_i])
        for si, c in enumerate(CLASSES, 1):
            ii = np.flatnonzero(noncontrol & (cls == c)); np.add.at(a, (gmap[ii], si), counts[r][ii])
        return a
    AG = {r: agg(r) for r in RULES}; EP = ['prmscore', 'f1_correct', 'f1_error', 'flag_rate']
    boot = {r: np.empty((args.draws, len(strata), len(EP)), np.float32) for r in RULES}; pos_ = 0
    while pos_ < args.draws:
        nb = min(1000, args.draws - pos_); W = rng.multinomial(G, np.full(G, 1 / G), size=nb).astype(float)
        for r in RULES:
            pp = prm_parts(np.einsum('bg,gkc->bkc', W, AG[r])); boot[r][pos_:pos_ + nb] = np.stack([pp[e] for e in EP], -1)
        pos_ += nb
    point = {r: np.stack([prm_parts(AG[r].sum(0))[e] for e in EP], -1) for r in RULES}
    def q(x, a): x = x[np.isfinite(x)]; return [float(np.quantile(x, a / 2)), float(np.quantile(x, 1 - a / 2))] if len(x) else [np.nan, np.nan]
    crow = []
    for r in FAMILY:
        for si, sname in enumerate(strata):
            for ei, e in enumerate(EP):
                d = boot[r][:, si, ei].astype(float) - boot['R0_frozen'][:, si, ei].astype(float); pt = float(point[r][si, ei] - point['R0_frozen'][si, ei])
                crow.append({'contrast': f'{r} - R0_frozen', 'stratum': sname, 'endpoint': e, 'delta': pt, 'ci95': q(d, .05),
                             'ci_bonf9': q(d, .05 / 9) if sname in CLASSES else None, 'p_two_sided': float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if np.isfinite(d).any() else None})
    CT = pd.DataFrame(crow)
    prim = CT[(CT.stratum == 'total') & (CT.endpoint == 'prmscore')].copy().sort_values('p_two_sided', kind='stable')
    m = len(prim); holm = []; still = True
    for rank_, (_, row) in enumerate(prim.iterrows()):   # step-down: once a contrast is not rejected, none after it is
        a = .05 / (m - rank_); d = boot[row.contrast.split(' - ')[0]][:, 0, 0].astype(float) - boot['R0_frozen'][:, 0, 0].astype(float)
        lo, hi = q(d, a); rejected = still and (lo > 0 or hi < 0); still = rejected
        holm.append({'contrast': row.contrast, 'delta': row.delta, 'p_two_sided': row.p_two_sided, 'holm_alpha': a, 'holm_ci': [lo, hi],
                     'rejected': bool(rejected), 'improves': bool(rejected and lo > 0)})
    CT.to_csv(OUT / 'CONTRASTS.csv', index=False); dump(OUT / 'HOLM_PRIMARY_FAMILY.json', holm)
    timing['bootstrap_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ nulls (decisions fixed, labels permuted)
    t = time.perf_counter(); rng2 = np.random.default_rng(20261003); ncA = np.flatnonzero(noncontrol); stepmask_nc = noncontrol[aid]
    def prm_from_labels(lab, r):
        v = ~flags[r][stepmask_nc]; g = ~lab[stepmask_nc]
        return float(prm_parts(np.array([(v & g).sum(), (v & ~g).sum(), (~v & ~g).sum(), (~v & g).sum()]))['prmscore'])
    obs = {r: official[r]['prmscore'] - official['R0_frozen']['prmscore'] for r in FAMILY}
    nullA = {r: [] for r in FAMILY}; nullB = {r: [] for r in FAMILY}
    keyB = pd.DataFrame({'i': ncA, 'len': ns[ncA], 'fold': fold[ncA]})
    for _ in range(args.perms):
        o = np.lexsort((rng2.random(S_), aid))          # sorted by answer (aid is non-decreasing), random order inside each answer
        labA = labels[o]                                 # so position t keeps its answer and receives a random label of that answer
        labB = labels.copy()
        for _, grp in keyB.groupby(['len', 'fold']).i:
            idx = grp.to_numpy()
            if len(idx) < 2: continue
            perm = rng2.permutation(len(idx)); src = idx[np.roll(perm, 1)]; dst = idx[perm]
            for d_, s_ in zip(dst, src): labB[off[d_]:off[d_ + 1]] = labels[off[s_]:off[s_ + 1]]
        b0A = prm_from_labels(labA, 'R0_frozen'); b0B = prm_from_labels(labB, 'R0_frozen')
        for r in FAMILY:
            nullA[r].append(prm_from_labels(labA, r) - b0A); nullB[r].append(prm_from_labels(labB, r) - b0B)
    nulls = {r: {'observed': obs[r], 'within_answer_null_mean': float(np.mean(nullA[r])), 'within_answer_null_sd': float(np.std(nullA[r])),
                 'swap_null_mean': float(np.mean(nullB[r])), 'swap_null_sd': float(np.std(nullB[r])),
                 'residual_vs_within_answer': obs[r] - float(np.mean(nullA[r])), 'residual_vs_swap': obs[r] - float(np.mean(nullB[r])),
                 'allocation_beyond_length': float(np.mean(nullA[r]) - np.mean(nullB[r]))} for r in FAMILY}
    sizes = keyB.groupby(['len', 'fold']).i.transform('size'); unswapped = keyB[sizes < 2].i.to_numpy()
    dump(OUT / 'NULLS.json', {'perms': args.perms,
                              'interpretation': 'within-answer null keeps each answer error count, so a pure allocation gain survives it (observed - its mean = placement part); the swap null keeps only length/fold structure (observed - its mean = everything answer-specific, allocation and placement); within-answer mean - swap mean = allocation beyond length',
                              'unswapped_answers_alone_in_their_length_fold_cell': int(len(unswapped)), 'unswapped_steps': int(ns[unswapped].sum()), 'rules': nulls})
    timing['nulls_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ random-score baseline for the clean-answer panel
    t = time.perf_counter(); rng3 = np.random.default_rng(20261004); base = []
    for b in range(args.baseline_draws):
        Xr = rng3.standard_normal(Xs.shape); Sr = rng3.standard_normal(S_)
        try:
            fr, _, _ = build_rules(Xr, Sr, None, DS_SEED)
        except Exception as e:   # the DS fit may fail on structureless marks; the baseline records it instead of stopping
            base.append({'draw': b, 'error': repr(e)}); continue
        rowb = {'draw': b}
        for r in RULES:
            fl = np.bincount(aid, weights=fr[r], minlength=n)
            rowb[r] = {'controls_share_with_flag': float((fl[control] > 0).mean()), 'controls_step_flag_rate': float(fl[control].sum() / ns[control].sum()),
                       'noncontrol_prmscore': float(prm_parts(np.array([((~fr[r]) & good & noncontrol[aid]).sum(), ((~fr[r]) & ~good & noncontrol[aid]).sum(),
                                                                          (fr[r] & ~good & noncontrol[aid]).sum(), (fr[r] & good & noncontrol[aid]).sum()]))['prmscore'])}
        base.append(rowb)
    dump(OUT / 'RANDOM_BASELINE.json', base); timing['baseline_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ ProcessBench official F1 (secondary)
    t = time.perf_counter(); PBc = sorted(set(cells[pb])); pred = {}
    if len(PBc) != 8: stop(f'expected 8 ProcessBench cells, found {len(PBc)}')
    for r in RULES:
        pr = np.full(n, -2)
        for i in np.flatnonzero(pb):
            fi = np.flatnonzero(flags[r][off[i]:off[i + 1]]); pr[i] = int(fi[0]) if len(fi) else -1
        pred[r] = pr
    Pb = np.flatnonzero(pb); Gpb, gpi = np.unique(groups[Pb], return_inverse=True); cix = np.array([PBc.index(c) for c in cells[Pb]])
    def pb_counts(r):
        a = np.zeros((len(Gpb), len(PBc), 4)); err = target[Pb] >= 0; hit = pred[r][Pb] == target[Pb]
        np.add.at(a, (gpi, cix, 0), (hit & err).astype(float)); np.add.at(a, (gpi, cix, 1), err.astype(float))
        np.add.at(a, (gpi, cix, 2), (hit & ~err).astype(float)); np.add.at(a, (gpi, cix, 3), (~err).astype(float))
        return a
    def pb_f1(a):   # official convention: both accuracies 0 -> F1 0 (not NaN, which nanmean would drop)
        ae = ratio(a[..., 0], a[..., 1]); ac = ratio(a[..., 2], a[..., 3]); f = np.where((ae == 0) & (ac == 0), 0.0, ratio(2 * ae * ac, ae + ac))
        return f, ae, ac
    PA = {r: pb_counts(r) for r in RULES}; rng4 = np.random.default_rng(20261002); pbb = {r: np.empty(args.draws) for r in RULES}; pos_ = 0
    while pos_ < args.draws:
        nb = min(1000, args.draws - pos_); W = rng4.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float)
        for r in RULES: pbb[r][pos_:pos_ + nb] = np.nanmean(pb_f1(np.einsum('bg,gkc->bkc', W, PA[r]))[0], 1)
        pos_ += nb
    pbrows = []
    for r in RULES:
        f, ae, ac = pb_f1(PA[r].sum(0))
        pbrows.append({'rule': r, 'f1_macro8': float(np.nanmean(f)), 'acc_erroneous_macro8': float(np.nanmean(ae)), 'acc_correct_macro8': float(np.nanmean(ac)),
                       'per_cell_f1': dict(zip(PBc, map(float, f))), 'delta_vs_R0': float(np.nanmean(f) - np.nanmean(pb_f1(PA['R0_frozen'].sum(0))[0])),
                       'delta_ci95': q(pbb[r] - pbb['R0_frozen'], .05)})
    dump(OUT / 'PB_OFFICIAL_F1.json', pbrows); timing['pb_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ estimates vs truth (label-using diagnostic, after all decisions)
    diag = {}
    for k, r_ in rec.items():
        c = (k + 1) % 5; fitm = ~np.isin(step_fold, [k, c]); prm_fit = fitm & prm_step
        Zg = (Xs - r_['mu']) / r_['sd']; m01 = Zg >= r_['mark_thresholds']; y = labels[fitm]
        yp = labels[prm_fit]
        diag[k] = {**{kk: vv for kk, vv in r_.items() if kk not in ('mu', 'sd', 'mark_thresholds')}, 'true_prevalence_fit_MIXED_PB_PLACEHOLDER': float(y.mean()),
                   'true_psi_MIXED_PB_PLACEHOLDER': m01[fitm][y].mean(0), 'true_eta_MIXED_PB_PLACEHOLDER': (~m01[fitm][~y]).mean(0),
                   'true_prevalence_fit_prm': float(yp.mean()), 'true_psi_prm': m01[prm_fit][yp].mean(0), 'true_eta_prm': (~m01[prm_fit][~yp]).mean(0)}
    dump(OUT / 'DS_ESTIMATES.json', {'channels': [names[j] for j in surv], 'ds_population': args.ds_population,
                                     'note': 'OOF labels mark every ProcessBench step True (PB uses target); only the *_prm truth is meaningful', 'folds': diag})
    timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing); dump(OUT / 'GATES.json', gates)
    dump(OUT / 'CODE_MANIFEST.json', {'script_sha256': sha(Path(__file__)), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json'),
                                      'inputs': {k: {'path': str(v), 'sha256': sha(v)} for k, v in INPUTS.items()}, 'prm_metadata_sha256': sha(meta_path),
                                      'er_stage_a_sha256': sha(ROOT / 'scripts/experiments/er_stage_a.py'),
                                      'cvf_em_sha256': sha(MAIN / '.worktrees/cumulative-vote-fusion-v2/scripts/experiments/cvf_v2/em.py'),
                                      'cvf_core_sha256': sha(MAIN / '.worktrees/cumulative-vote-fusion-v2/scripts/experiments/cvf_v2/core.py'),
                                      'prmbench_sha256': sha(DEPTH / 'spectral_utils/prmbench.py'),
                                      'lsml_gate_locator_research_sha256': sha(DEPTH / 'spectral_utils/lsml_gate_locator_research.py'),
                                      'git_head': __import__('subprocess').run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()})
    status.update({'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds')}); dump(OUT / 'RUN_STATUS.json', status)
    pd.set_option('display.width', 250)
    print(pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)} for r in rows]).to_string(index=False))
    print(json.dumps(holm, indent=1)); print(json.dumps({r: {k: round(v, 4) for k, v in x.items()} for r, x in nulls.items()}, indent=1))
    print(json.dumps([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'per_cell_f1'} for r in pbrows], indent=1)); print(json.dumps(timing, indent=1))
    return OUT


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        _ap = argparse.ArgumentParser(); _ap.add_argument('run_id', nargs='?', default='run_20260929')
        rid = _ap.parse_known_args()[0].run_id
        (STAGE / rid).mkdir(parents=True, exist_ok=True)
        dump(STAGE / rid / 'RUN_STATUS.json', {'status': 'FAILED', 'error': repr(exc), 'at': datetime.now().isoformat(timespec='seconds')}); raise
