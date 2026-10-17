"""PRMScore decomposition on frozen predictions (protocol results/expectation_realization_v1/prmscore_decomposition_v1/
PROTOCOL.json, frozen in commit a6a7ea493 before any result). No fitting and no threshold choice: our rows use the per-fold
thresholds their own fold models produced (THRESHOLDS.json of the byte-identical threshold re-runs of stage B / B2);
realized_drv and the Qwen PRM use the same fold rule (or the PRM's own 0.5 rule). Labels enter only the evaluation.

Panels: P1 overall table, P2 official classifications and PRMScore components with paired source-group intervals,
P3 within-answer ranking vs between-answer comparison vs threshold decisions (with the clean-answer panel),
P4 complementarity, S1 secondary length / first-error-position strata.

    python -B scripts/experiments/er_prmscore_decomposition.py [--draws 10000] [--draws-auc 2000]
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
from scipy import sparse
from scipy.stats import rankdata

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.prmbench import CATEGORY_OF, prmbench_evaluate  # noqa: E402

STAGE = ROOT / 'results/expectation_realization_v1'; OUT = STAGE / 'prmscore_decomposition_v1'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
FROZEN = {'B': STAGE / 'run_20260927_stage_b', 'B2': STAGE / 'run_20260927_stage_b2'}
THR = {'B': STAGE / 'run_20260927_stage_b_thr', 'B2': STAGE / 'run_20260927_stage_b2_thr'}
SEED = 20260927
OURS = {'B13_equal': 'B', 'S_equal': 'B', 'G1_sml': 'B', 'B_sml__merge': 'B2', 'B13_lsml': 'B', 'S_lsml': 'B',
        'fam421': 'B', 'ct7': 'B', 'step_index': 'B'}
FIXED = ['B13_equal', 'fam421', 'ct7', 'step_index']          # fixed-weight rows: used to check the threshold rule
ORDER = ['B13_equal', 'S_equal', 'G1_sml', 'B_sml__merge', 'B13_lsml', 'S_lsml', 'realized_drv', 'fam421', 'ct7', 'step_index',
         'PRM_native', 'PRM_raw_q80', 'PRM_z_q80']
ROW = {  # (display name, kind, calibration condition)
    'B13_equal': ('average of all 13 channels', 'algorithmic, label-free', 'answer-z, q80 of the same model on fold (k+1)%5'),
    'S_equal': ('DS filter, then average of the survivors', 'algorithmic, label-free', 'answer-z, q80 of the same model on fold (k+1)%5'),
    'G1_sml': ('binary rule: DS filter + discovered groups + DS group weights', 'algorithmic, label-free', 'answer-z, q80 of the same model on fold (k+1)%5'),
    'B_sml__merge': ('binary rule, level sub-groups merged', 'MECHANISM TEST (hard-coded level list), not algorithmic', 'answer-z, q80 of the same model on fold (k+1)%5'),
    'B13_lsml': ('L-SML on all 13 channels', 'algorithmic, label-free', 'answer-z, q80 of the same model on fold (k+1)%5'),
    'S_lsml': ('L-SML on the DS survivors', 'algorithmic, label-free', 'answer-z, q80 of the same model on fold (k+1)%5'),
    'realized_drv': ('realized_drv alone', 'single channel chosen post hoc (best of 22 on this data)', 'answer-z, q80 on fold (k+1)%5'),
    'fam421': ('fam421', 'reference developed on this data', 'answer-z, q80 on fold (k+1)%5'),
    'ct7': ('CT7', 'reference developed on this data', 'answer-z, q80 on fold (k+1)%5'),
    'step_index': ('step index (position only)', 'position reference', 'answer-z, q80 on fold (k+1)%5'),
    'PRM_native': ('Qwen2.5-Math-PRM-7B, its own 0.5 threshold', 'supervised reference, different access', 'reward >= 0.5 is valid (manufacturer rule)'),
    'PRM_raw_q80': ('Qwen2.5-Math-PRM-7B, q80 on its own scale', 'supervised reference, different access', 'raw risk, q80 on fold (k+1)%5'),
    'PRM_z_q80': ('Qwen2.5-Math-PRM-7B, exactly our rule', 'supervised reference, different access', 'answer-z, q80 on fold (k+1)%5'),
}
PRIMARY = [('S_equal', 'B13_equal'), ('S_equal', 'realized_drv'), ('B13_equal', 'realized_drv'),
           ('S_equal', 'PRM_z_q80'), ('S_equal', 'PRM_raw_q80'), ('S_equal', 'PRM_native')]
SECONDARY = [('B13_lsml', 'B13_equal'), ('S_lsml', 'S_equal'), ('G1_sml', 'S_equal'), ('B_sml__merge', 'S_equal'),
             ('S_equal', 'fam421'), ('S_equal', 'ct7'), ('realized_drv', 'PRM_z_q80')]
ENDPOINTS = ['prmscore', 'f1_correct', 'f1_error', 'recall_correct', 'recall_error', 'precision_correct', 'precision_error',
             'flag_rate', 'within_auc', 'any_error_hit', 'error_detected']
CLASSES = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence',
           'missing_condition', 'deception', 'multi_solutions']
GROUP3 = {g: [c for c in CLASSES if CATEGORY_OF[c] == g] for g in ('simplicity', 'soundness', 'sensitivity')}
S1_ARMS = ['B13_equal', 'S_equal', 'realized_drv', 'ct7', 'PRM_z_q80', 'PRM_raw_q80', 'step_index']
S1_CONTRASTS = [('S_equal', 'B13_equal'), ('S_equal', 'realized_drv'), ('S_equal', 'PRM_z_q80'), ('S_equal', 'PRM_raw_q80')]
PAIRS4 = [('S_equal', 'realized_drv'), ('S_equal', 'PRM_z_q80'), ('S_equal', 'PRM_raw_q80'), ('realized_drv', 'PRM_z_q80'), ('B13_equal', 'S_equal')]


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def earliest_argmax(v): return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])


def ratio(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b != 0)


def count_metrics(c):
    """c[..., 4] = TP, FP, TN, FN with positive = correct step predicted valid (official convention). NaN where undefined."""
    tp, fp, tn, fn = np.moveaxis(np.asarray(c, float), -1, 0)
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn); p2, r2 = ratio(tn, tn + fn), ratio(tn, tn + fp)
    f1, f1e = ratio(2 * p * r, p + r), ratio(2 * p2 * r2, p2 + r2)
    return {'prmscore': (f1 + f1e) / 2, 'f1_correct': f1, 'f1_error': f1e, 'recall_correct': r, 'recall_error': r2,
            'precision_correct': p, 'precision_error': p2, 'flag_rate': ratio(tn + fn, tp + fp + tn + fn)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--draws', type=int, default=10_000); ap.add_argument('--draws-auc', type=int, default=2_000)
    args = ap.parse_args(); T0 = time.perf_counter(); timing = {}; gates = {}
    OUT.mkdir(parents=True, exist_ok=True)
    status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'draws': args.draws, 'draws_auc': args.draws_auc}
    dump(OUT / 'RUN_STATUS.json', status)

    def stop(msg):
        status.update({'status': 'STOPPED', 'reason': msg, 'gates': gates}); dump(OUT / 'RUN_STATUS.json', status); dump(OUT / 'GATES.json', gates)
        raise SystemExit('STOP: ' + msg)

    # ------------------------------------------------------------------ population
    ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz')
    off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S = int(off[-1])
    prm = ~ans.cell.str.startswith('pb_').to_numpy(); fold = ans.fold.to_numpy(); ids = ans.id.to_numpy(); groups = ans.source_group.to_numpy()
    freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta_path = Path(freeze['prm_metadata']['path'])
    meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
    cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
    control = prm & (cls == 'correct'); noncontrol = prm & ~control
    has_err = np.array([prm[i] and labels[off[i]:off[i + 1]].any() for i in range(n)])
    eligible = np.array([prm[i] and labels[off[i]:off[i + 1]].any() and (~labels[off[i]:off[i + 1]]).any() for i in range(n)])
    P = np.flatnonzero(prm)
    lab_ok = all(np.array_equal(labels[off[i]:off[i + 1]], np.isin(np.arange(ns[i]) + 1, meta[ids[i]]['error_steps'])) for i in P)
    gates['population'] = {'prm_answers': int(prm.sum()), 'controls': int(control.sum()), 'noncontrol': int(noncontrol.sum()),
                           'erroneous': int(has_err.sum()), 'erroneous_noncontrol': int((has_err & noncontrol).sum()),
                           'noncontrol_without_in_range_error': int((noncontrol & ~has_err).sum()), 'within_auc_eligible': int(eligible.sum()),
                           'noncontrol_steps': int(ns[noncontrol].sum()), 'noncontrol_error_steps': int(sum(labels[off[i]:off[i + 1]].sum() for i in np.flatnonzero(noncontrol))),
                           'control_steps': int(ns[control].sum()), 'labels_equal_official_error_steps': bool(lab_ok),
                           'classes': {c: int((cls == c).sum()) for c in sorted(set(cls[prm]))}}
    ms = noncontrol & (cls == 'multi_solutions'); inert = noncontrol & ~has_err & ~ms   # amendment A1
    inert_ok = all(len(meta[ids[i]]['error_steps']) > 0 and min(meta[ids[i]]['error_steps']) > ns[i] for i in np.flatnonzero(inert))
    gates['population'].update({'multi_solutions': int(ms.sum()), 'inert_error_annotation': int(inert.sum()),
                                'inert_error_annotation_ids': ids[inert].tolist(), 'inert_all_indices_past_last_step': bool(inert_ok)})
    exp = {'prm_answers': 6969, 'controls': 758, 'noncontrol': 6211, 'within_auc_eligible': 6030, 'erroneous_noncontrol': 6035,
           'multi_solutions': 160, 'inert_error_annotation': 16}
    class_ok = set(cls[prm]) == set(CLASSES) | {'correct'} and not (ms & has_err).any()
    if any(gates['population'][k] != v for k, v in exp.items()) or not (lab_ok and inert_ok and class_ok):
        stop('population differs from the protocol (amendment A1)')
    has_ok = np.array([prm[i] and (~labels[off[i]:off[i + 1]]).any() for i in range(n)])

    # ------------------------------------------------------------------ gate 1: threshold re-runs byte-identical to the frozen runs
    t = time.perf_counter(); g1 = {}
    for st in ('B', 'B2'):
        fz, tz = np.load(FROZEN[st] / 'STEP_SCORES.npz'), np.load(THR[st] / 'STEP_SCORES.npz')
        g1[st] = {'arrays': len(fz.files), 'same_names': sorted(fz.files) == sorted(tz.files),
                  'identical': all(np.array_equal(fz[k], tz[k], equal_nan=True) for k in fz.files if k in tz.files),
                  'inputs_identical': json.loads((FROZEN[st] / 'INPUT_MANIFEST.json').read_text(encoding='utf8')) == json.loads((THR[st] / 'INPUT_MANIFEST.json').read_text(encoding='utf8')),
                  'rerun_complete': json.loads((THR[st] / 'RUN_STATUS.json').read_text(encoding='utf8'))['status'] == 'COMPLETE',
                  'metrics_csv_equal': pd.read_csv(FROZEN[st] / 'METRICS.csv').equals(pd.read_csv(THR[st] / 'METRICS.csv'))}
    gates['frozen_scores_identical'] = g1
    if not all(all(v[k] for k in ('same_names', 'identical', 'inputs_identical', 'rerun_complete', 'metrics_csv_equal')) for v in g1.values()):
        stop('threshold re-run differs from the frozen run')

    # ------------------------------------------------------------------ scores, decision scales, thresholds, valid flags
    def answer_z(s):
        z = np.full(S, np.nan)
        for i in P: z[off[i]:off[i + 1]] = zt(s[off[i]:off[i + 1]])
        return z
    step_fold = np.repeat(fold, ns); prm_step = np.repeat(prm, ns)
    def q80_rule(z):   # threshold of eval fold k = 0.8 quantile of the decision-scale scores of PRMBench steps in fold (k+1)%5
        return {k: float(np.quantile(z[prm_step & (step_fold == (k + 1) % 5)], .8)) for k in range(5)}
    def valid_from(z, tau):
        v = np.zeros(S, bool)
        for k in range(5): sel = prm_step & (step_fold == k); v[sel] = z[sel] < tau[k]
        return v
    score, scale, tau, valid = {}, {}, {}, {}
    thr_saved = {st: json.loads((THR[st] / 'THRESHOLDS.json').read_text(encoding='utf8')) for st in THR}
    for m, st in OURS.items():
        s = np.load(FROZEN[st] / 'STEP_SCORES.npz')[m]; score[m] = s; scale[m] = answer_z(s)
        tau[m] = {int(k): float(v) for k, v in thr_saved[st][m].items()}; valid[m] = valid_from(scale[m], tau[m])
    ch = np.load(THR['B'] / 'CHANNELS.npz'); names = [str(x) for x in ch['names']]
    fz_names = json.loads((FROZEN['B'] / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))['channels']
    dev_ch = float(np.max(np.abs(ch['values'].mean(1) - np.load(FROZEN['B'] / 'STEP_SCORES.npz')['B13_equal'])))
    gates['channels_tied_to_frozen'] = {'names_equal_frozen_manifest': names == fz_names, 'equal_mean_minus_frozen_B13_equal_max_abs': dev_ch}
    if names != fz_names or not dev_ch <= 1e-12: stop('CHANNELS.npz not tied to the frozen run')
    s = ch['values'][:, names.index('realized_drv')].astype(float); score['realized_drv'] = s; scale['realized_drv'] = answer_z(s)
    tau['realized_drv'] = q80_rule(scale['realized_drv']); valid['realized_drv'] = valid_from(scale['realized_drv'], tau['realized_drv'])
    reward = np.full(S, np.nan)
    for i in P:
        r = np.asarray(meta[ids[i]]['rewards'], float)
        if len(r) != ns[i] or not np.isfinite(r).all(): stop(f'PRM reward length/NaN mismatch on {ids[i]}')
        reward[off[i]:off[i + 1]] = r
    risk = 1 - reward
    score['PRM_native'] = score['PRM_raw_q80'] = score['PRM_z_q80'] = risk
    scale['PRM_native'] = scale['PRM_raw_q80'] = risk; scale['PRM_z_q80'] = answer_z(risk)
    tau['PRM_native'] = {k: 0.5 for k in range(5)}; valid['PRM_native'] = prm_step & (reward >= 0.5)
    tau['PRM_raw_q80'] = q80_rule(risk); valid['PRM_raw_q80'] = valid_from(risk, tau['PRM_raw_q80'])
    tau['PRM_z_q80'] = q80_rule(scale['PRM_z_q80']); valid['PRM_z_q80'] = valid_from(scale['PRM_z_q80'], tau['PRM_z_q80'])
    gates['threshold_rule_reproduces_saved_fixed_rows'] = {m: max(abs(q80_rule(scale[m])[k] - tau[m][k]) for k in range(5)) for m in FIXED}
    if max(gates['threshold_rule_reproduces_saved_fixed_rows'].values()) > 1e-12: stop('fold rule does not reproduce the saved thresholds')
    timing['scores_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ per-answer quantities (PRMBench answers)
    t = time.perf_counter(); nP = len(P); good = ~labels
    per = {}
    for m in ORDER:
        s, v = score[m], valid[m]
        cnt = np.zeros((nP, 4)); auc = np.full(nP, np.nan); hit = np.full(nP, np.nan); det = np.full(nP, np.nan)
        nflag = np.zeros(nP); nflag_ok = np.zeros(nP)
        for j, i in enumerate(P):
            a, b = off[i:i + 2]; vv = v[a:b]; gg = good[a:b]
            cnt[j] = [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
            nflag[j] = (~vv).sum(); nflag_ok[j] = (~vv & gg).sum()
            if eligible[i]: auc[j] = within_auc(labels[a:b], s[a:b])
            if has_err[i]: hit[j] = float(labels[a + earliest_argmax(s[a:b])]); det[j] = float((~vv & ~gg).any())
        per[m] = {'cnt': cnt, 'auc': auc, 'hit': hit, 'det': det, 'nflag': nflag, 'nflag_ok': nflag_ok}
    timing['per_answer_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ official scorer + gate 2 (frozen METRICS replay)
    t = time.perf_counter(); official = {}
    for m in ORDER:
        res = prmbench_evaluate([{'idx': ids[i], 'labels': valid[m][off[i]:off[i + 1]].astype(int).tolist()} for i in P], [meta[ids[i]] for i in P])
        official[m] = {'total': res['total'], 'by_classification': {k: res['by_classification'][k] for k in ('f1', 'negative_f1', 'recall', 'negative_recall')},
                       'used_redundancy_head': res['used_redundancy_head']}
    nc = noncontrol[P]
    replay = {}
    for m, st in OURS.items():
        M = pd.read_csv(FROZEN[st] / 'METRICS.csv'); M = M[(M.method == m) & (M.benchmark == 'prm') & (M.stratum == 'all')].set_index('metric').estimate
        mine = {'within_auc': float(np.nanmean(per[m]['auc'])), 'prmscore': 0.5 * (official[m]['total']['f1'] + official[m]['total']['negative_f1']),
                'any_error_hit': float(np.nanmean(per[m]['hit']))}
        replay[m] = {k: {'frozen': float(M[k]), 'recomputed': mine[k], 'abs_diff': abs(float(M[k]) - mine[k])} for k in mine}
    gates['frozen_metrics_replay'] = replay
    rep_dev = [r[k]['abs_diff'] for r in replay.values() for k in r]
    tot_counts = {m: count_metrics(per[m]['cnt'][nc].sum(0))['prmscore'] for m in ORDER}
    tot_dev = [abs(float(tot_counts[m]) - 0.5 * (official[m]['total']['f1'] + official[m]['total']['negative_f1'])) for m in ORDER]
    cls_dev = []
    for m in ORDER:
        for c in CLASSES:
            mm = count_metrics(per[m]['cnt'][nc & (cls[P] == c)].sum(0)); o = official[m]['by_classification']
            cls_dev.append(abs(mm['f1_correct'] - o['f1'][c]))
            if c != 'multi_solutions':   # the only class without error steps: error-F1 undefined (NaN here, 0 or -1 officially)
                cls_dev.append(abs(mm['f1_error'] - o['negative_f1'][c]))
    if not (np.isfinite(rep_dev).all() and np.isfinite(tot_dev).all() and np.isfinite(cls_dev).all()): stop('non-finite gate deviation')
    gates['frozen_metrics_replay_max_abs_diff'] = float(max(rep_dev)); gates['counts_equal_official_total'] = float(max(tot_dev))
    gates['counts_equal_official_by_class'] = float(max(cls_dev)); gates['counts_compared_by_class'] = len(cls_dev)
    gates['used_redundancy_head'] = {m: official[m]['used_redundancy_head'] for m in ORDER}
    gates['realized_drv_within_auc'] = float(np.nanmean(per['realized_drv']['auc']))
    gates['step437_reference'] = {'PRM_native': 0.6545676654529491, 'PRM_raw_q80': 0.6803513244980721, 'PRM_z_q80': 0.6732722285202598,
                                  'within_auc_prm': 0.8011803689223952}
    gates['prm_here'] = {m: 0.5 * (official[m]['total']['f1'] + official[m]['total']['negative_f1']) for m in ('PRM_native', 'PRM_raw_q80', 'PRM_z_q80')}
    gates['prm_here']['within_auc_prm'] = float(np.nanmean(per['PRM_z_q80']['auc']))
    if gates['frozen_metrics_replay_max_abs_diff'] > 1e-12 or gates['counts_equal_official_total'] > 1e-12 or gates['counts_equal_official_by_class'] > 1e-12:
        stop('frozen metrics or official scorer not reproduced')
    if abs(gates['prm_here']['PRM_native'] - 0.6545676654529491) > 1e-12: stop('PRM native threshold not reproduced')
    dump(OUT / 'GATES.json', gates); timing['official_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ strata and bootstrap machinery
    t = time.perf_counter()
    gP = groups[P]; G_u, g_idx = np.unique(gP, return_inverse=True); G = len(G_u)
    strata_main = ['total_noncontrol'] + CLASSES + ['controls']
    def stratum_index(labels_by_answer, names_):
        return np.array([names_.index(x) if x in names_ else -1 for x in labels_by_answer])
    main_lab = np.where(control[P], 'controls', cls[P])
    def aggregate(m, sidx, K):
        d = per[m]; A = {'cnt': np.zeros((G, K, 4)), 'auc_s': np.zeros((G, K)), 'auc_n': np.zeros((G, K)), 'hit_s': np.zeros((G, K)),
                         'hit_n': np.zeros((G, K)), 'det_s': np.zeros((G, K)), 'det_n': np.zeros((G, K))}
        ok = sidx >= 0
        np.add.at(A['cnt'], (g_idx[ok], sidx[ok]), d['cnt'][ok])
        for key, arr in (('auc', d['auc']), ('hit', d['hit']), ('det', d['det'])):
            f = ok & np.isfinite(arr); np.add.at(A[key + '_s'], (g_idx[f], sidx[f]), arr[f]); np.add.at(A[key + '_n'], (g_idx[f], sidx[f]), 1.0)
        return A
    def endpoints_from(Cn, As, An, Hs, Hn, Ds, Dn):   # arrays with a leading draw/point axis and stratum axis
        e = count_metrics(Cn); e['within_auc'] = ratio(As, An); e['any_error_hit'] = ratio(Hs, Hn); e['error_detected'] = ratio(Ds, Dn)
        return np.stack([e[k] for k in ENDPOINTS], -1)   # (..., K, E)
    def with_total(A, K_names, total_members):
        """Append pooled strata (e.g. total over non-control classes) as extra columns."""
        out = {}
        for key, arr in A.items():
            extra = [arr[:, [K_names.index(c) for c in mem]].sum(1, keepdims=True) for mem in total_members]
            out[key] = np.concatenate([arr] + extra, 1)
        return out
    def run_boot(arms, sidx, names_, pooled, draws, seed):
        K = len(names_); agg = {m: with_total(aggregate(m, sidx, K), names_, [p[1] for p in pooled]) for m in arms}
        allnames = names_ + [p[0] for p in pooled]
        point = {m: endpoints_from(agg[m]['cnt'].sum(0), agg[m]['auc_s'].sum(0), agg[m]['auc_n'].sum(0), agg[m]['hit_s'].sum(0),
                                   agg[m]['hit_n'].sum(0), agg[m]['det_s'].sum(0), agg[m]['det_n'].sum(0)) for m in arms}
        rng = np.random.default_rng(seed); boot = {m: np.empty((draws, len(allnames), len(ENDPOINTS)), np.float32) for m in arms}; pos = 0
        while pos < draws:
            nb = min(1000, draws - pos); W = rng.multinomial(G, np.full(G, 1 / G), size=nb).astype(float)
            for m in arms:
                a = agg[m]; e = lambda k: np.einsum('bg,gk->bk', W, a[k])
                boot[m][pos:pos + nb] = endpoints_from(np.einsum('bg,gkc->bkc', W, a['cnt']), e('auc_s'), e('auc_n'), e('hit_s'), e('hit_n'), e('det_s'), e('det_n'))
            pos += nb
        return allnames, point, boot, agg

    main_names = CLASSES + ['controls']
    main_all, point, boot, agg = run_boot(ORDER, stratum_index(main_lab, main_names), main_names,
                                          [('total_noncontrol', CLASSES)] + [(f'{g}__pooled', mem) for g, mem in GROUP3.items()], args.draws, SEED)
    timing['bootstrap_main_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ P1 / P2 tables
    t = time.perf_counter()
    def ci(x, q):
        x = x[np.isfinite(x)]; return (float(np.quantile(x, q[0])), float(np.quantile(x, q[1]))) if len(x) else (np.nan, np.nan)
    rows = []
    for m in ORDER:
        for si, sname in enumerate(main_all):
            for ei, en in enumerate(ENDPOINTS):
                lo, hi = ci(boot[m][:, si, ei].astype(float), (.025, .975))
                rows.append({'arm': m, 'stratum': sname, 'endpoint': en, 'estimate': float(point[m][si, ei]), 'ci95_lo': lo, 'ci95_hi': hi})
    ARMS_T = pd.DataFrame(rows)
    # group3 strata as the official convention: mean of member-class values (MS PRMScore undefined -> skipped)
    for g, mem in GROUP3.items():
        for m in ORDER:
            for ei, en in enumerate(ENDPOINTS):
                idx = [main_all.index(c) for c in mem]
                pv = np.nanmean(point[m][idx, ei]) if np.isfinite(point[m][idx, ei]).any() else np.nan
                bv = np.nanmean(boot[m][:, idx, ei].astype(float), 1) if np.isfinite(point[m][idx, ei]).any() else np.full(args.draws, np.nan)
                lo, hi = ci(bv, (.025, .975))
                ARMS_T.loc[len(ARMS_T)] = {'arm': m, 'stratum': f'{g}__member_mean', 'endpoint': en, 'estimate': float(pv), 'ci95_lo': lo, 'ci95_hi': hi}
    sizes = {}
    for si, sname in enumerate(main_all):
        a = agg['S_equal']; c = a['cnt'].sum(0)[si]
        sizes[sname] = {'answers': None, 'steps': int(c.sum()), 'error_steps': int(c[1] + c[2]),
                        'within_auc_answers': int(a['auc_n'].sum(0)[si]), 'erroneous_answers': int(a['hit_n'].sum(0)[si])}
    for sname in main_names:
        sizes[sname]['answers'] = int((main_lab == sname).sum())
    sizes['total_noncontrol']['answers'] = int(nc.sum())
    for g, mem in GROUP3.items(): sizes[f'{g}__pooled']['answers'] = int(np.isin(main_lab, mem).sum())
    ARMS_T['n_answers'] = ARMS_T.stratum.map(lambda x: sizes.get(x.replace('__member_mean', '__pooled'), {}).get('answers'))
    ARMS_T.to_csv(OUT / 'P2_ARMS_BY_STRATUM.csv', index=False)

    bonf = (.05 / 9 / 2, 1 - .05 / 9 / 2); crow = []
    for (a, b), fam in [(p, 'primary') for p in PRIMARY] + [(p, 'secondary') for p in SECONDARY]:
        for si, sname in enumerate(main_all):
            for ei, en in enumerate(ENDPOINTS):
                d = (boot[a][:, si, ei].astype(float) - boot[b][:, si, ei].astype(float))
                pt = float(point[a][si, ei] - point[b][si, ei]); lo, hi = ci(d, (.025, .975))
                blo, bhi = ci(d, bonf) if sname in CLASSES else (np.nan, np.nan)
                crow.append({'contrast': f'{a} - {b}', 'family': fam, 'stratum': sname, 'endpoint': en, 'delta': pt, 'ci95_lo': lo, 'ci95_hi': hi,
                             'ci_bonf9_lo': blo, 'ci_bonf9_hi': bhi, 'n_answers': sizes.get(sname, {}).get('answers'),
                             'valid_draws': int(np.isfinite(d).sum())})
    CT = pd.DataFrame(crow); CT.to_csv(OUT / 'P2_CONTRASTS.csv', index=False)

    P1 = []
    ti = main_all.index('total_noncontrol')
    for m in ORDER:
        r = {'arm': m, 'name': ROW[m][0], 'kind': ROW[m][1], 'calibration': ROW[m][2]}
        for ei, en in enumerate(ENDPOINTS):
            r[en] = float(point[m][ti, ei]); r[en + '_ci95'] = ci(boot[m][:, ti, ei].astype(float), (.025, .975))
        r['official_prmscore'] = 0.5 * (official[m]['total']['f1'] + official[m]['total']['negative_f1'])
        r['thresholds'] = tau[m]
        P1.append(r)
    timing['tables_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ P3: pooled and cross-answer AUC, decision errors, clean answers
    t = time.perf_counter()
    nc_steps = np.repeat(noncontrol, ns); y_all = labels
    st_ans = np.repeat(np.arange(n), ns)
    gmap = np.full(n, -1); gmap[P] = g_idx; st_group = gmap[st_ans]
    within_pairs_sum = {}; p3 = []
    rng = np.random.default_rng(SEED + 1); Wauc = rng.multinomial(G, np.full(G, 1 / G), size=args.draws_auc).astype(float)
    pooled_draws = {}
    for m in ORDER:
        z = scale[m][nc_steps]; y = y_all[nc_steps]; gs = st_group[nc_steps]
        rk = rankdata(z); P_ = int(y.sum()); N_ = len(y) - P_
        U_pool = rk[y].sum() - P_ * (P_ + 1) / 2
        U_w = 0.0; pairs_w = 0.0
        for i in np.flatnonzero(noncontrol & eligible):
            a, b = off[i:i + 2]; yy = labels[a:b]; n1 = int(yy.sum()); n0 = len(yy) - n1
            U_w += rankdata(scale[m][a:b])[yy].sum() - n1 * (n1 + 1) / 2; pairs_w += n1 * n0
        pooled = U_pool / (P_ * N_); cross = (U_pool - U_w) / (P_ * N_ - pairs_w); within_pair = U_w / pairs_w
        # group-weighted pooled AUC per draw (ties at half weight)
        u, inv = np.unique(z, return_inverse=True)
        Mp = sparse.csr_matrix((np.ones(int(y.sum())), (inv[y], gs[y])), shape=(len(u), G))
        Mn = sparse.csr_matrix((np.ones(int((~y).sum())), (inv[~y], gs[~y])), shape=(len(u), G))
        dr = np.empty(args.draws_auc)
        for s0 in range(0, args.draws_auc, 100):
            Wb = Wauc[s0:s0 + 100].T; Pw = np.asarray(Mp @ Wb); Nw = np.asarray(Mn @ Wb)
            below = np.cumsum(Nw, 0) - Nw
            dr[s0:s0 + 100] = (Pw * (below + 0.5 * Nw)).sum(0) / (Pw.sum(0) * Nw.sum(0))
        pooled_draws[m] = dr
        cn = per[m]['cnt'][nc].sum(0); tp, fp, tn, fn = cn
        def clean(mask):
            sel = mask[P]; nfl = per[m]['nflag_ok'][sel]; nsteps = ns[P][sel]
            return {'answers': int(sel.sum()), 'steps': int(nsteps.sum()), 'step_false_flag_rate': float(nfl.sum() / nsteps.sum()),
                    'share_answers_with_false_flag': float((nfl > 0).mean())}
        err_ans = has_err[P] & nc; err_ok = err_ans & has_ok[P]
        p3.append({'arm': m, 'within_auc': float(np.nanmean(per[m]['auc'])), 'pooled_auc_decision_scale': float(pooled),
                   'pooled_auc_ci95': ci(dr, (.025, .975)), 'cross_answer_auc': float(cross), 'within_auc_pair_weighted': float(within_pair),
                   'share_pairs_within_answer': float(pairs_w / (P_ * N_)), 'share_pairs_cross_answer': float(1 - pairs_w / (P_ * N_)),
                   'decision_scale': 'raw risk' if m in ('PRM_native', 'PRM_raw_q80') else 'answer-z',
                   'false_flags_on_correct_steps': int(fn), 'false_flag_rate_correct_steps': float(fn / (tp + fn)),
                   'missed_error_steps': int(fp), 'miss_rate_error_steps': float(fp / (fp + tn)),
                   'predicted_error_fraction': float((tn + fn) / cn.sum()),
                   'controls_outside_prmscore': clean(control), 'multi_solutions_in_prmscore': clean(ms), 'inert_error_annotation_in_prmscore': clean(inert),
                   'erroneous_answers_correct_steps': {'answers_with_a_correct_step': int(err_ok.sum()),
                                                      'step_false_flag_rate': float(per[m]['nflag_ok'][err_ok].sum() / sum((~labels[off[i]:off[i + 1]]).sum() for i in P[err_ok])),
                                                      'share_answers_with_false_flag_on_a_correct_step': float((per[m]['nflag_ok'][err_ok] > 0).mean())},
                   'mean_flags_per_answer': float(per[m]['nflag'][nc].mean())})
    p3c = []
    for (a, b), fam in [(p, 'primary') for p in PRIMARY] + [(p, 'secondary') for p in SECONDARY]:
        d = pooled_draws[a] - pooled_draws[b]; pa = next(r for r in p3 if r['arm'] == a); pb_ = next(r for r in p3 if r['arm'] == b)
        p3c.append({'contrast': f'{a} - {b}', 'family': fam, 'endpoint': 'pooled_auc_decision_scale',
                    'delta': pa['pooled_auc_decision_scale'] - pb_['pooled_auc_decision_scale'], 'ci95': ci(d, (.025, .975))})
    dump(OUT / 'P3_RANKING_DECISIONS.json', {'arms': p3, 'contrasts': p3c})
    timing['p3_s'] = time.perf_counter() - t

    # ------------------------------------------------------------------ P4 complementarity on erroneous non-control answers
    err_ans = has_err[P] & nc; p4 = []
    for a, b in PAIRS4:
        for key in ('hit', 'det'):
            A_, B_ = per[a][key][err_ans] > 0.5, per[b][key][err_ans] > 0.5; cl = cls[P][err_ans]
            for sname in ['all'] + CLASSES[:-1]:
                sel = np.ones(len(A_), bool) if sname == 'all' else cl == sname
                p4.append({'pair': f'{a} | {b}', 'measure': 'any_error_hit (threshold-free)' if key == 'hit' else 'error step flagged at the frozen threshold',
                           'stratum': sname, 'n': int(sel.sum()), 'both': int((A_ & B_ & sel).sum()), 'only_first': int((A_ & ~B_ & sel).sum()),
                           'only_second': int((~A_ & B_ & sel).sum()), 'neither': int((~A_ & ~B_ & sel).sum())})
    pd.DataFrame(p4).to_csv(OUT / 'P4_COMPLEMENTARITY.csv', index=False)

    # ------------------------------------------------------------------ S1 secondary: length quartiles, first-error position
    t = time.perf_counter()
    q = np.quantile(ns[noncontrol], [.25, .5, .75])
    len_lab = np.array(['' if not nc[j] else f'len_q{int(np.searchsorted(q, ns[i], side="left")) + 1}' for j, i in enumerate(P)])
    len_names = ['len_q1', 'len_q2', 'len_q3', 'len_q4']
    def pos_bin(i):
        f = int(np.flatnonzero(labels[off[i]:off[i + 1]])[0]); rel = f / (ns[i] - 1) if ns[i] > 1 else 0.0
        return 'first_step' if f == 0 else 'early' if rel <= 1 / 3 else 'middle' if rel <= 2 / 3 else 'late'
    pos_lab = np.array([pos_bin(i) if (nc[j] and has_err[i]) else '' for j, i in enumerate(P)])
    pos_names = ['first_step', 'early', 'middle', 'late']
    s1 = []
    for kind, lab_, names_ in (('length_quartile', len_lab, len_names), ('first_error_position', pos_lab, pos_names)):
        alln, pt1, bt1, ag1 = run_boot(S1_ARMS, stratum_index(lab_, names_), names_, [], args.draws, SEED + 2)
        for si, sname in enumerate(alln):
            nans = int((lab_ == sname).sum())
            for m in S1_ARMS:
                for en in ('prmscore', 'within_auc', 'any_error_hit', 'f1_error', 'flag_rate'):
                    ei = ENDPOINTS.index(en); lo, hi = ci(bt1[m][:, si, ei].astype(float), (.025, .975))
                    s1.append({'kind': kind, 'stratum': sname, 'n_answers': nans, 'row': m, 'endpoint': en, 'estimate': float(pt1[m][si, ei]), 'ci95_lo': lo, 'ci95_hi': hi})
            for a, b in S1_CONTRASTS:
                for en in ('prmscore', 'within_auc', 'any_error_hit'):
                    ei = ENDPOINTS.index(en); d = bt1[a][:, si, ei].astype(float) - bt1[b][:, si, ei].astype(float); lo, hi = ci(d, (.025, .975))
                    s1.append({'kind': kind, 'stratum': sname, 'n_answers': nans, 'row': f'{a} - {b}', 'endpoint': en, 'estimate': float(pt1[a][si, ei] - pt1[b][si, ei]), 'ci95_lo': lo, 'ci95_hi': hi})
    pd.DataFrame(s1).to_csv(OUT / 'S1_LENGTH_POSITION.csv', index=False)
    dump(OUT / 'S1_BINS.json', {'length_quartile_edges_steps': q.tolist(), 'length_bins_label_free': True,
                                'length_bin_answers': {b: int((len_lab == b).sum()) for b in len_names},
                                'position_bins': 'first_step: first error at step 0; early: rel <= 1/3; middle: rel <= 2/3; late: rel > 2/3 (rel = first error index / (n-1))',
                                'position_bins_use_labels': 'yes - descriptive stratification only',
                                'position_bin_answers': {b: int((pos_lab == b).sum()) for b in pos_names}, 'seed': SEED + 2})
    timing['s1_s'] = time.perf_counter() - t

    for r in P1:
        q3 = next(x for x in p3 if x['arm'] == r['arm'])
        r.update({k: q3[k] for k in ('pooled_auc_decision_scale', 'pooled_auc_ci95', 'cross_answer_auc', 'within_auc_pair_weighted', 'decision_scale')})
        r['used_redundancy_head'] = official[r['arm']]['used_redundancy_head']
    dump(OUT / 'P1_OVERALL.json', {'rows': P1, 'population': gates['population'], 'strata_sizes': sizes})
    timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
    dump(OUT / 'CODE_MANIFEST.json', {'script': str(Path(__file__).relative_to(ROOT)), 'script_sha256': sha(Path(__file__)),
                                      'prmbench_module_sha256': sha(DEPTH / 'spectral_utils/prmbench.py'), 'protocol_sha256': sha(OUT / 'PROTOCOL.json'),
                                      'thresholds_sha256': {st: sha(THR[st] / 'THRESHOLDS.json') for st in THR}, 'prm_metadata': str(meta_path), 'prm_metadata_sha256': sha(meta_path)})
    status.update({'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds')}); dump(OUT / 'RUN_STATUS.json', status)
    pd.set_option('display.width', 250)
    print(pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ['arm'] + ENDPOINTS[:3] + ['within_auc', 'any_error_hit', 'flag_rate']} for r in P1]).to_string(index=False))
    print(json.dumps(timing, indent=1))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:   # leave a visible FAILED status instead of RUNNING
        dump(OUT / 'RUN_STATUS.json', {'status': 'FAILED', 'error': repr(exc), 'at': datetime.now().isoformat(timespec='seconds')}); raise
