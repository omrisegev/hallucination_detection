"""Stage 2 (labels read here first): parity gate, policy table, success/failure diagnostics, clean detection,
label-chosen tau curve (reference), bootstraps. Rules come frozen from STAGE1.npz (DECLARATION.md)."""
import csv, json
import numpy as np
from scipy.stats import rankdata, spearmanr
from gt_common import *  # noqa
from common_ec import load_meta, labels, evaluate, per_class, prmscore  # noqa: E402
import common as pc  # noqa: E402
from spectral_utils.external_generalization.evaluation import confusion, within_auc  # noqa: E402
from spectral_utils.family_external_metrics import summary  # noqa: E402

Z = dict(np.load(HERE / 'STAGE1.npz'))
POL6 = [f'{t}-{f}' for t in ('T1', 'T2') for f in ('F0', 'FQ', 'F1')]
SEC = [f'{t}-{f}' for t in ('T3', 'DW') for f in ('F0', 'FQ', 'F1')]


def wcsv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(HERE / name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def auroc(pos, s):
    pos = np.asarray(pos, bool); s = np.asarray(s, float); ok = np.isfinite(s); pos, s = pos[ok], s[ok]
    n1 = pos.sum(); n0 = len(pos) - n1
    if not n1 or not n0:
        return np.nan
    return float((rankdata(s)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strat_auroc(pos, s, n, min_n=20):
    vals, ws = [], []
    for m in np.unique(n):
        i = n == m
        if i.sum() >= min_n and 0 < pos[i].sum() < i.sum():
            vals.append(auroc(pos[i], s[i])); ws.append(i.sum())
    if not vals:
        return np.nan, 0
    return float(np.average(vals, weights=ws)), int(np.sum(ws))


def partial_spearman(q, succ, n):
    r = lambda a, b: np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    rqs, rqn, rsn = r(q, succ), r(q, n), r(succ, n)
    return float((rqs - rqn * rsn) / np.sqrt((1 - rqn ** 2) * (1 - rsn ** 2)))


def logit_fit(X, yb, iters=50):
    X = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]); g = X.T @ (yb - p)
        step = np.linalg.solve(H, g); b += step
        if np.max(np.abs(step)) < 1e-10: break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


# =============================== PRMBench ===============================
off = Z['PB__off']; ids = Z['PB__ids']; n_pb = np.diff(off); A = len(n_pb)
meta = load_meta(); y, cnt, nonc, cls, cat = labels(ids, off, meta)
P = pc.load_population(); assert np.array_equal(P['off'], off) and np.array_equal(P['y'], y)
groups = P['group']; sm = np.repeat(nonc, n_pb); aid = np.repeat(np.arange(A), n_pb)
PAR = []
ref = {(r['arm'], r['rule']): float(r['prmscore']) for r in csv.DictReader(open(EC_DIR / 'RESULTS.csv'))}
OF = dict(np.load(OLD_FLAGS))
OLDN = {'B_CHAIN_N16': 'BOOST_CHAIN_N16', 'T_LSML_CONT': 'TELE_LSML_CONT', 'T_ALL_EQ': 'TELE_ALL_EQ_ORIENTED'}
for arm, old in OLDN.items():
    z = Z[f'PB__{arm}__z']
    fl = {'Q80': z > np.quantile(z, .8), 'E1_GAP': topk_flags(z, off, Z[f'PB__{arm}__count__GAP']),
          'K2': topk_flags(z, off, 2), 'ORACLE': topk_flags(z, off, cnt)}
    for r, f in fl.items():
        ps = prmscore(y, f, sm)
        PAR.append(dict(bench='prmbench', arm=arm, rule=r, local=ps, reported=ref[(old, r)], diff=ps - ref[(old, r)],
                        flags_identical_to_old_run=bool(np.array_equal(f, OF[f'{old}__{r}'])),
                        source='results/error_count_label_free_v1/RESULTS.csv'))
    # FQ fallback placed by ranking must reproduce Q80 exactly
    assert np.array_equal(topk_flags(z, off, Z[f'PB__{arm}__Q80count']), fl['Q80'])

# =============================== Socratic ===============================
gold = {g['uid']: g for g in json.loads(SOC_GOLD.read_text(encoding='utf8'))}
socref = {(r['model'], r['arm']): (float(r['official_metric']), float(r['within_auc']))
          for r in csv.DictReader(open(SOC_REF))}
SOCD = {}
for model in ('qwen_prm', 'qwen_prm_72b'):
    S = dict(np.load(SOC_DIR / f'SCORES_{model}.npz'))
    uids = [str(u) for u in S['uids']]; lens = S['nonempty_lengths']; ne = S['nonempty'].astype(bool)
    assert all(len(gold[u]['correct']) == l for u, l in zip(uids, lens))
    corr = np.concatenate([np.asarray(gold[u]['correct'], bool) for u in uids])
    inc = np.concatenate([np.asarray(gold[u]['include'], bool) for u in uids])
    st = np.r_[0, np.cumsum(lens)]; afull = np.repeat(np.arange(len(uids)), lens); nei = np.flatnonzero(ne)
    SOCD[model] = dict(S=S, uids=uids, lens=lens, ne=ne, corr=corr, inc=inc, st=st, afull=afull, nei=nei,
                       cat=np.array([gold[u]['category'] for u in uids]))


def soc_conf(D, flags_ne):
    """Vectorized confusion per answer (POSITIVE = correct step; pred 1 = valid). Empty steps predicted incorrect."""
    p = np.zeros(len(D['corr']), bool); p[D['nei']] = ~flags_ne
    c, i, a = D['corr'], D['inc'], D['afull']; A_ = len(D['uids'])
    return np.stack([np.bincount(a, weights=(i & c & p), minlength=A_), np.bincount(a, weights=(i & ~c & p), minlength=A_),
                     np.bincount(a, weights=(i & ~c & ~p), minlength=A_), np.bincount(a, weights=(i & c & ~p), minlength=A_)],
                    1).astype(np.int64)


def soc_metric(C):
    return 100 * summary(C, 'socratic')['metric']


SOC_ARMS = [f'{m}__{a}' for m in ('qwen_prm', 'qwen_prm_72b') for a in ('CHAIN_N16', 'TELE_N16')]
for key in SOC_ARMS:
    model, arm = key.split('__'); D = SOCD[model]; z = Z[f'SOC__{key}__z']; offn = Z[f'SOC__{key}__off']
    fq = z >= np.quantile(z, .8)
    Cv = soc_conf(D, fq)
    pv = np.zeros(len(D['corr']), int); pv[D['nei']] = (~fq).astype(int)
    full_s = np.full(len(D['corr']), np.nan); full_s[D['nei']] = D['S'][f'score__{arm}']
    Co, Aw = [], []
    for i, u in enumerate(D['uids']):
        a, b = D['st'][i], D['st'][i + 1]
        Co.append(confusion(gold[u]['correct'], pv[a:b], D['inc'][a:b]))
        v = within_auc(gold[u]['correct'], full_s[a:b], D['inc'][a:b] & D['ne'][a:b]); Aw.append(np.nan if v is None else v)
    Co = np.array(Co)
    assert np.array_equal(Co, Cv), 'vectorized confusion mismatch'
    om, ow = summary(Co, 'socratic')['metric'], float(np.nanmean(Aw)); rm, rw = socref[(model, f'{arm}_Q80')]
    PAR.append(dict(bench='socratic', arm=key, rule='Q80', local=100 * om, reported=100 * rm, diff=100 * (om - rm),
                    within_auc_local=ow, within_auc_reported=rw, d_within=ow - rw,
                    source='results/sml_weight_estimator_v1/socratic_sign/RESULTS_socratic_signed.csv'))
    assert np.array_equal(topk_flags(z, offn, Z[f'SOC__{key}__Q80count']), fq)
wcsv('PARITY.csv', PAR)
for r in PAR: print('PARITY', r['bench'], r['arm'], r['rule'], round(r['local'], 4), round(r['reported'], 4), r.get('flags_identical_to_old_run', ''), round(r.get('d_within', 0), 9))
assert all(abs(r['diff']) < 1e-9 for r in PAR) and all(abs(r.get('d_within', 0)) < 1e-9 for r in PAR)
assert all(r.get('flags_identical_to_old_run', True) for r in PAR)
print('PARITY PASS')

# =============================== policies ===============================
ROWS, BEST, DIAG, DEC, LOGIT, CLEAN, TAU, CONTR, PERCAT = [], {}, [], [], [], [], [], [], []
FLAGS = {}


def access_of(r):
    return ('LABEL (ceiling)' if r == 'ORACLE' else 'benchmark prior' if r == 'K2' else
            'label-free reference' if r in ('Q80', 'GAP') else 'label-free (primary family)' if r in POL6 else
            'label-free (secondary, declared)')


for arm in OLDN:
    key = f'PB__{arm}'; z = Z[f'{key}__z']; nn = Z[f'{key}__Q__n']
    rules = {'Q80': z > np.quantile(z, .8), 'GAP': topk_flags(z, off, Z[f'{key}__count__GAP']),
             'K2': topk_flags(z, off, 2), 'ORACLE': topk_flags(z, off, cnt)}
    for r in POL6 + SEC:
        rules[r] = topk_flags(z, off, Z[f'{key}__count__{r}'])
    FLAGS[key] = rules
    for r, f in rules.items():
        e = evaluate(y, cnt, nonc, off, f)
        t = r.split('-')[0]
        ps_ = Z[f'{key}__pass__{t}'] if t in ('T1', 'T2', 'T3', 'DW') else None
        ROWS.append(dict(bench='prmbench', arm=arm, rule=r, metric=e['prmscore'], count_mae=e['count_mae'],
                         mean_pred=e['mean_pred'], mean_true=e['mean_true'], zero_acc=e['zero_acc'],
                         zero_recall=e['zero_recall'], exact_count_acc=float(np.mean(np.add.reduceat(f.astype(int), off[:-1])[nonc] == cnt[nonc])),
                         pass_share_noncontrol=float(ps_[nonc].mean()) if ps_ is not None else np.nan,
                         pass_share_all=float(ps_.mean()) if ps_ is not None else np.nan, access=access_of(r)))
    sc = {r['rule']: r['metric'] for r in ROWS if r['bench'] == 'prmbench' and r['arm'] == arm}
    BEST[key] = max(POL6, key=lambda r: sc[r])

for key in SOC_ARMS:
    model, arm = key.split('__'); D = SOCD[model]; z = Z[f'SOC__{key}__z']; offn = Z[f'SOC__{key}__off']
    err_ne = ~D['corr'][D['nei']]; tc = np.add.reduceat(err_ne.astype(int), offn[:-1])
    rules = {'Q80': z >= np.quantile(z, .8), 'GAP': topk_flags(z, offn, Z[f'SOC__{key}__count__GAP']),
             'K2': topk_flags(z, offn, 2), 'ORACLE': topk_flags(z, offn, tc)}
    for r in POL6 + SEC:
        rules[r] = topk_flags(z, offn, Z[f'SOC__{key}__count__{r}'])
    FLAGS[f'SOC__{key}'] = rules
    for r, f in rules.items():
        k = np.add.reduceat(f.astype(int), offn[:-1]); t = r.split('-')[0]
        ps_ = Z[f'SOC__{key}__pass__{t}'] if t in ('T1', 'T2', 'T3', 'DW') else None
        ROWS.append(dict(bench='socratic', arm=key, rule=r, metric=soc_metric(soc_conf(D, f)),
                         count_mae=float(np.mean(np.abs(k - tc))), mean_pred=float(k.mean()), mean_true=float(tc.mean()),
                         zero_acc=float(np.mean((k == 0) == (tc == 0))), zero_recall=float(np.mean(k[tc == 0] == 0)) if (tc == 0).any() else np.nan,
                         exact_count_acc=float(np.mean(k == tc)),
                         pass_share_noncontrol=float(ps_.mean()) if ps_ is not None else np.nan,
                         pass_share_all=float(ps_.mean()) if ps_ is not None else np.nan, access=access_of(r)))
    sc = {r['rule']: r['metric'] for r in ROWS if r['bench'] == 'socratic' and r['arm'] == key}
    BEST[f'SOC__{key}'] = max(POL6, key=lambda r: sc[r])
for r in ROWS:
    kk = ('PB__' if r['bench'] == 'prmbench' else 'SOC__') + r['arm']
    r['best_of_6_label_selected'] = r['rule'] == BEST[kk]
wcsv('RESULTS.csv', ROWS)

# =============================== diagnostics ===============================
QN = ['G1', 'R1', 'R2', 'neg_p_null', 'neg_p_R1']


def diag_block(bench, arm, Qd, n, k, tp, c, kin, tpin, cin, pop, groups_or_none, rng_seed):
    succ_set = (tpin == kin) & (kin == cin); succ_cnt = k == c
    prec = tp / np.maximum(k, 1); rec = np.where(c > 0, tp / np.maximum(c, 1), np.nan)
    for popname, pm in (('all_n>=3', pop), ('erroneous_only', pop & (c >= 1))):
        for target, sv in (('SUCCESS_SET', succ_set), ('SUCCESS_COUNT', succ_cnt)):
            for q in QN:
                v = Qd[q][pm]; s_ = sv[pm]
                sa, cov = strat_auroc(s_, v, n[pm])
                DIAG.append(dict(bench=bench, arm=arm, population=popname, target=target, quantity=q,
                                 n_answers=int(pm.sum()), success_rate=float(s_.mean()),
                                 median_success=float(np.median(v[s_])) if s_.any() else np.nan,
                                 median_failure=float(np.median(v[~s_])),
                                 auroc_pooled=auroc(s_, v), auroc_exact_n_strat=sa, strat_coverage=cov,
                                 partial_spearman_given_n=partial_spearman(v, s_, n[pm]),
                                 spearman_with_n=float(spearmanr(v, n[pm]).correlation),
                                 success_rate_spearman_n=float(spearmanr(s_, n[pm]).correlation)))
    # deciles (primary population)
    for q in ('G1', 'R1', 'R2', 'neg_p_null'):
        v = Qd[q][pop]; edges = np.quantile(v, np.linspace(0, 1, 11)); d = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, 9)
        for j in range(10):
            i = d == j
            DEC.append(dict(bench=bench, arm=arm, quantity=q, decile=j + 1, lo=edges[j], hi=edges[j + 1], n_answers=int(i.sum()),
                            success_set_rate=float(succ_set[pop][i].mean()), pooled_flag_precision=float(tp[pop][i].sum() / k[pop][i].sum()),
                            mean_recall=float(np.nanmean(rec[pop][i])), median_n=float(np.median(n[pop][i])),
                            mean_kstar=float(k[pop][i].mean()), mean_true=float(c[pop][i].mean())))
    # logistic (Amendment A)
    pm = pop
    for second in ('R1', 'R2'):
        X = np.column_stack([np.log(n[pm]), np.log(Qd['G1'][pm]), np.log(Qd[second][pm])])
        X = (X - X.mean(0)) / X.std(0)
        b, se = logit_fit(X, succ_set[pm].astype(float))
        LOGIT.append(dict(bench=bench, arm=arm, model=f'SUCCESS_SET ~ z(log n) + z(log G1) + z(log {second})',
                          b_logn=b[1], z_logn=b[1] / se[1], b_logG1=b[2], z_logG1=b[2] / se[2],
                          b_second=b[3], z_second=b[3] / se[3], n_answers=int(pm.sum())))
    # bootstrap stratified AUROC differences (primary population, SUCCESS_SET)
    rng = np.random.default_rng(rng_seed); idx = np.flatnonzero(pop)
    if groups_or_none is not None:
        gu, gi = np.unique(groups_or_none[idx], return_inverse=True)
    dif = {'G1-R1': [], 'G1-R2': []}
    for _ in range(1000):
        if groups_or_none is not None:
            mult = np.bincount(rng.integers(0, len(gu), len(gu)), minlength=len(gu))[gi]
            rep = np.repeat(idx, mult)
        else:
            rep = idx[rng.integers(0, len(idx), len(idx))]
        aa = {q: strat_auroc(succ_set[rep], Qd[q][rep], n[rep])[0] for q in ('G1', 'R1', 'R2')}
        dif['G1-R1'].append(aa['G1'] - aa['R1']); dif['G1-R2'].append(aa['G1'] - aa['R2'])
    base = {q: strat_auroc(succ_set[pop], Qd[q][pop], n[pop])[0] for q in ('G1', 'R1', 'R2')}
    out = {}
    for kd, v in dif.items():
        a_, b_ = kd.split('-'); v = np.asarray(v)
        out[kd] = (base[a_] - base[b_], float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975)))
    return out, succ_set, prec, rec


BOOTDIAG = []
for arm in OLDN:
    key = f'PB__{arm}'
    Qd = {q: Z[f'{key}__Q__{q}'] for q in ('G1', 'R1', 'R2', 'n', 'kstar')}
    Qd['neg_p_null'] = -Z[f'{key}__Q__p_null']; Qd['neg_p_R1'] = -Z[f'{key}__Q__p_R1']
    fl = FLAGS[key]['GAP']; k = Qd['kstar']; tp = np.add.reduceat((fl & y).astype(int), off[:-1])
    pop = nonc & (n_pb >= 3)
    o, succ_set, prec, rec = diag_block('prmbench', arm, Qd, n_pb, k, tp, cnt, k, tp, cnt, pop, groups, 20261021)
    for kd, (d, lo, hi) in o.items():
        BOOTDIAG.append(dict(bench='prmbench', arm=arm, contrast=f'strat AUROC {kd}', diff=d, lo=lo, hi=hi, resample='source groups'))
    # clean-answer detection
    clean = cnt == 0
    for q, s_ in (('p_null', Z[f'{key}__Q__p_null']), ('neg_G1', -Qd['G1']), ('neg_R1', -Qd['R1']), ('neg_R2', -Qd['R2']),
                  ('p_R1', Z[f'{key}__Q__p_R1'])):
        sa, cov = strat_auroc(clean[pop], s_[pop], n_pb[pop])
        CLEAN.append(dict(bench='prmbench', arm=arm, quantity=q, auroc_clean_pooled=auroc(clean[pop], s_[pop]),
                          auroc_clean_exact_n_strat=sa, strat_coverage=cov, n_clean=int(clean[pop].sum()), n_answers=int(pop.sum())))
    for t in ('T1', 'T2', 'T3', 'DW'):
        pm_ = Z[f'{key}__pass__{t}']
        CLEAN.append(dict(bench='prmbench', arm=arm, quantity=f'pass_{t}',
                          pass_rate_clean=float(pm_[nonc & clean].mean()), pass_rate_multi_solutions=float(pm_[nonc & (cls == 'multi_solutions')].mean()),
                          pass_rate_erroneous=float(pm_[nonc & ~clean].mean()), n_clean=int((nonc & clean).sum())))
    # label-chosen tau curve (reference)
    G1 = Qd['G1']; taus = np.nanquantile(G1[pop], np.arange(0, .99, .02)); q80c = Z[f'{key}__Q80count']
    for qi, tau in zip(np.arange(0, .99, .02), taus):
        pm_ = (n_pb >= 3) & (np.nan_to_num(G1, nan=-1) >= tau)
        for fb in ('F0', 'FQ', 'F1'):
            kk = policy_counts({'kstar': k}, pm_, fb, q80c)
            TAU.append(dict(bench='prmbench', arm=arm, tau_quantile=round(float(qi), 2), tau=float(tau), fallback=fb,
                            metric=prmscore(y, topk_flags(Z[f'{key}__z'], off, kk), sm), pass_share_noncontrol=float(pm_[nonc].mean())))
    # per-category
    for r in ('Q80', 'GAP', 'K2', 'ORACLE', 'T1-F0', BEST[key]):
        for c_, v in per_class(y, cnt, nonc, off, FLAGS[key][r], cls).items():
            PERCAT.append(dict(bench='prmbench', arm=arm, rule=r + (' (best)' if r == BEST[key] else ''), category=c_, **v))

for key in SOC_ARMS:
    model, arm = key.split('__'); D = SOCD[model]; offn = Z[f'SOC__{key}__off']; nn = np.diff(offn)
    Qd = {q: Z[f'SOC__{key}__Q__{q}'] for q in ('G1', 'R1', 'R2', 'n', 'kstar')}
    Qd['neg_p_null'] = -Z[f'SOC__{key}__Q__p_null']; Qd['neg_p_R1'] = -Z[f'SOC__{key}__Q__p_R1']
    err = ~D['corr'][D['nei']]; incn = D['inc'][D['nei']]
    fl = FLAGS[f'SOC__{key}']['GAP']; k = Qd['kstar']
    R = lambda x: np.add.reduceat(x.astype(int), offn[:-1])
    tp, c = R(fl & err), R(err); kin, tpin, cin = R(fl & incn), R(fl & err & incn), R(err & incn)
    pop = nn >= 3
    o, *_ = diag_block('socratic', key, Qd, nn, k, tp, c, kin, tpin, cin, pop, None, 20261018)
    for kd, (d, lo, hi) in o.items():
        BOOTDIAG.append(dict(bench='socratic', arm=key, contrast=f'strat AUROC {kd}', diff=d, lo=lo, hi=hi, resample='answers'))
    clean = c == 0
    CLEAN.append(dict(bench='socratic', arm=key, quantity='p_null', auroc_clean_pooled=auroc(clean[pop], Z[f'SOC__{key}__Q__p_null'][pop]),
                      n_clean=int(clean[pop].sum()), n_answers=int(pop.sum())))
    G1 = Qd['G1']; taus = np.nanquantile(G1[pop], np.arange(0, .99, .02)); q80c = Z[f'SOC__{key}__Q80count']
    for qi, tau in zip(np.arange(0, .99, .02), taus):
        pm_ = (nn >= 3) & (np.nan_to_num(G1, nan=-1) >= tau)
        for fb in ('F0', 'FQ', 'F1'):
            kk = policy_counts({'kstar': k}, pm_, fb, q80c)
            TAU.append(dict(bench='socratic', arm=key, tau_quantile=round(float(qi), 2), tau=float(tau), fallback=fb,
                            metric=soc_metric(soc_conf(D, topk_flags(Z[f'SOC__{key}__z'], offn, kk))), pass_share_noncontrol=float(pm_.mean())))
    for r in ('Q80', 'GAP', 'ORACLE', BEST[f'SOC__{key}']):
        f = FLAGS[f'SOC__{key}'][r]; C = soc_conf(D, f)
        for c_ in sorted(set(D['cat'])):
            PERCAT.append(dict(bench='socratic', arm=key, rule=r + (' (best)' if r == BEST[f'SOC__{key}'] else ''), category=c_,
                               prmscore=soc_metric(C[D['cat'] == c_]), n=int((D['cat'] == c_).sum())))

# =============================== policy bootstraps ===============================
gu, ginv = np.unique(groups, return_inverse=True)
rng = np.random.default_rng(20261021); B = 2000
W = np.stack([np.bincount(rng.integers(0, len(gu), len(gu)), minlength=len(gu)) for _ in range(B)])


def conf_by_group(fl):
    parts = [(~y & ~fl & sm), (y & ~fl & sm), (y & fl & sm), (~y & fl & sm)]
    return np.stack([np.bincount(ginv[aid], weights=p.astype(float), minlength=len(gu)) for p in parts], 1)


def ps_from(c):
    tp, fp, tn, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return 100 * .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))


for arm in OLDN:
    key = f'PB__{arm}'; g = {r: conf_by_group(FLAGS[key][r]) for r in ('GAP', 'Q80', 'K2', 'T1-F0', BEST[key])}
    for a_, b_ in [('GAP', 'Q80'), ('T1-F0', 'GAP'), ('T1-F0', 'Q80'), (BEST[key], 'GAP'), (BEST[key], 'Q80'), (BEST[key], 'K2')]:
        d = ps_from(g[a_].sum(0)) - ps_from(g[b_].sum(0)); db = ps_from(W @ g[a_]) - ps_from(W @ g[b_])
        CONTR.append(dict(bench='prmbench', arm=arm, contrast=f'{a_} - {b_}', best_label_selected=a_ == BEST[key],
                          diff=d, lo=np.quantile(db, .025), hi=np.quantile(db, .975), resample='source groups, 2000'))
rng = np.random.default_rng(20261018)
for key in SOC_ARMS:
    model = key.split('__')[0]; D = SOCD[model]; A_ = len(D['uids'])
    Wm = np.stack([np.bincount(rng.integers(0, A_, A_), minlength=A_) for _ in range(B)])
    bk_ = BEST[f'SOC__{key}']; Cs = {r: soc_conf(D, FLAGS[f'SOC__{key}'][r]).astype(float) for r in ('GAP', 'Q80', 'T1-F0', bk_)}
    for a_, b_ in [('GAP', 'Q80'), ('T1-F0', 'GAP'), ('T1-F0', 'Q80'), (bk_, 'GAP'), (bk_, 'Q80')]:
        d = soc_metric(Cs[a_].sum(0, keepdims=True).astype(np.int64)) - soc_metric(Cs[b_].sum(0, keepdims=True).astype(np.int64))
        db = ps_from(Wm @ Cs[a_]) - ps_from(Wm @ Cs[b_])
        CONTR.append(dict(bench='socratic', arm=key, contrast=f'{a_} - {b_}', best_label_selected=a_ == bk_,
                          diff=d, lo=np.quantile(db, .025), hi=np.quantile(db, .975), resample='answers, 2000'))

for name, rows in (('DIAGNOSTICS.csv', DIAG), ('DECILES.csv', DEC), ('LOGIT.csv', LOGIT), ('CLEAN.csv', CLEAN),
                   ('TAU_CURVE.csv', TAU), ('CONTRASTS.csv', CONTR), ('PER_CATEGORY.csv', PERCAT), ('BOOT_DIAG.csv', BOOTDIAG)):
    wcsv(name, rows)
(HERE / 'BEST.json').write_text(json.dumps(BEST, indent=1))
print('BEST', BEST)
for r in ROWS:
    print(f"{r['bench']:9s} {r['arm']:26s} {r['rule']:7s} PS={r['metric']:.2f} MAE={r['count_mae']:.3f} pass={r['pass_share_noncontrol']:.3f} zacc={r['zero_acc']:.3f}")
for r in CONTR: print('CONTRAST', r['bench'], r['arm'], r['contrast'], f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
for r in BOOTDIAG: print('BOOTDIAG', r['bench'], r['arm'], r['contrast'], f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
