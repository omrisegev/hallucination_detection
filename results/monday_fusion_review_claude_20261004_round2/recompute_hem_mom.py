"""Independent recompute of the highlighted HEM_MOM input comparison and its controls.
Reads only sealed per-answer outcome arrays (chunks/*.stats.npz) and the frozen answers table.
Own aggregation code, own bootstrap seed, own label nulls. No refit, no new arm."""
import json, datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent
RUN = {'input': ROOT / 'results/estimator_input_v1/run_20261004_resume', 'level': ROOT / 'results/level_reduction_v1/run_20261004_resume'}
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
VARIANTS = ['groups_answer', 'channels_answer', 'groups_cell', 'channels_cell']
RECIPES = ['HEM_MOM', 'MEAN_MOM', 'HEM_EQ', 'MEAN_EQ', 'DS_FILTER_EQ', 'ALL_EQ', 'HEM_DS', 'MEAN_DS']
DRAWS = 20000; SEED = 7_2026_1004; FAMILY = 40
man = json.loads((RUN['level'] / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
answers = pd.read_csv(man['inputs']['oof_answers']['path'], usecols=['id', 'uid', 'cell', 'source_group', 'fold', 'target'])
offsets = np.load(man['inputs']['joined']['path'])['offsets']; ns = np.diff(offsets)
assert len(answers) == 13769 and len(ns) == 13769
metrics = pd.read_csv(RUN['input'] / 'METRICS.csv')
metrics = metrics[(metrics.position_variant == 'CONTENT') & (metrics.decision_rule == 'ANSWER_Z_Q80')]
pb_cells = sorted(c for c in answers.cell.unique() if c.startswith('pb_')); prm_cell = 'prmbench_qwen3_8b'
report = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'draws': DRAWS, 'seed': SEED, 'family_size': FAMILY, 'scope': {}}

# ---------------- ProcessBench: per-answer outcomes ----------------
pb_idx = np.flatnonzero(answers.cell.str.startswith('pb_').to_numpy())
pbA = answers.iloc[pb_idx].reset_index(drop=True); target = pbA.target.to_numpy(); err = target >= 0
groups, gi = np.unique(pbA.source_group, return_inverse=True); cells, ci = np.unique(pbA.cell, return_inverse=True)
local = {int(a): j for j, a in enumerate(pb_idx)}
G, C, B = len(groups), len(cells), len(BANKS)
report['scope']['pb'] = {'answers': int(len(pbA)), 'erroneous': int(err.sum()), 'clean': int((~err).sum()), 'source_groups': int(G), 'cells': int(C), 'banks': B}
# outcomes[variant][recipe] -> dict of per-answer x bank arrays
outcomes = {}; metric_check = []
for variant in VARIANTS:
    outcomes[variant] = {}
    for recipe in RECIPES:
        hit_arg = np.zeros((len(pbA), B)); hit_first = np.zeros((len(pbA), B)); clean_flag = np.zeros((len(pbA), B)); clean_ok = np.zeros((len(pbA), B))
        for b, bk in enumerate(BANKS):
            for ce in cells:
                with np.load(RUN['input'] / 'chunks' / f'{ce}__{bk}__{variant}.stats.npz') as z:
                    ai = z['answer_indices']; li = np.array([local[int(a)] for a in ai])
                    assert (pbA.cell.to_numpy()[li] == ce).all()
                    t = target[li]
                    hit_arg[li, b] = z[recipe + '__ANSWER_Z_Q80__argmax'] == t
                    hit_first[li, b] = z[recipe + '__ANSWER_Z_Q80__first'] == t
                    clean_flag[li, b] = z[recipe + '__ANSWER_Z_Q80__flag_count'] > 0
                    clean_ok[li, b] = z[recipe + '__ANSWER_Z_Q80__first'] == -1
                    e = err[li]
                    m = metrics[(metrics.bank == bk) & (metrics.variant == variant) & (metrics.recipe == recipe) & (metrics.cell == ce)].iloc[0]
                    ae = hit_first[li, b][e].mean(); ac = clean_ok[li, b][~e].mean(); f1 = 2 * ae * ac / (ae + ac) if ae + ac > 0 else 0.
                    metric_check.append({'variant': variant, 'recipe': recipe, 'bank': bk, 'cell': ce,
                                         'd_first_error': abs(hit_arg[li, b][e].mean() - m.pb_first_error_accuracy), 'd_f1': abs(f1 - m.pb_official_f1),
                                         'd_clean_flag': abs(clean_flag[li, b][~e].mean() - m.clean_flag_rate), 'd_decision_acc': abs(ae - m.pb_decision_error_accuracy),
                                         'coverage': m.coverage, 'failures': m.failures, 'n_answers': m.n_answers})
        outcomes[variant][recipe] = {'hit_arg': hit_arg, 'hit_first': hit_first, 'clean_flag': clean_flag, 'clean_ok': clean_ok}
mc = pd.DataFrame(metric_check); mc.to_csv(OUT / 'PB_METRICS_RECHECK.csv', index=False)
report['pb_metrics_recheck'] = {'rows': int(len(mc)), 'max_abs_difference': float(mc[['d_first_error', 'd_f1', 'd_clean_flag', 'd_decision_acc']].max().max()),
                                'all_coverage_1': bool((mc.coverage == 1).all()), 'all_failures_0': bool((mc.failures == 0).all())}

# per-group sufficient statistics: S[g, c, b, k] with k in (argmax hits, first-flag hits, clean no-flag, clean flagged); denominators D[g, c, 0]=n_err, D[g,c,1]=n_clean
def suff(o):
    S = np.zeros((G, C, B, 4))
    for b in range(B):
        np.add.at(S[:, :, b, 0], (gi[err], ci[err]), o['hit_arg'][err, b])
        np.add.at(S[:, :, b, 1], (gi[err], ci[err]), o['hit_first'][err, b])
        np.add.at(S[:, :, b, 2], (gi[~err], ci[~err]), o['clean_ok'][~err, b])
        np.add.at(S[:, :, b, 3], (gi[~err], ci[~err]), o['clean_flag'][~err, b])
    return S
D = np.zeros((G, C, 2)); np.add.at(D[:, :, 0], (gi[err], ci[err]), 1); np.add.at(D[:, :, 1], (gi[~err], ci[~err]), 1)
SUFF = {(v, r): suff(outcomes[v][r]) for v in VARIANTS for r in RECIPES}

def pb_endpoints(S, Dw):
    """S: (..., C, B, 4) weighted sums; Dw: (..., C, 2). Returns dict of equal-bank, equal-cell means."""
    n_err = Dw[..., 0][..., None]; n_clean = Dw[..., 1][..., None]
    acc_arg = S[..., 0] / n_err; ae = S[..., 1] / n_err; ac = S[..., 2] / n_clean; cf = S[..., 3] / n_clean
    f1 = np.where(ae + ac > 0, 2 * ae * ac / np.where(ae + ac > 0, ae + ac, 1), 0.)
    return {'pb_first_error_argmax': acc_arg.mean((-1, -2)), 'pb_first_flag_accuracy': ae.mean((-1, -2)), 'pb_official_f1': f1.mean((-1, -2)), 'clean_flag_rate': cf.mean((-1, -2))}
point = {k: pb_endpoints(S.sum(0), D.sum(0)) for k, S in SUFF.items()}
# bootstrap over PB source groups, shared draws
rng = np.random.default_rng(SEED); keys = list(SUFF); flat = np.stack([SUFF[k].reshape(G, -1) for k in keys], 1)  # G x K x (C*B*4)
boot = {k: {e: np.empty(DRAWS) for e in point[k]} for k in keys}
for s in range(0, DRAWS, 500):
    n = min(500, DRAWS - s); w = rng.multinomial(G, np.full(G, 1 / G), size=n).astype(float)
    Dw = (w @ D.reshape(G, -1)).reshape(n, C, 2)
    for j, k in enumerate(keys):
        Sw = (w @ flat[:, j, :]).reshape(n, C, B, 4)
        for e, v in pb_endpoints(Sw, Dw).items(): boot[k][e][s:s + n] = v
def contrast(a, b, e, fam=FAMILY):
    d = boot[a][e] - boot[b][e]; lo, hi = np.quantile(d, [.025, .975]); blo, bhi = np.quantile(d, [.025 / fam, 1 - .025 / fam])
    return {'delta': float(point[a][e] - point[b][e]), 'lo95': float(lo), 'hi95': float(hi), 'lo_bonferroni_40': float(blo), 'hi_bonferroni_40': float(bhi),
            'bootstrap_sd': float(d.std()), 'p_two_sided_sign_flip': float(2 * min((d <= 0).mean(), (d >= 0).mean()))}
pb_contrasts = {}
pairs = [('HEM_MOM channels_cell vs groups_answer (both changes)', ('channels_cell', 'HEM_MOM'), ('groups_answer', 'HEM_MOM')),
         ('HEM_MOM channels_answer vs groups_answer (channel votes only)', ('channels_answer', 'HEM_MOM'), ('groups_answer', 'HEM_MOM')),
         ('HEM_MOM channels_cell vs channels_answer (cell quota only)', ('channels_cell', 'HEM_MOM'), ('channels_answer', 'HEM_MOM')),
         ('HEM_MOM groups_cell vs groups_answer (cell quota, group votes)', ('groups_cell', 'HEM_MOM'), ('groups_answer', 'HEM_MOM')),
         ('HEM_MOM channels_cell vs DS_FILTER_EQ original (plain filtered mean)', ('channels_cell', 'HEM_MOM'), ('groups_answer', 'DS_FILTER_EQ')),
         ('HEM_MOM channels_cell vs ALL_EQ original', ('channels_cell', 'HEM_MOM'), ('groups_answer', 'ALL_EQ')),
         ('HEM_MOM channels_cell vs HEM_EQ original (matched EQ)', ('channels_cell', 'HEM_MOM'), ('groups_answer', 'HEM_EQ')),
         ('HEM_MOM groups_answer vs DS_FILTER_EQ original', ('groups_answer', 'HEM_MOM'), ('groups_answer', 'DS_FILTER_EQ')),
         ('MEAN_MOM channels_cell vs groups_answer (both changes)', ('channels_cell', 'MEAN_MOM'), ('groups_answer', 'MEAN_MOM')),
         ('MEAN_MOM channels_cell vs DS_FILTER_EQ original', ('channels_cell', 'MEAN_MOM'), ('groups_answer', 'DS_FILTER_EQ')),
         ('HEM_DS channels_cell vs groups_answer', ('channels_cell', 'HEM_DS'), ('groups_answer', 'HEM_DS')),
         ('MEAN_DS channels_cell vs groups_answer', ('channels_cell', 'MEAN_DS'), ('groups_answer', 'MEAN_DS'))]
for name, a, b in pairs:
    pb_contrasts[name] = {e: contrast(a, b, e) for e in point[a]}
report['pb_contrasts'] = pb_contrasts
report['pb_absolute'] = {f'{v}/{r}': {e: float(x) for e, x in point[(v, r)].items()} for v, r in keys}
# per-cell and per-bank breakdown of the highlighted argmax delta
S1, S0 = SUFF[('channels_cell', 'HEM_MOM')].sum(0), SUFF[('groups_answer', 'HEM_MOM')].sum(0); Dn = D.sum(0)
cellbank = (S1[:, :, 0] - S0[:, :, 0]) / Dn[:, 0][:, None]
report['pb_argmax_delta_breakdown'] = {'per_cell_mean_over_banks': {c: float(v) for c, v in zip(cells, cellbank.mean(1))},
                                       'per_bank_mean_over_cells': {b: float(v) for b, v in zip(BANKS, cellbank.mean(0))},
                                       'cell_bank_positive': int((cellbank > 0).sum()), 'cell_bank_total': int(cellbank.size),
                                       'n_erroneous_per_cell': {c: int(v) for c, v in zip(cells, Dn[:, 0])}}
report['pb_absolute_per_bank_HEM_MOM'] = {v: {b: float(x) for b, x in zip(BANKS, (SUFF[(v, 'HEM_MOM')].sum(0)[:, :, 0] / Dn[:, 0][:, None]).mean(0))} for v in ['groups_answer', 'channels_cell']}
report['pb_absolute_per_bank_DS_FILTER_EQ'] = {b: float(x) for b, x in zip(BANKS, (SUFF[('groups_answer', 'DS_FILTER_EQ')].sum(0)[:, :, 0] / Dn[:, 0][:, None]).mean(0))}

# ---------------- PB label nulls for the highlighted delta (argmax arrays fixed, targets randomized) ----------------
arg = {}
for variant in ['channels_cell', 'groups_answer']:
    A = np.zeros((len(pbA), B), int)
    for b, bk in enumerate(BANKS):
        for ce in cells:
            with np.load(RUN['input'] / 'chunks' / f'{ce}__{bk}__{variant}.stats.npz') as z:
                li = np.array([local[int(a)] for a in z['answer_indices']]); A[li, b] = z['HEM_MOM__ANSWER_Z_Q80__argmax']
    arg[variant] = A
def acc_delta(t):
    e = t >= 0; out = []
    for variant in ['channels_cell', 'groups_answer']:
        h = (arg[variant] == t[:, None]) & e[:, None]
        num = np.zeros((C, B)); np.add.at(num, ci[e], h[e]); den = np.bincount(ci[e], minlength=C).astype(float)
        out.append((num / den[:, None]).mean())
    return out[0] - out[1], out[0], out[1]
observed = acc_delta(target); pb_ns = ns[pb_idx]
nulls = {}
for mode, seed in [('within_answer_shuffle', 11), ('whole_answer_same_length_swap', 12)]:
    r = np.random.default_rng(seed); sample = []
    for _ in range(200):
        t = target.copy()
        if mode == 'within_answer_shuffle': t[err] = r.integers(0, pb_ns[err])
        else:
            for c in range(C):
                for length in np.unique(pb_ns[ci == c]):
                    ix = np.flatnonzero((ci == c) & (pb_ns == length))
                    if len(ix) > 1: perm = r.permutation(ix); t[perm] = target[np.roll(perm, 1)]
        sample.append(acc_delta(t))
    sample = np.asarray(sample)
    nulls[mode] = {'delta_null_mean': float(sample[:, 0].mean()), 'delta_null_sd': float(sample[:, 0].std()), 'delta_null_max_abs': float(np.abs(sample[:, 0]).max()),
                   'p_plus_one_delta_ge_observed': float((1 + (sample[:, 0] >= observed[0]).sum()) / 201),
                   'channels_cell_null_mean_accuracy': float(sample[:, 1].mean()), 'groups_answer_null_mean_accuracy': float(sample[:, 2].mean()),
                   'p_plus_one_channels_cell_accuracy': float((1 + (sample[:, 1] >= observed[1]).sum()) / 201), 'permutations': 200}
report['pb_nulls_HEM_MOM'] = {'observed_delta': float(observed[0]), 'observed_channels_cell': float(observed[1]), 'observed_groups_answer': float(observed[2]), **nulls}

# ---------------- PRMBench within-answer AUC ----------------
prm_idx = np.flatnonzero((answers.cell == prm_cell).to_numpy()); prmA = answers.iloc[prm_idx].reset_index(drop=True)
pg, pgi = np.unique(prmA.source_group, return_inverse=True); plocal = {int(a): j for j, a in enumerate(prm_idx)}
auc = {}; eligible = None
for variant in VARIANTS:
    for recipe in RECIPES:
        M = np.full((len(prmA), B), np.nan)
        for b, bk in enumerate(BANKS):
            with np.load(RUN['input'] / 'chunks' / f'{prm_cell}__{bk}__{variant}.stats.npz') as z:
                li = np.array([plocal[int(a)] for a in z['answer_indices']]); M[li, b] = z[recipe + '__ANSWER_Z_Q80__auc']
        fin = np.isfinite(M).all(1)
        if eligible is None: eligible = fin
        assert np.array_equal(fin, eligible) and np.array_equal(np.isfinite(M).any(1), eligible)
        auc[(variant, recipe)] = M.mean(1)
        m = metrics[(metrics.variant == variant) & (metrics.recipe == recipe) & (metrics.cell == prm_cell)].set_index('bank').within_answer_auc
        assert max(abs(np.nanmean(M[eligible, b]) - m[bk]) for b, bk in enumerate(BANKS)) < 1e-12
report['scope']['prm'] = {'answers': int(len(prmA)), 'eligible_auc_answers': int(eligible.sum()), 'source_groups': int(len(pg)), 'eligible_source_groups': int(len(np.unique(pgi[eligible])))}
PG = len(pg); cnt = np.bincount(pgi[eligible], minlength=PG).astype(float)
sums = {k: np.bincount(pgi[eligible], weights=v[eligible], minlength=PG) for k, v in auc.items()}
rng = np.random.default_rng(SEED + 1); pkeys = list(sums); mat = np.stack([sums[k] for k in pkeys], 1); pboot = np.empty((DRAWS, len(pkeys)))
for s in range(0, DRAWS, 1000):
    n = min(1000, DRAWS - s); w = rng.multinomial(PG, np.full(PG, 1 / PG), size=n).astype(float)
    pboot[s:s + n] = (w @ mat) / (w @ cnt)[:, None]
ppoint = {k: float(sums[k].sum() / cnt.sum()) for k in pkeys}
def pcontrast(a, b):
    d = pboot[:, pkeys.index(a)] - pboot[:, pkeys.index(b)]; lo, hi = np.quantile(d, [.025, .975]); blo, bhi = np.quantile(d, [.025 / FAMILY, 1 - .025 / FAMILY])
    return {'delta': ppoint[a] - ppoint[b], 'lo95': float(lo), 'hi95': float(hi), 'lo_bonferroni_40': float(blo), 'hi_bonferroni_40': float(bhi), 'bootstrap_sd': float(d.std())}
report['prm_contrasts'] = {name: pcontrast(a, b) for name, a, b in pairs}
report['prm_absolute'] = {f'{v}/{r}': ppoint[(v, r)] for v, r in pkeys}
report['prm_per_bank_HEM_MOM'] = {v: {bk: float(metrics[(metrics.variant == v) & (metrics.recipe == 'HEM_MOM') & (metrics.cell == prm_cell) & (metrics.bank == bk)].within_answer_auc.iloc[0]) for bk in BANKS} for v in ['groups_answer', 'channels_cell']}
# PRMScore from confusion counts (q80 decision), equal banks
def prmscore(c):
    tp, fp, tn, fn = c
    return .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))
ps = {}
for variant in ['groups_answer', 'channels_cell']:
    vals = []
    for bk in BANKS:
        with np.load(RUN['input'] / 'chunks' / f'{prm_cell}__{bk}__{variant}.stats.npz') as z: vals.append(prmscore(z['HEM_MOM__ANSWER_Z_Q80__confusion'].sum(0)))
    ps[variant] = float(np.mean(vals))
report['prm_prmscore_HEM_MOM_q80'] = {**ps, 'delta_channels_cell_minus_groups_answer': ps['channels_cell'] - ps['groups_answer']}
# Codex reference values for comparison
codex = pd.read_csv(ROOT / 'results/monday_fusion_review_response_20261004/DIRECT_CONTRASTS.csv')
q = codex[(codex.recipe == 'HEM_MOM') & (codex.comparison == 'both_changes')]
report['codex_reference'] = {r.benchmark: {'delta': r.delta, 'lo95': r.lo95, 'hi95': r.hi95, 'lo_bonferroni': r.lo_bonferroni, 'hi_bonferroni': r.hi_bonferroni} for r in q.itertuples()}
(OUT / 'HEM_MOM_RECOMPUTE.json').write_text(json.dumps(report, indent=1, default=float), encoding='utf8')
print(json.dumps({k: report[k] for k in ['scope', 'pb_metrics_recheck', 'pb_nulls_HEM_MOM', 'prm_prmscore_HEM_MOM_q80', 'codex_reference', 'pb_argmax_delta_breakdown']}, indent=1, default=float))
for name in pb_contrasts:
    print('PB ', name); print('    ', {e: {k: round(v, 6) for k, v in d.items() if k in ('delta', 'lo95', 'hi95', 'lo_bonferroni_40', 'hi_bonferroni_40')} for e, d in pb_contrasts[name].items()})
    print('PRM', {k: round(v, 6) for k, v in report['prm_contrasts'][name].items() if k in ('delta', 'lo95', 'hi95', 'lo_bonferroni_40', 'hi_bonferroni_40')})
print('PB absolute:', {k: {e: round(x, 5) for e, x in v.items()} for k, v in report['pb_absolute'].items()})
print('PRM absolute:', {k: round(v, 6) for k, v in report['prm_absolute'].items()})
