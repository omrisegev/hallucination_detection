"""CAT34 vs Z17 on Socratic-PRMBench (local CPU). Same arms, transforms and L-SML as ../auto_raw_plus_z.py.

Inputs and code path are those of the stored Socratic AUTO_N16 (results/sml_weight_estimator_v1/socratic_sign/
prm_placement.py, mode 'signed' = shipped SML): bank_v2 FEATURES.npz + REWARDS.npz per PRM (PRM-backbone
telemetry, mask1), the frozen adapter's N16 answer-z bank, z(PRM) = answer-z of logit(1 - reward), and
lsml_continuous_guarded (constant views dropped; identical to lsml_continuous otherwise) with the fused score
oriented toward z(PRM). Scoring is results/prm_boost_ablation5_v1/eval_socratic_ablation5.py: official Socratic
PRMScore = (F1_correct + F1_error)/2 over included steps (0-1 scale), ANSWER_Z_Q80 on non-empty steps, empty
steps predicted incorrect, within-answer AUC on included non-empty steps.
  Z17    = AUTO_N16_signed (parity: must equal the stored RESULTS_socratic_placement.csv row exactly)
  CAT34  = Z17's 17 views + the same 17 channels through a global rank -> normal transform over all non-empty
           Socratic steps of that PRM (digit channels over active steps, 0 elsewhere)
  PRM_native, PRMPROB_Q80 (= PRM_Q80_PROB), POS_ONLY (relative step position, Q80) as references.
Labels enter only metrics, bootstrap and the class-conditional diagnostic.

    python auto_raw_plus_z_socratic.py <repo> <tag> <bank dir> <gold socratic.json> <stored placement csv> <out dir>
"""
import csv, importlib.util, json, sys, types
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

repo, tag, bank, goldp, stored, out = (Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]),
                                        Path(sys.argv[5]), Path(sys.argv[6]))
sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / 'results/auto_raw_plus_z_v1'))
import auto_raw_plus_z as AR  # noqa: E402  (N16, DIG, rank_gauss, auc, lead_share)
_src = repo / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
EV = types.ModuleType('ev'); EV.__file__ = str(repo / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
sys.modules['ev'] = EV
exec(compile(_src.read_text(), str(_src), 'exec'), EV.__dict__)
from spectral_utils.fusion_utils import lsml_continuous_guarded  # noqa: E402
from spectral_utils.family_external_metrics import summary  # noqa: E402
from spectral_utils.external_generalization.evaluation import confusion, within_auc  # noqa: E402

DRAWS, SEED = 1000, 20261103
out.mkdir(parents=True, exist_ok=True)
F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
assert np.array_equal(F['uids'], R['uids'])
ne = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64); lens = F['nonempty_lengths']
F['cells'] = np.array([EV.CELL] * len(F['uids'])); F['row_ids'] = F['uids']
fs = importlib.util.spec_from_file_location('prm_fuse', repo / 'results/prm_boosting_v1/fuse.py')
FUSE = importlib.util.module_from_spec(fs); fs.loader.exec_module(FUSE); ext = FUSE.ext
assert list(ext.BN['N16']) == AR.N16
ans = np.arange(len(F['uids']))
cb = ext.Population(F, ['N16']).cell_bank('N16', ans)
rew = np.clip(R['rewards'][ne], 1e-6, 1 - 1e-6)
prm = EV.risk(rew); zp = EV.az(prm, o); prob = 1 - np.asarray(R['rewards'][ne], float)
raw = np.asarray(cb['raw'], float); act = F['digit_active'].astype(bool)
RG = np.column_stack([AR.rank_gauss(raw[:, j], act[:, AR.DIG.index(c)] if c in AR.DIG else None) for j, c in enumerate(AR.N16)])
V17 = np.column_stack([zp, np.asarray(cb['V'], float)])
V34 = np.column_stack([V17, AR.rank_gauss(prm), RG])
names17 = ['PRM'] + AR.N16
names34 = ['z:' + c for c in names17] + ['raw:' + c for c in names17]
nst = np.diff(o); pos = np.concatenate([(np.arange(k) + .5) / k for k in nst])


def lsml(V):
    s, d = lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])])
    s = np.asarray(s, float)
    if np.corrcoef(s, V[:, 0])[0, 1] < 0: s = -s
    return s, d


arms, fits = {'PRM': prm, 'PRMPROB': prob, 'POS_ONLY': EV.az(pos, o)}, {}
arms['Z17'], fits['Z17'] = lsml(V17)
arms['CAT34'], fits['CAT34'] = lsml(V34)

# ---- scoring, verbatim logic of eval_socratic_ablation5.py
gold = {g['uid']: g for g in json.loads(goldp.read_text(encoding='utf8'))}
uids = [str(u) for u in F['uids']]; starts = np.r_[0, np.cumsum(lens)]


def az_q80(s, off):
    az = np.zeros_like(s)
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return az >= np.quantile(az, .8)


def per_answer(score_ne, pred_valid_all):
    full_s = np.full(int(lens.sum()), np.nan); full_s[np.flatnonzero(ne)] = score_ne
    C, A = [], []
    for i, u in enumerate(uids):
        g = gold[u]; a, b = starts[i], starts[i + 1]; inc = np.asarray(g['include'], bool)
        C.append(confusion(g['correct'], pred_valid_all[a:b], inc))
        v = within_auc(g['correct'], full_s[a:b], inc & ne[a:b]); A.append(np.nan if v is None else v)
    return np.array(C, float), np.array(A, float)


res = {}
for n, s in arms.items():
    pv = np.zeros(int(lens.sum()), int); pv[np.flatnonzero(ne)] = (~az_q80(s, o)).astype(int)
    res[f'{n}_Q80'] = per_answer(s, pv)
res['PRM_native'] = per_answer(prm, R['native_labels'].astype(int))
score_of = {k: arms[k[:-4]] if k.endswith('_Q80') else prm for k in res}

# step labels on non-empty steps (error = not correct), inclusion mask
corr_all = np.concatenate([np.asarray(gold[u]['correct'], bool) for u in uids])
inc_all = np.concatenate([np.asarray(gold[u]['include'], bool) for u in uids])
err = ~corr_all[ne]; inc = inc_all[ne]
ans_err = np.array([(err[a:b] & inc[a:b]).any() for a, b in zip(o[:-1], o[1:])])
ans_ok = np.array([inc[a:b].any() for a, b in zip(o[:-1], o[1:])])

rows = []
for k, (C, A) in res.items():
    s = score_of[k]
    mx = np.array([s[a:b].max() for a, b in zip(o[:-1], o[1:])])
    rows.append({'benchmark': 'socratic', 'model': tag, 'arm': k, 'official_metric': summary(C, 'socratic')['metric'],
                 'within_auc': float(np.nanmean(A)), 'n_auc': int(np.isfinite(A).sum()),
                 'pooled_auc': float(AR.auc(err[inc], s[inc])), 'answer_auc_max': float(AR.auc(ans_err[ans_ok], mx[ans_ok])),
                 'n_answers': len(uids)})

# parity against the stored AUTO_N16_signed_Q80 row
st = {r['arm']: r for r in csv.DictReader(open(stored)) if r['model'] == {'qwen_prm_7b': 'qwen_prm'}.get(tag, tag)}
z = [r for r in rows if r['arm'] == 'Z17_Q80'][0]; ref = st['AUTO_N16_signed_Q80']
par = {'stored_official': float(ref['official_metric']), 'stored_within_auc': float(ref['within_auc']),
       'z17_official': z['official_metric'], 'z17_within_auc': z['within_auc']}
par['pass'] = abs(par['stored_official'] - par['z17_official']) < 1e-9 and abs(par['stored_within_auc'] - par['z17_within_auc']) < 1e-9
print(tag, 'PARITY', json.dumps(par), flush=True)
if not par['pass']:
    (out / f'PARITY_FAIL_{tag}.json').write_text(json.dumps(par, indent=1)); raise SystemExit('HARD STOP: parity')

# answer-level bootstrap (multinomial reweighting of answers), differences vs Z17
n = len(uids); rng = np.random.default_rng(SEED)
W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=DRAWS)])


def boot_stats(C, A):
    t = W @ C; tp, fp, tn, fn = t.T
    with np.errstate(divide='ignore', invalid='ignore'):
        pc, rc, pe, re = tp / (tp + fp), tp / (tp + fn), tn / (tn + fn), tn / (tn + fp)
        m = (2 * pc * rc / (pc + rc) + 2 * pe * re / (pe + re)) / 2
    f = np.isfinite(A)
    return m, (W[:, f] @ A[f]) / W[:, f].sum(1)


B = {k: boot_stats(*v) for k, v in res.items()}
assert abs(B['Z17_Q80'][0][0] - z['official_metric']) < 1e-9 and abs(B['Z17_Q80'][1][0] - z['within_auc']) < 1e-9
boot = []
for k in ('CAT34_Q80', 'PRMPROB_Q80', 'PRM_Q80', 'PRM_native', 'POS_ONLY_Q80'):
    for j, mt in enumerate(('official_metric', 'within_auc')):
        d = B[k][j] - B['Z17_Q80'][j]
        boot.append({'benchmark': 'socratic', 'model': tag, 'unit': 'answer', 'draws': DRAWS, 'contrast': f'{k} - Z17_Q80',
                     'metric': mt, 'delta': d[0], 'ci_low': np.quantile(d[1:], .025), 'ci_high': np.quantile(d[1:], .975)})

# diagnostics (included non-empty steps)
c34 = np.asarray(fits['CAT34']['c'], int); c17 = np.asarray(fits['Z17']['c'], int)
diag = []
for i, c in enumerate(names17):
    zi, ri = V34[:, i], V34[:, 17 + i]
    row = {'model': tag, 'channel': c, 'Z17_group': int(c17[i]), 'CAT34_group_z': int(c34[i]),
           'CAT34_group_raw': int(c34[17 + i]), 'paired_in_CAT34': bool(c34[i] == c34[17 + i])}
    for lab, m in (('clean', inc & ~err), ('error', inc & err), ('all', inc)):
        row[f'abs_corr_raw_z_{lab}'] = float(abs(np.corrcoef(zi[m], ri[m])[0, 1]))
    diag.append(row)
fitd = {'parity': par}
for k, V, nm in (('Z17', V17, names34[:17]), ('CAT34', V34, names34)):
    c = np.asarray(fits[k]['c'], int)
    fitd[k] = {'K': int(fits[k]['K']), 'groups': {int(g): [nm[j] for j in np.flatnonzero(c == g)] for g in np.unique(c)},
               'dropped_constant': [nm[j] for j in fits[k].get('dropped_constant', [])],
               'lead_eig_share_all_inputs': AR.lead_share(V),
               'n_pairs_same_group': int(sum(r['paired_in_CAT34'] for r in diag)) if k == 'CAT34' else None}

for path, rr in ((out / f'RESULTS_{tag}.csv', rows), (out / f'BOOTSTRAP_{tag}.csv', boot), (out / f'DIAG_{tag}.csv', diag)):
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rr for k in r))); wr.writeheader(); wr.writerows(rr)
(out / f'FIT_{tag}.json').write_text(json.dumps(fitd, indent=1))
# self-contained input file, same format as the PRMBench bundle (non-empty steps only)
sidx = np.concatenate([np.arange(k) for k in nst])
np.savez_compressed(out / f'INPUTS_{tag}.npz', row_ids=np.array(uids), offsets=o, step_index_nonempty=sidx,
                    answer_of_step=np.repeat(np.arange(n), nst), step_error=err, step_included=inc,
                    nonempty=ne, nonempty_lengths=lens, reward=np.asarray(R['rewards'][ne], float),
                    prm_native_valid=R['native_labels'][ne].astype(bool),
                    raw17=np.column_stack([prm, raw]), raw17_names=np.array(names17),
                    digit_active=act, digit_names=np.array(AR.DIG), X34=V34, names34=np.array(names34),
                    position=pos, **{f'score__{k}': v for k, v in arms.items()})
for r in rows: print(tag, r['arm'], round(r['official_metric'], 4), round(r['within_auc'], 4), round(r['pooled_auc'], 4), round(r['answer_auc_max'], 4), flush=True)
