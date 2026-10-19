"""Top-50 approximations of self-certainty (INTUITOR, arXiv 2505.19590) on Socratic PRM telemetry (CPU, 2026-10-20).

Self-certainty of one token = KL(U || p) = -log V - (1/V) * sum_j log p_j over the FULL vocabulary V.
The raw telemetry keeps only the top-50 log-probs per token, so it cannot be computed exactly. Two
approximations are computed per token, and then averaged over the tokens of each step:
  SC_HEAD  self-certainty of the renormalized top-50 distribution: -log 50 - mean_k log p~_k
           (how flat the head is; ignores the tail completely)
  SC_LB    the uniform-tail LOWER bound: the tail mass m = 1 - sum(top-50) spread uniformly over the other
           V-50 tokens (Jensen). Since V >> 50 it is essentially -log m, so it should track tail-mass channels.
Higher = more certain. Steps follow the bank builder exactly: mask1 (forced token 0 dropped, spans shifted),
already-empty steps dropped, records sorted by uid. The step mask is checked against bank_v2/FEATURES.npz.

LABEL-USING DIAGNOSTIC: the AUC and LR columns read the Socratic gold labels. Nothing here is a method.

    .venv/bin/python self_certainty_approx.py <raw root with full_<prm>_socratic/external_telemetry.pkl> \
        <banks root with full_<prm>_socratic/bank_v2> <socratic gold json>
"""
import csv, json, pickle, sys, types
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
M = HERE.parents[1]
sys.path.insert(0, str(M))
from spectral_utils.first_token_mask import shift_spans  # noqa: E402
_src = M / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
EV = types.ModuleType('ev'); EV.__file__ = str(M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
sys.modules['ev'] = EV
exec(compile(_src.read_text(), str(_src), 'exec'), EV.__dict__)
import importlib.util  # noqa: E402
fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext

# lm_head rows of each backbone family (only shifts SC_LB by a near-constant; within-answer metrics unaffected)
VOCAB = {'qwen_prm': 152064, 'qwen_prm_72b': 152064, 'pure_prm': 152064, 'rlhflow_prm': 128256, 'math_shepherd': 32000}
raw_root, bank_root, gold_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
gold = {g['uid']: g for g in json.loads(gold_path.read_text(encoding='utf8'))}


def token_sc(lp, V):
    lp = np.asarray(lp, np.float64)
    lse = np.logaddexp.reduce(lp, axis=1, keepdims=True)
    head = -np.log(lp.shape[1]) - (lp - lse).mean(1)
    m = np.clip(1.0 - np.exp(lp).sum(1), 1e-12, 1.0)
    k = lp.shape[1]
    lb = -np.log(V) - (lp.sum(1) + (V - k) * np.log(m / (V - k))) / V
    return head, lb, np.log(m)


def wauc(s, y, o, keep):
    out = []
    for a, b in zip(o[:-1], o[1:]):
        k = keep[a:b]; yy = y[a:b][k]; ss = s[a:b][k]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0: out.append((rankdata(ss)[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out))


def within_rank(v, o):
    r = np.empty_like(v, float)
    for a, b in zip(o[:-1], o[1:]): r[a:b] = (rankdata(v[a:b]) - .5) / max(b - a, 1)
    return r


rows_auc, rows_corr, rows_lr = [], [], []
for prm, V in VOCAB.items():
    bank = bank_root / f'full_{prm}_socratic/bank_v2'
    F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
    ne_bank = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64); uids = [str(u) for u in F['uids']]
    raw = pickle.load(open(raw_root / f'full_{prm}_socratic/external_telemetry.pkl', 'rb'))
    recs = {str(e['idx']): e for e in (raw.values() if isinstance(raw, dict) else raw)}
    feats = {'SC_HEAD': [], 'SC_LB': [], 'LOG_TAIL_MASS': []}; ne_mine = []
    for u in uids:
        e = recs[u]; lp = np.asarray(e['top_k_logprobs']['logprobs'])[1:]   # mask1: drop forced token 0
        sp = e['step_token_spans']; keep = [k for k, s in enumerate(sp) if s[1] > s[0]]
        shifted = shift_spans([tuple(sp[k]) for k in keep])
        head, lb, ltm = token_sc(lp, V)
        flags = np.zeros(len(sp), bool); flags[keep] = True; ne_mine.append(flags)
        for (lo, hi) in shifted:
            for name, arr in (('SC_HEAD', head), ('SC_LB', lb), ('LOG_TAIL_MASS', ltm)):
                feats[name].append(float(arr[lo:hi].mean()))
    ne_mine = np.concatenate(ne_mine)
    assert np.array_equal(ne_mine, ne_bank), f'{prm}: step mask differs from bank_v2'
    X = {k: np.asarray(v) for k, v in feats.items()}
    assert all(len(v) == len(F['values']) and np.isfinite(v).all() for v in X.values())
    del raw, recs

    corr = np.concatenate([np.asarray(gold[u]['correct'], bool) for u in uids])
    inc = np.concatenate([np.asarray(gold[u]['include'], bool) for u in uids])
    y, keep = ~corr[ne_bank], inc[ne_bank]
    prm_risk = EV.risk(np.clip(R['rewards'][ne_bank], 1e-6, 1 - 1e-6))

    # step-error AUC: risk direction = LOW certainty, i.e. -SC
    for name, v in [('prm_risk', prm_risk)] + [(k, -v) for k, v in X.items()]:
        a = wauc(v, y, o, keep)
        rows_auc.append({'prm': prm, 'signal': name if name == 'prm_risk' else f'-{name}', 'within_auc': a,
                         'within_auc_oriented': max(a, 1 - a), 'supervised_sign': 1 if a >= .5 else -1})
    # within-answer Spearman with every bank channel and the PRM
    names = [str(n) for n in F['names']]; Vb = np.asarray(F['values'], float)
    others = [('prm_risk', prm_risk)] + [(n, Vb[:, j]) for j, n in enumerate(names)]
    for name, v in X.items():
        rv = within_rank(v, o)
        cs = sorted(((abs(np.corrcoef(rv, within_rank(w, o))[0, 1]), np.corrcoef(rv, within_rank(w, o))[0, 1], n)
                     for n, w in others if np.std(w) > 0), reverse=True)
        for r_abs, r, n in cs[:5]:
            rows_corr.append({'prm': prm, 'approx': name, 'channel': n, 'within_answer_spearman': r})
        rows_corr.append({'prm': prm, 'approx': name, 'channel': 'prm_risk', 'within_answer_spearman':
                          float(np.corrcoef(rv, within_rank(prm_risk, o))[0, 1])})
    # does it add anything beyond PRM + N16? supervised LR, same folds as diagnose.py
    F2 = dict(F); F2['cells'] = np.array([EV.CELL] * len(uids)); F2['row_ids'] = F2['uids']
    V16 = np.asarray(ext.Population(F2, ['N16']).cell_bank('N16', np.arange(len(uids)))['V'], float)
    zp = EV.az(prm_risk, o); ZX = np.column_stack([EV.az(X['SC_HEAD'], o), EV.az(X['SC_LB'], o)])
    src = np.array([u.rsplit(':', 1)[-1] for u in uids]); step_ans = np.repeat(np.arange(len(uids)), np.diff(o))
    folds = list(GroupKFold(n_splits=5).split(np.arange(len(uids)), groups=src))
    for arm, Xa in (('LR_PRM_N16', np.column_stack([zp, V16])), ('LR_PRM_N16_SC', np.column_stack([zp, V16, ZX])),
                    ('LR_PRM_SC', np.column_stack([zp, ZX]))):
        oof = np.full(len(Xa), np.nan)
        for tr_a, te_a in folds:
            tr = np.isin(step_ans, tr_a) & keep; te = np.isin(step_ans, te_a)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000))
            clf.fit(Xa[tr], y[tr]); oof[te] = clf.decision_function(Xa[te])
        rows_lr.append({'prm': prm, 'arm': arm, 'oof_within_auc': wauc(oof, y, o, keep)})
    print(prm, {r['signal']: round(r['within_auc'], 3) for r in rows_auc if r['prm'] == prm},
          {r['arm']: round(r['oof_within_auc'], 4) for r in rows_lr if r['prm'] == prm}, flush=True)

for path, rr in ((HERE / 'SELF_CERTAINTY_AUC.csv', rows_auc), (HERE / 'SELF_CERTAINTY_CORR.csv', rows_corr),
                 (HERE / 'SELF_CERTAINTY_LR.csv', rows_lr)):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rr[0])); w.writeheader(); w.writerows(rr)
print('DONE')
