"""Socratic-PRMBench diagnosis for PRM boosting (CPU, 2026-10-19). LABEL-USING: nothing here is a method.

Question: on Socratic the label-free fusions lose to the PRM on 4 of 5 PRMs. Is that because the
telemetry has no extra step-error signal there, or because the label-free fusion uses it badly?

Part A, supervised LR ceiling (oracle access, NOT label-free):
  balanced logistic regression, C=1, standardized, 5-fold GroupKFold by source question (the part of the
  uid after the last ':', e.g. OlympiadBench#208; one source has up to 6 perturbations). Out-of-fold
  decision scores on every non-empty step. Features: z(PRM) and/or the answer-z, label-free-oriented
  bank channels exactly as L-SML sees them (N13 digit-free, N16 with 3 digit channels, S3 = LSML_CONT).
  Scores are written as SCORES_<prm>.npz for ../prm_boost_external_v1/eval_external.py.
Part B, per-channel direction (label-using diagnostic):
  for every N16 + S3 channel and the PRM: within-answer AUC of the channel as L-SML sees it against the
  error label, its supervised sign, and the sign/weight the label-free TELE_N16 L-SML fit gives it
  (least-squares effective weight of the fused score on the channels). PRMBench reference for Qwen-7B is
  results/prm_boosting_v1/CHANNEL_ATLAS_prm_backbone.csv (same bank code, same orientation).

    .venv/bin/python diagnose.py <banks root with full_<prm>_socratic/bank_v2> <socratic gold json> <scores out dir>
"""
import csv, importlib.util, json, sys, types
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
M = HERE.parents[1]
sys.path.insert(0, str(M))
_src = M / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
EV = types.ModuleType('ev'); EV.__file__ = str(M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
sys.modules['ev'] = EV
exec(compile(_src.read_text(), str(_src), 'exec'), EV.__dict__)
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402
fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext

PRMS = ('qwen_prm', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd')
root, gold_path, sout = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sout.mkdir(parents=True, exist_ok=True)
gold = {g['uid']: g for g in json.loads(gold_path.read_text(encoding='utf8'))}
S3 = list(EV.S3)
ext.BN = dict(ext.BN, TRANSFER=S3)


def wauc(s, y, o, keep):
    """Mean within-answer AUC (higher score = error) over answers with both classes among kept steps."""
    out = []
    for a, b in zip(o[:-1], o[1:]):
        k = keep[a:b]; yy = y[a:b][k]; ss = s[a:b][k]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0:
            r = np.argsort(np.argsort(ss, kind='mergesort'), kind='mergesort') + 1.
            # average ranks for ties
            _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
            r = (np.bincount(inv, r) / cnt)[inv]
            out.append((r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out)), len(out)


chan_rows, lr_rows = [], []
for a in PRMS:
    bank = root / f'full_{a}_socratic/bank_v2'
    F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
    ne = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64); uids = [str(u) for u in F['uids']]
    F['cells'] = np.array([EV.CELL] * len(uids)); F['row_ids'] = F['uids']
    pop = ext.Population(F, ['TRANSFER', 'N13', 'N16'])
    ans = np.arange(len(uids))
    B = {k: pop.cell_bank(k, ans) for k in ('TRANSFER', 'N13', 'N16')}
    prm = EV.risk(np.clip(R['rewards'][ne], 1e-6, 1 - 1e-6)); zp = EV.az(prm, o)
    # step labels on the non-empty rows (error = not correct), and the include mask
    corr = np.concatenate([np.asarray(gold[u]['correct'], bool) for u in uids])
    inc = np.concatenate([np.asarray(gold[u]['include'], bool) for u in uids])
    assert len(corr) == len(ne) == int(F['nonempty_lengths'].sum())
    y, keep = ~corr[ne], inc[ne]
    src = np.array([u.rsplit(':', 1)[-1] for u in uids]); step_ans = np.repeat(ans, np.diff(o))

    # ---- Part B: per-channel direction
    V16 = np.asarray(B['N16']['V'], float); n16 = list(B['N16']['names'])
    tele, dt = lsml_continuous(*[V16[:, j] for j in range(V16.shape[1])])
    tele = np.asarray(tele, float)
    if np.corrcoef(tele, V16.mean(1))[0, 1] < 0: tele = -tele
    w = np.linalg.lstsq(np.column_stack([V16, np.ones(len(V16))]), tele, rcond=None)[0][:-1]
    au_tele, _ = wauc(tele, y, o, keep)
    cols = [('prm_risk', prm)] + [(c, V16[:, j]) for j, c in enumerate(n16)] + \
           [(c, np.asarray(B['TRANSFER']['V'], float)[:, j]) for j, c in enumerate(B['TRANSFER']['names']) if c not in n16]
    for c, v in cols:
        au, n = wauc(v, y, o, keep)
        j = n16.index(c) if c in n16 else None
        chan_rows.append({'prm': a, 'channel': c, 'within_auc': au, 'n_answers': n, 'supervised_sign': 1 if au >= .5 else -1,
                          'within_auc_oriented': max(au, 1 - au),
                          'tele_n16_weight': float(w[j]) if j is not None else '',
                          'tele_n16_uses_right_direction': (bool(np.sign(w[j]) == (1 if au >= .5 else -1)) if j is not None else '')})
    chan_rows.append({'prm': a, 'channel': 'TELE_N16 (label-free fused)', 'within_auc': au_tele, 'n_answers': '',
                      'supervised_sign': 1 if au_tele >= .5 else -1, 'within_auc_oriented': max(au_tele, 1 - au_tele),
                      'tele_n16_weight': '', 'tele_n16_uses_right_direction': ''})

    # ---- Part A: supervised LR ceiling, out-of-fold by source question
    feats = {'LR_PRM': [zp], 'LR_N13': [], 'LR_N16': [], 'LR_PRM_N13': [zp], 'LR_PRM_N16': [zp], 'LR_PRM_S3': [zp]}
    V13 = np.asarray(B['N13']['V'], float)
    VS3 = np.column_stack([np.asarray(B['TRANSFER']['V'], float)[:, list(B['TRANSFER']['names']).index(c)] for c in S3])
    extra = {'LR_N13': V13, 'LR_N16': V16, 'LR_PRM_N13': V13, 'LR_PRM_N16': V16, 'LR_PRM_S3': VS3}
    arms = {'PRM': prm}
    folds = list(GroupKFold(n_splits=5).split(ans, groups=src))
    for name, base in feats.items():
        X = np.column_stack(base + ([extra[name]] if name in extra else []))
        oof = np.full(len(X), np.nan)
        for tr_a, te_a in folds:
            tr = np.isin(step_ans, tr_a) & keep; te = np.isin(step_ans, te_a)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000))
            clf.fit(X[tr], y[tr]); oof[te] = clf.decision_function(X[te])
        assert np.isfinite(oof).all()
        arms[name] = oof
        lr_rows.append({'prm': a, 'arm': name, 'n_features': X.shape[1], 'oof_within_auc': wauc(oof, y, o, keep)[0]})
    np.savez_compressed(sout / f'SCORES_{a}.npz', uids=F['uids'], nonempty=ne, nonempty_lengths=F['nonempty_lengths'],
                        native_valid=R['native_labels'].astype(int), **{f'score__{k}': v for k, v in arms.items()})
    print(a, {r['arm']: round(r['oof_within_auc'], 4) for r in lr_rows if r['prm'] == a}, flush=True)

for path, rr in ((HERE / 'CHANNEL_DIRECTION_socratic.csv', chan_rows), (HERE / 'LR_CEILING_withinauc.csv', lr_rows)):
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rr for k in r))); wr.writeheader(); wr.writerows(rr)
print('DONE')
