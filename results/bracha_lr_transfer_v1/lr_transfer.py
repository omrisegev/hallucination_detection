"""Bracha's experiment, as frozen in PROTOCOL.json: supervised LR trained on PRM800K, applied unchanged to 7 benchmarks,
against label-free AUTO / CHAIN / TELE16 and the PRM. CPU only, stored scores.

    python lr_transfer.py      # -> RESULTS.csv, WEIGHTS.csv, BOOTSTRAP.csv, OVERLAP.json, PARITY.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import glob, json, pickle, re, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prm_sub5_all_benchmarks_v1'))
import sub5_all as A  # noqa: E402
E = A.E
sys.path.insert(0, str(A.ROOT / 'results/prm_boost_math_rigor_v1'))
import stage8_auto_forced_k as S8  # noqa: E402

PROT = json.loads((HERE / 'PROTOCOL.json').read_text(encoding='utf-8'))
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
TESTS = PROT['test_sets']
CGRID = [0.01, 0.1, 1.0, 10.0]
TXT = A.DATA / 'crop_ares_v1' / 'inputs_text'
MAIN = {'prmbench': 'PRMScore_GAP', 'processbench': 'avg_f1_NC', 'hard2verify': 'balanced_f1_GAP'}
S8_REF = {(r['bench'], r['prm']): r['A']['auto'] for r in
          json.loads((A.ROOT / 'results/prm_boost_math_rigor_v1/stage8/AUTO_FORCEDK_RESULTS.json').read_text())}
AUTO_NEW = {(r['benchmark'], r['prm'], r['arm'], r['metric']): float(r['value']) for r in E.read_csv(A.HERE / 'AUTO_NEW.csv')}
SUB5 = {(r['benchmark'], r['mode'], r['prm'], r['arm'], r['metric']): float(r['value']) for r in E.read_csv(A.HERE / 'SUB5_ALL.csv')}
ROWS, PAR, W, BOOT = [], [], [], []
norm = lambda s: re.sub(r'\s+', ' ', str(s or '')).strip().lower()


def gate(name, got, ref, tol=1e-6):
    d = abs(float(got) - float(ref)); PAR.append(dict(check=name, got=float(got), ref=float(ref), abs_diff=d, ok=d <= tol))
    print(('PASS ' if d <= tol else 'FAIL ') + name, round(float(got), 6), round(float(ref), 6), flush=True)


# ------------------------------------------------------------------ problem overlap (test problems removed from training)
def overlap():
    from datasets import load_dataset
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    q8 = {a['uid']: norm(a['question']) for a in json.load(open(TXT / 'prm800k/answers.json', encoding='utf8'))}
    test = {'prmbench': {norm(x) for x in load_dataset('hitsmy/PRMBench_Preview', split='train')['original_question']}}
    pq = set()
    for f in glob.glob(str(A.ROOT.parent / 'hallucination_detection/dataset_cache/four_localization/pb_prm_qwen25math7b_full/*.pkl')):
        pq |= {norm(r['problem']) for r in pickle.load(open(f, 'rb')).values()}
    test['processbench'] = pq
    for b in ('deltabench', 'crv_arith', 'crv_gsm8k'):
        test[b] = {norm(a['question']) for a in json.load(open(TXT / f'{b}/answers.json', encoding='utf8'))}
    probs = set(q8.values()); bad = set().union(*[probs & t for t in test.values()])
    info = dict(prm800k_answers=len(q8), prm800k_problems=len(probs), removed_problems=len(bad),
                removed_answers=sum(q in bad for q in q8.values()),
                overlap_problems={b: len(probs & t) for b, t in test.items()},
                not_checked=['socratic', 'hard2verify'])
    return q8, bad, info


# ------------------------------------------------------------------ cells and features
def load(bench, prm, mode):
    if mode == 'prm' and bench in ('prmbench', 'socratic', 'hard2verify'):
        c = E.load(bench, prm)
    elif mode == 'sa' and bench == 'prmbench':
        c = E.load('prmbench', f'{prm}__standalone')
    else:
        key = f'cab|{mode}|{bench}'; A.REG[key] = dict(mode=mode, bench=bench, common=A.common_uids(bench))
        c = E.load(key, prm)
    if not hasattr(c, 'n_gold'): c.n_gold = len(c.ids)
    return c


def zprm(c, bench):
    return E.answer_z(c.arms['PRM_Q80' if bench == 'prmbench' else 'PRM'], c.off)


def feats(c, bench, which):
    V = E.channels(c, 'N16'); T = np.column_stack([E.answer_z(V[:, j], c.off) for j in range(V.shape[1])])
    zp = zprm(c, bench) if which in ('a', 'c') else None
    X = np.column_stack([zp, T]) if which == 'a' else T if which == 'b' else zp[:, None]
    return X, V, zp


def fit(X, y, g):
    best = None
    for C in CGRID:
        aucs = []
        for tr, te in GroupKFold(5).split(X, y, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000)).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.decision_function(X[te])))
        if best is None or np.mean(aucs) > best[1]: best = (C, float(np.mean(aucs)))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], class_weight='balanced', max_iter=3000)).fit(X, y)
    return m, best


# ------------------------------------------------------------------ evaluation
def metric_from_counts(c, C):
    v = float(E.official_from_counts(C, c.bench))
    return v if c.bench == 'prmbench' else 100 * v


def evaluate(c, bench, prm, mode, arm_names, probs):
    if bench == 'prmbench':
        scale = 'PRM_Q80_PROB' if A.META[f'prmbench/{prm}']['prm_reference_scale'] == '1-reward' else 'PRM_Q80'
        arms = [('PRM native', 'PRM_native', 'native'), ('PRM', scale, 'GAP')]
    else:
        arms = [('PRM native', 'PRM', 'native'), ('PRM', A.prm_scale_arm(bench, prm), 'GAP')]
    arms += [(a, a, 'GAP') for a in arm_names]
    if bench == 'prmbench': R = A.rows_prmbench(c, arms)
    elif bench == 'processbench': R = A.rows_pb(c, arms)
    else:
        R = A.rows_cab(c, mode, bench, prm, arms)
        if bench in ('crv_arith', 'crv_gsm8k'): R += A.rows_crop(c, arms)
    if bench == 'processbench': R += A.rows_crop(c, arms)
    out = [dict(r, rule='GAP' if r['arm'] != 'PRM native' else 'native') for r in R]
    keep = c.keep_gold if bench == 'processbench' else c.keep
    for label, arm, _ in arms:            # pooled step AUROC where the row functions do not give it
        if bench in ('prmbench', 'processbench') and label != 'PRM native':
            out.append(dict(arm=label, metric='pooled_auroc', value=float(roc_auc_score(c.err[keep], c.arms[arm][keep])), rule='GAP'))
    if bench != 'processbench':
        for a, p in probs.items():        # THR05: label-trained threshold on the LR probability
            out.append(dict(arm=a, metric=MAIN.get(bench, 'macro_f1_GAP').replace('GAP', 'THR05'),
                            value=metric_from_counts(c, E.counts_by_answer(c, p > .5).sum(0)), rule='THR05'))
    for r in out: r.update(benchmark=bench, prm=prm, telemetry={'prm': 'backbone', 'sa': 'standalone'}[mode])
    return out


def boot(c, bench, prm, a, b, B=1000):
    Ca, Cb = E.counts_by_answer(c, c.flags(a, 'GAP')), E.counts_by_answer(c, c.flags(b, 'GAP'))
    ug, gi = np.unique(np.asarray(c.groups), return_inverse=True)
    Ga, Gb = np.zeros((len(ug), 4)), np.zeros((len(ug), 4)); np.add.at(Ga, gi, Ca); np.add.at(Gb, gi, Cb)
    pt = metric_from_counts(c, Ga.sum(0)) - metric_from_counts(c, Gb.sum(0))
    rng = np.random.default_rng(20261028); d = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        d.append(metric_from_counts(c, w @ Ga) - metric_from_counts(c, w @ Gb))
    lo, hi = np.percentile(d, [2.5, 97.5])
    BOOT.append(dict(benchmark=bench, prm=prm, contrast=f'{a} - {b}', delta=pt, ci_lo=float(lo), ci_hi=float(hi),
                     n_groups=len(ug), grouping='source group' if bench == 'prmbench' else 'answer'))
    return metric_from_counts(c, Ga.sum(0))


def main():
    q8, bad, info = overlap(); json.dump(info, open(HERE / 'OVERLAP.json', 'w'), indent=1); print(info, flush=True)
    for prm in PRMS:
        models = {}
        for mode, sets in (('prm', ('a', 'c')), ('sa', ('b',))):
            c = load('prm800k', prm, mode); y = c.err.astype(int)
            prob = np.array([q8[u] for u in c.ids]); g = np.repeat(np.unique(prob, return_inverse=True)[1], np.diff(c.off))
            clean = np.repeat(~np.isin(prob, list(bad)), np.diff(c.off))
            for which in sets:
                X, _, _ = feats(c, 'prm800k', which)
                for tag, m in (('clean', clean),) + ((('all', np.ones_like(clean)),) if which == 'a' else ()):
                    model, (C, cvauc) = fit(X[m], y[m], g[m]); models[(which, tag)] = model
                    coef = model[-1].coef_[0]
                    W.append(dict(prm=prm, inputs=which, training=tag, C=C, cv_auroc=cvauc, n_steps=int(m.sum()),
                                  prm_share=float(abs(coef[0]) / np.abs(coef).sum()) if which == 'a' else None,
                                  **({'coef_PRM': float(coef[0])} if which in ('a', 'c') else {}),
                                  **{f'coef_{n}': float(v) for n, v in zip(A.NAMES, coef[1:] if which == 'a' else coef if which == 'b' else [])}))
                    print(prm, which, tag, 'C', C, 'cv AUROC', round(cvauc, 4), 'PRM share', W[-1]['prm_share'], flush=True)
            del c
        for bench in TESTS:
            # ---- backbone cell: (a) clean/all, (c), AUTO, CHAIN, backbone TELE/MEAN
            c = load(bench, prm, 'prm'); probs = {}
            for nm, key in (('LR_a_clean', ('a', 'clean')), ('LR_a_all', ('a', 'all')), ('LR_c_clean', ('c', 'clean'))):
                X, V, zp = feats(c, bench, key[0]); c.arms[nm] = models[key].decision_function(X); probs[nm] = models[key].predict_proba(X)[:, 1]
            cs = c                                                    # cell on which the label-free AUTO is fitted
            if bench == 'hard2verify':     # stage8 fits AUTO on each PRM's full readable set, then scores the 177 common answers
                cs = A._load('hard2verify', prm); X, _, _ = feats(cs, bench, 'a')
                cs.arms['LR_a_clean'] = models[('a', 'clean')].decision_function(X)
            cs.arms['AUTO_N16'], _ = S8.auto(E.channels(cs, 'N16'), zprm(cs, bench))
            if cs is not c:
                pos = {u: j for j, u in enumerate(cs.ids)}
                c.arms['AUTO_N16'] = np.concatenate([cs.arms['AUTO_N16'][cs.off[pos[u]]:cs.off[pos[u] + 1]] for u in c.ids])
            R = evaluate(c, bench, prm, 'prm', ['TELE_N16', 'MEAN_N16', 'CHAIN_N16', 'AUTO_N16', 'LR_a_clean', 'LR_a_all', 'LR_c_clean'], probs)
            get = lambda arm, m: next(r['value'] for r in R if r['arm'] == arm and r['metric'] == m)
            mn = MAIN.get(bench, 'macro_f1_GAP')
            ref = S8_REF[(bench, prm)]['gap_prmscore'] if bench in ('prmbench', 'socratic', 'hard2verify') else AUTO_NEW[(bench, prm, 'AUTO_N16', mn)]
            gate(f'{prm} {bench} AUTO main metric', get('AUTO_N16', mn), ref)
            gate(f'{prm} {bench} PRM (GAP) main metric vs SUB5_ALL', get('PRM', mn), SUB5[(bench, 'backbone', prm, 'PRM', mn)])
            if bench in ('prmbench', 'socratic', 'hard2verify'):     # stage8's least-squares PRM share, same function
                share = lambda s: E.prm_weight_share(cs, 'CHAIN', 'N16', score=s)['prm_weight_share']
                for a in ('AUTO_N16', 'LR_a_clean', 'CHAIN_N16'):
                    W.append(dict(prm=prm, inputs='lstsq share on ' + bench, training=a, prm_share=float(share(cs.arms[a]))))
                gate(f'{prm} {bench} AUTO lstsq PRM share vs stage8', W[-3]['prm_share'], S8_REF[(bench, prm)]['lstsq_prm_share'])
            if bench != 'processbench':
                pa = boot(c, bench, prm, 'LR_a_clean', 'AUTO_N16')
                gate(f'{prm} {bench} bootstrap point = row metric (LR_a_clean)', pa, get('LR_a_clean', mn))
            ROWS.extend(R); del c, cs
            # ---- standalone cell: (b) vs stored standalone TELE16 / MEAN
            c = load(bench, prm, 'sa'); X, _, _ = feats(c, bench, 'b')
            c.arms['LR_b_clean'] = models[('b', 'clean')].decision_function(X)
            R = evaluate(c, bench, prm, 'sa', ['TELE_N16', 'MEAN_N16', 'LR_b_clean'], {'LR_b_clean': models[('b', 'clean')].predict_proba(X)[:, 1]})
            ROWS.extend(R); del c
            print(prm, bench, 'done', flush=True)
    E.write_csv(HERE / 'RESULTS.csv', ROWS); E.write_csv(HERE / 'WEIGHTS.csv', W); E.write_csv(HERE / 'BOOTSTRAP.csv', BOOT)
    fails = [p['check'] for p in PAR if not p['ok']]
    json.dump(dict(n_checks=len(PAR), failed=fails, checks=PAR), open(HERE / 'PARITY.json', 'w'), indent=1, default=float)
    print('rows', len(ROWS), 'parity', len(PAR), 'failed', fails)


if __name__ == '__main__':
    main()
