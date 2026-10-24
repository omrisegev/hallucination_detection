"""DUFS-LIU on PRMBench localization (pre-registration: PREREG.md in this directory).

    python run.py parity          evaluator vs tables/STATUS_MAIN.csv of prm_boost_collection_v1 -> PARITY.csv
    python run.py fit <prm>|all   fit every pre-registered arm -> scores/SCORES_<prm>.npz, scores/FIT_<prm>.json
    python run.py evaluate        metrics, contrasts, manifold diagnostics -> RESULTS.csv, CONTRASTS.csv, DIAGNOSTICS.csv

Label access. The label-free arms (IU_*, DUFSLIU_*) and the PRM pseudo-label arms (PALIU, PRHOLIU, PRHOPCR) are
fitted from the channels and the PRM score only; `fit_label_free` never receives a label. The label arms (TALIU,
RHOLIU, RHOPCR, LR) are cross-fitted by source-group fold and read labels only on their calibration answers.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '2')
os.environ.setdefault('COLLECTION_DATA', '/Users/osegev/Desktop/hd_collection_data')
import csv, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
import evaluate_collection as E  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.anchored_liu import (  # noqa: E402
    covariance_target, liu_external_target, orient_to, pooled_zscore_features, registered_dufs_liu_fit,
    split_liu_weights, REGISTERED_EPOCHS, REGISTERED_K, REGISTERED_LAMBDA, REGISTERED_SEEDS)
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS, build_graph_from_features  # noqa: E402
from spectral_utils.target_anchored_laplacian import (  # noqa: E402
    positive_correlation_gates, pseudo_anchor_laplacian_fit, target_anchored_laplacian_fit)
from spectral_utils.upcr import upcr_fit  # noqa: E402

DATA = Path(os.environ['COLLECTION_DATA'])
SCORES = HERE / 'scores'
FOLD_SEED, BUDGET_SEED, N_FOLDS = 20261021, 20261024, 5
BUDGETS = {'64': 5, '512': 5, 'FULL': 1}            # calibration answers -> number of draws
LABEL_ARMS = ('TALIU_PRM16', 'RHOLIU_PRM16', 'RHOPCR_PRM16', 'LR_PRM16')
SINGLE_ARMS = ('IU_N16', 'DUFSLIU_N16', 'IU_PRM16', 'DUFSLIU_PRM16', 'PALIU_PRM16', 'PRHOLIU_PRM16', 'PRHOPCR_PRM16')
REFS = ('CHAIN_N16', 'PRMG_N16', 'TELE_N16', 'EQ_N16', 'MEAN_N16', 'PRM_Q80', 'PRM_Q80_PROB', 'LSML_CONT')
# qwen_prm_7b, pure_prm, rlhflow_prm and math_shepherd ran the canonical calls (registered_dufs_liu_fit,
# pseudo_anchor_laplacian_fit, target_anchored_laplacian_fit). Their per-call ARPACK connectivity diagnostic stalled
# on qwen_prm_72b, which runs the same fits split into parts (env TALIU_SPLIT=1): identical weights
# (tests/test_anchored_liu.py; real-data check `python run.py talius_parity`), no connectivity diagnostic.
TALIU_CANONICAL_CALL = os.environ.get('TALIU_SPLIT') != '1'


def sha256(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


# ------------------------------------------------------------------------------------------------ inputs
def load_unlabeled(prm):
    """Channels, PRM score and identities. No label is read here."""
    S = np.load(DATA / 'prmbench' / f'ABL5_SCORES_{prm}.npz'); Ch = np.load(DATA / 'prmbench' / f'ABL5_CHANNELS_{prm}.npz')
    off = S['offsets'].astype(np.int64)
    assert np.array_equal(Ch['offsets'], off) and np.array_equal(Ch['row_ids'], S['row_ids'])
    ch = Ch['channels__N16'].astype(float); names = [str(x) for x in Ch['channel_names__N16']]
    zp = E.answer_z(S['score__PRM_Q80'].astype(float), off)
    return dict(off=off, ch=ch, names=names, zp=zp, ids=S['row_ids'].astype(str), groups=S['source_group'].astype(str),
                files={f'ABL5_SCORES_{prm}.npz': sha256(DATA / 'prmbench' / f'ABL5_SCORES_{prm}.npz'),
                       f'ABL5_CHANNELS_{prm}.npz': sha256(DATA / 'prmbench' / f'ABL5_CHANNELS_{prm}.npz')})


def step_labels(ids, off):
    """Step error labels exactly as evaluate_collection.load_prmbench builds them."""
    meta = E.prmbench_meta(); err = np.zeros(off[-1], bool)
    for j, i in enumerate(ids):
        a, b = off[j], off[j + 1]; assert b - a == meta[i]['n_steps'], i
        err[a:b] = np.isin(np.arange(1, b - a + 1), meta[i]['error_steps'])
    return err


def source_folds(groups):
    ug = np.unique(groups); rng = np.random.default_rng(FOLD_SEED)
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % N_FOLDS))
    return np.array([fold_of[g] for g in groups])


def steps_of(answers, off):
    return np.concatenate([np.arange(off[j], off[j + 1]) for j in answers]) if len(answers) else np.zeros(0, int)


def named(w, keep, names):
    return {n: float(v) for n, v in zip([n for n, k in zip(names, keep) if k], w)}


# ------------------------------------------------------------------------------------------------ fit
def fit_label_free(U):
    """Label-free and PRM pseudo-label arms. Receives no label."""
    off, ch, zp = U['off'], U['ch'], U['zp']
    names16, names17 = U['names'], ['PRM_zp'] + U['names']
    F16, k16 = pooled_zscore_features(ch); F17, k17 = pooled_zscore_features(np.column_stack([zp, ch]))
    tele_anchor = ch.mean(1)
    yp = E.topk_flags(zp, off, E.gap_counts(zp, off)).astype(float)       # the PRM's own GAP flags
    out, fit = {}, {'kept_N16': k16.tolist(), 'kept_X17': k17.tolist(), 'pseudo_label_rate': float(yp.mean())}

    def record(name, w, F, keep, nm, anchor, extra=None):
        s, flipped = orient_to(w @ F, anchor)
        out[name] = s
        fit[name] = dict(weights=named(w * (-1 if flipped else 1), keep, nm), flipped=flipped, **(extra or {}))

    def dufs_liu(F):
        """Registered DUFS-LIU -> (w, raw gate probabilities, gate diagnostics, graph, fit diagnostics)."""
        if TALIU_CANONICAL_CALL:
            d = registered_dufs_liu_fit(F)
            return d.fit.w, d.gate_diagnostics, d.graph, d.fit.diagnostics
        gates, gd = adapted_dufs_soft_gates(F, seeds=REGISTERED_SEEDS, epochs=REGISTERED_EPOCHS)
        w, graph, diag = split_liu_weights(F, gates)
        return w, gd, graph, diag

    iu16 = upcr_fit(F16, **IU_FIT_DEFAULTS); record('IU_N16', iu16.w, F16, k16, names16, tele_anchor)
    w16, gd16, _, fd16 = dufs_liu(F16)
    record('DUFSLIU_N16', w16, F16, k16, names16, tele_anchor,
           dict(gates=named(gd16['raw_probabilities'], k16, names16),
                gate_effective_feature_count=gd16['effective_feature_count'],
                weight_cosine_vs_iu=fd16['weight_cosine_vs_iu'], graph_components=fd16['n_components']))
    iu17 = upcr_fit(F17, **IU_FIT_DEFAULTS)
    record('IU_PRM16', iu17.w, F17, k17, names17, zp, dict(rho_hat=named(iu17.rho_hat, k17, names17)))
    w17, gd17, graph17, fd17 = dufs_liu(F17)
    record('DUFSLIU_PRM16', w17, F17, k17, names17, zp,
           dict(gates=named(gd17['raw_probabilities'], k17, names17),
                gate_effective_feature_count=gd17['effective_feature_count'],
                weight_cosine_vs_iu=fd17['weight_cosine_vs_iu'], graph_components=fd17['n_components']))
    if TALIU_CANONICAL_CALL:
        pa = pseudo_anchor_laplacian_fit(F17, yp, lambda_=REGISTERED_LAMBDA, k=REGISTERED_K,
                                         baseline_kwargs=dict(IU_FIT_DEFAULTS))
        w_pa, pa_gate, pa_cos = pa.fit.w, pa.gate_result, pa.fit.diagnostics['weight_cosine_vs_iu']
    else:
        pa_gate = positive_correlation_gates(F17, yp, indices=None, require_two_classes=False)
        w_pa, _, pa_diag = split_liu_weights(F17, pa_gate.gates); pa_cos = pa_diag['weight_cosine_vs_iu']
    record('PALIU_PRM16', w_pa, F17, k17, names17, zp,
           dict(gates=named(pa_gate.gates, k17, names17), correlations=named(pa_gate.correlations, k17, names17),
                gate_fallback=pa_gate.fallback_reason, weight_cosine_vs_iu=pa_cos))
    rho_p = covariance_target(F17, yp)
    for name, lam in (('PRHOLIU_PRM16', REGISTERED_LAMBDA), ('PRHOPCR_PRM16', 0.0)):
        r = liu_external_target(F17, rho_p, graph=graph17, lambda_=lam)
        record(name, r.w, F17, k17, names17, zp, dict(rho=named(rho_p, k17, names17), **r.diagnostics))
    return out, fit, dict(F17=F17, k17=k17, graph17=graph17, rho_hat17=iu17.rho_hat)


def fit_label_arms(U, err, cache):
    """Cross-fitted label arms. Labels are read only on each fit's calibration steps."""
    off, zp, F17, k17, graph = U['off'], U['zp'], cache['F17'], cache['k17'], cache['graph17']
    names17 = ['PRM_zp'] + U['names']
    folds = source_folds(U['groups']); n_ans = len(off) - 1; n = off[-1]
    scores = {f'{a}@{b}': np.zeros((d, n)) for a in LABEL_ARMS for b, d in BUDGETS.items()}
    fits = {}
    for budget, draws in BUDGETS.items():
        for d in range(draws):
            for f in range(N_FOLDS):
                train = np.flatnonzero(folds != f); test_steps = steps_of(np.flatnonzero(folds == f), off)
                if budget == 'FULL':
                    cal = train
                else:
                    cal = np.sort(np.random.default_rng([BUDGET_SEED, int(budget), d, f]).choice(train, int(budget), replace=False))
                cal_steps = steps_of(cal, off)
                y = np.zeros(n); y[cal_steps] = err[cal_steps]                 # only calibration labels exist here
                key = f'{budget}/d{d}/f{f}'
                rec = {'n_cal_answers': int(len(cal)), 'n_cal_steps': int(len(cal_steps)),
                       'cal_error_rate': float(y[cal_steps].mean())}
                if TALIU_CANONICAL_CALL:
                    ta = target_anchored_laplacian_fit(F17, y, cal_steps, lambda_=REGISTERED_LAMBDA, k=REGISTERED_K,
                                                       baseline_kwargs=dict(IU_FIT_DEFAULTS))
                    w_ta, gate = ta.fit.w, ta.gate_result
                else:   # the same fit split into its parts; skips laplacian_iu_path's ARPACK diagnostic (identical w)
                    gate = positive_correlation_gates(F17, y, indices=cal_steps, require_two_classes=True)
                    w_ta = liu_external_target(F17, cache['rho_hat17'], lambda_=REGISTERED_LAMBDA,
                                               graph=build_graph_from_features(F17, gates=gate.gates, k=REGISTERED_K)).w
                s, flipped = orient_to(w_ta @ F17, zp)
                scores[f'TALIU_PRM16@{budget}'][d, test_steps] = s[test_steps]
                rec['TALIU'] = dict(gates=named(gate.gates, k17, names17), fallback=gate.fallback_reason,
                                    flipped=flipped, weights=named(w_ta * (-1 if flipped else 1), k17, names17))
                rho = covariance_target(F17, y, indices=cal_steps)
                for arm, lam in (('RHOLIU_PRM16', REGISTERED_LAMBDA), ('RHOPCR_PRM16', 0.0)):
                    r = liu_external_target(F17, rho, graph=graph, lambda_=lam)
                    scores[f'{arm}@{budget}'][d, test_steps] = (r.w @ F17)[test_steps]
                    rec[arm.split('_')[0]] = dict(weights=named(r.w, k17, names17), **r.diagnostics)
                lr = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000)
                lr.fit(F17[:, cal_steps].T, err[cal_steps].astype(int))
                scores[f'LR_PRM16@{budget}'][d, test_steps] = lr.decision_function(F17[:, test_steps].T)
                rec['LR'] = dict(weights=named(lr.coef_[0], k17, names17), intercept=float(lr.intercept_[0]))
                fits[key] = rec
    return scores, fits


def fit(prm):
    t0 = time.time(); SCORES.mkdir(parents=True, exist_ok=True)
    U = load_unlabeled(prm)
    single, fit_single, cache = fit_label_free(U)
    t1 = time.time()
    err = step_labels(U['ids'], U['off'])                                   # labels enter only after the label-free arms
    label_scores, label_fits = fit_label_arms(U, err, cache)
    arrays = {f'score__{k}': v for k, v in single.items()}
    arrays.update({f'score__{k}': v for k, v in label_scores.items()})
    np.savez_compressed(SCORES / f'SCORES_{prm}.npz', offsets=U['off'], row_ids=U['ids'], **arrays)
    meta = dict(prm=prm, inputs=U['files'], label_free=fit_single, label=label_fits,
                seconds_label_free=round(t1 - t0, 1), seconds_total=round(time.time() - t0, 1),
                budgets=BUDGETS, fold_seed=FOLD_SEED, budget_seed=BUDGET_SEED)
    (SCORES / f'FIT_{prm}.json').write_text(json.dumps(meta, indent=1))
    print(prm, 'done', meta['seconds_total'], 's', flush=True)


# ------------------------------------------------------------------------------------------------ parity
def parity():
    rows = [r for r in E.read_csv(ROOT / 'results/prm_boost_collection_v1/tables/STATUS_MAIN.csv')
            if r['benchmark'] == 'prmbench' and r['row'] == r['arm']
            and r['arm'] in ('CHAIN_N16', 'TELE_N16', 'EQ_N16', 'PRM_Q80', 'PRM_Q80_PROB')]
    out, worst = [], 0.
    for prm in E.PRMS:
        c = E.load_prmbench(prm)
        for r in rows:
            if r['prm'] != prm: continue
            m, _, _ = E.measure(c, r['arm'], 'GAP', fast=True)
            for col, mine in (('GAP', m['official']), ('within_auc', m['within_auc'])):
                d = abs(mine - float(r[col])); worst = max(worst, d)
                out.append(dict(prm=prm, arm=r['arm'], metric=col, published=float(r[col]), recomputed=mine, abs_diff=d))
    E.write_csv(HERE / 'PARITY.csv', out)
    print('parity rows', len(out), 'max abs diff', worst)
    if worst > 1e-9: raise SystemExit('PARITY FAILED')
    print('PARITY PASS')


def talius_parity(prm='qwen_prm_7b'):
    """Real-data check: the split calls reproduce the canonical-call scores saved for `prm` (label-free arms and
    TALIU@FULL)."""
    global TALIU_CANONICAL_CALL
    TALIU_CANONICAL_CALL = False
    U = load_unlabeled(prm); split_scores, _, cache = fit_label_free(U)
    Z = np.load(SCORES / f'SCORES_{prm}.npz')
    label_free = {a: float(np.max(np.abs(split_scores[a] - Z[f'score__{a}']))) for a in split_scores}
    err = step_labels(U['ids'], U['off']); folds = source_folds(U['groups']); F17, zp = cache['F17'], U['zp']
    saved = Z['score__TALIU_PRM16@FULL'][0]; worst = 0.
    for f in range(N_FOLDS):
        cal = steps_of(np.flatnonzero(folds != f), U['off']); test = steps_of(np.flatnonzero(folds == f), U['off'])
        y = np.zeros(U['off'][-1]); y[cal] = err[cal]
        gate = positive_correlation_gates(F17, y, indices=cal, require_two_classes=True)
        w = liu_external_target(F17, cache['rho_hat17'], lambda_=REGISTERED_LAMBDA,
                                graph=build_graph_from_features(F17, gates=gate.gates, k=REGISTERED_K)).w
        s, _ = orient_to(w @ F17, zp); worst = max(worst, float(np.max(np.abs(s[test] - saved[test]))))
    out = dict(prm=prm, taliu_full_folds=N_FOLDS, taliu_full_max_abs_score_diff=worst,
               label_free_max_abs_score_diff=label_free,
               passed=worst < 1e-8 and max(label_free.values()) < 1e-8)
    (HERE / 'TALIU_SPLIT_PARITY.json').write_text(json.dumps(out, indent=1)); print(out)
    if not out['passed']: raise SystemExit('TA-LIU split parity FAILED')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'parity'
    if cmd == 'parity':
        parity()
    elif cmd == 'talius_parity':
        talius_parity()
    elif cmd == 'fit':
        prms = E.PRMS if sys.argv[2] == 'all' else [sys.argv[2]]
        for p in prms: fit(p)
    elif cmd == 'evaluate':
        from evaluate_dufs import main as _m; _m()
    else:
        raise SystemExit(f'unknown command {cmd}')
