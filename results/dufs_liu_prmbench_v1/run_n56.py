"""Amendment 1 of PREREG.md: the same arms on the Qwen3-8B telemetry, banks N16 and N56.

    python run_n56.py parity            rebuilt banks + canonical scorers vs published thinkfix_mask1 numbers,
                                        and the CHAIN port vs the exported CHAIN_N16 -> n56/PARITY.csv
    python run_n56.py fit <bank> <set>  bank in {N16, N56}, set in {tele, prm} -> n56/scores/SCORES_<bank>_<set>.npz
    python run_n56.py evaluate          -> n56/RESULTS.csv, n56/CONTRASTS.csv, n56/DIAGNOSTICS.csv, n56/VERDICT.json

Label access as in run.py: label-free and PRM pseudo-label arms never receive a label; label arms are cross-fitted by
source-group fold and read labels only on calibration answers.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '2')
os.environ.setdefault('COLLECTION_DATA', '/Users/osegev/Desktop/hd_collection_data')
import json, sys, time
from pathlib import Path
import numpy as np

from run import (BUDGETS, BUDGET_SEED, DATA, E, HERE, N_FOLDS, ROOT, covariance_target, liu_external_target,
                 named, orient_to, pooled_zscore_features, registered_dufs_liu_fit, sha256, source_folds, steps_of,
                 IU_FIT_DEFAULTS, REGISTERED_K, REGISTERED_LAMBDA, upcr_fit, LogisticRegression,
                 pseudo_anchor_laplacian_fit, target_anchored_laplacian_fit)
from spectral_utils.bank_rebuild import build_nested_bank, chain_lsml, level_score
from spectral_utils.fusion_utils import lsml_continuous
from spectral_utils.laplacian_upcr import build_graph_from_features
from spectral_utils.target_anchored_laplacian import positive_correlation_gates

OUT = HERE / 'n56'; SC = OUT / 'scores'
FEATURES = Path('/Users/osegev/Desktop/hallucination_detection/results/advisor_completion_2026-10-11/inputs/'
                'features_thinkfix_mask1.npz')
BANKS_JSON = ROOT / 'results/thinkfix_rescore_v1/thinkfix_mask1/nested_banks/BANKS.json'
PUBLISHED = Path('/Users/osegev/Desktop/hallucination_detection/results/thinkfix_rescore_v1/thinkfix_mask1/'
                 'cont_baselines/ABSOLUTE.csv')
CELL, PRM = 'prmbench_qwen3_8b', 'qwen_prm_7b'
LABEL_ARMS = ('TALIU', 'RHOLIU', 'RHOPCR', 'LR')


# ------------------------------------------------------------------------------------------------ inputs
def load_inputs(bank):
    """Rebuilt Qwen3-8B bank in the evaluator's answer order, the level score and the PRM channel. No labels."""
    Fz = np.load(FEATURES); cells = Fz['cells'].astype(str); off_all = Fz['offsets'].astype(np.int64)
    ans = np.flatnonzero(cells == CELL); ids = Fz['row_ids'][ans].astype(str)
    rows = np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in ans])
    off = np.r_[0, np.cumsum(np.diff(off_all)[ans])].astype(np.int64)
    cols = json.loads(BANKS_JSON.read_text())[bank]
    V = build_nested_bank(Fz['values'][rows], Fz['names'], Fz['digit_active'][rows], off, cols)
    lev = level_score(Fz['values'][rows], Fz['names'], off)
    S = np.load(DATA / 'prmbench' / f'ABL5_SCORES_{PRM}.npz'); eids = S['row_ids'].astype(str); eoff = S['offsets']
    pos = {u: j for j, u in enumerate(ids)}; order = np.array([pos[u] for u in eids])
    lens = np.diff(off)[order]; assert np.array_equal(lens, np.diff(eoff)), 'step counts differ from the evaluator'
    steps = np.concatenate([np.arange(off[j], off[j + 1]) for j in order])
    zp = E.answer_z(S['score__PRM_Q80'].astype(float), eoff)
    return dict(V=V[steps], lev=lev[steps], zp=zp, off=eoff.astype(np.int64), ids=eids, names=list(cols),
                groups=S['source_group'].astype(str), files={FEATURES.name: sha256(FEATURES)})


def orient(V, lev):
    """Label-free orientation by the sign of each channel's pooled correlation with the level score."""
    sg = np.array([np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1. if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
    return V * sg, sg


# ------------------------------------------------------------------------------------------------ parity
def parity():
    OUT.mkdir(parents=True, exist_ok=True)
    pub = {(r['bank'], r['scorer']): float(r['prmb_within_auc']) for r in E.read_csv(PUBLISHED)}
    c = E.load_prmbench(PRM); rows, worst = [], 0.
    for bank in ('N16', 'N56'):
        I = load_inputs(bank); V = I['V']
        sd = V.std(0); keep = sd > 0; F = ((V[:, keep] - V[:, keep].mean(0)) / sd[keep]).T
        iu = upcr_fit(F, **IU_FIT_DEFAULTS).w @ F
        ls, _ = lsml_continuous(*[V[:, j] for j in range(V.shape[1])])
        for name, s in (('IU_PCR', iu), ('LSML_CONT', np.asarray(ls, float))):
            mine = float(np.nanmean(E.per_answer_auc(c.err, s, c.off)))
            d = abs(mine - pub[(bank, name)]); worst = max(worst, d)
            rows.append(dict(check=f'{bank} {name} within-AUC vs published', published=pub[(bank, name)], recomputed=mine, abs_diff=d))
    Ch = np.load(DATA / 'prmbench' / f'ABL5_CHANNELS_{PRM}.npz')
    ch = Ch['channels__N16'].astype(float); zp = E.answer_z(c.arms['PRM_Q80'], c.off)
    chain, _, _ = chain_lsml(zp, ch, c.off)
    d_chain = float(np.max(np.abs(chain - c.arms['CHAIN_N16'])))
    rows.append(dict(check='CHAIN port vs exported CHAIN_N16 (PRM backbone, Qwen-7B)', published=0., recomputed=d_chain, abs_diff=d_chain))
    E.write_csv(OUT / 'PARITY.csv', rows)
    for r in rows: print(r)
    if worst > 1e-9 or d_chain > 1e-6: raise SystemExit('PARITY FAILED')
    print('PARITY PASS')


# ------------------------------------------------------------------------------------------------ fit
def cross_fit(F, keep, names, anchor, err, off, groups, graph, rho_hat, scores, fits, prefix):
    """TA-LIU here is target_anchored_laplacian_fit split into its parts: the unchanged gate function and graph
    builder, then the projected solve with IU-PCR's rho_hat (identical weights; tests/test_anchored_liu.py). This
    skips the per-fit ARPACK connectivity diagnostic of laplacian_iu_path, which is not used."""
    folds = source_folds(groups); n = off[-1]
    for budget, draws in BUDGETS.items():
        for a in LABEL_ARMS: scores[f'{prefix}{a}@{budget}'] = np.zeros((draws, n))
        for d in range(draws):
            for f in range(N_FOLDS):
                train = np.flatnonzero(folds != f); test = steps_of(np.flatnonzero(folds == f), off)
                cal = train if budget == 'FULL' else np.sort(
                    np.random.default_rng([BUDGET_SEED, int(budget), d, f]).choice(train, int(budget), replace=False))
                cs = steps_of(cal, off); y = np.zeros(n); y[cs] = err[cs]
                rec = {'n_cal_answers': int(len(cal)), 'cal_error_rate': float(y[cs].mean())}
                gate = positive_correlation_gates(F, y, indices=cs, require_two_classes=True)
                ta_graph = build_graph_from_features(F, gates=gate.gates, k=REGISTERED_K)
                ta = liu_external_target(F, rho_hat, graph=ta_graph, lambda_=REGISTERED_LAMBDA)
                s, flipped = orient_to(ta.w @ F, anchor)
                scores[f'{prefix}TALIU@{budget}'][d, test] = s[test]
                rec['TALIU'] = dict(gates=named(gate.gates, keep, names), flipped=flipped,
                                    fallback=gate.fallback_reason)
                rho = covariance_target(F, y, indices=cs)
                for arm, lam in (('RHOLIU', REGISTERED_LAMBDA), ('RHOPCR', 0.0)):
                    r = liu_external_target(F, rho, graph=graph, lambda_=lam)
                    scores[f'{prefix}{arm}@{budget}'][d, test] = (r.w @ F)[test]
                    rec[arm] = dict(weights=named(r.w, keep, names), **r.diagnostics)
                lr = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000)
                lr.fit(F[:, cs].T, err[cs].astype(int))
                scores[f'{prefix}LR@{budget}'][d, test] = lr.decision_function(F[:, test].T)
                rec['LR'] = dict(weights=named(lr.coef_[0], keep, names))
                fits[f'{prefix}{budget}/d{d}/f{f}'] = rec


def fit(bank, which):
    t0 = time.time(); SC.mkdir(parents=True, exist_ok=True)
    I = load_inputs(bank); Vo, sg = orient(I['V'], I['lev']); off, zp = I['off'], I['zp']
    scores, fits = {}, {'orientation_signs': named(sg, np.ones(len(sg), bool), I['names'])}
    if which == 'tele':
        X, names, anchor, prefix = Vo, I['names'], Vo.mean(1), ''
        s, _ = lsml_continuous(*[I['V'][:, j] for j in range(I['V'].shape[1])])
        scores['LSML'] = np.asarray(s, float)                                  # published recipe (parity row)
        scores['EQ'] = Vo.mean(1)
    else:
        X, names, anchor, prefix = np.column_stack([zp, Vo]), ['PRM_zp'] + I['names'], zp, 'PRM_'
        chain, tele, meta = chain_lsml(zp, I['V'], off)
        scores['CHAIN'] = chain; scores['EQ_PRM'] = X.mean(1)
        fits['CHAIN_groups'] = int(len(set(np.asarray(meta['c']).tolist())))
    F, keep = pooled_zscore_features(X)
    base = upcr_fit(F, **IU_FIT_DEFAULTS)
    scores[f'IU{"_PRM" if prefix else ""}'], fl0 = orient_to(base.w @ F, anchor)
    dl = registered_dufs_liu_fit(F)
    scores[f'DUFSLIU{"_PRM" if prefix else ""}'], fl1 = orient_to(dl.fit.w @ F, anchor)
    fits['IU'] = dict(weights=named(base.w, keep, names), flipped=fl0)
    fits['DUFSLIU'] = dict(weights=named(dl.fit.w, keep, names), flipped=fl1,
                           gates=named(dl.gate_diagnostics['raw_probabilities'], keep, names),
                           weight_cosine_vs_iu=dl.fit.diagnostics['weight_cosine_vs_iu'])
    if prefix:
        yp = E.topk_flags(zp, off, E.gap_counts(zp, off)).astype(float)
        pa = pseudo_anchor_laplacian_fit(F, yp, lambda_=REGISTERED_LAMBDA, k=REGISTERED_K,
                                         baseline_kwargs=dict(IU_FIT_DEFAULTS))
        scores['PALIU_PRM'], fl2 = orient_to(pa.fit.w @ F, zp)
        r = liu_external_target(F, covariance_target(F, yp), graph=dl.graph, lambda_=REGISTERED_LAMBDA)
        scores['PRHOLIU_PRM'], fl3 = orient_to(r.w @ F, zp)
        fits['PALIU'] = dict(gates=named(pa.gate_result.gates, keep, names), flipped=fl2,
                             weight_cosine_vs_iu=pa.fit.diagnostics['weight_cosine_vs_iu'])
        fits['PRHOLIU'] = dict(weights=named(r.w, keep, names), flipped=fl3, **r.diagnostics)
    t1 = time.time()
    err = E.load_prmbench(PRM).err                                              # labels enter only here
    cross_fit(F, keep, names, anchor, err, off, I['groups'], dl.graph, base.rho_hat, scores, fits, prefix)
    tag = f'{bank}_{which}'
    np.savez_compressed(SC / f'SCORES_{tag}.npz', offsets=off, row_ids=I['ids'],
                        **{f'score__{k}_{bank}': v for k, v in scores.items()})
    fits.update(seconds_label_free=round(t1 - t0, 1), seconds_total=round(time.time() - t0, 1), inputs=I['files'])
    (SC / f'FIT_{tag}.json').write_text(json.dumps(fits, indent=1))
    print(tag, 'done', fits['seconds_total'], 's', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'parity':
        parity()
    elif cmd == 'fit':
        fit(sys.argv[2], sys.argv[3])
    elif cmd == 'evaluate':
        from evaluate_n56 import main as _m; _m()
    else:
        raise SystemExit(f'unknown command {cmd}')
