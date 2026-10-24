"""Amendment 2 of PREREG.md: the PRM defines the IU-PCR subspace (5 PRMs, PRM-backbone N16).

    python run_prm_plane.py fit <prm>|all   -> plane/scores/SCORES_<prm>.npz, plane/scores/FIT_<prm>.json
    python run_prm_plane.py evaluate        -> plane/RESULTS.csv, plane/CONTRASTS.csv, plane/DIAGNOSTICS.csv,
                                               plane/VERDICT.json

No label enters a fit. The registered DUFS graph of X17 is rebuilt through the split call and must reproduce the
saved DUFSLIU_PRM16 scores (max |diff| < 1e-8) before the new arms are scored.
"""
from __future__ import annotations
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

from run import (E, HERE, IU_FIT_DEFAULTS, REGISTERED_EPOCHS, REGISTERED_K, REGISTERED_LAMBDA, REGISTERED_SEEDS,
                 SCORES, adapted_dufs_soft_gates, build_graph_from_features, liu_external_target, load_unlabeled,
                 named, orient_to, pooled_zscore_features, sha256, upcr_fit)

OUT = HERE / 'plane'; SC = OUT / 'scores'
ARMS = ('PPLIU_PRM16', 'PPREF_PRM16')


def fit(prm):
    U = load_unlabeled(prm); zp = U['zp']; names = ['PRM_zp'] + U['names']
    F, keep = pooled_zscore_features(np.column_stack([zp, U['ch']]))
    assert keep[0], 'the PRM channel must be kept'
    base = upcr_fit(F, **IU_FIT_DEFAULTS)
    gates, _ = adapted_dufs_soft_gates(F, seeds=REGISTERED_SEEDS, epochs=REGISTERED_EPOCHS)
    graph = build_graph_from_features(F, gates=gates, k=REGISTERED_K)
    s_dufs, _ = orient_to(liu_external_target(F, base.rho_hat, graph=graph, lambda_=REGISTERED_LAMBDA).w @ F, zp)
    parity = float(np.max(np.abs(s_dufs - np.load(SCORES / f'SCORES_{prm}.npz')['score__DUFSLIU_PRM16'])))
    if parity > 1e-8: raise SystemExit(f'{prm}: rebuilt DUFS graph does not reproduce DUFSLIU_PRM16 ({parity})')
    C = F @ F.T / F.shape[1]; c = C[1:, 0]
    B = np.zeros((C.shape[0], 2)); B[0, 0] = 1.0; B[1:, 1] = c                # PRM axis + PRM-aligned telemetry
    smc = float(c @ np.linalg.solve(C[1:, 1:], c))                              # PRM's squared multiple correlation
    targets = {'PPLIU_PRM16': base.rho_hat, 'PPREF_PRM16': np.r_[smc, c]}
    scores, fits = {}, {'graph_parity_max_abs_diff_vs_DUFSLIU_PRM16': parity, 'g2_hat': float(base.g2_hat),
                        'prm_smc': smc, 'r2_prm_on_aligned_composite': float((c @ c) ** 2 / (c @ C[1:, 1:] @ c) / (c @ c)),
                        'inputs': U['files']}
    for arm, rho in targets.items():
        r = liu_external_target(F, rho, graph=graph, lambda_=REGISTERED_LAMBDA, subspace=B)
        s, flipped = orient_to(r.w @ F, zp); scores[arm] = s
        w = r.w * (-1 if flipped else 1)
        fits[arm] = dict(weights=named(w, keep, names), flipped=flipped, prm_weight=float(w[0]),
                         prm_share=float(abs(w[0]) / np.abs(w).sum()), rho=named(rho, keep, names), **r.diagnostics)
    SC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SC / f'SCORES_{prm}.npz', offsets=U['off'], row_ids=U['ids'],
                        **{f'score__{k}': v for k, v in scores.items()})
    (SC / f'FIT_{prm}.json').write_text(json.dumps(fits, indent=1))
    print(prm, 'parity', parity, {a: round(fits[a]['prm_weight'], 4) for a in ARMS}, flush=True)


def evaluate():
    from evaluate_dufs import contrast, diagnostics, prm_best, prm_share
    results, contrasts, diags, verdict = [], [], [], {}
    for prm in E.PRMS:
        c = E.load_prmbench(prm)
        P = np.load(SC / f'SCORES_{prm}.npz'); M = np.load(SCORES / f'SCORES_{prm}.npz')
        for Z in (P, M): assert np.array_equal(Z['offsets'], c.off) and np.array_equal(Z['row_ids'].astype(str), c.ids)
        for a in ARMS: c.arms[a] = P[f'score__{a}'].astype(float)
        c.arms['DUFSLIU_PRM16'] = M['score__DUFSLIU_PRM16'].astype(float)
        best = prm_best(c); W = E.boot_weights(c); cache = {}
        ch = E.channels(c, 'N16'); zp = E.answer_z(c.arms['PRM_Q80'], c.off)
        pos_z = E.answer_z(c.extra['position'], c.off); len_z = E.answer_z(c.extra['log_step_tokens'], c.off)
        F, _ = pooled_zscore_features(np.column_stack([zp, ch]))
        lr = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000).fit(F.T, c.err.astype(int))
        lr_z = E.answer_z(lr.decision_function(F.T), c.off)
        for arm in (best, 'CHAIN_N16', 'EQ_N16', 'DUFSLIU_PRM16') + ARMS:
            g, _, _ = E.measure(c, arm, 'GAP', fast=True); q, _, _ = E.measure(c, arm, 'Q80', fast=True)
            results.append(dict(prm=prm, arm=arm, within_auc=g['within_auc'], GAP=g['official'], Q80=q['official'],
                                n_auc=g['n_auc'], prm_weight_share=prm_share(c, zp, ch, c.arms[arm]), is_prm_best=arm == best))
            diags.append(dict(prm=prm, arm=arm, **diagnostics(c, arm, pos_z, len_z, lr_z)))
        for a in ARMS:
            for b in ('CHAIN_N16', 'DUFSLIU_PRM16', best, 'EQ_N16'):
                contrasts += contrast(c, a, b, W, cache)
        print(prm, 'evaluated', flush=True)
    E.write_csv(OUT / 'RESULTS.csv', results); E.write_csv(OUT / 'CONTRASTS.csv', contrasts)
    E.write_csv(OUT / 'DIAGNOSTICS.csv', diags)
    get = {(r['prm'], r['contrast'], r['metric']): r for r in contrasts}
    for a in ARMS:
        o = [get[(p, f'{a} - CHAIN_N16', 'official')] for p in E.PRMS]
        w = [get[(p, f'{a} - CHAIN_N16', 'within_auc')] for p in E.PRMS]
        d = [get[(p, f'{a} - DUFSLIU_PRM16', 'within_auc')] for p in E.PRMS]
        noninf = all(x['ci_low'] > -0.5 for x in o) and all(x['ci_low'] > -0.005 for x in w)
        better = sum(x['ci_low'] > 0 for x in o)
        verdict[a] = dict(
            replacement='replaces and improves' if noninf and better >= 3 else 'can replace' if noninf else 'cannot replace',
            prmscore_vs_chain=[round(x['delta'], 2) for x in o], within_vs_chain=[round(x['delta'], 4) for x in w],
            within_vs_dufsliu=[round(x['delta'], 4) for x in d],
            prm_subspace_helps_dufsliu=sum(x['ci_low'] > 0 for x in d) >= 3, n_ci_above0_vs_dufsliu=sum(x['ci_low'] > 0 for x in d))
    verdict['primary'] = 'PPLIU_PRM16'
    verdict['inputs'] = {p.name: sha256(p) for p in sorted(SC.glob('*.npz'))}
    (OUT / 'VERDICT.json').write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items() if k != 'inputs'}, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        for p in (E.PRMS if sys.argv[2] == 'all' else [sys.argv[2]]): fit(p)
    elif sys.argv[1] == 'evaluate':
        evaluate()
    else:
        raise SystemExit('usage: run_prm_plane.py fit <prm>|all | evaluate')
