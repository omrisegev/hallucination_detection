"""Ablation on PRMBench (Omri, 2026-10-24): GAP flag rule x AUTO placement x rank-one SML fit. CPU only, label-free scoring.

Question: with GAP, with AUTO (L-SML groups the PRM with the channels itself, the PRM is not forced into its own group),
and with the off-diagonal rank-one fit of SML (fusion_utils.rank1_offdiag_fit, v >= 0) instead of the eigenvector of the
zero-diagonal covariance, what PRMScore do we get, and what does each change add?

Inputs (stage-1 exports of results/prm_boost_collection_v1, the same files evaluate_collection.py reads):
  $COLLECTION_DATA/prmbench/ABL5_SCORES_<prm>.npz   reward, offsets, shipped arm scores (parity anchors)
  $COLLECTION_DATA/prmbench/ABL5_CHANNELS_<prm>.npz channels__N16 / channels__N13 (answer-z, oriented bank)
Arms (bank B = N16 and N13; z(PRM) = answer-z of the logit risk, as in ablation5.py):
  CHAIN_<B>_<mode>  L-SML on the channels, then SML over [z(PRM), answer-z group scores]  (shipped placement)
  AUTO_<B>_<mode>   L-SML over [z(PRM), channels] with its own group detection
  mode signed = shipped sml_fuse_signed + anchor flip toward z(PRM); nnr1 = rank1_offdiag_fit (v >= 0), no flip.
  Same construction as results/sml_weight_estimator_v1/socratic_sign/prm_placement.py.
Rules Q80 / GAP / K2 and every metric come from evaluate_collection.py (unchanged code, imported).
Parity: CHAIN_<B>_signed must equal the exported score__CHAIN_<B> (max |diff| reported, gate 1e-9).

    COLLECTION_DATA=<dir> python run.py [prm ...]
"""
import csv, json, os, sys, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
import evaluate_collection as EC  # noqa: E402
import spectral_utils.fusion_utils as FU  # noqa: E402
from scipy.linalg import eigh  # noqa: E402

SHIPPED = FU.sml_fuse_signed


def sml_nnr1(*c, gates=None, small_m_guard=False):
    X = np.column_stack(c); w = FU.rank1_offdiag_fit(np.cov(X.T), nonneg=True)
    if not np.any(w > 0): w = np.ones(X.shape[1])
    return X @ w, w


MODES = {'signed': SHIPPED, 'nnr1': sml_nnr1}


def az(v, off):
    return EC.answer_z(np.asarray(v, float), off)


def risk(r):
    p = np.clip(1 - np.asarray(r, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def lsml(V, mode, groups=None, anchor=None):
    FU.sml_fuse_signed = MODES[mode]
    try:
        s, d = FU.lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    finally:
        FU.sml_fuse_signed = SHIPPED
    s = np.asarray(s, float)
    if mode == 'signed':
        ref = V[:, 0] if anchor is None else anchor
        if np.corrcoef(s, ref)[0, 1] < 0: s = -s
    return s, d


def build(prm):
    S = dict(np.load(EC.DATA / 'prmbench' / f'ABL5_SCORES_{prm}.npz'))
    C = dict(np.load(EC.DATA / 'prmbench' / f'ABL5_CHANNELS_{prm}.npz'))
    o = S['offsets'].astype(np.int64); assert np.array_equal(C['offsets'], o)
    zp = az(risk(S['reward']), o)
    arms, info = {}, {}
    for bk in ('N16', 'N13'):
        V = np.asarray(C[f'channels__{bk}'], float); VP = np.column_stack([zp, V])
        for mode in MODES:
            st, dt = lsml(V, mode, anchor=V.mean(1))
            vg = np.asarray(dt['virtual_classifiers'], float)
            VC = np.column_stack([zp] + [az(vg[:, g], o) for g in range(vg.shape[1])])
            arms[f'CHAIN_{bk}_{mode}'], _ = lsml(VC, mode, groups=np.arange(VC.shape[1]))
            arms[f'AUTO_{bk}_{mode}'], da = lsml(VP, mode)
            ca = np.asarray(da['c'], int)
            info[f'AUTO_{bk}_{mode}'] = {'K': int(da['K']), 'prm_group_size': int((ca == ca[0]).sum()),
                                         'prm_alone_in_group': bool((ca == ca[0]).sum() == 1)}
            for k in (f'CHAIN_{bk}_{mode}', f'AUTO_{bk}_{mode}'):
                w = np.linalg.lstsq(VP, arms[k], rcond=None)[0]
                info.setdefault(k, {}).update(prm_weight_share=float(abs(w[0]) / np.abs(w).sum()),
                                              fit_r2=float(1 - np.var(arms[k] - VP @ w) / np.var(arms[k])))
        info[f'parity_CHAIN_{bk}'] = float(np.max(np.abs(arms[f'CHAIN_{bk}_signed'] - S[f'score__CHAIN_{bk}'])))
    return arms, info


def main():
    prms = sys.argv[1:] or EC.PRMS
    out = Path(os.environ.get('ABL_OUT', HERE)); out.mkdir(parents=True, exist_ok=True)
    rows, cons, infos = [], [], {}
    for prm in prms:
        c = EC.load_prmbench(prm)
        new, info = build(prm); infos[prm] = info
        for bk in ('N16', 'N13'):
            if os.environ.get('ABL_SMOKE') == '1': break           # synthetic smoke data has no shipped scores
            assert info[f'parity_CHAIN_{bk}'] < 1e-9, (prm, bk, info[f'parity_CHAIN_{bk}'])
        c.arms.update(new)
        ref = 'PRM_Q80' if prm in ('rlhflow_prm', 'math_shepherd') else 'PRM_Q80_PROB'   # GAP-best scale, as STATUS_MAIN
        arms = [ref, 'EQ_N16', 'TELE_N16'] + list(new)
        for a in arms:
            for rule in ('Q80', 'GAP', 'K2'):
                r, _, _ = EC.measure(c, a, rule, fast=True); r['prm_ref_scale'] = ref; rows.append(r)
            print(prm, a, {rr['rule']: round(rr['official'], 2) for rr in rows[-3:]}, round(rows[-1]['within_auc'], 4), flush=True)
        W = EC.boot_weights(c)
        steps = [  # (label, a, b): the ablation path for N16, then the other cells against the PRM
            ('1 shipped: CHAIN signed, Q80 vs PRM', ('CHAIN_N16_signed', 'Q80'), (ref, 'Q80')),
            ('2 +GAP (CHAIN signed)', ('CHAIN_N16_signed', 'GAP'), ('CHAIN_N16_signed', 'Q80')),
            ('3 +AUTO (signed, GAP)', ('AUTO_N16_signed', 'GAP'), ('CHAIN_N16_signed', 'GAP')),
            ('4 +nnr1 (AUTO, GAP)', ('AUTO_N16_nnr1', 'GAP'), ('AUTO_N16_signed', 'GAP')),
            ('nnr1 on CHAIN (GAP)', ('CHAIN_N16_nnr1', 'GAP'), ('CHAIN_N16_signed', 'GAP')),
            ('AUTO nnr1 vs CHAIN signed (GAP)', ('AUTO_N16_nnr1', 'GAP'), ('CHAIN_N16_signed', 'GAP')),
            ('AUTO nnr1 vs CHAIN signed (K2)', ('AUTO_N16_nnr1', 'K2'), ('CHAIN_N16_signed', 'K2')),
        ]
        for a in new:
            for rule in ('GAP', 'K2', 'Q80'):
                steps.append((f'{a} vs PRM ({rule})', (a, rule), (ref, rule)))
        for lab, a, b in steps:
            for r in EC.contrast(c, a, b, W): r['step'] = lab; cons.append(r)
    EC.write_csv(out / 'RESULTS.csv', rows); EC.write_csv(out / 'CONTRASTS.csv', cons)
    (out / 'DIAG.json').write_text(json.dumps(infos, indent=1))
    print('DONE', out)


if __name__ == '__main__':
    main()
