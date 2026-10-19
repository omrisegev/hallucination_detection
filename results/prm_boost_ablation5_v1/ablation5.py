"""Five-arm ablation of PRM boosting on PRMBench (Omri, 2026-10-19). CPU only, label-free fusion.

Omri's five arms, per PRM and per bank (N13 digit-free, N16 with 3 digit channels, S5 label-selected reference):
  1 TELE_<bk>   L-SML on the bank channels only (no PRM)                 (= bank sweep TELE_<bk>)
  2 MEAN_<bk>   simple mean of the same answer-z, label-free-oriented channels (no PRM)      NEW
  3 PRM_native  the PRM as it is, with its own 0.5 threshold
  4 CHAIN_<bk>  L-SML boosting: PRM + bank, PRM its own group (= bank sweep CHAIN_<bk>); PRMG_<bk> also kept
  5 HALF_<bk>   1/2 z(PRM) + 1/2 mean(bank)                                  (= controls HALF_N16)
    EQ_<bk>     mean of z(PRM) and every channel, PRM weight 1/(K+1)                               NEW
P-variants (NEW): HALFP/EQP/CHAINP_<bank> = HALF/EQ/CHAIN with the PRM channel as z(1 - reward) instead of z(logit(1 - reward)).
Extra controls (NEW): PRM_Q80 / PRM_Q80_PROB = the PRM score (logit risk / 1 - reward) with the same Q80 flag rule as every fused arm. It splits
"score is better" from "decision rule is different".
Same inputs, orientation, Q80 rule and paired source-group bootstrap as ../prm_boosting_bank_sweep_v1/lsml_bank_sweep.py.

    python3 ablation5.py <tag> <FEATURES.npz> <telemetry.pkl> <out_dir>
"""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np

NESTED = {'N11': ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
                  'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0']}
NESTED['N13'] = NESTED['N11'] + ['realized_z', 'realized_drv']
NESTED['N16'] = NESTED['N13'] + ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
S3 = ['energy_level', 'hist_entropy_cusum_abs_series', 'realized_drv']
BANKS = {'N13': NESTED['N13'], 'N16': NESTED['N16'], 'S5': S3 + ['ct7_H0lim_prefix_innovation', 'digit_spread']}


def main():
    M = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1/eval_mirror_prm_gen_v1')
    spec = importlib.util.spec_from_file_location('ev', M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
    EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
    sys.path.insert(0, str(M))
    from spectral_utils.fusion_utils import lsml_continuous

    tag, feat, tele, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    SE = M / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
    out.mkdir(parents=True, exist_ok=True)
    with open(SE / 'OOF_ANSWERS.csv', newline='', encoding='utf-8-sig') as f: A = list(csv.DictReader(f))
    F = dict(np.load(feat)); off_all = F['offsets'].astype(np.int64)
    ans = np.flatnonzero(F['cells'].astype(str) == EV.CELL)
    ids = F['row_ids'][ans].astype(str); o = np.r_[0, np.cumsum(np.diff(off_all)[ans])]
    T = EV.metadata_from(tele)
    meta = [dict(T[i], idx=i) for i in ids]
    for r in meta:
        if 'native_labels' not in r: r['native_labels'] = [int(v > .5) for v in r['rewards']]
    prm = np.concatenate([EV.risk(r['rewards']) for r in meta])
    native = np.concatenate([np.asarray(r['native_labels'], int) == 0 for r in meta])
    with np.load(SE / 'OOF_STEP_SCORES.npz') as z:
        lab_all = z['labels'].astype(bool); assert np.array_equal(z['offsets'], off_all)
    labels = lab_all[np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in ans])]
    err = np.zeros(o[-1], bool)
    for j, m in enumerate(meta):
        for e in m['error_steps']:
            if 1 <= e <= o[j + 1] - o[j]: err[o[j] + e - 1] = True

    fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
    FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
    ext.BN = dict(ext.BN, TRANSFER=list(S3), **BANKS)
    pop = ext.Population(F, ['TRANSFER'] + list(BANKS))
    zp = EV.az(prm, o)

    def lsml(V, groups=None, anchor=None):
        s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
        s = np.asarray(s, float)
        ref = V[:, 0] if anchor is None else anchor
        if np.corrcoef(s, ref)[0, 1] < 0: s = -s
        return s, d

    prob = np.concatenate([1 - np.asarray(r['rewards'], float) for r in meta])
    # PRM_Q80 answer-z's the logit risk (the same zp the fusions use); PRM_Q80_PROB answer-z's 1 - reward.
    # The two give different Q80 flags, so both are reported (voters analysis used the prob scale).
    zq = EV.az(prob, o)
    arms, diag = {'PRM_native': prm, 'PRM_Q80': prm, 'PRM_Q80_PROB': prob}, {}
    bt = pop.cell_bank('TRANSFER', ans)
    arms['LSML_CONT'], _ = lsml(np.column_stack([zp] + [bt['V'][:, bt['names'].index(c)] for c in S3]))
    for bk in BANKS:
        V = np.asarray(pop.cell_bank(bk, ans)['V'], float)
        arms[f'TELE_{bk}'], dt = lsml(V, anchor=V.mean(1))
        arms[f'MEAN_{bk}'] = V.mean(1)
        cn = np.asarray(dt['c'], int)
        VP = np.column_stack([zp, V])
        arms[f'PRMG_{bk}'], _ = lsml(VP, groups=np.r_[cn.max() + 1, cn])
        vg = np.asarray(dt['virtual_classifiers'], float)
        VC = np.column_stack([zp] + [EV.az(vg[:, g], o) for g in range(vg.shape[1])])
        arms[f'CHAIN_{bk}'], _ = lsml(VC, groups=np.arange(VC.shape[1]))
        arms[f'HALF_{bk}'] = .5 * zp + .5 * V.mean(1)
        arms[f'EQ_{bk}'] = VP.mean(1)
        # P-variants: the same fusions with the PRM channel on the probability scale, z(1 - reward), instead of z(logit(1 - reward))
        arms[f'HALFP_{bk}'] = .5 * zq + .5 * V.mean(1)
        arms[f'EQP_{bk}'] = np.column_stack([zq, V]).mean(1)
        VCq = np.column_stack([zq] + list(VC[:, 1:].T))
        arms[f'CHAINP_{bk}'], _ = lsml(VCq, groups=np.arange(VCq.shape[1]))
        # corr of each arm with z(PRM): how much of the PRM ranking each fused score keeps
        diag[bk] = {'n_channels': V.shape[1], 'tele_K': int(dt['K']),
                    'zero_var_channels': [str(n) for n, sd in zip(pop.cell_bank(bk, ans)['names'], V.std(0)) if sd < 1e-9],
                    'corr_with_zprm': {a: float(np.corrcoef(arms[f'{a}_{bk}'], zp)[0, 1])
                                       for a in ('TELE', 'MEAN', 'PRMG', 'CHAIN', 'HALF', 'EQ')},
                    'corr_TELE_MEAN': float(np.corrcoef(arms[f'TELE_{bk}'], arms[f'MEAN_{bk}'])[0, 1])}

    flags = {n: native if n == 'PRM_native' else EV.flags_for(s, o) for n, s in arms.items()}
    rows = []
    for n, f in flags.items():
        r, _ = EV.measure(arms[n], f, labels, o, ids, meta)
        rows.append({'model': tag, 'arm': n, 'rule': 'native' if n == 'PRM_native' else 'Q80',
                     'flag_rate': float(f.mean()), **r})

    groups = np.array([A[i]['source_group'] for i in ans])
    unique, inv = np.unique(groups, return_inverse=True)
    rb = np.random.default_rng(EV.SEED)
    W = np.vstack([np.ones(len(unique)), rb.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=EV.DRAWS)])
    B = {}
    for n, f in flags.items():
        _, au = EV.measure(arms[n], f, labels, o, ids, meta)
        c4 = np.zeros((len(ids), 4))
        for j, m in enumerate(meta):
            if m['classification'] == 'correct': continue
            y, ff = err[o[j]:o[j + 1]], f[o[j]:o[j + 1]]
            c4[j] = [(~y & ~ff).sum(), (y & ~ff).sum(), (y & ff).sum(), (~y & ff).sum()]
        G = np.zeros((len(unique), 6)); np.add.at(G, inv, np.column_stack([np.where(np.isfinite(au), au, 0), np.isfinite(au), c4]))
        t = W @ G; tp, fp, tn, fn = t[:, 2:].T
        B[n] = (t[:, 0] / t[:, 1], 50 * (tp / (tp + .5 * (fp + fn)) + tn / (tn + .5 * (fp + fn))))

    pairs = [('PRM_Q80', 'PRM_native'), ('PRM_Q80_PROB', 'PRM_native')]
    for bk in BANKS:
        for a in ('TELE', 'MEAN', 'PRMG', 'CHAIN', 'HALF', 'EQ'):
            pairs += [(f'{a}_{bk}', 'PRM_native'), (f'{a}_{bk}', 'PRM_Q80'), (f'{a}_{bk}', 'PRM_Q80_PROB')]
        pairs += [(f'TELE_{bk}', f'MEAN_{bk}'), (f'CHAIN_{bk}', f'HALF_{bk}'), (f'CHAIN_{bk}', f'EQ_{bk}'),
                  (f'PRMG_{bk}', f'HALF_{bk}'), (f'CHAIN_{bk}', f'TELE_{bk}'), (f'HALF_{bk}', f'MEAN_{bk}'),
                  (f'CHAIN_{bk}', 'LSML_CONT')]
        for a in ('HALF', 'EQ', 'CHAIN'):
            pairs += [(f'{a}P_{bk}', f'{a}_{bk}'), (f'{a}P_{bk}', 'PRM_Q80_PROB'), (f'{a}P_{bk}', 'PRM_native')]
        pairs += [(f'EQ_{bk}', f'MEAN_{bk}'), (f'EQP_{bk}', f'MEAN_{bk}')]
    con = []
    for a, b in pairs:
        for j, mt in enumerate(('within_auc', 'prmscore')):
            dd = B[a][j] - B[b][j]
            con.append({'model': tag, 'contrast': f'{a} - {b}', 'metric': mt, 'delta': dd[0],
                        'ci_low': np.quantile(dd[1:], .025), 'ci_high': np.quantile(dd[1:], .975)})
    for path, rr in ((out / f'ABL5_RESULTS_{tag}.csv', rows), (out / f'ABL5_BOOTSTRAP_{tag}.csv', con)):
        with open(path, 'w', newline='') as f:
            wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rr for k in r))); wr.writeheader(); wr.writerows(rr)
    (out / f'ABL5_DIAG_{tag}.json').write_text(json.dumps(diag, indent=1))
    for r in rows: print(tag, r['arm'], r['rule'], round(r['within_auc'], 4), round(r['prmscore'], 2), flush=True)


if __name__ == '__main__':
    main()
