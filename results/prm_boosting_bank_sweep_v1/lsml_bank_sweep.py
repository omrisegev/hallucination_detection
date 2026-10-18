"""L-SML of the PRM output with each telemetry bank of at most 16 channels, plus a per-answer
oracle-count ceiling (Omri, 2026-10-18).

No fixed-weight averaging with the PRM. For each bank B and each PRM, the PRM risk view and
the B channels enter ONE continuous L-SML (spectral_utils.fusion_utils.lsml_continuous):
  PRMG_<B>   the PRM is forced into its own group; the B groups are detected label-free on B alone.
             This is the hierarchical form: SML inside each telemetry group, then SML across the
             group scores and the PRM.
  AUTO_<B>   automatic grouping over PRM + B (the PRM may be absorbed into a telemetry group).
  TELE_<B>   L-SML on B alone (no PRM), for reference.
Plus PRM_native and LSML_CONT (the 3 label-chosen channels; replays results/prm_boosting_controls_v1).

Banks: the nested family N11, N13, N14, N16 (results/monday_nested_banks_v2/BANKS.json), every
ext.BN bank with <= 16 channels if present (B13, B16), and the label-selected S5, S6
(docs/reviews/integration_completion_20261017/replay_verification/FROZEN_BANKS.json), reported as
label-selected. S3 is LSML_CONT. N11 and N13 have no digit channels; N14 and N16 do.

Decision rules, each reported for every arm:
  Q80      ANSWER_Z_Q80, the deployed label-free rule.
  ORACLE   per answer, flag the top-k steps of the arm's score, where k is the TRUE number of
           error steps in that answer (0 for correct answers). Label-using ceiling: it measures
           how much PRMScore a perfect count would add on top of the arm's ordering.
Diagnostics: L-SML groups, the PRM's least-squares weight share in the fused score (as in
eval_lsml4/lsml_four.py), and whether the PRM sits alone in its group.
Labels are used only for scoring and for the ORACLE count.

    python3 lsml_bank_sweep.py <tag> <FEATURES.npz> <telemetry.pkl> <out_dir>
"""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np

NESTED = {
    'N11': ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
            'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0'],
}
NESTED['N13'] = NESTED['N11'] + ['realized_z', 'realized_drv']
NESTED['N14'] = NESTED['N11'] + ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
NESTED['N16'] = NESTED['N13'] + ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
S3 = ['energy_level', 'hist_entropy_cusum_abs_series', 'realized_drv']
LABEL_SELECTED = {'S5': S3 + ['ct7_H0lim_prefix_innovation', 'digit_spread'],
                  'S6': S3 + ['ct7_H0lim_prefix_innovation', 'digit_spread', 'hist_topk_varentropy_series']}
MAX_BANK = 16


def oracle_flags(score, err, o):
    """Per answer, flag the top-k steps of score where k = true number of error steps."""
    f = np.zeros(o[-1], bool)
    for j in range(len(o) - 1):
        a, b = o[j], o[j + 1]; k = int(err[a:b].sum())
        if k: f[a + np.argsort(-score[a:b], kind='stable')[:k]] = True
    return f


def prm_share(fused, V):
    """Least-squares weight share of view 0 (the PRM) in the fused score, as in lsml_four.py."""
    w = np.linalg.lstsq(np.column_stack([V, np.ones(len(V))]), fused, rcond=None)[0][:V.shape[1]]
    return float(abs(w[0]) / np.abs(w).sum())


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
    names_avail = set(str(x) for x in F['names'])
    extra = {k: list(v) for k, v in ext.BN.items() if k.startswith('B') and len(v) <= MAX_BANK}
    banks, kind, skipped = {}, {}, {}
    for src, label in ((NESTED, 'nested'), (extra, 'estimator_bank'), (LABEL_SELECTED, 'label_selected')):
        for k, v in src.items():
            miss = [c for c in v if (c[4:] if c.startswith('lf__') else c) not in names_avail]
            if miss: skipped[k] = miss; continue
            banks[k], kind[k] = v, label
    print(tag, 'banks', {k: len(v) for k, v in banks.items()}, 'skipped', skipped, flush=True)
    ext.BN = dict(ext.BN, TRANSFER=list(S3), **banks)
    pop = ext.Population(F, ['TRANSFER'] + list(banks))
    zp = EV.az(prm, o)

    def lsml(V, groups=None, anchor=None):
        s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
        s = np.asarray(s, float)
        ref = V[:, 0] if anchor is None else anchor
        if np.corrcoef(s, ref)[0, 1] < 0: s = -s
        return s, d

    arms, diag, bank_of = {'PRM_native': prm}, {}, {'PRM_native': '', 'LSML_CONT': 'S3'}
    bt = pop.cell_bank('TRANSFER', ans)
    V4 = np.column_stack([zp] + [bt['V'][:, bt['names'].index(c)] for c in S3])
    arms['LSML_CONT'], d = lsml(V4)
    diag['LSML_CONT'] = {'K': int(d['K']), 'groups': np.asarray(d['c']).tolist(), 'prm_share': prm_share(arms['LSML_CONT'], V4),
                         'prm_alone': bool((np.asarray(d['c']) == d['c'][0]).sum() == 1)}
    for bk in banks:
        V = np.asarray(pop.cell_bank(bk, ans)['V'], float)
        arms[f'TELE_{bk}'], dt = lsml(V, anchor=V.mean(1))
        cn = np.asarray(dt['c'], int)
        VP = np.column_stack([zp, V])
        arms[f'PRMG_{bk}'], dg = lsml(VP, groups=np.r_[cn.max() + 1, cn])
        arms[f'AUTO_{bk}'], da = lsml(VP)
        ca = np.asarray(da['c'], int)
        diag[bk] = {'kind': kind[bk], 'n_channels': len(banks[bk]), 'channels': banks[bk],
                    'tele_groups': cn.tolist(), 'tele_K': int(dt['K']),
                    'PRMG_K': int(dg['K']), 'PRMG_prm_share': prm_share(arms[f'PRMG_{bk}'], VP),
                    'AUTO_K': int(da['K']), 'AUTO_groups': ca.tolist(), 'AUTO_prm_alone': bool((ca == ca[0]).sum() == 1),
                    'AUTO_prm_share': prm_share(arms[f'AUTO_{bk}'], VP),
                    'degenerate': {'TELE': bool(dt.get('degenerate', False)), 'PRMG': bool(dg.get('degenerate', False)),
                                   'AUTO': bool(da.get('degenerate', False))}}
        for a in ('TELE', 'PRMG', 'AUTO'): bank_of[f'{a}_{bk}'] = bk

    flags = {}
    for n, s in arms.items():
        flags[(n, 'Q80')] = native if n == 'PRM_native' else EV.flags_for(s, o)
        flags[(n, 'ORACLE')] = oracle_flags(s, err, o)

    rows = []
    for (n, rule), f in flags.items():
        r, _ = EV.measure(arms[n], f, labels, o, ids, meta)
        rows.append({'model': tag, 'arm': n, 'bank': bank_of[n], 'rule': rule, 'flag_rate': float(f.mean()), **r})

    groups = np.array([A[i]['source_group'] for i in ans])
    unique, inv = np.unique(groups, return_inverse=True)
    rb = np.random.default_rng(EV.SEED)
    W = np.vstack([np.ones(len(unique)), rb.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=EV.DRAWS)])
    B = {}
    for (n, rule), f in flags.items():
        _, au = EV.measure(arms[n], f, labels, o, ids, meta)
        c4 = np.zeros((len(ids), 4))
        for j, m in enumerate(meta):
            if m['classification'] == 'correct': continue
            y, ff = err[o[j]:o[j + 1]], f[o[j]:o[j + 1]]
            c4[j] = [(~y & ~ff).sum(), (y & ~ff).sum(), (y & ff).sum(), (~y & ff).sum()]
        G = np.zeros((len(unique), 6)); np.add.at(G, inv, np.column_stack([np.where(np.isfinite(au), au, 0), np.isfinite(au), c4]))
        t = W @ G; tp, fp, tn, fn = t[:, 2:].T
        B[(n, rule)] = (t[:, 0] / t[:, 1], 50 * (tp / (tp + .5 * (fp + fn)) + tn / (tn + .5 * (fp + fn))))

    pairs = []
    for n in arms:
        for ref in ('PRM_native', 'LSML_CONT'):
            if n != ref: pairs.append(((n, 'Q80'), (ref, 'Q80')))
        if n.startswith('PRMG_') or n.startswith('AUTO_'):
            pairs.append(((n, 'Q80'), ('TELE_' + bank_of[n], 'Q80')))
        pairs.append(((n, 'ORACLE'), (n, 'Q80')))
    con = []
    for a, b in pairs:
        for j, mt in enumerate(('within_auc', 'prmscore')):
            if mt == 'within_auc' and a[0] == b[0]: continue  # same ordering, AUC identical
            dd = B[a][j] - B[b][j]
            con.append({'model': tag, 'contrast': f'{a[0]}[{a[1]}] - {b[0]}[{b[1]}]', 'metric': mt, 'delta': dd[0],
                        'ci_low': np.quantile(dd[1:], .025), 'ci_high': np.quantile(dd[1:], .975)})
    for path, rr in ((out / f'SWEEP_RESULTS_{tag}.csv', rows), (out / f'SWEEP_BOOTSTRAP_{tag}.csv', con)):
        with open(path, 'w', newline='') as f:
            wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rr for k in r))); wr.writeheader(); wr.writerows(rr)
    (out / f'SWEEP_DIAG_{tag}.json').write_text(json.dumps({'skipped_banks': skipped, **diag}, indent=1))
    for r in rows: print(tag, r['arm'], r['rule'], round(r['within_auc'], 4), round(r['prmscore'], 2), flush=True)


if __name__ == '__main__':
    main()
