"""AUTO L-SML on answer-z channels plus their raw (globally rank-gaussed) values (Omri, 2026-11-03).

Same inputs, L-SML code, orientation and Q80 rule as results/prm_boosting_bank_sweep_v1/lsml_bank_sweep.py.
Arms (PRMBench, one PRM per call):
  Z17         AUTO_N16 of the sweep: L-SML, automatic grouping, over [z(PRM risk), 16 answer-z N16 channels].
              Parity gate: must reproduce the stored AUTO_N16 Q80 within-AUC and PRMScore exactly.
  CAT34       one L-SML, automatic grouping, over Z17's 17 views + the same 17 channels RAW, each raw
              channel mapped through a GLOBAL rank -> normal-score transform over all PRMBench steps
              (digit channels: over their active steps only, 0 elsewhere, as in the answer-z masked_std).
              The transform is monotone increasing, so every raw view keeps the orientation of its z view.
  PRM_native  the PRM's own step labels (reference).
  PRM_Q80_PROB  answer-z(1 - reward) under the same Q80 rule (reference, as in prm_boost_ablation5_v1).
Nothing is tuned on labels; labels enter metrics, bootstrap and the class-conditional diagnostic only.

    python3 auto_raw_plus_z.py <tag> <FEATURES.npz> <telemetry.pkl> <stored SWEEP_RESULTS csv> <out_dir>
"""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

N16 = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
       'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0',
       'realized_z', 'realized_drv', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
DRAWS, SEED = 1000, 20261103
M = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1/eval_mirror_prm_gen_v1')


def rank_gauss(x, mask=None):
    """Global rank -> standard normal scores, ties averaged; outside mask -> 0."""
    out = np.zeros(len(x))
    m = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    r = rankdata(x[m]); out[m] = ndtri((r - .5) / m.sum())
    return out


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def lead_share(V):
    ev = np.linalg.eigvalsh(np.corrcoef(V, rowvar=False))
    return float(ev[-1] / ev.sum())


def build_inputs(feat, tele):
    """Load one PRM's inputs exactly as the run does; returns (EV, dict of arrays and metadata)."""
    spec = importlib.util.spec_from_file_location('ev', M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
    EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
    sys.path.insert(0, str(M))
    SE = M / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
    with open(SE / 'OOF_ANSWERS.csv', newline='', encoding='utf-8-sig') as f: A = list(csv.DictReader(f))
    F = dict(np.load(feat)); off_all = F['offsets'].astype(np.int64)
    ans = np.flatnonzero(F['cells'].astype(str) == EV.CELL)
    ids = F['row_ids'][ans].astype(str); o = np.r_[0, np.cumsum(np.diff(off_all)[ans])]
    T = EV.metadata_from(tele)
    meta = [dict(T[i], idx=i) for i in ids]
    for r in meta:
        if 'native_labels' not in r: r['native_labels'] = [int(v > .5) for v in r['rewards']]
    rewards = np.concatenate([np.asarray(r['rewards'], float) for r in meta])
    prm = np.concatenate([EV.risk(r['rewards']) for r in meta])
    native = np.concatenate([np.asarray(r['native_labels'], int) == 0 for r in meta])
    with np.load(SE / 'OOF_STEP_SCORES.npz') as z:
        lab_all = z['labels'].astype(bool); assert np.array_equal(z['offsets'], off_all)
    rows_ = np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in ans])
    fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
    FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
    ext.BN = dict(ext.BN, N16=list(N16))
    cb = ext.Population(F, ['N16']).cell_bank('N16', ans)
    assert np.array_equal(cb['rows'], rows_)
    raw = np.asarray(cb['raw'], float); act = F['digit_active'].astype(bool)[rows_]
    zp = EV.az(prm, o)
    R = np.column_stack([rank_gauss(raw[:, j], act[:, DIG.index(c)] if c in DIG else None) for j, c in enumerate(N16)])
    V17 = np.column_stack([zp, np.asarray(cb['V'], float)])
    V34 = np.column_stack([V17, rank_gauss(prm), R])
    return EV, {'A': A, 'ans': ans, 'ids': ids, 'o': o, 'meta': meta, 'rewards': rewards, 'prm': prm,
                'native': native, 'labels': lab_all[rows_], 'raw': raw, 'act': act, 'V17': V17, 'V34': V34}


def main():
    sys.path.insert(0, str(M))
    from spectral_utils.fusion_utils import lsml_continuous
    tag, feat, tele, stored, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    out.mkdir(parents=True, exist_ok=True)
    EV, I = build_inputs(feat, tele)
    A, ans, ids, o, meta, prm, native, labels = (I[k] for k in ('A', 'ans', 'ids', 'o', 'meta', 'prm', 'native', 'labels'))
    prob = 1 - I['rewards']; V17, V34 = I['V17'], I['V34']
    names17 = ['PRM'] + N16
    names34 = ['z:' + c for c in names17] + ['raw:' + c for c in names17]

    def lsml(V):
        s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])])
        s = np.asarray(s, float)
        if np.corrcoef(s, V[:, 0])[0, 1] < 0: s = -s   # same anchor as the sweep's AUTO arm: z(PRM)
        return s, d

    arms, fits = {'PRM_native': prm, 'PRM_Q80_PROB': prob}, {}
    arms['Z17'], fits['Z17'] = lsml(V17)
    arms['CAT34'], fits['CAT34'] = lsml(V34)
    flags = {n: (native if n == 'PRM_native' else EV.flags_for(s, o)) for n, s in arms.items()}

    err_any = np.array([labels[a:b].any() for a, b in zip(o[:-1], o[1:])])
    res, au_per = [], {}
    for n, s in arms.items():
        r, au = EV.measure(s, flags[n], labels, o, ids, meta); au_per[n] = au
        mx = np.array([s[a:b].max() for a, b in zip(o[:-1], o[1:])])
        res.append({'model': tag, 'arm': n, 'rule': 'native' if n == 'PRM_native' else 'Q80',
                    'flag_rate': float(flags[n].mean()), 'pooled_auc': float(auc(labels, s)),
                    'answer_auc_max': float(auc(err_any, mx)), **r})

    # parity: Z17 must equal the stored AUTO_N16 Q80 row exactly
    with open(stored, newline='') as f:
        st = [r for r in csv.DictReader(f) if r['arm'] == 'AUTO_N16' and r['rule'] == 'Q80'][0]
    z17 = [r for r in res if r['arm'] == 'Z17'][0]
    par = {'stored_within_auc': float(st['within_auc']), 'stored_prmscore': float(st['prmscore']),
           'z17_within_auc': z17['within_auc'], 'z17_prmscore': z17['prmscore']}
    par['pass'] = bool(abs(par['stored_within_auc'] - par['z17_within_auc']) < 1e-9 and
                       abs(par['stored_prmscore'] - par['z17_prmscore']) < 1e-9)
    print(tag, 'PARITY', json.dumps(par), flush=True)
    if not par['pass']:
        (out / f'PARITY_FAIL_{tag}.json').write_text(json.dumps(par, indent=1)); raise SystemExit('HARD STOP: parity')

    # bootstrap of differences vs Z17: (a) answer-level resampling, (b) source-group resampling
    err = np.zeros(o[-1], bool)
    for j, m in enumerate(meta):
        for e in m['error_steps']:
            if 1 <= e <= o[j + 1] - o[j]: err[o[j] + e - 1] = True
    groups = np.array([A[i]['source_group'] for i in ans])
    stats = {}
    for n in arms:
        c4 = np.zeros((len(ids), 4))
        for j, m in enumerate(meta):
            if m['classification'] == 'correct': continue
            y, ff = err[o[j]:o[j + 1]], flags[n][o[j]:o[j + 1]]
            c4[j] = [(~y & ~ff).sum(), (y & ~ff).sum(), (y & ff).sum(), (~y & ff).sum()]
        au = au_per[n]
        stats[n] = np.column_stack([np.where(np.isfinite(au), au, 0), np.isfinite(au), c4])
    boot = []
    for unit, key in (('answer', np.arange(len(ids))), ('source_group', groups)):
        uq, inv = np.unique(key, return_inverse=True)
        rng = np.random.default_rng(SEED)
        W = np.vstack([np.ones(len(uq)), rng.multinomial(len(uq), np.full(len(uq), 1 / len(uq)), size=DRAWS)])
        B = {}
        for n in arms:
            G = np.zeros((len(uq), 6)); np.add.at(G, inv, stats[n]); t = W @ G; tp, fp, tn, fn = t[:, 2:].T
            B[n] = (t[:, 0] / t[:, 1], 50 * (tp / (tp + .5 * (fp + fn)) + tn / (tn + .5 * (fp + fn))))
        assert abs(B['Z17'][1][0] - z17['prmscore']) < 1e-6
        for n in ('CAT34', 'PRM_Q80_PROB', 'PRM_native'):
            for j, mt in enumerate(('within_auc', 'prmscore')):
                d = B[n][j] - B['Z17'][j]
                boot.append({'model': tag, 'unit': unit, 'n_units': len(uq), 'draws': DRAWS, 'contrast': f'{n} - Z17',
                             'metric': mt, 'delta': d[0], 'ci_low': np.quantile(d[1:], .025),
                             'ci_high': np.quantile(d[1:], .975)})

    # diagnostics: class-conditional |corr(raw_i, z_i)|, CAT34 grouping, leading-eigenvalue share
    c34 = np.asarray(fits['CAT34']['c'], int); c17 = np.asarray(fits['Z17']['c'], int)
    diag = []
    for i, c in enumerate(names17):
        zi, ri = V34[:, i], V34[:, 17 + i]
        row = {'model': tag, 'channel': c, 'Z17_group': int(c17[i]), 'CAT34_group_z': int(c34[i]),
               'CAT34_group_raw': int(c34[17 + i]), 'paired_in_CAT34': bool(c34[i] == c34[17 + i])}
        for lab, m in (('clean', ~labels), ('error', labels), ('all', np.ones(len(labels), bool))):
            row[f'abs_corr_raw_z_{lab}'] = float(abs(np.corrcoef(zi[m], ri[m])[0, 1]))
        diag.append(row)
    fitd = {}
    for n, V, nm in (('Z17', V17, ['z:' + c for c in names17]), ('CAT34', V34, names34)):
        d, c = fits[n], np.asarray(fits[n]['c'], int)
        fitd[n] = {'K': int(d['K']), 'groups': {int(g): [nm[j] for j in np.flatnonzero(c == g)] for g in np.unique(c)},
                   'lead_eig_share_all_inputs': lead_share(V),
                   'lead_eig_share_per_group': {int(g): (lead_share(V[:, c == g]) if (c == g).sum() > 1 else 1.0) for g in np.unique(c)},
                   'degenerate': bool(d.get('degenerate', False)),
                   'n_pairs_same_group': int(sum(r['paired_in_CAT34'] for r in diag)) if n == 'CAT34' else None}
    fitd['parity'] = par

    for path, rr in ((out / f'RESULTS_{tag}.csv', res), (out / f'BOOTSTRAP_{tag}.csv', boot), (out / f'DIAG_{tag}.csv', diag)):
        with open(path, 'w', newline='') as f:
            wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rr for k in r))); wr.writeheader(); wr.writerows(rr)
    (out / f'FIT_{tag}.json').write_text(json.dumps(fitd, indent=1))
    np.savez_compressed(out / f'SCORES_{tag}.npz', row_ids=ids, offsets=o, **{f'score__{n}': s for n, s in arms.items()})
    for r in res: print(tag, r['arm'], round(r['within_auc'], 4), round(r['pooled_auc'], 4), round(r['answer_auc_max'], 4), round(r['prmscore'], 2), flush=True)


if __name__ == '__main__':
    main()
