"""Post-hoc reasoning checks for the view-selection candidates (2026-10-26).

Re-reads existing stage-0/1/2 outputs; no new search and no new fusion method. Three kinds of computation:
- lookups in stage1/SEARCH.npz and stage2/SEARCH.npz;
- label-free statistics on the stage-1 light caches (vote covariance, L-SML groups, signs);
- replays of recorded fits of named subsets through stage1.py / stage2.py evaluate(), with parity against SEARCH.npz.
Labels enter only through the stage-0 VAL_* columns and the recorded PRMScores, as in stages 0-2.
All statistics use the 8 cells without Math-Shepherd (Omri's decision, 2026-10-26).

    python checks.py            # writes the CSV/JSON files next to this script (about 3 min, CPU)
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
VS = HERE.parent
ROOT = VS.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(VS / 'stage1')); sys.path.insert(0, str(VS / 'stage2'))
import csv  # noqa: E402
from spectral_utils import prm_anchored_lsml as A  # noqa: E402
import stage1 as S1, stage2 as S2  # noqa: E402

N16 = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
       'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
       'digit_spread', 'digit_alternative_innovation']
# Families fixed from the channel definitions (spectral_utils/claude_feature_bank_v1.py, external_banks_v4.py,
# digit_feature_family.py), not from data.
FAMILY = {'uncertainty': ['q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level'],
          'realized': ['chosen_surprisal', 'realized_z', 'realized_drv'],
          'dynamics': ['energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0'],
          'digit': ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']}
FAM_OF = {n: f for f, v in FAMILY.items() for n in v}
FAM_IDX = np.array([list(FAMILY).index(FAM_OF[n]) for n in N16])
STORED_SIGN = {'logprob_margin': -1, 'energy_level': -1, 'dominant_freq16': -1}   # bank11 FEATURE_RISK_SIGNS; others +1
ALL10 = [f'{b}/{p}' for b, p in S1.CELLS]
# Math-Shepherd dropped by Omri's decision (2026-10-26): its PRM is position-driven; 8 cells remain
C8 = [i for i, c in enumerate(ALL10) if 'math_shepherd' not in c]
CELLS = [ALL10[i] for i in C8]
NAMED = {
    'SUB5': ['q15_H1', 'chosen_surprisal', 'top15_turnover', 'dominant_freq16', 'bocpd_p0', 'digit_spread'],
    'S1_9of10': ['q15_VE1', 'chosen_surprisal', 'top15_turnover', 'dominant_freq16', 'bocpd_p0'],
    'S2_best': ['q15_VE1', 'chosen_surprisal', 'logprob_margin', 'energy_innovation', 'top50_js'],
    'SUB4': ['q15_VE1', 'chosen_surprisal', 'energy_innovation', 'top15_turnover', 'realized_z', 'digit_alternative',
             'digit_alternative_innovation'],
    'SUB5_minus_dom': ['q15_H1', 'chosen_surprisal', 'top15_turnover', 'bocpd_p0', 'digit_spread'],
    'SUB5_minus_t15': ['q15_H1', 'chosen_surprisal', 'dominant_freq16', 'bocpd_p0', 'digit_spread'],
    'SUB5_minus_t15_dom': ['q15_H1', 'chosen_surprisal', 'bocpd_p0', 'digit_spread'],
    'SUB5_cs_to_rz': ['q15_H1', 'realized_z', 'top15_turnover', 'dominant_freq16', 'bocpd_p0', 'digit_spread'],
    'RULE_pooled': ['true_tail50', 'chosen_surprisal', 'energy_innovation', 'digit_spread'],
}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def mask_of(names):
    return sum(1 << N16.index(n) for n in names)


def rand_index(a, b):
    iu = np.triu_indices(len(a), 1)
    sa, sb = (a[:, None] == a[None, :])[iu], (b[:, None] == b[None, :])[iu]
    return float((sa == sb).mean())


def main():
    out = {}
    caches = {c: np.load(VS / 'stage1' / 'cache' / f'{c.replace("/", "__")}.npz') for c in CELLS}

    # 1. per-channel table: stage-1 signs, PRM-vote correlation, stored-orientation within-AUC (stage 0), stage-0 loadings
    corr = {c: (lambda R: (R / np.sqrt(np.outer(np.diag(R), np.diag(R))))[0, 1:])(caches[c]['R']) for c in CELLS}
    sign1 = {c: caches[c]['sign'][1:] for c in CELLS}
    s0 = [r for r in read_csv(VS / 'stage0' / 'BINARY_VIEWS.csv') if r['view'] != 'PRM' and r['prm'] != 'math_shepherd']
    rows = []
    for k, n in enumerate(N16):
        row = dict(channel=n, family=FAM_OF[n], stored_risk_sign=STORED_SIGN.get(n, 1),
                   stage1_plus_signs_of_8=int(sum(sign1[c][k] > 0 for c in CELLS)))
        for b in ('prmbench', 'socratic'):
            cc = [c for c in CELLS if c.startswith(b)]
            x = np.array([corr[c][k] for c in cc])
            row[f'{b}_prm_vote_corr_median'] = round(float(np.median(x)), 3)
            v = [r for r in s0 if r['benchmark'] == b and r['view'] == n]
            auc = np.array([float(r['VAL_within_auc_cont']) for r in v]); sg = np.array([int(r['sign']) for r in v])
            stored = np.where(sg > 0, auc, 1 - auc)          # within-AUC of the channel in its stored orientation
            row[f'{b}_stored_auc_min'] = round(float(stored.min()), 3)
            row[f'{b}_stored_auc_median'] = round(float(np.median(stored)), 3)
            row[f'{b}_stored_auc_max'] = round(float(stored.max()), 3)
            row[f'{b}_stored_auc_below_half'] = int((stored < .5).sum())
            row[f'{b}_fa_u_median'] = round(float(np.median([float(r['fa_u']) for r in v])), 3)
            row[f'{b}_fa_lambda_median'] = round(float(np.median([float(r['fa_lambda']) for r in v])), 3)
        rows.append(row)
    write_csv(HERE / 'CHANNEL_TABLE.csv', rows)

    # 2. code-defined families vs the label-free L-SML grouping of all 16 channels (stage-1 vote covariance)
    rng = np.random.default_rng(0)
    fam_rows, pick_rows = [], []
    for c in CELLS:
        R = caches[c]['R']; K, g = A.groups_from_cov(R[1:, 1:])
        ri = rand_index(FAM_IDX, g)
        null = np.array([rand_index(rng.permutation(FAM_IDX), g) for _ in range(2000)])
        C = R[1:, 1:] / np.sqrt(np.outer(np.diag(R[1:, 1:]), np.diag(R[1:, 1:]))); iu = np.triu_indices(16, 1)
        same = (FAM_IDX[:, None] == FAM_IDX[None, :])[iu]
        fam_rows.append(dict(cell=c, K=K, rand=round(ri, 3), perm_null_mean=round(float(null.mean()), 3),
                             perm_p=float((null >= ri).mean()), vote_corr_within_family=round(float(C[iu][same].mean()), 3),
                             vote_corr_between_family=round(float(C[iu][~same].mean()), 3),
                             groups=' | '.join('+'.join(N16[i] for i in range(16) if g[i] == j) for j in range(K))))
        # 3. the rule, per cell: in each L-SML group keep the channel with the largest PRM-vote correlation
        picks = [N16[max(np.where(g == j)[0], key=lambda i: corr[c][i])] for j in range(K)]
        # strict form of the argument: covariance r_0k = v_0 v_k (Jaffe 2016 Lemma 1), not correlation
        picks_cov = [N16[max(np.where(g == j)[0], key=lambda i: R[0, 1 + i])] for j in range(K)]
        pick_rows.append(dict(cell=c, K=K, picks='+'.join(picks), picks_cov='+'.join(picks_cov)))
    write_csv(HERE / 'FAMILY_VS_LSML.csv', fam_rows); write_csv(HERE / 'RULE_PICKS_PER_CELL.csv', pick_rows)
    for key, col in (('rule_pick_frequency_of_8', 'picks'), ('rule_pick_frequency_of_8_cov', 'picks_cov')):
        allp = [p for r in pick_rows for p in r[col].split('+')]
        out[key] = {n: allp.count(n) for n in N16 if allp.count(n)}
    # pooled rule with code-defined families: median over the 8 cells of the PRM-vote correlation
    med = {n: float(np.median([corr[c][N16.index(n)] for c in CELLS])) for n in N16}
    out['rule_pooled_family_picks'] = {f: max(v, key=lambda n: med[n]) for f, v in FAMILY.items()}
    out['rule_pooled_median_corr'] = {n: round(v, 3) for n, v in med.items()}

    # 4. lookups of named subsets in both searches
    P = json.loads((VS / 'stage1' / 'PARITY.json').read_text()); prm = np.array([P[c]['prm_exact'] for c in CELLS])
    D1, D2 = np.load(VS / 'stage1' / 'SEARCH.npz'), np.load(VS / 'stage2' / 'SEARCH.npz')
    pos1 = {int(m): i for i, m in enumerate(D1['masks'])}; pos2 = {int(m): i for i, m in enumerate(D2['masks'])}
    look = []
    for name, chans in NAMED.items():
        i, j = pos1[mask_of(chans)], pos2[mask_of(chans)]
        for ci, c in enumerate(CELLS):
            col = ALL10.index(c)
            look.append(dict(subset=name, cell=c, s1_valid=bool(D1['valid'][i, col]), s1_K=int(D1['K'][i, col]),
                             s1_fused_minus_prm=round(float(D1['ps_fused'][i, col] - prm[ci]), 3),
                             s1_fused_minus_mean=round(float(D1['ps_fused'][i, col] - D1['ps_mean'][i, col]), 3),
                             s1_prm_share=round(float(D1['w_prm_share'][i, col]), 3),
                             s2_valid=bool(D2['valid'][j, col]),
                             s2_fused_minus_mean=round(float(D2['ps_fused'][j, col] - D2['ps_mean'][j, col]), 3),
                             s2_fused_minus_prm=round(float(D2['ps_fused'][j, col] - prm[ci]), 3)))
    write_csv(HERE / 'NAMED_LOOKUP.csv', look)

    # 5. stage 2: is "beats its own mean" driven by the stored-orientation channels below chance?
    m = D2['masks']; v2 = D2['valid'][:, C8].all(1); f2, mu2 = D2['ps_fused'][:, C8], D2['ps_mean'][:, C8]
    succ = v2 & (f2 > mu2).all(1)
    has = lambda n: ((m >> N16.index(n)) & 1) == 1
    ei, js = has('energy_innovation'), has('top50_js')
    out['stage2'] = dict(
        valid=int(v2.sum()), successes=int(succ.sum()),
        share_with_EI_or_JS_valid=round(float((ei | js)[v2].mean()), 3), share_with_EI_or_JS_successes=round(float((ei | js)[succ].mean()), 3),
        success_rate_with_EI=round(float(succ[v2 & ei].mean()), 4), success_rate_without_EI=round(float(succ[v2 & ~ei].mean()), 4),
        median_mean_by_cell_with_EI=np.round(np.median(mu2[v2 & ei], 0), 2).tolist(),
        median_mean_by_cell_without_EI=np.round(np.median(mu2[v2 & ~ei], 0), 2).tolist(),
        median_fused_by_cell_with_EI=np.round(np.median(f2[v2 & ei], 0), 2).tolist(),
        median_fused_by_cell_without_EI=np.round(np.median(f2[v2 & ~ei], 0), 2).tolist())

    # 6. stage 1: does same-family redundancy predict the PRM's weight or the margins? (valid subsets)
    v1 = D1['valid'][:, C8].all(1); f1, mu1, sh = D1['ps_fused'][:, C8], D1['ps_mean'][:, C8], D1['w_prm_share'][:, C8]
    bits = np.array([[(int(x) >> i) & 1 for i in range(16)] for x in D1['masks']])
    nA = bits[:, [N16.index(n) for n in FAMILY['uncertainty']]].sum(1)
    nfam = sum((bits[:, [N16.index(n) for n in v]].sum(1) > 0).astype(int) for v in FAMILY.values())
    marg = np.minimum(f1 - prm, f1 - mu1)                    # 8-cell order: 0-3 PRMBench, 4-7 Socratic
    out['stage1_redundancy'] = {
        lab: dict(rho_socratic_median_prm_share=round(float(spearmanr(x[v1], np.median(sh[v1][:, 4:], 1))[0]), 3),
                  rho_socratic_min_margin=round(float(spearmanr(x[v1], marg[v1][:, 4:].min(1))[0]), 3),
                  rho_prmbench_min_margin=round(float(spearmanr(x[v1], marg[v1][:, :4].min(1))[0]), 3))
        for lab, x in (('n_uncertainty_family_channels', nA), ('n_families_covered', nfam))}

    # 7. replays with detail: groups and between-group weight shares (parity vs SEARCH.npz)
    rep = []
    for stage, name in ((1, 'SUB5'), (1, 'SUB5_minus_dom'), (1, 'SUB5_minus_t15'), (2, 'S2_best')):
        S = [N16.index(n) for n in NAMED[name]]; D = D1 if stage == 1 else D2
        i = list(D['masks']).index(mask_of(NAMED[name])); off = 1 if stage == 1 else 0
        for b, p in S1.CELLS:
            if p == 'math_shepherd': continue
            ci = ALL10.index(f'{b}/{p}')
            cell = (S1.CellData.from_light(VS / 'stage1' / 'cache' / f'{b}__{p}.npz') if stage == 1
                    else S2.TeleCell.load(VS / 'stage2' / 'cache' / f'{b}__{p}.npz'))
            o = cell.evaluate(S, want_detail=True)
            if not o['valid']:
                rep.append(dict(stage=stage, subset=name, cell=ALL10[ci], valid=False, K=o['K'])); continue
            cols = np.asarray(S) + off
            groups = ['+'.join(N16[k - off] for k in cols[o['groups'] == g]) for g in range(o['K'])]
            wb = o['between']['w']; share = wb / wb.sum()
            labels = (['PRM'] if stage == 1 else []) + groups
            rep.append(dict(stage=stage, subset=name, cell=ALL10[ci], valid=True, K=o['K'],
                            parity_abs_diff=float(abs(o['ps_fused'] - D['ps_fused'][i, ci])),
                            shares=' | '.join(f'{l}:{s:.3f}' for l, s in zip(labels, share))))
    write_csv(HERE / 'REPLAY_WEIGHTS.csv', rep)
    # cells after an invalid one have no search value (pre-registered early exit): nan, not compared
    par = np.array([r['parity_abs_diff'] for r in rep if 'parity_abs_diff' in r])
    out['replay_parity'] = dict(compared=int(np.isfinite(par).sum()), not_in_search=int((~np.isfinite(par)).sum()),
                                max_abs_diff=float(np.nanmax(par)))
    (HERE / 'CHECKS.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
