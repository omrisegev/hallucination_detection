"""Evaluation of amendment 1 (Qwen3-8B telemetry, banks N16 and N56). Called by `python run_n56.py evaluate`.

Every Qwen3-8B arm is added to the Qwen-7B PRM cell of the evaluator under the prefix `Q3:` (labels, groups and
the PRM come from that cell). Writes n56/RESULTS.csv, n56/CONTRASTS.csv, n56/DIAGNOSTICS.csv,
n56/SUBSPACE_POSTHOC.csv, n56/VERDICT.json.
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

import evaluate_collection as E
from evaluate_dufs import boot_arm, contrast, diagnostics, draws_of, subspace_content
from run import BUDGETS, sha256
from run_n56 import OUT, PRM, SC, load_inputs, orient
from spectral_utils.anchored_liu import pooled_zscore_features

BANKS = ('N16', 'N56')
TELE = ['LSML', 'EQ', 'IU', 'DUFSLIU'] + [f'{a}@{b}' for a in ('TALIU', 'RHOLIU', 'RHOPCR', 'LR') for b in BUDGETS]
FUSED = ['CHAIN', 'EQ_PRM', 'IU_PRM', 'DUFSLIU_PRM', 'PALIU_PRM', 'PRHOLIU_PRM'] + \
        [f'PRM_{a}@{b}' for a in ('TALIU', 'RHOLIU', 'RHOPCR', 'LR') for b in BUDGETS]


def q(arm, bank):
    return f'Q3:{arm}_{bank}'


def main():
    c = E.load_prmbench(PRM); W = E.boot_weights(c); cache = {}
    fits = {}
    for bank in BANKS:
        for which in ('tele', 'prm'):
            S = np.load(SC / f'SCORES_{bank}_{which}.npz')
            assert np.array_equal(S['offsets'], c.off) and np.array_equal(S['row_ids'].astype(str), c.ids)
            for k in S.files:
                if not k.startswith('score__'): continue
                v = S[k].astype(float); name = 'Q3:' + k[len('score__'):]
                if v.ndim == 1: c.arms[name] = v
                else:
                    for d in range(v.shape[0]): c.arms[f'{name}#d{d}'] = v[d]
            fits[f'{bank}_{which}'] = json.loads((SC / f'FIT_{bank}_{which}.json').read_text())
    pos_z = E.answer_z(c.extra['position'], c.off); len_z = E.answer_z(c.extra['log_step_tokens'], c.off)
    results, diags, subspace = [], [], []
    for bank in BANKS:
        I = load_inputs(bank); Vo, _ = orient(I['V'], I['lev'])
        for which, arms, X in (('tele', TELE, Vo), ('prm', FUSED, np.column_stack([I['zp'], Vo]))):
            F, _ = pooled_zscore_features(X)
            lr = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=2000).fit(F.T, c.err.astype(int))
            lr_z = E.answer_z(lr.decision_function(F.T), c.off)          # in-sample supervised direction (diagnostic)
            subspace += [dict(r, bank=bank, set=which) for r in subspace_content(PRM, F, {
                'supervised_score': lr_z, 'prm_zp': I['zp'], 'position': pos_z, 'log_length': len_z,
                'error_label': c.err.astype(float)})]
            for arm in arms:
                name = q(arm, bank); names = draws_of(c, name)
                per = {rule: [E.measure(c, n, rule, fast=True)[0] for n in names] for rule in ('GAP', 'Q80')}
                results.append(dict(bank=bank, set=which, arm=arm, n_draws=len(names),
                                    within_auc=float(np.mean([r['within_auc'] for r in per['GAP']])),
                                    GAP=float(np.mean([r['official'] for r in per['GAP']])),
                                    Q80=float(np.mean([r['official'] for r in per['Q80']])),
                                    within_auc_draw_sd=float(np.std([r['within_auc'] for r in per['GAP']])),
                                    n_auc=per['GAP'][0]['n_auc']))
                diags.append(dict(bank=bank, set=which, arm=arm, **diagnostics(c, name, pos_z, len_z, lr_z)))
    for arm in ('PRM_Q80_PROB', 'PRM_Q80'):
        r = E.measure(c, arm, 'GAP', fast=True)[0]
        results.append(dict(bank='-', set='prm', arm=arm, n_draws=1, within_auc=r['within_auc'], GAP=r['official'],
                            Q80=E.measure(c, arm, 'Q80', fast=True)[0]['official'], n_auc=r['n_auc']))
    pairs = []
    for b in BANKS:
        pairs += [(q('DUFSLIU_PRM', b), q('CHAIN', b)), (q('IU_PRM', b), q('CHAIN', b)), (q('DUFSLIU_PRM', b), q('IU_PRM', b)),
                  (q('DUFSLIU_PRM', b), q('EQ_PRM', b)), (q('CHAIN', b), 'PRM_Q80_PROB'), (q('DUFSLIU_PRM', b), 'PRM_Q80_PROB'),
                  (q('DUFSLIU', b), q('LSML', b)), (q('IU', b), q('LSML', b)), (q('DUFSLIU', b), q('IU', b)),
                  (q('DUFSLIU', b), q('EQ', b)),
                  (q('PALIU_PRM', b), q('DUFSLIU_PRM', b)), (q('PALIU_PRM', b), q('CHAIN', b)),
                  (q('PRHOLIU_PRM', b), q('DUFSLIU_PRM', b)), (q('PRHOLIU_PRM', b), q('CHAIN', b))]
        for k in BUDGETS:
            pairs += [(q(f'TALIU@{k}', b), q('DUFSLIU', b)), (q(f'TALIU@{k}', b), q(f'LR@{k}', b)),
                      (q(f'RHOLIU@{k}', b), q('DUFSLIU', b)), (q(f'RHOLIU@{k}', b), q(f'LR@{k}', b)),
                      (q(f'LR@{k}', b), q('LSML', b)),
                      (q(f'PRM_TALIU@{k}', b), q('DUFSLIU_PRM', b)), (q(f'PRM_TALIU@{k}', b), q(f'PRM_LR@{k}', b)),
                      (q(f'PRM_RHOLIU@{k}', b), q('DUFSLIU_PRM', b)), (q(f'PRM_RHOLIU@{k}', b), q(f'PRM_LR@{k}', b)),
                      (q(f'PRM_LR@{k}', b), q('CHAIN', b))]
    for arm in TELE + FUSED:                                            # D56: N56 - N16 for every arm
        pairs.append((q(arm, 'N56'), q(arm, 'N16')))
    contrasts = [r for a, b in pairs for r in contrast(c, a, b, W, cache)]
    E.write_csv(OUT / 'RESULTS.csv', results); E.write_csv(OUT / 'CONTRASTS.csv', contrasts)
    E.write_csv(OUT / 'DIAGNOSTICS.csv', diags); E.write_csv(OUT / 'SUBSPACE_POSTHOC.csv', subspace)
    get = {(r['contrast'], r['metric']): r for r in contrasts}

    def ci(a, b, m='within_auc'):
        r = get[(f'{a} - {b}', m)]; return [round(r['delta'], 4), round(r['ci_low'], 4), round(r['ci_high'], 4)]

    def replace(a, b):
        o, w = get[(f'{a} - {b}', 'official')], get[(f'{a} - {b}', 'within_auc')]
        ok = o['ci_low'] > -0.5 and w['ci_low'] > -0.005
        return dict(prmscore=ci(a, b, 'official'), within=ci(a, b),
                    verdict='replaces and improves' if ok and o['ci_low'] > 0 else 'can replace' if ok else 'cannot replace')

    verdict = {}
    for b in BANKS:
        verdict[b] = {
            'A_replace_fused': replace(q('DUFSLIU_PRM', b), q('CHAIN', b)),
            'A_replace_tele': replace(q('DUFSLIU', b), q('LSML', b)),
            'B_tele_TALIU512_vs_DUFSLIU': ci(q('TALIU@512', b), q('DUFSLIU', b)),
            'B_tele_TALIU512_vs_LR512': ci(q('TALIU@512', b), q('LR@512', b)),
            'B_fused_TALIU512_vs_DUFSLIU': ci(q('PRM_TALIU@512', b), q('DUFSLIU_PRM', b)),
            'B_fused_TALIU512_vs_LR512': ci(q('PRM_TALIU@512', b), q('PRM_LR@512', b)),
            'B_secondary_RHOLIU512_vs_DUFSLIU_tele': ci(q('RHOLIU@512', b), q('DUFSLIU', b)),
            'C_PALIU_vs_DUFSLIU': ci(q('PALIU_PRM', b), q('DUFSLIU_PRM', b)),
            'C_PALIU_prmscore_vs_DUFSLIU': ci(q('PALIU_PRM', b), q('DUFSLIU_PRM', b), 'official'),
        }
    verdict['D56_more_channels_within'] = {arm: ci(q(arm, 'N56'), q(arm, 'N16')) for arm in TELE + FUSED}
    verdict['inputs'] = {p.name: sha256(p) for p in sorted(SC.glob('*.npz'))}
    (OUT / 'VERDICT.json').write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items() if k != 'inputs'}, indent=1))


if __name__ == '__main__':
    main()
