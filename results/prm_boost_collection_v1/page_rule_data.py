"""Ablation-page data under a chosen count rule, in the schema of results/prm_boost_ablation5_v1/page/build_data.py
(arms[a] = {score, auc, simp/sound/sens, d, da, df, daf}; pairs[<X-Y>_<bank>] = {d, da}, each [delta, lo, hi]).

    python page_rule_data.py   ->  tables/PAGE_RULE_DATA.json  (GAP for PRMBench / Socratic / Hard2Verify; Q80 for Hard2Verify)

d/da: vs PRM_native (its own rule); df/daf: vs the PRM on 1-reward under the SAME rule (page arm 3, PRM_Q80_PROB).
Paired bootstrap as evaluate_collection.contrast (2,000 draws; PRMBench by source group, external by answer).
Official metrics are on a 0-100 scale for every benchmark.
"""
import json, sys
import numpy as np
import evaluate_collection as E

sys.path.insert(0, str(E.ROOT / 'results/prm_boost_ablation5_v1/page'))
PAGE_ARMS = ['PRM_native', 'PRM_Q80', 'PRM_Q80_PROB', 'LSML_CONT'] + [f'{a}_{b}' for b in ('N16', 'N13', 'S5') for a in (
    'TELE', 'MEAN', 'CHAIN', 'PRMG', 'HALF', 'EQ', 'HALFP', 'EQP', 'CHAINP', 'TELERES', 'MEANRES', 'EQRES', 'CHAINRES',
    'TELEPERM', 'EQPERM', 'CHAINPERM')] + ['POS_ONLY', 'LEN_ONLY', 'PRMPOS', 'PRMPOSLEN']
PAIRS = ['TELE-MEAN', 'EQ-MEAN', 'EQP-MEAN', 'HALF-MEAN', 'CHAIN-HALF', 'CHAIN-TELE', 'CHAIN-EQ', 'PRMG-HALF', 'EQP-EQ',
         'HALFP-HALF', 'CHAINP-CHAIN', 'TELERES-TELE', 'MEANRES-MEAN', 'EQRES-EQ', 'CHAINRES-CHAIN', 'TELEPERM-TELE',
         'EQPERM-EQ', 'CHAINPERM-CHAIN', 'EQRES-MEANRES', 'CHAINRES-TELERES']
EXT = {'PRM_native': 'PRM', 'PRM_Q80': 'PRM', 'PRM_Q80_PROB': 'PRMPROB'}


def name(c, a):
    return a if c.bench == 'prmbench' else EXT.get(a, a)


def cell_data(c, rule):
    k = 1. if c.bench == 'prmbench' else 100.; W = E.boot_weights(c)
    nat = (name(c, 'PRM_native'), 'native'); prob = (name(c, 'PRM_Q80_PROB'), rule)
    arms = {}
    for a in PAGE_ARMS:
        na = name(c, a)
        if na not in c.arms: continue
        ru = 'native' if a == 'PRM_native' else rule
        r, _, _ = E.measure(c, na, ru, fast=True)
        e = {'score': k * r['official'], 'auc': r['within_auc'], 'src': f'collection_v1/{rule}'}
        if c.bench == 'prmbench':
            cs = E.category_scores(c, c.flags(na, ru)); e.update(simp=cs['Simplicity'], sound=cs['Soundness'], sens=cs['Sensitivity'])
        for ref, sfx in ((nat, ''), (prob, 'f')):
            if (na, ru) == ref: continue
            for x in E.contrast(c, (na, ru), ref, W):
                s = k if x['metric'] == 'official' else 1.
                e[('d' if x['metric'] == 'official' else 'da') + sfx] = [s * x['delta'], s * x['ci_low'], s * x['ci_high']]
        arms[a] = e
    pairs = {}
    for pr in PAIRS:
        x, y = pr.split('-')
        for bk in ('N16', 'N13', 'S5'):
            a, b = f'{x}_{bk}', f'{y}_{bk}'
            if a not in c.arms or b not in c.arms: continue
            for r in E.contrast(c, (a, rule), (b, rule), W):
                s = k if r['metric'] == 'official' else 1.
                pairs.setdefault(f'{pr}_{bk}', {})['d' if r['metric'] == 'official' else 'da'] = [s * r['delta'], s * r['ci_low'], s * r['ci_high']]
    return {'arms': arms, 'pairs': pairs}


def main():
    from collection_parity import h2v_common_uids
    common = h2v_common_uids(); out = {'GAP': {}, 'Q80': {}}
    for rule, benches in (('GAP', E.BENCHES), ('Q80', ['hard2verify'])):
        for b in benches:
            out[rule][b] = {}
            for prm in E.PRMS:
                c = E.load(b, prm, **({'uids_keep': common} if b == 'hard2verify' else {}))
                if b != 'prmbench':
                    v = E.load_frozen_lsml(c)
                    if v is not None: c.arms['LSML_CONT'] = v
                out[rule][b][prm] = cell_data(c, rule); print(rule, b, prm, len(out[rule][b][prm]['arms']), flush=True)
    E.TABLES.mkdir(parents=True, exist_ok=True)
    (E.TABLES / 'PAGE_RULE_DATA.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
