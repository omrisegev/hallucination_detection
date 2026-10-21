"""Plan section 2.1 (status snapshot): tables per benchmark x 5 PRMs, written to tables/.

    python evaluate_collection.py status

Outputs (every row carries benchmark, PRM, arm, rule and access label):
  STATUS_MAIN.csv          rows = PRM native, PRM same rule (both scales; 'PRM_best' = the better scale under GAP),
                           TELE/MEAN/EQ/CHAIN_N16, CHAIN_N13, LSML_CONT (label-selected), HALF_N16 (control);
                           columns = within-AUC, pooled AUROC, official metric under GAP / Q80 / K2 / ORACLE
  STATUS_CONTRASTS.csv     paired bootstrap: CHAIN-PRM (GAP), CHAIN-EQ, TELE-MEAN, EQ-PRM, CHAIN_N13-PRM; GAP-Q80 per arm
  ABLATION_GAP.csv         PRM -> EQ -> CHAIN and MEAN -> TELE -> CHAIN steps under GAP, plus controls
  ABLATION_RANKS.csv       rank of every arm per cell under GAP (and Q80)
  PRMBENCH_CATEGORIES.csv  9 categories + S/S/S, PRM (GAP, better scale) vs CHAIN_N16 (GAP), 5 PRMs
  PRM_WEIGHT_SHARE.csv     least-squares PRM weight share in CHAIN/PRMG fits, shipped vs sign-abs, sign flips
"""
import json
import numpy as np
import evaluate_collection as E

T = E.TABLES
MAIN_ARMS = ['TELE_N16', 'MEAN_N16', 'EQ_N16', 'CHAIN_N16', 'CHAIN_N13', 'LSML_CONT', 'HALF_N16']
CONTROL_ARMS = ['TELERES_N16', 'MEANRES_N16', 'EQRES_N16', 'CHAINRES_N16', 'TELEPERM_N16', 'EQPERM_N16', 'CHAINPERM_N16',
                'POS_ONLY', 'LEN_ONLY', 'PRMPOS', 'PRMPOSLEN']
SCALE_ARMS = {'prmbench': ('PRM_Q80', 'PRM_Q80_PROB'), 'socratic': ('PRM', 'PRMPROB'), 'hard2verify': ('PRM', 'PRMPROB')}
SCALE_NAME = {'PRM_Q80': 'logit(1-reward)', 'PRM_Q80_PROB': '1-reward', 'PRM': 'logit(1-reward)', 'PRMPROB': '1-reward'}
RULE_COLS = ['GAP', 'Q80', 'K2', 'ORACLE']


def cells():
    common = None
    for bench in E.BENCHES:
        for prm in E.PRMS:
            kw = {}
            if bench == 'hard2verify':
                if common is None:
                    from collection_parity import h2v_common_uids
                    common = h2v_common_uids()
                kw['uids_keep'] = common
            c = E.load(bench, prm, **kw)
            if bench != 'prmbench':
                v = E.load_frozen_lsml(c)
                if v is not None: c.arms['LSML_CONT'] = v
            yield c


def scale(c):
    return 1. if c.bench == 'prmbench' else 100.       # report every official metric on a 0-100 scale


def main_rows(c, check_official=True):
    rows, k = [], scale(c)
    native = 'PRM_native' if c.bench == 'prmbench' else 'PRM'
    r, _, _ = E.measure(c, native, 'native')
    rows.append(dict(benchmark=c.bench, prm=c.prm, row='PRM native', arm=native, scale='', access=E.ACCESS['PRM'],
                     within_auc=r['within_auc'], pooled_auroc=r['pooled_auroc'], native=k * r['official'],
                     n_answers=r['n_answers']))
    gap_by_scale = {}
    for a in SCALE_ARMS[c.bench] + tuple(x for x in MAIN_ARMS if x in c.arms):
        row = dict(benchmark=c.bench, prm=c.prm, row=a, arm=a, scale=SCALE_NAME.get(a, ''), access=E.access_of(a))
        for rule in RULE_COLS:
            rr, _, _ = E.measure(c, a, rule, fast=True)
            if check_official and c.bench == 'prmbench' and rule == 'GAP':
                full, _, _ = E.measure(c, a, rule)                 # official prmbench_evaluate path
                assert abs(full['official'] - rr['official']) < 1e-9, (c.prm, a)
            row[rule] = k * rr['official']
            row.setdefault('within_auc', rr['within_auc']); row.setdefault('pooled_auroc', rr['pooled_auroc'])
            row[f'flag_rate_{rule}'] = rr['flag_rate']
        row['n_answers'] = len(c.ids)
        if a in SCALE_ARMS[c.bench]: gap_by_scale[a] = row['GAP']
        rows.append(row)
    best = max(gap_by_scale, key=gap_by_scale.get)
    src = next(r for r in rows if r['arm'] == best and r['row'] != 'PRM native')   # external native row shares arm 'PRM'
    rows.append(dict(src, row='PRM same rule (better scale under GAP)',
                     note='scale picked with labels after scoring (favours the PRM reference)'))
    return rows, best


def contrasts(c, best):
    W = E.boot_weights(c); out = []
    pairs = [(('CHAIN_N16', 'GAP'), (best, 'GAP')), (('CHAIN_N13', 'GAP'), (best, 'GAP')),
             (('CHAIN_N16', 'GAP'), ('EQ_N16', 'GAP')), (('TELE_N16', 'GAP'), ('MEAN_N16', 'GAP')),
             (('EQ_N16', 'GAP'), (best, 'GAP')), (('CHAIN_N16', 'GAP'), ('TELE_N16', 'GAP')),
             (('MEAN_N16', 'GAP'), (best, 'GAP')), (('TELE_N16', 'GAP'), (best, 'GAP')),
             (('HALF_N16', 'GAP'), (best, 'GAP')), (('CHAIN_N16', 'GAP'), ('PRM_native' if c.bench == 'prmbench' else 'PRM', 'native'))]
    if 'LSML_CONT' in c.arms: pairs.append((('LSML_CONT', 'GAP'), (best, 'GAP')))
    for a in list(SCALE_ARMS[c.bench]) + [x for x in MAIN_ARMS if x in c.arms]:
        pairs.append(((a, 'GAP'), (a, 'Q80')))
    for a, b in pairs:
        for r in E.contrast(c, a, b, W):
            if c.bench != 'prmbench' and r['metric'] == 'official':
                for kk in ('delta', 'delta_bootstrap_point', 'ci_low', 'ci_high'): r[kk] *= 100
            r['note'] = 'PRM reference scale chosen post hoc' if best in (a[0], b[0]) else ''
            out.append(r)
    return out


def ablation(c, best):
    steps = [('PRM -> EQ (add telemetry, equal weight)', ('EQ_N16', 'GAP'), (best, 'GAP')),
             ('EQ -> CHAIN (L-SML weights)', ('CHAIN_N16', 'GAP'), ('EQ_N16', 'GAP')),
             ('MEAN -> TELE (L-SML on telemetry)', ('TELE_N16', 'GAP'), ('MEAN_N16', 'GAP')),
             ('TELE -> CHAIN (add PRM)', ('CHAIN_N16', 'GAP'), ('TELE_N16', 'GAP'))]
    for a in CONTROL_ARMS:
        if a in c.arms:
            base = a.replace('RES', '').replace('PERM', '') if ('RES' in a or 'PERM' in a) else best
            steps.append((f'control {a} vs {base}', (a, 'GAP'), (base, 'GAP')))
    W = E.boot_weights(c); rows = []
    for name, a, b in steps:
        for r in E.contrast(c, a, b, W):
            if c.bench != 'prmbench' and r['metric'] == 'official':
                for kk in ('delta', 'delta_bootstrap_point', 'ci_low', 'ci_high'): r[kk] *= 100
            rows.append(dict(step=name, access_a=E.access_of(a[0]), **r))
    return rows


def ranks(c):
    out = []
    arms = [a for a in c.arms if a not in ('PRM_native',)]
    for rule in ('GAP', 'Q80'):
        vals = {a: scale(c) * E.measure(c, a, rule, fast=True)[0]['official'] for a in arms}
        for i, (a, v) in enumerate(sorted(vals.items(), key=lambda kv: -kv[1])):
            out.append(dict(benchmark=c.bench, prm=c.prm, rule=rule, rank=i + 1, arm=a, official=v, access=E.access_of(a)))
    return out


def categories(c, best):
    rows = []
    for arm, label in ((best, 'PRM (GAP, better scale)'), ('CHAIN_N16', 'CHAIN_N16 (GAP)'), ('CHAIN_N13', 'CHAIN_N13 (GAP)'),
                       ('PRM_native', 'PRM native')):
        f = c.flags(arm, 'native' if arm == 'PRM_native' else 'GAP')
        rows.append(dict(prm=c.prm, arm=arm, label=label, access=E.access_of(arm),
                         Overall=E.official(c, f, arm == 'PRM_native')[0], **E.category_scores(c, f)))
    return rows


def weight_share(c):
    """PRM weight share of the shipped fits and of the sign-abs variant (same input views, other fused score)."""
    variants = {'shipped': c}
    try:
        variants['signabs'] = (E.load_prmbench(c.prm, 'prmbench_signabs') if c.bench == 'prmbench' else
                               E.load_external(c.bench, c.prm, variant={'socratic': 'socratic_signabs', 'hard2verify': 'h2v_signabs'}[c.bench],
                                               uids_keep=set(c.ids) if c.bench == 'hard2verify' else None))
        assert np.array_equal(variants['signabs'].off, c.off)
    except FileNotFoundError:
        pass
    rows = []
    for name, cv in variants.items():
        for bank in ('N13', 'N16'):
            for arm in ('CHAIN', 'PRMG'):
                key = f'{arm}_{bank}'
                if key not in cv.arms: continue
                ws = E.prm_weight_share(c, arm, bank, score=cv.arms[key])
                if ws is None: continue
                rows.append(dict(benchmark=c.bench, prm=c.prm, variant=name, fit=key,
                                 corr_with_TELE=float(np.corrcoef(cv.arms[key], cv.arms[f'TELE_{bank}'])[0, 1]),
                                 within_auc=float(np.nanmean(E.within(cv, key))), **ws))
    return rows


def sign_flip_meta(WS):
    """Anchor flip = the fused score ends anti-correlated with the telemetry it was built from (corr(fit, TELE) < 0): the
    flip toward z(PRM) makes the PRM weight positive by construction, so a negative PRM weight is NOT the detector."""
    out = {}
    for v in ('shipped', 'signabs'):
        out[f'anchor_flips_{v}'] = {b: dict(n_fits=sum(r['benchmark'] == b and r['variant'] == v for r in WS),
                                            flipped=[f"{r['prm']}:{r['fit']}" for r in WS if r['benchmark'] == b and r['variant'] == v
                                                     and float(r['corr_with_TELE']) < 0]) for b in E.BENCHES}
    return out


def main_only():
    """Recompute STATUS_MAIN.csv and the sign-flip metadata only (fast)."""
    M, META = [], json.loads((T / 'STATUS_META.json').read_text())
    for c in cells():
        rows, best = main_rows(c, check_official=False); M += rows
    E.write_csv(T / 'STATUS_MAIN.csv', M)
    META = {k: v for k, v in META.items() if not k.startswith(('sign_flips', 'anchor_flips'))}
    META.update(sign_flip_meta(E.read_csv(T / 'PRM_WEIGHT_SHARE.csv')))
    (T / 'STATUS_META.json').write_text(json.dumps(META, indent=1))
    print(json.dumps({k: v for k, v in META.items() if k.startswith('anchor')}, indent=1))


def main():
    T.mkdir(parents=True, exist_ok=True)
    M, CON, AB, RK, CAT, WS, META = [], [], [], [], [], [], {}
    for c in cells():
        print('cell', c.bench, c.prm, len(c.ids), 'answers', c.off[-1], 'scored steps', flush=True)
        rows, best = main_rows(c); M += rows
        META[f'{c.bench}/{c.prm}'] = dict(n_answers=len(c.ids), n_scored_steps=int(c.off[-1]),
                                          n_metric_steps=int(c.keep.sum()), prm_reference_scale=SCALE_NAME[best],
                                          lsml_cont_available='LSML_CONT' in c.arms)
        CON += contrasts(c, best); AB += ablation(c, best); RK += ranks(c)
        if c.bench == 'prmbench': CAT += categories(c, best)
        WS += weight_share(c)
    E.write_csv(T / 'STATUS_MAIN.csv', M); E.write_csv(T / 'STATUS_CONTRASTS.csv', CON)
    E.write_csv(T / 'ABLATION_GAP.csv', AB); E.write_csv(T / 'ABLATION_RANKS.csv', RK)
    E.write_csv(T / 'PRMBENCH_CATEGORIES.csv', CAT); E.write_csv(T / 'PRM_WEIGHT_SHARE.csv', WS)
    META.update(sign_flip_meta(WS))
    (T / 'STATUS_META.json').write_text(json.dumps(META, indent=1))
    print(json.dumps(META['sign_flips_shipped'], indent=1))


if __name__ == '__main__':
    import sys
    main_only() if 'main_only' in sys.argv else main()
