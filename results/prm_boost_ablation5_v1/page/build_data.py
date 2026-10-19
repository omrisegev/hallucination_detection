"""Collect the five-arm ablation numbers into one JSON for the page (no computation beyond S/S/S averages).

PRMBench sources, first match wins per arm:
  ../ABL5_RESULTS_<prm>.csv + ABL5_BOOTSTRAP (AIRCC ablation5.py; all arms)          when present
  ../../prm_boosting_bank_sweep_v1/SWEEP_*  (PRM_native, LSML_CONT, TELE/PRMG/CHAIN_<bank>, Q80 rows)
  ../../prm_boosting_controls_v1/CONTROLS_* (HALF_N16)
Socratic: ../socratic/{RESULTS,BOOTSTRAP}_socratic.csv (ablation5_socratic.py + eval_socratic_ablation5.py).
Math-Shepherd comes from the digit-fixed rebuild when present (../digitfix_v1/, AIRCC job 288932): PRMBench
digitfix/ABL5_*_math_shepherd.*, Socratic digitfix_v1/socratic/*. The committed zero-digit files stay as records.
PRMBench S/S/S follow the paper: Simplicity = mean(redundancy, circular); Soundness = mean(counterfactual,
step contradiction, domain inconsistency, confidence); Sensitivity = mean(missing condition, deception, MS),
with MS = 2 x prmscore__multi_solutions (valid-F1 only on all-correct answers).

    python build_data.py   -> data.json
"""
import csv, json
from pathlib import Path

H = Path(__file__).resolve().parent; A5 = H.parent; R = A5.parent; DIGITFIX = A5 / 'digitfix_v1/digitfix'
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
SOC = {'qwen_prm_7b': 'qwen_prm'}
NAMES = {'qwen_prm_7b': 'Qwen2.5-Math-PRM-7B', 'qwen_prm_72b': 'Qwen2.5-Math-PRM-72B', 'pure_prm': 'PURE-PRM-7B',
         'rlhflow_prm': 'RLHFlow-PRM-8B', 'math_shepherd': 'Math-Shepherd-PRM-7B'}
ARMS = ['PRM_native', 'PRM_Q80', 'PRM_Q80_PROB', 'LSML_CONT'] + [f'{a}_{b}' for b in ('N16', 'N13', 'S5')
                                                                   for a in ('TELE', 'MEAN', 'CHAIN', 'PRMG', 'HALF', 'EQ', 'HALFP', 'EQP', 'CHAINP',
                                                                             'TELERES', 'MEANRES', 'EQRES', 'CHAINRES',
                                                                             'TELEPERM', 'EQPERM', 'CHAINPERM')] + ['POS_ONLY', 'LEN_ONLY', 'PRMPOS', 'PRMPOSLEN']
PAIRS = ['TELE-MEAN', 'EQ-MEAN', 'EQP-MEAN', 'HALF-MEAN', 'CHAIN-HALF', 'CHAIN-TELE', 'CHAIN-EQ', 'PRMG-HALF',
         'EQP-EQ', 'HALFP-HALF', 'CHAINP-CHAIN', 'TELERES-TELE', 'MEANRES-MEAN', 'EQRES-EQ', 'CHAINRES-CHAIN',
         'TELEPERM-TELE', 'EQPERM-EQ', 'CHAINPERM-CHAIN', 'EQRES-MEANRES', 'CHAINRES-TELERES']


def rd(p):
    with open(p, newline='') as f: return list(csv.DictReader(f))


def sss(r):
    g = lambda k: float(r['prmscore__' + k])
    return {'simp': (g('redundency') + g('circular')) / 2,
            'sound': (g('counterfactual') + g('step_contradiction') + g('domain_inconsistency') + g('confidence')) / 4,
            'sens': (g('missing_condition') + g('deception') + 2 * g('multi_solutions')) / 3}


def prmbench(p):
    arms, src = {}, {}
    def add(rows, boot, tag, keep=lambda r: True, cname=lambda a, b: f'{a} - {b}'):
        bd = {(b['contrast'], b['metric']): b for b in boot}
        for r in rows:
            a = r['arm']
            if a in arms or a not in ARMS or not keep(r): continue
            e = {'score': float(r['prmscore']), 'auc': float(r['within_auc']), **sss(r), 'src': tag}
            if 'flag_rate' in r and r['flag_rate']: e['flag_rate'] = float(r['flag_rate'])
            for ref, sfx in (('PRM_native', ''), ('PRM_Q80_PROB', 'f')):
                for mt, k in (('prmscore', 'd'), ('within_auc', 'da')):
                    b = bd.get((cname(a, ref), mt))
                    if b: e[k + sfx] = [float(b['delta']), float(b['ci_low']), float(b['ci_high'])]
            arms[a] = e
    src = DIGITFIX if p == 'math_shepherd' and (DIGITFIX / f'ABL5_RESULTS_{p}.csv').exists() else A5
    ab = src / f'ABL5_RESULTS_{p}.csv'
    if ab.exists(): add(rd(ab), rd(src / f'ABL5_BOOTSTRAP_{p}.csv'), 'digitfix' if src == DIGITFIX else 'ablation5')
    sw = R / 'prm_boosting_bank_sweep_v1'
    add(rd(sw / f'SWEEP_RESULTS_{p}.csv'), rd(sw / f'SWEEP_BOOTSTRAP_{p}.csv'), 'bank_sweep',
        keep=lambda r: r['rule'] == 'Q80', cname=lambda a, b: f'{a}[Q80] - {b}[Q80]')
    co = R / 'prm_boosting_controls_v1'
    add(rd(co / f'CONTROLS_RESULTS_{p}.csv'), rd(co / f'CONTROLS_BOOTSTRAP_{p}.csv'), 'controls')
    pairs = {}
    boots = []
    if ab.exists(): boots += [(b, '') for b in rd(src / f'ABL5_BOOTSTRAP_{p}.csv')]
    boots += [(b, '[Q80]') for b in rd(sw / f'SWEEP_BOOTSTRAP_{p}.csv')]
    for b, sfx in boots:
        for pr in PAIRS:
            x, y = pr.split('-')
            for bk in ('N16', 'N13', 'S5'):
                if b['contrast'] == f'{x}_{bk}{sfx} - {y}_{bk}{sfx}':
                    k = f'{pr}_{bk}'; m = 'd' if b['metric'] == 'prmscore' else 'da'
                    pairs.setdefault(k, {}).setdefault(m, [float(b['delta']), float(b['ci_low']), float(b['ci_high'])])
    return arms, pairs


def socratic(p):
    q = SOC.get(p, p); d = R / 'prm_boost_ablation5_v1/socratic'
    if p == 'math_shepherd' and (A5 / 'digitfix_v1/socratic/RESULTS_socratic.csv').exists():
        d = A5 / 'digitfix_v1/socratic'
    bd = {(b['contrast'], b['metric']): b for b in rd(d / 'BOOTSTRAP_socratic.csv') if b['model'] == q}
    nm = lambda a: a if a == 'PRM_native' else {'PRM_Q80': 'PRM', 'PRM_Q80_PROB': 'PRMPROB'}.get(a, a) + '_Q80'
    arms = {}
    for r in rd(d / 'RESULTS_socratic.csv'):
        if r['model'] != q: continue
        a = 'PRM_native' if r['arm'] == 'PRM_native' else r['arm'][:-4]
        a = {'PRM': 'PRM_Q80', 'PRMPROB': 'PRM_Q80_PROB', 'PRMPOS': 'PRMPOS'}.get(a, a)
        e = {'score': 100 * float(r['official_metric']), 'auc': float(r['within_auc']), 'src': 'socratic'}
        for ref, sfx in (('PRM_native', ''), ('PRMPROB_Q80', 'f')):
            for mt, k, s in (('official_metric', 'd', 100), ('within_auc', 'da', 1)):
                b = bd.get((f'{nm(a)} - {ref}', mt))
                if b: e[k + sfx] = [s * float(b['delta']), s * float(b['ci_low']), s * float(b['ci_high'])]
        arms[a] = e
    pairs = {}
    for pr in PAIRS:
        x, y = pr.split('-')
        for bk in ('N16', 'N13'):
            for mt, k, s in (('official_metric', 'd', 100), ('within_auc', 'da', 1)):
                b = bd.get((f'{x}_{bk}_Q80 - {y}_{bk}_Q80', mt))
                if b: pairs.setdefault(f'{pr}_{bk}', {})[k] = [s * float(b['delta']), s * float(b['ci_low']), s * float(b['ci_high'])]
    return arms, pairs


def fill_fair_auc(arms):
    # within-AUC is rank-based inside each answer, and 1 - reward is a monotone map of the native risk, so the
    # same-rule PRM has the same per-answer AUC as PRM_native: its within-AUC contrast equals the native one.
    for e in arms.values():
        if 'da' in e and 'daf' not in e: e['daf'] = e['da']


out = {'prms': PRMS, 'names': NAMES, 'bench': {}}
for bench, fn in (('prmbench', prmbench), ('socratic', socratic)):
    out['bench'][bench] = {}
    for p in PRMS:
        arms, pairs = fn(p)
        fill_fair_auc(arms)
        out['bench'][bench][p] = {'arms': arms, 'pairs': pairs}
(H / 'data.json').write_text(json.dumps(out, indent=1))
for b in out['bench']:
    for p in PRMS:
        a = out['bench'][b][p]['arms']
        print(b, p, len(a), sorted(a)[:30], list(out['bench'][b][p]['pairs']))
