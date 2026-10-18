"""Build the README tables of the PRM bank sweep (AIRCC job 288319) from SWEEP_* files. Read-only."""
import csv, json
from pathlib import Path

D = Path(__file__).resolve().parent
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
NAME = {'qwen_prm_7b': 'Qwen-7B', 'qwen_prm_72b': 'Qwen-72B', 'pure_prm': 'PURE',
        'rlhflow_prm': 'RLHFlow', 'math_shepherd': 'MathShepherd'}
BANKS = ['N11', 'N13', 'N14', 'N16', 'S5', 'S6']
R = {m: {(r['arm'], r['rule']): r for r in csv.DictReader(open(D / f'SWEEP_RESULTS_{m}.csv'))} for m in PRMS}
B = {m: {(r['contrast'], r['metric']): r for r in csv.DictReader(open(D / f'SWEEP_BOOTSTRAP_{m}.csv'))} for m in PRMS}
G = {m: json.loads((D / f'SWEEP_DIAG_{m}.json').read_text()) for m in PRMS}
out = []
hdr = '| arm | ' + ' | '.join(NAME[m] for m in PRMS) + ' |\n|---|' + '---|' * len(PRMS)


def cell(m, arm, rule='Q80'):
    r = R[m].get((arm, rule))
    return '-' if r is None else f"{float(r['within_auc']):.4f} / {float(r['prmscore']):.2f}"


def ci(m, con, metric, fmt):
    r = B[m].get((con, metric))
    if r is None: return '-'
    d, lo, hi = float(r['delta']), float(r['ci_low']), float(r['ci_high'])
    star = '' if lo <= 0 <= hi else '*'
    return f"{d:+{fmt}} [{lo:+{fmt}}, {hi:+{fmt}}]{star}"


out.append('### Table 1. within-AUC / PRMScore (Q80)\n\n' + hdr)
rows = ['PRM_native', 'LSML_CONT'] + [f'{a}_{b}' for b in BANKS for a in ('TELE', 'PRMG', 'AUTO', 'CHAIN')]
for a in rows: out.append(f'| {a} | ' + ' | '.join(cell(m, a) for m in PRMS) + ' |')

out.append('\n### Table 2. PRMScore contrast vs PRM_native and vs LSML_CONT (Q80), 95% source-group CI\n\n' + hdr)
for b in BANKS:
    for a in ('PRMG', 'AUTO', 'CHAIN'):
        for ref in ('PRM_native', 'LSML_CONT'):
            con = f'{a}_{b}[Q80] - {ref}[Q80]'
            out.append(f'| {a}_{b} - {ref} | ' + ' | '.join(ci(m, con, 'prmscore', '.2f') for m in PRMS) + ' |')

out.append('\n### Table 3. PRM weight share in the fused score (least squares, |w_PRM| / sum |w|)\n\n'
           '| bank | arm | ' + ' | '.join(NAME[m] for m in PRMS) + ' |\n|---|---|' + '---|' * len(PRMS))
for b in BANKS:
    for a in ('PRMG', 'AUTO', 'CHAIN'):
        vals = []
        for m in PRMS:
            g = G[m].get(b, {})
            v = g.get(f'{a}_prm_share')
            extra = '' if a != 'AUTO' or v is None else (' (alone)' if g.get('AUTO_prm_alone') else ' (merged)')
            vals.append('-' if v is None else f'{v:.3f}{extra}')
        out.append(f'| {b} | {a} | ' + ' | '.join(vals) + ' |')

out.append('\n### Table 4. CHAIN - PRMG (Q80) with the PRM cross-stage share in each\n\n'
           'Cell: dPRMScore [CI] ; dAUC ; PRM cross share PRMG -> CHAIN ; K+1 (u = CHAIN_undetermined)\n\n' + hdr.replace('| arm |', '| bank |', 1))
for b in BANKS:
    vals = []
    for m in PRMS:
        g = G[m].get(b)
        if g is None: vals.append('-'); continue
        con = f'CHAIN_{b}[Q80] - PRMG_{b}[Q80]'
        u = ' u' if g.get('CHAIN_undetermined') else ''
        vals.append(f"{ci(m, con, 'prmscore', '.2f')} ; {float(B[m][(con, 'within_auc')]['delta']):+.4f} ; "
                    f"{g['PRMG_prm_cross_share']:.2f} -> {g['CHAIN_prm_cross_share']:.2f} ; {g['CHAIN_n_views']}{u}")
    out.append(f'| {b} | ' + ' | '.join(vals) + ' |')

out.append('\n### Table 5. ORACLE - Q80 PRMScore (count headroom; ORACLE uses labels)\n\n' + hdr)
for a in rows:
    vals = []
    for m in PRMS:
        q, o = R[m].get((a, 'Q80')), R[m].get((a, 'ORACLE'))
        vals.append('-' if q is None or o is None else
                    f"{float(o['prmscore']):.2f} ({float(o['prmscore']) - float(q['prmscore']):+.2f})")
    out.append(f'| {a} | ' + ' | '.join(vals) + ' |')

out.append('\n### Table 6. Groups found (K) and skipped banks\n\n'
           '| bank | ' + ' | '.join(NAME[m] for m in PRMS) + ' |\n|---|' + '---|' * len(PRMS))
for b in BANKS:
    vals = []
    for m in PRMS:
        g = G[m].get(b)
        vals.append('-' if g is None else
                    f"TELE {g['tele_K']}, PRMG {g['PRMG_K']}, AUTO {g['AUTO_K']}"
                    + (' deg:' + ','.join(k for k, v in g['degenerate'].items() if v) if any(g['degenerate'].values()) else ''))
    out.append(f'| {b} | ' + ' | '.join(vals) + ' |')
out.append('\nskipped_banks: ' + '; '.join(f"{NAME[m]}: {G[m]['skipped_banks'] or 'none'}" for m in PRMS))
(D / 'TABLES.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('\n'.join(out))
