"""One PRMBench table per PRM: the PRM, its backbone telemetry and the standalone base model's telemetry
(plan docs/experiments/PRM_STANDALONE_TELEMETRY_2026-10-22.md). Local, labels stay on this machine.

Standalone telemetry is reported alone (MEAN / L-SML). It is not fused with the PRM score (Omri, 2026-10-22).

Inputs (COLLECTION_DATA, same root as ../prm_boost_collection_v1/evaluate_collection.py):
  prmbench/ABL5_SCORES_<prm>.npz                         backbone run (collection stage 1)
  prmbench_standalone/ABL5_SCORES_<prm>__standalone.npz  ablation5 on the standalone bank (this experiment)

    python standalone_table.py         ->  tables/STANDALONE_MAIN.csv, STANDALONE_CONTRASTS.csv, STANDALONE_TABLES.md
    python standalone_table.py --smoke-no-parity   (synthetic smoke test only: skips the STATUS_MAIN parity)

Parity: every backbone row must reproduce ../prm_boost_collection_v1/tables/STATUS_MAIN.csv; the standalone file must
have the same answers, offsets, rewards and source groups as the backbone file. Any mismatch stops the script.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
COLL = HERE.parent / 'prm_boost_collection_v1'
sys.path.insert(0, str(COLL))
import evaluate_collection as E  # noqa: E402

TABLES = HERE / 'tables'
RULES = ['Q80', 'GAP', 'K2', 'ORACLE']
STANDALONE_MODEL = {'qwen_prm_7b': 'Qwen2.5-Math-7B-Instruct', 'qwen_prm_72b': 'Qwen2.5-Math-72B-Instruct',
                    'pure_prm': 'Qwen2.5-Math-7B', 'rlhflow_prm': 'Llama-3.1-8B-Instruct', 'math_shepherd': 'Mistral-7B-v0.1'}
# (row label, arm in the merged cell, uses the PRM score, telemetry source)
ROWS = [('PRM, native threshold', 'PRM_native', True, '-'),
        ('PRM alone, same rule', 'PRM_best', True, '-'),
        ('Backbone telemetry, plain mean', 'MEAN_N16', False, 'backbone'),
        ('Backbone telemetry, L-SML', 'TELE_N16', False, 'backbone'),
        ('Standalone telemetry, plain mean', 'SA_MEAN_N16', False, 'standalone'),
        ('Standalone telemetry, L-SML', 'SA_TELE_N16', False, 'standalone'),
        ('PRM + backbone telemetry, plain mean', 'EQ_N16', True, 'backbone'),
        ('PRM + backbone telemetry, L-SML', 'CHAIN_N16', True, 'backbone')]
CONTRASTS = [('Standalone vs backbone, L-SML', 'SA_TELE_N16', 'TELE_N16'),
             ('Standalone vs backbone, plain mean', 'SA_MEAN_N16', 'MEAN_N16'),
             ('Standalone: L-SML vs plain mean', 'SA_TELE_N16', 'SA_MEAN_N16'),
             ('Standalone L-SML vs the PRM alone', 'SA_TELE_N16', 'PRM_best'),
             ('Backbone L-SML vs the PRM alone', 'TELE_N16', 'PRM_best')]


def merged_cell(prm):
    c = E.load_prmbench(prm)
    s = E.load_prmbench(f'{prm}__standalone', 'prmbench_standalone')
    for k in ('ids', 'off', 'err', 'groups'):
        if not np.array_equal(getattr(c, k), getattr(s, k)): raise SystemExit(f'{prm}: standalone {k} differs from backbone')
    if not np.array_equal(c.extra['reward'], s.extra['reward']): raise SystemExit(f'{prm}: rewards differ')
    for bank in ('N16', 'N13'):
        for arm in ('MEAN', 'TELE'):
            c.arms[f'SA_{arm}_{bank}'] = s.arms[f'{arm}_{bank}']
    # 'PRM same rule' = the PRM scale that scores better under GAP (as collection_status.scale)
    best = max(('PRM_Q80', 'PRM_Q80_PROB'), key=lambda a: E.measure(c, a, 'GAP', fast=True)[0]['official'])
    c.arms['PRM_best'] = c.arms[best]
    return c, best


def status_reference():
    ref = {}
    for r in E.read_csv(COLL / 'tables/STATUS_MAIN.csv'):
        if r['benchmark'] == 'prmbench' and not r['note']: ref.setdefault((r['prm'], r['arm']), r)
    return ref


def main(parity=True):
    TABLES.mkdir(exist_ok=True); ref = status_reference() if parity else {}; main_rows, con_rows, md = [], [], []
    for prm in E.PRMS:
        c, best = merged_cell(prm)
        md += [f'## {E.PRM_LABEL[prm]} (standalone model: {STANDALONE_MODEL[prm]}; PRM scale: {best})', '',
               '| arm | uses PRM | telemetry | within-AUC | Q80 | GAP | flag 2 | true count |', '|---|---|---|---|---|---|---|---|']
        for label, arm, uses_prm, tel in ROWS:
            rules = ['native'] if arm == 'PRM_native' else RULES
            vals = {}
            for rule in rules:
                row, _, _ = E.measure(c, arm, rule, fast=True); vals[rule] = row['official']
            wa = float(np.nanmean(E.within(c, arm)))
            src_arm = best if arm == 'PRM_best' else arm
            if parity and not arm.startswith('SA_'):                         # parity with the collection tables
                r = ref[(prm, src_arm)]
                if abs(float(r['within_auc']) - wa) > 1e-9: raise SystemExit(f'PARITY within {prm} {arm}')
                for rule in rules:
                    key = 'native' if rule == 'native' else rule
                    if abs(float(r[key]) - vals[rule]) > 1e-9: raise SystemExit(f'PARITY {prm} {arm} {rule}')
            main_rows.append(dict(prm=prm, standalone_model=STANDALONE_MODEL[prm], row=label, arm=src_arm if arm != 'PRM_best' else f'PRM_best={best}',
                                  uses_prm=uses_prm, telemetry=tel, within_auc=wa, **{r: vals.get(r) for r in ['native'] + RULES}))
            cells = [f'{wa:.3f}'] + ([f"{vals['native']:.2f}", '', '', ''] if arm == 'PRM_native' else [f'{vals[r]:.2f}' for r in RULES])
            md.append(f"| {label} | {'yes' if uses_prm else 'no'} | {tel} | " + ' | '.join(cells) + ' |')
        md += ['', '| contrast (GAP, 95% CI) | PRMScore | within-AUC |', '|---|---|---|']
        W = E.boot_weights(c)
        for label, a, b in CONTRASTS:
            out = E.contrast(c, (a, 'GAP'), (b, 'GAP'), W)
            for o in out: con_rows.append(dict(o, label=label))
            o1, o2 = out
            md.append(f"| {label} | {o1['delta']:+.2f} [{o1['ci_low']:+.2f}, {o1['ci_high']:+.2f}] | "
                      f"{o2['delta']:+.3f} [{o2['ci_low']:+.3f}, {o2['ci_high']:+.3f}] |")
        md.append('')
        print(f'[standalone] {prm} done' + (f' (parity PASS on {sum(1 for r in ROWS if not r[1].startswith("SA_"))} backbone rows)' if parity else ' (NO PARITY: smoke)'), flush=True)
    E.write_csv(TABLES / 'STANDALONE_MAIN.csv', main_rows); E.write_csv(TABLES / 'STANDALONE_CONTRASTS.csv', con_rows)
    (TABLES / 'STANDALONE_TABLES.md').write_text('\n'.join(md), encoding='utf8')
    print('wrote', TABLES)


if __name__ == '__main__':
    main(parity='--smoke-no-parity' not in sys.argv)
