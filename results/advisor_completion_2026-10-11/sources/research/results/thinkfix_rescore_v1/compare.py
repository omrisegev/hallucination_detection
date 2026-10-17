"""Build the OLD vs NEW comparison tables for thinkfix_rescore_v1.

    python results/thinkfix_rescore_v1/compare.py --frozen-root <main checkout> --root <thinkfix_rescore_v1 dir>

Columns per (table, bank, scorer, group, metric):
  frozen          the frozen Monday result (monday_nested_banks_v2 / monday_cont_baselines_v1 / monday_lr_reference_v1)
  old_replay      this pipeline on the bank rebuilt from the OLD raw cells (must equal frozen; replay_minus_frozen ~ 0)
  thinkfix        rebuilt from the think-fix teacher-forced cells (empty think block in the prefix)
  thinkfix_mask1  thinkfix with forced token 0 removed (primary; Omri 2026-10-10)
Mind the Gap reference rows are absent from the re-scores (EVIDENCE_DROP.npz no longer exists) and are dropped.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VARIANTS = ['old_replay', 'thinkfix', 'thinkfix_mask1']
VDIR = {'old_replay': 'old', 'thinkfix': 'thinkfix', 'thinkfix_mask1': 'thinkfix_mask1'}
TABLES = {  # name: (frozen path, variant-relative path, key columns, metric columns)
    'panel_a': ('results/monday_nested_banks_v2/PANEL_A_ABSOLUTE.csv', 'nested_banks/PANEL_A_ABSOLUTE.csv', ['bank', 'scorer'],
                ['prmb_within_auc', 'prmb_top1_on_error', 'prmb_prmscore_top1', 'pb_argmax_hit']),
    'prmbench_by_category': ('results/monday_nested_banks_v2/PANEL_B_PRMBENCH_TYPES.csv', 'nested_banks/PANEL_B_PRMBENCH_TYPES.csv',
                             ['group', 'bank', 'scorer'], ['within_auc', 'top1_on_error', 'prmscore_top1', 'f1_correct_top1', 'f1_error_top1']),
    'prmbench_panel_b': ('results/monday_nested_banks_v2/PANEL_B_PRMBENCH.csv', 'nested_banks/PANEL_B_PRMBENCH.csv', ['bank', 'scorer'],
                         ['within_auc', 'top1_on_error', 'prmscore_top1', 'prmscore_own']),
    'cont_baselines': ('results/monday_cont_baselines_v1/ABSOLUTE.csv', 'cont_baselines/ABSOLUTE.csv', ['bank', 'scorer'],
                       ['prmb_within_auc', 'prmb_top1_on_error', 'prmb_prmscore_top1', 'pb_argmax_hit']),
    'lr_reference': ('results/monday_lr_reference_v1/LR_SCORES.csv', 'lr_reference/LR_SCORES.csv', ['bank', 'scorer'],
                     ['prmb_within_auc', 'prmb_top1_on_error', 'prmb_prmscore_top1', 'pb_argmax_hit']),
    'lr_gaps': ('results/monday_lr_reference_v1/GAPS.csv', 'lr_reference/GAPS.csv', ['bank'],
                ['lr_within_auc', 'best_within_auc', 'gap_vs_best', 'gap_vs_MEAN_MOM_ga', 'gap_vs_ALL_EQ', 'pb_lr_hit', 'pb_gap_vs_MEAN_MOM_ga']),
    'processbench_strata': ('results/monday_nested_banks_v2/PANEL_B_PROCESSBENCH.csv', 'nested_banks/PANEL_B_PROCESSBENCH.csv',
                            ['stratum', 'bank', 'scorer'], ['argmax_hit']),
    'processbench_f1': ('results/monday_nested_banks_v2/PANEL_B_PROCESSBENCH_F1.csv', 'nested_banks/PANEL_B_PROCESSBENCH_F1.csv',
                        ['bank', 'scorer', 'rule'], ['official_f1', 'error_hit', 'clean_correct']),
}


def long(df, keys, metrics):
    df = df[~df.apply(lambda c: c.astype(str).str.contains('Mind the Gap')).any(axis=1)]
    m = [c for c in metrics if c in df.columns]
    out = df[keys + m].melt(id_vars=keys, value_vars=m, var_name='metric', value_name='value')
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--frozen-root', type=Path, required=True); ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args(); frames = []; missing = []
    for name, (frozen, rel, keys, metrics) in TABLES.items():
        fp = a.frozen_root / frozen
        if not fp.exists(): missing.append(str(fp)); continue
        base = long(pd.read_csv(fp), keys, metrics).rename(columns={'value': 'frozen'})
        for v in VARIANTS:
            p = a.root / VDIR[v] / rel
            if not p.exists(): missing.append(str(p)); base[v] = np.nan; continue
            base = base.merge(long(pd.read_csv(p), keys, metrics).rename(columns={'value': v}), on=keys + ['metric'], how='outer')
        base.insert(0, 'table', name)
        for c in ('group', 'stratum', 'rule'):
            if c not in base.columns: base[c] = ''
        frames.append(base)
    df = pd.concat(frames, ignore_index=True)
    for v in VARIANTS:
        df[v] = pd.to_numeric(df[v], errors='coerce')
    df['frozen'] = pd.to_numeric(df['frozen'], errors='coerce')
    df['replay_minus_frozen'] = df.old_replay - df.frozen
    df['thinkfix_minus_old'] = df.thinkfix - df.old_replay
    df['mask1_minus_old'] = df.thinkfix_mask1 - df.old_replay
    cols = ['table', 'bank', 'scorer', 'group', 'stratum', 'rule', 'metric', 'frozen', 'old_replay', 'thinkfix', 'thinkfix_mask1',
            'replay_minus_frozen', 'thinkfix_minus_old', 'mask1_minus_old']
    df = df[cols]
    prm = df[~df.table.isin(['processbench_strata', 'processbench_f1']) & ~df.metric.str.startswith('pb_')]
    pb = df[df.table.isin(['processbench_strata', 'processbench_f1']) | df.metric.str.startswith('pb_')]
    prm.to_csv(a.root / 'PRMBENCH_COMPARISON.csv', index=False); pb.to_csv(a.root / 'PROCESSBENCH_COMPARISON.csv', index=False)
    rep = df.replay_minus_frozen.abs()
    summary = {'rows': len(df), 'replay_rows_compared': int(rep.notna().sum()), 'replay_max_abs_diff': float(rep.max()),
               'replay_rows_over_1e-9': int((rep > 1e-9).sum()), 'missing_inputs': missing}
    pd.Series(summary).to_json(a.root / 'COMPARISON_SUMMARY.json', indent=1)
    print(summary)
    key = prm[(prm.table == 'panel_a') & (prm.metric.isin(['prmb_within_auc', 'prmb_prmscore_top1'])) & (prm.bank == 'N16')]
    print(key.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
