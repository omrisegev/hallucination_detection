"""Aggregate the AIRCC CROP-controls output (crop_controls_v1/<prm>/CROP_CONTROLS_<prm>.csv) into tables and a figure.

    python crop_controls_report.py <dir with the 5 per-PRM folders>

Outputs (results/prm_boost_collection_v1/):
  CROP_CONTROLS_PARITY.csv            approx-subset rows (controls off) vs tables/COMP_CROP_T6.csv; must be exact
  tables/CROP_C1_MAIN.csv             C1 subset, alpha .05, every proxy x grid: AUROC, kept, risk, empty, mean +- sd over 10 seeds
  tables/CROP_C4_INDUCTIVE.csv        transductive vs inductive fused arms, side by side
  tables/CROP_C5_SHUFFLES.csv         none / label shuffle / order shuffle, alpha .05
  tables/CROP_C6_ALPHA.csv            alpha sweep, both grids: kept, risk, empty-prefix rate, infeasible seeds, 1/(n+1)>alpha mark
  tables/CROP_C7_BOUNDARY.csv         |M-O|/T, over-withholding, unsafe overshoot
  figures/crop_controls_c6_kept_vs_alpha.png (+ data CSV)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent.parent; T = H / 'tables'; FIG = H / 'figures'
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
COLS = ['auroc', 'kept', 'risk', 'empty', 'bdev', 'over_withhold', 'overshoot']
PARITY_MAP = {'Random': 'fixed101_raw', 'Direct PRM (1-reward)': 'fixed101_raw', 'Position/length LR (supervised)': 'fixed101_raw',
              'Telemetry13+PRM LR (supervised)': 'fixed101_raw', 'TELE_N16': 'quantile201', 'CHAIN_N16': 'quantile201',
              'CHAIN_N13': 'quantile201', 'LSML_CONT': 'quantile201'}


def agg(df, by):
    g = df.groupby(by, sort=False)
    out = g[COLS].mean().add_suffix('_mean').join(g[['kept', 'risk', 'auroc']].std(ddof=1).add_suffix('_sd'))
    out['n_seeds'] = g.size(); out['n_infeasible'] = g['feasible'].apply(lambda s: int((~s.astype(bool)).sum()))
    out['impossible_1_over_n1'] = g['impossible'].apply(lambda s: bool(s.astype(bool).any()))
    return out.reset_index()


def main(d):
    D = pd.concat([pd.read_csv(Path(d) / p / f'CROP_CONTROLS_{p}.csv') for p in PRMS if (Path(d) / p / f'CROP_CONTROLS_{p}.csv').exists()])
    D['feasible'] = D['feasible'].astype(str).eq('True'); D['impossible'] = D['impossible'].astype(str).eq('True')
    # ---- parity vs COMP_CROP_T6 (approx subset, controls off, alpha .05, the grid that table used)
    ref = pd.read_csv(T / 'COMP_CROP_T6.csv'); rows = []
    A = agg(D[(D.subset == 'approx') & (D.control == 'none') & (D.alpha == .05)], ['prm', 'proxy', 'grid'])
    for _, r in ref.iterrows():
        nm = r['proxy']; g = 'quantile201' if nm.endswith('quantile grid') else PARITY_MAP.get(nm)
        base = nm.replace(', quantile grid', '')
        m = A[(A.prm == r['prm']) & (A.proxy == base) & (A.grid == g)]
        for k, mk in (('auroc_mean', 'auroc_mean'), ('kept_mean', 'kept_mean'), ('risk_mean', 'risk_mean')):
            ours = float(m[mk].iloc[0]) if len(m) else np.nan
            rows.append(dict(prm=r['prm'], proxy=nm, grid=g, quantity=k, ours=ours, reference=float(r[k]), diff=ours - float(r[k]),
                             status='PASS' if len(m) and abs(ours - float(r[k])) < 1e-9 else 'FAIL'))
    P = pd.DataFrame(rows); P.to_csv(H / 'CROP_CONTROLS_PARITY.csv', index=False)
    print('parity', P.status.value_counts().to_dict(), 'max |diff|', float(P['diff'].abs().max()))
    # ---- C1 main, C4, C5, C6, C7
    C1 = D[D.subset == 'C1']
    main_ = agg(C1[(C1.control == 'none') & (C1.alpha == .05)], ['prm', 'proxy', 'grid']); main_.to_csv(T / 'CROP_C1_MAIN.csv', index=False)
    ind = main_[main_.proxy.str.contains('LSML_CONT|TELE_N16|CHAIN_N1')].copy()
    ind['fit'] = np.where(ind.proxy.str.contains('inductive'), 'inductive (60% train traces, frozen)', 'transductive (all 6,969 answers)')
    ind['arm'] = ind.proxy.str.replace(' (inductive, C4)', '', regex=False)
    ind.to_csv(T / 'CROP_C4_INDUCTIVE.csv', index=False)
    agg(C1[C1.alpha == .05], ['prm', 'control', 'proxy', 'grid']).to_csv(T / 'CROP_C5_SHUFFLES.csv', index=False)
    C6 = agg(C1[C1.control == 'none'], ['prm', 'proxy', 'grid', 'alpha']); C6.to_csv(T / 'CROP_C6_ALPHA.csv', index=False)
    C6[C6.proxy.isin(['Direct PRM (1-reward)', 'Telemetry13+PRM LR (supervised)', 'CHAIN_N16', 'LSML_CONT', 'C3 Format/position+PRM LR (supervised)'])][
        ['prm', 'proxy', 'grid', 'alpha', 'bdev_mean', 'over_withhold_mean', 'overshoot_mean', 'kept_mean', 'risk_mean']].to_csv(T / 'CROP_C7_BOUNDARY.csv', index=False)
    # ---- figure: C6 kept vs alpha (quantile201 and fixed101_minmax)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    show = {'Direct PRM (1-reward)': '#2a78d6', 'Telemetry13+PRM LR (supervised)': '#6250d6', 'C3 Format/position+PRM LR (supervised)': '#8a8983',
            'CHAIN_N16': '#eb6834', 'LSML_CONT': '#1baf7a', 'CHAIN_N16 (inductive, C4)': '#eda100'}
    fig, axs = plt.subplots(2, 5, figsize=(18, 6.5), sharey=True)
    fd = C6[C6.proxy.isin(show)]
    for i, grid in enumerate(['quantile201', 'fixed101_minmax']):
        for j, p in enumerate(PRMS):
            ax = axs[i, j]
            for nm, c in show.items():
                s = fd[(fd.prm == p) & (fd.proxy == nm) & (fd.grid == grid)].sort_values('alpha')
                ax.plot(s.alpha, s.kept_mean, color=c, marker='o', ms=3, lw=2, ls='--' if 'inductive' in nm else '-', label=nm)
            ax.set_title(f'{p} ({grid})', fontsize=8, loc='left'); ax.grid(color='#e6e5e0'); ax.tick_params(labelsize=7)
            for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    axs[1, 0].set_xlabel('alpha', fontsize=8); axs[0, 0].set_ylabel('prefix kept (%)', fontsize=8); axs[1, 0].set_ylabel('prefix kept (%)', fontsize=8)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='lower center', ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(.5, -.04))
    fig.savefig(FIG / 'crop_controls_c6_kept_vs_alpha.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    fd.to_csv(FIG / 'data' / 'crop_controls_c6_kept_vs_alpha.csv', index=False)
    with open(FIG / 'CAPTIONS.md', 'a', encoding='utf-8') as fh:
        fh.write('\n### crop_controls_c6_kept_vs_alpha.png\nPRMBench, CROP Table 6 protocol on the exact Hub-order first-5,000 subset (4,995 scored traces; '
                 'C1), 10 seeds, test split. Prefix kept vs alpha on two grids (C2): train-quantile 201 and fixed {0,.01,..,1} after a '
                 'train min-max map. Every proxy\'s lambda is calibrated with labels (CROP access). Direct PRM: trained PRM. LRs: supervised. '
                 'CHAIN_N16 / LSML_CONT: label-free fusion (LSML_CONT channels label-selected); dashed = inductive fit on the 60% '
                 'training traces (C4).\nData: `figures/data/crop_controls_c6_kept_vs_alpha.csv`.\n')
    print('C1 rows', len(main_))


if __name__ == '__main__':
    main(sys.argv[1])
