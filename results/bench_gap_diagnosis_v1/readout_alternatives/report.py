"""Figures and the summary table for the readout alternatives (reads T_bench.csv / T_cells.csv)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
G = pd.read_csv(HERE / 'T_bench.csv')
READS = ['top10', 'top10_lenres', 'top10_drop1', 'mean', 'top20pct', 'p90', 'median', 'frac_above']
BENCHES = ['prmbench', 'socratic', 'hard2verify', 'processbench']
COL = dict(top10='#4a4a4a', top10_lenres='#9a9a9a', top10_drop1='#7b6fb0', mean='#2a7fb8', top20pct='#5aa9d6',
           p90='#e08a2c', median='#2e9e6a', frac_above='#c4495a')


def get(b, sig, r=None):
    m = (G.bench == b) & (G.signal == sig)
    if r is not None: m &= G.readout == r
    return G[m].iloc[0]


def summary(sig='avg11'):
    rows = []
    for b in BENCHES:
        prm = get(b, 'PRM'); pz = get(b, 'PRM_answer_z')
        for r in READS:
            x = get(b, sig, r)
            rows.append(dict(bench=b, readout=r, within=x.within, within_minus_PRM=x.within - prm.within,
                             pooled=x.pooled, pooled_minus_PRM=x.pooled - prm.pooled, fe_strat=x.fe_strat,
                             fe_strat_minus_PRM=x.fe_strat - prm.fe_strat, within_nolast=x.within_nolast,
                             last_correct=x.last_level_correct, last_wrong=x.last_level_wrong,
                             pair_a=x.pair_a, pair_b=x.pair_b, pair_c=x.pair_c, pair_d=x.pair_d))
        rows.append(dict(bench=b, readout='PRM', within=prm.within, pooled=prm.pooled, fe_strat=prm.fe_strat,
                         within_nolast=prm.within_nolast, last_correct=pz.last_level_correct, last_wrong=pz.last_level_wrong,
                         pair_a=prm.pair_a, pair_b=prm.pair_b, pair_c=prm.pair_c, pair_d=prm.pair_d))
    return pd.DataFrame(rows)


def main():
    for sig in ('avg11', 'avg3'):
        summary(sig).to_csv(HERE / f'T_summary_{sig}.csv', index=False)
    S = summary('avg11')
    # fig1: gap to PRM per metric
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    for i, b in enumerate(BENCHES):
        s = S[(S.bench == b) & (S.readout != 'PRM')]
        mets = ['pooled_minus_PRM', 'fe_strat_minus_PRM'] + ([] if b == 'processbench' else ['within_minus_PRM'])
        w = .8 / len(READS)
        for k, r in enumerate(READS):
            v = s[s.readout == r][mets].values[0]
            ax[i].bar(np.arange(len(mets)) + k * w - .4 + w / 2, v, w, color=COL[r], label=r if i == 0 else None)
        ax[i].axhline(0, color='k', lw=.8); ax[i].set_xticks(range(len(mets)))
        ax[i].set_xticklabels([m.replace('_minus_PRM', '') for m in mets]); ax[i].set_title(b)
        ax[i].grid(axis='y', alpha=.3)
    ax[0].set_ylabel('11-channel average minus PRM\n(AUC, mean of 4 PRMs)')
    fig.legend(loc='lower center', ncol=8, frameon=False); fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(HERE / 'fig1_gap_to_prm_by_readout.png', dpi=130); plt.close(fig)
    # fig2: final step level
    fig, ax = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for i, b in enumerate(BENCHES):
        s = S[S.bench == b].set_index('readout').loc[READS + ['PRM']]
        x = np.arange(len(s))
        ax[i].bar(x - .2, s.last_correct, .4, color='#2e9e6a', label='final step correct')
        ax[i].bar(x + .2, s.last_wrong, .4, color='#c4495a', label='final step wrong')
        ax[i].axhline(0, color='k', lw=.8); ax[i].set_xticks(x); ax[i].set_xticklabels(s.index, rotation=60, ha='right', fontsize=8)
        ax[i].set_title(b); ax[i].grid(axis='y', alpha=.3)
    ax[0].set_ylabel('answer-z level of the final step (mean of 4 PRMs)')
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(HERE / 'fig2_final_step_level.png', dpi=130); plt.close(fig)
    # fig3: pair types
    fig, ax = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    lab = dict(pair_a='(a) 1st err vs\ncorrect before', pair_b='(b) 1st err vs\ncorrect after',
               pair_c='(c) later err vs\ncorrect after', pair_d='(d) later err vs\ncorrect before')
    for i, b in enumerate(BENCHES):
        s = S[S.bench == b].set_index('readout'); w = .8 / (len(READS) + 1)
        for k, r in enumerate(READS + ['PRM']):
            ax[i].bar(np.arange(4) + k * w - .4 + w / 2, s.loc[r, list(lab)].values.astype(float), w,
                      color=COL.get(r, 'k'), label=r if i == 0 else None, hatch='//' if r == 'PRM' else None,
                      edgecolor='white' if r == 'PRM' else None)
        ax[i].axhline(.5, color='k', lw=.8); ax[i].set_xticks(range(4)); ax[i].set_xticklabels(lab.values(), fontsize=7)
        ax[i].set_title(b); ax[i].set_ylim(.3, 1); ax[i].grid(axis='y', alpha=.3)
    ax[0].set_ylabel('P(error step scored above correct step)')
    fig.legend(loc='lower center', ncol=9, frameon=False); fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(HERE / 'fig3_pair_types.png', dpi=130); plt.close(fig)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(S.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
