"""Ablation charts: one PNG per benchmark, 4 small multiples (one per PRM pairing), each a dot + 95% CI per arm,
arms in ablation order. Color = arm family (no PRM / no PRM + length fix / uses the PRM). Reads BOOT.csv."""
import csv, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'charts'; OUT.mkdir(exist_ok=True)
PRMS = [('qwen_prm_7b', 'Qwen-PRM-7B'), ('qwen_prm_72b', 'Qwen-PRM-72B'), ('pure_prm', 'PURE'), ('rlhflow_prm', 'RLHFlow')]
NOPRM, FIX, PRM = '#2a78d6', '#1baf7a', '#8a8984'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
ARMS = {  # label, family
    'GSP-T own count': ('A  GSP no PRM, own count', NOPRM),
    'GSP-T + GAP': ('B  GSP no PRM, GAP', NOPRM),
    'GSP-T hyb10 own count': ('D  A + length fix (<10 tok)', FIX),
    'GSP-T hyb10 + GAP': ('D  B + length fix (<10 tok)', FIX),
    'GSP-T hyb30 own count': ('D\'  A + length fix (<30 tok)', FIX),
    'GSP-T hyb30 + GAP': ('D\'  B + length fix (<30 tok)', FIX),
    'CP GSP-T rank (free EM)': ('E  GSP no PRM, change-point (EM)', NOPRM),
    'CP GSP-T rank (free, pi = 1/2)': ('E  GSP no PRM, change-point (pi=1/2)', NOPRM),
    'CP GSP-T hyb10 rank (free, pi = 1/2)': ('E  + length fix, change-point', FIX),
    'GSP full (PRM + tele) own count': ('C  GSP with PRM, own count', PRM),
    'GSP full (PRM + tele) + GAP': ('C  GSP with PRM, GAP', PRM),
    'CP PRM + GSP full rank (anchored)': ('E  GSP with PRM, change-point', PRM),
    'PRM native': ('C  PRM alone, own threshold', PRM),
    'PRM + GAP': ('C  PRM alone, GAP', PRM),
    'CP PRM (anchored)': ('E  PRM alone, change-point', PRM),
}
TITLE = {'prmbench': ('PRMBench', 'official PRMScore'), 'socratic': ('Socratic-PRMBench', 'official PRMScore'),
         'hard2verify': ('Hard2Verify', 'official balanced F1'), 'processbench': ('ProcessBench', 'official average F1')}


def main():
    B = defaultdict(dict)
    for r in csv.DictReader(open(HERE / 'BOOT.csv')):
        if r['kind'] == 'abs':
            B[(r['bench'], r['prm'])][r['arm']] = tuple(float(r[k]) for k in ('value', 'lo', 'hi'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
    for bench, (name, metric) in TITLE.items():
        arms = [a for a in ARMS if any(a in B[(bench, p)] for p, _ in PRMS)]
        if not arms:
            continue
        fig, axs = plt.subplots(1, 4, figsize=(13, 0.42 * len(arms) + 1.6), sharey=True)
        for ax, (p, pl) in zip(axs, PRMS):
            for i, a in enumerate(arms):
                if a not in B[(bench, p)]:
                    continue
                v, lo, hi = B[(bench, p)][a]; col = ARMS[a][1]
                ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle='round')
                ax.plot(v, i, 'o', ms=7, color=col, mec=SURF, mew=2)
                ax.text(hi + .6, i, f'{v:.1f}', va='center', fontsize=8, color=INK)
            ax.set_title(pl, fontsize=10, color=INK, loc='left')
            ax.grid(axis='x', color=GRID, lw=.8); ax.set_axisbelow(True)
            for s in ('top', 'right', 'left'):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis='y', length=0)
            lo_all = min(B[(bench, p)][a][1] for a in arms if a in B[(bench, p)])
            hi_all = max(B[(bench, p)][a][2] for a in arms if a in B[(bench, p)])
            ax.set_xlim(max(0, lo_all - 3), hi_all + 6)
            ax.set_xlabel(metric)
        axs[0].set_yticks(range(len(arms))); axs[0].set_yticklabels([ARMS[a][0] for a in arms])
        axs[0].invert_yaxis()
        fig.suptitle(f'{name}: GSP ablation on base-model telemetry (dot = value, line = 95% bootstrap CI)',
                     x=.01, ha='left', fontsize=11, color=INK)
        h = [plt.Line2D([], [], marker='o', ls='', color=c, label=l) for c, l in
             ((NOPRM, 'no PRM'), (FIX, 'no PRM + step-length fix'), (PRM, 'uses the PRM (reference)'))]
        fig.legend(handles=h, loc='upper right', ncol=3, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(OUT / f'{bench}.png', dpi=160)
        plt.close(fig)
        print('wrote', bench)


if __name__ == '__main__':
    main()
