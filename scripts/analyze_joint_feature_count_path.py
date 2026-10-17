"""Post-hoc full-population N curve from already fitted Joint deletion paths.

No refitting, no automatic method choice. Maxima are development-selected.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digitfree_broad50_v1 import load_data
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator


def main():
    out=ROOT/'results/joint_feature_selection_bocpd_v1';data=load_data();x=data['x']
    rowfold=np.repeat(data['folds'],np.diff(data['offsets']));scores=np.full((len(x),43),np.nan)
    counts=list(range(8,51))
    for f in range(5):
        d=json.loads((out/f'B50_fold{f}.json').read_text());by_count={len(s['active']):s for s in d['selection']['path']}
        assert all(n in by_count for n in counts)
        weights=np.column_stack([by_count[n]['weights'] for n in counts]);scores[rowfold==f]=x[rowfold==f]@weights
    rows=[]
    for j,n in enumerate(counts):
        m=score_locator(scores[:,j],data['gate'],data);rows.append(dict(retained=n,removed=50-n,pb=m['pb'],within=m['within']))
    report=dict(status='COMPLETE',scope='post-hoc descriptive; already fitted paths; full population',
        winner_is_development_selected=True,rows=rows,best_pb=max(rows,key=lambda r:r['pb']),
        best_within=max(rows,key=lambda r:r['within']),above40=[r['retained'] for r in rows if r['pb']>.4])
    dump(out/'COUNT_PATH_DIAGNOSTIC.json',report)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(counts,[100*r['pb'] for r in rows],marker='.',label='Joint deletion path')
    axes[0].axhline(39.831353,color='gray',linestyle='--',label='Historical innovation5')
    axes[0].axhline(40.367635,color='green',linestyle=':',label='Historical BOCPD correction')
    axes[0].set(ylabel='ProcessBench macro score (%)',xlabel='Retained features',title='Full benchmark: fixed feature counts')
    axes[1].plot(counts,[r['within'] for r in rows],marker='.')
    axes[1].axhline(.760292625,color='gray',linestyle='--')
    axes[1].axhline(.763222892,color='green',linestyle=':')
    axes[1].set(ylabel='PRMB within-answer AUROC',xlabel='Retained features',title='Same frozen folds and gate')
    axes[0].legend(fontsize=8);fig.savefig(out/'COUNT_PATH.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
