"""Post-evaluation held-out loss diagnostic, no fitting or candidate selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base,csv_write


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);args=p.parse_args();source=args.source_root
    out=ROOT/'results/rbm_supervision_matched_v1';bench=source/'results/localization_full_benchmark_v3/evaluation'
    records=json.loads((bench/'JOINED.json').read_text())['records'];j=np.load(bench/'JOINED.npz');off=j['offsets']
    fmap=json.loads((source/'results/localization_source_group_audit_v1/FOLDS_V2.json').read_text())['outer'];folds=np.array([fmap[r['group_id']] for r in records])
    cells=np.array([r['cell'] for r in records]);labels=j['labels'].copy()
    for i,r in enumerate(records):
        if not r['cell'].startswith('pb_'):continue
        y=labels[off[i]:off[i+1]];y[:]=0;t=int(j['target'][i])
        if t>=0:y[t]=1;y[t+1:]=-1
    with np.load(out/'SCORES.npz') as z:scores={m:z['steps__'+m] for m in ('unsupervised_update','supervised_update')}
    rows=[]
    for cell in sorted(set(cells)):
        for fold in sorted(set(folds)):
            answer=np.flatnonzero((cells==cell)&(folds==fold));idx=[];mass=[]
            for i in answer:
                selected=np.arange(off[i],off[i+1]);selected=selected[labels[selected]>=0]
                idx.extend(selected);mass.extend(np.full(len(selected),1/len(selected)))
            idx=np.array(idx);mass=np.array(mass);y=labels[idx]
            for k in (0,1):mass[y==k]*=.5/mass[y==k].sum()
            result=dict(cell=cell,fold=int(fold),answers=len(answer),known_steps=len(idx))
            for m,s in scores.items():result[m]=float(np.sum(mass*np.where(y==1,np.logaddexp(0,-s[idx]),np.logaddexp(0,s[idx]))))
            result['delta']=result['supervised_update']-result['unsupervised_update'];rows.append(result)
    summary={}
    for task in ('pb','prm'):
        subset=[r for r in rows if r['cell'].startswith(task)]
        summary[task]=dict(folds=len(subset),improved=sum(r['delta']<0 for r in subset),
            mean_unsupervised=float(np.mean([r['unsupervised_update'] for r in subset])),mean_supervised=float(np.mean([r['supervised_update'] for r in subset])))
    csv_write(out/'HELDOUT_LOSS.csv',rows)
    base.atomic_json(out/'HELDOUT_LOSS.json',dict(scope='Post-evaluation descriptive diagnostic. Class-balanced BCE on held-out known step labels, answer-balanced before class balancing. No refit or tuning.',
        constant_probability_half_bce=float(np.log(2)),
        caution='Even updated BCE exceeds the .5 constant-probability loss. Reduction versus the latent-RBM proxy is not proof of good calibration. A shared additive correction cannot cancel arbitrary answer-specific base coefficients.',summary=summary))
    print(json.dumps(summary))


if __name__=='__main__':main()
