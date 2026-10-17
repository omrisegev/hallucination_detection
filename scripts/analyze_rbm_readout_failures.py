"""Descriptive failure inspection after the frozen readout result; no new arm."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base, csv_write, stat, NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    base.old.configure_source_root(a.source_root)
    out=ROOT/'results/rbm_data_diagnostics_v1'
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    joined=np.load(base.old.BENCH/'evaluation/JOINED.npz');scores=np.load(out/'SCORES.npz')
    target=joined['target'];details=[];summary={}
    for m in NAMES:
        if m in ('length','random'):continue
        old=scores['steps__'+m+'__old'];new=scores['steps__'+m+'__near']
        op=scores['prediction__'+m+'__old'];npred=scores['prediction__'+m+'__near']
        vs=scores['valid__'+m+'__near'];row=[]
        for i,r in enumerate(records):
            lo,hi=joined['offsets'][i:i+2];s=old[lo:hi];sn=new[lo:hi]
            if not np.isfinite(s).all():continue
            near=s>=s.max()-.25*s.std();peak=int(np.argmax(s));npeak=int(np.argmax(sn));t=int(target[i])
            pb=r['cell'].startswith('pb_');err=pb and t>=0
            gained=err and npred[i]==t and op[i]!=t
            lost=err and op[i]==t and npred[i]!=t
            item=dict(method=m,uid=r['uid'],cell=r['cell'],steps=len(s),near_count=int(near.sum()),
                near_fraction=float(near.mean()),old_peak=peak,new_peak=npeak,peak_shift=npeak-peak,
                target=t,gained=bool(gained),lost=bool(lost),step_sd=float(s.std()),
                minimum=float(s.min()),maximum=float(s.max()),exact_one_steps=int((s==1).sum()),
                pb=pb,error=err,old_prediction=int(op[i]),new_prediction=int(npred[i]))
            row.append(item)
            if gained or lost:details.append(item)
        summary[m]=dict(n=len(row),
            pb_changed=sum(r['pb'] and r['old_peak']!=r['new_peak'] for r in row),
            prm_changed=sum(not r['pb'] and r['old_peak']!=r['new_peak'] for r in row),
            gained=sum(r['gained'] for r in row),lost=sum(r['lost'] for r in row),
            lost_earlier=sum(r['lost'] and r['new_peak']<r['target'] for r in row),
            lost_later=sum(r['lost'] and r['new_peak']>r['target'] for r in row),
            gate_changed=sum((r['old_prediction']==-1)!=(r['new_prediction']==-1) for r in row if r['pb']),
            near_fraction=stat([r['near_fraction'] for r in row]),
            near_fraction_pb=stat([r['near_fraction'] for r in row if r['pb']]),
            near_fraction_lost=stat([r['near_fraction'] for r in row if r['lost']]),
            near_count_pb=stat([r['near_count'] for r in row if r['pb']]),
            lost_shift=stat([r['peak_shift'] for r in row if r['lost']]),
            answers_with_exact_one=sum(r['exact_one_steps']>0 for r in row),
            lost_with_exact_one=sum(r['lost'] and r['exact_one_steps']>0 for r in row))
    base.atomic_json(out/'READOUT_FORENSICS.json',dict(methods=summary,
        scope='Post-result descriptive diagnosis, not a new tuned readout or causal proof. Exact-one counts report numerical saturation without selecting a new threshold.'))
    csv_write(out/'READOUT_CHANGED_SUCCESSES.csv',details)
    for m in ('rbm6','rbm12','initial6','initial12','entropy','var15','var50'):
        s=summary[m];print(m,'gained/lost',s['gained'],s['lost'],'near fraction PB',s['near_fraction_pb']['mean'],
                           'lost earlier',s['lost_earlier'],'gate changed',s['gate_changed'])


if __name__=='__main__':main()
