"""Post-fit task diagnostics; labels never feed back to the fitted model."""
import argparse
import csv
import io
import json
from pathlib import Path
import sqlite3
import numpy as np


def summary(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)) if len(x) else None,mean=float(np.mean(x)) if len(x) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);args=p.parse_args()
    out=Path(__file__).resolve().parents[1]/'results/rbm_position_fusion_v1_overlap_fix'
    records=json.loads((args.source_root/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    by_uid={r['uid']:r for r in records}
    con=sqlite3.connect((out/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    info={json.loads(s)['uid']:json.loads(s) for s, in con.execute('SELECT info FROM answers')};con.close()
    buckets={};earlylate={};changes=dict(same_half=0,early_to_late=0,late_to_early=0)
    with (out/'ANSWER_CHOICES.csv').open(newline='',encoding='utf8') as f:
        for row in csv.DictReader(f):
            if not row['cell'].startswith('pb_') or int(row['target'])<0:continue
            uid=row['uid'];truth=int(row['target']);split=(by_uid[uid]['steps']+1)//2
            new=int(row['position__max_prediction']);old=int(row['rbm12__logit_old_prediction'])
            valid=row['position__max_valid']=='True';old_valid=row['rbm12__logit_old_valid']=='True'
            hit=valid and new==truth;old_hit=old_valid and old==truth
            category='gained' if hit and not old_hit else 'lost' if old_hit and not hit else 'both_hit' if hit else 'both_miss'
            b=buckets.setdefault(category,dict(n=0,objective_improvement=[],relative_delta_norm=[],same_half=0,early_to_late=0,late_to_early=0,invalid=0))
            b['n']+=1;d=info[uid]['diagnostics'].get('position')
            if d:
                b['objective_improvement'].append(d['objective_initial']-d['objective_final']);b['relative_delta_norm'].append(d['relative_delta_norm'])
            apeak,bpeak=int(row['position__max_peak']),int(row['rbm12__logit_old_peak'])
            if not valid:b['invalid']+=1
            elif apeak!=bpeak:
                kind='same_half' if (apeak<split)==(bpeak<split) else 'early_to_late' if bpeak<split else 'late_to_early'
                b[kind]+=1;changes[kind]+=1
            region='early' if truth<split else 'late';g=earlylate.setdefault(region,dict(n=0,original_hits=0,conditional_hits=0,gained=0,lost=0))
            g['n']+=1;g['original_hits']+=int(old_hit);g['conditional_hits']+=int(hit);g['gained']+=int(category=='gained');g['lost']+=int(category=='lost')
    for b in buckets.values():
        for k in ('objective_improvement','relative_delta_norm'):b[k]=summary(b[k])
    result=dict(scope='Post-evaluation descriptive diagnostics on PB erroneous answers; no method selection or refit.',
        versus_frozen=buckets,true_error_half=earlylate,changed_peaks=changes)
    (out/'TASK_DIAGNOSTICS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    with np.load(args.source_root/'results/localization_full_benchmark_v3/evaluation/JOINED.npz') as z:
        labels,offsets=z['labels'],z['offsets']
    con=sqlite3.connect((out/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True);values=[]
    for i,r in enumerate(records):
        if r['cell'].startswith('pb_'):continue
        lab=labels[offsets[i]:offsets[i+1]];split=(len(lab)+1)//2
        if not all(np.any(y==0) and np.any(y==1) for y in (lab[:split],lab[split:])):continue
        blob,s=con.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone();d=json.loads(s)
        if 'position' in d['failures']:continue
        with np.load(io.BytesIO(blob)) as z:
            cols=z['columns'];delta=z['position::delta'];w=z['w'];ori=d['orientation']
            if 0 not in cols or 3 not in cols:continue
            h=int(np.flatnonzero(cols==0)[0]);a=int(np.flatnonzero(cols==3)[0])
            values.append(dict(relative_coefficient_change=float(2*ori*(delta[a]-delta[h])),
                selected_early=float(ori*(w[a]-delta[a])),selected_late=float(ori*(w[a]+delta[a])),
                entropy_early=float(ori*(w[h]-delta[h])),entropy_late=float(ori*(w[h]+delta[h]))))
    con.close()
    weight_report=dict(scope='Descriptive coefficients on the same PRMB eligibility rule as the prior matched early/late AUC diagnostic; coefficients with correlated powers are not standalone reliability estimates.',
        answers=len(values),median_relative_coefficient_change=float(np.median([v['relative_coefficient_change'] for v in values])),
        fraction_relative_selected_weight_increases_late=float(np.mean([v['relative_coefficient_change']>0 for v in values])),
        medians={k:float(np.median([v[k] for v in values])) for k in ('selected_early','selected_late','entropy_early','entropy_late')})
    (out/'MATCHED_WEIGHT_DIAGNOSTIC.json').write_text(json.dumps(weight_report,indent=2)+'\n')


if __name__=='__main__':main()
