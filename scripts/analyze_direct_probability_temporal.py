"""Explain completed temporal results from saved predictions; no new fitting."""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils.direct_probability_temporal import METHODS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    args=p.parse_args();out=ROOT/'results/direct_probability_temporal_v3'
    result=json.loads((out/'METRICS.json').read_text())
    scores=np.load(out/'SCORES.npz',allow_pickle=False)
    bench=args.source_root/'results/localization_full_benchmark_v3/evaluation'
    records=json.loads((bench/'JOINED.json').read_text())['records'];joined=np.load(bench/'JOINED.npz')
    target,offsets=joined['target'],joined['offsets'];pb=np.array([r['cell'].startswith('pb_') for r in records])
    if result['n_answers']!=len(records):raise ValueError('incomplete results')
    base='current__iu';oldpred=scores['prediction__'+base];oldvalid=scores['valid__'+base]
    oldpeak=np.array([np.argmax(scores['steps__'+base][offsets[i]:offsets[i+1]]) if oldvalid[i] else -1 for i in range(len(records))])
    contrasts={}
    for m in METHODS:
        if m==base:continue
        valid=scores['valid__'+m];pred=scores['prediction__'+m]
        peak=np.array([np.argmax(scores['steps__'+m][offsets[i]:offsets[i+1]]) if valid[i] else -1 for i in range(len(records))])
        oldhit=pb & oldvalid & (oldpred==target);newhit=pb & valid & (pred==target)
        lost=np.flatnonzero(oldhit & ~newhit);gained=np.flatnonzero(newhit & ~oldhit)
        cases=[]
        for change,indices in [('lost',lost),('gained',gained)]:
            for i in indices:
                reason=('fit_failure' if not valid[i] else 'clean_decision' if pred[i]<0 else
                    'earlier_step' if peak[i]<target[i] else 'later_step' if peak[i]>target[i] else 'correct_location')
                cases.append(dict(uid=records[i]['uid'],cell=records[i]['cell'],change=change,
                    target=int(target[i]),baseline_peak=int(oldpeak[i]),new_peak=int(peak[i]),
                    baseline_prediction=int(oldpred[i]),new_prediction=int(pred[i]),reason=reason))
        contrasts[m]=dict(gained=int(len(gained)),lost=int(len(lost)),cases=cases,
            peak_shift_median_on_changed_errors=float(np.median((peak-oldpeak)[pb & (target>=0) & valid & oldvalid & (peak!=oldpeak)]))
                if np.any(pb & (target>=0) & valid & oldvalid & (peak!=oldpeak)) else None)
    con=sqlite3.connect(out/'CHECKPOINT.sqlite')
    lag_weights={m:[] for m in METHODS if m.startswith('lag8__')}
    negative={m:[] for m in lag_weights};counts={m:0 for m in lag_weights}
    for _,blob in con.execute('SELECT idx,payload FROM answers ORDER BY idx'):
        with np.load(io.BytesIO(blob)) as row:
            W=row['weights']
            for m in lag_weights:
                w=W[METHODS.index(m)]
                if not np.isfinite(w).all():continue
                total=np.abs(w).sum()
                if total==0:continue
                w=w.reshape(8,17)/total
                lag_weights[m].append(np.abs(w).sum(axis=1));negative[m].append(np.maximum(-w,0).sum(axis=1));counts[m]+=1
    weights={m:dict(valid_answers=counts[m],mean_abs_weight_share_by_lag=np.mean(v,axis=0).tolist() if v else None,
        mean_negative_weight_share_by_lag=np.mean(negative[m],axis=0).tolist() if v else None)
        for m,v in lag_weights.items()}
    payload=dict(scope='Descriptive full-development error and standardized-weight analysis. Not a causal attribution.',
        baseline=base,contrasts=contrasts,lag_weights=weights)
    (out/'ERROR_ANALYSIS.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for m,c in contrasts.items():print(m,'gained',c['gained'],'lost',c['lost'],'median changed-peak shift',c['peak_shift_median_on_changed_errors'])
    con.close()


if __name__=='__main__':main()
