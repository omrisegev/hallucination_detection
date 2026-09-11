"""Preserve and correct the old diagnostic's bootstrap and readout attribution."""
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_rbm_literature_completion as run


def main():
    source=ROOT.parents[1];out=run.PROGRAM/'diagnostic_correction';out.mkdir(parents=True,exist_ok=True)
    records,joined,reference=run.load_contract(source)
    diag=source/'.worktrees/rbm-data-diagnostics-v1/results/rbm_data_diagnostics_v1'
    paths=[Path(__file__),diag/'BANK6_DIAGNOSTICS.npz',diag/'BANK12_DIAGNOSTICS.npz',
           run.parent(source)/'SCORES.npz',run.base.old.BENCH/'evaluation/JOINED.json',
           run.base.old.BENCH/'evaluation/JOINED.npz',run.base.old.FOLDS,
           run.base.old.FIXED_GATE/'METRICS.json',run.base.old.FIXED_GATE/'DETECTORS.npz']
    manifest=dict(schema='literature-diagnostic-attribution-correction-v1',draws=10000,
        hash_by_path={str(p):run.base.old.sha256_file(p) for p in paths})
    run.base.atomic_json(out/'MANIFEST.json',manifest)
    lookup={r['uid']:i for i,r in enumerate(records)}
    groups,inv=np.unique([r['group_id'] for r in records],return_inverse=True);ng=len(groups)
    detector,threshold=run.base.old._gate_contract(records);target=joined['target']
    pb=np.array([r['cell'].startswith('pb_') for r in records]);offsets=joined['offsets']
    summaries=[];cases=[];allvalues=[];allmasks=[]
    for bank in (6,12):
        name=f'rbm{bank}__'+('old' if bank==6 else 'logit_old')
        flat=reference[name]
        peaks=np.array([np.argmax(flat[a:b]) for a,b in zip(offsets[:-1],offsets[1:])])
        pred=np.where(detector>=threshold,peaks,-1)
        with np.load(diag/f'BANK{bank}_DIAGNOSTICS.npz') as z:
            order=np.array([lookup[str(u)] for u in z['uid']])
            assert len(np.unique(order))==13769
            for j,i in enumerate(order):
                assert str(z['group_id'][j])==records[i]['group_id']
                assert str(z['cell'][j])==records[i]['cell']
            delta=z['lag'][:,0,0,:]-z['permuted'][:,0,0,:]
            good=np.isfinite(delta).any(axis=1)
            serial=np.full(13769,np.nan);serial[order[good]]=np.nanmean(delta[good],axis=1)
            off=~np.eye(bank,dtype=bool)
            corr=np.full(13769,np.nan);corr[order]=np.nanmean(np.abs(z['corr'][:,off]),axis=1)
        category=np.full(13769,'PRMB',dtype=object)
        for i in np.flatnonzero(pb):
            t=int(target[i]);p=int(pred[i])
            if t<0:cat='clean_correct' if p<0 else 'false_alarm'
            elif p<0:cat='gate_miss'
            elif p==t:cat='exact'
            else:cat='early' if p<t else 'late'
            category[i]=cat
            cases.append(dict(bank=bank,method=name,uid=records[i]['uid'],group_id=records[i]['group_id'],
                cell=records[i]['cell'],target=t,prediction=p,peak=int(peaks[i]),category=cat,
                serial_excess=float(serial[i]) if np.isfinite(serial[i]) else None,
                residual_abs_corr=float(corr[i]) if np.isfinite(corr[i]) else None,
                correct_peak_suppressed=bool(t>=0 and peaks[i]==t and p<0)))
        masks={'all_answers':np.ones(13769,bool),'PB_all':pb,
               **{c:category==c for c in ('exact','early','late','gate_miss','clean_correct','false_alarm')}}
        for metric,values in [('lag1_excess',serial),('residual_abs_corr',corr)]:
            for group,mask in masks.items():
                valid=mask&np.isfinite(values)
                summaries.append(dict(bank=bank,method=name,metric=metric,category=group,n_rows=int(mask.sum()),
                    n_valid=int(valid.sum()),n_groups=int(len(set(inv[valid]))),
                    mean=float(values[valid].mean()) if valid.any() else None))
                allvalues.append(values);allmasks.append(valid)
    numerator=np.column_stack([np.bincount(inv[m],weights=v[m],minlength=ng) for v,m in zip(allvalues,allmasks)])
    denominator=np.column_stack([np.bincount(inv[m],minlength=ng) for m in allmasks])
    draws=[];rng=np.random.default_rng(2026091201)
    with threadpool_limits(limits=1):
        for begin in range(0,10000,128):
            n=min(128,10000-begin);w=rng.multinomial(ng,np.full(ng,1/ng),size=n)
            den=w@denominator;num=w@numerator
            draws.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0))
    draws=np.concatenate(draws)
    for j,row in enumerate(summaries):
        row.update(ci95=np.nanquantile(draws[:,j],[.025,.975]).tolist(),
                   valid_draws=int(np.isfinite(draws[:,j]).sum()),bootstrap='canonical-source-group')
    corrections=[
        'Original report sampled answers independently; source-group dependence is now retained.',
        'Original PB case categories were posterior/first_near_max, not the retained current baselines.',
        'Current categories use bank6 posterior/max and bank12 logit/max, with the unchanged gate.',
        'Category n_rows and n_valid_serial are now separate; NaN serial statistics are not counted as observations.',
        'This reanalysis does not fit a detector, test exact boundary-token adjacency, or establish task improvement.',
        'Global c-STG/family-router findings concern their original targets; they do not close local RBM graph/temporal models.',
        'Existing position-conditioned RBM was already tested and negative; a repeated marginal reliability reversal is not new joint predictive evidence.'
    ]
    run.base.atomic_json(out/'CORRECTED_DIAGNOSTICS.json',dict(status='COMPLETE',n_answers=13769,
        banks=[6,12],draws=10000,n_source_groups=ng,summaries=summaries,corrections=corrections,
        scope='Retrospective descriptive attribution; no new benchmark scores or model fit.'))
    run.csv_write(out/'SUMMARY.csv',summaries);run.csv_write(out/'PB_CASES.csv',cases)
    run.base.atomic_json(out/'REVIEW.json',dict(status='PASS',uid_group_cell_matches=27538,
        pb_cases=len(cases),cases_per_bank=6800,scope='Identity assertions, retained baseline attribution, explicit coverage, grouped resampling.'))
    print(json.dumps([r for r in summaries if r['category']=='all_answers'],indent=2))


if __name__=='__main__':main()
