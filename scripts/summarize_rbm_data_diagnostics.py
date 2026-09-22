"""Prespecified diagnostic contrasts, coverage, source-group uncertainty."""
import csv
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import csv_write, stat, base
from spectral_utils.rbm_data_diagnostics import STRATA, RESIDUAL_NAMES
OUT=ROOT/'results/rbm_data_diagnostics_v1'


def mean_intervals(values,groups):
    """Conditional descriptive 95% intervals, fixed saved models and four draws."""
    x=np.asarray(values,float);_,inv=np.unique(groups,return_inverse=True);ng=inv.max()+1
    num=np.zeros((ng,x.shape[1]));den=num.copy()
    for j in range(x.shape[1]):
        good=np.isfinite(x[:,j]);num[:,j]=np.bincount(inv[good],weights=x[good,j],minlength=ng)
        den[:,j]=np.bincount(inv[good],minlength=ng)
    samples=[];rng=np.random.default_rng(20260911259)
    for start in range(0,10000,128):
        w=rng.multinomial(ng,np.full(ng,1/ng),size=min(128,10000-start))
        d=w@den;n=w@num
        samples.append(np.divide(n,d,out=np.full_like(n,np.nan),where=d>0))
    return np.nanpercentile(np.concatenate(samples),[2.5,97.5],axis=0).T


def main():
    metrics=json.loads((OUT/'METRICS.json').read_text());diags=json.loads((OUT/'DIAGNOSTICS.json').read_text())
    rows=[];effect_arrays=[];descriptors=[];errors=list(csv.DictReader((OUT/'ERROR_CASES.csv').open()))
    for bank in (6,12):
        with np.load(OUT/f'BANK{bank}_DIAGNOSTICS.npz') as z:
            groups=z['group_id'];features=z['feature'];n=len(groups)
            def add(values,**info):
                effect_arrays.append(values);descriptors.append(dict(bank=bank,**info,**stat(values)))
            excess=z['resid']-z['null'].mean(axis=1)
            for j,name in enumerate(RESIDUAL_NAMES):add(excess[:,j],question='residual_model_excess',feature=name,stratum='all')
            for si,st in enumerate(STRATA):
                for j,name in enumerate(features[:-1]):
                    add(z['logvar'][:,si,j],question='class_log_variance_ratio',feature=str(name),stratum=st)
                delta=np.nanmean(z['lag'][:,si]-z['permuted'][:,si],axis=2)
                for lag in range(3):add(delta[:,lag],question='lag_excess',feature=f'lag{lag+1}',stratum=st)
            # Difference-of-differences: does feature-vs-fusion AUC advantage
            # change between regimes WITHIN the same answer? No feature choice.
            for lo,hi,label in [(1,2,'high_minus_low_entropy'),(3,4,'long_minus_short'),(5,6,'late_minus_early')]:
                rel=z['rel'];difference=(rel[:,hi,:-1]-rel[:,hi,-1,None])-(rel[:,lo,:-1]-rel[:,lo,-1,None])
                for j,name in enumerate(features[:-1]):add(difference[:,j],question='reliability_regime_interaction',feature=str(name),stratum=label)
            # Relate model misfit to failures within each benchmark cell. This
            # association is descriptive; it does not establish a causal fix.
            bm=[e for e in errors if int(e['bank'])==bank]
            lookup={str(uid):i for i,uid in enumerate(z['uid'])}
            for cell in sorted({e['cell'] for e in bm}):
                for mode in ('exact','early','late','gate_miss'):
                    ix=[lookup[e['uid']] for e in bm if e['cell']==cell and e['category']==mode]
                    rows.append(dict(bank=bank,question='residual_by_error_type',cell=cell,category=mode,
                                     feature='mean_abs_correlation_excess',**stat(excess[ix,0])))
            for relation in ('readout','learning'):
                for state in ('gained','lost'):
                    es=[e for e in bm if e[f'{state}_{relation}']=='True']
                    for mode in ('exact','early','late','gate_miss'):
                        rows.append(dict(bank=bank,question=f'{relation}_{state}',category=mode,n=sum(e['category']==mode for e in es),
                                         total=len(es)))
            for longest in ('True','False'):
                es=[e for e in bm if e['truth_longest']==longest]
                rows.append(dict(bank=bank,question='truth_is_longest',stratum=longest,n=len(es),
                    exact=sum(e['category']=='exact' for e in es),early=sum(e['category']=='early' for e in es),
                    late=sum(e['category']=='late' for e in es),gate_miss=sum(e['category']=='gate_miss' for e in es)))
    print('[diagnostic uncertainty] 10000 group draws;',len(descriptors),'descriptive contrasts',flush=True)
    # Chunk columns so the full matrix product remains moderate on one CPU.
    for start in range(0,len(descriptors),48):
        ci=mean_intervals(np.column_stack(effect_arrays[start:start+48]),groups)
        for d,interval in zip(descriptors[start:start+48],ci):
            rows.append(dict(**d,ci_low=interval[0],ci_high=interval[1],ci_level=.95,draws=10000))
    csv_write(OUT/'EVIDENCE.csv',rows)
    base.atomic_json(OUT/'EVIDENCE.json',dict(rows=rows,
        note='Descriptive source-group intervals, not multiplicity-adjusted discovery tests. '
             'Four synthetic replicates condition on saved parameters; parameter-estimation and earlier selection uncertainty excluded. '
             'Feature means vs top10 fusion: a diagnostic aggregation difference, not a controlled fusion ablation.'))
    lines=['# RBM saved-data diagnostics','',
           'Full cached development data. No model retraining. First-near-max is a research readout, not an independently confirmed improvement.','',
           '| Method | PB old | PB near | PRMB within old | PRMB within near | PRMScore near |',
           '|---|---:|---:|---:|---:|---:|']
    for name,label in metrics['names'].items():
        old=metrics['metrics'][name+'__old'];new=metrics['metrics'].get(name+'__near',old)
        lines.append(f"| {label} | {old['pb_all8']*100:.3f}% | {new['pb_all8']*100:.3f}% | {old['prm_within']:.5f} | {new['prm_within']:.5f} | {new['prmscore_q08']:.5f} |")
    lines+=['','Primary readout changes:']
    for bank in (6,12):
        c=metrics['contrasts'][f'rbm{bank}__near_minus_rbm{bank}__old']
        lines.append(f"- RBM{bank}: PB change {c['pb_delta']*100:.3f} percentage points; 97.5% CI {np.array(c['pb_ci'])*100}. Within-answer AUC change {c['prm_within_delta_common']:.5f}; CI {c['prm_within_ci']}.")
        s=diags['banks'][str(bank)]
        lines.append(f"  Readout gains/losses: {s['readout_gained']}/{s['readout_lost']}; learning gains/losses under near readout: {s['learning_gained']}/{s['learning_lost']}.")
    lines+=['','Evidence: EVIDENCE.csv/JSON; full per-answer arrays BANK6/12_DIAGNOSTICS.npz; ERROR_CASES.csv; original/new per-cell tables PB_CELLS.csv.',
            'No new architecture is authorized by a model mismatch alone. Consult the reviewed task-linked priority assessment before new training.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Mean of empty slice')
        warnings.filterwarnings('ignore',message='All-NaN slice encountered')
        main()
