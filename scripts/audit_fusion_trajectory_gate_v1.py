"""Post-evaluation component exchanges; diagnostic, not candidate selection."""
from collections import defaultdict
import copy
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ROOT/'scripts/run_fusion_trajectory_imm_v1.py','gate_exchange_driver')
metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','gate_exchange_metrics')
OUT=d.OUT


def exchange(rows,peak_arm,gate_arm):
    """Keep each named component's actual output; never choose using targets."""
    copied=copy.deepcopy(rows);cells=defaultdict(lambda:dict(clean=0,error=0,clean_correct=0,error_exact=0,valid=0))
    predictions=[]
    for row in copied:
        valid=bool(row['valid'][peak_arm] and row['decision_valid'][gate_arm])
        prediction=(row['peaks'][peak_arm] if row['predictions'][gate_arm]!=-1 else -1) if valid else None
        row['decision_valid']['exchange']=valid;row['predictions']['exchange']=prediction
        clean=row['target']==-1;c=cells[row['cell']];c['clean' if clean else 'error']+=1;c['valid']+=valid
        c['clean_correct' if clean else 'error_exact']+=bool(valid and prediction==row['target'])
        predictions.append(dict(uid=row['uid'],valid=valid,prediction=prediction))
    values=[]
    for c in cells.values():
        a=c['clean_correct']/c['clean'];b=c['error_exact']/c['error'];c['f1']=0. if a+b==0 else 2*a*b/(a+b);values.append(c['f1'])
    point=float(np.mean(values));independent=metrics.pb(copied,'exchange')['macro_f1']
    np.testing.assert_allclose(point,independent,atol=1e-14,rtol=0)
    return dict(peak=peak_arm,gate=gate_arm,pb=point,clean_correct=sum(c['clean_correct'] for c in cells.values()),
                error_exact=sum(c['error_exact'] for c in cells.values()),cells=dict(cells),predictions=predictions)


def main():
    manifest=d.verify();review=d.load(OUT/'REVIEW.json');evaluation=d.load(OUT/'EVALUATION.json')
    assert review['status']=='PASS'
    for p,h in review['hashes'].items():assert d.sha(p)==h,p
    rows=[r for r in evaluation['rows'] if r['cell'].startswith('pb')];assert len(rows)==86
    families=list(d.PAIRS)+list(d.SINGLES);exchanges={};checks=0;records=[]
    for family in families:
        hold=f'traj_{family}__hold';imm=f'traj_{family}__imm';combos=[]
        for peak in (hold,imm):
            for gate in (hold,imm):
                result=exchange(rows,peak,gate);combos.append(result);checks+=1
                if peak==gate:
                    np.testing.assert_allclose(result['pb'],evaluation['metrics'][peak]['pb']['macro_f1'],atol=1e-14,rtol=0)
                    assert all(v['prediction']==r['predictions'][peak] for v,r in zip(result['predictions'],rows));checks+=1
        exchanges[family]=combos
    for row in rows:
        meta=d.load(OUT/'scores'/(row['uid']+'.json'))
        with np.load(OUT/'scores'/(row['uid']+'.npz'),allow_pickle=False) as a:
            fi=a['fit_indices']
            for family in families:
                for readout in ('hold','imm'):
                    arm=f'traj_{family}__{readout}';x=a[arm+'__window'][fi];before=x[:-1];after=x[1:]
                    centered0=before-before.mean();centered1=after-after.mean();den=np.linalg.norm(centered0)*np.linalg.norm(centered1)
                    rho=None if den<=1e-14 else float(np.clip(centered0@centered1/den,-1,1))
                    if rho is not None:np.testing.assert_allclose(rho,np.corrcoef(before,after)[0,1],atol=1e-12,rtol=1e-12)
                    method=meta['methods'][arm];bic=method['gate']['bic']
                    record=dict(uid=row['uid'],cell=row['cell'],family=family,readout=readout,arm=arm,clean=row['target']==-1,
                        windows=len(fi),lag1=rho,bic_advantage_two=float(bic[0]-bic[1]),gate_open=row['predictions'][arm]!=-1,
                        peak=row['peaks'][arm],target=row['target'],prediction=row['predictions'][arm],
                        peak_step_offset=None if row['target']==-1 else row['peaks'][arm]-row['target'])
                    records.append(record)
    summaries=[]
    for family in families:
        for readout in ('hold','imm'):
            for clean in (True,False):
                chosen=[r for r in records if r['family']==family and r['readout']==readout and r['clean']==clean]
                rho=[r['lag1'] for r in chosen if r['lag1'] is not None]
                summaries.append(dict(family=family,readout=readout,clean=clean,answers=len(chosen),
                    median_lag1=float(np.median(rho)) if rho else None,
                    median_bic_advantage_two=float(np.median([r['bic_advantage_two'] for r in chosen])),
                    gate_open=sum(r['gate_open'] for r in chosen)))
    result=dict(status='POST_EVALUATION_DIAGNOSTIC_REVIEWED',exchanges=exchanges,records=records,summaries=summaries,
        checks=checks,lag_replays=len(records),
        interpretation='All 2x2 peak/gate exchanges are retrospective diagnostics, not new registered candidates, oracle selections or causal mediation estimates. Direct per-cell harmonic metrics match the independent benchmark helper. Lag1 correlations are descriptive; no iid BIC correction or semantic-error null is established.',
        hashes={str(OUT/n):d.sha(OUT/n) for n in ('EVALUATION.json','REVIEW.json','SCORES_FROZEN.json')},
        dependencies={str(p):d.sha(p) for p in [Path(__file__),Path(metrics.__file__)]})
    d.save(OUT/'GATE_AUDIT.json',result)
    for x in exchanges['iu_joint_graph']:print(x['peak'].split('__')[-1],x['gate'].split('__')[-1],x['clean_correct'],x['error_exact'],x['pb'])
    for x in summaries:
        if x['family']=='iu_joint_graph':print(x)
    print('Gate exchange audit PASS',checks,'metric/component checks;',len(records),'lag replays.',flush=True)


if __name__=='__main__':main()
