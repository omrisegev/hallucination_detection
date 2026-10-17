"""Frozen full 2x2 score/readout study using saved six/twelve-feature RBMs."""
import argparse
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base,connect,csv_write,stat,METRIC_KEYS,NAMES
from spectral_utils.higher_moment_fusion import representation_order
from spectral_utils.direct_probability_fusion import zscore_columns
from spectral_utils.rbm_logit_readout import saved_scores

OUT=ROOT/'results/rbm_logit_readout_v1'
ROOTS=('rbm6','initial6','rbm12','initial12')
NEW=tuple(m+'__logit_'+r for m in ROOTS for r in ('old','near'))


def parent(source):return source/'.worktrees/rbm-data-diagnostics-v1/results/rbm_data_diagnostics_v1'
def model_dir(source):return source/'.worktrees/higher-moment-fusion-v1/results/higher_moment_fusion_v1'


def manifest_for(source):
    previous=json.loads((parent(source)/'MANIFEST.json').read_text())
    for path,digest in previous['hashes'].items():
        if base.old.sha256_file(Path(path))!=digest:raise ValueError('Source changed: '+path)
    for filename in ('RESULT_REVIEW.json','DIAGNOSTIC_REVIEW.json'):
        assert json.loads((parent(source)/filename).read_text())['status']=='PASS'
    assert json.loads((parent(source)/'RUN_STATE.json').read_text())['status']=='COMPLETE'
    paths=[Path(__file__),ROOT/'spectral_utils/rbm_logit_readout.py',
           ROOT/'scripts/test_rbm_logit_readout.py',ROOT/'scripts/verify_rbm_logit_readout.py',
           ROOT/'docs/experiments/RBM_LOGIT_READOUT_V1.md',
           ROOT/'scripts/run_direct_probability_temporal.py',ROOT/'scripts/run_direct_probability_fusion_v2.py',
           ROOT/'scripts/run_rbm_data_diagnostics.py',ROOT/'scripts/run_higher_moment_fusion.py',
           ROOT/'spectral_utils/higher_moment_fusion.py',ROOT/'spectral_utils/moment_rbm_fusion.py',
           ROOT/'spectral_utils/rbm_data_diagnostics.py',ROOT/'spectral_utils/direct_probability_fusion.py']
    paths += [parent(source)/n for n in ('MANIFEST.json','SCORES.npz','METRICS.json','RESULT_REVIEW.json','DIAGNOSTIC_REVIEW.json')]
    paths += [model_dir(source)/'CHECKPOINT.sqlite']
    hashes=previous['hashes'].copy()
    for path in paths:hashes[str(path)]=base.old.sha256_file(path)
    return dict(schema='rbm-logit-readout-v1',base_commit='7575cb237',hashes=hashes,
                new_methods=NEW,refits=0,bootstrap=10000)


def scan(source,con,records,joined,reference,smoke):
    done={r[0] for r in con.execute('SELECT idx FROM answers')};start=time.perf_counter()
    src=sqlite3.connect((model_dir(source)/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    detector,_=base.old._gate_contract(records)
    try:
        for cell,path,kind,dataset in base.source_specs():
            indices=[i for i,r in enumerate(records) if r['cell']==cell and i not in done]
            if not indices:continue
            print('[load]',cell,flush=True)
            rows=base.old._source_row_map(base.old.load_pickle(path),kind=kind,dataset=dataset)
            if smoke:
                ordered=sorted(indices,key=lambda i:len(rows[records[i]['row_id']]['token_entropies']))
                indices=[ordered[j] for j in sorted({0,len(ordered)//2,int(.95*(len(ordered)-1))})]
            for i in indices:
                r=records[i];row=rows[r['row_id']];uid=r['uid']
                lp=np.asarray(base.old._topk_payload(row)['logprobs'],float)
                entropy=np.asarray(row['token_entropies'],float);spans=np.asarray(row['step_token_spans'],int)
                assert lp.shape==(len(entropy),50) and spans.shape==(r['steps'],2)
                with np.load(base.old.BENCH/'scores'/f'{uid}.npz') as z:
                    np.testing.assert_array_equal(spans[:,0],z['step_starts']);np.testing.assert_array_equal(spans[:,1],z['step_ends'])
                if kind=='pb':np.testing.assert_allclose(entropy.mean(),detector[i],atol=1e-12,rtol=0)
                blob,info=src.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone();info=json.loads(info)
                assert info['uid']==uid
                sl=slice(joined['offsets'][i],joined['offsets'][i+1]);out={};telemetry={};failures={}
                with np.load(io.BytesIO(blob)) as state:
                    for degree,bank in ((3,6),(6,12)):
                        x=representation_order(lp,row['token_spilled_energies'],degree)
                        z,keep,mean,scale=zscore_columns(x)
                        for solver,name in [('rbm',f'rbm{bank}'),('rbm_initial',f'initial{bank}')]:
                            key=f'd{degree}__{solver}'
                            if key not in info['diagnostics']:
                                failures[name]=info['failures'].get(key,'missing saved state')
                                for readout in ('old','near'):out[name+'__logit_'+readout]=np.full(len(spans),np.nan)
                                continue
                            d=info['diagnostics'][key]
                            np.testing.assert_array_equal(np.flatnonzero(keep),d['columns'])
                            np.testing.assert_array_equal(mean,d['normalization_mean']);np.testing.assert_array_equal(scale,d['normalization_scale'])
                            scores,t=saved_scores(z,state[key+'::w'],state[key+'::b'],d['orientation'],spans)
                            np.testing.assert_array_equal(scores['posterior'],reference[name+'__old'][sl])
                            np.testing.assert_array_equal(scores['posterior_near'],reference[name+'__near'][sl])
                            out[name+'__logit_old']=scores['logit'];out[name+'__logit_near']=scores['logit_near']
                            telemetry[name]=t
                con.execute('INSERT INTO answers VALUES (?,?,?)',(i,base.packed(**out),base.dumps(dict(uid=uid,telemetry=telemetry,failures=failures))))
                done.add(i)
                if len(done)%100==0 or i==indices[-1]:
                    con.commit();base.atomic_json(OUT/('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'),dict(
                        status='SMOKE' if smoke else 'RUNNING',stage='score_replay',completed=len(done),expected=len(records),pid=os.getpid(),
                        elapsed_seconds=time.perf_counter()-start))
                    print('[checkpoint]',len(done),'/',len(records),round(time.perf_counter()-start,1),'s',flush=True)
            del rows
    finally:src.close()


def expressions():
    result={};primary=set()
    def add(name,weights,main=False):
        result[name]=weights
        if main:primary.add(name)
    for root in ROOTS:
        po,pn,lo,ln=root+'__old',root+'__near',root+'__logit_old',root+'__logit_near'
        if root.startswith('rbm'):add(root+'_primary',{ln:1,po:-1},True)
        add(root+'_logit_at_max',{lo:1,po:-1});add(root+'_logit_at_near',{ln:1,pn:-1})
        add(root+'_near_posterior',{pn:1,po:-1});add(root+'_near_logit',{ln:1,lo:-1})
        add(root+'_interaction',{ln:1,lo:-1,pn:-1,po:1})
    for bank in (6,12):
        for suffix in ('old','near','logit_old','logit_near'):
            add(f'bank{bank}_learning_{suffix}',{f'rbm{bank}__{suffix}':1,f'initial{bank}__{suffix}':-1})
        for readout in ('old','near'):
            for ref in ('var50','var15_iu'):
                add(f'rbm{bank}_logit_{readout}_vs_{ref}',{f'rbm{bank}__logit_{readout}':1,f'{ref}__{readout}':-1})
    return result,primary


def bootstrap(records,joined,metrics,per):
    expr,primary=expressions();names=sorted({m for e in expr.values() for m in e});keys=list(expr)
    _,inv=np.unique([r['group_id'] for r in records],return_inverse=True);ng=int(inv.max()+1)
    cells=np.array([r['cell'] for r in records]);target=joined['target']
    pb_cells=sorted({c for c in cells if c.startswith('pb_')})
    counts=np.zeros((ng,8,2));success=np.zeros((ng,len(names),8,2));coef=np.zeros((len(keys),len(names)))
    for k,key in enumerate(keys):
        for m,w in expr[key].items():coef[k,names.index(m)]=w
    for ci,cell in enumerate(pb_cells):
        for cl,mask in enumerate(((cells==cell)&(target<0),(cells==cell)&(target>=0))):
            counts[:,ci,cl]=np.bincount(inv[mask],minlength=ng)
            for j,m in enumerate(names):
                hit=mask&per[m]['decision_valid']&(per[m]['prediction']==target)
                success[:,j,ci,cl]=np.bincount(inv[hit],minlength=ng)
    wn=np.zeros((ng,len(keys)));wd=np.zeros_like(wn);result={}
    for j,key in enumerate(keys):
        e=expr[key];good=np.logical_and.reduce([np.isfinite(per[m]['within']) for m in e])
        d=sum(w*per[m]['within'][good] for m,w in e.items())
        wn[:,j]=np.bincount(inv[good],weights=d,minlength=ng);wd[:,j]=np.bincount(inv[good],minlength=ng)
        result[key]=dict(coefficients=e,primary=key in primary,common_prm_answers=int(good.sum()),
            pb_delta=sum(w*metrics[m]['pb_all8'] for m,w in e.items()),
            prm_within_delta_common=float(d.mean()) if len(d) else None)
    pd=[];ad=[];rng=np.random.default_rng(202609112701)
    for start in range(0,10000,128):
        n=min(128,10000-start);weights=rng.multinomial(ng,np.full(ng,1/ng),size=n).astype(float)
        den=(weights@counts.reshape(ng,-1)).reshape(n,8,2)
        num=(weights@success.reshape(ng,-1)).reshape(n,len(names),8,2)
        rates=np.divide(num,den[:,None],out=np.full_like(num,np.nan),where=den[:,None]>0)
        ca,ea=rates[:,:,:,0],rates[:,:,:,1]
        f=np.divide(2*ca*ea,ca+ea,out=np.zeros_like(ca),where=(ca+ea)>0)
        f[~np.isfinite(ca)|~np.isfinite(ea)]=np.nan
        pd.append(f.mean(axis=2)@coef.T)
        dena=weights@wd;ad.append(np.divide(weights@wn,dena,out=np.full_like(dena,np.nan),where=dena>0))
    pd,ad=np.concatenate(pd),np.concatenate(ad)
    for j,key in enumerate(keys):
        ci=.975 if key in primary else .95;q=[(1-ci)*50,100-(1-ci)*50]
        result[key].update(ci_level=ci,bootstrap_draws=10000,pb_ci=np.nanpercentile(pd[:,j],q).tolist(),
            prm_within_ci=np.nanpercentile(ad[:,j],q).tolist(),
            valid_pb_draws=int(np.isfinite(pd[:,j]).sum()),valid_prm_draws=int(np.isfinite(ad[:,j]).sum()))
    # Small draw matrix permits an independent check of the interaction arithmetic.
    np.savez_compressed(OUT/'BOOTSTRAP_DRAWS.npz',pb=pd,prm=ad,names=np.array(keys))
    return result


def evaluate(con,source,records,joined,reference):
    scores=reference.copy();telemetry={}
    for m in NEW:scores[m]=np.full(int(joined['offsets'][-1]),np.nan)
    for i,blob,info in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        sl=slice(joined['offsets'][i],joined['offsets'][i+1]);telemetry[i]=json.loads(info)
        with np.load(io.BytesIO(blob)) as z:
            for m in NEW:scores[m][sl]=z[m]
    assert len(scores)==32
    print('[evaluate] 32 arms',flush=True);metrics,per=base.evaluate_arrays(records,joined,scores)
    previous=json.loads((parent(source)/'METRICS.json').read_text())
    for m in reference:
        for key in METRIC_KEYS:np.testing.assert_allclose(metrics[m][key],previous['metrics'][m][key],atol=1e-12,rtol=0)
    print('[bootstrap] 10000 group draws; paired contrasts and interactions',flush=True)
    contrasts=bootstrap(records,joined,metrics,per)
    base.atomic_json(OUT/'METRICS.json',dict(n_answers=len(records),n_steps=int(joined['offsets'][-1]),metrics=metrics,contrasts=contrasts,
        scope='Cached full development; no refit; identical external entropy gate; q0.8 PRMScore thresholds recomputed on other folds.'))
    np.savez_compressed(OUT/'SCORES.npz',**{'steps__'+m:x for m,x in scores.items()},
        **{'prediction__'+m:p['prediction'] for m,p in per.items()},**{'valid__'+m:p['valid'] for m,p in per.items()})
    csv_write(OUT/'SUMMARY.csv',[dict(method=m,name=NAMES[m.split('__')[0]],**{k:x[k] for k in METRIC_KEYS},valid_answers=x['valid_answers']) for m,x in metrics.items()])
    csv_write(OUT/'PB_CELLS.csv',[dict(method=m,cell=c,**v) for m,x in metrics.items() for c,v in x['pb_cells'].items()])
    rows=[];summary={};target=joined['target']
    for root in ROOTS:
        rr=[]
        for i,r in enumerate(records):
            sl=slice(joined['offsets'][i],joined['offsets'][i+1]);t=int(target[i]);pb=r['cell'].startswith('pb_')
            row=dict(method=root,uid=r['uid'],group_id=r['group_id'],cell=r['cell'],target=t,**telemetry[i]['telemetry'].get(root,{}))
            for suffix in ('old','near','logit_old','logit_near'):
                m=root+'__'+suffix;s=scores[m][sl];p=per[m]
                row[suffix+'_peak']=int(p['peak'][i]);row[suffix+'_prediction']=int(p['prediction'][i]);row[suffix+'_valid']=bool(p['decision_valid'][i])
                row[suffix+'_peak_shift_from_original']=int(p['peak'][i]-per[root+'__old']['peak'][i]) if p['valid'][i] and per[root+'__old']['valid'][i] else None
                if suffix in ('old','logit_old') and np.isfinite(s).all():
                    row[suffix+'_near_fraction']=float(np.mean(s>=s.max()-.25*s.std()))
                    row[suffix+'_near_count']=int(np.sum(s>=s.max()-.25*s.std()))
                    row[suffix+'_exact_max_count']=int(np.sum(s==s.max()))
                row[suffix+'_suppressed']=bool(pb and t>=0 and p['valid'][i] and p['peak'][i]==t and p['prediction'][i]==-1)
            if pb and t>=0:
                po,pn,ln=(row[s+'_prediction'] for s in ('old','near','logit_near'))
                row['posterior_near_lost']=po==t and pn!=t
                row['rescued_by_logit_near']=row['posterior_near_lost'] and ln==t
                row['primary_gain']=ln==t and po!=t;row['primary_loss']=ln!=t and po==t
                row['primary_distance']=row['logit_near_peak']-t if row['logit_near_valid'] else None
            rr.append(row)
        err=[r for r in rr if r['cell'].startswith('pb_') and r['target']>=0]
        lost=[r for r in err if r['posterior_near_lost']];rescued=[r for r in lost if r['rescued_by_logit_near']]
        pbr=[r for r in rr if r['cell'].startswith('pb_')]
        summary[root]=dict(primary_gained=sum(r['primary_gain'] for r in err),primary_lost=sum(r['primary_loss'] for r in err),
            posterior_near_lost=len(lost),rescued_by_logit_near=len(rescued),
            near_fraction_posterior=stat([r.get('old_near_fraction') for r in pbr]),
            near_fraction_logit=stat([r.get('logit_old_near_fraction') for r in pbr]),
            rescued_with_smaller_near_set=sum(r['logit_old_near_count']<r['old_near_count'] for r in rescued),
            rescued_with_token_ties=sum(r.get('collapsed_token_pairs',0)>0 for r in rescued),
            primary_losses_early=sum(r['primary_loss'] and r['primary_distance'] is not None and r['primary_distance']<0 for r in err),
            primary_losses_late=sum(r['primary_loss'] and r['primary_distance'] is not None and r['primary_distance']>0 for r in err),
            primary_invalid=sum(not r['logit_near_valid'] for r in err))
        rows.extend(rr)
    csv_write(OUT/'ANSWER_DIAGNOSTICS.csv',rows);base.atomic_json(OUT/'FORENSICS.json',summary)
    base.atomic_json(OUT/'RUN_STATE.json',dict(status='SCORED_AWAITING_REVIEW',completed=len(records),expected=len(records),refits=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();source=a.source_root.resolve();base.old.configure_source_root(source);OUT.mkdir(parents=True,exist_ok=True)
    print('[manifest] verify frozen inputs',flush=True);manifest=manifest_for(source)
    con=connect(OUT/('SMOKE.sqlite' if a.smoke else 'CHECKPOINT.sqlite'),manifest)
    base.atomic_json(OUT/('SMOKE_MANIFEST.json' if a.smoke else 'MANIFEST.json'),manifest)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    with np.load(base.old.BENCH/'evaluation/JOINED.npz') as z:joined={k:z[k] for k in z.files}
    with np.load(parent(source)/'SCORES.npz') as z:reference={k[7:]:z[k] for k in z.files if k.startswith('steps__')}
    assert len(records)==13769 and len(reference)==24 and len({r['uid'] for r in records})==13769
    with threadpool_limits(limits=1):
        scan(source,con,records,joined,reference,a.smoke)
        n=con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
        if a.smoke:base.atomic_json(OUT/'SMOKE.json',dict(status='PASS',n_answers=n,scope='mechanics/runtime only'))
        elif n==len(records):evaluate(con,source,records,joined,reference)
    con.close()


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);base.atomic_json(OUT/'ERROR.json',dict(error=repr(e)));raise
