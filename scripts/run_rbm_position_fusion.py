"""Full fixed-contract evaluation of one conditional weight-correction candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_rbm_logit_readout as previous
from scripts.run_rbm_data_diagnostics import base,connect,csv_write,METRIC_KEYS,stat
from spectral_utils.higher_moment_fusion import representation_order,feature_names
from spectral_utils.direct_probability_fusion import zscore_columns,step_top_mean
from spectral_utils.rbm_position_fusion import contexts,fit_correction,MODES

OUT=ROOT/'results/rbm_position_fusion_v1_overlap_fix'
NEW=tuple(m+'__max' for m in MODES)


def parent(source):return source/'.worktrees/rbm-logit-readout-v1/results/rbm_logit_readout_v1'
def model_db(source):return source/'.worktrees/higher-moment-fusion-v1/results/higher_moment_fusion_v1/CHECKPOINT.sqlite'


def manifest_for(source):
    m=json.loads((parent(source)/'MANIFEST.json').read_text())
    for path,h in m['hashes'].items():
        if base.old.sha256_file(Path(path))!=h:raise ValueError('frozen input changed: '+path)
    assert json.loads((parent(source)/'RUN_STATE.json').read_text())['status']=='COMPLETE'
    for n in ('RESULT_REVIEW.json','REPLAY_REVIEW.json','CHOICE_REVIEW.json'):
        assert json.loads((parent(source)/n).read_text())['status']=='PASS'
    paths=[Path(__file__),ROOT/'spectral_utils/rbm_position_fusion.py',ROOT/'scripts/test_rbm_position_fusion.py',
        ROOT/'scripts/verify_rbm_position_fusion.py',ROOT/'docs/experiments/RBM_POSITION_FUSION_V1.md',
        ROOT/'scripts/run_rbm_logit_readout.py',ROOT/'scripts/run_direct_probability_temporal.py',
        ROOT/'scripts/run_direct_probability_fusion_v2.py',ROOT/'scripts/run_rbm_data_diagnostics.py',
        ROOT/'spectral_utils/higher_moment_fusion.py',ROOT/'spectral_utils/direct_probability_fusion.py']
    paths += [parent(source)/n for n in ('MANIFEST.json','METRICS.json','SCORES.npz','RUN_STATE.json')]
    for path in paths:m['hashes'][str(path)]=base.old.sha256_file(path)
    m.pop('refits',None)
    m.update(schema='rbm-position-fusion-v1',base_commit='a9cda144e',new_methods=NEW,new_fits_per_answer=3,
        fit_scope='one answer, no correctness labels',bank=12,readout='top10 mean then original argmax',
        ridge='.1 + active_P / min(early_tokens,late_tokens)',maxiter=100)
    return m


def worker(task):
    i,uid,X,spans,a,w,b,orientation,keep=task
    context,partition=contexts(len(X),spans,uid)
    ridge=.1+X.shape[1]/max(1,min(partition['early_tokens'],partition['late_tokens']))
    out={};diag={};failures={}
    for mode in MODES:
        try:
            score,delta,d=fit_correction(X,context[mode],a,w,b,ridge,orientation,single_step=partition['single_step'])
            out[mode+'__max']=step_top_mean(score,spans[:,0],spans[:,1],count=10)
            out[mode+'::delta']=delta;diag[mode]=d
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as e:
            out[mode+'__max']=np.full(len(spans),np.nan);failures[mode]=repr(e)
    out.update(a=a,w=w,b=np.asarray(b),columns=np.flatnonzero(keep))
    return i,base.packed(**out),base.dumps(dict(uid=uid,orientation=orientation,partition=partition,diagnostics=diag,failures=failures))


def scan(source,con,records,joined,refs,smoke,workers):
    done={r[0] for r in con.execute('SELECT idx FROM answers')};start=time.perf_counter()
    src=sqlite3.connect(model_db(source).as_uri()+'?mode=ro',uri=True);detector,_=base.old._gate_contract(records)
    try:
        with ProcessPoolExecutor(max_workers=workers,initializer=base.worker_init) as pool:
            for cell,path,kind,dataset in base.source_specs():
                indices=[i for i,r in enumerate(records) if r['cell']==cell and i not in done]
                if not indices:continue
                print('[load]',cell,flush=True);rows=base.old._source_row_map(base.old.load_pickle(path),kind=kind,dataset=dataset)
                if smoke:
                    order=sorted(indices,key=lambda i:len(rows[records[i]['row_id']]['token_entropies']))
                    selected=[order[j] for j in sorted({0,len(order)//2,int(.95*(len(order)-1))})]
                    for i in indices:
                        spans=np.asarray(rows[records[i]['row_id']]['step_token_spans'],int)
                        if np.any(spans[1:,0]<spans[:-1,1]):selected.append(i)
                    indices=sorted(set(selected))
                for offset in range(0,len(indices),8):
                    tasks=[]
                    for i in indices[offset:offset+8]:
                        r=records[i];row=rows[r['row_id']];spans=np.asarray(row['step_token_spans'],int)
                        lp=np.asarray(base.old._topk_payload(row)['logprobs'],float)
                        with np.load(base.old.BENCH/'scores'/f'{r["uid"]}.npz') as z:
                            np.testing.assert_array_equal(spans[:,0],z['step_starts']);np.testing.assert_array_equal(spans[:,1],z['step_ends'])
                        if kind=='pb':np.testing.assert_allclose(np.mean(row['token_entropies']),detector[i],atol=1e-12,rtol=0)
                        blob,info=src.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone();info=json.loads(info)
                        assert info['uid']==r['uid'];key='d6__rbm';d=info['diagnostics'][key]
                        X=representation_order(lp,row['token_spilled_energies'],6);Z,keep,mean,scale=zscore_columns(X)
                        np.testing.assert_array_equal(np.flatnonzero(keep),d['columns']);np.testing.assert_array_equal(mean,d['normalization_mean']);np.testing.assert_array_equal(scale,d['normalization_scale'])
                        with np.load(io.BytesIO(blob)) as state:a,w,b=state[key+'::a'],state[key+'::w'],float(state[key+'::b'])
                        replay=step_top_mean(d['orientation']*(b+Z@w),spans[:,0],spans[:,1],count=10)
                        sl=slice(joined['offsets'][i],joined['offsets'][i+1]);np.testing.assert_array_equal(replay,refs['rbm12__logit_old'][sl])
                        tasks.append((i,r['uid'],Z,spans,a,w,b,d['orientation'],keep))
                    for item in pool.map(worker,tasks,chunksize=1):con.execute('INSERT INTO answers VALUES (?,?,?)',item);done.add(item[0])
                    con.commit()
                    base.atomic_json(OUT/('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'),dict(status='RUNNING',completed=len(done),expected=len(records),elapsed_seconds=time.perf_counter()-start,pid=os.getpid(),workers=workers))
                    if len(done)%100<8 or offset+8>=len(indices):print('[checkpoint]',len(done),'/',len(records),round(time.perf_counter()-start,1),'s',flush=True)
                del rows
    finally:src.close()


def expressions():
    result={};primary={'position_vs_frozen','position_vs_shared'}
    for key,ref in [('frozen','rbm12__logit_old'),('shared','shared__max'),('permuted','permuted__max'),
                    ('iu','var15_iu__old'),('entropy','entropy__old'),('var15','var15__old'),('var50','var50__old'),('rbm6','rbm6__old')]:
        result['position_vs_'+key]={'position__max':1,ref:-1}
    result['shared_vs_frozen']={'shared__max':1,'rbm12__logit_old':-1}
    result['permuted_vs_frozen']={'permuted__max':1,'rbm12__logit_old':-1}
    return result,primary


def evaluate(source,con,records,joined,refs):
    scores=refs.copy();infos=[];weights={m:[] for m in MODES}
    for m in NEW:scores[m]=np.full(int(joined['offsets'][-1]),np.nan)
    for i,blob,info in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        info=json.loads(info);infos.append(info);sl=slice(joined['offsets'][i],joined['offsets'][i+1])
        with np.load(io.BytesIO(blob)) as z:
            for mode in MODES:
                scores[mode+'__max'][sl]=z[mode+'__max']
                if mode+'::delta' in z:
                    w=np.zeros((3,12));idx=z['columns'];ori=info['orientation'];w[0,idx]=ori*z['w']
                    w[1,idx]=ori*(z['w']+z[mode+'::delta'] if mode=='shared' else z['w']-z[mode+'::delta']);w[2,idx]=ori*(z['w']+z[mode+'::delta'])
                    weights[mode].append(w)
    print('[evaluate] 35 configurations; original max for every new arm',flush=True)
    metrics,per=base.evaluate_arrays(records,joined,scores)
    old=json.loads((parent(source)/'METRICS.json').read_text())['metrics']
    for m in refs:
        for k in METRIC_KEYS:np.testing.assert_allclose(metrics[m][k],old[m][k],atol=1e-12,rtol=0)
    print('[bootstrap] 10000 canonical group draws',flush=True)
    previous.OUT=OUT;previous.expressions=expressions;contrasts=previous.bootstrap(records,joined,metrics,per)
    base.atomic_json(OUT/'METRICS.json',dict(n_answers=len(records),n_steps=int(joined['offsets'][-1]),metrics=metrics,contrasts=contrasts))
    np.savez_compressed(OUT/'SCORES.npz',**{'steps__'+m:s for m,s in scores.items()},
        **{'prediction__'+m:p['prediction'] for m,p in per.items()},**{'valid__'+m:p['valid'] for m,p in per.items()})
    csv_write(OUT/'SUMMARY.csv',[dict(method=m,**{k:v[k] for k in METRIC_KEYS},valid_answers=v['valid_answers']) for m,v in metrics.items()])
    csv_write(OUT/'PB_CELLS.csv',[dict(method=m,cell=c,**v) for m,x in metrics.items() for c,v in x['pb_cells'].items()])
    health={};weight_rows=[]
    for mode in MODES:
        ds=[s['diagnostics'][mode] for s in infos if mode in s['diagnostics']]
        health[mode]=dict(valid=len(ds),failures=len(infos)-len(ds),nonconverged=sum(not d['converged'] for d in ds),
            single_step=sum(s['partition']['single_step'] for s in infos),
            **{k:stat([d[k] for d in ds]) for k in ('iterations','ridge','relative_delta_norm','gradient_max','seconds')},
            objective_improvement=stat([d['objective_initial']-d['objective_final'] for d in ds]))
        if weights[mode]:
            w=np.stack(weights[mode])
            for j,name in enumerate(feature_names(6)):
                weight_rows.append(dict(method=mode,feature=name,base_median=float(np.median(w[:,0,j])),
                    minus_median=float(np.median(w[:,1,j])),plus_median=float(np.median(w[:,2,j])),
                    median_absolute_delta=float(np.median(np.abs(w[:,2,j]-w[:,0,j])))))
    base.atomic_json(OUT/'FIT_HEALTH.json',health);csv_write(OUT/'WEIGHT_SUMMARY.csv',weight_rows)
    rows=[];forensics={};target=joined['target'];pb=np.array([r['cell'].startswith('pb_') for r in records]);err=pb&(target>=0)
    for name,coef in expressions()[0].items():
        pos=next(m for m,w in coef.items() if w==1);neg=next(m for m,w in coef.items() if w==-1)
        a,b=per[pos],per[neg];ah=a['decision_valid']&(a['prediction']==target);bh=b['decision_valid']&(b['prediction']==target)
        gained=err&ah&~bh;lost=err&~ah&bh
        forensics[name]=dict(gained=int(gained.sum()),lost=int(lost.sum()),
            lost_early=int(np.sum(lost&a['valid']&(a['peak']<target))),lost_late=int(np.sum(lost&a['valid']&(a['peak']>target))),
            lost_invalid=int(np.sum(lost&~a['valid'])),raw_peak_changed=int(np.sum(pb&(a['peak']!=b['peak']))))
    for i,r in enumerate(records):
        row=dict(uid=r['uid'],cell=r['cell'],group_id=r['group_id'],target=int(target[i]))
        for m in (*NEW,'rbm12__logit_old'):
            row[m+'_peak']=int(per[m]['peak'][i]);row[m+'_prediction']=int(per[m]['prediction'][i]);row[m+'_valid']=bool(per[m]['decision_valid'][i])
        rows.append(row)
    csv_write(OUT/'ANSWER_CHOICES.csv',rows);base.atomic_json(OUT/'FORENSICS.json',forensics)
    base.atomic_json(OUT/'RUN_STATE.json',dict(status='SCORED_AWAITING_REVIEW',completed=len(records),expected=len(records),new_configurations=3))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    source=args.source_root.resolve();base.old.configure_source_root(source);OUT.mkdir(parents=True,exist_ok=True)
    print('[manifest] verify frozen source artifacts',flush=True);manifest=manifest_for(source)
    con=connect(OUT/('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'),manifest)
    base.atomic_json(OUT/('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'),manifest)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    with np.load(base.old.BENCH/'evaluation/JOINED.npz') as z:joined={k:z[k] for k in z.files}
    with np.load(parent(source)/'SCORES.npz') as z:refs={k[7:]:z[k] for k in z.files if k.startswith('steps__')}
    assert len(records)==13769 and len(refs)==32
    with threadpool_limits(limits=1):
        scan(source,con,records,joined,refs,args.smoke,args.workers)
        if args.smoke:base.atomic_json(OUT/'SMOKE.json',dict(status='PASS',answers=con.execute('SELECT COUNT(*) FROM answers').fetchone()[0],scope='mechanics only, not comparative evidence'))
        else:
            assert con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]==len(records)
            evaluate(source,con,records,joined,refs)
    con.close()


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);base.atomic_json(OUT/'ERROR.json',dict(error=repr(e)));raise
