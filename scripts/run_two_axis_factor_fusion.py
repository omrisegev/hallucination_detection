"""Shared low-rank feature/time fusion, fixed raw tokens/Top10 and nested folds."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils import two_axis_factor_fusion as model
from spectral_utils.direct_probability_fusion import step_top_mean
from scripts import run_rbm_hierarchical_time as hy
from scripts.test_two_axis_factor_fusion import run as fixtures
old=hy.old;base=hy.base
OUT=ROOT/'results/two_axis_factor_fusion_v1'
METHODS=('stationary','rank1','rank2','rank2_shuffled')
PRIMARY=[('rank1','stationary'),('rank2','rank1')]
START=0.

def emit(name,value):base.atomic_json(OUT/name,value)
def progress(stage,**kwargs):
    emit('RUN_STATE.json',dict(status=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-START,**kwargs))
    print(stage,kwargs,flush=True)
    if time.monotonic()-START>8*3600:raise hy.InvocationCap()

def manifest(source,smoke):
    inherited=hy.manifest(source)
    paths=[Path(__file__),ROOT/'spectral_utils/two_axis_factor_fusion.py',
           ROOT/'scripts/test_two_axis_factor_fusion.py',ROOT/'docs/experiments/TWO_AXIS_FACTOR_FUSION_V1.md']
    inherited['hashes'].update({str(p):base.old.sha256_file(p) for p in paths})
    return dict(schema='two-axis-factor-fusion-v1',base='2ec11dd0c',smoke=smoke,
                methods=METHODS,hashes=inherited['hashes'])

def connect(freeze,smoke):
    con=sqlite3.connect(OUT/('SMOKE.sqlite' if smoke else 'CHECKPOINT.sqlite'))
    con.execute('pragma journal_mode=WAL')
    for sql in ['create table if not exists manifest(payload text)',
                'create table if not exists stats(idx integer primary key,uid text,payload blob)',
                'create table if not exists models(key text primary key,payload blob,info text)',
                'create table if not exists scores(idx integer primary key,payload blob)']:
        con.execute(sql)
    prior=con.execute('select payload from manifest').fetchone()
    if prior:assert json.loads(prior[0])==json.loads(base.dumps(freeze)),'frozen manifest changed'
    else:con.execute('insert into manifest values (?)',(base.dumps(freeze),));con.commit()
    return con

def selection(path,records,fold,smoke):
    with np.load(path) as z:
        ids=z['ids'].copy();lengths=np.diff(z['token_offsets'])
    if not smoke:return list(map(int,ids))
    positions=[];used=set();order=np.argsort(lengths,kind='stable')
    for fraction in (0.,.5,.95):
        pool=[int(k) for k in order if fold[int(ids[k])] not in used]
        chosen=pool[min(len(pool)-1,int(len(pool)*fraction))]
        positions.append(int(ids[chosen]));used.add(fold[int(ids[chosen])])
    return positions

def load_cache(path):
    while hy.free_gib()<4.:
        progress('WAITING_FOR_RAM',available_gib=hy.free_gib());time.sleep(30)
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def answer(cache,k,records,joined,reference):
    i=int(cache['ids'][k]);assert str(cache['uid'][k])==records[i]['uid']
    ta,tb=cache['token_offsets'][k:k+2];sa,sb=cache['step_offsets'][k:k+2]
    x=cache['x'][ta:tb];spans=cache['spans'][sa:sb]-ta
    assert x.shape[1]==12 and np.isfinite(x).all()
    active=cache['active'][k]
    np.testing.assert_allclose(x[:,active].mean(axis=0),0.,atol=1e-10)
    np.testing.assert_allclose(x[:,active].std(axis=0),1.,atol=1e-10)
    assert spans.shape==(records[i]['steps'],2) and np.all(spans[:,1]>spans[:,0])
    assert np.min(spans)>=0 and np.max(spans)<=len(x)
    ell=cache['orientation'][k]*(cache['b'][k]+x@cache['w'][k])
    top=step_top_mean(ell,spans[:,0],spans[:,1],10)
    sl=slice(joined['offsets'][i],joined['offsets'][i+1])
    np.testing.assert_allclose(top,reference['rbm12__logit_old'][sl],atol=1e-12,rtol=0)
    return i,x,spans,top

def extract(con,paths,selected,records,joined,reference):
    done={r[0] for r in con.execute('select idx from stats')}
    for path in paths:
        with np.load(path) as z:ids=z['ids'].copy()
        todo=[k for k,i in enumerate(ids) if int(i) in selected and int(i) not in done]
        if not todo:continue
        cache=load_cache(path)
        for k in todo:
            i,x,spans,top=answer(cache,k,records,joined,reference)
            real,shuffle=model.sufficient_statistics(x,spans,records[i]['uid'])
            con.execute('insert into stats values(?,?,?)',(i,records[i]['uid'],base.packed(real=real,shuffle=shuffle)))
            con.commit();done.add(i)
            if len(done)%100==0:progress('EXTRACTING_TOKEN_MOMENTS',completed=len(done),expected=len(selected))
        del cache;gc.collect()
    assert done==set(selected)
    progress('STATISTICS_COMPLETE',completed=len(done))

def train(con,records,fold,selected):
    saved={k for k, in con.execute('select key from models')}
    for cell in sorted({records[i]['cell'] for i in selected}):
        ids=[i for i in selected if records[i]['cell']==cell]
        cell_folds=sorted({fold[i] for i in ids})
        exclusions=[(f,) for f in cell_folds]
        if not cell.startswith('pb_'):
            exclusions+=[(f,h) for f in cell_folds for h in cell_folds if f<h]
        stats={i:hy.load_blob(con.execute('select payload from stats where idx=?',(i,)).fetchone()[0]) for i in ids}
        for excluded in exclusions:
            key=cell+'__exclude_'+'_'.join(map(str,excluded))
            if key in saved:continue
            train_ids=[i for i in ids if fold[i] not in excluded]
            assert train_ids
            weights=hy.group_weights(records,train_ids)
            held_groups={records[i]['group_id'] for i in selected if fold[i] in excluded}
            groups=sorted({records[i]['group_id'] for i in train_ids})
            assert not held_groups.intersection(groups)
            moments={name:sum(weights[i]*stats[i][name] for i in train_ids) for name in ('real','shuffle')}
            arrays={};info=dict(cell=cell,excluded_folds=excluded,training_ids=train_ids,training_groups=groups,fits={})
            for name in METHODS:
                try:
                    a,d=model.fit(moments['shuffle' if name=='rank2_shuffled' else 'real'],
                                  2 if name.startswith('rank2') else 1,stationary=name=='stationary')
                    arrays.update({name+'__'+k:v for k,v in a.items()});info['fits'][name]=d
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
                    arrays[name+'__coefficients']=np.full((16,12),np.nan);info['fits'][name]=dict(failure=str(e))
                progress('FITTING',model=key,arm=name,converged=info['fits'][name].get('converged'),
                         training_answers=len(train_ids))
            con.execute('insert into models values(?,?,?)',(key,base.packed(**arrays),base.dumps(info)));con.commit();saved.add(key)
        del stats;gc.collect()

def model_key(cell,excluded):return cell+'__exclude_'+'_'.join(map(str,sorted(excluded)))

def score(con,paths,selected,records,joined,reference,fold):
    done={i for i, in con.execute('select idx from scores')}
    models={key:hy.load_blob(payload) for key,payload in con.execute('select key,payload from models')}
    for path in paths:
        with np.load(path) as z:ids=z['ids'].copy()
        todo=[k for k,i in enumerate(ids) if int(i) in selected and int(i) not in done]
        if not todo:continue
        cache=load_cache(path)
        for k in todo:
            i,x,spans,top=answer(cache,k,records,joined,reference);cell=records[i]['cell'];f=fold[i]
            payload=dict(top10=top)
            excluded_sets=[(f,)]
            if not cell.startswith('pb_'):
                other={fold[j] for j in selected if records[j]['cell']==cell and fold[j]!=f}
                excluded_sets += [tuple(sorted((f,h))) for h in sorted(other)]
            for excluded in excluded_sets:
                fit=models[model_key(cell,excluded)]
                suffix='' if len(excluded)==1 else '__inner_for_'+str(next(h for h in excluded if h!=f))
                for name in METHODS:
                    payload[name+suffix]=model.step_scores(x,spans,fit[name+'__coefficients'],records[i]['uid'],
                                                         shuffled=name=='rank2_shuffled')
            con.execute('insert into scores values(?,?)',(i,base.packed(**payload)));con.commit();done.add(i)
            if len(done)%100==0:progress('SCORING_TOP10',completed=len(done),expected=len(selected))
        del cache;gc.collect()
    assert done==set(selected)
    progress('SCORES_COMPLETE',completed=len(done))

def review_exclusions(con,records,fold,selected):
    count=0
    for key,text in con.execute('select key,info from models'):
        info=json.loads(text);ex=set(info['excluded_folds'])
        expected=[i for i in selected if records[i]['cell']==info['cell'] and fold[i] not in ex]
        assert info['training_ids']==expected
        assert set(info['training_groups'])=={records[i]['group_id'] for i in expected}
        assert not set(info['training_groups']).intersection(records[i]['group_id'] for i in selected if fold[i] in ex)
        count+=1
    # Fit moments depend only on saved unlabeled token statistics; no labels are read by train/fit.
    return dict(status='PASS',models=count,all_exclusions_rederived=True,fit_api_accepts_no_labels=True)

def evaluate(con,records,joined,reference,fold,selected):
    assert len(selected)==13769
    total=int(joined['offsets'][-1]);scores={name:np.full(total,np.nan) for name in ('top10',)+METHODS}
    predictions={i:hy.load_blob(blob) for i,blob in con.execute('select idx,payload from scores')}
    for i,a in predictions.items():
        sl=slice(joined['offsets'][i],joined['offsets'][i+1])
        for name in scores:scores[name][sl]=a[name]
    for name,a in reference.items():scores['reference__'+name]=a
    thresholds={name:{} for name in scores};coverage=[]
    for f in sorted(set(fold.values())):
        ids=[i for i in selected if not records[i]['cell'].startswith('pb_') and fold[i]!=f]
        assert not {records[i]['group_id'] for i in ids}.intersection(records[i]['group_id'] for i in selected if fold[i]==f)
        for name in scores:
            values=[]
            for i in ids:
                sl=slice(joined['offsets'][i],joined['offsets'][i+1])
                a=predictions[i][name+'__inner_for_'+str(f)] if name in METHODS else scores[name][sl]
                if np.isfinite(a).all():values.append(a)
            if not values:raise ValueError(f'no calibration for {name} fold{f}')
            thresholds[name][str(f)]=float(np.quantile(np.concatenate(values),.8))
            coverage.append(dict(method=name,outer_fold=f,answers=len(values),expected=len(ids)))
    metrics,per=base.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True)
    prm=np.array([not r['cell'].startswith('pb_') for r in records]);step_prm=np.repeat(prm,np.diff(joined['offsets']))
    for name,flat in scores.items():
        mask=step_prm & np.repeat(per[name]['valid'],np.diff(joined['offsets'])) & (joined['labels']>=0) & np.isfinite(flat)
        metrics[name]['prm_pooled']=base.old.auc(joined['labels'][mask]==1,flat[mask])
    previous=json.loads((old.parent(SOURCE)/'METRICS.json').read_text())['metrics']
    for name in reference:
        for metric in ('pb_all8','prm_within','prm_pooled','prmscore_q08'):
            np.testing.assert_allclose(metrics['reference__'+name][metric],previous[name][metric],atol=1e-12,rtol=0)
    np.testing.assert_allclose(scores['top10'],reference['rbm12__logit_old'],atol=1e-12,rtol=0)
    pairs=PRIMARY+[('rank2','rank2_shuffled')]+[(name,'top10') for name in METHODS]
    progress('BOOTSTRAP',draws=10000)
    contrasts=base.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(PRIMARY),primary_ci=.975)
    review=hy.independent_review(records,joined,scores,metrics,thresholds,per)
    hy.METHODS=('top10',)+METHODS
    transitions=hy.error_and_weight_tables(records,joined,{},[],per)
    np.savez_compressed(OUT/'SCORES.npz',**scores)
    emit('METRICS.json',metrics);emit('CONTRASTS.json',contrasts)
    emit('CALIBRATION.json',dict(thresholds=thresholds,coverage=coverage));emit('ERROR_TRANSITIONS.json',transitions)
    old.csv_write(OUT/'COMPARISON.csv',[dict(method=n,**{k:v for k,v in a.items() if not isinstance(v,dict)}) for n,a in metrics.items()])
    old.csv_write(OUT/'PER_CELL.csv',[dict(method=n,cell=cell,**a) for n,d in metrics.items() for cell,a in d['pb_cells'].items()])
    emit('RESULT_REVIEW.json',dict(status='PASS',answers=len(selected),independent_metrics=review,
                                 frozen_references_reproduced=True,exclusions=review_exclusions(con,records,fold,selected)))
    report=['# Two-axis factor fusion (RBM12 bank, Top10 fixed)','',
            'Full development benchmark; other-answer unlabelled fitting. Rank means loading-map rank, not latent classes.',
            '', '| Method | PB % | PRMB within AUC | PRMScore | Coverage |', '|---|---:|---:|---:|---:|']
    for n in ('top10',)+METHODS:
        d=metrics[n];report.append(f'| {n} | {100*d["pb_all8"]:.3f} | {d["prm_within"]:.5f} | {d["prmscore_q08"]:.5f} | {d["valid_answers"]}/13769 |')
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    progress('COMPLETE_REVIEWED',completed=13769)

def main():
    global START,OUT,SOURCE
    START=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--phase',choices=['smoke','full'],default='smoke');args=p.parse_args();SOURCE=args.source_root
    smoke=args.phase=='smoke'
    if smoke:OUT=OUT/'smoke'
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'RUN.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        with threadpool_limits(limits=1):
            emit('UNIT_REVIEW.json',fixtures())
            records,joined,reference=old.load_contract(SOURCE)
            folds=json.loads(base.old.FOLDS.read_text())['outer'];fold={i:int(folds[r['group_id']]) for i,r in enumerate(records)}
            paths=sorted(old.caches(SOURCE).glob('cache_*.npz'))
            selected=sorted(i for path in paths for i in selection(path,records,fold,smoke))
            freeze=manifest(SOURCE,smoke);con=connect(freeze,smoke);emit('MANIFEST.json',freeze)
            extract(con,paths,selected,records,joined,reference);train(con,records,fold,selected)
            score(con,paths,selected,records,joined,reference,fold)
            fits=[dict(model=key,arm=n,**d) for key,text in con.execute('select key,info from models') for n,d in json.loads(text)['fits'].items()]
            emit('FIT_HEALTH.json',fits)
            if smoke:
                failures={n:0 for n in METHODS}
                for blob, in con.execute('select payload from scores'):
                    a=hy.load_blob(blob)
                    for n in METHODS:failures[n]+=int(not np.isfinite(a[n]).all())
                assert not any(failures.values()),failures
                emit('SMOKE_REVIEW.json',dict(status='PASS',answers=len(selected),failures=failures,
                     nonconverged=sum(d.get('converged') is False for d in fits),fits=len(fits),
                     exclusions=review_exclusions(con,records,fold,selected),purpose='feasibility only'))
                progress('SMOKE_COMPLETE',completed=len(selected))
            else:evaluate(con,records,joined,reference,fold,selected)
            con.close()
    except hy.InvocationCap:emit('RUN_STATE.json',dict(status='CHECKPOINTED_INVOCATION_CAP'))
    except Exception:
        emit('RUN_STATE.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
    finally:lock.unlink()

if __name__=='__main__':main()
