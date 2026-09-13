"""Frozen-RBM temporal weights, single process, resumable and nested-calibrated."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import ctypes
import gc
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils import rbm_hierarchical_time as model
from scripts import run_rbm_literature_completion as old
from scripts.test_rbm_hierarchical_time import run as fixtures
base=old.base
OUT=ROOT/'results/rbm_hierarchical_time_v1'
METHODS=('top10','all_mean','contiguous10','local','shared','hierarchical','shuffled','supervised',
         'drop_first_top10','drop_first_hierarchical')
POOLED=('shared','hierarchical','shuffled','supervised','drop_first_hierarchical')
PRIMARY=[('shared','local'),('hierarchical','shared')]


def emit(name,obj):
    base.atomic_json(OUT/name,obj)


def free_gib():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('cannot check available RAM')
    return m.available/2**30


class InvocationCap(Exception):pass


def checkpoint(stage,**kwargs):
    emit('RUN_STATE.json',dict(status=stage,elapsed_seconds=time.monotonic()-START,**kwargs))
    print(stage,kwargs,flush=True)
    if time.monotonic()-START>8*3600:
        raise InvocationCap()


def manifest(source):
    paths=[Path(__file__),ROOT/'spectral_utils/rbm_hierarchical_time.py',
           ROOT/'scripts/test_rbm_hierarchical_time.py',ROOT/'docs/experiments/RBM_HIERARCHICAL_TIME_V1.md',
           ROOT/'scripts/run_rbm_literature_completion.py',ROOT/'scripts/run_direct_probability_temporal.py',
           ROOT/'scripts/run_direct_probability_fusion_v2.py',ROOT/'spectral_utils/direct_probability_fusion.py',
           ROOT/'spectral_utils/rbm_literature_completion.py',ROOT/'spectral_utils/historical_fusion_evaluation.py',
           ROOT/'spectral_utils/prmbench.py',base.old.BENCH/'evaluation/JOINED.json',
           base.old.BENCH/'evaluation/JOINED.npz',base.old.FOLDS,base.old.PRMB_LABELS,
           base.old.FIXED_GATE/'DETECTORS.npz',base.old.FIXED_GATE/'METRICS.json',
           old.modeldb(source),old.parent(source)/'SCORES.npz',old.parent(source)/'METRICS.json',
           old.parent(source)/'RESULT_REVIEW.json',old.caches(source)/'MANIFEST.json',
           old.caches(source)/'ZERO_UPDATE_REVIEW.json',*sorted(old.caches(source).glob('cache_*.npz'))]
    assert len(list(old.caches(source).glob('cache_*.npz')))==9
    return dict(schema='rbm-hierarchical-time-v1',base='113c7eb79',methods=METHODS,
                bins=16,bootstrap=10000,hashes={str(p):base.old.sha256_file(p) for p in paths})


def open_database(path,freeze):
    con=sqlite3.connect(path)
    con.execute('pragma journal_mode=WAL')
    con.execute('create table if not exists manifest (payload text)')
    prior=con.execute('select payload from manifest').fetchone()
    if prior is not None:
        assert json.loads(prior[0])==json.loads(base.dumps(freeze)), 'frozen code/input manifest changed'
    else:
        con.execute('insert into manifest values (?)',(base.dumps(freeze),));con.commit()
    con.execute('create table if not exists profiles (idx integer primary key, uid text, payload blob)')
    con.execute('create table if not exists models (key text primary key, payload blob, info text)')
    con.execute('create table if not exists scores (key text primary key, payload blob, info text)')
    return con


def load_blob(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:return {k:z[k] for k in z.files}


def extract(con,source,records,joined,reference,smoke=False):
    done={r[0] for r in con.execute('select idx from profiles')}
    src=sqlite3.connect(f'file:{old.modeldb(source).as_posix()}?mode=ro',uri=True)
    selected=[]
    source_folds=json.loads(base.old.FOLDS.read_text(encoding='utf8'))['outer']
    for path in sorted(old.caches(source).glob('cache_*.npz')):
        while free_gib()<4.:
            checkpoint('WAITING_FOR_RAM',required_gib=4.,available_gib=free_gib(),cache=path.name)
            time.sleep(30)
        with np.load(path,allow_pickle=False) as z:
            ids=z['ids'];lengths=np.diff(z['token_offsets'])
            order=np.argsort(lengths,kind='stable')
            if smoke:
                # Length coverage while retaining three distinct source folds, so
                # nested exclusion of two folds still leaves training data.
                positions=[];used=set()
                for fraction in (0.,.5,.95):
                    candidates=[int(k) for k in order if records[int(ids[k])]['steps']>1
                                and source_folds[records[int(ids[k])]['group_id']] not in used]
                    if not candidates:raise ValueError('smoke needs three distinct source folds')
                    chosen=candidates[min(len(candidates)-1,int(fraction*len(candidates)))]
                    positions.append(chosen);used.add(source_folds[records[int(ids[chosen])]['group_id']])
            else:positions=list(range(len(ids)))
            selected.extend(int(ids[k]) for k in positions)
            todo=[k for k in positions if int(ids[k]) not in done]
            if not todo:continue
            cache={k:z[k] for k in z.files}
        for k in todo:
            i,uid,spans,anchor,banks=old.prepare_answer(k,cache,src,records,joined,reference)
            b=banks[12];a,w,bias=old.model.unpack(b['theta'],b['x'].shape[1],1)
            ell=b['orientation']*(float(bias[0])+b['x']@w[:,0])
            profile=model.make_profiles(ell,spans,uid)
            sl=slice(joined['offsets'][i],joined['offsets'][i+1])
            np.testing.assert_allclose(profile['top'],reference['rbm12__logit_old'][sl],atol=1e-12,rtol=0)
            con.execute('insert into profiles values (?,?,?)',(i,uid,base.packed(**profile)))
            con.commit();done.add(i)
            if len(done)%50==0:checkpoint('EXTRACTING',completed=len(done),expected=27 if smoke else 13769)
        del cache,banks,b,ell,profile;gc.collect()
    src.close()
    assert set(selected)==done
    checkpoint('PROFILES_COMPLETE',completed=len(done))
    return selected


def label_vectors(records,joined,ids):
    out={}
    for i in ids:
        r=records[i];s=r['steps']
        if r['cell'].startswith('pb_'):
            target=int(joined['target'][i]);y=np.full(s,-1,int)
            if target<0:y[:]=0
            else:y[:target]=0;y[target]=1
        else:y=joined['labels'][joined['offsets'][i]:joined['offsets'][i+1]].astype(int).copy()
        out[i]=y
    return out


def group_weights(records,ids):
    grouped={}
    for i in ids:grouped.setdefault(records[i]['group_id'],[]).append(i)
    return {i:1./(len(grouped)*len(members)) for members in grouped.values() for i in members}


def training_inputs(records,profiles,ids,labels):
    covariance_ids=[i for i in ids if len(profiles[i]['raw'])>1]
    if not covariance_ids:raise ValueError('no multi-step training answers')
    weights=group_weights(records,covariance_ids)
    covariance=sum(weights[i]*profiles[i]['covariance'] for i in covariance_ids)
    shuffled=sum(weights[i]*profiles[i]['shuffled_covariance'] for i in covariance_ids)
    labelled=[i for i in ids if np.any(labels[i]>=0)]
    weights=group_weights(records,labelled)
    xs=[];ys=[];ws=[]
    for i in labelled:
        known=labels[i]>=0;n=int(known.sum())
        xs.append(profiles[i]['normalized'][known]);ys.append(labels[i][known]);ws.append(np.full(n,weights[i]/n))
    return covariance,shuffled,np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)


def cached_model(con,cell,excluded,records,profiles,fold,labels):
    key=cell+'__exclude_'+'_'.join(map(str,sorted(excluded)))
    prior=con.execute('select payload,info from models where key=?',(key,)).fetchone()
    if prior:return load_blob(prior[0]),json.loads(prior[1])
    ids=[i for i in profiles if records[i]['cell']==cell and fold[i] not in excluded]
    held={records[i]['group_id'] for i in profiles if fold[i] in excluded}
    groups=sorted({records[i]['group_id'] for i in ids})
    assert not held.intersection(groups)
    c,cs,x,y,sw=training_inputs(records,profiles,ids,labels)
    arrays=dict(covariance=c,shuffled_covariance=cs);info=dict(cell=cell,excluded_folds=sorted(excluded),
        training_ids=ids,training_groups=groups,training_group_sha256=hashlib.sha256('\n'.join(groups).encode()).hexdigest(),fits={})
    for name,cov in [('shared',c),('shuffled',cs)]:
        try:arrays[name],info['fits'][name]=model.fit_covariance(cov)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
            arrays[name]=np.full(16,np.nan);info['fits'][name]=dict(failure=str(e))
    try:arrays['supervised'],info['fits']['supervised']=model.fit_supervised(x,y,sw)
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
        arrays['supervised']=np.full(16,np.nan);info['fits']['supervised']=dict(failure=str(e))
    con.execute('insert into models values (?,?,?)',(key,base.packed(**arrays),base.dumps(info)));con.commit()
    checkpoint('MODELS',model=key,training_answers=len(ids))
    return arrays,info


def score_answer(con,i,excluded,records,profiles,fold,labels,inner=False):
    key=f'{i}__exclude_'+('_'.join(map(str,sorted(excluded))))
    prior=con.execute('select payload,info from scores where key=?',(key,)).fetchone()
    if prior:return load_blob(prior[0]),json.loads(prior[1])
    p=profiles[i];s=len(p['raw']);cell=records[i]['cell']
    shared,provenance=cached_model(con,cell,excluded,records,profiles,fold,labels)
    info=dict(uid=records[i]['uid'],excluded_folds=sorted(excluded),inner_calibration=inner,fits={})
    arrays={}
    if not inner:
        arrays.update(top10=p['top'],all_mean=p['raw'].mean(axis=1),contiguous10=p['contiguous'],drop_first_top10=p['drop_top'])
        try:
            if s==1:u=np.full(16,1/16);d=dict(reason='ONE_STEP_UNIFORM')
            else:u,d=model.fit_covariance(p['covariance'])
            arrays['local']=p['raw']@u;arrays['weight__local']=u;info['fits']['local']=d
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
            arrays['local']=np.full(s,np.nan);info['fits']['local']=dict(failure=str(e))
    for name in ('shared','supervised'):
        u=shared[name];arrays[name]=p['raw']@u;arrays['weight__'+name]=u
        info['fits'][name]=provenance['fits'][name]
    for name,raw,cov,train,base_weight in (
        ('hierarchical','raw','covariance','covariance','shared'),
        ('shuffled','shuffled','shuffled_covariance','shuffled_covariance','shuffled')):
        try:
            ch,alpha=model.hierarchical(p[cov],shared[train],s)
            if s==1:u=shared[base_weight];d=dict(reason='ONE_STEP_SHARED')
            else:u,d=model.fit_covariance(ch)
            arrays[name]=p[raw]@u;arrays['weight__'+name]=u;info['fits'][name]=dict(alpha=alpha,**d)
            if name=='hierarchical':
                arrays['drop_first_hierarchical'],info['boundary']=model.drop_first(arrays[name],u,p)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
            arrays[name]=np.full(s,np.nan);info['fits'][name]=dict(failure=str(e))
            if name=='hierarchical':arrays['drop_first_hierarchical']=np.full(s,np.nan)
    con.execute('insert into scores values (?,?,?)',(key,base.packed(**arrays),base.dumps(info)));con.commit()
    return arrays,info


def full_scoring(con,records,profiles,fold,labels,smoke):
    started=time.monotonic()
    for n,i in enumerate(profiles):
        score_answer(con,i,{fold[i]},records,profiles,fold,labels)
        if (n+1)%25==0:checkpoint('SCORING',completed=n+1,expected=len(profiles),seconds=time.monotonic()-started)
    checkpoint('OUTER_SCORES_COMPLETE',completed=len(profiles),seconds=time.monotonic()-started)
    # Only PRMBench needs nested score calibration; models are cached by exclusion set.
    for f in sorted(set(fold.values())):
        calibration_ids=[i for i in profiles if not records[i]['cell'].startswith('pb_') and fold[i]!=f]
        for n,i in enumerate(calibration_ids):
            score_answer(con,i,{f,fold[i]},records,profiles,fold,labels,inner=True)
            if (n+1)%50==0:checkpoint('INNER_CALIBRATION',outer_fold=f,completed=n+1,expected=len(calibration_ids))
    checkpoint('ALL_SCORES_COMPLETE',completed=len(profiles))


def thresholds_and_scores(con,records,joined,profiles,fold,labels,reference):
    scores={name:np.full(int(joined['offsets'][-1]),np.nan) for name in METHODS}
    details=[]
    for i in profiles:
        arrays,info=score_answer(con,i,{fold[i]},records,profiles,fold,labels)
        sl=slice(joined['offsets'][i],joined['offsets'][i+1])
        for name in METHODS:scores[name][sl]=arrays[name]
        details.append(dict(idx=i,**info))
    for name,value in reference.items():scores['reference__'+name]=value
    thresholds={m:{} for m in scores};coverage=[]
    for f in sorted(set(fold.values())):
        train=[i for i in profiles if not records[i]['cell'].startswith('pb_') and fold[i]!=f]
        streams={name:[] for name in scores}
        for i in train:
            inner,_=score_answer(con,i,{f,fold[i]},records,profiles,fold,labels,inner=True)
            sl=slice(joined['offsets'][i],joined['offsets'][i+1])
            for name in scores:
                a=inner[name] if name in POOLED else scores[name][sl]
                if np.isfinite(a).all():streams[name].append(a)
        for name,values in streams.items():
            if not values:raise ValueError(f'no calibration values: {name} fold {f}')
            thresholds[name][str(f)]=float(np.quantile(np.concatenate(values),.8))
            coverage.append(dict(method=name,outer_fold=f,answers=len(values),expected=len(train),steps=sum(map(len,values))))
    return scores,thresholds,coverage,details


def independent_review(records,joined,scores,metrics,thresholds,per):
    from sklearn.metrics import roc_auc_score
    cells=np.array([r['cell'] for r in records]);target=joined['target'];offsets=joined['offsets']
    detector,gate=base.old._gate_contract(records)
    review={}
    folds=json.loads(base.old.FOLDS.read_text(encoding='utf8'))['outer']
    raw_labels={str(r['idx']):r for r in base.old.load_pickle(base.old.PRMB_LABELS).values()}
    for name,flat in scores.items():
        cell_f1=[];within=[];correct=[];counts=np.zeros(4,dtype=int)
        for i,r in enumerate(records):
            s=flat[offsets[i]:offsets[i+1]];valid=np.isfinite(s).all()
            predicted=int(np.argmax(s)) if valid and detector[i]>=gate[i] else -1
            correct.append(bool(valid and np.isfinite(detector[i]) and np.isfinite(gate[i]) and predicted==target[i]))
            if not r['cell'].startswith('pb_') and valid:
                y=joined['labels'][offsets[i]:offsets[i+1]];known=y>=0
                if len(np.unique(y[known]))==2:within.append(roc_auc_score(y[known]==1,s[known]))
                errors={int(e)-1 for e in raw_labels[str(r['row_id'])]['error_steps']}
                truth=np.array([j in errors for j in range(len(s))])
                np.testing.assert_array_equal(truth,y==1)
                risk=s>=thresholds[name][str(folds[r['group_id']])]
                # Official PRMScore excludes synthetic all-correct control rows
                # from its pooled F1, while AUC still includes those rows.
                if raw_labels[str(r['row_id'])]['classification']!='correct':
                    counts+=np.array([np.sum(~truth & ~risk),np.sum(truth & ~risk),
                                      np.sum(truth & risk),np.sum(~truth & risk)])
        correct=np.asarray(correct)
        for cell in sorted(set(cells[np.char.startswith(cells,'pb_')])):
            clean=(cells==cell)&(target<0);error=(cells==cell)&(target>=0)
            c=correct[clean].mean();e=correct[error].mean()
            cell_f1.append(2*c*e/(c+e) if c+e else 0.)
        np.testing.assert_allclose(np.mean(cell_f1),metrics[name]['pb_all8'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.mean(within),metrics[name]['prm_within'],atol=1e-12,rtol=0)
        tp,fp,tn,fn=counts
        prm=.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
        np.testing.assert_allclose(prm,metrics[name]['prmscore_conditional'],atol=1e-12,rtol=0)
        review[name]=dict(pb_rederived=True,within_rederived=True,prmscore_rederived=True)
    return review


def exclusion_review(con,records,profiles,fold,labels):
    """All saved train sets are disjoint; refit representative models after label flips."""
    checks=[];refitted=set()
    for key,blob,text in con.execute('select key,payload,info from models order by key'):
        info=json.loads(text);excluded=set(info['excluded_folds']);ids=info['training_ids']
        expected=[i for i in profiles if records[i]['cell']==info['cell'] and fold[i] not in excluded]
        assert ids==expected
        held={records[i]['group_id'] for i in profiles if fold[i] in excluded}
        assert not held.intersection(info['training_groups'])
        category=('pb' if info['cell'].startswith('pb_') else 'prm',len(excluded))
        checked=False
        if category not in refitted:
            altered={i:(np.where(y>=0,1-y,y) if fold[i] in excluded else y.copy()) for i,y in labels.items()}
            original=training_inputs(records,profiles,ids,labels)
            changed=training_inputs(records,profiles,ids,altered)
            for a,b in zip(original,changed):np.testing.assert_array_equal(a,b)
            saved=load_blob(blob)
            if np.isfinite(saved['supervised']).all():
                w,_=model.fit_supervised(changed[2],changed[3],changed[4])
                np.testing.assert_allclose(w,saved['supervised'],atol=1e-12,rtol=0)
            checked=True;refitted.add(category)
        checks.append(dict(model=key,training_groups=len(info['training_groups']),
                           excluded_folds=sorted(excluded),held_label_perturbation_refitted=checked))
    return dict(status='PASS',models=checks,refitted_categories=[list(x) for x in sorted(refitted)])


def error_and_weight_tables(records,joined,profiles,details,per):
    target=joined['target'];pb=np.array([r['cell'].startswith('pb_') for r in records])
    rows=[];weight_rows=[]
    for name in METHODS:
        baseline=per['top10'];current=per[name]
        old_hit=pb & baseline['decision_valid'] & (baseline['prediction']==target)
        hit=pb & current['decision_valid'] & (current['prediction']==target)
        lost=old_hit & ~hit;gained=hit & ~old_hit
        invalid=lost & ~current['decision_valid']
        gate=lost & ~invalid & (current['prediction']==-1)
        early=lost & ~invalid & ~gate & (current['prediction']<target)
        late=lost & ~invalid & ~gate & (current['prediction']>target)
        assert np.array_equal(invalid.astype(int)+gate+early+late,lost.astype(int))
        rows.append(dict(method=name,gained=int(gained.sum()),lost=int(lost.sum()),
                         invalid=int(invalid.sum()),gate=int(gate.sum()),early=int(early.sum()),late=int(late.sum())))
    return rows


def evaluate(con,source,records,joined,profiles,fold,labels,reference):
    assert len(profiles)==13769
    scores,thresholds,calibration_coverage,details=thresholds_and_scores(con,records,joined,profiles,fold,labels,reference)
    metrics,per=base.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True)
    prm=np.array([not r['cell'].startswith('pb_') for r in records]);step_prm=np.repeat(prm,np.diff(joined['offsets']))
    for name,flat in scores.items():
        mask=step_prm & np.repeat(per[name]['valid'],np.diff(joined['offsets'])) & (joined['labels']>=0) & np.isfinite(flat)
        metrics[name]['prm_pooled']=base.old.auc(joined['labels'][mask]==1,flat[mask])
    previous=json.loads((old.parent(source)/'METRICS.json').read_text(encoding='utf8'))
    previous=previous['metrics']
    for name in reference:
        for metric in ('pb_all8','prm_within','prm_pooled','prmscore_q08'):
            np.testing.assert_allclose(metrics['reference__'+name][metric],previous[name][metric],atol=1e-12,rtol=0)
    for metric in ('pb_all8','prm_within','prm_pooled','prmscore_q08'):
        np.testing.assert_allclose(metrics['top10'][metric],metrics['reference__rbm12__logit_old'][metric],atol=1e-12,rtol=0)
    pairs=PRIMARY+[(m,'top10') for m in METHODS if m!='top10']
    checkpoint('BOOTSTRAP',draws=10000)
    contrasts=base.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(PRIMARY),primary_ci=.975)
    checkpoint('INDEPENDENT_REVIEW')
    verified=independent_review(records,joined,scores,metrics,thresholds,per)
    exclusions=exclusion_review(con,records,profiles,fold,labels)
    emit('EXCLUSION_REVIEW.json',exclusions)
    errors=error_and_weight_tables(records,joined,profiles,details,per)
    weight_rows=[];fit_summary={}
    for i in profiles:
        a,info=score_answer(con,i,{fold[i]},records,profiles,fold,labels)
        for name in ('local','shared','hierarchical','shuffled','supervised'):
            d=info['fits'][name];s=fit_summary.setdefault(name,dict(fits=0,nonconverged=0,failures=0,reasons={}))
            s['fits']+=1;s['nonconverged']+=int(d.get('converged') is False);s['failures']+=int('failure' in d)
            reason=d.get('reason','FAILURE');s['reasons'][reason]=s['reasons'].get(reason,0)+1
            if 'weight__'+name not in a:continue
            w=a['weight__'+name]
            row=dict(idx=i,uid=records[i]['uid'],cell=records[i]['cell'],method=name,
                     alpha=(records[i]['steps']-1)/(records[i]['steps']-1+16),
                     distance_uniform=float(np.abs(w-1/16).sum()),
                     distance_shared=float(np.abs(w-a['weight__shared']).sum()))
            row.update({f'region_{j+1}':float(x) for j,x in enumerate(w)})
            weight_rows.append(row)
    np.savez_compressed(OUT/'SCORES.npz',**scores)
    emit('METRICS.json',metrics);emit('CONTRASTS.json',contrasts);emit('CALIBRATION.json',dict(thresholds=thresholds,coverage=calibration_coverage))
    emit('FIT_HEALTH.json',fit_summary);emit('ERROR_TRANSITIONS.json',errors)
    old.csv_write(OUT/'WEIGHTS.csv',weight_rows)
    old.csv_write(OUT/'METRICS.csv',[dict(method=name,**{k:v for k,v in row.items() if not isinstance(v,dict)}) for name,row in metrics.items()])
    emit('RESULT_REVIEW.json',dict(status='PASS',answers=13769,fixtures=fixtures(),independent_metrics=verified,
                                 fixed_references_reproduced=True,calibration_coverage=calibration_coverage))
    checkpoint('COMPLETE_REVIEWED',completed=13769)


def main():
    global START
    START=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--phase',choices=['smoke','full'],default='smoke');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'RUN.lock'
    # Exclusive creation; stale locks require inspection, never automatic override.
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        with threadpool_limits(limits=1):
            emit('UNIT_REVIEW.json',fixtures())
            records,joined,reference=old.load_contract(args.source_root)
            freeze=manifest(args.source_root);freeze['phase']=args.phase
            smoke_id=hashlib.sha256(base.dumps(freeze).encode()).hexdigest()[:12]
            con=open_database(OUT/(f'SMOKE_{smoke_id}.sqlite' if args.phase=='smoke' else 'CHECKPOINT.sqlite'),freeze)
            extract(con,args.source_root,records,joined,reference,smoke=args.phase=='smoke')
            profiles={i:load_blob(blob) for i,uid,blob in con.execute('select idx,uid,payload from profiles order by idx')
                      if uid==records[i]['uid']}
            folds=json.loads(base.old.FOLDS.read_text(encoding='utf8'))['outer']
            fold={i:int(folds[records[i]['group_id']]) for i in profiles}
            labels=label_vectors(records,joined,profiles)
            full_scoring(con,records,profiles,fold,labels,args.phase=='smoke')
            if args.phase=='smoke':
                failures={m:0 for m in METHODS}
                for i in profiles:
                    a,_=score_answer(con,i,{fold[i]},records,profiles,fold,labels)
                    for m in METHODS:failures[m]+=int(not np.isfinite(a[m]).all())
                assert all(failures[m]==0 for m in METHODS if m!='supervised'),failures
                assert failures['supervised']<len(profiles),'no successful supervised smoke fits'
                excluded=exclusion_review(con,records,profiles,fold,labels)
                emit('SMOKE_REVIEW.json',dict(status='PASS',answers=len(profiles),failures=failures,exclusions=excluded,
                                             elapsed_seconds=time.monotonic()-START,
                                             purpose='feasibility only; no benchmark inference'))
                checkpoint('SMOKE_COMPLETE',completed=len(profiles))
            else:evaluate(con,args.source_root,records,joined,profiles,fold,labels,reference)
            con.close()
    except InvocationCap:
        emit('RUN_STATE.json',dict(status='CHECKPOINTED_INVOCATION_CAP',elapsed_seconds=time.monotonic()-START))
    except Exception:
        emit('RUN_STATE.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
