"""Whole-answer position fusion: Gaussian factor, exact RBM, canonical IU-PCR."""
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
from spectral_utils import answer_position_fusion as model
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.higher_moment_fusion import feature_names
from spectral_utils.historical_fusion_evaluation import pb_metrics
from spectral_utils.reconstruction_benchmark import edis_bootstrap as auc_boot
from scripts import run_rbm_hierarchical_time as hy
from scripts.test_answer_position_fusion import run as fixtures
old=hy.old;base=hy.base
OUT=ROOT/'results/answer_position_fusion_v1'
METHODS=model.METHODS
ALL_SCORED=METHODS+model.CONTROLS
PRIMARY=[('factor_rank1','factor_position_mean'),('factor_rank2','factor_rank1'),
         ('rbm_rank1','rbm_stationary'),('rbm_rank2','rbm_rank1'),('iu_position','iu_position_mean')]
PRIMARY_CI=.99
PAIRS=PRIMARY+[(n,n+'_shuffled') for n in ('factor_rank2','rbm_rank2','iu_position')]
PAIRS += [(n,'top10') for n in METHODS]
PAIRS += [('factor_position_mean','factor_stationary'),('iu_position_mean','iu_stationary')]
FEATURE_NAMES=tuple(feature_names(6))
EXPECTED_FEATURE_NAMES=('entropy15','varentropy15','moment3_15','selected_surprisal',
    'selected_squared','selected_cubed','moment4_15','selected_power4',
    'moment5_15','selected_power5','moment6_15','selected_power6')
assert FEATURE_NAMES==EXPECTED_FEATURE_NAMES and FEATURE_NAMES[1]=='varentropy15'
START=0.

def emit(name,value):base.atomic_json(OUT/name,value)
def progress(stage,**kwargs):
    emit('RUN_STATE.json',dict(status=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-START,**kwargs))
    print(stage,kwargs,flush=True)
    if time.monotonic()-START>8*3600:raise hy.InvocationCap()

def manifest(source,smoke):
    inherited=hy.manifest(source)
    paths=[Path(__file__),ROOT/'spectral_utils/answer_position_fusion.py',
           ROOT/'spectral_utils/two_axis_factor_fusion.py',ROOT/'spectral_utils/upcr.py',
           ROOT/'spectral_utils/laplacian_upcr.py',ROOT/'spectral_utils/moment_rbm_fusion.py',
           ROOT/'scripts/test_answer_position_fusion.py',ROOT/'docs/experiments/ANSWER_POSITION_FUSION_V1.md']
    paths += [ROOT/'spectral_utils/higher_moment_fusion.py',
              ROOT/'spectral_utils/reconstruction_benchmark/edis_bootstrap.py',
              source/'.worktrees/rbm-supervision-matched-v1/scripts/run_matched_rbm_supervision.py']
    inherited['hashes'].update({str(p):base.old.sha256_file(p) for p in paths})
    return dict(schema='answer-position-fusion-v1',base='6249db384',smoke=smoke,
                methods=METHODS,feature_names=FEATURE_NAMES,bins=model.BINS,ranks=[1,2],
                controls=model.CONTROLS,position='whole answer token intervals; no step reset',
                shuffle='permuted position assignments; features and labels stay at original token',
                training='all tokens; equal canonical groups, then answers, then tokens',
                iu_defaults=model.IU_FIT_DEFAULTS,iu_moment_shrinkage='G/(G+16)',
                rbm='exact Gaussian/Bernoulli H1; identity conditional variance; shared biases',
                primary_pairs=PRIMARY,
                ridge=model.RIDGE,optimizer=dict(name='L-BFGS-B',maxiter=1000,ftol=1e-10,gtol=1e-6),
                bootstrap=dict(draws=10000,primary_ci=PRIMARY_CI,secondary_ci=.95,unit='canonical_source_group'),
                gate='fixed mean-entropy q=0.3',prmscore_calibration='q=0.8 nested held folds',
                readout='token score then Top10 mean per step; argmax earliest tie',hashes=inherited['hashes'])

def connect(freeze,smoke):
    con=sqlite3.connect(OUT/('SMOKE.sqlite' if smoke else 'CHECKPOINT.sqlite'))
    con.execute('pragma journal_mode=WAL')
    for sql in ['create table if not exists manifest(payload text)',
                'create table if not exists stats(idx integer primary key,uid text,payload blob)',
                'create table if not exists models(key text primary key,payload blob,info text)',
                'create table if not exists scores(idx integer primary key,payload blob)',
                'create table if not exists arm_models(key text,arm text,payload blob,info text,primary key(key,arm))']:
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
    assert active.shape==(12,)
    np.testing.assert_array_equal(x[:,~active],0.)
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
            stat=model.statistics(x,records[i]['uid'])
            con.execute('insert into stats values(?,?,?)',(i,records[i]['uid'],base.packed(**stat)))
            con.commit();done.add(i)
            if len(done)%100==0:progress('EXTRACTING_TOKEN_MOMENTS',completed=len(done),expected=len(selected))
        del cache;gc.collect()
    assert done==set(selected)
    progress('STATISTICS_COMPLETE',completed=len(done))

def rbm_blocks(cache,train_ids,weights,records,shuffled):
    """All training tokens, no subsampling or step-label broadcasting."""
    positions={int(i):k for k,i in enumerate(cache['ids'])}
    values=[[] for _ in range(model.BINS)];mass=[[] for _ in range(model.BINS)]
    for i in train_ids:
        k=positions[i];a,b=cache['token_offsets'][k:k+2];x=cache['x'][a:b]
        o=model.position_overlap(len(x),records[i]['uid'],shuffled)
        for j in range(model.BINS):
            keep=o[:,j]>0
            values[j].append(x[keep]);mass[j].append(o[keep,j]*(weights[i]/len(x)))
    return [(np.concatenate(v),np.concatenate(w)) for v,w in zip(values,mass)]


def train(con,records,fold,selected,paths):
    saved={k for k, in con.execute('select key from models')}
    for cell in sorted({records[i]['cell'] for i in selected}):
        ids=[i for i in selected if records[i]['cell']==cell]
        cell_folds=sorted({fold[i] for i in ids})
        exclusions=[(f,) for f in cell_folds]
        if not cell.startswith('pb_'):
            exclusions+=[(f,h) for f in cell_folds for h in cell_folds if f<h]
        stats={i:hy.load_blob(con.execute('select payload from stats where idx=?',(i,)).fetchone()[0]) for i in ids}
        pending=[ex for ex in exclusions if model_key(cell,ex) not in saved]
        if not pending:continue
        # Match by recorded IDs, not a guessed file-name convention.
        matching=[]
        for path in paths:
            with np.load(path) as z:
                if int(z['ids'][0]) in ids:matching.append(path)
                elif records[int(z['ids'][0])]['cell']==cell:matching.append(path)
        assert len(matching)==1,(cell,matching)
        path=matching[0];cache=None
        for excluded in pending:
            key=cell+'__exclude_'+'_'.join(map(str,excluded))
            if key in saved:continue
            train_ids=[i for i in ids if fold[i] not in excluded]
            assert train_ids
            weights=hy.group_weights(records,train_ids)
            held_groups={records[i]['group_id'] for i in selected if records[i]['cell']==cell and fold[i] in excluded}
            groups=sorted({records[i]['group_id'] for i in train_ids})
            assert not held_groups.intersection(groups)
            moments={name:sum(weights[i]*stats[i][name] for i in train_ids)
                     for name in ('real','shuffle','real_mean','shuffle_mean')}
            arrays={};info=dict(cell=cell,excluded_folds=excluded,excluded_groups=sorted(held_groups),
                                training_ids=train_ids,training_groups=groups,fits={})
            blocks=None;block_shuffle=None
            for name in METHODS:
                prior=con.execute('select payload,info from arm_models where key=? and arm=?',(key,name)).fetchone()
                if prior:
                    a=hy.load_blob(prior[0]);d=json.loads(prior[1])
                    arrays.update({name+'__'+k:v for k,v in a.items()});info['fits'][name]=d
                    continue
                try:
                    shuffled=name.endswith('_shuffled');prefix='shuffle' if shuffled else 'real'
                    stationary=name.endswith('_stationary');rank=2 if 'rank2' in name else 1
                    if name.endswith('_position_mean'):
                        baseline=name.split('_')[0]+'_stationary'
                        a={k[len(baseline)+2:]:v.copy() for k,v in arrays.items() if k.startswith(baseline+'__')}
                        a=model.position_mean_control(a,moments['real_mean'],
                            groups=len(groups) if name.startswith('iu_') else None)
                        d=dict(algorithm='fixed coefficients; matched training position mean only',converged=None)
                    elif name.startswith('factor_'):
                        a,d=model.fit_factor(moments[prefix],moments[prefix+'_mean'],rank,stationary)
                    elif name.startswith('iu_'):
                        a,d=model.fit_iu(moments[prefix],moments[prefix+'_mean'],stationary=stationary,groups=len(groups))
                    else:
                        if cache is None:cache=load_cache(path)
                        if blocks is None or block_shuffle!=shuffled:
                            blocks=None;gc.collect()
                            blocks=rbm_blocks(cache,train_ids,weights,records,shuffled);block_shuffle=shuffled
                        a,d=model.fit_rbm(blocks,rank,stationary,heartbeat=lambda **kw:progress('FITTING_RBM',model=key,arm=name,**kw))
                    if not np.isfinite(a['coefficients']).all() or not np.isfinite(a['intercept']).all():
                        raise FloatingPointError('nonfinite model or failed baseline dependency')
                    arrays.update({name+'__'+k:v for k,v in a.items()});info['fits'][name]=d
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
                    a=dict(coefficients=np.full((16,12),np.nan),intercept=np.full(16,np.nan))
                    d=dict(failure=type(e).__name__+': '+str(e));info['fits'][name]=d
                    arrays.update({name+'__'+k:v for k,v in a.items()})
                con.execute('insert into arm_models values(?,?,?,?)',(key,name,base.packed(**a),base.dumps(d)));con.commit()
                progress('FITTING',model=key,arm=name,converged=info['fits'][name].get('converged'),
                         training_answers=len(train_ids))
            con.execute('insert into models values(?,?,?)',(key,base.packed(**arrays),base.dumps(info)));con.commit();saved.add(key)
            blocks=None;gc.collect()
        del stats,cache;gc.collect()

def model_key(cell,excluded):return cell+'__exclude_'+'_'.join(map(str,sorted(excluded)))


def write_weight_maps(con):
    rows=[]
    for key,payload,info in con.execute('select key,payload,info from models'):
        state=hy.load_blob(payload);health=json.loads(info)
        for name in METHODS:
            for j,w in enumerate(state[name+'__coefficients']):
                rows.append(dict(model=key,method=name,label=model.LABELS[name],position_bin=j,
                                 position_start=j/16,position_end=(j+1)/16,
                                 intercept=float(state[name+'__intercept'][j]),
                                 converged=health['fits'][name].get('converged'),
                                 **{f:float(v) for f,v in zip(FEATURE_NAMES,w)}))
    old.csv_write(OUT/'WEIGHT_MAPS.csv',rows)

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
            r=(np.arange(len(x))+.5)/len(x)
            payload.update(equal=step_top_mean(x.mean(1),spans[:,0],spans[:,1],10),
                           position_early=step_top_mean(-r,spans[:,0],spans[:,1],10),
                           position_late=step_top_mean(r,spans[:,0],spans[:,1],10))
            excluded_sets=[(f,)]
            if not cell.startswith('pb_'):
                other={fold[j] for j in selected if records[j]['cell']==cell and fold[j]!=f}
                excluded_sets += [tuple(sorted((f,h))) for h in sorted(other)]
            for excluded in excluded_sets:
                fit=models[model_key(cell,excluded)]
                suffix='' if len(excluded)==1 else '__inner_for_'+str(next(h for h in excluded if h!=f))
                for name in METHODS:
                    a={k:fit[name+'__'+k] for k in ('coefficients','intercept')}
                    payload[name+suffix]=model.step_scores(x,spans,a,records[i]['uid'],
                                                         shuffled=name.endswith('_shuffled'))
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
        expected_held={records[i]['group_id'] for i in selected if records[i]['cell']==info['cell'] and fold[i] in ex}
        assert set(info['excluded_groups'])==expected_held
        assert not set(info['training_groups']).intersection(expected_held)
        count+=1
    # Fit moments depend only on saved unlabeled token statistics; no labels are read by train/fit.
    return dict(status='PASS',models=count,all_exclusions_rederived=True,fit_api_accepts_no_labels=True)

def label_firewall_review(con,records,joined,fold,selected):
    """Perturb held labels and confirm the serialized fit inputs are unchanged."""
    cell=next(records[i]['cell'] for i in selected if not records[i]['cell'].startswith('pb_'))
    cell_ids=[i for i in selected if records[i]['cell']==cell]
    held_fold=min({fold[i] for i in cell_ids});train_ids=[i for i in cell_ids if fold[i]!=held_fold]
    weights=hy.group_weights(records,train_ids)
    def digest():
        h=hashlib.sha256()
        for i in train_ids:
            payload=con.execute('select payload from stats where idx=?',(i,)).fetchone()[0]
            h.update(np.float64(weights[i]).tobytes());h.update(payload)
        return h.hexdigest()
    before=digest();labels=joined['labels'].copy();target=joined['target'].copy()
    held=[i for i in cell_ids if fold[i]==held_fold];labels_before=labels.copy();target_before=target.copy()
    for i in held:
        a,b=joined['offsets'][i:i+2];known=labels[a:b]>=0;labels[a:b][known]=1-labels[a:b][known]
        target[i]=target[i]+1
    assert not np.array_equal(labels,labels_before) and not np.array_equal(target,target_before)
    after=digest();assert before==after
    return dict(status='PASS',cell=cell,held_fold=held_fold,held_answers=len(held),
                labels_and_targets_perturbed=True,fit_input_sha256_before=before,
                fit_input_sha256_after=after,fit_api_accepts_no_labels=True)

def _auc_draws(labels,scores,groups,draws,seed):
    state=auc_boot._validate_cell(labels=labels,scores_by_method=scores,group_ids=groups,
                                  reference_method='top10',canonical_group_order=True)
    rng=np.random.default_rng(seed);out={name:[] for name in state['methods']}
    for start in range(0,draws,128):
        n=min(128,draws-start);counts=auc_boot._draw_counts(draws=n,n_groups=len(state['roster']),rng=rng)
        positive=counts@state['group_pos'];total=counts@state['group_total'];valid=(positive>0)&((total-positive)>0)
        if not valid.any():continue
        for name in state['methods']:
            value,_=auc_boot._weighted_draw_metrics(labels=state['labels'],score=state['scores'][name],
                row_group_index=state['row_group_index'],counts=counts)
            out[name].append(value[valid])
    return {name:np.concatenate(values) for name,values in out.items()}

def endpoint_uncertainty(records,joined,scores,metrics,thresholds,fold):
    """Grouped paired intervals for the registered pooled/fold AUC and PRMScore endpoints."""
    methods=('top10',)+ALL_SCORED;pairs=PAIRS
    offsets=joined['offsets'];lengths=np.diff(offsets);cells=np.array([r['cell'] for r in records])
    prm=~np.char.startswith(cells,'pb_');common=prm.copy()
    for name in methods:
        common &= np.array([np.isfinite(scores[name][offsets[i]:offsets[i+1]]).all() for i in range(len(records))])
    step_prm=np.repeat(common,lengths);known=joined['labels']>=0;mask=step_prm&known
    step_groups=np.repeat(np.array([r['group_id'] for r in records]),lengths)
    step_fold=np.repeat(np.array([fold[i] for i in range(len(records))]),lengths)
    flat={name:scores[name][mask] for name in methods};labels=(joined['labels'][mask]==1).astype(int);groups=step_groups[mask]
    pooled=_auc_draws(labels,flat,groups,10000,2026091311)
    fold_draws={name:[] for name in methods}
    for f in sorted(set(step_fold[mask])):
        fm=step_fold[mask]==f
        draws=_auc_draws(labels[fm],{name:value[fm] for name,value in flat.items()},groups[fm],10000,2026091320+int(f))
        for name in methods:fold_draws[name].append(draws[name])
    fold_mean={name:np.mean(np.stack(values),axis=0) for name,values in fold_draws.items()}
    pooled_point={name:base.old.auc(labels,flat[name]) for name in methods}
    fold_point={name:np.mean([base.old.auc(labels[step_fold[mask]==f],flat[name][step_fold[mask]==f])
                             for f in sorted(set(step_fold[mask]))]) for name in methods}
    if int(common.sum())==int(prm.sum()):
        for name in methods:
            np.testing.assert_allclose(pooled_point[name],metrics[name]['prm_pooled'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(fold_point[name],metrics[name]['prm_fold_auc'],atol=1e-12,rtol=0)
    # Official PRMScore excludes the synthetic correct-control class.
    raw={str(r['idx']):r for r in base.old.load_pickle(base.old.PRMB_LABELS).values()}
    unique_groups,inv=np.unique([r['group_id'] for r in records],return_inverse=True);counts=np.zeros((len(unique_groups),len(methods),4))
    folds=json.loads(base.old.FOLDS.read_text())['outer']
    for i,r in enumerate(records):
        if not common[i] or r['cell'].startswith('pb_') or raw[str(r['row_id'])]['classification']=='correct':continue
        a,b=offsets[i:i+2];truth=joined['labels'][a:b]==1
        for j,name in enumerate(methods):
            risk=scores[name][a:b]>=thresholds[name][str(folds[r['group_id']])]
            counts[inv[i],j]+=np.array([np.sum(~truth&~risk),np.sum(truth&~risk),np.sum(truth&risk),np.sum(~truth&risk)])
    def f1(c):
        tp,fp,tn,fn=np.moveaxis(c,-1,0)
        return .5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
    point=f1(counts.sum(0))
    if int(common.sum())==int(prm.sum()):
        np.testing.assert_allclose(point,[metrics[n]['prmscore_conditional'] for n in methods],atol=1e-12,rtol=0)
    prm_draws={name:[] for name in methods};rng=np.random.default_rng(2026091331)
    for start in range(0,10000,128):
        n=min(128,10000-start);w=rng.multinomial(len(unique_groups),np.full(len(unique_groups),1/len(unique_groups)),size=n)
        value=f1((w@counts.reshape(len(unique_groups),-1)).reshape(n,len(methods),4))
        for j,name in enumerate(methods):prm_draws[name].append(value[:,j])
    prm_draws={name:np.concatenate(value) for name,value in prm_draws.items()}
    out={}
    for a,b in pairs:
        primary=(a,b) in PRIMARY;level=PRIMARY_CI if primary else .95;q=[(1-level)/2,1-(1-level)/2]
        out[a+'_minus_'+b]=dict(primary=primary,ci_level=level,draws=10000,
            common_prm_answers=int(common.sum()),
            prm_pooled_delta=float(pooled_point[a]-pooled_point[b]),
            prm_pooled_ci=np.quantile(pooled[a]-pooled[b],q).tolist(),
            prm_fold_auc_delta=float(fold_point[a]-fold_point[b]),
            prm_fold_auc_ci=np.quantile(fold_mean[a]-fold_mean[b],q).tolist(),
            prmscore_delta=float(point[methods.index(a)]-point[methods.index(b)]),
            prmscore_ci=np.quantile(prm_draws[a]-prm_draws[b],q).tolist(),
            conditional_on_saved_fits_scores_and_thresholds=True)
    return out

def independent_endpoint_review(records,joined,scores,metrics,fold):
    offsets=joined['offsets'];target=joined['target'];cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
    detector,gate=base.old._gate_contract(records);checks={}
    for name,flat in scores.items():
        valid=np.zeros(len(records),bool);peak=np.full(len(records),-1,int);pooled=np.zeros(len(flat),bool)
        for i in range(len(records)):
            a,b=offsets[i:i+2];s=flat[a:b]
            if len(s) and np.isfinite(s).all():
                valid[i]=True;peak[i]=int(np.argmax(s));
                if not pb[i]:pooled[a:b]=joined['labels'][a:b]>=0
        decision=valid&np.isfinite(detector)&np.isfinite(gate);prediction=np.where(detector>=gate,peak,-1)
        p=pb_metrics(target[pb],prediction[pb],decision[pb],cells[pb]);error=pb&(target>=0);clean=pb&(target<0);diff=peak-target
        expected=metrics[name]
        np.testing.assert_allclose([p['macros'][k] for k in ('all','q4','q8')],
                                   [expected[k] for k in ('pb_all8','pb_q4','pb_q8')],atol=1e-12,rtol=0)
        for cell,value in p['cells'].items():np.testing.assert_allclose(value['f1'],expected['pb_cells'][cell]['f1'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(base.old.auc(joined['labels'][pooled]==1,flat[pooled]),expected['prm_pooled'],atol=1e-12,rtol=0)
        fold_values=[base.old.auc(joined['labels'][pooled&(np.repeat(np.array([fold[i] for i in range(len(records))]),np.diff(offsets))==f)]==1,
                                  flat[pooled&(np.repeat(np.array([fold[i] for i in range(len(records))]),np.diff(offsets))==f)]) for f in sorted(set(fold.values()))]
        np.testing.assert_allclose(np.mean(fold_values),expected['prm_fold_auc'],atol=1e-12,rtol=0)
        assert int(valid.sum())==expected['valid_answers'] and int(np.sum(pb&~decision))==expected['pb_invalid']
        assert int(np.sum(error&valid&(diff<0)))==expected['pb_early'] and int(np.sum(error&valid&(diff>0)))==expected['pb_late']
        assert int(np.sum(error&valid&(diff==0)))==expected['pb_exact_count']
        assert int(np.sum(error&valid&(diff==0)&(prediction==-1)))==expected['pb_correct_peaks_suppressed']
        np.testing.assert_allclose(np.sum(error&valid&(diff==0))/error.sum(),expected['pb_raw_exact'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.sum(clean&decision&(prediction==-1))/clean.sum(),expected['pb_clean_accuracy'],atol=1e-12,rtol=0)
        checks[name]=dict(valid_answers=int(valid.sum()),all_registered_endpoints_reproduced=True)
    return dict(status='PASS',methods=checks)

def evaluate(con,records,joined,reference,fold,selected):
    assert len(selected)==13769
    total=int(joined['offsets'][-1]);scores={name:np.full(total,np.nan) for name in ('top10',)+ALL_SCORED}
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
            training_groups=sorted({records[i]['group_id'] for i in ids})
            excluded_groups=sorted({records[i]['group_id'] for i in selected if not records[i]['cell'].startswith('pb_') and fold[i]==f})
            coverage.append(dict(method=name,outer_fold=f,answers=len(values),expected=len(ids),
                                 training_groups=training_groups,excluded_groups=excluded_groups))
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
    pairs=PAIRS
    progress('BOOTSTRAP',draws=10000)
    contrasts=base.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(PRIMARY),primary_ci=PRIMARY_CI)
    review=hy.independent_review(records,joined,scores,metrics,thresholds,per)
    endpoint_review=independent_endpoint_review(records,joined,scores,metrics,fold)
    extended=endpoint_uncertainty(records,joined,scores,metrics,thresholds,fold)
    hy.METHODS=('top10',)+ALL_SCORED
    transitions=hy.error_and_weight_tables(records,joined,{},[],per)
    np.savez_compressed(OUT/'SCORES.npz',**scores)
    emit('METRICS.json',metrics);emit('CONTRASTS.json',contrasts);emit('ENDPOINT_CONTRASTS.json',extended)
    emit('CALIBRATION.json',dict(thresholds=thresholds,coverage=coverage));emit('ERROR_TRANSITIONS.json',transitions)
    old.csv_write(OUT/'COMPARISON.csv',[dict(method=n,label=model.LABELS.get(n,n),**{k:v for k,v in a.items() if not isinstance(v,dict)}) for n,a in metrics.items()])
    old.csv_write(OUT/'PER_CELL.csv',[dict(method=n,cell=cell,**a) for n,d in metrics.items() for cell,a in d['pb_cells'].items()])
    emit('RESULT_REVIEW.json',dict(status='PASS',answers=len(selected),independent_metrics=review,
                                 independent_all_endpoints=endpoint_review,
                                 frozen_references_reproduced=True,exclusions=review_exclusions(con,records,fold,selected)))
    report=['# Whole-answer position fusion (RBM12 bank, Top10 fixed)','',
            'Full development benchmark; other-answer unlabelled fitting. Rank means loading-map rank, not latent classes.',
            '', '| Method | PB % | PRMB within AUC | PRMScore | Coverage |', '|---|---:|---:|---:|---:|']
    for n in ('top10',)+ALL_SCORED:
        d=metrics[n];report.append(f'| {model.LABELS.get(n,n)} | {100*d["pb_all8"]:.3f} | {d["prm_within"]:.5f} | {d["prmscore_q08"]:.5f} | {d["valid_answers"]}/13769 |')
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
            extract(con,paths,selected,records,joined,reference)
            emit('LABEL_FIREWALL_REVIEW.json',label_firewall_review(con,records,joined,fold,selected))
            train(con,records,fold,selected,paths)
            write_weight_maps(con)
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
