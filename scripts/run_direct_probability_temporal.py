"""Full cached localization test of temporal probability fusion; resumable, no HTML."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_fusion_v2 as old
from spectral_utils.direct_probability_fusion_v2 import augmented_probability_risk
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.direct_probability_temporal import METHODS, fit_all, seed_for
from spectral_utils.historical_fusion_evaluation import pb_metrics
from spectral_utils.prmbench import prmbench_evaluate

OUT = ROOT / 'results/direct_probability_temporal_v3'
REFERENCES = {'current__equal':'augmented_equal', 'current__iu':'augmented_iu', 'current__joint_lw':'augmented_joint_lw'}


def dumps(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(dumps(payload)+'\n',encoding='utf8');os.replace(tmp,path)


def packed(**arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()


def worker_init():
    global THREAD_LIMIT
    THREAD_LIMIT=threadpool_limits(limits=1)


def score_one(task):
    index,uid,X,entropy,spans,reference = task
    started=time.perf_counter()
    fits,failures,seconds=fit_all(X,entropy,uid=uid)
    S=np.full((len(spans),len(METHODS)),np.nan)
    W=np.full((len(METHODS),136),np.nan)
    diagnostics={}
    tokens={}
    for j,name in enumerate(METHODS):
        if name not in fits:
            continue
        fit=fits[name]
        S[:,j]=step_top_mean(fit.score,spans[:,0],spans[:,1],count=10)
        W[j,:len(fit.weights)]=fit.weights
        diagnostics[name]=fit.diagnostics
        if seed_for(uid)%251==0:
            tokens[name]=fit.score
    # Exact saved v2 remains a separate reference. Any pure-fit failure is
    # explicit; no substitute scores are inserted into this new method.
    replay={}
    for name,expected in reference.items():
        actual=S[:,METHODS.index(name)]
        replay[name]=None if not np.isfinite(actual).all() else float(np.max(np.abs(actual-expected)))
        if replay[name] is not None:
            np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-9,err_msg=f'{uid} {name} baseline drift')
    info=dict(uid=uid,n_tokens=len(X),failures=failures,seconds=seconds,
        wall_seconds=time.perf_counter()-started,diagnostics=diagnostics,baseline_max_error=replay)
    return index,packed(steps=S,weights=W,**{f'tokens__{k}':v for k,v in tokens.items()}),dumps(info)


def input_manifest(source, v2):
    audit=json.loads((v2/'results/direct_probability_fusion_v2_selected_tail/DATA_AUDIT.json').read_text(encoding='utf8'))
    files={}
    for item in audit['localization']+[audit['prmbench_label_join']]:
        path=source/item['artifact']
        print('[verify]',item['artifact'],flush=True)
        digest=old.sha256_file(path)
        if digest!=item['sha256']:
            raise ValueError(f'frozen raw source drift: {path}')
        files[str(path)]=digest
    small=[old.BENCH/'evaluation/JOINED.json',old.BENCH/'evaluation/JOINED.npz',
        old.FIXED_GATE/'DETECTORS.npz',old.FIXED_GATE/'METRICS.json',old.FOLDS,old.STEP334,
        v2/'results/direct_probability_fusion_v2_selected_tail/LOCALIZATION_SCORES.npz',
        v2/'results/direct_probability_fusion_v2_selected_tail/LOCALIZATION.json']
    freeze=json.loads((v2/'docs/experiments/DIRECT_PROBABILITY_FUSION_V2_SELECTED_TAIL_INPUT_FREEZE.json').read_text(encoding='utf8'))
    # Source hashes are also bound on every resume; benchmark equality below
    # is checked directly against saved v2 predictions and identifiers.
    for path in small:
        digest=old.sha256_file(path)
        if path.is_relative_to(source) and not path.is_relative_to(v2):
            relative=path.relative_to(source).as_posix()
            if relative not in freeze['source_files'] or digest!=freeze['source_files'][relative]:
                raise ValueError(f'v2 benchmark/fold/gate freeze mismatch: {relative}')
        files[str(path)]=digest
    for path in [Path(__file__),ROOT/'docs/experiments/DIRECT_PROBABILITY_TEMPORAL_V3.md',
        *[ROOT/'spectral_utils'/n for n in ('direct_probability_temporal.py','direct_probability_fusion.py',
          'direct_probability_fusion_v2.py','upcr.py','laplacian_upcr.py','shrinkage_iu.py',
          'historical_fusion_evaluation.py','prmbench.py')],ROOT/'scripts/run_direct_probability_fusion_v2.py']:
        files[str(path)]=old.sha256_file(path)
    return dict(schema='temporal-probability-v3',methods=list(METHODS),hashes=files,source_root=str(source),v2_root=str(v2))


def connect(path, manifest):
    path.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(path)
    con.execute('CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
    con.execute('CREATE TABLE IF NOT EXISTS answers (idx INTEGER PRIMARY KEY, payload BLOB NOT NULL, info TEXT NOT NULL)')
    existing=con.execute('SELECT payload FROM manifest WHERE id=1').fetchone()
    if existing and json.loads(existing[0])!=manifest:
        con.close()
        raise ValueError('checkpoint code/input/protocol manifest mismatch; do not overwrite frozen checkpoint')
    if not existing:
        con.execute('INSERT INTO manifest VALUES (1,?)',(dumps(manifest),));con.commit()
    return con


def source_specs():
    specs=[(f'pb_{d}_{m}',p/f'processbench_{d}.pkl','pb',d) for m,p in old.PB_DIRS.items()
        for d in ('gsm8k','math','olympiadbench','omnimath')]
    return specs+[('prmbench_qwen3_8b',old.PRMB_TELEMETRY,'prm',None)]


def scoring(con, records, joined, reference, *, workers, smoke, max_answers):
    done={r[0] for r in con.execute('SELECT idx FROM answers')}
    offsets=joined['offsets'];detector,_=old._gate_contract(records)
    processed=0;started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init) as pool:
        for cell,path,kind,dataset in source_specs():
            indexes=[i for i,r in enumerate(records) if r['cell']==cell and i not in done]
            if not indexes:continue
            print('[load]',cell,len(indexes),'remaining',flush=True)
            source=old._source_row_map(old.load_pickle(path),kind=kind,dataset=dataset)
            if smoke:
                order=sorted(indexes,key=lambda i:len(source[records[i]['row_id']]['token_entropies']))
                indexes=[order[j] for j in sorted({0,len(order)//2,min(len(order)-1,int(.95*len(order)))})]
            for start in range(0,len(indexes),32):
                batch=[]
                for i in indexes[start:start+32]:
                    r=records[i];row=source[r['row_id']]
                    entropy=np.asarray(row['token_entropies'],float)
                    X=augmented_probability_risk(old._topk_payload(row),row['token_spilled_energies'],k=15)
                    if len(X)!=len(entropy) or not np.isfinite(entropy).all():raise ValueError('raw token alignment')
                    if kind=='pb' and not np.isclose(entropy.mean(),detector[i],atol=1e-12,rtol=0):
                        raise ValueError(f'gate source row mismatch {r["uid"]}')
                    spans=np.asarray(row['step_token_spans'],int)
                    if spans.shape!=(r['steps'],2):raise ValueError(f'span mismatch {r["uid"]}')
                    ref={k:reference['steps__'+v][offsets[i]:offsets[i+1]] for k,v in REFERENCES.items()}
                    batch.append((i,r['uid'],X,entropy,spans,ref))
                for item in pool.map(score_one,batch,chunksize=1):
                    con.execute('INSERT INTO answers VALUES (?,?,?)',item)
                con.commit();processed+=len(batch);done.update(x[0] for x in batch)
                state=dict(status='SMOKE' if smoke else 'RUNNING',completed=len(done),expected=len(records),
                    last_cell=cell,seconds_this_launch=time.perf_counter()-started,workers=workers)
                atomic_json(OUT/('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'),state)
                print('[checkpoint]',len(done),'/',len(records),'elapsed',round(state['seconds_this_launch'],1),flush=True)
                if max_answers and processed>=max_answers:return
            del source
    if not smoke and len(done)==len(records):
        atomic_json(OUT/'RUN_STATE.json',dict(status='SCORING_COMPLETE',completed=len(done),expected=len(records)))


def load_scored(con, records, offsets):
    rows=con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
    if rows!=len(records):raise ValueError(f'full evaluation requires {len(records)} answers, got {rows}')
    scores={m:np.full(int(offsets[-1]),np.nan) for m in METHODS}
    telemetry={m:dict(failures=[],fit_seconds=0.,alphas=[],weights=[]) for m in METHODS}
    for i,blob,info_json in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        info=json.loads(info_json)
        if info['uid']!=records[i]['uid']:raise ValueError('checkpoint ID mismatch')
        with np.load(io.BytesIO(blob),allow_pickle=False) as row:
            if row['steps'].shape!=(int(offsets[i+1]-offsets[i]),len(METHODS)):raise ValueError('checkpoint shape')
            for j,m in enumerate(METHODS):
                scores[m][offsets[i]:offsets[i+1]]=row['steps'][:,j]
                t=telemetry[m];t['fit_seconds']+=info['seconds'][m]
                if m in info['failures']:t['failures'].append(dict(uid=info['uid'],reason=info['failures'][m]))
                else:
                    t['weights'].append(row['weights'][j].copy())
                    alpha=info['diagnostics'][m].get('alpha')
                    if alpha is not None:t['alphas'].append(alpha)
    return scores,telemetry


def evaluate_arrays(records, joined, scores):
    offsets,labels,target=joined['offsets'],joined['labels'],joined['target']
    cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
    detector,thresholds=old._gate_contract(records)
    folds=json.loads(old.FOLDS.read_text(encoding='utf8'))['outer']
    outer=np.array([int(folds[r['group_id']]) for r in records])
    prm_label_map={str(r['idx']):r for r in old.load_pickle(old.PRMB_LABELS).values()}
    metrics,per={},{}
    for name,flat in scores.items():
        valid=np.zeros(len(records),bool);peak=np.full(len(records),-1,int);within=np.full(len(records),np.nan)
        pooled=np.zeros(len(flat),bool)
        for i in range(len(records)):
            sl=slice(offsets[i],offsets[i+1]);s=flat[sl]
            if not len(s) or not np.isfinite(s).all():continue
            valid[i]=True;peak[i]=int(np.argmax(s))
            if not pb[i]:
                usable=labels[sl]>=0;y=labels[sl][usable]==1
                pooled[sl]=usable
                if y.any() and (~y).any():within[i]=old.auc(y,s[usable])
        decision_valid=valid & np.isfinite(detector) & np.isfinite(thresholds)
        pred=np.where(detector>=thresholds,peak,-1)
        result=pb_metrics(target[pb],pred[pb],decision_valid[pb],cells[pb])
        error=pb & (target>=0);clean=pb & (target<0);diff=peak-target
        m=dict(pb_all8=result['macros']['all'],pb_q4=result['macros']['q4'],pb_q8=result['macros']['q8'],
            pb_cells=result['cells'],valid_answers=int(valid.sum()),prm_valid_answers=int((valid & ~pb).sum()),
            pb_clean_accuracy=float(np.sum(clean & decision_valid & (pred==-1))/clean.sum()),
            pb_raw_exact=float(np.sum(error & valid & (diff==0))/error.sum()),
            pb_early=int(np.sum(error & valid & (diff<0))),pb_late=int(np.sum(error & valid & (diff>0))),
            pb_exact_count=int(np.sum(error & valid & (diff==0))),pb_error_count=int(error.sum()),
            pb_invalid=int(np.sum(pb & ~decision_valid)),
            pb_correct_peaks_suppressed=int(np.sum(error & valid & (diff==0) & (pred==-1))),
            prm_within=float(np.nanmean(within)) if np.isfinite(within).any() else None,
            prm_within_n=int(np.isfinite(within).sum()),prm_pooled=old.auc(labels[pooled]==1,flat[pooled]) if pooled.any() else None,
            prm_pooled_steps=int(pooled.sum()))
        # Invalid answers are never relabeled clean. A partial PRMScore is
        # explicitly conditional and the headline full PRMScore is unavailable.
        predicted={};q_by_fold={}
        for f in sorted(set(outer[~pb])):
            train=np.flatnonzero(~pb & valid & (outer!=f));test=np.flatnonzero(~pb & valid & (outer==f))
            if not len(train):continue
            q=float(np.quantile(np.concatenate([flat[offsets[i]:offsets[i+1]] for i in train]),.8))
            q_by_fold[str(f)]=q
            assert not set(records[i]['group_id'] for i in train)&set(records[i]['group_id'] for i in test)
            for i in test:predicted[i]=(~(flat[offsets[i]:offsets[i+1]]>=q)).astype(int).tolist()
        prm=prmbench_evaluate([dict(idx=records[i]['row_id'],labels=p) for i,p in predicted.items()],
            [prm_label_map[str(records[i]['row_id'])] for i in predicted]) if predicted else None
        value=.5*(prm['total']['f1']+prm['total']['negative_f1']) if prm else None
        m.update(prmscore_q08=value if len(predicted)==int((~pb).sum()) else None,
            prmscore_conditional=value,prmscore_answers=len(predicted),prmscore_thresholds=q_by_fold)
        metrics[name]=m
        per[name]=dict(valid=valid,decision_valid=decision_valid,prediction=pred,peak=peak,within=within)
    return metrics,per


def paired_bootstrap(records, joined, per, *, draws=10000, pairs=None, primary_pairs=None):
    cells=np.array([r['cell'] for r in records]);target=joined['target']
    groups,inv=np.unique([r['group_id'] for r in records],return_inverse=True);ng=len(groups)
    base='current__iu'
    if pairs is None:
        pairs=[(m,base) for m in METHODS if m!=base]+[
            ('lag8__iu','shuffled_lag8__iu'),('current__chain_liu','current__permuted_chain_liu')]
    if primary_pairs is None:
        primary_pairs={('lag8__iu',base),('delta__iu',base)}
    names=sorted({m for pair in pairs for m in pair});pb_cells=sorted(set(cells[np.char.startswith(cells,'pb_')]))
    counts=np.zeros((ng,8,2));success=np.zeros((ng,len(names),8,2))
    for c,cell in enumerate(pb_cells):
        for k,mask in enumerate(((cells==cell)&(target<0),(cells==cell)&(target>=0))):
            counts[:,c,k]=np.bincount(inv[mask],minlength=ng)
            for j,m in enumerate(names):
                hit=mask & per[m]['decision_valid'] & (per[m]['prediction']==target)
                success[:,j,c,k]=np.bincount(inv[hit],minlength=ng)
    within_num=np.zeros((ng,len(pairs)));within_den=np.zeros_like(within_num)
    out={}
    for j,(a,b) in enumerate(pairs):
        mask=np.isfinite(per[a]['within']) & np.isfinite(per[b]['within'])
        difference=per[a]['within'][mask]-per[b]['within'][mask]
        within_num[:,j]=np.bincount(inv[mask],weights=difference,minlength=ng)
        within_den[:,j]=np.bincount(inv[mask],minlength=ng)
        out[a+'_minus_'+b]=dict(primary=((a,b) in primary_pairs),
            common_prm_answers=int(mask.sum()),prm_within_delta_common=float(difference.mean()) if len(difference) else None)
    pb_draws=[[] for _ in pairs];within_draws=[[] for _ in pairs]
    rng=np.random.default_rng(20260910136)
    for start in range(0,draws,128):
        n=min(128,draws-start);W=rng.multinomial(ng,np.full(ng,1/ng),size=n).astype(float)
        den=(W@counts.reshape(ng,-1)).reshape(n,8,2)
        num=(W@success.reshape(ng,-1)).reshape(n,len(names),8,2)
        rates=np.divide(num,den[:,None,:,:],out=np.full_like(num,np.nan),where=den[:,None,:,:]>0)
        ca,ea=rates[:,:,:,0],rates[:,:,:,1]
        cell_f1=np.divide(2*ca*ea,ca+ea,out=np.zeros_like(ca),where=ca+ea>0)
        cell_f1[~np.isfinite(ca) | ~np.isfinite(ea)]=np.nan
        f1=cell_f1.mean(axis=2)
        wd=W@within_den;wn=W@within_num
        wa=np.divide(wn,wd,out=np.full_like(wn,np.nan),where=wd>0)
        for j,(a,b) in enumerate(pairs):
            pb_draws[j].extend((f1[:,names.index(a)]-f1[:,names.index(b)]).tolist())
            within_draws[j].extend(wa[:,j].tolist())
    for j,(a,b) in enumerate(pairs):
        o=out[a+'_minus_'+b];q=[1.25,98.75] if o['primary'] else [2.5,97.5]
        wd=np.asarray(within_draws[j]);wd=wd[np.isfinite(wd)]
        o.update(ci_level=.975 if o['primary'] else .95,pb_ci=np.nanpercentile(pb_draws[j],q).tolist(),
            prm_within_ci=np.percentile(wd,q).tolist() if len(wd) else None,bootstrap_draws=draws,
            prm_valid_bootstrap_draws=len(wd))
    return out


def evaluation(con,records,joined,reference,v2,draws):
    scores,telemetry=load_scored(con,records,joined['offsets'])
    scores['entropy']=reference['steps__entropy']
    for k,v in REFERENCES.items():scores['saved_v2__'+v]=reference['steps__'+v]
    metrics,per=evaluate_arrays(records,joined,scores)
    previous=json.loads((v2/'results/direct_probability_fusion_v2_selected_tail/LOCALIZATION.json').read_text(encoding='utf8'))
    for m,ref in [('entropy','entropy')]+[('saved_v2__'+v,v) for v in REFERENCES.values()]:
        for key in ('pb_all8','prm_within','prm_pooled','prmscore_q08'):
            np.testing.assert_allclose(metrics[m][key],previous['methods'][ref][key],atol=1e-12,rtol=0)
    print('[evaluate] full-population continuity verified; bootstrap',draws,flush=True)
    contrasts=paired_bootstrap(records,joined,per,draws=draws)
    for key,c in contrasts.items():
        a,b=key.split('_minus_');c['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    for m,t in telemetry.items():
        weights=np.stack(t.pop('weights')) if t['weights'] else np.zeros((0,136))
        # Preserve padding as null, not a fictitious learned zero coefficient.
        mean=[]
        for j in range(136):
            vals=weights[:,j];vals=vals[np.isfinite(vals)];mean.append(float(vals.mean()) if len(vals) else None)
        alpha=t.pop('alphas');t.update(mean_weights=mean,mean_alpha=float(np.mean(alpha)) if alpha else None)
    payload=dict(schema='temporal-probability-full-localization-v3',n_answers=len(records),
        n_steps=int(joined['offsets'][-1]),metrics=metrics,contrasts=contrasts,telemetry=telemetry,
        historical_references=previous['frozen_references'],mind_gap_reference=previous['comparators'],
        scope='Full cached development. Fusion fits within answer; gates/calibration external. No historical24 temporal result yet.')
    atomic_json(OUT/'METRICS.json',payload)
    np.savez_compressed(OUT/'SCORES.npz',**{'steps__'+m:s for m,s in scores.items()},
        **{'prediction__'+m:p['prediction'] for m,p in per.items()},
        **{'valid__'+m:p['valid'] for m,p in per.items()})
    lines=['method,PB_all8,PB_Q4,PB_Q8,PRMB_within,PRMB_pooled,PRMScore,valid_answers']
    for m,x in metrics.items():
        lines.append(','.join(str(v) for v in [m,x['pb_all8'],x['pb_q4'],x['pb_q8'],x['prm_within'],x['prm_pooled'],x['prmscore_q08'],x['valid_answers']]))
    (OUT/'SUMMARY.csv').write_text('\n'.join(lines)+'\n',encoding='utf8')
    atomic_json(OUT/'RUN_STATE.json',dict(status='COMPLETE',completed=len(records),expected=len(records)))
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--v2-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-answers',type=int,default=0);p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--bootstrap',type=int,default=10000)
    args=p.parse_args();source=args.source_root.resolve();v2=args.v2_root.resolve()
    if args.bootstrap!=10000:raise ValueError('full protocol fixes bootstrap draws at 10000')
    old.configure_source_root(source);OUT.mkdir(parents=True,exist_ok=True)
    manifest=input_manifest(source,v2)
    con=connect(OUT/('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'),manifest)
    atomic_json(OUT/('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'),manifest)
    records=json.loads((old.BENCH/'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined=np.load(old.BENCH/'evaluation/JOINED.npz',allow_pickle=False)
    if len(records)!=13769 or len({r['uid'] for r in records})!=len(records):raise ValueError('full roster mismatch')
    reference=np.load(v2/'results/direct_probability_fusion_v2_selected_tail/LOCALIZATION_SCORES.npz',allow_pickle=False)
    with threadpool_limits(limits=1):
        if not args.evaluate_only:scoring(con,records,joined,reference,workers=args.workers,smoke=args.smoke,max_answers=args.max_answers)
        n=con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
        if not args.smoke and n==len(records):evaluation(con,records,joined,reference,v2,args.bootstrap)
        elif args.smoke:
            stats=[json.loads(r[0]) for r in con.execute('SELECT info FROM answers')]
            atomic_json(OUT/'SMOKE.json',dict(status='FEASIBILITY_ONLY',n_answers=n,rows=stats))
            print('[smoke] no benchmark ranking from subset; per-answer runtime and failures saved',flush=True)
    con.close()


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        name='SMOKE_STATE.json' if '--smoke' in sys.argv else 'RUN_STATE.json'
        state_path=OUT/name
        state=json.loads(state_path.read_text(encoding='utf8')) if state_path.exists() else {}
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}')
        atomic_json(state_path,state)
        raise
