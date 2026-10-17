"""Full saved-state RBM diagnosis. Deliberately contains no model fitting."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import warnings
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_temporal as base
from scripts.run_higher_moment_fusion import atomic_json_retry
from spectral_utils.higher_moment_fusion import representation_order, feature_names
from spectral_utils.direct_probability_fusion import zscore_columns, step_top_mean
from spectral_utils.rbm_data_diagnostics import (
    first_near_max, step_means, strata, diagnose_bank, STRATA, RESIDUAL_NAMES)

OUT = ROOT / 'results/rbm_data_diagnostics_v1'
base.atomic_json = atomic_json_retry
METRIC_KEYS = ('pb_all8','pb_q4','pb_q8','prm_within','prm_pooled','prmscore_q08')
NAMES = {'rbm6':'RBM, six features', 'initial6':'RBM before learning, six features',
         'rbm12':'RBM, twelve features', 'initial12':'RBM before learning, twelve features',
         'entropy':'Entropy', 'var15':'Varentropy, top 15', 'var50':'Varentropy, top 50',
         'var15_equal':'Varentropy contributions, equal weights',
         'var15_iu':'Varentropy contributions, IU-PCR',
         'shrinkage':'RBM with weight shrinkage',
         'diagonal':'RBM with shared diagonal variance',
         'length':'Longest step control', 'random':'Random step control'}


def dirs(source):
    def p(wt, result=None):
        return source/'.worktrees'/wt/'results'/(result or wt.replace('-','_'))
    return dict(higher=p('higher-moment-fusion-v1'),
                var=p('varentropy-contribution-fusion-v1'),
                shrink=p('rbm-weight-shrinkage-v1'),
                diagonal=p('rbm-diagonal-variance-v1'),
                claude=p('readout-provenance-v1','readout_length_control_and_provenance_v1'))


def csv_write(path, rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def frozen_manifest(source):
    m=base.input_manifest(source,source/'.worktrees/direct-probability-fusion-v2')
    m.update(schema='rbm-data-diagnostics-v1',base_commit='44f9bced8',
             no_refit=True,banks=[6,12],replicates=4,bootstrap=10000)
    extra=[Path(__file__),ROOT/'spectral_utils/rbm_data_diagnostics.py',
           ROOT/'spectral_utils/higher_moment_fusion.py',ROOT/'spectral_utils/moment_rbm_fusion.py',
           ROOT/'scripts/test_rbm_data_diagnostics.py',
           ROOT/'scripts/verify_rbm_data_diagnostics.py',
           ROOT/'scripts/summarize_rbm_data_diagnostics.py',
           ROOT/'docs/experiments/RBM_DATA_DIAGNOSTICS_V1.md',
           source/'.worktrees/readout-provenance-v1/spectral_utils/provenance_readout.py']
    for name,d in dirs(source).items():
        state=json.loads((d/'RUN_STATE.json').read_text())
        if state['status']!='COMPLETE':
            raise ValueError('Incomplete source: '+str(d))
        if name!='claude':
            review=json.loads((d/'RESULT_REVIEW.json').read_text())
            if review['status']!='PASS':raise ValueError('Unreviewed source: '+str(d))
            extra += [d/'RESULT_REVIEW.json']
        extra += [d/'MANIFEST.json', d/'METRICS.json', d/'SCORES.npz']
    extra += [dirs(source)['higher']/'CHECKPOINT.sqlite']
    for path in extra:
        m['hashes'][str(path)]=base.old.sha256_file(path)
    return m


def connect(path,manifest):
    con=sqlite3.connect(path)
    con.execute('CREATE TABLE IF NOT EXISTS contract (data TEXT)')
    previous=con.execute('SELECT data FROM contract').fetchone()
    serial=base.dumps(manifest)
    if previous is None:con.execute('INSERT INTO contract VALUES (?)',(serial,))
    elif previous[0]!=serial:raise ValueError('Checkpoint contract changed; preserve old result and identify correction')
    con.execute('CREATE TABLE IF NOT EXISTS answers (idx INTEGER PRIMARY KEY, payload BLOB, info TEXT)')
    con.commit()
    return con


def references(source, records, joined):
    mapping={'rbm6':('higher','d3__rbm'),'initial6':('higher','d3__rbm_initial'),
             'rbm12':('higher','d6__rbm'),'initial12':('higher','d6__rbm_initial'),
             'entropy':('var','entropy'),'var15':('var','k15__raw'),
             'var50':('var','k50__raw'),'var15_equal':('var','k15__equal'),
             'var15_iu':('var','k15__iu'),'shrinkage':('shrink','rbm_shrinkage'),
             'diagonal':('diagonal','rbm_diagonal'),
             'length':('claude','length'),'random':('claude','random_step')}
    scores={};prior={};ds=dirs(source)
    for name,(family,method) in mapping.items():
        with np.load(ds[family]/'SCORES.npz') as z:scores[name+'__old']=z['steps__'+method]
        prior[name]=json.loads((ds[family]/'METRICS.json').read_text())['metrics'][method]
        if name in ('length','random'):continue
        a=scores[name+'__old'].copy()
        for lo,hi in zip(joined['offsets'][:-1],joined['offsets'][1:]):
            if np.isfinite(a[lo:hi]).all():a[lo:hi]=first_near_max(a[lo:hi])
        scores[name+'__near']=a
    # Full-vector Claude replay independently binds step order and exact epsilon.
    with np.load(ds['claude']/'SCORES.npz') as z:
        for our,their in [('entropy','entropy'),('var50','varentropy')]:
            np.testing.assert_allclose(scores[our+'__old'],z['steps__'+their+'_top10'],atol=1e-12,rtol=0)
            claude_old=z['steps__'+their+'_top10'];claude_new=z['steps__'+their+'_first_near_max']
            replay=claude_old.copy()
            for lo,hi in zip(joined['offsets'][:-1],joined['offsets'][1:]):
                replay[lo:hi]=first_near_max(replay[lo:hi])
            np.testing.assert_array_equal(replay,claude_new)
            # Older independently saved top10 reductions differ by up to 9e-16.
            # Exact same-input replay above; numerical comparator continuity here.
            np.testing.assert_allclose(scores[our+'__near'],claude_new,atol=1e-12,rtol=0)
    return scores,prior


def evaluate(source, records, joined, scores, prior):
    print('[evaluate] 24 score arms; unchanged gate and folds',flush=True)
    metrics,per=base.evaluate_arrays(records,joined,scores)
    for name,old in prior.items():
        for key in METRIC_KEYS:
            np.testing.assert_allclose(metrics[name+'__old'][key],old[key],atol=1e-12,rtol=0,
                                       err_msg=f'{name}: {key} original replay')
    cm=json.loads((dirs(source)['claude']/'METRICS.json').read_text())['metrics']
    for our,their in [('entropy','entropy'),('var50','varentropy')]:
        for key in METRIC_KEYS:
            np.testing.assert_allclose(metrics[our+'__near'][key],cm[their+'_first_near_max'][key],atol=1e-12,rtol=0)
    pairs=[(name+'__near',name+'__old') for name in NAMES if name not in ('length','random')]
    pairs += [('rbm6__near','initial6__near'),('rbm12__near','initial12__near'),
              ('rbm6__near','var50__near'),('rbm12__near','var50__near')]
    print('[bootstrap] 10000 canonical-source-group resamples',flush=True)
    contrasts=base.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,
        primary_pairs={('rbm6__near','rbm6__old'),('rbm12__near','rbm12__old')},primary_ci=.975)
    for key,c in contrasts.items():
        a,b=key.split('_minus_');c['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    payload=dict(schema='rbm-data-diagnostics-v1',n_answers=len(records),n_steps=int(joined['offsets'][-1]),
                 metrics=metrics,contrasts=contrasts,names=NAMES,
                 scope='Full cached development; saved answer-local models; external fixed entropy gate. No refit.')
    base.atomic_json(OUT/'METRICS.json',payload)
    np.savez_compressed(OUT/'SCORES.npz',**{'steps__'+m:s for m,s in scores.items()},
        **{'prediction__'+m:p['prediction'] for m,p in per.items()},
        **{'valid__'+m:p['valid'] for m,p in per.items()})
    csv_write(OUT/'SUMMARY.csv',[dict(method=m,name=NAMES[m.split('__')[0]],readout=m.split('__')[1],
        **{key:x[key] for key in METRIC_KEYS},valid_answers=x['valid_answers']) for m,x in metrics.items()])
    csv_write(OUT/'PB_CELLS.csv',[dict(method=m,cell=c,**v) for m,x in metrics.items() for c,v in x['pb_cells'].items()])
    print('[evaluate] original metrics and Claude full replay PASS',flush=True)
    return metrics,per


def scan(source, con, records, joined, scores, smoke):
    started=time.perf_counter();done={x[0] for x in con.execute('SELECT idx FROM answers')}
    src=sqlite3.connect((dirs(source)['higher']/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    detector,_=base.old._gate_contract(records)
    try:
        for cell,path,kind,dataset in base.source_specs():
            indices=[i for i,r in enumerate(records) if r['cell']==cell and i not in done]
            if not indices:continue
            print('[load]',cell,len(indices),flush=True)
            rows=base.old._source_row_map(base.old.load_pickle(path),kind=kind,dataset=dataset)
            if smoke:
                ordered=sorted(indices,key=lambda i:len(rows[records[i]['row_id']]['token_entropies']))
                indices=[ordered[j] for j in sorted({0,len(ordered)//2,int(.95*(len(ordered)-1))})]
            for i in indices:
                r=records[i];row=rows[r['row_id']];uid=r['uid']
                lp=np.asarray(base.old._topk_payload(row)['logprobs'],float)
                entropy=np.asarray(row['token_entropies'],float);spans=np.asarray(row['step_token_spans'],int)
                if lp.shape!=(len(entropy),50) or spans.shape!=(r['steps'],2):raise ValueError('shape mismatch '+uid)
                if (spans[:,0]<0).any() or (spans[:,1]>len(lp)).any() or (spans[:,1]<=spans[:,0]).any():raise ValueError('invalid span '+uid)
                with np.load(base.old.BENCH/'scores'/f'{uid}.npz') as z:
                    np.testing.assert_array_equal(spans[:,0],z['step_starts'])
                    np.testing.assert_array_equal(spans[:,1],z['step_ends'])
                if kind=='pb':np.testing.assert_allclose(entropy.mean(),detector[i],atol=1e-12,rtol=0)
                saved,info=src.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone()
                info=json.loads(info)
                if info['uid']!=uid:raise ValueError('checkpoint identity mismatch')
                sl=slice(joined['offsets'][i],joined['offsets'][i+1])
                labels=joined['labels'][sl].copy() if kind=='prm' else np.full(len(spans),-1)
                lengths=spans[:,1]-spans[:,0];se=step_means(entropy,spans);masks=strata(se,lengths)
                arrays=dict(lengths=lengths,step_entropy=se,masks=masks)
                metadata=dict(uid=uid,cell=cell,n_tokens=len(lp),failures={},banks={})
                with np.load(io.BytesIO(saved)) as state:
                    for degree,bank in ((3,6),(6,12)):
                        key=f'd{degree}__rbm';prefix=f'b{bank}__'
                        if key not in info['diagnostics']:
                            metadata['failures'][str(bank)]=info['failures'].get(key,'missing saved fit');continue
                        d=info['diagnostics'][key]
                        x=representation_order(lp,row['token_spilled_energies'],degree)
                        z,keep,mean,scale=zscore_columns(x)
                        np.testing.assert_array_equal(np.flatnonzero(keep),d['columns'])
                        np.testing.assert_array_equal(mean,d['normalization_mean'])
                        np.testing.assert_array_equal(scale,d['normalization_scale'])
                        a,w,b=state[key+'::a'],state[key+'::w'],float(state[key+'::b'])
                        token=expit(b+z@w)
                        if d['orientation']==-1:token=1-token
                        replay=step_top_mean(token,spans[:,0],spans[:,1],count=10)
                        np.testing.assert_array_equal(replay,scores[f'rbm{bank}__old'][sl])
                        initial=step_top_mean(expit(z@np.full(z.shape[1],2/z.shape[1])),spans[:,0],spans[:,1],count=10)
                        np.testing.assert_array_equal(initial,scores[f'initial{bank}__old'][sl])
                        result=diagnose_bank(z,a,w,b,spans,labels,masks,uid,bank,replay)
                        # Preserve original feature-coordinate indices when a column is dropped.
                        result.update(columns=np.flatnonzero(keep),weights=w*d['orientation'])
                        arrays.update({prefix+k:v for k,v in result.items()})
                        initial_w=np.full(len(w),2/len(w));oriented=w*d['orientation']
                        norm=np.linalg.norm(oriented)
                        metadata['banks'][str(bank)]=dict(active_columns=int(keep.sum()),
                            converged=d['converged'],iterations=d['iterations'],
                            nll_improvement=d['nll_initial']-d['nll_final'],
                            weight_cosine_initial=float(oriented@initial_w/(norm*np.linalg.norm(initial_w))) if norm else None,
                            weight_norm=float(norm),negative_weights=int((oriented<0).sum()),
                            orientation=d['orientation'],gradient_max=d.get('gradient_max'),
                            score_sd=d['score_sd'])
                con.execute('INSERT INTO answers VALUES (?,?,?)',(i,base.packed(**arrays),base.dumps(metadata)))
                done.add(i)
                if len(done)%25==0 or i==indices[-1]:
                    con.commit()
                    base.atomic_json(OUT/('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'),dict(
                        status='SMOKE' if smoke else 'RUNNING',stage='data_scan',completed=len(done),expected=len(records),
                        elapsed_seconds=time.perf_counter()-started,pid=os.getpid()))
                    print('[diagnostics]',len(done),'/',len(records),round(time.perf_counter()-started,1),'seconds',flush=True)
            del rows
    finally:
        src.close()


def stat(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return dict(n=int(len(a)),mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p10=float(np.quantile(a,.1)) if len(a) else None,
                p90=float(np.quantile(a,.9)) if len(a) else None)


def aggregate(con,records,joined,per):
    n=len(records);target=joined['target'];cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
    all_lengths=np.zeros(int(joined['offsets'][-1]),int);details=[];summary={};tables=[];errors=[]
    B={bank:dict(rel=np.full((n,7,bank+1),np.nan),mean=np.full((n,7,bank),np.nan),
        logvar=np.full((n,7,bank),np.nan),zero=np.zeros((n,7,2,bank),bool),
        counts=np.zeros((n,7,2),int),resid=np.full((n,5),np.nan),null=np.full((n,4,5),np.nan),
        corr=np.full((n,bank,bank),np.nan),eigen=np.full((n,bank),np.nan),
        lag=np.full((n,7,3,bank),np.nan),permuted=np.full((n,7,3,bank),np.nan),
        lagcounts=np.zeros((n,7,3),int)) for bank in (6,12)}
    for i,payload,s in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        info=json.loads(s);details.append(dict(idx=i,**info))
        sl=slice(joined['offsets'][i],joined['offsets'][i+1])
        with np.load(io.BytesIO(payload)) as z:
            all_lengths[sl]=z['lengths']
            for bank in (6,12):
                pre=f'b{bank}__'
                if pre+'columns' not in z:continue
                cols=z[pre+'columns'];v=B[bank]
                for dst,key in [('mean','class_mean_delta'),('logvar','class_log_variance_ratio')]:
                    v[dst][i][:,cols]=z[pre+key]
                v['rel'][i][:,np.r_[cols,bank]]=z[pre+'reliability']
                v['zero'][i][:,:,cols]=z[pre+'class_zero_variance'];v['counts'][i]=z[pre+'class_counts']
                v['resid'][i]=z[pre+'residual_observed'];v['null'][i]=z[pre+'residual_synthetic']
                v['corr'][i][np.ix_(cols,cols)]=z[pre+'residual_correlation']
                v['eigen'][i,:len(cols)]=z[pre+'residual_eigenvalues']
                v['lag'][i][:,:,cols]=z[pre+'lag']
                v['permuted'][i][:,:,cols]=np.nanmean(z[pre+'lag_permutations'],axis=0)
                v['lagcounts'][i]=z[pre+'lag_counts']
    for bank,v in B.items():
        learned,initial=per[f'rbm{bank}__near'],per[f'initial{bank}__near']
        old=per[f'rbm{bank}__old'];err=pb&(target>=0)
        hit=err&learned['decision_valid']&(learned['prediction']==target)
        oldhit=err&old['decision_valid']&(old['prediction']==target)
        ihit=err&initial['decision_valid']&(initial['prediction']==target)
        names=list(feature_names(3 if bank==6 else 6))+['fusion_top10']
        for i in np.flatnonzero(pb):
            t=int(target[i]);sl=slice(joined['offsets'][i],joined['offsets'][i+1]);lengths=all_lengths[sl]
            p=int(learned['prediction'][i]);pk=int(learned['peak'][i])
            errors.append(dict(bank=bank,uid=records[i]['uid'],group_id=records[i]['group_id'],cell=cells[i],target=t,
                old_peak=int(old['peak'][i]),peak=pk,prediction=p,initial_peak=int(initial['peak'][i]),
                valid=bool(learned['decision_valid'][i]),peak_distance=pk-t if t>=0 else None,
                category=('invalid' if not learned['decision_valid'][i] else
                          'clean_correct' if t<0 and p==-1 else 'false_alarm' if t<0 else
                          'gate_miss' if p==-1 else 'exact' if p==t else 'early' if p<t else 'late'),
                gained_readout=bool(hit[i] and not oldhit[i]),lost_readout=bool(oldhit[i] and not hit[i]),
                gained_learning=bool(hit[i] and not ihit[i]),lost_learning=bool(ihit[i] and not hit[i]),
                truth_longest=bool(lengths[t]==lengths.max()) if t>=0 else None,
                truth_length=int(lengths[t]) if t>=0 else None,steps=len(lengths)))
        null=v['null'].mean(axis=1);excess=v['resid']-null
        summary[str(bank)]=dict(valid_fits=int(np.isfinite(v['resid'][:,0]).sum()),
            readout_gained=int((hit&~oldhit).sum()),readout_lost=int((oldhit&~hit).sum()),
            learning_gained=int((hit&~ihit).sum()),learning_lost=int((ihit&~hit).sum()),
            residual={name:dict(observed=stat(v['resid'][:,j]),synthetic=stat(null[:,j]),excess=stat(excess[:,j]),
                pb_hits_excess=stat(excess[hit,j]),pb_misses_excess=stat(excess[err&~hit,j])) for j,name in enumerate(RESIDUAL_NAMES)},
            optimization={key:stat([d['banks'][str(bank)].get(key) for d in details if str(bank) in d['banks']])
                for key in ('nll_improvement','weight_cosine_initial','negative_weights','iterations','gradient_max','score_sd')})
        for si,st in enumerate(STRATA):
            for j,name in enumerate(names):
                tables.append(dict(bank=bank,diagnostic='feature_auc',stratum=st,feature=name,**stat(v['rel'][:,si,j])))
                if j==bank:continue
                for key in ('mean','logvar'):
                    tables.append(dict(bank=bank,diagnostic=key,stratum=st,feature=name,**stat(v[key][:,si,j])))
                tables.append(dict(bank=bank,diagnostic='zero_variance',stratum=st,feature=name,
                    n=int((v['counts'][:,si].min(axis=1)>=2).sum()),
                    zero_correct=int(v['zero'][:,si,0,j].sum()),zero_error=int(v['zero'][:,si,1,j].sum())))
            for lag in range(3):
                delta=v['lag'][:,si,lag]-v['permuted'][:,si,lag]
                for subset,mask in [('all',np.ones(n,bool)),('pb_hit',hit),('pb_miss',err&~hit)]:
                    per_answer=np.nanmean(delta,axis=1)
                    tables.append(dict(bank=bank,diagnostic='lag_excess',stratum=st,feature='mean_signed_feature_correlation',
                                       lag=lag+1,subset=subset,**stat(per_answer[mask])))
        # Preserve per-answer arrays, including covariance matrices, eigen spectra,
        # uncertainty across model replicas, source IDs and diagnostic denominators.
        np.savez_compressed(OUT/f'BANK{bank}_DIAGNOSTICS.npz',**v,uid=np.array([r['uid'] for r in records]),
            group_id=np.array([r['group_id'] for r in records]),cell=cells,feature=np.array(names))
    base.atomic_json(OUT/'DIAGNOSTICS.json',dict(n_answers=n,banks=summary,
        interpretation='Step labels only. Conditional model check with four draws; no refit, no calibrated model-test p-values. '
                       'Diagnostic distributions are descriptive; p10/p90 are answer quantiles, not confidence intervals.',
        strata=STRATA,feature_auc_aggregation='Step means for features; original top10 for fusion.'))
    base.atomic_json(OUT/'MODEL_DETAILS.json',details)
    csv_write(OUT/'DIAGNOSTIC_SUMMARY.csv',tables);csv_write(OUT/'ERROR_CASES.csv',errors)
    np.save(OUT/'STEP_LENGTHS.npy',all_lengths)
    return summary


def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args();source=a.source_root.resolve();base.old.configure_source_root(source)
    OUT.mkdir(parents=True,exist_ok=True)
    print('[manifest] verifying frozen inputs',flush=True)
    manifest=frozen_manifest(source)
    con=connect(OUT/('SMOKE.sqlite' if a.smoke else 'CHECKPOINT.sqlite'),manifest)
    base.atomic_json(OUT/('SMOKE_MANIFEST.json' if a.smoke else 'MANIFEST.json'),manifest)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined=np.load(base.old.BENCH/'evaluation/JOINED.npz')
    assert len(records)==13769 and len({r['uid'] for r in records})==13769
    scores,prior=references(source,records,joined)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Mean of empty slice')
        if a.smoke:
            scan(source,con,records,joined,scores,True)
            base.atomic_json(OUT/'SMOKE.json',dict(status='PASS',n_answers=con.execute('SELECT COUNT(*) FROM answers').fetchone()[0],
                                                 scope='mechanics only; no subset performance conclusions'))
            return
        metrics,per=evaluate(source,records,joined,scores,prior)
        if not a.evaluate_only:scan(source,con,records,joined,scores,False)
        count=con.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
        if count==len(records):
            aggregate(con,records,joined,per)
            base.atomic_json(OUT/'RUN_STATE.json',dict(status='SCORED_AWAITING_REVIEW',completed=count,expected=len(records)))
    con.close()


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True)
        base.atomic_json(OUT/('SMOKE_ERROR.json' if '--smoke' in sys.argv else 'ERROR.json'),dict(error=repr(e)))
        raise
