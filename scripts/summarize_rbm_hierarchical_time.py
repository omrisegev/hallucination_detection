"""Read-only post-fit diagnostics and compatible historical references; no refits."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse
import json
import sqlite3
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_rbm_hierarchical_time as run

NAMES=dict(top10='Frozen RBM12: Top10 mean',all_mean='Frozen RBM12: all-token mean',
    contiguous10='Frozen RBM12: best consecutive 10',local='Time fusion: current answer',
    shared='Time fusion: other answers',hierarchical='Time fusion: shared + local',
    shuffled='Time fusion: shuffled tokens',supervised='Supervised time weights',
    drop_first_top10='Top10: remove first token',drop_first_hierarchical='Hierarchy: remove first token')
NAMES.update({'reference__'+key:value for key,value in {
    'rbm6__old':'RBM6: posterior + Top10','rbm6__logit_old':'RBM6: logit + Top10',
    'rbm12__old':'RBM12: posterior + Top10','rbm12__logit_old':'RBM12: logit + Top10',
    'initial6__old':'RBM6 initialization: posterior + Top10',
    'initial12__old':'RBM12 initialization: posterior + Top10',
    'entropy__old':'Token entropy: Top10','var15__old':'Varentropy15: Top10',
    'var50__old':'Varentropy50: Top10','var15_iu__old':'Varentropy15 contributions: IU-PCR',
    'var15_equal__old':'Varentropy15 contributions: equal weights',
    'shrinkage__old':'Saved RBM shrinkage reference','diagonal__old':'Saved RBM diagonal variance reference'}.items()})


def prm_intervals(records,joined,scores,metrics,thresholds):
    """Fixed nested thresholds, resampling the same canonical groups as PB."""
    groups,inv=np.unique([r['group_id'] for r in records],return_inverse=True);ng=len(groups)
    methods=list(run.METHODS);counts=np.zeros((ng,len(methods),4))
    raw={str(r['idx']):r for r in run.base.old.load_pickle(run.base.old.PRMB_LABELS).values()}
    folds=json.loads(run.base.old.FOLDS.read_text())['outer']
    for i,r in enumerate(records):
        if r['cell'].startswith('pb_') or raw[str(r['row_id'])]['classification']=='correct':continue
        a,b=joined['offsets'][i:i+2];truth=joined['labels'][a:b]==1
        for j,method in enumerate(methods):
            s=scores[method][a:b]
            if not np.isfinite(s).all():continue
            risk=s>=thresholds[method][str(folds[r['group_id']])]
            counts[inv[i],j]+=np.array([np.sum(~truth & ~risk),np.sum(truth & ~risk),
                                      np.sum(truth & risk),np.sum(~truth & risk)])
    def metric(c):
        tp,fp,tn,fn=np.moveaxis(c,-1,0)
        return .5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
    point=metric(counts.sum(axis=0))
    np.testing.assert_allclose(point,[metrics[m]['prmscore_conditional'] for m in methods],atol=1e-12,rtol=0)
    pairs=run.PRIMARY+[(m,'top10') for m in methods if m!='top10']
    draws={a+'_minus_'+b:[] for a,b in pairs};rng=np.random.default_rng(20260910136)
    for start in range(0,10000,128):
        n=min(128,10000-start);weights=rng.multinomial(ng,np.full(ng,1/ng),size=n).astype(float)
        values=metric((weights@counts.reshape(ng,-1)).reshape(n,len(methods),4))
        for a,b in pairs:draws[a+'_minus_'+b].extend((values[:,methods.index(a)]-values[:,methods.index(b)]).tolist())
    result={}
    for a,b in pairs:
        primary=(a,b) in run.PRIMARY;level=.975 if primary else .95;key=a+'_minus_'+b
        result[key]=dict(primary=primary,delta=float(point[methods.index(a)]-point[methods.index(b)]),
                         ci=np.quantile(draws[key],[(1-level)/2,1-(1-level)/2]).tolist(),ci_level=level,
                         draws=10000,conditional_on_saved_fits_and_thresholds=True)
    return result


def per_cell_table(metrics):
    rows=[]
    for method,m in metrics.items():
        for cell,c in m['pb_cells'].items():
            rows.append(dict(method=method,name=NAMES.get(method,method),cell=cell,benchmark='ProcessBench',
                             pb_f1=c['f1'],answers=c['answers'],valid_answers=c['valid_decisions'],
                             clean_accuracy=c['clean_accuracy'],error_exact_accuracy=c['error_exact_accuracy']))
        rows.append(dict(method=method,name=NAMES.get(method,method),cell='prmbench_qwen3_8b',benchmark='PRMBench',
                         answers=6969,valid_answers=m['prm_valid_answers'],within_auc=m['prm_within'],
                         pooled_auc=m['prm_pooled'],fold_mean_auc=m['prm_fold_auc'],prmscore=m['prmscore_q08']))
    run.old.csv_write(run.OUT/'PER_CELL.csv',rows)


def control_intervals(records,joined,scores,metrics):
    """Descriptive matched contrasts for the registered mechanism controls."""
    per={};detector,gate=run.base.old._gate_contract(records)
    pb=np.array([r['cell'].startswith('pb_') for r in records])
    for name,flat in scores.items():
        valid=np.zeros(len(records),bool);peak=np.full(len(records),-1,int);within=np.full(len(records),np.nan)
        for i in range(len(records)):
            a,b=joined['offsets'][i:i+2];s=flat[a:b]
            if not np.isfinite(s).all():continue
            valid[i]=True;peak[i]=int(np.argmax(s))
            if not pb[i]:
                y=joined['labels'][a:b];known=y>=0
                if len(np.unique(y[known]))==2:within[i]=run.base.old.auc(y[known]==1,s[known])
        np.testing.assert_allclose(np.nanmean(within),metrics[name]['prm_within'],atol=1e-12,rtol=0)
        per[name]=dict(valid=valid,decision_valid=valid & np.isfinite(detector) & np.isfinite(gate),
                       peak=peak,prediction=np.where(detector>=gate,peak,-1),within=within)
    pairs=[('hierarchical','shuffled'),('shared','all_mean'),('supervised','shared'),
           ('drop_first_hierarchical','hierarchical')]
    result=run.base.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set())
    for a,b in pairs:result[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    return result


def extra_references(source,records,joined):
    specs=[(source/'.worktrees/dufs-moment-selection-v1/results/dufs_moment_selection_v1',
            {'dufs6__rbm':'RBM with DUFS: 6 features','correlation6__rbm':'RBM with low-correlation selection: 6 features'}),
           (source/'.worktrees/rbm-literature-completion-v1/results/rbm_literature_completion_v1/variance',
            {'b12_variance_shared_logit':'RBM12 shared variance: logit',
             'b12_variance_shared_posterior':'RBM12 shared variance: posterior'})]
    paths=[run.base.old.BENCH/'evaluation/JOINED.json',run.base.old.BENCH/'evaluation/JOINED.npz',
           run.base.old.FOLDS,run.base.old.FIXED_GATE/'DETECTORS.npz',run.base.old.FIXED_GATE/'METRICS.json']
    required=[run.base.old.sha256_file(p) for p in paths]
    rows=[];audit=[]
    for directory,names in specs:
        source_manifest=json.loads((directory/'MANIFEST.json').read_text())
        assert all(h in source_manifest['hashes'].values() for h in required),'reference contract differs'
        assert json.loads((directory/'RESULT_REVIEW.json').read_text())['status']=='PASS'
        historical=json.loads((directory/'METRICS.json').read_text())['metrics']
        with np.load(directory/'SCORES.npz') as z:scores={name:z['steps__'+name] for name in names}
        metrics,_=run.base.evaluate_arrays(records,joined,scores)
        for name,m in metrics.items():
            for key in ('pb_all8','prm_within','prm_pooled','prmscore_q08'):
                np.testing.assert_allclose(m[key],historical[name][key],atol=1e-12,rtol=0)
            rows.append(dict(method=name,name=names[name],access='answer-local unlabelled; external gate and PRM calibration',
                             source=str(directory),**{k:v for k,v in m.items() if not isinstance(v,dict)}))
        audit.append(dict(source=str(directory),contract_hashes_match=True,metrics_reproduced=True,
                          hashes={p.name:run.base.old.sha256_file(p) for p in
                                  [directory/'MANIFEST.json',directory/'METRICS.json',directory/'SCORES.npz',directory/'RESULT_REVIEW.json']}))
    return rows,audit


def summarize(source):
    out=run.OUT
    assert json.loads((out/'RESULT_REVIEW.json').read_text())['status']=='PASS'
    records,joined,_=run.old.load_contract(source)
    metrics=json.loads((out/'METRICS.json').read_text());contrast=json.loads((out/'CONTRASTS.json').read_text())
    per_cell_table(metrics)
    con=sqlite3.connect(f'file:{(out/"CHECKPOINT.sqlite").as_posix()}?mode=ro',uri=True)
    freeze=json.loads(con.execute('select payload from manifest').fetchone()[0])
    for path,digest in freeze['hashes'].items():
        assert run.base.old.sha256_file(Path(path))==digest,f'source drift: {path}'
    folds=json.loads(run.base.old.FOLDS.read_text())['outer']
    detector,threshold=run.base.old._gate_contract(records)
    groups={};rows=[];failures=[];profile_sizes=[];step_lengths=[]
    for i,uid,blob in con.execute('select idx,uid,payload from profiles order by idx'):
        p=run.load_blob(blob);r=records[i];f=int(folds[r['group_id']])
        a_blob,meta=con.execute('select payload,info from scores where key=?',(f'{i}__exclude_{f}',)).fetchone()
        a=run.load_blob(a_blob);meta=json.loads(meta);target=int(joined['target'][i])
        baseline=int(np.argmax(p['top']));profile_sizes.append(len(p['raw']));step_lengths.extend(p['lengths'].tolist())
        for method in run.METHODS:
            valid=bool(np.isfinite(a[method]).all());peak=int(np.argmax(a[method])) if valid else -1
            gate_open=bool(detector[i]>=threshold[i]);prediction=peak if gate_open else -1
            row=dict(idx=i,uid=uid,cell=r['cell'],group=r['group_id'],fold=f,method=method,
                     target=target,steps=r['steps'],baseline_peak=baseline,peak=peak,
                     shift=peak-baseline if valid else None,valid=valid,gate_open=gate_open,
                     prediction=prediction,selected_length=int(p['lengths'][peak]) if valid else None,
                     raw_exact=bool(valid and target>=0 and peak==target),
                     correct=bool(valid and prediction==target))
            rows.append(row)
            wk='weight__'+method
            if wk not in a:continue
            w=a[wk]
            d=meta['fits'][method]
            key=(method,r['cell']);g=groups.setdefault(key,dict(weights=[],distances=[],local_shifts=[],misfit=[],conditions=[],floors=[]))
            if np.isfinite(w).all():
                g['weights'].append(w);g['distances'].append(float(np.abs(w-1/16).sum()))
                g['local_shifts'].append(float(np.abs(w-a['weight__shared']).sum()))
            if 'residual_relative' in d:g['misfit'].append(d['residual_relative'])
            if 'condition' in d:g['conditions'].append(d['condition'])
            if 'noise' in d:g['floors'].append(float(np.mean(np.asarray(d['noise'])<=d['noise_floor']*(1+1e-8))))
            if d.get('converged') is False or 'failure' in d:
                failures.append(dict(idx=i,method=method,details=d))
    summaries=[]
    length=np.asarray(step_lengths);sizes=np.asarray(profile_sizes)
    run.emit('TOKEN_REGION_COVERAGE.json',dict(steps=len(length),steps_shorter_than_16=int(np.sum(length<16)),
        single_token_steps=int(np.sum(length==1)),step_length_quartiles=np.quantile(length,[.25,.5,.75]).tolist(),
        answer_steps_quartiles=np.quantile(sizes,[.25,.5,.75]).tolist(),
        alpha_quartiles=np.quantile((sizes-1)/(sizes-1+16),[.25,.5,.75]).tolist(),
        note='Short-step regions overlap the same tokens. Binning creates no independent observations.'))
    for (method,cell),g in groups.items():
        w=np.asarray(g['weights']);summary=dict(method=method,cell=cell,answers=len(w))
        if len(w):summary.update({f'mean_region_{j+1}':float(v) for j,v in enumerate(w.mean(axis=0))})
        for key in ('distances','local_shifts','misfit','conditions','floors'):
            v=np.asarray(g[key]);v=v[np.isfinite(v)]
            summary[key+'_quartiles']=np.quantile(v,[.25,.5,.75]).tolist() if len(v) else []
        summaries.append(summary)
    unique_model_health=[]
    expected_records=len(records)+4*sum(not r['cell'].startswith('pb_') for r in records)
    assert con.execute('select count(*) from scores').fetchone()[0]==expected_records
    for key,text in con.execute('select key,info from models order by key'):
        info=json.loads(text)
        for method,d in info['fits'].items():
            unique_model_health.append(dict(model=key,method=method,excluded_folds=info['excluded_folds'],
                                           training_answers=len(info['training_ids']),
                                           converged=d.get('converged'),reason=d.get('reason'),failure=d.get('failure')))
    run.emit('COMPUTE_ACCOUNTING.json',dict(
        note='FIT_HEALTH fits counts answer-level uses, including named uniform rules; shared models are reused.',
        answer_exclusion_records=int(con.execute('select count(*) from scores').fetchone()[0]),
        unique_training_sets=int(con.execute('select count(*) from models').fetchone()[0]),
        unique_shared_model_health=unique_model_health))
    reference_rows,reference_audit=extra_references(source,records,joined)
    with np.load(out/'SCORES.npz') as z:full_scores={m:z[m] for m in run.METHODS}
    thresholds=json.loads((out/'CALIBRATION.json').read_text())['thresholds']
    run.emit('PRMSCORE_CONTRASTS.json',prm_intervals(records,joined,full_scores,metrics,thresholds))
    run.emit('CONTROL_CONTRASTS.json',control_intervals(records,joined,full_scores,metrics))
    run.emit('ADDITIONAL_REFERENCE_REVIEW.json',dict(status='PASS',sources=reference_audit))
    run.old.csv_write(out/'ADDITIONAL_REFERENCES.csv',reference_rows)
    run.old.csv_write(out/'ANSWER_DIAGNOSTICS.csv',rows)
    run.emit('WEIGHT_SUMMARY.json',summaries);run.emit('FLAGGED_FITS.json',failures)
    simple=[]
    for method,m in metrics.items():
        access=('other-answer labelled diagnostic' if method=='supervised' else
                'other-answer unlabelled + current-answer adaptation' if method in ('hierarchical','shuffled','drop_first_hierarchical') else
                'other-answer unlabelled' if method=='shared' else 'answer-local; external gate and PRM calibration')
        simple.append(dict(method=method,name=NAMES.get(method,method),access=access,
                           **{k:v for k,v in m.items() if not isinstance(v,dict)}))
    run.old.csv_write(out/'COMPARISON.csv',simple+reference_rows)
    lines=['# Frozen RBM12 hierarchical time fusion','',
           'Full development population: 13,769 answers. Fixed RBM12 coefficients and entropy gate. No first_near_max.',
           'Other-answer time weights use source-group exclusion; PRMScore uses nested calibration. No untouched confirmation.','',
           '| Method | ProcessBench % | PRMB within AUC | PRMScore | Valid answers |',
           '|---|---:|---:|---:|---:|']
    for method in run.METHODS:
        m=metrics[method];prm=m['prmscore_q08'];prm='unavailable' if prm is None else f'{prm:.5f}'
        lines.append(f'| {NAMES[method]} | {100*m["pb_all8"]:.3f} | {m["prm_within"]:.5f} | {prm} | {m["valid_answers"]} |')
    lines+=['','Primary contrasts (paired source groups; 10,000 draws; 97.5% intervals):','']
    for a,b in run.PRIMARY:
        c=contrast[a+'_minus_'+b]
        lines.append(f'- {NAMES[a]} minus {NAMES[b]}: PB delta {100*(metrics[a]["pb_all8"]-metrics[b]["pb_all8"]):.3f} pp, '
                     f'CI {[round(100*x,3) for x in c["pb_ci"]]}; within-AUC delta {c["prm_within_delta_common"]:.5f}, CI {c["prm_within_ci"]}.')
    lines+=['','See COMPARISON.csv for historical references and per-cell metrics in METRICS.json.',
            'WEIGHT_SUMMARY.json records temporal weights, adaptation distance, covariance misfit and conditioning.',
            'ANSWER_DIAGNOSTICS.csv and ERROR_TRANSITIONS.json retain exact peaks, shifts and exclusive losses.',
            'FLAGGED_FITS.json lists numerical failures and finite nonconverged fits. No method is silently substituted.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    run.emit('POSTFIT_REVIEW.json',dict(status='PASS',answers=len(profile_sizes),source_hashes_rechecked=True,
                                      historical_references=len(reference_rows),diagnostic_rows=len(rows)))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    summarize(p.parse_args().source_root)
