"""Join, evaluate and review a complete corrected historical first panel."""
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/historical_fusion_refit_v3'
PARENT=ROOT/'results/localization_full_benchmark_v3'
FOLDS=ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json'
spec=importlib.util.spec_from_file_location('historical_metric_core',ROOT/'spectral_utils/historical_fusion_evaluation.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
DRAWS=1000


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(value),indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def job_name(job):
    return job['cell']+'/outer'+str(job['outer'])+'/'+('test' if job['inner'] is None else 'inner'+str(job['inner']))


def join(manifest):
    reference=load(PARENT/'evaluation/JOINED.json')
    assert sha(PARENT/'evaluation/JOINED.npz')==reference['arrays_sha256']
    with np.load(PARENT/'evaluation/JOINED.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
    records=manifest['selected'];assert records==reference['records']
    old_arms=reference['arms'];new=manifest['arms'];arms=old_arms+new;n=len(records);b=len(old_arms);k=len(arms)
    a={key:old[key].copy() for key in ('offsets','labels','target')}
    for key,dtype,fill in [('scores',float,np.nan),('valid',bool,False),('decision',bool,False),
                         ('predictions',int,-2),('peaks',int,-2),('within',float,np.nan)]:
        a[key]=np.full((len(old[key]),k),fill,dtype=dtype);a[key][:,:b]=old[key]
    detector=np.full((n,len(new)),np.nan);seen=np.zeros(n,int)
    folds=load(FOLDS);outer=np.array([folds['outer'][r['group_id']] for r in records])
    cells=np.array([r['cell'] for r in records]);global_index={(r['cell'],r['row']):i for i,r in enumerate(records)}
    calibration={outer:dict(detector=[],peaks=[],valid=[],indices=[]) for outer in range(5)}
    mh=sha(OUT/'MANIFEST.json');files={};failure_counts={arm:0 for arm in new};fit_review=[]
    for job in manifest['jobs']:
        path=OUT/'fits'/job_name(job);mp=path.with_suffix('.json');ap=path.with_suffix('.npz')
        meta=load(mp);assert meta['manifest_sha256']==mh and meta['job']==job
        assert sha(ap)==meta['array_sha256'];files[str(ap)]=sha(ap);files[str(mp)]=sha(mp)
        assert not meta['labels_accessed']
        assert set(meta['methods']) | set(meta['failures']) == set(new)
        assert set(meta['methods']).isdisjoint(meta['failures'])
        assert set(meta['train_groups']).isdisjoint(meta['evaluation_groups'])
        assert all(folds['outer'][g]!=job['outer'] for g in meta['train_groups'])
        if job['inner'] is not None:
            assert all(folds['outer'][g]!=job['outer'] and folds['inner'][str(job['outer'])][g]==job['inner'] for g in meta['evaluation_groups'])
        with np.load(ap,allow_pickle=False) as z:
            rows=z['rows'];ix=np.array([global_index[job['cell'],int(r)] for r in rows])
            local_records=[r for r in records if r['cell']==job['cell']]
            local_records=sorted(local_records,key=lambda r:r['row'])
            mask_train=np.array([folds['outer'][r['group_id']]!=job['outer'] and
                (job['inner'] is None or folds['inner'][str(job['outer'])][r['group_id']]!=job['inner']) for r in local_records])
            np.testing.assert_array_equal(z['train_rows'],np.flatnonzero(mask_train))
            assert set(meta['train_groups'])=={local_records[i]['group_id'] for i in z['train_rows']}
            assert set(meta['evaluation_groups'])=={records[i]['group_id'] for i in ix}
            if 'fit_indices' in z:
                token_offsets=np.load(PARENT/'inputs'/job['cell']/'token_offsets.npy',mmap_mode='r')
                owners=np.searchsorted(token_offsets[1:],z['fit_indices'],side='right')
                assert np.all(mask_train[owners]) and len(owners)<=60000
            if job['inner'] is None:
                assert np.all(outer[ix]==job['outer']);seen[ix]+=1
            lo=np.r_[0,np.cumsum([records[i]['steps'] for i in ix])]
            local_offsets=np.r_[0,np.cumsum([r['steps'] for r in local_records])]
            expected_steps=np.concatenate([np.arange(local_offsets[row],local_offsets[row+1]) for row in rows])
            np.testing.assert_array_equal(z['steps'],expected_steps)
            det=np.full((len(ix),len(new)),np.nan);peaks=np.full(det.shape,-2,int);valid=np.zeros(det.shape,bool)
            for j,arm in enumerate(new):
                if arm in meta['failures']:
                    failure_counts[arm]+=1;assert arm+'__w' not in z;continue
                assert arm+'__w' in z
                top=z[arm+'__top10'];span=z[arm+'__spanmax'];d=z[arm+'__detector']
                assert len(top)==len(span)==lo[-1] and len(d)==len(ix)
                det[:,j]=d
                for q,index in enumerate(ix):
                    u,v=lo[q:q+2];valid[q,j]=bool(np.isfinite(top[u:v]).all() and np.isfinite(span[u:v]).all() and np.isfinite(d[q]))
                    if valid[q,j]:peaks[q,j]=int(np.argmax(top[u:v]))
                    if job['inner'] is None:
                        u0,v0=a['offsets'][index:index+2]
                        if valid[q,j]:
                            a['scores'][u0:v0,b+j]=span[u:v]
                            a['valid'][index,b+j]=True;a['peaks'][index,b+j]=peaks[q,j]
                            detector[index,j]=d[q]
            if job['inner'] is not None:
                c=calibration[job['outer']]
                for key,value in [('detector',det),('peaks',peaks),('valid',valid),('indices',ix)]:c[key].append(value)
        fit_review.append(dict(job=job,status='PASS',failures=meta['failures']))
    assert np.all(seen==1)
    thresholds={}
    for fold in range(5):
        c={key:np.concatenate(v) for key,v in calibration[fold].items()};ix=c['indices']
        np.testing.assert_array_equal(np.sort(ix),np.flatnonzero((outer!=fold)&np.char.startswith(cells,'pb_')))
        assert len(ix)==len(set(ix.tolist()))
        test=(outer==fold)&np.char.startswith(cells,'pb_')
        for j,arm in enumerate(new):
            result=metrics.calibrate(c['detector'][:,j],c['peaks'][:,j],c['valid'][:,j],a['target'][ix],cells[ix])
            thresholds[str(fold)+'/'+arm]=result
            if result['status']=='CALIBRATED':
                a['decision'][test,b+j]=a['valid'][test,b+j]
                a['predictions'][test,b+j]=np.where(detector[test,j]>=result['threshold'],a['peaks'][test,b+j],-1)
                a['predictions'][test&~a['decision'][:,b+j],b+j]=-2
    for i,rec in enumerate(records):
        if not rec['cell'].startswith('prm'):continue
        u,v=a['offsets'][i:i+2]
        for j in range(b,k):
            if a['valid'][i,j]:
                value=metrics.auc(a['labels'][u:v],a['scores'][u:v,j])
                if value is not None:a['within'][i,j]=value
    # Verify actual v3 label files, not merely the already joined labels.
    release=load(ROOT/'results/localization_prm_label_audit_v1/RELEASE_V3.json')
    checked=0
    for cell,info in release['cells'].items():
        with np.load(info['label_path'],allow_pickle=False) as z:
            # Row IDs bind the historical readouts to v3 targets.
            by_id={r['row_id']:i for i,r in enumerate(records) if r['cell']==cell}
            for q,rid in enumerate(z['row_ids']):
                i=by_id[str(rid)]
                if cell.startswith('prm'):
                    u,v=z['step_flag_offsets'][q:q+2];u0,v0=a['offsets'][i:i+2]
                    np.testing.assert_array_equal(z['step_error_flags'][u:v],a['labels'][u0:v0])
                else:assert int(z['first_error'][q])==a['target'][i]
                checked+=1
    assert checked==n
    save(OUT/'SCORES_FROZEN.json',dict(files=files,manifest_sha256=mh))
    save(OUT/'THRESHOLDS.json',thresholds)
    with (OUT/'JOINED.npz').open('wb') as f:np.savez_compressed(f,**a,outer=outer)
    save(OUT/'JOINED.json',dict(records=records,arms=arms,arrays_sha256=sha(OUT/'JOINED.npz')))
    return records,arms,a,outer,dict(fits=fit_review,checked_labels=checked,failures= failure_counts)


def point_metrics(records,arms,a,outer):
    cells=np.array([r['cell'] for r in records]);prm=np.char.startswith(cells,'prm');owner=np.repeat(np.arange(len(records)),np.diff(a['offsets']))
    result={}
    for j,arm in enumerate(arms):
        fold_auc=[];fold_coverage=[]
        for fold in range(5):
            rows=prm & (outer==fold);valid=rows & a['valid'][:,j];mask=valid[owner]
            fold_auc.append(metrics.auc(a['labels'][mask],a['scores'][mask,j]) if mask.any() else None)
            fold_coverage.append(dict(total=int(rows.sum()),valid=int(valid.sum())))
        within=a['within'][prm,j];finite=np.isfinite(within)
        pb=metrics.pb_metrics(a['target'],a['predictions'][:,j],a['decision'][:,j],cells)
        for cell,value in pb['cells'].items():
            rows=(cells==cell)&(a['target']>=0)
            value['raw_peak_accuracy']=float((a['valid'][rows,j]&(a['peaks'][rows,j]==a['target'][rows])).mean())
        result[arm]=dict(access='current_answer_only' if j<19 else 'other_training_answers; PB nested label calibration',
            prm=dict(fold_aucs=fold_auc,fold_mean_auc=float(np.mean(fold_auc)) if None not in fold_auc else None,
            within_answer_auc=float(within[finite].mean()) if finite.any() else None,mixed_answers=int(finite.sum()),
            total_answers=int(prm.sum()),valid_answers=int((prm&a['valid'][:,j]).sum()),fold_coverage=fold_coverage),pb=pb)
    return result


def bootstrap(records,arms,a,outer,new):
    group_names=sorted({r['group_id'] for r in records});lookup={g:i for i,g in enumerate(group_names)}
    gi=np.array([lookup[r['group_id']] for r in records]);n=len(group_names)
    draws=np.random.default_rng(20260907328).multinomial(n,np.full(n,1/n),size=DRAWS)
    cells=np.array([r['cell'] for r in records]);prm=np.char.startswith(cells,'prm');owner=np.repeat(np.arange(len(records)),np.diff(a['offsets']))
    roster=['dual__iu']+list(new)
    pairs=[(arm,control) for arm in new for control in ('dual__iu','equal_all23') if arm!=control]
    plans={};pair_info={};distribution={arm:np.full((DRAWS,3),np.nan) for arm in roster}
    prm_distribution={pair:np.full((DRAWS,2),np.nan) for pair in pairs}
    for pair in pairs:
        js=[arms.index(arm) for arm in pair]
        common=prm & a['valid'][:,js[0]] & a['valid'][:,js[1]]
        mixed=common & np.isfinite(a['within'][:,js[0]]) & np.isfinite(a['within'][:,js[1]])
        keys=[];points=[]
        for j in js:
            key=(j,hashlib.sha256(common.tobytes()).hexdigest());keys.append(key)
            if key not in plans:
                plans[key]=[]
                for fold in range(5):
                    mask=(common & (outer==fold))[owner]
                    plans[key].append(metrics.auc_plan(a['labels'][mask],a['scores'][mask,j],gi[owner][mask]) if mask.any() else None)
            fold_points=[metrics.weighted_auc(plan,np.ones(n)) if plan is not None else np.nan for plan in plans[key]]
            points.append([float(np.mean(fold_points)),float(a['within'][mixed,j].mean()) if mixed.any() else np.nan])
        pair_info[pair]=dict(keys=keys,mixed=mixed,indices=js,common_answers=int(common.sum()),
            common_mixed_answers=int(mixed.sum()),common_point_difference=(np.array(points[0])-points[1]).tolist())
    for d,w in enumerate(draws):
        rowweights=w[gi]
        cached_auc={key:np.mean([metrics.weighted_auc(plan,w) if plan is not None else np.nan for plan in current]) for key,current in plans.items()}
        for pair,info in pair_info.items():
            mixed=info['mixed'];den=rowweights[mixed].sum();j0,j1=info['indices']
            delta_within=float(np.dot(rowweights[mixed],a['within'][mixed,j0]-a['within'][mixed,j1])/den) if den else np.nan
            prm_distribution[pair][d]=[cached_auc[info['keys'][0]]-cached_auc[info['keys'][1]],delta_within]
        for arm in roster:
            j=arms.index(arm)
            pb=metrics.pb_metrics(a['target'],a['predictions'][:,j],a['decision'][:,j],cells,rowweights)['macros']
            distribution[arm][d]=[pb['q4'],pb['q8'],pb['all']]
        if (d+1)%100==0:print('Historical paired bootstrap',d+1,'/',DRAWS,flush=True)
    endpoints=['prm_fold_mean_auc','prm_within_auc','pb_q4','pb_q8','pb_all']
    comparisons=[]
    for arm,control in pairs:
        delta=np.column_stack([prm_distribution[arm,control],distribution[arm]-distribution[control]])
        info=pair_info[arm,control]
        comparisons.append(dict(candidate=arm,control=control,prm_common_answers=info['common_answers'],
            prm_common_mixed_answers=info['common_mixed_answers'],prm_common_point_difference=info['common_point_difference'],
            intervals={endpoint:dict(low=float(np.nanquantile(delta[:,j],.025)),high=float(np.nanquantile(delta[:,j],.975)),
            finite_draws=int(np.isfinite(delta[:,j]).sum())) for j,endpoint in enumerate(endpoints)}))
    return dict(draws=DRAWS,groups=n,scope='fixed predictions and calibrated thresholds; no model refits',
                selection_uncertainty_included=False,contrasts=comparisons)


def render(result,intervals,review):
    esc=html.escape
    def number(value,percent=False):return 'N/A' if value is None else (f'{100*value:.2f}%' if percent else f'{value:.4f}')
    rows=[]
    for arm,d in result.items():
        p=d['prm'];b=d['pb']['macros']
        rows.append('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in [arm,d['access'],number(p['fold_mean_auc']),
            number(p['within_answer_auc']),str(p['valid_answers'])+'/'+str(p['total_answers']),
            number(b['q4'],True),number(b['q8'],True),number(b['all'],True)])+'</tr>')
    contrasts=[]
    for d in intervals['contrasts']:
        for endpoint,v in d['intervals'].items():
            contrasts.append('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in [d['candidate'],d['control'],endpoint,
                number(v['low']),number(v['high']),v['finite_draws']])+'</tr>')
    percell=[]
    for arm,d in result.items():
        for cell,c in d['pb']['cells'].items():
            percell.append('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in [arm,cell,number(c['f1'],True),
                number(c['clean_accuracy'],True),number(c['error_exact_accuracy'],True),number(c['raw_peak_accuracy'],True),
                int(c['valid_decisions']),int(c['answers'])])+'</tr>')
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Full historical fusion comparison</title>
<style>body{font:16px/1.5 system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#172838}table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #cdd8e0;text-align:left}th{background:#e7eff5}p{max-width:1000px}.scroll{overflow-x:auto}a{color:#075ca5}</style>
<h1>Full historical fusion comparison: first controls</h1>
<p>All 13,769 model-answer rows are evaluated. These are exposed development data. This panel does not declare a publication winner or finish the historical benchmark.</p>
<p>The 19 answer-only anchors fit each answer separately. The five historical controls fit on other training answers. Their ProcessBench threshold uses labels from nested training folds. Both panels use the same evaluation rows and targets; their access and internal readout differ.</p>
<p>PRMBench AUC below is the mean of five held-out-fold AUCs. It differs from earlier pooled-AUC tables. The old 34–35% historical ProcessBench numbers used a different source-fold/calibration protocol and are context only. See the complete protocol and saved metrics for coverage and remaining comparators.</p>
<p><a href="../../docs/experiments/HISTORICAL_FUSION_REFIT_V3.md">Protocol</a> · <a href="METRICS.json">Metrics</a> · <a href="REVIEW.json">Review</a> · <a href="THRESHOLDS.json">Calibration</a> · <a href="../localization_full_benchmark_v3/METHOD_REGISTRY.json">Continuing registry</a></p>
<h2>Matched full-development results</h2><div class="scroll"><table><tr><th>Method</th><th>Fit / decision access</th><th>PRMB fold AUC</th><th>PRMB within-answer AUC</th><th>PRMB coverage</th><th>PB Q4</th><th>PB Q8</th><th>PB all</th></tr>'''+''.join(rows)+'''</table></div>
<h2>Fixed paired contrasts</h2><p>1,000 joint source-group draws. PRMB contrasts use each pair's common valid answers; within-answer contrasts also require both AUCs to be defined. Full-population coverage remains visible above. Every PB answer stays in its denominator. Differences are in score units (0.01 = one percentage point for ProcessBench). Intervals condition on fitted weights and thresholds; they do not cover refitting or method-selection uncertainty.</p><div class="scroll"><table><tr><th>Candidate</th><th>Control</th><th>Endpoint</th><th>95% low</th><th>95% high</th><th>Valid draws</th></tr>'''+''.join(contrasts)+'''</table></div>
<h2>Every ProcessBench cell</h2><div class="scroll"><table><tr><th>Method</th><th>Cell</th><th>Score</th><th>Clean accuracy</th><th>First-error accuracy</th><th>Raw peak accuracy</th><th>Valid decisions</th><th>All answers</th></tr>'''+''.join(percell)+'''</table></div>
<h2>Limits and next work</h2><p>The preflight replay checked all five methods against one complete historical GSM8K-Q8 fold; it did not select methods by performance. All fits were checked for source-group separation and all targets were joined to v3 labels. Review is automated and same-session, not external scientific or browser validation.</p>
<p>Corrected Joint graph/LIU/permutation refits and other historical localizers remain open. The full answer-only shortlist is a separate running stage. Small future runs check feasibility only; candidate selection requires the full matched benchmark, followed by untouched confirmation.</p></html>'''
    (OUT/'REPORT.html').write_text(content,encoding='utf-8')


def main():
    started=time.time();manifest=load(OUT/'MANIFEST.json')
    for path,h in manifest['hashes'].items():assert sha(path)==h,path
    fixtures=metrics.review_fixtures();records,arms,a,outer,checks=join(manifest)
    result=point_metrics(records,arms,a,outer)
    # Historical controls and anchor recipes share row IDs, not fitting access.
    intervals=bootstrap(records,arms,a,outer,manifest['arms'])
    review=dict(status='PASS',scope='same-session automated fidelity, isolation, joins and metric checks',
        external_review=False,browser_review=False,fixtures=fixtures,checks=checks,rows=len(records),
        arms=len(arms),fits=len(manifest['jobs']),seconds=time.time()-started)
    save(OUT/'METRICS.json',dict(status='FULL_DEVELOPMENT_FIRST_PANEL',metrics=result))
    save(OUT/'INTERVALS.json',intervals);save(OUT/'REVIEW.json',review);render(result,intervals,review)
    for path,h in manifest['hashes'].items():assert sha(path)==h,path
    print('Full historical first-panel evaluation and automated review PASS',flush=True)


if __name__=='__main__':main()
