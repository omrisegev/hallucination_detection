"""Background current110 entropy-coverage comparison with review and report."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from copy import deepcopy
import csv
import hashlib
import html
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_entropy_sampling_v1'
ORIGINAL=ROOT/'results/fusion_replication_v1';SAMPLE=ROOT/'results/fusion_sampling_replication_v1'
PARENT=ROOT/'results/fusion_trajectory_imm_v1';RELEASE=ROOT/'results/localization_prm_label_audit_v1/RELEASE_V3.json'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_entropy_sampling import SELECTORS,NEW_ARMS,CORES,JOINT_CORES,choose,score_selected
from spectral_utils.fusion_window_sampling import budget
from spectral_utils.answer_localization_v2 import moment_plan,json_safe
from spectral_utils.fusion_token_gap import apply_readout
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals

DISPLAY_SELECTORS=('full','uniform','risk_top',*SELECTORS)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def replace(tmp,path):
    for attempt in range(8):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.05*2**attempt,2))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False),encoding='utf-8');replace(tmp,path)


def priority():
    if os.name!='nt':return 'one worker; platform default priority'
    import ctypes
    result=ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    return 'BELOW_NORMAL' if result else 'priority change unavailable; one worker'


def name(selector,core):return f'sample_{selector}__{core}'


def pairs():
    result=[]
    for selector in SELECTORS:
        for core in ('iu','graph010','equal_graph_perm'):
            for baseline in ('full','uniform','risk_top'):result.append((name(selector,core),name(baseline,core)))
        for other in ('joint0','graph_perm','iu','equal_graph010'):result.append((name(selector,'graph010'),name(selector,other)))
        result.append((name(selector,'iu'),name(selector,'equal')))
    for core in ('iu','graph010','equal_graph_perm'):result.append((name(SELECTORS[0],core),name(SELECTORS[1],core)))
    assert len(result)==len(set(result))==31
    return result


def inputs(rec):
    uid=rec['uid'];meta=load(SAMPLE/'scores'/(uid+'.json'));original=load(ORIGINAL/'scores'/(uid+'.json'))
    with np.load(SAMPLE/'scores'/(uid+'.npz'),allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    plan=moment_plan(rec['tokens'],8)
    for k,v in [('window_starts',plan.starts),('window_ends',plan.ends),('fit_indices',plan.fit_indices)]:np.testing.assert_array_equal(arrays[k],v)
    assert meta['row_id']==rec['row_id'] and meta['group_id']==rec['group_id']
    return arrays,meta,original,plan


def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def prepare():
    if (OUT/'MANIFEST.json').exists():verify();print('Existing frozen entropy-sampling manifest verified.');return
    priority_status=priority();tests=module(ROOT/'tests/test_fusion_entropy_sampling.py','entropy_selector_tests')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    save(OUT/'TESTS.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,output=stream.getvalue()))
    assert result.wasSuccessful(),stream.getvalue()
    source=load(SAMPLE/'MANIFEST.json');oldfreeze=load(SAMPLE/'SCORES_FROZEN.json');parent=load(PARENT/'EVALUATION.json')
    assert load(SAMPLE/'REVIEW.json')['status']=='PASS' and load(PARENT/'REVIEW.json')['status']=='PASS'
    assert oldfreeze['manifest_sha256']==sha(SAMPLE/'MANIFEST.json')
    selected=source['selected'];assert len(selected)==110 and len(parent['metrics'])==176
    cases={};eligible=0;paths=[Path(__file__),ROOT/'spectral_utils/fusion_entropy_sampling.py',
        ROOT/'tests/test_fusion_entropy_sampling.py',ROOT/'docs/experiments/FUSION_ENTROPY_SAMPLING_V1.md',
        OUT/'TESTS.json',SAMPLE/'MANIFEST.json',SAMPLE/'SCORES_FROZEN.json',SAMPLE/'REVIEW.json',
        PARENT/'EVALUATION.json',PARENT/'REVIEW.json',RELEASE,ROOT/'scripts/run_answer_localization_v2.py']
    release=load(RELEASE);population={(cell,r['row_id']):r for cell,c in release['cells'].items() for r in c['rows']}
    for rec in selected:
        for k in ('row','group_id','tokens','steps'):assert rec[k]==population[rec['cell'],rec['row_id']][k]
        for ext in ('.json','.npz'):
            p=SAMPLE/'scores'/(rec['uid']+ext);assert sha(p)==oldfreeze['files'][str(p)];paths.append(p)
        paths.append(ORIGINAL/'scores'/(rec['uid']+'.json'))
        meta=load(SAMPLE/'scores'/(rec['uid']+'.json'))
        if meta['diagnostics']['eligible']:
            eligible+=1;key=(meta['diagnostics']['bank'],meta['diagnostics']['fits']['risk_top']['joint_valid'])
            cases.setdefault(key,rec)
    assert eligible==72
    checked=[]
    for rec in list(cases.values())[:3]:
        a,meta,original,plan=inputs(rec);selected_rows=a['risk_top__selected']
        identity=source['scoring_namespace']+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
        out,methods,_=score_selected(a['features'],meta['diagnostics']['names'],plan,a['step_starts'],a['step_ends'],selected_rows,
                                      identity,original['methods']['moment__iu'])
        for core in CORES:
            old=meta['methods'][name('risk_top',core)]
            for k in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction'):assert methods[core].get(k)==old.get(k),(rec['uid'],core,k)
            if old['valid']:
                for suffix in ('window','risk'):np.testing.assert_allclose(out[core+'__'+suffix],a[name('risk_top',core)+'__'+suffix],atol=1e-10,rtol=1e-10)
        checked.append(rec['uid'])
    save(OUT/'PREFLIGHT.json',dict(status='PASS',high_only_replays=checked,cores_per_replay=7,eligible=eligible,source_joins=110))
    paths.append(OUT/'PREFLIGHT.json')
    paths += [Path(info['label_path']) for cell,info in release['cells'].items() if cell in {r['cell'] for r in selected}]
    for n,mod in list(sys.modules.items()):
        if n.startswith('spectral_utils') and getattr(mod,'__file__',None):paths.append(Path(mod.__file__).resolve())
    save(OUT/'MANIFEST.json',dict(status='FROZEN_DEVELOPMENT_ENTROPY_COVERAGE',selected=selected,
        release_id=release['release_id'],scoring_namespace=source['scoring_namespace'],new_arms=NEW_ARMS,
        external_arms=list(parent['metrics']),arms=list(parent['metrics'])+list(NEW_ARMS),contrasts=pairs(),
        expected_eligible=72,workers=1,score_seconds_cap=3600,priority=priority_status,labels_used_for_scoring=False,
        hashes={str(p):sha(p) for p in paths},created_unix=time.time()))
    print('PreflightPASS:4 selector tests,',len(checked),'high-only replays; frozen14 new outputs and176 anchors.',flush=True)


def one(rec,m,digest):
    uid=rec['uid'];npz=OUT/'scores'/(uid+'.npz');jp=npz.with_suffix('.json');started=time.monotonic()
    if jp.exists():
        old=load(jp);assert old['manifest_sha256']==digest and old['array_sha256']==sha(npz);return
    a,old,original,plan=inputs(rec);values=a['features'];names=old['diagnostics']['names'];reference=original['methods']['moment__iu']
    identity=m['scoring_namespace']+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
    out={k:a[k] for k in ('features','window_starts','window_ends','fit_indices','step_starts','step_ends')};methods={}
    diagnostic=dict(bank=old['diagnostics']['bank'],names=names,eligible=old['diagnostics']['eligible'],
                    original_rows=len(plan.fit_indices),budget=budget(len(plan.fit_indices)),fits={},labels_used=False)
    for selector in SELECTORS:
        indices=plan.fit_indices[choose(values[plan.fit_indices,0],selector)];out[selector+'__selected']=indices
        if np.array_equal(indices,plan.fit_indices):
            diagnostic['fits'][selector]=dict(replay=True,joint_valid=old['diagnostics']['original_joint_valid'])
            for core in CORES:
                arm=name(selector,core);source=name('full',core);methods[arm]=deepcopy(old['methods'][source])
                methods[arm].update(source_arm=source,anchor_replay=True)
                if methods[arm]['valid']:
                    for suffix in ('window','risk'):out[arm+'__'+suffix]=a[source+'__'+suffix].copy()
            continue
        fitted,native,detail=score_selected(values,names,plan,a['step_starts'],a['step_ends'],indices,identity,reference)
        diagnostic['fits'][selector]=dict(detail,replay=False)
        out.update({selector+'__'+k:v for k,v in fitted.items()})
        for core in CORES:
            arm=name(selector,core);methods[arm]=deepcopy(native[core]);methods[arm].update(anchor_replay=False,
                source_arm=selector+'__native_'+methods[arm]['source_core'],bank=diagnostic['bank'])
            if methods[arm]['valid']:
                for suffix in ('window','risk'):out[arm+'__'+suffix]=fitted[core+'__'+suffix].copy()
    assert set(methods)==set(NEW_ARMS)
    npz.parent.mkdir(parents=True,exist_ok=True);tmp=npz.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**out)
    replace(tmp,npz);save(jp,dict(**rec,methods=methods,diagnostics=diagnostic,routing=original['routing'],
         manifest_sha256=digest,array_sha256=sha(npz),seconds=time.monotonic()-started,labels_used=False))


def state(phase,completed,**extra):
    save(OUT/'RUN_STATE.json',dict(phase=phase,pid=os.getpid(),completed=completed,total=110,
        workers=1,updated_unix=time.time(),**extra))


def scores():
    m=verify();digest=sha(OUT/'MANIFEST.json');started=time.monotonic();p=priority();done=0;state('SCORING',0,priority=p)
    for rec in m['selected']:
        if time.monotonic()-started>m['score_seconds_cap']:state('PAUSED_AT_CAP',done);return False
        one(rec,m,digest);done+=1;state('SCORING',done,seconds=time.monotonic()-started,priority=p)
        if done%10==0:print('Entropy sampling',done,'/110',flush=True)
    verify();files={str(OUT/'scores'/(r['uid']+ext)):sha(OUT/'scores'/(r['uid']+ext)) for r in m['selected'] for ext in ('.json','.npz')}
    save(OUT/'SCORES_FROZEN.json',dict(status='COMPLETE',manifest_sha256=digest,files=files,labels_used=False,seconds=time.monotonic()-started))
    return True


def mm():return module(ROOT/'scripts/run_answer_localization_v2.py','entropy_metric_reference')
def fixed(rows):return [{**r,'decision_valid':r['fixed_iu_valid'],'predictions':r['fixed_iu_predictions']} for r in rows]


def evaluate():
    state('EVALUATING',110);m=verify();f=load(OUT/'SCORES_FROZEN.json');assert f['status']=='COMPLETE'
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json')
    for p,h in f['files'].items():assert sha(p)==h,p
    previous=load(PARENT/'EVALUATION.json');rows=deepcopy(previous['rows']);release=load(RELEASE)
    for cell in sorted({r['cell'] for r in rows}):
        with np.load(release['cells'][cell]['label_path'],allow_pickle=False) as labels:
            index={str(v):i for i,v in enumerate(labels['row_ids'])}
            assert len(index)==len(labels['row_ids'])
            for row in (r for r in rows if r['cell']==cell):
                i=index[row['row_id']]
                if cell.startswith('prm'):
                    lo,hi=labels['step_flag_offsets'][i:i+2];np.testing.assert_array_equal(row['target'],labels['step_error_flags'][lo:hi])
                else:assert row['target']==int(labels['first_error'][i])
                meta=load(OUT/'scores'/(row['uid']+'.json'));assert meta['routing']==row['routing'] and meta['group_id']==row['group_id']
                row['entropy_sampling']={k:meta['diagnostics'][k] for k in ('bank','eligible','original_rows','budget')}
                with np.load(OUT/'scores'/(row['uid']+'.npz'),allow_pickle=False) as a:
                    for arm,d in meta['methods'].items():
                        for k in ('valid','decision_valid','fixed_iu_valid'):row[k][arm]=d[k]
                        for dst,src in [('predictions','prediction'),('fixed_iu_predictions','fixed_iu_prediction'),('peaks','peak')]:row[dst][arm]=d.get(src)
                        row['sources'][arm]=d.get('source_arm')
                        if d['valid']:
                            risk=a[arm+'__risk'];assert risk.shape==(row['steps'],) and np.isfinite(risk).all();row['scores'][arm]=risk.tolist()
    metric=mm()
    def bundle(subset,arms):
        fx=fixed(subset);return {a:dict(prm=metric.prm_metric(subset,a),pb=metric.pb_metric(subset,a),pb_common_iu_gate=metric.pb_metric(fx,a)) for a in arms}
    metrics=bundle(rows,m['arms']);eligible=[r for r in rows if r['entropy_sampling']['eligible']];assert len(eligible)==72
    for arm in m['external_arms']:assert metrics[arm]==previous['metrics'][arm],arm
    save(OUT/'EVALUATION.json',dict(status='DEVELOPMENT_COMPARISON',release_id=m['release_id'],rows=rows,metrics=metrics,
        eligible_metrics=bundle(eligible,[name(s,c) for s in DISPLAY_SELECTORS for c in CORES]),scores_sha256=sha(OUT/'SCORES_FROZEN.json')))


def contrasts():
    state('PAIRED_COMPARISONS',110);m=verify();e=load(OUT/'EVALUATION.json');metric=mm();path=OUT/'CONTRASTS.json'
    out=load(path) if path.exists() else dict(evaluation_sha256=sha(OUT/'EVALUATION.json'),pairs={})
    assert out['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    for left,right in m['contrasts']:
        key=left+' minus '+right
        if key in out['pairs']:continue
        common=[r for r in e['rows'] if r['valid'][left] and r['valid'][right]]
        out['pairs'][key]=dict(left=left,right=right,left_prm=metric.prm_metric(common,left),right_prm=metric.prm_metric(common,right),
            left_pb=metric.pb_metric(e['rows'],left),right_pb=metric.pb_metric(e['rows'],right),
            uncertainty=paired_source_group_intervals(e['rows'],left,right))
        save(path,out)
    out['status']='COMPLETE';save(path,out)


def independent_auc(y,x):
    y=np.asarray(y);x=np.asarray(x);p=int(np.sum(y==1));n=int(np.sum(y==0))
    if not p or not n:return None
    return float((rankdata(x)[y==1].sum()-p*(p+1)/2)/(p*n))


def review():
    state('REVIEWING',110);m=verify();e=load(OUT/'EVALUATION.json');checks=dict(selections=0,linear_outputs=0,dense_readouts=0,refitted_banks=0,metrics=0)
    refits=set(load(OUT/'PREFLIGHT.json')['high_only_replays']);maximum=0.
    for rec in m['selected']:
        a,old,original,plan=inputs(rec);meta=load(OUT/'scores'/(rec['uid']+'.json'))
        assert meta['array_sha256']==sha(OUT/'scores'/(rec['uid']+'.npz'))
        with np.load(OUT/'scores'/(rec['uid']+'.npz'),allow_pickle=False) as saved:
            entropy=a['features'][plan.fit_indices,0];n=len(entropy);count=min(n,max(32,(n+1)//2))
            for selector in SELECTORS:
                if count==n:expected=list(range(n))
                elif selector=='entropy_tails':
                    low=sorted(range(n),key=lambda i:(entropy[i],i))[:count//2]
                    high=sorted(set(range(n))-set(low),key=lambda i:(-entropy[i],i))[:count-len(low)];expected=sorted(low+high)
                else:
                    order=sorted(range(n),key=lambda i:(entropy[i],i));expected=sorted(order[((2*j+1)*n)//(2*count)] for j in range(count))
                indices=plan.fit_indices[expected];np.testing.assert_array_equal(saved[selector+'__selected'],indices);checks['selections']+=1
                fit=meta['diagnostics']['fits'][selector];z=None
                if not fit['replay'] and 'z' in {k.removeprefix(selector+'__') for k in saved.files if k.startswith(selector+'__')}:
                    shared=fit['shared'];columns=[old['diagnostics']['names'].index(k) for k in shared['active_features']]
                    values=a['features'][:,columns];mean=values[indices].mean(0);sd=values[indices].std(0)
                    np.testing.assert_allclose(mean,shared['mean'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(sd,shared['sd'],atol=1e-12,rtol=1e-12)
                    z=(values-mean)/sd;z-=z[indices].mean(0);z*=shared['feature_signs']
                    np.testing.assert_allclose(z,saved[selector+'__z'],atol=1e-10,rtol=1e-10)
                for core in CORES:
                    arm=name(selector,core);detail=meta['methods'][arm]
                    if not detail['valid']:continue
                    risk=saved[arm+'__window']
                    if fit['replay']:
                        np.testing.assert_array_equal(risk,a[name('full',core)+'__window'])
                    else:
                        assert detail['fallback_to_sample_iu']==(core in JOINT_CORES and not fit['joint_valid'])
                        expected_risk=-z@np.asarray(detail['standardized_weights']);delta=float(np.max(np.abs(expected_risk-risk)))
                        maximum=max(maximum,delta);np.testing.assert_allclose(expected_risk,risk,atol=1e-10,rtol=1e-10);checks['linear_outputs']+=1
                    step,replayed=apply_readout(risk,detail,plan,a['step_starts'],a['step_ends'],original['methods']['moment__iu'])
                    np.testing.assert_allclose(step,saved[arm+'__risk'],atol=1e-10,rtol=1e-10)
                    for k in ('decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):assert replayed.get(k)==detail.get(k),(rec['uid'],arm,k)
                    checks['dense_readouts']+=1
                if rec['uid'] in refits and not fit['replay']:
                    identity=m['scoring_namespace']+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
                    result,methods,_=score_selected(a['features'],old['diagnostics']['names'],plan,a['step_starts'],a['step_ends'],indices,identity,original['methods']['moment__iu'])
                    for core in CORES:
                        arm=name(selector,core);assert methods[core]['valid']==meta['methods'][arm]['valid']
                        if methods[core]['valid']:np.testing.assert_allclose(result[core+'__window'],saved[arm+'__window'],atol=1e-10,rtol=1e-10)
                    checks['refitted_banks']+=1
    for arm in NEW_ARMS:
        valid=[r for r in e['rows'] if r['cell'].startswith('prm') and r['valid'][arm]]
        auc=independent_auc(np.concatenate([r['target'] for r in valid]),np.concatenate([r['scores'][arm] for r in valid])) if valid else None
        target=e['metrics'][arm]['prm']['auroc']
        if auc is None:assert target is None
        else:np.testing.assert_allclose(auc,target,atol=1e-14,rtol=0)
        for cell,detail in e['metrics'][arm]['pb']['cells'].items():
            rows=[r for r in e['rows'] if r['cell']==cell];clean=[r for r in rows if r['target']==-1];err=[r for r in rows if r['target']!=-1]
            c=sum(r['decision_valid'][arm] and r['predictions'][arm]==-1 for r in clean)/len(clean)
            b=sum(r['decision_valid'][arm] and r['predictions'][arm]==r['target'] for r in err)/len(err)
            np.testing.assert_allclose([c,b,2*c*b/(c+b) if c+b else 0],[detail['clean_accuracy'],detail['error_exact_accuracy'],detail['f1']],atol=1e-14,rtol=0)
        checks['metrics']+=1
    assert len(load(OUT/'CONTRASTS.json')['pairs'])==31
    save(OUT/'REVIEW.json',dict(status='PASS',checks=checks,max_linear_discrepancy=maximum,evaluation_sha256=sha(OUT/'EVALUATION.json'),
        contrasts_sha256=sha(OUT/'CONTRASTS.json'),scope='Same-session independent selection/linear/rank/decision arithmetic; representative shared-kernel fits and reused audited bootstrap. Not external review.'))


def report():
    e=load(OUT/'EVALUATION.json');reviewed=load(OUT/'REVIEW.json');assert reviewed['status']=='PASS'
    pretty={'full':'All windows','uniform':'Uniform in time','risk_top':'High entropy only','entropy_tails':'Half low + half high','entropy_quantiles':'Across entropy quantiles'}
    records=[];rendered=[]
    for selector in DISPLAY_SELECTORS:
        for core in CORES:
            arm=name(selector,core);m=e['metrics'][arm];eligible=e['eligible_metrics'][arm]
            record=dict(selector=selector,core=core,prm_auc=m['prm']['auroc'],within_auc=m['prm']['within_answer_auc'],pb_f1=m['pb']['macro_f1'],
                prm_valid=m['prm']['answers'],pb_valid=sum(v['valid_decisions'] for v in m['pb']['cells'].values()),
                eligible_prm_auc=eligible['prm']['auroc'],eligible_pb_f1=eligible['pb']['macro_f1'])
            records.append(record)
            def fmt(value,percent=False):return '-' if value is None else (f'{100*value:.2f}%' if percent else f'{value:.4f}')
            rendered.append('<tr><td>'+pretty[selector]+'</td><td>'+core+'</td><td>'+fmt(record['prm_auc'])+'</td><td>'+fmt(record['within_auc'])+'</td><td>'+fmt(record['pb_f1'],True)+'</td><td>'+str(record['prm_valid'])+'/24; '+str(record['pb_valid'])+'/86</td></tr>')
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (OUT/'METRICS.csv').write_text(buf.getvalue(),encoding='utf-8')
    contrasts=load(OUT/'CONTRASTS.json');comparison=[]
    for key,d in contrasts['pairs'].items():
        u=d['uncertainty'];comparison.append('<tr><td>'+html.escape(key)+'</td><td>'+html.escape(str(u['prm_common_valid_ci95']))+'</td><td>'+html.escape(str(u['prm_within_answer_common_valid_ci95']))+'</td><td>'+html.escape(str(u['pb_all_population_ci95']))+'</td></tr>')
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Entropy coverage sampling</title>
<style>body{font:16px/1.55 system-ui;color:#193044;background:#f6f8fa;margin:24px auto;max-width:1250px;padding:20px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #d8e1e8}th{background:#e6edf4}.scroll{overflow:auto}.note{padding:18px;border-left:5px solid #ae722e;background:white}a{color:#136c98}</style>
<h1>Does entropy coverage help answer-only fusion?</h1><p>Completed development comparison; scientific checks PASS. Same110 answers:24 PRMBench and86 ProcessBench-Qwen3-8B.72 eligible for reduction;38 replay all windows. This does not complete the separate full benchmark or historical refits.</p>
<div class="note">Two new selectors use the same window budget as high-only: half low plus half high entropy, or evenly spaced entropy mid-quantiles. Both fit IU/Joint using selected windows and score EVERY original window. Normalization also changes with selection. Entropy is not an error label. There is no sparse-inference or weight-only causal claim.</div>
<p>Seven cores include the unchanged condition100 Joint lambda0/graph/permutation and equal graph controls. Joint fit failures fall back to selected-row IU. All176 prior entries remain in the evaluation JSON;14 additions give190 entries, not190 independent discoveries.</p>
<h2>Matched selector table</h2><p>PB is a harmonic first-error/no-error score, not general classification F1. PRMB pooled AUROC is not official PRMScore. Also inspect within-answer AUROC and coverage. Invalid PB decisions count as failures.</p>
<div class="scroll"><table><thead><tr><th>Selection</th><th>Fusion</th><th>PRMB pooled AUC</th><th>Within-answer AUC</th><th>PB score</th><th>Valid PRMB/PB</th></tr></thead><tbody>TABLE</tbody></table></div>
<h2>All31 registered paired comparisons</h2><p>95% exploratory source-group bootstrap intervals for left minus right, in native0-1 units. PB includes all86 answers; PRMB uses common valid answers. These are not multiplicity-adjusted confirmation. No publication winner is selected by this driver.</p>
<div class="scroll"><table><thead><tr><th>Comparison</th><th>Pooled AUC delta</th><th>Within-answer delta</th><th>PB delta</th></tr></thead><tbody>PAIRS</tbody></table></div>
<p><a href="METRICS.csv">Metrics CSV, including eligible-only endpoints</a> | <a href="EVALUATION.json">All190 metric bundles and per-answer outputs</a> | <a href="CONTRASTS.json">Paired details</a> | <a href="REVIEW.json">Review scope</a> | <a href="MANIFEST.json">Frozen manifest</a> | <a href="../../docs/experiments/FUSION_ENTROPY_SAMPLING_V1.md">Protocol</a></p>
<p>Reused audited numerical kernels/metrics; independent selection, linear-score and rank/decision checks plus representative refits. No external review or browser rendering claimed.</p></html>'''
    (OUT/'REPORT.html').write_text(text.replace('TABLE',''.join(rendered)).replace('PAIRS',''.join(comparison)),encoding='utf-8')
    assert len(records)==35 and len(comparison)==31
    save(OUT/'ARTIFACT_CHECKS.json',dict(status='PASS',metric_rows=35,paired_rows=31,report_sha256=sha(OUT/'REPORT.html'),browser_rendered=False))
    state('COMPLETE',110,report=str(OUT/'REPORT.html'));print('Entropy sampling COMPLETE, reviewPASS:',OUT/'REPORT.html',flush=True)


def run():
    try:
        if scores():evaluate();contrasts();review();report()
    except Exception as exc:
        save(OUT/'FAILURE.json',dict(reason=repr(exc),time=time.time()))
        state('FAILED_CHECKPOINTS_PRESERVED',len(list((OUT/'scores').glob('*.json'))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','run'],required=True)
    args=parser.parse_args();prepare() if args.phase=='prepare' else run()
