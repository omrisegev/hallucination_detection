"""Automatic full pass2a evaluation using the reviewed full-anchor kernels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from collections import Counter
import csv
import hashlib
import html
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/localization_full_shortlist_v3'
OUT=RUN/'evaluation'
PARENT=ROOT/'results/localization_full_benchmark_v3/evaluation'
PILOT=ROOT/'results/fusion_entropy_sampling_v1/EVALUATION.json'


def reference():
    spec=importlib.util.spec_from_file_location('reviewed_full_anchor_kernels',ROOT/'scripts/evaluate_localization_full_anchors_v3.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    ref.RUN=RUN;ref.OUT=OUT
    return ref


def extra_pairs():
    pairs=[('dual__cond100_graph010',r) for r in ('dual__cond100','dual__cond100_graph_perm','dual__equal_graph010','dual__graph010')]
    components={'iu_joint_graph':('dual__iu','dual__cond100_graph010'),
                'iu_joint0':('dual__iu','dual__cond100'), 'iu_joint_perm':('dual__iu','dual__cond100_graph_perm'),
                'equal_graph':('dual__equal','dual__equal_graph010'),'equal_perm':('dual__equal','dual__equal_graph_perm')}
    for family,sources in components.items():
        modes=('mean','gls','hold','imm') if family=='iu_joint_graph' else ('mean','gls')
        for mode in modes:
            for source in sources:
                if source!='dual__iu':pairs.append((f'traj_{family}__{mode}',source))
    for mode in ('mean','gls'):
        for other in ('iu_joint0','iu_joint_perm','equal_graph','equal_perm'):
            pairs.append((f'traj_iu_joint_graph__{mode}',f'traj_{other}__{mode}'))
    pairs.append(('traj_iu_joint_graph__imm','traj_iu_joint_graph__hold'))
    pairs.append(('traj_iu_joint_graph__hold','traj_iu_joint_graph__gls'))
    assert len(pairs)==len(set(pairs))
    return pairs


def supplementary_intervals(ref,records,arms,a,metrics,output):
    groups=sorted({r['group_id'] for r in records});lookup={g:i for i,g in enumerate(groups)}
    gi=np.array([lookup[r['group_id']] for r in records]);ng=len(groups)
    weights=np.random.default_rng(ref.SEED).multinomial(ng,np.full(ng,1/ng),size=ref.DRAWS)
    owner=np.repeat(np.arange(len(records)),np.diff(a['offsets']))
    prm=np.array([r['cell'].startswith('prm') for r in records]);cache={};pb={}
    def distribution(arm,common):
        j=arms.index(arm);key=(arm,np.packbits(common).tobytes())
        if key in cache:return cache[key]
        mask=prm&common;steps=mask[owner];mixed=mask&np.isfinite(a['within'][:,j])
        plan=ref.auc_plan(a['labels'][steps],a['scores'][steps,j],gi[owner[steps]])
        points=np.array([ref.weighted_auc(plan,w) for w in weights])
        numer=np.bincount(gi[mixed],weights=a['within'][mixed,j],minlength=ng)
        denom=np.bincount(gi[mixed],minlength=ng)
        within=(weights@numer)/(weights@denom)
        result=(points,within,dict(answers=int(mask.sum()),mixed_answers=int(mixed.sum()),
             auroc=ref.weighted_auc(plan,np.ones(ng,int)),within_answer_auc=float(a['within'][mixed,j].mean())))
        cache[key]=result;return result
    def pb_distribution(arm):
        if arm in pb:return pb[arm]
        j=arms.index(arm);draws={};success=a['decision'][:,j]&(a['predictions'][:,j]==a['target'])
        for cell in metrics[arm]['pb']['cells']:
            rows=np.array([r['cell']==cell for r in records]);clean=rows&(a['target']==-1);err=rows&(a['target']>=0)
            count=lambda mask:weights@np.bincount(gi[mask],minlength=ng)
            ca,ea=count(clean&success)/count(clean),count(err&success)/count(err)
            draws[cell]=np.divide(2*ca*ea,ca+ea,out=np.zeros_like(ca),where=ca+ea!=0)
        pb[arm]={panel:np.mean([v for cell,v in draws.items() if panel=='all' or cell.endswith(panel)],axis=0) for panel in ('q4','q8','all')}
        return pb[arm]
    def ci(x):
        x=x[np.isfinite(x)];return dict(ci95=np.quantile(x,[.025,.975]).tolist(),valid_draws=len(x))
    pairs=extra_pairs()
    for i,(left,right) in enumerate(pairs):
        common=a['valid'][:,arms.index(left)]&a['valid'][:,arms.index(right)]
        dl,wl,pl=distribution(left,common);dr,wr,pr=distribution(right,common)
        output['paired'][left+' minus '+right]=dict(left=left,right=right,prm_left=pl,prm_right=pr,
            prm_delta=pl['auroc']-pr['auroc'],within_delta=pl['within_answer_auc']-pr['within_answer_auc'],
            prm=ci(dl-dr),within=ci(wl-wr),pb={panel:dict(delta=metrics[left]['pb']['macros'][panel]-metrics[right]['pb']['macros'][panel],
                **ci(pb_distribution(left)[panel]-pb_distribution(right)[panel])) for panel in ('q4','q8','all')})
        ref.save(OUT/'INTERVALS.json',output);ref.state('NEW_CANDIDATE_CONTRASTS',completed=i+1,total=len(pairs))
    return output


def main():
    OUT.mkdir(exist_ok=True);ref=reference();start=time.time();ref.bootstrap_preflight()
    m=ref.load(RUN/'MANIFEST.json');f=ref.load(RUN/'SCORES_FROZEN.json')
    assert f['status']=='COMPLETE_SHORTLIST_PASS2A' and f['rows']==13769
    assert f['manifest_sha256']==ref.sha(RUN/'MANIFEST.json')
    for path,h in m['hashes'].items():assert ref.sha(path)==h,path
    if (OUT/'JOINED.json').exists():
        joined=ref.load(OUT/'JOINED.json');assert joined['records']==m['selected'] and joined['arms']==m['arms']
        assert joined['scores_freeze_sha256']==ref.sha(RUN/'SCORES_FROZEN.json')
        assert joined['arrays_sha256']==ref.sha(OUT/'JOINED.npz')
        records,arms,sources=joined['records'],joined['arms'],joined['sources']
        with np.load(OUT/'JOINED.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    else:records,arms,sources,a=ref.prepare_inputs(m,f)
    ref.state('METRICS');metrics={arm:ref.metric(records,a,j) for j,arm in enumerate(arms)}
    parent=ref.load(PARENT/'METRICS.json')['metrics']
    for arm in m['external_arms']:assert metrics[arm]==parent[arm],arm
    pilot=ref.load(PILOT);lookup={r['uid']:i for i,r in enumerate(records)};selected=np.zeros(len(records),bool)
    for row in pilot['rows']:
        i=lookup[row['uid']];selected[i]=True;lo,hi=a['offsets'][i:i+2]
        for j,arm in enumerate(arms):
            assert a['valid'][i,j]==row['valid'][arm] and a['decision'][i,j]==row['decision_valid'][arm]
            if row['valid'][arm]:np.testing.assert_allclose(a['scores'][lo:hi,j],row['scores'][arm],atol=1e-10,rtol=1e-10)
            if row['decision_valid'][arm]:assert a['predictions'][i,j]==row['predictions'][arm]
    for j,arm in enumerate(arms):
        got=ref.metric(records,a,j,selected);old=pilot['metrics'][arm]
        for key in ('auroc','within_answer_auc'):assert abs(got['prm'][key]-old['prm'][key])<1e-14,(arm,key)
        assert abs(got['pb']['macros']['q8']-old['pb']['macro_f1'])<1e-14,arm
        mask=(a['labels']>=0)&np.repeat(a['valid'][:,j],np.diff(a['offsets']))
        assert abs(ref.roc_auc_score(a['labels'][mask],a['scores'][mask,j])-metrics[arm]['prm']['auroc'])<1e-14
    ref.save(OUT/'METRICS.json',dict(status='POINT_METRICS_VERIFIED_INTERVALS_PENDING',metrics=metrics,sources=sources))
    print('Full shortlist point metrics verified; computing paired intervals.',flush=True)
    uncertainty=ref.intervals(records,arms,a,metrics)
    uncertainty=supplementary_intervals(ref,records,arms,a,metrics,uncertainty)
    expected_pairs=len(arms)-1+8+len(extra_pairs());assert len(uncertainty['paired'])==expected_pairs
    uncertainty['status']='COMPLETE';ref.save(OUT/'INTERVALS.json',uncertainty)
    # Reconcile source provenance with actual validity, including each task/cell.
    coverage=Counter()
    for i,rec in enumerate(records):
        d=ref.load(RUN/'scores'/(rec['uid']+'.json'))
        for j,arm in enumerate(arms):
            info=d['methods'][arm]
            assert bool(a['valid'][i,j])==info['valid'] and bool(a['decision'][i,j])==info['decision_valid']
            coverage[(rec['cell'],arm,info.get('source_arm') or 'UNSPECIFIED',info['valid'],info['decision_valid'])]+=1
    coverage_rows=[dict(cell=k[0],method=k[1],source_arm=k[2],score_valid=k[3],decision_valid=k[4],answers=v) for k,v in sorted(coverage.items())]
    ref.save(OUT/'SOURCE_VALIDITY_COVERAGE.json',dict(records=coverage_rows))
    rows=[];rendered=[]
    for arm,d in metrics.items():
        row=dict(method=arm,prm_auc=d['prm']['auroc'],within_auc=d['prm']['within_answer_auc'],prm_valid=d['prm']['answers'],
            pb_q4=d['pb']['macros']['q4'],pb_q8=d['pb']['macros']['q8'],pb_all=d['pb']['macros']['all'],
            pb_valid=sum(c['valid_decisions'] for c in d['pb']['cells'].values()))
        rows.append(row);rendered.append('<tr><td>'+arm+'</td><td>'+f'{row["prm_auc"]:.4f}'+'</td><td>'+f'{row["within_auc"]:.4f}'+'</td><td>'+str(row['prm_valid'])+'/6969</td><td>'+f'{100*row["pb_q4"]:.2f}%'+'</td><td>'+f'{100*row["pb_q8"]:.2f}%'+'</td><td>'+f'{100*row["pb_all"]:.2f}%'+'</td><td>'+str(row['pb_valid'])+'/6800</td></tr>')
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'METRICS.csv').write_text(buf.getvalue(),encoding='utf-8')
    comparisons=['<tr><td>'+html.escape(k)+'</td><td>'+str(d['prm_left']['answers'])+'</td><td>'+str(d['prm']['ci95'])+'</td><td>'+str(d['within']['ci95'])+'</td><td>'+str(d['pb']['q8']['ci95'])+'</td></tr>' for k,d in uncertainty['paired'].items()]
    report='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full localization shortlist: pass2a</title><style>body{font:16px/1.55 system-ui;max-width:1300px;margin:30px auto;padding:20px;color:#21364c}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #cdd8e2;text-align:left}.scroll{overflow:auto}.note{background:#fff2d5;padding:20px}</style><h1>Full localization shortlist: pass2a</h1><p class="note">All13769 registered model-answer rows;19 original anchors and17 previously tested additions. Conditioning, graph controls and trajectory combinations are evaluated here. Token/window sampling remains pass2b, and corrected historical refits remain pending. This is exposed development data, not untouched confirmation.</p><p>Original dual__joint0/graph010 use condition1000. New dual__cond100 maps use condition100. Equal-graph controls use identity covariance and uniform loading; trajectory equal_graph/equal_perm combine plain equal with the corresponding graph-equal curve. Fixed original routes and IU fallback are retained. Every fit is within the current answer.</p><p>PRMB: pooled and within-answer AUROC, not official PRMScore. Invalid PRMB scores have explicit coverage. PB: first-error/no-error harmonic score, averaged per cell; all invalid decisions are failures. Q4/Q8 are four-cell macros. All110 pilot cases replay, and all19 full anchor metric bundles remain unchanged.</p><div class="scroll"><table><tr><th>Method</th><th>PRMB pooled</th><th>Within answer</th><th>PRMB valid</th><th>PB Q4</th><th>PB Q8</th><th>PB all</th><th>PB valid</th></tr>ROWS</table></div><h2>Paired exploratory95% intervals</h2><p>Left minus right; native0-1 units. PRMB uses common-valid answers.1000 joint canonical-group draws preserve repeated scorers, repeated answers and cross-task source links. No multiplicity adjustment or automatic publication-winner claim.</p><div class="scroll"><table><tr><th>Comparison</th><th>Common PRMB</th><th>Pooled delta</th><th>Within delta</th><th>PB Q8 delta</th></tr>PAIRS</table></div><p><a href="METRICS.csv">Summary CSV</a> · <a href="METRICS.json">All cell metrics</a> · <a href="INTERVALS.json">Absolute and paired intervals</a> · <a href="SOURCE_VALIDITY_COVERAGE.json">Source/validity coverage</a> · <a href="REVIEW.json">Review</a></p></html>'''
    report=report.replace('ROWS',''.join(rendered)).replace('PAIRS',''.join(comparisons));(OUT/'REPORT.html').write_text(report,encoding='utf-8')
    class Parsed(HTMLParser):
        def __init__(self):super().__init__();self.rows=0;self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='tr':self.rows+=1
            if tag=='a':self.links.append(dict(attrs)['href'])
    parsed=Parsed();parsed.feed(report);assert parsed.rows==len(arms)+expected_pairs+2
    for path,h in m['hashes'].items():assert ref.sha(path)==h,path
    ref.save(OUT/'METRICS.json',dict(status='COMPLETE_REVIEWED_PASS2A',metrics=metrics,sources=sources))
    review=dict(status='PASS',joined_rows=len(records),source_file_hashes=len(f['files']),pilot_answers_replayed=110,
        pilot_metric_bundles=len(arms),unchanged_full_anchor_bundles=19,independent_full_auc_checks=len(arms),
        weighted_auc_tie_tests=4,source_groups=uncertainty['source_groups'],paired_contrasts=expected_pairs,
        source_validity_groups=len(coverage_rows),report_rows=parsed.rows,browser_rendered=False,
        seconds=time.time()-start,scope='Same-session source, metric, input and numerical replay review; not external confirmation.',
        report_sha256=ref.sha(OUT/'REPORT.html'),metrics_sha256=ref.sha(OUT/'METRICS.json'),intervals_sha256=ref.sha(OUT/'INTERVALS.json'))
    ref.save(OUT/'REVIEW.json',review)
    for link in parsed.links:assert (OUT/link).is_file()
    ref.state('COMPLETE_REVIEWED_PASS2A',rows=len(records),methods=len(arms),historical_comparison_complete=False)
    print('Full shortlist evaluation and review PASS.',flush=True)


if __name__=='__main__':main()
