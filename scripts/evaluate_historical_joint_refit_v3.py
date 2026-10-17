"""Evaluate the combined full historical control and Joint panels."""
import hashlib
import html
import importlib.util
from pathlib import Path
import re
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/historical_joint_refit_v3'
BASE=ROOT/'results/historical_fusion_refit_v3'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    output=importlib.util.module_from_spec(spec);spec.loader.exec_module(output)
    return output


base=module('historical_original_evaluator','scripts/evaluate_historical_fusion_refit_v3.py')
base.OUT=OUT
mechanism=module('historical_joint_metric_core','spectral_utils/historical_joint_evaluation.py')


def source_control_review(manifest):
    original=base.load(BASE/'MANIFEST.json');checked=0;fallbacks={};status={};convergence={}
    for job in manifest['jobs']:
        name=base.job_name(job);path=OUT/'fits'/name;source=BASE/'fits'/name
        meta=base.load(path.with_suffix('.json'));old=base.load(source.with_suffix('.json'))
        assert meta['source_control_array_sha256']==old['array_sha256']==base.sha(source.with_suffix('.npz'))
        assert meta['source_control_metadata_sha256']==base.sha(source.with_suffix('.json'))
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as current, np.load(source.with_suffix('.npz'),allow_pickle=False) as reference:
            for key in reference.files:np.testing.assert_array_equal(current[key],reference[key]);checked+=1
        for arm in original['arms']:
            assert meta['methods'].get(arm)==old['methods'].get(arm)
            assert meta['failures'].get(arm)==old['failures'].get(arm)
        audit=meta['joint_audit'];label=str(audit['internal_grouping_status'])
        status[label]=status.get(label,0)+1
        for event in audit['fallback_events']:
            key=event['row']+'/'+event['fallback'];fallbacks[key]=fallbacks.get(key,0)+1
        for arm,d in meta['methods'].items():
            if 'joint_converged' in d:
                key=arm+'/'+str(d['joint_converged']);convergence[key]=convergence.get(key,0)+1
    return dict(status='PASS',exact_control_arrays=checked,grouping_counts=status,
                provenance_fallback_events=fallbacks,reported_convergence=convergence,
                limitation='Only convergence flags exposed by the historical entry point are counted; no added Jacobian gate')


def render(result,intervals,mechanisms,review):
    esc=html.escape
    def f(value,pct=False):
        if value is None or not np.isfinite(value):return 'N/A'
        return f'{value*100:.2f}%' if pct else f'{value:.4f}'
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    main_rows=[];cell_rows=[]
    for arm,d in result.items():
        p=d['prm'];b=d['pb']['macros']
        main_rows.append([arm,d['access'],f(p['fold_mean_auc']),f(p['within_answer_auc']),
            str(p['valid_answers'])+'/'+str(p['total_answers']),f(b['q4'],True),f(b['q8'],True),f(b['all'],True)])
        for cell,c in d['pb']['cells'].items():
            cell_rows.append([arm,cell,f(c['f1'],True),f(c['clean_accuracy'],True),f(c['error_exact_accuracy'],True),
                f(c['raw_peak_accuracy'],True),str(int(c['valid_decisions']))+'/'+str(int(c['answers']))])
    mechanism_rows=[]
    for c in mechanisms['contrasts']:
        for endpoint,e in c['endpoints'].items():
            mechanism_rows.append([c['candidate'],c['control'],endpoint,f(e['difference']),f(e['low']),f(e['high']),c['prm_common_answers']])
    control_rows=[]
    for c in intervals['contrasts']:
        for endpoint,e in c['intervals'].items():control_rows.append([c['candidate'],c['control'],endpoint,f(e['low']),f(e['high']),e['finite_draws']])
    audit=review['source_controls'];fit_rows=[]
    for key,count in audit['grouping_counts'].items():fit_rows.append(['Grouping status',key,count])
    for key,count in audit['provenance_fallback_events'].items():fit_rows.append(['Fallback event',key,count])
    for key,count in audit['reported_convergence'].items():fit_rows.append(['Exposed convergence flag',key,count])
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Full Joint and historical fusion benchmark</title>
<style>body{font:16px/1.55 system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#172838}p{max-width:1050px}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #ccd8e2;text-align:left}th{background:#e7eff6}.scroll{overflow-x:auto}a{color:#075ca5}summary{cursor:pointer;font-weight:bold;margin:20px 0}</style>
<h1>Full Joint and historical fusion benchmark</h1>
<p>This table evaluates all 13,769 model-answer records: 19 original answer-only anchors, five corrected historical controls and ten historical Joint/L-SML variants. These are exposed development data. The broader benchmark and untouched confirmation are still unfinished.</p>
<p>The answer-only methods learn from each answer independently. Historical controls learn from other training answers, and their ProcessBench no-error threshold uses labels inside nested training folds. Evaluation targets, source groups and held-out rows are shared. The methods' access, representation and readout differ; this end-to-end comparison does not isolate the fusion equation.</p>
<p>PRMBench uses the mean of five held-out-fold AUCs, plus within-answer AUC. Paired PRMB comparisons use common valid answers. Every ProcessBench failure remains in the denominator. The reported source-group intervals condition on fitted weights and calibrated thresholds; they do not cover refit or method-selection uncertainty.</p>
<p><a href="../../docs/experiments/HISTORICAL_JOINT_REFIT_V3.md">Protocol</a> · <a href="METRICS.json">Metrics</a> · <a href="MECHANISM_CONTRASTS.json">Mechanism contrasts</a> · <a href="REVIEW.json">Review</a> · <a href="../localization_full_benchmark_v3/METHOD_REGISTRY.json">Remaining comparator registry</a></p>
<h2>Full matched evaluation</h2>'''
    content+=table(['Method','Fit / decision access','PRMB fold AUC','PRMB within-answer','PRMB coverage','PB Q4','PB Q8','PB all'],main_rows)
    content+='''<h2>What belongs to the graph contribution?</h2><p>Internal Joint uses hierarchical weights. Model-inverse lambda0 uses a different map and is the proper zero-penalty reference for LIU and diagonal variants. A gain over hierarchical Joint alone does not establish a graph benefit. The node-permutation control tests geometry at lambda .1; there is no matched .5 permutation in this panel. Diagonal penalties do not introduce graph geometry.</p><p>Differences below are in score units: 0.01 means one ProcessBench percentage point. These are fixed exploratory comparisons, not multiplicity-adjusted confirmation.</p>'''
    content+=table(['Candidate','Control','Endpoint','Difference','95% low','95% high','Common PRMB answers'],mechanism_rows)
    content+='<details><summary>All comparisons against answer-only IU and equal active23</summary>'+table(['Candidate','Control','Endpoint','95% low','95% high','Valid draws'],control_rows)+'</details>'
    content+='<details><summary>Every ProcessBench dataset and scorer</summary>'+table(['Method','Cell','PB score','Clean accuracy','Exact error','Raw peak','Decision coverage'],cell_rows)+'</details>'
    content+='<h2>Fitting and fallback audit</h2>'+table(['Check','Value','Fits / events'],fit_rows)
    content+='''<p>The adapter preserves the historical grouping, minimum group size, gate learning, random seeds and provenance fallback rules. It does not add a new structural admissibility or Jacobian guard. Only convergence flags exposed by the historical function are counted. Every previously computed control array is checked for exact equality before evaluation.</p>
<p>Review is automated and same-session: source hashes, full joins, train/test isolation, old-fold replay and metric fixtures. No external scientific or browser review is claimed. Full sampling, other historical feature contracts and dedicated localizers remain in the continuing registry. Small future runs are feasibility checks only; method selection requires the full benchmark and later untouched confirmation.</p></html>'''
    (OUT/'REPORT.html').write_text(content,encoding='utf-8')
    for link in re.findall(r'href="([^"]+)"',content):assert (OUT/link).resolve().exists(),link


def main():
    started=time.time();manifest=base.load(OUT/'MANIFEST.json')
    for path,h in manifest['hashes'].items():assert base.sha(path)==h,path
    fixtures=base.metrics.review_fixtures();controls=source_control_review(manifest)
    records,arms,a,outer,checks=base.join(manifest)
    result=base.point_metrics(records,arms,a,outer)
    intervals=base.bootstrap(records,arms,a,outer,manifest['arms'])
    mechanisms=mechanism.contrasts(records,arms,a,outer,base.metrics)
    review=dict(status='PASS',scope='same-session automated control replay, isolation, joins and metrics',
        rows=len(records),arms=len(arms),fits=len(manifest['jobs']),fixtures=fixtures,source_controls=controls,
        checks=checks,external_review=False,browser_review=False,seconds=time.time()-started)
    base.save(OUT/'METRICS.json',dict(status='FULL_DEVELOPMENT_HISTORICAL_JOINT_EXTENSION',metrics=result))
    base.save(OUT/'INTERVALS.json',intervals);base.save(OUT/'MECHANISM_CONTRASTS.json',mechanisms)
    base.save(OUT/'REVIEW.json',review);render(result,intervals,mechanisms,review)
    for path,h in manifest['hashes'].items():assert base.sha(path)==h,path
    print('Full historical Joint evaluation and automated review PASS',flush=True)


if __name__=='__main__':main()
