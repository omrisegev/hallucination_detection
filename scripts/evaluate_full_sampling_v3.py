"""Full sampling table, historical controls, paired uncertainty and review."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from collections import Counter, defaultdict
import csv
import html
from html.parser import HTMLParser
import importlib.util
import io
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/localization_full_sampling_v3'
OUT = RUN/'evaluation'
SHORTLIST = ROOT/'results/localization_full_shortlist_v3/evaluation'
HISTORICAL = ROOT/'results/historical_fusion_refit_v3'
sys.path.insert(0, str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.full_sampling_evaluation import contrasts, fold_points, intervals, SELECTORS
from spectral_utils.fusion_token_gap import ANCHORS


def reference():
    spec = importlib.util.spec_from_file_location('full_sampling_metric_reference', ROOT/'scripts/evaluate_localization_full_anchors_v3.py')
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    ref.RUN, ref.OUT = RUN, OUT
    return ref


def append_reference(ref, records, arrays, arms, path, take):
    meta = ref.load(path/'JOINED.json')
    assert meta['records'] == records
    assert ref.sha(path/'JOINED.npz') == meta['arrays_sha256']
    indices = [meta['arms'].index(arm) for arm in take]
    assert not set(arms) & set(take)
    with np.load(path/'JOINED.npz', allow_pickle=False) as z:
        for key in ('offsets', 'labels', 'target'): np.testing.assert_array_equal(arrays[key], z[key])
        for key in ('scores','valid','decision','predictions','peaks','within'):
            arrays[key] = np.concatenate((arrays[key], z[key][:, indices]), axis=1)
    arms.extend(take)


def diagnostics(ref, records, arms, a):
    eligible = np.zeros(len(records), bool); native = np.zeros_like(a['valid'])
    coverage = Counter(); support = defaultdict(list); stability = defaultdict(list); timing = []
    for i, rec in enumerate(records):
        meta = ref.load(RUN/'scores'/(rec['uid']+'.json')); diag = meta['diagnostics']
        assert meta['labels_used'] is False
        eligible[i] = diag['eligible']; timing.append(meta['seconds'])
        for arm, detail in meta['methods'].items():
            j = arms.index(arm); native[i,j] = bool(detail.get('joint_fit_valid', False))
            assert detail['valid'] == bool(a['valid'][i,j])
            assert detail['decision_valid'] == bool(a['decision'][i,j])
            coverage[(rec['cell'],arm,detail.get('source_arm') or detail.get('reason','UNSPECIFIED'),
                      detail['valid'],detail['decision_valid'],bool(detail.get('fallback_to_sample_iu',False)))]+=1
        if not eligible[i]: continue
        with np.load(RUN/'scores'/(rec['uid']+'.npz'), allow_pickle=False) as z:
            for selector in SELECTORS:
                key = selector+'__step_support_fraction'
                if key not in z: continue
                fractions = z[key]; ss, ee = z['step_starts'], z['step_ends']
                if rec['cell'].startswith('pb') and a['target'][i]>=0:
                    j = a['target'][i]; length = ee[j]-ss[j]
                    support[(rec['cell'],selector,'all_error')].append(float(fractions[j]))
                    if length<=32: support[(rec['cell'],selector,'error_le32_tokens')].append(float(fractions[j]))
                elif rec['cell'].startswith('prm'):
                    lo, hi = a['offsets'][i:i+2]
                    support[(rec['cell'],selector,'error_steps')].extend(fractions[a['labels'][lo:hi]==1].tolist())
        for selector, detail in diag.get('selectors',{}).items():
            stability[selector].extend(v['jaccard'] for v in detail.get('block_perturbation',[]) if v.get('jaccard') is not None)
    result = dict(eligible_answers=int(eligible.sum()), total_answers=len(records),
        support_scope='sampling-eligible only; fitting support is not sparse scoring recall',
        support=[dict(cell=k[0],selector=k[1],stratum=k[2],n=len(v),any_support=sum(x>0 for x in v),
                      mean_token_fraction=float(np.mean(v))) for k,v in sorted(support.items())],
        stability={k:dict(observations=len(v),mean_jaccard=float(np.mean(v))) for k,v in stability.items() if v},
        stability_scope='two within-window perturbations; entropy tails/quantiles were not in the original perturbation contract',
        scoring_seconds=dict(sum=float(sum(timing)),median=float(np.median(timing)),p95=float(np.quantile(timing,.95)),
                             caveat='includes all selectors and diagnostics per answer; machine contention varies'),
        coverage=[dict(cell=k[0],method=k[1],source=k[2],score_valid=k[3],decision_valid=k[4],fallback=k[5],answers=v)
                  for k,v in sorted(coverage.items())])
    return eligible,native,result


def independent_review(ref, records, arms, a, folds, metrics):
    """Alternative AUC arithmetic, direct PB counting and exact baseline replay."""
    owner = np.repeat(np.arange(len(records)),np.diff(a['offsets']))
    prm = np.array([r['cell'].startswith('prm') for r in records]); checked = 0
    for j, arm in enumerate(arms):
        mask = prm & a['valid'][:,j]
        for fold in range(5):
            steps = (mask & (folds==fold))[owner]
            expected = ref.roc_auc_score(a['labels'][steps],a['scores'][steps,j])
            np.testing.assert_allclose(metrics[arm]['prm']['fold_aucs'][fold],expected,atol=1e-14,rtol=0)
            checked += 1
        for cell, d in metrics[arm]['pb']['cells'].items():
            rows = [i for i,r in enumerate(records) if r['cell']==cell]
            clean = [i for i in rows if a['target'][i]==-1]
            error = [i for i in rows if a['target'][i]>=0]
            success = lambda ids:sum(bool(a['decision'][i,j]) and int(a['predictions'][i,j])==int(a['target'][i]) for i in ids)
            assert d['clean_accuracy']==success(clean)/len(clean)
            assert d['error_exact_accuracy']==success(error)/len(error)
            checked += 1
    alias_checks = 0
    for core, anchor in ANCHORS.items():
        j,k = arms.index('sample_full__'+core),arms.index(anchor)
        for key in ('scores','valid','decision','predictions','peaks','within'):
            np.testing.assert_allclose(a[key][:,j],a[key][:,k],atol=1e-10,rtol=1e-10,equal_nan=True)
        alias_checks += 1
    return dict(metric_bundles=checked,full_population_aliases=alias_checks)


def render(ref, arms, metrics, uncertainty, diag, historical_arms):
    def number(v,percent=False):
        return 'N/A' if v is None else (f'{100*v:.2f}%' if percent else f'{v:.4f}')
    rows=[]; rendered=[]
    for arm in arms:
        d=metrics[arm];p=d['prm'];b=d['pb']
        row=dict(method=arm,access=d['access'],prm_fold=p['fold_mean_auc'],prm_pooled=p['auroc'],
            prm_within=p['within_answer_auc'],prm_valid=p['answers'],pb_q4=b['macros']['q4'],
            pb_q8=b['macros']['q8'],pb_all=b['macros']['all'],pb_valid=sum(v['valid_decisions'] for v in b['cells'].values()))
        rows.append(row)
        rendered.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (arm,d['access'],number(row['prm_fold']),number(row['prm_pooled']),number(row['prm_within']),
             str(row['prm_valid'])+'/6969',number(row['pb_q4'],True),number(row['pb_q8'],True),
             number(row['pb_all'],True),str(row['pb_valid'])+'/6800'))+'</tr>')
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'METRICS.csv').write_text(buffer.getvalue(),encoding='utf-8')
    comparison=[]
    for d in uncertainty['paired']:
        fields=(d['left']+' minus '+d['right'],d['scope'],d['prm_common_answers'],
                d['intervals']['prm_fold']['ci95'],d['intervals']['within']['ci95'],d['intervals']['pb_q8']['ci95'])
        comparison.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in fields)+'</tr>')
    report='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full localization sampling benchmark</title><style>body{font:16px/1.55 system-ui;max-width:1450px;margin:30px auto;padding:20px;color:#20344b}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #ccd6e0;text-align:left}.scroll{overflow:auto}.note{padding:20px;background:#fff2d5}</style>
<h1>Full localization sampling benchmark</h1><p class="note">13,769 registered model-answer records: all cached PRMBench and ProcessBench Q4/Q8. This is exposed development evidence. The earlier small runs validate code and feasibility; they do not establish improvement. No automatic winner or untouched confirmation is claimed.</p>
<p>Eight fitting-row selectors cross seven fusion cores. All windows are still scored. The seven sample_full entries are aliases of existing references, so the 56 entries contain 49 additions. The table also preserves all 36 full shortlist entries and five historical controls: 97 displayed entries, including aliases. Other historical Joint and dedicated localizer comparisons remain in the project registry.</p>
<p>Full uses every fitting window. Uniform spreads the budget over time. Risk selects high mean entropy. Entropy tails takes low and high entropy; entropy quantiles covers its distribution. Transposed DUFS selects window coordinates, with a permuted control. Window diffusion selects spread-out points on the window graph. Each non-full budget is min(N,max(32,ceil(N/2))). No error labels choose windows. Joint model-fit failure uses the declared sampled-IU fallback; other failures stay visible.</p>
<p>PRMB fold-mean AUROC matches the historical ranking convention; pooled and within-answer AUROC remain visible. These are localization diagnostics, not official PRMScore. PB evaluates the exact first-error step or no error, with failures in the denominator and a macro across cells. The five historical rows fit other training answers and use nested PB label calibration. Their different access prevents attributing the entire gap to fusion.</p>
<div class="scroll"><table><tr><th>Method</th><th>Fit/calibration access</th><th>PRMB fold mean</th><th>PRMB pooled</th><th>Within answer</th><th>PRMB valid</th><th>PB Q4</th><th>PB Q8</th><th>PB all</th><th>PB valid</th></tr>ROWS</table></div>
<h2>Registered paired comparisons</h2><p>Exploratory 95% intervals in 0-1 units; 1,000 joint canonical-source draws. PRMB uses common-valid answers; PB includes all failures in the stated population. Eligible-only and both-native comparisons are conditional diagnostics, not full-population evidence. Intervals condition on fitted predictions and do not account for selecting the best variant or multiple comparisons.</p>
<div class="scroll"><table><tr><th>Comparison</th><th>Population</th><th>Common PRMB</th><th>Fold AUC delta</th><th>Within AUC delta</th><th>PB Q8 delta</th></tr>PAIRS</table></div>
<p><a href="METRICS.csv">Summary CSV</a> · <a href="METRICS.json">All cell metrics</a> · <a href="INTERVALS.json">All intervals</a> · <a href="DIAGNOSTICS.json">Support, coverage and runtime</a> · <a href="REVIEW.json">Review</a> · <a href="../../historical_fusion_refit_v3/REPORT.html">Historical fit protocol and results</a></p></html>'''
    report=report.replace('ROWS',''.join(rendered)).replace('PAIRS',''.join(comparison))
    (OUT/'REPORT.html').write_text(report,encoding='utf-8')
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.rows=0;self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='tr':self.rows+=1
            if tag=='a':self.links.append(dict(attrs)['href'])
    parser=Parser();parser.feed(report)
    assert parser.rows==len(arms)+len(uncertainty['paired'])+2
    return parser


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);ref=reference();ref.bootstrap_preflight()
    manifest=ref.load(RUN/'MANIFEST.json');frozen=ref.load(RUN/'SCORES_FROZEN.json')
    assert frozen['status']=='COMPLETE_FULL_SAMPLING' and frozen['rows']==13769
    assert frozen['manifest_sha256']==ref.sha(RUN/'MANIFEST.json')
    assert ref.load(SHORTLIST/'REVIEW.json')['status']=='PASS'
    assert ref.load(HISTORICAL/'REVIEW_SUPPLEMENT.json')['status']=='PASS'
    for path,h in manifest['hashes'].items():assert ref.sha(path)==h,path
    if (OUT/'JOINED.json').exists():
        joined=ref.load(OUT/'JOINED.json');assert joined['records']==manifest['selected'] and joined['arms']==manifest['arms']
        assert joined['scores_freeze_sha256']==ref.sha(RUN/'SCORES_FROZEN.json')
        assert joined['arrays_sha256']==ref.sha(OUT/'JOINED.npz')
        records,arms=joined['records'],joined['arms']
        with np.load(OUT/'JOINED.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    else:records,arms,_,a=ref.prepare_inputs(manifest,frozen)
    arms=list(arms)
    append_reference(ref,records,a,arms,SHORTLIST,manifest['shortlist_arms'])
    append_reference(ref,records,a,arms,HISTORICAL,manifest['historical_arms'])
    assert len(arms)==97
    foldmap=ref.load(ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json')['outer']
    folds=np.array([foldmap[r['group_id']] for r in records])
    metrics={}
    for j,arm in enumerate(arms):
        d=ref.metric(records,a,j);d['prm'].update(fold_points(ref,records,a,j,folds))
        d['access']='pooled; PB nested labels' if arm in manifest['historical_arms'] else 'answer only; no labels'
        for cell,info in d['pb']['cells'].items():
            mask=np.array([r['cell']==cell for r in records]) & (a['target']>=0)
            peak=a['valid'][:,j] & (a['peaks'][:,j]==a['target'])
            info['raw_peak_accuracy']=float(peak[mask].mean())
            info['correct_peaks_suppressed']=int((mask & peak & a['decision'][:,j] & (a['predictions'][:,j]==-1)).sum())
        metrics[arm]=d
    for source,roster in ((SHORTLIST,manifest['shortlist_arms']),(HISTORICAL,manifest['historical_arms'])):
        old=ref.load(source/'METRICS.json')['metrics']
        for arm in roster:
            for panel in ('q4','q8','all'):assert metrics[arm]['pb']['macros'][panel]==old[arm]['pb']['macros'][panel]
            assert metrics[arm]['prm']['within_answer_auc']==old[arm]['prm']['within_answer_auc']
            key='fold_mean_auc' if source==HISTORICAL else 'auroc'
            assert metrics[arm]['prm'][key]==old[arm]['prm'][key]
    eligible,native,diag=diagnostics(ref,records,arms,a)
    review=independent_review(ref,records,arms,a,folds,metrics)
    ref.save(OUT/'METRICS.json',dict(status='VERIFIED_POINTS_INTERVALS_PENDING',metrics=metrics))
    ref.save(OUT/'DIAGNOSTICS.json',diag)
    def progress(phase,done,total,output):
        ref.state(phase,completed=done,total=total)
        if done%10==0 or done==total:
            ref.save(OUT/'INTERVALS.json',output);print(phase,done,'/',total,flush=True)
    uncertainty=intervals(ref,records,arms,a,folds,eligible,native,progress)
    assert len(uncertainty['paired'])==len(manifest['contrasts']) and manifest['contrasts']==contrasts()
    for arm,d in uncertainty['absolute'].items():
        for name,value in [('prm_fold',metrics[arm]['prm']['fold_mean_auc']),('prm_pooled',metrics[arm]['prm']['auroc']),
                           ('within',metrics[arm]['prm']['within_answer_auc']),('pb_q8',metrics[arm]['pb']['macros']['q8'])]:
            np.testing.assert_allclose(d['point'][name],value,atol=1e-14,rtol=0)
    ref.save(OUT/'INTERVALS.json',uncertainty)
    parser=render(ref,arms,metrics,uncertainty,diag,manifest['historical_arms'])
    for path,h in manifest['hashes'].items():assert ref.sha(path)==h,path
    ref.save(OUT/'METRICS.json',dict(status='COMPLETE_REVIEWED_FULL_SAMPLING',metrics=metrics))
    ref.save(OUT/'REVIEW.json',dict(status='PASS',**review,records=len(records),displayed_entries=len(arms),
        paired_contrasts=len(uncertainty['paired']),report_rows=parser.rows,source_file_hashes=len(frozen['files']),
        report_sha256=ref.sha(OUT/'REPORT.html'),metrics_sha256=ref.sha(OUT/'METRICS.json'),
        interval_sha256=ref.sha(OUT/'INTERVALS.json'),seconds=time.time()-started,
        browser_rendered=False,scope='same-session numerical/provenance review; no external confirmation'))
    for link in parser.links:assert (OUT/link).is_file(),link
    ref.state('COMPLETE_REVIEWED_FULL_SAMPLING',records=len(records),methods=len(arms))
    print('Full sampling evaluation and review PASS',flush=True)


if __name__=='__main__':main()
