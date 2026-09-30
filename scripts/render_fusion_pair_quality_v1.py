"""Report the checked-pair quality experiment without promoting fit coverage."""
from collections import Counter
import hashlib
from html import escape
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_pair_quality_v1';PARENT=ROOT/'results/fusion_replication_v1'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render():
    m,f,e,c,review=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','REVIEW.json')]
    assert review['status']=='PASS'
    for p,h in {**review['hashes'],**review['review_dependencies']}.items():assert sha(p)==h,p
    assert sha(ROOT/'scripts/review_fusion_pair_quality_v1.py')==review['review_script_sha256']
    helper=ROOT/'scripts/render_fusion_replication_v1.py';spec=importlib.util.spec_from_file_location('report_tables',helper)
    view=importlib.util.module_from_spec(spec);spec.loader.exec_module(view)
    table,mdtable,pct=view.table,view.mdtable,view.pct
    previous=load(PARENT/'EVALUATION.json');oldrows={r['uid']:r for r in previous['rows']}
    transitions=Counter();switches=Counter();conditioning={}
    for r in e['rows']:
        old=oldrows[r['uid']]['routing']['routes']['dual'];new=r['routing']['routes']['dual'];transitions[(old,new)]+=1
        if (old=='context_joint')!=(new=='context_joint'):switches[r['cell']]+=1
    for bank in ('moment','context'):
        current=[];common=[];relative_ridge=[]
        for rec in m['selected']:
            meta=load(OUT/'scores'/(rec['uid']+'.json'));new=meta['methods']['pair_'+bank+'__joint0'];old=meta['methods'][bank+'__joint0']
            if new['valid']:
                d=new['inverse'];current.append(d['condition_after']);relative_ridge.append(d['ridge']/d['raw_max_eigenvalue'])
                if old['valid']:common.append([old['inverse']['condition_after'],d['condition_after']])
        conditioning[bank]={'valid':len(current),'condition_at_least_999':sum(x>=999 for x in current),
            'new_condition_quantiles':np.quantile(current,[0,.5,1]).tolist(),'median_ridge_over_max_eigenvalue':float(np.median(relative_ridge)),
            'common_old_new_count':len(common),'common_old_new_median_condition':np.median(common,axis=0).tolist()}
    diagnostic={'status':'POST_EVALUATION_DIAGNOSTIC_NO_NEW_METHOD','route_transitions':[{'old':a,'new':b,'answers':n} for (a,b),n in transitions.items()],
        'bank_switches_by_cell':dict(switches),'bank_switches':sum(switches.values()),'conditioning':conditioning,
        'evaluation_sha256':sha(OUT/'EVALUATION.json'),'scores_sha256':sha(OUT/'SCORES_FROZEN.json')}
    (OUT/'DIAGNOSTICS.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    sections=[];md=['# Checked pair groups: localization results\n\n2026-09-07. Development experiment. Independent review PASS.\n']
    def section(title,text,head=None,rows=None,extra='',collapsed=False):
        content=(table(head,rows) if head else '')+extra
        if collapsed:content='<details><summary>Show complete table</summary>'+content+'</details>'
        sections.append('<section><h2>'+escape(title)+'</h2><p>'+escape(text)+'</p>'+content+'</section>')
        md.append('\n## '+title+'\n\n'+text+'\n'+(mdtable(head,rows) if head else ''))
    section('Decision: retain the existing incumbents',
        'The pair extension increases valid Joint fits, but this fixed localization recipe is not an improvement. The dual-bank '
        'graph route falls from PRMB AUC 0.63350 / PB F1 29.94% to 0.58974 / 19.29% on the same answers. Its exploratory paired '
        'intervals are below zero on both primary endpoints. Single-bank fallback is closer to its old result, but adds no '
        'consistent gain. Do not promote the new pair-routing policy. Keep the mathematical repair and the negative result as '
        'part of developing our fusion, not as evidence that the entire Joint/graph family is finished.')
    section('Our fusion method is still the center',
        'The architecture remains one answer -> N windows x P features -> IU-PCR / Joint L-SML -> fused trajectory -> official '
        'step and no-error decisions. This experiment changes the feature-group minimum and its safe covariance/Jacobian '
        'treatment. It does not replace fusion with an auxiliary detector. More admissible partitions and more valid fits '
        'are useful engineering properties; they are not a correctness objective.',
        extra='<div class="flow"><div><strong>Before: minimum 3</strong><br>Dual route: 78 moment Joint<br>29 context Joint<br>3 moment IU</div><div><strong>After: checked pairs</strong><br>Dual route: 106 moment Joint<br>4 context Joint<br>0 IU fallbacks</div><div><strong>Measured consequence</strong><br>More moment fits become usable.<br>Bank selection changes.<br>Localization does not improve.</div></div>')
    section('What we can show our advisors',
        'We have adapted our existing fusion architecture to observations inside one answer, tested feature banks and '
        'graph penalties with matched controls, and repaired two mathematical edge cases in admitting feature pairs. '
        'The quality test also identifies a weakness in using fit eligibility to select the feature bank. These are '
        'concrete steps in developing IU-PCR and Joint L-SML. They do not yet establish a consistently better localizer. '
        'IMM, LOCA, Diverging Flows, KalmanNet and token sampling remain possible supporting components. Each needs '
        'the same fusion with and without it, plus simple aggregation with the component, to show what fusion contributes.')
    section('Frozen contract and what was compared',
        'All 110 previously evaluated development answers were retained: 24 PRMB, 16 GSM8K, 24 MATH, 22 OlympiadBench and '
        '24 Omni-MATH. These are one-pass teacher-forced gray-box traces of fixed official answers. Fitting stays inside '
        'each answer, with the declared negative-entropy anchor. Width eight, 27 moment/context coordinates, four chronological '
        'blocks, K={3,4,6,8}, held admissibility 0.95, five starts, 5000 sweeps, inverse condition 1000 and all seeds are unchanged. '
        'Graph lambda is 0 or 0.1, with the same node-permutation control. Predictions were frozen before this evaluator read '
        'labels; existing data exposure is still disclosed. No untouched confirmation is claimed.')
    section('Read the new method names',
        'The 19 original arms are copied exactly as anchors. Fourteen new arms bring the total to 33. pair_moment and pair_context '
        'are pure checked-pair Joint on the respective bank. joint0 means no graph penalty; graph010 means lambda 0.1; '
        'graph_perm uses the permuted graph. pair_single uses moment Joint then moment IU on invalid fits. pair_dual tries '
        'context Joint between them. pair_dual__iu/equal use the same new bank route, but apply the unchanged IU/equal map on '
        'that bank. A no-error decision or failed readout never triggers another route. The fixed-IU gate remains diagnostic.')
    focus=['moment__iu','dual__iu','dual__graph010','single__joint0','pair_single__joint0','pair_dual__joint0','pair_dual__graph010','pair_dual__iu','pair_dual__equal','context__equal']
    def metric_rows(arms):
        return [[arm,str(review['coverage'][arm])+'/110',str(e['metrics'][arm]['prm']['answers'])+'/24',
            f"{e['metrics'][arm]['prm']['auroc']:.5f}",f"{e['metrics'][arm]['prm']['within_answer_auc']:.5f}",
            pct(e['metrics'][arm]['pb']['macro_f1']),pct(e['metrics'][arm]['pb_common_iu_gate']['macro_f1'])] for arm in arms]
    headers=['Arm','Fit coverage','Valid PRMB','Pooled PRMB AUC','Within-answer AUC','PB native F1','PB fixed-IU gate: diagnostic']
    section('Matched headline rows',
        'Higher is better. Every row here covers all 110 answers. PRMB pooled AUC ranks steps across answers; within-answer '
        'AUC uses the 16 answers with both step labels. PB is the mean of four subset-level harmonic means of clean and '
        'exact-first-error accuracy. Invalid decisions remain failures on all 86 PB answers.',headers,metric_rows(focus))
    section('All 33 arms, including pure failures',
        'Pure moment Joint coverage rises 78 -> 106; context 102 -> 108. Pure PRMB scores use their own valid answers, so '
        'their full-row AUCs are not automatically matched comparisons. Native and fixed-IU readouts are separate endpoints. '
        'All valid fits in this run have valid native decisions. The two proposed routes retain all answers.',headers,metric_rows(m['arms']),collapsed=True)
    keypairs=['pair_dual__graph010 minus dual__graph010','pair_dual__joint0 minus dual__joint0',
        'pair_single__joint0 minus single__joint0','pair_dual__iu minus dual__iu',
        'pair_moment__joint0 minus moment__joint0','pair_context__joint0 minus context__joint0',
        'pair_dual__graph010 minus pair_dual__joint0','pair_dual__graph010 minus pair_dual__graph_perm',
        'pair_dual__graph010 minus pair_dual__iu']
    def paired_rows(keys):
        result=[]
        for key in keys:
            p=c['pairs'][key];u=p['uncertainty']
            result.append([key,p['left_prm']['answers'],view.f(p['left_prm']['auroc']-p['right_prm']['auroc'],signed=True),
                view.interval(u['prm_common_valid_ci95']),view.f(p['left_pb']['macro_f1']-p['right_pb']['macro_f1'],100,True),
                view.interval(u['pb_all_population_ci95'],100),u['pb_all_population_valid_draws']])
        return result
    ch=['New minus reference','Common valid PRMB','Delta AUC','95% AUC CI','Delta PB (pp)','95% PB CI (pp)','Defined PB draws']
    section('Paired evidence, with the same IDs',
        'All 63 comparisons were frozen before new quality evaluation. Intervals use 1000 source-group bootstrap draws, '
        'stratified by cell, and are exploratory and unadjusted across this roster. The new dual graph versus its old version '
        'has AUC delta -0.04376, interval [-0.08151,-0.01527], and PB delta -10.65 points, interval [-21.04,-1.51]. '
        'Within-answer and fixed-IU diagnostic intervals for that comparison include zero. The diagnostics do not replace '
        'the primary native endpoints.',ch,paired_rows(keypairs))
    section('Why coverage alone gave the wrong expectation',
        'The dual route changes banks on 31 answers: 28 move from context to moment, and three move from moment to context. '
        'Three further answers move from moment IU fallback to moment Joint, with the bank unchanged. The IU routing control '
        'isolates bank choice: its IU maps never change, yet its PRMB/PB falls from 0.63797/30.16% to 0.59222/26.10%. '
        'That is evidence that fit validity is insufficient as a rule for choosing the more useful representation. It does '
        'not prove that all of the Joint regression is caused by routing, because its fitted weights can also change.',
        ['Old dual route','New dual route','Answers'],[[a,b,n] for (a,b),n in transitions.items()])
    section('Do not misread the pure context comparison',
        'Context Joint0\'s unmatched full AUC falls from 0.68397 on 21 answers to 0.64353 on 23. On the 20 common valid '
        'answers, however, its paired AUC change is +0.00109, and its mean within-answer change is exactly zero. Its PB '
        'full-population change is +3.92 points, with interval [0.00,+11.50], not a clear two-task win. Moment Joint0 has '
        'common-ID AUC delta -0.00196 on 16 answers and PB delta +0.66 points, both intervals including zero. The paired '
        'tables distinguish representation/weight changes from who could be scored.')
    pbdata=[]
    pb=[r for r in e['rows'] if r['cell'].startswith('pb')]
    for arm in focus:
        clean=sum(r['decision_valid'][arm] and r['predictions'][arm]==-1 for r in pb if r['target']==-1)
        error=sum(r['decision_valid'][arm] and r['predictions'][arm]==r['target'] for r in pb if r['target']!=-1)
        peak=sum(r['valid'][arm] and r['peaks'][arm]==r['target'] for r in pb if r['target']!=-1)
        pbdata.append([arm,str(clean)+'/33',str(error)+'/53',str(peak)+'/53']+
            [pct(e['metrics'][arm]['pb']['cells'][cell]['f1']) for cell in ('pb_gsm8k_q8','pb_math_q8','pb_olympiadbench_q8','pb_omnimath_q8')])
    section('ProcessBench: what changed in actual decisions',
        'The cohort contains 33 clean and 53 erroneous answers. The old dual graph gets 17 clean and 13 exact errors right; '
        'the new dual graph gets 10 and 11. Its raw peak hits 14 erroneous answers, versus 18 before. Both the no-error '
        'decision and localization need work. Subset-balanced F1 and total successes are different summaries; both are visible.',
        ['Arm','Clean hits','Exact-error hits','Raw peak hits','GSM F1','MATH F1','Olympiad F1','Omni F1'],pbdata)
    history=[]
    for arm in previous['metrics']:
        old=e['older_58_answer_metrics'][arm];now=e['metrics'][arm]
        history.append([arm,old['prm']['answers'],f"{old['prm']['auroc']:.5f}",pct(old['pb']['macro_f1']),
                        now['prm']['answers'],f"{now['prm']['auroc']:.5f}",pct(now['pb']['macro_f1'])])
    section('Historical context remains visible',
        'All 19 prior 110-answer endpoint bundles replay exactly. The older 58-answer cohort is shown separately below. '
        'Changes across these two question cohorts are not algorithmic gains. New pair arms have no older-58 result in this '
        'experiment. Claude\'s multi-answer scores remain a separate fitting protocol requiring corrected-fold refits; this '
        'answer-only comparison does not repair those scores.',
        ['Frozen anchor','Old valid PRMB','Old-58 AUC','Old-58 PB F1','Current valid PRMB','Current-110 AUC','Current-110 PB F1'],history,collapsed=True)
    section('All 63 registered comparisons',
        'The JSON includes all four intervals for each comparison, including within-answer AUC and the fixed-IU gate '
        'diagnostic. Undefined bootstrap draws are counted, not silently treated as zero.',ch,paired_rows(list(c['pairs'])),collapsed=True)
    section('Independent review and runtime',
        'Review passes for 110 direct label/source-group joins, 4710 exact parent arrays, 2090 parent metadata records, '
        '220 audit groupings, 219 fitted covariance/factor replays, 67 independent pair-product Jacobians, 642 graph/native '
        'inverse projections, step maps and GMM decisions, 880 route inheritances, all 33 metric bundles and all 63 paired '
        'point bundles. Six explicit 1000-draw bootstraps match all four intervals and defined counts. Same-answer gates '
        'replay in 176 bank fits; representative newly computed gate recipes are refitted. The Joint optimizer, DUFS and '
        'graph-builder kernels were reused; the Laplacian, inverse projection, decisions and endpoints were reconstructed '
        'independently. The two pre-freeze tests supplement the eleven preceding pair/audit tests.',
        ['Measurement','Value'],[['Scoring wall time, three workers',f"{f['seconds_this_invocation']:.2f} s"],
            ['All 63 contrasts',f"{c['seconds_this_invocation']:.2f} s"],['Independent review',f"{review['seconds']:.2f} s"],
            ['Largest reconstructed risk difference',review['max_reconstructed_risk_difference']],['Browser visual inspection','Not run; structural and link checks only']])
    section('Next short question: the native inverse',
        'Do not replace the incumbent routes with the pair-eligibility route. The next bounded experiment should keep the '
        'original Joint fits, feature groups and bank routing fixed and test stronger native inverse conditioning. A post-evaluation '
        'unlabeled diagnostic shows that 104/106 valid pair-moment maps and 97/108 pair-context maps sit at the condition-1000 '
        'cap; the median old/new condition on their common fits is also 1000. This motivates a conditioning test, not a claim '
        'that conditioning caused the observed errors. Start with the lambda-zero native map to isolate that change, retaining '
        'the existing graph and IU/equal anchors; add graph-dose interaction only if justified. Full comparator coverage, '
        'corrected-fold multi-answer replay, supporting fusion ideas, untouched confirmation and historical24 transfer remain open.')
    links=[('MANIFEST.json','Frozen protocol/source registry'),('EVALUATION.json','All labels, scores and metrics'),('CONTRASTS.json','All paired intervals'),
        ('REVIEW.json','Independent review'),('DIAGNOSTICS.json','Post-evaluation routing/conditioning diagnostics'),('TESTS.txt','Observed two-test PASS record'),
        ('../joint_pair_identifiability_audit_v1/REPORT.html','Pair covariance/Jacobian explanation'),('../fusion_replication_v1/REPORT.html','Prior 110-answer results'),
        ('../../spectral_utils/fusion_pair_quality.py','Checked pair-fusion scoring and routing'),('../../docs/experiments/JOINT_PAIR_LOCALIZATION_QUALITY_V1.md','Frozen quality protocol'),
        ('../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html','Visual guide to our fusion method')]
    sections.append('<section><h2>Evidence and code</h2><ul>'+''.join('<li><a href="'+p+'">'+escape(t)+'</a></li>' for p,t in links)+'</ul></section>')
    md.append('\n## Evidence and code\n\n'+'\n'.join('- ['+t+']('+p+')' for p,t in links))
    css='''*{box-sizing:border-box}body{margin:0;background:#f4f6f3;color:#193d45;font:16px/1.65 system-ui,Segoe UI,sans-serif}header{background:#153f48;color:white;padding:45px max(22px,calc((100vw - 1160px)/2))}h1{font-size:clamp(32px,5vw,50px);line-height:1.15}main{max-width:1204px;margin:auto;padding:20px 22px 50px}section{margin:22px 0;padding:24px;background:white;border:1px solid #dce5df;border-radius:13px}h2{font-size:26px;line-height:1.25}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.flow>div{padding:18px;border-top:4px solid #197f70;background:#eaf3ed;border-radius:8px}.flow>div:last-child{background:#fff1df;border-color:#b87719}.table{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce5df;vertical-align:top}th{background:#eaf3ed}td:first-child{font-weight:650;white-space:nowrap}summary{cursor:pointer;font-weight:650;padding:10px 0}a{color:#075c90}a:focus-visible,summary:focus-visible{outline:3px solid #b57715}@media(max-width:720px){.flow{grid-template-columns:1fr}main{padding:12px}section{padding:16px}}@media print{body{background:white;font-size:11pt}header{padding:15px;background:white;color:#193d45}section{border:0;padding:10px}h2{break-after:avoid}.table{overflow:visible}table{font-size:8pt}th,td{padding:5px;white-space:normal!important}}'''
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Checked pair Joint: localization quality</title><style>'+css+'</style></head><body><header><p>07 September 2026 · Reviewed development experiment</p><h1>More Joint fits did not<br>give better localization.</h1><p>33 methods on the same 110 answers. The negative result tells us what to preserve and what to test next.</p></header><main>'+''.join(sections)+'</main></body></html>\n'
    (OUT/'REPORT.html').write_text(html,encoding='utf-8',newline='\n');(OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8',newline='\n')
    class Check(HTMLParser):
        def __init__(self):super().__init__();self.stack=[];self.links=[]
        def handle_starttag(self,t,a):
            if t not in ('meta','br','link','hr','input','img'):self.stack.append(t)
            if t=='a':self.links.append(dict(a)['href'])
        def handle_endtag(self,t):assert self.stack.pop()==t,t
    check=Check();check.feed(html);check.close();assert not check.stack
    for p in check.links:assert (OUT/p).resolve().exists(),p
    prov={'status':'PASS','source_hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','REVIEW.json','DIAGNOSTICS.json')},
        'report_hashes':{str(OUT/n):sha(OUT/n) for n in ('REPORT.html','REPORT.md')},'renderer_sha256':sha(__file__),'helper_sha256':sha(helper)}
    (OUT/'REPORT_PROVENANCE.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
    (OUT/'ARTIFACT_VALIDATION.json').write_text(json.dumps({'status':'PASS','html_structure':'PASS','local_links':len(check.links),
        'all_33_arms_and_63_comparisons':True,'browser_visual_check':'NOT_RUN','provenance_sha256':sha(OUT/'REPORT_PROVENANCE.json')},indent=2),encoding='utf-8')
    print('Reports rendered; 33 arms, 63 comparisons,',len(check.links),'verified local links.')


if __name__=='__main__':render()
