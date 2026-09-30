"""Render the reviewed native-conditioning experiment and preserve all controls."""
from html import escape
from html.parser import HTMLParser
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_native_conditioning_v1'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render():
    m,f,e,c,r=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','REVIEW.json')]
    assert r['status']=='PASS'
    for p,h in {**r['hashes'],**r['review_dependencies']}.items():assert sha(p)==h,p
    assert sha(ROOT/'scripts/review_fusion_native_conditioning_v1.py')==r['review_script_sha256']
    helper=ROOT/'scripts/render_fusion_replication_v1.py';spec=importlib.util.spec_from_file_location('native_condition_report_tables',helper)
    view=importlib.util.module_from_spec(spec);spec.loader.exec_module(view)
    table,mdtable,pct=view.table,view.mdtable,view.pct
    sections=[];md=['# Original Joint inverse conditioning\n\n2026-09-07. Development evidence. Independent review PASS.\n']
    def section(title,text,head=None,rows=None,extra='',collapsed=False):
        content=(table(head,rows) if head else '')+extra
        if collapsed:content='<details><summary>Show complete table</summary>'+content+'</details>'
        sections.append('<section><h2>'+escape(title)+'</h2><p>'+escape(text)+'</p>'+content+'</section>')
        md.append('\n## '+title+'\n\n'+text+'\n'+(mdtable(head,rows) if head else ''))
    section('Decision: keep developing Joint, retain the references',
        'Stronger native conditioning modestly improves Joint over its original condition-1000 map. Dual Joint at '
        'condition 30 gives PRMB AUC 0.63639 and PB F1 28.43%, versus original Joint0 0.62884 and 25.33%. However, '
        'both paired improvement intervals include zero, and dual IU remains higher at 0.63797 and 30.16%. This '
        'is a useful direction for the same fusion method, not a demonstrated two-task winner or an optimal setting.')
    section('What changed inside our fusion',
        'The original Joint model supplies a covariance C and global loading v. Its native weights solve '
        '(C + alpha I) w = v. Diagonal ridge alpha reduces the influence of directions with very small variance. '
        'A smaller allowed condition number generally requires more ridge. We tested 1000, 300, 100 and 30, with '
        '1000 the unchanged reference. This parameter is different from graph lambda; all new heads here have lambda zero.',
        extra='<div class="flow"><div><strong>Hold fixed</strong><br>One answer and its windows<br>Feature bank, signs and groups<br>Original Joint fit and bank route</div><div><strong>Change one operation</strong><br>Native inverse condition cap<br>1000 → 300 → 100 → 30<br>Same C and v; different diagonal ridge</div><div><strong>Measure the result</strong><br>Fused trajectory and official steps<br>Same GMM gate recipe<br>First error or no error</div></div>')
    section('A controlled extension of the original method',
        'All 110 answers are the same previously evaluated development examples: 24 PRMB, and 16 GSM8K, 24 MATH, '
        '22 OlympiadBench and 24 Omni-MATH. These are fixed official answers scored with one teacher-forced gray-box '
        'model pass. Normalization, groups, Joint weights and gates are fitted inside each answer. The declared '
        'negative-entropy anchor remains. The original fit did not persist C and v, so we reproduced each valid fit '
        'on its original matrix and partition, requiring the condition-1000 weights, scores and decisions to replay '
        'before new scoring. We now save C, v and group loadings for reuse. All 180 valid fits passed; the 40 invalid '
        'bank fits remain invalid. No new pair admission or bank switching took place.')
    headers=['Arm','Fit coverage','Valid PRMB','Pooled AUC','Within-answer AUC','PB native F1','PB fixed-IU: diagnostic']
    def metric_rows(arms):
        return [[a,str(r['coverage'][a])+'/110',str(e['metrics'][a]['prm']['answers'])+'/24',
            view.f(e['metrics'][a]['prm']['auroc']),view.f(e['metrics'][a]['prm']['within_answer_auc']),
            pct(e['metrics'][a]['pb']['macro_f1']),pct(e['metrics'][a]['pb_common_iu_gate']['macro_f1'])] for a in arms]
    focus=['moment__iu','dual__equal','dual__iu','dual__joint0','dual__cond300','dual__cond100','dual__cond30',
           'dual__graph010','single__joint0','single__cond30','context__equal']
    section('Full-coverage comparisons on the same answers',
        'All rows in this table cover the entire cohort. Dual means the ORIGINAL route: moment Joint if valid, '
        'otherwise context Joint if valid, otherwise moment IU. The matched dual IU and equal controls use the same '
        'bank route. Pure moment/context failures appear in the complete table below. The within-answer endpoint '
        'uses 16 mixed-label PRMB answers here; pooled AUC includes cross-answer comparisons.',headers,metric_rows(focus))
    def plot(metric,title,low,high,scale):
        x=[66,176,286,396];arms=['dual__joint0','dual__cond300','dual__cond100','dual__cond30']
        def val(a):return e['metrics'][a]['prm']['auroc'] if metric=='prm' else e['metrics'][a]['pb']['macro_f1']
        def yy(value):return 190-(value-low)/(high-low)*145
        svg='<svg viewBox="0 0 510 265" role="img" aria-label="'+escape(title)+'"><title>'+escape(title)+'</title>'
        for tick in np.linspace(low,high,5):
            y=yy(tick);svg+=f'<line x1="55" y1="{y}" x2="420" y2="{y}" stroke="#d6e3df"/><text x="8" y="{y+4}" font-size="12">{tick*scale:.2f}</text>'
        for a,label,color in [('dual__iu','IU','#8052a0'),('dual__graph010','Graph 0.1','#bc7419'),('dual__equal','Equal','#72817b')]:
            y=yy(val(a));svg+=f'<line x1="55" y1="{y}" x2="420" y2="{y}" stroke="{color}" stroke-dasharray="5 4"/>'
        points=' '.join(f'{xx},{yy(val(a))}' for xx,a in zip(x,arms))
        svg+='<polyline points="'+points+'" fill="none" stroke="#067d78" stroke-width="3"/>'
        for xx,a,label in zip(x,arms,['1000','300','100','30']):
            y=yy(val(a));svg+=f'<circle cx="{xx}" cy="{y}" r="5" fill="#067d78"/><text x="{xx}" y="{y-10}" text-anchor="middle" font-size="12">{val(a)*scale:.3f}</text><text x="{xx}" y="216" text-anchor="middle" font-size="13">{label}</text>'
        return svg+'<text x="240" y="244" text-anchor="middle" font-size="12">Condition cap: stronger ridge to the right</text></svg><p style="font-size:13px"><span style="color:#067d78">● Joint, no graph</span> &nbsp; <span style="color:#8052a0">-- IU</span> &nbsp; <span style="color:#bc7419">-- Graph 0.1</span> &nbsp; <span style="color:#72817b">-- Equal</span></p>'
    section('The measured dose response',
        'The strongest of the three new tested settings has the highest dual Joint point estimates. It is not a '
        'proven optimum. Plots use narrow vertical ranges to show small changes, and equally spaced tested settings '
        'rather than a numeric horizontal scale. These are point estimates; paired uncertainty is shown next.',
        extra='<div class="plots"><div><h3>Dual Joint: PRMB AUC</h3>'+plot('prm','Dual Joint PRMB AUC at four condition caps',.62,.645,1)+'</div><div><h3>Dual Joint: PB F1 (%)</h3>'+plot('pb','Dual Joint PB F1 at four condition caps',.24,.32,100)+'</div></div>')
    ch=['New minus reference','Common PRMB','Delta AUC','95% AUC CI','Delta PB (pp)','95% PB CI (pp)','Defined PB draws']
    def pair_rows(keys):
        result=[]
        for k in keys:
            p=c['pairs'][k];u=p['uncertainty'];result.append([k,p['left_prm']['answers'],
                view.f(p['left_prm']['auroc']-p['right_prm']['auroc'],signed=True),view.interval(u['prm_common_valid_ci95']),
                view.f(p['left_pb']['macro_f1']-p['right_pb']['macro_f1'],100,True),view.interval(u['pb_all_population_ci95'],100),u['pb_all_population_valid_draws']])
        return result
    keypairs=['dual__cond30 minus dual__joint0','dual__cond30 minus dual__iu','dual__cond30 minus dual__equal',
        'dual__cond30 minus dual__graph010','dual__cond30 minus dual__graph_perm','single__cond30 minus single__joint0',
        'single__cond30 minus moment__iu','context__cond30 minus context__equal','moment__cond30 minus moment__joint0']
    section('Paired evidence limits the conclusion',
        'Dual condition30 minus original Joint0 is +0.00755 AUC, interval [-0.00062,+0.01777], and +3.10 PB points, '
        'interval [-2.58,+9.74]. Both include zero. Against dual IU and matched equal, both primary intervals also '
        'include zero. These are 1000-draw source-group intervals, exploratory and unadjusted across 69 registered '
        'comparisons. A positive within-answer interval against the permuted-graph control is a secondary finding, '
        'not a two-task win against IU or the real graph.',ch,pair_rows(keypairs))
    section('Keep the pure-bank coverage caveat',
        'Context condition30 has full-row PRMB AUC 0.68799 on 21 valid answers, versus context equal 0.66971 on '
        'all 24. That ordering is misleading as a method comparison: on the SAME 21 answers, context equal scores '
        '0.69658. The paired Joint-minus-equal difference is -0.00859, interval [-0.01748,-0.00100]. Stronger '
        'conditioning has not removed this weakness. Pure moment retains 78/110 fits and context 102/110. '
        'The original single route stays 78 Joint/32 IU, and dual stays 78 moment Joint/29 context Joint/3 IU.')
    pb=[x for x in e['rows'] if x['cell'].startswith('pb')];pbarms=['moment__iu','dual__iu','single__joint0','single__cond30','dual__joint0','dual__cond30','dual__graph010'];pbtable=[]
    for a in pbarms:
        clean=sum(x['decision_valid'][a] and x['predictions'][a]==-1 for x in pb if x['target']==-1)
        exact=sum(x['decision_valid'][a] and x['predictions'][a]==x['target'] for x in pb if x['target']!=-1)
        peak=sum(x['valid'][a] and x['peaks'][a]==x['target'] for x in pb if x['target']!=-1)
        pbtable.append([a,f'{clean}/33',f'{exact}/53',f'{peak}/53']+[pct(v['f1']) for v in e['metrics'][a]['pb']['cells'].values()])
    section('ProcessBench: gating and location still differ',
        'Dual condition30 gets 16 clean answers and 12 exact first errors right, versus original Joint0 14 and 11. '
        'The total raw-peak hit count stays 17/53; identical totals do not mean identical answer-level hits. '
        'With the fixed-original-IU gate, PB falls from 29.25% to 28.73%. Within-answer PRMB AUC rises '
        '0.63034 -> 0.67117, but its paired improvement interval includes zero. The native PB increase alone '
        'does not demonstrate better peak localization.',
        ['Arm','Clean hits','Exact-error hits','Raw peak hits','GSM F1','MATH F1','Olympiad F1','Omni F1'],pbtable)
    section('All 45 arms',
        'All 33 preceding arms replay exactly, including the negative checked-pair controls. Twelve new heads '
        'change only native conditioning. Invalid fits and decisions remain visible. Pure-bank AUCs use their '
        'own valid IDs; use paired common-ID contrasts to compare them.',headers,metric_rows(m['arms']),collapsed=True)
    history=[]
    for a,old in e['older_58_answer_metrics'].items():
        current=e['metrics'][a];history.append([a,old['prm']['answers'],view.f(old['prm']['auroc']),pct(old['pb']['macro_f1']),
            current['prm']['answers'],view.f(current['prm']['auroc']),pct(current['pb']['macro_f1'])])
    section('Historical context, with separate populations',
        'The older 58-answer results are retained below. Changing the evaluated cohort is not an algorithmic gain. '
        'The three new conditioning doses have no old-58 result in this experiment. Claude\'s multi-answer results '
        'use another fitting protocol and still require corrected-fold refits. This answer-only experiment does '
        'not repair those historical fits.',
        ['Anchor','Old valid PRMB','Old-58 AUC','Old-58 PB','Current valid PRMB','Current-110 AUC','Current-110 PB'],history,collapsed=True)
    section('All 69 registered paired comparisons',
        'Every comparison is retained. CONTRASTS.json additionally includes within-answer and fixed-IU-gate '
        'intervals. Undefined bootstrap draws are counted instead of being imputed as zero.',ch,pair_rows(list(c['pairs'])),collapsed=True)
    section('Code and findings review',
        'Three pre-freeze tests pass: analytical ridge behavior, original score/route replay, and preservation of '
        'failed new heads/readouts. Independent review passes for 110 label/group joins, 8411 exact parent arrays, '
        '3630 parent metadata records, 180 covariance constructions, 720 inverse/step/GMM reconstructions, 660 '
        'fixed-route inheritances, 45 metric bundles and 69 paired point bundles. Ten representative original '
        'fits replay. Six explicit 1000-draw bootstraps match all four intervals and defined counts. The optimizer '
        'and sklearn GMM kernels are reused. The initial representative refit differed at about 1e-11 when using '
        'an independently ordered normalization; exact source-recipe replay now preserves the reduction order. '
        'Independent input/algebra checks remain, and no frozen scientific source, score or tolerance was changed.',
        ['Measurement','Value'],[['Scoring, three CPU workers',f"{f['seconds_this_invocation']:.2f} s"],
        ['69 contrasts',f"{c['seconds_this_invocation']:.2f} s"],['Final review',f"{r['seconds']:.2f} s"],
        ['Maximum reconstructed risk difference',r['max_reconstructed_risk_difference']],['Browser visual inspection','Not run; structure and local links checked']])
    section('Next bounded question: does graph structure still help?',
        'The conditioning effect is large enough to change Joint scores and some decisions, but does not establish '
        'superiority. Keep all three frozen caps, the original fits and routes, and test their interaction with the '
        'existing graph lambda 0.1 and its permutation control. Reuse the saved original covariances and same-answer '
        'DUFS gates. This separates graph structure from generic inverse regularization without widening K or '
        'searching new graph doses. Do not choose a best cap from these labels or promote a two-task winner. '
        'The broader IU/Joint program, supporting ideas, corrected-fold refits, full comparator coverage, '
        'untouched confirmation and historical24 transfer remain open.')
    links=[('MANIFEST.json','Frozen methods and comparisons'),('EVALUATION.json','All metrics and decisions'),('CONTRASTS.json','All paired intervals'),
        ('REVIEW.json','Independent review'),('TESTS.txt','Captured test output'),('TEST_EXECUTION.json','Test command and source hashes'),
        ('../fusion_pair_quality_v1/REPORT.html','Previous checked-pair result'),('../fusion_replication_v1/REPORT.html','Original 110-answer fusion references'),
        ('../../spectral_utils/fusion_native_conditioning.py','Conditioning and fixed routing code'),('../../docs/experiments/FUSION_NATIVE_CONDITIONING_V1.md','Frozen protocol'),
        ('../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html','Visual guide to our fusion')]
    sections.append('<section><h2>Evidence and code</h2><ul>'+''.join('<li><a href="'+p+'">'+escape(t)+'</a></li>' for p,t in links)+'</ul></section>')
    md.append('\n## Evidence and code\n\n'+'\n'.join('- ['+t+']('+p+')' for p,t in links))
    css='''*{box-sizing:border-box}body{margin:0;background:#f4f6f3;color:#193d45;font:16px/1.65 system-ui,Segoe UI,sans-serif}header{background:#153f48;color:white;padding:45px max(22px,calc((100vw - 1160px)/2))}h1{font-size:clamp(32px,5vw,50px);line-height:1.15}main{max-width:1204px;margin:auto;padding:20px 22px 50px}section{margin:22px 0;padding:24px;background:white;border:1px solid #dce5df;border-radius:13px}h2{font-size:26px;line-height:1.25}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.flow>div{padding:18px;border-top:4px solid #197f70;background:#eaf3ed;border-radius:8px}.flow>div:nth-child(2){background:#fff1df;border-color:#b87719}.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}.plots svg{display:block;width:100%;height:auto}.table{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce5df;vertical-align:top}th{background:#eaf3ed}td:first-child{font-weight:650;white-space:nowrap}summary{cursor:pointer;font-weight:650;padding:10px 0}a{color:#075c90}a:focus-visible,summary:focus-visible{outline:3px solid #b57715}@media(max-width:720px){.flow,.plots{grid-template-columns:1fr}main{padding:12px}section{padding:16px}}@media print{body{background:white;font-size:11pt}header{padding:15px;background:white;color:#193d45}section{border:0;padding:10px}h2{break-after:avoid}.table{overflow:visible}table{font-size:8pt}th,td{padding:5px;white-space:normal!important}}'''
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Original Joint: native inverse conditioning</title><style>'+css+'</style></head><body><header><p>07 September 2026 · Reviewed development experiment</p><h1>Stabilizing the weights<br>inside our Joint fusion.</h1><p>The original fits and routes stay fixed. Stronger ridge helps some points, but a two-task advantage remains unproven.</p></header><main>'+''.join(sections)+'</main></body></html>\n'
    (OUT/'REPORT.html').write_text(html,encoding='utf-8',newline='\n');(OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8',newline='\n')
    class Check(HTMLParser):
        def __init__(self):super().__init__();self.stack=[];self.links=[];self.svg_count=0
        def handle_starttag(self,t,a):
            if t not in ('meta','br','link','hr','input','img'):self.stack.append(t)
            if t=='a':self.links.append(dict(a)['href'])
            if t=='svg':self.svg_count+=1
        def handle_endtag(self,t):assert self.stack.pop()==t,t
    check=Check();check.feed(html);check.close();assert not check.stack
    for p in check.links:assert (OUT/p).resolve().is_file(),p
    assert check.svg_count==2 and all(a in html for a in m['arms']) and all(k in html for k in c['pairs'])
    prov={'status':'PASS','source_hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','REVIEW.json')},
        'report_hashes':{str(OUT/n):sha(OUT/n) for n in ('REPORT.html','REPORT.md')},'renderer_sha256':sha(__file__),'helper_sha256':sha(helper)}
    (OUT/'REPORT_PROVENANCE.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
    (OUT/'ARTIFACT_VALIDATION.json').write_text(json.dumps({'status':'PASS','html_structure':'PASS','local_links':len(check.links),'svg_plots':check.svg_count,
        'all_45_arms_and_69_comparisons':True,'browser_visual_check':'NOT_RUN','provenance_sha256':sha(OUT/'REPORT_PROVENANCE.json')},indent=2),encoding='utf-8')
    print('Reports rendered: 45 arms, 69 comparisons, two SVG plots and',len(check.links),'verified local links.')


if __name__=='__main__':render()
