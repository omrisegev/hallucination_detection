"""Report graph/conditioning interaction with matched simple controls."""
from html import escape
from html.parser import HTMLParser
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_graph_conditioning_v1'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render():
    m,f,e,c,r=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','REVIEW.json')]
    assert r['status']=='PASS'
    for p,h in {**r['hashes'],**r['review_dependencies']}.items():assert sha(p)==h,p
    assert sha(ROOT/'scripts/review_fusion_graph_conditioning_v1.py')==r['review_script_sha256']
    helper=ROOT/'scripts/render_fusion_replication_v1.py';spec=importlib.util.spec_from_file_location('graph_condition_report_tables',helper)
    view=importlib.util.module_from_spec(spec);spec.loader.exec_module(view);table,mdtable,pct=view.table,view.mdtable,view.pct
    overlap=load(OUT/'ERROR_OVERLAP.json');assert overlap['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    pb=[x for x in e['rows'] if x['cell'].startswith('pb')];err=[x for x in pb if x['target']!=-1]
    # Independently validate the post-evaluation overlap artifact as Boolean sets.
    for arm,info in overlap['comparisons'].items():
        for category in ('clean','erroneous'):
            subset=[x for x in pb if (x['target']==-1)==(category=='clean')]
            a=np.array([x['decision_valid']['dual__iu'] and x['predictions']['dual__iu']==x['target'] for x in subset])
            b=np.array([x['decision_valid'][arm] and x['predictions'][arm]==x['target'] for x in subset])
            assert info[category]['answers']==len(subset)
            actual=[int((a&b).sum()),int((a&~b).sum()),int((~a&b).sum()),int((~a&~b).sum())]
            assert actual==[info[category]['native_decisions'][k] for k in ('both_correct','iu_only_correct','joint_only_correct','neither_correct')]
        aa=np.array([x['valid']['dual__iu'] and x['peaks']['dual__iu']==x['target'] for x in err])
        bb=np.array([x['valid'][arm] and x['peaks'][arm]==x['target'] for x in err])
        assert [int((aa&bb).sum()),int((aa&~bb).sum()),int((~aa&bb).sum()),int((~aa&~bb).sum())]==[info['erroneous']['raw_peaks'][k] for k in ('both_hit','iu_only_hit','joint_only_hit','neither_hit')]
    heads=['dual__iu','dual__graph010']+[f'dual__cond{k}_graph010' for k in (30,100,300)]
    hits={a:{x['uid'] for x in err if x['valid'][a] and x['peaks'][a]==x['target']} for a in heads}
    assert all(hits[a]==hits['dual__graph010'] for a in heads[2:])
    union=set.union(*hits.values());assert len(union)==22 and len(err)==53
    diagnostics={'status':'POST_EVALUATION_LABEL_USING_DIAGNOSTIC','evaluation_sha256':sha(OUT/'EVALUATION.json'),
        'overlap_artifact_sha256':sha(OUT/'ERROR_OVERLAP.json'),'overlap_boolean_recheck':'PASS',
        'joint_raw_hit_sets_equal_across_caps':True,'union_existing_peak_hits':22,'erroneous_answers':53,
        'scope':'A selector restricted to existing peak locations. Not an upper bound for score fusion, reranking or other features/readouts.'}
    (OUT/'DIAGNOSTICS.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    sections=[];md=['# Graph structure after native conditioning\n\n2026-09-07. Development evidence. Independent review PASS.\n']
    def section(title,text,head=None,rows=None,extra='',collapsed=False):
        content=(table(head,rows) if head else '')+extra
        if collapsed:content='<details><summary>Show complete table</summary>'+content+'</details>'
        sections.append('<section><h2>'+escape(title)+'</h2><p>'+escape(text)+'</p>'+content+'</section>')
        md.append('\n## '+title+'\n\n'+text+'\n'+(mdtable(head,rows) if head else ''))
    section('Decision: retain Joint with graphs, without declaring a winner',
        'Dual Joint with graph0.1 and condition100 gives PRMB AUC 0.63847 and PB F1 30.22%. Dual IU gives '
        '0.63797 and 30.16%. This crosses IU at the point-estimate level, but the differences are only +0.00050 '
        'AUC and +0.06 PB points. Their paired intervals are [-0.03409,+0.02323] and [-7.75,+8.18] points. '
        'There is no demonstrated superiority or optimal cap. The result keeps our Joint/graph direction viable '
        'while showing that more work is needed for a consistent two-task advantage.')
    section('The same fusion, with two distinct controls',
        'Native Joint solves a regularized inverse using its fitted covariance and global loading. Here graph '
        'lambda remains 0.1; condition caps remain 30/100/300. We hold original C/v/u, feature groups, bank routes '
        'and normalization fixed. A node permutation tests whether graph alignment matters. The additional '
        'graph-smoothed equal control substitutes identity covariance and uniform loading under the SAME graph '
        'mechanism, testing what Joint contributes beyond simple aggregation with a graph.',
        extra='<div class="flow"><div><strong>Joint0</strong><br>Original covariance and loading<br>Diagonal ridge only<br>Three frozen condition caps</div><div><strong>Joint + graph</strong><br>Same covariance and loading<br>Aligned or permuted graph<br>Same three caps</div><div><strong>Equal + graph</strong><br>Identity covariance, equal loading<br>Same aligned/permuted graph<br>Simple-control adaptation</div></div>')
    section('Fixed benchmark and fitting scope',
        'All 110 previously evaluated development answers remain: 24 PRMB and 86 PB over GSM8K, MATH, OlympiadBench '
        'and Omni-MATH. Qwen3-8b scores fixed official answers through one teacher-forced gray-box model pass. '
        'The width-eight banks, same-answer fitting and declared negative-entropy anchor remain. All 45 prior '
        'arms replay exactly; 24 native graph heads and eight simple graph controls bring the total to 77. '
        'No Joint fit is rerun. Original real/permuted graph scores at condition1000 replay before new scoring. '
        'Original DUFS gates replay in 182 bank records; the same recipe is computed in 38 remaining banks '
        'for simple controls. New predictions were frozen before this evaluator read labels; this is not untouched confirmation.')
    def pairpoint(a):return view.f(e['metrics'][a]['prm']['auroc'])+' / '+pct(e['metrics'][a]['pb']['macro_f1'])
    grid=[]
    for cap in (1000,300,100,30):
        names=['dual__joint0','dual__graph010','dual__graph_perm'] if cap==1000 else [f'dual__cond{cap}',f'dual__cond{cap}_graph010',f'dual__cond{cap}_graph_perm']
        grid.append([cap]+[pairpoint(a) for a in names])
    section('Interaction matrix: dual Joint',
        'Each entry is PRMB AUC / PB native F1. All cells cover the same 110 answers with the original dual route. '
        'The graph improves PB points most at caps 100 and 300, but point ordering is not a significance test. '
        'All three caps were retained; no best cap is selected from these labels.',
        ['Condition cap','No graph','Our graph, lambda0.1','Permuted graph, lambda0.1'],grid)
    headers=['Arm','Fit coverage','Valid PRMB','Pooled AUC','Within-answer AUC','PB native F1','PB fixed-IU: diagnostic']
    def metric_rows(arms):
        return [[a,str(r['coverage'][a])+'/110',str(e['metrics'][a]['prm']['answers'])+'/24',
            view.f(e['metrics'][a]['prm']['auroc']),view.f(e['metrics'][a]['prm']['within_answer_auc']),
            pct(e['metrics'][a]['pb']['macro_f1']),pct(e['metrics'][a]['pb_common_iu_gate']['macro_f1'])] for a in arms]
    focus=['moment__iu','dual__iu','dual__equal','dual__graph010','dual__cond30_graph010','dual__cond100_graph010',
           'dual__cond300_graph010','dual__equal_graph010','dual__equal_graph_perm','context__equal']
    section('Matched full-coverage references',
        'The new graph-smoothed equal control also stays in the comparison. Its permuted version has higher '
        'PB than native Joint at condition100, but lower PRMB AUC. Different endpoint leaders do not establish '
        'a single consistently better method. The fixed-IU gate is diagnostic only.',headers,metric_rows(focus))
    ch=['New minus reference','Common PRMB','Delta AUC','95% AUC CI','Delta PB (pp)','95% PB CI (pp)','Defined PB draws']
    def pair_rows(keys):
        out=[]
        for k in keys:
            p=c['pairs'][k];u=p['uncertainty'];out.append([k,p['left_prm']['answers'],
                view.f(p['left_prm']['auroc']-p['right_prm']['auroc'],signed=True),view.interval(u['prm_common_valid_ci95']),
                view.f(p['left_pb']['macro_f1']-p['right_pb']['macro_f1'],100,True),view.interval(u['pb_all_population_ci95'],100),u['pb_all_population_valid_draws']])
        return out
    keys=[]
    for cap in (30,100,300):
        a=f'dual__cond{cap}_graph010'
        keys += [a+' minus '+b for b in (f'dual__cond{cap}',f'dual__cond{cap}_graph_perm','dual__iu','dual__equal_graph010')]
    section('Paired evidence for graph structure and fusion',
        'At condition100 the aligned graph adds +4.75 PB points over both no graph and permutation, with interval '
        '[0.00,+10.37]. At condition300 it beats permutation by +5.22 points, interval [+0.61,+11.28], while '
        'the PRMB interval still includes zero. Comparisons against IU and the matched equal-graph control '
        'do not establish a two-task advantage. These are unadjusted exploratory 1000-draw source-group intervals '
        'across 101 registered comparisons; the isolated positive PB interval is not a confirmatory claim.',ch,pair_rows(keys))
    section('What the simple graph control tells us',
        'Dual equal-graph gives PRMB 0.62993 / PB 26.62%, versus plain dual equal 0.62649 / 25.55%. Both improvement '
        'intervals include zero. Its permuted control gives 0.62170 / 31.32%, so the real graph is not uniformly '
        'better for every fusion core. Joint at condition100 exceeds real equal-graph in both points, but both '
        'paired intervals include zero. We have not established that the learned Joint structure adds a reliable '
        'advantage under this graph. The simple control is exactly equal fusion at lambda zero; its graph condition '
        'is at most 3.18289 here, below every cap, so it is identical across the tested caps.',ch,
        pair_rows(['dual__equal_graph010 minus dual__equal','dual__equal_graph010 minus dual__equal_graph_perm']))
    section('Preserve the pure-bank comparison rule',
        'Pure moment/native graph retains 78/110 valid fits and context 102/110; composites and simple controls '
        'cover all answers. Context condition100 graph AUC is 0.68452 on 21 valid PRMB answers. Context equal '
        'on the same 21 scores 0.69658, giving delta -0.01206, interval [-0.02432,-0.00124]. Its full-row '
        'comparison against equal on all 24 would conceal this weakness. Use shared valid IDs, while PB keeps '
        'invalid decisions as failures on the full population.')
    decisions=[]
    for a in ['dual__iu','dual__graph010','dual__cond30_graph010','dual__cond100_graph010','dual__cond100_graph_perm','dual__equal_graph010','dual__equal_graph_perm']:
        clean=sum(x['decision_valid'][a] and x['predictions'][a]==-1 for x in pb if x['target']==-1)
        exact=sum(x['decision_valid'][a] and x['predictions'][a]==x['target'] for x in err)
        peak=sum(x['valid'][a] and x['peaks'][a]==x['target'] for x in err)
        decisions.append([a,f'{clean}/33',f'{exact}/53',f'{peak}/53']+[pct(v['f1']) for v in e['metrics'][a]['pb']['cells'].values()])
    section('ProcessBench: inspect the actual decisions',
        'Dual IU gets 18 clean and 12 exact errors right; Joint condition100 with the real graph gets 17 and 13. '
        'Both therefore get 30 answers correct in total, while subset-balanced F1 differs slightly. A tiny macro '
        'lead is not a uniform improvement. Raw peaks hit 20 errors for IU and 18 for Joint; fixed-IU PB is '
        '30.37% versus 29.53%. Within-answer PRMB is also higher for IU (0.67470 versus 0.66244).',
        ['Arm','Clean hits','Exact-error hits','Raw peak hits','GSM F1','MATH F1','Olympiad F1','Omni F1'],decisions)
    bar='<div class="overlap" role="img" aria-label="Of 53 erroneous answers: both peaks hit 16, only IU hits 4, only Joint hits 2, neither hits 31">'
    for n,label,color in [(16,'Both','#087d78'),(4,'IU only','#8052a0'),(2,'Joint only','#bf7b22'),(31,'Neither','#6b7a83')]:
        bar+=f'<div style="flex:{n};background:{color}" title="{label}: {n}">{n}</div>'
    bar+='</div><p class="legend">Teal: both hit · Purple: IU only · Amber: Joint only · Gray: neither hits</p>'
    section('A more useful next clue: shared misses',
        'This is a POST-EVALUATION diagnostic that uses error labels, not a new scoring method. Of 53 erroneous '
        'PB answers, both IU and Joint raw peaks hit 16; only IU hits four; only Joint hits two; neither hits 31. '
        'The Joint hit sets are identical across condition1000/300/100/30. Even a perfect chooser restricted to '
        'these existing peak locations could hit only 22/53. This is NOT a ceiling for combining full trajectories, '
        'reranking steps or adding features: those operations can produce other locations. It shows why simply '
        'switching between these heads is unlikely to solve the main localization problem.',extra=bar)
    section('All 77 arms and their failures',
        'All 45 preceding arms are exact historical anchors. Twenty-four conditioned native graph heads and eight '
        'simple graph controls are new. No-error and readout failures cannot change the original route. Equal-graph '
        'routes follow the original bank choice, using moment equal-graph where the native route falls back to IU.',
        headers,metric_rows(m['arms']),collapsed=True)
    history=[]
    for a,old in e['older_58_answer_metrics'].items():
        now=e['metrics'][a];history.append([a,old['prm']['answers'],view.f(old['prm']['auroc']),pct(old['pb']['macro_f1']),
            now['prm']['answers'],view.f(now['prm']['auroc']),pct(now['pb']['macro_f1'])])
    section('History remains visible under its original contract',
        'Older-58 and current-110 scores use different question cohorts and are shown separately. Cross-cohort '
        'changes are not algorithmic gains. The new graph-condition arms have no old-58 result here. Claude\'s '
        'multi-answer results still require corrected-fold refits; this answer-only experiment does not repair them.',
        ['Anchor','Old valid PRMB','Old-58 AUC','Old-58 PB','Current valid PRMB','Current-110 AUC','Current-110 PB'],history,collapsed=True)
    section('All 101 registered comparisons',
        'All four interval families, including within-answer AUC and fixed-IU PB, are in CONTRASTS.json. Undefined '
        'bootstrap draws are counted, not set to zero.',ch,pair_rows(list(c['pairs'])),collapsed=True)
    section('Code and findings review',
        'Three pre-freeze tests pass. Independent review verifies 110 label/group joins, 11351 exact parent arrays, '
        '4950 metadata records, 182 gate replays, 220 reconstructed source graphs and zero-graph equal replays, '
        '440 Laplacians and equal-control cap-invariance checks, 1880 inverse/step/GMM reconstructions, 1760 route '
        'inheritances, all 77 metric bundles and 101 paired point bundles. Seven explicit 1000-draw bootstraps '
        'match all four intervals and defined counts. New gate recipes receive ten representative refits. '
        'Graph-builder, DUFS and GMM kernels are reused; Laplacians, trace matching, inverses and endpoints are '
        'reconstructed independently. No Joint refits. The later overlap diagnostic is separately rechecked from '
        'Boolean success sets by this renderer; it is not represented as a preregistered comparison.',
        ['Measurement','Value'],[['Scoring, three CPU workers',f"{f['seconds_this_invocation']:.2f} s"],['101 contrasts',f"{c['seconds_this_invocation']:.2f} s"],
        ['Independent review',f"{r['seconds']:.2f} s"],['Maximum reconstructed risk difference',r['max_reconstructed_risk_difference']],
        ['Browser visual inspection','Not run; HTML structure and local links checked']])
    section('Next bounded direction: complementary information for fusion',
        'Keep IU and the original/conditioned Joint graph recipes as frozen references. Do not widen the same dose '
        'grid just to chase a tiny lead on these 110 answers. The next short stage should audit the old AR/Kalman '
        'innovation code, then test one same-answer prediction-residual view inside the existing feature matrix, '
        'with unchanged-core and matched equal controls. Old final-answer scalar innovations were highly correlated '
        'with entropy; the new view must demonstrate additional information rather than rename that old signal. '
        'This supports the requested KalmanNet/flow track without claiming that a simple predictor implements either '
        'named method. It also does not rule out improved trajectory readout. Broader supporting tracks, corrected-fold '
        'multi-answer refits, full comparators, untouched confirmation and historical24 transfer remain open.')
    links=[('MANIFEST.json','Frozen experiment'),('EVALUATION.json','All metrics and decisions'),('CONTRASTS.json','All paired intervals'),('REVIEW.json','Independent review'),
        ('ERROR_OVERLAP.json','Post-evaluation overlap counts'),('DIAGNOSTICS.json','Separate overlap validation and scope'),('TESTS.txt','Captured tests'),
        ('../fusion_native_conditioning_v1/REPORT.html','Preceding native-conditioning results'),('../fusion_replication_v1/REPORT.html','Original fusion references'),
        ('../../spectral_utils/fusion_graph_conditioning.py','Graph, simple-control and routing code'),('../../docs/experiments/FUSION_GRAPH_CONDITIONING_V1.md','Frozen protocol'),
        ('../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html','Visual guide to our fusion')]
    sections.append('<section><h2>Evidence and code</h2><ul>'+''.join('<li><a href="'+p+'">'+escape(t)+'</a></li>' for p,t in links)+'</ul></section>')
    md.append('\n## Evidence and code\n\n'+'\n'.join('- ['+t+']('+p+')' for p,t in links))
    css='''*{box-sizing:border-box}body{margin:0;background:#f4f6f3;color:#193d45;font:16px/1.65 system-ui,Segoe UI,sans-serif}header{background:#153f48;color:white;padding:45px max(22px,calc((100vw - 1160px)/2))}h1{font-size:clamp(32px,5vw,50px);line-height:1.15}main{max-width:1204px;margin:auto;padding:20px 22px 50px}section{margin:22px 0;padding:24px;background:white;border:1px solid #dce5df;border-radius:13px}h2{font-size:26px;line-height:1.25}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.flow>div{padding:18px;border-top:4px solid #197f70;background:#eaf3ed;border-radius:8px}.flow>div:last-child{background:#fff1df;border-color:#b87719}.table{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce5df;vertical-align:top}th{background:#eaf3ed}td:first-child{font-weight:650;white-space:nowrap}summary{cursor:pointer;font-weight:650;padding:10px 0}a{color:#075c90}a:focus-visible,summary:focus-visible{outline:3px solid #b57715}.overlap{display:flex;border-radius:8px;overflow:hidden;color:white;text-align:center;font-weight:700;margin-top:20px}.overlap>div{padding:16px 0;min-width:22px}.legend{font-size:13px}@media(max-width:720px){.flow{grid-template-columns:1fr}main{padding:12px}section{padding:16px}}@media print{body{background:white;font-size:11pt}header{padding:15px;background:white;color:#193d45}section{border:0;padding:10px}h2{break-after:avoid}.table{overflow:visible}table{font-size:8pt}th,td{padding:5px;white-space:normal!important}}'''
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Joint graph after conditioning</title><style>'+css+'</style></head><body><header><p>07 September 2026 · Reviewed development experiment</p><h1>Joint with a graph<br>reaches IU\'s point scores.</h1><p>The lead is too small to establish superiority. Shared localization misses point to the next useful question.</p></header><main>'+''.join(sections)+'</main></body></html>\n'
    (OUT/'REPORT.html').write_text(html,encoding='utf-8',newline='\n');(OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8',newline='\n')
    class Check(HTMLParser):
        def __init__(self):super().__init__();self.stack=[];self.links=[]
        def handle_starttag(self,t,a):
            if t not in ('meta','br','link','hr','input','img'):self.stack.append(t)
            if t=='a':self.links.append(dict(a)['href'])
        def handle_endtag(self,t):assert self.stack.pop()==t,t
    check=Check();check.feed(html);check.close();assert not check.stack
    for p in check.links:assert (OUT/p).resolve().is_file(),p
    assert all(a in html for a in m['arms']) and all(k in html for k in c['pairs'])
    prov={'status':'PASS','source_hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','REVIEW.json','ERROR_OVERLAP.json','DIAGNOSTICS.json')},
        'report_hashes':{str(OUT/n):sha(OUT/n) for n in ('REPORT.html','REPORT.md')},'renderer_sha256':sha(__file__),'helper_sha256':sha(helper)}
    (OUT/'REPORT_PROVENANCE.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
    (OUT/'ARTIFACT_VALIDATION.json').write_text(json.dumps({'status':'PASS','html_structure':'PASS','local_links':len(check.links),
        'all_77_arms_and_101_comparisons':True,'overlap_boolean_recheck':'PASS','browser_visual_check':'NOT_RUN','provenance_sha256':sha(OUT/'REPORT_PROVENANCE.json')},indent=2),encoding='utf-8')
    print('Reports rendered: 77 arms, 101 comparisons, overlap recheck and',len(check.links),'verified local links.')


if __name__=='__main__':render()
