"""Check static/interactive numbers, scientific hashes, links and report assets."""
import ast
from decimal import Decimal,ROUND_HALF_UP
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ROOT/'scripts/run_fusion_trajectory_imm_v1.py','trajectory_artifact_driver');OUT=d.OUT
Page=module(ROOT/'scripts/validate_fusion_sampling_report_v1.py','trajectory_page_parser').Page
def n(x,digits=5):return 'n/a' if x is None else f'{x:.{digits}f}'
def pct(x):return 'n/a' if x is None else f'{100*x:.2f}%'
def ci(v,scale=1):return 'n/a' if v is None else '['+', '.join(f'{x*scale:+.4f}' for x in v)+']'


def main():
    m=d.verify();e,c,r,dg,g=[d.load(OUT/x) for x in ('EVALUATION.json','CONTRASTS.json','REVIEW.json','DIAGNOSTICS.json','GATE_AUDIT.json')]
    assert r['status']=='PASS';hashes=dict(m['hashes'])
    for group in [d.load(OUT/'SCORES_FROZEN.json')['files'],r['hashes'],r['dependencies'],g['hashes'],g['dependencies']]:
        for p,h in group.items():assert d.sha(p)==h,p;hashes[p]=h
    html=(OUT/'REPORT.html').read_text(encoding='utf-8');page=Page();page.feed(html)
    assert len(page.ids)==len(set(page.ids))
    for link in page.links:assert (OUT/link).resolve().is_file(),link
    payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert payload==dict(arms=m['arms'],new_arms=m['new_arms'],metrics=e['metrics'])
    names=['dual__iu','dual__cond100_graph010','sample_risk_top__equal_graph_perm']+m['new_arms']
    assert len(page.tables['headline'])==len(names)+1
    for name,row in zip(names,page.tables['headline'][1:]):
        mm=e['metrics'][name];o=dg['pb_outcomes'][name]
        assert row==[name,n(mm['prm']['auroc']),n(mm['prm']['within_answer_auc']),pct(mm['pb']['macro_f1']),pct(mm['pb_common_iu_gate']['macro_f1'])]+[str(o[x]) for x in ('clean_correct','error_exact','raw_peak_exact')]
    expected=[]
    for family,values in g['exchanges'].items():
        for x in values:expected.append([family,x['peak'].split('__')[-1],x['gate'].split('__')[-1],str(x['clean_correct']),str(x['error_exact']),pct(x['pb'])])
    assert page.tables['gate-exchanges'][1:]==expected
    assert page.tables['serial'][1:]==[[x['family'],x['readout'],'clean' if x['clean'] else 'error',str(x['answers']),n(x['median_lag1'],3),n(x['median_bic_advantage_two'],2),str(x['gate_open'])] for x in g['summaries']]
    assert page.tables['transitions'][1:]==[[p]+[str(x[k]) for k in ('gained','lost','peak_changed','gate_changed')] for p,x in dg['pb_transitions'].items()]
    for row,p in zip(page.tables['contrasts'][1:],c['pairs'].values()):
        u=p['uncertainty'];a,b=p['left_prm']['auroc'],p['right_prm']['auroc'];ap,bp=p['left_pb']['macro_f1'],p['right_pb']['macro_f1']
        assert row==[p['left']+' minus '+p['right'],p['scope'],str(len(p['selected_ids'])),n(a-b) if a is not None and b is not None else 'n/a',ci(u['prm_common_valid_ci95']),
            n(100*(ap-bp),3) if ap is not None and bp is not None else 'n/a',ci(u['pb_all_population_ci95'],100),ci(u['prm_within_answer_common_valid_ci95']),str(u['pb_all_population_valid_draws'])]
    assert len(page.tables['contrasts'])==51
    import numpy as np
    for row,family in zip(page.tables['models'][1:],list(d.PAIRS)+list(d.SINGLES)):
        x=[r for r in dg['models'] if r['family']==family];corr=[r['source_correlation'] for r in x if r['source_correlation'] is not None]
        assert row==[family,str(len(x)),str(r['duplicate_collapses'].get(family,0)),str(r['inherited_fallbacks'].get(family,0)),n(float(np.median(corr)),3) if corr else 'single input',
            n(float(np.median([r['condition'] for r in x])),2),str(sum(any(v<0 for v in r['weights']) for r in x))]
    node=r"""
const fs=require('fs'),vm=require('vm'),html=fs.readFileSync(process.argv[1],'utf8');
const payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n');
const ids={payload:{textContent:payload},scope:{value:'all'},search:{value:''},'method-rows':{innerHTML:''},count:{textContent:''}};
for(const x of Object.values(ids))x.addEventListener=()=>{};
const context={document:{getElementById:id=>{if(!(id in ids))throw Error(id);return ids[id];}}};vm.createContext(context);vm.runInContext(code,context);
let out=[];for(const scope of ['all','new'])for(const query of ['','imm','traj_iu_joint','dual__iu','sample_risk','NO_SUCH_METHOD']){
ids.scope.value=scope;ids.search.value=query;vm.runInContext('render()',context);out.push({scope,query,html:ids['method-rows'].innerHTML,count:ids.count.textContent});}
process.stdout.write(JSON.stringify(out));
"""
    result=subprocess.run([r'C:\Program Files\nodejs\node.exe','-e',node,str(OUT/'REPORT.html')],capture_output=True,text=True,check=True)
    cases=json.loads(result.stdout);numeric=0
    def js(x,digits=5,scale=1):return 'n/a' if x is None else format(Decimal.from_float(float(x*scale)).quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP),'f')
    def jsp(x):return 'n/a' if x is None else js(x,2,100)+'%'
    for case in cases:
        roster=m['arms'] if case['scope']=='all' else m['new_arms'];names=[n for n in roster if case['query'].lower() in n.lower()]
        p=Page();p.feed('<table id="rendered">'+case['html']+'</table>');assert len(p.tables['rendered'])==len(names)
        assert case['count']==f'{len(names)} of {len(roster)} entries | same110 answers'
        for name,row in zip(names,p.tables['rendered']):
            mm=e['metrics'][name];assert row==[name,str(mm['prm']['answers']),js(mm['prm']['auroc']),js(mm['prm']['within_answer_auc']),jsp(mm['pb']['macro_f1']),
                str(sum(x['valid_decisions'] for x in mm['pb']['cells'].values())),jsp(mm['pb_common_iu_gate']['macro_f1'])];numeric+=1
    files=[ROOT/'spectral_utils/fusion_trajectory_imm.py',ROOT/'tests/test_fusion_trajectory_imm.py']+[ROOT/'scripts'/n for n in
        ['run_fusion_trajectory_imm_v1.py','review_fusion_trajectory_imm_v1.py','audit_fusion_trajectory_gate_v1.py','render_fusion_trajectory_imm_v1.py','validate_fusion_trajectory_report_v1.py']]
    for p in files:ast.parse(p.read_text(encoding='utf-8'))
    for name in ('trajectory_points','gate_peak_exchange'):
        assert '<svg' in (OUT/(name+'.svg')).read_text(encoding='utf-8');assert (OUT/(name+'.png')).stat().st_size>10000
        files.extend([OUT/(name+'.svg'),OUT/(name+'.png')])
    guide=Page();guide.feed((ROOT/'docs/reviews/joint_lsml_visual_guide_2026-09-06.html').read_text(encoding='utf-8'))
    assert 'trajectory-imm-update' in guide.ids and len(guide.ids)==len(set(guide.ids))
    report=dict(status='PASS',report_sha256=d.sha(OUT/'REPORT.html'),verified_hashes=len(hashes),local_links_images=len(page.links),
        static_numeric_rows=sum(len(v)-1 for k,v in page.tables.items() if k!='all-methods'),node_dom_cases=len(cases),node_numeric_rows=numeric,
        python_ast_files=7,guide_unique_ids=len(guide.ids),files={str(p):d.sha(p) for p in files+[OUT/'REPORT.html',OUT/'REPORT.md',OUT/'GATE_AUDIT.json']},
        visual_scope='Actual report JavaScript exercised through Node VM/DOM fixtures; both exported PNG figures separately inspected. No browser rendering claimed.')
    d.save(OUT/'ARTIFACT_VALIDATION.json',report);print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
