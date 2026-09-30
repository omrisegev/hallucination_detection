"""Validate report values, links, hashes, and actual JavaScript via Node DOM fixtures."""
import ast
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('sampling_artifact_driver',ROOT/'scripts/run_fusion_sampling_replication_v1.py')
d=importlib.util.module_from_spec(s);s.loader.exec_module(d);OUT=d.OUT


class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.links=[];self.tables={};self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag in ('a','img') and ('href' in a or 'src' in a):self.links.append(a.get('href',a.get('src')))
        if tag=='table':self.table=a.get('id');self.tables[self.table]=[]
        if tag=='tr':self.row=[]
        if tag in ('th','td'):self.cell=''
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('th','td'):
            self.row.append(self.cell);self.cell=None
        if tag=='tr' and self.table is not None:self.tables[self.table].append(self.row);self.row=None
        if tag=='table':self.table=None


def main():
    m=d.verify();e=d.load(OUT/'EVALUATION.json');dg=d.load(OUT/'DIAGNOSTICS.json');r=d.load(OUT/'REVIEW.json');c=d.load(OUT/'CONTRASTS.json')
    assert r['status']=='PASS'
    hashes=dict(m['hashes'])
    for group in (d.load(OUT/'SCORES_FROZEN.json')['files'],r['hashes'],r['dependencies']):
        for p,h in group.items():assert d.sha(p)==h,p;hashes[p]=h
    html=(OUT/'REPORT.html').read_text(encoding='utf-8');page=Page();page.feed(html)
    assert len(page.ids)==len(set(page.ids))
    for link in page.links:
        path,_,fragment=link.partition('#')
        if path:assert (OUT/path).resolve().exists(),link
    source=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert source==dict(arms=m['arms'],metrics=e['metrics'],eligible_metrics=e['eligible_metrics'])
    assert len(page.tables['headline'])==7 and len(page.tables['contrasts'])==94
    assert len(page.tables['affine'])==5 and len(page.tables['support'])==7 and len(page.tables['transitions'])==11
    def number(x,digits=5):return 'n/a' if x is None else f'{x:.{digits}f}'
    def percent(x):return 'n/a' if x is None else f'{100*x:.2f}%'
    def interval(v,scale=1):return 'n/a' if v is None else '['+', '.join(f'{scale*x:+.4f}' for x in v)+']'
    for sel,row in zip(d.SELECTORS,page.tables['headline'][1:]):
        iu,joint,eq=[e['metrics'][d.arm_name(sel,core)] for core in ('iu','graph010','equal_graph_perm')]
        assert row[1:]==[number(iu['prm']['auroc']),number(iu['prm']['within_answer_auc']),percent(iu['pb']['macro_f1']),number(joint['prm']['auroc']),
            percent(joint['pb']['macro_f1']),number(eq['prm']['auroc']),percent(eq['pb']['macro_f1']),str(r['native_joint_valid'][sel])+'/110']
    for actual,pair in zip(page.tables['contrasts'][1:],c['pairs'].values()):
        u=pair['uncertainty'];ap,bp=pair['left_prm']['auroc'],pair['right_prm']['auroc'];ab,bb=pair['left_pb']['macro_f1'],pair['right_pb']['macro_f1']
        expected=[pair['left']+' minus '+pair['right'],pair['scope'],str(len(pair['selected_ids'])),number(ap-bp) if ap is not None and bp is not None else 'n/a',
            interval(u['prm_common_valid_ci95']),number(100*(ab-bb),3) if ab is not None and bb is not None else 'n/a',interval(u['pb_all_population_ci95'],100),
            interval(u['prm_within_answer_common_valid_ci95']),str(u['pb_all_population_valid_draws'])]
        assert actual==expected,(actual,expected)
    for actual,pair in zip(page.tables['affine'][1:],dg['affine_diagnostic']):
        expected=[pair['core']]+[number(pair[k]['auroc']) for k in ('full','sample','sample_ranking_full_scale','full_ranking_sample_scale')]+[number(pair['full']['within_answer_auc']),number(pair['sample']['within_answer_auc'])]
        assert actual==expected
    for actual,(arm,cells) in zip(page.tables['transitions'][1:],dg['pb_transitions'].items()):
        assert actual==[arm]+[str(sum(v[k] for v in cells.values())) for k in ('predictions_changed','peaks_changed','gained','lost')]
    for selector,actual in zip(d.SELECTORS,page.tables['support'][1:]):
        records=[v for v in dg['records'] if v['selector']==selector and v['eligible']];errors=[v for v in records if 'first_error_tokens' in v];short=[v for v in errors if v['first_error_tokens']<=32]
        assert actual[1]==str(len(records)) and actual[3]==str(len(errors)) and actual[7]==str(len(short))
        assert actual[4]==str(sum(v['first_error_fit_support']>0 for v in errors)) and actual[8]==str(sum(v['first_error_fit_support']>0 for v in short))
    node_script=r"""
const fs=require('fs'),vm=require('vm');const html=fs.readFileSync(process.argv[1],'utf8');
const payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n');
const ids={payload:{textContent:payload},scope:{value:'metrics'},search:{value:''},'method-rows':{innerHTML:''},count:{textContent:''}};
for(const x of Object.values(ids))x.addEventListener=()=>{};
const sandbox={document:{getElementById:id=>{if(!(id in ids))throw Error('Unknown ID '+id);return ids[id];}},console};
vm.createContext(sandbox);vm.runInContext(code,sandbox);let results=[];
for(const scope of ['metrics','eligible_metrics'])for(const query of ['','sample_risk_top','sample_dufs','dual__iu','gap__','NO_SUCH_METHOD']){
ids.scope.value=scope;ids.search.value=query;vm.runInContext('render()',sandbox);
results.push({scope,query,html:ids['method-rows'].innerHTML,count:ids.count.textContent});}
process.stdout.write(JSON.stringify(results));
"""
    result=subprocess.run([r'C:\Program Files\nodejs\node.exe','-e',node_script,str(OUT/'REPORT.html')],capture_output=True,text=True,check=True)
    cases=json.loads(result.stdout);numeric_rows=0
    # JS toFixed rounds exact positive halfway cases upward; Python's format
    # uses ties-to-even. Compare the actual JS contract at binary64 precision.
    def js_number(x,digits=5,scale=1):
        if x is None:return 'n/a'
        return format(Decimal.from_float(float(x*scale)).quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP),'f')
    def js_percent(x):return 'n/a' if x is None else js_number(x,2,100)+'%'
    for case in cases:
        names=[name for name in m['arms'] if case['query'].lower() in name.lower()]
        parser=Page();parser.feed('<table id="rendered">'+case['html']+'</table>');actual=parser.tables['rendered'];assert len(actual)==len(names)
        assert case['count']==f'{len(names)} of 149 entries | '+('110 answers' if case['scope']=='metrics' else '72 eligible answers')
        for name,row in zip(names,actual):
            mm=e[case['scope']][name];expected=[name,str(mm['prm']['answers']),js_number(mm['prm']['auroc']),js_number(mm['prm']['within_answer_auc']),js_percent(mm['pb']['macro_f1']),
                str(sum(v['valid_decisions'] for v in mm['pb']['cells'].values())),js_percent(mm['pb_common_iu_gate']['macro_f1'])]
            assert row==expected,(case['scope'],name,row,expected);numeric_rows+=1
    files=[ROOT/'spectral_utils/fusion_sampling_replication.py',ROOT/'tests/test_fusion_sampling_replication.py',
           ROOT/'scripts/run_fusion_sampling_replication_v1.py',ROOT/'scripts/review_fusion_sampling_replication_v1.py',
           ROOT/'scripts/render_fusion_sampling_replication_v1.py',Path(__file__)]
    for path in files:ast.parse(path.read_text(encoding='utf-8'))
    guide=Page();guide.feed((ROOT/'docs/reviews/joint_lsml_visual_guide_2026-09-06.html').read_text(encoding='utf-8'))
    assert len(guide.ids)==len(set(guide.ids)) and 'sampling-replication-update' in guide.ids
    for name in ('sampling_points','selected_windows'):
        svg=(OUT/(name+'.svg')).read_text(encoding='utf-8');assert '<svg' in svg and (OUT/(name+'.png')).stat().st_size>10000
    report=dict(status='PASS',report_sha256=d.sha(OUT/'REPORT.html'),verified_hashes=len(hashes),local_links_images=len(page.links),
        static_numeric_rows=6+93+4+10+6,node_dom_cases=len(cases),node_numeric_rows=numeric_rows,python_ast_files=len(files),guide_unique_ids=len(guide.ids),
        visual_scope='Actual HTML JavaScript executed in Node VM with DOM fixtures; exported PNG plots separately inspected. No browser rendering claim.',
        files={str(p):d.sha(p) for p in files+[OUT/'REPORT.html',OUT/'REPORT.md',OUT/'sampling_points.svg',OUT/'selected_windows.svg',OUT/'sampling_points.png',OUT/'selected_windows.png']})
    d.save(OUT/'ARTIFACT_VALIDATION.json',report);print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
