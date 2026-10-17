"""Final-only structural, task-inventory and displayed-number validation."""
from pathlib import Path
from html.parser import HTMLParser
import csv,hashlib,json,re

P=Path(__file__).resolve().parents[1]
class Document(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.href=[];self.external=[];self.tables=[]
        self.table=None;self.row=None;self.cell=None;self.text=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.href.append(a['href'])
        if tag in ('script','img','link'):
            u=a.get('src',a.get('href',''))
            if u.startswith(('http:','https:','//')):self.external.append(u)
        if tag=='table':self.table=[]
        if tag=='tr':self.row=[]
        if tag in ('th','td'):self.cell=[]
    def handle_data(self,data):
        self.text.append(data)
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ('th','td') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()));self.cell=None
        if tag=='tr' and self.row is not None:
            self.table.append(self.row);self.row=None
        if tag=='table' and self.table is not None:
            self.tables.append(self.table);self.table=None

receipt=json.loads((P/'report/BUILD_RECEIPT.json').read_text())
assert receipt['status']=='FINAL'
path=P/'advisor_review_v11.html';content=path.read_text();digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest==receipt['html_sha256']
d=Document();d.feed(content)
assert len(d.ids)==len(set(d.ids)), 'Duplicate IDs including embedded SVG markers'
assert not d.external,'Runtime depends on external resources'
bad=[]
for href in d.href:
    if href.startswith(('#','http:','https:','mailto:','data:')):continue
    target=P/href.split('#')[0].split('?')[0]
    if not target.exists():bad.append(href)
assert not bad,bad
required=['1','2','3','4','5','5א','6','7','8','9','9א','9ב','10','11','12','13']
tasks=next(t for t in d.tables if t[0][0]=='Task' and 'Requirement' in t[0])
assert {r[0] for r in tasks[1:]}==set(required)
assert any(h in tasks[0] for h in ['Previous state','Before','Prior status']),tasks[0]
assert any(h in tasks[0] for h in ['Action completed','Action done','Action']),tasks[0]
rows=json.loads((P/'report_data.json').read_text())['task_status']
assert {r['id'] for r in rows}==set(required)
header=next(t for t in d.tables if 'Within-answer AUC' in t[0] and 'Error flag rule' in t[0])
assert ['65.381','0.779','59.85'] == header[1][1:4],header
assert ['65.359','0.780','59.52'] == header[2][1:4],header
assert ['65.457','0.801','57.78'] == header[3][1:4],header
matched=next(t for t in d.tables if t[0][:3]==['Category','Matched N','L-SML'])
raw=list(csv.DictReader((P/'controls/NUISANCE_CONTROLS.csv').open()))
lookup={(r['classification'],r['method'],r['control']):r for r in raw}
for row in matched[1:]:
    typ=row[0].replace('redundancy','redundency');expected=[]
    for m in ['LSML_CONT','IU_PCR','PRM_NATIVE','STEP_INDEX','TOKEN_LENGTH','TEXT_REPETITION']:
        value=lookup[typ,m,'position_length'];expected.append(f"{float(value['within_auc']):.3f}")
        assert row[1]==value['n_auc_answers'],row
    assert row[2:]==expected,(row,expected)
text=' '.join(d.text)
for phrase in ['Sensitivity includes MS','internal validation','not untouched confirmation',
               'partial score','supervised-selected diagnostic','no Git commit, push or Claude update asserted']:
    assert phrase.lower() in text.lower(),phrase
assert '\u200e' not in content and '\u200f' not in content
result=dict(status='PASS',html_sha256=digest,builder_sha256=receipt['builder_sha256'],
            n_tables=len(d.tables),n_ids=len(d.ids),n_task_rows=len(tasks)-1,
            external_runtime_resources=d.external,broken_local_links=bad,
            checks=['exact headline points','same position-length pairs/denominators in displayed control table',
                    'all16 task IDs and prior/action fields','no duplicate SVG/section IDs',
                    'local evidence links resolve','development and failed-arm disclosures',
                    'uniform MS/Sensitivity convention','no bidi control characters'])
(P/'review/HTML_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
