"""Validate the saved presentation interface; numerical audits are separate."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent

class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.links=[];self.runtime=[];self.svg_count=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='svg':self.svg_count+=1
        if (tag in ['script','img','iframe'] and a.get('src','').startswith(('https:','http:','//'))):self.runtime.append(a['src'])
        if tag=='link' and a.get('rel')=='stylesheet' and a.get('href','').startswith(('https:','http:','//')):self.runtime.append(a['href'])

def main():
    path=PACKAGE/'advisor_review_v11.html';text=path.read_text();p=Page();p.feed(text)
    duplicates=[k for k,v in Counter(p.ids).items() if v>1]
    broken_anchors=[];missing_files=[]
    for raw in p.links:
        u=urlsplit(raw)
        if u.scheme or u.netloc:continue
        if not u.path and u.fragment and u.fragment not in p.ids:broken_anchors.append(raw)
        if u.path and not (PACKAGE/unquote(u.path)).exists():missing_files.append(raw)
    expected=['results','changed','leaderboard','features','decision','tasks','appendix','lr-diagnostics','simplicity-controls','registered-rules','dependence','mom-quota','internal-validation']
    absent=[i for i in expected if i not in p.ids]
    checks={'duplicate_ids':duplicates,'broken_internal_anchors':broken_anchors,'missing_local_evidence_files':missing_files,
            'external_runtime_dependencies':p.runtime,'missing_analysis_sections':absent,
            'unicode_direction_markers':any(c in text for c in ['\u200e','\u200f'])}
    checks.update({'too_few_scientific_figures':p.svg_count<10,
                   'missing_failed_arm_disclosure':'undefined · failed arm' not in text,
                   'missing_task_execution_history':not ('Previous state' in text and 'Action completed' in text),
                   'missing_full_category_rules':'All nine categories under each registered rule' not in text})
    result={'status':'FAIL' if any(checks.values()) else 'PASS','scope':'HTML structure, evidence-link existence and visible contract disclosures; arithmetic verified by separate analysis audits',
            'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'svg_count':p.svg_count,'link_count':len(p.links),'checks':checks}
    (HERE/'HTML_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert result['status']=='PASS',json.dumps(checks,indent=2)

if __name__=='__main__':main()
