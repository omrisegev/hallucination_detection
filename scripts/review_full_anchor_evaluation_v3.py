"""Supplement the completed anchor aggregation with provenance/coverage review."""
from collections import Counter, defaultdict
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/localization_full_benchmark_v3'
OUT=RUN/'evaluation'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    assert load(OUT/'RUN_STATE.json')['phase']=='COMPLETE_ANCHOR_EVALUATION'
    assert load(OUT/'REVIEW.json')['status']=='PASS'
    manifest,joined=load(RUN/'MANIFEST.json'),load(OUT/'JOINED.json')
    assert joined['records']==manifest['selected'] and joined['arms']==manifest['arms']
    assert joined['scores_freeze_sha256']==sha(RUN/'SCORES_FROZEN.json')
    assert joined['arrays_sha256']==sha(OUT/'JOINED.npz')
    release=load(ROOT/'results/localization_prm_label_audit_v1/RELEASE_V3.json')
    for cell,info in release['cells'].items():
        path=Path(info['label_path']);assert sha(path)==manifest['hashes'][str(path)]
    with np.load(OUT/'JOINED.npz',allow_pickle=False) as z:
        valid=z['valid'];decision=z['decision']
    coverage=defaultdict(Counter)
    for i,rec in enumerate(joined['records']):
        meta=load(RUN/'scores'/(rec['uid']+'.json'))
        for j,arm in enumerate(joined['arms']):
            m=meta['methods'][arm]
            assert valid[i,j]==m['valid'] and decision[i,j]==m['decision_valid']
            key=(rec['cell'],arm,m.get('source_arm') or 'UNSPECIFIED',bool(m['valid']),bool(m['decision_valid']))
            coverage[key]['answers']+=1
    records=[dict(cell=key[0],method=key[1],source_arm=key[2],score_valid=key[3],decision_valid=key[4],answers=c['answers']) for key,c in sorted(coverage.items())]
    metrics=load(OUT/'METRICS.json')['metrics']
    for arm,m in metrics.items():
        expected=sum(x['answers'] for x in records if x['method']==arm and x['cell'].startswith('prm') and x['score_valid'])
        assert expected==m['prm']['answers']
        for cell,detail in m['pb']['cells'].items():
            got=sum(x['answers'] for x in records if x['method']==arm and x['cell']==cell and x['decision_valid'])
            assert got==detail['valid_decisions']
    (OUT/'SOURCE_VALIDITY_COVERAGE.json').write_text(json.dumps(dict(records=records,
        note='Per-cell source provenance crossed with actual score/decision validity; source names alone are not native-valid coverage.'),indent=2),encoding='utf-8')
    class Parsed(HTMLParser):
        def __init__(self):super().__init__();self.rows=0;self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='tr':self.rows+=1
            if tag=='a':self.links.append(dict(attrs)['href'])
    h=Parsed();h.feed((OUT/'REPORT.html').read_text(encoding='utf-8'))
    assert h.rows==47
    for link in h.links:assert (OUT/link).is_file()
    report=dict(status='PASS',joined_provenance=True,label_hashes_rechecked=9,
        method_validity_records=len(joined['records'])*len(joined['arms']),
        source_validity_groups=len(records),metric_coverage_reconciled=True,
        report_rows=h.rows,local_links=len(h.links),browser_rendered=False,
        report_sha256=sha(OUT/'REPORT.html'),metrics_sha256=sha(OUT/'METRICS.json'),
        intervals_sha256=sha(OUT/'INTERVALS.json'),script_sha256=sha(Path(__file__)),
        scope='Same-session supplementary review; second-agent code/contract inspection found no blocking metric issue. Not external scientific confirmation.')
    (OUT/'REVIEW_SUPPLEMENT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
