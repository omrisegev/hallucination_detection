"""Reproduce the literature-claim audit from archived primary sources."""
from pathlib import Path
from html.parser import HTMLParser
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
KEYS = ['overall', 'NR', 'NCL', 'simplicity', 'ES', 'SC', 'DC', 'CI', 'soundness', 'PS', 'DR', 'MS', 'sensitivity']

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.row = None
        self.cell = None
        self.rows = []
    def handle_starttag(self, tag, attrs):
        if tag == 'table' and dict(attrs).get('id') == 'S3.T3.2.1': self.active = True
        if not self.active: return
        if tag == 'tr': self.row = []
        if tag in ('td', 'th'): self.cell = []
    def handle_data(self, data):
        if self.active and self.cell is not None: self.cell.append(data)
    def handle_endtag(self, tag):
        if not self.active: return
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        if tag == 'table': self.active = False

def summarize(rows):
    cat = lambda r: r['simplicity'] < min(r['soundness'], r['sensitivity'])
    both = lambda r: max(r['NR'], r['NCL']) < r['overall']
    return {
        'n_rows': len(rows),
        'simplicity_strictly_below_both_other_category_means': sum(cat(r) for r in rows),
        'both_NR_and_NCL_strictly_below_own_overall': sum(both(r) for r in rows),
        'category_counterexamples': [r for r in rows if not cat(r)],
        'subtype_counterexamples': [r for r in rows if not both(r)],
    }

def main():
    parser = TableParser()
    parser.feed((ROOT/'sources/prmbench_arxiv_v1.html').read_text())
    paper = []
    for row in parser.rows:
        if len(row) == 14 and row[0] != 'Avg.':
            try: vals = [float(v) for v in row[1:]]
            except ValueError: continue
            paper.append({'model': row[0], **dict(zip(KEYS, vals))})
    assert len(paper) == 15, f'Paper row parse failed: {len(paper)}'
    live_source = json.loads((ROOT/'sources/prmbench_leaderboard.json').read_text())
    live = [{'model': r['Model'], 'class': r['Class'], 'overall': float(r['Overall']),
             'simplicity': float(r['S1']), 'soundness': float(r['S2']), 'sensitivity': float(r['S3']),
             **{k: float(r[k]) for k in ['NR','NCL','ES','SC','DC','CI','PS','DR','MS']}}
            for r in live_source if r['Class'] != 'human']
    sources = []
    urls = {'prmbench_arxiv_v1.html':'https://arxiv.org/html/2501.03124v1',
            'prmbench_homepage.html':'https://prmbench.github.io/',
            'prmbench_leaderboard.js':'https://prmbench.github.io/static/js/prmbench_leaderboard.js',
            'prmbench_leaderboard.json':'https://prmbench.github.io/static/data/prmbench_leaderboard.json'}
    for name,url in urls.items():
        f=ROOT/'sources'/name
        sources.append({'path':str(f.relative_to(ROOT.parent)), 'url':url,
                        'sha256':hashlib.sha256(f.read_bytes()).hexdigest(), 'bytes':f.stat().st_size,
                        'access_date':'2026-10-11'})
    out = {'status':'UNIVERSAL_40_ROW_CLAIM_WITHDRAWN', 'comparison_definitions':
           ['Simplicity mean < both Soundness and Sensitivity means',
            'NR < own Overall and NCL < own Overall; strict comparisons on published rounded values'],
           'paper_v1':summarize(paper), 'live_snapshot':summarize(live),
           'paper_rows':paper, 'live_rows':live, 'sources':sources,
           'interpretation':'Simplicity is frequently weaker, but neither the paper nor the current official leaderboard supports an all-model universal claim. Published model rows are not independent replications.',
           'construction':'PRM800K correct paths are edited by GPT-4o and filtered; QwQ provides correct alternative paths for MS. Simplicity errors can preserve final answer correctness.',
           'mechanism_status':'The construction defines the target but does not causally establish why a model fails. Correctness-focused training is a hypothesis, not a verified causal explanation.',
           'dataset_counts':{'paper_answers':6216,'paper_steps':83456,'local_noncontrol_answers':6211,'local_noncontrol_steps':83371,'local_added_controls':758},
           'o1_mini_n':394}
    (ROOT/'LITERATURE.json').write_text(json.dumps(out,indent=2)+'\n')
    with (ROOT/'paper_table3.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model']+KEYS);w.writeheader();w.writerows(paper)
    print(json.dumps({k:out[k] for k in ['status','paper_v1','live_snapshot']},indent=2))

if __name__=='__main__': main()
