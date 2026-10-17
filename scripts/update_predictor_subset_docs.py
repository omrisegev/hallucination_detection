"""Record completed Step389 without rewriting historical document bytes."""
from pathlib import Path
import json,html
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/predictor_subset_iu_v1'

def read(p):return p.read_bytes().decode('utf8')
def save(p,s):p.write_bytes(s.encode('utf8'))

def run():
    state=json.loads(read(OUT/'RUN_STATE.json'))
    if state['status']!='COMPLETE_REVIEWED':raise ValueError('Full results required')
    review=json.loads(read(OUT/'REVIEW.json'));d=json.loads(read(OUT/'METRICS.json'))
    summary=review['summary_en'].strip()+'\n'
    p=ROOT/'PROGRESS.md';s=read(p)
    start=s.index('ACTIVE — Step389');end=s.index('LATEST — Step388',start)
    s=s[:start]+'LATEST — Step389 COMPLETE_REVIEWED\r\n'+summary.replace('\n','\r\n')+'\r\n'+s[end:].replace('LATEST — Step388','PREVIOUS — Step388',1)
    save(p,s)
    p=ROOT/'Research_Directions.md';s=read(p)
    save(p,'## 2026-09-15 — compare all predictor subsets (Step389)\r\n\r\n'+summary.replace('\n','\r\n')+'\r\n'+s.replace('LATEST — Step388','PREVIOUS — Step388',1))
    p=ROOT/'HISTORY.md';s=read(p)
    if '### Step 389' in s:raise ValueError('Step389 already recorded')
    entry='''### Step 389 — Compare all three-, four- and five-predictor IU combinations [Codex]

**What**: User authorized all available predictor subsets. Fit answer-local
canonical IU weights to standardized signed predictor residuals and compare
with equal weights on the same columns. Keep the innovation base, token
Top10, signed correction and gate fixed. Reuse the saved excluded-group fits.

**Why**: Determine whether predictor fusion improves the existing method,
and isolate predictor selection from the contribution of learned weights.

**Result**:
'''+summary+'''
**Validation and storage**: Three new tests including45 canonical covariance
comparisons; all48 metric bundles independently verified; all32 readouts
replayed independently over the full population. Singleton predictions
replayed under every outer and nested exclusion. Complete provenance and
per-cell results are stored. Recovered full disk by removing regenerable
bytecode, retiring the clean binary-moment worktree while preserving its
branch and unique ignored results, and hardlinking27 byte-identical inactive
CLI copies. No research results deleted; see STORAGE.json.

**Files changed**:
- spectral_utils/predictor_subset_fusion.py and tests/test_predictor_subset_fusion.py.
- scripts/{run,evaluate,report,update}_predictor_subset* — execution and review.
- docs/experiments/PREDICTOR_SUBSET_IU_20260915.md — frozen comparison contract.
- results/predictor_subset_iu_v1/ — complete metrics, intervals, reports and hashes.
- PROGRESS, Research_Directions and central execution report — findings and scope.
Large residual arrays, weights, score archives and SQLite remain local with hashes.

---

'''
    save(p,s+entry.replace('\n','\r\n'))
    paragraphs=review['paragraphs_he'];names=[]
    for n in [d['leaders']['pb_all8'],d['leaders']['prm_within'],d['leaders_by_head']['iu']['pb_all8'],'tcn__real','ridge','bocpd','innovation5']:
        if n not in names:names.append(n)
    rows=[[n,f"{100*d['metrics'][n]['pb_all8']:.4f}%",f"{d['metrics'][n]['prm_within']:.6f}",f"{d['metrics'][n]['prmscore_q08']:.6f}"] for n in names]
    headers=['תצורה','PB F1','within-AUC','PRMScore']
    table='| '+' | '.join(headers)+' |\n|---|---:|---:|---:|\n'+''.join('| '+' | '.join(r)+' |\n' for r in rows)
    links='[כל 32 התצורות וההשוואות](../../results/predictor_subset_iu_v1/REPORT.html) · [מדדים ורווחי סמך](../../results/predictor_subset_iu_v1/METRICS.json).'
    p=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.md';s=read(p);i=s.index('\n',s.index('# '))+1
    addition='\n## כל צירופי המנבאים — Step389\n\n'+'\n\n'.join(paragraphs)+'\n\n'+table+'\n'+links+'\n\n'
    save(p,s[:i]+addition+s[i:])
    p=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.html';s=read(p);i=s.index('</h1>')+5
    section='<section id="predictor-subsets-step389"><h2>כל צירופי המנבאים — Step389</h2>'+''.join('<p>'+html.escape(x)+'</p>' for x in paragraphs)
    section+='<table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'
    section+='<p><a href="../../results/predictor_subset_iu_v1/REPORT.html">כל 32 התצורות וההשוואות</a> · <a href="../../results/predictor_subset_iu_v1/METRICS.json">מדדים ורווחי סמך</a></p></section>'
    save(p,s[:i]+section+s[i:])

if __name__=='__main__':run()
