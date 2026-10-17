"""All-answer text/trajectory explorer for the corrected-label diagnostic."""
from html import escape
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('forensics_report_driver',ROOT/'scripts/audit_fusion_localization_forensics_v3.py')
d=importlib.util.module_from_spec(sp);sp.loader.exec_module(d)
OUT=d.OUT
load,save,sha=d.load,d.save,d.sha


def main():
    a=load(OUT/'AUDIT.json');review=load(OUT/'REVIEW.json');assert review['status']=='PASS'
    for p,h in review['hashes'].items():assert sha(Path(p))==h,p
    s=a['summary'];metrics=a['historical_metrics'];lead=['dual__iu','dual__cond100_graph010','dual__equal_graph_perm','ar1__iu','ar1__graph010']
    facts={'answers':len(a['records']),'tokens':sum(r['tokens'] for r in a['records']),
        'steps':sum(r['steps'] for r in a['records']),'separator_tokens':sum(r['separator_tokens'] for r in a['records'])}
    lines=['# Where our fusion localizers fail — corrected-label forensics','',
        'Step314. Frozen scores; no new detector or claimed algorithm gain. Review PASS.','',
        f'All {facts["answers"]} answers / {facts["tokens"]:,} tokens / {facts["steps"]:,} steps pass exact token-ID and span replay. Three original scalar streams match the extracted input exactly. There are {facts["separator_tokens"]} legitimate separator tokens outside official spans. All1,760 method/answer score projections replay within2.67e-15.','',
        '## The two bottlenecks','',
        '| Fusion | Clean decisions correct /33 | Raw peak exact /53 errors | Exact after gate /53 | Exact peaks hidden by gate | False alarms /33 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in lead:
        q=s[arm];c=q['native_categories'];lines.append(f'| {arm} | {c.get("clean_correct",0)} | {q["raw_error_peak_exact"]} | {c.get("error_exact",0)} | {q["exact_peak_hidden_by_gate"]} | {c.get("clean_false_alarm",0)} |')
    lines += ['', '## Boundary ties are real, but a limited part of this failure','',
        'IU and graph100 each have16/86 PB answers with tied highest-scoring steps; those ties have shared window support. IU hits20/53 first errors with its actual peak; the correct step occurs somewhere in the tied top set in22/53. Joint gives18/53 and21/53. A perfect label-using tie choice with the existing gate raises IU PB from30.16% to31.77%. This is an oracle diagnostic, not a deployable gain.','',
        'Both original IU and Joint miss31/53 first errors using their actual peaks. Even their combined tied top sets miss28/53. This restricts choosing among those existing peaks, not the potential of full-trajectory fusion or new measurements.','',
        '## What changed with prediction residuals','',
        'AR+Joint changes16 binary gate decisions and13 peak locations. Only2/13 changed peaks had an old top-two gap <=0.1 score SD. It loses10 previously correct complete decisions and gains2. Four previously correct raw peaks are lost and two gained. Its weaker PB result is therefore not explained solely by almost-tied maxima or solver failure. This is descriptive attribution to changed outputs, not a proven cause inside the weights.','',
        '## Component-conditional oracles, not new methods','',
        '| Fusion | Actual PB | Perfect binary gate; same peak | Perfect locator; same gate | Perfect top-tie choice; same gate |',
        '|---|---:|---:|---:|---:|']
    for arm in lead:
        q=s[arm]['oracle_diagnostics'];lines.append(f'| {arm} | {100*metrics[arm]["pb"]["macro_f1"]:.2f}% | '+ ' | '.join(f'{100*q[k]["macro_f1"]:.2f}%' for k in ('perfect_binary_gate','perfect_locator','top_tie_locator'))+' |')
    lines += ['', 'These columns use the answer labels deliberately. A perfect gate cannot fix a wrong peak; a perfect locator cannot fix false alarms or a closed gate. They are not achievable-performance forecasts.','',
        '## Decision for the next bounded experiment','',
        'Keep IU-PCR, Joint graph100 and the simple permuted-graph control. Do not make boundary tie-breaking or a finer lambda grid the main next experiment: the measured recoverable set is small and both gate and risk localization fail. Inspect whether existing single-pass token confidence carries correctness evidence that our uncertainty-oriented fused trajectory suppresses, then freeze one supporting feature/readout change with matched controls. Audit earlier implementations before calling it new. No new candidate is selected here.','',
        '## Limits and review','',
        'The review re-read raw pickle metadata and used the independent existing alignment API, incidence-matrix score projection, pairwise AUC, PB counts and all48 oracle endpoint reconstructions. All98 corrected metric bundles,16 native summaries and6 transition bundles pass. Shared tokenizer, raw metadata reader and saved fusion scores are disclosed. This validates downstream alignment for these110 answers, not the original model logit-position slice or every top-K-derived feature. No new inference or multi-answer fitting.','',
        'The broader corrected historical bridges/refits, full comparators, IMM/LOCA/Flows/KalmanNet adaptations, untouched confirmation and historical24 transfer remain open.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    table=[]
    for arm,q in s.items():
        c=q['native_categories'];table.append('<tr><td>'+escape(arm)+'</td>'+''.join('<td>'+str(v)+'</td>' for v in [c.get('clean_correct',0),q['raw_error_peak_exact'],c.get('error_exact',0),q['exact_peak_hidden_by_gate'],c.get('clean_false_alarm',0),q['all_pb_shared_top_plateaus']])+'</tr>')
    oracle=[]
    for arm in d.ARMS:
        q=s[arm]['oracle_diagnostics'];vals=[metrics[arm]['pb']['macro_f1']]+[q[k]['macro_f1'] for k in ('perfect_binary_gate','perfect_locator','top_tie_locator')]
        oracle.append('<tr><td>'+escape(arm)+'</td>'+''.join('<td>'+f'{100*v:.2f}%'+'</td>' for v in vals)+'</tr>')
    anchors=[]
    for arm,m in metrics.items():
        anchors.append('<tr><td>'+escape(arm)+'</td><td>'+str(m['prm']['answers'])+'/24</td><td>'+('unavailable' if m['prm']['auroc'] is None else f'{m["prm"]["auroc"]:.5f}')+'</td><td>'+f'{100*m["pb"]["macro_f1"]:.2f}%'+'</td></tr>')
    data=[]
    for r in a['records']:
        row={k:r[k] for k in ('uid','cell','row_id','target','tokens','steps','question','text_steps','spans','classification')}
        row['methods']={arm:{k:q[k] for k in ('step_scores','peak','prediction','bic_advantage_two','numerical_top_steps','shared_peak_plateaus','window_starts','window_ends','window_risk')} for arm,q in r['methods'].items()}
        data.append(row)
    blob=json.dumps({'records':data,'cases':a['cases']},ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    introduction=''.join('<p>'+escape(line)+'</p>' for line in lines if line.startswith(('IU and graph100','Both original','AR+Joint changes')))
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Where fusion misses the first error</title><style>
body{font:17px/1.6 system-ui,sans-serif;color:#24374a;background:#f3f6f8;margin:0}main{max-width:1200px;margin:auto;padding:30px}h1{font-size:2.4rem;line-height:1.15}h2{margin-top:2em}.note{padding:18px;border-left:5px solid #127a71;background:#e5f3ef}.warn{background:#fff1d7;border-color:#b97a18}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:9px;border-bottom:1px solid #d9e1e6;text-align:left;vertical-align:top}th{background:#e2eaf1;position:sticky;top:0}td:first-child{overflow-wrap:anywhere}.scroll{overflow:auto;max-height:570px}select,input,button{font:inherit;padding:8px;max-width:100%;margin:4px}button,summary{cursor:pointer}.card{background:white;padding:18px;border:1px solid #d9e1e6;border-radius:9px;margin:15px 0}.controls{display:flex;flex-wrap:wrap;gap:10px}svg{width:100%;height:auto;background:white}.gold{background:#fff0d0}.chosen{border-left:4px solid #137a72}.steptext{white-space:pre-wrap;min-width:280px;max-width:680px}a{color:#146496}.muted{color:#546578;font-size:14px}details{margin:20px 0}.legend{display:flex;gap:18px;flex-wrap:wrap}.legend b{border-bottom:3px solid}code{background:#e2eaf1;padding:2px}</style></head><body><main>
<p>IU-PCR / Joint L-SML · Step314 · corrected v3 labels</p><h1>Where our fusion misses the first error</h1><div class="note">All110 answers,71,385 tokens and1,112 official steps pass downstream alignment checks. All1,760 frozen trajectory projections replay. This report explains existing failures; it does not introduce a new localizer or claim an improvement.</div>
<h2>Both the risk peak and the no-error decision matter</h2><p>The fusion produces a risk score along the answer. The current readout chooses the step with the highest score. A separate GMM gate decides whether to report that step or return <code>no error</code>. A mixture models score distributions; its states are not correctness labels.</p><div class="scroll"><table><thead><tr><th>Fusion</th><th>Clean correct /33</th><th>Raw peak exact /53 errors</th><th>Exact after gate /53</th><th>Exact peaks hidden</th><th>False alarms /33</th><th>Shared top plateau /86</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div>'''+introduction+'''
<h2>Inspect the actual answer and its scores</h2><p>Step numbers below start at1 for readability. The gold highlight marks the first error on ProcessBench, or every annotated error on PRMBench. Later ProcessBench steps are not certified correct. Text is taken from the original cache, not generated by this report.</p><div class="controls"><label>Example category <select id="category"><option value="all">All110 answers</option><option value="shared_boundary_peak_with_gold_in_tie">Shared boundary peak</option><option value="AR_graph_loses_correct_peak">AR graph loses a correct peak</option><option value="AR_graph_new_clean_false_alarm">AR graph adds a clean false alarm</option><option value="both_originals_miss_outside_top_tie">Both original peaks miss</option></select></label><label>Answer <select id="answer"></select></label><label>Compare with IU <select id="method"></select></label></div><div id="caseinfo" class="card"></div><div id="question" class="card"></div><div class="legend"><b style="border-color:#137a72">Original IU</b><b style="border-color:#8257b0">Selected comparison</b><span>Gold shading: annotated error step</span></div><svg id="trajectory" viewBox="0 0 1100 290" role="img" aria-label="Frozen risk score by token position, with official step boundaries"></svg><p class="muted">Scores use the existing window-overlap mean. Vertical lines are official step boundaries. The end of a trace can have overlapping scoring windows; this plot preserves that averaging.</p><div class="scroll"><table><thead><tr><th>Step</th><th>Tokens</th><th>IU score</th><th>Comparison score</th><th>Original step text</th></tr></thead><tbody id="steps"></tbody></table></div>
<h2>Oracle diagnostics: these are not proposed methods</h2><div class="note warn">These columns deliberately use the true labels. They isolate bottlenecks while holding another component fixed. They are not achievable-performance forecasts and must not be shown as experimental gains.</div><div class="scroll"><table><thead><tr><th>Fusion</th><th>Actual PB</th><th>Perfect gate, same peak</th><th>Perfect locator, same gate</th><th>Perfect tied-top choice, same gate</th></tr></thead><tbody>'''+''.join(oracle)+'''</tbody></table></div>
<h2>Decision for the next short experiment</h2><p>Keep IU-PCR, Joint graph100 and the simple permuted-graph control. Boundary tie-breaking alone has limited measured headroom. Before adding another feature family or widening lambda, inspect whether the existing single-pass token confidence contains correctness evidence that the uncertainty-based fusion suppresses. Audit older implementations, then freeze one supporting feature or readout change with the same fusion and simple controls. No candidate is selected by this audit.</p>
<details><summary>All98 corrected historical anchors</summary><p>PRMB rows with different coverage are not a matched leaderboard. The score contract and label correction are in the linked bridge report.</p><div class="scroll"><table><thead><tr><th>Method</th><th>PRMB coverage</th><th>Corrected PRMB AUC</th><th>PB F1</th></tr></thead><tbody>'''+''.join(anchors)+'''</tbody></table></div></details>
<h2>Review and limits</h2><p>Review re-read the original metadata and checked110 raw text/token/span/label joins,330 primitive streams,1,760 alternate score projections,98 independent metrics,48 oracle metrics,16 native summaries and6 transition bundles. It passed. This is same-session review using a shared tokenizer, metadata reader and frozen fusion scores, with independent downstream alignment/metric calculations.</p><p>This does not verify the original model's logit-position slice or every top-K-derived feature. Sixteen separator tokens legitimately fall outside official steps. These are development answers; corrected historical refits, full comparators, named auxiliary methods, untouched confirmation and24-cell transfer remain open.</p><ul><li><a href="AUDIT.json">Complete diagnostic records</a></li><li><a href="REVIEW.json">Review evidence</a></li><li><a href="MANIFEST.json">Frozen inputs and scope</a></li><li><a href="REPORT.md">Text report</a></li><li><a href="../localization_prm_label_audit_v1/REPORT.html">Corrected benchmark bridge</a></li></ul></main><script id="data" type="application/json">'''+blob+'''</script><script>
const payload=JSON.parse(document.getElementById('data').textContent), rows=payload.records;
const el=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
function option(value,label){const x=document.createElement('option');x.value=value;x.textContent=label;return x;}
function chosenRow(){return rows.find(r=>r.uid===el('answer').value);}
function isGold(r,i){return Array.isArray(r.target)?r.target[i]===1:r.target===i;}
function tokenValues(q,n){const sums=Array(n).fill(0),counts=Array(n).fill(0);q.window_risk.forEach((v,j)=>{for(let t=q.window_starts[j];t<q.window_ends[j];t++){sums[t]+=v;counts[t]++;}});return sums.map((v,t)=>v/counts[t]);}
function sv(tag,attrs,text){const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;el('trajectory').append(n);return n;}
function render(){const r=chosenRow();if(!r)return;const arm=el('method').value,u=r.methods.dual__iu,q=r.methods[arm];el('question').textContent=r.question;
const pred=x=>x.prediction===-1?'no error':'step '+(x.prediction+1);el('caseinfo').textContent=r.row_id+' | '+r.cell+' | '+(Array.isArray(r.target)?'PRMB annotated steps: '+r.target.flatMap((x,i)=>x?[i+1]:[]).join(', '):r.target===-1?'Gold: no error':'Gold first error: step '+(r.target+1))+' | IU: '+pred(u)+'; comparison: '+pred(q)+' | GMM BIC advantage for two components: IU '+u.bic_advantage_two.toFixed(2)+', comparison '+q.bic_advantage_two.toFixed(2);
el('steps').replaceChildren();r.text_steps.forEach((text,i)=>{const tr=document.createElement('tr');if(isGold(r,i))tr.classList.add('gold');if(i===q.peak)tr.classList.add('chosen');[i+1,r.spans[i][1]-r.spans[i][0],u.step_scores[i].toFixed(3),q.step_scores[i].toFixed(3),text].forEach((v,j)=>{const td=document.createElement('td');td.textContent=v;if(j===4)td.className='steptext';tr.append(td);});el('steps').append(tr);});
el('trajectory').replaceChildren();const uv=tokenValues(u,r.tokens),qv=tokenValues(q,r.tokens),lo=Math.min(...uv,...qv),hi=Math.max(...uv,...qv),x=t=>55+1000*t/r.tokens,y=v=>240-200*(v-lo)/Math.max(hi-lo,1e-9);
r.spans.forEach(([a,b],i)=>{if(isGold(r,i))sv('rect',{x:x(a),y:25,width:x(b)-x(a),height:225,fill:'#fff0d0'});sv('line',{x1:x(a),x2:x(a),y1:25,y2:250,stroke:'#c6d0d9','stroke-width':1});if(r.steps<=25)sv('text',{x:x((a+b)/2),y:270,'text-anchor':'middle','font-size':12},i+1);});
[[uv,'#137a72'],[qv,'#8257b0']].forEach(([values,color])=>{const points=values.flatMap((v,t)=>[[x(t),y(v)],[x(t+1),y(v)]]).map(p=>p.map(v=>v.toFixed(2)).join(',')).join(' ');sv('polyline',{points,fill:'none',stroke:color,'stroke-width':1.8});});sv('text',{x:5,y:35,'font-size':12},hi.toFixed(2));sv('text',{x:5,y:245,'font-size':12},lo.toFixed(2));sv('text',{x:550,y:287,'font-size':13,'text-anchor':'middle'},'Official step numbers; horizontal distance follows token position');}
function filter(){const cat=el('category').value,ids=cat==='all'?null:new Set(payload.cases[cat].eligible_ids);el('answer').replaceChildren();rows.filter(r=>ids===null||ids.has(r.uid)).forEach(r=>el('answer').append(option(r.uid,r.cell+' · '+r.row_id)));if(el('answer').options.length)el('answer').selectedIndex=0;render();}
Object.keys(rows[0].methods).forEach(a=>el('method').append(option(a,a)));el('method').value='dual__cond100_graph010';el('category').addEventListener('change',filter);el('answer').addEventListener('change',render);el('method').addEventListener('change',render);filter();
</script></body></html>'''
    (OUT/'REPORT.html').write_text(html,encoding='utf-8')
    save(OUT/'REPORT_PROVENANCE.json',{'report_sha256':sha(OUT/'REPORT.html'),'renderer_sha256':sha(__file__),
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('AUDIT.json','REVIEW.json','MANIFEST.json')},'facts':facts,
        'embedded_answers':len(data),'method_choices':len(d.ARMS),'historical_anchor_rows':len(metrics)})
    print('Rendered110-answer explorer,16 methods and98 historical anchors.',flush=True)


if __name__=='__main__':main()
