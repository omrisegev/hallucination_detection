"""Advisor-readable label correction report with complete historical bridges."""
from html import escape
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('label_report_driver',ROOT/'scripts/repair_localization_prm_labels_v3.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
OUT=d.OUT
load,save,sha=d.load,d.save,d.sha


def fmt(x,n=5):return 'unavailable' if x is None else f'{x:.{n}f}'


def impact_inventory():
    labels={r['row_id']:r for r in load(OUT/'LABEL_AUDIT.json')['rows']};records=[]
    for path in sorted((ROOT/'results').glob('*/EVALUATION*.json')):
        e=load(path);rows=e.get('rows',[])
        if not isinstance(rows,list):continue
        summary=dict(path=str(path),sha256=sha(path),prm_rows=0,old_only=0,corrected_only=0,
                     same_under_both=0,other_contract=0,missing_raw_id=0)
        for r in rows:
            if not isinstance(r,dict) or not str(r.get('cell','')).startswith('prm'):continue
            summary['prm_rows']+=1;gold=labels.get(r.get('row_id'))
            if gold is None:summary['missing_raw_id']+=1;continue
            old=r.get('target')==gold['previous_flags'];new=r.get('target')==gold['corrected_flags']
            field='same_under_both' if old and new else 'old_only' if old else 'corrected_only' if new else 'other_contract'
            summary[field]+=1
        if summary['prm_rows']:records.append(summary)
    result={'scope':'Only results/*/EVALUATION*.json with row-level cell/row_id/target fields; not a complete project/Claude/Drive exposure inventory.',
            'rows':records,'label_audit_sha256':sha(OUT/'LABEL_AUDIT.json')}
    save(OUT/'IMPACT_INVENTORY.json',result)
    return result


def main():
    review=load(OUT/'REVIEW.json');assert review['status']=='PASS'
    for p,h in review['hashes'].items():assert sha(p)==h,p
    audit=load(OUT/'LABEL_AUDIT.json');count=audit['counts'];impact=impact_inventory()
    data={name:load(OUT/(name+'_EVALUATION_V3.json')) for name in d.COHORTS}
    contrasts={name:load(OUT/(name+'_CONTRASTS_V3.json')) for name in d.COHORTS}
    current=data['current110'];metric=current['metrics'];iu=metric['dual__iu'];graph=metric['dual__cond100_graph010']
    augmented={a:m for a,m in metric.items() if a.startswith(('ar1__','last__','ema32__'))}
    below_iu=sum(m['prm']['auroc']<iu['prm']['auroc'] and m['pb']['macro_f1']<iu['pb']['macro_f1'] for m in augmented.values())
    below_graph=sum(m['prm']['auroc']<graph['prm']['auroc'] and m['pb']['macro_f1']<graph['pb']['macro_f1'] for m in augmented.values())
    assert len(augmented)==21 and below_iu==20 and below_graph==9
    heads=['dual__equal','dual__iu','dual__cond100','dual__cond100_graph010','dual__equal_graph_perm',
           'ar1__iu','ar1__graph010','last__equal_graph_perm']
    title='A one-step label error changes our PRMBench comparison'
    lines=['# '+title,'','Step313. Corrected labels; exactly the same fusion scores. Review PASS.','',
        f'{count["changed_answers"]:,}/{count["answers"]:,} cached answers have changed target arrays; {count["changed_step_labels"]:,}/{count["steps"]:,} step labels change. The old rule omitted valid final-step errors in some answers; {count["old_all_correct_new_has_error"]} previously all-correct arrays now contain an error. All eight PB label files are unchanged.','',
        'The raw annotations count steps from 1. Claude v2 wrote flags[step] instead of flags[step-1]. Recent Codex runs inherited that NPZ. The official evaluator and our existing port subtract one. Earlier review checked the derived NPZ, which missed this contract error.','',
        'Official source: [PRMBench task evaluator](https://github.com/ssmisya/PRMBench/blob/main/mr_eval/tasks/prmtest_classified/task.py).','',
        '## Current110 — same 24 PRMB and 86 PB answers','',
        '| Method | Previous PRMB (superseded) | Corrected PRMB | Corrected within-answer AUC | PB F1 unchanged |',
        '|---|---:|---:|---:|---:|']
    for arm in heads:
        m=metric[arm];old=current['previous_metrics'][arm]
        lines.append(f'| {arm} | {fmt(old["prm"]["auroc"])} | {fmt(m["prm"]["auroc"])} | {fmt(m["prm"]["within_answer_auc"])} | {100*m["pb"]["macro_f1"]:.2f}% |')
    lines += ['', '## What changes in the research conclusion','',
        '- Joint graph100 is no longer nearly tied with IU on PRMB: 0.65545 versus 0.68131. Their paired difference interval still includes zero; no proven two-task winner.',
        '- The original permuted-graph equal-weight control reaches 0.69226 / 31.32%, above IU on both point estimates. This is a control, not evidence that meaningful graph structure or learned Joint weights caused a gain. Do not hide it or promote a label-selected winner.',
        '- The old claim that all21 augmented recipes trail both anchors on both points is withdrawn: 20/21 trail IU and 9/21 trail graph100. All21 still have lower PB than both anchors. Last+equal+permuted graph has PRMB0.69733 but PB25.45%.',
        '- AR+IU PRMB regression is no longer resolved by its exploratory interval. The AR+Joint-graph PB regression remains because PB is unchanged. Fit-validity and group-count observations remain valid.',
        '- PRMB pooled AUROC remains a project endpoint, not the official PRMB category score. All123 method/cohort bundles and 207 registered comparisons are retained. Two cohorts stay separate; this is adaptive development, not untouched confirmation.','',
        '## Next action','',
        'Use RELEASE_V3.json for every new label read, while retaining v2 source groups/folds and the original graph seed namespace. Resume the interrupted raw-text/peak audit on this contract, then choose a bounded fusion/readout experiment. Other score sets need bridges before comparison. Claude multi-answer fitting/selection needs new labels AND new source-group folds; this bridge cannot repair already trained or selected pipelines. Full comparator coverage, named auxiliary methods, untouched confirmation and historical24 transfer remain open.','',
        '## Review','',
        f'Five contract tests. Direct raw-pickle and official-port checks on all6969 rows, eight PB file identities, 168 unchanged row-field bundles, {review["counts"]["exact_method_prediction_score_replays"]} score/prediction replays, 123 independent metric bundles and 207 paired point/scope checks. Five explicit1000-draw bootstrap checks. Same-session review with shared metadata reader/official port; no external reviewer.','',
        'The original forensics run deliberately stopped on the mismatch. Its six geometry tests and 110-row raw extraction completed; its full alignment/peak AUDIT did not. No complete alignment verdict is claimed.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    def table(cohort):
        e=data[cohort];parts=[];nprm=sum(r['cell'].startswith('prm') for r in e['rows'])
        for arm,m in e['metrics'].items():
            old=e['previous_metrics'][arm];coverage=sum(r['valid'][arm] for r in e['rows'])
            parts.append('<tr><td>'+escape(arm)+'</td><td>'+f'{coverage}/{len(e["rows"])}'+'</td><td>'+f'{m["prm"]["answers"]}/{nprm}'+'</td><td>'+fmt(old['prm']['auroc'])+'</td><td>'+fmt(m['prm']['auroc'])+'</td><td>'+fmt(m['prm']['within_answer_auc'])+'</td><td>'+f'{100*m["pb"]["macro_f1"]:.2f}%'+'</td></tr>')
        return '<table><thead><tr><th>Method</th><th>Fit coverage</th><th>PRMB answers</th><th>Old PRMB</th><th>Corrected PRMB</th><th>Within-answer</th><th>PB F1, unchanged</th></tr></thead><tbody>'+''.join(parts)+'</tbody></table>'
    def pairs_table(cohort):
        rows=[]
        for p in contrasts[cohort]['pairs'].values():
            u=p['uncertainty'];ci=u['prm_common_valid_ci95'];pb=u['pb_all_population_ci95']
            diff=p['left_prm']['auroc']-p['right_prm']['auroc'] if p['left_prm']['auroc'] is not None and p['right_prm']['auroc'] is not None else None
            rows.append('<tr><td>'+escape(d.key(p))+'</td><td>'+fmt(diff)+'</td><td>'+escape(str([round(v,5) for v in ci]) if ci else 'undefined')+'</td><td>'+escape(str([round(100*v,3) for v in pb]) if pb else 'undefined')+'</td></tr>')
        return '<table><thead><tr><th>Left minus right [scope]</th><th>PRMB difference</th><th>PRMB 95% interval</th><th>PB interval, percentage points</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    bars=[]
    for j,arm in enumerate(['dual__iu','dual__cond100_graph010','ar1__iu','ar1__graph010']):
        old=current['previous_metrics'][arm]['prm']['auroc'];new=metric[arm]['prm']['auroc'];y=40+j*75
        bars.append(f'<text x="10" y="{y+5}" font-size="14">{escape(arm)}</text><rect x="280" y="{y-15}" width="{old*500:.2f}" height="15" fill="#a3adba"/><rect x="280" y="{y+7}" width="{new*500:.2f}" height="15" fill="#147d69"/><text x="{285+new*500:.2f}" y="{y+20}" font-size="14">{new:.3f}</text>')
    svg='<svg viewBox="0 0 840 345" role="img" aria-label="Old and corrected PRMB AUC on a zero to one scale"><text x="280" y="15" font-size="13">Gray: old labels · Green: corrected labels · scale 0 to 1</text>'+''.join(bars)+'<line x1="280" y1="330" x2="780" y2="330" stroke="#555"/><text x="280" y="344">0</text><text x="765" y="344">1</text></svg>'
    example=next(r for r in load(ROOT/'results/fusion_localization_forensics_v1/RAW_METADATA.json')['rows'] if r['uid']=='prmbench_qwen3_8b__04382527b670caf7')
    case='<ol>'+''.join('<li class="'+('gold' if i==7 else 'oldlabel' if i==8 else '')+'">'+escape(t)+('<br><strong>Actual annotated error: step 8.</strong>' if i==7 else '<br><strong>Old array incorrectly marked this step.</strong>' if i==8 else '')+'</li>' for i,t in enumerate(example['text_steps']))+'</ol>'
    prose=''.join('<p>'+escape(x)+'</p>' for x in lines[lines.index('## What changes in the research conclusion')+2:lines.index('## Next action')] if x)
    links=['RELEASE_V3.json','LABEL_AUDIT.json','current110_EVALUATION_V3.json','original58_EVALUATION_V3.json',
           'current110_CONTRASTS_V3.json','original58_CONTRASTS_V3.json','IMPACT_INVENTORY.json','REVIEW.json','TESTS.json','REPORT.md']
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRMB label correction · Fusion research</title>
<style>body{font:17px/1.6 system-ui,sans-serif;color:#203047;background:#f4f6f9;margin:0}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:2.1rem;line-height:1.2}h2{margin-top:2em}.banner{background:#fff0d5;border-left:6px solid #c97611;padding:20px}.cards{display:flex;gap:12px;flex-wrap:wrap}.cards div{background:white;flex:1;min-width:180px;padding:18px;border-radius:10px}.cards b{display:block;font-size:1.8rem}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:9px;border-bottom:1px solid #d8dee6;text-align:left}th{background:#e3eaf2;position:sticky;top:0}td:first-child{overflow-wrap:anywhere}.scroll{overflow:auto;max-height:600px}.gold{background:#e1f3e9;border-left:4px solid #147d69}.oldlabel{background:#fff0d5;border-left:4px solid #c97611}li{padding:8px}svg{max-width:100%;background:white}button,input{font:inherit;padding:8px;margin:5px}button{cursor:pointer}a{color:#166798}code{background:#e3eaf2;padding:2px}details{margin:20px 0}summary{cursor:pointer;font-weight:bold}.muted{color:#596575}.hide{display:none}</style>
<main><p>IU-PCR / Joint L-SML · Step 313 · September 7, 2026</p><h1>'''+title+'''</h1><div class="banner"><strong>This is a correction, not a new algorithm gain.</strong> The recent PRMB labels were shifted by one step. Old PRMB numbers are superseded. We rebuilt the labels and evaluated exactly the same frozen fusion scores. ProcessBench stayed unchanged.</div>
<div class="cards"><div><b>6,035 / 6,969</b>answers with changed target arrays</div><div><b>15,147 / 94,203</b>step labels change</div><div><b>123</b>method/cohort metric bundles reviewed</div><div><b>207</b>registered comparisons recomputed</div></div>
<h2>What went wrong</h2><p>The raw annotation <code>error_steps = [8]</code> means the eighth step. A Python array counts from zero, so the flag belongs at <code>flags[7]</code>. Claude v2 wrote <code>flags[8]</code>. Codex inherited this label file. Earlier reviews checked the derived label file; they did not independently verify this conversion against raw annotations.</p><p>The <a href="https://github.com/ssmisya/PRMBench/blob/main/mr_eval/tasks/prmtest_classified/task.py">official PRMBench evaluator</a> subtracts one, as does our existing metric port. Out-of-range annotations remain inert. We did not invent or re-annotate labels.</p><details><summary>A real answer: the error is in step 8, not step 9</summary>'''+case+'''</details>
<h2>Same scores, corrected measurement</h2><p>These 110 development answers contain 24 PRMB and 86 PB answers. Each answer fits its own fusion; this is not a pooled training fit.</p>'''+svg+prose+'''
<h2>Complete comparison tables</h2><p>Each cohort stays separate. Different fit coverage must not be ranked as if it used the same population. Old PRMB is shown only to explain the correction; PB is the official subset-macro harmonic score used by this project.</p><label>Filter method or comparison: <input id="filter" placeholder="e.g. dual__iu or graph" type="search"></label>
<details open><summary>Current110: all 98 methods</summary><div class="scroll searchable">'''+table('current110')+'''</div></details><details><summary>Original58: all 25 methods</summary><div class="scroll searchable">'''+table('original58')+'''</div></details>
<details><summary>Current110: all 175 registered comparisons</summary><p>Unadjusted exploratory 95% source-group intervals, 1,000 draws.</p><div class="scroll searchable">'''+pairs_table('current110')+'''</div></details><details><summary>Original58: all 32 registered comparisons</summary><div class="scroll searchable">'''+pairs_table('original58')+'''</div></details>
<h2>What is complete, and what still needs work</h2><p>Five label-contract tests and the raw-source review pass. All 6,969 labels match the existing official evaluator port. Eight PB label files are unchanged. Review checked 168 row bundles, 11,594 exact method-score/prediction records, 123 metric bundles, 207 comparisons, and five explicit 1,000-draw bootstrap reconstructions. This was same-session review with independent metric calculations, not a second researcher.</p><p>Use <code>RELEASE_V3.json</code> for new work. Source-question groups and folds remain v2; the original graph random-seed namespace stays fixed. Old files remain intact for audit. Other historical score sets need their own bridge before comparison. Multi-answer models selected or trained under old labels/folds require refitting.</p><p>The text/peak forensics correctly stopped at this mismatch. Its full alignment and error-geometry analysis remains unfinished. Resume it on v3 before choosing another fusion/readout addition. The broader comparator, named-method, untouched confirmation and historical 24-cell transfer requirements remain open.</p>
<h2>Evidence files</h2><ul>'''+''.join('<li><a href="'+n+'">'+n+'</a></li>' for n in links)+'''</ul><p class="muted">Standalone local report. No external scripts or tracking. PRMB pooled AUROC is our project endpoint, not a reproduction of the official category-level PRMB score.</p></main>
<script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('.searchable tbody tr').forEach(r=>r.hidden=!r.cells[0].textContent.toLowerCase().includes(q));});</script></html>'''
    (OUT/'REPORT.html').write_text(html,encoding='utf-8')
    save(OUT/'REPORT_PROVENANCE.json',{'report_sha256':sha(OUT/'REPORT.html'),'renderer_sha256':sha(__file__),
        'source_hashes':{str(OUT/n):sha(OUT/n) for n in ['LABEL_AUDIT.json','REVIEW.json','IMPACT_INVENTORY.json']+[c+s for c in d.COHORTS for s in ('_EVALUATION_V3.json','_CONTRASTS_V3.json')]},
        'augmented_counts':{'below_iu_both':below_iu,'below_graph_both':below_graph,'total':21}})
    print('Rendered both cohorts, all123 methods /207 contrasts.',flush=True)


if __name__=='__main__':main()
