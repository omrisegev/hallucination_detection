"""Visual report with all anchors, matched scopes and sampling caveats."""
from collections import Counter
from html import escape
import importlib.util
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('sampling_report_driver',ROOT/'scripts/run_fusion_sampling_replication_v1.py')
d=importlib.util.module_from_spec(s);s.loader.exec_module(d);OUT=d.OUT
SELECTOR_LABELS=dict(full='All windows',uniform='Uniform',risk_top='Entropy risk',dufs_transposed='Transposed DUFS',
                    dufs_permuted='Permuted DUFS',window_diffusion='Window diffusion')
def n(x,digits=5):return 'n/a' if x is None else f'{x:.{digits}f}'
def pct(x):return 'n/a' if x is None else f'{100*x:.2f}%'
def ci(x,scale=1):return 'n/a' if x is None else '['+', '.join(f'{v*scale:+.4f}' for v in x)+']'
def table(headers,rows,identity):
    return '<div class="scroll"><table id="'+identity+'"><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    m=d.verify();e=d.load(OUT/'EVALUATION.json');c=d.load(OUT/'CONTRASTS.json');r=d.load(OUT/'REVIEW.json');dg=d.load(OUT/'DIAGNOSTICS.json');f=d.load(OUT/'SCORES_FROZEN.json')
    assert r['status']=='PASS'
    for source in ('hashes','dependencies'):
        for p,h in r[source].items():assert d.sha(p)==h,p
    metric=e['metrics'];summary=[]
    for sel in d.SELECTORS:
        iu=metric[d.arm_name(sel,'iu')];joint=metric[d.arm_name(sel,'graph010')];eq=metric[d.arm_name(sel,'equal_graph_perm')]
        summary.append([SELECTOR_LABELS[sel],n(iu['prm']['auroc']),n(iu['prm']['within_answer_auc']),pct(iu['pb']['macro_f1']),
            n(joint['prm']['auroc']),pct(joint['pb']['macro_f1']),n(eq['prm']['auroc']),pct(eq['pb']['macro_f1']),str(r['native_joint_valid'][sel])+'/110'])
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
    colors={'iu':'#2264ac','graph010':'#cc6622','equal_graph_perm':'#487b43'}
    labels={'iu':'IU-PCR','graph010':'Joint + graph','equal_graph_perm':'Equal + permuted graph'}
    for ax,task,field,scale,title in [(axes[0],'prm','auroc',1,'PRMBench pooled AUROC'),(axes[1],'pb','macro_f1',100,'ProcessBench macro F1 (%)')]:
        for core,offset in [('iu',-.18),('graph010',0),('equal_graph_perm',.18)]:
            vals=[metric[d.arm_name(sel,core)][task][field]*scale for sel in d.SELECTORS]
            ax.scatter(vals,np.arange(6)+offset,label=labels[core],color=colors[core],s=48)
        ax.set_yticks(range(6),[SELECTOR_LABELS[x] for x in d.SELECTORS] if ax is axes[0] else ['']*6)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.08),frameon=False,ncol=1)
    fig.suptitle('Same 110 development answers - v3 labels\nDescriptive points; paired uncertainty below',fontsize=13)
    fig.savefig(OUT/'sampling_points.svg',bbox_inches='tight');fig.savefig(OUT/'sampling_points.png',dpi=175,bbox_inches='tight');plt.close(fig)
    example=next(row for row in e['rows'] if row['cell'].startswith('pb') and row['sampling']['eligible'] and row['target']!=-1)
    with np.load(OUT/'scores'/(example['uid']+'.npz'),allow_pickle=False) as a:
        fig,ax=plt.subplots(figsize=(12,3.8),layout='constrained')
        gold=example['target'];ax.axvspan(a['step_starts'][gold],a['step_ends'][gold],alpha=.15,color='#b33434',label='Official first-error step')
        for y,sel in enumerate(d.SELECTORS):
            starts=a['window_starts'][a[sel+'__selected']]
            ax.scatter(starts+4,np.full(len(starts),y),marker='|',s=100,color='#245d91')
        ax.set_yticks(range(6),[SELECTOR_LABELS[x] for x in d.SELECTORS]);ax.invert_yaxis();ax.set_xlim(0,example['tokens']);ax.set_xlabel('Token position in this answer')
        ax.set_title('Selected fitting windows; every window is still scored\nFirst eligible PB error answer in registry order (illustration)')
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.2),fontsize=9,frameon=False);ax.spines[['top','right']].set_visible(False)
        fig.savefig(OUT/'selected_windows.svg',bbox_inches='tight');fig.savefig(OUT/'selected_windows.png',dpi=175,bbox_inches='tight');plt.close(fig)
    support=[]
    for sel in d.SELECTORS:
        records=[v for v in dg['records'] if v['selector']==sel and v['eligible']]
        errors=[v for v in records if 'first_error_tokens' in v];short=[v for v in errors if v['first_error_tokens']<=32]
        stability=[x for v in records for x in v['perturbation_jaccard'] if x is not None]
        support.append([SELECTOR_LABELS[sel],len(records),n(float(np.mean([v['token_fraction'] for v in records])),3),
            len(errors),sum(v['first_error_fit_support']>0 for v in errors),sum(v['first_error_fit_support']==1 for v in errors),
            n(float(np.mean([v['first_error_fit_support'] for v in errors])),3) if errors else 'n/a',len(short),
            sum(v['first_error_fit_support']>0 for v in short),n(float(np.mean(stability)),3) if stability else 'n/a'])
    changes=[]
    for arm,cells in dg['pb_transitions'].items():
        changes.append([arm]+[sum(v[k] for v in cells.values()) for k in ('predictions_changed','peaks_changed','gained','lost')])
    aff=[]
    for x in dg['affine_diagnostic']:
        aff.append([x['core']]+[n(x[key]['auroc']) for key in ('full','sample','sample_ranking_full_scale','full_ranking_sample_scale')]+
            [n(x['full']['within_answer_auc']),n(x['sample']['within_answer_auc'])])
    comparison=[]
    for p in c['pairs'].values():
        u=p['uncertainty'];av,bv=p['left_prm']['auroc'],p['right_prm']['auroc'];ap,bp=p['left_pb']['macro_f1'],p['right_pb']['macro_f1']
        comparison.append([p['left']+' minus '+p['right'],p['scope'],len(p['selected_ids']),n(av-bv) if av is not None and bv is not None else 'n/a',
            ci(u['prm_common_valid_ci95']),n(100*(ap-bp),3) if ap is not None and bp is not None else 'n/a',ci(u['pb_all_population_ci95'],100),
            ci(u['prm_within_answer_common_valid_ci95']),u['pb_all_population_valid_draws']])
    current_head=['Selector','IU AUC','IU within-answer','IU PB','Joint graph AUC','Joint graph PB','Equal perm-graph AUC','Equal perm-graph PB','Native Joint fits']
    payload=json.dumps(dict(arms=m['arms'],metrics=e['metrics'],eligible_metrics=e['eligible_metrics']),ensure_ascii=True,allow_nan=False).replace('</','<\\/')
    links=[('Protocol','../../docs/experiments/FUSION_SAMPLING_REPLICATION_V1.md'),('Fusion code','../../spectral_utils/fusion_sampling_replication.py'),
        ('Scoring driver','../../scripts/run_fusion_sampling_replication_v1.py'),('Review code','../../scripts/review_fusion_sampling_replication_v1.py'),
        ('Review evidence','REVIEW.json'),('All metrics','EVALUATION.json'),('All paired comparisons','CONTRASTS.json'),('Diagnostics and transitions','DIAGNOSTICS.json'),
        ('Corrected old58 history','../localization_history_bridge_v3/REPORT.html'),('Previous current110 reference','../fusion_token_gap_v1/REPORT.html'),
        ('Export points (SVG)','sampling_points.svg'),('Export selected windows (SVG)','selected_windows.svg')]
    content='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sampling inside IU-PCR and Joint L-SML - Step317</title><style>
body{margin:0;background:#f2f6fa;color:#1b2c41;font:17px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px 24px 70px}h1{font-size:2.1rem;line-height:1.2}h2{margin-top:2.2rem}.card,.scroll{background:white;border:1px solid #d7e0eb;border-radius:12px;padding:18px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #e5ebf3;text-align:left;white-space:nowrap}th{background:#eaf0f7}img{max-width:100%;height:auto}.flow{display:flex;gap:12px;flex-wrap:wrap}.flow div{flex:1;min-width:165px;background:#e3edf8;border-radius:9px;padding:13px}.muted{color:#53657a;font-size:.92rem}input,select{font:inherit;padding:8px;margin:8px;border:1px solid #aebfd3;border-radius:7px}.verdict{border-left:6px solid #bd6b24}a{color:#1d5e9b}
</style></head><body><main><p class="muted">September 7, 2026 | Step317 | Development evidence | Fusion remains the method</p>
<h1>Choosing windows changes the scores.<br>It has not yet produced a better localizer on both tasks.</h1>
<div class="card verdict"><strong>Retain the original IU and Joint graph references. No candidate is promoted.</strong>
<p>Entropy-risk selection raises pooled PRMB AUROC substantially, including with equal fusion. IU and Joint graph have lower PB points than their full-grid versions. Transposed DUFS and direct window diffusion do not establish a consistent gain. A strong simple control, entropy-risk + equal + permuted graph, reaches 0.76842 PRMB /33.92% PB, but its PB improvement interval includes zero. It is a visible comparator, not evidence that learned Joint or correct graph alignment caused the gain.</p></div>
<h2>One supporting change in our existing fusion</h2>
<div class="flow"><div>One provided answer<br>existing gray-box telemetry</div><div>All eight-token windows<br>same27 features and bank</div><div>Select fitting rows<br>learn IU / Joint fusion</div><div>Score every window<br>step + no-error decision</div></div>
<p>The original81 moment /29 context bank route stays fixed. The same selected rows feed equal, IU, Joint0, Joint graph and matched graph controls. Joint uses condition100 and lambda0.1. A failed Joint model fit can fall back explicitly to selected-row IU in that same bank. A selector or graph/readout failure is not a license to switch methods silently.</p>
<p>For N original fitting windows, retain min(N,max(32,ceil(N/2))).72 answers can reduce rows;38 retain every row and replay the exact references. The selectors see only the current answer. Fusion normalization, grouping and weights use selected rows. The GMM sees scores on ALL original fitting windows, preserving the first sampling pilot's decision rule. Scoring still uses every window, including the final scoring-only window. This is offline answer-only fitting with the declared entropy orientation anchor.</p>
<p><strong>The graph operations are different:</strong> transposed DUFS gates windows while its graph nodes are feature coordinates; window diffusion builds a graph over windows; Joint's graph penalty changes fusion weights. None is automatically the other.</p>
<h2>Matched full-cohort results</h2><img src="sampling_points.png" alt="IU, Joint graph and equal permuted-graph points across six window selectors, on PRMBench and ProcessBench.">'''
    content+=table(current_head,summary,'headline')
    content+='''<p>All displayed recipes return valid final scores and decisions for all110 answers (24 PRMB/86 PB). Full coverage includes declared IU fallbacks; the native Joint column exposes this difference. PB averages four subsets' harmonic means of clean-answer accuracy and exact first-error accuracy. It is not overall accuracy. Every answer in this cache has prior development exposure.</p>
<p>Entropy-risk IU minus full IU: PRMB delta+0.07705, exploratory95% interval[+0.04689,+0.11188]; within-answer delta+0.01011, interval[-0.00325,+0.02690]; PB delta-1.811 points, interval[-7.987,+3.071]. Entropy-risk equal-permuted-graph minus its full reference: PB delta+2.598 points, interval[-3.293,+8.029]. These93 comparisons use unadjusted intervals and do not select a publication winner.</p>
<h2>How much of the PRMB jump is between-answer scaling?</h2>
<p>This diagnostic was added AFTER inspecting the points. For each answer, align the sampled score's mean and standard deviation over all original fitting windows to the full-grid score; also do the reverse with the original score. Positive affine changes preserve within-answer ranking. The resulting pooled AUCs help distinguish local ordering from between-answer score location and scale. They are diagnostics, not additional candidates or new PB predictions.</p>'''
    content+=table(['Core','Full AUC','Sampled AUC','Sample ranking, full scale','Full ranking, sample scale','Full within-answer','Sample within-answer'],aff,'affine')
    content+='''<p><strong>The large pooled jump is consistent mainly with between-answer location/scale changes.</strong> Keeping IU's original ranking but assigning the sampled answer's location and scale raises AUC from0.68131 to0.74876. Keeping sampled IU ranking but restoring the full-grid location and scale gives0.68938, rather than0.75835. For Joint graph the corresponding values are0.73585 and0.66564. The sampled within-answer improvement is only0.01011 for IU and0.00071 for Joint graph. This is not evidence for a0.758 first-error localizer.</p>
<p>Matching two moments does not fully decompose nonlinear AUROC or prove a causal explanation. Use the within-answer endpoint and exact first-error decisions beside pooled AUC.</p>
<h2>Which fitting windows survived?</h2><img src="selected_windows.png" alt="Six sampling policies plotted at their selected token positions, with the official first-error step shaded.">'''
    content+='<p class="muted">Illustration ID: '+escape(example['row_id'])+'. Chosen by first eligible error answer in the fixed registry order, not by which selector succeeds.</p>'
    content+=table(['Selector','Eligible answers','Mean token fraction','Eligible PB errors','Any error overlap','Full error covered','Mean error coverage','Errors <=32 tokens','Short errors with overlap','Perturbation Jaccard'],support,'support')
    content+='''<p>There are38 erroneous PB answers among the56 sampling-eligible PB answers, and <strong>none has a first-error step of32 tokens or fewer</strong>. Short-error retention is still untested. Entropy-risk selection has no fitting-window overlap with one of those38 error steps. Coverage here means overlap between an official error step and selected FIT windows. Unselected windows are still scored, so this is not sparse-detector recall. Perturbations resample two-token blocks within each original eight-token window. For context banks, recomputing EMA propagates those changes to later windows. Stability is descriptive, not a criterion used to pick a winner.</p>
<h2>Exact PB decisions gained and lost</h2>'''
    content+=table(['Arm vs its full-grid core','Predictions changed','Peaks changed','Correct gained','Correct lost'],changes,'transitions')
    content+='''<p>Detailed per-subset and per-answer transitions are saved in DIAGNOSTICS.json. Macro F1 can improve without increasing the total number of correct decisions if successes move between subsets or between clean/error classes.</p>
<h2>All149 entries, including107 historical anchors</h2><p>The42 selector/core entries include seven exact full-grid aliases. Only35 recipes can introduce new fits. Counts are not counts of independent new algorithms. The eligible-only view contains72 answers (16 PRMB/56 PB) and supplements the primary full110 view.</p>
<label>Population <select id="scope"><option value="metrics">Full110</option><option value="eligible_metrics">Eligible72</option></select></label>
<label>Filter methods <input id="search" type="search" placeholder="e.g. sample_risk_top or dual__iu"></label><p id="count" class="muted"></p>
<div class="scroll"><table id="all-methods"><thead><tr><th>Arm</th><th>PRMB answers</th><th>Pooled AUROC</th><th>Within-answer AUC</th><th>PB F1</th><th>PB valid decisions</th><th>Fixed-IU gate: diagnostic</th></tr></thead><tbody id="method-rows"></tbody></table></div>
<h2>All93 registered paired comparisons</h2><p>PRMB compares common valid answers. PB keeps every answer in the registered scope, including failures. The native-only scope additionally requires original and sampled Joint model fits to be valid; it cannot replace the full-population result. Intervals use1000 source-group draws.</p>'''
    content+=table(['Comparison','Scope','Answers','PRMB delta','PRMB95% CI','PB delta pp','PB95% CI pp','Within-answer95% CI','Valid PB draws'],comparison,'contrasts')
    content+=f'''<h2>What changed relative to the first sampling pilot?</h2><p>The original58 results are separate development evidence. After correcting labels, entropy-risk IU gave0.66850/19.05% versus full-grid0.64753/17.71%, with both improvement intervals including zero. Its within-answer AUC fell slightly. The current110 uses the original routed banks and condition100 references; the first pilot used only the moment bank and earlier Joint settings. Compare every selector to its matched full-grid reference within its cohort, not 19.05% to28.35% as an algorithmic improvement.</p>
<h2>Review and runtime</h2><p>Review PASS: {r['counts']['raw_label_span_joins']} raw-label/span joins, {r['counts']['selected_normalization_replays']} selected-row normalizations and IU refits, {r['counts']['independent_native_weights']} native weight projections, {r['counts']['step_output_replays']} final trajectory/output checks, {r['counts']['independent_metric_bundles']} metric bundles across full/eligible populations, and93 paired scope/point replays. Representative grouping/Joint/DUFS and selector-perturbation replays and five explicit1000-draw bootstraps pass. Maximum native risk reconstruction difference: {r['maximum_risk_difference']:.3g}.</p>
<p>Scoring took{f['seconds']:.2f}s with three CPU workers, including selector perturbations; comparisons{c['seconds']:.2f}s; review{r['seconds']:.2f}s. This is total experiment wall time, not an end-to-end speedup over full-grid fusion. Dense telemetry and features were still needed. No new model inference or Claude-worktree change was made.</p>
<p>{escape(r['scope'])}</p>
<h2>Research decision</h2><p>Do not promote graph observation selection or the current risk-sampled Joint/IU recipe. Preserve the stronger simple control. Before another sampling or lambda sweep, distinguish between-answer calibration effects from actual peak and no-error improvements. Any next change must retain fusion, both task endpoints, the same IDs, and matched equal controls. Short-error and sparse end-to-end sampling claims still need direct evidence.</p>
<p>Corrected multi-answer refits, full published comparator coverage, the named supporting tracks, untouched confirmation and eventual historical24 transfer remain open. This stage does not complete the broader research goal.</p>
<h2>Evidence and code</h2><ul>'''+''.join('<li><a href="'+escape(path)+'">'+escape(label)+'</a></li>' for label,path in links)+'''</ul>
<script id="payload" type="application/json">'''+payload+'''</script>
<script>
const payload=JSON.parse(document.getElementById('payload').textContent);
function format(x,percent=false){return x===null?'n/a':percent?(100*x).toFixed(2)+'%':x.toFixed(5);}
function render(){const scope=document.getElementById('scope').value;const query=document.getElementById('search').value.toLowerCase();
const arms=payload.arms.filter(arm=>arm.toLowerCase().includes(query));const data=payload[scope];
document.getElementById('method-rows').innerHTML=arms.map(arm=>{const m=data[arm];const valid=Object.values(m.pb.cells).reduce((n,c)=>n+c.valid_decisions,0);
return '<tr><td>'+arm+'</td><td>'+m.prm.answers+'</td><td>'+format(m.prm.auroc)+'</td><td>'+format(m.prm.within_answer_auc)+'</td><td>'+format(m.pb.macro_f1,true)+'</td><td>'+valid+'</td><td>'+format(m.pb_common_iu_gate.macro_f1,true)+'</td></tr>';}).join('');
document.getElementById('count').textContent=arms.length+' of '+payload.arms.length+' entries | '+(scope==='metrics'?'110 answers':'72 eligible answers');}
document.getElementById('scope').addEventListener('change',render);document.getElementById('search').addEventListener('input',render);render();
</script></main></body></html>'''
    (OUT/'REPORT.html').write_text(content,encoding='utf-8')
    md=['# Fixed-bank sampling replication - Step317','','Review PASS. No consistent IU/Joint improvement on both tasks. Same110 development answers, v3 labels/v2 groups.','',
        '| Selector | IU AUC | IU within | IU PB | Joint graph AUC | Joint graph PB | Equal perm AUC | Equal perm PB | Native Joint |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    md+=['| '+' | '.join(row)+' |' for row in summary]
    md+=['','149 entries include107 anchors and seven full aliases. All42 final outputs cover110; native Joint counts above include the unchanged short-answer fits.',
        'Risk-IU pooled AUC improves strongly, within-answer change is small/uncertain, PB falls. Equal-permuted-graph risk sampling has higher points on both tasks, but PB improvement CI includes zero.',
        'Post-evaluation affine diagnostic, exact PB transitions, eligible-only metrics, short-error fit support and all93 comparisons are in REPORT.html / DIAGNOSTICS.json.',
        f'Scoring {f["seconds"]:.2f}s; contrasts {c["seconds"]:.2f}s; review {r["seconds"]:.2f}s. Dense features/scoring retained; no inference saving claimed.',
        'No candidate promotion or full-goal completion. Preserve anchors and inspect calibration/peak/gate attribution before another sweep.']
    (OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('Rendered HTML, markdown and two SVG/PNG scientific figures.',flush=True)


if __name__=='__main__':main()
