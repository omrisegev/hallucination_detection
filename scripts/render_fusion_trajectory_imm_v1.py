"""Full reviewed trajectory comparison and explicit post-evaluation gate audit."""
from html import escape
import importlib.util
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('trajectory_report_driver',ROOT/'scripts/run_fusion_trajectory_imm_v1.py')
d=importlib.util.module_from_spec(s);s.loader.exec_module(d);OUT=d.OUT
def number(x,digits=5):return 'n/a' if x is None else f'{x:.{digits}f}'
def percent(x):return 'n/a' if x is None else f'{100*x:.2f}%'
def interval(x,scale=1):return 'n/a' if x is None else '['+', '.join(f'{scale*v:+.4f}' for v in x)+']'
def table(headers,rows,identity):
    return '<div class="scroll"><table id="'+identity+'"><thead><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    m=d.verify();e,c,r,dg,g,f=[d.load(OUT/n) for n in ('EVALUATION.json','CONTRASTS.json','REVIEW.json','DIAGNOSTICS.json','GATE_AUDIT.json','SCORES_FROZEN.json')]
    assert r['status']=='PASS' and g['status']=='POST_EVALUATION_DIAGNOSTIC_REVIEWED'
    for obj in (r,g):
        for group in ('hashes','dependencies'):
            for p,h in obj[group].items():assert d.sha(p)==h,p
    metric=e['metrics'];names=['dual__iu','dual__cond100_graph010','sample_risk_top__equal_graph_perm']+m['new_arms'];summary=[]
    for name in names:
        x=metric[name];o=dg['pb_outcomes'][name]
        summary.append([name,number(x['prm']['auroc']),number(x['prm']['within_answer_auc']),percent(x['pb']['macro_f1']),
            percent(x['pb_common_iu_gate']['macro_f1']),o['clean_correct'],o['error_exact'],o['raw_peak_exact']])
    plot_names=['dual__iu','dual__cond100_graph010','sample_risk_top__equal_graph_perm','traj_iu_joint_graph__mean',
                'traj_iu_joint_graph__gls','traj_iu_joint_graph__hold','traj_iu_joint_graph__imm','traj_iu_joint_graph__imm_permuted','traj_iu__imm','traj_equal_perm__imm']
    plot_labels=['IU reference','Joint graph reference','Risk equal + permuted graph','IU + Joint: mean','IU + Joint: GLS','IU + Joint: GLS hold',
                 'IU + Joint: IMM','IU + Joint: shuffled-time IMM','Single IU: IMM','Equal + permuted graph: IMM']
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,task,field,scale,title in [(axes[0],'prm','auroc',1,'PRMB pooled AUROC'),(axes[1],'pb','macro_f1',100,'ProcessBench benchmark score (%)')]:
        values=[metric[n][task][field]*scale for n in plot_names]
        ax.scatter(values,np.arange(len(values)),s=55,color=['#536c80']*3+['#15668b']*3+['#c35827']*4)
        ax.set_yticks(range(len(values)),plot_labels if ax is axes[0] else ['']*len(values));ax.invert_yaxis()
        ax.grid(axis='x',alpha=.2);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Current110 / corrected v3 labels — descriptive points, not confidence intervals',fontsize=12)
    for ext in ('svg','png'):fig.savefig(OUT/('trajectory_points.'+ext),dpi=175,bbox_inches='tight')
    plt.close(fig)
    combos=g['exchanges']['iu_joint_graph'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    labels=[x['peak'].split('__')[-1]+' peak / '+x['gate'].split('__')[-1]+' gate' for x in combos]
    for ax,field,den,title in [(axes[0],'clean_correct',33,'Clean answers correctly accepted'),(axes[1],'error_exact',53,'Exact first-error decisions')]:
        values=[x[field] for x in combos];ax.barh(range(4),values,color=['#15668b','#d69755','#77a6bc','#ba562d'])
        ax.set_yticks(range(4),labels if ax is axes[0] else ['']*4);ax.invert_yaxis();ax.set_xlim(0,den);ax.set_xlabel('Correct answers / '+str(den));ax.set_title(title)
        for y,v in enumerate(values):ax.text(v+.4,y,str(v),va='center')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Post-evaluation component exchanges — diagnostics, not registered candidates',fontsize=12)
    for ext in ('svg','png'):fig.savefig(OUT/('gate_peak_exchange.'+ext),dpi=175,bbox_inches='tight')
    plt.close(fig)
    contrast_rows=[]
    for p in c['pairs'].values():
        u=p['uncertainty'];a,b=p['left_prm']['auroc'],p['right_prm']['auroc'];ap,bp=p['left_pb']['macro_f1'],p['right_pb']['macro_f1']
        contrast_rows.append([p['left']+' minus '+p['right'],p['scope'],len(p['selected_ids']),number(a-b) if a is not None and b is not None else 'n/a',
            interval(u['prm_common_valid_ci95']),number(100*(ap-bp),3) if ap is not None and bp is not None else 'n/a',
            interval(u['pb_all_population_ci95'],100),interval(u['prm_within_answer_common_valid_ci95']),u['pb_all_population_valid_draws']])
    exchanges=[[family,x['peak'].split('__')[-1],x['gate'].split('__')[-1],x['clean_correct'],x['error_exact'],percent(x['pb'])] for family,values in g['exchanges'].items() for x in values]
    serial=[[x['family'],x['readout'],'clean' if x['clean'] else 'error',x['answers'],number(x['median_lag1'],3),number(x['median_bic_advantage_two'],2),x['gate_open']] for x in g['summaries']]
    models=[]
    for family in list(d.PAIRS)+list(d.SINGLES):
        chosen=[x for x in dg['models'] if x['family']==family];corr=[x['source_correlation'] for x in chosen if x['source_correlation'] is not None]
        models.append([family,len(chosen),r['duplicate_collapses'].get(family,0),r['inherited_fallbacks'].get(family,0),
            number(float(np.median(corr)),3) if corr else 'single input',number(float(np.median([x['condition'] for x in chosen])),2),
            sum(any(v<0 for v in x['weights']) for x in chosen)])
    transitions=[[p]+[x[k] for k in ('gained','lost','peak_changed','gate_changed')] for p,x in dg['pb_transitions'].items()]
    payload=json.dumps(dict(arms=m['arms'],new_arms=m['new_arms'],metrics=e['metrics']),ensure_ascii=True,allow_nan=False).replace('</','<\\/')
    links=[('Frozen protocol','../../docs/experiments/FUSION_TRAJECTORY_IMM_V1.md'),('Implementation history audit','../../docs/reviews/trajectory_fusion_history_audit_2026-09-07.md'),
        ('Fusion code','../../spectral_utils/fusion_trajectory_imm.py'),('Scoring driver','../../scripts/run_fusion_trajectory_imm_v1.py'),('Review code','../../scripts/review_fusion_trajectory_imm_v1.py'),
        ('Review evidence','REVIEW.json'),('Registered evaluation','EVALUATION.json'),('Paired comparisons','CONTRASTS.json'),('Diagnostics','DIAGNOSTICS.json'),
        ('Post-evaluation gate audit','GATE_AUDIT.json'),('Previous sampling report','../fusion_sampling_replication_v1/REPORT.html'),('Corrected older58 report','../localization_history_bridge_v3/REPORT.html')]
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IU / Joint trajectories and IMM — Step318</title>
<style>body{margin:0;background:#f4f7fa;color:#1a2c3a;font:17px/1.65 system-ui,sans-serif}main{max-width:1230px;margin:auto;padding:32px 24px 80px}h1{line-height:1.2}h2{margin-top:2rem}a{color:#145e82}.note{background:white;border-left:5px solid #bd652d;padding:14px 20px;margin:20px 0}.flow{padding:20px;background:#e5f0f5;border-radius:10px;text-align:center}.muted{color:#526776}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dce4eb;text-align:left;vertical-align:top}th{background:#e5eef4}figure{margin:25px 0}img{width:100%;height:auto}input,select{font:inherit;padding:7px;margin:8px}code{font-size:.87em}li{margin:7px 0}</style></head><body><main>
<p class="muted">7 September 2026 · Same-answer offline fusion · Current110 development cohort · Review PASS</p>
<h1>Combining IU and Joint trajectories does not yet give a reliable improvement.</h1>
<p>GLS has a small positive point difference from IU on both benchmarks, with wide uncertainty. The tested IMM level readout weakens ProcessBench. Mean/GLS combinations can change the peak, but they remain pointwise combinations of existing feature weights. Only the IMM stage here models chronological state evolution. All additions support our existing fusion method.</p>
<div class="flow">One answer → N windows × P features → IU-PCR and Joint L-SML → full-trajectory mean / GLS → optional IMM → step peak and no-error decision</div>
<h2>What was actually tested</h2><p>Retain the same24 PRMB and86 PB answers, v3 labels/v2 source groups, original81 moment/29 context bank route, width8 windows and graph seeds. No feature/Joint refit or new inference. Five paired families, each with mean/GLS/hold/IMM; three single-source families with hold/IMM; one shuffled-time primary IMM. This gives27 additions and149 unchanged anchors:176 entries, not176 new algorithms.</p>
<p>Both inputs observe a declared common scalar state with correlated noise. GLS accounts for their estimated dependence; its weights can be negative. The noise covariance is a first-difference heuristic fitted from this answer, not identified semantic-error noise. Two IMM modes describe slow and faster change, with fixed Q=(.01R,R) and transition self-probability.95. They are not labelled correct/error states.</p>
<p>The scalar GLS statistic is mathematically sufficient for this common-level, shared-covariance vector model. A direct vector IMM replay verifies that equivalence. Exact positive-affine duplicates collapse to one observation; opposite duplicates fail. Three original Joint fallbacks are inherited, and their IU/Joint pairs collapse. All27 final recipes have110 valid outputs. No new failure fallback is introduced.</p>
<p>IMM updates only nonoverlapping fit windows; the optional overlapping end window receives the last filtered value. The matched hold control uses that same tail rule. Output normalization uses all original fit rows. Source fitting and normalization use the full current answer, so this is not causal online detection.</p>
<figure><img src="trajectory_points.png" alt="Matched PRMB and ProcessBench point estimates for references and trajectory combinations"><figcaption>Both endpoints use identical development cases. All registered paired intervals appear below.</figcaption></figure>
<h2>All new results and mandatory headline references</h2><p>PRMB pooled AUROC compares steps across answers; within-answer AUC reports local ranking. PB is the macro across four subsets of the harmonic mean of clean-answer accuracy and exact first-error accuracy. It is not overall accuracy. The fixed-IU gate column uses the original moment-IU binary decision and is diagnostic only.</p>'''
    html+=table(['Method','PRMB pooled AUC','Within-answer AUC','PB native','PB fixed-IU: diagnostic','Clean correct /33','Error exact /53','Raw exact peaks /53'],summary,'headline')
    html+='''<div class="note"><strong>No promotion.</strong> Primary GLS versus IU: PRMB difference +.00208,95% interval[-.02612,+.02882]; PB +.1885pp, interval[-8.7608,+9.3277]pp. Its within-answer AUC is .79012 versus .76881, also uncertain. Primary IMM gives .69325/16.69%, versus static hold .68043/30.35%. The strong risk equal+permuted graph reference remains .76842/33.92%; its prior PB improvement interval includes zero. None is a confirmed fusion winner.</div>
<h2>Separate the peak from the no-error decision</h2><p>The primary IMM gains4 correct PB decisions and loses13 versus its hold control. Raw exact peaks change18→17, while final exact error decisions change12→7 and correct clean decisions15→11. A small change in raw peak counts can hide different cases; we therefore exchange the two fixed components in both directions below.</p>
<p>These four combinations per family were computed after evaluation to diagnose the failure. They are not newly registered candidates, label-perfect choices, or a causal decomposition. Every answer uses the same named peak and gate components. No component is selected using its target.</p>
<figure><img src="gate_peak_exchange.png" alt="Four fixed peak and gate exchanges, showing separate clean and exact first-error counts"><figcaption>Primary IU+Joint graph family. The complete eight-family exchange table follows.</figcaption></figure>'''
    html+=table(['Family','Peak source','Gate source','Clean correct /33','Error exact /53','PB diagnostic'],exchanges,'gate-exchanges')
    html+='''<h2>Serial dependence and mixture evidence: descriptive audit</h2><p>The current no-error gate asks whether a two-Gaussian mixture has lower BIC than one Gaussian. That is evidence about score shape under this recipe, not a proof of a reasoning error. Filtering changes serial dependence as well as the marginal score distribution. The table reports lag1 correlation and BIC evidence without fitting or selecting a new gate. It does not establish a calibrated error null or justify substituting an effective-sample-size formula blindly.</p>'''
    html+=table(['Family','Readout','Answer type','Answers','Median lag1','Median BIC1−BIC2','Gate open'],serial,'serial')
    html+='''<h2>Model dependence and inherited coverage</h2><p>First-difference covariance is regularized to condition≤100. Very similar source trajectories are not independent evidence. Negative GLS weights are permitted by the declared estimator; these diagnostics were not used to choose a method.</p>'''
    html+=table(['Family','Valid models','Duplicate collapse','Inherited Joint fallback','Median source correlation','Median noise condition','Any negative weight'],models,'models')
    html+='''<h2>Actual PB changes in all registered full-population comparisons</h2><p>Gains and losses count exact correct decisions. Macro PB can move differently from the total because it weights subsets and clean/error accuracies separately.</p>'''
    html+=table(['Comparison','Correct gained','Correct lost','Peak changed','Gate changed'],transitions,'transitions')
    html+='''<h2>All176 current entries</h2><p>Historical anchors are preserved exactly. Older58 experiments remain separate context, not directly comparable absolute numbers. Filter by name or show only the27 additions.</p>
<label>Scope <select id="scope"><option value="all">All176</option><option value="new">New27</option></select></label><label>Filter <input id="search" type="search" placeholder="e.g. imm or dual__iu"></label><p id="count"></p>
<div class="scroll"><table id="all-methods"><thead><tr><th>Method</th><th>PRMB answers</th><th>Pooled AUC</th><th>Within-answer AUC</th><th>PB native</th><th>Valid PB decisions</th><th>PB fixed-IU: diagnostic</th></tr></thead><tbody id="method-rows"></tbody></table></div>
<h2>All50 registered paired comparisons</h2><p>1000 source-group bootstrap draws, stratified by cell; exploratory and unadjusted for multiplicity. Native-parent-Joint scope107 excludes the original3 fallback answers; it is diagnostic and cannot replace the full110 primary scope. Undefined PB resamples are counted explicitly.</p>'''
    html+=table(['Comparison','Scope','Answers','PRMB delta','PRMB95% CI','PB delta pp','PB95% CI pp','Within-answer95% CI','Valid PB draws'],contrast_rows,'contrasts')
    html+=f'''<h2>History, review and limitations</h2><p>Scalar IMM after each individual fusion was already tested on the original58 answers (Step299; corrected results in Step316). Old IU IMM gave .61291/14.79% versus IU peak .64753/17.71%, under a retained parent gate. Step318 adds combined full trajectories and native gates on current110. This is not a new IMM invention or an actual KalmanNet test.</p>
<p>Seven tests pass. Review checks110 raw label/span joins,16390 unchanged anchor row records,880 independent R/GLS reconstructions,990 direct vector IMM replays,2970 final trajectory/step/GMM outputs,176 metric bundles and50 paired scopes/points, plus five explicit1000-draw bootstraps. Maximum reconstructed window difference{r['maximum_window_difference']:.3g}. Post-evaluation exchange audit adds{g['checks']} metric/component checks and{g['lag_replays']} lag-correlation replays.</p>
<p>Scoring{f['seconds']:.2f}s; comparisons{c['seconds']:.2f}s; successful scientific review{r['seconds']:.2f}s. These are stage times, not a model-inference saving. {escape(r['scope'])}</p>
<p>There are{len(dg['pb_first_errors_le32'])} first-error steps of≤32 tokens in this PB cohort. A broad short-error robustness claim remains untested. All examples were already exposed during project development; no untouched confirmation or published-baseline completeness is claimed.</p>
<h2>Next bounded question</h2><p>Audit the existing no-error methods and assess how the current mixture decision responds to serial dependence before adding more temporal smoothing. Preserve each fusion's actual score trajectory and peak when testing a gate change, and keep native/fixed-gate comparisons explicit. A gate must still be fitted within this answer without correctness labels; treating two modes as semantic error needs evidence.</p>
<p>Joint/graph representation work, IU improvement, both fusion axes, actual KalmanNet/LOCA/Flows adaptations, sparse/short-error sampling, corrected multi-answer refits, complete comparators, untouched confirmation and historical24 transfer remain open. No family is closed by this single fixed experiment.</p>
<h2>Evidence and code</h2><ul>'''+''.join('<li><a href="'+escape(path)+'">'+escape(label)+'</a></li>' for label,path in links)+'''</ul>
<script id="payload" type="application/json">'''+payload+'''</script><script>
const payload=JSON.parse(document.getElementById('payload').textContent);
function format(x,percent=false){return x===null?'n/a':percent?(100*x).toFixed(2)+'%':x.toFixed(5);}
function render(){const scope=document.getElementById('scope').value,query=document.getElementById('search').value.toLowerCase();
const roster=scope==='new'?payload.new_arms:payload.arms,arms=roster.filter(a=>a.toLowerCase().includes(query));
document.getElementById('method-rows').innerHTML=arms.map(arm=>{const m=payload.metrics[arm],valid=Object.values(m.pb.cells).reduce((n,c)=>n+c.valid_decisions,0);
return '<tr><td>'+arm+'</td><td>'+m.prm.answers+'</td><td>'+format(m.prm.auroc)+'</td><td>'+format(m.prm.within_answer_auc)+'</td><td>'+format(m.pb.macro_f1,true)+'</td><td>'+valid+'</td><td>'+format(m.pb_common_iu_gate.macro_f1,true)+'</td></tr>';}).join('');
document.getElementById('count').textContent=arms.length+' of '+roster.length+' entries | same110 answers';}
document.getElementById('scope').addEventListener('change',render);document.getElementById('search').addEventListener('input',render);render();
</script></main></body></html>'''
    (OUT/'REPORT.html').write_text(html,encoding='utf-8')
    md=['# Full-trajectory fusion and IMM - Step318','','Review PASS. No consistent two-task winner. Same110 development answers, corrected v3 labels/v2 groups.','',
        '| Method | PRMB AUC | Within-answer | PB native | Fixed-IU gate | Clean /33 | Exact /53 | Raw peak /53 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    md+=['| '+' | '.join(map(str,row))+' |' for row in summary]
    md+=['','27 additions +149 unchanged anchors =176 total;50 registered contrasts. All final outputs valid110;3 original Joint fallbacks inherited.',
        'GLS vsIU intervals include0 on both primary endpoints. Primary IMM loses13 correct PB decisions/gains4 versus hold. Its two modes are dynamical, not semantic labels.',
        'Post-evaluation fixed peak/gate exchanges and serial-dependence summaries are diagnostics, not new candidates. See GATE_AUDIT.json and REPORT.html.',
        f'Scoring{f["seconds"]:.2f}s; contrasts{c["seconds"]:.2f}s; review{r["seconds"]:.2f}s. Seven tests, vector IMM/math/raw-label/metric and bootstrap review PASS.',
        'Next audit the no-error decision under serially dependent fused trajectories before a bounded gate-only change. Keep all wider research obligations active.']
    (OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print('Rendered full trajectory report and two scientific figures.',flush=True)


if __name__=='__main__':main()
