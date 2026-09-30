"""Render a bounded negative result with all matched corrected anchors."""
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
s=importlib.util.spec_from_file_location('gap_report_driver',ROOT/'scripts/run_fusion_token_gap_v1.py')
d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
OUT=d.OUT

LABELS={'equal':'Equal fusion','iu':'IU-PCR','joint0':'Joint, no graph','graph010':'Joint + graph',
    'graph_perm':'Joint + permuted graph','equal_graph010':'Equal + graph','equal_graph_perm':'Equal + permuted graph'}
def number(x,digits=5):return 'n/a' if x is None else f'{x:.{digits}f}'
def percent(x):return 'n/a' if x is None else f'{100*x:.2f}%'
def interval(v,scale=1):return 'n/a' if v is None else '['+', '.join(f'{scale*x:+.4f}' for x in v)+']'
def table(headers,rows,identity=''):
    return '<div class="scroll"><table'+(f' id="{identity}"' if identity else '')+'><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def main():
    manifest=d.verify();e=d.load(OUT/'EVALUATION.json');c=d.load(OUT/'CONTRASTS.json');review=d.load(OUT/'REVIEW.json')
    raw=d.load(OUT/'RAW_CONFIDENCE_AUDIT.json');frozen=d.load(OUT/'SCORES_FROZEN.json')
    assert review['status']=='PASS' and c['status']=='COMPLETE'
    for p,h in review['hashes'].items():assert d.sha(p)==h,p
    for p,h in review['dependencies'].items():assert d.sha(p)==h,p
    metrics=e['metrics'];pb=[r for r in e['rows'] if r['cell'].startswith('pb')]
    headline=['dual__iu','gap__iu','dual__cond100','gap__joint0','dual__cond100_graph010','gap__graph010',
        'dual__equal_graph_perm','gap__equal_graph_perm','gap_scalar','surprisal_scalar']
    def metric_row(arm):
        m=metrics[arm]
        return [arm,m['prm']['answers'],number(m['prm']['auroc']),number(m['prm']['within_answer_auc']),
            percent(m['pb']['macro_f1']),sum(x['valid_decisions'] for x in m['pb']['cells'].values()),percent(m['pb_common_iu_gate']['macro_f1'])]
    headers=['Arm','PRMB answers','PRMB pooled AUC','Within-answer AUC','PB native F1','PB valid decisions','PB fixed-IU gate: diagnostic']
    all_table=table(headers,[metric_row(a) for a in manifest['arms']],'all-arms')
    comparisons=[]
    for p in c['pairs'].values():
        u=p['uncertainty'];ap,bp=p['left_prm']['auroc'],p['right_prm']['auroc'];ab,bb=p['left_pb']['macro_f1'],p['right_pb']['macro_f1']
        comparisons.append([p['left']+' minus '+p['right'],p['scope'],len(p['selected_ids']),
            number(ap-bp),interval(u['prm_common_valid_ci95']),number(100*(ab-bb),3),interval(u['pb_all_population_ci95'],100),
            interval(u['prm_within_answer_common_valid_ci95']),u['pb_all_population_valid_draws']])
    outcomes=[]
    for arm in headline:
        err=[r for r in pb if r['target']!=-1];clean=[r for r in pb if r['target']==-1]
        outcomes.append([arm,sum(r['peaks'][arm]==r['target'] for r in err),
            sum(r['predictions'][arm]==r['target'] for r in err),sum(r['predictions'][arm]==-1 for r in clean),
            sum(r['predictions'][arm]!=-1 for r in clean),sum(r['predictions'][arm]==-1 for r in err)])
    transitions={}
    for new,old in [('gap__iu','dual__iu'),('gap__graph010','dual__cond100_graph010')]:
        transitions[new+' minus '+old]={
            'prediction_changes':sum(r['predictions'][new]!=r['predictions'][old] for r in pb),
            'peak_changes':sum(r['peaks'][new]!=r['peaks'][old] for r in pb),
            'lost_correct':sum(r['predictions'][old]==r['target'] and r['predictions'][new]!=r['target'] for r in pb),
            'gained_correct':sum(r['predictions'][new]==r['target'] and r['predictions'][old]!=r['target'] for r in pb)}
    # Use a standard plotting tool; preserve exportable SVG and PNG artifacts.
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,task,field,title,scale in [(axes[0],'prm','auroc','PRMBench pooled AUROC',1),(axes[1],'pb','macro_f1','ProcessBench macro F1 (%)',100)]:
        for y,core in enumerate(d.CORES):
            before=metrics[d.ANCHORS[core]][task][field]*scale;after=metrics['gap__'+core][task][field]*scale
            ax.plot([before,after],[y,y],color='#a1aab7',lw=1.8)
            ax.scatter(before,y,color='#235b95',marker='o',s=42,label='Original bank' if y==0 else None,zorder=3)
            ax.scatter(after,y,color='#cb6517',marker='x',s=52,label='Gap replacement' if y==0 else None,zorder=4)
        ax.set_yticks(range(7),[LABELS[x] for x in d.CORES] if ax is axes[0] else ['']*7)
        ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,-.27),ncol=2,frameon=False)
    fig.suptitle('Same 110 development answers; corrected v3 labels\nPoints are descriptive; paired uncertainty is reported separately',fontsize=12)
    fig.savefig(OUT/'matched_fusion_points.svg',bbox_inches='tight');fig.savefig(OUT/'matched_fusion_points.png',dpi=180,bbox_inches='tight');plt.close(fig)
    source_links=[('Frozen protocol','../../docs/experiments/FUSION_TOKEN_GAP_V1.md'),('Fusion implementation','../../spectral_utils/fusion_token_gap.py'),
        ('Experiment driver','../../scripts/run_fusion_token_gap_v1.py'),('Review code','../../scripts/review_fusion_token_gap_v1.py'),
        ('Corrected prior benchmark','../localization_prm_label_audit_v1/REPORT.html'),('Text and peak explorer','../fusion_localization_forensics_v3/REPORT.html'),
        ('Review evidence','REVIEW.json'),('Raw confidence audit','RAW_CONFIDENCE_AUDIT.json'),('All metrics','EVALUATION.json'),
        ('All paired intervals','CONTRASTS.json'),('Export figure (SVG)','matched_fusion_points.svg')]
    counts=raw['counts']
    content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Provided-token gap inside IU-PCR and Joint L-SML — Step 315</title><style>
body{{margin:0;background:#f3f6fa;color:#172b45;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:34px 24px 70px}}
h1{{font-size:2.1rem;line-height:1.15}}h2{{margin-top:2.2rem}}p{{max-width:95ch}}.card,.scroll{{background:white;border:1px solid #d8e1ec;border-radius:12px;padding:20px;margin:18px 0}}
.verdict{{border-left:6px solid #ae5514}}.muted{{color:#52677e;font-size:.93rem}}.flow{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}.node{{background:#eaf1fa;padding:12px;border-radius:8px;flex:1;min-width:155px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 10px;border-bottom:1px solid #dde4ed;text-align:left;vertical-align:top}}th{{background:#eef3f9;position:sticky;top:0}}.scroll{{overflow:auto;padding:0}}td:first-child{{overflow-wrap:anywhere;min-width:175px}}
code{{background:#e9eef5;padding:2px 4px}}a{{color:#145d9b}}img{{max-width:100%;height:auto}}input{{padding:10px;font:inherit;max-width:90%;width:420px;border:1px solid #8198b3;border-radius:6px}}[hidden]{{display:none!important}}
@media print{{body{{background:white}}main{{max-width:none}}input,.filter{{display:none}}.scroll{{overflow:visible}}th{{position:static}}}}
</style></head><body><main><p class="muted">Step 315 · September 7, 2026 · Reviewed development experiment · 110 answers · one cached teacher-forced pass</p>
<h1>Does the provided-token preference gap help our fusion?</h1>
<div class="card verdict"><strong>No consistent improvement. Keep the original IU-PCR and Joint graph references.</strong>
<p>Replacing surprisal with the preference gap does not improve IU on either primary endpoint. Joint graph keeps exactly the same PB F1 and has lower PRMB AUC. No candidate is promoted. All107 comparison rows and25 registered paired contrasts are retained below.</p></div>
<h2>One change inside the existing method</h2><p>The model can be uncertain about many plausible next tokens, or can disagree with the token actually present in the provided answer. The new representation subtracts the preferred token's surprisal from the provided token's surprisal:</p>
<div class="card"><code>gap = -log p(provided token) + log p(preferred token)</code><p>This equals the difference between their logits when both probabilities use the same full, unwarped distribution. Zero means the provided token is one of the most probable tokens. A large value means model disagreement; it does not prove a reasoning error.</p></div>
<div class="flow"><div class="node">One provided answer<br>existing token telemetry</div><span aria-hidden="true">→</span><div class="node">8-token windows<br>replace 3 of27 features</div><span aria-hidden="true">→</span><div class="node">IU-PCR / Joint L-SML<br>same-answer fusion</div><span aria-hidden="true">→</span><div class="node">Fused trajectory<br>step + no-error decision</div></div>
<p>The original bank route stays81 moment /29 context answers. P stays27. Groups, normalization and weights are fitted from the current answer. Graph lambda stays0.1, inverse condition100, and the exact original permutation seeds are retained. Only a failed Joint model fit permits the declared IU fallback. This is offline full-answer fitting with a declared entropy sign anchor, not causal online or anchor-free fitting.</p>
<p>The old bank already contained both input streams. This is a reparameterization of existing information. The new window standard deviation can encode their covariance; it is not an independent measurement or a literature novelty claim.</p>
<h2>Matched results</h2><img src="matched_fusion_points.png" alt="Paired original and gap fusion results on PRMBench and ProcessBench; no consistent two-task improvement.">
{table(headers,[metric_row(a) for a in headline])}
<p>All headline rows cover24 PRMB and86 PB answers. PB F1 averages four subsets' harmonic means of clean-answer accuracy and exact-first-error accuracy. It is not ordinary binary F1 or overall accuracy. The fixed original IU gate is a diagnostic, not a newly selected deployment rule.</p>
<p>Gap-IU versus original IU: PRMB delta−0.001119, exploratory95% interval[−0.005004,+0.001971]; PB delta−1.337 points, interval[−5.265,0]. Its within-answer PRMB AUC is unchanged. Gap-Joint graph versus original graph: PRMB delta−0.005515, interval[−0.032549,+0.014061]; PB delta0, interval[0,0]. These intervals are unadjusted across25 comparisons and do not turn this inspected sample into confirmation.</p>
<h2>Where it still fails</h2>{table(['Arm','Raw exact peaks /53','Exact errors after gate /53','Clean correct /33','Clean false alarms /33','Error gates closed /53'],outcomes)}
<p>Gap-IU changes two PB predictions, losing one correct decision and gaining none. Gap-Joint graph changes one prediction, but that answer remains wrong, so every PB exact-success indicator is unchanged. Identical F1 does not mean identical scores or predictions.</p>
<p>Both scalar controls open the error gate on every PB answer. They find nine first-error peaks but falsely flag all33 clean answers, giving PB F1=0. This is a measured failure of these scalar-plus-GMM recipes, not proof that provided-token confidence has no useful information. They are attribution controls, not replacements for fusion.</p>
<h2>Fit health and source fidelity</h2><p>Native Joint is valid on{review['native_joint_valid']}/110 answers versus107/110 originally. Ten answers use the declared gap-IU fallback: nine have no admissible partition and one fails a fit guard. Of101 selected partitions,69 have K=3 and32 have K=4; selected partitions and accepted fits are distinct counts. All nine final outputs retain full coverage.</p>
<p>Direct original-cache review covers{counts['tokens']:,} tokens in{counts['raw_answer_and_label_joins']} answers. The provided token appears in top50 at{counts['provided_in_top50']:,} positions;{counts['provided_outside_top50']:,} positions are outside it. At retained positions, the saved provided-token log probability matches its top50 entry (maximum difference{raw['maximum_provided_logprob_difference']:.3g}). All six top50-derived streams and three original scalar streams replay. {100*counts['gap_near_zero']/counts['tokens']:.2f}% of gaps are within1e-5 of zero.</p>
<p>Top50 cannot independently reconstruct the omitted provided-token probability. That value comes from its separately saved channel, with consistency supported by the teacher-forced writer source. This review is not a new model forward pass or an empirical verification of original logit positions. Do not apply this formula to raw top1 plus a differently warped provided-token probability.</p>
<h2>Review, runtime and remaining work</h2><p>Review PASS:110 independent feature matrices,110 normalization checks,110 IU refits, native covariance/Jacobian checks,220 Laplacians,990 step/gate/output replays,10,780 exact inherited method rows,107 independent metric bundles and25 paired point/scope checks. Five representative grouping, Joint and DUFS refits and five explicit1000-draw bootstrap checks pass. The review shares disclosed scientific fitting kernels; it is not an external review.</p>
<p>CPU scoring took{frozen['seconds']:.2f}s with three workers; paired comparisons{c['seconds']:.2f}s; successful review{review['seconds']:.2f}s. Raw caches were loaded sequentially after workers exited, with a RAM guard. No inference, Claude-worktree edit or old-artifact overwrite was required.</p>
<p>The initial review harness stopped because it called the official label port incorrectly. Its return contract was corrected; all raw targets were then checked using the official port's per-step outcomes. Frozen experiment code, labels and predictions were unchanged.</p>
<p><strong>Next bounded stage:</strong> bridge the earlier unique trajectory/readout, sampling and regularization outputs to v3 labels and v2 source groups before using their PRMB results to choose another method. The current98-anchor bridge does not include every unique old arm. Preserve original artifacts and method/ID contracts; rescoring cannot repair multi-answer models fitted or selected under bad folds or labels. Those refits remain separate.</p>
<p>Joint/graph representation, feature and trajectory fusion, IMM/LOCA/Flows/KalmanNet, task-aware sampling, full comparator coverage, untouched confirmation and the later24-cell transfer remain open. This negative representation test does not close those families or achieve the full research goal.</p>
<h2>All107 corrected comparison rows</h2><p>These are98 inherited method entries plus9 new outputs; they are not107 newly trained independent methods. Pure historical Joint arms can have incomplete PRMB coverage. Use matched-ID contrasts before comparing them.</p>
<p class="filter"><label for="arm-search">Find an arm: </label><input id="arm-search" type="search" placeholder="For example: gap__, dual__iu, ar1"><span id="visible-count" aria-live="polite"></span></p>{all_table}
<h2>All25 registered paired comparisons</h2><p>PRMB uses common valid answers; PB includes failures in its selected population. Native scopes restrict model-fit eligibility explicitly. Intervals resample source groups within dataset cells using seed2026090706. Draws missing either PB class are excluded and their valid counts are shown.</p>
{table(['Left minus right','Scope','Selected answers','PRMB ΔAUC','PRMB95% interval','PB Δpoints','PB95% interval (points)','Within-answer95% interval','Valid PB draws /1000'],comparisons,'paired')}
<h2>Sources and artifacts</h2><ul>{''.join('<li><a href="'+escape(h)+'">'+escape(t)+'</a></li>' for t,h in source_links)}</ul>
<p class="muted">Browser visual rendering has not been checked. HTML structure, links, displayed values, provenance and the actual table-filter JavaScript are checked separately.</p>
</main><script>
const input=document.getElementById('arm-search');
const rows=Array.from(document.querySelectorAll('#all-arms tbody tr'));
function filterArms(){{const term=input.value.trim().toLowerCase();let visible=0;for(const row of rows){{row.hidden=!row.textContent.toLowerCase().includes(term);if(!row.hidden)visible++;}}document.getElementById('visible-count').textContent=' '+visible+' / '+rows.length;}}
input.addEventListener('input',filterArms);filterArms();
</script></body></html>'''
    (OUT/'REPORT.html').write_text(content,encoding='utf-8')
    markdown=['# Provided-token gap inside our fusion — Step315','',
        'Reviewed development result: no consistent improvement. Retain original IU and Joint graph references.',
        'Same110 answers, v3 labels, v2 groups, exact original graph seed namespace. P27, width8, original81/29 bank route.',
        '','| Arm | PRMB AUC | Within-answer AUC | PB F1 |','|---|---:|---:|---:|']
    for arm in headline:
        q=metrics[arm];markdown.append(f"| {arm} | {q['prm']['auroc']:.8f} | {q['prm']['within_answer_auc']:.8f} | {percent(q['pb']['macro_f1'])} |")
    markdown+=['',f"Native Joint {review['native_joint_valid']}/110 versus107; ten explicit fit-failure fallbacks. All9 final outputs valid.",
        'Gap-IU loses one correct PB decision; gap-Joint graph changes one incorrect prediction and preserves all exact-success indicators.',
        'Scalar gap and surprisal controls both flag all33 clean PB answers and have0% PB F1 under this fixed GMM readout.',
        f"Raw top50 audit: {counts['provided_in_top50']}/{counts['tokens']} provided tokens retained; {counts['provided_outside_top50']} outside. Omitted probabilities are not independently recoverable from top50.",
        'Review PASS; complete107 rows,25 comparisons, failure/coverage counts and five explicit bootstraps in REPORT.html.',
        'No new inference, untouched confirmation or overall goal completion.',
        'Next: bridge earlier unique readout/sampling/regularization methods to v3 labels and v2 source groups before reusing their conclusions. Multi-answer refits and the full research mandate remain open.']
    (OUT/'REPORT.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    d.save(OUT/'REPORT_PROVENANCE.json',{'status':'REVIEWED_DEVELOPMENT','transitions':transitions,
        'source_hashes':{str(OUT/n):d.sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','REVIEW.json','RAW_CONFIDENCE_AUDIT.json')},
        'renderer_sha256':d.sha(Path(__file__)),'report_sha256':d.sha(OUT/'REPORT.html'),
        'figure_sha256':d.sha(OUT/'matched_fusion_points.svg')})
    print('Rendered107 arms,25 comparisons, two exportable figures and source audit.',flush=True)


if __name__=='__main__':main()
